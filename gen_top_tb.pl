#!/usr/bin/env perl
use strict;
use warnings;
use Getopt::Long qw(GetOptions);
use File::Basename qw(basename);

# ------------------------------------------------------------
# gen_top_tb_and_instr.pl
#
# Generates:
#   1) Verilog testbench
#   2) Hex instruction file
#
# Behavior:
#   - Input image file is decimal text, comma/space/newline separated
#   - Default logical image row width is 224
#   - A single zero-padding border is added around the full image
#   - Kernels are loaded first
#   - Each kernel block is:
#         kernel values + bias + zero padding to multiple of 8
#   - Image data is streamed as sliding convolution windows
#   - Each window is flattened row-major and padded to multiple of 8
#   - Writes only happen when waitrequest_write is low
#   - read is held high so outputs can drain
#   - Output words are displayed when available
#   - Expected convolution outputs are emitted as comments
#   - You can limit how many convolution windows are streamed
#   - Instruction stream is generated in 32-bit hex
#
# Example:
#   perl gen_top_tb_and_instr.pl \
#       --input sample_data.txt \
#       --kernel k0.txt:0.125 \
#       --kernel k1.txt:-0.75 \
#       --output TOP_TB_gen.v \
#       --instr-output instr.hex \
#       --max-convs 100 \
#       --relu
#
# Instruction format:
#   opcode    : bits 31:29
#   start_loc : bits 28:20
#   length    : bits 19:10
#   param_loc : bits 9:1
#   unused    : bit 0
#
# Opcodes:
#   MAC  = 3'b000
#   LOAD = 3'b001
#   RELU = 3'b010
#   END  = 3'b011
# ------------------------------------------------------------

my $input_file    = '';
my @kernel_args   = ();
my $output_file   = 'TOP_TB.v';
my $instr_output  = 'instr.hex';

my $row_width     = 224;
my $chunk_size    = 8;
my $rst_cycles    = 1;
my $pad_border    = 1;
my $max_convs     = 0;   # 0 => unlimited
my $do_relu       = 0;

# Address planning
# Each address stores 8 x 32-bit numbers.
# Kernel blocks go into parameter address space starting here.
# Window data blocks go into data address space starting here.
my $param_base_addr = 0;
my $data_base_addr  = 0;

GetOptions(
    'input=s'         => \$input_file,
    'kernel=s@'       => \@kernel_args,
    'output=s'        => \$output_file,
    'instr-output=s'  => \$instr_output,
    'row-width=i'     => \$row_width,
    'chunk-size=i'    => \$chunk_size,
    'rst-cycles=i'    => \$rst_cycles,
    'pad-border=i'    => \$pad_border,
    'max-convs=i'     => \$max_convs,
    'relu!'           => \$do_relu,
    'param-base=i'    => \$param_base_addr,
    'data-base=i'     => \$data_base_addr,
) or die "Bad arguments\n";

die "Missing --input\n" unless $input_file;
die "Need at least one --kernel file:bias\n" unless @kernel_args;
die "--row-width must be > 0\n" unless $row_width > 0;
die "--chunk-size must be > 0\n" unless $chunk_size > 0;
die "--chunk-size must be 8 for this instruction format\n" unless $chunk_size == 8;
die "--pad-border must be >= 0\n" unless $pad_border >= 0;
die "--max-convs must be >= 0\n" unless $max_convs >= 0;
die "--param-base must be >= 0\n" unless $param_base_addr >= 0;
die "--data-base must be >= 0\n" unless $data_base_addr >= 0;

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

sub make_tb_module_name {
    my ($path) = @_;
    my $base = basename($path);

    $base =~ s/\.[^.]+$//;
    $base =~ s/[^A-Za-z0-9_]/_/g;

    if ($base =~ /^[0-9]/) {
        $base = "_" . $base;
    }

    $base = "generated_tb" if $base eq '';
    return $base;
}

sub read_decimal_values {
    my ($file) = @_;
    open my $fh, '<', $file or die "Cannot open '$file': $!";

    local $/;
    my $text = <$fh>;
    close $fh;

    $text =~ s/\/\/.*$//mg;
    $text =~ s/#.*$//mg;

    my @vals;
    for my $tok (split /[\s,]+/, $text) {
        next if $tok eq '';
        die "Non-numeric token '$tok' in '$file'\n"
            unless $tok =~ /^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$/;
        push @vals, 0.0 + $tok;
    }

    die "No numeric values found in '$file'\n" unless @vals;
    return \@vals;
}

sub float_to_hex32 {
    my ($f) = @_;
    my $bin = pack('f>', $f);
    return uc unpack('H8', $bin);
}

sub parse_kernel_arg {
    my ($arg) = @_;
    my ($file, $bias) = split /:/, $arg, 2;

    die "Kernel argument must be file:bias, got '$arg'\n"
        unless defined $file && defined $bias && $file ne '';

    die "Bias for '$file' is not numeric: '$bias'\n"
        unless $bias =~ /^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$/;

    my $vals  = read_decimal_values($file);
    my $count = scalar @$vals;
    my $size  = int(sqrt($count));

    die "Kernel '$file' has $count values; not a perfect square\n"
        unless $size * $size == $count;

    return {
        file   => $file,
        size   => $size,
        values => $vals,
        bias   => 0.0 + $bias,
    };
}

sub make_image_rows {
    my ($vals_ref, $width) = @_;
    my @vals = @$vals_ref;

    my $height = int((@vals + $width - 1) / $width);
    my @rows;

    for my $r (0 .. $height - 1) {
        my @row;
        for my $c (0 .. $width - 1) {
            my $idx = $r * $width + $c;
            push @row, ($idx < @vals) ? $vals[$idx] : 0.0;
        }
        push @rows, \@row;
    }

    return \@rows;
}

sub add_zero_border {
    my ($rows_ref, $pad) = @_;
    return $rows_ref if $pad == 0;

    my $orig_h = scalar @$rows_ref;
    my $orig_w = scalar @{ $rows_ref->[0] };

    my $new_w = $orig_w + 2 * $pad;
    my @rows;

    for (1 .. $pad) {
        push @rows, [ (0.0) x $new_w ];
    }

    for my $r (0 .. $orig_h - 1) {
        my @new_row = (
            ((0.0) x $pad),
            @{ $rows_ref->[$r] },
            ((0.0) x $pad),
        );
        push @rows, \@new_row;
    }

    for (1 .. $pad) {
        push @rows, [ (0.0) x $new_w ];
    }

    return \@rows;
}

sub pad_list_to_multiple {
    my ($vals_ref, $multiple, $pad_value) = @_;
    my @vals = @$vals_ref;
    my $rem = @vals % $multiple;
    if ($rem != 0) {
        my $need = $multiple - $rem;
        push @vals, ($pad_value) x $need;
    }
    return \@vals;
}

sub build_kernel_stream {
    my ($kernels_ref, $chunk_size, $param_base_addr) = @_;
    my @stream;
    my @meta;

    my $addr = $param_base_addr;

    for my $k (@$kernels_ref) {
        my @block = (@{ $k->{values} }, $k->{bias});
        my $raw_words = scalar @block;
        my $padded = pad_list_to_multiple(\@block, $chunk_size, 0.0);
        my $padded_words = scalar(@$padded);

        die "Kernel block padded_words must be divisible by 8\n"
            if $padded_words % 8 != 0;

        my $addr_words = $padded_words / 8;

        push @stream, @$padded;
        push @meta, {
            file         => $k->{file},
            size         => $k->{size},
            bias         => $k->{bias},
            raw_words    => $raw_words,
            padded_words => $padded_words,
            base_addr    => $addr,
            addr_words   => $addr_words,
        };

        $addr += $addr_words;
    }

    return (\@stream, \@meta);
}

sub build_windowed_image_stream {
    my ($rows_ref, $kernels_ref, $chunk_size, $max_convs, $data_base_addr) = @_;

    my $img_h = scalar @$rows_ref;
    my $img_w = scalar @{ $rows_ref->[0] };

    my $k = $kernels_ref->[0]{size};
    for my $ker (@$kernels_ref) {
        die "All kernels must have the same size for shared image window streaming\n"
            if $ker->{size} != $k;
    }

    die "Image too small for ${k}x${k} kernel\n" if $img_h < $k || $img_w < $k;

    my $out_h = $img_h - $k + 1;
    my $out_w = $img_w - $k + 1;
    my $total_possible = $out_h * $out_w;

    my @stream;
    my @meta;
    my $count = 0;

    my $raw_words_per_window = $k * $k;
    my $padded_words_per_window =
        scalar @{ pad_list_to_multiple([ (0.0) x $raw_words_per_window ], $chunk_size, 0.0) };

    die "Window padded_words must be divisible by 8\n"
        if $padded_words_per_window % 8 != 0;

    my $addr_words_per_window = $padded_words_per_window / 8;

    OUTER:
    for my $r (0 .. $out_h - 1) {
        for my $c (0 .. $out_w - 1) {
            last OUTER if $max_convs && $count >= $max_convs;

            my @patch;
            for my $kr (0 .. $k - 1) {
                for my $kc (0 .. $k - 1) {
                    push @patch, $rows_ref->[$r + $kr][$c + $kc];
                }
            }

            my $padded = pad_list_to_multiple(\@patch, $chunk_size, 0.0);

            push @stream, @$padded;
            push @meta, {
                out_row       => $r,
                out_col       => $c,
                raw_words     => scalar(@patch),
                padded_words  => scalar(@$padded),
                window_index  => $count,
                base_addr     => $data_base_addr,          # same buffer reused every window
                addr_words    => $addr_words_per_window,   # same number every window
            };

            $count++;
        }
    }

    return (\@stream, \@meta, $out_h, $out_w, $k, $count, $total_possible);
}

sub conv2d_valid {
    my (%args) = @_;
    my $img   = $args{image};
    my $kern  = $args{kernel};
    my $bias  = $args{bias};

    my $h = scalar @$img;
    my $w = scalar @{ $img->[0] };
    my $k = $kern->{size};

    die "Image too small for kernel '$kern->{file}'\n" if $h < $k || $w < $k;

    my $out_h = $h - $k + 1;
    my $out_w = $w - $k + 1;

    my @out;
    for my $r (0 .. $out_h - 1) {
        my @row;
        for my $c (0 .. $out_w - 1) {
            my $sum = $bias;
            for my $kr (0 .. $k - 1) {
                for my $kc (0 .. $k - 1) {
                    $sum += $img->[$r + $kr][$c + $kc] *
                            $kern->{values}[ $kr * $k + $kc ];
                }
            }
            push @row, $sum;
        }
        push @out, \@row;
    }

    return {
        out_h => $out_h,
        out_w => $out_w,
        data  => \@out,
    };
}

sub encode_instr {
    my (%args) = @_;
    my $opcode    = $args{opcode}    // 0;
    my $start_loc = $args{start_loc} // 0;
    my $length    = $args{length}    // 0;
    my $param_loc = $args{param_loc} // 0;

    die "opcode out of range: $opcode\n"    if $opcode    < 0 || $opcode    > 0x7;
    die "start_loc out of range: $start_loc\n" if $start_loc < 0 || $start_loc > 0x1FF;
    die "length out of range: $length\n"    if $length    < 0 || $length    > 0x3FF;
    die "param_loc out of range: $param_loc\n" if $param_loc < 0 || $param_loc > 0x1FF;

    my $word = 0;
    $word |= ($opcode    & 0x7)   << 29;
    $word |= ($start_loc & 0x1FF) << 20;
    $word |= ($length    & 0x3FF) << 10;
    $word |= ($param_loc & 0x1FF) << 1;
    # bit 0 unused => 0

    return sprintf("%08X", $word);
}

sub generate_instruction_words {
    my (%args) = @_;

    my $kernel_meta = $args{kernel_meta};
    my $image_meta  = $args{image_meta};
    my $kernels     = $args{kernels};
    my $do_relu     = $args{do_relu};

    my @instrs;
    my @comments;

    my $OP_MAC  = 0b000;
    my $OP_LOAD = 0b001;
    my $OP_RELU = 0b010;
    my $OP_END  = 0b011;

    # --------------------------------------------------
    # 1) Load each kernel block once
    #    LOAD only uses start_loc
    # --------------------------------------------------
    for my $ki (0 .. $#$kernel_meta) {
        my $km = $kernel_meta->[$ki];

        for my $ofs (0 .. $km->{addr_words} - 1) {
            my $addr = $km->{base_addr} + $ofs;

            my $hex = encode_instr(
                opcode    => $OP_LOAD,
                start_loc => $addr,
                length    => 0,
                param_loc => 0,
            );

            push @instrs, $hex;
            push @comments, sprintf(
                "LOAD kernel %d chunk %d -> addr %d",
                $ki, $ofs, $addr
            );
        }
    }

    # --------------------------------------------------
    # 2) For each window:
    #    - LOAD window chunks into the SAME reusable data buffer
    #    - MAC once per kernel using same start_loc
    # --------------------------------------------------
    for my $wi (0 .. $#$image_meta) {
        my $wm = $image_meta->[$wi];

        # Load current window into reusable data buffer
        for my $ofs (0 .. $wm->{addr_words} - 1) {
            my $addr = $wm->{base_addr} + $ofs;

            my $hex = encode_instr(
                opcode    => $OP_LOAD,
                start_loc => $addr,
                length    => 0,
                param_loc => 0,
            );

            push @instrs, $hex;
            push @comments, sprintf(
                "LOAD window %d chunk %d -> addr %d (out=%d,%d)",
                $wi, $ofs, $addr, $wm->{out_row}, $wm->{out_col}
            );
        }

        # Reuse the same loaded window for every kernel
        for my $ki (0 .. $#$kernel_meta) {
            my $km = $kernel_meta->[$ki];
            my $kernel_len = $kernels->[$ki]{size} * $kernels->[$ki]{size};

            my $hex = encode_instr(
                opcode    => $OP_MAC,
                start_loc => $wm->{base_addr},   # reusable window buffer
                length    => $kernel_len,
                param_loc => $km->{base_addr},   # kernel+bias base
            );

            push @instrs, $hex;
            push @comments, sprintf(
                "MAC window %d kernel %d start_loc=%d len=%d param_loc=%d",
                $wi, $ki, $wm->{base_addr}, $kernel_len, $km->{base_addr}
            );

            if ($do_relu) {
                my $relu_hex = encode_instr(
                    opcode    => $OP_RELU,
                    start_loc => 0,
                    length    => 0,
                    param_loc => 0,
                );

                push @instrs, $relu_hex;
                push @comments, sprintf(
                    "RELU after MAC window %d kernel %d",
                    $wi, $ki
                );
            }
        }
    }

    # --------------------------------------------------
    # 3) END
    # --------------------------------------------------
    my $end_hex = encode_instr(
        opcode    => $OP_END,
        start_loc => 0,
        length    => 0,
        param_loc => 0,
    );

    push @instrs, $end_hex;
    push @comments, "END";

    return (\@instrs, \@comments);
}

sub write_instruction_file {
    my (%args) = @_;
    my $file     = $args{file};
    my $instrs   = $args{instrs};
    my $comments = $args{comments};

    open my $fh, '>', $file or die "Cannot write instruction file '$file': $!";

    for my $i (0 .. $#$instrs) {
        my $hex = $instrs->[$i];
        my $cmt = $comments->[$i];
        print $fh "$hex";
        print $fh "    // $cmt" if defined $cmt;
        print $fh "\n";
    }

    close $fh;
}

sub emit_tb {
    my (%args) = @_;

    my $outfile            = $args{outfile};
    my $module_name        = $args{module_name};
    my $kernel_stream      = $args{kernel_stream};
    my $kernel_meta        = $args{kernel_meta};
    my $image_stream       = $args{image_stream};
    my $image_meta         = $args{image_meta};
    my $kernels            = $args{kernels};
    my $conv_results       = $args{conv_results};
    my $row_width          = $args{row_width};
    my $unpadded_height    = $args{unpadded_height};
    my $padded_width       = $args{padded_width};
    my $padded_height      = $args{padded_height};
    my $chunk_size         = $args{chunk_size};
    my $rst_cycles         = $args{rst_cycles};
    my $input_file         = $args{input_file};
    my $pad_border         = $args{pad_border};
    my $window_out_h       = $args{window_out_h};
    my $window_out_w       = $args{window_out_w};
    my $window_kernel_size = $args{window_kernel_size};
    my $max_convs          = $args{max_convs};
    my $actual_convs       = $args{actual_convs};
    my $total_possible     = $args{total_possible_convs};
    my $instr_output       = $args{instr_output};
    my $instr_count        = $args{instr_count};

    open my $fh, '>', $outfile or die "Cannot write '$outfile': $!";

    print $fh "module $module_name ();\n\n";

    print $fh "\treg clk, rst;\n";
    print $fh "\treg reset_write, reset_reset;\n";
    print $fh "\treg [4:0] burstcount;\n";
    print $fh "\treg write;\n";
    print $fh "\treg [31:0] writedata;\n";
    print $fh "\treg read;\n\n";

    print $fh "\twire waitrequest_write;\n";
    print $fh "\twire [63:0] readdata;\n";
    print $fh "\twire waitrequest_read;\n\n";

    print $fh "\tparameter Default = 5'b00000, Init = 5'b00001, Test = 5'b00010, Done = 5'b01111;\n";
    print $fh "\treg [4:0] Present_state = Default;\n\n";

    print $fh "\tTOP dut (\n";
    print $fh "\t\t.clk(clk),\n";
    print $fh "\t\t.reset({31'b0, rst}),\n";
    print $fh "\t\t.reset_write(reset_write),\n";
    print $fh "\t\t.reset_reset(reset_reset),\n";
    print $fh "\t\t.burstcount(burstcount),\n";
    print $fh "\t\t.waitrequest_write(waitrequest_write),\n";
    print $fh "\t\t.write(write),\n";
    print $fh "\t\t.writedata(writedata),\n";
    print $fh "\t\t.readdata(readdata),\n";
    print $fh "\t\t.read(read),\n";
    print $fh "\t\t.waitrequest_read(waitrequest_read)\n";
    print $fh "\t);\n\n";

    print $fh "\tinitial begin\n";
    print $fh "\t\tclk = 0;\n";
    print $fh "\t\tforever #10 clk = ~clk;\n";
    print $fh "\tend\n\n";

    print $fh "\talways @(posedge clk) begin\n";
    print $fh "\t\tcase (Present_state)\n";
    print $fh "\t\t\tDefault : Present_state <= Init;\n";
    print $fh "\t\t\tInit    : Present_state <= Test;\n";
    print $fh "\t\t\tTest    : Present_state <= Test;\n";
    print $fh "\t\t\tDone    : Present_state <= Done;\n";
    print $fh "\t\t\tdefault : Present_state <= Default;\n";
    print $fh "\t\tendcase\n";
    print $fh "\tend\n\n";

    print $fh "\tinitial begin\n";
    print $fh "\t\tread = 1'b1;\n";
    print $fh "\tend\n\n";

    print $fh "\ttask write_word;\n";
    print $fh "\t\tinput [31:0] data_word;\n";
    print $fh "\t\tbegin\n";
    print $fh "\t\t\t\@(posedge clk);\n";
    print $fh "\t\t\twhile (waitrequest_write) begin\n";
    print $fh "\t\t\t\twrite <= 1'b0;\n";
    print $fh "\t\t\t\twritedata <= 32'h00000000;\n";
    print $fh "\t\t\t\t\@(posedge clk);\n";
    print $fh "\t\t\tend\n\n";
    print $fh "\t\t\twrite <= 1'b1;\n";
    print $fh "\t\t\twritedata <= data_word;\n";
    print $fh "\t\t\t\@(posedge clk);\n\n";
    print $fh "\t\t\twrite <= 1'b0;\n";
    print $fh "\t\t\twritedata <= 32'h00000000;\n";
    print $fh "\t\tend\n";
    print $fh "\tendtask\n\n";

    print $fh "\talways \@(posedge clk) begin\n";
    print $fh "\t\tif (read && !waitrequest_read) begin\n";
    print $fh "\t\t\t\$display(\"T=%0t OUTPUT = %h\", \$time, readdata);\n";
    print $fh "\t\tend\n";
    print $fh "\tend\n\n";

    print $fh "\tinitial begin\n";
    print $fh "\t\trst = 1'b0;\n";
    print $fh "\t\treset_write = 1'b0;\n";
    print $fh "\t\treset_reset = 1'b0;\n";
    print $fh "\t\tburstcount = 5'd1;\n";
    print $fh "\t\twrite = 1'b0;\n";
    print $fh "\t\twritedata = 32'h00000000;\n\n";

    print $fh "\t\t#5;\n";
    print $fh "\t\trst = 1'b1;\n";
    print $fh "\t\treset_write = 1'b1;\n";
    print $fh "\t\treset_reset = 1'b1;\n";
    print $fh "\t\t#" . (20 * $rst_cycles) . ";\n";
    print $fh "\t\trst = 1'b0;\n";
    print $fh "\t\treset_write = 1'b0;\n";
    print $fh "\t\treset_reset = 1'b0;\n";
    print $fh "\t\t#20;\n\n";

    print $fh "\t\t// --------------------------------------------------\n";
    print $fh "\t\t// Load kernels + bias + zero pad data stream\n";
    print $fh "\t\t// --------------------------------------------------\n";
    for my $i (0 .. $#$kernel_stream) {
        my $hex = float_to_hex32($kernel_stream->[$i]);
        print $fh "\t\twrite_word(32'h$hex);\n";
    }

    print $fh "\n";
    print $fh "\t\t// --------------------------------------------------\n";
    print $fh "\t\t// Load zero-padded sliding windows from input file: $input_file\n";
    print $fh "\t\t// Each window is flattened row-major and padded to a multiple of $chunk_size\n";
    print $fh "\t\t// --------------------------------------------------\n";
    for my $i (0 .. $#$image_stream) {
        my $hex = float_to_hex32($image_stream->[$i]);
        print $fh "\t\twrite_word(32'h$hex);\n";
    }

    print $fh "\n";
    print $fh "\t\t#1000;\n";
    print $fh "\t\t\$finish;\n";
    print $fh "\tend\n\n";

    print $fh "\t// --------------------------------------------------\n";
    print $fh "\t// Metadata\n";
    print $fh "\t// --------------------------------------------------\n";
    print $fh "\t// TB module name      : $module_name\n";
    print $fh "\t// Input file          : $input_file\n";
    print $fh "\t// Instruction file    : $instr_output\n";
    print $fh "\t// Instruction count   : $instr_count\n";
    print $fh "\t// Original width      : $row_width\n";
    print $fh "\t// Original height     : $unpadded_height\n";
    print $fh "\t// Zero border         : $pad_border\n";
    print $fh "\t// Padded width        : $padded_width\n";
    print $fh "\t// Padded height       : $padded_height\n";
    print $fh "\t// Chunk size          : $chunk_size\n";
    print $fh "\t// Window kernel size  : ${window_kernel_size}x${window_kernel_size}\n";
    print $fh "\t// Window output size  : ${window_out_h} x ${window_out_w}\n";
    print $fh "\t// Total possible convs: $total_possible\n";
    print $fh "\t// Max conv limit      : " . ($max_convs ? $max_convs : "unlimited") . "\n";
    print $fh "\t// Actual convs stream : $actual_convs\n\n";

    for my $i (0 .. $#$kernel_meta) {
        my $m = $kernel_meta->[$i];
        print $fh "\t// Kernel $i file=$m->{file}, size=$m->{size}x$m->{size}, bias=$m->{bias}, raw_words=$m->{raw_words}, padded_words=$m->{padded_words}, base_addr=$m->{base_addr}, addr_words=$m->{addr_words}\n";
    }
    print $fh "\n";

    for my $i (0 .. $#$image_meta) {
        my $m = $image_meta->[$i];
        print $fh "\t// Window $i at output ($m->{out_row},$m->{out_col}): raw_words=$m->{raw_words}, padded_words=$m->{padded_words}, base_addr=$m->{base_addr}, addr_words=$m->{addr_words}\n";
    }
    print $fh "\n";

    print $fh "\t// --------------------------------------------------\n";
    print $fh "\t// Expected convolution outputs (reference only)\n";
    print $fh "\t// --------------------------------------------------\n";
    for my $ki (0 .. $#$kernels) {
        my $k = $kernels->[$ki];
        my $res = $conv_results->[$ki];

        print $fh "\t// Kernel $ki : $k->{file}, size=$k->{size}x$k->{size}, bias=$k->{bias}\n";
        print $fh "\t// Output size: $res->{out_h} x $res->{out_w}\n";
        for my $r (0 .. $res->{out_h} - 1) {
            my @hexrow = map { "0x" . float_to_hex32($_) } @{ $res->{data}[$r] };
            print $fh "\t// OUT[$ki][$r] = " . join(", ", @hexrow) . "\n";
        }
        print $fh "\n";
    }

    print $fh "endmodule\n";
    close $fh;
}

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

my $tb_module_name = make_tb_module_name($output_file);

my $input_vals        = read_decimal_values($input_file);
my @kernels           = map { parse_kernel_arg($_) } @kernel_args;
my $image_rows        = make_image_rows($input_vals, $row_width);
my $orig_image_h      = scalar @$image_rows;
my $padded_image_rows = add_zero_border($image_rows, $pad_border);
my $padded_image_h    = scalar @$padded_image_rows;
my $padded_image_w    = scalar @{ $padded_image_rows->[0] };

my ($kernel_stream, $kernel_meta) =
    build_kernel_stream(\@kernels, $chunk_size, $param_base_addr);

my $kernel_addr_end = $param_base_addr;
for my $km (@$kernel_meta) {
    my $end = $km->{base_addr} + $km->{addr_words};
    $kernel_addr_end = $end if $end > $kernel_addr_end;
}

# Shared address space: window buffer starts after all kernel blocks
$data_base_addr = $kernel_addr_end;

my ($image_stream, $image_meta, $window_out_h, $window_out_w, $window_kernel_size,
    $actual_convs, $total_possible_convs) =
    build_windowed_image_stream(
        $padded_image_rows,
        \@kernels,
        $chunk_size,
        $max_convs,
        $data_base_addr,
    );

my @conv_results;
for my $k (@kernels) {
    push @conv_results, conv2d_valid(
        image  => $padded_image_rows,
        kernel => $k,
        bias   => $k->{bias},
    );
}

my ($instr_words, $instr_comments) = generate_instruction_words(
    kernel_meta => $kernel_meta,
    image_meta  => $image_meta,
    kernels     => \@kernels,
    do_relu     => $do_relu,
);

write_instruction_file(
    file     => $instr_output,
    instrs   => $instr_words,
    comments => $instr_comments,
);

emit_tb(
    outfile              => $output_file,
    module_name          => $tb_module_name,
    input_file           => $input_file,
    instr_output         => $instr_output,
    instr_count          => scalar(@$instr_words),
    kernel_stream        => $kernel_stream,
    kernel_meta          => $kernel_meta,
    image_stream         => $image_stream,
    image_meta           => $image_meta,
    kernels              => \@kernels,
    conv_results         => \@conv_results,
    row_width            => $row_width,
    unpadded_height      => $orig_image_h,
    padded_width         => $padded_image_w,
    padded_height        => $padded_image_h,
    chunk_size           => $chunk_size,
    rst_cycles           => $rst_cycles,
    pad_border           => $pad_border,
    window_out_h         => $window_out_h,
    window_out_w         => $window_out_w,
    window_kernel_size   => $window_kernel_size,
    max_convs            => $max_convs,
    actual_convs         => $actual_convs,
    total_possible_convs => $total_possible_convs,
);

print "Generated $output_file\n";
print "Generated $instr_output\n";
print "TB module name         : $tb_module_name\n";
print "Image values           : " . scalar(@$input_vals) . "\n";
print "Original image size    : ${row_width}x${orig_image_h}\n";
print "Zero border            : $pad_border\n";
print "Padded image size      : ${padded_image_w}x${padded_image_h}\n";
print "Kernel stream words    : " . scalar(@$kernel_stream) . "\n";
print "Image window words     : " . scalar(@$image_stream) . "\n";
print "Window output size     : ${window_out_w}x${window_out_h}\n";
print "Total possible convs   : $total_possible_convs\n";
print "Max conv limit         : " . ($max_convs ? $max_convs : "unlimited") . "\n";
print "Actual convs streamed  : $actual_convs\n";
print "Instruction words      : " . scalar(@$instr_words) . "\n";
for my $i (0 .. $#kernels) {
    my $k = $kernels[$i];
    my $r = $conv_results[$i];
    print "Kernel $i => $k->{size}x$k->{size}, bias=$k->{bias}, output=$r->{out_w}x$r->{out_h}\n";
}