module PE(
    input clk, rst, i_vld, bias_add,
	 input [1:0] bias_loc, 
	 input [1:0] cntrl_in,
	 input [31:0] pc_in,
    input [31:0] in0, in1, in2, in3, in4, in5, in6, in7, 
	 output out_valid,
	 output reg [1:0] cntrl_clock,
	 output reg [31:0] pc_clock,
    output [31:0] out_node
);

	 wire [31:0] mul0_ia, mul0_ib, mul1_ia, mul1_ib, mul2_ia, mul2_ib, mul3_ia, mul3_ib;

    wire mul0_overflow, mul1_overflow, mul2_overflow, mul3_overflow;
    wire mul0_o_res_vld, mul1_o_res_vld, mul2_o_res_vld, mul3_o_res_vld;
    wire [31:0] mul0_o_res, mul1_o_res, mul2_o_res, mul3_o_res;

    wire add0_overflow, add1_overflow, add2_overflow;
    wire add0_o_res_vld, add1_o_res_vld, add2_o_res_vld;
    wire [31:0] add0_o_res, add1_o_res, add2_o_res;

    wire add_acc_o_res_vld;
    wire add_acc_overflow;
	 
	 wire [31:0] out_reg_out;

    reg [31:0] out_reg;
	 
	 wire acc_vld;
	 reg [31:0] accumulate_in;
	 reg [17:0] bias_add_clk;
	 reg [11:0] in_valid_clk;
	 reg [31:0] pc_clk_reg_0, pc_clk_reg_1, pc_clk_reg_2, pc_clk_reg_3, pc_clk_reg_4, pc_clk_reg_5, pc_clk_reg_6, pc_clk_reg_7, pc_clk_reg_8, pc_clk_reg_9, pc_clk_reg_10, pc_clk_reg_11;
	 reg [1:0] cntrl_clk_reg_0, cntrl_clk_reg_1, cntrl_clk_reg_2, cntrl_clk_reg_3, cntrl_clk_reg_4, cntrl_clk_reg_5, cntrl_clk_reg_6, cntrl_clk_reg_7, cntrl_clk_reg_8, cntrl_clk_reg_9, cntrl_clk_reg_10, cntrl_clk_reg_11;
	 
	 reg [31:0] alt_acc;


	always @(posedge clk) begin
		if (rst) begin
			out_reg <= 32'd0;
			bias_add_clk <= 18'd0;
			in_valid_clk <= 12'd0;
		end 
		else begin
			if (bias_add_clk[3] == 1'b1) begin
				out_reg <= 32'd0;
			end else begin
				out_reg <= in_valid_clk[0] ? out_reg_out : out_reg;
			end
			bias_add_clk <= {bias_add, bias_add_clk[17], bias_add_clk[16], bias_add_clk[15], bias_add_clk[14], bias_add_clk[13], bias_add_clk[12], bias_add_clk[11], bias_add_clk[10], bias_add_clk[9], bias_add_clk[8], bias_add_clk[7], bias_add_clk[6], bias_add_clk[5], bias_add_clk[4], bias_add_clk[3], bias_add_clk[2], bias_add_clk[1]};
			in_valid_clk <= {i_vld, in_valid_clk[11], in_valid_clk[10], in_valid_clk[9], in_valid_clk[8], in_valid_clk[7], in_valid_clk[6], in_valid_clk[5], in_valid_clk[4], in_valid_clk[3], in_valid_clk[2], in_valid_clk[1]};
			
			if (bias_add_clk[8]) begin
				pc_clock <= pc_clk_reg_1;
				cntrl_clock <= cntrl_clk_reg_1;
				alt_acc <= out_reg_out;
				out_reg <= 32'd0;
			end
			else if (bias_add_clk[7]) begin
				out_reg <= out_reg_out;
				accumulate_in <= alt_acc;
			end 
			else if (bias_add_clk[6]) begin
				alt_acc <= out_reg_out;
				out_reg <= 32'd0;
			end
			else if (bias_add_clk[4]) begin
				out_reg <= out_reg_out;
				accumulate_in <= alt_acc;
			end
			else begin
				accumulate_in <= add2_o_res;
			end

		
		end
		

		
		pc_clk_reg_11 <= pc_in;
		pc_clk_reg_10 <= pc_clk_reg_11;
		pc_clk_reg_9 <= pc_clk_reg_10;
		pc_clk_reg_8 <= pc_clk_reg_9;
		pc_clk_reg_7 <= pc_clk_reg_8;
		pc_clk_reg_6 <= pc_clk_reg_7;
		pc_clk_reg_5 <= pc_clk_reg_6;
		pc_clk_reg_4 <= pc_clk_reg_5;
		pc_clk_reg_3 <= pc_clk_reg_4;
		pc_clk_reg_2 <= pc_clk_reg_3;
		pc_clk_reg_1 <= pc_clk_reg_2;
		pc_clk_reg_0 <= pc_clk_reg_1;
		//pc_clock <= pc_clk_reg_0;
		
		cntrl_clk_reg_11 <= cntrl_in;
		cntrl_clk_reg_10 <= cntrl_clk_reg_11;
		cntrl_clk_reg_9 <= cntrl_clk_reg_10;
		cntrl_clk_reg_8 <= cntrl_clk_reg_9;
		cntrl_clk_reg_7 <= cntrl_clk_reg_8;
		cntrl_clk_reg_6 <= cntrl_clk_reg_7;
		cntrl_clk_reg_5 <= cntrl_clk_reg_6;
		cntrl_clk_reg_4 <= cntrl_clk_reg_5;
		cntrl_clk_reg_3 <= cntrl_clk_reg_4;
		cntrl_clk_reg_2 <= cntrl_clk_reg_3;
		cntrl_clk_reg_1 <= cntrl_clk_reg_2;
		cntrl_clk_reg_0 <= cntrl_clk_reg_1;
		//cntrl_clock <= cntrl_clk_reg_0;
		
    end
	 
	 assign mul0_ia = (bias_add && (bias_loc == 2'b00)) ? 32'h3f800000: in0;
	 assign mul0_ib = in1;
	 assign mul1_ia = (bias_add && (bias_loc == 2'b01)) ? 32'h3f800000: in2;
	 assign mul1_ib = in3;
	 assign mul2_ia = (bias_add && (bias_loc == 2'b10)) ? 32'h3f800000: in4;
	 assign mul2_ib = in5;
	 assign mul3_ia = (bias_add && (bias_loc == 2'b11)) ? 32'h3f800000: in6;
	 assign mul3_ib = in7;
	 
	 reg [2:0] count0, count1;
	 	 
	//3 latency
    custom_ip_mult multiplier0(
        .clk(clk),
		  .areset(rst),
        .a(mul0_ia),
        .b(mul0_ib),
        .q(mul0_o_res)
    );

    custom_ip_mult multiplier1(
        .clk(clk),
		  .areset(rst),
        .a(mul1_ia),
        .b(mul1_ib),
        .q(mul1_o_res)
    );

    custom_ip_mult multiplier2(
        .clk(clk),
		  .areset(rst),
        .a(mul2_ia),
        .b(mul2_ib),
        .q(mul2_o_res)
    );

    custom_ip_mult multiplier3(
        .clk(clk),
		  .areset(rst),
        .a(mul3_ia),
        .b(mul3_ib),
        .q(mul3_o_res)
    );

	 //3 latency
    custom_ip_add adder0(
        .clk(clk),
		  .areset(rst),
        .a(mul0_o_res),
        .b(mul1_o_res),
        .q(add0_o_res)
    );

    custom_ip_add adder1(
        .clk(clk),
		  .areset(rst),
        .a(mul2_o_res),
        .b(mul3_o_res),
        .q(add1_o_res)
    );
	 
	 // 3 latency
    custom_ip_add adder2(
        .clk(clk),
		  .areset(rst),
        .a(add0_o_res),
        .b(add1_o_res),
        .q(add2_o_res)
    );
	 	 
	 
	 // 3 latency
    custom_ip_add adder_acumulate(
        .clk(clk),
		  .areset(rst),
        .a(out_reg),
        .b(accumulate_in),
        .q(out_reg_out)
    );

    assign out_node = out_reg_out;
	 assign out_valid = bias_add_clk[0];

endmodule
