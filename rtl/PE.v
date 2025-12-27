module PE(
    input clk, rst, i_vld, bias_add,  
    input [31:0] in0, in1, in2, in3, in4, in5, in6, in7, bias_val, 
    output [31:0] out_node
);

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
	 wire [31:0] accumulate_in;

    always @(posedge clk) begin
        if (rst) begin
            out_reg = 32'd0;
        end else begin
            out_reg = out_reg_out;
		  end
    end

    multiplier_32bit multiplier0(
        .clk(clk),
        .rst(rst),
        .i_a(in0),
        .i_b(in1),
        .i_vld(i_vld),
        .o_res(mul0_o_res),
        .o_res_vld(mul0_o_res_vld),
        .overflow(mul0_overflow)
    );

    multiplier_32bit multiplier1(
        .clk(clk),
        .rst(rst),
        .i_a(in2),
        .i_b(in3),
        .i_vld(i_vld),
        .o_res(mul1_o_res),
        .o_res_vld(mul1_o_res_vld),
        .overflow(mul1_overflow)
    );

    multiplier_32bit multiplier2(
        .clk(clk),
        .rst(rst),
        .i_a(in4),
        .i_b(in5),
        .i_vld(i_vld),
        .o_res(mul2_o_res),
        .o_res_vld(mul2_o_res_vld),
        .overflow(mul2_overflow)
    );

    multiplier_32bit multiplier3(
        .clk(clk),
        .rst(rst),
        .i_a(in6),
        .i_b(in7),
        .i_vld(i_vld),
        .o_res(mul3_o_res),
        .o_res_vld(mul3_o_res_vld),
        .overflow(mul3_overflow)
    );

    assign i_vld_adder0 = mul0_o_res_vld & mul1_o_res_vld;
    assign i_vld_adder1 = mul2_o_res_vld & mul3_o_res_vld;


    adder_32bit adder0(
        .clk(clk),
        .rst(rst),
        .i_a(mul0_o_res),
        .i_b(mul1_o_res),
        .i_vld(i_vld_adder0),
        .o_res(add0_o_res),
        .o_res_vld(add0_o_res_vld),
        .overflow(add0_overflow)
    );

    adder_32bit adder1(
        .clk(clk),
        .rst(rst),
        .i_a(mul2_o_res),
        .i_b(mul3_o_res),
        .i_vld(i_vld_adder1),
        .o_res(add1_o_res),
        .o_res_vld(add1_o_res_vld),
        .overflow(add1_overflow)
    );

    adder_32bit adder2(
        .clk(clk),
        .rst(rst),
        .i_a(add0_o_res),
        .i_b(add1_o_res),
        .i_vld(add1_o_res_vld),
        .o_res(add2_o_res),
        .o_res_vld(add2_o_res_vld),
        .overflow(add2_overflow)
    );
	 
	 assign accumulate_in = bias_add ? bias_val : add2_o_res;
	 assign acc_vld = bias_add ? bias_add : add2_o_res_vld;

    adder_32bit adder_acumulate(
        .clk(clk),
        .rst(rst),
        .i_a(out_reg),
        .i_b(accumulate_in),
        .i_vld(acc_vld),
        .o_res(out_reg_out),
        .o_res_vld(add_acc_o_res_vld),
        .overflow(add_acc_overflow)
    );

    assign out_node = out_reg_out;

endmodule