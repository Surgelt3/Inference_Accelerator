module PE(
    input clk, rst, i_vld, bias_add,
	 input [1:0] bias_loc, 
    input [31:0] in0, in1, in2, in3, in4, in5, in6, in7, 
	 output out_valid,
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
	 wire [31:0] accumulate_in;
	 reg [2:0] bias_add_clk;

    always @(posedge clk) begin
        if (rst) begin
            out_reg <= 32'd0;
				bias_add_clk <= 3'b000;
        end else begin
            out_reg <= out_reg_out;
		  end
		  bias_add_clk = {bias_add, bias_add_clk[2], bias_add_clk[1]};
    end
	 
	 assign mul0_ia = (bias_add && (bias_loc == 2'b00)) ? in0: 1'b1;
	 assign mul0_ib = in1;
	 assign mul1_ia = (bias_add && (bias_loc == 2'b01)) ? in2: 1'b1;
	 assign mul1_ib = in3;
	 assign mul2_ia = (bias_add && (bias_loc == 2'b10)) ? in4: 1'b1;
	 assign mul2_ib = in5;
	 assign mul3_ia = (bias_add && (bias_loc == 2'b11)) ? in6: 1'b1;
	 assign mul3_ib = in7;


    multiplier_32bit multiplier0(
        .clk(clk),
        .rst(rst),
        .i_a(mul0_ia),
        .i_b(mul0_ib),
        .i_vld(i_vld),
        .o_res(mul0_o_res),
        .o_res_vld(mul0_o_res_vld),
        .overflow(mul0_overflow)
    );

    multiplier_32bit multiplier1(
        .clk(clk),
        .rst(rst),
        .i_a(mul1_ia),
        .i_b(mul1_ib),
        .i_vld(i_vld),
        .o_res(mul1_o_res),
        .o_res_vld(mul1_o_res_vld),
        .overflow(mul1_overflow)
    );

    multiplier_32bit multiplier2(
        .clk(clk),
        .rst(rst),
        .i_a(mul2_ia),
        .i_b(mul2_ib),
        .i_vld(i_vld),
        .o_res(mul2_o_res),
        .o_res_vld(mul2_o_res_vld),
        .overflow(mul2_overflow)
    );

    multiplier_32bit multiplier3(
        .clk(clk),
        .rst(rst),
        .i_a(mul3_ia),
        .i_b(mul3_ib),
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
	 
	 assign accumulate_in = add2_o_res;
	 assign acc_vld = add2_o_res_vld;

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
	 assign out_valid = bias_add_clk[0] && add_acc_o_res_vld;

endmodule