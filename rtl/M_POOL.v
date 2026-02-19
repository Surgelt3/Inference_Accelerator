module M_POOL #(parameter integer NUM_INPUTS = 49)(
	input clk, reset, 
	input i_vld,
	input [31:0] a,
	output o_res_vld,
	output [31:0] c
);

	reg [31:0] count;
	reg final, busy;
	reg [31:0] b, out_add;
	reg valid_out_add;
	reg o_res_vld_add, overflow_add;
	reg overflow_div;
	wire i_vld_add;
	
	assign i_vld_add = i_vld && !busy;
	
	always @(posedge clk) begin
		if (reset) begin
			count <= 32'd0;
			b <= 32'b0;
			final <= 1'b0;
			busy <= 1'b0;
		end else begin
			if (i_vld_add) begin
				busy <= 1'b1;
				final <= (count == NUM_INPUTS-1);
			end
			if (o_res_vld_add) begin
				b <= out_add;
				count <= final ? 32'd0 : (count + 32'd1);
				busy <= 1'b0;
			end
		end
	end
	
	adder_32bit GAP_ADDER(
		clk,
		reset,
		a, b,
		i_vld_add,
		out_add,
		o_res_vld_add,
		overflow_add
	)
	
	assign valid_out_add = (final && o_res_vld_add);
	
	multiplier_32bit GAP_DIV(
		clk,
		reset,
		out_add,
		32'h3ca72f05,
		valid_out_add,
		c,
		o_res_vld,
		overflow_div 
	);
	
	

endmodule
