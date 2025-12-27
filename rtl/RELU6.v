module RELU6(
	input clk,
	input [31:0] in_val,
	output [31:0] out_val,
);

	localparam [31:0] FP_ZERO = 32'h00000000;
	localparam [31:0] FP_SIX = 32'h40C00000;

	
	always @(posedge clk) begin
		if (in_val > FP_SIX)
			out_val <= FP_SIX;
		end else if (in_val[31])
			out_val <= FP_ZERO;
		end else 
			out_val <= in_val;
		end
		
	end


endmodule