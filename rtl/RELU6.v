module RELU6(
	input clk,
	input in_valid,
	input [31:0] in_data,
	output out_valid,
	output [31:0] out_data
);

	localparam [31:0] FP_ZERO = 32'h00000000;
	localparam [31:0] FP_SIX = 32'h40C00000;

	
	always @(posedge clk) begin
		if (in_valid) begin
			if (in_data > FP_SIX)
				out_data <= FP_SIX;
			end else if (in_data[31])
				out_data <= FP_ZERO;
			end else 
				out_data <= in_data;
			end
			out_valid <= 1'b1;
		end
		
	end


endmodule