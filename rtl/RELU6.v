module RELU6(
	input clk,
	input in_valid,
	input use_relu,
	input [31:0] pc_in,
	input [31:0] in_data,
	output reg out_valid,
	output reg [31:0] pc_out,
	output reg [31:0] out_data
);

	localparam [31:0] FP_ZERO = 32'h00000000;
	localparam [31:0] FP_SIX = 32'h40C00000;

	
	always @(posedge clk) begin
		out_valid <= 1'b0;
		if (in_valid) begin
			pc_out <= pc_in;
			if (use_relu) begin
				if (in_data > FP_SIX) begin
					out_data <= FP_SIX;
				end else if (in_data[31]) begin
					out_data <= FP_ZERO;
				end else begin
					out_data <= in_data;
				end
			end
			else begin
				out_data <= in_data;
			end
			out_valid <= 1'b1;
		end
		
	end

endmodule
