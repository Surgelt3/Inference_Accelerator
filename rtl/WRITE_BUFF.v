module WRITE_BUFF(
	input clk, rst,
	input in_valid,
	input [31:0] in_data, 
	input [31:0] pc_in,
	output reg out_valid,
	output reg [63:0] out_data
);

	localparam [31:0] offset = 32'd120;

	always @(posedge clk) begin
		out_valid <= 1'b0;
		if (in_valid) begin
			out_data <= {in_data, pc_in + offset};
			out_valid <= 1'b1;
		end		
	end

endmodule
