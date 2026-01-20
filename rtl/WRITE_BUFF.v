
module WRITE_BUFF(
	input clk, rst,
	input in_valid,
	input [31:0] in, 
	output reg out_valid,
	output reg [255:0] out_data
);

	reg [3:0] ptr;
	reg [255:0] data;

	always @(posedge clk) begin
		if (rst) begin
			out_data <= 256'd0;
			data <= 256'd0;
			ptr <= 4'd0;
			out_valid <= 1'b0;
		end else begin
			out_valid <= 1'b0;
			if (in_valid) begin
				if (ptr == 4'd7) begin
					out_valid <= 1'b1;
					out_data <= {in, data[255:32]};
					ptr <= 4'd0;
				end
				else begin
					ptr <= ptr + 4'd1;
				end
				data <= {in, data[255:32]};
			end
		end
		
	end
		
// First store value then 
		
		
endmodule