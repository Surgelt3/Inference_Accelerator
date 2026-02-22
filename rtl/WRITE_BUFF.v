
module WRITE_BUFF(
	input clk, rst,
	input out_ready,
	input in_valid,
	input [31:0] in_data, 
	input [8:0] in_address,
	output in_ready,
	output reg out_valid,
	output reg [8:0] out_address,
	output reg [255:0] out_data
);


	reg [3:0] ptr;
	reg [255:0] data;

	always @(posedge clk) begin
		if (rst) begin
			data <= 256'd0;
			ptr <= 4'd0;
			out_valid <= 1'b0;
		end else begin
			out_valid <= 1'b0;
			if (in_valid) begin
				if (ptr == 4'd7) begin
					out_valid <= 1'b1;
					out_data <= {in_data, data[255:32]};
					out_address <= in_address;
					ptr <= 4'd0;
				end
				else begin
					ptr <= ptr + 4'd1;
				end
				data <= {in_data, data[255:32]};
			end
		end
		
	end
		
// First store value then 
		
		
endmodule
