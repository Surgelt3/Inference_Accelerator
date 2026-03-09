
module READ_BUFF(
	input clk, rst,
	input out_ready,
	input in_valid,
	input [31:0] in_data, 
	output in_ready,
	output reg out_valid,
	output reg [255:0] out_data
);


	reg [2:0] ptr;
	reg [255:0] data;
	
	wire allow_in;
	wire allow_out;
	
	assign in_ready = !(out_valid && (ptr == 3'd7));
	assign allow_in = in_valid && in_ready;
	assign allow_out = out_valid && out_ready;
	//assign out_valid = allow_out ? 1'b0 : ((allow_in && ptr == 3'd7) ? 1'b1 : 1'b0 ); 

	always @(posedge clk) begin
		if (rst) begin
			data <= 256'd0;
			ptr <= 4'd0;
			out_valid <= 1'b0;
			out_data <= 256'd0;
		end else begin
		
			if (allow_out) begin
				out_valid <= 1'b0;
			end
			
			if (allow_in) begin
				data <= {in_data, data[255:32]};
				if (ptr == 3'd7) begin
					out_data <= {in_data, data[255:32]};
					out_valid <= 1'b1;
					ptr <= 3'd0;
				end 
				else begin
					ptr <= ptr + 3'd1;
				end
			end
			
		end
		
	end
				
		
endmodule
