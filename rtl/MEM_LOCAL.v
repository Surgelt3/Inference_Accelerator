
module MEM_LOCAL(
	input clk, rst,
	input write_en,
	input bias_add,
	input [8:0] address, param_loc, 
	input [31:0] write_data,
	output reg [255:0] read_data
);

	reg [31:0] mem [0:511];
	
	always @(posedge clk) begin
		if (rst)
			for (integer i = 0; i < 512; i++) begin
				mem[i] <= 32'd0;
			end
		else begin
			if (write_en)
				mem[address] <= write_data;
			else
				if (bias_add) begin
					read_data <= {
							 224{1'b0}, mem[param_loc+1]
						};
				end 
				else begin
					read_data <= {
							mem[param_loc], mem[address+3], mem[param_loc], mem[address+2], 
							mem[param_loc], mem[address+1], mem[param_loc], mem[address]
						};
				end
		end
	end
	

endmodule