
module MEM_LOCAL(
	input clk, rst,
	input write_en,
	input [1:0] read_size, 
	input [8:0] address, param_loc, 
	input [31:0] write_data,
	output reg [767:0] read_data
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
				case (read_size)
					2'b01: begin
								read_data <= {
										{512{1'b0}},
										mem[param_loc+3], mem[address+3], mem[param_loc+2], mem[address+2], 
										mem[param_loc+1], mem[address+1], mem[param_loc], mem[address]
									};
							end
					2'b10: begin
								read_data <= {
										{256{1'b0}},
										mem[param_loc+7], mem[address+7], mem[param_loc+6], mem[address+6], 
										mem[param_loc+5], mem[address+5], mem[param_loc+4], mem[address+4], 
										mem[param_loc+3], mem[address+3], mem[param_loc+2], mem[address+2], 
										mem[param_loc+1], mem[address+1], mem[param_loc], mem[address]
									};
							end
					2'b11: begin
								read_data <= {
										mem[param_loc+11], mem[address+11], mem[param_loc+10], mem[address+10], 
										mem[param_loc+9], mem[address+9], mem[param_loc+8], mem[address+8], 
										mem[param_loc+7], mem[address+7], mem[param_loc+6], mem[address+6], 
										mem[param_loc+5], mem[address+5], mem[param_loc+4], mem[address+4], 
										mem[param_loc+3], mem[address+3], mem[param_loc+2], mem[address+2], 
										mem[param_loc+1], mem[address+1], mem[param_loc], mem[address]
									};
							end
					default: begin
									read_data <= {768{1'b0}};
								end
				endcase
		end
	end
	

endmodule