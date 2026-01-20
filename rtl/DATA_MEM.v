module DATA_MEM(
	input clk,
	input read, write,
	input [8:0] addr,
	input [31:0] data_in,
	output reg [31:0] data_out
);

	always @(posedge clk) begin
		if(read)
			data_out <= memory[addr];
		else if (write)
			memory[addr] <= data_in;
	end


endmodule