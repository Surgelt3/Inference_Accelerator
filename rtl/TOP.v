module TOP(
	input clk, rst,
);

	wire [31:0] pc;
	wire [31:0] instr;
	
	wire read_en, write_en;
	wire [8:0] addr;
	wire [31:0] data_write, data_read;

	INSTR_MEM instr_mem(
		 .pc(pc),
		 .instr(instr),
	);
	
	DATA_MEM data_mem(
		.clk(clk),
		.read(read_en), .write(write_en),
		.addr(addr),
		.data_in(data_write),
		.data_out(data_read)
	);
	
	




endmodule