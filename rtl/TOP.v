module TOP (
	input clk,
	input [31:0] reset,
	input reset_write, 
	input reset_reset,
	input [4:0] burstcount, 
	output waitrequest_write,
	input write,
	input [31:0] writedata, 

	output [63:0] readdata,
	input read,
	output waitrequest_read
);	

	wire in_valid, in_ready;
	wire [31:0] in_data;
	wire out_valid, out_ready;
	wire [63:0] out_data;
	
	assign readdata = out_data;
	assign waitrequest_read = !out_valid;
	assign out_ready = read;
	
	assign in_data = writedata;
	assign in_valid = write;
	assign waitrequest_write = !in_ready;


	wire [31:0] pc;
	wire instr_valid;
	wire [31:0] instr;
	
	wire rst = reset_reset | (reset_write ? |reset : rst);


	NPU npu(
		clk, rst, 
		instr,
		instr_valid, 
		
		in_valid,
		in_data, 
		in_ready,
		
		pc,
		out_ready,
		out_valid,
		out_data
	);
	
	
	INSTR_MEM instr_mem(
		 pc,
		 instr_valid,
		 instr
	);




endmodule
