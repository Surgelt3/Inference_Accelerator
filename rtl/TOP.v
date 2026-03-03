module TOP (
	input clk, rst,
	input load_data_in_valid,
	input [31:0] load_data_in_data, 
	output load_data_in_ready,
	output out_valid,
	output [63:0] out_data
	
);
	

	wire [31:0] pc;
	wire instr_valid;
	wire [31:0] instr;

	NPU npu(
		clk, rst, 
		instr,
		instr_valid, 
		
		load_data_in_valid,
		load_data_in_data, 
		load_data_in_ready,
		
		pc,
		
		out_valid,
		out_data
	);
	
	
	INSTR_MEM instr_mem(
		 pc,
		 instr_valid,
		 instr
	);




endmodule
