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
	
	input instr_in_valid,
	input [31:0] in_instr,
	output in_instr_ready
	
);	

	wire in_valid, in_ready;
	wire [31:0] in_data;
	wire out_valid, out_ready;
	wire [63:0] out_data;
	
	assign readdata = out_data;
	//assign readdatavalid = out_valid;
	//assign waitrequest_read = !out_valid;
	
	
	//assign waitrequest_read = 1'b0;
		
	assign out_ready = read;
	
	assign in_data = writedata;
	assign in_valid = write;
	assign waitrequest_write = !in_ready;


	wire [31:0] pc;
	wire instr_valid;
	wire [31:0] instr;
	
	wire rst = reset_reset | (reset_write ? |reset : 1'b0);
	
	reg [31:0] tmp;
	
	/*
	
	always @(posedge clk) begin
		//waitrequest_read <= 1'b0;
		if (rst) begin
			waitrequest_read <= 1'b0;
			tmp <= 32'd0;
			readdata <= 64'd0;
		end else begin
			waitrequest_read <= 1'b1;
			readdata <= out_data;
		end
	end
	*/



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
		.clk(clk), .rst(rst),
		.instr_in_valid(instr_in_valid),
		.in_ready(in_instr_ready),
		.in_instr(in_instr),

		.pc(pc),
		.instr_valid(instr_valid),
		.instr(instr)
	);

	





endmodule
