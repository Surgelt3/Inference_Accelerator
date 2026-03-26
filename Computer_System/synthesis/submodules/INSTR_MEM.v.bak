module INSTR_MEM(
	 input clk, rst,
	 input instr_in_valid,
	 output in_ready,
	 input [31:0] in_instr,

    input [31:0] pc,
	 output reg instr_valid,
    output [31:0] instr
);
      
	reg [31:0] instructions_preload[131071:0];
	reg [31:0] instructions_temp[2047:0];
	
	reg [31:0] instr_counter;
	wire [10:0] addr_read, addr_write;
	wire put_instr;

	initial begin
		$readmemh("C:/Users/lucas/Desktop/ELEC_49X/Inference_Accelerator/rtl/instr.txt", instructions_preload);
	end

	assign addr_read = pc[10:0];
	assign addr_write = instr_counter[10:0];
	assign instr = (pc > 32'd131071) ? instructions_temp[addr_read] : instructions_preload[pc[16:0]];
	assign in_ready = pc[10:0] >= instr_counter[10:0];
	assign put_instr = in_ready && instr_in_valid;
	
	always @(posedge clk) begin
		if (rst) begin
			instr_counter <= 32'd131072;
		end 
		else if (put_instr) begin
			instructions_temp[addr_write] <= in_instr;
			instr_counter <= instr_counter + 1;
		end
	end
 
	always @(*) begin
		if (instr[31:29] == 3'b011) begin
			instr_valid = 1'b0;
		end
		else 
			instr_valid = 1'b1;
	end

	always @(*) begin 
		$monitor("Instruction = %b", instr); 
	end


endmodule
