module INSTR_MEM(
    input [31:0] pc,
	 output reg instr_valid,
    output [31:0] instr
);
      
	reg [31:0] instructions[256:0];

	initial begin
		$readmemh("C:/Users/lucas/Desktop/ELEC_49X/Inference_Accelerator/rtl/instr.txt", instructions);
	end

	assign instr = instructions[pc];
 
	always @(*) begin
		if (instr == 32'd0) begin
			instr_valid = 1'b0;
		end
		else 
			instr_valid = 1'b1;
	end

	always @(*) begin 
		$monitor("Instruction = %b", instr); 
	end


endmodule
