module INSTR_MEM(
    input [31:0] pc,
	 output instr_valid,
    output [31:0] instr
);
      
    reg [31:0] instructions[256:0];

    initial begin
        $readmemh("C:/Users/lucas/Desktop/ELEC_49X/Inference_Accelerator/rtl/instr.txt", instructions);
    end
	 
    assign read_data = instructions[pc];
	 assign instr_valid = 1'b1;
    
    always @(*) begin 
        $monitor("Instruction = %b", instr); 
    end


endmodule
