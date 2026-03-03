module INSTR_MEM(
    input [31:0] pc,
    output [31:0] instr
);
      
    reg [31:0] instructions[31:0];

    initial begin
        $readmemh("C:/\Users/\lucas/\Desktop/\ELEC_49X/\Inference_Accelerator/\instr.txt", instructions);
    end

    assign read_data = instructions[pc];
    
    always @(*) begin 
        $monitor("Instruction = %b", instr); 
    end


endmodule