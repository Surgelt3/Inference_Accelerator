module INSTR_DECODER(
	input clk, rst, 
	input [2:0] opcode,
	input [8:0] start_loc, length, param_loc,
	input [1:0] pe_busy, 
	input [31:0] pc_in,
	output [2:0] opcode_out,
	output classifier_bit_out,
	output [3:0] op_len,
	output [31:0] pc_out
	
);

	always @(posedge clk) begin
		if (rst) begin
			pc_out <= 32'd0;
			opcode_out <= 3'd0;
			classifier_bit_out <= 1'b0;
			op_len <= 4'd0;
		end
		if (!pe_busy[0] | !pe_busy[1]) begin
			opcode_out <= opcode;
			if (opcode == 3'b000) begin
				// MAC Operation
				if (!pe_busy[0]) begin
					classifier_bit_out <= 0;
				else if (!pe_busy[1]) begin
					classifier_bit_out <= 1;
				end 
				pc_out <= pc_in + 1;
			end
			else if (opcode == 3'b001) begin
				// LOAD Operation
				pc_out <= pc_in + 1;
			end
			else if (opcode == 3'b010) begin
				
				pc_out <= pc_in + 1;
			end
			else if (opcode == 3'b011) begin
				// END Operation
			end
		end
	
	end


endmodule