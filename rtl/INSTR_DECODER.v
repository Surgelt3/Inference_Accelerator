module INSTR_DECODER(
	input clk, rst, 
	input [2:0] opcode,
	input [8:0] start_loc, param_loc, 
	input [9:0] length,
	input [1:0] pe_busy, 
	input [31:0] pc_in,
	output reg classifier_bit_out,
	output reg [2:0] opcode_out,
	output reg [8:0] start_loc_out, param_loc_out, 
	output reg [9:0] length_out,
	output reg [31:0] pc_out
);

	always @(posedge clk) begin
		if (rst) begin
			pc_out <= 32'd0;
			classifier_bit_out <= 1'b0;
		end
		opcode_out <= opcode;
		start_loc_out <= start_loc;
		param_loc_out <= param_loc;
		length_out <= length;
		if (opcode == 3'b000) begin
			// MAC Operation
			if (!pe_busy[0] | !pe_busy[1]) begin
				if (!pe_busy[0]) begin
					classifier_bit_out <= 0;
				end 
				else if (!pe_busy[1]) begin
					classifier_bit_out <= 1;
				end 
				pc_out <= pc_in + 1;
			end
		end
		else if (opcode == 3'b001) begin
			// LOAD Operation
			opcode_out <= opcode;
			start_loc_out <= start_loc;
			param_loc_out <= param_loc;
			length_out <= length;
			pc_out <= pc_in + 1;
		end
		else if (opcode == 3'b010) begin
			// RELU Operation
			opcode_out <= opcode;
			start_loc_out <= start_loc;
			param_loc_out <= param_loc;
			length_out <= length;
			pc_out <= pc_in + 1;
		end
		else if (opcode == 3'b011) begin
			// POOL Operation
			opcode_out <= opcode;
			start_loc_out <= start_loc;
			param_loc_out <= param_loc;
			length_out <= length;
			pc_out <= pc_in + 1;
		end

	end


endmodule
