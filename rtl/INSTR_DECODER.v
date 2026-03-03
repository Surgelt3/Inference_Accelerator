module INSTR_DECODER(
	input clk, rst, 
	input instr_valid,
	input [2:0] opcode,
	input [8:0] start_loc, param_loc, 
	input [9:0] length,
	input [1:0] pe_busy, 
	input [31:0] pc_in,
	output reg classifier_bit_out,
	output reg instr_valid_out, 
	output reg relu_signal, 
	output reg [2:0] opcode_out,
	output reg [8:0] start_loc_out, param_loc_out, 
	output reg [9:0] length_out,
	output reg [31:0] pc_clock,
	output reg [31:0] pc_out
);

	always @(posedge clk) begin
		relu_signal <= 1'b0;
		pc_clock <= pc_in;
		instr_valid_out <= instr_valid;
		opcode_out <= opcode;
		start_loc_out <= start_loc;
		param_loc_out <= param_loc;
		length_out <= length;

		if (rst) begin
			pc_out <= 32'd0;
			classifier_bit_out <= 1'b0;
		end else if (instr_valid) begin
			if (opcode == 3'b000) begin
				// MAC Operation
				case (pe_busy) 
					2'b00: begin
								classifier_bit_out <= !classifier_bit_out;
								pc_out <= pc_in + 1;
							end
					2'b01: begin
								classifier_bit_out <= 1;
								pc_out <= pc_in + 1;
							end
					2'b10: begin
								classifier_bit_out <= 0;
								pc_out <= pc_in + 1;
							end
					default: begin
									pc_out <= pc_in;
									instr_valid_out <= 1'b0;
							end
				endcase
				
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
				relu_signal <= 1'b1;
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
		else begin
			classifier_bit_out <= 1'b0;
			opcode_out <= opcode;
			start_loc_out <= start_loc;
			param_loc_out <= param_loc;
			length_out <= length;
			pc_out <= pc_in;

		end

	end


endmodule
