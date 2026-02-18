
module MEM_LOCAL(
	input clk, rst,
	input write_en, read_en, 
	input bias_add, 
	input classifier_bit, 
	input [1:0] read_size, bias_loc, 
	input [8:0] write_address, 
	input [8:0] address, param_loc, 
	input [767:0] write_data,
	output reg classifier_bit_out,
	output reg read_valid_out, 
	output reg bias_add_out, 
	output reg [1:0] bias_loc_out, read_size_out, 
	output reg [767:0] read_data
);

	reg [31:0] mem [0:511];
	
	initial begin
		#20 $readmemh("C:/\Users/\lucas/\Desktop/\ELEC_49X/\Inference_Accelerator/\mem.hex", mem);
	end
	
	always @(posedge clk) begin
		read_valid_out <= 1'b0;
		if (rst) begin
		
		end	
		else begin
			bias_loc_out <= bias_loc;
			classifier_bit_out <= classifier_bit;
			bias_add_out <= bias_add;
			read_size_out <= read_size;
			if (write_en) begin
			  for (integer i = 0; i < 16; i = i + 1) begin
					mem[write_address + i] <= write_data[i*32 +: 32];
			  end
			end
			else if (read_en) begin
				read_valid_out <= 1'b1;
				case (read_size)
					2'b01: begin
								read_data <= {
										{512{1'b0}},
										mem[param_loc+3], mem[address+3], mem[param_loc+2], mem[address+2], 
										mem[param_loc+1], mem[address+1], mem[param_loc], mem[address]
									};
							end
					2'b10: begin
								read_data <= {
										{256{1'b0}},
										mem[param_loc+7], mem[address+7], mem[param_loc+6], mem[address+6], 
										mem[param_loc+5], mem[address+5], mem[param_loc+4], mem[address+4], 
										mem[param_loc+3], mem[address+3], mem[param_loc+2], mem[address+2], 
										mem[param_loc+1], mem[address+1], mem[param_loc], mem[address]
									};
							end
					2'b11: begin
								read_data <= {
										mem[param_loc+11], mem[address+11], mem[param_loc+10], mem[address+10], 
										mem[param_loc+9], mem[address+9], mem[param_loc+8], mem[address+8], 
										mem[param_loc+7], mem[address+7], mem[param_loc+6], mem[address+6], 
										mem[param_loc+5], mem[address+5], mem[param_loc+4], mem[address+4], 
										mem[param_loc+3], mem[address+3], mem[param_loc+2], mem[address+2], 
										mem[param_loc+1], mem[address+1], mem[param_loc], mem[address]
									};
							end
					default: begin
									read_data <= {768{1'b0}};
								end
				endcase
			end
		end
	end
	

endmodule