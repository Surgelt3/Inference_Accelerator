module CONTROL_UNIT(
	input clk, rst, 
	input instr_valid,
	input [2:0] opcode,
	input classifier_bit,
	input [8:0] address, param_loc, 
	input [9:0] length,
	input [31:0] pc_in,
	input [255:0] load_data,
	output reg [1:0] pe_fifo_bias_loc,
	output reg [1:0] read_size, 
	output reg bias_add, 
	output reg [8:0] mem_local_address, mem_local_param_loc, 
	output reg curr_classifier_bit, 
	output reg load_signal, pool_signal, mac_signal,
	output reg [1:0] pe_busy,
	output reg [31:0] pc_clock,
	output reg [255:0] write_data
);
	
	reg [1:0] pe_busy_next;
	reg [8:0] pe0_address, pe0_address_next, pe1_address, pe1_address_next;
	reg [8:0] pe0_param_loc, pe0_param_loc_next, pe1_param_loc, pe1_param_loc_next;
	reg [9:0] pe0_length, pe0_length_next, pe1_length, pe1_length_next;
	reg [9:0] pe_length_next;
	reg [9:0] curr_length, curr_length_next;
	reg [8:0] curr_address_next, curr_param_loc_next;

	reg [1:0] pe_fifo_bias_loc_next;
	reg bias_add_next;
	reg [1:0] read_size_next;
	reg [8:0] mem_local_param_loc_next, mem_local_address_next;

	reg curr_classifier_bit_next;


	wire [8:0] curr_adress, curr_param_loc;
	reg mem_calc_ready; 
	

	always @(posedge clk) begin
		
		pc_clock <= pc_in;
		write_data <= load_data;
		load_signal <= 1'b0;
		pool_signal <= 1'b0;
		mac_signal <= 1'b0;
		
		mem_local_address <= mem_local_address_next;
		mem_local_param_loc <= mem_local_param_loc_next;
		read_size <= read_size_next;
		bias_add <= bias_add_next;
		pe_fifo_bias_loc <= pe_fifo_bias_loc_next;
		pe_busy <= pe_busy_next;
		curr_classifier_bit <= curr_classifier_bit_next;
		
		pe1_length <= pe1_length_next;
		pe1_address <= pe1_address_next;
		pe1_param_loc <= pe1_param_loc_next;
		pe0_length <= pe0_length_next;
		pe0_address <= pe0_address_next;
		pe0_param_loc <= pe0_param_loc_next;

		if (rst) begin
			pe_busy <= 2'b00;
			curr_classifier_bit <= 1'b0;
			read_size <= 2'b00;
		end
		else if (instr_valid) begin
			if (opcode == 3'b000) begin
				// MAC Operation
				mac_signal <= 1'b1;
				mem_local_address <= mem_local_address_next;
				mem_local_param_loc <= mem_local_param_loc_next;
				read_size <= read_size_next;
				bias_add <= bias_add_next;
				pe_fifo_bias_loc <= pe_fifo_bias_loc_next;
				pe_busy <= pe_busy_next;
				curr_classifier_bit <= curr_classifier_bit_next;
				
				pe1_length <= pe1_length_next;
				pe1_address <= pe1_address_next;
				pe1_param_loc <= pe1_param_loc_next;
				pe0_length <= pe0_length_next;
				pe0_address <= pe0_address_next;
				pe0_param_loc <= pe0_param_loc_next;


				end
			else if (opcode == 3'b001) begin
				// LOAD
				load_signal <= 1'b1;
			end
			else if (opcode == 3'b010) begin
				// RELU Operation
				//relu_signal <= 1'b1;
			end
			else if (opcode == 3'b011) begin
				// POOL Operation
				pool_signal <= 1'b1;
			end
		end

	end
	
	always @(*) begin
		
		if (pe_busy[classifier_bit] == 0 && instr_valid && opcode == 3'b000) begin
			curr_length_next = length;
			curr_address_next = address;
			curr_param_loc_next = param_loc;
			curr_classifier_bit_next = classifier_bit;
			pe_busy_next = pe_busy | (2'b01 << classifier_bit); 
		end
		else if (|pe_busy) begin
			if (instr_valid) begin
				curr_classifier_bit_next = !curr_classifier_bit;
			end 
			else begin
				curr_classifier_bit_next = pe_busy[curr_classifier_bit] ? curr_classifier_bit : !curr_classifier_bit;
			end
			
			if (curr_classifier_bit_next) begin
				curr_length_next = pe1_length;
				curr_address_next = pe1_address;
				curr_param_loc_next = pe1_param_loc;
			end
			else begin
				curr_length_next = pe0_length;
				curr_address_next = pe0_address;
				curr_param_loc_next = pe0_param_loc;
			end
		end
		else begin
			mem_local_address_next = address;
			mem_local_param_loc_next = param_loc;
			read_size_next = read_size;
			bias_add_next = 1'b0;
			pe_fifo_bias_loc_next = pe_fifo_bias_loc;
			pe_busy_next = pe_busy;
			curr_classifier_bit_next = curr_classifier_bit;
			pe1_length_next = pe1_length;
			pe1_address_next = pe1_address;
			pe1_param_loc_next = pe1_param_loc;
			pe0_length_next = pe0_length;
			pe0_address_next = pe0_address;
			pe0_param_loc_next = pe0_param_loc;
		end
		
		if (pe_busy_next == 2'b00) begin
			read_size_next = 2'b00;
		end
		else begin
			if (curr_length_next > 10'd4) begin
				read_size_next = 2'b10;
			end 
			else begin
				read_size_next = 2'b01;
			end
			
			mem_local_address_next = curr_address_next;
			mem_local_param_loc_next = curr_param_loc_next;
			pe_length_next = curr_length_next;
			if (curr_length_next < 10'd8) begin
			// -1 is index 3, -2 is index 2, -3 is index 1, -4 is index 0
				pe_fifo_bias_loc_next = 3'd4 - ((read_size_next * 4)-curr_length_next);
				
				bias_add_next = 1'b1;
				pe_busy_next[curr_classifier_bit_next] = 1'b0;
				if (curr_classifier_bit_next) begin
					pe1_length_next = curr_length_next;
					pe1_address_next = curr_address_next;
					pe1_param_loc_next = curr_param_loc_next;
					pe0_length_next = pe0_length;
					pe0_address_next = pe0_address;
					pe0_param_loc_next = pe0_param_loc;
				end 
				else begin		
					pe0_length_next = curr_length_next;
					pe0_address_next = curr_address_next;
					pe0_param_loc_next = curr_param_loc_next;
					pe1_length_next = pe1_length;
					pe1_address_next = pe1_address;
					pe1_param_loc_next = pe1_param_loc;
				end

			end
			else begin
				pe_fifo_bias_loc_next = pe_fifo_bias_loc;
				if (curr_classifier_bit_next) begin
					pe1_length_next = curr_length_next - (read_size_next * 4);
					pe1_address_next = curr_address_next + 1;
					pe1_param_loc_next = curr_param_loc_next + 1;
					pe0_length_next = pe0_length;
					pe0_address_next = pe0_address;
					pe0_param_loc_next = pe0_param_loc;
				end 
				else begin		
					pe0_length_next = curr_length_next - (read_size_next * 4);
					pe0_address_next = curr_address_next + 1;
					pe0_param_loc_next = curr_param_loc_next + 1;
					pe1_length_next = pe1_length;
					pe1_address_next = pe1_address;
					pe1_param_loc_next = pe1_param_loc;
				end

			end

		end
		
		
		
		
		
	end


endmodule
