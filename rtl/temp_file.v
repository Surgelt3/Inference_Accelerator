module CONTROL_UNIT(
	input clk, rst, 
	input opcode,
	input classifier_bit,
	input [6:0] address_row0, address_row1, address_row2, 
	input [5:0] param_loc, 
	input [3:0] length, 
	output reg [1:0] pe_fifo_bias_loc,
	output reg [1:0] read_size, 
	output reg [2:0] bias_add, 
	input [5:0] param_loc, 
	input [6:0] mwmaddress_row0, address_row1, address_row2, 
	output reg curr_classifier_bit, swap_mem_signal_out
	output reg [1:0] pe_busy,
	output run_end
);
	
	
		
	reg [8:0] pe0_address, pe1_address;
	reg [8:0] pe0_param_loc, pe1_param_loc;
	reg [8:0] pe0_length, pe1_length;
	reg [8:0] cur_length;
	
	reg curr_classifier_bit;
	wire [8:0] curr_adress, curr_param_loc;
	reg mem_calc_ready; 
	
	assign curr_adress = curr_classifier_bit ? pe1_address : pe0_address;
	assign curr_param_loc = curr_classifier_bit ? pe1_param_loc : pe0_param_loc;
	assign curr_length = curr_classifier_bit ? pe1_length : pe0_length;

	always @(posedge clk) begin
	
		swap_mem_signal_out <= 1'b0;
		if (rst) begin
			pe0_fifo_out_ready <= 1'b0;
			pe1_fifo_out_ready <= 1'b0;
			pe0_bias_add <= 1'b0;
			pe1_bias_add <= 1'b0;
			run_end <= 1'b0;
			pe_busy <= 2'b00;
			
			curr_classifier_bit <= 1'b0;
		end
		else if (opcode == 3'b000) begin
			// MAC Operation
			mem_local_address <= mem_local_address_next;
			mem_local_param_loc <= mem_local_param_loc_next;
			pe0_length <= pe0_length_next;
			pe1_length <= pe1_length_next;
			read_size <= read_size_next;
			bias_add <= bias_add_next;
			pe_fifo_bias_loc <= pe_fifo_bias_loc_next;
			pe_busy <= pe_busy_next;
			curr_classifier_bit <= curr_classifier_bit_next;
		end
		else if (opcode == 3'b001) begin
			// Switch Local Mem
			swap_mem_signal_out <= 1'b1;
		end
		else if (opcode == 3'b011) begin
			// END Operation
			if (pe_busy == 2'b00)
				run_end <= 1'b1;
		end

	end
	
	always @(*) begin
		if (pe_busy[classifier_bit] == 0) begin
			pe_busy_next[classifier_bit] = 1'b1;
			if (curr_classifier_bit) begin
				pe1_address = address;
				pe1_param_loc = param_loc;
				pe1_length = length;
			end 
			else begin
				pe0_address = address;
				pe0_param_loc = param_loc;
				pe0_length = length;
			end
			curr_classifier_bit_next = classifier_bit;
		end
		else begin
			curr_classifier_bit_next = !curr_classifier_bit;
		end
	
		if (curr_length > 8'd8) begin
			read_size_next = 2'b11;
		end
		else if (curr_length > 8'd4) begin
			read_size_next = 2'b10;
		end 
		else begin
			read_size_next = 2'b01;
		end
		
		mem_local_address_next = curr_adress + (read_size_next * 4);
		mem_local_param_loc_next = curr_adress + (read_size_next * 4);
		
		pe_length = curr_length - (read_size_next * 4);
		if (curr_classifier_bit) begin
			pe1_length_next = pe_length;
		end 
		else begin
			pe0_length_next = pe_length;
		end
		
		if (pe_length < 'd0) begin
		// -1 is index 3, -2 is index 2, -3 is index 1, -4 is index 0
			pe_fifo_bias_loc_next = pe_length + 'd4;
			bias_add_next = 1'b1;
			pe_busy_next[curr_classifier_bit] = 1'b0;
		end
		else begin
			bias_add_next = 1'b0;
		end
		
		
		
	end


endmodule