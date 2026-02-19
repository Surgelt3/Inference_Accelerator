module CONTROL_UNIT(
	input clk, rst, 
	input [2:0] opcode,
	input classifier_bit,
	input [8:0] address, param_loc, 
	input [9:0] length,
	output reg [1:0] pe_fifo_bias_loc,
	output reg [1:0] read_size, 
	output reg [2:0] bias_add, 
	output reg [8:0] mem_local_address, mem_local_param_loc, 
	output reg curr_classifier_bit, 
	output reg load_signal, relu_signal, pool_signal, mac_signal,
	output reg [1:0] pe_busy
);
	
	reg [1:0] pe_busy_next;
	reg [8:0] pe0_address, pe1_address;
	reg [8:0] pe0_param_loc, pe1_param_loc;
	reg [9:0] pe0_length, pe0_length_next, pe1_length, pe1_length_next;
	reg [9:0] pe_length;
	reg [9:0] cur_length;

	reg [1:0] pe_fifo_bias_loc_next;
	reg [2:0] bias_add_next;
	reg [1:0] read_size_next;
	reg [8:0] mem_local_param_loc_next, mem_local_address_next;

	reg curr_classifier_bit_next;


	wire [8:0] curr_adress, curr_param_loc;
	reg mem_calc_ready; 
	
	assign curr_adress = curr_classifier_bit ? pe1_address : pe0_address;
	assign curr_param_loc = curr_classifier_bit ? pe1_param_loc : pe0_param_loc;
	assign curr_length = curr_classifier_bit ? pe1_length : pe0_length;

	always @(posedge clk) begin
	
		load_signal <= 1'b0;
		relu_signal <= 1'b0;
		pool_signal <= 1'b0;
		mac_signal <= 1'b0;
		if (rst) begin
			pe_busy <= 2'b00;
			
			curr_classifier_bit <= 1'b0;
		end
		else if (opcode == 3'b000) begin
			// MAC Operation
			mac_signal <= 1'b1;
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
			// LOAD
			load_signal <= 1'b1;
		end
		else if (opcode == 3'b010) begin
			// RELU Operation
			relu_signal <= 1'b1;
		end
		else if (opcode == 3'b011) begin
			// POOL Operation
			pool_signal <= 1'b1;
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
