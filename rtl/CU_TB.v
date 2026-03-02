module CU_TB();
	
	
	reg clk, rst;
	reg [31:0] instr;
	reg classifier_bit;

	wire [1:0] pe_fifo_bias_loc, read_size, pe_busy;
	wire bias_add, curr_classifier_bit, load_signal, relu_signal, pool_signal, mac_signal;
	wire [8:0] mem_local_address, mem_local_param_loc;
	reg instr_valid;

		
	 CONTROL_UNIT DUT(
			clk, rst, 
			instr_valid,
			instr[31:29],
			classifier_bit,
			instr[28:20], instr[9:1],
			instr[19:10],
			pe_fifo_bias_loc,
			read_size, 
			bias_add, 
			mem_local_address, mem_local_param_loc, 
			curr_classifier_bit, 
			load_signal, relu_signal, pool_signal, mac_signal,
			pe_busy
		);

	parameter Default = 5'b00000, Init = 5'b00001, Test = 5'b00010, Done = 5'b01111;
	reg [4:0] Present_state = Default;
	
	initial begin
			clk = 0;
			forever #10 clk = ~ clk;
	end	
	
	always @(posedge clk) 
	begin
		case (Present_state)
			Default : Present_state = Init;
			Init : Present_state = Test;
			Test : Present_state = Test;
			Done: Present_state = Done;
		endcase
	end
	
	// instr breakdown
	// opcode (3 bits): 31-29
	// start_loc (9 bits): 28-20
	// length (10 bits): 19-10
	// param_loc (9 bits): 9-1
	// unused: 0
	
	// MAC opcode: 3'b000
	// LOAD opcode: 3'b001
	// START opcode: 3'b010
	// END opcode: 3'b011
	
		
	always @(posedge clk) begin
		case (Present_state)
			Init: begin
					rst <= 1;
					instr_valid <= 1'b0;
					#20 rst <= 0;
					classifier_bit <= 1'b0;
					instr_valid <= 1'b1;
					instr <= 32'b00000000000000000010010011011110;
					#20;
					classifier_bit <= 1'b1;
					instr <= 32'b00000000000000000010010011011110;
					#20; 
					classifier_bit <= 1'b0;
					instr <= 32'b00000000000000000010010011011110;
					#20;
					classifier_bit <= 1'b1;
					instr <= 32'b00000000000000000010010011011110;
					#20;
					instr_valid <= 1'b0;

			end
			Test: begin
				
			end
			
			Done: begin
				
			end
		endcase
	end




endmodule
