module NPU_TB(

);


	
	
	reg clk, rst;
	reg [31:0] instr;
	wire npu_instr_ready;
	reg instr_valid;
	
	reg load_data_in_valid;
	reg [8:0] load_data_in_address;
	reg [31:0] load_data_in_data;
	wire load_data_in_ready;
	
	wire out_valid;
	wire [31:0] out_data;
		
		
	NPU DUT(
		clk, rst, 
		instr,
		instr_valid, 
		npu_instr_ready, 
		load_data_in_valid,
		load_data_in_address,
		load_data_in_data, 
		load_data_in_ready,
		out_valid,
		out_data
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
					instr_valid <= 1'b0;
					load_data_in_valid <= 1'b0; 
					load_data_in_address <= 1'b0;
					load_data_in_data <= 32'b0;
					rst <= 1;
					#20 rst <= 0;
					instr <= 32'b00000000000000000010010011011110;
					instr_valid <= 1'b1;
					#20;
					instr <= 32'b00000000000000000010010011011110;
					instr_valid <= 1'b1;
					#20; 
					instr <= 32'b00000000000000000010010011011110;
					instr_valid <= 1'b1;
					#20;
					instr <= 32'b00000000000000000010010011011110;
					instr_valid <= 1'b1;
					instr_valid <= 1'b0;
					
			end
			Test: begin
				
			end
			
			Done: begin
				
			end
		endcase
	end



endmodule