module PE_tb();
	
	
	reg clk, rst;
	reg i_vld, bias_add;
	reg [31:0] instr;
	reg [1:0] bias_loc;
	reg [31:0] in0, in1, in2, in3, in4, in5, in6, in7;

	wire out_valid;
	wire [31:0] out_node;
	
		
	 PE DUT(
		 clk, rst, i_vld, bias_add,
		 bias_loc, 
		 in0, in1, in2, in3, in4, in5, in6, in7, 
		 out_valid,
		 out_node
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
					#20 rst <= 0;
					in0 <= 32'h00000000; in1 <= 32'h3AF8AE7D; in2 <= 32'h3E38B8AE; in3 <= 32'hBA2300D4; in4 <= 32'h3E38B8AE; in5 <= 32'h3B0BC921;
					in6 <= 32'h00000000; in7 <= 32'h3B9A904F;
					i_vld <= 1'b1; bias_add <= 1'b0; bias_loc <= 2'b01;
					#20;
					in0 <= 32'h3E048494; in1 <= 32'hBB6F62CF; in2 <= 32'h3E008073; in3 <= 32'h3A815442; in4 <= 32'h00000000; in5 <= 32'h3AC6F547;
					in6 <= 32'h3D808081; in7 <= 32'hBBB018CC;
					i_vld <= 1'b1; bias_add <= 1'b0; bias_loc <= 2'b01;
					#20;
					in0 <= 32'h3D808081; in1 <= 32'hBB0AE11A; in2 <= 32'h00000000; in3 <= 32'h00000000; in4 <= 32'h00000000; in5 <= 32'h00000000;
					in6 <= 32'h00000000; in7 <= 32'h00000000;
					i_vld <= 1'b1; bias_add <= 1'b1; bias_loc <= 2'b01;
					#20;
					i_vld <= 1'b0; bias_add <= 1'b0; bias_loc <= 2'b01;
					
			end
			Test: begin
				
			end
			
			Done: begin
				
			end
		endcase
	end




endmodule
