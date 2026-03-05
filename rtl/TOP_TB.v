module TOP_TB ();

	reg clk, rst;
	
	reg load_data_in_valid;
	reg [31:0] load_data_in_data;
	reg write_buff_out_ready;
	wire load_data_in_ready;
	
	wire out_valid;
	wire [63:0] out_data;
		
			
	TOP dut(
		clk, rst,
		load_data_in_valid,
		load_data_in_data, 
		load_data_in_ready,
		write_buff_out_ready,
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
	
	
	always @(posedge clk) begin
		case (Present_state)
			Init: begin
					load_data_in_valid <= 1'b0; 
					load_data_in_data <= 32'b0;
					write_buff_out_ready <= 1'b1;
					rst <= 1;
					#20 
					rst <= 0;
					write_buff_out_ready <= 1'b1;
					load_data_in_data <= 32'h00000000;
					load_data_in_valid <= 1'b1;
					#20
					load_data_in_data <= 32'h3E38B8AE;
					#20
					load_data_in_data <= 32'h3E38B8AE;
					#20
					load_data_in_data <= 32'h00000000;
					#20
					load_data_in_data <= 32'h3E048494;
					#20
					load_data_in_data <= 32'h3E008073;
					#20
					load_data_in_data <= 32'h00000000;
					#20
					load_data_in_data <= 32'h3E008073;
					#20
					load_data_in_data <= 32'h3E008073;
					#20
					load_data_in_data <= 32'h00000000;
					#20
					load_data_in_data <= 32'h00000000;
					#20
					load_data_in_data <= 32'h00000000;
					#20
					load_data_in_data <= 32'h00000000;
					#20
					load_data_in_data <= 32'h00000000;
					#20
					load_data_in_data <= 32'h00000000;
					#20
					load_data_in_data <= 32'h00000000;
					#20
					load_data_in_valid <= 1'b0;
					



					
			end
			Test: begin
				
			end
			
			Done: begin
				
			end
		endcase
	end



endmodule
