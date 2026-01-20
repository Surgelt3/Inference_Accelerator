module CONTROL_UNIT(
	input clk, rst, 
	input [2:0] opcode,
	input classifier_bit,
	input [5:0] op_len,
	output pe0_fifo_out_ready, pe1_fifo_out_ready,
	output pe0_bias_add, pe1_bias_add, 
	output run_end,
	output [1:0] pe_busy, 
	output [5:0] pe_counter
);
	
	reg [5:0] op0_len, op1_len;
	reg [5:0] pe0_counter, pe1_counter;

	always @(posedge clk) begin
		if (rst) begin
			pe0_fifo_out_ready <= 1'b0;
			pe1_fifo_out_ready <= 1'b0;
			pe0_bias_add <= 1'b0;
			pe1_bias_add <= 1'b0;
			run_end <= 1'b0;
			pe_busy <= 2'b00;
		end
		else if (opcode == 3'b000) begin
			// MAC Operation
			
			case(classifier_bit)
				1'b0: begin
							pe0_fifo_out_ready <= 1'b1;
							pe_busy <= pe_busy | 2'b01;
							op0_len <= op_len;
						end
				1'b1: begin
							pe1_fifo_out_ready <= 1'b1;
							pe_busy <= pe_busy | 2'b10;
							op1_len <= op_len;
						end
			endcase
			
			if (op0_len == pe0_counter) begin
				pe0_counter <= '0;
				pe0_bias_add <= 1'b1;
				pe_busy <= pe_busy & 2'b10;
			end 
			else begin
				pe0_counter <= pe0_counter+4;
				pe0_bias_add <= 1'b0;
			end
			
			if (op1_len == pe1_counter) begin
				pe1_counter <= '0;
				pe1_bias_add <= 1'b1;
				pe_busy <= pe_busy & 2'b01;
			end 
			else begin
				pe1_counter <= pe1_counter+4;
				pe1_bias_add <= 1'b0;
			end
			
		end
		else if (opcode == 3'b011) begin
			// END Operation
			if (pe_busy == 2'b00)
				run_end <= 1'b1;
		end
	
	end


endmodule