module IN_DATA_BUFFER(
	input clk, rst,
	input out_ready, 
	input in_valid, 
	input [8:0] in_address,
	input [31:0] in_data, 
	output reg out_valid,
	output in_ready, 
	output [8:0] out_address,
	output [511:0] out_data
);
	
	wire push, pop;
	reg [511:0] fifo_data [0:15];
	reg [8:0] fifo_address [0:15];
	
	reg [3:0] wptr, rptr;
	
	reg [3:0] buffer_counter;
	reg [31:0] in_buffer [0:15];
	
	assign in_ready = (wptr + 1) != rptr;
	assign push = in_valid && in_ready;
	
	assign not_empty = !(wptr == rptr);
	assign pop = not_empty && out_ready; 
	
	assign out_data = fifo_data[rptr];
	assign out_address = fifo_address[rptr];

	always @(posedge clk) begin
		if (rst) begin
			buffer_counter <= 3'd0;
			wptr <= 0;
			rptr <= 0;
		end
		else begin
			if (push) begin
				in_buffer[buffer_counter] <= in_data;
				buffer_counter <= buffer_counter + 4'd1;
				if (buffer_counter == 4'b1111) begin
					fifo_data[wptr] <= {in_data, in_buffer[14], in_buffer[13], in_buffer[12], in_buffer[11], in_buffer[10], in_buffer[9], in_buffer[8], 
												in_buffer[7], in_buffer[6], in_buffer[5], in_buffer[4], in_buffer[3], in_buffer[2], in_buffer[1], in_buffer[0]};
					fifo_address[wptr] <= in_address;
					wptr <= wptr + 4'd1;
					buffer_counter <= 4'b0000;
				end
			end
			
			if (pop) begin
				out_valid <= 1'b1;
				rptr <= rptr + 4'd1;
			end
			else begin
				out_valid <= 1'b0;
			end

		end
	end
	



endmodule
