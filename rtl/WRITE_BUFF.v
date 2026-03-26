module WRITE_BUFF(
	input clk, rst,
	input out_ready, 
	input in_valid,
	input [31:0] in_data, 
	input [31:0] pc_in,
	output reg out_valid,
	output in_ready_pre, 
	output reg [63:0] out_data
);

	//localparam [31:0] offset = 32'd120;
		

	wire pop, push;
	
	reg [3:0] rptr, wptr;
	reg [4:0] counter;
	reg [63:0] fifo [0:15];
	
	wire in_ready;
	
	assign in_ready_pre = (counter < 5'd7);
	assign in_ready = (counter != 5'b10000);
	
	
	assign pop = out_valid && out_ready;
	assign push = in_valid && in_ready;
	
	
	
	always @(posedge clk) begin
		 if (rst) begin
			  out_valid <= 1'b0;
			  out_data  <= 64'd0;
			  counter   <= 5'd0;
			  wptr      <= 4'd0;
			  rptr      <= 4'd0;
		 end else begin
			  if (push)
					fifo[wptr] <= {pc_in, in_data};

			  case ({push, pop})
					2'b10: begin
						 wptr <= wptr + 4'd1;
						 counter <= counter + 5'd1;
					end
					2'b01: begin
						 rptr <= rptr + 4'd1;
						 counter <= counter - 5'd1;
					end
					2'b11: begin
						 wptr <= wptr + 4'd1;
						 rptr <= rptr + 4'd1;
					end
			  endcase

			  out_valid <= (counter != 'd0);
			  if (counter != 'd0)
					out_data <= fifo[rptr];
			  else
					out_data <= 64'd0;
		 end
	end

endmodule
