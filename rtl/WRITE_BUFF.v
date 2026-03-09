module WRITE_BUFF(
	input clk, rst,
	input out_ready, 
	input in_valid,
	input [31:0] in_data, 
	input [31:0] pc_in,
	output out_valid,
	output in_ready_pre, 
	output [63:0] out_data
);

	localparam [31:0] offset = 32'd120;
		

	reg [3:0] rptr, wptr;
	reg [4:0] counter;
	reg [63:0] fifo [0:15];
	
	wire in_ready;
	
	assign in_ready_pre = (counter < 'd7);
	assign out_valid = (counter > 0);
	assign in_ready = (counter != 5'b10000);
	
	
	assign pop = out_valid && out_ready;
	assign push = in_valid && in_ready;
	
	
	assign out_data = out_valid ? fifo[rptr]: 64'd0;

	always @(posedge clk) begin
		if (rst) begin
			counter <= 'd0;
			wptr <= 'd0;
			rptr <= 'd0;
		end else begin
		
			if (pop) begin
				rptr <= rptr + 'd1;
			end
			
			
			if (push) begin
				fifo[wptr] <= {in_data, pc_in + offset};
				wptr <= wptr + 'd1;
			end
			
			case ({push, pop})
				2'b01: counter <= counter - 'd1;
				2'b10: counter <= counter + 'd1;
				default: counter <= counter;
			endcase
			
			
		end
		
	end


endmodule
