module FIFO #(
	parameter integer W = 256,
	parameter integer length = 32
)(
	input clk, reset,
	input in_valid, out_ready, 
	input [W-1:0] in_data,
	output in_ready, out_valid,
	output [W-1:0] out_data
);

	localparam integer ADDR_W = $clog2(length);
	reg [ADDR_W-1:0] wptr, rptr;
	reg [W-1:0] fifo [0:length-1];
	reg [ADDR_W:0] count;
	wire push, pop;
	
	assign push = in_valid && in_ready;
	assign pop = out_valid && out_ready; 
	
	assign in_ready = count < length;
	assign out_valid = count > 0;
	assign out_data = fifo[rptr];
	
	always @(posedge clk) begin
		if (reset) begin
			wptr <= '0;
			rptr <= '0;
			count <= '0;
		end 
		else begin
			
			if (push) begin
				fifo[wptr] <= in_data;
				wptr <= wptr + 'd1;
			end
			
			if (pop) begin
				rptr <= rptr + 'd1;
			end
			
			case ({push, pop})
				2'b01: count <= count - 'd1;
				2'b10: count <= count + 'd1;
				default: count <= count;
			endcase
		end

	end

endmodule