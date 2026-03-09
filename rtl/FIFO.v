module FIFO #(
	parameter integer W = 256,
	parameter integer length = 32
)(
	input clk, reset,
	input out_ready, 
	input [1:0] in_valid, 
	input [36:0] extra_in0, extra_in1, 
	input [W-1:0] in_data0, in_data1, 
	output out_valid,
	output reg [1:0] in_ready, 
	output [36:0] extra_out,
	output [W-1:0] out_data
);

	localparam integer ADDR_W = $clog2(length);
	reg [ADDR_W-1:0] wptr, rptr;
	reg [W-1:0] fifo [0:length-1];
	reg [36:0] extra_fifo [0:length-1];
	reg [ADDR_W:0] count;
	wire [1:0] push;
	wire pop;
	
	assign push = (in_valid > in_ready) ? in_ready : in_valid;
	assign pop = out_valid && out_ready; 
	
	assign out_valid = count > 0;
	assign out_data = out_valid ? fifo[rptr] : 'd0;
	assign extra_out = out_valid ? extra_fifo[rptr] : 37'd0;
	
	always @(*) begin
		if (count+3 < length)
			in_ready = 2'b11;
		else if (count + 2 < length)
			in_ready = 2'b10;
		else if (count + 1 < length)
			in_ready = 2'b01;
		else 
			in_ready = 2'b00;
	end
	
	always @(posedge clk) begin
		if (reset) begin
			wptr <= 5'd0;
			rptr <= 5'd0;
			count <= 6'd0;
		end 
		else begin
						
			case (push)
				2'b01: begin
							fifo[wptr] <= in_data0;
							extra_fifo[wptr] <= extra_in0;
							wptr <= wptr + 'd1;
						end
				2'b10: begin
							fifo[wptr] <= in_data0;
							fifo[wptr+1] <= in_data1;
							extra_fifo[wptr] <= extra_in0;
							extra_fifo[wptr+1] <= extra_in1;
							wptr <= wptr + 'd2;
						end
			endcase
			
			if (pop) begin
				rptr <= rptr + 'd1;
			end
			
			case ({push, pop})
				3'b001: count <= count - 'd1;
				3'b010: count <= count + 'd1;
				3'b100: count <= count + 'd2;
				3'b101: count <= count + 'd1;
				3'b110: count <= count + 'd3;
				3'b111: count <= count + 'd2;
				default: count <= count;
			endcase
		end

	end

endmodule
