module avalon_mm_master_interface (
	input clk, reset, waitrequest, readdatavalid, 
	input [255:0] readdata, 
	output read, write, flush, 
	output [2:0] burstcount,
	output [31:0] byteenable, 
	output [31:0] address, writedata
);

	
	always @(posedge clk) begin
			
			if (reset) begin
			
			end else begin
				
				
			
			end
	
	end

	


endmodule
