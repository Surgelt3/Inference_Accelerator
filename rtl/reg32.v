module regN #(parameter int DATA_W = 32)(
	input wire clock,
	input wire resetn,
	input wire [DATA_W-1:0] D,
	input wire [DATA_W/8-1:0] byteenable,
	output reg [DATA_W-1:0] Q
);
	integer i;
	always @(posedge clock) begin
		if (!resetn) begin
			Q <= '0;
		end else begin
			for (i = 0; i < DATA_W/8; i = i + 1) begin
				if (byteenable[i])
					Q[i*8 +: 8] <= D[i*8 +: 8];
				end
		end
	end
endmodule
