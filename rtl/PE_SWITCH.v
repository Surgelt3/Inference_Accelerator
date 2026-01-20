module PE_SWITCH(
	input [1:0] sel,
	input[31:0] in1, in2,
	output out_valid,
	output [31:0] out
);

	always @(*) begin
		case (sel)
			2'b00: begin
						out_valid = 1'b0;
						out = '0;
					end
			2'b01: begin 
						out_valid = 1'b1;
						out = in1;
					end
			2'b10: begin 
						out_valid = 1'b1;
						out = in2;
					end
			2'b11: begin 
						out_valid = 1'b1;
						out = in1;
					end
		
		endcase
	end

endmodule