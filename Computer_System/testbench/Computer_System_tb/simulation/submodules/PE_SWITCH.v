module PE_SWITCH(
	input clk, rst, 
	input out_taken,
	input [1:0] relu_signal,
	input [1:0] in_valid,
	input [31:0] pc0, pc1,
	input [31:0] in0, in1,
	output reg out_valid,
	output reg relu_out,
	output reg [31:0] out, pc_out
);

	reg [31:0] pe0_res, pe1_res;
	reg [31:0] pe0_pc, pe1_pc;
	reg [1:0] res_valid;
	reg [1:0] relu_pe;
	
	reg out_valid_next, relu_out_next;
	reg [31:0] out_next, pc_out_next;

	always @(posedge clk) begin
		out_valid <= out_valid_next;
		relu_out <= relu_out_next;
		out <= out_next;
		pc_out <= pc_out_next;
		
	end
	
	
	always @(*) begin
		
		if (rst) begin
			res_valid = 2'b00;
		end
		
		case (in_valid)
			2'b01: begin 
						pe0_res = in0;
						pe0_pc = pc0;
						relu_pe[0] = relu_signal[0];
						res_valid = res_valid | 2'b01;
					end
			2'b10: begin
						pe1_res = in1;
						pe1_pc = pc1;
						relu_pe[1] = relu_signal[1];
						res_valid = res_valid | 2'b10;
					end
			2'b11: begin 
							pe0_res = in0;
							pe1_res = in1;
							pe0_pc = pc0;
							pe1_pc = pc1;
							relu_pe = relu_signal;
							res_valid = res_valid | 2'b11;
					end
		endcase
		
		if (res_valid[0]) begin
			out_next = pe0_res;
			pc_out_next = pe0_pc;
			relu_out_next = relu_pe[0];
			out_valid_next = 1'b1;
			if (out_taken) begin
				res_valid = res_valid & 2'b10;
			end
		end
		else if (res_valid[1]) begin
			out_next = pe1_res;
			pc_out_next = pe1_pc;
			relu_out_next = relu_pe[1];
			out_valid_next = 1'b1;
			if (out_taken) begin
				res_valid = res_valid & 2'b01;
			end
		end
		else begin
			out_valid_next = 1'b0;
			relu_out_next <= relu_out;
			out_next <= out;
			pc_out_next <= pc_out;
		end
		
	

	end


endmodule
