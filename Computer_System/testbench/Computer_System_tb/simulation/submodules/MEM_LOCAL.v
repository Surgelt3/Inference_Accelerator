
module MEM_LOCAL(
	input clk, rst,
	input write_en, read_en, 
	input bias_add, 
	input classifier_bit, 
	input [1:0] control_signals,
	input [1:0] read_size, bias_loc, 
	input [8:0] address, param_loc, 
	input [31:0] pc_in,
	input [255:0] write_data,
	output reg classifier_bit_out,
	output reg read_valid_out, 
	output reg bias_add_out, 
	output reg [1:0] bias_loc_out, read_size_out, 
	output reg [1:0] control_signals_clock,
	output reg [31:0] pc_clock,
	output reg [511:0] read_data
);

	
	reg [31:0] mem0 [0:63];
	reg [31:0] mem1 [0:63];
	reg [31:0] mem2 [0:63];
	reg [31:0] mem3 [0:63];
	reg [31:0] mem4 [0:63];
	reg [31:0] mem5 [0:63];
	reg [31:0] mem6 [0:63];
	reg [31:0] mem7 [0:63];
	
	//initial begin
	//	#20 $readmemh("C:/Users/lucas/Desktop/ELEC_49X/Inference_Accelerator/rtl/memory.hex", mem1);
	//end
	
	integer i;
	
	always @(posedge clk) begin
		read_valid_out <= 1'b0;
		pc_clock <= pc_in;
		control_signals_clock <= control_signals;
		if (rst) begin
		
		end	
		else begin
			bias_loc_out <= bias_loc;
			classifier_bit_out <= classifier_bit;
			bias_add_out <= bias_add;
			read_size_out <= read_size;
			if (write_en) begin
				  mem0[address[5:0]] <= write_data[31:0];
				  mem1[address[5:0]] <= write_data[63:32];
				  mem2[address[5:0]] <= write_data[95:64];
				  mem3[address[5:0]] <= write_data[127:96];
				  mem4[address[5:0]] <= write_data[159:128];
				  mem5[address[5:0]] <= write_data[191:160];
				  mem6[address[5:0]] <= write_data[223:192];
				  mem7[address[5:0]] <= write_data[255:224];
			end
			else if (read_en) begin
				
				if (read_size == 2'b00) begin
					read_valid_out <= 1'b0;
				end
				else begin
					read_data <= {
							mem7[param_loc[5:0]], mem7[address[5:0]], mem6[param_loc[5:0]], mem6[address[5:0]], 
							mem5[param_loc[5:0]], mem5[address[5:0]], mem4[param_loc[5:0]], mem4[address[5:0]], 
							mem3[param_loc[5:0]], mem3[address[5:0]], mem2[param_loc[5:0]], mem2[address[5:0]], 
							mem1[param_loc[5:0]], mem1[address[5:0]], mem0[param_loc[5:0]], mem0[address[5:0]]
						};
					read_valid_out <= 1'b1;
					
				end
			end
		end
	end
	

endmodule
