
module NPU(
	input clk, rst, 
	input [31:0] instr,
	output npu_instr_ready, 
);

	
	
	
	
	//FIFO inputs
	wire pe0_fifo_in_valid, pe1_fifo_in_valid;
	wire pe0_fifo_out_ready, pe1_fifo_out_ready;
	wire [256:0] pe0_fifo_in_data, pe1_fifo_in_data;
	//FIFO Output
	wire pe0_fifo_in_ready, pe1_fifo_in_ready;
	//FIFO Output / PE Input
	wire pe0_in_valid, pe1_in_valid;
	wire [255:0] pe0_data, pe1_data;
	//Wire for PE units
	wire pe0_reset, pe1_reset;
	wire pe0_out_valid, pe1_out_valid;
	wire pe0_bias_add, pe1_bias_add;
	wire [31:0] pe0_out, pe1_out;
	//wires for PE Switch and Activation Function
	wire pe_sWitch_valid;
	wire [1:0] pe_sWitch_sel;
	wire relu_valid;
	wire [31:0] relu_in;
	wire [31:0] relu_out;
	
	reg local_mem_read_num, local_mem_write_num;
	
	//wires for mem_local
	wire mem_local0_wire_en, mem_local1_wire_en;
	wire mem_local0_bias_add, mem_local1_bias_add;
	wire [8:0] mem_local0_address, mem_local1_address;
	wire [31:0] mem_local0_write_data, mem_local1_write_data;
	wire [255:0] mem_local0_read_data, mem_local1_read_data;
	
	wire run_end;
	wire [2:0] pe_busy;
	
	wire [31:0] pc_in, pc_out;
	wire classifier_bit;
	wire [2:0] curr_opcode;
	wire [3:0] op_len;
	
	always @(posedge clk) begin
		if (rst) begin
			local_mem_read_num <= 1'b0;
			local_mem_write_num <= 1'b1;
		end
			
	end
	
	assign mem_local0_wire_en = local_mem_write_num ? 1'b0 : relu_valid;
	assign mem_local1_wire_en = local_mem_write_num ? relu_valid : 1'b0;
	assign mem_local0_address = local_mem_write_num ? ;
	assign mem_local1_address = local_mem_write_num ? ;
	
	assign mem_local0_bias_add = (!local_mem_read_num) ? (pe0_bias_add) : 1'b0;
	assign mem_local1_bias_add = (local_mem_read_num) ? (pe1_bias_add) : 1'b0;

	
	assign pe0_fifo_in_data = local_mem_read_num ? mem_local1_read_data : mem_local0_read_data;
	assign pe1_fifo_in_data = local_mem_read_num ? mem_local1_read_data : mem_local0_read_data;

	INSTR_DECODER instr_decoder(
		.clk(clk), .rst(rst), 
		.opcode(instr[2:0]),
		input [8:0] start_loc, length, param_loc,
		.pe_busy(pe_busy), 
		.pc_in(pc_in),
		.opcode_out(curr_opcode),
		.classifier_bit_out(classifier_bit),
		.op_len(op_len),
		.pc_out(pc_out)
		
	);

	
	
	
	CONTROL_UNIT control_unit(
		.clk(clk), .rst(rst), 
		.opcode(curr_opcode),
		.classifier_bit(classifier_bit),
		.op_len(op_len), 
		.pe0_fifo_out_ready(pe0_fifo_out_ready), .pe1_fifo_out_ready(pe1_fifo_out_ready),
		.pe0_bias_add(pe0_bias_add), .pe1_bias_add(pe1_bias_add), 
		.run_end(run_end),
		.pe_busy(pe_busy)
	);

	
	MEM_LOCAL mem_local0(
		.clk(clk), .rst(rst),
		.write_en(mem_local0_wire_en),
		.bias_add(mem_local0_bias_add), 
		.address(mem_local0_address), .param_loc(), 
		.write_data(mem_local0_write_data),
		.read_data(mem_local0_read_data)
	);
	


	MEM_LOCAL mem_local1(
		.clk(clk), .rst(rst),
		.write_en(mem_local1_wire_en),
		.bias_add(mem_local1_bias_add), 
		.address(mem_local1_address), .param_loc(), 
		.write_data(mem_local1_write_data),
		.read_data(mem_local1_read_data)
	);

	
	FIFO pe0_fifo #(
		W = 256,
		length = 32
	)(
		.clk(clk), .reset(rst),
		.in_valid(pe0_fifo_in_valid), .out_ready(pe0_fifo_out_ready), 
		.in_data(pe0_fifo_in_data),
		.in_ready(pe0_fifo_in_ready), .out_valid(pe0_in_valid),
		.out_data(pe0_data)
	);

	PE pe0(
		.clk(clk), .rst(pe0_reset), .i_vld(pe0_in_valid), .bias_add(pe0_bias_add),  
		.in0(pe0_data[0:31]), .in1(pe0_data[32:63]), .in2(pe0_data[64:95]), .in3(pe0_data[96:127]), 
		.in4(pe0_data[128:159]), .in5(pe0_data[160:191]), .in6(pe0_data[192:223]), .in7(pe0_data[224:255]), 
		.bias_val(pe0_data[0:31]), 
		.out_valid(pe0_out_valid), 
		.out_node(pe0_out)
	);
	
	
	FIFO pe1_fifo #(
		W = 256,
		length = 32
	)(
		.clk(clk), .reset(rst),
		.in_valid(pe1_fifo_in_valid), .out_ready(pe1_fifo_out_ready), 
		.in_data(pe1_fifo_in_data),
		.in_ready(pe1_fifo_in_ready), .out_valid(pe1_in_valid),
		.out_data(pe1_data)
	);
	
	
	PE pe1(
		.clk(clk), .rst(pe1_reset), .i_vld(pe1_in_valid), .bias_add(pe1_bias_add),  
		.in0(pe1_data[0:31]), .in1(pe1_data[32:63]), .in2(pe1_data[64:95]), .in3(pe1_data[96:127]), 
		.in4(pe1_data[128:159]), .in5(pe1_data[160:191]), .in6(pe1_data[192:223]), .in7(pe1_data[224:255]), 
		.bias_val(pe1_data[0:31]),  
		.out_valid(pe1_out_valid), 
		.out_node(pe1_out)
	);
	
	
	assign pe_sWitch_sel = {pe0_out_valid, pe1_out_valid};
	
	
	PE_SWITCH pe_switch(
		.sel(pe_sWitch_sel),
		.in1(pe0_out), .in2(pe1_out),
		.out_valid(pe_sWitch_valid),
		.out(relu_in)
	);
	
	RELU6 activation_func(
		.clk(clk),
		.in_valid(pe_sWitch_valid), 
		.in_data(relu_in),
		.out_valid(relu_valid),
		.out_data(relu_out)
	);
	

	
	

	

endmodule