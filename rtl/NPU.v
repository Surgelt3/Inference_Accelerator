
module NPU(
	input clk, rst, 
	input [31:0] instr,
	output npu_instr_ready, 
	
	input load_data_in_valid,
	input [8:0] load_data_in_address,
	input [31:0] load_data_in_data, 
	output load_data_in_ready

);

	// instr breakdown
	// opcode (3 bits): 31-29
	// start_loc (9 bits): 28-20
	// length (10 bits): 19-10
	// param_loc (9 bits): 9-1
	// unused: 0
	
	// MAC opcode: 3'b000
	// LOAD opcode: 3'b001
	// START opcode: 3'b010
	// END opcode: 3'b011
	
	
	
	//FIFO inputs
	wire [1:0] pe0_fifo_in_valid, pe1_fifo_in_valid;
	wire pe0_fifo_out_ready, pe1_fifo_out_ready;
	wire [256:0] pe0_fifo_in_data, pe1_fifo_in_data;
	//FIFO Output
	wire pe0_fifo_bias_add0, pe0_fifo_bias_add1, pe0_fifo_bias_add2, pe1_fifo_bias_add0, pe1_fifo_bias_add1, pe1_fifo_bias_add2;
	wire [1:0] pe0_fifo_bias_loc, pe1_fifo_bias_loc;
	wire [1:0] pe0_fifo_in_ready, pe1_fifo_in_ready;
	//FIFO Output / PE Input
	wire pe0_bias_add, pe1_bias_add;
	wire [1:0] pe0_bias_loc, pe1_bias_loc;
	wire pe0_in_valid, pe1_in_valid;
	wire [255:0] pe0_data, pe1_data;
	//Wire for PE units
	wire pe0_reset, pe1_reset;
	wire pe0_out_valid, pe1_out_valid;
	wire [31:0] pe0_out, pe1_out;
	//wires for PE Switch and Activation Function
	wire pe_sWitch_valid;
	wire [1:0] pe_sWitch_sel;
	wire relu_valid;
	wire [31:0] relu_in;
	wire [31:0] relu_out;
	
	reg local_mem_read_num, local_mem_write_num;
	
	//wires for mem_local
	wire [1:0] mem_local0_read_size, mem_local1_read_size;
	wire mem_local0_write_en, mem_local1_write_en;
	wire mem_local0_bias_add, mem_local1_bias_add;
	wire [8:0] mem_local0_address, mem_local1_address;
	wire [31:0] mem_local0_write_data, mem_local1_write_data;
	wire [767:0] mem_local0_read_data, mem_local1_read_data;
	
	wire run_end;
	wire [2:0] pe_busy;
	
	wire [31:0] pc_in, pc_out;
	wire classifier_bit;
	wire [2:0] opcode_out;
	wire [8:0] start_loc_out, param_loc_out;
	wire [9:0] length_out;
	
	always @(posedge clk) begin
		if (rst) begin
			local_mem_read_num <= 1'b0;
			local_mem_write_num <= 1'b1;
		end
	end
	
		
	wire load_data_control_unit;
	wire load_data_out_valid;
	wire [8:0] load_data_write_address;
	wire [511:0] load_data_mem_write_data;
	
	WRITE_BUFF write_buff(
		.clk(clk), .rst(rst),
		.out_ready(load_data_control_unit), 
		.in_valid(load_data_in_valid), 
		.in_address(load_data_in_address),
		.in_data(load_data_in_data), 
		.out_valid(load_data_out_valid),
		.in_ready(load_data_in_ready), 
		.out_address(load_data_write_address),
		.out_data(load_data_mem_write_data)
	);

	
	INSTR_DECODER instr_decoder(
		.clk(clk), .rst(rst), 
		.opcode(instr[31:29]),
		.start_loc(instr[28:20]), .param_loc(instr[9:1]),
		.length(instr[19:10]),
		.pe_busy(pe_busy), 
		.pc_in(pc_in),
		.classifier_bit_out(classifier_bit),
		.opcode_out(curr_opcode),
		.start_loc_out(start_loc_out), .param_loc_out(param_loc_out),
		.length_out(length_out),
		.pc_out(pc_out)
		
	);
	
	wire bias_add;
	wire [8:0] out_address; 
	
	
	wire run_end_control_unit;
	wire [1:0] mem_local0_read_size_out, mem_local1_read_size_out;
	wire [31:0] write_data_mem_local;
	wire classifier_bit_fifo, mem_local0_classifier_bit_out, mem_local1_classifier_bit_out, curr_classifier_bit;
	
	wire [1:0] mem_local_bias_loc, mem_local0_bias_loc, mem_local1_bias_loc;
	
	wire pe_fifo_in_valid;
	wire fifo_bias_add;
	wire mem_read_en;

	
	assign mem_local0_write_en = load_data_out_valid && load_data_control_unit;
	
	assign mem_local0_bias_add = bias_add_control_unit;
	assign mem_local1_bias_add = bias_add_control_unit;
	assign mem_local0_read_size = read_size_control_unit;
	assign mem_local1_read_size = read_size_control_unit;
	assign mem_local0_address = local_mem_read_num ? out_address : mem_local_address_control_unit;
	assign mem_local1_address = local_mem_read_num ? mem_local_address_control_unit : out_address;
	assign mem_local0_param_loc = mem_local_param_loc_control_unit;
	assign mem_local1_param_loc = mem_local_param_loc_control_unit;
	//assign mem_local0_write_data = write_data_mem_local;
	//assign mem_local1_write_data = write_data_mem_local;
	
	assign classifier_bit_fifo = local_mem_read_num ? mem_local1_classifier_bit_out : mem_local0_classifier_bit_out;
	
	assign pe0_fifo_in_data0 = mem_local0_read_data[255:0];
	assign pe1_fifo_in_data0 = mem_local0_read_data[255:0];
	assign pe0_fifo_in_data1 = mem_local0_read_data[511:256];
	assign pe1_fifo_in_data1 = mem_local0_read_data[511:256];
	assign pe0_fifo_in_data2 = mem_local0_read_data[767:512];
	assign pe1_fifo_in_data2 = mem_local0_read_data[767:512];

	
	assign pe0_fifo_out_ready = 1'b1;
	assign pe1_fifo_out_ready = 1'b1;
	assign pe_fifo_in_valid = mem_local0_read_valid_out;
	assign mem_local_read_size_out = local_mem_read_num ? mem_local1_read_size_out : mem_local0_read_size_out;
	assign pe0_fifo_in_valid = ((classifier_bit_fifo == 0) && (pe_fifo_in_valid)) ? mem_local_read_size_out : 2'b00;
	assign pe1_fifo_in_valid = ((classifier_bit_fifo == 1) && (pe_fifo_in_valid)) ? mem_local_read_size_out : 2'b00;

	
	assign fifo_bias_add = mem_local0_bias_add_out;
	
	
	assign pe0_fifo_bias_add0 = ((mem_local_read_size_out == 2'b01) && (classifier_bit_fifo == 0)) ? fifo_bias_add: 1'b0;
	assign pe0_fifo_bias_add1 = ((mem_local_read_size_out == 2'b10) && (classifier_bit_fifo == 0)) ? fifo_bias_add: 1'b0;
	assign pe0_fifo_bias_add2 = ((mem_local_read_size_out == 2'b11) && (classifier_bit_fifo == 0)) ? fifo_bias_add: 1'b0;
	assign pe1_fifo_bias_add0 = ((mem_local_read_size_out == 2'b01) && (classifier_bit_fifo == 1)) ? fifo_bias_add: 1'b0;
	assign pe1_fifo_bias_add1 = ((mem_local_read_size_out == 2'b10) && (classifier_bit_fifo == 1)) ? fifo_bias_add: 1'b0;
	assign pe1_fifo_bias_add2 = ((mem_local_read_size_out == 2'b11) && (classifier_bit_fifo == 1)) ? fifo_bias_add: 1'b0;

	
	assign mem_local_bias_loc = local_mem_read_num ? mem_local1_bias_loc : mem_local0_bias_loc;
	assign pe0_fifo_bias_loc = (classifier_bit_fifo == 0) ? mem_local_bias_loc : 2'b00;
	assign pe1_fifo_bias_loc = (classifier_bit_fifo == 1) ? mem_local_bias_loc : 2'b00;
	
	assign mem_local0_write_data = {{256{1'b0}}, load_data_mem_write_data};
	
	
	wire load_signal, relu_signal, pool_signal, mac_signal;
	
	wire [1:0] pe_fifo_bias_loc_control_unit;
	wire [1:0] read_size_control_unit;
	wire [2:0] bias_add_control_unit;
	wire [8:0] mem_local_address_control_unit, mem_local_param_loc_control_unit;
	
	CONTROL_UNIT control_unit(
		.clk(clk), .rst(rst), 
		.opcode(opcode_out),
		.classifier_bit(classifier_bit),
		.address(start_loc_out), .length(length_out), .param_loc(param_loc_out), 
		.pe_fifo_bias_loc(pe_fifo_bias_loc_control_unit),
		.read_size(read_size_control_unit), 
		.bias_add(bias_add_control_unit), 
		.mem_local_address(mem_local_address_control_unit), .mem_local_param_loc(mem_local_param_loc_control_unit), 
		.curr_classifier_bit(curr_classifier_bit), 
		.load_signal(load_signal), .relu_signal(relu_signal), .pool_signal(pool_signal), .mac_signal(mac_signal), 
		.pe_busy(pe_busy)
	);
	
	assign mem_read_en = (classifier_bit_fifo == 1) ? pe1_fifo_in_ready : pe0_fifo_in_ready;
	
	wire read_en;
	assign read_en = mac_signal | pool_signal;
	
	
	MEM_LOCAL mem_local0(
		.clk(clk), .rst(rst),
		.write_en(mem_local0_write_en), .read_en(read_en), 
		.bias_add(mem_local0_bias_add), 
		.classifier_bit(curr_classifier_bit), 
		.read_size(mem_local0_read_size), .bias_loc(pe_fifo_bias_loc_control_unit), 
		.write_address(load_data_write_address),
		.address(mem_local0_address), .param_loc(mem_local0_param_loc), 
		.write_data(load_data_mem_write_data),
		.classifier_bit_out(mem_local0_classifier_bit_out), 
		.read_valid_out(mem_local0_read_valid_out), 
		.bias_add_out(mem_local0_bias_add_out), 
		.bias_loc_out(mem_local0_bias_loc), .read_size_out(mem_local0_read_size_out), 
		.read_data(mem_local0_read_data)
	);

	
	FIFO pe0_fifo (
		.clk(clk), .reset(rst),
		.out_ready(pe0_fifo_out_ready), 
		.in_valid(pe0_fifo_in_valid), 
		.extra_in0({pe0_fifo_bias_add0, pe0_fifo_bias_loc}), .extra_in1({pe0_fifo_bias_add1, pe0_fifo_bias_loc}), .extra_in2({pe0_fifo_bias_add2, pe0_fifo_bias_loc}),
		.in_data0(pe0_fifo_in_data0), .in_data1(pe0_fifo_in_data1), .in_data2(pe0_fifo_in_data2),
		.out_valid(pe0_in_valid),
		.in_ready(pe0_fifo_in_ready),
		.extra_out({pe0_bias_add, pe0_bias_loc}),
		.out_data(pe0_data)
	);
	
	PE pe0(
		.clk(clk), .rst(pe0_reset), .i_vld(pe0_in_valid), .bias_add(pe0_bias_add),  
		.bias_loc(pe0_bias_loc), 
		.in0(pe0_data[31:0]), .in1(pe0_data[63:32]), .in2(pe0_data[95:64]), .in3(pe0_data[127:96]), 
		.in4(pe0_data[159:128]), .in5(pe0_data[191:160]), .in6(pe0_data[223:192]), .in7(pe0_data[255:224]), 
		.out_valid(pe0_out_valid), 
		.out_node(pe0_out)
	);
	
	// FIFO Singls to assign
	// to be used pe1_fifo_in_ready
	
	
	FIFO pe1_fifo (
		.clk(clk), .reset(rst),
		.out_ready(pe1_fifo_out_ready), 
		.in_valid(pe1_fifo_in_valid),
		.extra_in0({pe1_fifo_bias_add0, pe1_fifo_bias_loc}), .extra_in1({pe1_fifo_bias_add0, pe1_fifo_bias_loc}), .extra_in2({pe1_fifo_bias_add0, pe1_fifo_bias_loc}),
		.in_data0(pe1_fifo_in_data0), .in_data1(pe1_fifo_in_data1), .in_data2(pe1_fifo_in_data2),
		.out_valid(pe1_in_valid),
		.in_ready(pe1_fifo_in_ready),
		.extra_out({pe1_bias_add, pe1_bias_loc}),
		.out_data(pe1_data)
	);
	
	
	PE pe1(
		.clk(clk), .rst(pe1_reset), .i_vld(pe1_in_valid), .bias_add(pe1_bias_add),  
		.bias_loc(pe1_bias_loc), 
		.in0(pe1_data[31:0]), .in1(pe1_data[63:32]), .in2(pe1_data[95:64]), .in3(pe1_data[127:96]), 
		.in4(pe1_data[159:128]), .in5(pe1_data[191:160]), .in6(pe1_data[223:192]), .in7(pe1_data[255:224]), 
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
	
	assign write_data_mem_local = relu_out;

	
	
	

	

endmodule
