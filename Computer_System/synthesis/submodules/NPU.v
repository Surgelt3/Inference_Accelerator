module NPU(
	input clk, rst, 
	input [31:0] instr,
	input instr_valid, 
	
	input load_data_in_valid,
	input [31:0] load_data_in_data, 
	output load_data_in_ready,
	
	output [31:0] pc_out,
	
	input write_buff_out_ready,
	output out_valid,
	output [63:0] out_data
);
	
	//Note: Each address stores 8 32 bit numbers so 7'd1 (stores 8 32 bit numbers) 7'd2 (stores 8 32 bit numbers)
	// instr breakdown
	// opcode (3 bits): 31-29
	// start_loc (9 bits) (only use bits 26-20): 28-20
	// length (10 bits) (size of kernel): 19-10
	// param_loc (9 bits) (location of kernel + bias) (only use bits 7-1): 9-1
	// unused: 0
	
	// MAC opcode: 3'b000
	// RELU (must be called the instruction after the MAC to apply relu to the MAC instruction result) opcode: 3'b010
	// LOAD (loads in 8 32 bit numbers at the start_loc address from the write data buffer) opcode: 3'b001
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
	wire [255:0] mem_local0_write_data, mem_local1_write_data;
	wire [511:0] mem_local0_read_data, mem_local1_read_data;
	
	wire run_end;
	wire [1:0] pe_busy;
	
	wire [31:0] pc_in;
	wire classifier_bit;
	wire [2:0] opcode_out;
	wire [8:0] start_loc_out, param_loc_out;
	wire [9:0] length_out;
	
	wire load_signal, relu_signal, pool_signal, mac_signal;
	
	wire [1:0] pe_fifo_bias_loc_control_unit;
	wire [1:0] read_size_control_unit;
	wire bias_add_control_unit;
	wire [8:0] mem_local_address_control_unit, mem_local_param_loc_control_unit;
	wire [1:0] mem_local_read_size_out;

	
	always @(posedge clk) begin
		if (rst) begin
			local_mem_read_num <= 1'b0;
			local_mem_write_num <= 1'b1;
		end
	end
	
	
	
		
	wire load_data_control_unit;
	wire load_data_out_valid;
	wire [8:0] load_data_write_address;
	wire [255:0] load_data_mem_write_data;
	wire [31:0] pc_instr_dec, pc_control_unit, pc_mem_local, pc_fifo0, pc_fifo1;
	
	wire mem_local0_read_valid_out;
	wire mem_local0_bias_add_out;
	
	wire instr_valid_control_unit;
	
	assign pc_in = pc_out;
	
	wire load_data_ready;
	
	wire write_buff_in_ready_pre;
	
	READ_BUFF read_buff(
		.clk(clk), .rst(rst),
		.out_ready(load_data_ready), 
		.in_valid(load_data_in_valid), 
		.in_data(load_data_in_data), 
		.out_valid(load_data_out_valid),
		.in_ready(load_data_in_ready), 
		.out_data(load_data_mem_write_data)
	);

	
	INSTR_DECODER instr_decoder(
		.clk(clk), .rst(rst), 
		.instr_valid(instr_valid),
		.load_out_valid(load_data_out_valid),
		.output_ready(write_buff_in_ready_pre), 
		.opcode(instr[31:29]),
		.start_loc(instr[28:20]), .param_loc(instr[9:1]),
		.length(instr[19:10]),
		.pe_busy(pe_busy), 
		.pc_in(pc_in),
		.classifier_bit_out(classifier_bit),
		.instr_valid_out(instr_valid_control_unit), 
		.relu_signal(relu_signal), .load_signal(load_data_ready), 
		.opcode_out(opcode_out),
		.start_loc_out(start_loc_out), .param_loc_out(param_loc_out),
		.length_out(length_out),
		.pc_clock(pc_instr_dec), 
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
	
	wire [255:0] pe0_fifo_in_data0, pe1_fifo_in_data0;
	wire [255:0] pe0_fifo_in_data1, pe1_fifo_in_data1;
	wire [255:0] pe0_fifo_in_data2, pe1_fifo_in_data2;
	
	wire [8:0] mem_local0_param_loc;
	
	assign mem_local0_write_en = load_signal;
	
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
	
	assign classifier_bit_fifo = mem_local0_classifier_bit_out;
	
	assign pe0_fifo_in_data0 = mem_local0_read_data[255:0];
	assign pe1_fifo_in_data0 = mem_local0_read_data[255:0];
	assign pe0_fifo_in_data1 = mem_local0_read_data[511:256];
	assign pe1_fifo_in_data1 = mem_local0_read_data[511:256];
	//assign pe0_fifo_in_data2 = mem_local0_read_data[767:512];
	//assign pe1_fifo_in_data2 = mem_local0_read_data[767:512];

	
	assign pe0_fifo_out_ready = 1'b1;
	assign pe1_fifo_out_ready = 1'b1;
	assign pe_fifo_in_valid = mem_local0_read_valid_out;
	assign mem_local_read_size_out = mem_local0_read_size_out;
	assign pe0_fifo_in_valid = ((classifier_bit_fifo == 0) && (pe_fifo_in_valid)) ? mem_local_read_size_out : 2'b00;
	assign pe1_fifo_in_valid = ((classifier_bit_fifo == 1) && (pe_fifo_in_valid)) ? mem_local_read_size_out : 2'b00;

	
	assign fifo_bias_add = mem_local0_bias_add_out;
	
	
	assign pe0_fifo_bias_add0 = ((mem_local_read_size_out == 2'b01) && (classifier_bit_fifo == 0)) ? fifo_bias_add: 1'b0;
	assign pe0_fifo_bias_add1 = ((mem_local_read_size_out == 2'b10) && (classifier_bit_fifo == 0)) ? fifo_bias_add: 1'b0;
	//assign pe0_fifo_bias_add2 = ((mem_local_read_size_out == 2'b11) && (classifier_bit_fifo == 0)) ? fifo_bias_add: 1'b0;
	assign pe1_fifo_bias_add0 = ((mem_local_read_size_out == 2'b01) && (classifier_bit_fifo == 1)) ? fifo_bias_add: 1'b0;
	assign pe1_fifo_bias_add1 = ((mem_local_read_size_out == 2'b10) && (classifier_bit_fifo == 1)) ? fifo_bias_add: 1'b0;
	//assign pe1_fifo_bias_add2 = ((mem_local_read_size_out == 2'b11) && (classifier_bit_fifo == 1)) ? fifo_bias_add: 1'b0;

	
	assign mem_local_bias_loc = local_mem_read_num ? mem_local1_bias_loc : mem_local0_bias_loc;
	assign pe0_fifo_bias_loc = (classifier_bit_fifo == 0) ? mem_local_bias_loc : 2'b00;
	assign pe1_fifo_bias_loc = (classifier_bit_fifo == 1) ? mem_local_bias_loc : 2'b00;
	
	
		
	CONTROL_UNIT control_unit(
		.clk(clk), .rst(rst),
		.instr_valid(instr_valid_control_unit), 
		.opcode(opcode_out),
		.classifier_bit(classifier_bit),
		.address(start_loc_out), .param_loc(param_loc_out), 
		.length(length_out),
		.pc_in(pc_instr_dec), 
		.load_data(load_data_mem_write_data), 
		.pe_fifo_bias_loc(pe_fifo_bias_loc_control_unit),
		.read_size(read_size_control_unit), 
		.bias_add(bias_add_control_unit), 
		.mem_local_address(mem_local_address_control_unit), .mem_local_param_loc(mem_local_param_loc_control_unit), 
		.curr_classifier_bit(curr_classifier_bit), 
		.load_signal(load_signal), .pool_signal(pool_signal), .mac_signal(mac_signal), 
		.pe_busy(pe_busy),
		.pc_clock(pc_control_unit), 
		.write_data(mem_local0_write_data)
	);
	
	assign mem_read_en = (classifier_bit_fifo == 1) ? pe1_fifo_in_ready : pe0_fifo_in_ready;
	
	wire read_en;
	assign read_en = mac_signal | pool_signal;
	wire [1:0] control_signals_mem_local, control_signals_fifo0, control_signals_fifo1;
	
	wire pool_signal_pe0, pool_signal_pe1;
	wire relu_signal_pe0, relu_signal_pe1;
	wire [31:0] pc_pe0, pc_pe1;
	
	wire pe_switch_out_taken;
	wire pe_switch_out_valid;
	wire pe_switch_relu_signal;
	wire [31:0] pe_switch_out;
	wire [31:0] pc_pe_switch_out;
	
	wire relu_in_valid;
	
	wire [31:0] pc_in_relu_out;
	
	
	MEM_LOCAL mem_local0(
		.clk(clk), .rst(rst),
		.write_en(mem_local0_write_en), .read_en(read_en), 
		.bias_add(mem_local0_bias_add), 
		.classifier_bit(curr_classifier_bit),
		.control_signals({pool_signal, relu_signal}),
		.read_size(mem_local0_read_size), .bias_loc(pe_fifo_bias_loc_control_unit), 
		.address(mem_local0_address), .param_loc(mem_local0_param_loc), 
		.pc_in(pc_control_unit), 
		.write_data(mem_local0_write_data),
		.classifier_bit_out(mem_local0_classifier_bit_out), 
		.read_valid_out(mem_local0_read_valid_out), 
		.bias_add_out(mem_local0_bias_add_out), 
		.bias_loc_out(mem_local0_bias_loc), .read_size_out(mem_local0_read_size_out),
		.control_signals_clock(control_signals_mem_local), 
		.pc_clock(pc_mem_local),
		.read_data(mem_local0_read_data)
	);

	
	FIFO pe0_fifo (
		.clk(clk), .reset(rst),
		.out_ready(pe0_fifo_out_ready), 
		.in_valid(pe0_fifo_in_valid), 
		.extra_in0({pe0_fifo_bias_add0, pe0_fifo_bias_loc, pc_mem_local, control_signals_mem_local}), .extra_in1({pe0_fifo_bias_add1, pe0_fifo_bias_loc, pc_mem_local, control_signals_mem_local}),
		.in_data0(pe0_fifo_in_data0), .in_data1(pe0_fifo_in_data1), 
		.out_valid(pe0_in_valid),
		.in_ready(pe0_fifo_in_ready),
		.extra_out({pe0_bias_add, pe0_bias_loc, pc_fifo0, control_signals_fifo0}),
		.out_data(pe0_data)
	);
	
	
	PE pe0(
		.clk(clk), .rst(rst), .i_vld(pe0_in_valid), .bias_add(pe0_bias_add),  
		.bias_loc(pe0_bias_loc), 
		.cntrl_in(control_signals_fifo0), 
		.pc_in(pc_fifo0), 
		.in0(pe0_data[31:0]), .in1(pe0_data[63:32]), .in2(pe0_data[95:64]), .in3(pe0_data[127:96]), 
		.in4(pe0_data[159:128]), .in5(pe0_data[191:160]), .in6(pe0_data[223:192]), .in7(pe0_data[255:224]), 
		.out_valid(pe0_out_valid), 
		.cntrl_clock({pool_signal_pe0, relu_signal_pe0}),
		.pc_clock(pc_pe0), 
		.out_node(pe0_out)
	);
	
	// FIFO Singls to assign
	// to be used pe1_fifo_in_ready
	
	
	FIFO pe1_fifo (
		.clk(clk), .reset(rst),
		.out_ready(pe1_fifo_out_ready), 
		.in_valid(pe1_fifo_in_valid),
		.extra_in0({pe1_fifo_bias_add0, pe1_fifo_bias_loc, pc_mem_local, control_signals_mem_local}), .extra_in1({pe1_fifo_bias_add0, pe1_fifo_bias_loc, pc_mem_local, control_signals_mem_local}), 
		.in_data0(pe1_fifo_in_data0), .in_data1(pe1_fifo_in_data1), 
		.out_valid(pe1_in_valid),
		.in_ready(pe1_fifo_in_ready),
		.extra_out({pe1_bias_add, pe1_bias_loc, pc_fifo1, control_signals_fifo1}),
		.out_data(pe1_data)
	);
	
	
	PE pe1(
		.clk(clk), .rst(rst), .i_vld(pe1_in_valid), .bias_add(pe1_bias_add),  
		.bias_loc(pe1_bias_loc), 
		.cntrl_in(control_signals_fifo1), 
		.pc_in(pc_fifo1),
		.in0(pe1_data[31:0]), .in1(pe1_data[63:32]), .in2(pe1_data[95:64]), .in3(pe1_data[127:96]), 
		.in4(pe1_data[159:128]), .in5(pe1_data[191:160]), .in6(pe1_data[223:192]), .in7(pe1_data[255:224]), 
		.out_valid(pe1_out_valid), 
		.cntrl_clock({pool_signal_pe1, relu_signal_pe1}),
		.pc_clock(pc_pe1), 
		.out_node(pe1_out)
	);
	
	
	PE_SWITCH pe_switch(
		.clk(clk), .rst(rst), 
		.relu_signal({relu_signal_pe1, relu_signal_pe0}),
		.in_valid({pe1_out_valid, pe0_out_valid}),
		.pc0(pc_pe0), .pc1(pc_pe1),
		.in0(pe0_out), .in1(pe1_out),
		.out_valid(pe_switch_out_valid),
		.relu_out(pe_switch_relu_signal),
		.out(pe_switch_out), .pc_out(pc_pe_switch_out)
	);
	
	assign relu_in_valid = pe_switch_out_valid;
	
	RELU6 activation_func(
		.clk(clk),
		.in_valid(relu_in_valid), 
		.use_relu(pe_switch_relu_signal), 
		.pc_in(pc_pe_switch_out), 
		.in_data(pe_switch_out),
		.out_valid(relu_valid),
		.pc_out(pc_in_relu_out), 
		.out_data(relu_out)
	);
	
	
	WRITE_BUFF write_buff(
		.clk(clk), .rst(rst),
		.out_ready(write_buff_out_ready),
		.in_valid(relu_valid),
		.in_data(relu_out), 
		.pc_in(pc_in_relu_out),
		.out_valid(out_valid),
		.in_ready_pre(write_buff_in_ready_pre),
		.out_data(out_data)
	);

	
	

endmodule
