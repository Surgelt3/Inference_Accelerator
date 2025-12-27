
`timescale 1ns/1ps


module PE_tb(
	
);
	 
    reg clk;
    reg rst;
	 reg i_vld;
	 reg [31:0] in0, in1, in2, in3, in4, in5, in6, in7;
	 wire [31:0] out_node;

	PE PE_mod(
		.clk(clk), .rst(rst), .i_vld(i_vld),
		.in0(in0), .in1(in1), .in2(in2), .in3(in3), .in4(in4), .in5(in5), .in6(in6), .in7(in7), 
		.out_node(out_node)
	);
		
    always #5 clk = ~clk;
    
    initial begin    
        clk = 0;
        rst = 1;
        i_vld = 0;
        #10 rst = 0;
		  
		  in0 = 32'h3a83126f; in1 = 32'h3aa1be2b; in2 = 32'h3e6f9db2; in3 = 32'h3d9fbe77; 
		  in4 = 32'h3f65a1cb; in5 = 32'h3da0663c; in6 = 32'h3d9fbe77; in7 = 32'h00000000;
		  i_vld = 1;
        
			    
    end
    
endmodule
