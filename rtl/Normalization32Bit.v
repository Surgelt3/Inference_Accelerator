`timescale 1ns / 1ps

module Normalization32Bit (
    input [24:0] res,
    input [7:0] exp_base,
    output reg [23:0] man_res,
    output reg [7:0] exp_res,
    output reg overflow
);
    reg [24:0] normalized_result;
    integer i;
    reg continue_shift;

    always @(*) begin
        normalized_result = res;
        exp_res = exp_base;
        overflow = 0;

        if (normalized_result == 0) begin //explicitly handle 0 case for lower runtime
            man_res = 0;
            exp_res = 0;
            overflow = 0;
        end 
        else begin
            if (normalized_result[24]) begin // If MSB is 1, shift right once and increment exponent
                normalized_result = normalized_result >> 1;
                exp_res = exp_base + 1;
            end 
            else begin // Shift left until bit 23 is 1 or exponent reaches 0, max 25 times
                repeat (24) begin                
                    if(normalized_result[23] == 0) begin
                        normalized_result = normalized_result << 1;
                        exp_res = exp_res - 1;
                    end
		        end		    
            man_res = normalized_result[23:0];
                if(exp_res==8'b11111111) begin
                     overflow=1;
                end
            end
        end
    end
endmodule
