`timescale 1ns / 1ps

module Addition32Bit (
    input sign_a,
    input sign_b,
    input [23:0] a,b,
    output reg [24:0] res,
    output reg sign_res
    );
    
    always @(*) begin
        if (sign_a == sign_b) begin
            sign_res = sign_a;   
            res = a+b;
        end
        else if (a >= b) begin
            res = a - b;
            sign_res = sign_a;
        end
        else begin
            res = b - a;
            sign_res = sign_b;
        end
    end
    
endmodule