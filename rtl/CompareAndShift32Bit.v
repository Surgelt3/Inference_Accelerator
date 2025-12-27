`timescale 1ns / 1ps

module CompareAndShift32Bit (
    input [7:0] exp_a,
    input [7:0] exp_b,
    input [23:0] man_a,
    input [23:0] man_b,
    output reg [23:0] al_man_a,
    output reg [23:0] al_man_b,
    output reg [7:0] shift
);
    always @(*) begin
        if (exp_a > exp_b) begin
            shift = exp_a - exp_b;
            al_man_a = man_a;
            al_man_b = man_b >> shift;
        end else begin
            shift = exp_b - exp_a;
            al_man_a = man_a >> shift;
            al_man_b = man_b;
        end
    end
endmodule