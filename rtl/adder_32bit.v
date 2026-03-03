`timescale 1ns / 1ps

module adder_32bit(
    input clk,
    inout rst,
    input [31:0] i_a,
    input [31:0] i_b,
    input i_vld,
    output reg [31:0] o_res,
    output reg o_res_vld,
    output reg overflow );

    wire [7:0] shift;
    wire [23:0] al_man_a, al_man_b;
    wire sign_a, sign_b, sign_res;
    wire [7:0] exp_a, exp_b, exp_res;
    wire [23:0] man_a, man_b, man_res;
    wire [24:0] res;
    wire operation_overflow;

    //unpacking the inputs - 32bit single precision
    assign sign_a = i_a[31];
    assign sign_b = i_b[31];
    assign exp_a = i_a[30:23];
    assign exp_b = i_b[30:23];
    assign man_a = (exp_a == 8'b0) ? {1'b0, i_a[22:0]} : {1'b1, i_a[22:0]}; //adding explicit 1 for normalized and 0 for denormalised numbers
    assign man_b = (exp_b == 8'b0) ? {1'b0, i_b[22:0]} : {1'b1, i_b[22:0]};

    wire is_nan_a = ((exp_a == 8'b11111111) && (man_a[22:0] != 0)); //only 23 bits of mantissa must be checked excluding the added 1
    wire is_nan_b = ((exp_b == 8'b11111111) && (man_b[22:0] != 0));
    wire is_inf_a = ((exp_a == 8'b11111111) && (man_a[22:0] == 0));
    wire is_inf_b = ((exp_b == 8'b11111111) && (man_b[22:0] == 0));
    wire is_zero_a = ((i_a[30:0] == 0)); //sign bit doesnt contribute anything
    wire is_zero_b = ((i_b[30:0] == 0));

    CompareAndShift32Bit u_CompareAndShift32Bit (
        .exp_a(exp_a),          //input
        .exp_b(exp_b),          //input
        .man_a(man_a),          //input
        .man_b(man_b),          //input
        .al_man_a(al_man_a),    //wire out
        .al_man_b(al_man_b),    //wire out
        .shift(shift)           //wire out
    );

    Addition32Bit u_Addition32Bit (
        .sign_a(sign_a),        //input
        .sign_b(sign_b),        //input
        .a(al_man_a),           //wire in from submodule
        .b(al_man_b),           //wire in from submodule
        .res(res),              //wire out
        .sign_res(sign_res)     //wire out
    );

    Normalization32Bit u_Normalization32Bit (
        .res(res),                                  //wire in from submodule
        .exp_base((exp_a > exp_b) ? exp_a : exp_b), //input
        .man_res(man_res),                          //wire out
        .exp_res(exp_res),                          //wire out
        .overflow(operation_overflow)               //wire out
    );

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            o_res_vld <= 1'b0;
            o_res <= 32'b0;
            overflow <=0;
        end
        else if (i_vld) begin
            if (is_nan_a || is_nan_b || (is_inf_a && is_inf_b && (sign_a != sign_b))) begin
                o_res <= 32'h7FC00000; // Quiet NaN
                overflow <= 1'b1;
            end 
            else if (is_inf_a || is_inf_b) begin
                o_res <= is_inf_a ? i_a : i_b; // Preserve Infinity
                overflow <= 1'b1;
            end 
            else if (is_zero_a && is_zero_b) begin
                o_res <= 32'b0; // Zero
            end 
            else begin
            o_res <= {sign_res, exp_res, man_res[22:0]};
            o_res_vld <= 1'b1;
            overflow <= operation_overflow;
            end
            o_res_vld <= 1'b1;
        end 
        else begin
            o_res_vld <= 1'b0;
            o_res <= 32'b0;
        end
    end

endmodule