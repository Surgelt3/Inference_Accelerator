// from https://github.com/Rishadd/32bit-FPU/blob/main/rtl/multiplier_32bit.v

`timescale 1ns / 1ps

module multiplier_32bit (
    input clk,
    input rst,
    input [31:0] i_a,
    input [31:0] i_b,
    input i_vld,
    output reg [31:0] o_res,
    output reg o_res_vld,
    output reg overflow );

    // Internal wires
    wire sign_a, sign_b, sign_res;
    wire [7:0] exp_a, exp_b, final_exp;
    wire [22:0] final_mantissa;
    wire result_overflow;
    wire [23:0] man_a, man_b;

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

    // Compute result sign (XOR of inout signs)
    assign sign_res = sign_a ^ sign_b;

    // Core multiplication and normalization
    Multiplication32bit u_Multiplication32bit (
        .man_a(man_a),
        .man_b(man_b),
        .exp_a(exp_a),
        .exp_b(exp_b),
        .final_mantissa(final_mantissa),
        .final_exp(final_exp),
        .overflow(result_overflow) );

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            o_res_vld <= 1'b0;
            o_res <= 32'b0;
            overflow <=0;
        end
        else if (i_vld) begin
            if (is_nan_a || is_nan_b || ((is_inf_a && is_zero_b) || (is_zero_a && is_inf_b))) begin
                o_res <= 32'h7FC00000; // NaN
                overflow <= 1'b1;
            end else if (is_inf_a || is_inf_b) begin
                o_res <= {sign_res, 8'hFF, 23'b0}; // Infinity
                overflow <= 1'b1;
            end else if (is_zero_a || is_zero_b) begin
                o_res <= {sign_res, 31'b0}; // Zero
                overflow <= 1'b0;
            end else if (result_overflow) begin
                o_res <= {sign_res, 8'hFF, 23'b0}; // Overflow → Inf
                overflow <= 1'b1;
            end else begin
                o_res <= {sign_res, final_exp, final_mantissa};
                overflow <= 1'b0;
            end
            o_res_vld <= 1'b1;
        end 
        else begin
            o_res_vld <= 1'b0;
            o_res <= 32'b0;
        end
    end

endmodule