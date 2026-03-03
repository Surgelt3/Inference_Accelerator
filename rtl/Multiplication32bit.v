// Taken from: https://github.com/Rishadd/32bit-FPU/blob/main/rtl/Multiplication32Bit.v

module Multiplication32bit(
    input [23:0] man_a,
    input [23:0] man_b,
    input [7:0] exp_a,
    input [7:0] exp_b,
    output [22:0] final_mantissa,
    output [7:0] final_exp,
    output overflow
);

    wire [47:0] product = man_a * man_b; // Multiply mantissas max (24x24 = 48 bits)

    // Normalization
    wire norm_shift = product[47]; //checking if number is normal or denormal
    wire [22:0] man_norm = norm_shift ? product[46:24] : product[45:23]; //nomalizing the mantissa product
    wire guard_bit = norm_shift ? product[23] : product[22]; //guard bit is the one after 10 bits from decimal point
    wire sticky_bit = norm_shift ? |product[22:0] : |product[21:0]; //checking if there is a 1 present after guard bit
    wire round = guard_bit & sticky_bit; //round = 1 if guard bit is 1 and sticky bit is 1

    // Add rounding (could overflow mantissa → handled later)
    wire [23:0] rounded = {1'b0, man_norm} + round; //24 bit wire to account for overflow
    wire carry = rounded[23]; // For overflow detection after rounding
    assign final_mantissa = carry ? rounded[23:1] : rounded[22:0]; //final 23 bits of the mantissa

    //Exponent computation
    wire [9:0] exp_raw = exp_a + exp_b - 8'd127 + norm_shift + carry; //127 is subtracted because it is added twice
    assign final_exp = exp_raw[7:0];

    assign overflow = exp_raw > 8'd254;

endmodule