// from https://github.com/Rishadd/32bit-FPU


module multiplier_32bit (
    input         clk,
    input         rst,
    input  [31:0] i_a,
    input  [31:0] i_b,
    input         i_vld,
    output reg [31:0] o_res,
    output reg    o_res_vld,
    output reg    overflow
);

	// ========================== FLOATING-POINT DECODING ========================
	wire        sign_a, sign_b;
	wire [7:0]  exp_a, exp_b;
	wire [23:0] man_a, man_b;

	assign sign_a = i_a[31];
	assign sign_b = i_b[31];
	assign exp_a  = i_a[30:23];
	assign exp_b  = i_b[30:23];
	assign man_a  = (exp_a == 8'b0) ? {1'b0, i_a[22:0]} : {1'b1, i_a[22:0]};
	assign man_b  = (exp_b == 8'b0) ? {1'b0, i_b[22:0]} : {1'b1, i_b[22:0]};

	// ========================== SPECIAL CASES DETECTION ========================
	wire is_nan_a  = (exp_a == 8'hFF) && (|i_a[22:0]);
	wire is_nan_b  = (exp_b == 8'hFF) && (|i_b[22:0]);
	wire is_inf_a  = (exp_a == 8'hFF) && (i_a[22:0] == 23'b0);
	wire is_inf_b  = (exp_b == 8'hFF) && (i_b[22:0] == 23'b0);
	wire is_zero_a = (i_a[30:0] == 31'b0);
	wire is_zero_b = (i_b[30:0] == 31'b0);

	// ========================== PIPELINE STAGE 1 ==============================
	reg         i_vld_r;
	reg         sign_a_r, sign_b_r;
	reg  [7:0]  exp_a_r, exp_b_r;
	reg  [23:0] man_a_r, man_b_r;
	reg         is_nan_a_r, is_nan_b_r;
	reg         is_inf_a_r, is_inf_b_r;
	reg         is_zero_a_r, is_zero_b_r;

	always @(posedge clk or posedge rst) begin
		 if (rst) begin
			  i_vld_r     <= 1'b0;
			  sign_a_r    <= 1'b0; sign_b_r    <= 1'b0;
			  exp_a_r     <= 8'b0; exp_b_r     <= 8'b0;
			  man_a_r     <= 24'b0; man_b_r    <= 24'b0;
			  is_nan_a_r  <= 1'b0; is_nan_b_r  <= 1'b0;
			  is_inf_a_r  <= 1'b0; is_inf_b_r  <= 1'b0;
			  is_zero_a_r <= 1'b0; is_zero_b_r <= 1'b0;
		 end else begin
			  i_vld_r     <= i_vld;
			  sign_a_r    <= sign_a;    sign_b_r    <= sign_b;
			  exp_a_r     <= exp_a;     exp_b_r     <= exp_b;
			  man_a_r     <= man_a;     man_b_r     <= man_b;
			  is_nan_a_r  <= is_nan_a;  is_nan_b_r  <= is_nan_b;
			  is_inf_a_r  <= is_inf_a;  is_inf_b_r  <= is_inf_b;
			  is_zero_a_r <= is_zero_a; is_zero_b_r <= is_zero_b;
		 end
	end

	// ========================== MULTIPLICATION CORE ============================
	wire [47:0] product_raw;  // 24x24 = 48-bit product
	wire        sign_res;
	wire [7:0]  exp_raw;
	wire [7:0]  exp_calc;
	wire        overflow_calc;
	wire [23:0] man_rounded;
	wire [22:0] man_final;

	assign sign_res = sign_a_r ^ sign_b_r;  // XOR of signs
	assign product_raw = man_a_r * man_b_r; // Mantissa multiplication

	// Exponent calculation: exp1 + exp2 - bias(127)
	assign exp_raw = exp_a_r + exp_b_r - 8'd127;

	// Normalization detection
	wire product_msb = product_raw[47];  // Check if result >= 2.0
	wire [47:0] product_shifted = product_msb ? product_raw >> 1 : product_raw;
	assign exp_calc = product_msb ? exp_raw + 1 : exp_raw;

	// Rounding: round to nearest even (simplified)
	wire guard_bit    = product_shifted[23];
	wire sticky_bit   = |product_shifted[22:0];
	wire round_up     = guard_bit && (sticky_bit || product_shifted[24]);

	// Apply rounding
	assign man_rounded = product_shifted[47:24] + round_up;
	wire carry_round = man_rounded[23];  // Check if rounding caused overflow

	// Final mantissa and exponent
	assign man_final = carry_round ? man_rounded[23:1] : man_rounded[22:0];
	wire [7:0] exp_final = carry_round ? exp_calc + 1 : exp_calc;

	// Overflow detection
	assign overflow_calc = (exp_final == 8'hFF) || (exp_raw > 8'd254);

	// ========================== PIPELINE STAGE 2 ==============================
	reg         i_vld_r2;
	reg         sign_res_r;
	reg [7:0]   exp_final_r;
	reg [22:0]  man_final_r;
	reg         overflow_calc_r;
	reg         is_nan_r, is_inf_r, is_zero_r;

	always @(posedge clk or posedge rst) begin
		 if (rst) begin
			  i_vld_r2        <= 1'b0;
			  sign_res_r      <= 1'b0;
			  exp_final_r     <= 8'b0;
			  man_final_r     <= 23'b0;
			  overflow_calc_r <= 1'b0;
			  is_nan_r        <= 1'b0;
			  is_inf_r        <= 1'b0;
			  is_zero_r       <= 1'b0;
		 end else begin
			  i_vld_r2        <= i_vld_r;
			  sign_res_r      <= sign_res;
			  exp_final_r     <= exp_final;
			  man_final_r     <= man_final;
			  overflow_calc_r <= overflow_calc;
			  // Combine special cases
			  is_nan_r        <= is_nan_a_r || is_nan_b_r || 
									  ((is_inf_a_r && is_zero_b_r) || (is_zero_a_r && is_inf_b_r));
			  is_inf_r        <= (is_inf_a_r || is_inf_b_r) && !is_nan_r;
			  is_zero_r       <= is_zero_a_r || is_zero_b_r;
		 end
	end

	// ========================== OUTPUT GENERATION ==============================
	always @(posedge clk or posedge rst) begin
		 if (rst) begin
			  o_res     <= 32'b0;
			  o_res_vld <= 1'b0;
			  overflow  <= 1'b0;
		 end else if (i_vld_r2) begin
			  // Handle special cases
			  if (is_nan_r) begin
					o_res <= 32'h7FC0_0000;  // Quiet NaN
					overflow <= 1'b1;
			  end
			  else if (is_inf_r) begin
					o_res <= {sign_res_r, 8'hFF, 23'b0};  // Infinity
					overflow <= 1'b1;
			  end
			  else if (is_zero_r) begin
					o_res <= {sign_res_r, 31'b0};  // Signed zero
					overflow <= 1'b0;
			  end
			  else if (overflow_calc_r) begin
					o_res <= {sign_res_r, 8'hFF, 23'b0};  // Overflow to infinity
					overflow <= 1'b1;
			  end
			  else begin
					o_res <= {sign_res_r, exp_final_r, man_final_r};
					overflow <= 1'b0;
			  end
			  o_res_vld <= 1'b1;
		 end else begin
			  o_res_vld <= 1'b0;
			  o_res     <= 32'b0;
			  overflow  <= 1'b0;
		 end
	end

endmodule
