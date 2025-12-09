use crate::*;

pub fn format_float(num: float, total_width: usize) -> String {
    debug_assert!(total_width > 0);

    // Format with significant digits in scientific notation
    let formatted = format!("{:.*e}", total_width - 1, num);

    // Convert back to float to remove unnecessary trailing zeros
    let formatted_num = formatted.parse::<float>().unwrap_or(num);

    // Determine precision based on integer part length
    let precision = num.trunc().to_string().len() + 1;

    // Effective total width of the formatted number
    let total_width = formatted_num.to_string().len();

    // If integer part consumes all width, return directly
    if precision >= total_width {
        format!("{formatted_num}")
    } else {
        // Otherwise format normally with computed width + precision
        format!(
            "{:width$.precision$}",
            num,
            width = total_width,
            precision = total_width - precision
        )
    }
}

pub fn format_with_separator<T: CCfloat>(n: T, sep: char) -> String {
    // Convert and round
    let n = n.float();
    let n = round(n, 3); // Round to 3 decimal places

    // Extract integer part
    let integer_part = n.trunc().i64();
    let formatted_integer = integer_part.to_string();

    // Extract decimal part (if any)
    let n_string = n.to_string();
    let formatted_decimal = if n_string.contains('.') {
        format!(".{}", n_string.split('.').collect::<Vec<&str>>()[1])
    } else {
        String::new()
    };

    // Insert thousand separators
    let mut formatted = String::new();
    let len = formatted_integer.len();

    for (i, c) in formatted_integer.chars().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) {
            formatted.push(sep);
        }
        formatted.push(c);
    }

    // Return combined number
    if formatted.len() <= 3 {
        format!("{formatted}{formatted_decimal}")
    } else {
        formatted.to_string()
    }
}

pub fn scientific_notation<T: CCfloat>(n: T, precision: usize) -> String {
    let n = n.float();

    // Special case for zero
    if n == 0.0 {
        return "0".to_string();
    }

    // Format in scientific notation with the requested precision
    let formatted = format!("{n:.precision$E}");

    // Split into mantissa and exponent
    let parts: Vec<&str> = formatted.split('E').collect();
    let exponent: i32 = parts[1].parse().unwrap();

    // Exponent formatting: always at least 2 digits
    let exp_str = format!("{exponent:02}");

    // Add a '+' sign for positive exponents (Rust omits it by default)
    let sign = if exponent >= 0 { "+" } else { "" };

    // Rebuild final scientific notation string
    format!("{}E{}{}", parts[0], sign, exp_str)
}
