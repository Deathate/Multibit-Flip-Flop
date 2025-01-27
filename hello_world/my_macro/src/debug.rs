#![cfg(feature = "always_assert")]
use crate::*;
use cast::*;
use castaway::cast as cast_special;
use easy_cast::Conv;

fn print_error<
    T: std::fmt::Display + std::fmt::LowerExp,
    U: std::fmt::Display + std::fmt::LowerExp,
>(
    min: U,
    max: U,
    value: T,
) -> String {
    format!(
        "cast x: {} to {}: expected {:.6e} <= x <= {:.6e}, found x = {:.6e}",
        std::any::type_name::<T>(),
        std::any::type_name::<U>(),
        min,
        max,
        value
    )
}

// CCbool
impl<T> CCbool for T {
    fn bool(&self) -> bool {
        if let Ok(value) = cast_special!(self, &bool) {
            return *value;
        } else if let Ok(value) = cast_special!(self, &i64) {
            return *value != 0;
        } else if let Ok(value) = cast_special!(self, &i32) {
            return *value != 0;
        } else if let Ok(value) = cast_special!(self, &usize) {
            return *value != 0;
        } else if let Ok(value) = cast_special!(self, &f32) {
            return *value != 0.0;
        } else if let Ok(value) = cast_special!(self, &f64) {
            return *value != 0.0;
        }
        panic!("Invalid type");
    }
}

// CCi32
impl<T> CCi32 for T {
    fn i32(&self) -> i32 {
        if let Ok(value) = cast_special!(self, &bool) {
            if *value {
                return 1;
            } else {
                return 0;
            }
        } else if let Ok(value) = cast_special!(self, &i64) {
            return i32::conv(*value);
        } else if let Ok(value) = cast_special!(self, &i32) {
            return *value;
        } else if let Ok(value) = cast_special!(self, &usize) {
            return i32::conv(*value);
        } else if let Ok(value) = cast_special!(self, &f32) {
            return i32(*value).expect(print_error(i32::MIN, i32::MAX, *value).as_str());
        } else if let Ok(value) = cast_special!(self, &f64) {
            return i32(*value).expect(print_error(i32::MIN, i32::MAX, *value).as_str());
        }
        panic!("Invalid type");
    }
}

// CCi64
impl<T> CCi64 for T {
    fn i64(&self) -> i64 {
        if let Ok(value) = cast_special!(self, &bool) {
            if *value {
                return 1;
            } else {
                return 0;
            }
        } else if let Ok(value) = cast_special!(self, &i64) {
            return *value;
        } else if let Ok(value) = cast_special!(self, &i32) {
            return i64::conv(*value);
        } else if let Ok(value) = cast_special!(self, &usize) {
            return i64::conv(*value);
        } else if let Ok(value) = cast_special!(self, &f32) {
            return i64(*value).expect(print_error(i64::MIN, i64::MAX, *value).as_str());
        } else if let Ok(value) = cast_special!(self, &f64) {
            return i64(*value).expect(print_error(i64::MIN, i64::MAX, *value).as_str());
        }
        panic!("Invalid type");
    }
}

// CCusize
impl<T> CCusize for T {
    fn usize(&self) -> usize {
        if let Ok(value) = cast_special!(self, &bool) {
            if *value {
                return 1;
            } else {
                return 0;
            }
        } else if let Ok(value) = cast_special!(self, &i64) {
            return usize::conv(*value);
        } else if let Ok(value) = cast_special!(self, &i32) {
            return usize::conv(*value);
        } else if let Ok(value) = cast_special!(self, &usize) {
            return *value;
        } else if let Ok(value) = cast_special!(self, &f32) {
            return usize(*value).expect(print_error(usize::MIN, usize::MAX, *value).as_str());
        } else if let Ok(value) = cast_special!(self, &f64) {
            return usize(*value).expect(print_error(usize::MIN, usize::MAX, *value).as_str());
        }
        panic!("Invalid type");
    }
}

// CCf32
impl<T> CCf32 for T {
    fn f32(&self) -> f32 {
        if let Ok(value) = cast_special!(self, &bool) {
            if *value {
                return 1.0;
            } else {
                return 0.0;
            }
        } else if let Ok(value) = cast_special!(self, &i64) {
            return f32::conv(*value);
        } else if let Ok(value) = cast_special!(self, &i32) {
            return f32::conv(*value);
        } else if let Ok(value) = cast_special!(self, &usize) {
            return f32::conv(*value);
        } else if let Ok(value) = cast_special!(self, &f32) {
            return *value;
        } else if let Ok(value) = cast_special!(self, &f64) {
            return f32(*value).expect(print_error(f32::MIN, f32::MAX, *value).as_str());
        }
        panic!("Invalid type");
    }
}

// CCf64
impl<T> CCf64 for T {
    fn f64(&self) -> f64 {
        if let Ok(value) = cast_special!(self, &bool) {
            if *value {
                return 1.0;
            } else {
                return 0.0;
            }
        } else if let Ok(value) = cast_special!(self, &i64) {
            return f64::conv(*value);
        } else if let Ok(value) = cast_special!(self, &i32) {
            return f64::conv(*value);
        } else if let Ok(value) = cast_special!(self, &usize) {
            return f64::conv(*value);
        } else if let Ok(value) = cast_special!(self, &f32) {
            return f64::conv(*value);
        } else if let Ok(value) = cast_special!(self, &f64) {
            return *value;
        }
        panic!("Invalid type {}", std::any::type_name::<T>());
    }
}
