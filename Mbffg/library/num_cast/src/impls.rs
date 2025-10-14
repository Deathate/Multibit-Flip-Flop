use crate::*;
use easy_cast::{Conv, ConvApprox, ConvFloat};

// CCbool
impl CCbool for bool {
    fn bool(&self) -> bool {
        *self
    }
}
#[duplicate_item(
    type_name;
    [i32];
    [i64];
    [u32];
    [u64];
    [usize];
    [f32];
    [f64];
)]
impl CCbool for type_name {
    fn bool(&self) -> bool {
        *self != 0 as type_name
    }
}

// CCi32
impl CCi32 for bool {
    fn i32(&self) -> i32 {
        if *self {
            return 1;
        } else {
            return 0;
        }
    }
}
#[duplicate_item(
    type_name;
    [i32];
    [i64];
    [u32];
    [u64];
    [usize];
)]
impl CCi32 for type_name {
    fn i32(&self) -> i32 {
        i32::conv(*self)
    }
}
#[duplicate_item(
    type_name;
    [f32];
    [f64];
)]
impl CCi32 for type_name {
    fn i32(&self) -> i32 {
        i32::conv_trunc(*self)
    }
}

// CCi64
impl CCi64 for bool {
    fn i64(&self) -> i64 {
        if *self {
            return 1;
        } else {
            return 0;
        }
    }
}
#[duplicate_item(
    type_name;
    [i32];
    [i64];
    [u32];
    [u64];
    [usize];
)]
impl CCi64 for type_name {
    fn i64(&self) -> i64 {
        i64::conv(*self)
    }
}
#[duplicate_item(
    type_name;
    [f32];
    [f64];
)]
impl CCi64 for type_name {
    fn i64(&self) -> i64 {
        i64::conv_trunc(*self)
    }
}

//CCu32
impl CCu32 for bool {
    fn u32(&self) -> u32 {
        if *self {
            return 1;
        } else {
            return 0;
        }
    }
}
#[duplicate_item(
    type_name;
    [i32];
    [i64];
    [u32];
    [u64];
    [usize];
)]
impl CCu32 for type_name {
    fn u32(&self) -> u32 {
        u32::conv(*self)
    }
}
#[duplicate_item(
    type_name;
    [f32];
    [f64];
)]
impl CCu32 for type_name {
    fn u32(&self) -> u32 {
        u32::conv_trunc(*self)
    }
}

// CCu64
impl CCu64 for bool {
    fn u64(&self) -> u64 {
        if *self {
            return 1;
        } else {
            return 0;
        }
    }
}
#[duplicate_item(
    type_name;
    [i32];
    [i64];
    [u32];
    [u64];
    [usize];
)]
impl CCu64 for type_name {
    fn u64(&self) -> u64 {
        u64::conv(*self)
    }
}
#[duplicate_item(
    type_name;
    [f32];
    [f64];
)]
impl CCu64 for type_name {
    fn u64(&self) -> u64 {
        u64::conv_trunc(*self)
    }
}

// CCusize
impl CCusize for bool {
    fn usize(&self) -> usize {
        if *self {
            return 1;
        } else {
            return 0;
        }
    }
}
#[duplicate_item(
    type_name;
    [i32];
    [i64];
    [u32];
    [u64];
    [usize];
)]
impl CCusize for type_name {
    fn usize(&self) -> usize {
        usize::conv(*self)
    }
}
#[duplicate_item(
    type_name;
    [f32];
    [f64];
)]
impl CCusize for type_name {
    fn usize(&self) -> usize {
        usize::conv_trunc(*self)
    }
}

// CCf32
impl CCf32 for bool {
    fn f32(&self) -> f32 {
        if *self {
            return 1.0;
        } else {
            return 0.0;
        }
    }
}
#[duplicate_item(
    type_name;
    [i32];
    [i64];
    [u32];
    [u64];
    [usize];
    [f32];
)]
impl CCf32 for type_name {
    fn f32(&self) -> f32 {
        f32::conv(*self)
    }
}
impl CCf32 for f64 {
    fn f32(&self) -> f32 {
        f32::conv_approx(*self)
    }
}

// CCf64
impl CCf64 for bool {
    fn f64(&self) -> f64 {
        if *self {
            return 1.0;
        } else {
            return 0.0;
        }
    }
}
#[duplicate_item(
    type_name;
    [i32];
    [i64];
    [u32];
    [u64];
    [usize];
    [f32];
    [f64];
)]
impl CCf64 for type_name {
    fn f64(&self) -> f64 {
        f64::conv(*self)
    }
}
#[duplicate_item(
    type_name;
    [&i32];
    [&i64];
    [&u32];
    [&u64];
    [&usize];
    [&f32];
    [&f64];
)]
impl CCf64 for type_name {
    fn f64(&self) -> f64 {
        (*self).f64()
    }
}
#[duplicate_item(
    type_name;
    [i32];
    [i64];
    [u32];
    [u64];
    [usize];
    [f32];
    [f64];
)]
impl CCint for type_name {
    fn int(&self) -> int {
        #[cfg(feature = "integer_as_i64")]
        return self.i64();
        #[cfg(not(feature = "integer_as_i64"))]
        return self.i32();
    }
}
#[duplicate_item(
    type_name;
    [i32];
    [i64];
    [u32];
    [u64];
    [usize];
    [f32];
    [f64];
)]
impl CCuint for type_name {
    fn uint(&self) -> uint {
        #[cfg(feature = "integer_as_i64")]
        return self.u64();
        #[cfg(not(feature = "integer_as_i64"))]
        return self.u32();
    }
}
#[duplicate_item(
    type_name;
    [i32];
    [i64];
    [u32];
    [u64];
    [usize];
    [f32];
    [f64];
)]
impl CCfloat for type_name {
    fn float(&self) -> float {
        #[cfg(feature = "float_as_f64")]
        return self.f64();
        #[cfg(not(feature = "float_as_f64"))]
        return self.f32();
    }
}
