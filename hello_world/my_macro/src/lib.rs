use cast::*;
use castaway::cast as cast_special;
use duplicate::duplicate_item;
use easy_cast::Conv;
use std::fmt::Debug;
pub trait CC {
    fn i32(&self) -> i32;
    fn i64(&self) -> i64;
    fn usize(&self) -> usize;
}
impl CC for bool {
    fn i32(&self) -> i32 {
        *self as i32
    }
    fn i64(&self) -> i64 {
        (*self) as i64
    }
    fn usize(&self) -> usize {
        (*self) as usize
    }
}
impl CC for i32 {
    fn i32(&self) -> i32 {
        *self
    }
    fn i64(&self) -> i64 {
        i64::conv(*self)
    }
    fn usize(&self) -> usize {
        usize::conv(*self)
    }
}
impl CC for i64 {
    fn i32(&self) -> i32 {
        i32::conv(*self)
    }
    fn i64(&self) -> i64 {
        *self
    }
    fn usize(&self) -> usize {
        usize::conv(*self)
    }
}
impl CC for usize {
    fn i32(&self) -> i32 {
        i32::conv(*self)
    }
    fn i64(&self) -> i64 {
        i64::conv(*self)
    }
    fn usize(&self) -> usize {
        *self
    }
}
#[duplicate_item(
    type_name;
    [f32];
    [f64];
  )]
impl CC for type_name {
    fn i32(&self) -> i32 {
        i32(*self).unwrap()
    }
    fn i64(&self) -> i64 {
        i64(*self).unwrap()
    }
    fn usize(&self) -> usize {
        usize(*self).unwrap()
    }
}
pub trait CCf32 {
    fn f32(&self) -> f32;
}
impl CCf32 for bool {
    fn f32(&self) -> f32 {
        if *self {
            1.0
        } else {
            0.0
        }
    }
}
#[duplicate_item(
    type_name;
    [i64];
    [i32];
    [usize];
  )]
impl CCf32 for type_name {
    fn f32(&self) -> f32 {
        #[cfg(feature = "always_assert")]
        return f32::conv(*self);
        #[cfg(not(feature = "always_assert"))]
        return (*self) as f32;
    }
}
impl CCf32 for f64 {
    fn f32(&self) -> f32 {
        #[cfg(feature = "always_assert")]
        return f32(*self).unwrap();
        #[cfg(not(feature = "always_assert"))]
        return (*self) as f32;
    }
}
pub trait CCf64 {
    fn f64(&self) -> f64;
}
impl CCf64 for bool {
    fn f64(&self) -> f64 {
        if *self {
            1.0
        } else {
            0.0
        }
    }
}
#[duplicate_item(
    type_name;
    [i64];
    [i32];
    [usize];
    [f32];
    [f64];
  )]
impl CCf64 for type_name {
    fn f64(&self) -> f64 {
        #[cfg(feature = "always_assert")]
        return f64::conv(*self);
        #[cfg(not(feature = "always_assert"))]
        return (*self) as f64;
    }
}
