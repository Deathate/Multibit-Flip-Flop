use castaway::cast as cast_special;
use easy_cast::Conv;
pub trait CustomCast {
    fn i32(&self) -> i32;
    fn i64(&self) -> i64;
    fn usize(&self) -> usize;
    fn f32(&self) -> f32;
}
impl<T: Copy> CustomCast for T
where
    i32: easy_cast::Conv<T>,
    i64: easy_cast::Conv<T>,
    usize: easy_cast::Conv<T>,
    f32: easy_cast::Conv<T>,
{
    fn i32(&self) -> i32 {
        if let Ok(value) = cast_special!(self, &bool) {
            *value as i32
        } else if let Ok(value) = cast_special!(self, &i32) {
            *value
        } else {
            i32::conv(*self)
        }
    }
    fn i64(&self) -> i64 {
        if let Ok(value) = cast_special!(self, &bool) {
            *value as i64
        } else if let Ok(value) = cast_special!(self, &i64) {
            *value
        } else {
            i64::conv(*self)
        }
    }
    fn usize(&self) -> usize {
        if let Ok(value) = cast_special!(self, &bool) {
            *value as usize
        } else if let Ok(value) = cast_special!(self, &usize) {
            *value
        } else {
            // use easy_cast::*;
            usize::conv(*self)
        }
    }
    fn f32(&self) -> f32 {
        if let Ok(value) = cast_special!(self, &f32) {
            *value
        } else {
            f32::conv(*self)
        }
    }
}
