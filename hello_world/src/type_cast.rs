use castaway::cast as cast_special;
use funty::Fundamental;
pub trait CustomCast {
    fn i32(&self) -> i32;
}
impl<T: Fundamental> CustomCast for T
where
    i32: easy_cast::Conv<T>,
{
    fn i32(&self) -> i32 {
        if let Ok(value) = cast_special!(self, &bool) {
            *value as i32
        } else if let Ok(value) = cast_special!(self, &i32) {
            *value
        } else {
            use easy_cast::*;
            (*self).cast()
        }
    }
}
