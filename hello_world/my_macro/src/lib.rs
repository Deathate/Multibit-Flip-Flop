mod debug;
mod release;
pub trait CCbool {
    fn bool(&self) -> bool;
}
pub trait CCi32 {
    fn i32(&self) -> i32;
}
pub trait CCi64 {
    fn i64(&self) -> i64;
}
pub trait CCusize {
    fn usize(&self) -> usize;
}
pub trait CCf32 {
    fn f32(&self) -> f32;
}
pub trait CCf64 {
    fn f64(&self) -> f64;
}
