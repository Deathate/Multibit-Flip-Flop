#![feature(specialization)]
#![feature(negative_impls)]
use castaway::cast as cast_special;
use duplicate::duplicate_item;
use easy_cast::Conv;
pub trait IsBool {}
pub trait NotBool {}
impl IsBool for bool {}
impl<T: IsBool> !NotBool for T {}
#[duplicate_item(
    name  type_name func_name;
    [ CustomCastI32 ]    [ i32 ] [i32];
    [ CustomCastI64 ]    [ i64 ] [i64];
    [ CustomCastUsize ]   [ usize ] [usize];
    [ CustomCastF32 ]    [ f32 ] [f32];
    [ CustomCastF64 ]    [ f64 ] [f64];
  )]
pub trait name {
    fn func_name(&self) -> type_name;
}
#[duplicate_item(
    name  type_name func_name;
    [ CustomCastI32 ]    [ i32 ] [i32];
    [ CustomCastI64 ]    [ i64 ] [i64];
    [ CustomCastUsize ]   [ usize ] [usize];
  )]
impl<T> name for T
where
    T: Copy + easy_cast::Cast<type_name>,
{
    fn func_name(&self) -> type_name {
        if let Ok(value) = cast_special!(self, &bool) {
            *value as type_name
        } else if let Ok(value) = cast_special!(self, &type_name) {
            *value
        } else {
            (*self).cast()
        }
    }
}
// ----------------

// impl CustomCastI32 for bool {
//     fn i32(&self) -> i32 {
//         *self as i32
//     }
// }
// impl<T: Copy + NotBool + easy_cast::Cast<i32>> CustomCastI32 for T {
//     default fn i32(&self) -> i32 {
//         if let Ok(value) = cast_special!(self, &i32) {
//             *value
//         } else {
//             (*self).cast()
//         }
//     }
// }

#[duplicate_item(
    name  type_name func_name;
    [ CustomCastF32 ]    [ f32 ] [f32];
    [ CustomCastF64 ]    [ f64 ] [f64];
  )]
impl<T> name for T
where
    T: Copy + easy_cast::Cast<type_name>,
{
    fn func_name(&self) -> type_name {
        if let Ok(value) = cast_special!(self, &type_name) {
            *value
        } else {
            (*self).cast()
        }
    }
}
