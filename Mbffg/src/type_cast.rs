use castaway::cast as cast_special;
use duplicate::duplicate_item;
use easy_cast::Conv;
// trait MyTrait {
//     fn do_something(&self);
// }

// // Default implementation
// impl<T> MyTrait for T
// where
//     T: Copy,
// {
//     default fn do_something(&self) {
//         println!("Default implementation");
//     }
// }

// // Specialized implementation for a specific type
// impl MyTrait for u32 {
//     fn do_something(&self) {
//         println!("Specialized implementation for u32: {}", self);
//     }
// }

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
// #[duplicate_item(
//     name  type_name func_name;
//     [ CustomCastI32 ]    [ i32 ] [i32];
//     [ CustomCastI64 ]    [ i64 ] [i64];
//     [ CustomCastUsize ]   [ usize ] [usize];
//   )]
// impl name for bool
// where
//     T: Copy,
// {
//     fn func_name(&self) -> type_name {
//         *value as type_name
//     }
// }
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
// // use main::NotBool;
// impl<T: Copy + easy_cast::Cast<i32>> CustomCastI32 for T {
//     default fn i32(&self) -> i32 {
//         if let Ok(value) = cast_special!(self, &bool) {
//             *value as i32
//         } else if let Ok(value) = cast_special!(self, &i32) {
//             *value
//         } else {
//             // (*self).cast()

//             1
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
