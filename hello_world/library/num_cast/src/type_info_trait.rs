use duplicate::duplicate_item;
pub trait TypeInfo: Copy {
    fn type_of(&self) -> &'static str;
}
#[duplicate_item(
    type_name result_name;
    [bool] ["bool"];
    [i32] ["i32"];
    [i64] ["i64"];
    [usize] ["usize"];
    [f32] ["f32"];
    [f64] ["f64"];
)]
impl TypeInfo for type_name {
    fn type_of(&self) -> &'static str {
        result_name
    }
}
pub trait TypeCheck:
    Copy + PartialOrd + PartialEq + std::ops::Sub + std::ops::Add + std::ops::Mul + std::ops::Div
{
    fn is_integer(&self) -> bool;
    fn is_float(&self) -> bool;
}
#[duplicate_item(
    type_name;
    [i32];
    [i64];
    [usize];
)]
impl TypeCheck for type_name {
    fn is_integer(&self) -> bool {
        true
    }
    fn is_float(&self) -> bool {
        false
    }
}
#[duplicate_item(
    type_name;
    [f32];
    [f64];
)]
impl TypeCheck for type_name {
    fn is_integer(&self) -> bool {
        false
    }
    fn is_float(&self) -> bool {
        true
    }
}
// pub fn is_integer<T: TypeInfo>(value: T) -> bool {
//     value.type_of() == "i32" || value.type_of() == "i64" || value.type_of() == "usize"
// }
// pub fn is_float<T: TypeInfo>(value: T) -> bool {
//     value.type_of() == "f32" || value.type_of() == "f64"
// }
