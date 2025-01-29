use crate::*;
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
pub fn is_integer<T: TypeInfo>(value: T) -> bool {
    value.type_of() == "i32" || value.type_of() == "i64" || value.type_of() == "usize"
}
pub fn is_float<T: TypeInfo>(value: T) -> bool {
    value.type_of() == "f32" || value.type_of() == "f64"
}
