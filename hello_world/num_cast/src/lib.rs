#![feature(specialization)]
mod debug;
mod release;
use duplicate::duplicate_item;
#[duplicate_item(
    trait_name type_name;
    [CCbool] [bool];
    [CCi32] [i32];
    [CCi64] [i64];
    [CCusize] [usize];
    [CCf32] [f32];
    [CCf64] [f64];
)]
pub trait trait_name {
    fn type_name(&self) -> type_name;
}
