// #![feature(specialization)]
mod debug;
mod release;
mod tovec;
use duplicate::duplicate_item;
pub use tovec::Collectible;
mod type_info_trait;
pub use type_info_trait::*;
mod inner {
    cfg_if::cfg_if! {
        if #[cfg(feature = "integer_as_i64")] {
            pub type int = i64;
            pub type uint = u64;
        } else {
            pub type int = i32;
            pub type uint = u32;
        }
    }
    #[cfg(feature = "float_as_f64")]
    pub type float = f64;
    #[cfg(not(feature = "float_as_f64"))]
    pub type float = f32;
}
pub use inner::{float, int, uint};
#[duplicate_item(
    trait_name type_name;
    [CCbool] [bool];
    [CCi32] [i32];
    [CCi64] [i64];
    [CCusize] [usize];
    [CCf32] [f32];
    [CCf64] [f64];
    [CCint] [int];
    [CCfloat] [float];
)]
pub trait trait_name {
    fn type_name(&self) -> type_name;
}
