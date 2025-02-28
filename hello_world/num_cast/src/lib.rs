// #![feature(specialization)]
mod debug;
mod release;
// mod tovec;
use duplicate::duplicate_item;
// pub use tovec::Collectible;
mod type_info_trait;
pub use type_info_trait::*;
// mod inner {
// }
// pub use inner::{float, int, uint};
cfg_if::cfg_if! {
    if #[cfg(feature = "integer_as_i64")] {
        pub use i64 as int;
        pub use u64 as uint;
    } else {
        pub use i32 as int;
        pub use u32 as uint;
    }
}
#[cfg(not(feature = "float_as_f64"))]
pub use f32 as float;
#[cfg(feature = "float_as_f64")]
pub use f64 as float;
#[duplicate_item(
    trait_name type_name;
    [CCbool] [bool];
    [CCi32] [i32];
    [CCi64] [i64];
    [CCu32] [u32];
    [CCu64] [u64];
    [CCusize] [usize];
    [CCf32] [f32];
    [CCf64] [f64];
    [CCint] [int];
    [CCfloat] [float];
)]
pub trait trait_name {
    fn type_name(&self) -> type_name;
}
