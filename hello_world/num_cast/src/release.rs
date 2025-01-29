#![cfg(not(feature = "always_assert"))]
use crate::*;
use duplicate::duplicate_item;

// CCbool
#[duplicate_item(
    type_name;
    [ i32 ];
    [ i64 ];
    [ usize ];
    [ f32 ];
    [ f64 ];
)]
impl CCbool for type_name {
    fn bool(&self) -> bool {
        return (*self) != 0 as type_name;
    }
}

// CCi32
#[duplicate_item(
    type_name;
    [ bool ];
    [ i64 ];
    [usize];
    [ f32 ];
    [ f64 ];
)]
impl CCi32 for type_name {
    fn i32(&self) -> i32 {
        return (*self) as i32;
    }
}

// CCi64
#[duplicate_item(
    type_name;
    [ bool ];
    [ i32 ];
    [usize];
    [ f32 ];
    [ f64 ];
)]
impl CCi64 for type_name {
    fn i64(&self) -> i64 {
        return (*self) as i64;
    }
}

// CCusize
#[duplicate_item(
    type_name;
    [ bool ];
    [ i32 ];
    [ i64 ];
    [ f32 ];
    [ f64 ];
)]
impl CCusize for type_name {
    fn usize(&self) -> usize {
        return (*self) as usize;
    }
}

// CCf32
impl CCf32 for bool {
    fn f32(&self) -> f32 {
        if *self {
            1.0
        } else {
            0.0
        }
    }
}
#[duplicate_item(
    type_name;
    [ i32 ];
    [ i64 ];
    [usize];
    [ f64 ];
)]
impl CCf32 for type_name {
    fn f32(&self) -> f32 {
        return (*self) as f32;
    }
}

// CCf64
impl CCf64 for bool {
    fn f64(&self) -> f64 {
        if *self {
            1.0
        } else {
            0.0
        }
    }
}
#[duplicate_item(
    type_name;
    [ i32 ];
    [ i64 ];
    [usize];
    [f32];
)]
impl CCf64 for type_name {
    fn f64(&self) -> f64 {
        return (*self) as f64;
    }
}
