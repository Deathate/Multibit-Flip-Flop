#![feature(specialization)]
use std::fmt;

// Define a trait with a method to print values
pub trait MyPrint {
    fn print(&self);
}
// Implement the trait for any single value that implements Display
impl<T: fmt::Display> MyPrint for T {
    fn print(&self) {
        println!("{self}");
    }
}
// Implement the trait for slices of values that implement Display
impl<T: fmt::Display> MyPrint for [T] {
    fn print(&self) {
        print!("[");
        for (i, elem) in self.iter().enumerate() {
            if i == self.len() - 1 {
                print!("{elem}");
            } else {
                print!("{elem}, ");
            }
        }
        print!("]");
    }
    // fn println(&self) {
    //     print!("[");
    //     for (i, elem) in self.iter().enumerate() {
    //         if i == self.len() - 1 {
    //             print!("{elem}");
    //         } else {
    //             print!("{elem}, ");
    //         }
    //     }
    //     println!("]");
    // }
}

pub trait MySPrint {
    fn prints(&self);
}
impl MySPrint for String {
    fn prints(&self) {
        println!("{self}");
    }
}
impl<T: fmt::Debug> MySPrint for T {
    default fn prints(&self) {
        println!("{self:#?}");
    }
}
