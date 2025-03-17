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
        print!("]\n");
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
    fn prints_with(&self, start: &str);
}
impl MySPrint for String {
    fn prints(&self) {
        println!("{self}");
    }
    fn prints_with(&self, start: &str) {
        println!("{start} {self}");
    }
}

impl<T: fmt::Debug> MySPrint for T {
    default fn prints(&self) {
        println!("{self:#?}");
    }
    default fn prints_with(&self, start: &str) {
        println!("{start} {self:#?}");
    }
}

pub trait MySPrintIter: Iterator {
    fn iter_print(self);
    fn iter_print_reverse(self);
}
impl<T, I> MySPrintIter for T
where
    T: Iterator<Item = I> + std::iter::DoubleEndedIterator,
    I: fmt::Display,
{
    fn iter_print(self) {
        print!("[");
        self.for_each(|elem| print!("   {elem}, \n"));
        print!("]\n");
    }
    fn iter_print_reverse(self) {
        print!("[");
        self.rev().for_each(|elem| print!("   {elem}, \n"));
        print!("]\n");
    }
}
