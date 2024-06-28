use std::fmt::{Debug, Display};
pub fn print_type_of<T>(_: &T) -> &'static str {
    println!("{}", std::any::type_name::<T>());
    return std::any::type_name::<T>();
}
// Define a trait with a method to print values
pub trait MyPrint {
    fn print(&self);
}
// Implement the trait for any single value that implements Display
impl<T: Display> MyPrint for T {
    fn print(&self) {
        println!("{self}");
        // if print_type_of(self) == "i32" {
        //     println!("I am an i32");
        // }
        // match self {
        //     i32 => println!("i32"),
        //     _ => println!("Not i32"),
        // }
        // if TypeId::of::<Self>() == TypeId::of::<String>() {
        //     println!("I am a string");
        // }
    }
}
// Implement the trait for slices of values that implement Display
impl<T: Display> MyPrint for [T] {
    fn print(&self) {
        print!("[");
        for (i, elem) in self.iter().enumerate() {
            if i == self.len() - 1 {
                print!("{elem}");
            } else {
                print!("{elem}, ");
            }
        }
        println!("]");
    }
}
pub trait MySPrint {
    fn prints(&self);
}
impl<T: Debug> MySPrint for T {
    fn prints(&self) {
        println!("{self:#?}");
    }
}
