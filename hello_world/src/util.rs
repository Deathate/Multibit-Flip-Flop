use std::cell::RefCell;
use std::fmt::{Debug, Display};
pub use std::rc::{Rc, Weak};
pub type Reference<T> = Rc<RefCell<T>>;
pub type WeakReference<T> = Weak<RefCell<T>>;
pub type Dict<T, K> = fxhash::FxHashMap<T, K>;
// pub type Dict = fxhash::FxHashMap;
pub fn build_ref<T>(value: T) -> Reference<T> {
    Rc::new(RefCell::new(value))
}
pub fn build_weak_ref<T>() -> WeakReference<T> {
    Weak::new()
}
pub fn clone_ref<T>(value: &Reference<T>) -> Reference<T> {
    Rc::clone(value)
}
pub fn clone_weak_ref<T>(value: &Reference<T>) -> WeakReference<T> {
    Rc::downgrade(&value)
}
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
