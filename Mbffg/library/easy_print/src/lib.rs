#![feature(decl_macro)]
use castaway::cast;
use std::fmt;

pub struct InlinePrintable<'a, T: ?Sized>(&'a T);
impl<'a, T: fmt::Debug + ?Sized> fmt::Display for InlinePrintable<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Ok(string) = cast!(self, &String) {
            write!(f, "'{}'", string)
        } else if let Ok(string) = cast!(self, &str) {
            write!(f, "'{}'", string)
        } else {
            write!(f, "{:?}", self.0)
        }
    }
}
// helper
pub fn inline<T: ?Sized>(t: &T) -> InlinePrintable<'_, T> {
    InlinePrintable(t)
}
/// Python-like print macro with optional `sep` and `end`.
/// Defaults: `sep = " "`, `end = "\n"`.
pub macro py_print {
    // sep + end (either order)
    (sep = $sep:expr, end = $end:expr, $( $x:expr ),* $(,)? ) => {
        $crate::py_print!(@impl $sep, $end, $( $x ),*);
    },
    (end = $end:expr, sep = $sep:expr, $( $x:expr ),* $(,)? ) => {
        $crate::py_print!(@impl $sep, $end, $( $x ),*);
    },

    // sep only
    (sep = $sep:expr, $( $x:expr ),* $(,)? ) => {
        $crate::py_print!(@impl $sep, "\n", $( $x ),*);
    },

    // end only
    (end = $end:expr, $( $x:expr ),* $(,)? ) => {
        $crate::py_print!(@impl " ", $end, $( $x ),*);
    },

    // defaults: sep = " ", end = "\n"
    ( $( $x:expr ),* $(,)? ) => {
        $crate::py_print!(@impl " ", "\n", $( $x ),*);
    },

    // internal implementation
    (@impl $sep:expr, $end:expr, $( $x:expr ),* ) => {{
        let sep = $sep;
        let end = $end;
        let mut __first = true;
        $(
            if !__first { print!("{}", sep); }
            print!("{}", $crate::inline(&(&$x)));
            __first = false;
        )*
        print!("{}", end);
    }},
}

pub trait DebugPrintExt {
    fn print(&self);
    fn pprint(&self);
}
impl<T: fmt::Debug + ?Sized> DebugPrintExt for T {
    fn print(&self) {
        if let Ok(string) = cast!(self, &String) {
            println!("{string}");
        } else if let Ok(string) = cast!(self, &str) {
            println!("{string}");
        } else {
            println!("{self:?}");
        }
    }
    fn pprint(&self) {
        if let Ok(string) = cast!(self, &String) {
            println!("'{string}'");
        } else if let Ok(string) = cast!(self, &str) {
            println!("'{string}'");
        } else {
            println!("{self:#?}");
        }
    }
}

pub trait IterPrintExt: Iterator {
    fn iter_print(self)
    where
        Self: Sized,
        Self::Item: fmt::Display;

    fn iter_print_reverse(self)
    where
        Self: Sized + DoubleEndedIterator,
        Self::Item: fmt::Display;
}
impl<T> IterPrintExt for T
where
    T: Iterator,
{
    fn iter_print(self)
    where
        Self: Sized,
        Self::Item: fmt::Display,
    {
        println!("[");
        self.for_each(|elem| println!("   {elem},"));
        println!("]");
    }

    fn iter_print_reverse(self)
    where
        Self: Sized + DoubleEndedIterator,
        Self::Item: fmt::Display,
    {
        println!("[");
        self.rev().for_each(|elem| println!("   {elem},"));
        println!("]");
    }
}
