use std::cell::RefCell;
use std::rc::{Rc, Weak};
#[derive(Debug)]
pub struct A {
    name: String,
    bs: Vec<Rc<RefCell<B>>>,
}

#[derive(Debug)]
pub struct B {
    parent: Weak<RefCell<A>>, // Weak reference to A to avoid a reference
    name: String,
}

impl A {
    pub fn new(name: &str) -> Rc<RefCell<Self>> {
        let parent = Rc::new(RefCell::new(A {
            name: name.to_string(),
            bs: Vec::new(),
        }));
        let mut bs = Vec::new();
        let parent_weak = Rc::downgrade(&parent);
        let b = Rc::new(RefCell::new(B {
            parent: parent_weak,
            name: name.to_string(),
        }));
        bs.push(b);
        parent.borrow_mut().bs = bs;
        parent
    }
}

fn main() {
    // Create instance of A
    let a = A::new("Parent A");
}
