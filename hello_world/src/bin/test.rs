use std::rc;

use ahash::HashMapExt;
use hello_world::*;
use internment::Intern;
#[derive(Default, Debug)]
struct Owner {
    name: String,
    refer: Vec<Reference<Sheep>>,
}
impl Owner {
    fn new() -> Self {
        Owner {
            name: "Alice".to_string(),
            refer: Vec::new(),
        }
    }
}
#[derive(Default, Debug)]
pub struct Sheep {
    owner: WeakReference<Owner>,
}

impl Sheep {
    fn new() -> Self {
        Default::default()
    }

    fn update_name(&self, new_name: &str) {
        self.owner.upgrade().unwrap().borrow_mut().name = new_name.to_string();
    }
}

fn main() {
    let mut arr = Vec::new();
    let reference1 = build_ref(Sheep::new());
    let reference2 = build_ref(Sheep::new());
    arr.push(reference1);
    arr.push(reference2);
    {
        let owner = build_ref(Owner::new());
        owner.borrow_mut().refer.push(clone_ref(&arr[0]));
        owner.borrow_mut().refer.push(clone_ref(&arr[1]));
        arr[0].borrow_mut().owner = clone_weak_ref(&owner);
        arr[1].borrow_mut().owner = clone_weak_ref(&owner);
        arr[0].borrow_mut().update_name("Dolly");
        println!("Owner name: {}", owner.borrow().name);
        // reference2.borrow_mut().update_name("Blackie");
        // println!("Owner name: {}", owner.borrow().name);
        // println!("{}",owner.borrow().name);
    }
    arr[0].borrow_mut().owner.upgrade().prints();
}
