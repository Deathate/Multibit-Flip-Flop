use rustlib::*;
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
struct Sheep {
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
    let reference1 = build_ref(Sheep::new());
    let reference2 = build_ref(Sheep::new());
    {
        let owner = build_ref(Owner::new());
        owner.borrow_mut().refer.push(clone_ref(&reference1));
        owner.borrow_mut().refer.push(clone_ref(&reference2));
        reference1.borrow_mut().owner = clone_weak_ref(&owner);
        reference2.borrow_mut().owner = clone_weak_ref(&owner);
        reference1.borrow_mut().update_name("Dolly");
        println!("Owner name: {}", owner.borrow().name);
        reference2.borrow_mut().update_name("Blackie");
        println!("Owner name: {}", owner.borrow().name);
        println!("{}",owner.borrow().name);
    }
    reference1.borrow_mut().owner.upgrade().prints();
    // println!("Reference 1 to: {}", reference1.borrow().get_name());
    // println!("Reference 2 to: {}", reference2.borrow().get_name());
}
