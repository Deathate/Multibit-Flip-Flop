pub trait ToVec<T> {
    fn to_vec(self) -> Vec<T>;
}

// Implementation for a tuple of two elements
impl<T> ToVec<T> for (T, T) {
    fn to_vec(self) -> Vec<T> {
        vec![self.0, self.1]
    }
}

// Implementation for a tuple of three elements
impl<T> ToVec<T> for (T, T, T) {
    fn to_vec(self) -> Vec<T> {
        vec![self.0, self.1, self.2]
    }
}

// Implementation for a tuple of four elements
impl<T> ToVec<T> for (T, T, T, T) {
    fn to_vec(self) -> Vec<T> {
        vec![self.0, self.1, self.2, self.3]
    }
}

#[macro_export]
macro_rules! tuple_to_vec {
    ($($elem:expr),*) => {
        vec![$($elem),*]
    };
}
