pub trait Collectible {
    type Item;
    fn to_vec(self) -> Vec<Self::Item>;
}

impl<T, I> Collectible for I
where
    I: Iterator<Item = T>,
{
    type Item = T;

    fn to_vec(self) -> Vec<Self::Item> {
        self.collect()
    }
}