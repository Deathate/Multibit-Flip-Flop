use itertools::Itertools;
use num_cast::CCf64;
use ordered_float::*;
use std::cmp::Reverse;
mod boolean_mask;
pub use boolean_mask::*;
mod fancy_index;
pub use fancy_index::*;
pub trait VecUtil {
    fn sum(&self) -> f64;
    fn mean(&self) -> f64;
    // fn mode(&self) -> f64;
    fn median(&self) -> f64;
    fn variance(&self) -> f64;
    fn std_dev(&self) -> f64;
    fn sum_of_squares(&self) -> f64;
    fn min(&self) -> f64;
    fn max(&self) -> f64;
}
impl<T> VecUtil for [T]
where
    T: CCf64,
{
    fn sum(&self) -> f64 {
        self.iter().map(|x| x.f64()).sum()
    }
    fn mean(&self) -> f64 {
        self.sum() / self.len() as f64
    }
    // fn mode(&self) -> f64 {
    //     let counts = self.iter().counts();
    //     let max = **counts.values().max().as_ref().unwrap();
    //     counts
    //         .into_iter()
    //         .find(|(_, v)| *v == max)
    //         .map(|(k, _)| k.f64())
    //         .unwrap()
    // }
    fn median(&self) -> f64 {
        let mut sorted = self.to_vec();
        sorted.sort_unstable_by_key(|x| OrderedFloat(x.f64()));
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1].f64() + sorted[mid].f64()) / 2.0
        } else {
            sorted[mid].f64()
        }
    }
    fn variance(&self) -> f64 {
        let mean = self.mean();
        self.iter().map(|x| (x.f64() - mean).powi(2)).sum::<f64>() / self.len() as f64
    }
    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
    fn sum_of_squares(&self) -> f64 {
        self.iter().map(|x| x.f64().powi(2)).sum()
    }
    fn min(&self) -> f64 {
        self.iter()
            .map(|x| x.f64())
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }
    fn max(&self) -> f64 {
        self.iter()
            .map(|x| x.f64())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
    }
}

pub trait VecSort: Iterator {
    fn sort(&mut self) -> std::vec::IntoIter<<Self as Iterator>::Item>;
    fn sort_reverse(&mut self) -> std::vec::IntoIter<<Self as Iterator>::Item>;
}
impl<T> VecSort for T
where
    T: Iterator<Item: CCf64>,
{
    fn sort(&mut self) -> std::vec::IntoIter<<Self as Iterator>::Item> {
        self.sorted_unstable_by_key(|x| OrderedFloat(x.f64()))
    }
    fn sort_reverse(&mut self) -> std::vec::IntoIter<<Self as Iterator>::Item> {
        self.sorted_unstable_by_key(|x| Reverse(OrderedFloat(x.f64())))
    }
}
