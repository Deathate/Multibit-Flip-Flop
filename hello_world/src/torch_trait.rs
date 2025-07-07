use crate::*;
use libloading::Library;
use tch::Kind;
use tch::Tensor;
pub fn load_cuda_lib() {
    unsafe {
        Library::new("/home/deathate/libtorch/lib/libtorch_cuda.so")
            .expect("Failed to load libtorch_cuda.so");
    }
}
pub trait TorchUtil {
    fn float(&self) -> f32;
    fn tolist(&self) -> Vec<f32>;
    fn norm1(&self, other: &Self) -> Tensor;
}

// Implement the trait for tch::Tensor
impl TorchUtil for Tensor {
    fn float(&self) -> f32 {
        f32::try_from(self)
            .unwrap_or_else(|_| panic!("Failed to convert Tensor to f32: {:?}", self))
    }
    fn tolist(&self) -> Vec<f32> {
        Vec::try_from(self).unwrap()
    }
    fn norm1(&self, other: &Self) -> Tensor {
        self.rsub(other).abs()
    }
}
pub trait TorchVecUtil {
    fn values(&self) -> Vec<f32>;
    fn max_value(&self) -> f32;
    fn max(&self) -> Tensor;
    fn soft_max(&self, temperature: f32) -> Tensor;
    fn sum(&self) -> Tensor;
}
// Implement the trait for Vec<Tensor>
impl TorchVecUtil for Vec<Tensor> {
    fn values(&self) -> Vec<f32> {
        self.iter().map(|tensor| tensor.float()).collect()
    }
    fn max_value(&self) -> f32 {
        self.iter()
            .map(|tensor| tensor.float())
            .reduce(|a, b| a.max(b))
            .unwrap()
    }
    /// Calculates the elementwise maximum across a vector of tensors.
    /// Assumes all tensors have the same shape.
    fn max(&self) -> Tensor {
        assert!(!self.is_empty(), "tensors must not be empty");
        // Start with a clone of the first tensor (so autograd is preserved)
        // self.iter()
        //     .skip(1)
        //     .fold(self[0].shallow_clone(), |acc, t| acc.max_other(t))
        Tensor::stack(&self, 0).max()
    }
    fn soft_max(&self, temperature: f32) -> Tensor {
        let temperature = Tensor::from(temperature);
        (Tensor::stack(self, 0) * &temperature).logsumexp(0, true) / &temperature
    }
    fn sum(&self) -> Tensor {
        self.iter().sum()
    }
}
#[allow(unused_must_use)]
pub fn freeze_indices_grad(tensor: &Tensor, indices: &[i64]) {
    let grad = tensor.grad();
    for &idx in indices {
        grad.get(idx).fill_(0.0);
    }
}
#[allow(unused_must_use)]
pub fn freeze_indices_grad_mask(tensor: &Tensor, mask: &Tensor) {
    tensor.grad().masked_fill_(&mask, 0.0);
}
