use crate::*;
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::*;
pub fn run_python_script<A>(function_name: &str, args: A)
where
    A: for<'py> IntoPyObject<'py, Target = PyTuple>,
{
    Python::with_gil(|py| {
        let script = c_str!(include_str!("script.py")); // Include the script as a string
        let module = PyModule::from_code(py, script, c_str!(""), c_str!(""))?;

        let result = module.getattr(function_name)?.call1(args)?;
        Ok::<(), PyErr>(())
    })
    .unwrap()
}
pub fn run_python_script_with_return<A, R: Clone>(function_name: &str, args: A) -> R
where
    A: for<'py> IntoPyObject<'py, Target = PyTuple>,
    R: for<'py> FromPyObject<'py>,
{
    Python::with_gil(|py| {
        let script = c_str!(include_str!("script.py")); // Include the script as a string
        let module = PyModule::from_code(py, script, c_str!(""), c_str!(""))?;

        let result = module.getattr(function_name)?.call1(args)?.extract()?;
        Ok::<R, PyErr>(result)
    })
    .unwrap()
}
#[pyclass(get_all)]
pub struct Pyo3Cell {
    pub name: String,
    pub x: float,
    pub y: float,
    pub width: float,
    pub height: float,
    pub walked: bool,
    pub highlighted: bool,
    pub pins: Vec<Pyo3Pin>,
}
impl Pyo3Cell {
    pub fn new(inst: &Reference<Inst>) -> Self {
        let name = inst.borrow().name.clone();
        let x = inst.borrow().x;
        let y = inst.borrow().y;
        let width = inst.borrow().width();
        let height = inst.borrow().height();
        let walked = inst.borrow().walked;
        let highlighted = inst.borrow().highlighted;
        let pins = Vec::new();
        Self {
            name,
            x,
            y,
            width,
            height,
            walked,
            highlighted,
            pins,
        }
    }
}
#[pyclass(get_all)]
pub struct Pyo3Net {
    pub pins: Vec<Pyo3Pin>,
    pub is_clk: bool,
}
#[derive(Clone)]
#[pyclass(get_all)]
pub struct Pyo3Pin {
    pub name: String,
    pub x: float,
    pub y: float,
}
#[pyclass(get_all)]
pub struct Pyo3KMeansResult {
    pub points: Vec<float>,
    pub cluster_centers: Vec<float>,
    pub labels: Vec<usize>,
}
