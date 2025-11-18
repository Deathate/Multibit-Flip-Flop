use crate::*;
pub use pyo3::ffi::c_str;
pub use pyo3::prelude::*;
pub use pyo3::types::*;
pub fn run_python_script<A>(function_name: &str, args: A)
where
    A: for<'py> IntoPyObject<'py, Target = PyTuple>,
{
    Python::with_gil(|py| {
        let script = c_str!(include_str!("script.py")); // Include the script as a string
        let module = PyModule::from_code(py, script, c_str!(""), c_str!(""))?;
        module.getattr(function_name)?.call1(args)?;
        Ok::<(), PyErr>(())
    })
    .unwrap();
}
pub fn run_python_script_with_return<A, R>(function_name: &str, args: A) -> R
where
    A: for<'py> IntoPyObject<'py, Target = PyTuple>,
    R: for<'py> FromPyObject<'py> + Clone,
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
    pub fn new(inst: &SharedInst) -> Self {
        let name = inst.get_name().to_string();
        let (x, y) = inst.pos();
        let width = inst.get_width();
        let height = inst.get_height();
        let walked = inst.get_walked();
        let highlighted = inst.get_highlighted();
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
#[pyclass(get_all)]
// #[derive(new)]
#[derive(Debug, Builder, Clone)]
pub struct PyExtraVisual {
    /// cirlce, rect, line
    #[builder(into)]
    pub id: String,
    /// circle: [(x, y)], rect: [(x, y) * 4], line: [(x1, y1), (x2, y2)]
    pub points: Vec<(float, float)>,
    #[builder(default = 1)]
    pub line_width: int,
    #[builder(default = (0, 0, 0))]
    pub color: (int, int, int),
    #[builder(default = 0)]
    pub angle: int,
    #[builder(default = 3)]
    pub radius: int,
    #[builder(default = true)]
    pub arrow: bool,
}
