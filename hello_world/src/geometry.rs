use crate::*;

#[derive(new, Serialize, Deserialize, Debug)]
pub struct Rect {
    pub xmin: float,
    pub ymin: float,
    pub xmax: float,
    pub ymax: float,
}
impl Rect {
    pub fn coords(&self) -> (float, float, float, float) {
        (self.xmin, self.ymin, self.xmax, self.ymax)
    }
}
