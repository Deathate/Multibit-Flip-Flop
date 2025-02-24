use crate::*;

#[derive(new, Serialize, Deserialize, Debug, Default)]
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
    pub fn area(&self) -> float {
        (self.xmax - self.xmin) * (self.ymax - self.ymin)
    }
    pub fn center(&self) -> (float, float) {
        ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)
    }
}
