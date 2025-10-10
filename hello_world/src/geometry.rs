use crate::*;
#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct Rect {
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
}
impl Rect {
    pub fn from_bbox(bbox: [[float; 2]; 2]) -> Self {
        Self {
            xmin: bbox[0][0],
            ymin: bbox[0][1],
            xmax: bbox[1][0],
            ymax: bbox[1][1],
        }
    }
    pub fn from_size(xmin: float, ymin: float, width: float, height: float) -> Self {
        Self {
            xmin,
            ymin,
            xmax: xmin + width,
            ymax: ymin + height,
        }
    }
    pub fn erosion(&self, delta: float) -> Self {
        Self {
            xmin: self.xmin + delta,
            ymin: self.ymin + delta,
            xmax: self.xmax - delta,
            ymax: self.ymax - delta,
        }
    }
    pub fn intersection_area(&self, other: &Rect) -> float {
        use geo::algorithm::bool_ops::BooleanOps;
        use geo::Area;
        let r1 = geo::Rect::new(
            geo::coord !(x : self.xmin, y : self.ymin),
            geo::coord !(x : self.xmax, y : self.ymax),
        );
        let r2 = geo::Rect::new(
            geo::coord !(x : other.xmin, y : other.ymin),
            geo::coord !(x : other.xmax, y : other.ymax),
        );
        r1.to_polygon()
            .intersection(&r2.to_polygon())
            .unsigned_area()
    }
    pub fn bbox(&self) -> [[float; 2]; 2] {
        [[self.xmin, self.ymin], [self.xmax, self.ymax]]
    }
}
