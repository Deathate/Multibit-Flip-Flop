use crate::*;
#[derive(new, Serialize, Deserialize, Debug, Default, Clone)]
pub struct Rect {
    pub xmin: float,
    pub ymin: float,
    pub xmax: float,
    pub ymax: float,
}
impl Rect {
    pub fn from_coords(coords: [(float, float); 2]) -> Self {
        Self {
            xmin: coords[0].0,
            ymin: coords[0].1,
            xmax: coords[1].0,
            ymax: coords[1].1,
        }
    }
    pub fn coords(&self) -> (float, float, float, float) {
        (self.xmin, self.ymin, self.xmax, self.ymax)
    }
    pub fn area(&self) -> float {
        (self.xmax - self.xmin) * (self.ymax - self.ymin)
    }
    pub fn center(&self) -> (float, float) {
        ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)
    }
    // Compute intersection of two rectangles
    pub fn intersection(&self, other: &Rect) -> Option<Rect> {
        let xmin = self.xmin.max(other.xmin);
        let ymin = self.ymin.max(other.ymin);
        let xmax = self.xmax.min(other.xmax);
        let ymax = self.ymax.min(other.ymax);
        if xmin < xmax && ymin < ymax {
            Some(Rect {
                xmin,
                ymin,
                xmax,
                ymax,
            })
        } else {
            None
        }
    }
    pub fn to_4_corners(&self) -> [(float, float); 4] {
        [
            (self.xmin, self.ymin),
            (self.xmin, self.ymax),
            (self.xmax, self.ymax),
            (self.xmax, self.ymin),
        ]
    }
}
// Function to compute intersection of a set of rectangles
pub fn intersection_of_rects(rects: &Vec<Rect>) -> Option<Rect> {
    if rects.is_empty() {
        return None;
    }
    let mut intersection = rects[0].clone();
    for rect in &rects[1..] {
        match intersection.intersection(rect) {
            Some(new_intersection) => intersection = new_intersection,
            None => return None, // If any two rectangles do not overlap, return None
        }
    }
    Some(intersection)
}
