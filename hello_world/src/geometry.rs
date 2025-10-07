use crate::*;
#[derive(Serialize, Deserialize, Debug, Default, Clone)]
pub struct Rect {
    pub xmin: float,
    pub ymin: float,
    pub xmax: float,
    pub ymax: float,
    pub is_manhattan: bool,
}
impl Rect {
    pub fn from_coords(coords: [(float, float); 2]) -> Self {
        Self {
            xmin: coords[0].0,
            ymin: coords[0].1,
            xmax: coords[1].0,
            ymax: coords[1].1,
            is_manhattan: false,
        }
    }
    pub fn from_bbox(bbox: [[float; 2]; 2]) -> Self {
        Self {
            xmin: bbox[0][0],
            ymin: bbox[0][1],
            xmax: bbox[1][0],
            ymax: bbox[1][1],
            is_manhattan: false,
        }
    }
    pub fn coords(&self) -> (float, float, float, float) {
        (self.xmin, self.ymin, self.xmax, self.ymax)
    }
    pub fn from_center_and_size(
        middle: (float, float),
        width: float,
        height: float,
        is_manhattan: bool,
    ) -> Self {
        let half_width = width / 2.0;
        let half_height = height / 2.0;
        if !is_manhattan {
            Self {
                xmin: middle.0 - half_width,
                ymin: middle.1 - half_height,
                xmax: middle.0 + half_width,
                ymax: middle.1 + half_height,
                is_manhattan,
            }
        } else {
            assert!(
                width == height,
                "Width and height must be equal for Manhattan rectangles"
            );
            Self {
                xmin: middle.0,
                ymin: middle.1 - half_width,
                xmax: middle.0,
                ymax: middle.1 + half_width,
                is_manhattan,
            }
        }
    }
    pub fn from_size(xmin: float, ymin: float, width: float, height: float) -> Self {
        Self {
            xmin,
            ymin,
            xmax: xmin + width,
            ymax: ymin + height,
            is_manhattan: false,
        }
    }
    pub fn area(&self) -> float {
        (self.xmax - self.xmin) * (self.ymax - self.ymin)
    }
    pub fn center(&self) -> (float, float) {
        ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)
    }
    // Compute intersection of two rectangles
    pub fn intersection(&self, other: &Rect) -> Option<Rect> {
        assert!(
            !self.is_manhattan && !other.is_manhattan,
            "Intersection is only defined for non-Manhattan rectangles"
        );
        // Calculate the overlapping area
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
                is_manhattan: false,
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
    pub fn to_2_corners(&self) -> [(float, float); 2] {
        [(self.xmin, self.ymin), (self.xmax, self.ymax)]
    }
    pub fn wh(&self) -> (float, float) {
        (self.xmax - self.xmin, self.ymax - self.ymin)
    }
    /// Returns the bounding box of the rectangle with a small buffer
    pub fn bbox(&self) -> [[float; 2]; 2] {
        let buffer = 0.1;
        [
            [self.xmin + buffer, self.ymin + buffer],
            [self.xmax - buffer, self.ymax - buffer],
        ]
    }
    /// Returns the bounding box of the rectangle without a buffer
    pub fn bbox_p(&self) -> [[float; 2]; 2] {
        [[self.xmin, self.ymin], [self.xmax, self.ymax]]
    }
    pub fn inside(&self, bbox: [[float; 2]; 2]) -> bool {
        self.xmin >= bbox[0][0]
            && self.ymin >= bbox[0][1]
            && self.xmax <= bbox[1][0]
            && self.ymax <= bbox[1][1]
    }
    pub fn dilation(&self, delta: float) -> Self {
        Self {
            xmin: self.xmin - delta,
            ymin: self.ymin - delta,
            xmax: self.xmax + delta,
            ymax: self.ymax + delta,
            is_manhattan: self.is_manhattan,
        }
    }
    pub fn erosion(&self, delta: float) -> Self {
        Self {
            xmin: self.xmin + delta,
            ymin: self.ymin + delta,
            xmax: self.xmax - delta,
            ymax: self.ymax - delta,
            is_manhattan: self.is_manhattan,
        }
    }
}
pub fn manhattan_square(middle: (float, float), half: float) -> [(float, float); 4] {
    [
        (middle.0, middle.1 - half),
        (middle.0 - half, middle.1),
        (middle.0, middle.1 + half),
        (middle.0 + half, middle.1),
    ]
}
pub fn rotate_point_45(x: float, y: float) -> (float, float) {
    (x + y, y - x)
}
pub fn rotate_point_inv_45(x: float, y: float) -> (float, float) {
    ((x - y) / 2.0, (x + y) / 2.0)
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
pub fn joint_manhattan_square(rects: Vec<Rect>, rotate_back: bool) -> Option<[(f64, f64); 2]> {
    assert!(
        rects.iter().all(|r| r.is_manhattan),
        "All rectangles must be Manhattan squares"
    );
    let cells: Vec<geometry::Rect> = rects
        .into_iter()
        .map(|rect| {
            geometry::Rect::from_coords([
                rotate_point_45(rect.xmin, rect.ymin),
                rotate_point_45(rect.xmax, rect.ymax),
            ])
        })
        .collect();
    match intersection_of_rects(&cells) {
        Some(x) => {
            let coord = x.to_2_corners();
            // Rotate the coordinates back by -45 degrees
            if rotate_back {
                return coord
                    .iter()
                    .map(|x| geometry::rotate_point_inv_45(x.0, x.1))
                    .collect_vec()
                    .try_into()
                    .ok();
            }
            // If not rotating back, return the coordinates as they are
            return Some(coord);
        }
        None => return None,
    }
}
