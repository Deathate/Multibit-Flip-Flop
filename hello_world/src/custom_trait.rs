use crate::Vector2;
pub trait SmallShiftTrait {
    fn small_shift(&self) -> Vector2;
}
impl SmallShiftTrait for Vector2 {
    fn small_shift(&self) -> Vector2 {
        (self.0 + 0.1, self.1 + 0.1)
    }
}
