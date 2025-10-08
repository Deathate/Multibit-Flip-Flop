use crate::Vector2;
pub trait SmallShiftExt {
    fn small_shift(&self) -> Vector2;
}
impl SmallShiftExt for Vector2 {
    fn small_shift(&self) -> Vector2 {
        (self.0 - 0.01, self.1 - 0.01)
    }
}
