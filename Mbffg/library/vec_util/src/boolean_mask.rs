pub trait BooleanMaskOptimized<'a, T> {
    fn boolean_mask_ref(&'a self, mask: &[bool]) -> Vec<&'a T>;
}

impl<'a, T> BooleanMaskOptimized<'a, T> for [T] {
    fn boolean_mask_ref(&'a self, mask: &[bool]) -> Vec<&'a T> {
        assert_eq!(self.len(), mask.len(), "Data and mask length must match");

        // Precompute how many `true` values we have to preallocate capacity
        let count = mask.iter().filter(|&&flag| flag).count();
        let mut result = Vec::with_capacity(count);

        for (val, flag) in self.iter().zip(mask.iter()) {
            if *flag {
                result.push(val);
            }
        }

        result
    }
}
pub trait BooleanMaskClone<T> {
    fn boolean_mask_clone(&self, mask: &[bool]) -> Vec<T>;
}

impl<T: Clone> BooleanMaskClone<T> for [T] {
    fn boolean_mask_clone(&self, mask: &[bool]) -> Vec<T> {
        assert_eq!(self.len(), mask.len(), "Data and mask length must match");
        let count = mask.iter().filter(|&&flag| flag).count();
        let mut result = Vec::with_capacity(count);

        for (val, flag) in self.iter().zip(mask.iter()) {
            if *flag {
                result.push(val.clone());
            }
        }

        result
    }
}
pub trait BooleanMaskOwned<T> {
    fn boolean_mask(self, mask: &[bool]) -> Vec<T>;
}

impl<T> BooleanMaskOwned<T> for Vec<T> {
    fn boolean_mask(self, mask: &[bool]) -> Vec<T> {
        assert_eq!(self.len(), mask.len(), "Data and mask length must match");

        let mut iter = self.into_iter();
        let mut result = Vec::with_capacity(mask.iter().filter(|&&flag| flag).count());

        for flag in mask {
            let val = iter.next().unwrap();
            if *flag {
                result.push(val);
            }
            // else val gets dropped automatically
        }

        result
    }
}
pub trait BooleanMaskInPlace {
    fn boolean_mask_in_place(&mut self, mask: &[bool]);
}

impl<T> BooleanMaskInPlace for Vec<T> {
    fn boolean_mask_in_place(&mut self, mask: &[bool]) {
        assert_eq!(self.len(), mask.len(), "Data and mask length must match");

        let mut i = 0;
        self.retain(|_| {
            let keep = mask[i];
            i += 1;
            keep
        });
    }
}
