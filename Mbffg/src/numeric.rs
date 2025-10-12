pub fn argmin<T, F, Q>(array: &Vec<T>, f: F) -> usize
where
    T: Copy,
    F: Fn(T) -> Q,
    Q: PartialOrd,
{
    let mut min_index = 0;
    let mut min_value = f(array[0]);
    for (i, &value) in array.iter().enumerate() {
        let value = f(value);
        if value < min_value {
            min_value = value;
            min_index = i;
        }
    }
    min_index
}
