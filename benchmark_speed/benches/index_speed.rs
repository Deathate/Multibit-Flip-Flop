use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{seq::IteratorRandom, thread_rng};
use std::collections::HashMap; // <-- IMPORT SliceRandom HERE

// The maximum size for the data structures
const MAX_SIZE: usize = 100_000;

// Setup function to create the data structures
fn setup_data() -> (Vec<u32>, HashMap<u32, u32>, Vec<usize>) {
    // 1. Create a Vec where the index is effectively the key
    let vec: Vec<u32> = (0..MAX_SIZE as u32).collect();

    // 2. Create a HashMap with the same keys and values
    let hashmap: HashMap<u32, u32> = vec.iter().map(|&i| (i, i)).collect();

    // 3. Create a set of random indices to look up (to avoid predictability)
    let indices: Vec<usize> = (0..MAX_SIZE).collect();
    let mut rng = thread_rng();

    // The use of choose_multiple is now valid
    let random_indices: Vec<usize> = indices.into_iter().choose_multiple(&mut rng, 1000);

    (vec, hashmap, random_indices)
}

fn bench_direct_access(c: &mut Criterion) {
    let (vec, hashmap, indices) = setup_data();
    let mut group = c.benchmark_group("O(1) Access Speed");

    // Benchmark Vec Direct Indexing
    group.bench_function(BenchmarkId::new("Vec_Direct_Index", MAX_SIZE), |b| {
        let mut i = 0;
        b.iter(|| {
            // Access a random element using its index
            let index = indices[i % indices.len()];
            let value = vec[index];
            i += 1;
            black_box(value);
        })
    });

    // Benchmark HashMap Lookup
    group.bench_function(BenchmarkId::new("HashMap_Lookup_Key", MAX_SIZE), |b| {
        let mut i = 0;
        b.iter(|| {
            // Access the HashMap using a u32 key
            let key = indices[i % indices.len()] as u32;
            let value = hashmap.get(&key).unwrap();
            i += 1;
            black_box(value);
        })
    });

    group.finish();
}

criterion_group!(benches, bench_direct_access);
criterion_main!(benches);
