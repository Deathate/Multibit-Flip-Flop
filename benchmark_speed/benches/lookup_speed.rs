use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::collections::HashMap;

// Generates a dataset of a specified size.
fn generate_data(size: usize) -> Vec<(String, u32)> {
    (0..size)
        .map(|i| (format!("Key_{}", i), i as u32))
        .collect()
}

// Benchmarks key-based lookup in a HashMap
fn bench_hashmap_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("Lookup Speed");

    for size in [10, 100, 1000, 10000, 100000].iter() {
        let data = generate_data(*size);
        let hashmap: HashMap<String, u32> = data.iter().cloned().collect();

        // The key we will look up. We choose a key near the end/middle to avoid
        // artificially fast results from the linear search case.
        let lookup_key = format!("Key_{}", size / 2);

        group.bench_with_input(BenchmarkId::new("HashMap", size), &lookup_key, |b, key| {
            b.iter(|| {
                // Use black_box to prevent the compiler from optimizing away the operation
                black_box(hashmap.get(key));
            });
        });

        group.bench_with_input(
            BenchmarkId::new("Vec_LinearSearch", size),
            &lookup_key,
            |b, key| {
                b.iter(|| {
                    // Linear search in a Vec (Vec<(Key, Value)>)
                    let result = data.iter().find(|(k, _)| k == key);
                    black_box(result);
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_hashmap_lookup);
criterion_main!(benches);
