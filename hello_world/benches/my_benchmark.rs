use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hello_world::*;

fn compare_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_comparison");
    group
        // .sample_size(1000)
        .warm_up_time(std::time::Duration::from_millis(300))
        .measurement_time(std::time::Duration::from_secs(1));
    let result_one =
        group.bench_function("test 1", |b| b.iter(|| test1(black_box(20), black_box(15))));
    let result_two =
        group.bench_function("test 2", |b| b.iter(|| test2(black_box(20), black_box(15))));
    group.finish();
}

criterion_group!(benches, compare_functions);
criterion_main!(benches);
