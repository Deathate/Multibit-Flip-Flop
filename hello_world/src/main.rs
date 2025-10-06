// #![allow(dead_code)]
// unused_variables, unused_mut
// unused_imports
use hello_world::*;
mod scipy;
use pretty_env_logger;
static GLOBAL_RECTANGLE: LazyLock<Mutex<Vec<PyExtraVisual>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));
fn get_case(case: &str) -> (&str, &str) {
    // Mapping case identifiers to corresponding file paths
    let case_map: Dict<&str, (&str, &str)> = [
        (
            "c1_1",
            (
                "../cases/testcase1_0812.txt",
                "../tools/binary001/001_case1.txt",
            ),
        ),
        (
            "c2_1",
            (
                "../cases/testcase2_0812.txt",
                "../tools/binary001/001_case2.txt",
            ),
        ),
        (
            "c3_1",
            ("../cases/testcase3.txt", "../tools/binary001/001_case3.txt"),
        ),
        (
            "c1_2",
            (
                "../cases/hiddencases/hiddencase01.txt",
                "../tools/binary001/001_hidden1.txt",
            ),
        ),
        (
            "c2_2",
            (
                "../cases/hiddencases/hiddencase02.txt",
                "../tools/binary001/001_hidden2.txt",
            ),
        ),
        (
            "c2_3",
            (
                "../cases/hiddencases/hiddencase03.txt",
                "../tools/binary001/001_hidden3.txt",
            ),
        ),
        (
            "c3_2",
            (
                "../cases/hiddencases/hiddencase04.txt",
                "../tools/binary001/001_hidden4.txt",
            ),
        ),
    ]
    .into_iter()
    .collect();

    // Lookup the case or panic with an error
    *case_map
        .get(case)
        .unwrap_or_else(|| panic!("Unknown case: {}", case))
}

fn top1_test(case: &str, move_to_center: bool) {
    let (file_name, top1_name) = get_case(case);
    info!("File name: {}", file_name);
    info!("Top1 name: {}", top1_name);
    let mut mbffg = MBFFG::new(file_name, DebugConfig::builder().build());
    // check(&mut mbffg, true, false);
    mbffg.load(top1_name);
    if move_to_center {
        mbffg.move_ffs_to_center();
    }
    mbffg.visualize_layout(&format!("top1"), VisualizeOption::builder().build());
    mbffg.check(true, false);
    // for i in [1, 2, 4] {
    //     visualize_layout(
    //         &mbffg,
    //         "",
    //         2,
    //         VisualizeOption::builder().dis_of_origin(i).build(),
    //     );
    // }

    // let (ffs, timings) = mbffg.get_ffs_sorted_by_timing();
    // timings.iter().iter_print_reverse();
    // run_python_script("describe", (timings,));
    exit();
}
fn initial_score() {
    let file_names = ["c1_1", "c1_2", "c2_1", "c2_2", "c2_3", "c3_1", "c3_2"];
    for file_name in file_names {
        let mut mbffg = MBFFG::new(get_case(file_name).0, DebugConfig::builder().build());
        mbffg.check(true, false);
    }
    exit();
}
#[stime(it = "Merge Flip-Flops")]
fn merge(mbffg: &mut MBFFG) {
    info!("Analyzing placement resources");
    let move_to_center = false;
    let mut uncovered_place_locator =
        UncoveredPlaceLocator::new(&mbffg, &mbffg.find_all_best_library(), move_to_center);

    const METHOD: i32 = 0;
    if METHOD == 0 {
        mbffg.debank_all_multibit_ffs();
        mbffg.replace_1_bit_ffs();
        mbffg.merge(
            &mbffg.get_clock_groups()[0]
                .iter()
                .map(|x| x.inst())
                .collect_vec(),
            6,
            4,
            &mut uncovered_place_locator,
        );
    } else if METHOD == 1 {
        mbffg.merge_kmeans(&mut uncovered_place_locator);
        // mbffg.visualize_layout(
        //     stage_to_name(STAGE::Merging),
        //     VisualizeOption::builder().shift_of_merged(true).build(),
        // );
    }
}
#[stime(it = "Optimize Timing")]
fn optimize_timing(mbffg: &mut MBFFG) {
    mbffg.timing_optimization(0.5, false);
    mbffg.timing_optimization(1.0, true);
}
fn actual_main() {
    let tmr = timer!("Total Runtime");
    const TESTCASENAME: &str = "c2_1";
    const CURRENT_STAGE: STAGE = STAGE::Complete;
    let output_filename = format!("tmp/{}.out", TESTCASENAME);
    let (file_name, top1_name) = get_case(TESTCASENAME);
    let debug_config = DebugConfig::builder()
        // .debug_update_query_cache(true)
        // .debug_banking_utility(true)
        // .debug_placement(true)
        // .debug_timing_opt(true)
        // .visualize_placement_resources(true)
        .build();
    let mut mbffg = MBFFG::new(file_name, debug_config);

    if CURRENT_STAGE == STAGE::Merging {
        mbffg.visualize_layout(
            stage_to_name(STAGE::Merging),
            VisualizeOption::builder().build(),
        );
        // merge the flip-flops
        merge(&mut mbffg);
        mbffg.output(&output_filename);
        mbffg.visualize_layout(
            stage_to_name(STAGE::Merging),
            VisualizeOption::builder().shift_of_merged(true).build(),
        );
        mbffg.timing_analysis();
    } else if CURRENT_STAGE == STAGE::TimingOptimization {
        mbffg.load(&output_filename);
        mbffg.check(true, false);
        optimize_timing(&mut mbffg);
    } else if CURRENT_STAGE == STAGE::Complete {
        merge(&mut mbffg);
        optimize_timing(&mut mbffg);
    } else {
        panic!("Unknown stage: {:?}", CURRENT_STAGE);
    }
    finish!(tmr);
    mbffg.check(true, true);
}
fn main() {
    pretty_env_logger::init();
    actual_main();
}
