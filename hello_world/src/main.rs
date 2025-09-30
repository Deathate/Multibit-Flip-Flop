#![allow(dead_code, unused_imports, unused_variables, unused_mut)]
use colored::*;
use core::time;
// use geo::algorithm::bool_ops::BooleanOps;
// use geo::{coord, Intersects, Polygon, Rect, Vector2DOps};
use hello_world::*;
use rand::prelude::*;
use rustworkx_core::petgraph::graph::Node;
use rustworkx_core::petgraph::{graph::NodeIndex, Directed, Direction, Graph};
mod scipy;
use kiddo::{ImmutableKdTree, KdTree, Manhattan};
use pretty_env_logger;
use pyo3::types::PyNone;
use std::num::NonZero;
static GLOBAL_RECTANGLE: LazyLock<Mutex<Vec<PyExtraVisual>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));
static DEBUG: bool = true;
fn debug() {
    let file_name = "../cases/sample/sample_exp_comb3.txt";
    let file_name = "../cases/sample/sample_exp_comb2.txt";
    let file_name = "../cases/sample/sample_exp_comb5.txt";
    let file_name = "../cases/sample/sample_exp_mbit.txt";
    let file_name = "../cases/sample/sample_exp_comb4.txt";
    let file_name = "../cases/sample/sample_exp_comb6.txt";
    let file_name = "../cases/sample/sample_exp.txt";
    let file_name = "../cases/sample/sample_exp_mbit.txt";
    let file_name = "../cases/sample/sample_1.txt";
    let file_name = "../cases/sample/sample_6.txt";
    // let file_name = "../cases/sample/sample_2.txt";
    let mut mbffg = MBFFG::new(&file_name);
    mbffg.filter_timing = false;
    mbffg.move_relative_util("C2", 20.0, 0.0);
    // mbffg.move_relative_util("C3", 10.0, 0.0);
    // mbffg.sta();
    // mbffg.get_pin_util("C3/D").get_timing_record().prints();
    // mbffg.get_pin_util("C3/D").get_origin_dist().prints();
    // mbffg.get_prev_ff_records(&mbffg.get_ff("C3")).prints();
    // mbffg.get_ff("C1").dpins()[0]
    //     .get_farest_timing_record()
    //     .prints();
    mbffg.visualize_layout("test", VisualizeOption::builder().build());
    mbffg.check(false, true);
    exit();
}
fn debug_bank() {
    let file_name = "../cases/sample/sample_exp_comb3.txt";
    let file_name = "../cases/sample/sample_exp_comb2.txt";
    let file_name = "../cases/sample/sample_exp_comb5.txt";
    let file_name = "../cases/sample/sample_exp_mbit.txt";
    let file_name = "../cases/sample/sample_exp_comb4.txt";
    let file_name = "../cases/sample/sample_exp_comb6.txt";
    let file_name = "../cases/sample/sample_exp.txt";
    let file_name = "../cases/sample/sample_1.txt";
    let file_name = "../cases/sample/sample_exp_mbit.txt";
    // let file_name = "../cases/sample/sample_2.txt";
    let mut mbffg = MBFFG::new(&file_name);
    mbffg.filter_timing = false;
    mbffg.bank_util("C1,C8", "FF2").move_to(0.0, 0.0);
    // mbffg.sta();
    mbffg.visualize_layout("test", VisualizeOption::builder().build());
    mbffg.check(false, true);
    exit();
}

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
    let mut mbffg = MBFFG::new(file_name);
    // check(&mut mbffg, true, false);
    mbffg.load(top1_name);
    if move_to_center {
        mbffg.move_ffs_to_center();
    }
    mbffg.visualize_layout(
        &format!("top1"),
        VisualizeOption::builder().shift_from_input(true).build(),
    );
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
        let mut mbffg = MBFFG::new(get_case(file_name).0);
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
            4,
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
#[tokio::main]
#[time(it = "Total Runtime")]
async fn actual_main() {
    const TESTCASENAME: &str = "c2_1";
    const CURRENT_STAGE: STAGE = STAGE::TimingOptimization;
    let output_filename = format!("tmp/{}.out", TESTCASENAME);
    let (file_name, top1_name) = get_case(TESTCASENAME);
    let mut mbffg = MBFFG::new(file_name);
    // mbffg.check(true, false);
    // let inst = mbffg
    //     .get_all_ffs()
    //     .sorted_by_key(|x| x.get_gid())
    //     .skip(1)
    //     .next()
    //     .unwrap()
    //     .clone();
    // let r = mbffg.calculate_incr_neg_slack_after_move(&inst, (0.0, 0.0));
    // mbffg.check(true, false);
    // r.print();
    // exit();
    mbffg.debug_config = DebugConfig::builder()
        // .debug_update_query_cache(true)
        // .debug_banking_utility(true)
        // .debug_placement(true)
        // .debug_timing_opt(true)
        // .visualize_placement_resources(true)
        .build();

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
        mbffg.check(true, true);
    } else if CURRENT_STAGE == STAGE::TimingOptimization {
        mbffg.load(&output_filename);
        mbffg.check(true, false);
        optimize_timing(&mut mbffg);
        mbffg.check(true, true);
    } else if CURRENT_STAGE == STAGE::Complete {
        merge(&mut mbffg);
        optimize_timing(&mut mbffg);
        mbffg.check(true, true);
    } else {
        panic!("Unknown stage: {:?}", CURRENT_STAGE);
    }
}
fn main() {
    pretty_env_logger::init();
    actual_main();
}
