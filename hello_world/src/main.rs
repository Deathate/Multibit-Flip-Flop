#![allow(dead_code, unused_imports, unused_variables, unused_mut)]
use colored::*;
use core::time;
// use geo::algorithm::bool_ops::BooleanOps;
// use geo::{coord, Intersects, Polygon, Rect, Vector2DOps};
use grb::prelude::*;
use hello_world::*;
use rand::prelude::*;
use rustworkx_core::petgraph::graph::Node;
use rustworkx_core::petgraph::{graph::NodeIndex, Directed, Direction, Graph};
mod scipy;
use kiddo::{ImmutableKdTree, KdTree, Manhattan};
use pretty_env_logger;
use pyo3::types::PyNone;
use std::num::NonZero;
static GLOBAL_RECTANGLE: once_cell::sync::Lazy<Mutex<Vec<PyExtraVisual>>> =
    once_cell::sync::Lazy::new(|| Mutex::new(Vec::new()));
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
    mbffg.visualize_timing();
    mbffg.compute_mean_shift_and_plot();
    mbffg.visualize_layout("", VisualizeOption::builder().shift_of_merged(true).build());
    mbffg.check(true, true);
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
#[derive(PartialEq, Debug)]
enum STAGE {
    Initial,
    Merging,
    TimingOptimization,
    DetailPlacement,
}
const fn stage_to_name(stage: STAGE) -> &'static str {
    match stage {
        STAGE::Initial => "stage_INITIAL",
        STAGE::Merging => "stage_MERGING",
        STAGE::TimingOptimization => "stage_TIMING_OPTIMIZATION",
        STAGE::DetailPlacement => "stage_DETAIL_PLACEMENT",
    }
}
#[tokio::main]
async fn actual_main() {
    // let mut queue: PriorityQueue<_, _> = Default::default();
    // queue.push("First", A { value: 1 });
    // queue.push("Second", A { value: 1 });
    // let a = queue.get_mut("First").unwrap();
    // // queue.change_priority_by("First", |p| {
    // //     p.test();
    // // });
    // queue.prints();
    // exit();
    let case_name = "c2_1";
    // initial_score();
    // top1_test(case_name, true);
    const STAGE_STATUS: STAGE = STAGE::Initial;
    const LOAD_FROM_FILE: &str = stage_to_name(STAGE_STATUS);

    let tmr = stimer!("MAIN");
    let (file_name, top1_name) = get_case(case_name);
    let mut mbffg = MBFFG::new(file_name);
    mbffg.debug_config = DebugConfig::builder()
        // .debug_update_query_cache(true)
        // .debug_banking_utility(true)
        // .debug_placement(true)
        // .debug_timing_opt(true)
        // .visualize_placement_resources(true)
        .build();
    if STAGE_STATUS == STAGE::Initial {
        mbffg.visualize_layout(
            stage_to_name(STAGE::Initial),
            VisualizeOption::builder().build(),
        );

        let debanked = mbffg.debank_all_multibit_ffs();
        mbffg.replace_1_bit_ffs();
        // {
        //     // This block is for debugging or visualizing the debanked flip-flops.
        //     // You can add custom debug/visualization logic here if needed.
        //     debanked.iter().for_each(|x| {
        //         x.set_walked(true);
        //     });
        //     visualize_layout(&mbffg, "integra", 1, VisualizeOption::builder().build());
        // }

        // merge the flip-flops
        let tmr = stimer!("Merging");
        const SELECTION: i32 = 0;
        info!("Merge the flip-flops");
        if SELECTION == 0 {
            let move_to_center = false;
            let mut uncovered_place_locator =
                UncoveredPlaceLocator::new(&mbffg, &mbffg.find_all_best_library(), move_to_center);
            // uncovered_place_locator.describe().print();
            mbffg.merge(
                &mbffg.get_clock_groups()[0]
                    .iter()
                    .map(|x| x.inst())
                    .collect_vec(),
                2,
                &mut uncovered_place_locator.clone(),
            );
            mbffg.get_all_ffs().filter(|x| x.bits() == 1).for_each(|x| {
                uncovered_place_locator.update_uncovered_place(1, x.pos());
            });
            // {
            //     mbffg.visualize_placement_resources(
            //         &uncovered_place_locator.get(4).unwrap(),
            //         mbffg
            //             .find_best_library_by_bit_count(4)
            //             .borrow()
            //             .ff_ref()
            //             .size(),
            //     );
            //     exit();
            // }
            // uncovered_place_locator.describe().print();
            mbffg.debug_config.debug_banking_best = true;
            mbffg.debug_config.debug_banking_moving = true;
            mbffg.merge(
                &mbffg.get_clock_groups()[0]
                    .iter()
                    .map(|x| x.inst())
                    .filter(|x| x.bits() == 2)
                    .collect_vec(),
                2,
                &mut uncovered_place_locator,
            );
            mbffg.visualize_layout(
                stage_to_name(STAGE::Merging),
                VisualizeOption::builder().shift_of_merged(true).build(),
            );
        }
        finish!(tmr, "Merging done");
        mbffg.check(true, false);
        exit();
        mbffg.output(stage_to_name(STAGE::Merging));
        mbffg.visualize_layout(
            stage_to_name(STAGE::Merging),
            VisualizeOption::builder().shift_of_merged(true).build(),
        );
        mbffg.timing_analysis();
    } else if STAGE_STATUS == STAGE::Merging {
        // mbffg.load(LOAD_FROM_FILE);
        // mbffg.check(true, false);
        // {
        //     let tmr = stimer!("TIMING_OPTIMIZATION");
        //     info!("Timing optimization");
        //     let timing = mbffg.get_ffs_sorted_by_timing();
        //     let num_timing = timing.len();
        //     info!("Number of timing critical flip-flops: {}", num_timing);
        //     let negative_timing_slacks = timing
        //         .iter()
        //         .map(|x| mbffg.negative_timing_slack_inst(x))
        //         .collect_vec();
        //     let ratio_count = count_to_reach_percent(&negative_timing_slacks, 0.8);
        //     info!(
        //         "Number of timing critical flip-flops to reach 80%: {}",
        //         ratio_count
        //     );
        //     for ff in timing.iter() {
        //         ff.set_optimized_pos(ff.pos());
        //     }
        //     for op_group in &timing.iter().take(2500).chunks(500) {
        //         let optimized_pos =
        //             gurobi::optimize_multiple_timing(&mbffg, &op_group.collect_vec(), 0.3).unwrap();
        //         for (ff_id, pos) in optimized_pos.iter() {
        //             let ff = mbffg.get_node(*ff_id);
        //             ff.move_to_pos(*pos);
        //         }
        //         mbffg.check(true, false);
        //     }
        //     mbffg.output(stage_to_name(STAGE::TimingOptimization));
        //     mbffg.visualize_layout(
        //         stage_to_name(STAGE::TimingOptimization),
        //         VisualizeOption::builder().shift_from_optimized(4).build(),
        //     );
        //     finish!(tmr, "Timing optimization done");
        //     mbffg.timing_analysis();
        // }
    } else if STAGE_STATUS == STAGE::TimingOptimization {
        // mbffg.load(LOAD_FROM_FILE);
        // for ff in mbffg.get_all_ffs() {
        //     ff.set_optimized_pos(ff.pos());
        // }
        // let mut gate_rtree = mbffg.generate_gate_map();
        // let rows = mbffg.placement_rows();
        // let die_size = mbffg.die_size();
        // for bit in mbffg.unique_library_bit_widths() {
        //     let lib_size = mbffg
        //         .find_best_library_by_bit_count(bit)
        //         .borrow()
        //         .ff_ref()
        //         .size();
        //     let tmr = stimer!("Placement Resources Evaluation");
        //     let positions = helper::evaluate_placement_resources_from_size(
        //         &gate_rtree,
        //         rows,
        //         die_size,
        //         lib_size,
        //     );
        //     finish!(tmr, "Placement Resources Evaluation done");
        //     mbffg.visualize_placement_resources(&positions, lib_size);
        //     let ffs = mbffg.get_ffs_by_bit(bit).collect_vec();
        //     let result = gurobi::assignment_problem(&ffs, positions, 50).unwrap();
        //     for (i, pos) in result.iter().enumerate() {
        //         let ff = &ffs[i];
        //         ff.move_to_pos(*pos);
        //         gate_rtree.insert_bbox(ff.bbox());
        //     }
        //     mbffg.visualize_layout(
        //         stage_to_name(STAGE::DetailPlacement),
        //         VisualizeOption::builder().shift_from_optimized(bit).build(),
        //     );
        // }
        // mbffg.check(true, true);
        // mbffg.output(stage_to_name(STAGE::DetailPlacement));
        // mbffg.timing_analysis();
    } else {
        panic!("Unknown stage: {:?}", STAGE_STATUS);
    }
}
fn main() {
    pretty_env_logger::init();
    actual_main();
}
