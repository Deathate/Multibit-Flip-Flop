use log::{Level, LevelFilter};
use mbffg::*;
use pretty_env_logger::formatted_builder;

fn get_case(case: &str) -> (&str, &str, &str) {
    // Mapping case identifiers to corresponding file paths
    let case_map: Dict<&str, (&str, &str, &str)> = [
        (
            "c1_1",
            (
                "../cases/testcase1_0812.txt",
                "../tools/binary001/001_case1.txt",
                "Testcase 1",
            ),
        ),
        (
            "c2_1",
            (
                "../cases/testcase2_0812.txt",
                "../tools/binary001/001_case2.txt",
                "Testcase 2",
            ),
        ),
        (
            "c3_1",
            (
                "../cases/testcase3.txt",
                "../tools/binary001/001_case3.txt",
                "Testcase 3",
            ),
        ),
        (
            "c1_2",
            (
                "../cases/hiddencases/hiddencase01.txt",
                "../tools/binary001/001_hidden1.txt",
                "Hidden Testcase 1",
            ),
        ),
        (
            "c2_2",
            (
                "../cases/hiddencases/hiddencase02.txt",
                "../tools/binary001/001_hidden2.txt",
                "Hidden Testcase 2",
            ),
        ),
        (
            "c2_3",
            (
                "../cases/hiddencases/hiddencase03.txt",
                "../tools/binary001/001_hidden3.txt",
                "Hidden Testcase 3",
            ),
        ),
        (
            "c3_2",
            (
                "../cases/hiddencases/hiddencase04.txt",
                "../tools/binary001/001_hidden4.txt",
                "Hidden Testcase 4",
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
// #[allow(dead_code)]
// fn top1_test(case: &str) -> ExportSummary {
//     let (file_name, top1_name, _) = get_case(case);
//     info!("File name: {}", file_name);
//     info!("Top1 name: {}", top1_name);
//     let mut mbffg = MBFFG::builder().input_path(file_name).build();
//     // check(&mut mbffg, true, false);
//     mbffg.load(Some(top1_name));
//     mbffg.visualize_layout(&format!("top1"), VisualizeOption::builder().build());
//     mbffg.evaluate_and_report().call()
// }
#[time]
#[builder]
fn perform_stage<'a>(
    design_context: &'a DesignContext,
    load_file: Option<&str>,
    load_snapshot: Option<SnapshotData>,
    current_stage: Stage,
    ffs_locator: Option<&'a mut UncoveredPlaceLocator>,
    #[builder(default = 0.0)] pa_bits_exp: float,
    #[builder(default = false)] debug: bool,
    #[builder(default = false)] quiet: bool,
) -> MBFFG<'a> {
    let debug_config = DebugConfig::builder()
        // .debug_update_query_cache(true)
        // .debug_banking_utility(true)
        // .debug_banking_best(true)
        // .debug_placement(true)
        // .visualize_placement_resources(true)
        // .debug_timing_optimization(true)
        .build();
    let mut mbffg = MBFFG::builder()
        .design_context(design_context)
        .debug_config(debug_config)
        .build();
    mbffg.pa_bits_exp = pa_bits_exp;
    if let Some(filename) = load_file {
        mbffg.load_layout(Some(filename));
    }
    if let Some(snapshot) = load_snapshot {
        mbffg.load_snapshot(snapshot);
    }

    match current_stage {
        Stage::Merging => {
            mbffg.merge_flipflops(ffs_locator.unwrap(), quiet);
            if debug {
                mbffg.export_layout(None);
                mbffg.visualize_layout(
                    Stage::Merging.to_string(),
                    VisualizeOption::builder().build(),
                );
                mbffg.export_layout(None);
                mbffg.evaluate_and_report().call();
            }
        }
        Stage::TimingOptimization => {
            mbffg.optimize_timing(quiet);
        }
        Stage::Complete => {
            mbffg.merge_flipflops(ffs_locator.unwrap(), quiet);
            mbffg.optimize_timing(quiet);
            if debug {
                mbffg.export_layout(None);
                mbffg
                    .evaluate_and_report()
                    // .external_eval_opts(ExternalEvaluationOptions { quiet: true })
                    .call();
            }
        }
    }
    mbffg
    // mbffg.evaluate_and_report(true, use_evaluator, output_name, false)
}
// #[allow(dead_code)]
// #[builder]
// fn full_test(testcases: Vec<&str>, run_top1_binary: bool) {
//     let mut summaries = IndexMap::default();
//     for &testcase in &testcases {
//         let summary = if run_top1_binary {
//             top1_test(testcase)
//         } else {
//             perform_main_stage()
//                 .testcase(testcase)
//                 .current_stage(Stage::Complete)
//                 .use_evaluator(false)
//                 .quiet(true)
//                 .call()
//         };
//         summaries.insert(get_case(testcase).2, summary);
//     }
//     {
//         println!(
//             "{}",
//             "\nFinal Report Sheet:".bold().underline().bright_blue()
//         );
//         let column_name = "TNS, Power, Area, Utilization, Score, 1-bit, 2-bit, 4-bit";
//         println!("{}", column_name.bold().dimmed().underline());
//         for (name, summary) in summaries {
//             println!(
//                 "{}",
//                 format!(
//                     "{}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {}, {}, {}",
//                     format!("{}", name.bold().bright_yellow()),
//                     summary.tns,
//                     summary.power,
//                     summary.area,
//                     summary.utilization,
//                     summary.score,
//                     summary.ff_1bit,
//                     summary.ff_2bit,
//                     summary.ff_4bit
//                 )
//             );
//         }
//     }
// }
fn init_logger_with_target_filter() {
    formatted_builder()
        // Set the default log level for the application (e.g., WARN)
        .filter_level(LevelFilter::Info)
        // Set the specific target/module to OFF
        .filter_module("internal", LevelFilter::Off)
        // You can set the level for your main module (e.g., "my_app") to INFO
        // .filter_module("my_app", LevelFilter::Info)
        .init();
}

use malloc_best_effort::BEMalloc;
#[global_allocator]
static GLOBAL: BEMalloc = BEMalloc::new();

#[cfg_attr(feature = "hotpath", hotpath::main)]
fn main() {
    {
        // mbffg.pa_bits_exp = match testcase {
        //     "c1_1" => 1.05,
        //     "c1_2" => 1.05,
        //     "c2_1" => 0.5,
        //     "c2_2" => 0.6,
        //     "c2_3" => 1.05,
        //     "c3_1" => 1.05,
        //     "c3_2" => 1.05,
        //     _ => unreachable!(),
        // };

        // Test different stages of the MBFF optimization pipeline

        // Testcase 1
        // perform_main_stage()
        //     .testcase("c1_1")
        //     .pa_bits_exp(1.05)
        //     .current_stage(Stage::Complete)
        //     .call();

        // Testcase 1 hidden
        // perform_main_stage()
        //     .testcase("c1_2")
        //     .current_stage(Stage::Complete)
        //     .call();

        // Testcase 2
        // {
        //     formatted_builder().filter_level(LevelFilter::Info).init();
        //     let tmr = timer!(Level::Info; "Full MBFFG Process");
        //     let design_context = DesignContext::new(get_case("c2_1").0);
        //     let mut ffs_locator = UncoveredPlaceLocator::new(&design_context, true);
        //     let mut mbffg = perform_stage()
        //         .design_context(&design_context)
        //         .ffs_locator(&mut ffs_locator)
        //         .pa_bits_exp(0.3)
        //         .current_stage(Stage::Merging)
        //         .call();
        //     finish!(tmr);
        //     mbffg.evaluate_and_report().call();
        //     mbffg.evaluate_and_report().call();
        //     return;
        // }

        // Testcase 2 hidden
        // perform_main_stage()
        //     .testcase("c2_2")
        //     .current_stage(Stage::Merging)
        //     .call();

        // Testcase 2 hidden
        // perform_main_stage()
        //     .testcase("c2_3")
        //     .current_stage(Stage::Merging)
        //     .call();

        // Testcase 3 cases
        // perform_main_stage()
        //     .testcase("c3_1")
        //     .current_stage(Stage::Complete)
        //     .call();

        // Testcase 3 hidden
        // perform_main_stage()
        //     .testcase("c3_2")
        //     .current_stage(Stage::Merging)
        //     .call();
    }
    {
        init_logger_with_target_filter();
        let tmr = timer!(Level::Info; "Full MBFFG Process");
        let design_context = DesignContext::new(get_case("c2_1").0);
        let ffs_locator = UncoveredPlaceLocator::new(&design_context, true);
        thread::scope(|s| {
            let handles = [0.3, 0.5, 1.0, 1.05]
                .into_iter()
                .map(|pa_bits_exp| {
                    let design_context_ref = &design_context;
                    let mut ffs_locator = ffs_locator.clone();
                    s.spawn(move || {
                        let mut mbffg = perform_stage()
                            .design_context(design_context_ref)
                            .pa_bits_exp(pa_bits_exp)
                            .current_stage(Stage::Merging)
                            .ffs_locator(&mut ffs_locator)
                            .quiet(true)
                            .call();
                        let (total, w_tns) = (mbffg.sum_weighted_score(), mbffg.sum_neg_slack());
                        (mbffg.create_snapshot(), (total, w_tns))
                    })
                })
                .collect::<Vec<_>>();
            let mut mbffg = MBFFG::builder().design_context(&design_context).build();
            let best_snap_shot = {
                let mut merging_results = handles
                    .into_iter()
                    .map(|h| h.join().unwrap())
                    .collect::<Vec<_>>();
                // merging_results.iter().for_each(|(_, (total, w_tns))| {
                //     info!(
                //         "Merging Result - Total Cost: {:.3}, Weighted TNS: {:.3}",
                //         total, w_tns
                //     );
                // });
                let mut best_idx = 0;
                for (i, result) in merging_results.iter().skip(1).enumerate() {
                    let (_, (best_total, best_tns)) = &merging_results[best_idx];
                    let (_, (total, w_tns)) = result;
                    let diff = (total - best_total).abs() / best_total;
                    if diff > 0.05 {
                        best_idx = i;
                    } else {
                        if (diff - 1.0).abs() < 0.05 {
                            if w_tns > best_tns {
                                best_idx = i;
                            }
                        }
                    }
                }
                std::mem::take(&mut merging_results.get_mut(best_idx).unwrap().0)
            };
            mbffg.load_snapshot(best_snap_shot);
            mbffg.optimize_timing(true);
            mbffg.export_layout(None);
            finish!(tmr);
            mbffg
                .evaluate_and_report()
                .external_eval_opts(ExternalEvaluationOptions { quiet: true })
                .call();
        });
    }
    // full_test()
    //     .testcases(
    //         // vec!["c1_1", "c1_2", "c2_1", "c2_2", "c2_3", "c3_1", "c3_2"],
    //         // vec!["c1_1", "c1_2", "c2_1", "c2_2", "c2_3", "c3_1", "c3_2"],
    //         vec!["c1_1"],
    //     )
    //     .run_top1_binary(false)
    //     .call();
}
