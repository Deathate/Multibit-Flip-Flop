use mbffg::*;
use pretty_env_logger;

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
#[allow(dead_code)]
fn top1_test(case: &str) -> ExportSummary {
    let (file_name, top1_name, _) = get_case(case);
    info!("File name: {}", file_name);
    info!("Top1 name: {}", top1_name);
    let mut mbffg = MBFFG::new(file_name, DebugConfig::builder().build());
    // check(&mut mbffg, true, false);
    mbffg.load(Some(top1_name));
    mbffg.visualize_layout(&format!("top1"), VisualizeOption::builder().build());
    mbffg.evaluate_and_report().call()
}
fn display_progress_step(step: int) {
    match step {
        1 => println!(
            "{} {}",
            "[1/4]".bold().dimmed(),
            "⠋ Initializing MBFFG...".bold().bright_yellow()
        ),
        2 => println!(
            "{} {}",
            "[2/4]".bold().dimmed(),
            "⠙ Merging Flip-Flops...".bold().bright_yellow()
        ),
        3 => println!(
            "{} {}",
            "[3/4]".bold().dimmed(),
            "⠴ Optimizing Timing...".bold().bright_yellow()
        ),
        4 => println!(
            "{} {}",
            "[4/4]".bold().dimmed(),
            "✔ Done".bold().bright_green()
        ),
        _ => unreachable!(),
    }
}
#[time]
#[builder]
fn perform_stage(
    testcase: &str,
    load_file: Option<&str>,
    pa_bits_exp: float,
    current_stage: Stage,
    #[builder(default = false)] quiet: bool,
) -> MBFFG {
    let (file_name, _, _) = get_case(testcase);

    let debug_config = DebugConfig::builder()
        // .debug_update_query_cache(true)
        // .debug_banking_utility(true)
        // .debug_banking_best(true)
        // .debug_placement(true)
        // .visualize_placement_resources(true)
        // .debug_timing_optimization(true)
        .build();
    display_progress_step(1);
    let mut mbffg = MBFFG::new(file_name, debug_config);
    mbffg.pa_bits_exp = pa_bits_exp;
    if let Some(filename) = load_file {
        mbffg.load(Some(filename));
    }

    match current_stage {
        Stage::Merging => {
            display_progress_step(2);
            mbffg.merge_flipflops(quiet);
            #[cfg(debug_assertions)]
            {
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
            display_progress_step(3);
            mbffg.optimize_timing(quiet);
        }
        Stage::Complete => {
            display_progress_step(2);
            mbffg.merge_flipflops(quiet);
            display_progress_step(3);
            mbffg.optimize_timing(quiet);
            #[cfg(debug_assertions)]
            {
                mbffg.export_layout(None);
                display_progress_step(4);
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

use malloc_best_effort::BEMalloc;
#[global_allocator]
static GLOBAL: BEMalloc = BEMalloc::new();

#[cfg_attr(feature = "hotpath", hotpath::main)]
fn main() {
    {
        use std::env;
        // enable info level logging
        if env::var("RUST_LOG").is_err() {
            unsafe {
                env::set_var("RUST_LOG", "debug");
            }
        }

        pretty_env_logger::init();
    }
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
        // let mut mbffg = perform_main_stage()
        //     .testcase("c2_1")
        //     .pa_bits_exp(0.5)
        //     .current_stage(Stage::Complete)
        //     .quiet(true)
        //     .call();
        let tmr = timer!("Full MBFFG Process");
        let merging_results = [0.5]
            .into_par_iter()
            .enumerate()
            .map(|(i, par)| {
                redirect_output_to_null(false, || {
                    let mut mbffg = perform_stage()
                        .testcase("c2_1")
                        .pa_bits_exp(par)
                        .current_stage(Stage::Merging)
                        .quiet(true)
                        .call();
                    let filename = format!("par{:.2}", i);
                    mbffg.export_layout(Some(&filename));
                    mbffg
                        .evaluate_and_report()
                        .external_eval_opts(ExternalEvaluationOptions { quiet: false })
                        .call();
                    (filename, par, mbffg.calculate_weighted_cost())
                })
                .unwrap()
            })
            .collect::<Vec<_>>();
        exit();
        let best = merging_results
            .into_iter()
            .min_by_key(|x| OrderedFloat(x.2))
            .unwrap();
        let (file_name, _, _) = get_case("c2_1");
        let mut mbffg = MBFFG::new(file_name, DebugConfig::builder().build());
        best.0.print();
        mbffg.load(Some(best.0.as_str()));
        mbffg
            .evaluate_and_report()
            .external_eval_opts(ExternalEvaluationOptions { quiet: false })
            .call();

        finish!(tmr);
        // let mut mbffg = perform_main_stage()
        //     .testcase("c2_1")
        //     .load_file(&best.0)
        //     .pa_bits_exp(best.1)
        //     .current_stage(Stage::TimingOptimization)
        //     .quiet(true)
        //     .call();
        // mbffg
        //     .evaluate_and_report()
        //     .external_eval_opts(ExternalEvaluationOptions { quiet: false })
        //     .call();
        // mbffg.evaluate_and_report().call();

        // let mut handles = vec![];
        // for i in [0.5, 1.05] {
        //     let handle = thread::spawn(move || {
        //         redirect_output_to_null(true, || {
        //             let mut mbffg = perform_main_stage()
        //                 .testcase("c2_1")
        //                 .pa_bits_exp(i)
        //                 .current_stage(Stage::Complete)
        //                 .call();
        //             mbffg.calculate_weighted_cost()
        //         })
        //         .unwrap()
        //     });
        //     handles.push(handle);
        // }
        // for handle in handles {
        //     let result = handle.join().unwrap();
        //     result.print();
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
    // full_test()
    //     .testcases(
    //         // vec!["c1_1", "c1_2", "c2_1", "c2_2", "c2_3", "c3_1", "c3_2"],
    //         // vec!["c1_1", "c1_2", "c2_1", "c2_2", "c2_3", "c3_1", "c3_2"],
    //         vec!["c1_1"],
    //     )
    //     .run_top1_binary(false)
    //     .call();
}
