use clap::Parser;
use log::{Level, LevelFilter};
use malloc_best_effort::BEMalloc;
use mbffg::*;
use pretty_env_logger::formatted_builder;
#[global_allocator]
static GLOBAL: BEMalloc = BEMalloc::new();

struct CaseConfig {
    description: &'static str,
    input_path: &'static str,
}
fn get_case(case: &str) -> CaseConfig {
    // Mapping case identifiers to corresponding file paths
    match case {
        "c1" => CaseConfig {
            description: "Testcase 1",
            input_path: "../cases/testcase1_0812.txt",
        },
        "c2" => CaseConfig {
            description: "Testcase 2",
            input_path: "../cases/testcase2_0812.txt",
        },
        "c3" => CaseConfig {
            description: "Testcase 3",
            input_path: "../cases/testcase3.txt",
        },
        "c4" => CaseConfig {
            description: "Hidden 1",
            input_path: "../cases/hiddencases/hiddencase01.txt",
        },
        "c5" => CaseConfig {
            description: "Hidden 2",
            input_path: "../cases/hiddencases/hiddencase02.txt",
        },
        "c6" => CaseConfig {
            description: "Hidden 3",
            input_path: "../cases/hiddencases/hiddencase03.txt",
        },
        "c7" => CaseConfig {
            description: "Hidden 4",
            input_path: "../cases/hiddencases/hiddencase04.txt",
        },
        _ => panic!("Unknown case: {}", case),
    }
}

#[time]
#[builder]
fn perform_stage<'a>(
    mut mbffg: MBFFG<'a>,
    current_stage: Stage,
    ffs_locator: Option<&'a mut UncoveredPlaceLocator>,
    #[builder(default = false)] debug: bool,
    #[builder(default = false)] quiet: bool,
) -> MBFFG<'a> {
    match current_stage {
        Stage::Merging => {
            mbffg.merge_flipflops(ffs_locator.unwrap(), quiet);

            if debug {
                mbffg.export_layout(None);

                mbffg.visualize(
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

            mbffg.update_delay();

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
}

fn init_logger_with_target_filter() {
    let _ = formatted_builder()
        // Set the default log level for the application (e.g., WARN)
        .filter_level(LevelFilter::Info)
        // Set the specific target/module to OFF
        .filter_module("internal", LevelFilter::Off)
        // You can set the level for your main module (e.g., "my_app") to INFO
        // .filter_module("my_app", LevelFilter::Info)
        .try_init();
}

fn perform_mbffg_optimization(case: &str, pa_bits_exp: float) {
    let tmr = timer!(Level::Info; "Full MBFFG Process");

    formatted_builder().filter_level(LevelFilter::Debug).init();

    let design_context = DesignContext::new(get_case(case).input_path);
    let mut ffs_locator = UncoveredPlaceLocator::new(&design_context, true);

    let mut mbffg = MBFFG::builder().design_context(&design_context).build();

    mbffg.pa_bits_exp = pa_bits_exp;

    let mut mbffg = perform_stage()
        .mbffg(mbffg)
        .ffs_locator(&mut ffs_locator)
        .current_stage(Stage::Complete)
        .call();

    mbffg.export_layout(None);

    finish!(tmr);

    mbffg
        .evaluate_and_report()
        // .external_eval_opts(ExternalEvaluationOptions { quiet: false })
        .call();
}

fn perform_mbffg_optimization_parallel(
    case: &str,
    report: bool,
    quiet: bool,
) -> Option<ExportSummary> {
    if log::max_level() == LevelFilter::Off {
        init_logger_with_target_filter();
    }

    let tmr = timer!(Level::Info; "Full MBFFG Process");
    let design_context = DesignContext::new(get_case(case).input_path);
    let ffs_locator = UncoveredPlaceLocator::new(&design_context, true);

    let summary = thread::scope(|s| {
        let params = vec![-2.0, 0.4, 1.05];
        let mut handles = Vec::with_capacity(params.len());
        let design_context_ref = &design_context;

        for pa_bits_exp in params {
            let mut ffs_locator = ffs_locator.clone();
            handles.push(s.spawn(move || {
                let mut mbffg = MBFFG::builder().design_context(design_context_ref).build();

                mbffg.pa_bits_exp = pa_bits_exp;

                let mut mbffg = perform_stage()
                    .mbffg(mbffg)
                    .current_stage(Stage::Merging)
                    .ffs_locator(&mut ffs_locator)
                    .quiet(true)
                    .call();

                mbffg.update_delay();

                (
                    mbffg.create_snapshot(),
                    (mbffg.sum_weighted_score(), mbffg.sum_neg_slack()),
                )
            }));
        }
        let mut mbffg = MBFFG::builder().design_context(&design_context).build();

        let mut merging_results = Vec::with_capacity(handles.len());
        for h in handles {
            merging_results.push(h.join().unwrap());
        }

        let best_idx = {
            let mut best_idx = 0;

            for (i, result) in merging_results.iter().skip(1).enumerate() {
                let (_, (best_total, _)) = &merging_results[best_idx];
                let (_, (total, _)) = result;

                if total < best_total {
                    best_idx = i + 1;
                    // info!(
                    //     "New Best Result Found - Total Cost: {:.3}, Weighted TNS: {:.3}",
                    //     total, w_tns
                    // );
                }
            }

            if !quiet {
                merging_results.iter().for_each(|(_, (total, w_tns))| {
                    info!(
                        "Merging Result - Total Cost: {:.3}, Weighted TNS: {:.3}",
                        total, w_tns
                    );
                });

                info!("Best Merging Result Selected: {}", best_idx);
            }

            best_idx
        };

        let best_snap_shot = &merging_results[best_idx];

        mbffg.load_snapshot(&best_snap_shot.0);

        let ratio = best_snap_shot.1.1 / best_snap_shot.1.0;
        let skip_timing_optimization = ratio < 0.0005;

        if skip_timing_optimization {
            info!(
                "The best merging result has very low weighted TNS to total cost ratio ({:.5}). Early stopping.",
                ratio
            );
        } else {
            mbffg.update_delay();

            mbffg.optimize_timing(true);
        }

        mbffg.export_layout(None);

        finish!(tmr);

        if report {
            Some(
                mbffg
                    .evaluate_and_report()
                    .external_eval_opts(ExternalEvaluationOptions { quiet: true })
                    .call(),
            )
        } else {
            None
        }
    });

    summary
}

#[allow(dead_code)]
#[builder]
fn full_test(testcases: Vec<&str>, report: bool) {
    init_logger_with_target_filter();

    // If not reporting, just run all cases and return early.
    if !report {
        for &testcase in &testcases {
            let _ = perform_mbffg_optimization_parallel(testcase, false, true);
        }

        return;
    }

    // Collect summaries when reporting is requested.
    let mut summaries: Vec<(&str, ExportSummary)> = Vec::with_capacity(testcases.len());

    for &testcase in &testcases {
        let label = get_case(testcase).description;

        match perform_mbffg_optimization_parallel(testcase, true, true) {
            Some(summary) => {
                summaries.push((label, summary));
            }
            None => {
                log::warn!("No summary produced for {}", label);
            }
        }
    }

    if summaries.is_empty() {
        return;
    }

    println!(
        "{}",
        "\nFinal Report Sheet:".bold().underline().bright_blue()
    );

    let column_name = "TNS, Power, Area, Utilization, Score, 1-bit, 2-bit, 4-bit";

    println!("{}", column_name.bold().dimmed().underline());

    for (name, summary) in summaries {
        println!(
            "{}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {}, {}, {}",
            format!("{}", name.bold().bright_yellow()),
            summary.tns,
            summary.power,
            summary.area,
            summary.utilization,
            summary.score,
            summary.ff_1bit,
            summary.ff_2bit,
            summary.ff_4bit
        );
    }
}

#[derive(Parser)]
struct Cli {
    /// The testcase to look for
    #[arg(default_value_t = String::from(""))]
    testcase: String,
}

#[cfg_attr(feature = "hotpath", hotpath::main)]
fn main() {
    {
        // Test the MBFF optimization pipeline

        // perform_mbffg_optimization("c1", 1.05); // Testcase 1
        // perform_mbffg_optimization("c2", 0.4); // Testcase 2
        // perform_mbffg_optimization("c3", 1.05); // Testcase 3 cases
        // perform_mbffg_optimization("c4", -2.0); // Testcase 1 hidden
        // perform_mbffg_optimization("c5", 0.4); // Testcase 2 hidden
        // perform_mbffg_optimization("c6", 1.05); // Testcase 2 hidden
        // perform_mbffg_optimization("c7", 1.05); // Testcase 3 hidden

        // Test the MBFF optimization pipeline in parallel
        perform_mbffg_optimization_parallel("c1", true, false);
        exit();
    }
    let args = Cli::parse();
    if args.testcase.is_empty() {
        full_test()
            .testcases(vec!["c1", "c2", "c3", "c4", "c5", "c6", "c7"])
            .report(false)
            .call();
    } else {
        full_test()
            .testcases(vec![&args.testcase])
            .report(true)
            .call();
    }
}
