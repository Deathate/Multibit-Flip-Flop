// #![allow(dead_code)]
// unused_variables, unused_mut
// unused_imports
use hello_world::*;
use pretty_env_logger;
use prettytable::{Cell, Row, Table};
static GLOBAL_RECTANGLE: LazyLock<Mutex<Vec<PyExtraVisual>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));
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
    mbffg.load(top1_name);
    mbffg.visualize_layout(&format!("top1"), VisualizeOption::builder().build());
    mbffg.perform_evaluation(true, true)
}
/// merge the flip-flops
#[stime(it = "Merge Flip-Flops")]
fn merge(mbffg: &mut MBFFG) {
    mbffg.debank_all_multibit_ffs();
    mbffg.replace_1_bit_ffs();
    mbffg.force_directed_placement();
    let mut uncovered_place_locator = UncoveredPlaceLocator::new(mbffg, false);
    // Statistics for merged flip-flops
    let mut statistics = Dict::new();
    let pbar = ProgressBar::new(mbffg.num_ff());
    pbar.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] {bar:60.cyan/blue} {pos:>7}/{len:7} {msg}",
        )
        .unwrap()
        .progress_chars("##-"),
    );
    let clk_groups = mbffg.get_clock_groups();
    for group in clk_groups {
        let bits_occurrences = mbffg.merge(
            group.iter_map(|x| x.inst()).collect_vec(),
            6,
            4,
            &mut uncovered_place_locator,
            &pbar,
        );
        for (bit, occ) in bits_occurrences {
            *statistics.entry(bit).or_insert(0) += occ;
        }
    }
    pbar.finish();

    // Print statistics
    info!("Flip-Flop Merge Statistics:");
    for (bit, occ) in statistics.iter().sorted_by_key(|&(bit, _)| *bit) {
        info!("{}-bit → {:>10} merged", bit, occ);
    }
    mbffg.check_on_site();
    mbffg.update_delay_all();
}
#[stime(it = "Optimize Timing")]
fn optimize_timing(mbffg: &mut MBFFG) {
    let clk_groups = mbffg.get_clock_groups();
    let single_clk = clk_groups.len() == 1;
    for group in clk_groups {
        let group = group
            .into_iter_map(|x| x.inst().dpins())
            .flatten()
            .collect_vec();
        mbffg.refine_timing(&group, 0.5, false, single_clk);
        mbffg.refine_timing(&group, 1.0, true, single_clk);
    }
}
fn show_step(step: int) {
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
fn perform_main_stage(testcase: &str, current_stage: STAGE, use_evaluator: bool) -> ExportSummary {
    let tmr = timer!("Total Runtime");
    let intermediate_output_filename = format!("tmp/{}.out", testcase);
    let (file_name, _, _) = get_case(testcase);

    let debug_config = DebugConfig::builder()
        // .debug_update_query_cache(true)
        // .debug_banking_utility(true)
        // .debug_placement(true)
        // .debug_timing_opt(true)
        // .visualize_placement_resources(true)
        .build();
    show_step(1);
    let mut mbffg = MBFFG::new(file_name, debug_config);
    if current_stage == STAGE::Merging {
        mbffg.visualize_layout(
            stage_to_name(STAGE::Merging),
            VisualizeOption::builder().build(),
        );
        show_step(2);
        merge(&mut mbffg);
        mbffg.output(&intermediate_output_filename);
        mbffg.visualize_layout(
            stage_to_name(STAGE::Merging),
            VisualizeOption::builder().build(),
        );
    } else if current_stage == STAGE::TimingOptimization {
        mbffg.load(&intermediate_output_filename);
        mbffg.perform_evaluation(true, false);
        show_step(3);
        optimize_timing(&mut mbffg);
    } else if current_stage == STAGE::Complete {
        show_step(2);
        merge(&mut mbffg);
        show_step(3);
        optimize_timing(&mut mbffg);
        let output_name = PathLike::new(file_name)
            .with_extension("out")
            .name()
            .unwrap();
        mbffg.output(format!("tmp/{}", output_name).as_str());
    } else {
        panic!("Unknown stage: {:?}", current_stage);
    }
    show_step(4);
    finish!(tmr);
    mbffg.perform_evaluation(true, use_evaluator)
}
fn full_test(testcases: Vec<&str>, top1: bool) {
    let mut summaries = IndexMap::default();
    for &testcase in &testcases {
        let summary = if top1 {
            top1_test(testcase)
        } else {
            perform_main_stage(testcase, STAGE::Complete, false)
        };
        summaries.insert(get_case(testcase).2, summary);
    }
    {
        println!(
            "{}",
            "\nFinal Report Sheet:".bold().underline().bright_blue()
        );
        let column_name = "TNS, Power, Area, Utilization, Score, 1-bit, 2-bit, 4-bit";
        println!("{}", column_name.bold().dimmed().underline());
        for (name, summary) in summaries {
            println!(
                "{}",
                format!(
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
                )
            );
        }
    }
}
fn main() {
    pretty_env_logger::init();
    // perform_main_stage("c1_1", STAGE::Complete, true);
    // perform_main_stage("c1_2", STAGE::Complete, true);
    // perform_main_stage("c2_1", STAGE::Complete, true);
    // perform_main_stage("c2_2", STAGE::Complete, true);
    // perform_main_stage("c2_3", STAGE::Complete, true);
    // perform_main_stage("c3_1", STAGE::Merging, true);
    // perform_main_stage("c3_2", STAGE::Complete, true);
    full_test(
        // vec!["c1_1", "c1_2", "c2_1", "c2_2", "c2_3", "c3_1", "c3_2"],
        vec!["c2_1"],
        false,
    );
}
