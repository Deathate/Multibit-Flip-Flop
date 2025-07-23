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
fn legalize_flipflops_multilevel(
    mbffg: &MBFFG,
    ffs: &Vec<SharedInst>,
    pcell_array: &PCellArray,
    range: ((usize, usize), (usize, usize)),
    (bits, full_ffs): (uint, &Vec<&LegalizeCell>),
    mut step: [usize; 2],
) -> Vec<LegalizeCell> {
    type Ele = Vec<(((usize, usize), (usize, usize)), [usize; 2], Vec<usize>)>;
    let mut legalization_lists = Vec::new();
    let mut queue = vec![(range, step, (0..full_ffs.len()).collect_vec())];
    let lib_list = &pcell_array.lib;
    let mut depth = 0;
    loop {
        let processed_elements = queue
            .iter_mut()
            .map(|(range, mut step, solution)| {
                let mut element_wrapper: (Ele, Vec<LegalizeCell>) = Default::default();
                let horizontal_span = range.0 .1 - range.0 .0;
                let vertical_span = range.1 .1 - range.1 .0;
                assert!(horizontal_span > 0);
                assert!(vertical_span > 0);
                let ffs = fancy_index_1d(full_ffs, &solution);
                let mut legalization_list = Vec::new();
                if depth == 1 || (horizontal_span == 1 && vertical_span == 1) {
                    let sub = pcell_array
                        .elements
                        .slice((range.0 .0..range.0 .1, range.1 .0..range.1 .1));
                    let mut group = PCellGroup::new();
                    group.add(sub);
                    GLOBAL_RECTANGLE.lock().unwrap().push(
                        PyExtraVisual::builder()
                            .id("rect")
                            .points(group.rect.to_2_corners().to_vec())
                            .line_width(10)
                            .color((255, 0, 100))
                            .build(),
                    );

                    let positions = pcell_array
                        .elements
                        .slice((range.0 .0..range.0 .1, range.1 .0..range.1 .1))
                        .into_iter()
                        .flat_map(|x| x.get(bits.i32()).iter().map(|x| &x.positions).collect_vec())
                        .flatten()
                        .collect_vec();

                    let mut min_value = f64::MAX;
                    let mut max_value = f64::MIN;
                    let mut items: Vec<(i32, Vec<f64>)> = ffs
                        .iter()
                        .map(|ff| {
                            let value_list: Vec<f64> = positions
                                .iter()
                                .map(|position| {
                                    let dis = norm1(ff.pos, **position);
                                    min_value = min_value.min(dis);
                                    max_value = max_value.max(dis);
                                    dis
                                })
                                .collect();
                            (1, value_list)
                        })
                        .collect();

                    for (item, ff) in items.iter_mut().zip(ffs.iter()) {
                        for value in item.1.iter_mut() {
                            *value = map_distance_to_value(*value, min_value, max_value).powf(0.9)
                                * ff.influence_factor.float();
                        }
                    }

                    let knapsack_capacities = vec![1; positions.len()];
                    // let knapsack_solution =
                    //     gurobi::solve_mutiple_knapsack_problem(&items, &knapsack_capacities);
                    let knapsack_solution = ffi::solveMultipleKnapsackProblem(
                        items.into_iter().map(Into::into).collect_vec(),
                        knapsack_capacities.into(),
                    );
                    for (&position, solution) in positions.into_iter().zip_eq(knapsack_solution) {
                        if solution.len() > 0 {
                            assert!(solution.len() == 1);
                            let index = solution[0].usize();
                            legalization_list.push(LegalizeCell {
                                index: ffs[index].index,
                                pos: position,
                                lib_index: 0,
                                influence_factor: 0,
                            });
                        }
                    }
                    element_wrapper.1 = legalization_list;
                } else {
                    if horizontal_span < step[0] {
                        if horizontal_span == 1 {
                            step[0] = 1;
                        } else {
                            step[0] = max(horizontal_span - 1, 2);
                        }
                    }
                    if vertical_span < step[1] {
                        if vertical_span == 1 {
                            step[1] = 1;
                        } else {
                            step[1] = max(vertical_span - 1, 2);
                        }
                    }

                    let sequence_range_column =
                        numpy::linspace(range.0 .0, range.0 .1, step[0] + 1);
                    let sequence_range_row = numpy::linspace(range.1 .0, range.1 .1, step[1] + 1);
                    let pcell_groups = sequence_range_column
                        .windows(2)
                        .flat_map(|cols| {
                            sequence_range_row.windows(2).map(move |rows| {
                                let (c1, c2) = (cols[0], cols[1]);
                                let (r1, r2) = (rows[0], rows[1]);
                                let sub = pcell_array.elements.slice((c1..c2, r1..r2));

                                let mut group = PCellGroup::new();
                                group.range = ((c1, c2), (r1, r2));
                                group.add(sub);
                                group
                            })
                        })
                        .collect_vec();
                    let mut min_value = f64::MAX;
                    let mut max_value = f64::MIN;
                    let mut items: Vec<(i32, Vec<f64>)> = ffs
                        .iter()
                        .map(|ff| {
                            let value_list: Vec<f64> = pcell_groups
                                .iter()
                                .map(|group| {
                                    if group.capacity(bits.i32()) > 0 {
                                        let value = group.distance(ff.pos);
                                        min_value = min_value.min(value);
                                        max_value = max_value.max(value);
                                        value
                                    } else {
                                        0.0
                                    }
                                })
                                .collect();
                            (1, value_list)
                        })
                        .collect();

                    for ((_, value_list), ff) in items.iter_mut().zip(&ffs) {
                        let factor = ff.influence_factor.float();
                        for value in value_list.iter_mut() {
                            *value = if *value > 0.0 {
                                map_distance_to_value(*value, min_value, max_value).powf(0.9)
                                    * factor
                            } else {
                                0.0
                            };
                        }
                    }
                    let knapsack_capacities = pcell_groups
                        .iter()
                        .map(|x| x.capacity(bits.i32()).i32())
                        .collect_vec();
                    let knapsack_solution = ffi::solveMultipleKnapsackProblem(
                        items.into_iter().map(Into::into).collect_vec(),
                        knapsack_capacities.into(),
                    );
                    // let knapsack_solution =
                    //     gurobi::solve_mutiple_knapsack_problem(&items, &knapsack_capacities);
                    let knapsack_results: Ele = knapsack_solution
                        .into_iter()
                        .enumerate()
                        .filter(|x| x.1.len() > 0)
                        .map(|(i, solution)| {
                            let range = pcell_groups[i].range;
                            (
                                range,
                                step,
                                solution.iter().map(|&x| ffs[x.usize()].index).collect_vec(),
                            )
                        })
                        .collect_vec();
                    element_wrapper.0 = knapsack_results;
                }
                element_wrapper
            })
            .collect::<Vec<_>>();
        queue.clear();
        processed_elements
            .into_iter()
            .for_each(|(knapsack_results, legalization_list)| {
                if knapsack_results.len() > 0 {
                    queue.extend(knapsack_results);
                    // queue.push(knapsack_results[0].clone());
                }
                legalization_lists.extend(legalization_list);
            });
        depth += 1;
        if depth == 1 {
            println!("Legalization depth: {}", depth);

            let gr_clone = GLOBAL_RECTANGLE.lock().unwrap().clone();
            GLOBAL_RECTANGLE.lock().unwrap().clear();
            for q in queue.iter() {
                let range = q.0;
                let ids = &q.2;
                let sub = pcell_array
                    .elements
                    .slice((range.0 .0..range.0 .1, range.1 .0..range.1 .1));
                let mut group = PCellGroup::new();
                group.add(sub);
                for id in ids {
                    ffs[*id].borrow_mut().move_to_pos(group.center());
                }
                GLOBAL_RECTANGLE.lock().unwrap().push(
                    PyExtraVisual::builder()
                        .id("rect")
                        .points(group.rect.to_2_corners().to_vec())
                        .line_width(10)
                        .color((255, 0, 100))
                        .build(),
                );
            }
            mbffg.visualize_layout(
                "",
                VisualizeOption::builder()
                    .shift_from_optimized(bits)
                    .build(),
            );
            GLOBAL_RECTANGLE.lock().unwrap().extend(gr_clone);
        }
        if queue.len() == 0 {
            break;
        }
    }
    legalization_lists
}
fn legalize_flipflops_full_place(
    mbffg: &MBFFG,
    ffs: &Vec<SharedInst>,
    pcell_array: &PCellArray,
    range: ((usize, usize), (usize, usize)),
    (bits, full_ffs): (uint, &Vec<&LegalizeCell>),
    mut step: [usize; 2],
    num_knapsacks: usize,
) -> Vec<LegalizeCell> {
    let mut group = PCellGroup::new();
    group.add_pcell_array(pcell_array);
    let num_items = full_ffs.len();
    let entries = group.get(bits.i32()).map(|x| [x.0, x.1]).collect_vec();
    let kdtree = ImmutableKdTree::new_from_slice(&entries);
    let positions = full_ffs.iter().map(|x| x.pos).collect_vec();
    let ffs_n100 = positions
        .iter()
        .map(|x| kdtree.nearest_n::<Manhattan>(&(*x).into(), NonZero::new(num_knapsacks).unwrap()))
        .collect_vec();
    let item_to_index_map: Dict<_, Vec<_>> = ffs_n100
        .iter()
        .enumerate()
        .flat_map(|(i, ff_list)| {
            ff_list
                .iter()
                .enumerate()
                .map(move |(j, ff)| (ff.item, (i, j)))
        })
        .fold(Dict::new(), |mut acc, (key, value)| {
            acc.entry(key).or_default().push(value);
            acc
        });
    let mut legalization_lists = Vec::new();
    let gurobi_output: grb::Result<_> = crate::redirect_output_to_null(false, || {
        let mut model = redirect_output_to_null(true, || {
            let env = Env::new("")?;
            let mut model = Model::with_env("multiple_knapsack", env)?;
            model.set_param(param::LogToConsole, 0)?;
            Ok::<_, grb::Error>(model)
        })
        .unwrap()
        .unwrap();

        let mut x: Vec<Vec<Var>> = (0..num_items)
            .map(|i| {
                (0..num_knapsacks)
                    .map(|j| add_binvar!(model, name: &format!("x_{}_{}", i, j)))
                    .collect::<Result<Vec<_>, _>>() // collect inner results
            })
            .collect::<Result<Vec<_>, _>>()?; // collect outer results
        for i in 0..num_items {
            model.add_constr(
                &format!("item_assignment_{}", i),
                c!((&x[i]).grb_sum() == 1),
            )?;
        }
        for (key, values) in &item_to_index_map {
            let constr_expr = values.iter().map(|(i, j)| x[*i][*j]).grb_sum();
            model.add_constr(&format!("knapsack_capacity_{}", key), c!(constr_expr <= 1))?;
        }
        // let min_distance = ffs_n100
        //     .iter()
        //     .flat_map(|x| x.iter().map(|ff| ff.distance))
        //     .fold(f64::MAX, |acc, x| acc.min(x));
        let (min_distance, max_distance) = ffs_n100
            .iter()
            .flat_map(|x| x.iter().map(|ff| ff.distance))
            .fold((f64::MAX, f64::MIN), |acc, x| (acc.0.min(x), acc.1.max(x)));

        let obj = (0..num_items)
            .map(|i| {
                let n100_flip_flop = &ffs_n100[i];
                (0..num_knapsacks)
                    .map(|j| {
                        let ff = n100_flip_flop[j];
                        let dis = ff.distance;
                        let value = map_distance_to_value(dis, min_distance, max_distance);
                        value * x[i][j]
                    })
                    .collect_vec()
            })
            .flatten()
            .grb_sum();
        model.set_objective(obj, Maximize)?;
        model.optimize()?;
        // Check the optimization result
        match model.status()? {
            Status::Optimal => {
                let mut result = vec![vec![false; num_knapsacks]; num_items];
                for i in 0..num_items {
                    for j in 0..num_knapsacks {
                        let val: f64 = model.get_obj_attr(attr::X, &x[i][j])?;
                        if val > 0.5 {
                            legalization_lists.push(LegalizeCell {
                                index: full_ffs[i].index,
                                pos: entries[ffs_n100[i][j].item.usize()].into(),
                                lib_index: 0,
                                influence_factor: 0,
                            });
                        }
                    }
                }
                return Ok(result);
            }
            Status::Infeasible => {
                error!("No feasible solution found.");
            }
            _ => {
                error!("Optimization was stopped with status {:?}", model.status()?);
            }
        }
        panic!();
    })
    .unwrap();
    legalization_lists
}
fn legalize_with_setup(
    mbffg: &mut MBFFG,
    ((row_step, col_step), pcell_array, _): &(Vector2, PCellArray, Vec<String>),
    num_knapsacks: usize,
) {
    info!("Legalization start");
    let mut placeable_bits = Vec::new();
    info!("Evaluate potential space");

    let mut group = PCellGroup::new();
    group.add_pcell_array(&pcell_array);
    let potential_space = group.summarize();
    let mut required_space = Dict::new();
    for ff in mbffg.get_free_ffs() {
        *required_space.entry(ff.borrow().bits()).or_insert(0) += 1;
    }
    for (bits, &count) in required_space.iter().sorted_by_key(|x| x.0) {
        let ev_space = *potential_space.get(&bits.i32()).or(Some(&0)).unwrap();
        if ev_space >= count {
            placeable_bits.push(*bits);
        }
        if DEBUG {
            debug!(
                "#{}-bit spaces: {} {} {} units",
                bits,
                ev_space,
                if ev_space >= count {
                    ">=".green()
                } else {
                    "< ".red()
                },
                count
            );
        }
    }
    if DEBUG {
        debug!("Placeable bits: {:?}", placeable_bits);
    }

    let ffs_classified = mbffg
        .get_ffs_classified()
        .into_iter()
        .filter(|(bits, _)| placeable_bits.contains(bits))
        .collect::<Dict<_, _>>();
    let shape = pcell_array.elements.shape();
    let classified_legalized_placement = ffs_classified
        .iter()
        .map(|(bits, ffs)| {
            let ffs = ffs
                .iter()
                .enumerate()
                .map(|(i, x)| LegalizeCell {
                    index: i,
                    pos: *x.get_optimized_pos(),
                    lib_index: 0,
                    influence_factor: 1,
                })
                .collect_vec();
            (bits, ffs)
        })
        .collect_vec();
    let classified_legalized_placement = classified_legalized_placement
        .into_iter()
        .map(|(bits, ffs_legalize_cell)| {
            // println!("# {}-bits: {}", bits, ffs_legalize_cell.len());
            let legalized_placement = redirect_output_to_null(false, || {
                legalize_flipflops_full_place(
                    &mbffg,
                    &ffs_classified[bits],
                    &pcell_array,
                    ((0, shape.0), (0, shape.1)),
                    (*bits, &ffs_legalize_cell.iter().collect_vec()),
                    [10, 10],
                    num_knapsacks,
                )
            })
            .unwrap();
            (bits, legalized_placement)
        })
        .collect::<Vec<_>>();

    for (bits, legalized_placement) in classified_legalized_placement {
        let ffs = &ffs_classified[bits];
        for x in legalized_placement {
            let ff = &ffs[x.index];
            // assert!(mbffg.is_on_site(x.pos));
            ff.move_to(x.pos.0, x.pos.1);
            // println!(
            //     "Legalized {}-bit FF {} to ({}, {})",
            //     bits,
            //     ff.borrow().name,
            //     x.pos.0,
            //     x.pos.1
            // );
            ff.set_legalized(true);
        }
    }
}
fn evaluate_placement_resource(
    mbffg: &MBFFG,
    restart: bool,
    candidates: Vec<uint>,
    includes: Option<Vec<uint>>,
) -> (Vector2, PCellArray, Vec<String>) {
    info!("Evaluate potential space");
    let (row_step, col_step) = mbffg
        .find_best_library_by_bit_count(4)
        .borrow()
        .ff_ref()
        .grid_coverage(&mbffg.setting.placement_rows[0]);
    let row_step = row_step.int() * 10;
    let col_step = col_step.int() * 10;
    let lib_candidates = candidates
        .iter()
        .map(|x| mbffg.find_best_library_by_bit_count(*x))
        .collect_vec();
    let ((row_step, col_step), pcell_array) = mbffg.evaluate_placement_resource(
        lib_candidates.clone(),
        includes.clone(),
        (row_step, col_step),
    );

    if DEBUG {
        let mut group = PCellGroup::new();
        group.add_pcell_array(&pcell_array);
        let potential_space = group.summarize();

        for (bits, &count) in potential_space.iter().sorted_by_key(|x| x.0) {
            debug!("#{}-bit spaces: {} units", bits, count);
        }

        let mut shaded_area = Vec::new();
        let num_placement_rows = mbffg.setting.placement_rows.len().i64();
        for i in (0..num_placement_rows).step_by(row_step.usize()) {
            let range_x =
                (i..min(i + row_step, mbffg.setting.placement_rows.len().i64())).collect_vec();
            let (min_pcell_y, max_pcell_y) = (
                mbffg.setting.placement_rows[range_x[0].usize()].y,
                mbffg.setting.placement_rows[range_x.last().unwrap().usize()].y
                    + mbffg.setting.placement_rows[range_x.last().unwrap().usize()].height,
            );
            let placement_row = &mbffg.setting.placement_rows[i.usize()];
            for j in (0..placement_row.num_cols).step_by(col_step.usize()) {
                let range_y = (j..min(j + col_step, placement_row.num_cols)).collect_vec();
                let (min_pcell_x, max_pcell_x) = (
                    placement_row.x + range_y[0].float() * placement_row.width,
                    placement_row.x + (range_y.last().unwrap() + 1).float() * placement_row.width,
                );
                shaded_area.push(
                    PyExtraVisual::builder()
                        .id("rect")
                        .points(vec![(min_pcell_x, min_pcell_y), (max_pcell_x, max_pcell_y)])
                        .line_width(10)
                        .color((0, 0, 0))
                        .build(),
                );
            }
        }
        let mut pcell_group = PCellGroup::new();
        let shape = pcell_array.elements.shape();
        pcell_group.add_pcell_array(&pcell_array);
        let mut ffs = Vec::new();
        for (lib_name, pos) in pcell_group.iter_named() {
            let lib = mbffg
                .retrieve_ff_libraries()
                .iter()
                .find(|x| x.borrow().ff_ref().name() == &lib_name)
                .unwrap();
            ffs.extend(pos.iter().map(|&x| Pyo3Cell {
                name: "FF".to_string(),
                x: x.0,
                y: x.1,
                width: lib.borrow().ff_ref().width(),
                height: lib.borrow().ff_ref().height(),
                walked: false,
                highlighted: false,
                pins: vec![],
            }));
        }
        Python::with_gil(|py| {
            let script = c_str!(include_str!("script.py")); // Include the script as a string
            let module = PyModule::from_code(py, script, c_str!("script.py"), c_str!("script"))?;

            let file_name = format!("tmp/potential_space_{:?}_{:?}.png", candidates, includes);
            let mut sticked_insts = mbffg.get_all_gate().map(|x| Pyo3Cell::new(x)).collect_vec();
            let includes_set = includes
                .unwrap_or_default()
                .iter()
                .cloned()
                .collect::<Set<_>>();
            sticked_insts.extend(
                mbffg
                    .get_free_ffs()
                    .filter(|x| includes_set.contains(&x.borrow().bits()))
                    .map(|x| Pyo3Cell::new(x)),
            );
            module.getattr("draw_layout")?.call1((
                false,
                &file_name,
                mbffg.setting.die_size.clone(),
                f32::INFINITY,
                f32::INFINITY,
                mbffg.setting.placement_rows.clone(),
                ffs,
                sticked_insts,
                mbffg.get_all_io().map(|x| Pyo3Cell::new(x)).collect_vec(),
                shaded_area,
            ))?;
            Ok::<(), PyErr>(())
        })
        .unwrap();
    }
    (
        (row_step.f64(), col_step.f64()),
        pcell_array,
        lib_candidates
            .iter()
            .map(|x| x.borrow().ff_ref().name().clone())
            .collect_vec(),
    )
}
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
    mbffg.create_prev_ff_cache();
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
fn load_placement_cache(mbffg: &MBFFG, force: bool) -> Vec<((f64, f64), PCellArray, Vec<String>)> {
    let placement_files = [
        ("placement4.json", 4),
        ("placement2.json", 2),
        ("placement1.json", 1),
    ];
    placement_files
        .into_iter()
        .map(|(file_name, bits)| {
            let stem = PathLike::new(&mbffg.input_path).stem().unwrap();
            let file_name = format!("{}_{}", stem, file_name);
            if force || !exist_file(&file_name).unwrap() {
                let evaluation = evaluate_placement_resource(mbffg, true, vec![bits], None);
                save_to_file(&evaluation, &file_name).unwrap();
                evaluation
            } else {
                load_from_file::<_>(&file_name).unwrap()
            }
        })
        .collect_vec()
}
fn placement(mbffg: &mut MBFFG, num_knapsacks: usize, cache: bool, force: bool) {
    if cache {
        let mut evaluations = load_placement_cache(mbffg, force);
        let mut placed_bits = Vec::new();
        for evaluation in evaluations.iter_mut() {
            let lib_name = &evaluation.2;
            crate::assert_eq!(lib_name.len(), 1);
            let lib = mbffg.get_lib(lib_name[0].as_str());
            let bit = lib.borrow().ff_ref().bits.usize();
            placed_bits.push(bit);
            info!("Legalizing {}-bit FFs", bit);
            let (w, h) = lib.borrow().ff_ref().size();
            let mut rtree = Rtree::new();
            let gates = mbffg.get_all_gate().map(|x| x.bbox());
            let ffs = mbffg.get_legalized_ffs().map(|x| x.bbox());
            let rects = gates.chain(ffs).collect_vec();
            rtree.bulk_insert(&rects);
            for pcell in evaluation.1.elements.iter_mut() {
                pcell.filter(&rtree, (w, h));
            }
            legalize_with_setup(mbffg, evaluation, num_knapsacks);
            mbffg.visualize_layout(
                &format!("leg_bit_{}", placed_bits.iter().join("_")),
                VisualizeOption::builder()
                    .bits(Some(placed_bits.clone()))
                    .build(),
            );
        }
    } else {
        let evaluation = evaluate_placement_resource(mbffg, true, vec![4], None);
        legalize_with_setup(mbffg, &evaluation, num_knapsacks);
        mbffg.visualize_layout("leg_bit_4", VisualizeOption::builder().build());

        let evaluation = evaluate_placement_resource(mbffg, true, vec![2], Some(vec![4]));
        legalize_with_setup(mbffg, &evaluation, num_knapsacks);
        mbffg.visualize_layout("leg_bit_2", VisualizeOption::builder().build());

        let evaluation = evaluate_placement_resource(mbffg, true, vec![1], Some(vec![4, 2]));
        legalize_with_setup(mbffg, &evaluation, num_knapsacks);
        mbffg.visualize_layout("leg_bit_1", VisualizeOption::builder().build());
    }
}
// fn placement_full_place(mbffg: &mut MBFFG, num_knapsacks: usize, force: bool) {
//     let evaluations = load_placement_cache(mbffg, force);
//     let mut group = PCellGroup::new();
//     for evaluation in &evaluations {
//         group.add_pcell_array(&evaluation.1);
//     }
//     let ffs_classified = mbffg.get_ffs_classified();
//     let lib_candidates: Dict<_, _> = evaluations
//         .iter()
//         .map(|x| {
//             assert!(x.2.len() == 1);
//             let lib = mbffg.get_lib(x.2[0].as_str());
//             let bit = lib.borrow().ff_ref().bits;
//             (bit, lib)
//         })
//         .collect();
//     let mut rtree = rtree_id::RtreeWithData::new();
//     let eps = 0.1;
//     let positions: Dict<_, _> = lib_candidates
//         .iter()
//         .map(|(&bits, _)| {
//             let pos_vec = group.get(bits.i32()).collect_vec();
//             (bits, pos_vec)
//         })
//         .collect();
//     let items = positions
//         .iter()
//         .flat_map(|(&bit, pos_vec)| {
//             let lib = lib_candidates[&bit.u64()].borrow();
//             pos_vec
//                 .iter()
//                 .enumerate()
//                 .map(move |(index, &(x, y))| ([x, y], (bit, index.u64())))
//         })
//         .collect_vec();
//     let bboxs: Dict<_, _> = items.iter().fold(Dict::new(), |mut acc, (bbox, id)| {
//         acc.insert(id.clone(), bbox.clone());
//         acc
//     });
//     rtree.bulk_insert(items);

//     let mut overlap_constrs = Dict::new();
//     let mut ffs_n100_map = Dict::new();
//     let gurobi_output: grb::Result<_> = crate::redirect_output_to_null(false, || {
//         let mut model = redirect_output_to_null(true, || {
//             let env = Env::new("")?;
//             let model = Model::with_env("multiple_knapsack", env)?;
//             // model.set_param(param::LogToConsole, 0)?;
//             Ok::<grb::Model, grb::Error>(model)
//         })
//         .unwrap()
//         .unwrap();
//         let mut objs = Vec::new();
//         let mut vars_x = Vec::new();
//         for (bits, _) in &lib_candidates {
//             let full_ffs = &ffs_classified[bits];
//             let num_items = full_ffs.len();
//             let num_knapsacks = 100;
//             let entries = positions[bits].iter().map(|&x| [x.0, x.1]).collect_vec();
//             let kdtree = kiddo::ImmutableKdTree::new_from_slice(&entries);
//             let positions = full_ffs.iter().map(|x| x.borrow().pos()).collect_vec();
//             let ffs_n100 = positions
//                 .iter()
//                 .map(|&x| {
//                     kdtree.nearest_n::<Manhattan>(&x.into(), NonZero::new(num_knapsacks).unwrap())
//                 })
//                 .collect_vec();
//             let item_to_index_map: Dict<_, Vec<_>> = ffs_n100
//                 .iter()
//                 .enumerate()
//                 .flat_map(|(i, ff_list)| {
//                     ff_list
//                         .iter()
//                         .enumerate()
//                         .map(move |(j, ff)| (ff.item, (i, j)))
//                 })
//                 .fold(Dict::new(), |mut acc, (key, value)| {
//                     acc.entry(key).or_default().push(value);
//                     acc
//                 });
//             {
//                 let mut x: Vec<Vec<Var>> = (0..num_items)
//                     .map(|i| {
//                         (0..num_knapsacks)
//                             .map(|j| add_binvar!(model, name: &format!("x_{}_{}", i, j)))
//                             .collect::<Result<Vec<_>, _>>() // collect inner results
//                     })
//                     .collect::<Result<Vec<_>, _>>()?; // collect outer results
//                 vars_x.push(x);
//             }
//             let mut x = vars_x.last().unwrap();

//             for i in 0..num_items {
//                 model.add_constr(
//                     &format!("item_assignment_{}", i),
//                     c!((&x[i]).grb_sum() == 1),
//                 )?;
//             }
//             let mut constrs = Dict::new();
//             for (key, values) in &item_to_index_map {
//                 let constr_expr = values.iter().map(|(i, j)| x[*i][*j]).grb_sum();
//                 let tmp_var = add_binvar!(model)?;
//                 model.add_constr("", c!(tmp_var == constr_expr))?;
//                 constrs.insert(*key, tmp_var);
//             }
//             overlap_constrs.insert(*bits, constrs);
//             let obj = (0..num_items)
//                 .map(|i| {
//                     let cands = &ffs_n100[i];
//                     (0..num_knapsacks).map(move |j| {
//                         let ff = cands[j];
//                         let dis = ff.distance / bits.f64();
//                         let value = -dis;
//                         value * x[i][j]
//                     })
//                 })
//                 .flatten()
//                 .grb_sum();
//             objs.push(obj);
//             ffs_n100_map.insert(*bits, ffs_n100);
//         }
//         let mut constr_group = Vec::new();
//         let mut hist = Vec::new();
//         for (&bit, constrs) in overlap_constrs.iter().sorted_by_key(|x| Reverse(x.1.len())) {
//             if overlap_constrs.len() > 1 && hist.len() == overlap_constrs.len() - 1 {
//                 break;
//             }
//             for (&key, _) in constrs {
//                 // model.add_constr(&format!("knapsack_capacity_{}", key), c!(constr <= 1))?;
//                 let id = (bit, key);
//                 let bbox = bboxs[&id];
//                 let intersects = rtree.intersection(bbox[0], bbox[1]);
//                 let mut group = Vec::new();
//                 for intersect in intersects {
//                     let intersect_id = intersect.data;
//                     if intersect_id.1 == id.1 {
//                         continue;
//                     }
//                     let intersect_bit = intersect_id.0;
//                     if !hist.contains(&intersect_bit) {
//                         group.push([id, intersect_id]);
//                     }
//                 }
//                 constr_group.extend(group);
//             }
//             hist.push(bit);
//         }
//         for group in constr_group.iter() {
//             let expr = group
//                 .into_iter()
//                 .filter_map(|(a, b)| overlap_constrs.get(&a).and_then(|m| m.get(&b)))
//                 .cloned()
//                 .grb_sum();
//             model.add_constr("", c!(expr <= 1))?;
//         }
//         model.set_objective(objs.grb_sum(), Maximize)?;
//         model.optimize()?;
//         // Check the optimization result
//         match model.status()? {
//             Status::Optimal => {
//                 for ((bits, _), vars) in lib_candidates.iter().zip(vars_x.iter()) {
//                     let full_ffs = &ffs_classified[bits];
//                     for (i, row) in vars.iter().enumerate() {
//                         for (j, var) in row.iter().enumerate() {
//                             if model.get_obj_attr(attr::X, var)? > 0.5 {
//                                 let ff = &full_ffs[i];
//                                 let pos = positions[bits][ffs_n100_map[bits][i][j].item.usize()];
//                                 ff.move_to_pos(*pos);
//                                 // println!(
//                                 //     "Legalized {}-bit FF {} to ({}, {})",
//                                 //     bits,
//                                 //     ff.borrow().name,
//                                 //     pos.0,
//                                 //     pos.1
//                                 // );
//                             }
//                         }
//                     }
//                 }
//                 return Ok(());
//             }
//             Status::Infeasible => {
//                 println!("No feasible solution found.");
//             }
//             _ => {
//                 println!("Optimization was stopped with status {:?}", model.status()?);
//             }
//         }
//         panic!("Optimization failed.");
//     })
//     .unwrap();
//     // let evaluation = evaluate_placement_resource(mbffg, true, vec![1], Some(vec![4, 2]));
//     // crate::redirect_output_to_null(false, || {
//     //     legalize_with_setup(mbffg, &evaluation, num_knapsacks)
//     // })
//     // .unwrap();
// }

// fn timing_debug(){
//     {
//         let mut q = vec![];
//         for ff in mbffg.get_all_ffs() {
//             let r = mbffg
//                 .get_prev_ff_records(ff)
//                 .iter()
//                 .map(|x| OrderedFloat(x.distance()))
//                 .collect_vec();
//             if r.is_empty() {
//                 continue;
//             }
//             let delta = r.iter().max().unwrap() - r.iter().min().unwrap();
//             q.push((delta, ff.slack(), ff.clone()));
//         }
//         q.sort_by_key(|x| Reverse(x.0));
//         // q.iter().take(100).for_each(|x| println!("{:?}", x));
//         q.iter().take(1).for_each(|x| {
//             mbffg.get_prev_ff_records(&x.2).iter().for_each(|record| {
//                 record.ff_q.as_ref().unwrap().0.borrow().set_walked(true);
//             });
//             x.2.set_highlighted(true);
//             mbffg
//                 .get_prev_ff_records(&x.2)
//                 .iter()
//                 .max_by_key(|x| OrderedFloat(x.distance()))
//                 .unwrap()
//                 .ff_q
//                 .as_ref()
//                 .unwrap()
//                 .0
//                 .borrow()
//                 .set_highlighted(true);
//             // mbffg.visualize_mindmap(&x.2.get_name(), true, Some(20));
//             let farest = mbffg
//                 .get_prev_ff_records(&x.2)
//                 .iter()
//                 .max_by_key(|x| OrderedFloat(x.distance()))
//                 .unwrap()
//                 .ff_q
//                 .as_ref()
//                 .unwrap()
//                 .0
//                 .inst();
//             // farest.move_relative(100.0, 100.0);
//             x.2.get_name().print();
//             mbffg
//                 .get_prev_ff_records_util("C98441")
//                 .iter()
//                 .for_each(|x| {
//                     x.ff_q.as_ref().unwrap().0.inst_name().print();
//                 });
//             // mbffg.move_util("C97567", 15300, 16800);
//             let f = x.2.pos();
//             let s = farest.pos();
//             mbffg.move_util("C98441", 15300.0, 1283100);
//             x.2.get_name().print();
//             // mbffg.move_relative_util("C97566", 100.0, 100.0);
//             // mbffg.move_relative_util("C98381", 100.0, 100.0);
//             // mbffg.visualize_mindmap("C98441", true, None);
//             // exit();
//         });
//     }
// }
fn initial_score() {
    let file_names = ["c1_1", "c1_2", "c2_1", "c2_2", "c2_3", "c3_1", "c3_2"];
    for file_name in file_names {
        let mut mbffg = MBFFG::new(get_case(file_name).0);
        mbffg.check(true, false);
    }
    exit();
}
fn debug_case2() {
    info!("Debugging case 2");
    let case_name = "c2_1";
    let (file_name, top1_name) = get_case(case_name);
    info!("File name: {}", file_name);
    info!("Top1 name: {}", top1_name);
    let mut mbffg = MBFFG::new(file_name);
    mbffg.filter_timing = false;
    // mbffg.load(top1_name);
    let cols = mbffg
        .get_all_ffs()
        .map(|x| (x.clone(), mbffg.get_next_ffs_count(x)))
        .sorted_by_key(|x| x.1)
        .collect_vec();
    cols.iter().for_each(|(ff, count)| {
        info!("{}: {}", ff.get_name(), count);
    });
    exit();
    cols.last().unwrap().0.prints();
    // mbffg.next_ffs_util("C106823").len().prints();
    // mbffg.get_next_ffs_util("C106823").len().prints();
    // exit();
    // let first = &cols[0].0;
    let last = mbffg.get_inst("C117042");
    let last = &cols.last().unwrap().0;
    last.prints();
    // mbffg.get_next_ffs(last).iter().for_each(|x| {
    //     x.full_name().print();
    // });
    // mbffg
    //     .get_prev_ff_records(&mbffg.get_inst("C117042"))
    //     .iter()
    //     .for_each(|x| {
    //         x.prints();
    //     });
    last.move_relative(-10000.0, 0.0);
    // mbffg.sta();
    mbffg.check(false, true);
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
        STAGE::Initial => "stage_initial",
        STAGE::Merging => "stage_merging",
        STAGE::TimingOptimization => "stage_timing_optimization",
        STAGE::DetailPlacement => "stage_detail_placement",
    }
}
#[tokio::main]
async fn actual_main() {
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
    let ff = mbffg.get_all_ffs().next().unwrap();
    // mbffg.prev_ffs_cache[&ff.dpins()[0]].prints();
    // cal_max_record(
    //     mbffg.prev_ffs_cache[&ff.dpins()[0]].iter(),
    //     mbffg.displacement_delay(),
    // )
    // .prints();
    ff.move_to(0, 0);
    // cal_max_record(
    //     mbffg.prev_ffs_cache[&ff.dpins()[0]].iter(),
    //     mbffg.displacement_delay(),
    // )
    // .prints();
    // exit();
    mbffg.check(false, true);
    exit();
    if STAGE_STATUS == STAGE::Initial {
        mbffg.visualize_layout(
            stage_to_name(STAGE::Initial),
            VisualizeOption::builder().build(),
        );
        let debanked = mbffg.debank_all_multibit_ffs();
        mbffg.replace_1_bit_ffs();
        {
            // let mut p = UncoveredPlaceLocator::new(&mbffg, &mbffg.find_all_best_library(), false);
            // let f = mbffg.get_ff("m_C118015");
            // let ori_pos = f.pos();
            // for ff in mbffg.get_all_ffs() {
            //     let pos = p.find_nearest_uncovered_place(1, ff.pos()).unwrap();
            //     p.update_uncovered_place(1, pos);
            // }
            // f.move_to_pos(
            //     p.find_nearest_uncovered_place(1, (ori_pos.0, ori_pos.1))
            //         .unwrap(),
            // );
            // mbffg.check(false, true);
            // exit();
        }
        {
            let tmr = stimer!("Initial Placement");
            mbffg.create_prev_ff_cache();
        }
        mbffg.structure_change = true;
        {
            let tmr = stimer!("Initial Placement");
            mbffg.create_prev_ff_cache();
        }
        // {
        //     let tmr = stimer!("Initial Placement");
        //     mbffg.create_prev_ff_cache();
        // }
        exit();
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
            mbffg.merge(
                &mbffg.get_clock_groups()[0]
                    .iter()
                    .map(|x| x.inst())
                    .collect_vec(),
                move_to_center,
                2,
            );
            exit();
            mbffg.merge(
                &mbffg.get_ffs_by_bit(2).cloned().collect_vec(),
                move_to_center,
                2,
            );
        } else if SELECTION == 1 {
            mbffg.gurobi_merge(
                &mbffg.get_clock_groups()[0]
                    .iter()
                    .map(|x| x.inst())
                    .collect_vec(),
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
        mbffg.load(LOAD_FROM_FILE);
        mbffg.check(true, false);
        {
            let tmr = stimer!("TIMING_OPTIMIZATION");
            info!("Timing optimization");
            let timing = mbffg.get_ffs_sorted_by_timing();
            let num_timing = timing.len();
            info!("Number of timing critical flip-flops: {}", num_timing);
            let negative_timing_slacks = timing
                .iter()
                .map(|x| mbffg.negative_timing_slack_inst(x))
                .collect_vec();
            let ratio_count = count_to_reach_percent(&negative_timing_slacks, 0.8);
            info!(
                "Number of timing critical flip-flops to reach 80%: {}",
                ratio_count
            );
            for ff in timing.iter() {
                ff.set_optimized_pos(ff.pos());
            }
            for op_group in &timing.iter().take(2500).chunks(500) {
                let optimized_pos =
                    gurobi::optimize_multiple_timing(&mbffg, &op_group.collect_vec(), 0.3).unwrap();
                for (ff_id, pos) in optimized_pos.iter() {
                    let ff = mbffg.get_node(*ff_id);
                    ff.move_to_pos(*pos);
                }
                mbffg.check(true, false);
            }
            mbffg.output(stage_to_name(STAGE::TimingOptimization));
            mbffg.visualize_layout(
                stage_to_name(STAGE::TimingOptimization),
                VisualizeOption::builder().shift_from_optimized(4).build(),
            );
            finish!(tmr, "Timing optimization done");
            mbffg.timing_analysis();
        }
    } else if STAGE_STATUS == STAGE::TimingOptimization {
        mbffg.load(LOAD_FROM_FILE);
        mbffg.create_prev_ff_cache();
        for ff in mbffg.get_all_ffs() {
            ff.set_optimized_pos(ff.pos());
        }
        let mut gate_rtree = mbffg.generate_gate_map();
        let rows = mbffg.placement_rows();
        let die_size = mbffg.die_size();
        for bit in mbffg.unique_library_bit_widths() {
            let lib_size = mbffg
                .find_best_library_by_bit_count(bit)
                .borrow()
                .ff_ref()
                .size();
            let tmr = stimer!("Placement Resources Evaluation");
            let positions = helper::evaluate_placement_resources_from_size(
                &gate_rtree,
                rows,
                die_size,
                lib_size,
            );
            finish!(tmr, "Placement Resources Evaluation done");
            mbffg.visualize_placement_resources(&positions, lib_size);
            let ffs = mbffg.get_ffs_by_bit(bit).collect_vec();
            let result = gurobi::assignment_problem(&ffs, positions, 50).unwrap();
            for (i, pos) in result.iter().enumerate() {
                let ff = &ffs[i];
                ff.move_to_pos(*pos);
                gate_rtree.insert_bbox(ff.bbox());
            }
            mbffg.visualize_layout(
                stage_to_name(STAGE::DetailPlacement),
                VisualizeOption::builder().shift_from_optimized(bit).build(),
            );
        }
        mbffg.check(true, true);
        mbffg.output(stage_to_name(STAGE::DetailPlacement));
        mbffg.timing_analysis();
    } else {
        panic!("Unknown stage: {:?}", STAGE_STATUS);
    }
}
fn main() {
    pretty_env_logger::init();
    actual_main();
}
