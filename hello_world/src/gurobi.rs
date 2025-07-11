use crate::class::*;
use crate::mbffg::*;
use crate::util::*;
use crate::*;
use easy_print::*;
use grb::prelude::*;
pub struct GRBLinExpr {
    expr: Expr,
}
impl GRBLinExpr {
    pub fn new() -> Self {
        Self {
            expr: Expr::from(0.0),
        }
    }
    pub fn from(vars: &Vec<Var>) -> Self {
        Self {
            expr: vars.iter().grb_sum(),
        }
    }
}
impl std::ops::AddAssign<Expr> for GRBLinExpr {
    fn add_assign(&mut self, other: Expr) {
        self.expr = self.expr.clone() + other;
    }
}
impl From<GRBLinExpr> for Expr {
    fn from(expr: GRBLinExpr) -> Self {
        expr.expr
    }
}
/// Solves the multiple knapsack problem.
///
/// Given a collection of n items, each with a specific weight and value. Additionally, you have m bins, where each bin has a maximum capacity. The goal is to pack a subset of the items into multiple bins such that:
/// 1. The total weight of the items in each bin does not exceed the bin's capacity.
/// 2. The total value of the items packed into the bins is maximized.
///
///
/// # Arguments
///
/// * `items` - Each tuple represents (weight, values for each bin) of an item
/// * `knapsack_capacities` - Capacities limits of the knapsacks
pub fn solve_mutiple_knapsack_problem(
    items: &Vec<(i32, Vec<f64>)>,
    knapsack_capacities: &Vec<i32>,
) -> Vec<Vec<usize>> {
    if items.is_empty() {
        return Vec::new();
    }
    let total_capacity: i32 = knapsack_capacities.iter().sum();
    assert!(items.len().i32() <= total_capacity, "Not enough knapsacks.");
    let gurobi_output: grb::Result<_> = crate::redirect_output_to_null(false, || {
        let num_items = items.len();
        let num_knapsacks = knapsack_capacities.len();
        // Create a new model
        let mut model = redirect_output_to_null(true, || {
            let env = Env::new("")?;
            let model = Model::with_env("multiple_knapsack", env)?;
            Ok::<grb::Model, grb::Error>(model)
        })
        .unwrap()
        .unwrap();
        model.set_param(param::LogToConsole, 0)?;
        // Decision variables: x[i][j] = 1 if item i is placed in knapsack j, else 0
        let mut x = vec![Vec::with_capacity(num_knapsacks); num_items];
        for i in 0..num_items {
            for j in 0..num_knapsacks {
                let var = add_binvar!(model, name: &format!("x_{}_{}", i, j))?;
                x[i].push(var);
            }
        }
        // Constraint 1: Each item can be assigned to at most one knapsack
        for i in 0..num_items {
            let assignment = GRBLinExpr::from(&x[i]);
            model.add_constr(&format!("item_assignment_{}", i), c!(assignment == 1))?;
        }
        // Constraint 2: The total weight of items in each knapsack must not exceed its capacity
        for j in 0..num_knapsacks {
            let mut total_weight = GRBLinExpr::new();
            for i in 0..num_items {
                total_weight += x[i][j] * items[i].0;
            }
            model.add_constr(
                &format!("knapsack_capacity_{}", j),
                c!(total_weight <= knapsack_capacities[j]),
            )?;
        }
        // Objective:  Maximize the total packed item values
        let mut obj = GRBLinExpr::new();
        for i in 0..num_items {
            for j in 0..num_knapsacks {
                obj += x[i][j] * items[i].1[j];
            }
        }
        model.set_objective(obj, Maximize)?;
        // Optimize the model
        model.optimize()?;
        // Check the optimization result
        match model.status()? {
            Status::Optimal => {
                // println!("Optimal solution found:");
                // for j in 0..num_knapsacks {
                //     println!("Knapsack {}:", j);
                //     for i in 0..num_items {
                //         let val: f64 = model.get_obj_attr(attr::X, &x[i][j])?;
                //         if val > 0.5 {
                //             println!(
                //                 "  Item {} (weight: {}, value: {})",
                //                 i + 1,
                //                 items[i].0,
                //                 items[i].1[j]
                //             );
                //         }
                //     }
                // }
                let mut result = vec![vec![false; num_knapsacks]; num_items];
                for i in 0..num_items {
                    for j in 0..num_knapsacks {
                        let val: f64 = model.get_obj_attr(attr::X, &x[i][j])?;
                        if val > 0.5 {
                            result[i][j] = true;
                        }
                    }
                }
                return Ok(result);
            }
            Status::Infeasible => {
                println!("No feasible solution found.");
            }
            _ => {
                println!("Optimization was stopped with status {:?}", model.status()?);
            }
        }
        panic!("Optimization failed.");
    })
    .unwrap();
    let gurobi_output = gurobi_output.unwrap();
    let shape = shape(&gurobi_output);
    let (num_items, num_knapsacks) = (shape.0, shape.1);
    let mut result = vec![Vec::new(); num_knapsacks];
    for i in 0..num_items {
        let mut assigned = false;
        for j in 0..num_knapsacks {
            let val = gurobi_output[i][j];
            if val == true {
                result[j].push(i);
                assigned = true;
            }
        }
        if assigned == false {
            println!("Number of items: {}", num_items);
            println!("Total capacity: {}", total_capacity);
            println!("Item {} is not assigned to any knapsack.", i);
            items[i].1.prints();
            panic!("Item not assigned.");
        }
    }
    result
}

struct Cell {
    x: Var,
    y: Var,
}
fn norm1(model: &mut Model, v1: (Expr, Expr), v2: (Expr, Expr)) -> grb::Result<Expr> {
    let (abs_delta_x, abs_delta_y) = (add_ctsvar!(model)?, add_ctsvar!(model)?);
    let var1 = add_ctsvar!(model, bounds: ..)?;
    let var2 = add_ctsvar!(model, bounds: ..)?;
    model.add_constr("", c!(var1 == v1.0 - v2.0))?;
    model.add_constr("", c!(var2 == v1.1 - v2.1))?;
    model.add_genconstr_abs("", abs_delta_x, var1)?;
    model.add_genconstr_abs("", abs_delta_y, var2)?;
    Ok(abs_delta_x + abs_delta_y)
}
pub fn optimize_single_timing(
    mbffg: &mut MBFFG,
    insts: &Vec<&SharedInst>,
) -> grb::Result<(float, float)> {
    let mut model = redirect_output_to_null(true, || {
        let env = Env::new("")?;
        let mut model = Model::with_env("", env)?;
        model.set_param(param::LogToConsole, 0)?;
        Ok::<_, grb::Error>(model)
    })
    .unwrap()
    .unwrap();
    let optimized_cell_ids: Set<usize> = insts.iter().map(|x| x.get_gid()).collect();
    let x_var = Cell {
        x: add_ctsvar!(model).unwrap(),
        y: add_ctsvar!(model).unwrap(),
    };
    let get_position = |pin: &SharedPhysicalPin| -> (Expr, Expr) {
        if optimized_cell_ids.contains(&pin.get_gid()) {
            let (x, y) = pin.relative_pos();
            (x_var.x + x, x_var.y + y)
        } else {
            let (x, y) = pin.pos();
            (Expr::from(x), Expr::from(y))
        }
    };

    let mut negative_delay_vars = Vec::new();
    let displacement_delay = mbffg.displacement_delay();
    let dpins = mbffg.get_effected_dpins(insts);
    let mut num_record = 0;
    for dpin in &dpins {
        let records = mbffg.get_prev_ff_records(dpin);
        let cloned_records = records.iter().cloned().collect_vec();
        if cloned_records.is_empty() {
            continue;
        }
        let max_record = cal_max_record(&cloned_records, displacement_delay);
        let max_delay = max_record.calculate_total_delay(displacement_delay);
        let slack = dpin.get_slack();
        let ori_delay = dpin.get_origin_delay();
        let ff_d_dist = max_record.ff_d_dist();
        let mut fixed_record = Vec::new();
        let mut dynamic_records = Vec::new();
        for record in records.iter() {
            if record.ff_q.is_some() {
                dynamic_records.push(record);
            } else {
                fixed_record.push(record);
            }
        }
        num_record += dynamic_records.len();
        let mut max_fixed_delay = 0.0;
        if !fixed_record.is_empty() {
            let max_fixed_record = fixed_record
                .iter()
                .max_by_key(|record| OrderedFloat(record.calculate_total_delay(displacement_delay)))
                .unwrap();
            max_fixed_delay = max_fixed_record.calculate_total_delay(displacement_delay)
                - ff_d_dist * displacement_delay;
        }
        let mut vars = Vec::new();
        for record in dynamic_records.iter() {
            let var = add_ctsvar!(model, bounds: ..).unwrap();
            let ff_q_pin = record.ff_q.as_ref().unwrap();
            let ff_q_expr = norm1(
                &mut model,
                get_position(&ff_q_pin.0),
                get_position(&ff_q_pin.1),
            )
            .unwrap();

            let unchanged_delay = record.calculate_total_delay(displacement_delay)
                - (record.ff_q_dist() + ff_d_dist) * displacement_delay;
            model
                .add_constr(
                    "",
                    c!(var >= unchanged_delay + ff_q_expr * displacement_delay),
                )
                .unwrap();
            vars.push(var);
        }
        let max_var = add_ctsvar!(model, bounds: ..)?;
        model.add_genconstr_max(
            &format!("max_vars_{}", dpin.get_gid()),
            max_var,
            vars,
            Some(max_fixed_delay),
        )?;
        let var = add_ctsvar!(model, bounds: ..)?;
        if let Some(ff_d) = max_record.ff_d.as_ref() {
            let ff_d_expr =
                norm1(&mut model, get_position(&ff_d.0), get_position(&ff_d.1)).unwrap();
            model
                .add_constr(
                    "",
                    c!(var >= max_var + ff_d_expr * displacement_delay - max_delay),
                )
                .unwrap();
        } else {
            model
                .add_constr("", c!(var <= slack + ori_delay - max_var))
                .unwrap();
        }
        model.add_constr("", c!(var <= 0.0)).unwrap();
        negative_delay_vars.push(var);
    }

    debug!("Solve {} objs...", negative_delay_vars.len());
    let obj = negative_delay_vars.iter().grb_sum();
    model.set_objective(obj, Maximize)?;
    model.optimize()?;
    match model.status()? {
        Status::Optimal => {
            let optimized_pos = (
                model.get_obj_attr(attr::X, &x_var.x)?,
                model.get_obj_attr(attr::X, &x_var.y)?,
            );
            for inst in insts {
                let pos = inst.pos();
                println!(
                    "{}: move ({}, {}) to ({}, {})",
                    inst.get_name(),
                    pos.0,
                    pos.1,
                    optimized_pos.0,
                    optimized_pos.1
                );
            }
            // let objective_value = model.get_attr(attr::ObjVal)?;
            // debug!("Objective value: {}", objective_value);
            debug!("dpins count: {}", dpins.len());
            debug!("Found {} previous flip-flop records", num_record,);
            return Ok(optimized_pos);
        }
        Status::InfOrUnbd => {
            println!("No feasible solution found.");
        }
        _ => {
            println!("Optimization was stopped with status {:?}", model.status()?);
        }
    }
    panic!("Optimization failed.");
}
pub fn optimize_multiple_timing(
    mbffg: &mut MBFFG,
    insts: &[&SharedInst],
    simplified_ratio: float,
) -> grb::Result<Dict<usize, (float, float)>> {
    let mut model = redirect_output_to_null(true, || {
        let env = Env::new("")?;
        let model = Model::with_env("", env)?;
        Ok::<_, grb::Error>(model)
    })
    .unwrap()
    .unwrap();
    model.set_param(param::LogToConsole, 0)?;
    let x_var: Dict<_, _> = insts
        .iter()
        .map(|x| {
            (
                x.get_gid(),
                Cell {
                    x: add_ctsvar!(model).unwrap(),
                    y: add_ctsvar!(model).unwrap(),
                },
            )
        })
        .collect();
    let get_position = |pin: &SharedPhysicalPin| -> (Expr, Expr) {
        if let Some(cell) = x_var.get(&pin.get_gid()) {
            let (x, y) = pin.relative_pos();
            (cell.x + x, cell.y + y)
        } else {
            let (x, y) = pin.pos();
            (Expr::from(x), Expr::from(y))
        }
    };

    let mut negative_delay_vars = Vec::new();
    let displacement_delay = mbffg.displacement_delay();
    let dpins = mbffg.get_effected_dpins(insts);
    let mut num_record = 0;
    for dpin in &dpins {
        let records = mbffg.get_prev_ff_records(dpin);
        num_record += records.len();
        if records.is_empty() {
            continue;
        }
        let slack = dpin.get_slack();
        let ori_delay = dpin.get_origin_delay();
        let ff_d = records.iter().next().unwrap();
        let mut fixed_record = Vec::new();
        let mut dynamic_records = Vec::new();
        for record in records.iter() {
            if record.ff_q.is_some() {
                dynamic_records.push(record);
            } else {
                fixed_record.push(record);
            }
        }
        if simplified_ratio < 1.0 && dynamic_records.len() > 100 {
            // debug!(
            //     "fixed_record: {}, dynamic_records: {}",
            //     fixed_record.len(),
            //     dynamic_records.len()
            // );
            // Extract the positions from the records into a vector of points
            let points = dynamic_records
                .iter()
                .filter_map(|record| record.ff_q.as_ref().map(|ff_q| ff_q.0.pos()))
                .collect_vec();

            // Determine how many random elements to pick (10% of the total)
            let num_to_pick = ((dynamic_records.len().float()) * simplified_ratio)
                .ceil()
                .usize();

            // Pick random elements from the records
            let mut rng = thread_rng();
            let picked: Set<_> = dynamic_records
                .choose_multiple(&mut rng, num_to_pick)
                .cloned()
                .collect();

            // Compute the convex hull indices based on the extracted points
            let hull_indices = convex_hull(&points);

            // Gather the records that correspond to the convex hull
            let hull_elements: Set<_> = dynamic_records
                .fancy_index_clone(&hull_indices)
                .into_iter()
                .collect();

            // Merge the picked random records and hull elements into the final selection
            dynamic_records = picked.union(&hull_elements).cloned().collect_vec();
        }
        let max_fixed_delay = fixed_record
            .iter()
            .max_by_key(|record| OrderedFloat(record.calculate_total_delay(displacement_delay)))
            .map(|record| {
                record.calculate_total_delay(displacement_delay)
                    - ff_d.ff_d_dist() * displacement_delay
            })
            .unwrap_or(0.0);
        let mut vars = Vec::new();
        for record in dynamic_records.iter() {
            let var = add_ctsvar!(model, bounds: ..).unwrap();
            let ff_q_pin = record.ff_q.as_ref().unwrap();
            let ff_q_expr = norm1(
                &mut model,
                get_position(&ff_q_pin.0),
                get_position(&ff_q_pin.1),
            )
            .unwrap();

            let unchanged_delay = record.calculate_total_delay(displacement_delay)
                - (record.ff_q_dist() + ff_d.ff_d_dist()) * displacement_delay;
            model
                .add_constr(
                    "",
                    c!(var >= unchanged_delay + ff_q_expr * displacement_delay),
                )
                .unwrap();
            vars.push(var);
        }
        let max_var = add_ctsvar!(model, bounds: ..)?;
        model.add_genconstr_max(
            &format!("max_vars_{}", dpin.get_gid()),
            max_var,
            vars,
            Some(max_fixed_delay),
        )?;
        let neg_slack_var = add_ctsvar!(model, bounds: ..)?;
        if let Some(ff_d) = ff_d.ff_d.as_ref() {
            let ff_d_expr = norm1(&mut model, get_position(&ff_d.0), get_position(&ff_d.1))?;
            model.add_constr(
                "",
                c!(neg_slack_var <= slack + ori_delay - (max_var + ff_d_expr * displacement_delay)),
            )?;
        } else {
            model.add_constr("", c!(neg_slack_var <= slack + ori_delay - max_var))?;
        }
        model.add_constr("", c!(neg_slack_var <= 0.0))?;
        negative_delay_vars.push(neg_slack_var);
    }
    debug!("Solve {} objs...", negative_delay_vars.len());
    let obj = negative_delay_vars.iter().grb_sum();
    model.set_objective(obj, Maximize)?;
    model.optimize()?;
    match model.status()? {
        Status::Optimal => {
            let optimized_pos: Dict<_, _> = x_var
                .iter()
                .map(|(id, cell)| {
                    let x = model.get_obj_attr(attr::X, &cell.x).unwrap();
                    let y = model.get_obj_attr(attr::X, &cell.y).unwrap();
                    (*id, (x, y))
                })
                .collect();
            // mbffg.check(true, false);
            for (gid, pos) in &optimized_pos {
                let inst = mbffg.get_node(*gid);
                let old_pos = inst.pos();
                if mbffg.debug_config.debug_timing_opt {
                    debug!(
                        "{}: move ({}, {}) to ({}, {})",
                        inst.get_name(),
                        old_pos.0,
                        old_pos.1,
                        pos.0,
                        pos.1
                    );
                }
                inst.move_to(pos.0, pos.1);
            }
            // mbffg.check(true, false);
            // let objective_value = model.get_attr(attr::ObjVal)?;
            // debug!("Objective value: {}", objective_value);
            // debug!("dpins count: {}", dpins.len());
            // debug!("Found {} previous flip-flop records", num_record,);
            return Ok(optimized_pos);
        }
        Status::InfOrUnbd => {
            println!("No feasible solution found.");
        }
        _ => {
            println!("Optimization was stopped with status {:?}", model.status()?);
        }
    }
    panic!("Optimization failed.");
}

pub fn solve_tiling_problem(
    cover_map: &Vec<Vec<CoverCell>>,
    tile_size: (uint, uint),
) -> grb::Result<()> {
    let (n, m) = (cover_map.len(), cover_map[0].len());
    let (tile_w, tile_h) = cast_tuple::<_, usize>(tile_size);

    let mut model = redirect_output_to_null(true, || {
        let env = Env::new("").unwrap();
        let model = Model::with_env("", env).unwrap();
        model
    })
    .unwrap();

    // Decision variables
    let x = vec![vec![add_binvar!(model).unwrap(); m]; n];

    // Coverage constraints
    for i in 0..n - tile_h {
        for j in 0..m - tile_w {
            let mut coverage = Vec::new();
            for r in i..i + tile_h {
                for c in j..j + tile_w {
                    coverage.push(x[r][c]);
                }
            }
            model.add_constr(
                &format!("coverage_{}_{}", i, j),
                c!(coverage.iter().grb_sum() <= 1),
            )?;
        }
    }

    // Objective: maximize total coverage
    model.set_objective(x.iter().map(|x| x.grb_sum()).grb_sum(), Maximize)?;

    // Solve the model
    model.optimize()?;

    if model.status()? == Status::Optimal {
        // get objective value
        let objective_value = model.get_attr(attr::ObjVal)?;
        println!("Optimal objective value: {}", objective_value);
        for i in 0..n {
            for j in 0..m {
                let var = x[i][j];
                let val = model.get_obj_attr(attr::X, &var)?;
                if val > 0.5 {}
            }
        }

        // if output {
        //     let mut total_coverage = 0.0;
        //     for k in 0..tile_sizes {
        //         total_coverage += (spatial_info_vec[k].capacity as usize
        //             * tile_infos[k].size.0
        //             * tile_infos[k].size.1) as f64;
        //     }
        //     println!("Optimal objective: {}", model.get_attr(attr::ObjVal)?);
        //     for k in 0..tile_sizes {
        //         println!(
        //             "Tile type {} ({}x{}): {}",
        //             k, tile_infos[k].size.1, tile_infos[k].size.0, spatial_info_vec[k].capacity
        //         );
        //     }
        //     println!("Total coverage: {}", total_coverage / (n * m) as f64);
        // }
        Ok(())
    } else {
        Ok(())
    }
}
