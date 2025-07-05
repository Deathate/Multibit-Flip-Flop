use crate::class::*;
use crate::mbffg::*;
use crate::util::*;
use easy_print::*;
use grb::prelude::*;
use num_cast::*;
use once_cell::sync::Lazy;
use std::sync::Arc;
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
    // assert!(
    //     items.iter().all(|x| x.1.iter().all(|&y| y >= 0.0)),
    //     "All values must be non-negative."
    // );
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
pub fn optimize_timing(
    mbffg: &mut MBFFG,
    insts: &Vec<&SharedInst>,
    joint: bool,
) -> grb::Result<()> {
    if mbffg.debug_config.debug_timing_opt {
        debug!("Optimizing timing...");
    }
    mbffg.create_prev_ff_cache();
    let mut model = redirect_output_to_null(true, || {
        let env = Env::new("")?;
        let model = Model::with_env("", env)?;
        // model.set_param(param::LogToConsole, 0)?;
        Ok::<_, grb::Error>(model)
    })
    .unwrap()
    .unwrap();
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

    let get_position = |x_var: &Dict<InstId, Cell>, pin: &SharedPhysicalPin| -> (Expr, Expr) {
        let gid = if joint { 0 } else { pin.get_gid() };
        x_var.get(&gid).map_or_else(
            || {
                let (x, y) = pin.borrow().pos();
                (Expr::from(x), Expr::from(y))
            },
            |cell| {
                let (x, y) = pin.borrow().relative_pos();
                (cell.x + x, cell.y + y)
            },
        )
    };
    let x = if !joint {
        Dict::from_iter(insts.iter().map(|ff| {
            (
                ff.get_gid(),
                Cell {
                    x: add_ctsvar!(model).unwrap(),
                    y: add_ctsvar!(model).unwrap(),
                },
            )
        }))
    } else {
        Dict::from_iter([(
            0,
            Cell {
                x: add_ctsvar!(model).unwrap(),
                y: add_ctsvar!(model).unwrap(),
            },
        )])
    };

    let mut negative_delay_vars = Vec::new();
    let mut dpins = Vec::new();
    for ff in insts.iter() {
        for dpin in ff.dpins() {
            let mut downstream_ffs = mbffg.get_next_ff_dpins(&dpin).iter().cloned().collect_vec();
            dpins.append(&mut downstream_ffs);
            dpins.push(dpin);
        }
    }
    debug!("Processing {} downstream flip-flops", dpins.len(),);
    let dpins = dpins.iter().unique().collect_vec();
    debug!("{} unique downstream flip-flops", dpins.len());
    let displacement_delay = mbffg.displacement_delay();
    for (i, dpin) in dpins.iter().enumerate() {
        let records = mbffg.get_prev_ff_records(&dpin);
        if !records.is_empty() {
            let mut max_delay: float = 0.0;
            let mut max_fixed_delay: float = 0.0;
            let mut pin_delays = Vec::new();
            let mut ff_d = None;
            for record in records.iter() {
                if !record.has_ff_q() {
                    max_fixed_delay =
                        max_fixed_delay.max(record.calculate_total_delay(displacement_delay));
                    continue;
                }
                let (ff_q, _) = record.ff_q.as_ref().unwrap();
                if ff_d.is_none() {
                    ff_d = record.ff_d.as_ref().map(|ff_d| ff_d.clone());
                }
                if !x.contains_key(&ff_q.get_gid()) {
                    let delay = record.calculate_total_delay(displacement_delay)
                        - record.ff_d_dist() * displacement_delay;
                    max_delay = max_delay.max(delay);
                } else {
                    let delay = record.qpin_delay()
                        + displacement_delay
                            * (record.travel_dist
                                + record.ff_q.as_ref().map_or(Expr::from(0.0), |ff_q| {
                                    norm1(
                                        &mut model,
                                        get_position(&x, &ff_q.0),
                                        get_position(&x, &ff_q.1),
                                    )
                                    .unwrap()
                                }));
                    let var = add_ctsvar!(model, bounds: ..).unwrap();
                    model.add_constr("", c!(var == delay)).unwrap();
                    pin_delays.push(var);
                }
            }
            // debug!(
            //     "max_delay: {}, max_fixed_delay: {}",
            //     max_delay, max_fixed_delay
            // );
            let prev_delay_wo_dpin = add_ctsvar!(model, bounds: ..)?;
            model.add_genconstr_max("", prev_delay_wo_dpin, pin_delays, Some(max_delay))?;
            if ff_d.is_none() {
                continue;
            }
            let ff_d_pin = ff_d.unwrap();
            let ff_d_expr = norm1(
                &mut model,
                get_position(&x, &ff_d_pin.0),
                get_position(&x, &ff_d_pin.1),
            )
            .unwrap()
                * displacement_delay;
            // debug!("Adding {} delays for {}", pin_delays.len(), dpin.get_gid());
            let ff_d_var = add_ctsvar!(model, bounds: ..)?;
            model.add_constr(
                &format!("ff_d_{}", dpin.get_gid()),
                c!(ff_d_var == prev_delay_wo_dpin + ff_d_expr),
            )?;
            let delay_var = add_ctsvar!(model, bounds: ..)?;
            model.add_constr("", c!(delay_var >= ff_d_var))?;
            model.add_constr("", c!(delay_var >= max_fixed_delay))?;

            let negative_delay = add_ctsvar!(model, bounds: ..)?;
            model.add_constr(
                &format!("negative_delay_{}", dpin.get_gid()),
                c!(negative_delay
                    == dpin.get_slack() + dpin.get_origin_delay() - (delay_var + ff_d_var)),
            )?;
            let negative_slack = add_ctsvar!(model, bounds: ..0)?;
            model.add_constr(
                &format!("negative_slack_{}", dpin.get_gid()),
                c!(negative_slack <= negative_delay),
            )?;
            negative_delay_vars.push(negative_slack);
        }
    }

    debug!("Solve {} objs...", negative_delay_vars.len());
    let obj = negative_delay_vars.iter().grb_sum();
    model.set_objective(obj, Maximize)?;
    model.optimize()?;
    match model.status()? {
        Status::Optimal => {
            for (gid, cell) in x {
                let x: f64 = model.get_obj_attr(attr::X, &cell.x)?;
                let y: f64 = model.get_obj_attr(attr::X, &cell.y)?;
                // if cell.is_fixed {
                //     (cell.x(), cell.y()).prints();
                // }
                let pos = mbffg.get_node(gid).borrow_mut().pos();
                println!("{}: move ({}, {}) to ({}, {})", gid, pos.0, pos.1, x, y);
                mbffg.get_node(gid).borrow_mut().move_to(x, y);
            }
            return Ok(());
        }
        Status::InfOrUnbd => {
            // "------------------------------------".prints();
            println!("No feasible solution found.");
        }
        _ => {
            println!("Optimization was stopped with status {:?}", model.status()?);
        }
    }
    panic!("Optimization failed.");
    Ok(())
}
pub fn optimize_single_timing(
    mbffg: &mut MBFFG,
    insts: &Vec<SharedInst>,
) -> grb::Result<(float, float)> {
    let mut model = redirect_output_to_null(true, || {
        let env = Env::new("")?;
        let model = Model::with_env("", env)?;
        // model.set_param(param::LogToConsole, 0)?;
        Ok::<_, grb::Error>(model)
    })
    .unwrap()
    .unwrap();
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
    let mut dpins: Set<_> = insts.iter().flat_map(|inst| inst.dpins()).collect();
    for dpin in &dpins.clone() {
        dpins.extend(mbffg.get_next_ff_dpins(&dpin).clone());
    }
    debug!("Processing {} downstream flip-flops", dpins.len(),);
    for dpin in dpins {
        let records = mbffg.get_prev_ff_records(&dpin);
        let max_record = &mbffg.prev_ffs_query_cache[&dpin.get_id()].0;
        let max_delay = max_record.calculate_total_delay(displacement_delay);
        let ff_d_dist = max_record.ff_d_dist();
        let mut fixed_record = Vec::new();
        let mut dynamic_records = Vec::new();
        for record in records.iter() {
            if let Some((ff_q, _)) = &record.ff_q {
                if optimized_cell_ids.contains(&ff_q.get_gid()) {
                    dynamic_records.push(record);
                } else {
                    fixed_record.push(record);
                }
            } else {
                fixed_record.push(record);
            }
        }
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
                .add_constr("", c!(var >= max_var - max_delay))
                .unwrap();
        }
        model.add_constr("", c!(var >= 0.0)).unwrap();
        negative_delay_vars.push(var);
    }

    debug!("Solve {} objs...", negative_delay_vars.len());
    let obj = negative_delay_vars.iter().grb_sum();
    model.set_objective(obj, Minimize)?;
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
            let objective_value = model.get_attr(attr::ObjVal)?;
            debug!("Objective value: {}", objective_value);
            mbffg.negative_timing_slack_dp(&insts[0]).prints();
            insts[0].move_to(optimized_pos.0, optimized_pos.1);
            mbffg.negative_timing_slack_dp(&insts[0]).prints();
            exit();
            return Ok(optimized_pos);
        }
        Status::InfOrUnbd => {
            // "------------------------------------".prints();
            println!("No feasible solution found.");
        }
        _ => {
            println!("Optimization was stopped with status {:?}", model.status()?);
        }
    }
    panic!("Optimization failed.");
}
