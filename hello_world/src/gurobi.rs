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
use crate::mbffg::*;
pub fn optimize_timing(mbffg: &MBFFG) -> grb::Result<()> {
    use geo::{polygon, ConvexHull, LineString, Polygon};

    let env = Env::new("")?;
    let mut model = Model::with_env("multiple_knapsack", env)?;
    // model.set_param(param::LogToConsole, 0)?;
    let num_ffs = mbffg.num_ff().usize();
    struct Cell {
        x: (Var, float),
        y: (Var, float),
        is_ff: bool,
        is_fixed: bool,
    }
    impl Cell {
        fn x(self: &Self) -> Expr {
            self.x.0 + self.x.1
        }
        fn y(self: &Self) -> Expr {
            self.y.0 + self.y.1
        }
    }
    fn create_fixed_cell(model: &mut Model, pos: (float, float)) -> grb::Result<Cell> {
        let cell = Cell {
            x: (add_ctsvar!(model, bounds: 0..0)?, pos.0),
            y: (add_ctsvar!(model, bounds:  0..0)?, pos.1),
            is_ff: false,
            is_fixed: true,
        };
        Ok(cell)
    }

    fn cityblock_variable(model: &mut Model, v1: &Cell, v2: &Cell) -> grb::Result<Expr> {
        let (abs_delta_x, abs_delta_y) = (add_ctsvar!(model)?, add_ctsvar!(model)?);
        let var1 = add_ctsvar!(model, bounds: ..)?;
        let var2 = add_ctsvar!(model, bounds: ..)?;
        model.add_constr("", c!(var1 == v1.x() - v2.x()))?;
        model.add_constr("", c!(var2 == v1.y() - v2.y()))?;
        model.add_genconstr_abs("", abs_delta_x, var1)?;
        model.add_genconstr_abs("", abs_delta_y, var2)?;
        Ok(abs_delta_x + abs_delta_y)
    }

    let mut x = Dict::new();
    let mut dist_vars = Vec::new();
    let all_ffs = mbffg.get_all_ffs().collect_vec();
    let split = 20;
    let mut selected_ids = all_ffs[..split]
        .iter()
        .map(|x| x.borrow().gid)
        .collect::<Set<_>>();
    for ff in &all_ffs[..split] {
        x.insert(
            ff.borrow().gid,
            Cell {
                x: (add_ctsvar!(model)?, 0.0),
                y: (add_ctsvar!(model)?, 0.0),
                is_ff: true,
                is_fixed: false,
            },
        );
    }
    let mut other_ff = Vec::new();
    for ff in &all_ffs[split..] {
        let ff_id = ff.borrow().gid;
        let record = mbffg
            .get_prev_ff_records(ff)
            .iter()
            .filter(|x| {
                x.ff_q.is_some()
                    && selected_ids.contains(&x.ff_q.as_ref().unwrap().0.borrow().gid())
            })
            .count();
        if record > 0 {
            let cell = Cell {
                x: (add_ctsvar!(model, bounds:0..0)?, ff.borrow().x),
                y: (add_ctsvar!(model, bounds:0..0)?, ff.borrow().y),
                is_ff: true,
                is_fixed: true,
            };
            x.insert(ff_id, cell);
            other_ff.push(ff_id);
        }
    }
    selected_ids.extend(other_ff);

    let displacement_delay = mbffg.setting.displacement_delay;
    let mut relationship = 0;
    for ff in &all_ffs {
        if !x.contains_key(&ff.borrow().gid) {
            continue;
        }
        for (dpin_prev, dpin) in mbffg.incomings(ff.borrow().gid) {
            let record = mbffg
                .get_prev_ff_records(&dpin_prev.borrow().inst())
                .iter()
                .filter(|x| x.ff_q.is_some())
                .collect_vec();
            let dpin_prev_gid = dpin_prev.borrow().gid();
            let dpin_gid = dpin.borrow().gid();
            if !x.contains_key(&dpin_prev_gid) {
                x.insert(
                    dpin_prev_gid,
                    create_fixed_cell(&mut model, dpin_prev.borrow().pos())?,
                );
            }
            assert!(x.contains_key(&dpin_gid));

            let ff_d_cityblock_distance =
                cityblock_variable(&mut model, &x[&dpin_prev_gid], &x[&dpin_gid])?;

            let original_dist = dpin
                .borrow()
                .farest_timing_record
                .as_ref()
                .unwrap()
                .distance();

            // let indices = if record.len() > 5 {
            //     let record_points = record
            //         .iter()
            //         .enumerate()
            //         .map(|(i, record)| (i, record.ff_q.as_ref().unwrap().0.borrow().pos()))
            //         .collect_vec();
            //     let poly = Polygon::new(
            //         LineString::from(record_points.iter().map(|(_, coord)| *coord).collect_vec()),
            //         vec![],
            //     );
            //     let hull = poly.convex_hull();
            //     let hull_points = hull
            //         .exterior()
            //         .points()
            //         .map(|p| (p.x(), p.y()))
            //         .collect_vec();
            //     let hull_indices = hull_points
            //         .iter()
            //         .filter_map(|p| {
            //             record_points
            //                 .iter()
            //                 .find(|(_, original)| original == p)
            //                 .map(|(idx, _)| *idx)
            //         })
            //         .collect_vec();
            //     assert!(hull_indices.len() == hull_points.len());
            //     hull_indices
            // } else {
            //     (0..record.len()).collect_vec()
            // };
            // let filtered_record = indices.iter().map(|i| record[*i]).collect_vec();

            // original_dist.prints_with("original distance:");
            let filtered_record = record;
            let constraint = if !dpin_prev.borrow().is_ff() && !filtered_record.is_empty() {
                let mut vars = Vec::new();
                for r in filtered_record {
                    let ff_q = r.ff_q.as_ref().unwrap();
                    let qpin = ff_q.0.borrow();
                    let qpin_next = ff_q.1.borrow();
                    let qpin_gid = qpin.gid();
                    let qpin_next_gid = qpin_next.gid();
                    if !(x.contains_key(&qpin_gid) || x.contains_key(&qpin_next_gid)) {
                        continue;
                    }
                    if !x.contains_key(&qpin_gid) {
                        x.insert(qpin_gid, create_fixed_cell(&mut model, qpin.pos())?);
                    }
                    if !x.contains_key(&qpin_next_gid) {
                        x.insert(
                            qpin_next_gid,
                            create_fixed_cell(&mut model, qpin_next.pos())?,
                        );
                    }
                    if x[&qpin_next_gid].is_fixed && x[&qpin_gid].is_fixed {
                        continue;
                    }
                    let ff_q_cityblock_distance =
                        cityblock_variable(&mut model, &x[&qpin_gid], &x[&qpin_next_gid])?;
                    let var = add_ctsvar!(model)?;
                    model.add_constr("", c!(var == ff_q_cityblock_distance))?;
                    vars.push(var);
                    relationship += 1;
                }

                if vars.len() > 0 {
                    let ff_q_max_distance = add_ctsvar!(model, bounds:..)?;
                    model.add_genconstr_max("", ff_q_max_distance, vars, None)?;
                    (original_dist - (ff_d_cityblock_distance + ff_q_max_distance))
                } else {
                    (original_dist - ff_d_cityblock_distance)
                }
            } else {
                (original_dist - ff_d_cityblock_distance)
            };

            let dist_var = add_ctsvar!(model, bounds: ..)?;
            model.add_constr(
                "",
                c!(dist_var == dpin.borrow().slack() + displacement_delay * constraint),
            )?;
            let b = add_ctsvar!(model, bounds: 0..)?;
            model.add_constr("", c!(b >= -dist_var))?;
            dist_vars.push(b);
        }
    }
    // relationship.prints_with("relationship:");
    // input();

    let mut obj = GRBLinExpr::new();
    for var in &dist_vars {
        obj += *var * 1.0;
    }

    // dist_vars.len().prints();
    // exit();
    model.set_objective(obj, Minimize)?;
    model.optimize()?;
    // model.status()?.prints();
    // exit();
    match model.status()? {
        Status::Optimal => {
            x.iter()
                .filter(|(_, cell)| cell.is_fixed)
                .count()
                .prints_with("fixed cells:");
            for (gid, cell) in x {
                let x: f64 = model.get_obj_attr(attr::X, &cell.x.0)?;
                let y: f64 = model.get_obj_attr(attr::X, &cell.y.0)?;
                // if cell.is_fixed {
                //     (cell.x(), cell.y()).prints();
                // }
                if cell.is_ff && !cell.is_fixed {
                    let pos = mbffg.get_node(gid).borrow_mut().pos();
                    println!("{}: move ({}, {}) to ({}, {})", gid, pos.0, pos.1, x, y);
                    mbffg.get_node(gid).borrow_mut().move_to(x, y);
                }
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
