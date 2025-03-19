use crate::util::*;
use easy_print::*;
use grb::prelude::*;
use num_cast::*;
use once_cell::sync::Lazy;
use std::sync::Arc;
struct GRBLinExpr {
    expr: Expr,
}
impl GRBLinExpr {
    fn new() -> Self {
        Self {
            expr: Expr::from(0.0),
        }
    }
    fn from(vars: &Vec<Var>) -> Self {
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
    assert!(
        items.iter().all(|x| x.1.iter().all(|&y| y >= 0.0)),
        "All values must be non-negative."
    );
    assert!(items.len().i32() <= total_capacity, "Not enough knapsacks.");
    let gurobi_output: grb::Result<_> = crate::redirect_output_to_null(true, || {
        let num_items = items.len();
        let num_knapsacks = knapsack_capacities.len();
        // Create a new model
        let env = Env::new("")?;
        let mut model = Model::with_env("multiple_knapsack", env)?;
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
    crate::assert_eq!(result.iter().map(|x| x.len()).sum::<usize>(), items.len());
    result
}
use crate::mbffg::*;
pub fn optimize_timing(mbffg: &MBFFG) -> grb::Result<()> {
    use geo::{polygon, ConvexHull, LineString, Polygon};

    let env = Env::new("")?;
    let mut model = Model::with_env("multiple_knapsack", env)?;
    model.set_param(param::LogToConsole, 0)?;
    let num_ffs = mbffg.num_ff().usize();
    struct Cell {
        x: (Var, float),
        y: (Var, float),
        dist: Var,
        is_ff: bool,
    }
    impl Cell {
        fn x(self: &Self) -> Expr {
            self.x.0 + self.x.1
        }
        fn y(self: &Self) -> Expr {
            self.y.0 + self.y.1
        }
    }

    fn cityblock_variable(model: &mut Model, v1: &Cell, v2: &Cell) -> grb::Result<Expr> {
        let (abs_delta_x, abs_delta_y) = (add_ctsvar!(model)?, add_ctsvar!(model)?);
        let exp1 = v1.x() - v2.x();
        let exp2 = v1.y() - v2.y();
        model.add_constr("", c!(abs_delta_x >= exp1.clone()))?;
        model.add_constr("", c!(abs_delta_x >= -exp1))?;
        model.add_constr("", c!(abs_delta_y >= exp2.clone()))?;
        model.add_constr("", c!(abs_delta_y >= -exp2))?;

        Ok(abs_delta_x + abs_delta_y)
    }
    fn create_fixed_cell(model: &mut Model, pos: (float, float)) -> grb::Result<Cell> {
        let cell = Cell {
            x: (add_ctsvar!(model)?, pos.0),
            y: (add_ctsvar!(model)?, pos.1),
            dist: add_ctsvar!(model)?,
            is_ff: false,
        };
        model.add_constr("", c!(cell.x.0 == 0.0))?;
        model.add_constr("", c!(cell.y.0 == 0.0))?;
        Ok(cell)
    }

    let mut x = Dict::with_capacity(num_ffs);
    let mut dist_vars = Vec::new();
    for ff in mbffg.get_all_ffs() {
        x.insert(
            ff.borrow().gid,
            Cell {
                x: (add_ctsvar!(model)?, 0.0),
                y: (add_ctsvar!(model)?, 0.0),
                dist: add_ctsvar!(model)?,
                is_ff: true,
            },
        );
    }
    for ff in mbffg.get_all_ffs() {
        if mbffg.get_prev_ff_records(ff).len() > 10 {
            // mbffg
            //     .get_prev_ff_records(ff)
            //     .iter()
            //     .filter(|record| record.ff_q.is_some())
            //     .map(|record| record.ff_q.as_ref().unwrap().0.borrow().pos())
            //     .collect_vec()
            //     .prints();
            let poly = Polygon::new(
                LineString::from(
                    mbffg
                        .get_prev_ff_records(ff)
                        .iter()
                        .filter(|record| record.ff_q.is_some())
                        .map(|record| record.ff_q.as_ref().unwrap().0.borrow().pos())
                        .collect_vec(),
                ),
                vec![],
            );
            let hull = poly.convex_hull();
            let points = hull.exterior().0.iter().map(|p| (p.x, p.y)).collect_vec();
            points.len().prints();
        }
        for (dpin_prev, dpin) in mbffg.incomings(ff.borrow().gid) {
            let dpin_prev_gid = dpin_prev.borrow().gid();
            let dpin_gid = dpin.borrow().gid();
            if !x.contains_key(&dpin_prev_gid) {
                x.insert(
                    dpin_prev_gid,
                    create_fixed_cell(&mut model, dpin_prev.borrow().pos())?,
                );
            }
            if !x.contains_key(&dpin_gid) {
                x.insert(
                    dpin_gid,
                    create_fixed_cell(&mut model, dpin.borrow().pos())?,
                );
            }
            let ff_d_cityblock_distance =
                cityblock_variable(&mut model, &x[&dpin_prev_gid], &x[&dpin_gid])?;

            let dist_var = add_ctsvar!(model);
            let record = &dpin.borrow().farest_timing_record;
            let farest_ff = &dpin.borrow().origin_farest_ff_pin;
            let original_dist = dpin
                .borrow()
                .farest_timing_record
                .as_ref()
                .unwrap()
                .distance();
            // dpin.borrow().full_name().prints();
            // original_dist.prints();
            let constrint = if let Some(farest_ff) = farest_ff {
                let qpin_gid = farest_ff.0.borrow().gid();
                let qpin_next_gid = farest_ff.1.borrow().gid();
                if !x.contains_key(&qpin_gid) {
                    x.insert(
                        qpin_gid,
                        create_fixed_cell(&mut model, farest_ff.0.borrow().pos())?,
                    );
                }
                if !x.contains_key(&qpin_next_gid) {
                    x.insert(
                        qpin_next_gid,
                        create_fixed_cell(&mut model, farest_ff.1.borrow().pos())?,
                    );
                }

                let ff_q_cityblock_distance =
                    cityblock_variable(&mut model, &x[&qpin_gid], &x[&qpin_next_gid]).unwrap();
                (ff_d_cityblock_distance + ff_q_cityblock_distance - original_dist)
            } else {
                (ff_d_cityblock_distance - original_dist)
            };
            let dist_var = add_ctsvar!(model)?;
            model.add_constr("", c!(dist_var == constrint))?;
            dist_vars.push(dist_var);
        }
    }
    let mut obj = GRBLinExpr::new();
    for var in &dist_vars {
        obj += *var * 1.0;
    }
    model.set_objective(obj, Minimize)?;
    model.optimize()?;
    match model.status()? {
        Status::Optimal => {
            for (gid, cell) in x {
                if cell.is_ff {
                    let x: f64 = model.get_obj_attr(attr::X, &cell.x.0)?;
                    let y: f64 = model.get_obj_attr(attr::X, &cell.y.0)?;
                    mbffg.get_node(gid).borrow_mut().move_to(x, y);
                }
            }
            return Ok(());
        }
        Status::Infeasible => {
            println!("No feasible solution found.");
        }
        _ => {
            println!("Optimization was stopped with status {:?}", model.status()?);
        }
    }
    panic!("Optimization failed.");
    Ok(())
}
