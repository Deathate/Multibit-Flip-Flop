use crate::util::shape;
use easy_print::*;
use grb::prelude::*;
use num_cast::*;
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
