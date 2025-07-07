use crate::*;
use tch::{
    index::IndexOp,
    nn,
    nn::{Module, OptimizerConfig},
    vision::{imagenet, resnet::resnet18},
    Device, Kind, TchError, Tensor,
};
pub fn test() -> Result<(), tch::TchError> {
    load_cuda_lib();
    println!("CUDA available: {}", tch::Cuda::is_available());
    println!("CUDA device count: {}", tch::Cuda::device_count());

    // Initialize the device (CPU or CUDA if available)
    let device = Device::Cuda(0);

    // Create a VarStore to manage variables and optimizer state
    let vs = nn::VarStore::new(device);

    // Initialize variables A and B with requires_grad = true
    // Use nn::var to register variables to the VarStore for optimizer tracking
    let a = vs.root().var("A", &[1], nn::Init::Const(2.0));
    let b = vs.root().var("B", &[1], nn::Init::Const(-3.0));

    // Create the SGD optimizer (can also use Adam or others)
    // let mut opt = nn::Sgd::default().build(&vs, 1e-3)?;
    let mut opt = nn::Sgd::default().build(&vs, 1e-2).unwrap();

    // Training loop
    for step in 0..1000 {
        // Compute the loss: sum of absolute values of A and B
        let loss = a.abs() + b.abs();

        // Backward and step optimizer
        opt.zero_grad();
        loss.backward();
        opt.step();

        // Print current state

        println!(
            "Step {}: A={:.4}, B={:.4}, A+B={:.4}, Loss={:.4}",
            step,
            a.double_value(&[0]),
            b.double_value(&[0]),
            (a.abs() + b.abs()).double_value(&[0]),
            loss.double_value(&[0])
        );
    }

    // Final values
    println!(
        "Final: A={}, B={}",
        a.double_value(&[0]),
        b.double_value(&[0])
    );

    Ok(())
}
pub fn test2() {
    // Load the CUDA library
    load_cuda_lib();
    // Set up a manual seed for reproducibility
    tch::manual_seed(42);
    let vs = nn::VarStore::new(Device::Cuda(0));
    // Create three trainable variables, initialized randomly
    // let vars = vs
    //     .root()
    //     .var(
    //         "vars",
    //         &[3],
    //         nn::Init::Randn {
    //             mean: (10.0),
    //             stdev: (5.0),
    //         },
    //     )
    //     .to_kind(Kind::Float);
    let vars = (0..3)
        .map(|i| {
            vs.root().var(
                &format!("i{}", i),
                &[1],
                nn::Init::Randn {
                    mean: (10.0),
                    stdev: (5.0),
                },
            )
        })
        .collect_vec();
    // Optimizer (Adam)
    let mut opt = nn::Adam::default().build(&vs, 1e-1).unwrap();

    let t = 10.0; // Temperature for softmax-max

    for step in 0..1000 {
        // Compute softmax-max (log-sum-exp) approximation
        let loss = ((&vars[0] + &vars[1] + &vars[2]) * t).logsumexp(0, true) / t;
        let loss = loss.relu();
        // Backward and optimize
        opt.zero_grad();
        loss.backward();
        // // create mask for indices where vars <= 0
        // let mask = vars.le(0.0);
        // // Freeze gradients for indices where vars <= 0
        // // Print progress
        // freeze_indices_grad_mask(&vars, &mask);
        if step % 10 == 0 || step == 99 {
            let v = vars.values();
            // let v = (0..vars.size1().unwrap())
            //     .map(|x| vars.float())
            //     .collect_vec();
            let actual_max = vars.max_value();
            println!(
                "Step {:>3}: vars = {:?}, softmax_max = {:.4}, actual_max = {:.4}",
                step,
                v,
                loss.float(),
                actual_max.float(),
            );
            // Vec::<bool>::try_from(&mask).unwrap().prints();
            // vars.grad().print();
        }
        opt.step();
        // opt.backward_step(&loss);
    }

    // Print final result
    // let v: Vec<f64> = vars.into();
    // println!("Final variables: {:?}", v);
}
fn norm1(ff_q_pin: &(SharedPhysicalPin, SharedPhysicalPin), x_var: &Dict<usize, Tensor>) -> Tensor {
    let p1 = get_position(&ff_q_pin.0, x_var);
    let p2 = get_position(&ff_q_pin.1, x_var);
    let v = (p1 - p2).abs();
    v.i(0) + v.i(1)
}

fn get_position(pin: &SharedPhysicalPin, x_var: &Dict<usize, Tensor>) -> Tensor {
    let result = if let Some(cell) = x_var.get(&pin.get_gid()) {
        let (x_off, y_off) = pin.relative_pos();
        cell + Tensor::from_slice(&[x_off, y_off])
    } else {
        let (x, y) = pin.pos();
        Tensor::from_slice(&[x, y])
    };
    result
}
// pub fn optimize_multiple_timing(mbffg: &mut MBFFG, insts: &Vec<SharedInst>) {
//     let device = Device::Cpu;
//     let vs = nn::VarStore::new(device);
//     let root = vs.root();
//     let x_var: Dict<_, _> = insts
//         .iter()
//         .map(|x| {
//             (
//                 x.get_gid(),
//                 Cell {
//                     x: root.var("x", &[], nn::Init::Const(x.get_x())),
//                     y: root.var("y", &[], nn::Init::Const(x.get_y())),
//                 },
//             )
//         })
//         .collect();
//     let mut negative_delay_vars = Vec::new();
//     let displacement_delay = mbffg.displacement_delay();
//     let dpins = mbffg.get_effected_dpins(&insts.iter().collect_vec());
//     let mut num_record = 0;
//     debug!("Number of dpins: {}", dpins.len());
//     for dpin in dpins.iter().tqdm() {
//         let records = mbffg.get_prev_ff_records(dpin);
//         if records.is_empty() {
//             continue;
//         }
//         let ori_delay = dpin.get_origin_delay();
//         let mut ff_d_dist = -1.0;
//         let mut fixed_record = Vec::new();
//         let mut dynamic_records = Vec::new();
//         let mut first_record = None;
//         for record in records.iter() {
//             if first_record.is_none() {
//                 first_record = Some(record.clone());
//                 ff_d_dist = record.ff_d_dist();
//             }
//             if record.ff_q.is_some() {
//                 dynamic_records.push(record);
//             } else {
//                 fixed_record.push(record);
//             }
//         }
//         num_record += dynamic_records.len();
//         let mut max_fixed_delay = 0.0;
//         if !fixed_record.is_empty() {
//             let max_fixed_record = fixed_record
//                 .iter()
//                 .max_by_key(|record| OrderedFloat(record.calculate_total_delay(displacement_delay)))
//                 .unwrap();
//             max_fixed_delay = max_fixed_record.calculate_total_delay(displacement_delay)
//                 - ff_d_dist * displacement_delay;
//         }
//         let mut vars = Vec::new();
//         for record in dynamic_records.iter() {
//             let ff_q_pin = record.ff_q.as_ref().unwrap();
//             let ff_q_expr = norm1(
//                 get_position(&ff_q_pin.0, &x_var),
//                 get_position(&ff_q_pin.1, &x_var),
//             );

//             let unchanged_delay = record.calculate_total_delay(displacement_delay)
//                 - (record.ff_q_dist() + ff_d_dist) * displacement_delay;
//             let delay_without_ffd = unchanged_delay + ff_q_expr * displacement_delay;
//             vars.push(delay_without_ffd);
//         }
//         let max_var = vars.max().max_other(&Tensor::from(max_fixed_delay));
//         let final_delay = if let Some(ff_d) = first_record.unwrap().ff_d.as_ref() {
//             let ff_d_expr = norm1(get_position(&ff_d.0, &x_var), get_position(&ff_d.1, &x_var));
//             max_var + ff_d_expr * displacement_delay
//         } else {
//             max_var
//         };
//         // final_delay.prints_with("Final Delay:");
//         // ori_delay.prints_with("Original Delay:");
//         let slack = dpin.get_slack() + ori_delay - final_delay;
//         // slack.prints_with("Slack:");
//         let neg_slack = (-slack).relu();
//         // neg_slack.prints_with("Negative Slack:");
//         negative_delay_vars.push(neg_slack);
//     }
//     let mut opt = nn::Adam::default().build(&vs, 1e-1).unwrap();
//     for step in (0..1000).tqdm() {
//         let loss = negative_delay_vars.sum();
//         // Backward and step optimizer
//         opt.zero_grad();
//         loss.backward();
//         opt.step();

//         if step % 10 == 0 || step == 999 {
//             println!("Step {}: Loss = {:.4}", step, loss.double_value(&[]));
//         }
//     }
// }
fn build_negative_delay_vars(
    mbffg: &MBFFG,
    x_var: &Dict<InstId, Tensor>,
    displacement_delay: f64,
    dpins: &Set<SharedPhysicalPin>,
) -> Vec<Tensor> {
    let mut negative_delay_vars = Vec::new();
    for dpin in dpins.iter() {
        let records = mbffg.get_prev_ff_records(dpin);
        if records.is_empty() {
            continue;
        }
        let ori_delay = dpin.get_origin_delay();
        let mut ff_d_dist = -1.0;
        let mut fixed_record = Vec::new();
        let mut dynamic_records = Vec::new();
        let mut first_record = None;
        for record in records.iter() {
            if first_record.is_none() {
                first_record = Some(record.clone());
                ff_d_dist = record.ff_d_dist();
            }
            if record.ff_q.is_some() {
                dynamic_records.push(record);
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
        // let tmr = stimer!("Processing dynamic records");
        for record in dynamic_records.iter() {
            let ff_q_pin = record.ff_q.as_ref().unwrap();
            let ff_q_expr = norm1(ff_q_pin, x_var);
            let unchanged_delay = record.calculate_total_delay(displacement_delay)
                - (record.ff_q_dist() + ff_d_dist) * displacement_delay;
            let delay_without_ffd = unchanged_delay + ff_q_expr * displacement_delay;
            vars.push(delay_without_ffd);
        }
        // finish!(tmr, "Processed {} dynamic records", dynamic_records.len());
        let max_var = if vars.is_empty() {
            Tensor::from(max_fixed_delay)
        } else {
            vars.max().max_other(&Tensor::from(max_fixed_delay))
        };
        let final_delay = if let Some(ff_d) = first_record.unwrap().ff_d.as_ref() {
            let ff_d_expr = norm1(ff_d, x_var);
            max_var + ff_d_expr * displacement_delay
        } else {
            max_var
        };
        let slack = dpin.get_slack() + ori_delay - final_delay;
        let neg_slack = (-slack).relu();
        negative_delay_vars.push(neg_slack);
    }

    negative_delay_vars
}

// --- Main function ---
pub fn optimize_multiple_timing(mbffg: &mut MBFFG, insts: &Vec<SharedInst>) {
    // Set up a manual seed for reproducibility
    tch::manual_seed(42);
    // let device = Device::Cuda(0);
    let device = Device::Cpu;
    let vs = nn::VarStore::new(device);
    let root = vs.root();
    let x_var: Dict<_, _> = insts
        .iter()
        .map(|x| {
            let t = Tensor::from_slice(&[x.get_x(), x.get_y()]);
            (x.get_gid(), root.var_copy("", &t))
        })
        .collect();
    let displacement_delay = mbffg.displacement_delay();
    let dpins = mbffg.get_effected_dpins(&insts.iter().collect_vec());
    let mut opt = nn::Adam::default().build(&vs, 1e-1).unwrap();
    for step in (0..1000).tqdm() {
        // --- Main autograd computation, cleanly separated ---
        let tmr = stimer!("");
        let negative_delay_vars =
            build_negative_delay_vars(mbffg, &x_var, displacement_delay, &dpins);
        finish!(tmr, "Step {}: Computation finished", step);
        let loss = negative_delay_vars.sum();

        // opt.zero_grad();
        // loss.backward();
        // opt.step();
        opt.backward_step(&loss);
        // for (gid, cell) in x_var.iter() {
        //     let x = cell.x.float();
        //     let y = cell.y.float();
        //     mbffg.get_node(*gid).move_to(x, y);
        //     // println!("Step {}: Inst {}: x = {:.4}, y = {:.4}", step, gid, x, y);
        // }
        if step % 1 == 0 || step == 999 {
            println!("Step {}: Loss = {:.4}", step, loss.double_value(&[]));
        }
    }
}
