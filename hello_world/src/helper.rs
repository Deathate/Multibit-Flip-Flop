use crate::*;
pub fn generate_coverage_map_from_size(
    gate_rtree: &Rtree,
    rows: &Vec<PlacementRows>,
    die_size: (float, float),
    size: (float, float),
) -> Vec<Vec<CoverCell>> {
    let (width, height) = size;
    let mut cover_map = Vec::new();
    let mut gate_rtree = gate_rtree.clone();
    let (die_width, die_height) = die_size;
    for row in rows.iter() {
        let row_bbox =
            geometry::Rect::from_size(row.x, row.y, row.width * row.num_cols.float(), height)
                .bbox_p();
        let row_intersection = gate_rtree.intersection_bbox(row_bbox);
        let mut row_rtee = Rtree::from(&row_intersection);
        let mut cover_cells = Vec::new();
        for j in 0..row.num_cols {
            let (x, y) = (row.x + j.float() * row.width, row.y);
            let bbox = geometry::Rect::from_size(x, y, width, height).bbox();
            // Check if the bounding box is within the row bounding box
            if bbox[1][0] > die_width || bbox[1][1] > die_height {
                cover_cells.push(CoverCell {
                    x,
                    y,
                    is_covered: true,
                });
            } else {
                let is_covered = row_rtee.count_bbox(bbox) > 0;
                if !is_covered {
                    row_rtee.insert_bbox(bbox);
                    gate_rtree.insert_bbox(bbox);
                }
                // Uncomment the following lines to check if the cover cell is covered by a gate
                // if !is_covered {
                //     // Check if the bounding box intersects with any gate
                //     let intersection = gate_rtree.intersection_bbox(bbox);
                //     if !intersection.is_empty() {
                //         row_intersection.prints();
                //         row_bbox.prints();
                //         panic!(
                //             "{}",
                //             self.error_message(format!(
                //                 "Cover cell {:?} is covered by gate, bbox: {:?}",
                //                 bbox, intersection
                //             ))
                //         );
                //     }
                // }
                cover_cells.push(CoverCell { x, y, is_covered });
            }
        }
        cover_map.push(cover_cells);
    }
    cover_map
}
pub fn evaluate_placement_resources_from_size(
    gate_rtree: &Rtree,
    rows: &Vec<PlacementRows>,
    die_size: (float, float),
    lib_size: (float, float),
) -> Vec<(f64, f64)> {
    let map = generate_coverage_map_from_size(gate_rtree, rows, die_size, lib_size);
    // run_python_script(
    //     "plot_binary_image",
    //     (
    //         map.iter()
    //             .map(|x| x.iter().map(|cell| cell.is_covered).collect_vec())
    //             .collect_vec(),
    //         -1,
    //         "cover_map",
    //         false,
    //     ),
    // );
    let available_placement_positions =
        apply_filter_map(&map, |x| if !x.is_covered { Some(x.pos()) } else { None })
            .into_iter()
            .flatten()
            .collect_vec();
    // if self.debug_config.visualize_placement_resources {
    //     let ffs = available_placement_positions
    //         .iter()
    //         .map(|&x| Pyo3Cell {
    //             name: "FF".to_string(),
    //             x: x.0,
    //             y: x.1,
    //             width: lib_width,
    //             height: lib_height,
    //             walked: false,
    //             highlighted: false,
    //             pins: vec![],
    //         })
    //         .collect_vec();

    //     Python::with_gil(|py| {
    //         let script = c_str!(include_str!("script.py")); // Include the script as a string
    //         let module = PyModule::from_code(py, script, c_str!("script.py"), c_str!("script"))?;

    //         let file_name = format!("tmp/potential_space_{}x{}.png", lib_width, lib_height);
    //         module.getattr("draw_layout")?.call1((
    //             false,
    //             &file_name,
    //             self.setting.die_size.clone(),
    //             f32::INFINITY,
    //             f32::INFINITY,
    //             self.placement_rows().clone(),
    //             ffs,
    //             self.get_all_gate().map(|x| Pyo3Cell::new(x)).collect_vec(),
    //             self.get_all_io().map(|x| Pyo3Cell::new(x)).collect_vec(),
    //             Vec::<PyExtraVisual>::new(),
    //         ))?;
    //         Ok::<(), PyErr>(())
    //     })
    //     .unwrap();
    //     // exit();
    // }

    // run_python_script("plot_binary_image", (bmap, -1, "cover_map", false));
    available_placement_positions
}
