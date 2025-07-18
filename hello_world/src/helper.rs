use crate::*;
pub fn generate_coverage_map_from_size(
    gate_rtree: &Rtree,
    rows: &Vec<PlacementRows>,
    die_size: Vector2,
    size: Vector2,
) -> Vec<Vec<CoverCell>> {
    let (width, height) = size;
    let mut cover_map = Vec::new();
    let mut gate_rtree = gate_rtree.clone();
    let (die_width, die_height) = die_size;
    let (bottom, top) = rows.split_at(rows.len() / 2);
    let bottom_rev = bottom.into_iter().rev().collect_vec();

    for row in bottom_rev.into_iter().chain(top.iter()) {
        let row_bbox =
            geometry::Rect::from_size(row.x, row.y, row.width * row.num_cols.float(), height)
                .bbox_p();
        let row_intersection = gate_rtree.intersection_bbox(row_bbox);
        let mut row_rtee = Rtree::from(&row_intersection);
        let mut cover_cells = Vec::new();
        let middle = row.num_cols / 2;
        for j in (0..middle).rev().chain(middle..row.num_cols) {
            let (x, y) = (row.x + j.float() * row.width, row.y);
            let bbox = geometry::Rect::from_size(x, y, width, height).bbox();
            // Check if the bounding box is within the row bounding box
            if bbox[1][0] > die_width || bbox[1][1] > die_height {
                cover_cells.push(CoverCell {
                    x,
                    y,
                    is_covered: true,
                });
                break;
            } else {
                let is_covered = row_rtee.count_bbox(bbox) > 0;
                if !is_covered {
                    row_rtee.insert_bbox(bbox);
                    gate_rtree.insert_bbox(bbox);
                } else {
                }
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
    die_size: Vector2,
    lib_size: Vector2,
) -> Vec<Vector2> {
    let map = generate_coverage_map_from_size(gate_rtree, rows, die_size, lib_size);
    let available_placement_positions =
        apply_filter_map(&map, |x| if !x.is_covered { Some(x.pos()) } else { None })
            .into_iter()
            .flatten()
            .collect_vec();
    // run_python_script("plot_binary_image", (bmap, -1, "cover_map", false));
    available_placement_positions
}
