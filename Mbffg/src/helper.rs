use crate::*;
pub fn generate_coverage_map_from_size_par(
    gate_rtree: &mut Rtree,
    rows: &[PlacementRows],
    die_size: Vector2,
    size: Vector2,
) -> Vec<Vector2> {
    let (width, height) = size;
    let mut cover_map = Vec::new();
    let (die_width, die_height) = die_size;
    let (bottom, top) = rows.split_at(rows.len() / 2);
    let bottom_rev = bottom.iter().rev();

    for row in bottom_rev.chain(top.iter()) {
        let row_bbox =
            geometry::Rect::from_size(row.x, row.y, row.width * row.num_cols.float(), height)
                .bbox();
        let row_intersection = gate_rtree.intersection_bbox(row_bbox);
        let row_rtee = Rtree::from(row_intersection);
        let middle = row.num_cols / 2;
        let step = (width / row.width).ceil().int();
        let middle_next = {
            let (x, y) = (row.x + middle.float() * row.width, row.y);
            if row_rtee.count_bbox(
                geometry::Rect::from_size(x, y, width, height)
                    .erosion(0.1)
                    .bbox(),
            ) == 0
            {
                middle + step
            } else {
                middle + 1
            }
        };
        let cover_cells = [(0, middle, true), (middle_next, row.num_cols - 1, false)]
            .into_par_iter()
            .map(|(start, end, rev)| {
                let mut cover_cells = Vec::new();
                let mut check = |cur: int| -> bool {
                    let (x, y) = (row.x + cur.float() * row.width, row.y);
                    let bbox = geometry::Rect::from_size(x, y, width, height)
                        .erosion(0.1)
                        .bbox();
                    // Check if the bounding box is within the row bounding box
                    if !(bbox[1][0] > die_width || bbox[1][1] > die_height) {
                        let is_covered = row_rtee.count_bbox(bbox) > 0;
                        if !is_covered {
                            cover_cells.push((x, y));
                            return true;
                        }
                    }
                    false
                };
                if rev {
                    let mut cur = end;
                    while cur >= start {
                        if check(cur) {
                            cur -= step;
                        } else {
                            cur -= 1;
                        }
                    }
                } else {
                    let mut cur = start;
                    while cur <= end {
                        if check(cur) {
                            cur += step;
                        } else {
                            cur += 1;
                        }
                    }
                }
                cover_cells
            })
            .flatten()
            .collect::<Vec<_>>();

        for cell in &cover_cells {
            let bbox = geometry::Rect::from_size(cell.0, cell.1, width, height)
                .erosion(0.1)
                .bbox();
            gate_rtree.insert_bbox(bbox);
        }
        cover_map.extend(cover_cells);
    }
    cover_map
}
pub fn evaluate_placement_resources_from_size(
    gate_rtree: &Rtree,
    rows: &[PlacementRows],
    die_size: Vector2,
    lib_size: Vector2,
) -> Vec<Vector2> {
    generate_coverage_map_from_size_par(&mut gate_rtree.clone(), rows, die_size, lib_size)
}
