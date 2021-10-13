use std::ops::Range;
use std::cmp::Ordering;

pub fn bounding_rect(pts : &[(usize, usize)]) -> (usize, usize, usize, usize) {
    let (mut min_y, mut max_y) = (usize::MAX, 0);
    let (mut min_x, mut max_x) = (usize::MAX, 0);
    for pt in pts.iter() {
        if pt.0 < min_y {
            min_y = pt.0;
        }
        if pt.0 > max_y {
            max_y = pt.0
        }
        if pt.1 < min_x {
            min_x = pt.1
        }
        if pt.1 > max_x {
            max_x = pt.1;
        }
    }
    (min_y, min_x, max_y - min_y, max_x - min_x)
}

pub struct RasterSearch {

}

impl RasterSearch {

}

const DIST_TOL : f64 = 16.0;

pub const MAX_POINT_DIST : usize = 4;

#[derive(Debug, Clone)]
pub struct PointStats {
    pub centroid : (usize, usize),
    pub avg_dist : f64,
    pub var_dist : f64
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FoundType {
    Left,
    Right,
    Both
}

impl PointStats {

    pub fn calculate(pts : &[(usize, usize)]) -> Self {
        let n = pts.len() as f64;
        let mut pt_avg = (0.0, 0.0);
        for pt in pts.iter() {
            pt_avg.0 += pt.0 as f64;
            pt_avg.1 += pt.1 as f64;
        }
        pt_avg.0 /= n;
        pt_avg.1 /= n;

        let mut avg_dist = 0.0;
        let centroid = (pt_avg.0 as usize, pt_avg.1 as usize);
        for pt in pts.iter() {
            avg_dist += point_dist(pt, &centroid);
        }
        avg_dist /= n;

        let mut var_dist = 0.0;
        for pt in pts.iter() {
            var_dist += (point_dist(pt, &centroid) - avg_dist).powf(2.);
        }
        var_dist /= n;

        PointStats {
            centroid,
            avg_dist,
            var_dist
        }
    }

}

pub fn point_dist(a : &(usize, usize), b : &(usize, usize)) -> f64 {
     ((a.0 as f64 - b.0 as f64).powf(2.) + (a.1 as f64 - b.1 as f64 ).powf(2.)).sqrt()
}

/// Crude shape estimate, based on simple statistics from its points.
pub struct ShapeStats {
    pub centroid : (usize, usize),
    pub avg_dist : f64
}

fn calc_centroid(found : &[(usize, usize)]) -> (usize, usize) {
    let n = found.len();
    let centr_sum = found.iter().fold((0, 0), |avg, el| (avg.0 + el.0, avg.1 + el.1) );
    (centr_sum.0 / n, centr_sum.1 / n)
}

fn calc_avg_dist(found : &[(usize, usize)], centroid : (usize, usize)) -> f64 {
    let n_inv = 1. / (found.len() as f64);
    found.iter().fold(0.0, |dist, el| dist + point_dist(&centroid, el) * n_inv )
}

impl ShapeStats {

    // Estimate center and average distance on a set of found points
    // In a circle, all distances to the estimated center should be mostly the same. Test this invariance
    // for each inserted point. This should keep outlier points away.
    fn from_pairs(found : &[(usize, usize)]) -> Self {
        let centroid = calc_centroid(found);
        let avg_dist = calc_avg_dist(found, centroid);
        ShapeStats{ centroid, avg_dist }
    }

    // Calculate from a single left or right edge.
    fn from_single(found : &[(usize, usize)], found_type : &FoundType, est_radius : f64) -> Self {
        let chord_center = calc_centroid(found);
        let centroid = match found_type {
            FoundType::Left => (chord_center.0, chord_center.1.saturating_add(est_radius as usize)),
            FoundType::Right => (chord_center.0, chord_center.1.saturating_sub(est_radius as usize)),
            _ => panic!("Invalid found type for single stats calculation")
        };
        let avg_dist = calc_avg_dist(found, centroid);
        ShapeStats{ centroid, avg_dist }
    }
}

pub fn is_horizontally_close(a : &(usize, usize), b : &(usize, usize)) -> bool {
    (a.1 as i32 - b.1 as i32).abs() as usize <= MAX_POINT_DIST
}

pub fn is_vertically_close(a : &(usize, usize), b : &(usize, usize)) -> bool {
    (a.0 as i32 - b.0 as i32).abs() as usize <= MAX_POINT_DIST
}

pub fn is_equidistant(stats : &ShapeStats, found_sub : &[(usize, usize)]) -> bool {
    found_sub.iter()
        .all(|found| (point_dist(&stats.centroid, found) - stats.avg_dist).abs() <= DIST_TOL )
}

/// Given a sequence of coordinates, verify all points are sequentially close
/// in both vertical and horizontal direction.
pub fn check_single_closeness(found_sub : &[(usize, usize)]) -> bool {
    for i in 0..(found_sub.len()-1) {
        let below_close_y = is_vertically_close(&found_sub[i], &found_sub[i+1]);
        let below_close_x = is_horizontally_close(&found_sub[i], &found_sub[i+1]);
        if !(below_close_y && below_close_x) {
            return false;
        }
    }
    true
}

/// Given a pair of sequences of coordinates (that assumes left elements are even indices;
/// and right elements are odd indices), verify if all points within each sequence are sequentially
/// close.
fn check_pair_closeness(found_sub : &[(usize, usize)]) -> bool {
    for i in 0..(found_sub.len() - 3) {
        let all_close = is_vertically_close(&found_sub[i], &found_sub[i+2]) &&
            is_vertically_close(&found_sub[i+1], &found_sub[i+3]) &&
            is_horizontally_close(&found_sub[i], &found_sub[i+2]) &&
            is_horizontally_close(&found_sub[i+1], &found_sub[i+3]);
        if !all_close {
            return false;
        }
    }
    true
}

/// Returns the lenght of the subgroup that is close, starting from element at i
fn define_if_sequence_close<F>(
    found : &[(usize, usize)],
    i : usize,
    min_group_size : usize,
    max_group_size : usize,
    close_fn : F
) -> Result<usize, ()>
where
    F : Fn(&[(usize, usize)])->bool
{
    let mut group_size = max_group_size;
    let mut all_close = false;
    while !all_close && group_size >= min_group_size {
        if let Some(found_sub) = found.get(i..i+group_size) { //.unwrap_or(&found[i..]);
            // all_close = check_single_closeness(found_sub);
            all_close = close_fn(found_sub);
            if !all_close {
                group_size /= 2;
            } else {
                group_size = found_sub.len();
            }
        } else {
            group_size /= 2;
        }
    }
    if all_close {
        Ok(group_size)
    } else {
        Err(())
    }
}

/// Found contains indices for the image; Groups contains pairs of [start, end] indices of the
/// first image corresponding to points where groups were defined. Found is assumed to be ordered
/// by natural search order.
fn decide_kept_groups<F>(
    found : &[(usize, usize)],
    group_a_ixs : &std::ops::Range<usize>,
    group_b_ixs : &std::ops::Range<usize>,
    criteria : F
) -> Keep
where
    F : Fn(&[(usize, usize)], &[(usize, usize)])-> bool
{
    let group_a = &found[group_a_ixs.clone()];
    let group_b = &found[group_b_ixs.clone()];
    if criteria(group_a, group_b) {
        Keep::Both
    } else {
        match group_a.len().cmp(&group_b.len()) {
            Ordering::Less => Keep::Second,
            Ordering::Equal => Keep::Both,
            Ordering::Greater => Keep::First
        }
    }
}

/// Elements are assumed aligned by natural search order. Each element is assumed to be on its own row
fn are_single_groups_aligned(group_a : &[(usize, usize)], group_b : &[(usize, usize)]) -> bool {
    match (group_a.last(), group_b.first()) {
        (Some(a), Some(b)) => {
            is_horizontally_close(&a, &b)
        },
        _ => false
    }
}

/// Elements are assumed aligned by natural search order. Each pair of consecutive elements are
/// from a different row.
fn are_paired_groups_aligned(group_a : &[(usize, usize)], group_b : &[(usize, usize)]) -> bool {
    let n_a = group_a.len();
    let n_b = group_b.len();
    if n_a < 2 || n_b < 2 {
        return false;
    }
    let left_close = is_horizontally_close(&group_a[n_a-2], &group_b[0]);
    let right_close = is_horizontally_close(&group_a[n_a-1], &group_b[1]);
    left_close && right_close
}

enum Keep {
    First,
    Second,
    Both
}

/// Receives a previously-set group vector, a new group range and verify if this
/// new group should be appended by comparing it to the last appended group. The
/// criteria will be different depending on whether this is a paired or single
/// point slice.
fn update_groups_vector<F>(
    found : &[(usize, usize)],
    groups : &mut Vec<std::ops::Range<usize>>,
    new_group : std::ops::Range<usize>,
    criteria : F
) where
    F : Fn(&[(usize, usize)], &[(usize, usize)])->bool
{
    let group_n = groups.len();
    if groups.len() >= 2 {
        match decide_kept_groups(found, &groups[group_n - 2], &groups[group_n - 1], criteria) {
            Keep::First => { },
            Keep::Second => { groups.remove(group_n - 1); groups.push(new_group); },
            Keep::Both => { groups.push(new_group); }
        }
    } else {
        groups.push(new_group);
    }
}

#[derive(Clone, Copy, Debug)]
pub enum PointError {
    FoundFewPoints,
    FoundManyPoints,
    FilteredFewPoints
}

pub fn filter_raster_diffs(
    filtered : &mut Vec<(usize, usize)>,
    groups : &mut Vec<std::ops::Range<usize>>,
    found : &[(usize, usize)],
    found_type : &FoundType,
    frame : usize,
    min_points : usize,
    max_points : usize
) -> Result<usize, PointError> {

    filtered.clear();

    if found.len() < min_points / 2 {
        return Err(PointError::FoundFewPoints);
    }

    match found_type {
        FoundType::Both => filter_pairs(filtered, groups, found),
        FoundType::Left | FoundType::Right => filter_single(filtered, groups, found)
    }

    // filter_by_hemisphere_closeness(filtered, found);

    let n = filtered.len();
    let has_filtered_min = (*found_type == FoundType::Both && n >= min_points) ||
        (*found_type != FoundType::Both && n >= min_points / 2);
    if has_filtered_min {
        if n <= max_points {
            Ok(n)
        } else {
            Err(PointError::FoundManyPoints)
        }
    } else {
        Err(PointError::FilteredFewPoints)
    }
}

fn filter_single(
    filtered : &mut Vec<(usize, usize)>,
    groups : &mut Vec<std::ops::Range<usize>>,
    found : &[(usize, usize)],
) {
    filtered.clear();
    groups.clear();
    let mut i = 0;
    let n = found.len();

    while i <= n-2 {
        // if let Ok(group_sz) = define_if_sequence_close(found, i, 8, 16, check_single_closeness) {
        if let Ok(group_sz) = define_if_sequence_close(found, i, 4, 36, check_single_closeness) {
            update_groups_vector(found, groups, i..i+group_sz, are_single_groups_aligned);
            i += group_sz;
        } else {
            i += 1;
        }
    }

    // We do not need to apply the is_equidistant condition here because when we reflect the points
    // they will be equidistant by construction.

    for group in groups.drain(0..) {
        if is_single_group_enclosing_rect_vertical(found, &group) {
            filtered.extend(found[group].iter().cloned());
        }
    }
}

/// Gets the bottom row and current raster row as long as the left and right samples
/// in the raster line are sufficiently close, irrespective of their absolute position.
fn filter_pairs(
    filtered : &mut Vec<(usize, usize)>,
    groups : &mut Vec<std::ops::Range<usize>>,
    found : &[(usize, usize)]
) {
    filtered.clear();
    groups.clear();
    let mut i = 0;
    let stats = ShapeStats::from_pairs(found);
    let n = found.len();

    while i <= n-4 {
        if let Ok(group_sz) = define_if_sequence_close(found, i, 8, 16, check_pair_closeness) {
            let new_group = i..i+group_sz;
            if is_equidistant(&stats, &found[new_group.clone()]) {
                update_groups_vector(found, groups, new_group, are_paired_groups_aligned);
                i += group_sz;
            } else {
                i += 2;
            }
        } else {
            i += 2;
        }
    }

    // enclosing rect vertical cannot apply here (only to separate left and right groups)
    for group in groups.drain(0..) {
        /*let g_left : Vec<_> = found[group.clone()].iter().step_by(2).cloned().collect();
        let g_right : Vec<_> = found[group.clone()].iter().skip(1).cloned().step_by(2).collect();

        let left_vert = is_single_group_enclosing_rect_vertical(&g_left[..], &(0..g_left.len()));
        let right_vert = is_single_group_enclosing_rect_vertical(&g_right[..], &(0..g_right.len()));

        if left_vert && right_vert {
            filtered.extend(found[group].iter().cloned());
        }*/
        if is_paired_group_enclosing_rect_vertical(found, &group) {
            filtered.extend(found[group].iter().cloned());
        }
    }

    /*while i <= n-4 {
        match define_if_sequence_close(found, i, 4, 16, check_pair_closeness) {
            Ok(group_sz) => {
                let equidistant = is_equidistant(&stats, &found[i..i+group_sz]);
                if equidistant {
                    for j in 0..group_sz {
                        filtered.push(found[i+j]);
                    }
                    i += group_sz;
                } else {
                    i += 2;
                }
            },
            _ => {
                i += 2;
            }
        }
    }*/
}

pub fn is_paired_group_enclosing_rect_vertical(found : &[(usize, usize)], group : &Range<usize>) -> bool {
    // Height is same for left and right groups
    let height = found[group.end.saturating_sub(1)].0 as i32 - found[group.start].0 as i32;

    // Width is separate for each group.
    let min_x_left = found[group.clone()].iter().step_by(2).min_by(|a, b| a.1.cmp(&b.1) ).unwrap();
    let max_x_left = found[group.clone()].iter().step_by(2).max_by(|a, b| a.1.cmp(&b.1) ).unwrap();

    let min_x_right = found[group.clone()].iter().skip(1).step_by(2).min_by(|a, b| a.1.cmp(&b.1) ).unwrap();
    let max_x_right = found[group.clone()].iter().skip(1).step_by(2).max_by(|a, b| a.1.cmp(&b.1) ).unwrap();

    let width_left = max_x_left.1 as i32 - min_x_left.1 as i32;
    let width_right = max_x_right.1 as i32 - min_x_right.1 as i32;

    let left_rect_ratio = height as f64 / width_left as f64;
    let right_rect_ratio = height as f64 / width_right as f64;

    // actually > 1.0 for strictly vertical; But we let some elements be slightly horizontal.
    left_rect_ratio > 1.0 && right_rect_ratio > 1.0
}

pub fn is_single_group_enclosing_rect_vertical(found : &[(usize, usize)],group : &Range<usize>) -> bool {
    let height = found[group.end.saturating_sub(1)].0 as i32 - found[group.start].0 as i32;
    let min_x = found[group.clone()].iter().min_by(|a, b| a.1.cmp(&b.1) ).unwrap();
    let max_x = found[group.clone()].iter().max_by(|a, b| a.1.cmp(&b.1) ).unwrap();
    let width = max_x.1 as i32 - min_x.1 as i32;
    let rect_ratio = height as f64 / width as f64;

    // actually > 1.0 for strictly vertical; But we let some elements be slightly horizontal.
    rect_ratio > 1.0
}

// TODO calculate outer (enclosing) circle for a set of points as:
// (1) Get largest point distance. This is the diameter.
// (2) Set center as halfway of this distance.
