// use either::Either;
use std::cmp::{Eq, PartialEq};
use itertools::Itertools;
use crate::image::{Image, Window, WindowMut};
use crate::feature::shape::{self, *};
use std::fmt;
use std::collections::HashMap;
use std::mem;
use std::default::Default;
use parry2d::shape::ConvexPolygon;
use nalgebra::geometry::Point2;
use std::cmp::Ordering;
use parry2d::query::PointQuery;
use std::borrow::Borrow;
use crate::feature::point;
use crate::feature::edge::euclidian;
use crate::image::index::index_distance;
use super::*;
use away::space::SpatialClustering;

// #[cfg(feature="opencvlib")]
// pub mod fgmm;

// #[cfg(feature="opencvlib")]
// pub mod mser;

/// The most general patch is a set of pixel positions with a homogeneous color
/// and a scale that was used for extraction. The patch is assumed to be
/// homonegeneous within a pixel spacing given by the scale field. TODO if the
/// patch is dense (has no holes in it), we can represent it with another
/// structure called contour, holding only external boundray pixels. If the patch
/// has holes, we must represent all pixels with this patch structure.
#[derive(Clone, Debug, Default)]
pub struct Patch {
    // Instead of storing all pixels, we can store a rect and only
    // the outer pixels. The rect is just characterized by a top-left
    // corner and an extension. The rect extension is increased
    // any time we have a set of inserted pixels that comptetely border
    // either its bottom or right borders.
    pub pxs : Vec<(usize, usize)>,

    // Outer rect, at patch scale
    pub outer_rect : (usize, usize, usize, usize),
    pub color : u8,
    pub scale : usize,
    pub img_height : usize,

    // Number of pixels inside it.
    area : usize
}

/*fn patch_from_grouped_rows(
    sorted_rows : &[usize],
    gr : HashMap<usize, Vec<usize>>,
    color : u8,
    scale : usize,
    img_height : usize
) -> Patch {
    let mut pxs = Vec::new();
    for r in sorted_rows.iter() {
        for c in gr[&r].iter() {
            pxs.push((*r, *c));
        }
    }
    let mut patch = Patch::new(pxs[0], color, scale, img_height);
    for px in pxs.iter().skip(1) {
        patch.expand(&[*px]);
    }
    patch
}*/

#[derive(Debug, Clone, Copy)]
struct Split {
    split_pos : usize,
    split_len : usize,
    fst_blob_width : usize,
    snd_blob_width : usize,
    fst_blob_height : usize,
    snd_blob_height : usize
}

// TODO take a took at spade=1.8.2 or acacia for RTrees.

#[test]
fn split_vertically() {
    let mut patch = Patch::new((0, 0), 255, 1, 16);

    for r in (0..8) {
        for c in (0..16) {
            if (r, c) != (0, 0) {
                // patch.expand(&[(r, c)]);
            }
        }
    }

    for r in (8..16) {
        for c in (4..8) {
            // patch.expand(&[(r, c)]);
        }
    }

    // println!("{:?}", patch.try_split_vertically(4, (4, 4)));
}

/*pub fn color_limit(win : &Window<'_, u8>, row : usize, center : usize, px_spacing : usize, max_diff : u8) -> (usize, usize) {
    win.horizontal_pixel_pairs(row, comp_dist)
        .take_while(|(c, px_a, px_b)|
            if c < center {

            } else {
            if c > center {

            }
        )
}*/

/// Approximate color momentum, which might work for nearly symmetrical objects and is faster.
pub fn approx_color_momentum(win : &Window<'_, u8>, px_spacing : usize, mode : ColorMode) -> Option<(usize, usize)> {
    let (mut sum_r, mut sum_c) = (0.0, 0.0);
    let mut n_matches = 0;
    for (r, c, color) in win.labeled_pixels(px_spacing) {
        if mode.matches(color) {
            sum_r += r as f32; //* weight;
            sum_c += c as f32; //* weight;
            n_matches += 1;
        }
    }
    if n_matches >= 1 {
        Some(((sum_r / n_matches as f32) as usize * px_spacing, (sum_c / n_matches as f32) as usize * px_spacing))
    } else {
        None
    }
}

/// Returns the pixel that centralizes the given color. If this color is distributed
/// in a homogeneous region, the center can be used as a seed for the patch.
/// If pixels matches mode not exactly, pixels are weighted in the inverse proportion of the target color.
pub fn color_momentum(win : &Window<'_, u8>, px_spacing : usize, mode : ColorMode) -> Option<(usize, usize)> {
    let mut n_matches = 0;
    let mut rows = HashMap::<usize, (usize, usize)>::new();
    let mut cols = HashMap::<usize, (usize, usize)>::new();
    for (r, c, color) in win.labeled_pixels(px_spacing) {
        if mode.matches(color) {
            if let Some(mut r) = rows.get_mut(&r) {
                r.0 += c;
                r.1 += 1;
            } else {
                rows.insert(r, (c as usize, 1));
            }
            if let Some(mut c) = cols.get_mut(&c) {
                c.0 += r;
                c.1 += 1;
            } else {
                cols.insert(c, (r as usize, 1));
            }
            n_matches += 1;
        }
    }

    if n_matches >= 1 {
        let avg_rows : usize = (cols.iter().map(|(_, rs)| rs.0 / rs.1 ).sum::<usize>() as f32 / cols.iter().count() as f32) as usize;
        let avg_cols : usize = (rows.iter().map(|(_, cs)| cs.0 / cs.1 ).sum::<usize>() as f32 / rows.iter().count() as f32) as usize;
        Some((avg_rows * px_spacing, avg_cols * px_spacing))
    } else {
        None
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum Strategy {
    Eager,
    Conservative(usize)
}

/// Calculates the color momentum, but if the momentum pixel is not at a valid color,
/// position the momentum to the closest pixel that also matches it (eager strategy)
/// or to the point average of the n-closest pixels that matches it (conservative strategy).
pub fn adjusted_color_momentum(
    win : &Window<'_, u8>,
    px_spacing : usize,
    mode : ColorMode,
    strategy : Strategy
) -> Option<(usize, usize)> {
    let mom = color_momentum(win, px_spacing, mode)?;

    let mut found = Vec::new();

    if mode.matches(win[mom]) && strategy == Strategy::Eager {
        Some(mom)
    } else {
        for (px_pos, px) in win.expanding_pixels(mom, px_spacing) {
            if mode.matches(*px) {
                match strategy {
                    Strategy::Eager => {
                        return Some(px_pos);
                    },
                    Strategy::Conservative(n_required) => {
                        found.push(px_pos);
                        if found.len() == n_required {
                            let stats = point::PointStats::calculate(&found[..]);
                            return Some(stats.centroid);
                        }
                    }
                }
            }
        }

        match strategy {
            // TODO decide logic if required number of points was not found.
            Strategy::Conservative(_) => {
                let stats = point::PointStats::calculate(&found[..]);
                Some(stats.centroid)
            },
            _ => None
        }

    }
}

#[test]
fn momentum() {
    let mut img = Image::new_constant(24, 24, 0);
    for r in 8..16 {
        for c in 8..16 {
            img[(r, c)] = 1;
        }
    }
    println!("{:?}", color_momentum(&img.full_window(), 1, ColorMode::Exact(1)));
}

#[test]
fn patch_growth() {
    let mut img = Image::new_constant(24, 24, 0);
    for r in 8..16 {
        for c in 8..16 {
            img[(r, c)] = 1;
        }
    }

    /*img[(6,5)] = 1;
    img[(6,6)] = 1;
    img[(7,5)] = 1;
    img[(7,6)] = 1;
    img[(7,7)] = 1;
    img[(7,8)] = 1;*/

    println!("{:?}", Patch::grow(&img.full_window(), (11, 11), 1, ColorMode::Exact(1), ReferenceMode::Constant, None, ExpansionMode::Contour));
}

/// Extract a single patch, using the color momentum as seed.
pub fn extract_main_patch(win : &Window<'_, u8>, px_spacing : usize, mode : ColorMode) -> Option<Patch> {
    let seed = color_momentum(win, px_spacing, mode)?;
    Some(Patch::grow(win, seed, 1, mode, ReferenceMode::Constant, None, ExpansionMode::Dense)).unwrap()
}

#[derive(Default, Clone)]
struct RowPair {
    curr : Vec<(usize, usize)>,
    past : Vec<(usize, usize)>
}

impl RowPair {

    fn new(px : (usize, usize)) -> Self {
        let (curr, mut past) = (Vec::with_capacity(64), Vec::with_capacity(64));
        past.push(px);
        Self { curr, past }
    }

}

/// Extracts patches from image based on dense, homogeneous regions.
/// Searches the 3d space or (row, col, color) for regions that are
/// very clustered together, therefore mostly ignoring regions that are
/// close but have non-homogeneous color, depending on the min_dist
/// and min_cluster_sz parameters chosen.
pub fn patches_from_dense_regions(
    win : &Window<'_, u8>,
    scale : usize,
    min_dist : f64,
    min_cluster_sz : usize,
    mode : ColorMode
) -> Vec<Patch> {
    let pxs : Vec<[f64; 3]> = win.labeled_pixels(scale)
        .filter(|(_, _, px)| mode.matches(*px) )
        .map(|(r, c, color)| [r as f64, c as f64, color as f64] )
        .collect::<Vec<_>>();
    let clust = SpatialClustering::cluster_linear(&pxs, min_dist, min_cluster_sz);
    let mut patches = Vec::new();
    for (_, clust) in clust.clusters.iter() {
        let mut color = (clust.iter().map(|[_, _, c]| c ).sum::<f64>() / clust.len() as f64) as u8;
        let outer_rect = (0, 0, 0, 0);
        let pxs : Vec<_> = clust.iter().map(|[r, c, _]| (*r as usize, *c as usize) ).collect();
        let mut patch = Patch {
            outer_rect,
            color,
            scale,
            img_height : win.height(),
            area : clust.len(),
            pxs
        };
        let mut row_pxs = patch.group_rows();
        let min_row = row_pxs.keys().min().unwrap();
        let max_row = row_pxs.keys().max().unwrap();

        let mut col_pxs = patch.group_cols();
        let min_col = col_pxs.keys().min().unwrap();
        let max_col = col_pxs.keys().max().unwrap();

        patch.outer_rect = (*min_row, *min_col, *max_row - *min_row, *max_col - *min_col);
        patches.push(patch);
    }

    patches
}

#[derive(Default)]
struct ExpansionFront {
    top : RowPair,
    left : RowPair,
    bottom : RowPair,
    right : RowPair
}

impl ExpansionFront {

    fn new(px : (usize, usize)) -> Self {
        let pair = RowPair::new(px);
        Self {
            top : pair.clone(),
            left : pair.clone(),
            bottom : pair.clone(),
            right : pair
        }
    }

}

fn pixel_horizontally_aligned_to_rect(outer_rect : &(usize, usize, usize, usize), (r, c) : (usize, usize)) -> bool {
    r >= outer_rect.0 /*.saturating_sub(1)*/ &&
        r <= (outer_rect.0 + outer_rect.2) /*.saturating_add(1)*/
}

fn pixel_vertically_aligned_to_rect(outer_rect : &(usize, usize, usize, usize), (r, c) : (usize, usize)) -> bool {
    c >= outer_rect.1 /*.saturating_sub(1)*/  &&
        c <= (outer_rect.1 + outer_rect.3) /*.saturating_add(1)*/
}

fn pixel_to_right_of_rect(outer_rect : &(usize, usize, usize, usize), (r, c) : (usize, usize)) -> bool {
    pixel_horizontally_aligned_to_rect(outer_rect, (r, c))
}

fn pixel_to_left_of_rect(outer_rect : &(usize, usize, usize, usize), (r, c) : (usize, usize)) -> bool {
    pixel_horizontally_aligned_to_rect(outer_rect, (r, c))
}

fn pixel_above_rect(outer_rect : &(usize, usize, usize, usize), (r, c) : (usize, usize)) -> bool {
    pixel_vertically_aligned_to_rect(outer_rect, (r, c))
}

fn pixel_below_rect(outer_rect : &(usize, usize, usize, usize), (r, c) : (usize, usize)) -> bool {
    pixel_horizontally_aligned_to_rect(outer_rect, (r, c))
}

fn pixel_neighbors_row(row : &[(usize, usize)], px : (usize, usize)) -> bool {
    row.binary_search_by(|row_px|
        if (row_px.1 as i16 - px.1 as i16).abs() <= 1 {
            Ordering::Equal
        } else {
            row_px.1.cmp(&px.1)
        }
    ).is_ok()
}

fn pixel_neighbors_col(col : &[(usize, usize)], px : (usize, usize)) -> bool {
    col.binary_search_by(|col_px|
        // if (col_px.0 as i16 - px.0 as i16).abs() <= 1 {
        //    Ordering::Equal
        //} else {
        col_px.0.cmp(&px.0)
        // }
    ).is_ok()
}

fn pixel_neighbors_last_at_row(row : &[(usize, usize)], px : (usize, usize)) -> bool {
    row.last().map(|last| (last.1 as i16 - px.1 as i16).abs() <= 1 ).unwrap_or(false)
}

fn pixel_neighbors_last_at_col(col : &[(usize, usize)], px : (usize, usize)) -> bool {
    col.last().map(|last| (last.0 as i16 - px.0 as i16).abs() <= 1 ).unwrap_or(false)
}

fn pixel_neighbors_top(exp_patch : &ExpansionFront, px : (usize, usize)) -> bool {
    pixel_neighbors_last_at_row(&exp_patch.top.curr[..], px) ||
        pixel_neighbors_row(&exp_patch.top.past[..], px)
}

fn pixel_neighbors_left(exp_patch : &ExpansionFront, px : (usize, usize)) -> bool {
    pixel_neighbors_last_at_col(&exp_patch.left.curr[..], px) ||
        pixel_neighbors_col(&exp_patch.left.past[..], px)
}

fn pixel_neighbors_bottom(exp_patch : &ExpansionFront, px : (usize, usize)) -> bool {
    pixel_neighbors_last_at_row(&exp_patch.bottom.curr[..], px) ||
        pixel_neighbors_row(&exp_patch.bottom.past[..], px)
}

fn pixel_neighbors_right(exp_patch : &ExpansionFront, px : (usize, usize)) -> bool {
    pixel_neighbors_last_at_col(&exp_patch.right.curr[..], px) ||
        pixel_neighbors_col(&exp_patch.right.past[..], px)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpansionMode {
    Rect,
    Contour,
    Dense
}

// Dense strategy here.
// TODO to build patch with only the external contour, verify which elements
// past_row and curr_row have in common. Drop all elements of past_row that also
// are present in current row, and push current_row and all remaining elements of
// past_row that were not dropped. Perhaps we can make this function generic over which
// allocation strategy to self.pxs we use here.
fn expand_patch(
    exp_patch : &mut ExpansionFront,
    patch : &mut Patch,
    outer_rect : (usize, usize, usize, usize),
    mode : ExpansionMode,
    win : &Window<'_, u8>
) {

    // Only push corner when those conditions are not met for the desired pair.
    let tl_joined = match (exp_patch.left.curr.first(), exp_patch.top.curr.first()) {
        (Some(left), Some(top)) => (left.0 as i32 - top.0 as i32).abs() <= 1 && (left.1 as i32 - top.1 as i32).abs() <= 1,
        _ => false
    };
    let tr_joined = match (exp_patch.top.curr.last(), exp_patch.right.curr.first()) {
        (Some(top), Some(right)) => (top.0 as i32 - right.0 as i32).abs() as i32 <= 1 && (top.1 as i32 - right.1 as i32).abs() <= 1,
        _ => false
    };
    let bl_joined = match (exp_patch.left.curr.last(), exp_patch.bottom.curr.first()) {
        (Some(left), Some(bottom)) => (left.0 as i32 - bottom.0 as i32).abs() <= 1 && (left.1 as i32 - bottom.1 as i32).abs() as i32 <= 1,
        _ => false
    };
    let br_joined = match (exp_patch.right.curr.last(), exp_patch.bottom.curr.last()) {
        (Some(right), Some(bottom)) => (right.0 as i32 - bottom.0 as i32).abs() <= 1 && (right.1 as i32 - bottom.1 as i32).abs() <= 1,
        _ => false
    };

    // Also iterate over this: if corners_joined[ix] push corner.
    let mut exp_fronts = [&mut exp_patch.top, &mut exp_patch.left, &mut exp_patch.bottom, &mut exp_patch.right];
    let corners_joined = [(tl_joined, tr_joined), (tl_joined, bl_joined), (bl_joined, br_joined), (tr_joined, br_joined)];

    for (mut ext, (fst_corner, snd_corner)) in exp_fronts.iter_mut().zip(corners_joined.iter()) {
        let is_seed = ext.past.len() == 1 && ext.past.get(0).cloned() == Some((patch.pxs[0]));
        if is_seed {
            ext.past.clear();
            mem::swap(&mut ext.curr, &mut ext.past);
            continue;
        }
        match mode {
            ExpansionMode::Rect => {
                ext.past.clear();
                patch.area = outer_rect.2 * outer_rect.3;
                mem::swap(&mut ext.curr, &mut ext.past);
            },
            ExpansionMode::Dense => {
                patch.area += ext.past.len();
                patch.pxs.extend(ext.past.drain(..));
                mem::swap(&mut ext.curr, &mut ext.past);
            },
            ExpansionMode::Contour => {
                if ext.curr.len() > 0 {
                    /*match ext.past.len() {
                        0 => { },
                        1 => {
                            patch.pxs.push(ext.past[0]);
                        },
                        _ => {
                            // Just push past border extreme pixels for non-final iterations.
                            patch.pxs.push(ext.past[0]);
                            patch.pxs.push(ext.past[ext.past.len()-1]);
                        }
                    }*/

                    // TODO Remove old corner pixels
                    // ext.past.remove(4 - ix);

                    // Only push corners when the distance between corners of edge limits
                    // are not sufficient to represent the shape in its current state.
                    if !*fst_corner {
                        patch.pxs.push(ext.past[0]);
                    }

                    let n = ext.past.len();
                    if n > 1 && !*snd_corner {
                        patch.pxs.push(ext.past[n-1]);
                    }

                    // Expand area
                    if n == 1 {
                        patch.area += 1;
                    } else {
                        if n > 1 {
                            let is_row_front = ext.past[n-1].0 == ext.past[0].0;
                            if is_row_front {
                                patch.area += ext.past[n-1].1 - ext.past[0].1;
                            } else {
                                patch.area += ext.past[n-1].0 - ext.past[0].0;
                            }
                        }
                    }

                    // Stop swapping at the last empty border so eventually we reach the final
                    // branch with the last past pixels that were found.
                    ext.past.clear();
                } else {
                    // The last patch iteration will take all remaining contour values.
                    patch.area += ext.past.len();
                    patch.pxs.extend(ext.past.drain(..));
                }
                mem::swap(&mut ext.curr, &mut ext.past);
            }
        }
    }
    // }

}

/*pub fn expand_contour_patch(exp_patch : &mut ExpansionFront, patch : &mut Patch, outer_rect : (usize, usize, usize, usize)) {
    for ext in [&mut exp_patch.top, &mut exp_patch.left, &mut exp_patch.bottom, &mut exp_patch.right] {
        if patch.pxs.len() > 1 || ext.past.get(0).cloned() != Some((patch.pxs[0])) {
            patch.pxs.extend(ext.past.drain(..));
        }
        patch.outer_rect = outer_rect;
        mem::swap(&mut ext.curr, &mut ext.past);
    }
}*/

fn close_contour(patch : &mut Patch, win : &Window<'_, u8>) {

    // Order by pairwise closeness.
    for ix in 1..patch.pxs.len() {
        let (closest_ix, _) = patch.pxs.iter().enumerate().skip(ix)
            .min_by(|(_, px1), (_, px2)| {
                index_distance(**px1, patch.pxs[ix-1], win.height()).0
                    .partial_cmp(&index_distance(**px2, patch.pxs[ix-1], win.height()).0)
                    .unwrap_or(Ordering::Equal)
            } ).unwrap();
        patch.pxs.swap(closest_ix, ix);

        /*let curr = patch.pxs[i-1];
        patch.pxs[i+1..].sort_by(|(_, px1), (_, px2)| {
        index_distance(**px1, curr, win.height()).0
            .partial_cmp(&index_distance(**px2, curr, win.height()).0)
            .unwrap_or(Ordering::Equal)
        });*/
    }

    // Last element will not be close to neighborhood, just remove it.
    patch.pxs.pop();
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceMode {
    Constant,
    Adaptive
}

fn update_stats(mode : &mut ColorMode, n_px : &mut usize, sum : &mut u64, sum_abs_dev : &mut u64, ref_mode : ReferenceMode, new : u8) {
    if ref_mode == ReferenceMode::Adaptive {
        *sum += new as u64;
        *n_px += 1;
        let mean = *sum / *n_px as u64;
        // let abs_dev = (mean as i64 - new as i64).abs() as u64;
        // *sum_abs_dev += abs_dev;
        mode.set_reference_color(mean as u8);
    }
}

fn expand_rect(rect : &mut (usize, usize, usize, usize), exp : &ExpansionFront) {
    let mut max_right = rect.1 + rect.3;
    let mut max_bottom = rect.0 + rect.2;
    if exp.top.curr.len() >= 1 {
        if exp.top.curr[0].0 < rect.0 {
            rect.0 = exp.top.curr[0].0;
        }
        if exp.top.curr[0].1 < rect.1 {
            rect.1 = exp.top.curr[0].1;
        }
        if exp.top.curr[exp.top.curr.len()-1].1 > max_right {
            max_right = exp.top.curr[exp.top.curr.len()-1].1;
        }
    }

    if exp.left.curr.len() >= 1 {
        if exp.left.curr[0].1 < rect.1 {
            rect.1 = exp.left.curr[0].1;
        }
        if exp.left.curr[0].0 < rect.0 {
            rect.0 = exp.left.curr[0].0;
        }
        if exp.left.curr[exp.left.curr.len()-1].0 > max_bottom {
            max_bottom = exp.left.curr[exp.left.curr.len()-1].0;
        }
    }

    if exp.bottom.curr.len() >= 1 {
        if exp.bottom.curr[0].0 > max_bottom {
            max_bottom = exp.bottom.curr[0].0;
        }
        if exp.bottom.curr[0].1 < rect.1 {
            rect.1 = exp.bottom.curr[0].1;
        }
        if exp.bottom.curr[exp.bottom.curr.len()-1].1 > max_right {
            max_right = exp.bottom.curr[exp.bottom.curr.len()-1].1;
        }
    }

    if exp.right.curr.len() >= 1 {
        if exp.right.curr[0].1 > max_right {
            max_right = exp.right.curr[0].1;
        }
        if exp.right.curr[0].0 < rect.0 {
            rect.0 = exp.right.curr[0].0;
        }
        if exp.right.curr[exp.right.curr.len()-1].0 > max_bottom {
            max_bottom = exp.right.curr[exp.right.curr.len()-1].0;
        }
    }

    rect.2 = max_bottom - rect.0;
    rect.3 = max_right - rect.1;
}

impl Patch {

    /*pub fn average_color(&self, win : &Window<'_, u8>, exp : ExpansionMode) -> u8 {
        match exp {
            ExpansionMode::Contour => {

            },
            _ => unimplemented!()
        }
    }*/

    pub fn center(&self) -> (usize, usize) {
        let rect = self.outer_rect();
        (rect.0 + rect.2 / 2, rect.1 + rect.3 / 2)
    }

    /// Number of pixels contained in the patch.
    pub fn area(&self) -> usize {
        self.area
    }

    /// Grows a patch from a pixel seed.
    /// cf. Connected Components (Szeliski, 2011, p. 131)
    pub fn grow(
        win : &Window<'_, u8>,
        seed : (usize, usize),
        px_spacing : usize,
        mut mode : ColorMode,
        ref_mode : ReferenceMode,
        max_area : Option<usize>,
        exp_mode : ExpansionMode
    ) -> Option<Self> {

        // if !mode.matches(win[seed]) {
        //    println!("Seed does not match desired color");
        //    return None;
        // }
        // mode.set_reference_color(win[seed]);

        let mut patch = Patch::new(seed, win[seed], 1, win.height());


        let (mut grows_left, mut grows_top, mut grows_right, mut grows_bottom) = (true, true, true, true);

        // Still need to alter pixel_neighbors_row and pixel_below_rect, etc to account for differing px_spacings.
        // Will check the absolute difference is <= px_spacing there.
        assert!(px_spacing == 1);

        let mut abs_dist = px_spacing;
        let mut exp_patch = ExpansionFront::new(seed);
        let mut outer_rect = (seed.0, seed.1, 1, 1);

        let mut n_px = 0;
        let mut sum_abs_dev : u64 = 0;
        let mut sum : u64 = win[seed] as u64;

        loop {
            assert!(exp_patch.top.curr.is_empty() && exp_patch.left.curr.is_empty() && exp_patch.right.curr.is_empty() && exp_patch.bottom.curr.is_empty());
            let left_col = if grows_left { seed.1.checked_sub(abs_dist) } else { None };
            let top_row = if grows_top { seed.0.checked_sub(abs_dist) } else { None };
            let right_col = if grows_right {
                if seed.1 + abs_dist < win.width() { Some(seed.1 + abs_dist) } else { None }
            } else {
                None
            };
            let bottom_row = if grows_bottom {
                if seed.0 + abs_dist < win.height() { Some(seed.0 + abs_dist) } else { None }
            } else {
                None
            };

            let row_end = (seed.0+abs_dist+1).min(win.height());
            let col_end = (seed.1+abs_dist+1).min(win.width());
            let row_range = (seed.0.saturating_sub(abs_dist)..( row_end )).step_by(px_spacing);
            let col_range = (seed.1.saturating_sub(abs_dist)..( col_end )).step_by(px_spacing);

            // TODO when pixel is not in rect range, we can break the loop.

            let mut grows_top = false;
            if exp_patch.top.past.len() >= 1 {
                let top_range = (exp_patch.top.past[0].1.saturating_sub(1))..((exp_patch.top.past[exp_patch.top.past.len()-1].1 + 2).min(win.width()));
                if let Some(r) = top_row {
                    for c in /*col_range.clone()*/ top_range.clone() {
                        let px = (r, c);
                        let is_corner = (c == top_range.start || c == top_range.end-1);
                        if mode.matches(win[px]) && (is_corner || ( /*pixel_above_rect(&outer_rect, px) &&*/ pixel_neighbors_top(&exp_patch, px))) {
                            if !grows_top {
                                // outer_rect.0 = r;
                                // outer_rect.2 += px_spacing;
                                grows_top = true;
                            }
                            exp_patch.top.curr.push(px);
                            update_stats(&mut mode, &mut n_px, &mut sum, &mut sum_abs_dev, ref_mode, win[px]);
                        }
                    }
                }
            }

            let mut grows_left = false;
            if exp_patch.left.past.len() >= 1 {
                let left_range = (exp_patch.left.past[0].0.saturating_sub(1))..((exp_patch.left.past[exp_patch.left.past.len()-1].0 + 2).min(win.height()));
                if let Some(c) = left_col {
                    for r in /*row_range.clone().skip(1).take(row_end-1)*/ left_range.clone() {
                        let px = (r, c);
                        let is_corner = (r == left_range.start || r == left_range.end-1);
                        if mode.matches(win[px]) && ( is_corner || ( /*pixel_to_left_of_rect(&outer_rect, px) &&*/ pixel_neighbors_left(&exp_patch, px))) {
                            if !grows_left {
                                // outer_rect.1 = c;
                                // outer_rect.3 += px_spacing;
                                grows_left = true;
                            }
                            exp_patch.left.curr.push(px);
                            update_stats(&mut mode, &mut n_px, &mut sum, &mut sum_abs_dev, ref_mode, win[px]);
                        }
                    }
                }
            }

            let mut grows_bottom = false;
            if exp_patch.bottom.past.len() >= 1 {
                let bottom_range = (exp_patch.bottom.past[0].1.saturating_sub(1))..((exp_patch.bottom.past[exp_patch.bottom.past.len()-1].1 + 2).min(win.width()));
                if let Some(r) = bottom_row {
                    for c in /*col_range*/ bottom_range.clone() {
                        let px = (r, c);
                        let is_corner = (r == bottom_range.start || r == bottom_range.end-1);
                        if mode.matches(win[px]) && (is_corner || ( /*pixel_below_rect(&outer_rect, px) &&*/ pixel_neighbors_bottom(&exp_patch, px))) {
                            if !grows_bottom {
                                // outer_rect.2 += px_spacing;
                                grows_bottom = true;
                            }
                            exp_patch.bottom.curr.push(px);
                            update_stats(&mut mode, &mut n_px, &mut sum, &mut sum_abs_dev, ref_mode, win[px]);
                        }
                    }
                }
            }

            let mut grows_right = false;
            if exp_patch.right.past.len() >= 1 {
                let right_range = (exp_patch.right.past[0].0.saturating_sub(1))..((exp_patch.right.past[exp_patch.right.past.len()-1].0 + 2).min(win.height()));
                if let Some(c) = right_col {
                    for r in /*row_range.skip(1).take(row_end-1)*/ right_range.clone() {
                        let px = (r, c);
                        let is_corner = (c == right_range.start || c == right_range.end-1);
                        if mode.matches(win[px]) && (is_corner || ( /*pixel_to_right_of_rect(&outer_rect, px) &&*/ pixel_neighbors_right(&exp_patch, px))) {
                            if !grows_right {
                                // outer_rect.3 += px_spacing;
                                grows_right = true;
                            }
                            exp_patch.right.curr.push(px);
                            update_stats(&mut mode, &mut n_px, &mut sum, &mut sum_abs_dev, ref_mode, win[px]);
                        }
                    }
                }
            }

            let grows_any = grows_left || grows_right || grows_bottom || grows_top;
            expand_rect(&mut outer_rect, &exp_patch);
            patch.outer_rect = outer_rect;
            expand_patch(&mut exp_patch, &mut patch, outer_rect, exp_mode, win);

            if let Some(area) = max_area {
                if /*patch.pxs.len()*/ outer_rect.2 * outer_rect.3 > area {
                    return None;
                }
            }

            if !grows_any {
                if exp_mode == ExpansionMode::Contour {

                    // Remove seed before closing patch
                    patch.pxs.swap_remove(0);

                    close_contour(&mut patch, win);
                }
                return Some(patch);
            }

            abs_dist += px_spacing;
        }

        Some(patch)
    }

    /// Starts a new patch.
    pub fn new(pt : (usize, usize), color : u8, scale : usize, img_height : usize ) -> Self {
        /*let pxs = if let Some(mut pxs) = pxs {
            pxs.clear();
            pxs.push(pt);
            pxs
        } else {*/
        let mut pxs = Vec::with_capacity(16);
        pxs.push(pt);
        // pxs
        // };
        Self {
            pxs,
            outer_rect : (pt.0, pt.1, 1, 1),
            color,
            scale,
            img_height,
            area : 1
        }
    }

    /// Outer rect, at image scale
    pub fn outer_rect(&self) -> (usize, usize, usize, usize) {
        (
            self.outer_rect.0 * self.scale,
            self.outer_rect.1 * self.scale,
            self.outer_rect.2 * self.scale,
            self.outer_rect.3 * self.scale
        )
    }

    /*/// Take the row with smallest width (within the set of rows of length smaller than split_max_width)
    /// and returns the patches above and below this row (excluding the row) as long as the
    /// resulting patches have width and height of at least output_min_rect.
    pub fn try_split_vertically(
        &self,
        split_max_width : usize,
        output_min_rect : (usize, usize)
    ) -> Option<(Patch, Patch)> {

        // Assume rows are ordered
        let mut gr = self.group_rows();
        let mut gc = self.group_cols();

        let mut best_split_row : Option<Split> = None;

        let mut sorted_rows : Vec<_> = gr.iter().map(|(k, _)| *k ).collect();
        let mut sorted_cols : Vec<_> = gc.iter().map(|(k, _)| *k ).collect();
        sorted_rows.sort();
        sorted_cols.sort();

        let indexed_row_iter = sorted_rows.iter()
            .enumerate()
            .skip(output_min_rect.0.saturating_sub(1))
            .take_while(|(ix, r)| sorted_rows.len() - ix >= output_min_rect.0 );

        for (split_row_ix, split_row) in indexed_row_iter {

            // It this row is sufficiently narrow, consider it for a split.
            if gr[split_row].len() <= split_max_width {

                // println!("Can split at {} with {} columns", split_row, gr[split_row].len());

                let prev_rows = &sorted_rows[0..split_row_ix];
                let post_rows = &sorted_rows[split_row_ix+1..];

                let largest_prev_row = prev_rows.iter()
                    .max_by(|row1, row2| gr[row1].len().cmp(&gr[row2].len()) )
                    .unwrap();
                let largest_post_row = post_rows.iter()
                    .max_by(|row1, row2| gr[row1].len().cmp(&gr[row2].len()) )
                    .unwrap();
                let prev_tall_enough = gr[&largest_prev_row].len() >= output_min_rect.0;
                let post_tall_enough = gr[&largest_post_row].len() >= output_min_rect.0;

                // This check is not strictly necessary because we already limit the indexed_row_iter
                // iterator to heights greater than required by the user. But we leave it here to
                // generalize the procedure to search by a column iterator later.

                // println!("prev_tall_enough = {}; post_tall_enough = {}", prev_tall_enough, post_tall_enough);
                if prev_tall_enough && post_tall_enough {

                    // Take all columns that compose the largest row in the patch
                    let min_prev_col_ix = sorted_cols.binary_search(gr[&largest_prev_row].first().unwrap()).unwrap();
                    let max_prev_col_ix = sorted_cols.binary_search(gr[&largest_prev_row].last().unwrap()).unwrap();
                    let min_post_col_ix = sorted_cols.binary_search(gr[&largest_post_row].first().unwrap()).unwrap();
                    let max_post_col_ix = sorted_cols.binary_search(gr[&largest_post_row].last().unwrap()).unwrap();
                    let prev_cols = &sorted_cols[min_prev_col_ix..max_prev_col_ix];
                    let post_cols = &sorted_cols[min_post_col_ix..max_post_col_ix];
                    // println!("prev cols = {:?}, post cols = {:?}", prev_cols, post_cols);

                    let largest_prev_col = prev_cols.iter()
                        .map(|col| (col, gc[col].iter().filter(|row| *row < split_row ).count()) )
                        .max_by(|(_, nrows1), (_, nrows2)| nrows1.cmp(&nrows2) ).unwrap();
                    let largest_post_col = post_cols.iter()
                        .map(|col| (col, gc[col].iter().filter(|row| *row > split_row ).count()) )
                        .max_by(|(_, nrows1), (_, nrows2)| nrows1.cmp(&nrows2) ).unwrap();
                    let prev_large_enough = largest_prev_col.1 >= output_min_rect.1;
                    let post_large_enough = largest_post_col.1 >= output_min_rect.1;

                    // println!("largest_prev_col = {:?}, largest_post_col = {:?}, prev_large_enough = {}; post_large_enough = {}", largest_prev_col, largest_post_col, prev_large_enough, post_large_enough);

                    if prev_large_enough && post_large_enough {

                        if let Some(split) = best_split_row {

                            // The shortness of the split row is an interesting criterion for hourglass-like blobs.
                            let shorter_split = gr[split_row].len() < split.split_len;

                            // For a vertical split, fst_blob_height + snd_blob_height = const. We leave it
                            // here to generalize later.
                            let taller_fst_blob = largest_prev_col.1 > split.fst_blob_height;
                            let taller_snd_blob = largest_prev_col.1 > split.snd_blob_height;
                            let larger_fst_blob = gr[&largest_prev_row].len() > split.fst_blob_width;
                            let larger_snd_blob = gr[&largest_post_row].len() > split.snd_blob_width;
                            let more_central = (largest_post_col.1 as i32 - largest_post_col.1 as i32).abs() <
                                (split.fst_blob_height as i32 - split.snd_blob_height as i32).abs();

                            // This split favors asymetry at width dimension. Other conditions are possible.
                            // In case of a tie regarding blob width, take the most central one.
                            let width_cmp = (gr[&largest_post_row].len() as i32 - gr[&largest_prev_row].len() as i32).abs().cmp(
                                &(split.snd_blob_width as i32 - split.fst_blob_width as i32).abs()
                            );
                            let req_update = match width_cmp {
                                Ordering::Less => false,
                                Ordering::Equal => more_central,
                                Ordering::Greater => true
                            };
                            if req_update {
                                best_split_row = Some(Split {
                                    split_pos : split_row_ix,
                                    split_len : gr[split_row].len(),
                                    fst_blob_height : largest_prev_col.1,
                                    snd_blob_height : largest_post_col.1,
                                    fst_blob_width : gr[&largest_prev_row].len(),
                                    snd_blob_width : gr[&largest_post_row].len()
                                });
                            }
                        } else {
                            best_split_row = Some(Split {
                                split_pos : split_row_ix,
                                split_len : gr[split_row].len(),
                                fst_blob_height : largest_prev_col.1,
                                snd_blob_height : largest_post_col.1,
                                fst_blob_width : gr[&largest_prev_row].len(),
                                snd_blob_width : gr[&largest_post_row].len()
                            });
                        }
                    }
                }
            }
        }

        if let Some(Split { split_pos, .. }) = best_split_row {
            let prev_patch = patch_from_grouped_rows(
                &sorted_rows[0..split_pos],
                gr.clone(),
                self.color,
                self.scale,
                self.img_height
            );
            let post_patch = patch_from_grouped_rows(
                &sorted_rows[split_pos+1..],
                gr,
                self.color,
                self.scale,
                self.img_height
            );
            Some((prev_patch, post_patch))
        } else {
            None
        }
    }*/

    // Use short-circuit to only iterate over pixels for verification when absolutely required.
    // Note: Outer rect right border should equal position of new pixel, since outer rect starts with
    // size 1.

    /// Pixel is inside rect area or near its bottom border
    pub fn pixel_not_far_below(&self, (r, c) : (usize, usize)) -> bool {
        // println!("rect={}; r = {}", self.outer_rect.0 + self.outer_rect.2, r);
        let rect_r = self.outer_rect.0 + self.outer_rect.2;

        // First is true when first pixel at row is being inserted; second after.
        // let is_below = (rect_r == r || rect_r == r+1);

        // assert!(self.pxs.is_sorted_by(|a, b| a.0.partial_cmp(&b.0 )));

        (rect_r as i32 - r as i32) >= -2
        //r >= 1 &&
            // is_below &&
            //pixel_below_rect(&self.outer_rect, (r, c)) //&&
            // self.pxs.iter().rev() /*.take_while(|px| px.0 >= r-1 ).*/ .any(|px| (px.0 == r || px.0 == r-1) && px.1 == c )
    }

    /// Pixel is inside rect area or near its border right
    pub fn pixel_not_far_right(&self, (r, c) : (usize, usize)) -> bool {
        // println!("rect={}; c= {}", self.outer_rect.0 + self.outer_rect.2, c);
        let rect_c = self.outer_rect.1 + self.outer_rect.3;

        (rect_c as i32 - c as i32) >= -2

        // First is true when first pixel at col is being inserted; second after.
        // let is_right = (rect_c == c || rect_c == c+1);
        //c >= 1 &&
            // is_right &&
        //    pixel_to_right_of_rect(&self.outer_rect, (r, c)) //&&
            // self.pxs.iter().rev().any(|px| px.0 == r && (px.1 == c || px.1 == c-1) )
    }

    /*pub fn pixel_is_left(&self, (r, c) : (usize, usize)) -> bool {
        let is_left = self.outer_rect.1 > 0 && self.outer_rect.1 - 1 == c;
        is_left &&
            pixel_to_left_of_rect(&self.outer_rect, (r, c)) //&&
            // self.pxs.iter().rev().any(|px| px.0 == r && (px.1 == c || px.1 == c+1) )
    }

    pub fn pixel_is_above(&self, (r, c) : (usize, usize)) -> bool {
        let is_above = self.outer_rect.0 > 0 && self.outer_rect.0 - 1 == r;
        is_above &&
        pixel_above_rect(&self.outer_rect, (r, c)) //&&
        // self.pxs.iter().rev() /*.take_while(|px| px.0 >= r-1 ).*/ .any(|px| (px.0 == r || px.0 == r+1) && px.1 == c )
    }*/

    pub fn add_to_right(&mut self, px : (usize, usize), exp_mode : ExpansionMode) {
        assert!(self.pixel_not_far_right(px), format!("Pixel not at right: {:?}", (self.outer_rect, &self.pxs, px)));

        match exp_mode {
            ExpansionMode::Rect => { },
            ExpansionMode::Dense => {
                self.pxs.push(px);
            },
            ExpansionMode::Contour => {
                let mut row_iter = self.pxs.iter().rev().take_while(|(r, _)| *r == px.0 );
                let has_2_this_row = row_iter.next().is_some() && row_iter.next().is_some();
                if has_2_this_row {
                    *(self.pxs.last_mut().unwrap()) = px;
                } else  {
                    self.pxs.push(px);
                }
            }
        }

        self.expand_rect(&[px]);
    }

    pub fn add_to_bottom(&mut self, px : (usize, usize), exp_mode : ExpansionMode) {
        assert!(self.pixel_not_far_below(px), format!("Pixel not at bottom: {:?}", (self.outer_rect, &self.pxs, px)));
        match exp_mode {
            ExpansionMode::Rect => { },
            ExpansionMode::Dense => {
                self.pxs.push(px);
            },
            ExpansionMode::Contour => {
                let mut col_iter = self.pxs.iter().rev().filter(|(_, c)| *c == px.1 );
                let has_2_this_col = col_iter.next().is_some() && col_iter.next().is_some();
                if has_2_this_col {
                    let mut col_iter_mut = self.pxs.iter_mut().rev().filter(|(_, c)| *c == px.1 );
                    // *(self.pxs.last_mut().unwrap()) = px;
                    *(col_iter_mut.next().unwrap()) = px;
                } else {
                    self.pxs.push(px);
                }
            }
        }

        self.expand_rect(&[px]);
    }

    pub fn merge(&mut self, other : Patch) {
        /*assert!(
            shape::rect_overlaps(&self.outer_rect, &other.outer_rect) ||
            shape::rect_contacts(&self.outer_rect, &other.outer_rect),
            format!("Rects not close {:?}", (self.outer_rect, other.outer_rect))
        );*/
        // assert!(shape::rect_contacts(&self.outer_rect, &other.outer_rect), format!("No contact {:?}", (self.outer_rect, other.outer_rect)));

        for pt in other.pxs.iter() {
            self.pxs.push(*pt);
        }

        self.expand_rect(&other.pxs);
    }

    pub fn expand_rect(&mut self, pts : &[(usize, usize)]) {
        for pt in pts.iter() {
            // self.pxs.push(*pt);
            if pt.0 < self.outer_rect.0 {
                self.outer_rect.2 = self.outer_rect.2 + (self.outer_rect.0 - pt.0);
                self.outer_rect.0 = pt.0;
            } else {
                if pt.0 > self.outer_rect.0 + self.outer_rect.2 {
                    self.outer_rect.2 = pt.0 - self.outer_rect.0;
                }
            }
            if pt.1 < self.outer_rect.1 {
                self.outer_rect.3 = self.outer_rect.3 + (self.outer_rect.1 - pt.1);
                self.outer_rect.1 = pt.1;
            } else {
                if pt.1 > self.outer_rect.1 + self.outer_rect.3 {
                    self.outer_rect.3 = pt.1 - self.outer_rect.1;
                }
            }
            self.area += 1;
        }
    }

    pub fn same_color(&self, other : &Self) -> bool {
        self.color == other.color
    }

    /// Maps rows to a (sorted) set of columns.
    pub fn group_rows(&self) -> HashMap<usize, Vec<usize>> {
        let mut row_pxs = HashMap::new();

        let mut row_sorted_pxs = self.pxs.clone();
        row_sorted_pxs.sort_unstable_by(|px_a, px_b| px_a.0.cmp(&px_b.0) );

        // pxs should be sorted by rows here.
        for (row, g_pxs) in row_sorted_pxs.iter().group_by(|px| px.0 ).into_iter() {
            let mut pxs_vec = g_pxs.map(|px| px.1 ).collect::<Vec<_>>();
            pxs_vec.sort_unstable();
            row_pxs.insert(row, pxs_vec);
        }
        row_pxs
    }

    /// Maps columns to a (sorted) set of rows
    pub fn group_cols(&self) -> HashMap<usize, Vec<usize>> {
        let mut col_sorted_pxs = self.pxs.clone();
        col_sorted_pxs.sort_unstable_by(|px_a, px_b| px_a.1.cmp(&px_b.1) );
        let mut col_pxs = HashMap::new();
        for (col, g_pxs) in col_sorted_pxs.iter().group_by(|px| px.1 ).into_iter() {
            let mut pxs_vec = g_pxs.map(|px| px.0 ).collect::<Vec<_>>();
            pxs_vec.sort_unstable();
            col_pxs.insert(col, pxs_vec);
        }
        col_pxs
    }

    pub fn num_regions(&self) -> usize {
        let row_pxs = self.group_rows();
        let mut n_regions = 0;
        for (_, cols) in row_pxs {
            n_regions += cols.len();
        }
        n_regions
    }

    // pub fn area(&self) -> usize {
    //    self.num_regions() * self.scale.pow(2)
    // }

    pub fn outer_points(&self, mode : ExpansionMode) -> Vec<(usize, usize)> {
        match mode {
            ExpansionMode::Dense => {
                let mut row_pxs = self.group_rows();
                let mut sorted_keys = row_pxs.iter().map(|(k, _)| k ).collect::<Vec<_>>();
                if sorted_keys.len() < 3 {
                    return Vec::new();
                }
                sorted_keys.sort();
                let n = sorted_keys.len();
                let mut pts : Vec<(usize, usize)> = Vec::new();

                // Points with "top" part of the patch
                let fst_row = sorted_keys[0];
                for col in row_pxs[fst_row].iter() {
                    pts.push((*fst_row * self.scale, *col * self.scale));
                }

                // Points with "right" part of the patch
                for row in sorted_keys[1..n-1].iter() {
                    pts.push((**row * self.scale, *row_pxs[row].last().unwrap() * self.scale));
                }

                // Points with "bottom" part of the patch
                let last_row = sorted_keys.last().unwrap();
                for col in row_pxs[last_row].iter().rev() {
                    pts.push((**last_row * self.scale, *col * self.scale));
                }

                // Points with "left" part of the patch
                for row in sorted_keys[1..n-1].iter().rev() {
                    pts.push((**row * self.scale, *row_pxs[row].first().unwrap() * self.scale));
                }
                pts
            },
            ExpansionMode::Contour => {
                self.pxs.iter().map(|px| (self.scale * px.0, self.scale * px.1) ).collect()
            },
            ExpansionMode::Rect => {
                unimplemented!()
            }
        }
    }

    pub fn polygon(&self) -> Option<ConvexPolygon> {
        let pts = self.outer_points(ExpansionMode::Dense);

        // Some(Polygon::from(pts))

        let float_pts : Vec<_> = pts.iter()
            .map(|pt| Point2::new(pt.1 as f32, pt.0 as f32 ) )
            .collect();
        ConvexPolygon::from_convex_hull(&float_pts[..])
    }

    pub fn contains(&self, other : &Self) -> Option<bool> {
        let this_poly = self.polygon()?;
        let other_poly = other.polygon()?;
        Some(other_poly.points().iter().all(|pt| this_poly.contains_local_point(pt)))
    }

    /// Order all pixels in this patch, first by row, then by column within rows.
    pub fn sort_by_raster(&mut self) {
        self.pxs.sort_unstable_by(|a, b| a.0.cmp(&b.0) );
        let min_row = self.pxs[0].0;
        let max_row = self.pxs[self.pxs.len()-1].0;
        let mut first_ix = 0;
        let mut last_ix = 0;
        for r in (min_row..max_row+1) {
            let last_ix = self.pxs[first_ix..].iter().position(|px| px.0 != r ).unwrap_or(self.pxs.len());
            self.pxs[first_ix..last_ix].sort_unstable_by(|a, b| a.1.cmp(&b.1) );
            first_ix = last_ix;
        }
    }

}

pub enum NestedPatch {
    Final(Patch),

    /// First patch encloses all others.
    Encloses(Patch, Vec<NestedPatch>)
}

// Must call recursively to build hierarchical organization.
pub fn build_nested(mut v : Vec<Patch>) -> Vec<NestedPatch> {
    let mut final_nested = Vec::new();
    while v.len() > 0 {
        let mut nested : Vec<Patch> = Vec::new();
        for n_ix in 1..v.len() {
            if v[0].contains(&v[n_ix]).unwrap() {
                nested.push(v.swap_remove(n_ix));
            }
        }
        let curr = v.swap_remove(0);
        if nested.len() == 0 {
            final_nested.push(NestedPatch::Final(curr));
        } else {
            // Now, we must call the function recursively over
            // the nested vector to examine if any elements at
            // nested contains any other.
            final_nested.push(NestedPatch::Encloses(curr, build_nested(nested)));
        }
    }
    final_nested
}

/*/// Unlike a Polygon, which has a precise mathematical description as a set of delimiting points,
/// a patch is an amorphous set of pixels known to have a certain color. To occupy a reasonable size,
/// patches are usually calculated by subsampling the image, and verifying the pixels closest to one of a few colors.
/// Neighborhoods might be superimposed to one-another. If image is subsampled by 2, scale will be two, and
/// all pixel coordinates at neighborhood should be multiplied by this value. Neighborhoods are each a 3x3 image
/// region, in raster order, each satisfying merges_with(ix-1) (left neighborhood) and/or merges_with(ix-ncol) (top neighborhood)
#[derive(Clone, Debug)]
pub struct BinaryPatch {

    // pub win : &'a Window<u8>,
    pub neighborhoods : Vec<Pattern>,

    pub color : u8,

    pub scale : u8
}

impl BinaryPatch {

    pub fn inner_polygon(&self) -> Polygon {
        unimplemented!()
    }

    pub fn outer_polygon(&self) -> Polygon {
        unimplemented!()
    }

    /// If the neighborhoods composing this patch are are at scale k,
    /// verifies if they can be simplified to have fewer neighborhoods
    /// at scale k+1.
    pub fn simplify(&mut self) {

    }

}*/

/// Local neighborhood, representing equality state between a center pixel and
/// its spacing=1 neighbors.
#[derive(Clone, Debug)]
pub struct Pattern {

    pub center : (usize, usize),

    pub color : u8,

    /// Whether outer pixels of the patch are equal to center starting from top-left and going row-wise,
    /// ignoring the center pixel:
    /// | 0 | 1 | 2 |
    /// | 3 | 4 | 5 |
    /// | 6 | 7 | 8 |
    pub pattern : [bool; 9]

}

fn symbol(has_px : bool) -> &'static str {
    if has_px {
        "X"
    } else {
        " "
    }
}

/*impl fmt::Display for Pattern {

    fn fmt(&self, f : &mut fmt::Formatter) -> Result<(), fmt::Error> {
        let mut layout = String::new();
        layout += &format!("|{}|{}|{}|\n", symbol(self.pattern[0]), symbol(self.pattern[1]), symbol(self.pattern[2]));
        layout += &format!("|{}|{}|{}|\n", symbol(self.pattern[3]), "X", symbol(self.pattern[4]));
        layout += &format!("|{}|{}|{}|\n", symbol(self.pattern[5]), symbol(self.pattern[6]), symbol(self.pattern[7]));
        write!(f, "{}", layout)
    }
}*/

impl Pattern {

    /*pub fn any_left(&self) -> bool {
        self.pattern[0] || self.pattern[3] || self.pattern[5]
    }

    pub fn all_left(&self) -> bool {
        self.pattern[0] && self.pattern[3] && self.pattern[5]
    }

    pub fn any_top(&self) -> bool {
        self.pattern[0] || self.pattern[1] || self.pattern[2]
    }

    pub fn all_top(&self) -> bool {
        self.pattern[0] && self.pattern[1] && self.pattern[2]
    }

    pub fn any_right(&self) -> bool {
        self.pattern[2] || self.pattern[4] || self.pattern[7]
    }

    pub fn any_bottom(&self) -> bool {
        self.pattern[5] || self.pattern[6] || self.pattern[7]
    }

    pub fn all_bottom(&self) -> bool {
        self.pattern[5] && self.pattern[6] && self.pattern[7]
    }

    pub fn any_horizontal_center(&self) -> bool {
        self.pattern[3] || self.pattern[4]
    }

    pub fn all_horizontal_center(&self) -> bool {
        self.pattern[3] && self.pattern[4]
    }

    pub fn any_vertical_center(&self) -> bool {
        self.pattern[1] || self.pattern[6]
    }

    pub fn all_vertical_center(&self) -> bool {
        self.pattern[1] && self.pattern[6]
    }

    pub fn all_right(&self) -> bool {
        self.pattern[2] && self.pattern[4] && self.pattern[7]
    }

    pub fn horizontal_extension(&self) -> u8 {
        if self.all_top() || self.all_horizontal_center() || self.all_bottom() {
            3
        } else {
            let top_2 = (self.pattern[0] && self.pattern[1]) || (self.pattern[1] && self.pattern[2]);
            let center_2 = self.pattern[3] || self.pattern[4];
            let bottom_2 = (self.pattern[5] && self.pattern[6]) || (self.pattern[6] && self.pattern[7]);
            if top_2 || center_2 || bottom_2 {
                2
            } else {
                1
            }
        }
    }

    pub fn vertical_extension(&self) -> u8 {
        if self.all_left() || self.all_vertical_center() || self.all_right() {
            3
        } else {
            let left_2 = (self.pattern[0] && self.pattern[3]) || (self.pattern[3] && self.pattern[5]);
            let center_2 = self.pattern[1] || self.pattern[6];
            let right_2 = (self.pattern[2] && self.pattern[4]) || (self.pattern[4] && self.pattern[7]);
            if left_2 || center_2 || right_2 {
                2
            } else {
                1
            }
        }
    }

    pub fn left_border(&self) -> [bool; 3] {
        [self.pattern[0], self.pattern[3], self.pattern[5]]
    }

    pub fn top_border(&self) -> [bool; 3] {
        [self.pattern[0], self.pattern[1], self.pattern[2]]
    }

    pub fn right_border(&self) -> [bool; 3] {
        [self.pattern[2], self.pattern[4], self.pattern[7]]
    }

    pub fn bottom_border(&self) -> [bool; 3] {
        [self.pattern[5], self.pattern[6], self.pattern[7]]
    }

    pub fn merges_left(&self, other : &Self) -> bool {
        let lb = self.left_border();
        let border_iter = lb.iter().zip(other.right_border());
        self.color == other.color &&
            self.left_border().iter().any(|l| *l ) &&
            border_iter.clone().all(|(l, r)| *l && r )
    }

    pub fn merges_top(&self, other : &Self) -> bool {
        let tb = self.top_border();
        let border_iter = tb.iter().zip(other.bottom_border());
        self.color == other.color &&
            self.top_border().iter().any(|l| *l ) &&
            border_iter.clone().all(|(t, b)| *t && b )
    }*/

}

pub struct PatchSegmentation {
    px_spacing : usize,
    patches : Vec<Patch>,
    n_patches : usize
}

#[derive(Debug, Clone, Copy)]
pub enum ColorMode {

    Exact(u8),

    Above(u8),

    Below(u8),

    // Carries a value and an absolute value tolerance around it
    Within(u8, u8),

    // Carries a value, an lower absolute tolerance and an upper absolute tolerance around it.
    Between(u8, u8, u8)
}

impl ColorMode {

    fn absolute_tolerance(&self) -> u8 {
        match &self {
            ColorMode::Exact(c) => 1,
            ColorMode::Above(c) => 255 - *c,
            ColorMode::Below(c) => *c,
            ColorMode::Within(c, tol) => c.saturating_add(*tol) - c.saturating_sub(*tol),
            ColorMode::Between(c, low_tol, high_tol) => c.saturating_add(*high_tol) - c.saturating_sub(*low_tol),
        }
    }

    fn color(&self) -> u8 {
        match &self {
            ColorMode::Exact(c) => *c,
            ColorMode::Above(c) => *c,
            ColorMode::Below(c) => *c,
            ColorMode::Within(c, _) => *c,
            ColorMode::Between(c, _, _) => *c,
        }
    }

    fn set_reference_color(&mut self, color : u8) {
        match self {
            ColorMode::Within(ref mut c, _) => *c = color,
            ColorMode::Exact(ref mut c) => *c = color,
            ColorMode::Above(ref mut c) => *c = color,
            ColorMode::Below(ref mut c) => *c = color,
            ColorMode::Between(ref mut c, _, _) => *c = color
        }
    }

    fn matches(&self, px_color : u8) -> bool {
        match self {
            ColorMode::Within(color, tol) => ((px_color as i16 - *color as i16).abs() as u8) < *tol,
            ColorMode::Exact(color) => px_color == *color,
            ColorMode::Above(color) => px_color >= *color,
            ColorMode::Below(color) => px_color <= *color,
            ColorMode::Between(color, low_tol, high_tol) => {
                px_color >= color.saturating_sub(*low_tol) && px_color <= color.saturating_add(*high_tol)
            }
        }
    }

}

impl PatchSegmentation {

    pub fn new(px_spacing : usize) -> Self {
        Self { patches : Vec::with_capacity(16), px_spacing, n_patches : 0 }
    }

    pub fn segment_all<'a>(&'a mut self, win : &WindowMut<'_, u8>, mode : ColorMode, exp_mode : ExpansionMode) -> &'a [Patch] {
        let src_win = unsafe { crate::image::create_immutable(&win) };
        let n_patches = full_color_patches(&mut self.patches, &src_win, self.px_spacing, mode, exp_mode);
        self.n_patches = n_patches;
        &self.patches[0..self.n_patches]
    }

    /// Returns only segments matching the given color.
    pub fn segment_single_color<'a>(&'a mut self, win : &Window<'_, u8>, mode : ColorMode, exp_mode : ExpansionMode) -> &'a [Patch] {
        // let src_win = unsafe { crate::image::create_immutable(&win) };

        // TODO remove this to avoid reallocating pixel vectors.
        // self.patches.clear();

        let n_patches = single_color_patches(&mut self.patches, &win, self.px_spacing, mode, exp_mode);
        self.n_patches = n_patches;
        assert!(self.patches[0..self.n_patches].iter().all(|patch| patch.color == mode.color() ));
        &self.patches[0..self.n_patches]
    }

    pub fn patches(&self) -> &[Patch] {
        &self.patches[0..self.n_patches]
    }

}

/// Keeps state of a patch search
struct PatchSearch {
    prev_row_patch_ixs : Vec<usize>,
    left_patch_ix : Option<usize>,
    top_patch_ix : Option<usize>,
    nrow : usize,
    ncol : usize
}

impl PatchSearch {

    fn new(win : &Window<'_, u8>, px_spacing : usize) -> Self {
        let (ncol, nrow) = (win.width() / px_spacing, win.height() / px_spacing);
        let mut prev_row_patch_ixs : Vec<usize> = Vec::with_capacity(ncol);
        for c in (0..ncol) {
            prev_row_patch_ixs.push(0);
        }
        let (left_patch_ix, top_patch_ix) : (Option<usize>, Option<usize>) = (None, None);
        Self { prev_row_patch_ixs, left_patch_ix, top_patch_ix, nrow, ncol }
    }

}

/// Search the image for disjoint color patches of a single user-specified color. If tol
/// is informed, any pixel witin patch+- tol is considered. If not, only pixels with strictly
/// the desired color are returned.
pub (crate) fn single_color_patches(
    patches : &mut Vec<Patch>,
    win : &Window<'_, u8>,
    px_spacing : usize,
    mode : ColorMode,
    exp_mode : ExpansionMode
) -> usize {
    let mut n_patches = 0;

    /// TODO avoid reallocating this structure, which has Vec<usize>.
    let mut search = PatchSearch::new(win, px_spacing);

    // TODO also avoid reallocating this structure.
    let mut prev_row_mask : Vec<bool> = Vec::with_capacity(win.width() / px_spacing);

    for c in (0..(win.width() / px_spacing)) {
        prev_row_mask.push(false);
    }

    let mut last_matching_col = None;
    for (r, c, px_color) in win.labeled_pixels(px_spacing) {
        // let (r, c) = (ix / win.width(), ix % win.width());
        // let px_color = win[(r*px_spacing, c*px_spacing)];
        search.top_patch_ix = if r >= 1 && prev_row_mask[c] {
            Some(search.prev_row_patch_ixs[c])
        } else {
            None
        };

        if c == 0 {
            last_matching_col = None;
        }

        let color_match = mode.matches(px_color);
        let merges_left = if let Some(last_c) = last_matching_col { c - last_c == 1 } else { false } && color_match && search.left_patch_ix.is_some();
        let merges_top = r >= 1 && prev_row_mask[c] && color_match && search.top_patch_ix.is_some();
        if color_match {
            append_or_update_patch(patches, &mut search, &mut n_patches, win, merges_left, merges_top, r, c, mode.color(), px_spacing, exp_mode);
            last_matching_col = Some(c);
            prev_row_mask[c] = true;
        } else {
            prev_row_mask[c] = false;
            search.left_patch_ix = None;
        }
    }

    if exp_mode == ExpansionMode::Contour {
        for mut patch in patches[0..n_patches].iter_mut() {
            close_contour(patch, win);
        }
    }

    n_patches
}

/// During pixel insertion in a patch, row raster order is preserved, but column raster order is not.
/// Returns up to which index of patches the new data is valid. We keep patches of a previous iteration
/// so the pixel vectors within patches do not get reallocated. In the public API, we use this quantity
/// to limit the index of the patch slice only to the points generated by the current iteration.
pub(crate) fn full_color_patches(
    patches : &mut Vec<Patch>,
    win : &Window<'_, u8>,
    px_spacing : usize,
    mut mode : ColorMode,
    exp_mode : ExpansionMode
) -> usize {

    /*// Recycle previous pixel vectors to avoid reallocations
    let mut prev_pxs : Vec<Vec<(usize, usize)>> = patches
        .iter_mut()
        .map(|mut patch| mem::take(&mut patch.pxs ) )
        .collect();
    patches.clear();*/
    let mut n_patches = 0;

    // let (ncol, nrow) = (win.width() / px_spacing, win.height() / px_spacing);

    // Maps each column to a patch index
    // let mut prev_row_patch_ixs : HashMap<usize, usize> = HashMap::with_capacity(ncol);

    // TODO also take this by &mut
    // let mut prev_row_patch_ixs : Vec<usize> = Vec::with_capacity(ncol);
    // for c in (0..ncol) {
        //prev_row_patch_ixs.insert(c, 0);
    //    prev_row_patch_ixs.push(0);
    // }

    /// TODO avoid reallocating this structure, which has Vec<usize>.
    let mut search = PatchSearch::new(win, px_spacing);

    for (r, c, color) in win.labeled_pixels(px_spacing) {
        // let (r, c) = (ix / win.width(), ix % win.width());
        // let color = win[(r*px_spacing, c*px_spacing)];

        let might_merge_top = if r >= 1 {
            mode.set_reference_color(win[((r-1)*px_spacing, c*px_spacing)]);
            mode.matches(color)
        } else {
            false
        };
        let might_merge_left = if c >= 1 {
            mode.set_reference_color(win[(r*px_spacing, (c-1)*px_spacing)]);
            mode.matches(color)
        } else {
            false
        };

        if c == 0 {
            search.left_patch_ix = None;
        }

        // println!("{},{}", r, c);
        // println!("{:?}", patches);

        /*// Get patch that contains the pixel above the current pixel
        // let top_patch_ix = patches.iter().position(|patch| patch.pxs.iter().any(|px| r >= 1 && px.0 == r-1 && px.1 == c ) );
        let top_patch_ix = patches.iter().rev()
            .position(|patch| patch.pixel_is_below((r, c)) )
            .map(|inv_pos| patches.len() - 1 - inv_pos);*/
        search.top_patch_ix = if r >= 1 { Some(search.prev_row_patch_ixs[c]) } else { None };
        // println!("{},{}", r, c);
        // Get patch that contains the pixel to the left of current pixel
        // let left_patch_ix = patches.iter().position(|patch| patch.pxs.iter().any(|px| c >= 1 && px.0 == r && px.1 == c-1 ) );
        /*let left_patch_ix = patches.iter().rev()
            .position(|patch| patch.pixel_is_right((r, c)) )
            .map(|inv_pos| patches.len() - 1 - inv_pos);*/
        // let left_patch_ix =

        // Verifies if patches have the same color
        /*let merges_top = might_merge_top && if let Some(top) = top_patch_ix.and_then(|ix| patches.get(ix) ) {
            top.color == color
        } else {
            false
        };
        let merges_left = might_merge_left && if let Some(left) = left_patch_ix.and_then(|ix| patches.get(ix) ) {
            left.color == color
        } else {
            false
        };*/
        let merges_left = might_merge_left && search.left_patch_ix.is_some();
        let merges_top = might_merge_top && search.top_patch_ix.is_some();
        append_or_update_patch(patches, &mut search, &mut n_patches, win, merges_left, merges_top, r, c, color, px_spacing, exp_mode);

        // patches.trim(n_patches);
        // println!("{:?}", patches.last().unwrap());
    }

    if exp_mode == ExpansionMode::Contour {
        for mut patch in patches[0..n_patches].iter_mut() {
            close_contour(patch, win);
        }
    }

    n_patches
}

// TODO review patch color strategy. For now, the patch color
// is the color that inaugurated the patch via its top-left pixel
// at the raster search strategy. This contrasts with the growth
// strategy where the patch color is the seed position color.
fn append_or_update_patch(
    patches : &mut Vec<Patch>,
    search : &mut PatchSearch,
    n_patches : &mut usize,
    win : &Window<'_, u8>,
    merges_left : bool,
    merges_top : bool,
    r : usize,
    c : usize,
    color : u8,
    px_spacing : usize,
    exp_mode : ExpansionMode
) {
    match (merges_left, merges_top) {
        (true, true) => {

            let top_differs_left = search.top_patch_ix.unwrap() != search.left_patch_ix.unwrap();
            if top_differs_left {
                merge_left_to_top_patch(patches, search, n_patches);
            }

            // Push new pixel to top patch
            patches[search.top_patch_ix.unwrap()].add_to_bottom((r, c), exp_mode);
            search.prev_row_patch_ixs[c] = search.top_patch_ix.unwrap();
            search.left_patch_ix = Some(search.top_patch_ix.unwrap());
        },
        (true, false) => {
            patches[search.left_patch_ix.unwrap()].add_to_right((r, c), exp_mode);
            search.prev_row_patch_ixs[c] = search.left_patch_ix.unwrap();
        },
        (false, true) => {
            patches[search.top_patch_ix.unwrap()].add_to_bottom((r, c), exp_mode);
            search.prev_row_patch_ixs[c] = search.top_patch_ix.unwrap();
            search.left_patch_ix = Some(search.top_patch_ix.unwrap());
        },
        (false, false) => {
            add_patch(patches, search, n_patches, win, r, c, color, px_spacing);
        }
    }
}

/// Since we might be recycling the patches Vec<_> from a previous iteration,
/// either update one of the junk patches to hold the new patch, or push
/// this new patch to the vector. In either case, n_patches (which hold the
/// valid patch slice size) will be incremented by one.
fn add_patch(
    patches : &mut Vec<Patch>,
    search : &mut PatchSearch,
    n_patches : &mut usize,
    win : &Window<'_, u8>,
    r : usize,
    c : usize,
    color : u8,
    px_spacing : usize
) {
    if *n_patches < patches.len() {
        patches[*n_patches].pxs.clear();
        patches[*n_patches].pxs.push((r, c));
        patches[*n_patches].outer_rect = (r, c, 1, 1);
        patches[*n_patches].color = color;
        patches[*n_patches].scale = px_spacing;
        patches[*n_patches].img_height = win.height();
    } else {
        patches.push(Patch::new((r, c), color, px_spacing, win.height()));
    }
    *n_patches += 1;
    search.prev_row_patch_ixs[c] = *n_patches - 1;
    search.left_patch_ix = Some(*n_patches - 1);
}

fn merge_left_to_top_patch(patches : &mut Vec<Patch>, search : &mut PatchSearch, n_patches : &mut usize) {
    // This method can't be called when the left and top patches are the same.
    assert!(search.left_patch_ix.unwrap() != search.top_patch_ix.unwrap());

    let left_patch = mem::take(&mut patches[search.left_patch_ix.unwrap()]);
    patches[search.top_patch_ix.unwrap()].merge(left_patch);

    // Remove old left patch (now merged with top).
    patches.remove(search.left_patch_ix.unwrap());

    if let Some(ref mut ix) = search.top_patch_ix {
        if *ix > search.left_patch_ix.unwrap() {
            *ix -= 1;
        }
    } else {
        panic!()
    }

    // Update indices of previous row patches, since the remove(.) invalidated
    // every index >= left patch.
    for mut ix in search.prev_row_patch_ixs.iter_mut() {
        if *ix == search.left_patch_ix.unwrap() {
            *ix = search.top_patch_ix.unwrap();
        } else {
            if *ix > search.left_patch_ix.unwrap() {
                *ix -= 1;
            }
        }
    }

    // Account for removed patch.
    *n_patches -= 1;
}

#[test]
fn test_patches() {
    let check = Image::<u8>::new_checkerboard(16, 4);
    println!("{}", check);
    for patch in check.full_window().patches(1).iter() {
        println!("{:?}", patch);
        println!("{:?}", patch.polygon());
    }
}

/*/// Returns the color patches of a given labeled window (such as retrurned by segment_colors).
pub fn binary_patches(label_win : &Window<'_, u8>, px_spacing : usize) -> Vec<BinaryPatch> {
    let mut curr_patch = 0;
    let mut neighborhoods : Vec<(Pattern, usize)> = Vec::new();
    let (ncol, nrow) = (label_win.width() / px_spacing, label_win.height() / px_spacing);
    assert!(px_spacing % 2 == 0, "Pixel spacing should be an even number");

    // Establish 4-neighborhoods by iterating over non-border allocations.
    for (ix_row, row) in (1..nrow-1).step_by(3).enumerate() {
        for (ix_col, col) in (1..ncol-1).step_by(3).enumerate() {
            let neighborhood = extract_neighborhood(label_win, (row, col));

            let merges_top = if row > 1 {
                let top_neighbor = &neighborhoods[ix_row*ncol + ix_col - ncol].0;
                neighborhood.merges_top(top_neighbor)
            } else {
                false
            };

            let merges_left = if col > 1 {
                let left_neighbor = &neighborhoods[ix_row*ncol + ix_col - 1].0;
                neighborhood.merges_left(left_neighbor)
            } else {
                false
            };

            match (merges_left, merges_top) {
                (true, true) => {
                    let left_neighbor_ix = neighborhoods[ix_row*ncol + ix_col - 1].1;
                    let top_neighbor_ix = neighborhoods[ix_row*ncol + ix_col - ncol].1;

                    // If both are true, and left and top are not merged, this means this
                    // neighborhood is a link element to the top and left patches. Attribute
                    // top neighbor (older) index to all patches matched to left neighbor index.
                    if left_neighbor_ix != top_neighbor_ix {
                        neighborhoods.iter_mut().for_each(|(_, patch_ix)| {
                            if *patch_ix == left_neighbor_ix {
                                *patch_ix = top_neighbor_ix;
                            }
                        });
                    }
                    neighborhoods.push((neighborhood, top_neighbor_ix));
                },
                (true, false) => {
                    let left_neighbor_ix = neighborhoods[ix_row*ncol + ix_col - 1].1;
                    neighborhoods.push((neighborhood, left_neighbor_ix));
                },
                (false, true) => {
                    let top_neighbor_ix = neighborhoods[ix_row*ncol + ix_col - ncol].1;
                    neighborhoods.push((neighborhood, top_neighbor_ix));
                },
                (false, false) => {
                    neighborhoods.push((neighborhood, curr_patch));
                    curr_patch += 1;
                }
            }
        }
    }

    let mut patches = Vec::new();
    let distinct_patch_indices = neighborhoods.iter().map(|(_, ix)| ix ).unique();

    for ix in distinct_patch_indices {

        let curr_neighborhoods : Vec<Neighborhood> = neighborhoods.iter()
            .filter(|(_, n_ix)| n_ix == ix )
            .map(|(n, _)| n )
            .cloned()
            .collect();
        let color = curr_neighborhoods[0].color;

        patches.push(BinaryPatch {
            neighborhoods : curr_neighborhoods,
            scale : px_spacing as u8,
            color
        });
    }

    patches
}*/

fn flatten_to_8bit(v : &f64) -> u8 {
    v.max(0.0).min(255.0) as u8
}

/// Transform each cluster mean to a valid 8-bit color value.
pub fn extract_mean_colors(km : &KMeans) -> Vec<u8> {
    km.means()
        .map(|m| flatten_to_8bit(&m[0]) )
        .collect::<Vec<_>>()
}

pub fn extract_extreme_colors(
    km : &KMeans,
    sample : impl Iterator<Item=impl Borrow<[f64]>> + Clone
) -> Vec<(u8, u8)> {
    (0..km.means().count()).map(|ix| bayes::fit::cluster::extremes(&km, sample.clone(), ix).unwrap() )
        .map(|(low, high)| (flatten_to_8bit(&low), flatten_to_8bit(&high)) )
        .collect::<Vec<_>>()
}

/// Calculates the central point of all pixels within a given color interval.
/// This quantity is meaningful when color represents a delimited region of space.
pub fn color_centroid(
    win : &Window<'_, u8>,
    min_color : u8,
    max_color : u8,
    subsampling : usize
) -> Option<(usize, usize)> {
    let (mut accum_y, mut accum_x) = (0, 0);
    let (mut n_y, mut n_x) = (0, 0);
    for (ix, px) in win.pixels(subsampling).enumerate() {
        if *px >= min_color && *px <= max_color {
            let (y, x) = (ix / win.height(), ix % win.height());
            accum_y += y;
            accum_x += x;
            n_y += 1;
            n_x += 1;
        }
    }
    if n_x > 0 && n_y > 0 {
        Some((accum_y / n_y, accum_x / n_x))
    } else {
        None
    }
}

/// colors : Sequence of representative colors returned by segment_colors ->extract_colors. Overwrites
/// all pixels of win with the closest color in the vector to each pixel.
pub fn write_segmented_colors_to_window<'a>(win : &'a mut WindowMut<'a, u8>, colors : &'a [u8]) {
    for px in win.pixels_mut(1) {
        let mut min_dist : (usize, u8) = (0, u8::MAX);
        for (ix_col, color) in colors.iter().enumerate() {
            let dist = ((*px as i16) - (*color as i16)).abs() as u8;
            if dist < min_dist.1 {
                min_dist = (ix_col, dist);
            }
        }
        *px = colors[min_dist.0];
    }
}

/*pub fn write_patches_to_window<'a>(win : &'a mut WindowMut<'a, u8>, patches : &'a [Patch]) {
    for patch in patches.iter() {
        assert!(patch.scale % 2 == 0, "Patch scale should be an even number");
        for neigh in patch.neighborhoods.iter() {
            let scaled_center = (neigh.center.0 * (patch.scale as usize), neigh.center.1 * (patch.scale as usize));
            let scaled_dim = (3*(patch.scale as usize), 3*(patch.scale as usize));
            let scaled_offset = (scaled_center.0 - scaled_dim.0 / 2, scaled_center.1 - scaled_dim.1 / 2);

            // This assumes the neighborhood is completely filled with the center color, ignoring neighborhood pattern.
            // *win` was mutably borrowed here in the previous iteration of the loop
            win.apply_to_sub_window(scaled_offset, scaled_dim, move |mut local : WindowMut<'_, u8>| { local.fill(neigh.color); } );
            // win.pixels_mut(2);
        }
    }
}*/

/*/// Verifies equality of neighbor (spacing=1 only) pixels to pixel centered at (row, col).
pub fn extract_pattern(label_img : &Window<'_, u8>, (row, col) : (usize, usize), mode : ColorMode) -> Pattern {
    let center = mode.matches(label_img[(row, col)]);
    let up = mode.matches(label_img[(row - 1, col)]);
    let down = mode.matches(label_img[(row + 1, col)]);
    let left = mode.matches(label_img[(row, col -1)]);
    let right = mode.matches(label_img[(row, col + 1)]);
    let up_left = mode.matches(label_img[(row - 1, col - 1)]);
    let up_right = mode.matches(label_img[(row - 1, col + 1)]);
    let down_left = mode.matches(label_img[(row + 1, col - 1)]);
    let down_right = mode.matches(label_img[(row + 1, col + 1)]);
    let pattern = [
        up_left,
        up,
        up_right,
        left,
        center,
        right,
        down_left,
        down,
        down_right
    ];
    Pattern { color : mode.reference_color(), pattern, center : (row, col) }
}*/

/*pub fn pattern_matches(img : &Window<'_, u8>, scale : usize, pattern : Pattern) -> Vec<Pattern> {

    for r in (scale..(win.height() - scale)).step_by(scale) {
        for c in (scale..(win.width() - scale)).step_by(scale) {
            if extract_pattern(img, (r, c))
        }
    }

}*/

/*/// Verifies equality of neighbor (spacing=2 only) pixels to pixel centered at (row, col).
pub fn extract_extended_neighborhood(label_img : &Window<'_, u8>, (row, col) : (usize, usize)) -> Option<Extended> {
    let center = label_img[(row, col)];
    let up = label_img[(row - 2, col)];
    let down = label_img[(row + 2, col)];
    let left = label_img[(row, col - 2)];
    let right = label_img[(row, col + 2)];

    // Verify if pixels at 3x3 cross are equal.
    if [up, down, left, right].iter().all(|px| *px == center ) {

        // Verify if borders at 3x3 (inner) box are all equal.
        let top_border = label_img.sub_row(row - 2, (col-1)..(col+2)).unwrap().pixels().all(|px| *px == center );
        let bottom_border = label_img.sub_row(row + 2, (col-1)..(col+2)).unwrap().pixels().all(|px| *px == center );
        let left_border = label_img.sub_col((row - 1)..(row+2), col-2).unwrap().pixels().all(|px| *px == center );
        let right_border = label_img.sub_col((row - 1)..(row+2), col+2).unwrap().pixels().all(|px| *px == center );

        // Verify if corners at 5x5 box are all equal.
        let top_left = label_img[(row-2, col-2)] == center;
        let top_right = label_img[(row-2, col+2)] == center;
        let bottom_left = label_img[(row+2, col-2)] == center;
        let bottom_right = label_img[(row+2, col+2)] == center;

        if top_border && left_border && right_border && bottom_border {
            if top_left && top_right && bottom_left && bottom_right {
                Some(Extended::TwentyFour)
            } else {
                Some(Extended::Twenty)
            }
        } else {
            Some(Extended::Twelve)
        }
    } else {
        None
    }
}*/

/*let kind = match local {
    Local::Eight => {
        match segmentation::extract_extended_neighborhood(label_win, (row, col)) {
            Some(ext) => Either::Right(ext),
            None => Either::Left(local)
        }
    },
    local => Either::Left(local)
 };*/

// TODO examine kmeans_colors crate


