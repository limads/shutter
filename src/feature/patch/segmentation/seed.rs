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
use std::ops::Range;
use std::convert::TryInto;
use std::convert::TryFrom;
use std::ops::Div;
use std::ops::Add;
use bayes;
use super::*;

/// Represents an iteration for one of the four expansion fronts for a convex shape
/// that is supposed to be grown from a pixel seed. "curr" holds pixels that match
/// the patch color for the current border. "past" holds pixels that matched the
/// patch color at the previous iteration.
#[derive(Default, Clone, Debug)]
struct RowPair {
    curr : Vec<(u16, u16)>,
    past_contiguous : Vec<(usize, usize)>,
    curr_contiguous : Vec<(usize, usize)>,
    past : Vec<(u16, u16)>
}

impl RowPair {

    /// Starts a new row pair without any values.
    fn new_empty() -> Self {
        Self {
            curr : Vec::with_capacity(32),
            past : Vec::with_capacity(32),
            past_contiguous : Vec::with_capacity(32),
            curr_contiguous : Vec::with_capacity(32),
        }
    }

    /// Starts a new row pair with a seed pixel value at the past row.
    fn new(px : (u16, u16)) -> Self {
        let (curr, curr_contiguous, mut past, mut past_contiguous) = (Vec::with_capacity(32), Vec::with_capacity(32), Vec::with_capacity(32), Vec::with_capacity(32));
        past.push(px);
        past_contiguous.push((0, 0));
        Self { curr, past, past_contiguous, curr_contiguous }
    }

    /// Starts a row pair for a new image seed, preserving the old allocations.
    pub fn reset(&mut self, px : (u16, u16)) {
        self.curr.clear();
        self.past.clear();
        self.past_contiguous.clear();
        self.curr_contiguous.clear();
        self.past.push(px);
        self.past_contiguous.push((0, 0));
    }

}

/// Represents an iteration for all four expansion fronts for a convex shape to be grown
/// for a pixel seed.
#[derive(Default, Clone, Debug)]
struct ExpansionFront {
    top : RowPair,
    left : RowPair,
    bottom : RowPair,
    right : RowPair
}

impl ExpansionFront {

    /// Starts a new front with clean allocations (no pixel seed).
    fn new_empty() -> Self {
        let top = RowPair::new_empty();
        let (left, bottom, right) = (top.clone(), top.clone(), top.clone());
        Self {
            top,
            left,
            bottom,
            right
        }
    }

    /// Starts all fronts with the "past" row populated with a seed pixel.
    fn new(px : (u16, u16)) -> Self {
        let pair = RowPair::new(px);
        Self {
            top : pair.clone(),
            left : pair.clone(),
            bottom : pair.clone(),
            right : pair
        }
    }

    /// Resets the iteration using the informed pixel, preserving old allocations.
    pub  fn reset(&mut self, px : (u16, u16)) {
        self.top.reset(px);
        self.left.reset(px);
        self.bottom.reset(px);
        self.right.reset(px);
    }

}

/// Convex-patch seed growing segmentation. If patch is not convex, this algorithm
/// might miss non-convex patch areas. To segment a full image, several SeedGrowth
/// structures should be instantiated, (one for each desired seed). If regions are
/// required to be non-overlapping, the SeedGrowth algorithm can be cleaned after each
/// iteration, and the borders of the previous iteration can be used as information to
/// where the next seed should be positioned, and the image should be trimmed taking only
/// the regions not yet considered at the previous iteration. TODO rename to SeedSegmenter or
/// FanSegmenter, since pixels are expanded in a fan-out fashion.
#[derive(Debug, Clone)]
pub struct SeedGrowth {
    front : ExpansionFront,
    patch : Patch,
    px_spacing : usize,
    max_area : Option<usize>,
    exp_mode : ExpansionMode,
    grown : bool
}

impl SeedGrowth {

    pub fn last_patch(&self) -> Option<&Patch> {
        if self.grown {
            Some(&self.patch)
        } else {
            None
        }
    }

    pub fn last_patch_mut(&mut self) -> Option<&mut Patch> {
        if self.grown {
            Some(&mut self.patch)
        } else {
            None
        }
    }

    pub fn new(px_spacing : usize, max_area : Option<usize>, exp_mode : ExpansionMode) -> Self {
        let patch = Patch::new_empty();
        Self { front : ExpansionFront::new_empty(), px_spacing, patch, max_area, exp_mode, grown : false }
    }

    pub fn grow<F>(&mut self, win : &Window<'_, u8>, seed : (u16, u16), comp : F, close_at_end : bool, adaptive : bool) -> Option<&Patch>
    where
        F : Fn(u8)->bool
    {

        if seed.0 / self.px_spacing as u16 > (win.height() - 1) as u16 || seed.1 as u16 / self.px_spacing as u16 > (win.width() - 1)  as u16 {
            // println!("Warning: Seed extrapolate image area");
            self.grown = false;
            return None;
        }

        self.front.reset(seed);
        self.patch.pxs.clear();

        if self.exp_mode == ExpansionMode::Dense {
            self.patch.pxs.push(seed);
        }

        self.patch.outer_rect = (seed.0, seed.1, 1, 1);
        self.patch.color = win[seed];
        self.patch.scale = self.px_spacing as u16;
        self.patch.img_height = win.height();
        self.patch.area = 1;

        if self.exp_mode == ExpansionMode::Contour {
            assert!(self.patch.pxs.len() == 0);
        } else {
            assert!(self.patch.pxs.len() == 1);
        }

        if grow(&mut self.patch, &mut self.front, &win, seed, self.px_spacing, comp, ReferenceMode::Constant, self.max_area, self.exp_mode, close_at_end, adaptive) {
            self.grown = true;
            Some(&self.patch)
        } else {
            self.grown = false;
            None
        }
    }

}

fn expand_rect(ext : &mut RowPair, patch : &mut Patch, outer_rect : (u16, u16, u16, u16), fst_corner : bool, snd_corner : bool) {
    ext.past.clear();
    patch.area = (outer_rect.2 * outer_rect.3) as usize;
    ext.past_contiguous.clear();
    mem::swap(&mut ext.curr, &mut ext.past);
    mem::swap(&mut ext.curr_contiguous, &mut ext.past_contiguous);
}

fn expand_dense(ext : &mut RowPair, patch : &mut Patch, outer_rect : (u16, u16, u16, u16), fst_corner : bool, snd_corner : bool) {
    patch.area += ext.past.len();
    patch.pxs.extend(ext.past.drain(..));
    ext.past_contiguous.clear();
    mem::swap(&mut ext.curr, &mut ext.past);
    mem::swap(&mut ext.curr_contiguous, &mut ext.past_contiguous);
}

fn expand_contour(ext : &mut RowPair, patch : &mut Patch, outer_rect : (u16, u16, u16, u16), fst_corner : bool, snd_corner : bool) {
    if ext.curr.len() > 0 {
        // TODO Remove old corner pixels
        // ext.past.remove(4 - ix);

        // Only push corners when the distance between corners of edge limits
        // are not sufficient to represent the shape in its current state.
        if !fst_corner {
            patch.pxs.push(ext.past[0]);
        }

        let n = ext.past.len();
        if n > 1 && !snd_corner {
            patch.pxs.push(ext.past[n-1]);
        }

        patch.area += ext.past.len();

        // Stop swapping at the last empty border so eventually we reach the final
        // branch with the last past pixels that were found.
        ext.past.clear();
    } else {
        // The last patch iteration will take all remaining contour values.
        patch.area += ext.past.len();
        patch.pxs.extend(ext.past.drain(..));
    }
    ext.past_contiguous.clear();
    mem::swap(&mut ext.curr, &mut ext.past);
    mem::swap(&mut ext.curr_contiguous, &mut ext.past_contiguous);
}

/// Computes a single iteration of the seed expansion algorithm.
fn expand_patch(
    exp_patch : &mut ExpansionFront,
    patch : &mut Patch,
    outer_rect : (u16, u16, u16, u16),
    // mode : ExpansionMode,
    win : &Window<'_, u8>,
    seed : (u16, u16),
    extension_func : fn(&mut RowPair, &mut Patch, (u16, u16, u16, u16), bool, bool)
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
        let past_is_seed = ext.past.len() == 1 && ext.past[0].clone() == /*Some((patch.pxs[0]))*/ seed;
        if past_is_seed {
            ext.past.clear();
            mem::swap(&mut ext.curr, &mut ext.past);
            continue;
        }
        extension_func(ext, patch, outer_rect, *fst_corner, *snd_corner);
    }
}

fn update_stats(n_px : &mut u16, sum : &mut u64, mean : &mut u64, sum_abs_dev : &mut u64, abs_dev : &mut u64, new : u8) {
    //if ref_mode == ReferenceMode::Adaptive {
    *sum += new as u64;
    *n_px += 1;
    *mean = *sum / *n_px as u64;
    let dev = (*mean as i64 - new as i64).abs() as u64;
    *sum_abs_dev += dev;
    *abs_dev = *sum_abs_dev / *n_px as u64;

    // mode.set_reference_color(mean as u8);
    // }
}

fn expand_rect_with_front(rect : &mut (u16, u16, u16, u16), exp : &ExpansionFront) {
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

fn pixel_neighbors_row_contiguous(front_part : &RowPair, px : (u16, u16)) -> bool {
    front_part.past_contiguous.iter()
        .any(|(from, to)| px.0 >= front_part.past[*from].0.saturating_sub(1) && px.0 <= front_part.past[*to].0 + 1 )
}

fn pixel_neighbors_col_contiguous(front_part : &RowPair, px : (u16, u16)) -> bool {
    front_part.past_contiguous.iter()
        .any(|(from, to)| px.1 >= front_part.past[*from].1.saturating_sub(1) && px.1 <= front_part.past[*to].1 + 1 )
}

fn pixel_neighbors_top(exp_patch : &ExpansionFront, px : (u16, u16)) -> bool {
    pixel_neighbors_last_at_row(&exp_patch.top.curr[..], px) ||
    //pixel_neighbors_row(&exp_patch.top.past[..], px)
    pixel_neighbors_row_contiguous(&exp_patch.top, px)
}

fn pixel_neighbors_left(exp_patch : &ExpansionFront, px : (u16, u16)) -> bool {
    pixel_neighbors_last_at_col(&exp_patch.left.curr[..], px) ||
    //pixel_neighbors_col(&exp_patch.left.past[..], px)
    pixel_neighbors_col_contiguous(&exp_patch.left, px)
}

fn pixel_neighbors_bottom(exp_patch : &ExpansionFront, px : (u16, u16)) -> bool {
    pixel_neighbors_last_at_row(&exp_patch.bottom.curr[..], px) ||
    //pixel_neighbors_row(&exp_patch.bottom.past[..], px)
    pixel_neighbors_row_contiguous(&exp_patch.bottom, px)
}

fn pixel_neighbors_right(exp_patch : &ExpansionFront, px : (u16, u16)) -> bool {
    pixel_neighbors_last_at_col(&exp_patch.right.curr[..], px) ||
    //pixel_neighbors_col(&exp_patch.right.past[..], px)
    pixel_neighbors_col_contiguous(&exp_patch.right, px)
}

fn update_contiguous(front_part : &mut RowPair, px : &(u16, u16)) {
    if let Some(last) = front_part.curr.last() {
        if px.0 - last.0 == 1 {
            front_part.curr_contiguous.last_mut().unwrap().1 += 1;
        } else {
            let ix = front_part.curr.len()-1;
            front_part.curr_contiguous.push((ix, ix));
        }
    } else {
        let ix = front_part.curr.len()-1;
        front_part.curr_contiguous.push((ix, ix));
    }
}

/// Grows a patch from a pixel seed.
/// cf. Connected Components (Szeliski, 2011, p. 131)
fn grow<F>(
    patch : &mut Patch,
    exp_patch : &mut ExpansionFront,
    win : &Window<'_, u8>,
    seed : (u16, u16),
    px_spacing : usize,
    comp : F,
    ref_mode : ReferenceMode,
    max_area : Option<usize>,
    exp_mode : ExpansionMode,
    close_at_end : bool,
    adaptive : bool
) -> bool
where
    F : Fn(u8)->bool
{

    // if !mode.matches(win[seed]) {
    //    println!("Seed does not match desired color");
    //    return None;
    // }
    // mode.set_reference_color(win[seed]);
    if !comp(win[seed]) {
        return false;
    }

    // let mut patch = Patch::new((seed.0 as u16, seed.1 as u16), win[seed], 1, win.height());
    // let seed = (seed.0 as u16, seed.1 as u16);
    // let mut exp_patch = ExpansionFront::new(seed);

    let (mut grows_left, mut grows_top, mut grows_right, mut grows_bottom) = (true, true, true, true);

    // Still need to alter pixel_neighbors_row and pixel_below_rect, etc to account for differing px_spacings.
    // Will check the absolute difference is <= px_spacing there.
    assert!(px_spacing == 1);

    let mut outer_rect = (seed.0, seed.1, 1, 1);

    let mut n_px = 0;
    let mut sum_abs_dev : u64 = 0;
    let mut abs_dev : u64 = 0;
    let mut sum : u64 = win[seed] as u64;
    let mut mean : u64 = 0;

    let (h, w) = (win.height() as u16, win.width() as u16);

    let extension_func = match exp_mode {
        ExpansionMode::Rect => {
            expand_rect
        },
        ExpansionMode::Dense => {
            expand_dense
        },
        ExpansionMode::Contour => {
            expand_contour
        }
    };

    // Loop counter, starting from seed and incrementing by px_spacing.
    let mut abs_dist = px_spacing as u16;

    debug_assert!(

        // Current front should be empty.
        exp_patch.top.curr.is_empty() &&
        exp_patch.left.curr.is_empty() &&
        exp_patch.right.curr.is_empty() &&
        exp_patch.bottom.curr.is_empty() &&

        // Past front should contain seed only.
        exp_patch.top.past.len() == 1 &&
        exp_patch.left.past.len() == 1 &&
        exp_patch.right.past.len() == 1 &&
        exp_patch.bottom.past.len() == 1
    );

    loop {

        debug_assert!(
            exp_patch.top.curr.is_empty() &&
            exp_patch.left.curr.is_empty() &&
            exp_patch.right.curr.is_empty() &&
            exp_patch.bottom.curr.is_empty()
        );

        let left_col = if grows_left { seed.1.checked_sub(abs_dist) } else { None };
        let top_row = if grows_top { seed.0.checked_sub(abs_dist) } else { None };
        let right_col = if grows_right {
            if seed.1 + abs_dist < w { Some(seed.1 + abs_dist) } else { None }
        } else {
            None
        };
        let bottom_row = if grows_bottom {
            if seed.0 + abs_dist < h { Some(seed.0 + abs_dist) } else { None }
        } else {
            None
        };

        let row_end = (seed.0+abs_dist+1).min(h);
        let col_end = (seed.1+abs_dist+1).min(w);
        let row_range = (seed.0.saturating_sub(abs_dist)..( row_end )).step_by(px_spacing as usize);
        let col_range = (seed.1.saturating_sub(abs_dist)..( col_end )).step_by(px_spacing as usize);

        // TODO when pixel is not in rect range, we can break the loop.

        // let matches_color = comp(win[px]);

        let mut grows_top = false;
        if exp_patch.top.past.len() >= 1 {
            let top_range = (exp_patch.top.past[0].1.saturating_sub(1))..((exp_patch.top.past[exp_patch.top.past.len()-1].1 + 2).min(w));
            if let Some(r) = top_row {
                for c in top_range.clone() {
                    let px = (r, c);
                    let is_corner = (c == top_range.start || c == top_range.end-1);
                    let match_adaptive = abs_dist / px_spacing as u16 > 2 && (win[px] as i16 - (mean as i16)).abs() < 12;
                    if (comp(win[px]) || (adaptive && match_adaptive)) && (is_corner || (pixel_neighbors_top(&exp_patch, px))) {
                        if !grows_top {
                            // outer_rect.0 = r;
                            // outer_rect.2 += px_spacing;
                            grows_top = true;
                        }
                        exp_patch.top.curr.push(px);
                        update_contiguous(&mut exp_patch.top, &px);
                        if adaptive {
                            update_stats(&mut n_px, &mut sum, &mut mean, &mut sum_abs_dev, &mut abs_dev, win[px]);
                        }
                    }
                }
            }
        }

        let mut grows_left = false;
        if exp_patch.left.past.len() >= 1 {
            let left_range = (exp_patch.left.past[0].0.saturating_sub(1))..((exp_patch.left.past[exp_patch.left.past.len()-1].0 + 2).min(h));
            if let Some(c) = left_col {
                for r in left_range.clone() {
                    let px = (r, c);
                    let is_corner = (r == left_range.start || r == left_range.end-1);
                    let match_adaptive = abs_dist / px_spacing as u16 > 2 && (win[px] as i16 - (mean as i16)).abs() < 12;
                    if (comp(win[px]) || (adaptive && match_adaptive)) && ( is_corner || (pixel_neighbors_left(&exp_patch, px))) {
                        if !grows_left {
                            // outer_rect.1 = c;
                            // outer_rect.3 += px_spacing;
                            grows_left = true;
                        }
                        exp_patch.left.curr.push(px);
                        update_contiguous(&mut exp_patch.left, &px);
                        if adaptive {
                            update_stats(&mut n_px, &mut sum, &mut mean, &mut sum_abs_dev, &mut abs_dev, win[px]);
                        }
                    }
                }
            }
        }

        let mut grows_bottom = false;
        if exp_patch.bottom.past.len() >= 1 {
            let bottom_range = (exp_patch.bottom.past[0].1.saturating_sub(1))..((exp_patch.bottom.past[exp_patch.bottom.past.len()-1].1 + 2).min(w));
            if let Some(r) = bottom_row {
                for c in bottom_range.clone() {
                    let px = (r, c);
                    let is_corner = (r == bottom_range.start || r == bottom_range.end-1);
                    let match_adaptive = abs_dist / px_spacing as u16 > 2 && (win[px] as i16 - (mean as i16)).abs() < 12;
                    if (comp(win[px]) || (adaptive && match_adaptive)) && (is_corner || ( pixel_neighbors_bottom(&exp_patch, px))) {
                        if !grows_bottom {
                            // outer_rect.2 += px_spacing;
                            grows_bottom = true;
                        }
                        exp_patch.bottom.curr.push(px);
                        update_contiguous(&mut exp_patch.bottom, &px);

                        if adaptive {
                            update_stats(&mut n_px, &mut sum, &mut mean, &mut sum_abs_dev, &mut abs_dev, win[px]);
                        }
                    }
                }
            }
        }

        let mut grows_right = false;
        if exp_patch.right.past.len() >= 1 {
            let right_range = (exp_patch.right.past[0].0.saturating_sub(1))..((exp_patch.right.past[exp_patch.right.past.len()-1].0 + 2).min(h));
            if let Some(c) = right_col {
                for r in right_range.clone() {
                    let px = (r, c);
                    let is_corner = (r == right_range.start || r == right_range.end-1);
                    let match_adaptive = abs_dist / px_spacing as u16 > 2 && (win[px] as i16 - (mean as i16)).abs() < 12;
                    if (comp(win[px]) || (adaptive && match_adaptive)) && (is_corner || ( pixel_neighbors_right(&exp_patch, px))) {
                        if !grows_right {
                            // outer_rect.3 += px_spacing;
                            grows_right = true;
                        }

                        exp_patch.right.curr.push(px);
                        update_contiguous(&mut exp_patch.right, &px);

                        if adaptive {
                            update_stats(&mut n_px, &mut sum, &mut mean, &mut sum_abs_dev, &mut abs_dev, win[px]);
                        }
                    }
                }
            }
        }

        let grows_any = grows_left || grows_right || grows_bottom || grows_top;
        expand_rect_with_front(&mut outer_rect, &exp_patch);
        patch.outer_rect = outer_rect;
        patch.color = mean as u8;
        expand_patch(exp_patch, patch, outer_rect, win, seed, extension_func);

        if let Some(area) = max_area {
            if outer_rect.2 * outer_rect.3 > area as u16 {
                return false;
            }
        }

        if !grows_any {
            if exp_mode == ExpansionMode::Contour {

                // Remove seed before closing patch
                // patch.pxs.swap_remove(0);

                // A contour should have at least three pixel points. A dense
                // patch can have any number of points >= 1.
                if patch.pxs.len() < 3 {
                    return false;
                }

                if close_at_end {
                    close_contour(patch, win);
                }

            }
            // return Some(patch);
            return true;
        }

        abs_dist += px_spacing as u16;
    }

    debug_assert!(patch.pxs.len() > 0);

    //if exp_mode == ExpansionMode::Contour {
    // Just make sure seed isn't inserted
    //    debug_assert!(patch.pxs.iter().find(|px| **px == seed ).is_none());
    //}

    // Some(patch)
    true
}


#[test]
fn test_seed() {

    use crate::feature::patch::seed::*;
    use crate::image::*;

	let mut img = Image::<u8>::new_constant(100, 100, 10);
    let radius : f32 = 20.;

	for i in 0..100 {
		for j in 0..100 {
			let dist_center = ((i as i32 - 50).pow(2) as f32 + (j as i32 - 50).pow(2) as f32).sqrt();
		    if dist_center < radius {
		        img[(i, j)] = 255;
		    }
		}
	}

	let mut growth = SeedGrowth::new(
		1,
		None,
		crate::feature::patch::ExpansionMode::Contour
	);

	let patch = growth.grow(
		&img.full_window(),
		(50, 50),
		|byte| byte == 255
	).unwrap();
	let rect = patch.outer_rect::<usize>();
	let pxs = patch.outer_points(crate::feature::patch::ExpansionMode::Contour);
	img.draw(Mark::Shape(pxs.clone(), 127));

	let (center, emp_radius) = crate::feature::shape::outer_circle(&pxs[..]);
	println!("patch area = {}", patch.area());
	println!("circle area = {}", std::f32::consts::PI * radius.powf(2.) );
	println!("emp circle center = {:?}", center);
	println!("circle area (emp radius) = {}", std::f32::consts::PI * emp_radius.powf(2.) );
	img.show();
}
