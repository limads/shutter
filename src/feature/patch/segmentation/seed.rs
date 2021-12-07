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

#[derive(Default, Clone, Debug)]
struct RowPair {
    curr : Vec<(u16, u16)>,
    past : Vec<(u16, u16)>
}

impl RowPair {

    fn new_empty() -> Self {
        Self {
            curr : Vec::with_capacity(32),
            past : Vec::with_capacity(32)
        }
    }

    fn new(px : (u16, u16)) -> Self {
        let (curr, mut past) = (Vec::with_capacity(32), Vec::with_capacity(32));
        past.push(px);
        Self { curr, past }
    }

    pub fn reset(&mut self, px : (u16, u16)) {
        self.curr.clear();
        self.past.clear();
        self.past.push(px);
    }

}

#[derive(Default, Clone, Debug)]
struct ExpansionFront {
    top : RowPair,
    left : RowPair,
    bottom : RowPair,
    right : RowPair
}

impl ExpansionFront {

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

    fn new(px : (u16, u16)) -> Self {
        let pair = RowPair::new(px);
        Self {
            top : pair.clone(),
            left : pair.clone(),
            bottom : pair.clone(),
            right : pair
        }
    }

    pub  fn reset(&mut self, px : (u16, u16)) {
        self.top.reset(px);
        self.left.reset(px);
        self.bottom.reset(px);
        self.right.reset(px);
    }

}

/// Convex-patch seed growing segmentation. If patch is not convex, this algorithm
/// will miss non-convex patch areas. TODO rename to SeedSegmenter
#[derive(Debug, Clone)]
pub struct SeedGrowth {
    front : ExpansionFront,
    patch : Patch,
    px_spacing : usize,
    max_area : Option<usize>,
    exp_mode : ExpansionMode
}

impl SeedGrowth {

    pub fn new(px_spacing : usize, max_area : Option<usize>, exp_mode : ExpansionMode) -> Self {
        let patch = Patch::new_empty();
        Self { front : ExpansionFront::new_empty(), px_spacing, patch, max_area, exp_mode }
    }

    pub fn grow<F>(&mut self, win : &Window<'_, u8>, seed : (u16, u16), comp : F) -> Option<&Patch>
    where
        F : Fn(u8)->bool
    {
        self.front.reset(seed);
        self.patch.pxs.clear();
        self.patch.pxs.push(seed);
        self.patch.outer_rect = (seed.0, seed.1, 1, 1);
        self.patch.color = win[seed];
        self.patch.scale = self.px_spacing as u16;
        self.patch.img_height = win.height();
        self.patch.area = 1;
        if grow(&mut self.patch, &mut self.front, &win, seed, self.px_spacing, comp, ReferenceMode::Constant, self.max_area, self.exp_mode) {
            Some(&self.patch)
        } else {
            None
        }
    }

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
    outer_rect : (u16, u16, u16, u16),
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
                patch.area = (outer_rect.2 * outer_rect.3) as usize;
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
                                patch.area += (ext.past[n-1].1 - ext.past[0].1) as usize;
                            } else {
                                patch.area += (ext.past[n-1].0 - ext.past[0].0) as usize;
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

fn update_stats(mode : &mut ColorMode, n_px : &mut u16, sum : &mut u64, sum_abs_dev : &mut u64, ref_mode : ReferenceMode, new : u8) {
    if ref_mode == ReferenceMode::Adaptive {
        *sum += new as u64;
        *n_px += 1;
        let mean = *sum / *n_px as u64;
        // let abs_dev = (mean as i64 - new as i64).abs() as u64;
        // *sum_abs_dev += abs_dev;
        mode.set_reference_color(mean as u8);
    }
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

fn pixel_neighbors_top(exp_patch : &ExpansionFront, px : (u16, u16)) -> bool {
    pixel_neighbors_last_at_row(&exp_patch.top.curr[..], px) ||
        pixel_neighbors_row(&exp_patch.top.past[..], px)
}

fn pixel_neighbors_left(exp_patch : &ExpansionFront, px : (u16, u16)) -> bool {
    pixel_neighbors_last_at_col(&exp_patch.left.curr[..], px) ||
        pixel_neighbors_col(&exp_patch.left.past[..], px)
}

fn pixel_neighbors_bottom(exp_patch : &ExpansionFront, px : (u16, u16)) -> bool {
    pixel_neighbors_last_at_row(&exp_patch.bottom.curr[..], px) ||
        pixel_neighbors_row(&exp_patch.bottom.past[..], px)
}

fn pixel_neighbors_right(exp_patch : &ExpansionFront, px : (u16, u16)) -> bool {
    pixel_neighbors_last_at_col(&exp_patch.right.curr[..], px) ||
        pixel_neighbors_col(&exp_patch.right.past[..], px)
}

/// Grows a patch from a pixel seed.
/// cf. Connected Components (Szeliski, 2011, p. 131)
fn grow<F>(
    patch : &mut Patch,
    exp_patch : &mut ExpansionFront,
    win : &Window<'_, u8>,
    seed : (u16, u16),
    px_spacing : usize,
    // mut mode : ColorMode,
    comp : F,
    ref_mode : ReferenceMode,
    max_area : Option<usize>,
    exp_mode : ExpansionMode
) -> bool
where
    F : Fn(u8)->bool
{

    // if !mode.matches(win[seed]) {
    //    println!("Seed does not match desired color");
    //    return None;
    // }
    // mode.set_reference_color(win[seed]);

    // let mut patch = Patch::new((seed.0 as u16, seed.1 as u16), win[seed], 1, win.height());
    // let seed = (seed.0 as u16, seed.1 as u16);
    // let mut exp_patch = ExpansionFront::new(seed);

    let (mut grows_left, mut grows_top, mut grows_right, mut grows_bottom) = (true, true, true, true);

    // Still need to alter pixel_neighbors_row and pixel_below_rect, etc to account for differing px_spacings.
    // Will check the absolute difference is <= px_spacing there.
    assert!(px_spacing == 1);

    let mut abs_dist = px_spacing as u16;
    let mut outer_rect = (seed.0, seed.1, 1, 1);

    let mut n_px = 0;
    let mut sum_abs_dev : u64 = 0;
    let mut sum : u64 = win[seed] as u64;

    let (h, w) = (win.height() as u16, win.width() as u16);
    loop {
        assert!(exp_patch.top.curr.is_empty() && exp_patch.left.curr.is_empty() && exp_patch.right.curr.is_empty() && exp_patch.bottom.curr.is_empty());
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
                for c in /*col_range.clone()*/ top_range.clone() {
                    let px = (r, c);
                    let is_corner = (c == top_range.start || c == top_range.end-1);
                    if comp(win[px]) && (is_corner || ( /*pixel_above_rect(&outer_rect, px) &&*/ pixel_neighbors_top(&exp_patch, px))) {
                        if !grows_top {
                            // outer_rect.0 = r;
                            // outer_rect.2 += px_spacing;
                            grows_top = true;
                        }
                        exp_patch.top.curr.push(px);
                        // update_stats(&mut mode, &mut n_px, &mut sum, &mut sum_abs_dev, ref_mode, win[px]);
                    }
                }
            }
        }

        let mut grows_left = false;
        if exp_patch.left.past.len() >= 1 {
            let left_range = (exp_patch.left.past[0].0.saturating_sub(1))..((exp_patch.left.past[exp_patch.left.past.len()-1].0 + 2).min(h));
            if let Some(c) = left_col {
                for r in /*row_range.clone().skip(1).take(row_end-1)*/ left_range.clone() {
                    let px = (r, c);
                    let is_corner = (r == left_range.start || r == left_range.end-1);
                    if comp(win[px]) && ( is_corner || ( /*pixel_to_left_of_rect(&outer_rect, px) &&*/ pixel_neighbors_left(&exp_patch, px))) {
                        if !grows_left {
                            // outer_rect.1 = c;
                            // outer_rect.3 += px_spacing;
                            grows_left = true;
                        }
                        exp_patch.left.curr.push(px);
                        // update_stats(&mut mode, &mut n_px, &mut sum, &mut sum_abs_dev, ref_mode, win[px]);
                    }
                }
            }
        }

        let mut grows_bottom = false;
        if exp_patch.bottom.past.len() >= 1 {
            let bottom_range = (exp_patch.bottom.past[0].1.saturating_sub(1))..((exp_patch.bottom.past[exp_patch.bottom.past.len()-1].1 + 2).min(w));
            if let Some(r) = bottom_row {
                for c in /*col_range*/ bottom_range.clone() {
                    let px = (r, c);
                    let is_corner = (r == bottom_range.start || r == bottom_range.end-1);
                    if comp(win[px]) && (is_corner || ( /*pixel_below_rect(&outer_rect, px) &&*/ pixel_neighbors_bottom(&exp_patch, px))) {
                        if !grows_bottom {
                            // outer_rect.2 += px_spacing;
                            grows_bottom = true;
                        }
                        exp_patch.bottom.curr.push(px);
                        // update_stats(&mut mode, &mut n_px, &mut sum, &mut sum_abs_dev, ref_mode, win[px]);
                    }
                }
            }
        }

        let mut grows_right = false;
        if exp_patch.right.past.len() >= 1 {
            let right_range = (exp_patch.right.past[0].0.saturating_sub(1))..((exp_patch.right.past[exp_patch.right.past.len()-1].0 + 2).min(h));
            if let Some(c) = right_col {
                for r in /*row_range.skip(1).take(row_end-1)*/ right_range.clone() {
                    let px = (r, c);
                    let is_corner = (c == right_range.start || c == right_range.end-1);
                    if comp(win[px]) && (is_corner || ( /*pixel_to_right_of_rect(&outer_rect, px) &&*/ pixel_neighbors_right(&exp_patch, px))) {
                        if !grows_right {
                            // outer_rect.3 += px_spacing;
                            grows_right = true;
                        }
                        exp_patch.right.curr.push(px);
                        // update_stats(&mut mode, &mut n_px, &mut sum, &mut sum_abs_dev, ref_mode, win[px]);
                    }
                }
            }
        }

        let grows_any = grows_left || grows_right || grows_bottom || grows_top;
        expand_rect_with_front(&mut outer_rect, &exp_patch);
        patch.outer_rect = outer_rect;
        expand_patch(exp_patch, patch, outer_rect, exp_mode, win);

        if let Some(area) = max_area {
            if /*patch.pxs.len()*/ outer_rect.2 * outer_rect.3 > area as u16 {
                return false;
            }
        }

        if !grows_any {
            if exp_mode == ExpansionMode::Contour {

                // Remove seed before closing patch
                patch.pxs.swap_remove(0);

                // A contour should have at least three pixel points. A dense
                // patch can have any number of points >= 1.
                if patch.pxs.len() < 3 {
                    return false;
                }

                close_contour(patch, win);
            }
            // return Some(patch);
            return true;
        }

        abs_dist += px_spacing as u16;
    }

    assert!(patch.pxs.len() > 0);
    // Some(patch)
    true
}


