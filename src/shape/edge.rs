// use crate::threshold::Threshold;
// use crate::image::Window;
// use crate::image::iter;
use crate::shape::Polygon;
use std::default::Default;
use crate::image::*;
// use crate::raster::*;
// use crate::draw::*;

#[derive(Debug, Copy, Clone)]
pub struct Threshold {
    pub min : u8,
    pub max : u8
}

impl Default for Threshold {

    fn default() -> Self {
        Self { min : 0, max : 255 }
    }

}


#[derive(Debug, Clone, Copy, Default)]
pub struct EdgeThreshold {
    pub lower : Threshold,
    pub upper : Threshold
}

pub fn is_any_unbounded_transition(
    max : &mut (usize, i16),
    comp_dist : usize,
    a : &u8,
    b : &u8,
    i : usize
) {
    let diff = (*a as i16 - *b as i16).abs();
    if diff > max.1 {
        *max = (i+(comp_dist/2), diff);
    }
}

pub fn is_white_dark_unbounded_transition(
    max : &mut (usize, i16),
    comp_dist : usize,
    a : &u8,
    b : &u8,
    i : usize
) {
    if *a > *b {
        let diff = (*a as i16 - *b as i16).abs();
        if diff > max.1 {
            *max = (i+(comp_dist/2), diff);
        }
    }
}

pub fn is_dark_white_unbounded_transition(
    max : &mut (usize, i16),
    comp_dist : usize,
    a : &u8,
    b : &u8,
    i : usize
) {
    if *a < *b {
        let diff = (*a as i16 - *b as i16).abs();
        if diff > max.1 {
            *max = (i+(comp_dist/2), diff);
        }
    }
}

/// Verify is bit levels a->b represent a white->dark transition
pub fn is_white_dark_bounded_transition(
    max_left : &mut (usize, i16),
    found : &mut bool,
    comp_dist : usize,
    a : &u8,
    b : &u8,
    i : usize,
    thr : &EdgeThreshold
) {
    let diff = (*a as i16 - *b as i16).abs();
    let a_match_up = *a > thr.upper.min && *a < thr.upper.max;
    let b_match_low = *b > thr.lower.min && *b < thr.lower.max;
    if a_match_up && b_match_low && diff > max_left.1 {
        *max_left = (i+(comp_dist/2), diff);
        *found = true;
    }
}

pub fn is_white_dark_diagonal_bounded_transition(
    max : &mut ((usize, usize), i16),
    found : &mut bool,
    to_right : bool,
    comp_dist : usize,
    a : &u8,
    b : &u8,
    thr : &EdgeThreshold
) {
    let diff = (*a as i16 - *b as i16).abs();
    let a_match_up = *a > thr.upper.min && *a < thr.upper.max;
    let b_match_low = *b > thr.lower.min && *b < thr.lower.max;
    if a_match_up && b_match_low && diff > max.1 {
        let y = max.0.0+(comp_dist/2);
        let x = if to_right {
            max.0.1+(comp_dist/2)
        } else {
            max.0.1.saturating_sub(comp_dist/2)
        };
        *max = ((y, x), diff);
        *found = true;
    }
}

pub fn is_dark_white_diagonal_bounded_transition(
    max : &mut ((usize, usize), i16),
    found : &mut bool,
    to_right : bool,
    comp_dist : usize,
    a : &u8,
    b : &u8,
    thr : &EdgeThreshold
) {
    let diff = (*a as i16 - *b as i16).abs();
    let a_match_low = *a > thr.lower.min && *a < thr.lower.max;
    let b_match_up =  *b > thr.upper.min && *b < thr.upper.max;
    if a_match_low && b_match_up && diff > max.1 {
        let y = max.0.0+(comp_dist/2);
        let x = if to_right {
            max.0.1+(comp_dist/2)
        } else {
            max.0.1.saturating_sub(comp_dist/2)
        };
        *max = ((y, x), diff);
        *found = true;
    }
}

pub fn diagonal_bounded_pair<'a>(
    diag_iter : impl Iterator<Item=((usize, usize), (&'a u8, &'a u8))>,
    to_right : bool,
    edge_thresh : &EdgeThreshold
) -> (Option<(usize, usize)>, Option<(usize, usize)>) {
    let (mut white_dark, mut dark_white) = (((0, 0), 0), ((0, 0), 0));
    let (mut white_dark_found, mut dark_white_found) = (false, false);
    diag_iter.for_each(|(ix, (a, b))| {
        is_white_dark_diagonal_bounded_transition(
            &mut white_dark,
            &mut white_dark_found,
            to_right,
            4,
            a,
            b,
            &edge_thresh
        );
        is_dark_white_diagonal_bounded_transition(
            &mut dark_white,
            &mut dark_white_found,
            to_right,
            4,
            a,
            b,
            &edge_thresh
        );
    });
    let white_dark = if white_dark_found {
        Some(white_dark.0)
    } else {
        None
    };
    let dark_white = if dark_white_found {
        Some(dark_white.0)
    } else {
        None
    };
    (white_dark, dark_white)
}

/// Verify is bit levels a->b represent a dark->white transition
pub fn is_dark_white_bounded_transition(
    max_right : &mut (usize, i16),
    found : &mut bool,
    comp_dist : usize,
    a : &u8,
    b : &u8,
    i : usize,
    thr : &EdgeThreshold
) {
    let diff = (*a as i16 - *b as i16).abs();
    let a_match_low = *a > thr.lower.min && *a < thr.lower.max;
    let b_match_up =  *b > thr.upper.min && *b < thr.upper.max;
    if a_match_low && b_match_up && diff > max_right.1 {
        *max_right = (i+(comp_dist/2), diff);
        *found = true;
    }
}

#[derive(Debug)]
pub enum ContourError {
    RasterSmall,
    InvalidThreshold,
    LowMissing,
    HighMissing,
    Direction,
    Symmetry
}

/// Returns the pair of column indices with the maximum negative then positive difference, as long as
/// the absolute value of the difference is above the informed threshold.
pub fn horizontal_max_pair_diff_index(
    row : &[u8],
    comp_dist : usize,
    edge_thr : &EdgeThreshold,
    row_ix : usize,
    frame : usize
) -> Result<(usize, usize), ContourError> {

    if row.len() < 4 {
        return Err(ContourError::RasterSmall);
    }

    // Stores the position of the difference and the difference. max_left starts at zero
    // and tends to get bigger (white-dark); max_right starts at zero but tends to
    // get smaller (dark-white).
    let mut max_left : (usize, i16) = (0, 0);
    let mut max_right : (usize, i16) = (0, 0);

    let (mut low_found, mut high_found) = (false, false);
    for (i, (a, b)) in horizontal_row_iterator(row, comp_dist) {
        is_white_dark_bounded_transition(&mut max_left, &mut low_found, comp_dist, a, b, i, &edge_thr);
        is_dark_white_bounded_transition(&mut max_right, &mut high_found, comp_dist, a, b, i, &edge_thr);
    }

    if !low_found {
        return Err(ContourError::LowMissing);
    }

    if !high_found {
        return Err(ContourError::HighMissing);
    }

    Ok((max_left.0, max_right.0))

}

pub fn horizontal_max_diff_index<F>(
    row : &[u8],
    comp_dist : usize,
    edge_thr : &EdgeThreshold,
    row_ix : usize,
    frame : usize,
    transition : F
) -> Option<usize>
where
    F : Fn(&mut (usize, i16),&mut bool,usize,&u8,&u8,usize,&EdgeThreshold)
{
    let mut max_right : (usize, i16) = (0, 0);
    if row.len() < 4 {
        // return Err(ContourError::RasterSmall);
        return None;
    }
    let mut found = false;
    // let upper_max_thr = upper_thr.saturating_add(intensity_tol);
    // let lower_max_thr = lower_thr.saturating_add(intensity_tol);
    for (i, (a, b)) in horizontal_row_iterator(row, comp_dist) {
        //is_dark_white_bounded_transition(&mut max_right, &mut found, a, b, i, &iris_thr);
        transition(&mut max_right, &mut found, comp_dist, a, b, i, &edge_thr)
    }
    if found {
        Some(max_right.0)
    } else {
        None
    }
}

/// Returns the (lower) row with maximum diff index for the informed column
pub fn vertical_max_diff_index<F>(
    win : &Window<'_, u8>,
    comp_dist : usize,
    edge_thr : &EdgeThreshold,
    col_ix : usize,
    transition : F
) -> Option<usize>
where
    F : Fn(&mut (usize, i16),&mut bool,usize,&u8,&u8,usize,&EdgeThreshold)
{

    // Holds (row, col, value)
    let mut max : (usize, i16) = (0, 0);
    //if row.len() < 4 {
    //    return Err(ContourError::RasterSmall);
    // }
    let mut found = false;
    for (i, (a, b)) in vertical_row_iterator(win.rows(), comp_dist, col_ix) {
        // is_dark_white_bounded_transition(&mut max, &mut found, a, b, i, &iris_thr);
        transition(&mut max, &mut found, comp_dist, a, b, i, &edge_thr)
    }

    if found {
        Some(max.0)
    } else {
        None
    }
}


