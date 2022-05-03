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
// use away::space::SpatialClustering;
use std::ops::Range;
use std::convert::TryInto;
use std::convert::TryFrom;
use std::ops::Div;
use std::ops::Add;
use bayes;
use serde::{Serialize, Deserialize};
use std::borrow::Cow;

/*pub trait  PixelCompare<T>
where
    Self : Fn(T)-> bool
{

    fn px_cmp(T) -> bool;

}

// Compare using current value
impl PixelCompare for fn(u8)->bool { }

// Compare using current value and immediate 1d neighbor
impl PixelCompare for fn((u8, u8))->bool { }

// Compare using current value and immediate 4-neighborhood
impl PixelCompare for fn((u8, [u8; 4]))->bool { }

// Compare using current value and immediate 8-neighborhood
impl PixelCompare for fn((u8, [u8; 8]))->bool { }*/

/*
(Klette & Rosenfeld (2004) For a binary image, a pair of pixels can be connected at the 4-neighborhood
or 8-neighborhood. A set of connected pixel pairs where each pair is connected to
at least one other defines a region, and can be conceptualized as an undirected
graph with pixels as the nodes and their connectedness as the edges. The RasterSegmenter
builds those graphs via the 4-neighborhood, the SeedSegmenter via the 8-neighborhood. RasterSegmenter
can be used to build many graphs from the same image in a single iteration, where each separate group
is a separate graph with the criteria "is this color" or "is not this color", where the colors are
defined as the iteration happens given a separation criteria.
*/

/*IppStatus ippiMarkSpeckles_<mod>(Ipp<datatype>* pSrcDst, int srcDstStep, IppiSize
roiSize, Ipp<datatype> speckleVal, int maxSpeckleSize, Ipp<datatype> maxPixDiff,
IppiNorm norm, Ipp8u* pBuffer );*/

/*IppStatus ippiSegmentWatershed_8u_C1IR(const Ipp8u* pSrc, int srcStep, Ipp8u* pMarker,
int markerStep, IppiSize roiSize, IppiNorm norm, int flag, Ipp8u* pBuffer );*/

/*IppStatus ippiSegmentGradient_8u_C1IR(const Ipp8u* pSrc, int srcStep, Ipp8u* pMarker,
int markerStep, IppiSize roiSize, IppiNorm norm, int flags, Ipp8u* pBuffer );*/

pub mod seed;

pub mod raster;

pub mod density;

pub mod ray;

pub mod pattern;

pub mod edge;

// use nohash-hasher;

// use crate::feature::shape;

// #[cfg(feature="opencvlib")]
// pub mod fgmm;

// #[cfg(feature="opencvlib")]
// pub mod mser;

pub fn intensity_below<const P : u8>(px : u8) -> bool {
    px <= P
}

pub fn intensity_above<const P : u8>(px : u8) -> bool {
    px >= P
}

pub struct ContourNeighborhood<'a> {

    begin : &'a (u16, u16),

    middle : &'a (u16, u16),

    end : &'a (u16, u16)

}

pub enum NeighborhoodType {
    Vertical,
    Horizontal,
    Diagonal,
    Generic
}

impl<'a> ContourNeighborhood<'a> {

    pub fn neighborhood_type(&self) -> NeighborhoodType {
        let vert_dist_begin = (self.middle.0 as i32 - self.begin.0 as i32).abs();
        let vert_dist_end =  (self.middle.0 as i32 - self.end.0 as i32).abs();
        let horiz_dist_begin = (self.middle.1 as i32 - self.begin.1 as i32).abs();
        let horiz_dist_end =  (self.middle.1 as i32 - self.end.1 as i32).abs();

        let vert_dist = vert_dist_begin + vert_dist_end;
        let horiz_dist = horiz_dist_begin + horiz_dist_end;

        if vert_dist == horiz_dist {
            return NeighborhoodType::Diagonal;
        } else {
            if vert_dist > 0 && horiz_dist == 0 {
                return NeighborhoodType::Vertical;
            } else {
                if vert_dist == 0 && horiz_dist > 0 {
                    return NeighborhoodType::Horizontal;
                }
            }
        }
        NeighborhoodType::Generic
    }

    pub fn below_seed(&self, seed : (u16, u16)) -> bool {
        self.middle.0 > seed.0 && self.begin.0 > seed.0 && self.end.0 > seed.0
    }

    pub fn above_seed(&self, seed : (u16, u16)) -> bool {
        self.middle.0 < seed.0 && self.begin.0 < seed.0 && self.end.0 < seed.0
    }

    pub fn to_right_of_seed(&self, seed : (u16, u16)) -> bool {
        self.middle.1 > seed.1 && self.begin.1 > seed.1 && self.end.1 > seed.1
    }

    pub fn to_left_of_seed(&self, seed : (u16, u16)) -> bool {
        self.middle.1 < seed.1 && self.begin.1 < seed.1 && self.end.1 < seed.1
    }

    pub fn strictly_below_seed(&self, seed : (u16, u16)) -> Option<bool> {
        match (self.below_seed(seed), self.above_seed(seed)) {
            (true, false) => Some(true),
            (false, true) => Some(false),
            _ => None
        }
    }

    pub fn strictly_to_right_of_seed(&self, seed : (u16, u16)) -> Option<bool> {
        match (self.to_left_of_seed(seed), self.to_right_of_seed(seed)) {
            (false, true) => Some(true),
            (true, false) => Some(false),
            _ => None
        }
    }

    pub fn closest_outside_neighbor(&self, win : &Window<'_, u8>, seed : (u16, u16)) -> Option<(u16, u16)> {
        let out = self.closest_outside_neighbor_unchecked(seed)?;
        if out.0 < win.height() as u16 && out.1 < win.width() as u16 {
            Some(out)
        } else {
            None
        }
    }

    pub fn closest_outside_neighbor_unchecked(&self, seed : (u16, u16)) -> Option<(u16, u16)> {
        let vert_incr = if self.strictly_below_seed(seed)? { 1 } else { -1 };
        let horiz_incr = if self.strictly_to_right_of_seed(seed)? { 1 } else { -1 };
        match self.neighborhood_type() {
            NeighborhoodType::Vertical => Some((self.middle.0, u16::try_from(self.middle.1 as i32 + horiz_incr).ok()?)),
            NeighborhoodType::Horizontal => Some((u16::try_from(self.middle.0 as i32 + vert_incr).ok()?, self.middle.1)),
            NeighborhoodType::Diagonal => {
                Some((u16::try_from(self.middle.0 as i32 + vert_incr).ok()?, u16::try_from(self.middle.1 as i32 + horiz_incr).ok()?))
            },
            NeighborhoodType::Generic => {
                let cand1 = [(0, horiz_incr), (vert_incr, 0), (horiz_incr, vert_incr)];
                let cand2 = candidate_increment::<2>(vert_incr, horiz_incr);
                let cand3 = candidate_increment::<3>(vert_incr, horiz_incr);
                let cand4 = candidate_increment::<4>(vert_incr, horiz_incr);
                let candidates = cand1.iter()
                    .chain(cand2.iter())
                    .chain(cand3.iter())
                    .chain(cand4.iter());

                let begin_dist = shape::point_euclidian_u16(*self.begin, seed);
                let end_dist = shape::point_euclidian_u16(*self.end, seed);
                let middle_dist = shape::point_euclidian_u16(*self.middle, seed);
                for cand in candidates{
                    if let (Ok(y), Ok(x)) = (u16::try_from(self.middle.0 as i32 + cand.0), u16::try_from(self.middle.1 as i32 + cand.1)) {
                        if (y, x) != *self.begin && (y, x) != *self.end && (y, x) != *self.middle {
                            let dist = shape::point_euclidian_u16((y, x), seed);
                            if dist > begin_dist && dist > end_dist && dist > middle_dist {
                                return Some((y, x));
                            }
                        }
                    }
                }
                None
            }
        }
    }

}

fn candidate_increment<const I : i32>(vert_incr : i32, horiz_incr : i32) -> [(i32, i32); 3] {
    [(I*vert_incr, horiz_incr), (vert_incr, I*horiz_incr), (I*vert_incr, I*horiz_incr)]
}

/// The most general patch is a set of pixel positions with a homogeneous color
/// and a scale that was used for extraction. The patch is assumed to be
/// homonegeneous within a pixel spacing given by the scale field. TODO if the
/// patch is dense (has no holes in it), we can represent it with another
/// structure called contour, holding only external boundray pixels. If the patch
/// has holes, we must represent all pixels with this patch structure.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Patch {
    // Instead of storing all pixels, we can store a rect and only
    // the outer pixels. The rect is just characterized by a top-left
    // corner and an extension. The rect extension is increased
    // any time we have a set of inserted pixels that comptetely border
    // either its bottom or right borders.
    pub pxs : Vec<(u16, u16)>,

    // Outer rect, at patch scale
    pub outer_rect : (u16, u16, u16, u16),
    pub color : u8,
    pub scale : u16,
    pub img_height : usize,

    // Number of pixels inside it.
    area : usize
}

/*impl deft::Interactive for Patch {

    #[export_name="register_Patch"]
    extern "C" fn interactive() -> Box<deft::TypeInfo> {
        deft::TypeInfo::builder::<Self>()
            .initializable()
            .fallible("outer_rect", |s : &mut Self| -> deft::ReplResult<rhai::Array> {
                let out = s.outer_rect::<i64>();
                Ok(vec![
                    rhai::Dynamic::from(out.0),
                    rhai::Dynamic::from(out.1),
                    rhai::Dynamic::from(out.2),
                    rhai::Dynamic::from(out.3)
                ])
            })
            .fallible("contour_points", |s : &mut Self| -> deft::ReplResult<rhai::Array> {
                let mut pts = Vec::new();
                for pt in s.outer_points::<i64>(ExpansionMode::Contour) {
                    pts.push(rhai::Dynamic::from(vec![
                        rhai::Dynamic::from(pt.0 as i64),
                        rhai::Dynamic::from(pt.1 as i64)
                    ]));
                }
                Ok(pts)
            })
            .fallible("dense_points", |s : &mut Self| -> deft::ReplResult<rhai::Array> {
                let mut pts = Vec::new();
                for pt in s.outer_points::<i64>(ExpansionMode::Dense) {
                    pts.push(rhai::Dynamic::from(vec![
                        rhai::Dynamic::from(pt.0 as i64),
                        rhai::Dynamic::from(pt.1 as i64)
                    ]));
                }
                Ok(pts)
            })
            .parseable()
            .build()
    }

}*/

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
pub fn approx_color_momentum(win : &Window<'_, u8>, px_spacing : usize, mut mode : ColorMode) -> Option<(usize, usize)> {
    let (mut sum_r, mut sum_c) = (0.0, 0.0);
    let mut n_matches = 0;
    for (r, c, color) in win.labeled_pixels::<usize, _>(px_spacing) {
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
pub fn hashmap_color_momentum(win : &Window<'_, u8>, px_spacing : usize, mut mode : ColorMode) -> Option<(usize, usize)> {
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

pub fn color_momentum(win : &Window<'_, u8>, px_spacing : usize, mut mode : ColorMode) -> Option<(usize, usize)> {

    let mut n_matches = 0;

    // Which rows of the image matched color at least once
    let mut row_ixs : Vec<usize> = Vec::new();

    // Stores the columns of all pixels that matched color
    let mut col_ixs : Vec<usize> = Vec::new();

    // Defines a contiguous index range over col_ixs linked to the same row.
    let mut row_ranges : Vec<Range<usize>> = Vec::new();

    for (r, c, color) in win.labeled_pixels(px_spacing) {
        if mode.matches(color) {
            if row_ixs.last().cloned() == Some(r) {
                col_ixs.push(c);
                row_ranges.last_mut().unwrap().end += 1;
            } else {
                row_ixs.push(r);
                col_ixs.push(c);
                let n = col_ixs.len();
                row_ranges.push(Range { start : n - 1, end : n });
            }
        }
    }
    assert!(row_ixs.len() == row_ranges.len());
    let nrows = row_ixs.len() as f32;

    let mut avg_col = 0.;
    let mut avg_row = 0.;
    for (row_ix, row_range) in row_ixs.drain(..).zip(row_ranges.drain(..)) {
        avg_row += row_ix as f32;
        let ncols = row_range.end - row_range.start;
        let sum_col_this_row = col_ixs[row_range]
            .iter()
            .fold(0, |sum_c, c| sum_c + c );
        avg_col += sum_col_this_row as f32 / ncols as f32;
    }
    avg_col /= nrows;
    avg_row /= nrows;

    Some((avg_row as usize, avg_col as usize))
}

/*/// If any pixels in the neighborhood of the contour do not differ by the neighboring
/// pixel by more than the diff threshold, aggregate the pixel into the contour. ext_step
/// is the angle tau used to extend the convex triplets of edge pixels.
fn smooth_contour(contour : &mut Patch, win : &Window<'_, u8>, max_diff : u8, max_extension : usize, tau_step : f64) {
    let n = contour.pxs.len();
    let triplet_iter = contour.pxs.iter().take(n-2)
        .zip(contour.pxs.iter().skip(1).take(n-1).iter().zip(contour.pxs.iter().skip(2)));
    assert!(tau_step > 0.0 && tau_step < 2. * f64::consts::PI);
    for (ext1, (center, ext2)) in triplet_iter {
        let theta_1 = shape::vertex_angle(ext1, center);
        let theta_2 = shape::vertex_angle(ext2, center);
        let theta_3 = f64::consts::PI - (theta_1 + theta_2);

        // This guarantees the extended point will be at the same normal away from the ext_1, ext_2 points
        // passing through the center.
        let new_theta_1 = theta_1 + tau_step;
        let new_theta_2 = theta_2 + tau_step;
        let new_theta_3 = theta_3 - 2.*tau_step;

        // This is the angle at the point center formed by the new triangle
        // between the old center point, the new center point
        // and one of the extremities (the angle will be the same for both sides).
        // This new triangle will have angles tau at the extremity, half_theta3_compl
        // at the corner, and new_theta_3 / 2.
        let half_theta3_compl = (2. * PI - theta_3) / 2.;

        // If extened point does not differ from color by more than max_diff,
        // substitute the inner point by the extended point.
    }
}*/

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
    mut mode : ColorMode,
    strategy : Strategy
) -> Option<(usize, usize)> {
    let mom = hashmap_color_momentum(win, px_spacing, mode)?;

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
    println!("{:?}", hashmap_color_momentum(&img.full_window(), 1, ColorMode::Exact(1)));
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

    // println!("{:?}", Patch::grow(&img.full_window(), (11, 11), 1, ColorMode::Exact(1), ReferenceMode::Constant, None, ExpansionMode::Contour));
}

// Extract a single patch, using the color momentum as seed.
// pub fn extract_main_patch(win : &Window<'_, u8>, px_spacing : u16, mode : ColorMode) -> Option<Patch> {
//    let seed = hashmap_color_momentum(win, px_spacing, mode)?;
//    Some(Patch::grow(win, seed, 1, mode, ReferenceMode::Constant, None, ExpansionMode::Dense)).unwrap()
// }

fn pixel_horizontally_aligned_to_rect(outer_rect : &(u16, u16, u16, u16), (r, c) : (u16, u16)) -> bool {
    r >= outer_rect.0 /*.saturating_sub(1)*/ &&
        r <= (outer_rect.0 + outer_rect.2) /*.saturating_add(1)*/
}

fn pixel_vertically_aligned_to_rect(outer_rect : &(u16, u16, u16, u16), (r, c) : (u16, u16)) -> bool {
    c >= outer_rect.1 /*.saturating_sub(1)*/  &&
        c <= (outer_rect.1 + outer_rect.3) /*.saturating_add(1)*/
}

fn pixel_to_right_of_rect(outer_rect : &(u16, u16, u16, u16), (r, c) : (u16, u16)) -> bool {
    pixel_horizontally_aligned_to_rect(outer_rect, (r, c))
}

fn pixel_to_left_of_rect(outer_rect : &(u16, u16, u16, u16), (r, c) : (u16, u16)) -> bool {
    pixel_horizontally_aligned_to_rect(outer_rect, (r, c))
}

fn pixel_above_rect(outer_rect : &(u16, u16, u16, u16), (r, c) : (u16, u16)) -> bool {
    pixel_vertically_aligned_to_rect(outer_rect, (r, c))
}

fn pixel_below_rect(outer_rect : &(u16, u16, u16, u16), (r, c) : (u16, u16)) -> bool {
    pixel_horizontally_aligned_to_rect(outer_rect, (r, c))
}

/// A pixel neighbor is a pixel with distance 1 to another. Searches the set at row/col
/// for a pixel with this distance, assuming the row is ordered by column
fn pixel_neighbors_row(row : &[(u16, u16)], px : (u16, u16)) -> bool {
    row.binary_search_by(|row_px|
        if (row_px.1 as i16 - px.1 as i16).abs() <= 1 {
            Ordering::Equal
        } else {
            row_px.1.cmp(&px.1)
        }
    ).is_ok()
}

/// See docs for pixel_neighbors_row. The same logic applies here.
fn pixel_neighbors_col(col : &[(u16, u16)], px : (u16, u16)) -> bool {
    col.binary_search_by(|col_px|
        // if (col_px.0 as i16 - px.0 as i16).abs() <= 1 {
        //    Ordering::Equal
        //} else {
        col_px.0.cmp(&px.0)
        // }
    ).is_ok()
}

fn pixel_neighbors_last_at_row(row : &[(u16, u16)], px : (u16, u16)) -> bool {
    row.last().map(|last| (last.1 as i16 - px.1 as i16).abs() <= 1 ).unwrap_or(false)
}

fn pixel_neighbors_last_at_col(col : &[(u16, u16)], px : (u16, u16)) -> bool {
    col.last().map(|last| (last.0 as i16 - px.0 as i16).abs() <= 1 ).unwrap_or(false)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpansionMode {
    Rect,
    Contour,
    Dense
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

    assert!(patch.pxs.len() >= 3);

    // Order by pairwise closeness.
    let h = u16::try_from(win.height()).unwrap();
    for ix in 1..patch.pxs.len() {
        let (closest_ix, _) = patch.pxs.iter().enumerate().skip(ix)
            .min_by(|(_, px1), (_, px2)| {
                index_distance(**px1, patch.pxs[ix-1], h).0
                    .partial_cmp(&index_distance(**px2, patch.pxs[ix-1], h).0)
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

#[derive(Debug, Clone, Copy)]
pub struct PatchBorder {
    inner : u8,
    outer : u8
}

impl Patch {

    /*/// This assumes the contour is ordered and complete (there is one per row) and convex (there aren't holes or openings).
    pub fn alt_inner_pixels(
        &'a self,
        win : &'a Window<'_, u8>,
        px_spacing : usize
    ) -> impl Iterator<Item=((u16, u16), u8)> + 'a {
        let mut last_left_ix = 0;
        let mut last_right_ix = 0;

        let mut rows = Vec::new();
        let mut r = self.outer_rect.0;

        while r < self.outer_rect.0 + self.outer_rect.2 {
            if let Some(pos1) = self.pxs.position(|px| px.0 == r ) {

                let left_slice = self.pxs[..pos1];
                let right_slice = self.pxs[(pos1+1)..];
                let opt_pos_2 = left_slice.position(|px| px.0 == r )
                    .or_else(|| right_slice.position(|px| px.0 == r ).map(|p| p + pos1+1) );

                // First pair of rows has been found.
                if let Some(pos2) = opt_pos_2 {
                    if pxs[pos1].1 < pxs[pos2].1 {
                        last_left_ix = pos1;
                        last_right_ix = pos2;
                    } else {
                        last_left_ix = pos2;
                        last_right_ix = pos1;
                    }

                    rows.push((pxs[last_left_ix].1..´pxs[last_right_ix].1).map(move |c| ((row, c), win[(row, c)]) ));
                }
            }
        }

        rows.into_iter().flatten()
    }*/

    // This works for a boundary too.
    pub fn inner_pixels<'a>(
        &'a self,
        win : &'a Window<'_, u8>,
        px_spacing : u16
    ) -> impl Iterator<Item=((u16, u16), u8)> + 'a {
        let mut pxs = self.pxs.clone();
        pxs.sort_by(|a, b| a.0.cmp(&b.0) );

        let mut rows = Vec::new();
        let mut inner = Vec::new();

        let mut last_row = None;
        for (row, row_px) in &pxs.into_iter().group_by(|px| px.0 ) {

            if let Some(last) = last_row {
                if row - last < px_spacing {
                    continue;
                }
            }

            // Sort by cols within rows
            inner.clear();
            inner.extend(row_px);
            inner.sort_by(|a, b| a.1.cmp(&b.1) );

            // Pairs of pixels within row (now sorted by col) must be contiguous in the patch.
            for (p1, p2) in inner.iter().clone().zip(inner.iter().skip(1)) {
                rows.push((p1.1..p2.1).step_by(px_spacing as usize).map(move |c| ((row, c), win[(row, c)]) ));
                last_row = Some(row as u16);
            }
        }

        rows.into_iter().flatten()
    }

    pub fn rect_border_color(&self, win : &Window<'_, u8>) -> Option<u8> {
        let mut out : u64 = 0;
        let mut n : u64 = 0;
        for px in win.rect_pixels(self.outer_rect::<usize>()) {
            out += px as u64;
            n += 1;
        }
        if n >= 1 {
            Some((out / n) as u8)
        } else {
            None
        }
    }

    pub fn border_color(&self, seed : (u16, u16), win : &Window<'_, u8>) -> Option<PatchBorder> {
        let mut inner : u64 = 0;
        let mut outer : u64 = 0;
        let mut n : u64 = 0;
        for (center_ix, neigh) in self.neighborhoods() {
            if let Some(out) = neigh.closest_outside_neighbor(&win, seed) {
                inner += win[self.pxs[center_ix]] as u64;
                outer += win[out] as u64;
                n += 1;
            }
        }
        if n >= 1 {
            Some(PatchBorder { inner : (inner / n) as u8, outer : (outer / n) as u8 })
        } else {
            None
        }
    }

    /// Expands this patch if its color is closer to the average color border than the
    /// outer color (if given) or the outer color calculated by taking the closest
    /// outer pixels to the border (if outer is not given).
    pub fn expand_to_closest_color(&mut self, win : Window<'_, u8>, seed : (u16, u16), outer : Option<u8>) -> usize {
        if let Some(color) = self.border_color(seed, &win) {
            let mut changes = Vec::new();
            for (ix, neigh) in self.neighborhoods() {
                if let Some(out) = neigh.closest_outside_neighbor(&win, seed) {
                    if out.0 < win.width() as u16 && out.1 < win.height() as u16 {
                        // let old_middle = (self.pxs[ix].0 as usize, self.pxs[ix].1 as usize);
                        let new_cand_middle = (out.0 as usize, out.1 as usize);
                        let diff_outer = (win[new_cand_middle] as i16 - outer.unwrap_or(color.outer) as i16).abs();
                        let diff_inner = (win[new_cand_middle] as i16 - color.inner as i16).abs();
                        if diff_inner < diff_outer {
                            changes.push((ix, out));
                        }
                    }
                }
            }
            self.expand_with_changes(changes)
        } else {
            0
        }
    }

    pub fn radial_expansion(
        &mut self,
        win : &Window<'_, u8>,
        px_tol : u8,
        adaptive : bool,
        seed : (u16, u16),
        limit : usize,
        min_angle : f32,
        max_angle : f32
    ) -> usize {
        let mut changes = Vec::new();
        for (ix, px) in self.pxs.iter().enumerate() {

            let dist = crate::feature::shape::point_euclidian_u16(*px, seed);

            // Automatically invert y axis
            let coord = (seed.0 as f32 - px.0 as f32, px.1 as f32 - seed.1 as f32);

            let theta = coord.0.atan2(coord.1);

            if theta < min_angle || theta > max_angle {
                continue;
            }

            let cos_theta = theta.cos();
            let sin_theta = theta.sin();
            let mut expand = 1;
            let mut old_coord = (px.0 as usize, px.1 as usize);
            loop {
                let new_coord = ((seed.0 as f32 - (sin_theta*(dist + expand as f32 + 1.))).ceil(), (seed.1 as f32 + (cos_theta*(dist + expand as f32 + 1.))).ceil());
                if new_coord.0 > 0.0 && new_coord.0 < (win.height() as f32 - 1.) && new_coord.1 > 0.0 && new_coord.1 < (win.width() as f32 - 1.) {
                    let new_coord_u = (new_coord.0 as usize, new_coord.1 as usize);
                    let is_match = if adaptive {
                        let matched = ((win[new_coord_u] as i16 - win[old_coord] as i16).abs() as u8) < px_tol;
                        old_coord = new_coord_u;
                        matched
                    } else {
                        let matched = ((win[new_coord_u] as i16 - win[*px] as i16).abs() as u8) < px_tol;
                        matched
                    };
                    if is_match {
                        if expand == 1 {
                            changes.push((ix, (new_coord_u.0 as u16, new_coord_u.1 as u16)));
                        } else {
                            *changes.last_mut().unwrap() = ((ix, (new_coord_u.0 as u16, new_coord_u.1 as u16)));
                        }
                        expand += 1;
                        if expand >= limit {
                            break;
                        }
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
        }
        self.expand_with_changes(changes)
    }

    /// Iterates over pixel index of the middle element and its local neighborhood.
    pub fn neighborhoods<'a>(&'a self) -> Vec<(usize, ContourNeighborhood<'a>)> {
        assert!(self.scale == 1);
        let mut neighs = Vec::new();
        for ix in 1..(self.pxs.len() - 1) {
            neighs.push((ix, ContourNeighborhood { begin : &self.pxs[ix-1], middle : &self.pxs[ix], end : &self.pxs[ix+1] }));
        }
        neighs
    }

    /*pub fn expand_to_similar(&mut self, seed : (u16, u16), px_tol : u8, win : Window<'_, u8>) -> usize {
        assert!(self.scale == 1);
        let mut changes = Vec::new();
        for (ix, neigh) in self.neighborhoods() {
            if let Some(out) = neigh.closest_outside_neighbor(&win, seed) {
                if out.0 < win.width() as u16 && out.1 < win.height() as u16 {
                    let old_middle = (self.pxs[ix].0 as usize, self.pxs[ix].1 as usize);
                    let mut new_cand_middle = (out.0 as usize, out.1 as usize);

                    let dy = (out.0 as i16 - self.pxs[ix].0 as i16).signum();
                    let dx = (out.1 as i16 - self.pxs[ix].1 as i16).signum();

                    let mut n_expanded = 0;
                    while (win[old_middle] as i16 - win[new_cand_middle] as i16).abs() as u8 <= px_tol {
                        if n_expanded == 0 {
                            changes.push((ix, out));
                        } else {
                            *changes.last_mut().unwrap() = (ix, out);
                        }
                        n_expanded += 1;
                        let new_y = out.0 as i16 + dy;
                        let new_x = out.1 as i16 + dx;
                        if new_y > 0 && new_y < win.height() && new_x > 0 && new_x < win.width() {
                            new_cand_middle.0 = new_y as usize;
                            new_cand_middle.1 = new_x as usize;
                            out = (new_y as u16, new_x as u16);
                        } else {
                            break;
                        }

                    }
                }
            }
        }

        self.expand_with_changes(changes)
    }*/

    fn expand_with_changes(&mut self, changes : Vec<(usize, (u16, u16))>) -> usize {
        let n_changes = changes.len();
        for (ch_ix, ch) in changes {
            let old = self.pxs[ch_ix];
            self.area += (ch.0.checked_sub(old.0).unwrap_or(0) + ch.1.checked_sub(old.1).unwrap_or(0)) as usize;
            self.pxs[ch_ix] = ch;
            if ch.0 < self.outer_rect.0 {
                self.outer_rect.2 = self.outer_rect.2 + (self.outer_rect.0 - ch.0);
                self.outer_rect.0 = ch.0;
            }
            if ch.1 < self.outer_rect.1 {
                self.outer_rect.3 = self.outer_rect.3 + (self.outer_rect.1 - ch.1);
                self.outer_rect.1 = ch.1;
            }
            self.outer_rect.2 = (ch.0 - self.outer_rect.0).max(self.outer_rect.2);
            self.outer_rect.3 = (ch.1 - self.outer_rect.1).max(self.outer_rect.3);
        }
        n_changes
    }

    pub fn new_empty() -> Self {
        Self {
            pxs : Vec::with_capacity(32),
            outer_rect : (0, 0, 0, 0),
            color : 0,
            scale : 1,
            img_height : 0,
            area : 0
        }
    }

    pub fn outer_points_with_angle<N>(&self, min_angle : f64, max_angle : f64) -> Vec<(N, N)>
    where
        N : Clone + Copy + From<u16> + 'static
    {
        let angles = self.inner_angles().collect::<Vec<_>>();
        let mut pxs = self.outer_points::<N>(ExpansionMode::Contour);
        let mut ix = 0;
        pxs.retain(|px| { let within = angles[ix] >= min_angle && angles[ix] <= max_angle; ix += 1; within });
        pxs
    }

    /// Returns all inner angles.
    pub fn inner_angles<'a>(&'a self) -> impl Iterator<Item=f64> +'a {
        let n = self.pxs.len();

        let first = shape::vertex_angle(
            (self.pxs[0].0 as usize, self.pxs[0].1 as usize),
            (self.pxs[n-1].0 as usize, self.pxs[n-1].1 as usize),
            (self.pxs[1].0 as usize, self.pxs[1].1 as usize),
        ).unwrap_or(std::f64::consts::PI);

        let last = shape::vertex_angle(
            (self.pxs[n-1].0 as usize, self.pxs[n-1].1 as usize),
            (self.pxs[n-2].0 as usize, self.pxs[n-2].1 as usize),
            (self.pxs[0].0 as usize, self.pxs[0].1 as usize),
        ).unwrap_or(std::f64::consts::PI);

        let triplet_iter = self.pxs.iter().take(n-2)
            .zip(self.pxs.iter().skip(1).take(n-1).zip(self.pxs.iter().skip(2)));

        let middle_iter = triplet_iter.map(|(ext1, (center, ext2))| {

            let opt_angle = shape::vertex_angle(
                (center.0 as usize, center.1 as usize),
                (ext1.0 as usize, ext1.1 as usize),
                (ext2.0 as usize, ext2.1 as usize)
            );

            // In the case the triplets are colinear, opt_angle will be None, and
            // since they are collinear their angle is PI (limit of a triangle
            // that was stretched out over its vertices).
            opt_angle.unwrap_or(std::f64::consts::PI)
        });
        std::iter::once(first).chain(middle_iter).chain(std::iter::once(last))
    }

    pub fn inner_angle_stats(&self) -> (f64, f64) {
        let angles = self.inner_angles().collect::<Vec<_>>();
        bayes::calc::running::mean_variance_from_slice(&angles[..], true)
    }

    pub fn unscaled_area(&self) -> f32 {
        self.area as f32
    }

    pub fn unscaled_perimeter(&self) -> f32 {
        shape::contour_perimeter(&self.pxs[..])
    }

    pub fn perimeter(&self) -> f32 {
        self.unscaled_perimeter() * self.scale as f32
    }

    pub fn circularity(&self) -> Option<f32> {
        // let poly = self.polygon()?;
        // let (area, _) = parry2d::mass_properties::details::convex_polygon_area_and_center_of_mass(&poly.points());
        // shape::circularity(self.unscaled_area() as f64, self.unscaled_perimeter() as f64) as f32
        unimplemented!()
    }

    /*pub fn average_color(&self, win : &Window<'_, u8>, exp : ExpansionMode) -> u8 {
        match exp {
            ExpansionMode::Contour => {

            },
            _ => unimplemented!()
        }
    }*/

    pub fn center<N>(&self) -> (N, N)
    where
        N : From<u16> + Copy + Div<Output=N> + Add<Output=N> + 'static
    {
        let rect = self.outer_rect::<N>();
        (rect.0 + rect.2 / N::from(2u16), rect.1 + rect.3 / N::from(2u16))
    }

    /// Number of pixels contained in the patch.
    pub fn area(&self) -> usize {
        self.area * self.scale as usize
    }

    /// Starts a new patch.
    pub fn new(pt : (u16, u16), color : u8, scale : u16, img_height : usize ) -> Self {
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
    pub fn outer_rect<N>(&self) -> (N, N, N, N)
    where
        N : Clone + Copy + From<u16> + 'static
    {
        (
            N::from(self.outer_rect.0 * self.scale),
            N::from(self.outer_rect.1 * self.scale),
            N::from(self.outer_rect.2 * self.scale),
            N::from(self.outer_rect.3 * self.scale)
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
    pub fn pixel_not_far_below(&self, (r, c) : (u16, u16)) -> bool {
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
    pub fn pixel_not_far_right(&self, (r, c) : (u16, u16)) -> bool {
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

    pub fn add_to_right(&mut self, px : (u16, u16), exp_mode : ExpansionMode) {
        // assert!(self.pixel_not_far_right(px), format!("Pixel not at right: {:?}", (self.outer_rect, &self.pxs, px)));
        match exp_mode {
            ExpansionMode::Rect => self.add_to_right_rect(px),
            ExpansionMode::Dense => self.add_to_right_dense(px),
            ExpansionMode::Contour => self.add_to_right_contour(px),
        }
    }

    pub fn add_to_right_contour(&mut self, px : (u16, u16)) {
        // TODO replace by counter of pixels within the current row.
        let mut this_row = self.pxs.iter().rev().take_while(|(r, _)| *r == px.0 ).take(2);
        let has_2_this_row = this_row.next().is_some() && this_row.next().is_some();
        if has_2_this_row {
            *(self.pxs.last_mut().unwrap()) = px;
        } else  {
            self.pxs.push(px);
        }
        self.expand_rect_and_area(px);
    }

    pub fn add_to_bottom_contour(&mut self, px : (u16, u16)) {
        if self.outer_rect.2 <= 1 {
            // Always add bottom pixels when the patch is still thin.
            self.pxs.push(px);
        } else {
            // If patch is not thin, verify if there are any pixels representing the
            // bottom patch border with the same col. If there are, we substitute it. If
            // there are not (the top row is represented only by the left and right pixels),
            // we add this pixel.

            // We iterate at least 2 rows because insertion of bottom pixels will leave the last row unsorted
            // (might have a few pixels of the row below it).
            let prev_row = self.pxs.iter_mut().rev()
                .enumerate()
                .take_while(|(_, (r, _))| *r >= px.0.saturating_sub(2) )
                .filter(|(_, (r, _))| *r == px.0 - 1 );

            let mut substituted = false;
            for (rev_ix, mut prev_px) in prev_row {
                if prev_px.1 == px.1 {
                    *prev_px = px;

                    // Keep last elements at the last row
                    let n = self.pxs.len();
                    self.pxs.swap(n - rev_ix - 1, n-1);

                    substituted = true;
                    break;
                }
            }

            // In this case, the color matched but the contour only represented
            // the previous row by their left and right border pixels. We push
            // the new pixel in this case, effectively expanding the row by one.
            if !substituted {
                self.pxs.push(px);
            }

            // let has_2_this_col = found_cols == 2;
            // if let Some(subs_px) = subs_col_px {
            // let mut col_iter_mut = self.pxs.iter_mut().rev().filter(|(_, c)| *c == px.1 );
            // *(col_iter_mut.next().unwrap()) = px;
            // *subs_px = px;
            // }
        }
        self.expand_rect_and_area(px);
    }

    pub fn add_to_right_dense(&mut self, px : (u16, u16)) {
        self.pxs.push(px);
        self.expand_rect_and_area(px);
    }

    pub fn add_to_bottom_dense(&mut self, px : (u16, u16)) {
        self.pxs.push(px);
        self.expand_rect_and_area(px);
    }

    pub fn add_to_right_rect(&mut self, px : (u16, u16)) {
        self.expand_rect(&[px]);
    }

    pub fn add_to_bottom_rect(&mut self, px : (u16, u16)) {
        self.expand_rect(&[px]);
    }

    pub fn add_to_bottom(&mut self, px : (u16, u16), exp_mode : ExpansionMode) {
        // assert!(self.pixel_not_far_below(px), format!("Pixel not at bottom: {:?}", (self.outer_rect, &self.pxs, px)));
        match exp_mode {
            ExpansionMode::Rect => self.add_to_bottom_rect(px),
            ExpansionMode::Dense => self.add_to_bottom_dense(px),
            ExpansionMode::Contour => self.add_to_bottom_contour(px)
        }
    }

    pub fn expand_rect_and_area(&mut self, px : (u16, u16)) {
        self.area += 1;
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

        // This is required because self.add_to_bottom assumes row ordering. But col ordering
        // within rows is unimportant. If the vertical extension of the top patch is not larger than
        // the vertical extension of the left patch, however, we do not need to worry about sorting,
        // since the pixels at the bottom will already be the most advanced ones.
        // if self.outer_rect.0 + self.outer_rect.2 > other.outer_rect.0 + other.outer_rect.2 {
        //    self.pxs.sort_by(|a, b| a.0.cmp(&b.0) );
        // }
        // assert!(self.outer_rect.0 + self.outer_rect.2 <= other.outer_rect.0 + other.outer_rect.2);
        // assert!()

        self.area += other.area;
        self.expand_rect(&other.pxs);
    }

    pub fn expand_rect(&mut self, pts : &[(u16, u16)]) {
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
        }
    }

    pub fn same_color(&self, other : &Self) -> bool {
        self.color == other.color
    }

    /// Maps rows to a (sorted) set of columns.
    pub fn group_rows(&self) -> HashMap<u16, Vec<u16>> {
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
    pub fn group_cols(&self) -> HashMap<u16, Vec<u16>> {
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

    pub fn outer_points_ref(&self) -> &[(u16, u16)] {
        assert!(self.scale == 1);
        &self.pxs[..]
    }

    pub fn outer_points_mut(&mut self) -> &mut [(u16, u16)] {
        assert!(self.scale == 1);
        &mut self.pxs[..]
    }

    // TODO return a Cow<[N, N]> here if the user asked for the points in u16 format and scale = 1.
    pub fn outer_points<N>(&self, mode : ExpansionMode) -> Vec<(N, N)>
    where
        N : Clone + Copy + From<u16> + 'static
    {
        match mode {
            ExpansionMode::Dense => {
                let mut row_pxs = self.group_rows();
                let mut sorted_keys = row_pxs.iter().map(|(k, _)| k ).collect::<Vec<_>>();
                if sorted_keys.len() < 3 {
                    return Vec::new();
                }
                sorted_keys.sort();
                let n = sorted_keys.len();
                let mut pts : Vec<(N, N)> = Vec::new();

                // Points with "top" part of the patch
                let fst_row = sorted_keys[0];
                for col in row_pxs[fst_row].iter() {
                    pts.push((N::from(*fst_row * self.scale), N::from(*col * self.scale)));
                }

                // Points with "right" part of the patch
                for row in sorted_keys[1..n-1].iter() {
                    pts.push((N::from(**row * self.scale), N::from(*row_pxs[row].last().unwrap() * self.scale)));
                }

                // Points with "bottom" part of the patch
                let last_row = sorted_keys.last().unwrap();
                for col in row_pxs[last_row].iter().rev() {
                    pts.push((N::from(**last_row * self.scale), N::from(*col * self.scale)));
                }

                // Points with "left" part of the patch
                for row in sorted_keys[1..n-1].iter().rev() {
                    pts.push((N::from(**row * self.scale), N::from(*row_pxs[row].first().unwrap() * self.scale)));
                }
                pts
            },
            ExpansionMode::Contour => {
                self.pxs.iter().map(|px| (N::from(self.scale * px.0), N::from(self.scale * px.1)) ).collect()
            },
            ExpansionMode::Rect => {
                unimplemented!()
            }
        }
    }

    pub fn unscaled_polygon(&self) -> Option<ConvexPolygon> {
        let pts = self.outer_points::<usize>(ExpansionMode::Dense);

        // Some(Polygon::from(pts))

        let float_pts : Vec<_> = pts.iter()
            .map(|pt| Point2::new(pt.1 as f32, pt.0 as f32 ) )
            .collect();
        // ConvexPolygon::from_convex_hull(&float_pts[..])
        unimplemented!()
    }

    pub fn contains(&self, other : &Self) -> Option<bool> {
        /*let this_poly = self.polygon()?;
        let other_poly = other.polygon()?;
        Some(other_poly.points().iter().all(|pt| this_poly.contains_local_point(pt)))*/
        unimplemented!()
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

#[test]
fn segmentation_test() {
    // script!("scripts/segmentation.rh")
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

#[derive(Debug, Clone, Copy)]
pub enum ColorMode {

    Exact(u8),

    Above(u8),

    Below(u8),

    // Carries a value and an absolute value tolerance around it
    Within(u8, u8),

    // Carries a value, an lower absolute tolerance and an upper absolute tolerance around it.
    Between(u8, u8, u8),

    /// The first color is the seed color; the second color is the current color; The third
    /// color is the difference threshold. At each comparison, the current color is updated
    /// with the comparison value IF it matches. Further comparisons are done relative to the
    /// current color. Might also carry a seed weight, specifying how much the seed impacts
    /// in a linear combination with the seed with any matching colors.
    Adaptive(u8, u8, u8)
}

impl ColorMode {

    fn absolute_tolerance(&self) -> u8 {
        match &self {
            ColorMode::Exact(c) => 1,
            ColorMode::Above(c) => 255 - *c,
            ColorMode::Below(c) => *c,
            ColorMode::Within(c, tol) => c.saturating_add(*tol) - c.saturating_sub(*tol),
            ColorMode::Between(c, low_tol, high_tol) => c.saturating_add(*high_tol) - c.saturating_sub(*low_tol),
            ColorMode::Adaptive(_, _, diff) => *diff,
        }
    }

    fn color(&self) -> u8 {
        match &self {
            ColorMode::Exact(c) => *c,
            ColorMode::Above(c) => *c,
            ColorMode::Below(c) => *c,
            ColorMode::Within(c, _) => *c,
            ColorMode::Between(c, _, _) => *c,
            ColorMode::Adaptive(seed, curr, _) => *curr,
        }
    }

    fn set_reference_color(&mut self, color : u8) {
        match self {
            ColorMode::Within(ref mut c, _) => *c = color,
            ColorMode::Exact(ref mut c) => *c = color,
            ColorMode::Above(ref mut c) => *c = color,
            ColorMode::Below(ref mut c) => *c = color,
            ColorMode::Between(ref mut c, _, _) => *c = color,
            ColorMode::Adaptive(ref mut c, _, _) => *c = color
        }
    }

    fn matches(&mut self, px_color : u8) -> bool {
        match self {
            ColorMode::Within(color, tol) => ((px_color as i16 - *color as i16).abs() as u8) < *tol,
            ColorMode::Exact(color) => px_color == *color,
            ColorMode::Above(color) => px_color >= *color,
            ColorMode::Below(color) => px_color <= *color,
            ColorMode::Between(color, low_tol, high_tol) => {
                px_color >= color.saturating_sub(*low_tol) && px_color <= color.saturating_add(*high_tol)
            },
            ColorMode::Adaptive(seed, ref mut current, _diff) => {

                // if ((px_color as i16 - *current as i16).abs() as u8) < *diff {
                if px_color < *current {
                    *current = *seed / 2 as u8 + px_color / 2 as u8;
                    true
                } else {
                    // *current = *seed;
                    false
                }
            }
        }
    }

}

/*/// Updates the contour state. If the strip (row or column) is already complete, returns the index
/// of the pxs vector that should be updated. If the row/column is new at the patch or it is still
/// incomplete (with a single pixel), return None.
fn update_contour_state(matches : &mut HashMap<usize, ContourStrip>, pos : usize, pxs_ix : usize) -> Option<usize> {
    match matches.get(r) {
        Some(ContourStrip::Incomplete) => {
            *self.n_matches_row.get_mut(r).unwrap() = ContourStrip::Complete(pxs_ix);
            None
        },
        Some(ContourStrip::Complete(pos)) => {
            Some(pos)
        },
        None => {
            self.n_matches_row.insert(r, ContourStrip::Incomplete);
            None
        }
    }
}

struct ContourState {
    states : Vec<PatchContourState>,
    n_elems : usize
}

impl ContourState {

    pub fn new() -> Self {
        ContourState {
            states : Vec::new(),
            n_elems : 0
        }
    }

    pub fn merge(&mut self, this_ix : usize, other_ix : usize) {
        assert!(this_ix < n_elems && other_ix < n_elems);
        let other = self.states.remove(other_ix);
        self.n_elems -= 1;
        self.states[this_ix].merge(other);
    }

    pub fn clear(&mut self) {
        self.states.iter_mut().for_each(|s| s.clear() );
        self.n_elems = 0;
    }

    fn retrieve_patch(&mut self, patch_ix : usize) -> &mut PatchContourState {
        assert!(patch_ix <= self.n_elems + 1);
        if let Some(state) = self.states.get_mut(patch_ix) {
            state
        } else {
            self.states.push(PatchContourState::new());
            self.n_elems += 1;
            &mut self.states[self.states.len() - 1]
        }
    }

    pub fn add_new(&mut self, patch_ix : usize, r : usize, c : usize, pxs_ix :usize) {
        self.update_row(patch_ix, r, pxs_ix);
        self.update_col(patch_ix, c, pxs_ix);
    }

    /// Updates contour strip, returning true if it is now complete.
    pub fn update_row(&mut self, patch_ix : usize, r : usize, pxs_ix : usize) -> Option<usize> {
        self.retrieve_patch(patch_ix).update_row(r, pxs_ix)
    }

    /// Updates contour strip, returning true if it is now complete.
    pub fn update_col(&mut self, patch_ix : usize, c : usize, pxs_ix : usize) -> Option<usize> {
        self.retrieve_patch(patch_ix).update_col(c, pxs_ix)
    }

}*/

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
    (0..km.means().count()).map(|ix| bayes::fit::cluster::center::extremes(&km, sample.clone(), ix).unwrap() )
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


