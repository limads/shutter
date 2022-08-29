use nalgebra::*;
use std::f32::consts::PI;
use crate::image::*;
use crate::local::*;
use std::collections::BTreeMap;

pub struct RectComplement {
    pub tl : (usize, usize, usize, usize),
    pub tr : (usize, usize, usize, usize),
    pub bl : (usize, usize, usize, usize),
    pub br : (usize, usize, usize, usize),
    pub top : (usize, usize, usize, usize),
    pub right : (usize, usize, usize, usize),
    pub bottom : (usize, usize, usize, usize),
    pub left : (usize, usize, usize, usize),
}

impl RectComplement {

    pub fn iter(&self) -> impl Iterator<Item=(usize, usize)> + 'static {
        [self.tl, self.top, left.tr, self.left, self.right, self.bl, self.bottom, self.br]
            .into_iter()
    }
    
}

pub fn rect_complement(
    inner : (usize, usize, usize, usize),
    outer : (usize, usize, usize, usize)
) -> Option<RectComplement> {
    let outside_bounds = inner.0 < outer.0 ||
        inner.0 > outer.2.checked_sub(inner.0)? ||
        inner.1 > outer.3.checked_sub(inner.1)? ||
        inner.1 < outer.1;
    if outside_bounds {
        return None;
    }
    let left_w = inner.1 - outer.1;
    let right_w = (outer.1 + outer.3) - (inner.1 + inner.3);
    let top_h = inner.0 - outer.0;
    let bottom_h = (outer.0 + outer.2) - (inner.0 + inner.2);
    let right_col_offset = inner.1 + inner.3;
    let bottom_row_offset = inner.0 + inner.2;
    Some(RectComplement {
        tl : (outer.0, outer.1, top_h, left_w),
        tr : (outer.0, right_col_offset, top_h, right_w),
        bl : (bottom_row_offset, outer.0, bottom_h, left_w),
        br : (bottom_row_offset, right_col_offset, bottom_h, right_w),
        top : (outer.0, inner.1, top_h, inner.3),
        right : (inner.0, right_col_offset, inner.2, right_w),
        bottom : (bottom_row_offset, inner.1, bottom_h, inner.3),
        left : (inner.0, outer.1, inner.2, left_w)
    })
}

#[derive(Debug, Clone)]
pub struct HoughCircle {

    angles : Vec<f32>,
    
    radii : Vec<f32>,
    
    // For each radius, holds the x and y increments at the cos and sin component
    // for each angle. Outer vecs has dimensions of radii, inner vecs the dimensions
    // of angles.
    deltas : Vec<Vec<Vector2<f32>>>,
    
    shape : (usize, usize),
    
    // Holds one matrix for each possible radius. Each matrix has the same dimension as the image.
    // Use float because eventually we will want to blur it, but it is actually an counter.
    accum : Vec<Image<f32>>
    
}

impl HoughCircle {

    pub fn new(
        n_sectors : usize, 
        radius_min : f32, 
        radius_max : f32, 
        n_radii : usize, 
        shape : (usize, usize)
    ) -> Self {
        let delta_sector = 2.0 * PI / n_sectors as f32;
        let angles : Vec<_> = (0..n_sectors)
            .map(|i| i as f32 * delta_sector ).collect();
        
        let delta_radius = (radius_max - radius_min) / n_radii as f32;
        let radii : Vec<_> = (0..n_radii)
            .map(|i| radius_min + i as f32 * delta_radius ).collect();
        
        let mut deltas = Vec::new();
        for r in &radii {
            let mut this_delta = Vec::with_capacity(angles.len());
            for angle in &angles {
                this_delta.push(Vector2::new(angle.cos() * r, angle.sin() * r));
            }
            deltas.push(this_delta);
        }
        
        let accum : Vec<_> = (0..n_radii)
            .map(|_| Image::new_constant(shape.0, shape.1, 0.) ).collect();
        Self { angles, radii, accum, shape, deltas }
    }
    
    /// Returns the n-highest circle peaks at least dist apart from each other.
    /// Points are in cartesian coordinates.
    pub fn calculate(
        &mut self, 
        pts : &[Point2<f32>], 
        n_expected : usize, 
        min_dist : usize
    ) -> Vec<((usize, usize), f32)> {
        for i in 0..self.radii.len() {
            let mut acc = &mut self.accum[i];
            acc.fill(0.);
            accumulate_for_radius(pts, &self.deltas[i], acc);
            *acc = accum.clone_owned().convolve(&crate::blur::GAUSS_5, Convolution::Linear);
        }
        let mut circles = Vec::new();
        let mut maxima = find_hough_maxima(&self.accum, n_expected, min_dist);
        for (rad_ix, mut centers) in maxima {
            circles.extend(centers.drain(..).map(|c| (c, self.radii[rad_ix])));
        }
        circles        
    }
    
}

fn accumulate_for_radius(
    pts : &[Point2<f32>], 
    deltas : &[Vector2<f32>], 
    accum : &mut Image<f32>
) {
    let shape = Vector2::new(accum.ncols() as f32, accum.nrows() as f32);
    for pt in pts {
        for d in deltas {
            let center = pt.clone() + d;
            let inside = center[0] > 0.0 && 
                center[1] > 0.0 && 
                center[0] < shape[0] && 
                center[1] < shape[1];
            if inside {
                accum[((shape[1] - center[1]) as usize, center[0] as usize)] += 1.0;
            }
        }
    }
}

// The returned map maps indices of the radius vector into a 
// set of likely circle centers.
// Perhaps blur found peaks a little bit.
// Take as parameter number of points in the circle edge (will equal the number of votes
// in a nearby blurred region).

// Find global maximum. Then define a neighborhood of size min_dist around it
// where no further searches will be done. Then repeat the search until
// n_expected is found.

// Repeat the above steps for each possible radius, then calculate the
// global maxima across all radii.
fn find_hough_maxima(
    accums : &[Image<f32>], 
    n_expected : usize, 
    min_dist : usize
) -> BTreeMap<usize, Vec<(usize, usize)> {
    let mut found = BTreeMap::new();
    for i in 0..n_expected {
        let mut max : ((usize, usize), f32) = ((0, 0), 0.);
        for rad_ix in 0..accums.len() {
            let outer_shape = accums[rad_ix].shape();
            if let Some(prev_found) = found.get(rad_ix) {
                let prev_pos = prev_found[0];
                let offy = prev_pos.0.saturating_sub(min_dist);
                let offx = prev_pos.1.saturating_sub(min_dist);
                let mut exclude = (
                    offy,
                    offx,
                    (2*min_dist).min(outer_shape.0-offy),
                    (2*min_dist).min(outer_shape.1-offx)
                );
                let compl = rect_complement(
                    exclude,
                    (0, 0, outer_shape.0, outer_shape.1)
                );
                for r in compl.iter() {
                    let opt_max = crate::local::min_max_idx(
                        &accums[i].sub_window((r.0, r.1), (r.2, r.3)).unwrap(), 
                        false,
                        true
                    );
                    if let (_, Some(new_max)) = opt_max {
                        if new_max.1 > max.1 {
                            max = ((new_max.0 + r.0, new_max.1 + r.1), new_max.1);
                        }
                    }
                }
            } else {
                let opt_max = crate::local::min_max_idx(accums[rad_ix].as_ref(), false, true);
                if let (_, Some(new_max)) = opt_max {
                    if new_max.1 > max.1 {
                        max = new_max;
                    }
                }
            }
        }
        found[j] = max;
    }
    
}

fn main() {
    
}
