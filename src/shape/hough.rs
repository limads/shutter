use nalgebra::*;
use std::f32::consts::PI;
use crate::image::*;
use crate::local::*;
use std::collections::BTreeMap;
use std::cmp::Ordering;
use std::ops::SubAssign;
// use crate::point::PointOp;
use crate::conv::*;

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

    pub fn as_array(&self) -> [(usize, usize, usize, usize); 8] {
        [
            self.tl, 
            self.top, 
            self.tr, 
            self.left, 
            self.right, 
            self.bl, 
            self.bottom, 
            self.br
        ]
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
    
    // Holds one matrix for each possible radius. Each matrix has the same dimension as the ImageBuf.
    // Use float because eventually we will want to blur it, but it is actually an counter.
    accum : Vec<ImageBuf<f32>>,
    
    accum_blurred : Vec<ImageBuf<f32>>,

    found : BTreeMap<usize, Vec<((usize, usize), f32)>>,
    
    // Local sums over the blurred ImageBuf.
    pub ws : ImageBuf<f32>
    
}

impl HoughCircle {

    pub fn new(
        n_sectors : usize, 
        radius_min : f32, 
        radius_max : f32, 
        n_radii : usize, 
        shape : (usize, usize)
    ) -> Self {
        assert!(n_radii >= 1);

        // Required for gaussian blurring.
        assert!(shape.0 > 5 && shape.1 > 5);

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
            .map(|_| ImageBuf::new_constant(shape.0, shape.1, 0.) ).collect();
        let accum_blurred : Vec<_> = (0..n_radii)
            .map(|_| ImageBuf::new_constant(shape.0 - 5 + 1, shape.1 - 5 + 1, 0.) ).collect();
        let ws = ImageBuf::new_constant(
            accum_blurred[0].height() / 4, 
            accum_blurred[0].width() / 4, 0.0f32);
        Self { angles, radii, accum, shape, deltas, found : BTreeMap::new(), accum_blurred, ws }
    }
    
    pub fn best_matched_accumulator(&self) -> Option<&ImageBuf<f32>> {
        let mut max = 0.0;
        let mut acc = None;
        for rad_ix in self.found.keys() {
            let best_at_this = self.found[&rad_ix].iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal) )
                .unwrap()
                .1;
            if best_at_this > max {
                max = best_at_this;
                acc = Some(&self.accum_blurred[*rad_ix]);
            }
        }
        acc
    }
    
    pub fn all_matched_accumulators(&self) -> Vec<&ImageBuf<f32>> {
        let mut acc = Vec::new();
        for rad_ix in self.found.keys() {
            acc.push(&self.accum_blurred[*rad_ix]);
        }
        acc
    }
    
    pub fn accumulators(&self) -> &[ImageBuf<f32>] {
        &self.accum_blurred[..]
    }
    
    pub fn found(&self) -> &BTreeMap<usize, Vec<((usize, usize), f32)>> {
        &self.found
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
            acc.full_window_mut().fill(0.);
            accumulate_for_radius(pts, &self.deltas[i], acc);
            acc.clone().full_window().convolve_mut(
                &blur::GAUSS, 
                Convolution::Linear, 
                &mut self.accum_blurred[i].full_window_mut()
            );
            
            /*let mut twice_blurred = self.accum_blurred[i].full_window()
                .convolve_padded(&blur::GAUSS, Convolution::Linear);
            twice_blurred = twice_blurred.full_window()
                .convolve_padded(&blur::GAUSS, Convolution::Linear);    
            self.accum_blurred[i].full_window_mut().sub_assign(twice_blurred.full_window());
            self.accum_blurred[i].full_window_mut().abs_mut();*/
            
            
        }
        let mut circles = Vec::new();
        find_hough_maxima(&self.accum_blurred[..], n_expected, min_dist, &mut self.found, self.ws.as_mut());
        for (rad_ix, centers) in &self.found {
            circles.extend(centers.clone().drain(..).map(|c| (c.0, self.radii[*rad_ix])));
        }
        circles        
    }
    
}

fn accumulate_for_radius(
    pts : &[Point2<f32>], 
    deltas : &[Vector2<f32>], 
    accum : &mut ImageBuf<f32>
) {
    let shape = Vector2::new(accum.width() as f32, accum.height() as f32);
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
    accums : &[ImageBuf<f32>], 
    n_expected : usize, 
    min_dist : usize,
    found : &mut BTreeMap<usize, Vec<((usize, usize), f32)>>,
    ws : &mut WindowMut<f32>
) {
    found.clear();
    for _ in 0..n_expected {
        let mut max : ((usize, usize), f32) = ((0, 0), 0.);
        let mut max_rad_ix = 0;
        for rad_ix in 0..accums.len() {
            let outer_shape = accums[rad_ix].shape();
            if let Some(prev_found) = found.get(&rad_ix) {
                let prev_pos = prev_found[0].0;
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
                ).unwrap();
                for r in compl.as_array() {
                    
                    // Use this for absolute global maximum.
                    /*let opt_max = crate::local::min_max_idx(
                        &accums[rad_ix].window((r.0, r.1), (r.2, r.3)).unwrap(), 
                        false,
                        true
                    );*/
                    let opt_max = contrasting_maximum(
                        &accums[rad_ix].window((r.0, r.1), (r.2, r.3)).unwrap(),
                        ws,
                        3
                    );
                    
                    if let (_, Some(new_max)) = opt_max {
                        if new_max.2 > max.1 {
                            max = ((new_max.0 + r.0, new_max.1 + r.1), new_max.2);
                            max_rad_ix = rad_ix;
                        }
                    }
                }
            } else {
                let opt_max = crate::local::min_max_idx(
                    accums[rad_ix].as_ref(), 
                    false, 
                    true
                );
                if let (_, Some(new_max)) = opt_max {
                    if new_max.2 > max.1 {
                        max = ((new_max.0, new_max.1), new_max.2);
                        max_rad_ix = rad_ix;
                    }
                }
            }
        }
        found.entry(max_rad_ix).or_insert(Vec::new()).push(max);
    }
}

fn contrasting_maximum(
    win : &Window<f32>,
    sum : &mut WindowMut<f32>,
    ext : i32
) -> (Option<(usize, usize, f32)>, Option<(usize, usize, f32)>) {
    crate::local::baseline_local_sum(win, sum);
    let mut max_contrast = (0, 0, 0.0);
    for i in (ext+1)..(sum.height() as i32-(ext+1)) {
        for j in (ext+1)..(sum.width() as i32-(ext+1)) {
            let center = sum[(i as usize, j as usize)];
            let mut contrast = 0.0;
            for dx in [-ext, 0, ext] {
                for dy in [-ext, 0, ext] {
                
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    
                    let surround = sum[((i+dy) as usize, (j+dx) as usize)];
                    // contrast += (center - surround) / (center + surround);
                    contrast += center - surround;
                }
            }
            // contrast /= 8.0;
            if contrast > 0.0 && contrast > max_contrast.2 {
                max_contrast = (i as usize * 4, j as usize * 4, contrast);
            }
        }
    }
    (None, Some(max_contrast))
}

mod test {

    use super::*;

    fn test_hough_at_range(
        true_radius : f32, 
        min_radius : f32, 
        max_radius : f32,
        n_radii : usize
    ) {
        let mut pts = Vec::new();
        let c = Point2::new(50.0, 50.0);
        for i in 0..100 {
            let theta = 2.0*PI / 100.0 * i as f32; 
            let delta = Vector2::new(
                theta.cos() * true_radius, 
                theta.sin() * true_radius
            );
            pts.push(c.clone() + delta);
        }
        let mut hough = HoughCircle::new(12, min_radius, max_radius, n_radii, (100,100));
        let ans = hough.calculate(&pts[..], 1, 1);
        println!("{:?}", ans);
        for i in 0..n_radii {
            hough.accum[i].show();
        }
    }
    
    fn test_hough_at_unique_radius(
        true_radius : f32
    ) {
        test_hough_at_range(true_radius, true_radius, true_radius, 1);
    }
    
    // cargo test -- test_hough --nocapture 
    #[test]
    fn test_hough() {
        // test_hough_at_unique_radius(20.0);
        test_hough_at_range(20.0, 10.0, 30.0, 5);
    }

}



