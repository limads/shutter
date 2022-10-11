use std::cmp::{PartialEq, Ordering};
use nalgebra::*;
use bayes::fit::cluster::{Manhattan, Metric};
use nalgebra::geometry::Rotation2;
use nalgebra::Vector2;
use std::cmp::Ord;
use std::ops::Add;
use nalgebra;
use std::mem;
use crate::image::*;
use itertools::Itertools;
use std::collections::BTreeMap;

pub mod point;

pub mod edge;

use edge::*;

pub mod ellipse;

// pub use ellipse::*;

pub mod hough;

// cargo test -- registration_test --nocapture
#[test]
fn registration_test() {

    let a = vec![(0, 10), (10, 10), (10, 0)];
    
    // Just scaled (perserve order)
    let b = vec![(0, 20), (20, 20), (20, 0)];
    
    // Scaled and permuted (0,1)
    let c = vec![(20, 20), (0, 20), (20, 0)];
    
    // Scaled and permuted (0,3), (1,2)
    let d = vec![(20, 20), (20, 0), (0, 20)];
    
    println!("{:?}", registration(&a, &b));
    println!("{:?}", registration(&a, &c));
    println!("{:?}", registration(&a, &d));
    
}

/* Given two equal length slices carryinig coordinates, permute elements at the second slice until
all elements within the second slice satisfy the same comparison relations with each other as the elements 
within the first slice. All elements at both slices are assumed distinct.
If this condition can be satisfied, return Some(col), else None. 
This is useful as a first approximation to
registration problems, since each index of current will be equivalent to the
same index at past. Returns None if slices are of different lengths or 
elements are not distinct. Corresponding points map exactly if they 
are scaled along the orthogonal coordinate axis, or if they are translated
to polar coordinates and suffer arbitrary scaling and rotations around the same axis.*/
pub fn registration(
    past : &[(usize, usize)], 
    current : &[(usize, usize)]
) -> Option<Vec<(usize, usize)>> {
    let n = past.len();
    if current.len() != n {
        return None;
    }
    let mut past_comp = BTreeMap::new();
    for i in 0..n {
        for j in (i+1)..n {
            if past[i] == past[j] {
                return None;
            }
            past_comp.insert((i, j), (past[i].0 > past[j].0, past[i].1 > past[j].1 ));
        }
    }
    'outer : for ixs in (0..n).permutations(n) {
        for i in 0..n {
            for j in (i+1)..n {
                let past = past_comp[&(i,j)];
                if ( (current[ixs[i]].0 > current[ixs[j]].0) != past.0 ) || 
                   ( (current[ixs[i]].1 > current[ixs[j]].1) != past.1 )
                {
                    continue 'outer;
                }
            }
        }
        
        // Exit the inner loops only if all comparisons are equivalent.
        let permuted : Vec<_> = (0..n).map(|i| current[ixs[i]] ).collect();
        return Some(permuted);
    }
    None
}

pub fn bounding_rect(pts : &[(usize, usize)]) -> (usize, usize, usize, usize) {
    let (mut min_y, mut max_y) = (usize::MAX, 0);
    let (mut min_x, mut max_x) = (usize::MAX, 0);
    for pt in pts.iter() {
        if pt.0 < min_y {
            min_y = pt.0;
        }
        if pt.0 > max_y {
            max_y = pt.0;
        }
        if pt.1 < min_x {
            min_x = pt.1;
        }
        if pt.1 > max_x {
            max_x = pt.1;
        }
    }
    (min_y, min_x, max_y - min_y, max_x - min_x)
}

pub mod coord {

    use nalgebra::{Vector2, Scalar, Point2};
    use num_traits::{AsPrimitive, Zero};
    use std::cmp::PartialOrd;

    pub fn coord_to_point<F>(coord : (usize, usize), shape : (usize, usize)) -> Option<Point2<F>>
    where
        usize : AsPrimitive<F>,
        F : Scalar + Copy
    {
        coord_to_vec(coord, shape).map(|v| Point2::from(v) )
    }

    // Maps coord to vector with strictly positive entries with origin at the bottom-left
    // pixel in the image.
    pub fn coord_to_vec<F>(coord : (usize, usize), shape : (usize, usize)) -> Option<Vector2<F>>
    where
        usize : AsPrimitive<F>,
        F : Scalar + Copy
    {
        Some(Vector2::new(coord.1.as_(), (shape.0.checked_sub(coord.0)?).as_()))
    }

    /* If pt is a point in the cartesian plane centered at the image bottom left
    pixel, returns the correspoinding image coordinate. */
    pub fn point_to_coord<F>(pt : &Vector2<F>, shape : (usize, usize)) -> Option<(usize, usize)>
    where
        F : AsPrimitive<usize> + PartialOrd + Zero
    {
        if pt[0] > F::zero() && pt[1] > F::zero() {
            let (row, col) : (usize, usize) = (pt[1].as_(), pt[0].as_());
            if col < shape.1 && row < shape.0 {
                return Some((shape.0 - row, col));
            }
        }
        None
    }

}

// Ramer-Douglas-Peucker line simplification, based on
// https://rosettacode.org/wiki/Ramer-Douglas-Peucker_line_simplification
mod rdp {
    #[derive(Copy, Clone)]
    struct Point {
        x: f64,
        y: f64,
    }

    // Returns the distance from point p to the line between p1 and p2
    fn perpendicular_distance(p: &Point, p1: &Point, p2: &Point) -> f64 {
        let dx = p2.x - p1.x;
        let dy = p2.y - p1.y;
        (p.x * dy - p.y * dx + p2.x * p1.y - p2.y * p1.x).abs() / dx.hypot(dy)
    }

    fn rdp(points: &[Point], epsilon: f64, result: &mut Vec<Point>) {
        let n = points.len();
        if n < 2 {
            return;
        }
        let mut max_dist = 0.0;
        let mut index = 0;
        for i in 1..n - 1 {
            let dist = perpendicular_distance(&points[i], &points[0], &points[n - 1]);
            if dist > max_dist {
                max_dist = dist;
                index = i;
            }
        }
        if max_dist > epsilon {
            rdp(&points[0..=index], epsilon, result);
            rdp(&points[index..n], epsilon, result);
        } else {
            result.push(points[n - 1]);
        }
    }

    fn ramer_douglas_peucker(points: &[Point], epsilon: f64) -> Vec<Point> {
        let mut result = Vec::new();
        if points.len() > 0 && epsilon >= 0.0 {
            result.push(points[0]);
            rdp(points, epsilon, &mut result);
        }
        result
    }
}

/*/// Convex hull implementation, based on
/// https://rosettacode.org/wiki/Convex_hull
mod convex {

    use nalgebra::Point2;
    
    pub fn convex_hull(points: &Vec<Point2<F>>) -> Vec<Point> {
        //There must be at least 3 points
        if points.len() < 3 { return points.clone(); }

        let mut hull = vec![];

        //Find the left most point in the polygon
        let (left_most_idx, _) = points.iter()
            .enumerate()
            .min_by(|lhs, rhs| lhs.1.x.partial_cmp(&rhs.1.x).unwrap())
            .expect("No left most point");


        let mut p = left_most_idx;
        let mut q = 0_usize;

        loop {
            //The left most point must be part of the hull
            hull.push(points[p].clone());

            q = (p + 1) % points.len();

            for i in 0..points.len() {
                if orientation(&points[p], &points[i], &points[q]) == 2 {
                    q = i;
                }
            }

            p = q;

            //Break from loop once we reach the first point again
            if p == left_most_idx { break; }
        }

        return hull;
    }
    
    fn orientation(p: &Point, q: &Point, r: &Point) -> usize {
        let val = (q.y - p.y) * (r.x - q.x) -
            (q.x - p.x) * (r.y - q.y);

        if val == 0. { return 0 };
        if val > 0. { return 1; } else { return 2; }
    }

}*/

/*
// Based on https://rosettacode.org/wiki/Find_the_intersection_of_two_lines
#[derive(Copy, Clone, Debug)]
struct Line(Point, Point);

impl Line {
    pub fn intersect(self, other: Self) -> Option<Point> {
        let a1 = self.1.y - self.0.y;
        let b1 = self.0.x - self.1.x;
        let c1 = a1 * self.0.x + b1 * self.0.y;

        let a2 = other.1.y - other.0.y;
        let b2 = other.0.x - other.1.x;
        let c2 = a2 * other.0.x + b2 * other.0.y;

        let delta = a1 * b2 - a2 * b1;

        if delta == 0.0 {
            return None;
        }

        Some(Point {
            x: (b2 * c1 - b1 * c2) / delta,
            y: (a1 * c2 - a2 * c1) / delta,
        })
    }
}

// Based on https://rosettacode.org/wiki/Line_circle_intersection
const EPS: f64 = 1e-14;

pub struct Point {
    x: f64,
    y: f64,
}

pub struct Line {
    p1: Point,
    p2: Point,
}

impl Line {
    pub fn circle_intersections(&self, mx: f64, my: f64, r: f64, segment: bool) -> Vec<Point> {
        let mut intersections: Vec<Point> = Vec::new();

        let x0 = mx;
        let y0 = my;
        let x1 = self.p1.x;
        let y1 = self.p1.y;
        let x2 = self.p2.x;
        let y2 = self.p2.y;

        let ca = y2 - y1;
        let cb = x1 - x2;
        let cc = x2 * y1 - x1 * y2;

        let a = ca.powi(2) + cb.powi(2);
        let mut b = 0.0;
        let mut c = 0.0;
        let mut bnz = true;

        if cb.abs() >= EPS {
            b = 2.0 * (ca * cc + ca * cb * y0 - cb.powi(2) * x0);
            c = cc.powi(2) + 2.0 * cb * cc * y0
                - cb.powi(2) * (r.powi(2) - x0.powi(2) - y0.powi(2));
        } else {
            b = 2.0 * (cb * cc + ca * cb * x0 - ca.powi(2) * y0);
            c = cc.powi(2) + 2.0 * ca * cc * x0
                - ca.powi(2) * (r.powi(2) - x0.powi(2) - y0.powi(2));
            bnz = false;
        }
        let mut d = b.powi(2) - 4.0 * a * c;
        if d < 0.0 {
            return intersections;
        }

        fn within(x: f64, y: f64, x1: f64, y1: f64, x2: f64, y2: f64) -> bool {
            let d1 = ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt(); // distance between end-points
            let d2 = ((x - x1).powi(2) + (y - y1).powi(2)).sqrt(); // distance from point to one end
            let d3 = ((x2 - x).powi(2) + (y2 - y).powi(2)).sqrt(); // distance from point to other end
            let delta = d1 - d2 - d3;
            return delta.abs() < EPS;
        }

        fn fx(x: f64, ca: f64, cb: f64, cc: f64) -> f64 {
            -(ca * x + cc) / cb
        }

        fn fy(y: f64, ca: f64, cb: f64, cc: f64) -> f64 {
            -(cb * y + cc) / ca
        }

        fn rxy(
            x: f64,
            y: f64,
            x1: f64,
            y1: f64,
            x2: f64,
            y2: f64,
            segment: bool,
            intersections: &mut Vec<Point>,
        ) {
            if !segment || within(x, y, x1, y1, x2, y2) {
                let point = Point { x: x, y: y };
                intersections.push(point);
            }
        }

        if d == 0.0 {
            if bnz {
                let x = -b / (2.0 * a);
                let y = fx(x, ca, cb, cc);
                rxy(x, y, x1, y1, x2, y2, segment, &mut intersections);
            } else {
                let y = -b / (2.0 * a);
                let x = fy(y, ca, cb, cc);
                rxy(x, y, x1, y1, x2, y2, segment, &mut intersections);
            }
        } else {
            d = d.sqrt();
            if bnz {
                let x = (-b + d) / (2.0 * a);
                let y = fx(x, ca, cb, cc);
                rxy(x, y, x1, y1, x2, y2, segment, &mut intersections);
                let x = (-b - d) / (2.0 * a);
                let y = fx(x, ca, cb, cc);
                rxy(x, y, x1, y1, x2, y2, segment, &mut intersections);
            } else {
                let y = (-b + d) / (2.0 * a);
                let x = fy(y, ca, cb, cc);
                rxy(x, y, x1, y1, x2, y2, segment, &mut intersections);
                let y = (-b - d) / (2.0 * a);
                let x = fy(y, ca, cb, cc);
                rxy(x, y, x1, y1, x2, y2, segment, &mut intersections);
            }
        }

        intersections.sort_unstable_by(|a, b| a.x.partial_cmp(&b.x).unwrap());
        intersections
    }
}
*/

pub trait Quadrilateral {

    fn top_left(&self) -> (usize, usize);

    fn size(&self) -> (usize, usize);

}

pub struct Square {

    pub tl : Vector2<usize>,

    pub side : usize

}

impl Quadrilateral for Square {

    fn top_left(&self) -> (usize, usize) {
        (self.tl[0], self.tl[1])
    }

    fn size(&self) -> (usize, usize) {
        (self.side, self.side)
    }

}

pub struct Rect {

    pub tl : Vector2<usize>,

    pub sides : Vector2<usize>

}

impl Quadrilateral for Rect {

    fn top_left(&self) -> (usize, usize) {
        (self.tl[0], self.tl[1])
    }

    fn size(&self) -> (usize, usize) {
        (self.sides[0], self.sides[1])
    }

}

pub fn point_manhattan_distance(a : &(usize, usize), b : &(usize, usize)) -> usize {
    a.0.abs_diff(b.0) + a.1.abs_diff(b.1)
}

pub fn pair_distance(a : (usize, usize), b : (usize, usize)) -> (i16, i16) {
    ((a.0 as i16 - b.0 as i16).abs(), (a.1 as i16 - b.1 as i16).abs())
}

pub fn is_pair_contiguous(a : (usize, usize), b : (usize, usize)) -> bool {
     pair_distance(a, b) == (1, 1)
}

pub fn is_contour_contigous(pts : &[(usize, usize)]) -> bool {
    pts.iter().take(pts.len()-1).zip(pts.iter().skip(1)).all(|(a, b)| is_pair_contiguous(*a, *b) )
}

pub fn make_contour_contiguous(pts : &mut [(usize, usize)]) -> usize {
    make_contour_contiguous_step(pts, &mut pts.len())
}

struct PointCandidate {
    ix : usize,
    is_dead_end : bool
}

/* Swaps points at the slice until a maximum set of contigous elements is found.
All elements before the returned index will be contigous. If no elements could be
made contiguos, this will be zero; if all could be made contiguous, return slice.len().
No guarantee is made about the order of the elements after this partition index.
The function should be started with ix = pts.len(). Assumes each point is unique. */
fn make_contour_contiguous_step(pts : &mut [(usize, usize)], partition_ix : &mut usize) -> usize {
    match pts.len() - *partition_ix {
        0..=1 => {
            *partition_ix
        },
        2 => {
            if is_pair_contiguous(pts[0], pts[1]) {
                *partition_ix
            } else {
                *partition_ix - 1
            }
        },
        n => {

            // Holds indices of pts that are contigous to the first point. This holds
            // at a maximum 8 points (assuming uniqueness), so could be an array/slice to reduce allocations.
            let mut candidates = Vec::new();
            for ix in 1..(*partition_ix) {
                if is_pair_contiguous(pts[0], pts[ix]) {
                    candidates.push(ix);
                }
            }

            let mut best_candidate : Option<PointCandidate> = None;

            // Keep only candidates that are also contiguous with any other points (excluding the
            // current one).
            for cand in candidates.drain(..).rev() {
                let mut other_points = pts[1..cand].iter().chain(pts[cand+1..*partition_ix].iter());
                let this_is_dead_end = !other_points.any(|other_pt| is_pair_contiguous(pts[cand], *other_pt) );
                if let Some(ref mut best_cand) = best_candidate {
                    match (this_is_dead_end, best_cand.is_dead_end) {

                        // If previous and this are both dead ends, swap new to end of slice.
                        (true, true) => {
                            pts.swap(cand, *partition_ix-1);
                            *partition_ix -= 1;
                        },

                        // This is dead end but old candidate is not. Swap new to end of slice as well.
                        (true, false) => {
                            pts.swap(cand, *partition_ix-1);
                            *partition_ix -= 1;
                        },

                        // Found a linking point. Replace by previous dead end candidate.
                        (false, true) => {
                            pts.swap(cand, 1);
                            *partition_ix -= 1;
                            best_cand.ix = cand;
                            best_cand.is_dead_end = false;
                        },

                        (false, false) => {
                            // Here, decide on the best index (both are valid). We must remove one of
                            // them to avoid forking paths at the contour. Remove new one by convention.
                            pts.swap(cand, *partition_ix-1);
                            *partition_ix -= 1;
                        }
                    }
                } else {
                    pts.swap(cand, 1);
                    best_candidate = Some(PointCandidate { is_dead_end : this_is_dead_end, ix : cand });
                }
            }

            match best_candidate {
                Some(cand) => {
                    if cand.is_dead_end {
                        *partition_ix
                    } else {
                        make_contour_contiguous_step(&mut pts[2..], partition_ix)
                    }
                },

                // Cannot find contiguous points anymore. Just set all points to unordered and stop search.
                None => {
                    *partition_ix - pts.len()
                }
            }
        }
    }
}

/* Verifies if one contours completes another (i.e. the extreme points at both ends are within
a maximum distance. */
pub fn contour_completes(a : &[(usize, usize)], b : &[(usize, usize)], dist : f32) -> bool {
    false
}

/* Split polygonal approximation (after Distante & Distante, 2020)
(1) Find farthest pair of points at contour and trace a bisection line;
(2) Find pair of points with largest distance perpendicular to the bisection line (where p1 is opposite to p2 wrt this line)
(3) With the four resulting points, consider the 4 closed remaining segments
*/

/* The area is the zeroth moment. The first moment is the pixel sum divided by area, or average. */
pub fn point_centroid(pts : &[(usize, usize)]) -> (f32, f32) {
    let sum : (f32, f32) = pts.iter().fold((0.0, 0.0), |avg, pt| (avg.0 + pt.0 as f32, avg.1 + pt.1 as f32) );
    let n = pts.len() as f32;
    (sum.0 / n, sum.1 / n)
}

#[cfg(feature="ipp")]
pub struct IppiCentralMoments {
    state : Vec<u8>
}

#[cfg(feature="ipp")]
impl IppiCentralMoments {

    pub fn new() -> Self {
        unsafe {
            let mut sz : i32 = 0;
            let alg = crate::foreign::ipp::ippi::IppHintAlgorithm_ippAlgHintNone;
            let ans = crate::foreign::ipp::ippi::ippiMomentGetStateSize_64f(
                alg,
                &mut sz as *mut _
            );
            assert!(ans == 0);
            let mut state = Vec::with_capacity(sz as usize);
            let ans = crate::foreign::ipp::ippi::ippiMomentInit_64f(
                mem::transmute(state.as_mut_ptr()),
                alg
            );
            assert!(ans == 0);
            Self { state }
        }
    }

    unsafe fn get_spatial_moment(&self, m : i32, n : i32) -> f32 {
        let channel = 0;
        let mut val : f64 = 0.;
        let offset = crate::foreign::ipp::ippi::IppiPoint { x : 0, y : 0 };
        let ans = crate::foreign::ipp::ippi::ippiGetSpatialMoment_64f(
            mem::transmute(self.state.as_ptr()),
            m,
            n,
            channel,
            offset,
            &mut val as *mut _
        );
        assert!(ans == 0);
        val as f32
    }

    unsafe fn get_central_moment(&self, m : i32, n : i32) -> f32 {
        let channel = 0;
        let mut val : f64 = 0.;
        let ans = crate::foreign::ipp::ippi::ippiGetCentralMoment_64f(
            mem::transmute(self.state.as_ptr()),
            m,
            n,
            channel,
            &mut val as *mut _
        );
        assert!(ans == 0);
        val as f32
    }

    pub fn calculate(&mut self, win : &Window<u8>) -> CentralMoments {
        unsafe {
            let (step, sz) = crate::image::ipputils::step_and_size_for_window(win);
            let ans = crate::foreign::ipp::ippi::ippiMoments64f_8u_C1R(
                win.as_ptr(),
                step,
                sz,
                mem::transmute(self.state.as_mut_ptr())
            );
            assert!(ans == 0);

            // 0th moment = area = number of pixels
            let zero = self.get_spatial_moment(0, 0);
            let center_x = self.get_spatial_moment(1, 0) / zero;
            let center_y = self.get_spatial_moment(0, 1) / zero;

            let zero = self.get_central_moment(2, 0);
            let xx = self.get_central_moment(2, 0);
            let yy = self.get_central_moment(0, 2);
            let xy = self.get_central_moment(1, 1);
            let xxy = self.get_central_moment(2, 1);
            let yyx = self.get_central_moment(1, 2);
            let xxx = self.get_central_moment(3, 0);
            let yyy = self.get_central_moment(0, 3);
            CentralMoments {
                center : (center_y, center_x),
                zero,
                xx,
                yy,
                xy,
                xxy,
                yyx,
                xxx,
                yyy
            }
        }
    }

}

// The central moments are translation invariant. Moments can be calculated over the
// shape edge only, since the edge is equivalent to a binary image giving weight 1
// to pixels at the edge and weight zero for pixels outside it.
pub struct CentralMoments {

    // (row, col) just convert to f32
    pub center : (f32, f32),

    pub zero : f32,

    pub xx : f32,

    pub yy : f32,

    pub xy : f32,

    pub xxx : f32,

    pub yyy : f32,

    pub xxy : f32,

    pub yyx : f32

}

impl CentralMoments {

    // Calculate, with the result in cartesian orientation.
    pub fn calculate_upright(pts : &[(usize, usize)], img_height : usize) -> Self {
        let corrected : Vec<_> = pts.iter().map(|pt| (img_height - pt.0, pt.1) ).collect();
        Self::calculate(&corrected, None)
    }

    pub fn calculate(pts : &[(usize, usize)], centroid : Option<(f32, f32)>) -> Self {
        let centroid = centroid.unwrap_or(point_centroid(pts));
        let diffs : Vec<(f32, f32)> = pts.iter()
            .map(|pt| (pt.0 as f32 - centroid.0 as f32, pt.1 as f32 - centroid.1 as f32) )
            .collect();
        let xx = diffs.iter().fold(0.0, |acc, d| acc + d.1.powf(2.) );
        let yy = diffs.iter().fold(0.0, |acc, d| acc + d.0.powf(2.) );
        let xy = diffs.iter().fold(0.0, |acc, d| acc + (d.0 * d.1) );
        let xxx = diffs.iter().fold(0.0, |acc, d| acc + d.1.powf(3.) );
        let yyy = diffs.iter().fold(0.0, |acc, d| acc + d.0.powf(3.) );
        let xxy = diffs.iter().fold(0.0, |acc, d| acc + (d.1 * d.1 * d.0) );
        let yyx = diffs.iter().fold(0.0, |acc, d| acc + (d.0 * d.0 * d.1) );
        Self { zero : diffs.len() as f32, xx, yy, xy, xxx, yyy, xxy, yyx, center : centroid }
    }

    // After Burger & Burge (11.25), resulting in the range [-pi/2, pi/2] (positive and negative RIGHT quadrant)
    // BUT inverted when the y coordinate of the image is inverted.
    pub fn major_axis_orientation(&self) -> f32 {
        0.5*( (2.0*self.xy) / (self.xx - self.yy) ).atan()
    }

    // Upright unit vector resulting from the orientation (Burger & Burge, 11.26),
    // points in the same direction as the major ellipse axis.
    pub fn oriented_unit_vector(&self) -> Vector2<f32> {

        // Multiply by -1 (reflect y coordinate) when y moments are inverted wrt cartesian plane.
        let theta = self.major_axis_orientation();

        Vector2::new(theta.cos(), theta.sin())
    }

    // Resulting major and minor ellipse vectors (Burger & Burge, 11.31 and 11.32).
    // num_pxs is the number of pixels in the region, required for
    // scaling back the value (zero-th moment). vectors are relative to the ellipse center.
    pub fn ellipse_vectors(&self) -> (Vector2<f32>, Vector2<f32>) {

        // TODO if only the boundary is used for calculation, then num_pxs
        // must be informed separately. num_pxs == area only if the pxs used
        // to calculate the moments are a dense representation.
        let num_pxs = self.zero;
        let mut major = self.oriented_unit_vector();
        //println!("{}", major);
        let mut minor = nalgebra::geometry::Rotation2::new(std::f32::consts::PI/2.) * major;
        //println!("{}", minor);

        let (lambda1, lambda2) = self.eigenvalues();
        assert!(lambda1 >= lambda2);

        major.scale_mut(2.*(lambda1 / num_pxs).sqrt());
        minor.scale_mut(2.*(lambda2 / num_pxs).sqrt());
        (major, minor)
    }

    pub fn ellipse(&self, img_height : usize) -> crate::shape::ellipse::Ellipse {
        let (major, minor) = self.ellipse_vectors();
        let center = self.centroid_point(img_height);
        crate::shape::ellipse::Ellipse { center, major, minor }
    }

    pub fn eigenvalues(&self) -> (f32, f32) {
        let exc = self.excentricity_matrix();
        let eigen = nalgebra::linalg::SymmetricEigen::new(exc);
        (eigen.eigenvalues[0], eigen.eigenvalues[1])
    }

    // Burger & Burge (11.30). The excentricity is a ratio of the largest to
    // smallest eigenvalue of the affine matrix. It is the length of the major axis
    // of the ellipse axis that best approximate the object. Therefore self.oriented_unit_vector()
    // scaled by the scalar resulting from this function gives the resulting ellipse.
    pub fn excentricity(&self) -> f32 {
        let (lambda1, lambda2) = self.eigenvalues();
        lambda1 / lambda2
    }

    // Burger & Burge (11.30)
    pub fn excentricity_matrix(&self) -> Matrix2<f32> {
        Matrix2::from_rows(&[
            RowVector2::new(self.xx, self.xy),
            RowVector2::new(self.xy, self.yy)
        ])
    }

    pub fn centroid_point(&self, img_height : usize) -> Vector2<f32> {
        let center = self.center();
        Vector2::new(center.1, img_height as f32 - center.0)
    }

    pub fn center(&self) -> (f32, f32) {
        self.center
    }

}

// Calculate scaled moments by dividing them by the zero-th moment (area), which
// generates scale-invariant moments (i.e. the same values irrespective of the
// distance between camera or object. The object is not, however, orientation-invariant).
pub struct NormalizedMoments {

    pub xx : f32,

    pub yy : f32,

    pub xy : f32,

    pub xxx : f32,

    pub yyy : f32,

    pub xxy : f32,

    pub yyx : f32

}

impl NormalizedMoments {

    // norm moment: mu_pq * (1/area)^((p + q + 2)/2.) for p+q >= 2
    pub fn calculate(pts : &[(usize, usize)], centroid : Option<(f32, f32)>, area : f32) -> Self {
        // area = pts.len()
        let CentralMoments { mut xx, mut xy, mut yy, mut xxx, mut yyy, mut xxy, mut yyx, .. } = CentralMoments::calculate(pts, centroid);
        let area2 = (1. / area).powf((0. + 2. + 2.) / 2.);
        let area3 = (1. / area).powf((1. + 2. + 2.) / 2.);
        xx /= area2;
        xy /= area2;
        yy /= area2;
        xxx /= area3;
        yyy /= area3;
        xxy /= area3;
        yyx /= area3;
        Self { xx, yy, xy, xxx, yyy, xxy, yyx }
    }

}

/*
IppStatus ippiGetHuMoments_64f(const IppiMomentState_64f* pState, int nChannel,
IppiHuMoment_64f pHm);
*/

// Calculate translation, scale and orientation-invariant moments from basic shapes (aka. Hu moments).
// Usually, the log of the moments is used. The moments are invariant to any in-plane rotations and
// scale changes.
pub struct IsotropicMoments {

    pub h1 : f32,

    pub h2 : f32,

    pub h3 : f32,

    pub h4 : f32,

    pub h5 : f32,

    pub h6 : f32,

    pub h7 : f32

}

impl IsotropicMoments {

    pub fn calculate(pts : &[(usize, usize)], centroid : Option<(f32, f32)>, area : f32) -> Self {
        let NormalizedMoments{ xx, xy, yy, xxx, yyy, xxy, yyx } = NormalizedMoments::calculate(pts, centroid, area);
        let h1 = xx + yy;
        let h2 = (xx - yy).powf(2.) + 4. * xy.powf(2.);
        let h3 = (xxx - 3.*yyx).powf(2.) + (3.*xxy - yyy).powf(2.);
        let h4 = (xxx + yyx).powf(2.) + (xxy + yyy).powf(2.);
        let h5 = (xxx - 3.*yyx) * (xxx + yyx) * ((xxx + yyx).powf(2.) - 3.*(xxy + yyy).powf(2.)) +
            (3.*xxy - yyy) * (xxy + yyy) * (3.*(xxx + yyx).powf(2.) - (xxy + yyy).powf(2.));
        let h6 = (xx - yy) * ((xxx + yyx).powf(2.) - (xxy + yyy).powf(2.)) +
            4.*xy*(xxx + yyx)*(xxy + yyy);
        let h7 = (3.*xxy - yyy)*(xxx + yyx)*((xxx+yyx).powf(2.) - 3.*(xxy+yyy).powf(2.)) +
            (3.*yyx - xxx)*(xxy + yyy)*(3.*(xxx + yyx).powf(2.) - (xxy + yyy).powf(2.));
        Self { h1, h2, h3, h4, h5, h6, h7 }
    }

}

#[cfg(feature="opencv")]
pub fn convex_hull(pts : &[(u16, u16)]) -> Option<Vec<(u16, u16)>> {

    use opencv::{core, imgproc};

    let mut pts_vec : core::Vector<core::Point2i> = core::Vector::new();
    for pt in pts.iter() {
        pts_vec.push(core::Point2i::new(pt.1 as i32, pt.0 as i32));
    }

    let mut pts_ix : core::Vector<i32> = core::Vector::new();
    let ans = imgproc::convex_hull(
        &pts_vec,
        &mut pts_ix,
        true,
        false
    );

    if let Ok(_) = ans {
        let mut out = Vec::new();
        for i in 0..pts_ix.len() {
            out.push(pts[pts_ix.get(i).unwrap() as usize]);
        }
        Some(out)
    } else {
        None
    }
}

/// Calculates the euclidian distance between two points. By convention the row coordinate will
/// come first, but since the distance is scalar (not vector) quantity, the order of the coordinate
/// isn't relevant.
pub fn point_euclidian(a : (usize, usize), b : (usize, usize)) -> f32 {
    // ((a.0 as f32 - b.0 as f32).powf(2.) + (a.1 as f32 - b.1 as f32).powf(2.)).sqrt()
    ((a.0 as f32 - b.0 as f32)).hypot((a.1 as f32 - b.1 as f32))
}

pub fn point_euclidian_u16(a : (u16, u16), b : (u16, u16)) -> f32 {
    //((a.0 as f32 - b.0 as f32).powf(2.) + (a.1 as f32 - b.1 as f32).powf(2.)).sqrt()
    ((a.0 as f32 - b.0 as f32)).hypot((a.1 as f32 - b.1 as f32))
}

pub fn point_euclidian_f32(a : (f32, f32), b : (f32, f32)) -> f32 {
    //((a.0 as f32 - b.0 as f32).powf(2.) + (a.1 as f32 - b.1 as f32).powf(2.)).sqrt()
    ((a.0 - b.0)).hypot((a.1 - b.1))
}

pub fn point_euclidian_float(a : (f64, f64), b : (f64, f64)) -> f32 {
    //((a.0 as f32 - b.0 as f32).powf(2.) + (a.1 as f32 - b.1 as f32).powf(2.)).sqrt()
    ((a.0 - b.0)).hypot((a.1 - b.1)) as f32
}

/// Calculate the angle formed by side_a and side_b given the opposite side using the law of cosines.
pub fn angle(side_a : f64, side_b : f64, opp_side : f64) -> f64 {
	let angle_cos = (side_a.powf(2.) + side_b.powf(2.) - opp_side.powf(2.)) / (2. * side_a * side_b);
	angle_cos.acos()
}

pub fn circle_points(center : Vector2<f64>, theta_diff : f64, radius : f64) -> Vec<Vector2<f64>> {
    assert!(theta_diff > 0. && theta_diff < 2.*std::f64::consts::PI);
    let mut unit_points = Vec::new();
    let n_angles = (2. * std::f64::consts::PI / theta_diff) as usize;
    for theta in (0..n_angles).map(|n| n as f64 * theta_diff) {
        unit_points.push(Vector2::from([theta.cos()*radius + center[0], theta.sin()*radius + center[1]]));
    }
    unit_points
}

pub fn circle_indices(center : (usize, usize), img_height : usize, theta_diff : f64, radius : f64) -> Vec<(usize, usize)> {
    let mut pts = circle_points(Vector2::from([center.1 as f64, (img_height - center.0) as f64]), theta_diff, radius);
    points_to_indices_dp(pts, img_height)
}

pub fn points_to_indices_dp(mut pts : Vec<Vector2<f64>>, img_height : usize) -> Vec<(usize, usize)> {
    pts.drain(..).map(|pt| (img_height as usize - pt[1] as usize, pt[0] as usize) ).collect()
}

pub fn points_to_indices_sp(mut pts : Vec<Vector2<f32>>, img_height : usize) -> Vec<(usize, usize)> {
    pts.drain(..).map(|pt| (img_height as usize - pt[1] as usize, pt[0] as usize) ).collect()
}

/*
Convex hull (Graham's scan) after Klette & Rosenfeld (2004):
(1) Start at a pivot point p that is known to be in the convex hull
(2) Sort the remaining points p_i in order of increasing angles. If the angle is
the same for more than one point, keep only the point furthest from p. Let the resulting
sorted sequence of points be q1..qm.
(3) Initialize the convex set C(S) by the edge between p and q1
(4) Scan through the sorted sequence. At each left turn, add a new edge to C(S). If there
is no turn (collinear points) skip the point. Backtrack at each right turn (i.e. remove the previous
points).

Rosenfeld-Pfaltz labeling algorithm labels all components of an image
in two scans (label propagation step and class equivalence step). Algo 2.2. of Klette & Rosenfeld.
This can be used for segmentation: (1) Median-filter an image (or max-filter, or min-filter, etc).
(2) For each resulting median: Label it.

*/
// The convex polygon of a contour is calculated by:
// (1) Iterate over all triplets of points
// (2) Remove points with angle > 180.
// (3) Repeat for triplets with a larger step far apart + 1
// Until there are no closed trianglges left.

/// Verifies if point is inside the circle.
pub fn circle_contains(circle : &((usize, usize), f32), pt : (usize, usize)) -> bool {
    point_euclidian(circle.0, pt) < circle.1
}

pub fn circle_from_diameter(a : (usize, usize), b : (usize, usize)) -> ((usize, usize), f32) {
    let r = point_euclidian(a, b) / 2.;

    // Guarantees reference point is always at x-positive quadrant.
    let (base_pt, ref_pt) = if a.1 < b.1 {
        (&a, &b)
    } else {
        (&b, &a)
    };
    let base = ref_pt.1 as f32 - base_pt.1 as f32;
    let side = ref_pt.0 as f32 - base_pt.0 as f32;
    let theta = side.atan2(base);

    let midpoint = (base_pt.0 + (theta.sin() * r) as usize, base_pt.1 + (theta.cos() * r) as usize);
    (midpoint, r)
}

pub fn point_distances(pts : &[(usize, usize)]) -> Vec<((usize, usize), (usize, usize), f32)> {
    let mut dists = Vec::new();
    for a in pts.iter() {
        for b in pts.iter() {
            if a != b {
                dists.push((*a, *b, point_euclidian(*a, *b)));
            }
        }
    }
    dists
}

pub fn outer_rect(pts : &[(usize, usize)]) -> (usize, usize, usize, usize) {

    let mut min_r = usize::MAX;
    let mut max_r = 0;
    let mut min_c = usize::MAX;
    let mut max_c = 0;
    
    for pt in pts {
        if pt.0 < min_r {
            min_r = pt.0;
        }
        if pt.1 < min_c {
            min_c = pt.1;
        }
        if pt.0 > max_r {
            max_r = pt.0;
        }
        if pt.1 > max_c {
            max_c = pt.1;
        }
    }
    
    (min_r, min_c, (max_r - min_r), (max_c - min_c))
    
}

/// Returns the outer circle that encloses a set of points (smallest circle with center at the shape
/// sutch that the shape is circumscribed in the circle).
pub fn outer_circle(pts : &[(usize, usize)]) -> ((usize, usize), f32) {
    assert!(pts.len() >= 2);
    let dists = point_distances(pts);
    let max_chord = dists.iter().max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
    circle_from_diameter(max_chord.0, max_chord.1)
}

/*/// Returns the inner circle by approximating it from the closest two points. This is cheaper,
/// but does not guarantee there will be other points of the shape inside it. It is at best an
/// upper bound for the inner circle.
pub fn approx_inner_circle(pts : &[(usize, usize)]) -> ((usize, usize), f32) {
    assert!(pts.len() >= 2);
    let dists = point_distances(pts);
    let min_chord = dists.iter().min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal)).unwrap();
    circle_from_diameter(min_chord.0, min_chord.1)
}*/

/// Returns center and radius of the largest inscribed circle with a pair of points
/// in the set that forms its diameter that does not contain any of the other points.
pub fn inner_circle(pts : &[(usize, usize)]) -> Option<((usize, usize), f32)> {
    assert!(pts.len() >= 2);
    let mut dists = point_distances(pts);

    // Order from largest to smallest distance.
    dists.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal) );

    // Remove points that repeat in more than one chord. This leaves us with
    // the largest chords such that no chord starts at the same point.
    let mut base_ix = 0;
    while base_ix < dists.len() - 1 {
        let (base_a, base_b) = (dists[base_ix].0, dists[base_ix].1);
        let mut ix = base_ix+1;
        while ix < dists.len() {
            let (a, b) = (dists[ix].0, dists[ix].1);
            if a == base_a || b == base_b || a == base_b || b == base_a {
                dists.remove(ix);
            } else {
                ix += 1;
            }
            // dbg!(base_ix, ix);
        }
        base_ix += 1;
    }

    if dists.len() < 2 {
        return None;
    }

    let mut ix = 0;
    while ix < dists.len() {
        let mut circle = circle_from_diameter(dists[ix].0, dists[ix].1);
        circle.1 = (circle.1 - (2.).sqrt()).max(1.);

        let mut point_inside = false;
        for test_ix in (0..ix).chain(ix+1..dists.len()) {
            if circle_contains(&circle, dists[test_ix].0) || circle_contains(&circle, dists[test_ix].1) {
                ix += 1;
                point_inside = true;
                break;
            }
        }

        if !point_inside {
            return Some(circle);
        }
    }

    None

}

#[test]
fn test_inner_circle() {
    let pts = [(22, 37), (22, 36), (23, 35), (23, 36), (23, 37), (23, 38), (24, 36), (25, 49), (25, 47), (25, 48), (25, 36),
    (25, 37), (25, 24), (25, 21), (25, 22), (26, 47), (26, 23), (26, 49), (26, 46), (26, 37), (26, 38), (26, 39),
    (26, 40), (26, 41), (26, 21), (26, 22), (26, 48), (26, 24), (26, 25), (26, 26), (27, 44), (27, 45), (27, 46), (27, 47),
    (27, 48), (27, 49), (27, 50), (27, 51), (27, 36), (27, 37), (27, 38), (27, 39), (27, 40), (27, 41), (27, 20), (27, 21), (27, 22),
    (27, 23), (27, 25), (27, 26), (27, 62), (28, 63), (28, 36), (28, 37), (28, 38), (28, 39), (28, 40), (28, 41), (28, 42),
    (28, 43), (28, 44), (28, 45), (28, 46), (28, 47), (28, 48), (28, 49), (28, 50), (28, 51), (28, 52)];
    println!("{:?}", inner_circle(&pts[..]));
}

/// Verifies if any rectangle is contained in the horizontal strip defined by the other
pub fn horizontally_aligned(r1 : &(usize, usize, usize, usize), r2 : &(usize, usize, usize, usize)) -> bool {
    let tl_1 = top_left_coordinate(r1);
    let tl_2 = top_left_coordinate(r2);
    let br_1 = bottom_right_coordinate(r1);
    let br_2 = bottom_right_coordinate(r2);
    (tl_1.0 >= tl_2.0 && tl_1.0 <= br_2.0) || (tl_2.0 >= tl_1.0 && tl_1.0 <= br_1.0)
}

/// Verifies if any rectangle is contained in the vertical strip defined by the other
pub fn vertically_aligned(r1 : &(usize, usize, usize, usize), r2 : &(usize, usize, usize, usize)) -> bool {
    let tl_1 = top_left_coordinate(r1);
    let tl_2 = top_left_coordinate(r2);
    let br_1 = bottom_right_coordinate(r1);
    let br_2 = bottom_right_coordinate(r2);
    (tl_1.1 >= tl_2.1 && tl_1.1 <= br_2.1) || (tl_2.1 >= tl_1.1 && tl_1.1 <= br_1.1)
}

pub fn rect_enclosing_pair(
    r1 : &(usize, usize, usize, usize), 
    r2 : &(usize, usize, usize, usize)
) -> (usize, usize, usize, usize) {
    let tl = (r1.0.min(r2.0), r1.1.min(r2.1));
    let br = ((r1.0 + r1.2).max(r2.0 + r2.2), (r1.1 + r1.3).max(r2.1 + r2.3));
    (tl.0, tl.1, br.0 - tl.0, br.1 - tl.1)
}

// This might be wrong?
pub fn rect_contacts2(r1 : &(usize, usize, usize, usize), r2 : &(usize, usize, usize, usize)) -> bool {

    // if rect_overlaps(r1, r2) {
    //    return false;
    // }

    let tl_1 = top_left_coordinate(r1);
    let tl_2 = top_left_coordinate(r2);
    let br_1 = bottom_right_coordinate(r1);
    let br_2 = bottom_right_coordinate(r2);

    ((br_1.0 as i32 - br_2.0 as i32 <= 1) && vertically_aligned(r1, r2)) ||
    ((br_1.1 as i32 - tl_2.1 as i32 <= 1) && horizontally_aligned(r1, r2)) ||
    ((tl_1.0 as i32 - br_2.0 as i32 <= 1) && vertically_aligned(r1, r2)) ||
    ((tl_1.1 as i32 - br_2.1 as i32 <= 1) && horizontally_aligned(r1, r2))
}

// This might be wrong?
pub fn rect_contacts(r1 : &(usize, usize, usize, usize), r2 : &(usize, usize, usize, usize)) -> bool {

    // if rect_overlaps(r1, r2) {
    //    return false;
    // }

    let tl_1 = top_left_coordinate(r1);
    let tl_2 = top_left_coordinate(r2);
    let br_1 = bottom_right_coordinate(r1);
    let br_2 = bottom_right_coordinate(r2);

    ((br_1.0 as i32 - br_2.0 as i32 <= 1) && vertically_aligned(r1, r2)) ||
    ((br_1.1 as i32 - tl_2.1 as i32 <= 1) && horizontally_aligned(r1, r2)) ||
    ((tl_1.0 as i32 - br_2.0 as i32 <= 1) && vertically_aligned(r1, r2)) ||
    ((tl_1.1 as i32 - br_2.1 as i32 <= 1) && horizontally_aligned(r1, r2))
}

pub fn top_left_coordinate<N>(r : &(N, N, N, N)) -> (N, N)
where
    N : Add<Output=N> + Copy
{
    (r.0, r.1)
}

pub fn bottom_right_coordinate<N>(r : &(N, N, N, N)) -> (N, N)
where
    N : Add<Output=N> + Copy
{
    (r.0 + r.2, r.1 + r.3)
}

/*#[test]
fn test_overlap() {
    let rects = [
        ((0, 0, 10, 10), (11, 11, 10, 10)),
        ((0, 0, 10, 10), (5, 5, 10, 10)),
        ((0, 0, 10, 10), (0, 5, 10, 10)),
        ((0, 0, 10, 10), (5, 0, 10, 10)),
        ((11, 11, 10, 10), (0, 0, 10, 10)),
        ((5, 5, 10, 10), (0, 0, 10, 10)),
        ((0, 5, 10, 10), (0, 0, 10, 10),),
        ((5, 0, 10, 10), (0, 0, 10, 10)),
    ];
    for (r1, r2) in rects {
        println!("{:?} {:?} = {:?}", &r1, &r2, shutter::feature::shape::rect_overlaps(&r1, &r2));
    }
}*/

pub fn rect_overlaps<N>(r1 : &(N, N, N, N), r2 : &(N, N, N, N)) -> bool
where
    N : Ord + Add<Output=N> + Copy
{

    r1.1 < (r2.1 + r2.3) && (r1.1 + r1.3) > r2.1 && (r1.0 < r2.0 + r2.2) && (r1.0 + r1.2) > r2.0

    /*let tl_vdist = (r1.0 as i32 - r2.0 as i32).abs();
    let tl_hdist = (r1.1 as i32 - r2.1 as i32).abs();
    let (h1, w1) = (r1.2 as i32, r1.3 as i32);
    let (h2, w2) = (r2.2 as i32, r2.3 as i32);
    (tl_vdist < h1 || tl_vdist < h2) && (tl_hdist < w1 || tl_hdist < w2)*/

    /*let tl_1 = top_left_coordinate(r1);
    let tl_2 = top_left_coordinate(r2);
    let br_1 = bottom_right_coordinate(r1);
    let br_2 = bottom_right_coordinate(r2);

    let to_left = br_2.1 < tl_1.1;
    let to_right = tl_2.1 > br_1.1;
    let to_top = br_2.0 < tl_1.0;
    let to_bottom = tl_2.0 > br_1.0;

    !(to_left || to_top || to_right || to_bottom)*/

    /*if tl_1.1 >= br_2.1 || tl_2.1 >= br_1.1 {
        return false;
    }

    if br_1.0 >= tl_2.0 || br_2.0 >= tl_1.0 {
        return false;
    }

    true*/
}

/// Assuming pts represents a closed shape, calculate its perimeter.
pub fn contour_perimeter<N>(pts : &[(N, N)]) -> f32
where
    usize : From<N>,
    N : Copy
{

    if pts.len() < 1 {
        return 0.0;
    }

    let mut perim = 0.;
    let n = pts.len();
    for (p1, p2) in pts[0..(n-1)].iter().zip(pts[1..n].iter()) {
        perim += point_euclidian((usize::from(p1.0), usize::from(p1.1)), (usize::from(p2.0), usize::from(p2.1)));
    }
    perim
}

/*/// Assuming pts represents a closed shape, calculate its area.
pub fn contour_area(pts : &[(usize, usize)], outer_rect : (usize, usize, usize, usize)) -> f32 {
    let mut area = outer_rect.2 * outer_rect.3;
    for row in rect.0..(rect.0+rect.2) {

    }
}*/

/// A cricle has circularity of 1; Other circular polygons (ellipses, conic sections) have circularity < 1.
/// (Isoperimetric inequality).
/// A circle has the largest area among all round shapes (i.e. shapes for which a circumference can be calculated)
/// with the same circumference.
pub fn circularity(area : f32, circumf : f32) -> Option<f32> {
    let circ = (4. * std::f32::consts::PI * area) / circumf.powf(2.);

    // This condition will fail when joining the points of the shape fails (i.e. shape path crosses
    // over any of the already-formed paths.
    if circumf < 1.0 {
        Some(circ)
    } else {
        None
    }

}

// Roundess gives how close to a circle an arbitrary polygon is. It is the ratio between the inscribed
// and circumscribed cricles of the shape.
// https://en.wikipedia.org/wiki/Roundness
pub fn roundness(pts : &[(usize, usize)]) -> Option<f32> {
    let (_, inner_radius) = inner_circle(&pts)?;
    let (_, outer_radius) = outer_circle(&pts);
    let round =  inner_radius / outer_radius;
    assert!(round < 1.0);
    Some(round)
}

// Measures how close to a closed shape a blob is.
// pub fn convexity() -> f64 {
//    area_blob / area_convex_hull
// }

// Polygon is just an edge that is interpreted to be closed.
#[derive(Debug, Clone)]
pub struct Polygon(Vec<(usize, usize)>);

impl From<Vec<(usize, usize)>> for Polygon {

    fn from(pts : Vec<(usize, usize)>) -> Self {
        assert!(pts.len() >= 3);
        Self(pts)
    }
}

impl From<Edge> for Polygon {

    fn from(edge : Edge) -> Self {
        assert!(edge.len() >= 3);
        Self(edge.into())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Triangle([(usize, usize); 3]);

impl Triangle {

    pub fn lines(&self) -> [Line; 3] {
        let l1 = Line::from([self.0[0], self.0[1]]);
        let l2 = Line::from([self.0[0], self.0[2]]);
        let l3 = Line::from([self.0[1], self.0[2]]);
        [l1, l2, l3]
    }

    /// Splits this triangle into two right triangles
    pub fn split_to_right(&self) -> (Triangle, Triangle) {
        let base = self.base();
        let perp = perp_line(&self, &base);
        let (b1, b2) = base.points();
        let (perp_pt, top) = perp.points();
        (Triangle([b1, top, perp_pt]), Triangle([b2, top, perp_pt]))
    }

    // Gets the triangle "base" (largest line)
    pub fn base(&self) -> Line {
        let mut lines = self.lines();
        lines.sort_by(|l1, l2| l1.length().partial_cmp(&l2.length()).unwrap_or(Ordering::Equal) );
        lines[2]
    }

    pub fn outer_rect(&self) -> (usize, usize, usize, usize) {
        let (r1, r2) = self.split_to_right();
        unimplemented!()
    }

    pub fn area(&self) -> f64 {
        let base = self.base();
        let perp = perp_line(&self, &base);
        base.length() * perp.length() / 2.
    }

    pub fn perimeter(&self) -> f64 {
        self.lines().iter().fold(0.0, |p, l| p + l.length() )
    }

    pub fn contains(&self, pt : &(usize, usize)) -> bool {

        // Verify if either left or right outer rect contains point. If neither contain it, returns false. Else:

        // If left contains it, verify if point is closer to outer corner (outside) or inner cordre (inside)

        // If right contains it, verify if point is closer to outer (outside) or inner corner (inside).

        unimplemented!()
    }

}

fn perp_line(tri : &Triangle, base : &Line) -> Line {
    let (b1, b2) = base.points();
    let top = tri.0.iter().find(|pt| **pt != b1 && **pt != b2 ).unwrap();
    base.perpendicular(*top)
}

impl Polygon {

    pub fn area(&self) -> f64 {
        self.triangles().iter().fold(0.0, |area, tri| area + tri.area() )
    }

    pub fn triangles<'a>(&'a self) -> Vec<Triangle> {
        let n = self.0.len();
        let last = self.0.last().unwrap().clone();
        let mut triangles = Vec::new();
        for (a, b) in self.0.iter().take(n-1).zip(self.0.iter().skip(1)) {
            triangles.push(Triangle([*a, *b, last]));
        }
        triangles
    }

    /// Check if point is inside polygon
    pub fn contains(&self, pt : (usize, usize)) -> bool {
        self.triangles().iter().any(|tri| tri.contains(&pt) )
    }

    // True when any of the triangles intersect. Triangles intersect when any of the points
    // of one triangle are inside the other.
    pub fn intersects(&self, other : &Self) -> bool {
        unimplemented!()
    }

    pub fn vertices<'a>(&'a self) -> impl Iterator<Item=&'a (usize, usize)> + 'a {
        self.0.iter()
    }

    pub fn vertex_pairs<'a>(&'a self) -> impl Iterator<Item=(&'a (usize, usize), &'a (usize, usize))> + 'a {
        self.0.iter().take(self.0.len()-1).zip(self.0.iter().skip(1))
    }

    // pub fn vertices(&self) -> impl Iterator<Item=(usize, usize)

    pub fn outer_rect(&self) -> (usize, usize, usize, usize) {
        unimplemented!()
    }

    pub fn inner_rect(&self) -> (usize, usize, usize, usize) {
        unimplemented!()
    }

    pub fn join(&mut self, other : &Self) {
        // Works when shapes intersect
    }

    pub fn perimeter(&self) -> f64 {
        let n = self.0.len();
        let mut per = 0.0;
        for ((a1, a2), (b1, b2)) in self.vertex_pairs() {
            per += euclidian(&[*a1 as f64, *a2 as f64], &[*b1 as f64, *b2 as f64]);
        }
        per
    }

    /// Check if this polygon completely contains another.
    pub fn encloses(&self, other : &Polygon) -> bool {
        other.0.iter().all(|pt| self.contains(*pt) )
    }

}

fn euclidian(a : &[f64], b : &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| (ai - bi).powf(2.) ).sum::<f64>().sqrt()
}

// impl AsRef<[(usize, usize)]> for Edge { }
// impl AsRef<[(usize, usize)]> for Polygon { }

/// Assume pts is an ordered vector over columns. Join any two points as long as they are close enough.
pub fn join_single_col_ordered(pts : &[(usize, usize)], max_dist : f64) -> Vec<[(usize, usize); 2]> {
    let mut clusters = Vec::new();
    if pts.len() <= 1 {
        return clusters;
    }
    for (p1, p2) in pts[0..pts.len()-1].iter().zip(pts[1..pts.len()].iter()) {
        // assert ordering
        assert!(p2.1 as i32 - p1.1 as i32 >= 0);
        if Manhattan::metric(&(p1.0 as f64, p1.1 as f64), &(p2.0 as f64, p2.1 as f64)) < max_dist {
            clusters.push([*p1, *p2]);
        }
    }
    clusters
}

pub fn join_pairs_col_ordered(pairs : &[[(usize, usize); 2]], max_dist : f64) -> Vec<[(usize, usize); 4]> {
    let mut clusters = Vec::new();
    if pairs.len() <= 1 {
        return clusters;
    }
    for (c1, c2) in pairs[0..pairs.len()-1].iter().zip(pairs[1..pairs.len()].iter()) {

        // assert ordering between extreme elements.
        assert!(c2[0].1 as i32 - c1[1].1 as i32 >= 0);
        if Manhattan::metric(&(c2[0].0 as f64, c2[0].1 as f64), &(c1[1].0 as f64, c1[1].1 as f64)) < max_dist {
            clusters.push([c1[0], c1[1], c2[0], c2[1]]);
        }
    }
    clusters
}

// Define if points lie on a diagonal
pub fn same_diagonal(a : (usize, usize), b : (usize, usize)) -> bool {
    (a.0  as i32 - b.0 as i32).abs() == (a.1 as i32 - b.1 as i32).abs()
}

pub fn colinear_pair(a : (usize, usize), b : (usize, usize)) -> bool {
    a.0 == b.0 || a.1 == b.1 || same_diagonal(a, b)
}

pub fn colinear_triplet(a : (usize, usize), b : (usize, usize), c : (usize, usize)) -> bool {
    (a.0 == b.0 && a.0 == c.0) || (a.1 == b.1 && b.1 == c.1 ) || (same_diagonal(a, b) && same_diagonal(a, c))
}

/// Returns the angle at the vertex p1 in the triangle [p1, p2, p3] using the law of cosines.
/// Reference https://stackoverflow.com/questions/1211212/how-to-calculate-an-angle-from-three-points
/// We invert the y-axis for all points to situate them in the analytical plane.
pub fn vertex_angle(pt1 : (usize, usize), pt2 : (usize, usize), pt3 : (usize, usize)) -> Option<f64> {

    if pt1 == pt2 || pt1 == pt3 || pt2 == pt3 {
        return None;
    }
    
    if colinear_triplet(pt1, pt2, pt3) {
        return None;
    }
    
    let dist_12 = euclidian(&[-1. * pt1.0 as f64, pt1.1 as f64], &[-1. * pt2.0 as f64, pt2.1 as f64]);
    let dist_13 = euclidian(&[-1. * pt1.0 as f64, pt1.1 as f64], &[-1. * pt3.0 as f64, pt3.1 as f64]);
    let dist_23 = euclidian(&[-1. * pt2.0 as f64, pt2.1 as f64], &[-1. * pt3.0 as f64, pt3.1 as f64]);
    // println!("sides = {:?}", (dist_12, dist_13, dist_23));
    // println!("num = {:?}", dist_12.powf(2.) + dist_13.powf(2.) - dist_23.powf(2.));
    // println!("den = {}", 2.*dist_12*dist_13);
    let c = (dist_12.powf(2.) + dist_13.powf(2.) - dist_23.powf(2.)) / (2.*dist_12*dist_13);
    // println!("div = {}", c);
    
    // If this fails, the user might have informed an invalid triangle.
    if c >= -1. && c <= 1. {
        Some(c.acos())
    } else {
        None
    }
    
}

/// Returns the side ab given vertex angles theta1 (ab), and the remaining sides b and c using the law of sines.
/// Reference https://en.wikipedia.org/wiki/Triangle#Sine,_cosine_and_tangent_rules
pub fn vertex_side(theta1 : f64, b : f64, c : f64) -> f64 {
    b.powf(2.) - c.powf(2.) - 2. * b * c * theta1.cos()
}

#[test]
fn line_intersect() {
    println!("{:?}", line_intersection_usize(((0, 0), (5, 5)), ((5, 0), (0, 5))) );
}

#[test]
fn vertex_opening() {
    println!("{:?}", vertex_angle((0, 0), (0, 5), (5, 0)));
    println!("{:?}", vertex_angle((1, 0), (0, 1), (1, 2)));
    // println!("{:?}", vertex_angle((1, 0), (0, 1), (1, 2)));
}

/// Calculate line intersection
/// Reference https://mathworld.wolfram.com/Line-LineIntersection.html
/// m1 = [x1 y1; x2 y2]
/// m2 = [x3 y3; x4 y4]
/// m3 = [x1 - x2, y1 - y2; x3 - x4, y3 - y4]
/// Calculates intersection of two lines, where each line is defined by two points
/// stacked into a 2x2 matrix (one at each row as [[x y], [x,y]]).
pub fn line_intersection(m1 : Matrix2<f64>, m2 : Matrix2<f64>) -> Option<Vector2<f64>> {
    // let diff_x_l1 = line1_pt1.1 as f64 - line1_pt2.1 as f64;
    let diff_x_l1 = m1[(0,0)] - m1[(1,0)];

    // let diff_x_l2 = line2_pt1.1 as f64 - line2_pt2.1 as f64;
    let diff_x_l2 = m2[(0,0)] - m2[(1,0)];

    // let diff_y_l1 = (line1_pt1.0 as f64 * -1.) - (line1_pt2.0 as f64 * -1.);
    let diff_y_l1 = m1[(0,1)] - m1[(1,1)];

    // let diff_y_l2 = (line2_pt1.0 as f64 * -1.) - (line2_pt2.0 as f64 * -1.);
    let diff_y_l2 = m2[(0,1)] - m2[(1,1)];

    let m3 = Matrix2::from_rows(&[
        RowVector2::new(diff_x_l1, diff_y_l1),
        RowVector2::new(diff_x_l2, diff_y_l2)
    ]);

    let m1_det = m1.determinant();
    let m2_det = m2.determinant();
    let m3_det = m3.determinant();

    let x = Matrix2::from_rows(&[
        RowVector2::new(m1_det, diff_x_l1),
        RowVector2::new(m2_det, diff_x_l2)
    ]).determinant() / m3_det;

    let mut y = Matrix2::from_rows(&[
        RowVector2::new(m1_det, diff_y_l1),
        RowVector2::new(m2_det, diff_y_l2)
    ]).determinant() / m3_det;

    if !x.is_nan() && !y.is_nan() {
        Some(Vector2::new(x, y))
    } else {
        None
    }
}

pub fn line_intersection_usize(
    (line1_pt1, line1_pt2) : ((usize, usize), (usize, usize)),
    (line2_pt1, line2_pt2) : ((usize, usize), (usize, usize))
) -> Option<(usize, usize)> {
    let m1 = Matrix2::from_rows(&[
        RowVector2::new(line1_pt1.1 as f64, line1_pt1.0 as f64 * -1.),
        RowVector2::new(line1_pt2.1 as f64, line1_pt2.0 as f64 * -1.)
    ]);
    let m2 = Matrix2::from_rows(&[
        RowVector2::new(line2_pt1.1 as f64, line2_pt1.0 as f64 * -1.),
        RowVector2::new(line2_pt2.1 as f64, line2_pt2.0 as f64 * -1.)
    ]);
    let out = line_intersection(m1, m2)?;
    let (x, mut y) = (out[0], out[1]);
    if y <= 0. {
        y *= -1.;
    }
    if !x.is_nan() && !y.is_nan() {
        Some(((y*-1.) as usize, x as usize))
    } else {
        None
    }
}

/// Slope is just height / width (for the line bounding box). Invert row (y) axis
/// to the geometric plane.
fn slope(pt1 : (usize, usize), pt2 : (usize, usize)) -> f64 {
    (pt1.0 as f64 * -1. - pt2.0 as f64 * -1.) / (pt1.1 as f64 - pt2.1 as f64)
}

/// Iterate over inner and border pixels.
pub fn rect_dense_iter(rect : (usize, usize, usize, usize)) -> impl Iterator<Item=(usize, usize)> + Clone {
    (0..=rect.2)
        .map(move |r| (0..=rect.3).map(move |c| (rect.0 + r, rect.1+c) ) )
        .flatten() 
}

/// Approximate an ellipse by taking two pairs of points with the largest vertical
/// and largest horizontal ellipse. If vert > horizontal distance, the ellipse will be tall;
/// if horiz > vert distance, the ellipse will be wide.
pub fn ellipse_extremes(
    pts : &[(usize, usize)]
) -> Option<(((usize, usize), (usize, usize)), ((usize, usize), (usize, usize)))> {
    let mut largest_horiz_distance : ((usize, usize), (usize, usize), f64) = ((0, 0), (0, 0), 0.0);
    let mut largest_vert_distance : ((usize, usize), (usize, usize), f64) = ((0, 0), (0, 0), 0.0);
    for pt1 in pts.iter() {
        for pt2 in pts.iter() {
            let v_dist = (pt1.0 as f64 - pt2.0 as f64).abs();
            let h_dist = (pt1.1 as f64 - pt2.1 as f64).abs();
            if v_dist > largest_vert_distance.2 {
                largest_vert_distance = (*pt1, *pt2, v_dist);
            }
            if h_dist > largest_horiz_distance.2 {
                largest_horiz_distance = (*pt1, *pt2, h_dist);
            }
        }
    }
    Some((
        (largest_horiz_distance.0, largest_horiz_distance.1),
        (largest_vert_distance.0, largest_vert_distance.1)
    ))
}

pub fn ellipse_axis_pair(
    pts : &[(usize, usize)]
) -> Option<(((usize, usize), (usize, usize)), ((usize, usize), (usize, usize)))> {
    /// Ellect the major ellipsis to be the farthest pair of points in the set.
    let mut largest_major_distance : ((usize, usize), (usize, usize), f64) = ((0, 0), (0, 0), 0.0);
    for pt1 in pts.iter() {
        for pt2 in pts.iter() {
            let dist = euclidian(&[pt1.0 as f64, pt1.1 as f64], &[pt2.0 as f64, pt2.1 as f64]);
            if dist > largest_major_distance.2 {
                largest_major_distance = (*pt1, *pt2, dist);
            }
        }
    }
    let (major_pt1, major_pt2) = (largest_major_distance.0, largest_major_distance.1);
    let major_slope = slope(major_pt1, major_pt2);
    let large_axis_dist = euclidian(&[major_pt1.0 as f64, major_pt1.1 as f64], &[major_pt2.0 as f64, major_pt2.1 as f64]);

    let mut perp_minor_line : ((usize, usize), (usize, usize), f64) = ((0, 0), (0, 0), std::f64::INFINITY);
    let mut found_minor = false;

    let major_midpoint = ((major_pt1.0 + major_pt2.0) / 2, (major_pt1.1 + major_pt2.1) / 2);
    for pt1 in pts.iter() {
        for pt2 in pts.iter() {

            let not_same = (pt1.0 != major_pt1.0 && pt2.0 != major_pt2.0 && pt1.1 != major_pt1.1 && pt2.1 != major_pt2.1) &&
                (pt2.0 != major_pt1.0 && pt1.0 != major_pt2.0 && pt2.1 != major_pt1.1 && pt1.1 != major_pt2.1);

            if not_same {
                let test_slope = slope(*pt1, *pt2);

                if let Some(intersection) = line_intersection_usize((major_pt1, major_pt2), (*pt1, *pt2)) {

                    // let dist_to_mid = euclidian(
                    //    &[intersection.0 as f64, intersection.1 as f64],
                    //    &[major_midpoint.0 as f64, major_midpoint.1 as f64]
                    // );
                    // println!("Dist to mid = {:?}", dist_to_mid);

                    // if dist_to_mid <= 32.0 {
                        // let pt1_ref_center = (pt1.0 as i32 - intersection.0 as i32, pt1.1 as i32 - intersection.1 as i32);
                        // let pt2_ref_center = (pt2.0 as i32 - intersection.0 as i32, pt2.1 as i32 - intersection.1 as i32);

                        // If they are at the same quadrant both vertically and horizontally, the points
                        // offset by the center will have same same sign.
                        // let reflected_vertically = (pt1_ref_center.0.signum() * pt2_ref_center.0.signum()) <= 0;
                        // let reflected_horizontally = (pt1_ref_center.1.signum() * pt2_ref_center.1.signum()) <= 0;

                        // if reflected_vertically || reflected_horizontally {

                        // Known up to the sign, which can either be positive or negative
                        let angle_tan = (major_slope - test_slope) / (1. + major_slope*test_slope);

                        let dist = euclidian(&[pt1.0 as f64, pt1.1 as f64], &[pt2.0 as f64, pt2.1 as f64]);

                        // This is taken from the maximum expected asymetry (minor_axis / major_axis).
                        if dist >= large_axis_dist * 0.75 {
                            // Test to see if tan(angle) is close to pi/2, in which case this pair of
                            // points is nearly perpendicular to the major ellipsis. If pair is closer
                            // to being perpendicular than last pair, (suggesting we
                            // are closer to the center of the ellipsis), this is the best guess for the minor axis of the
                            // ellipse.
                            let angle_from_perp = (angle_tan.atan().abs() - std::f64::consts::PI).abs();

                            // Get closest to perpendicular
                            if angle_from_perp < perp_minor_line.2 {
                                perp_minor_line = (*pt1, *pt2, angle_from_perp);
                                found_minor = true;
                            }
                        }

                        // Get closest to midpoint
                        // Alternatively, we check the normal from the midpoint of the major axis, and extend
                        // it to one of the external convex points.
                    }
                }
                // }
            // }
        }
    }
    if !found_minor {
        return None;
        println!("Did not found minor");
    }

    let (minor_pt1, minor_pt2) = (perp_minor_line.0, perp_minor_line.1);

    Some(((major_pt1, major_pt2), (minor_pt1, minor_pt2)))
}

/// Assumes set of points is convex. Calculate the best enclosing ellipse.
pub fn outer_ellipse(pts : &[(usize, usize)]) -> Option<Ellipse> {

    let ((major_pt1, major_pt2), (minor_pt1, minor_pt2)) = ellipse_axis_pair(pts)?;

    // Set minor axis to be the normal projecting from the major axis at the
    // point the line of perp_minor_line intersects with the major axis.

    let center = line_intersection_usize((major_pt1, major_pt2), (minor_pt1, minor_pt2))?;
    let large_axis = euclidian(&[major_pt1.0 as f64, major_pt1.1 as f64], &[major_pt2.0 as f64, major_pt2.1 as f64]);
    let small_axis = euclidian(&[minor_pt1.0 as f64, minor_pt1.1 as f64], &[minor_pt2.0 as f64, minor_pt2.1 as f64]);

    // Calculate angle of center with respect to x axis.
    let angle = vertex_angle(center, major_pt2, (center.0, center.1 + small_axis as usize)).unwrap();

    Some(Ellipse {
        center : (center.0 as f64, center.1 as f64),
        large_axis,
        small_axis,
        angle
    })
}

pub fn join_col_ordered(pts : &[(usize, usize)], max_dist : f64) -> Vec<[(usize, usize); 4]> {
    let pairs = join_single_col_ordered(pts, max_dist);
    join_pairs_col_ordered(&pairs[..], max_dist)
}

#[derive(Debug, Clone, Copy)]
pub struct Ellipse {
    pub center : (f64, f64),
    pub large_axis : f64,
    pub small_axis : f64,
    pub angle : f64
}

#[derive(Debug, Clone, Copy)]
pub struct EllipseAxis {
    pub major : (f64, f64),
    pub minor : (f64, f64)
}

impl EllipseAxis {

    pub fn vectors(&self, center : (f64, f64), img_height : usize) -> (Vector2<f64>, Vector2<f64>) {
        let major = Vector2::from([
            self.major.1 as f64 - center.1 as f64,
            img_height as f64 - self.major.0 as f64 - center.0 as f64
        ]);
        let minor = Vector2::from([
            self.minor.1 as f64 - center.1 as f64,
            img_height as f64 - self.minor.0 as f64 - center.0 as f64
        ]);
        (major, minor)
    }

}

impl Ellipse {

    /* Returns the affine matrix that maps points (assumed to be centered relative to this
    ellipse center) according to the major and minor axis of the ellipse. If the points are
    in the unit circe centered at the ellipise, this matrix gives the ellipse edges. */
    pub fn affine_matrix(&self, img_height : usize) -> Option<Matrix2<f64>> {
        let axis = self.axis()?;
        let (major, minor) = axis.vectors(self.center, img_height);
        Some(Matrix2::from_columns(&[major, minor]))
    }

    pub fn center_vector(&self, img_height : usize) -> Vector2<f64> {
        Vector2::from([self.center.1 as f64, (img_height as f64 - self.center.0) as f64])
    }

    pub fn from_axis(center : (f64, f64), axis : EllipseAxis, img_height : usize) -> Self {
        let (major, minor) = axis.vectors(center, img_height);

        // The angle in Opencv rotated rect is defined as the clockwise angle
        // between one of the axis and the horizontal axis. Adding 90 degrees
        // generates the other axis.
        let unit_x = Vector2::from([1.0, 0.0]);
        let angle = unit_x.angle(&major).to_degrees();
        Ellipse {
            center,
            large_axis : major.magnitude_squared(),

            // TODO minor?
            small_axis : major.magnitude_squared(),
            angle
        }
    }

    pub fn edge(&self, img_height : usize, theta_diff : f64) -> Option<Vec<(usize, usize)>> {
        let mut unit_pts = circle_points(Vector2::from([0.0, 0.0]), theta_diff, 1.0);
        let center = self.center_vector(img_height);
        let mtx = self.affine_matrix(img_height)?;
        let out : Vec<_> = unit_pts
            .drain(..)
            .map(|pt| mtx * pt + center )
            .map(|pt| (img_height - pt[1] as usize, pt[0] as usize) )
            .collect();
        Some(out)
    }

    /// Returns radial distance of the given point. If r < 1.0, point is inside the ellipse. If r == 1, point is
    /// exactly at edge of ellipse. If r > 1.0, point is outside ellipse.
    pub fn point_dist_radii(&self, pt : (usize, usize), img_height : usize) -> Option<f64> {
        let axis = self.axis()?;
        let (major, minor) = axis.vectors(self.center, img_height);
        let rot = Rotation2::new(self.positive_angle().to_radians());
        let rot_major = rot * &major;
        let rot_minor = rot * &minor;
        let pt = Vector2::from([
            pt.1 as f64 - self.center.1 as f64,
            img_height as f64 - pt.0 as f64 - self.center.0 as f64
        ]);
        let rot_pt = rot * &pt;

        // Derived from ellipse equation, (x-xc)^2/semi_major^2 + (y-yc)^2/semi_minor^2 = 1
        // (but ignoring (xc, yc) since everything was already centered at the conversion stage)
        Some((rot_pt[0].powf(2.) / rot_major[0].powf(2.) + rot_pt[1].powf(2.) / rot_minor[1].powf(2.)))
    }

    pub fn contains(&self, pt : (usize, usize), img_height : usize) -> Option<bool> {
        self.point_dist_radii(pt, img_height).map(|dist| dist <= 1.0 )
    }

    pub fn positive_angle(&self) -> f64 {
        let mut angle = self.angle;
        while angle > 90.0 {
            angle -= 90.;
        }
        angle
    }

    pub fn axis(&self) -> Option<EllipseAxis> {
        // println!("{}", self.angle);
        if self.angle <= 90. {
            let major = (
                self.center.0 + (-1. * (-self.angle).to_radians().sin() * (self.large_axis / 2.)),
                self.center.1 + ((-self.angle).to_radians().cos() * (self.large_axis / 2.))
            );
            let minor = (
                self.center.0 - (-1. * (-90. as f64 + self.angle).to_radians().sin() * (self.small_axis / 2.)),
                self.center.1 + ( (-90. as f64 + self.angle).to_radians().cos() * (self.small_axis / 2.))
            );
            Some(EllipseAxis { major, minor })
        } else {
            let mut p = self.clone();
            p.small_axis = self.large_axis;
            p.large_axis = self.small_axis;
            p.angle -= 90.;
            p.axis()
            // if self.angle > 90. && self.angle < 180.0 {
            //    let mut p = self.clone();
                //p.angle = p.angle - 90.;
                //p.axis()
            // } else {
            //    panic!()
            // }
        }
    }

    /*fn major_axis_points(&self) -> (usize, usize) {
        // large_axis * angle.to_radians().sin()
        unimplemented!()
    }

    fn minor_axis_points(&self) -> (usize, usize) {
        // small_axis * angle.to_radians().cos()
        unimplemented!()
    }*/
}

pub fn outer_square_for_circle(center : (usize, usize), radius : usize) -> Option<(usize, usize, usize, usize)> {
    let tl = (center.0.checked_sub(radius)?, center.1.checked_sub(radius)?);
    Some((tl.0, tl.1, 2*radius, 2*radius))
}

/// Returns the square the circle encloses.
pub fn inner_square_for_circle(center : (usize, usize), radius : usize) -> Option<(usize, usize, usize, usize)> {
    // Those quantities will be the same for squares, but can be generalized
    // later for axis-aligned ellipsoids for a non-symmetrical rect, by informing two
    // radii values. sin(45) = cos(45) for squares, but the angle will be different for
    // generic rects.
    let half_width = (std::f64::consts::FRAC_PI_4.cos() * radius as f64) as usize;
    let half_height = (std::f64::consts::FRAC_PI_4.sin() * radius as f64) as usize;
    Some((
        center.0.checked_sub(half_height)?,
        center.1.checked_sub(half_width)?,
        2*half_height,
        2*half_width
    ))
}

pub fn win_contains_circle(win_sz : &(usize, usize), circ : ((usize, usize), f32)) -> bool {
    (circ.0.0 + circ.1 as usize) < win_sz.0 && (circ.0.1 + circ.1 as usize) < win_sz.1
}

/// Returns the smallerst circle with center at the square such that the share is circumscribed in the shape.
pub fn outer_circle_for_square(
    sq : &(usize, usize, usize, usize),
    win_sz : (usize, usize)
) -> Option<((usize, usize), f32)> {
    if sq.2 > win_sz.0 || sq.3 > win_sz.1 {
        return None;
    }
    let tl = top_left_coordinate(sq);
    let br = bottom_right_coordinate(sq);
    let circ = outer_circle(&[tl, br]);
    if win_contains_circle(&win_sz, circ) {
        Some(circ)
    } else {
        None
    }
}

pub fn inner_circle_for_square(sq : &(usize, usize, usize, usize)) -> ((usize, usize), f32) {
    let center = (sq.0 + sq.2 / 2, sq.1 + sq.3 / 2);
    let radius = sq.2.min(sq.3) as f32 / 2.;
    (center, radius)
}

pub fn circumference_iter(circ : ((usize, usize), f32), step_rad : f32) -> impl Iterator<Item=(usize, usize)> {
    use std::f32::consts::PI;
    assert!(step_rad < 2. * PI);
    let n_steps = (2. * PI / step_rad) as usize;
    (0..n_steps).filter_map(move |step| {
        let (mut y, mut x) = ((step as f32 * step_rad).sin(), (step as f32 * step_rad).cos());
        y *= circ.1;
        x *= circ.1;
        y += circ.0.0 as f32;
        x += circ.0.1 as f32;
        if y > 0.0 && x > 0.0 {
            Some((y as usize, x as usize))
        } else {
            None
        }
    })
}

/*
Explores the projection matrix that explains the best projective transform between two planes.
See https://docs.opencv.org/3.4/d9/dab/tutorial_homography.html
*/
#[cfg(feature="opencv")]
pub fn find_homography(flat : &[(usize, usize)], proj : &[(usize, usize)], img_height : usize) -> Result<nalgebra::Matrix3<f64>, String> {

    use opencv::calib3d;
    use opencv::prelude::MatTrait;

    // Position y axis on the analytical plane.
    let flat : Vec<_> = flat.iter().map(|pt| (img_height - pt.0, pt.1) ).collect();
    let proj : Vec<_> = proj.iter().map(|pt| (img_height - pt.0, pt.1) ).collect();
    let mut flat = convert_points_to_opencv_float(&flat);
    let mut proj = convert_points_to_opencv_float(&proj);

    let mat = calib3d::find_homography(
        &flat,
        &proj,
        &mut opencv::core::no_array().unwrap(),
        calib3d::RANSAC,
        1.0
    ).map_err(|e| format!("{}",e ) )?;

    let mut out = nalgebra::Matrix3::zeros();

    for i in 0..3 {
        for j in 0..3 {
            out[(i, j)] = *mat.at_2d::<f64>(i as i32, j as i32).unwrap();
        }
    }

    /*let mut rotations = core::Vector::<core::Mat>::new();
    let mut translations = core::Vector::<core::Mat>::new();
    let mut normals = core::Vector::<core::Mat>::new();
    let dec_ans = calib3d::decompose_homography_mat(
        &homography_mat,
        &intrinsic_mat,
        &mut rotations,
        &mut translations,
        &mut normals
    );
    let mut possible_solutions = core::Vector::<i32>::new();
    calib3d::filter_homography_decomp_by_visible_refpoints(
        &rotations,
        &normals,
        &circ_pts,
        &ellipse_pts,
        &mut possible_solutions,
        &core::no_array().unwrap()
    ) -> Result<()>;*/

    Ok(out)
}

#[cfg(feature="opencv")]
pub fn convert_points_to_opencv_int(pts : &[(usize, usize)]) -> opencv::core::Vector<opencv::core::Point2i> {
    let mut pt_vec = opencv::core::Vector::new();
    for pt in pts.iter() {
        pt_vec.push(opencv::core::Point2i::new(pt.1 as i32, pt.0 as i32));
    }
    pt_vec
}

#[cfg(feature="opencv")]
pub fn convert_points_to_opencv_float(pts : &[(usize, usize)]) -> opencv::core::Vector<opencv::core::Point2f> {
    let mut pt_vec = opencv::core::Vector::new();
    for pt in pts.iter() {
        pt_vec.push(opencv::core::Point2f::new(pt.1 as f32, pt.0 as f32));
    }
    pt_vec
}

#[cfg(feature="opencv")]
pub mod cvellipse {

    use super::*;
    use opencv::core;
    use opencv::imgproc;
    use opencv::prelude::RotatedRectTrait;

    /*pub fn convert_points(pts : &[(usize, usize)]) -> core::Vector<core::Point2i> {
        let mut pt_vec = core::Vector::new();
        for pt in pts.iter() {
            pt_vec.push(core::Point2i::new(pt.1 as i32, pt.0 as i32));
        }
        pt_vec
    }*/

    pub fn fit_circle(pts : &[(usize, usize)], method : Method) -> Result<((usize, usize), usize), String> {
        let ellipse = EllipseFitting::new().fit_ellipse(pts, method)?;
        let radius = ((ellipse.large_axis*0.5 + ellipse.small_axis*0.5) / 2.) as usize;

        if ellipse.center.0 < 0. || ellipse.center.1 < 0. {
            return Err(format!("Invalid value"));
        }

        Ok(((ellipse.center.0 as usize, ellipse.center.1 as usize), radius))
    }

    // TODO make WindowMut
    pub fn draw_ellipse(window : Window<'_, u8>, el : &Ellipse, thickness : i32, color : u8) {
        let line_type = 8;
        let shift = 0;
        let mut m : core::Mat = window.into();
        imgproc::ellipse(
            &mut m,
            core::Point{ x : el.center.1 as i32, y : el.center.0 as i32 },
            core::Size{ width : (el.large_axis / 2.) as i32, height : (el.small_axis / 2.) as i32 },
            el.angle,
            0.0,
            360.0,
            core::Scalar::from((color as f64, color as f64, color as f64)),
            thickness,
            line_type,
            shift
        );
    }

    #[derive(Clone, Copy)]
    pub enum Method {

        /// Implemented in OpenCV with Andrew W Fitzgibbon and Robert B Fisher. A buyer's guide to conic fitting.
        /// In Proceedings of the 6th British conference on Machine vision (Vol. 2),
        /// pages 513–522. BMVA Press, 1995.
        LeastSquares,

        /// Implemented in OpenCV with Andrew Fitzgibbon, Maurizio Pilu, and Robert B. Fisher.
        /// Direct least square fitting of ellipses. IEEE Transactions on Pattern Analysis and Machine Intelligence, 21(5):476–480, 1999.
        Direct,

        /// Implemented in OpenCV with Gabriel Taubin. Estimation of planar curves, surfaces, and nonplanar space curves defined
        /// by implicit equations with applications to edge and range image segmentation.
        /// IEEE Transactions on Pattern Analysis and Machine Intelligence, 13(11):1115–1138, 1991.
        ApproxMeanSquare
    }

    #[derive(Debug)]
    pub struct EllipseFitting {
        pt_vec : core::Vector<core::Point2f>
    }

    impl EllipseFitting {

        pub fn new() -> Self {
            Self { pt_vec : core::Vector::new() }
        }

        /// Returns position and radius of fitted circle. Also see fit_ellipse_ams; fit_ellipse_direct.
        pub fn fit_ellipse(&mut self, pts : &[(usize, usize)], method : Method) -> Result<Ellipse, String> {
            self.pt_vec.clear();
            if pts.len() < 5 {
                return Err(String::from("Insufficient points"));
            }
            for pt in pts.iter() {
                self.pt_vec.push(core::Point2f::new(pt.1 as f32, pt.0 as f32));
            }
            self.fit_from_current(method)
        }

        pub fn fit_ellipse_u16(&mut self, pts : &[(u16, u16)], method : Method) -> Result<Ellipse, String> {
            self.pt_vec.clear();
            if pts.len() < 5 {
                return Err(String::from("Insufficient points"));
            }
            for pt in pts.iter() {
                self.pt_vec.push(core::Point2f::new(pt.1 as f32, pt.0 as f32));
            }
            self.fit_from_current(method)
        }

        fn fit_from_current(&mut self, method : Method) -> Result<Ellipse, String> {
            assert!(self.pt_vec.len() >= 5);
            let rotated_rect = match method {
                Method::LeastSquares => {
                    imgproc::fit_ellipse(&self.pt_vec)
                        .map_err(|e| format!("Ellipse fitting error ({})", e))?
                },
                Method::Direct => {
                    // Requires minimum of 5 points.
                    imgproc::fit_ellipse_direct(&self.pt_vec)
                        .map_err(|e| format!("Ellipse fitting error ({})", e))?
                },
                Method::ApproxMeanSquare => {
                    imgproc::fit_ellipse_ams(&self.pt_vec)
                        .map_err(|e| format!("Ellipse fitting error ({})", e))?
                }
            };
            let center_pt = rotated_rect.center();
            if center_pt.y < 0.0 || center_pt.x < 0.0 {
                return Err(format!("Circle outside image boundaries"));
            }
            let center = (center_pt.y as f64, center_pt.x as f64);
            let angle = rotated_rect.angle();
            let size = rotated_rect.size();
            if rotated_rect.size().width as i32 <= 0 || rotated_rect.size().height as i32 <= 0 {
                return Err(format!("Invalid ellipse dimension"));
            }
            let w = rotated_rect.size().width as f64;
            let h = rotated_rect.size().height as f64;
            //let large_axis = w.max(h);
            //let small_axis = w.min(h);
            let large_axis = w;
            let small_axis = h;
            // assert!(large_axis >= small_axis);
            Ok(Ellipse {
                center,
                large_axis,
                small_axis,
                angle : angle as f64
            })
        }

    }

    #[derive(Debug)]
    pub struct EnclosingCircle {

        pt_vec : core::Vector<core::Point2i>
    }

    impl EnclosingCircle {

        pub fn new() -> Self {
            let pt_vec = core::Vector::with_capacity(256);
            Self { pt_vec }
        }

        pub fn calculate(&mut self, pts : &[(usize, usize)]) -> Result<((usize, usize), usize), String> {
            self.pt_vec.clear();
            for pt in pts.iter() {
                self.pt_vec.push(core::Point2i::new(pt.1 as i32, pt.0 as i32));
            }
            let mut center = core::Point2f{ x : 0.0, y : 0.0 };
            let mut radius = 0.0;
            let ans = imgproc::min_enclosing_circle(
                &self.pt_vec,
                &mut center,
                &mut radius
            );
            match ans {
                Ok(_) => if center.x >= 0.0 && center.y >= 0.0 {
                    Ok(((center.y as usize, center.x as usize), radius as usize))
                } else {
                    Err(format!("Center outside valid region"))
                },
                Err(e) => Err(format!("{}", e))
            }
        }
    }

}

// pub enum EllipseError {
// }

pub struct CircleFit {
    pub center : Vector2<f32>,
    pub radius : f32
}

impl CircleFit {

    pub fn calculate_from_points(ptsf : &[Vector2<f32>]) -> Option<Self> {
        let n = ptsf.len() as f32;
        let (mut center_x, mut center_y) = ptsf.iter().fold((0.0, 0.0), |acc, pt| (acc.0 + pt[0], acc.1 + pt[1]) );
        center_x /= n;
        center_y /= n;
        let centered_ptsf : Vec<_> = ptsf.iter().map(|pt| Vector2::new(pt[0] - center_x, pt[1] - center_y)).collect();
        let x_sq = centered_ptsf.iter().fold(0., |acc, pt| acc + pt[0].powf(2.) );
        let y_sq = centered_ptsf.iter().fold(0., |acc, pt| acc + pt[1].powf(2.) );
        let x_cub = centered_ptsf.iter().fold(0., |acc, pt| acc + pt[0].powf(3.) );
        let y_cub = centered_ptsf.iter().fold(0., |acc, pt| acc + pt[1].powf(3.) );
        let xy = centered_ptsf.iter().fold(0., |acc, pt| acc + pt[0] * pt[1] );
        let yyx = centered_ptsf.iter().fold(0., |acc, pt| acc + pt[0] * pt[1].powf(2.) );
        let xxy = centered_ptsf.iter().fold(0., |acc, pt| acc + pt[0].powf(2.) * pt[1] );
        let m = Matrix2::from_rows(&[RowVector2::new(x_sq, xy), RowVector2::new(xy, y_sq)]);
        let b = Vector2::new(
            0.5 * (x_cub + yyx),
            0.5 * (y_cub + xxy)
        );
        let ans = LU::new(m).solve(&b)?;
        let center = Vector2::new(ans[0] + center_x, ans[1] + center_y);
        let radius = (ans[0].powf(2.) + ans[1].powf(2.) + (x_sq + y_sq) / n).sqrt();
        Some(Self { center, radius })
    }

    pub fn calculate(
        pts : &[(usize, usize)], 
        img_height : usize
    ) -> Option<Self> {
        let ptsf : Vec<Vector2<f32>> = pts.iter()
            .map(|pt| Vector2::new(pt.1 as f32, (img_height - pt.0) as f32) ).collect();
        Self::calculate_from_points(&ptsf[..])
    }

    pub fn center_coord(&self, img_height : usize) -> Option<(usize, usize)> {
        if (self.center[0] > 0. && self.center[1] > 0.) && (self.center[1] as usize) < img_height {
            Some((img_height - self.center[1] as usize, self.center[0] as usize))
        } else {
            None
        }
    }

    // variance of the random variable (dist(pt, center) - radius), which is a measure of fit quality.
    pub fn radius_variance(&self, pts : &[(u16, u16)], img_height : u16) -> f32 {
        let ptsf : Vec<Vector2<f32>> = pts.iter().map(|pt| Vector2::new(pt.1 as f32, (img_height - pt.0) as f32) ).collect();
        let n = ptsf.len() as f32;
        ptsf.iter().fold(0.0, |acc, pt| acc + ((pt - &self.center).magnitude() - self.radius).powf(2.)  ) / n
    }

}

