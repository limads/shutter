use std::cmp::{PartialEq, Ordering};
use nalgebra::*;
use bayes::fit::ca::{Manhattan, Metric};
use nalgebra::geometry::Rotation2;
use nalgebra::Vector2;
use std::cmp::Ord;
use std::ops::Add;
use nalgebra;
use std::mem;
use crate::image::*;
use itertools::Itertools;
use std::collections::BTreeMap;
use std::ops::Range;

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

/* Given two equal length slices carrying coordinates, permute elements at the second slice until
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

/** Holds a partitioned plane delimited by size. A plane can be split
into four rects:

| Base   | Right |
| Bottom | Complement |

With the base always having offset zero, and complement
having offset given by the offset field.

Offset is always guaranteed to be < size, and size should be non-zero.

Plane is a base structure used to map unsigned integer memory offsets used
to index images (type Offset = (usize, usize), y-down) into the analytical plane where
floating poing operations are performed (type Point = Vector2<f32>, y-up). The floating-point
counterpart to plane is Area, which carries a pair of vectors in the analytical plane
that delimit a Plane.
**/
#[derive(Debug, Clone)]
pub struct Plane {
    pub offset : (usize, usize),
    pub size : (usize, usize)
}

impl Plane {

    pub fn base(&self) -> Rect {
        Rect {
            offset : (0, 0),
            size : self.offset
        }
    }

    pub fn bottom(&self) -> Rect {
        Rect {
            offset : (self.offset.0 + self.size.0, 0),
            size : (self.size.0 - self.offset.0, self.offset.1)
        }
    }

    pub fn right(&self) -> Rect {
        Rect {
            offset : (0, self.offset.1 + self.size.1),
            size : (self.offset.1, self.size.1 - self.offset.1)
        }
    }

    pub fn complement(&self) -> Rect {
        Rect {
            offset : self.offset,
            size : (self.size.0 - self.offset.0, self.size.1 - self.offset.1)
        }
    }

    /* Returns the vector pointing to the offset of this Plane,
    assuming a cartesian plane startinig at the bottom-left
    portion of the plane. */
    pub fn point(&self) -> Option<Vector2<f32>> {
        coord::coord_to_vec::<f32>(self.offset, self.size)
    }

    pub fn offset(&self, pt : &Vector2<f32>) -> Option<(usize, usize)> {
        coord::point_to_coord(pt, self.size)
    }

}

/* A rectangle with offset an size.  */
#[derive(Debug, Clone)]
pub struct Rect {
    pub offset : (usize, usize),
    pub size : (usize, usize)
}

impl Rect {

    pub fn bottom_left(&self) -> (usize, usize) {
        (self.offset.0 + self.size.0, 0)
    }

    // pub fn bottom_right(&self) -> (usize, usize) {
    //    self.offset.0 +
    // }
}

impl From<Rect> for Region {

    fn from(r : Rect) -> Self {
        Region {
            rows : Range { start : r.offset.0, end : r.offset.0 + r.size.0 },
            cols : Range { start : r.offset.1, end : r.offset.1 + r.size.1 }
        }
    }

}

impl From<Region> for Rect {

    fn from(reg : Region) -> Self {
        Rect {
            offset : (reg.rows.start, reg.cols.start),
            size : (reg.rows.end - reg.rows.start, reg.cols.end - reg.cols.start)
        }
    }

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
        F : AsPrimitive<usize> + PartialOrd + Zero + num_traits::Float
    {
        if pt[0] > F::zero() && pt[1] > F::zero() {
            let (row, col) : (usize, usize) = (pt[1].round().as_(), pt[0].round().as_());
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

/* The floating-point equivalent to Rect, where size happens to be symmetric. */
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

/* The floating-point equivalent to Rect */
pub struct Rectangle {

    pub tl : Vector2<usize>,

    pub sides : Vector2<usize>

}

impl Quadrilateral for Rectangle {

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
#[derive(Clone)]
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

    pub unsafe fn get_spatial_moment(&self, m : i32, n : i32) -> Option<f32> {
        let channel = 0;
        let mut val : f64 = 0.;
        assert!(m + n <= 3 );
        let offset = crate::foreign::ipp::ippi::IppiPoint { x : 0, y : 0 };
        let ans = crate::foreign::ipp::ippi::ippiGetSpatialMoment_64f(
            mem::transmute(self.state.as_ptr()),
            m,
            n,
            channel,
            offset,
            &mut val as *mut _
        );
        if ans == crate::foreign::ipp::ippi::ippStsMoment00ZeroErr {
            return None;
        }
        assert!(ans == 0);
        Some(val as f32)
    }

    // Normalized variant on interval 0.0-1.0
    pub unsafe fn get_normalized_spatial_moment(&self, m : i32, n : i32) -> Option<f32> {
        let channel = 0;
        let mut val : f64 = 0.;
        let offset = crate::foreign::ipp::ippi::IppiPoint { x : 0, y : 0 };
        let ans = crate::foreign::ipp::ippi::ippiGetNormalizedSpatialMoment_64f(
            mem::transmute(self.state.as_ptr()),
            m,
            n,
            channel,
            offset,
            &mut val as *mut _
        );
        if ans == crate::foreign::ipp::ippi::ippStsMoment00ZeroErr {
            return None;
        }
        assert!(ans == 0);
        Some(val as f32)
    }

    pub unsafe fn get_central_moment(&self, m : i32, n : i32) -> Option<f32> {
        let channel = 0;
        let mut val : f64 = 0.;
        let ans = crate::foreign::ipp::ippi::ippiGetCentralMoment_64f(
            mem::transmute(self.state.as_ptr()),
            m,
            n,
            channel,
            &mut val as *mut _
        );
        if ans == crate::foreign::ipp::ippi::ippStsMoment00ZeroErr {
            return None;
        }
        assert!(ans == 0);
        Some(val as f32)
    }

    // Normalized variant on interval 0.0-1.0
    pub unsafe fn get_normalized_central_moment(&self, m : i32, n : i32) -> Option<f32> {
        let channel = 0;
        let mut val : f64 = 0.;
        let ans = crate::foreign::ipp::ippi::ippiGetNormalizedCentralMoment_64f(
            mem::transmute(self.state.as_ptr()),
            m,
            n,
            channel,
            &mut val as *mut _
        );
        if ans == crate::foreign::ipp::ippi::ippStsMoment00ZeroErr {
            return None;
        }
        assert!(ans == 0);
        Some(val as f32)
    }

    pub fn calculate(&mut self, win : &Window<u8>, snd : bool) -> Option<CentralMoments> {
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
            let zero = self.get_spatial_moment(0, 0)?;
            if zero == 0.0 {
                return None;
            }
            let m10 = self.get_spatial_moment(1, 0)?;
            let m01 = self.get_spatial_moment(0, 1)?;
            let center_x = m10 / zero;
            let center_y = m01 / zero;
            let (xx, yy, xy) = if snd {
                let xx = self.get_spatial_moment(2, 0)? / zero;
                let yy = self.get_spatial_moment(0, 2)? / zero;
                let xy = self.get_spatial_moment(1, 1)? / zero;
                (xx, yy, xy)
            } else {
                (0., 0., 0.)
            };

           // let disp_x = self.get_spatial_moment(2, 0)? / zero - center_x * m10;
           // let disp_y = self.get_spatial_moment(0, 2)? / zero - center_y * m01;

            // let disp_x = (self.get_spatial_moment(2, 0)? - self.get_spatial_moment(1, 0)?.powf(2.)) / zero;
            // let disp_y = (self.get_spatial_moment(0, 2)? - self.get_spatial_moment(0, 1)?.powf(2.)) / zero;

            // let center_x = self.get_normalized_central_moment(1, 0)?;
            // let center_y = self.get_normalized_central_moment(0, 1)?;
            // let xx = self.get_central_moment(2, 0)?;
            // let yy = self.get_central_moment(0, 2)?;
            /*let xy = self.get_normalized_central_moment(1, 1)?;
            let xxy = self.get_normalized_central_moment(2, 1)?;
            let yyx = self.get_normalized_central_moment(1, 2)?;
            let xxx = self.get_normalized_central_moment(3, 0)?;
            let yyy = self.get_normalized_central_moment(0, 3)?;*/

            Some(CentralMoments {
                center : (center_y, center_x),
                zero,
                xx,
                yy,
                xy,
                xxy : 0.0,
                yyx : 0.0,
                xxx : 0.0,
                yyy : 0.0
            })
        }
    }

}

// The central moments are translation invariant. Moments can be calculated over the
// shape edge only, since the edge is equivalent to a binary image giving weight 1
// to pixels at the edge and weight zero for pixels outside it. For a binary image,
// the moment 0,1/(0,0) and 1,0/(0,0) are the centroid. For a grayscale image,
// it is the center of gravity (point of equilibrium of the image terrain surface).
#[derive(Debug, Clone, Default)]
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

    pub fn orientation(&self, img_height : usize) -> Option<Orientation> {
        // let c = (m.center.0, m.center.1);
        let my = self.center.0;
        let mx = self.center.1;
        let mtx = Matrix2::from_row_slice(&[self.xx - mx.powf(2.), self.xy - mx*my, self.xy - mx*my, self.yy - my.powf(2.)]);
        let Some(sm) = nalgebra::SymmetricEigen::try_new(mtx, 0.0001, 100) else { return None };

        // The orientation of the fst eigenvec gives the ellipse orientation.
        let u1 = sm.eigenvectors.column(0).clone_owned();
        let u2 = sm.eigenvectors.column(1).clone_owned();

        let (lambda_1, lambda_2) = (sm.eigenvalues[0], sm.eigenvalues[1]);
        Some(Orientation {
            u1,
            u2,
            lambda_1,
            lambda_2,
            center : self.center,
            img_height,
            mass : self.zero as f32
        })
    }

    pub fn area(&self) -> f32 {
        self.zero
    }

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
        // assert!(lambda1 >= lambda2);

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

pub trait OffsetOps {

    fn midpoint(&self, other : &Self) -> Self;

    fn half(&self) -> Self;

    fn distance(&self, other : &Self) -> f32;

}

pub type Offset = (usize, usize);

impl OffsetOps for Offset {

    fn midpoint(&self, other : &Self) -> Offset {
        ((self.0 + other.0) / 2, (self.1 + other.1) / 2)
    }

    fn half(&self) -> Offset {
        (self.0 / 2, self.1 / 2)
    }

    fn distance(&self, other : &Self) -> f32 {
        ((self.0 as f32 - other.0 as f32)).hypot((self.1 as f32 - other.1 as f32))
    }

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

    assert!(pts.len() >= 1);

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

pub trait Set
where
    Self : Sized
{

    type Item;

    fn contains(&self, it : &Self::Item) -> bool;

    fn intersect(&self, other : &Self) -> Option<Self>;

    fn unite(&self, other : &Self) -> Self;

}

use std::ops::RangeInclusive;

// TODO rename to OrdSet, which satisfies Self : Set with the same Item. Move
// everything to top-level set module. RunLength code should satisfy set and ordset
// as well. Perhaps have associated Proximity type (1D or 2D spatial relationships).
// Where Region has Direction { vertical : Proximity, horizontal : Proximity }.
pub trait HalfOpen
where
    Self : Sized
{

    // If segments overlap, return their joint area (A || B).
    fn joint(&self, other : &Self) -> Option<Self>;

    // If segments overlap, return their common area (A && B).
    fn overlap(&self, other : &Self) -> Option<Self>;

    fn proximity(&self, other : &Self) -> Proximity;

    // perhaps rename to enclose so as to not conflict with std::ops::Range::contains
    fn contains(&self, inner : &Self) -> bool;

    fn contained(&self, outer : &Self) -> bool;

    // If self contains all inner intervals, return the
    // differences (disjoint intervals). Assume rs is sorted and
    // end >= start for each inner interval.
    fn difference(&self, rs : &[Self]) -> Vec<Self>;

    fn overlapping_range(&self, b_ranges : &[Self]) -> Option<RangeInclusive<usize>>;

}

impl HalfOpen for Range<usize>
where
    Self : Sized
{

    fn joint(&self, other : &Self) -> Option<Self> {
        if self.proximity(other) == Proximity::Exclude {
            None
        } else {
            Some(Range { start : self.start.min(other.start), end : self.end.max(other.end) })
        }
    }

    fn overlap(&self, other : &Self) -> Option<Self> {
        /*use rangetools::Rangetools;
        let int = self.intersection(other.clone());
        if int.is_empty() {
            None
        } else {
            Some(Range { start : int.start, end : int.end })
        }*/
        let joint = self.joint(other)?;
        let ovlp_start = self.start.max(other.start);
        let ovlp_len = joint.end.checked_sub(joint.start)?
            .checked_sub(self.start.abs_diff(other.start))?
            .checked_sub(self.end.abs_diff(other.end))?;
        if ovlp_len >= 1 {
            Some(Range { start : ovlp_start, end : ovlp_start + ovlp_len })
        } else {
            None
        }
    }

    fn overlapping_range(&self, b_ranges : &[Range<usize>]) -> Option<RangeInclusive<usize>> {
        // Bottom is not partitioned (will return false for all entries, all b greater than a)
        if let Some(fst_b) = b_ranges.first() {
            if fst_b.start > self.end && fst_b.end > self.end {
                return None;
            }
        }

        // First false entry overlaps w/ a. (since non-overlapping ones were excluded at prev. step).
        let smaller = b_ranges.partition_point(|b| b.start < self.start && b.end < self.start );

        // All elements of b smaller than a starting point (returns true for all entries,
        // see docs of partition point).
        if smaller == b_ranges.len() {
            return None;
        }

        // First false entry does not overlap w/ a.
        let mut larger = b_ranges[smaller..].partition_point(|b| b.start <= self.end );
        if larger < b_ranges[smaller..].len() {
            // i.e. last one returning true.
            larger -= 1;
            Some(RangeInclusive::new(smaller, larger + smaller))
        } else {
            // All returned true (take whole rest of slice).
            Some(RangeInclusive::new(smaller, b_ranges.len()-1))
        }
    }

    fn difference(&self, rs : &[Range<usize>]) -> Vec<Range<usize>> {
        let mut diffs = Vec::new();
        let mut last_end = self.start;

        // Add intervals before each range
        for r in rs {
            assert!(HalfOpen::contains(self, r));
            assert!(r.start >= last_end);
            assert!(r.end >= r.start);
            diffs.push(Range { start : last_end, end : r.start });
            last_end = r.end;
        }

        // Add interval after last range
        if let Some(lst) = rs.last() {
            diffs.push(Range { start : last_end, end : self.end });
        }
        diffs
    }

    fn contains(&self, inner : &Self) -> bool {
        self.start <= inner.start && self.end >= inner.end
    }

    fn contained(&self, outer : &Self) -> bool {
        self.start >= outer.start && self.end <= outer.end
    }

    fn proximity(&self, other : &Self) -> Proximity {

        if (self.start < other.start && self.end < other.start) ||
            (self.start > other.end && self.end > other.end)
        {
            return Proximity::Exclude;
        }

        match (self.start.cmp(&other.end), self.end.cmp(&other.start)) {

            // Contacts from left
            (Ordering::Equal, Ordering::Greater) => Proximity::Contact,

            // Contacts from right
            (Ordering::Less, Ordering::Equal) => Proximity::Contact,

            // Exclude
            (Ordering::Less, Ordering::Less) | (Ordering::Greater, Ordering::Greater) => Proximity::Exclude,

            _ => Proximity::Overlap

        }
    }

}

pub fn enclosing_rect_for_rects(
    qs : impl IntoIterator<Item=(usize, usize, usize, usize)>
) -> Option<(usize, usize, usize, usize)> {
    /*let mut qs = qs.into_iter();
    let fst = qs.next()?;
    let mut tl = (fst.0, fst.1);
    let mut br = (fst.0 + fst.2, fst.1 + fst.3);
    while let Some(q) = qs.next() {
        let this_tl = (q.0, q.1);
        let this_br = (q.0 + q.2, q.1 + q.3);
        if this_br.0 > br.0 {
            br.0 = this_br.0;
        }
        if this_br.1 > br.1 {
            br.1 = this_br.1;
        }
        if this_tl.0 < tl.0 {
            tl.0 = this_tl.0;
        }
        if this_tl.1 < tl.1 {
            tl.1 = this_tl.1;
        }
    }
    Some((tl.0, tl.1, br.0 - tl.0, br.1 - tl.1))*/
    let mut rects : Vec<_> = qs.into_iter().collect();
    let min_row = rects.iter().map(|r| r.0 ).min()?;
    let min_col = rects.iter().map(|r| r.1 ).min()?;
    let max_row = rects.iter().map(|r| r.0+r.2 ).max()?;
    let max_col = rects.iter().map(|r| r.1+r.3 ).max()?;
    Some((min_row, min_col, max_row - min_row, max_col - min_col))
}

#[test]
fn proximity() {

    let regions = [

        // Same region overlaps
        (
            Region { cols : Range { start : 0, end : 10 }, rows : Range { start : 0, end : 10 } },
            Region { cols : Range { start : 0, end : 10 }, rows : Range { start : 0, end : 10 } },
            Proximity::Overlap
        ),

        // Half width overlap
        (
            Region { cols : Range { start : 0, end : 10 }, rows : Range { start : 0, end : 10 } },
            Region { cols : Range { start : 5, end : 15 }, rows : Range { start : 0, end : 10 } },
            Proximity::Overlap
        ),

        // Half height overlap
        (
            Region { cols : Range { start : 0, end : 10 }, rows : Range { start : 0, end : 10 } },
            Region { cols : Range { start : 0, end : 10 }, rows : Range { start : 5, end : 15 } },
            Proximity::Overlap
        ),

        // Horizontal exclude
        (
            Region { cols : Range { start : 0, end : 10 }, rows : Range { start : 0, end : 10 } },
            Region { cols : Range { start : 30, end : 40 }, rows : Range { start : 0, end : 10 } },
            Proximity::Exclude
        ),

        // Vertical exclude
        (
            Region { cols : Range { start : 0, end : 10 }, rows : Range { start : 0, end : 10 } },
            Region { cols : Range { start : 0, end : 10 }, rows : Range { start : 30, end : 40 } },
            Proximity::Exclude
        ),

        // Both exclude
        (
            Region { cols : Range { start : 0, end : 10 }, rows : Range { start : 0, end : 10 } },
            Region { cols : Range { start : 30, end : 40 }, rows : Range { start : 30, end : 40 } },
            Proximity::Exclude
        ),

        // Horizontal contact (other overlap)
        (
            Region { cols : Range { start : 0, end : 10 }, rows : Range { start : 0, end : 10 } },
            Region { cols : Range { start : 10, end : 20 }, rows : Range { start : 0, end : 10 } },
            Proximity::Contact
        ),

        // Vertical contact (other overlap)
        (
            Region { cols : Range { start : 0, end : 10 }, rows : Range { start : 0, end : 10 } },
            Region { cols : Range { start : 0, end : 10 }, rows : Range { start : 10, end : 20 } },
            Proximity::Contact
        ),

        // Corner contact
        (
            Region { cols : Range { start : 0, end : 10 }, rows : Range { start : 0, end : 10 } },
            Region { cols : Range { start : 10, end : 20 }, rows : Range { start : 10, end : 20 } },
            Proximity::Contact
        ),

    ];

    let mut regions_reversed = regions.clone();
    regions_reversed.iter_mut().for_each(|(r1, r2, _)| std::mem::swap(r1, r2) );

    for (r1, r2, out) in regions {
        assert!(r1.proximity(&r2) == out, "{:?},{:?}={:?}", r1, r2, out);
    }
    for (r1, r2, out) in regions_reversed {
        assert!(r1.proximity(&r2) == out, "{:?},{:?}={:?}", r1, r2, out);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Proximity {
    Contact,
    Overlap,
    Exclude
}

/* Represents a subset of the cartesian plane by an origin (bottom-left point) and target
(top-right point). This is the floating-point, cartesian plane counterpart
of the image plane struct Region and Plane. */
#[derive(Debug, Clone, Copy)]
pub struct Area {
    pub origin : Vector2<f32>,
    pub target : Vector2<f32>
}

impl Area {

    pub fn rect(&self, size : (usize, usize)) -> Option<(usize, usize, usize, usize)> {
        let region = self.region(size)?;
        Some(region.to_rect_tuple())
    }

    pub fn region(&self, size : (usize, usize)) -> Option<crate::shape::Region> {
        let bl = (size.0.checked_sub(self.origin[1] as usize)?, self.origin[0] as usize);
        let tr = (size.0.checked_sub(self.target[1] as usize)?, self.target[0] as usize);
        if tr.0 < bl.0 && tr.1 > bl.1 {
            let off = (tr.0, bl.1);
            let sz = (bl.0.checked_sub(tr.0)?, tr.1.checked_sub(bl.1)?);
            Some(crate::shape::Region::new(off, sz))
        } else {
            None
        }
    }

}

/**
Used to index images. This is equivalent to a rect, containing
a pair of horizontal (h) and vertical (v) ranges.
**/
#[derive(Debug, Clone, Default, PartialEq, Eq, std::hash::Hash, serde::Serialize, serde::Deserialize)]
pub struct Region {
    pub rows : Range<usize>,
    pub cols : Range<usize>
}

impl From<(usize, usize, usize, usize)> for Region {

    fn from(r : (usize, usize, usize, usize)) -> Self {
        Self::from_rect_tuple(&r)
    }

}

impl Region {

    pub fn enclosing(pts : &[(usize, usize)]) -> Option<Self> {
        let mut r = Self::containing(*pts.get(0)?);
        for i in 1..pts.len() {
            r.expand(pts[i]);
        }
        Some(r)
    }

    // Creates a region that just encloses the given point.
    pub fn containing(pos_u : (usize, usize)) -> Self {
        Self {
            rows : Range { start : pos_u.0, end : pos_u.0+1 },
            cols : Range { start : pos_u.1, end : pos_u.1 + 1}
        }
    }

    // Expand this region so it contains this new point.
    pub fn expand(&mut self, pos_u : (usize, usize)) {
        if pos_u.0 < self.rows.start {
            self.rows.start = pos_u.0;
        }
        if pos_u.1 < self.cols.start {
            self.cols.start = pos_u.1;
        }
        if pos_u.0 >= self.rows.end {
            self.rows.end = pos_u.0 + 1;
        }
        if pos_u.1 >= self.cols.end {
            self.cols.end = pos_u.1 + 1;
        }
    }

    pub fn last_row(&self) -> usize {
        self.rows.end.saturating_sub(1)
    }

    pub fn last_col(&self) -> usize {
        self.cols.end.saturating_sub(1)
    }

    pub fn is_valid(&self) -> bool {
        self.rows.end > self.rows.start && self.cols.end > self.cols.start
    }

    pub fn offset(&self) -> (usize, usize) {
        (self.rows.start, self.cols.start)
    }

    pub fn size(&self) -> (usize, usize) {
        (self.rows.end - self.rows.start, self.cols.end - self.cols.start)
    }

    pub fn offset_by(&self, rows : usize, cols : usize) -> Region {
        Region {
            rows : Range { start : self.rows.start + rows, end: self.rows.end + rows },
            cols : Range { start : self.cols.start + cols, end: self.cols.end + cols },
        }
    }

    pub fn contains_point(&self, pt : &(usize, usize)) -> bool {
        self.rows.start <= pt.0 && self.rows.end > pt.0
            && self.cols.start <= pt.1 && self.cols.end > pt.1
    }

    pub fn center(&self) -> (usize, usize) {
        (
            self.rows.start + (self.rows.end - self.rows.start) / 2,
            self.cols.start + (self.cols.end - self.cols.start) / 2
        )
    }

    pub fn area(&self) -> usize {
        self.height() * self.width()
    }

    pub fn height(&self) -> usize {
        self.rows.end - self.rows.start
    }

    pub fn width(&self) -> usize {
        self.cols.end - self.cols.start
    }

    pub fn from_rect_tuple(r : &(usize, usize, usize, usize)) -> Self {
        Self {
            cols : Range { start : r.1, end : r.1 + r.3 + 1 },
            rows : Range { start : r.0, end : r.0 + r.2 + 1 }
        }
    }

    pub fn new_centered(center : (usize, usize), sz : (usize, usize)) -> Option<Self> {
        let off = (center.0.checked_sub(sz.0 / 2)?, center.1.checked_sub(sz.1 / 2)?);
        Some(Self::new(off, sz))
    }

    pub fn new(off : (usize, usize), sz : (usize, usize)) -> Self {
        Self::from_offset_size(off, sz)
    }

    pub fn from_offset_size(off : (usize, usize), sz : (usize, usize)) -> Self {
        Self {
            cols : Range { start : off.1, end : off.1 + sz.1 },
            rows : Range { start : off.0, end : off.0 + sz.0 }
        }
    }

    pub fn to_rect_tuple(&self) -> (usize, usize, usize, usize) {
        (
            self.rows.start,
            self.cols.start,
            self.rows.end.saturating_sub(self.rows.start),
            self.cols.end.saturating_sub(self.cols.start)
        )
    }

}

impl HalfOpen for Region
where
    Self : Sized
{

    fn overlap(&self, other : &Self) -> Option<Self> {
        let rows = self.rows.overlap(&other.rows)?;
        let cols = self.cols.overlap(&other.cols)?;
        Some(Region { rows, cols })
    }

    fn joint(&self, other : &Self) -> Option<Self> {
        let rows = self.rows.joint(&other.rows)?;
        let cols = self.cols.joint(&other.cols)?;
        Some(Region { rows, cols })
    }

    fn overlapping_range(&self, b_ranges : &[Self]) -> Option<RangeInclusive<usize>> {
        unimplemented!()
    }

    fn difference(&self, rs : &[Self]) -> Vec<Self> {
        unimplemented!()
    }

    fn contains(&self, inner : &Self) -> bool {
        HalfOpen::contains(&self.cols, &inner.cols) && HalfOpen::contains(&self.rows, &inner.rows)
    }

    fn contained(&self, outer : &Self) -> bool {
        self.cols.contained(&outer.cols) && self.rows.contained(&outer.rows)
    }

    fn proximity(&self, other : &Self) -> Proximity {
        match (self.cols.proximity(&other.cols), self.rows.proximity(&other.rows)) {
            (Proximity::Exclude, _) | (_, Proximity::Exclude) => Proximity::Exclude,
            (Proximity::Overlap | Proximity::Contact, Proximity::Contact) => Proximity::Contact,
            (Proximity::Contact, Proximity::Overlap | Proximity::Contact) => Proximity::Contact,
            _ => Proximity::Overlap
        }
    }

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

#[test]
fn rect_contact() {
    assert!([
        rect_contacts3(&(0,0,10,10), &(10,10,10,10)),
        rect_contacts3(&(10,10,10,10), &(0,0,10,10)),
        rect_contacts3(&(11,11,10,10), &(0,0,10,10)),
        rect_contacts3(&(0,0,10,10), &(11,11,10,10)),
    ] == [true, true, false, false]);
}

pub fn rect_contacts3(
    r1 : &(usize, usize, usize, usize), 
    r2 : &(usize, usize, usize, usize)
) -> bool {

    let tl_1 = top_left_coordinate(r1);
    let tl_2 = top_left_coordinate(r2);
    let br_1 = bottom_right_coordinate(r1);
    let br_2 = bottom_right_coordinate(r2);
    
	if tl_1.1 > br_2.1 || tl_2.1 > br_1.1 {
	    return false;
	}

	if tl_1.0 > br_2.0 || tl_2.0 > br_1.0 {
	    return false;
	}

	true
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
pub struct TriangleCoords([(usize, usize); 3]);

impl TriangleCoords {

    pub fn lines(&self) -> [Line; 3] {
        let l1 = Line::from([self.0[0], self.0[1]]);
        let l2 = Line::from([self.0[0], self.0[2]]);
        let l3 = Line::from([self.0[1], self.0[2]]);
        [l1, l2, l3]
    }

    /// Splits this triangle into two right triangles
    pub fn split_to_right(&self) -> (TriangleCoords, TriangleCoords) {
        let base = self.base();
        let perp = perp_line(&self, &base);
        let (b1, b2) = base.points();
        let (perp_pt, top) = perp.points();
        (TriangleCoords([b1, top, perp_pt]), TriangleCoords([b2, top, perp_pt]))
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

fn perp_line(tri : &TriangleCoords, base : &Line) -> Line {
    let (b1, b2) = base.points();
    let top = tri.0.iter().find(|pt| **pt != b1 && **pt != b2 ).unwrap();
    base.perpendicular(*top)
}

impl Polygon {

    pub fn area(&self) -> f64 {
        self.triangles().iter().fold(0.0, |area, tri| area + tri.area() )
    }

    pub fn triangles<'a>(&'a self) -> Vec<TriangleCoords> {
        let n = self.0.len();
        let last = self.0.last().unwrap().clone();
        let mut triangles = Vec::new();
        for (a, b) in self.0.iter().take(n-1).zip(self.0.iter().skip(1)) {
            triangles.push(TriangleCoords([*a, *b, last]));
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

// TODO polygon contains line
// https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/

/// Calculate line intersection
/// Reference https://mathworld.wolfram.com/Line-LineIntersection.html
/// m1 = [x1 y1; x2 y2]
/// m2 = [x3 y3; x4 y4]
/// m3 = [x1 - x2, y1 - y2; x3 - x4, y3 - y4]
/// Calculates intersection of two lines, where each line is defined by two points
/// stacked into a 2x2 matrix (one at each row as [[x y], [x,y]]).
pub fn line_intersection(m1 : Matrix2<f32>, m2 : Matrix2<f32>) -> Option<Vector2<f32>> {
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
        RowVector2::new(line1_pt1.1 as f32, line1_pt1.0 as f32 * -1.),
        RowVector2::new(line1_pt2.1 as f32, line1_pt2.0 as f32 * -1.)
    ]);
    let m2 = Matrix2::from_rows(&[
        RowVector2::new(line2_pt1.1 as f32, line2_pt1.0 as f32 * -1.),
        RowVector2::new(line2_pt2.1 as f32, line2_pt2.0 as f32 * -1.)
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

/*pub fn rect_to_polygon(r : (usize, usize, usize, usize)) -> parry2d::ConvexPolygon {
    parry2d::ConvexPolygon::from_convex_hull(&[
        Point2::new(r.1 as f32, r.0 as f32),
        Point2::new(r.1 as f32 + r.3 as f32, r.0 as f32),
        Point2::new(r.1 as f32 + r.3 as f32, r.0 as f32 + r.2 as f32),
        Point2::new(r.1 as f32, r.0 as f32 + r.2 as f32)
    ]).unwrap()
}*/

/*
Explores the projection matrix that explains the best projective transform between two planes.
See https://docs.opencv.org/3.4/d9/dab/tutorial_homography.html
*/
#[cfg(feature="opencv")]
pub fn find_homography(flat : &[(usize, usize)], proj : &[(usize, usize)], img_height : usize) -> Result<nalgebra::Matrix3<f64>, String> {

    use opencv::calib3d;
    use opencv::prelude::*;

    // Position y axis on the analytical plane.
    let flat : Vec<_> = flat.iter().map(|pt| (img_height - pt.0, pt.1) ).collect();
    let proj : Vec<_> = proj.iter().map(|pt| (img_height - pt.0, pt.1) ).collect();
    let mut flat = convert_points_to_opencv_float(&flat);
    let mut proj = convert_points_to_opencv_float(&proj);

    let mat = calib3d::find_homography(
        &flat,
        &proj,
        &mut opencv::core::no_array(),
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
            /*assert!(self.pt_vec.len() >= 5);
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
            })*/
            unimplemented!()
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

pub fn rect_area(a : &(usize, usize, usize, usize)) -> usize {
    a.2 * a.3
}

pub fn rect_intersection(a : &(usize, usize, usize, usize), b : &(usize, usize, usize, usize)) -> Option<(usize, usize, usize, usize)> {
    use parry2d::utils::Interval;
    let vint = Interval(a.0, a.0 + a.2).intersect(Interval(b.0, b.0 + b.2))?;
    let hint = Interval(a.1, a.1 + a.3).intersect(Interval(b.1, b.1 + b.3))?;
    Some((vint.0, hint.0, vint.1 - vint.0, hint.1 - hint.0))
}

// cargo test --lib -- enclosing_circle --nocapture
#[test]
fn enclosing_circle() {
    println!("{:?}", Circle::enclosing_from_points(&[Vector2::new(1.0, 0.0), Vector2::new(0.0, 1.0)]));
}

#[derive(Debug, Clone)]
pub struct CircleCoords {
    pub center : (usize, usize),
    pub radius : usize
}

#[repr(C)]
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct Circle {
    pub center : Vector2<f32>,
    pub radius : f32
}

impl Default for Circle {

    fn default() -> Self {
        Self { center : Vector2::new(0.0, 0.0), radius : 0.0 }
    }

}

pub fn centroid(ptsf : &[Vector2<f32>]) -> Vector2<f32> {
    let n = ptsf.len() as f32;
    let mut center = Vector2::new(0.0, 0.0);
    for pt in ptsf {
        center += pt;
    }
    center[0] /= n;
    center[1] /= n;
    center
}

/* A sector (subset of a disk) delimited by two angles. */
#[derive(Debug, Clone, Copy)]
pub struct Sector {
    pub circ : Circle,
    pub a1 : f64,
    pub a2 : f64
}

/* A triangle represents as a triplet of vertices. No
spatial relationship between the vertices is specified,
except that they must not be all collinear. */
pub struct Triangle {
    pub v1 : Vector2<f32>,
    pub v2 : Vector2<f32>,
    pub v3 : Vector2<f32>,
}

const SQRT_3 : f32 = 1.732050808;

fn height_equilateral(edge : f32) -> f32 {
    (SQRT_3 * edge) / 2.
}

impl Triangle {

    pub fn new(v1 : Vector2<f32>, v2 : Vector2<f32>, v3 : Vector2<f32>) -> Self {
        Self { v1, v2, v3 }
    }

    pub fn edge_lenghts(&self) -> [f32; 3] {
        [
            self.v1.metric_distance(&self.v2),
            self.v2.metric_distance(&self.v3),
            self.v3.metric_distance(&self.v1),
        ]
    }

    pub fn edge_midpoints(&self) -> [Vector2<f32>; 3] {
        [
            self.v1.scale(0.5) + self.v2.scale(0.5),
            self.v2.scale(0.5) + self.v3.scale(0.5),
            self.v3.scale(0.5) + self.v1.scale(0.5)
        ]
    }

    // Find the lengths of the triangle's median lines (mutually intersecting bisectors
    // from a vertex to the midpoint of the opposite side)
    pub fn bisector_lines(&self) -> [Line2; 3] {
        let midpoints = self.edge_midpoints();
        [
            Line2 { p1 : self.v3, p2 : midpoints[0] },
            Line2 { p1 : self.v1, p2 : midpoints[1] },
            Line2 { p1 : self.v2, p2 : midpoints[2] }
        ]
    }

    // Finds the Fermat-Torricelli point for a triangle.
    // https://en.wikipedia.org/wiki/Fermat_point
    // Original Code by Aaron Becker
    // (https://www.mathworks.com/matlabcentral/fileexchange/131254-calculate-fermat-point)
    // A=v3, B=v1, C=v2
    pub fn geometric_median(&self) -> Vector2<f32> {
        let [a, b, c] = self.edge_lenghts();
        // let (ax, ay) = self.v3[0], self.v3[1];
        // let (bx, by) = self.v1[0], self.v1[1];
        // let (cx, cy) = self.v2[0], self.v2[1];
        let fabc = first_isogonic_center(a,b,c);
        let fbca = first_isogonic_center(b,c,a);
        let fcab = first_isogonic_center(c,a,b);
        let fs = (fabc+fbca+fcab);
        let fic1 = fabc/fs;
        let fic2 = fbca/fs;

        // Barycentric coords of ffp
        let ffp1 = median3(0., fic1, 1.);
        let ffp2 = median3(0., fic2, 1.);
        let ffp3 = 1. - ffp1 - ffp2;

        // Convert barycentric to cartesian
        self.v3*ffp1 + self.v1*ffp2 + self.v2*ffp3

    }

    /*pub fn geometric_median(&self) -> Option<Vector2<f32>> {
        let d_ab = self.v1 - self.v2;
        let d_bc = self.v2 - self.v3;
        let mut p_ab = to_polar(&d_ab);
        let mut p_bc = to_polar(&d_bc);

        // Rotate edges by 60º to get opposite-side vertex. Counter-clockwise (positive angle)
        // if difference is at upper half-circle; clockwise (negative angle) if difference is at
        // lower-half circle.
        p_ab[1] += (std::f32::consts::PI/3.)*d_ab[1].signum();
        p_bc[1] += (std::f32::consts::PI/3.)*d_bc[1].signum();
        let vert_ab = to_cartesian(&p_ab) + self.v2;
        let vert_bc = to_cartesian(&p_bc) + self.v3;

        Line2 { p1 : vert_ab, p2 : self.v3 }.intersection(&Line2 { p1 : vert_bc, p2 : self.v1 })
    }*/

}

// cargo test --lib -- trigeomedian --nocapture
#[test]
fn trigeomedian() {
    let a = Vector2::new(-0.5, 0.0);
    let b = Vector2::new(0.5, 0.0);
    let c = Vector2::new(0.0, 0.86602);
    let tr = Triangle::new(a, b, c);
    let m = tr.geometric_median();
    println!("{}", m);
    println!("ma = {}", m.metric_distance(&a));
    println!("mb = {}", m.metric_distance(&b));
    println!("mc = {}", m.metric_distance(&c));
}

fn median3(a : f32, b : f32, c : f32) -> f32 {
    (2.*a + 2.*b + (a+b-2.*c - (a-b).abs()).abs() - (a+b-2.*c+(a-b).abs()).abs()) / 4.
}

fn first_isogonic_center(a : f32, b : f32, c : f32) -> f32 {
    a.powf(4.) - 2.*(b.powf(2.) - c.powf(2.)).powf(2.) +
        a.powf(2.)*(b.powf(2.) + c.powf(2.) +
            (3.*(-a+b+c)*(a+b-c)*(a-b+c)*(a+b+c)).sqrt()
        )
}

fn to_polar(v : &Vector2<f32>) -> Vector2<f32> {
    Vector2::new(v.magnitude(), v[1].atan2(v[0]))
}

fn to_cartesian(v : &Vector2<f32>) -> Vector2<f32> {
    Vector2::new(v[1].cos() * v[0], v[1].sin() * v[0])
}

#[derive(Debug, Clone, Copy)]
pub struct MedianPoint {
    pub median : Vector2<f32>,
    pub niter : u32,
    pub err :  f32
}

fn guess_or_avg(pts : &[Vector2<f32>], guess : Option<Vector2<f32>>) -> Vector2<f32> {
    if let Some(guess) = guess {
        guess
    } else {
        point_average(pts)
    }
}

pub fn point_average(pts : &[Vector2<f32>]) -> Vector2<f32> {
    let mut avg = pts.iter().fold(Vector2::new(0.0, 0.0), |s, pt| s + pt );
    avg.scale_mut(1. / (pts.len() as f32));
    avg
}

// This is f(x) at Cohen's paper.
fn sum_euclid(pts : &[Vector2<f32>], center : &Vector2<f32>) -> f32 {
    pts.iter().fold(0., |s, pt| s + pt.metric_distance(center) )
}

mod acc_median {

    use super::*;
    use nalgebra::Matrix2;

    // gt(x)
    fn penalty(pt : &Vector2<f32>, center : &Vector2<f32>, t : f32) -> f32 {
        (1.0 + t.powf(2.)*(center - pt).norm_squared()).sqrt()
    }

    // This is f_t(x) at Cohen's paper.
    fn penalized_euclidian_dist(pts : &[Vector2<f32>], center : &Vector2<f32>, t : f32) -> f32 {
        pts.iter().fold(0., |s, pt| {
            let t_sq = t.powf(2.);
            let m = (center - pt).norm_squared();
            let gt = (1.0 + t_sq*m).sqrt();
            s + gt - (1. + gt).ln()
        })
    }

    pub fn path_param_update(f_star : f32, i : usize) -> f32 {
        (1. / (400. * f_star)) * (1. + 1. / 600.).powf(i as f32 - 1.0)
    }

    // Return the eigenvector to the largest eigenvalue of PSD matrix A.
    fn power_method(a : &Matrix2<f32>, niter : usize) -> Vector2<f32> {
        use rand::distributions::Distribution;
        let mut n = statrs::distribution::Normal::new(0., 1.0).unwrap();
        let mut rng = rand::thread_rng();
        let mut y = Vector2::new(n.sample(&mut rng) as f32, n.sample(&mut rng) as f32);
        for i in 0..niter {
            let y_unorm = a.pow(i as u32) * &y;
            y = y_unorm.normalize();
        }
        y
    }

    // Contribution of point to the hessian.
    fn weight(pts : &[Vector2<f32>], center : &Vector2<f32>, t : f32) -> f32 {
        pts.iter().fold(0.0, |s, pt| s + (1.0 / (1.0 + penalty(pt, center, t))) )
    }

    // Lemma 13 of Cohen's paper
    fn path_gradient(pts : &[Vector2<f32>], x : &Vector2<f32>, t : f32) -> Vector2<f32> {
        let t_sq = t.powf(2.);
        pts.iter().fold(Vector2::new(0.0, 0.0), |grad, pt| {
            grad + (x - pt).scale(t_sq / (1. + penalty(pt, x, t)))
        })
    }

    // Lemma 13 of Cohen's paper
    fn path_hessian(pts : &[Vector2<f32>], x : &Vector2<f32>, t : f32) -> Matrix2<f32> {
        let t_sq = t.powf(2.);
        pts.iter().fold(Matrix2::zeros(), |hess, pt| {
            let d = x.clone() - pt;
            let pen = penalty(pt, x, t);
            let scale_out = t_sq / (1. + pen);
            let scale_in = t_sq / (pen * (1. + pen));
            hess + (Matrix2::identity() - (d * d.transpose()).scale(scale_in)).scale(scale_out)
        })
    }

    // This is alg. 3 of Cohen (2016)
    // lambda, v are returned by approx_min_eigen
    fn local_center(
        pts : &[Vector2<f32>],
        y : &Vector2<f32>,
        t : f32,
        eps : f32,
    ) -> Vector2<f32> {
        // TODO use x here instead of y?
        let (lambda, v) = approx_min_eigen(pts, y, t, eps);
        let wq = weight(pts, y, t);
        let weighted_id = Matrix2::identity().scale(wq);
        let t_sq = (t as f32).powf(2.);
        let q = t_sq * weighted_id - (v * v.transpose()).scale(t_sq * wq - lambda);
        let mut x_i = y.clone();

        for i in 1..=((64.0 * (1.0 / eps).ln()) as usize) {
            let delta_prev = y.clone() - x_i;

            // The last term  ||x(i) - x(i-1)||_Q^2 is simply (x(i) - x(i-1))^T Q (x(i) - x(i-1))
            // (since the norm under q is the square root of the Q quadratic form, i.e. a Q-induced norm)
            let norm_q = ((delta_prev.transpose() * &q) * &delta_prev);
            let grad = path_gradient(pts, &x_i, t);

            // by Lemma 23 we have that the minimizer has value ft0 (xt0 ) and is achieved in the range [ 6f˜⇤ , 6f˜⇤ ].
            // x_i = min(sum_euclid(pts, x_i)) + grad.dot(&delta_prev) + 4.0*norm_q;
        }

        x_i
    }

    // This is alg. 4 of Cohen (2016)
    // u : eigenvec of approx_min_eigen (bad direction)
    pub fn line_search(
        pts : &[Vector2<f32>],
        f_star : f32,
        y : &Vector2<f32>,
        t : f32,
        t_next : f32,
        u : &Vector2<f32>,
        eps : f32
    ) -> Vector2<f32> {
        let eps_star = (1.0 / 3.0)*eps;
        let lower = -6.0*f_star;
        let upper = 6.0*f_star;
        let n = pts.len() as f32;
        let e_zero = ((eps*eps_star) / (160. * n.powf(2.))).powf(2.);

        /* TODO local_center takes evals/evecs of iteration t, NOT iteration t_next */

        // q is the oracle function.
        let q = Box::new(|alpha : f32| -> f32 {
            let lc = local_center(pts, &(y.clone() + u.scale(alpha)), t_next, e_zero);
            penalized_euclidian_dist(pts, &lc, t_next)
        });

        /* Line search over g(alpha), a convex function, using a binary search */
        let alpha_next = one_dim_minimizer(q,lower,upper,e_zero,t_next*(pts.len() as f32));

        // TODO original paper uses alpha here (not alpha_next)
        local_center(pts, &(y.clone() + u.scale(alpha_next)), t_next, e_zero)
    }

    // (l, u) is an interval in the real line
    // eps is a target error.
    // This is a binary search over the line along the
    // eigenvector with the smallest eigenvale (the "bad" direction,
    // that is steep therefore making x vary the most.). The argmin (alpha)
    // is a scale factor applied to the eigenvector (https://en.wikipedia.org/wiki/Line_search)
    fn one_dim_minimizer<G : Fn(f32)->f32>(
        g : G,
        l : f32,
        u : f32,
        eps : f32,
        lipschitz_bound : f32
    ) -> f32 {
        let max_iter = ((lipschitz_bound*(u - l))/eps).log(3.0/2.0).ceil() as usize;
        let mut x_i = l;
        let mut yl_i = l;
        let mut yu_i = u;
        for i in 1..=max_iter {
            let zl_i = (2.0*yl_i + yu_i) / 3.0;
            let zu_i = (yl_i + 2.0*yu_i) / 3.0;
            if g(zl_i) <= g(zu_i) {
                yu_i = zu_i;
                if g(zl_i) <= g(x_i) {
                    x_i = zl_i;
                }
            } else {
                yl_i = zl_i;
                if g(zu_i) <= g(x_i) {
                    x_i = zu_i;
                }
            }
        }
        x_i
    }

    // This is alg. 2 of Cohen (2016)
    // Return (eigenval, eigenvec pair)
    pub fn approx_min_eigen(pts : &[Vector2<f32>], median : &Vector2<f32>, t : f32, eps : f32) -> (f32, Vector2<f32>) {
        let mut a = Matrix2::zeros();
        for pt in pts {
            let d = median - pt;
            let pen = penalty(pt, median, t);
            let mut sq : Matrix2<f32> = (d * &d.transpose());
            let s = t.powf(4.) / ((1. + pen).powf(2.) * pen);
            sq.scale_mut(s);
            a += sq;
        }
        let niter = (pts.len() as f32 / eps).ln() as usize;
        let u = power_method(&a, niter);
        let lambda = (u.transpose() * path_hessian(pts, median, t) * u)[0];
        (lambda, u)
    }

}

fn update_median_point(
    pt : &Vector2<f32>,
    median : &Vector2<f32>,
    w : f32,
    next_median : &mut Vector2<f32>,
    weight_sum : &mut f32
) {
    let mut diff_prev = pt - median;
    let l1 = diff_prev.lp_norm(1).max(1.0e-20);
    let weight = w / l1;
    *next_median += pt.scale(weight);
    *weight_sum += weight;
}

fn check_median_step(
    median : &mut Vector2<f32>,
    next_median : &mut Vector2<f32>,
    niter : &mut u32,
    weight_sum : f32,
    tol : f32
) -> Option<MedianPoint> {
    next_median.scale_mut(1.0 / weight_sum);
    let err = median.metric_distance(&next_median);
    if err <= tol {
        Some(MedianPoint { median : next_median.clone(), niter : *niter, err })
    } else {
        *median = *next_median;
        *niter += 1;
        None
    }
}

impl MedianPoint {

    pub fn average_radius(&self, pts_f : &[Vector2<f32>]) -> f32 {
        let mut rad = pts_f.iter().fold(0.0, |s, pt| s + pt.metric_distance(&self.median) );
        rad /= pts_f.len() as f32;
        rad
    }

    // Implementation of the accurate median algorithm of  Cohen, Lee & Miller
    // Cohen, M. B., Lee, Y. T., Miller, G., Pachocki, J., & Sidford, A. (2016, June).
    // Geometric median in nearly linear time. In Proceedings of the forty-eighth annual ACM symposium
    // on Theory of Computing (pp. 9-21).
    /*
    From the paper (p.4) Let x_t for t increasing during the optimization be
    // the central path.
    lim t->inf x_t = x* (where x* is the solution). For any t, rapid
    changes in xt must occur in the direction of the smallest eigenvector
    of the hessian of the penalized objective f_t(x). This direction is
    v_t, the bad direction. Starting at x_t, there is a point y obtained
    by moving x_t in the bad direction, such that y and x_t' are close
    enough so that a first order optimization can be applied to converge
    to x_t'. Then increase t by a multiplicative constant so that the
    algorithm converge in logarithmic time of log(n/eps) (number of points
    divided by desired accuracy). The algorithm applies a linear search
    along the bad path to find the next point on the central path.

    The region ||x-y||_2 is bounded by O(1/t) (decrease with t), so
    the line search is limited to a convex function g found approximately
    by a centering procedure, that can be operated on using binary search.

    This is alg. 1 of Cohen (2016)
    */
    pub fn calculate_accurate(
        &self,
        pts : &[Vector2<f32>],
        eps : f32,
        weights : Option<&[f32]>,
        guess : Option<Vector2<f32>>
    ) -> Option<Vector2<f32>> {
        assert!(eps > 0. && eps <= 1.0);
        let n = pts.len() as f32;
        let mut median = guess_or_avg(pts, guess);
        let f_star = sum_euclid(pts, &median);

        // Path parameter to penalized weight

        let eps_star = (1.0 / 3.0)*eps;
        let t_star = (2.0*n) / (eps_star * f_star);
        let e_v = (1.0/8.0)*(eps_star / (7.0*n)).powf(2.);
        let e_c = (e_v / 36.0).powf(3.0/2.0);

        let mut i = 1;
        let mut t_i = acc_median::path_param_update(f_star, i);
        median = acc_median::line_search(pts, f_star, &median, t_i, t_i, &Vector2::zeros(), e_c);
        while t_i <= t_star {
            t_i = acc_median::path_param_update(f_star, i);
            let t_i_next = acc_median::path_param_update(f_star, i+1);

            // u_i is an approximation to the bad direction.
            let (lambda_i, u_i) = acc_median::approx_min_eigen(pts, &median, t_i, e_v);

            median = acc_median::line_search(
                pts,
                f_star,
                &median,
                t_i,
                t_i_next,
                &u_i,
                e_c
            );
            t_i = t_i_next;
            i += 1;
        }
        Some(median)
    }

    // Weighted Weiszfeld algorithm.
    // The weighted generalization is presented at
    // Beck & Sabach (2015) Weisfeld's Method: Old and
    // new results (p. 5).
    pub fn calculate_weighted(
        pts : &[Vector2<f32>],
        weights : &[f32],
        tol : f32,
        maxiter : u32,
        guess : Option<Vector2<f32>>
    ) -> Option<Self> {

        // Start with the average as a first guess.
        let mut median = guess_or_avg(pts, guess);

        let mut niter = 0;
        while niter <= maxiter {
            let mut next_median = Vector2::new(0.0, 0.0);
            let mut weight_sum = 0.0;
            for (pt,w) in pts.iter().zip(weights.iter()) {
                update_median_point(pt, &median, *w, &mut next_median, &mut weight_sum);
            }
            if let Some(ans) = check_median_step(&mut median, &mut next_median, &mut niter, weight_sum,tol) {
                return Some(ans);
            }
        }
        None
    }

    // Calculate the geometric median with the Weiszfeld algorithm.
    // Iterates while dist(prev_median, curr_median) > tol.
    // https://en.wikipedia.org/wiki/Geometric_median
    pub fn calculate(pts : &[Vector2<f32>], tol : f32, maxiter : u32, guess : Option<Vector2<f32>>) -> Option<Self> {

        // Start with the average as a first guess.
        let mut median = guess_or_avg(pts, guess);

        let mut niter = 0;
        while niter <= maxiter {
            let mut next_median = Vector2::new(0.0, 0.0);
            let mut weight_sum = 0.0;
            for pt in pts {
                update_median_point(pt, &median, 1.0, &mut next_median, &mut weight_sum);
            }
            if let Some(ans) = check_median_step(&mut median, &mut next_median, &mut niter, weight_sum,tol) {
                return Some(ans);
            }
        }
        None
    }
}

#[derive(Debug, Clone)]
pub struct CircleEstimator {
    centered_ptsf : Vec<Vector2<f32>>,
    x_sq : Vec<f32>,
    y_sq : Vec<f32>
}

impl CircleEstimator {

    pub fn new() -> Self {
        Self {
            centered_ptsf : Vec::with_capacity(32),
            x_sq : Vec::with_capacity(32),
            y_sq : Vec::with_capacity(32),
        }
    }

    // This doesn't really work because the algorihtm expects
    // a mean estimate of the center. Using the median point
    // overestimates the radius.
    // Estimate with the median point as the center estimate.
    // The median point minimizes the l1 norms of the points, therefore
    // it is more robust to outliers, but requires an iterative
    // method to be found, therefore it is more computationally demanding.
    /*pub fn estimate_with_median(&mut self, ptsf : &[Vector2<f32>], tol : f32, max_iter : u32, guess : Option<Vector2<f32>>) -> Option<Circle> {
        let m = MedianPoint::calculate(ptsf, tol, max_iter, guess)?;
        self.estimate_with_centroid(ptsf, &m.median)
    }*/

    // Estimate with the centroid as the center estimate. The
    // centroid minimizes the lenght of the projections along the
    // x and y axes, making this method cheap but sensitive to
    // outliers.
    pub fn estimate(&mut self, ptsf : &[Vector2<f32>]) -> Option<Circle> {
        let center = centroid(ptsf);
        self.estimate_with_centroid(ptsf, &center)
    }

    pub fn estimate_with_centroid(&mut self, ptsf : &[Vector2<f32>], centroid : &Vector2<f32>) -> Option<Circle> {
        if ptsf.len() < 3 {
            return None;
        }
        self.centered_ptsf.clear();
        self.x_sq.clear();
        self.y_sq.clear();

        self.centered_ptsf.extend(ptsf.iter().map(|pt| pt.clone() - centroid ));
        self.x_sq.extend(self.centered_ptsf.iter().map(|pt| pt[0].powf(2.) ));
        self.y_sq.extend(self.centered_ptsf.iter().map(|pt| pt[1].powf(2.) ));

        let (mut x_sq, mut y_sq, mut x_cub, mut y_cub, mut xy, mut yyx, mut xxy) = (0., 0., 0., 0., 0., 0., 0.);
        for i in 0..self.centered_ptsf.len() {
            let x = self.centered_ptsf[i][0];
            let y = self.centered_ptsf[i][1];
            x_sq += self.x_sq[i];
            y_sq += self.y_sq[i];
            x_cub += x * self.x_sq[i];
            y_cub += y * self.y_sq[i];
            xy += x * y;
            yyx += x * self.y_sq[i];
            xxy += self.x_sq[i] * y;
        }

        let m = Matrix2::from_rows(&[RowVector2::new(x_sq, xy), RowVector2::new(xy, y_sq)]);
        let b = Vector2::new(
            0.5 * (x_cub + yyx),
            0.5 * (y_cub + xxy)
        );
        let ans = LU::new(m).solve(&b)?;
        // let center = Vector2::new(ans[0] + center_x, ans[1] + center_y);
        let center = ans + centroid;
        let n = ptsf.len() as f32;
        let radius = (ans[0].powf(2.) + ans[1].powf(2.) + (x_sq + y_sq) / n).sqrt();
        Some(Circle { center, radius })
    }

}

#[derive(Debug, Clone, Copy)]
pub struct Line2 {
    pub p1 : Vector2<f32>,
    pub p2 : Vector2<f32>
}

impl Line2 {

    pub fn intersection(&self, other : &Self) -> Option<Vector2<f32>> {
        let l1 = Matrix2::from_rows(&[self.p1.transpose(), self.p2.transpose()]);
        let l2 = Matrix2::from_rows(&[other.p1.transpose(), other.p2.transpose()]);
        line_intersection(l1, l2)
    }

    pub fn midpoint(&self) -> Vector2<f32> {
        self.p1.scale(0.5) + self.p2.scale(0.5)
    }

    pub fn new(p1 : Vector2<f32>, p2 : Vector2<f32>) -> Self {
        Self { p1, p2 }
    }

    pub fn collinear(&self, pt : &Vector2<f32>, tol : f32) -> bool {
        (self.angle() - (Line2 { p1 : self.p1, p2 : pt.clone() }).angle()).abs() < tol
    }

    pub fn length(&self) -> f32 {
        self.p1.metric_distance(&self.p2)
    }

    /* Build the vector p2 centered at p1 */
    pub fn offset(&self) -> Vector2<f32> {
        self.p2 - self.p1
    }

    pub fn angle(&self) -> f32 {
        let off = self.offset();
        off[1].atan2(off[0])
    }

}

#[derive(Clone, Debug, Copy)]
pub enum Incidence {
    Tangent(Vector2<f32>),
    Secant(Line2)
}

// cargo test --lib -- test_circle --nocapture
#[test]
fn test_circle() {
    let c = Circle { center : Vector2::new(10.0, 10.0), radius : 10.0 };
    let l = Line2 { p1 : Vector2::new(1.0, 1.0), p2 : Vector2::new(30.0, 30.0) };
    println!("{:?}", c.incidence(&l, 0.01));
}

pub fn circular_trace(
    center : (usize, usize),
    radius : usize,
    img_sz : (usize, usize),
    n_pts : usize
) -> Vec<(usize, usize)> {
    (0..n_pts).filter_map(|ix| circular_coord(ix, n_pts, center, radius, img_sz) ).collect()
}

const TWO_PI : f64 = 6.283185307;

pub fn circular_coord_at_theta(
    theta : f64,
    center : (usize, usize),
    radius : usize,
    (height, width) : (usize, usize)
) -> Option<(usize, usize)> {
    let x = center.1 as f64 + theta.cos() * radius as f64;
    let y = height.checked_sub(center.0)? as f64 + theta.sin() * radius as f64;
    if x >= 0.0 && x < width as f64 {
        if y >= 0.0 && y < height as f64 {
            let i = height.checked_sub(y as usize)?;
            let j = x as usize;
            Some((i, j))
        } else {
            None
        }
    } else {
        None
    }
}

// Ix: Coordinate over 0..n_points
pub(crate) fn circular_coord(
    ix : usize,
    n_points : usize,
    center : (usize, usize),
    radius : usize,
    (height, width) : (usize, usize)
) -> Option<(usize, usize)> {
    let theta = (ix as f64 / n_points as f64) * TWO_PI;
    circular_coord_at_theta(theta, center, radius, (height, width))
}

impl Circle {

    //pub fn contains(&self, pt : &Vector2<f32>) -> bool {
    //    pt.metric_distance(&self.center) < self.radius
    //}

    /* If line intersects the circle at two points, return the pair of points.
    The points are ordered by their distance to p1
    https://mathworld.wolfram.com/Circle-LineIntersection.html
    Tangent_tol is used to decide on whether the line is tangent or secant
    based on its proximity to circle. If the line is closer to the circle
    than tangent tol, it is considered a tangent. It is considered a secant
    otherwise. */
    pub fn incidence(&self, line : &Line2, tangent_tol : f32) -> Option<Incidence> {
        let line_c = Line2 {
            p1 : line.p1 - self.center,
            p2 : line.p2 - self.center
        };
        let dx = line_c.p2[0] - line_c.p1[0];
        let dy = line_c.p2[1] - line_c.p1[1];
        let dr = dx.hypot(dy);
        if dr == 0.0 {
            return None;
        }

        // This differs from f32::signum in that it cannot return zero.
        let sgn = |x : f32| if x < 0.0 { -1.0 } else { 1.0 };

        // Determinant of [[x1, x2], [y1, y2]]
        let det = line_c.p1[0] * line_c.p2[1] - line_c.p2[0] * line_c.p1[1];
        let incidence = self.radius.powf(2.)*dr.powf(2.) - det.powf(2.);
        if incidence < 0.0 {
            None
        } else if incidence.abs() < tangent_tol {
            let x = (det*dy) / dr.powf(2.);
            let y = ((-det)*dx) / dr.powf(2.);
            Some(Incidence::Tangent(Vector2::new(x, y) + self.center))
        } else {
            let incidence_root = incidence.sqrt();
            let x1 = (det*dy + sgn(dy)*dx * incidence_root) / dr.powf(2.);
            let x2 = (det*dy - sgn(dy)*dx * incidence_root) / dr.powf(2.);
            let y1 = ((-det)*dx + dy.abs() * incidence_root) / dr.powf(2.);
            let y2 = ((-det)*dx - dy.abs() * incidence_root) / dr.powf(2.);
            let v1 = Vector2::new(x1, y1) + self.center;
            let v2 = Vector2::new(x2, y2) + self.center;
            let l = if v1.metric_distance(&line.p1) < v2.metric_distance(&line.p1) {
                Line2 { p1 : v1, p2 : v2 }
            } else {
                Line2 { p1 : v2, p2 : v1 }
            };
            Some(Incidence::Secant(l))
        }
    }

    pub fn contains(&self, pt : &Vector2<f32>) -> bool {
        self.center.metric_distance(&pt) < self.radius
    }

    pub fn encloses(&self, other : &Self) -> bool {
        self.center.metric_distance(&other.center) + other.radius < self.radius
    }

    pub fn circumference(&self) -> f32 {
        2.0*std::f32::consts::PI*self.radius
    }

    pub fn area(&self) -> f32 {
        std::f32::consts::PI*self.radius.powf(2.)
    }

    pub fn coords(&self, img_sz : (usize, usize)) -> Option<CircleCoords> {
        if self.center[0] < 0.0 || self.center[1] < 0.0 {
            return None;
        }
        let center = (img_sz.0.checked_sub(self.center[1].round() as usize)?, self.center[0].round() as usize);
        let radius = self.radius.round() as usize;
        if center.0 + radius < img_sz.0 || center.1 + radius < img_sz.1 {
            Some(CircleCoords { center, radius })
        } else {
            None
        }
    }

    pub fn enclosing_from_points(pts : &[Vector2<f32>]) -> Option<Self> {
        if pts.len() < 2 {
            return None;
        }
        let mut farthest = ((0, 0), 0.0);
        for (i, a) in pts.iter().enumerate() {
            for (j, b) in pts[(i+1)..].iter().enumerate() {
                let dist = (a.clone() - b).magnitude();
                if dist > farthest.1 {
                    farthest = ((i, i + 1 + j), dist);
                }
            }
        }
        let mut diff = pts[farthest.0.0].clone() - &pts[farthest.0.1];
        let radius = diff.magnitude() / 2.0;
        diff.scale_mut(0.5);
        let center = diff + &pts[farthest.0.1];
        Some(Circle { center, radius })
    }

    pub fn enclosing(pts : &[(usize, usize)], img_height : usize) -> Option<Self> {
        let ptsf : Vec<Vector2<f32>> = pts.iter()
            .map(|pt| Vector2::new(pt.1 as f32, (img_height - pt.0) as f32) ).collect();
        Self::enclosing_from_points(&ptsf[..])
    }

    pub fn calculate(
        pts : &[(usize, usize)], 
        img_height : usize
    ) -> Option<Self> {
        /*let ptsf : Vec<Vector2<f32>> = pts.iter()
            .map(|pt| Vector2::new(pt.1 as f32, (img_height - pt.0) as f32) ).collect();
        Self::calculate_from_points(&ptsf[..])*/
        unimplemented!()
    }

    pub fn center_coord(&self, img_height : usize) -> Option<(usize, usize)> {
        if (self.center[0] > 0. && self.center[1] > 0.) && (self.center[1] as usize) < img_height {
            Some((img_height - self.center[1] as usize, self.center[0] as usize))
        } else {
            None
        }
    }

    pub fn radius_variance_from_pts(&self, ptsf : &[Vector2<f32>]) -> f32 {
        let n = ptsf.len() as f32;
        ptsf.iter().fold(0.0, |acc, pt| acc + ((pt - &self.center).magnitude() - self.radius).powf(2.)  ) / n
    }

    pub fn abs_errors(&self, ptsf : &[Vector2<f32>]) -> Vec<f32> {
        ptsf.iter().map(|pt| ((pt - &self.center).magnitude() - self.radius).abs()  ).collect()
    }

    pub fn avg_abs_error(&self, ptsf : &[Vector2<f32>]) -> f32 {
        let n = ptsf.len() as f32;
        ptsf.iter().fold(0.0, |acc, pt| acc + ((pt - &self.center).magnitude() - self.radius).abs()  ) / n
    }

    // variance of the random variable (dist(pt, center) - radius), which is a measure of fit quality.
    pub fn radius_variance(&self, pts : &[(u16, u16)], img_height : u16) -> f32 {
        let ptsf : Vec<Vector2<f32>> = pts.iter().map(|pt| Vector2::new(pt.1 as f32, (img_height - pt.0) as f32) ).collect();
        self.radius_variance_from_pts(&ptsf[..])
    }

    pub fn iter_circumference(&self, n_pts : u32) -> Vec<Vector2<f32>> {
        let sector_arc = (2.0 * std::f32::consts::PI) / n_pts as f32;
        (0..n_pts).map(|s| self.circumference_at(sector_arc*(s as f32)) ).collect()
    }

    pub fn circumference_at(&self, deg_rad : f32) -> Vector2<f32> {
       self.center.clone() + (Vector2::new(deg_rad.cos(), deg_rad.sin()) * self.radius)
    }

}

/* Packed interleaved indices (y, x, y, x, ... y, x) representation,
useful for vectorized ops. */
struct InterIndices(Vec<usize>);

/* Sequential indices (y, y, y, ... x, x, x) representation,
useful for vectorized ops. */
struct SeqIndices(Vec<usize>);

use petgraph::graph::NodeIndex;
use std::iter::FromIterator;

// TODO implement PartialOrd for Event by y-coordinate.
enum Event {

    // Sweep line met a new point
    Site(Vector2<f32>),

    // Sweep line met a circle lower point.
    Circle(Vector2<f32>, NodeIndex<usize>)
}

// Computes the Voronoi diagram using Fortune's algorithm (plane sweep)
fn voronoi_diagram(pts : &[Vector2<f32>]) {

    // Priority queue of events
    let mut queue : Vec<Event> = Vec::from_iter(pts.iter().map(|pt| Event::Site(pt.clone() )));
    // queue.sort_by(|a, b| a[1].cmp(&b[1]) );

    while !queue.is_empty() {
        match queue.pop().unwrap() {
            Event::Site(site) => {
                handle_site_event(&site)
            },
            Event::Circle(lower_pt, ix) => {
                handle_circle_event(&lower_pt)
            }
        }
    }

}

fn handle_site_event(site : &Vector2<f32>) {

}

fn handle_circle_event(lower_pt : &Vector2<f32>) {

}

/* Note orientation follows y-down convention */
#[derive(Debug, Clone)]
pub struct Orientation {
    pub u1 : Vector2<f32>,
    pub u2 : Vector2<f32>,
    pub center : (f32, f32),
    pub mass : f32,
    pub lambda_1 : f32,
    pub lambda_2 : f32,
    img_height : usize,
}

impl Orientation {

    pub fn bounded_region(&self) -> Region {
        let c_u = (self.center.0.round() as usize, self.center.1.round() as usize);
        let el = self.ellipse();
        let dy = el.major[1].max(el.minor[1]).abs().ceil() as usize;
        let dx = el.major[0].max(el.minor[0]).abs().ceil() as usize;
        Region::new(
            (c_u.0.saturating_sub(dy), c_u.1.saturating_sub(dx)),
            (2*dy, 2*dx)
        )
    }

    // The central point, in y-up orientation.
    pub fn center(&self) -> Vector2<f32> {
        Vector2::new(self.center.1, self.img_height as f32 - self.center.0)
    }

    // The return ellipse follows y-up orientation.
    pub fn ellipse(&self) -> ellipse::Ellipse {
        let center = self.center();
        let a1 = (2.*self.lambda_1.sqrt())*self.u1;
        let a2 = (2.*self.lambda_2.sqrt())*self.u2;
        let u1_rev = Vector2::new(a1[0], -1.0*a1[1]);
        let u2_rev = Vector2::new(a2[0], -1.0*a2[1]);
        ellipse::Ellipse { center, major : u1_rev, minor : u2_rev }
    }

    /* Gives angle over the top-right quadrant */
    pub fn angle(&self) -> f32 {
        self.u1[1].asin().abs()
    }

    // For circle, eccentricity=0
    // For ellipse, eccentricity \in [0,1]
    pub fn eccentricity(&self) -> f32 {
        (1.0 - (self.lambda_2 / self.lambda_1)).sqrt()
    }

    pub fn largest_radius(&self) -> f32 {
        let r = self.radii();
        r.0.max(r.1)
    }

    pub fn smallest_radius(&self) -> f32 {
        let r = self.radii();
        r.0.min(r.1)
    }

    pub fn radii(&self) -> (f32, f32) {
        (2.*self.lambda_1.sqrt(), 2.*self.lambda_2.sqrt())
    }

}

/*/* Convert to an ellipse in y-up coordinate space. */
impl Into<ellipse::Ellipse> for Orientation {

    fn into(self) -> ellipse::Ellipse {

    }

}*/

impl<S> Image<u8, S>
where
    S : crate::image::Storage<u8>
{

    #[cfg(feature="ipp")]
    pub fn orientation(
        &self,
        mt : Option<&mut IppiCentralMoments>,
    ) -> Option<Orientation> {
        let mut mt = Brief::from(mt, || IppiCentralMoments::new() );
        let m = mt.calculate(&self.full_window(), true)?;
        m.orientation(self.height())
    }

}

use std::ops::{Deref, DerefMut};

/* Similar to std::borrow::Cow, provides two variants, one for an allocated object
and another for its reference (but mutable in this case). On construction, takes a closure
that allocates lazily when a mutable reference is not informed. The object is
de-allocated after its use, hence its name. Useful when a method requires a work buffer-type
structure, when the allocation logic is known or can be inferred from the arguments,
and the user might not want/need to explicitly allocate it on some occasions (unique runs,
in which the user does not need to explicitly allocate the structure, therefore increasing
code simplicity and minimizing the errors from mismatches of the structure parameters
and the function parameteers), or might want to explicitly allocate it (method called many times). */
pub enum Brief<'a, T> {
    Borrowed(&'a mut T),
    Owned(T)
}

impl<'a, T> From<Option<&'a mut T>> for Brief<'a, T>
where
    T : Default
{

    fn from(opt_mut : Option<&'a mut T>) -> Self {
        match opt_mut {
            Some(m) => Brief::Borrowed(m),
            None => Brief::Owned(T::default())
        }
    }

}

impl<'a, T> Brief<'a, T> {

    pub fn from(opt_mut : Option<&'a mut T>, alloc : impl Fn()->T) -> Self {
        match opt_mut {
            Some(m) => Brief::Borrowed(m),
            None => Brief::Owned(alloc())
        }
    }

}

impl<'a, T> Deref for Brief<'a, T> {

    type Target = T;

    fn deref(&self) -> &T {
        match self {
            Brief::Borrowed(m) => &*m,
            Brief::Owned(ref m) => m
        }
    }

}

impl<'a, T> DerefMut for Brief<'a, T> {

    fn deref_mut(&mut self) -> &mut T {
        match self {
            Brief::Borrowed(m) => m,
            Brief::Owned(ref mut m) => m
        }
    }

}

pub fn path(src : (usize, usize), dst : (usize, usize), nrow : usize) -> Vec<(usize, usize)> {
    let (dist, theta) = crate::image::index::index_distance(src, dst, nrow);
    let d_max = dist as usize;
    (0..=d_max).map(|i| line_position(src, theta, i) ).collect()
}

/* Returns the position (y, x) when starting at src and walking
straight along angle theta and arriving at position i */
pub fn line_position(src : (usize, usize), theta : f64, i : usize) -> (usize, usize) {
    let x_incr = theta.cos() * i as f64;
    let y_incr = theta.sin() * i as f64;
    let x_pos = (src.1 as i32 + x_incr as i32) as usize;
    let y_pos = (src.0 as i32 - y_incr as i32) as usize;
    (y_pos, x_pos)
}

