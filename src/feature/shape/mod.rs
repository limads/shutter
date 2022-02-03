use crate::image::*;
use crate::feature::edge::*;
use std::cmp::{PartialEq, Ordering};
use nalgebra::*;
use away::{Manhattan, Metric};

/// Calculates the euclidian distance between two points. By convention the row coordinate will
/// come first, but since the distance is scalar (not vector) quantity, the order of the coordinate
/// isn't relevant.
pub fn point_euclidian(a : (usize, usize), b : (usize, usize)) -> f32 {
    ((a.0 as f32 - b.0 as f32).powf(2.) + (a.1 as f32 - b.1 as f32).powf(2.)).sqrt()
}

/// Calculate the angle formed by side_a and side_b given the opposite side using the law of cosines.
pub fn angle(side_a : f64, side_b : f64, opp_side : f64) -> f64 {
	let angle_cos = (side_a.powf(2.) + side_b.powf(2.) - opp_side.powf(2.)) / (2. * side_a * side_b);
	angle_cos.acos()
}

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

pub fn top_left_coordinate(r : &(usize, usize, usize, usize)) -> (usize, usize) {
    (r.0, r.1)
}

pub fn bottom_right_coordinate(r : &(usize, usize, usize, usize)) -> (usize, usize) {
    (r.0 + r.2, r.1 + r.3)
}

pub fn rect_overlaps(r1 : &(usize, usize, usize, usize), r2 : &(usize, usize, usize, usize)) -> bool {
    /*let tl_vdist = (r1.0 as i32 - r2.0 as i32).abs();
    let tl_hdist = (r1.1 as i32 - r2.1 as i32).abs();
    let (h1, w1) = (r1.2 as i32, r1.3 as i32);
    let (h2, w2) = (r2.2 as i32, r2.3 as i32);
    (tl_vdist < h1 || tl_vdist < h2) && (tl_hdist < w1 || tl_hdist < w2)*/

    let tl_1 = top_left_coordinate(r1);
    let tl_2 = top_left_coordinate(r2);
    let br_1 = bottom_right_coordinate(r1);
    let br_2 = bottom_right_coordinate(r2);

    let to_left = br_2.1 < tl_1.1;
    let to_right = tl_2.1 > br_1.1;
    let to_top = br_2.0 < tl_1.0;
    let to_bottom = tl_2.0 > br_1.0;

    !(to_left || to_top || to_right || to_bottom)

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
    println!("{:?}", line_intersection(((0, 0), (5, 5)), ((5, 0), (0, 5))) );
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
fn line_intersection(
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

    let diff_x_l1 = line1_pt1.1 as f64 - line1_pt2.1 as f64;
    let diff_x_l2 = line2_pt1.1 as f64 - line2_pt2.1 as f64;
    let diff_y_l1 = (line1_pt1.0 as f64 * -1.) - (line1_pt2.0 as f64 * -1.);
    let diff_y_l2 = (line2_pt1.0 as f64 * -1.) - (line2_pt2.0 as f64 * -1.);

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

                if let Some(intersection) = line_intersection((major_pt1, major_pt2), (*pt1, *pt2)) {

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

    let center = line_intersection((major_pt1, major_pt2), (minor_pt1, minor_pt2))?;
    let large_axis = euclidian(&[major_pt1.0 as f64, major_pt1.1 as f64], &[major_pt2.0 as f64, major_pt2.1 as f64]);
    let small_axis = euclidian(&[minor_pt1.0 as f64, minor_pt1.1 as f64], &[minor_pt2.0 as f64, minor_pt2.1 as f64]);

    // Calculate angle of center with respect to x axis.
    let angle = vertex_angle(center, major_pt2, (center.0, center.1 + small_axis as usize)).unwrap();

    Some(Ellipse {
        center,
        large_axis,
        small_axis,
        angle
    })
}

pub fn join_col_ordered(pts : &[(usize, usize)], max_dist : f64) -> Vec<[(usize, usize); 4]> {
    let pairs = join_single_col_ordered(pts, max_dist);
    join_pairs_col_ordered(&pairs[..], max_dist)
}

pub struct Ellipse {
    pub center : (usize, usize),
    pub large_axis : f64,
    pub small_axis : f64,
    pub angle : f64
}

impl Ellipse {

    fn major_axis_points(&self) -> (usize, usize) {
        // large_axis * angle.to_radians().sin()
        unimplemented!()
    }

    fn minor_axis_points(&self) -> (usize, usize) {
        // small_axis * angle.to_radians().cos()
        unimplemented!()
    }
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

#[cfg(feature="opencvlib")]
pub mod cvellipse {

    use super::*;
    use opencv::core;
    use opencv::imgproc;
    use opencv::prelude::RotatedRectTrait;

    pub fn convert_points(pts : &[(usize, usize)]) -> core::Vector<core::Point2i> {
        let mut pt_vec = core::Vector::new();
        for pt in pts.iter() {
            pt_vec.push(core::Point2i::new(pt.1 as i32, pt.0 as i32));
        }
        pt_vec
    }

    pub fn fit_circle(pts : &[(usize, usize)], method : Method) -> Result<((usize, usize), usize), String> {
        let ellipse = EllipseFitting::new().fit_ellipse(pts, method)?;
        let radius = ((ellipse.large_axis*0.5 + ellipse.small_axis*0.5) / 2.) as usize;
        Ok((ellipse.center, radius))
    }

    // TODO make WindowMut
    pub fn draw_ellipse(window : Window<'_, u8>, el : &Ellipse) {
        let thickness = 1;
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
            core::Scalar::from((255f64, 0f64, 0f64)),
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
        pt_vec : core::Vector<core::Point2i>
    }

    impl EllipseFitting {

        pub fn new() -> Self {
            Self { pt_vec : core::Vector::new() }
        }

        /// Returns position and radius of fitted circle. Also see fit_ellipse_ams; fit_ellipse_direct.
        pub fn fit_ellipse(&mut self, pts : &[(usize, usize)], method : Method) -> Result<Ellipse, String> {

            // let pt_vec = convert_points(pts);
            self.pt_vec.clear();
            for pt in pts.iter() {
                self.pt_vec.push(core::Point2i::new(pt.1 as i32, pt.0 as i32));
            }

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
            let center = (center_pt.y as usize, center_pt.x as usize);
            let angle = rotated_rect.angle();
            let size = rotated_rect.size();
            if rotated_rect.size().width as i32 <= 0 || rotated_rect.size().height as i32 <= 0 {
                return Err(format!("Invalid ellipse dimension"));
            }
            let w = rotated_rect.size().width as f64;
            let h = rotated_rect.size().height as f64;
            let large_axis = w.max(h);
            let small_axis = w.min(h);
            assert!(large_axis >= small_axis);
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
