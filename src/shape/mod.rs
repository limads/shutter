mod ellipse;

pub use ellipse::*;

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

    use nalgebra::{Vector2, Scalar};
    use num_traits::{AsPrimitive, Zero};
    use std::cmp::PartialOrd;

    // Maps coord to vector with strictly positive entries with origin at the bottom-left
    // pixel in the image.
    pub fn coord_to_point<F>(coord : (usize, usize), shape : (usize, usize)) -> Option<Vector2<F>>
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

// convex hull implementation, based on
// https://rosettacode.org/wiki/Convex_hull
mod convex {

    #[derive(Debug, Clone)]
    struct Point {
        x: f32,
        y: f32
    }

    fn calculate_convex_hull(points: &Vec<Point>) -> Vec<Point> {
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

    //Calculate orientation for 3 points
    //0 -> Straight line
    //1 -> Clockwise
    //2 -> Counterclockwise
    fn orientation(p: &Point, q: &Point, r: &Point) -> usize {
        let val = (q.y - p.y) * (r.x - q.x) -
            (q.x - p.x) * (r.y - q.y);

        if val == 0. { return 0 };
        if val > 0. { return 1; } else { return 2; }
    }

}

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