use crate::image::*;
use crate::edge::*;
use away::Metric;
use std::cmp::{PartialEq, Ordering};

pub fn rect_overlap(r1 : &(usize, usize, usize, usize), r2 : &(usize, usize, usize, usize)) -> bool {
    let tl_vdist = (r1.0 as i32 - r2.0 as i32).abs();
    let tl_hdist = (r1.1 as i32 - r2.1 as i32).abs();
    let (h1, w1) = (r1.2 as i32, r1.3 as i32);
    let (h2, w2) = (r2.2 as i32, r2.3 as i32);
    (tl_vdist < h1 || tl_vdist < h2) && (tl_hdist < w1 || tl_hdist < w2)
}

/// A cricle has circularity of 1; Other polygons have circularity < 1.
pub fn circularity(area : f64, perim : f64) -> f64 {
    (4. * std::f64::consts::PI * area) / perim.powf(2.)
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
        if (p1.0 as f64, p1.1 as f64).manhattan(&(p2.0 as f64, p2.1 as f64)) < max_dist {
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
        if (c2[0].0 as f64, c2[0].1 as f64).manhattan(&(c1[1].0 as f64, c1[1].1 as f64)) < max_dist {
            clusters.push([c1[0], c1[1], c2[0], c2[1]]);
        }
    }
    clusters
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

    pub fn fit_circle(pts : &[(usize, usize)]) -> Result<((usize, usize), usize), String> {
        let ellipse = fit_ellipse(pts)?;
        let radius = (ellipse.large_axis / 2.) as usize;
        Ok((ellipse.center, radius))
    }

    // TODO make WindowMut
    pub fn draw_ellipse(window : Window<'_, u8>, el : &Ellipse) {
        let thickness = 2;
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

    /// Returns position and radius of fitted circle. Also see fit_ellipse_ams; fit_ellipse_direct.
    pub fn fit_ellipse(pts : &[(usize, usize)]) -> Result<Ellipse, String> {

        let pt_vec = convert_points(pts);
        let rotated_rect = imgproc::fit_ellipse(&pt_vec)
            .map_err(|e| format!("Ellipse fitting error ({})", e))?;
        let center_pt = rotated_rect.center();
        if center_pt.y < 0.0 || center_pt.x < 0.0 {
            return Err(format!("Circle outside image boundaries"));
        }
        let center = (center_pt.y as usize, center_pt.x as usize);
        let angle = rotated_rect.angle();
        let size = rotated_rect.size();
        let angle = rotated_rect.angle();
        Ok(Ellipse {
            center,
            large_axis : rotated_rect.size().width as f64,
            small_axis : rotated_rect.size().height as f64,
            angle : angle as f64
        })
    }

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



