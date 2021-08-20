use opencv::imgproc;
use crate::image::*;
use crate::threshold;
use opencv::core;
use opencv::prelude::RotatedRectTrait;
use away::Metric;

// An edge is a Vec<(usize, usize)>. Two edges intersect if at least one
// of the instances of the cartesian product of their sub-edges (neighboring pairs of points forming edges of size 2)
// intersect. Edges for which their enclosing rectangle do not match have no change of matching and can be excluded.
// An edge intersection can be found if their enclosing rectangles match. The point is defined by the solution of
// the simple 2x2 linear system with the line equations for the edges. Intersecting edges define vertices.
pub struct Edge(Vec<(usize, usize)>);

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

fn convert_points(pts : &[(usize, usize)]) -> core::Vector<core::Point2i> {
    let mut pt_vec = core::Vector::new();
    for pt in pts.iter() {
        pt_vec.push(core::Point2i::new(pt.1 as i32, pt.0 as i32));
    }
    pt_vec
}

// pub enum EllipseError {
// }

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



