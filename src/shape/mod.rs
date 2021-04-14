use opencv::imgproc;
use crate::image::*;
use crate::threshold;
use opencv::core;
use opencv::prelude::RotatedRectTrait;

fn convert_points(pts : &[(usize, usize)]) -> core::Vector<core::Point2i> {
    let mut pt_vec = core::Vector::new();
    for pt in pts.iter() {
        pt_vec.push(core::Point2i::new(pt.1 as i32, pt.0 as i32));
    }
    pt_vec
}

// pub enum EllipseError {
// }

/// Returns position and radius of fitted circle. Also see fit_ellipse_ams; fit_ellipse_direct.
pub fn fit_ellipse(pts : &[(usize, usize)]) -> Result<((usize, usize), usize), String> {
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
    let width = size.width;
    let radius = (width / 2.) as usize;
    Ok((center, radius))
}

pub fn enclosing_circle(pts : &[(usize, usize)]) -> Result<((usize, usize), usize), String> {
    let pt_vec = convert_points(pts);
    let mut center = core::Point2f{ x : 0.0, y : 0.0 };
    let mut radius = 0.0;
    let ans = imgproc::min_enclosing_circle(
        &pt_vec,
        &mut center,
        &mut radius
    );
    match ans {
        Ok(_) => Ok(((center.y as usize, center.x as usize), radius as usize)),
        Err(e) => Err(format!("{}", e))
    }
}


