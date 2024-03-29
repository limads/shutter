use nalgebra::*;
use std::f32::consts::{SQRT_2, PI};
use nalgebra_lapack;
use serde::{Serialize, Deserialize};
use std::mem;
use nalgebra::linalg::SVD;
use std::collections::BTreeSet;
use crate::shape::Region;
use crate::shape::Area;

pub trait Ellipsoid {

    fn elongation(&self) -> f32;
    
    fn area(&self) -> f32;

    fn orientation(&self) -> f32;
    
    fn orientation_along_largest(&self) -> f32;
    
    fn coords(&self, size : (usize, usize)) -> Option<EllipseCoords>;
    
}

/* All ellipsis implement default by returning a unit-radius circle without a translation */

/** Represents a translated and axis-aligned ellipse, i.e. the simple geometric function r = x^2/a^2 + y^2/b^2
(as a function of cartesian coordinates) or (x, y) = (a*cos(theta), b*sin(theta)) (as a function of polar coordinates).
(Increasing a stretches the ellipse along the horizontal axis; increasing b stretches it along the vertical axis.
If a = b we have the circle of radius r=a=b as a special case. 
The effect is the same as applying a diagonal scale matrix to all points in a circle
circumference, where [a, b] is the diagonal of the matrix. **/
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignedEllipse {
    pub center : Vector2<f32>,
    pub major_scale : f32,
    pub minor_scale : f32
}

impl AlignedEllipse {

    fn area(&self) -> f32 {
        self.major_scale * self.minor_scale * PI
    }

    pub fn major_scale(&self) -> f32 {
        self.major_scale
    }
    
    pub fn minor_scale(&self) -> f32 {
        self.minor_scale
    }
    
    pub fn elongation(&self) -> f32 {
        if self.major_scale >= self.minor_scale {
            self.major_scale / self.minor_scale
        } else {
            self.minor_scale / self.major_scale
        }
    }
    
    pub fn orientation(&self) -> f32 {
        0.0
    }
    
    pub fn orientation_along_largest(&self) -> f32 {
        if self.major_scale >= self.minor_scale {
            0.0
        } else {
            std::f32::consts::PI / 2.0
        }
    }
    
    pub fn coords(&self, size : (usize, usize)) -> Option<EllipseCoords> {
        let el : OrientedEllipse = self.clone().into();
        el.coords(size)
    }
    
}

impl Default for AlignedEllipse {

    fn default() -> Self {
        Self { center : Vector2::zeros(), major_scale : 1.0, minor_scale : 1.0 }
    }
    
}

/// Represents a translated ellipse in terms of the lengths of its major and minor axis,
/// its translation center, and an orientation as the signed angle of the major axis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrientedEllipse {

    pub aligned : AlignedEllipse,

    /** Orientation of the major axis around the center, in [-pi, pi]. If theta=0,
    this is equivalent to the AlignedEllipse. Positive angles are counter-clockwise rotations; 
    negative angles are clockwise rotations). The rotation is defined to always be atan2(major_y, major_x) **/
    pub theta : f32
    
}

// cargo test --lib -- el_limits --nocapture
#[test]
fn el_limits() {
    let ori = OrientedEllipse::new(Vector2::new(0., 0.), 2., 1., 0.);
    let bounds = ori.bounds();
    println!("{:?}", bounds);

    let ori = OrientedEllipse::new(Vector2::new(0., 0.), 2., 1., PI/2.);
    let bounds = ori.bounds();
    println!("{:?}", bounds);
}

impl OrientedEllipse {

    pub fn bounds(&self) -> Area {
        let xs = self.x_limits();
        let ys = self.y_limits();
        Area {
            origin : Vector2::new(xs.0, ys.0),
            target : Vector2::new(xs.1, ys.1)
        }
    }

    // https://math.stackexchange.com/questions/91132/how-to-get-the-limits-of-rotated-ellipse
    fn x_limits(&self) -> (f32, f32) {
        let v = (self.aligned.major_scale.powf(2.) * self.theta.cos().powf(2.) +
            self.aligned.minor_scale.powf(2.) * self.theta.sin().powf(2.)).sqrt();
        (self.aligned.center[0] - v, self.aligned.center[1] + v)
    }

    // https://math.stackexchange.com/questions/91132/how-to-get-the-limits-of-rotated-ellipse
    fn y_limits(&self) -> (f32, f32) {
        let v = (self.aligned.major_scale.powf(2.) * self.theta.sin().powf(2.) +
            self.aligned.minor_scale.powf(2.) * self.theta.cos().powf(2.)).sqrt();
        (self.aligned.center[1] - v, self.aligned.center[1] + v)
    }

    fn area(&self) -> f32 {
        self.aligned.area()
    }

    pub fn new(center : Vector2<f32>, major_scale : f32, minor_scale : f32, angle : f32) -> Self {
        Self {
            aligned : AlignedEllipse { center, major_scale, minor_scale },
            theta : angle
        }
    }

    pub fn major_scale(&self) -> f32 {
        self.aligned.major_scale
    }
    
    pub fn minor_scale(&self) -> f32 {
        self.aligned.minor_scale
    }
    
    pub fn elongation(&self) -> f32 {
        self.aligned.elongation()
    }
    
    pub fn orientation(&self) -> f32 {
        self.theta
    }
    
    pub fn orientation_along_largest(&self) -> f32 {
        let el : Ellipse = self.clone().into();
        el.orientation_along_largest()
    }
    
    pub fn coords(&self, size : (usize, usize)) -> Option<EllipseCoords> {
        let el : Ellipse = self.clone().into();
        el.coords(size)
    }
    
}

impl Default for OrientedEllipse {

    fn default() -> Self {
        Self { aligned : AlignedEllipse::default(), theta : 0.0 }
    }
    
}

/// Represents an oriented and translated ellipse in terms of a triplet of vectors: A pair
/// of (major, minor) vector and a translation of their common origin (center).
/// This converts to an Oriented ellipse by first subtracting center from the major and
/// minor vectors, then by calculating the angle. Notice the "major" and "minor" denomination
/// do not mean one is larger than the other, but rather that the "major" is the axis that
/// should be aligned to the x-axis in the corresponding OrientedEllipse repr; and the "minor"
/// is the axis that should be aligned to the y-axis in the corresponding OrientedEllipse repr.
#[repr(C)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Ellipse {
    pub center : Vector2<f32>,
    pub major : Vector2<f32>,
    pub minor : Vector2<f32>
}

impl Default for Ellipse {

    fn default() -> Self {
        Self { 
            center : Vector2::zeros(),
            major : Vector2::new(1.0, 0.0),
            minor : Vector2::new(0.0, 1.0)
        }
    }
    
}

impl From<crate::shape::Circle> for Ellipse {

    fn from(c : crate::shape::Circle) -> Self {
        Ellipse { center : c.center, major : Vector2::new(c.radius, 0.0), minor : Vector2::new(0.0, c.radius) }
    }

}

impl From<AlignedEllipse> for OrientedEllipse {

    fn from(aligned : AlignedEllipse) -> Self {
        OrientedEllipse {
            aligned,
            theta : 0.0
        }
    }

}

impl From<AlignedEllipse> for Ellipse {

    fn from(aligned : AlignedEllipse) -> Self {
        Ellipse {
            center : Vector2::zeros(),
            major : aligned.major_scale * Vector2::new(1.0, 0.0),
            minor : aligned.minor_scale * Vector2::new(0.0, 1.0)
        }
    }

}

impl From<Ellipse> for OrientedEllipse {

    fn from(el : Ellipse) -> Self {
        OrientedEllipse {
            aligned : AlignedEllipse {
                major_scale : el.major_scale(),
                minor_scale : el.minor_scale(),
                center : el.center
            },
            theta : el.orientation(),
        }
    }

}

impl From<OrientedEllipse> for Ellipse {

    fn from(el : OrientedEllipse) -> Self {
        Ellipse::new(el.aligned.center, el.aligned.major_scale, el.aligned.minor_scale, el.theta)
    }

}

fn full_angle(a : &Vector2<f32>, b : &Vector2<f32>) -> f32 {
    // let unit_x = Vector2::new(1., 0.);

    // This gives the smallest angle between two vectors using the cosine rule:
    // acos((a.b)/(|a||b|)) in [0,pi] for argument in [-1,1];
    // to transform to [0-2pi]
    // let smallest_angle = centered.angle(&unit_x);

    // asin returns values in the range [-pi/2, pi/2] for values in the range = [-1,1]
    // if smallest_angle >

    let mut angle = a[1].atan2(a[0]) - b[1].atan2(b[0]);
    if angle < 0. {
        angle += 2. * PI;
    }
    angle
}

#[derive(Clone, Debug)]
pub struct EllipseEstimator {
    pts : Vec<Vector2<f32>>,
    A : DMatrix<f32>,
    b : DVector<f32>,
    A3 : DMatrix<f32>,
    b3 : DVector<f32>
}

fn vector_orientation(v : &Vector2<f32>) -> f32 {
    let normed = v.normalize();
    normed[1].atan2(normed[0])
}

impl EllipseEstimator {

    pub fn new() -> Self {
        Self {
            pts : Vec::new(),
            A : DMatrix::zeros(24, 5),
            b : DVector::zeros(24),
            A3 : DMatrix::zeros(24, 3),
            b3 : DVector::from_element(24, 1.0)
        }
    }
    
    pub fn estimate_from_coords(&mut self, coords : &[(usize, usize)], shape : (usize, usize)) -> Result<Ellipse, &'static str> {
        let mut pts = mem::take(&mut self.pts);
        pts.clear();
        for c in coords {
            let pt = super::coord::coord_to_vec(*c, shape).ok_or("Unable to convert coord")?;
            pts.push(pt);
        }
        let ans = self.estimate(&mut pts[..]);
        self.pts = pts;
        ans
    }
    
    pub fn estimate(&mut self, pts : &mut [Vector2<f32>]) -> Result<Ellipse, &'static str> {
        fit_ellipse_no_direct(pts, &mut self.A, &mut self.b, &mut self.A3, &mut self.b3)
    }
    
}

impl Ellipse {

    pub fn bounds(&self) -> Area {
        let ori : OrientedEllipse = self.clone().into();
        ori.bounds()
    }

    pub fn new_centered_symmetric(magn : f32) -> Self {
        Self {
            center : Vector2::zeros(),
            major : Vector2::new(1.0, 0.0).scale(magn),
            minor : Vector2::new(0.0, 1.0).scale(magn)
        }
    }

    pub fn new_symmetric(center : Vector2<f32>, magn : f32) -> Self {
        Self {
            center,
            major : Vector2::new(1.0, 0.0).scale(magn),
            minor : Vector2::new(0.0, 1.0).scale(magn)
        }
    }

    pub fn new_centered(major : Vector2<f32>, minor : Vector2<f32>) -> Self {
        Self { center : Vector2::zeros(), major, minor }
    }

    pub fn contains(&self, pt : &Vector2<f32>) -> bool {
        radial_deviation(self, pt) > 0.0
    }

    pub fn area(&self) -> f32 {
        self.major.magnitude() * self.minor.magnitude() * PI
    }

    /*// Returns a rect represented as an vector with origin at the ellipse
    // center and pointing to the top-right coordinate. Reflecting this vector
    // should give the negative position.
    pub fn enclosing_area(&self) -> crate::shape::Area {
        let EllipseAxes { major, minor } = self.axes();
        let (min_x, max_x) = if major[0] < minor[0] { (major[0], minor[0]) } else { (minor[0], major[0]) };
        let (min_y, max_y) = if major[1] < minor[1] { (major[1], minor[1]) } else { (minor[1], major[1]) };
        let origin = Vector2::new(min_x, min_y);
        let target = Vector2::new(max_x, max_y);
        crate::shape::Area { origin, target }
    }*/

    pub fn scaled_by(&self, s : f32) -> Ellipse {
        Ellipse {
            center : self.center.clone(),
            major : s * self.major.clone(),
            minor : s * self.minor.clone()
        }
    }

    pub fn rotate(&self, angle : f32) -> Self {
        let r = Rotation2::new(angle);
        let major = r.clone() * &self.major;
        let minor = r * &self.minor;
        Self {
            major,
            minor,
            center : self.center
        }
    }

    // Ratio of the largest to smallest axis (favovring either major or minor axis).
    pub fn elongation(&self) -> f32 {
        let major = self.major_scale();
        let minor = self.minor_scale();
        major.max(minor) / (major.min(minor) + std::f32::EPSILON)
    }
    
    /*pub fn estimate_iter_from_coords(coords : &[(usize, usize)], shape : (usize, usize), niter : usize, err_thr : f32) -> Result<Self, &'static str> {
        let mut coords = coords.to_vec();
        let mut el = Ellipse::default();
        for i in 0..niter {
            el = Self::estimate_from_coords(&coords, shape)?;
            let n_before = coords.len();
            for i in (0..n_before).rev() {
                if let Some(e) = circumference_coord_error(&el, coords[i], shape) {
                    println!("{}", e);
                    if e > err_thr {
                        coords.remove(i);
                    }
                }
            }
            if coords.len() == n_before {
                return Ok(el);
            }
        }
        Ok(el)
    }*/
    
    pub fn estimate_from_coords(coords : &[(usize, usize)], shape : (usize, usize)) -> Result<Self, &'static str> {
        EllipseEstimator::new().estimate_from_coords(coords, shape)
    }

    // This was based on the Opencv "direct" ellipse fitting implementation
    // available at modules/imgproc/src/shapedescr.cpp
    pub fn estimate(pts : &mut [Vector2<f32>]) -> Result<Self, &'static str> {
        // fit_ellipse_direct(pts)
        EllipseEstimator::new().estimate(pts)
    }

    /*// Returns the angle of the major axis on the right quadrant
    // (-pi/2, pi/2) (atan-like coordinates).
    /// Returns ellipse orientation using atan-like convention (-pi/2, pi/2)
    /// over the bottom and right quadrant).
    pub fn atan_orientation(&self) -> f32 {
        let angle = self.angle();
        if angle > 0.0 && angle <= PI / 2. {
            // In this case, we are ok (top-right quadrant is always the same).
            angle
        } else if angle > PI / 2. && angle < PI {
            // In this case, we are at the top-left quadrant. Reflect it
            // horizontally and vertically towards the negative-right quadrant
            angle + PI
        } else if angle > -PI && angle < 0.0 {
            // In this case, we are at the bottom-left quadrant. Reflect it
            // horizontally
            angle + PI/2
        } else {
            // We are now at the bottom-right quadrant. Reflect it
            // vertically.
            angle + PI / 2
        }
    }*/

    // Returns angle in the range [0,pi] for normalized scalar producut in [0,1]
    // fn cosine_orientation()

    // Retruns angle in the range [-pi, pi] for normalized scalar product in [0,1]
    // fn sin_orientation()

    /// Returns the orientation of the major axis of the ellipse around its center in [-pi, pi].
    /// The major axis is the one corresponding to the x-axis in the orientedellipse representation.
    pub fn orientation(&self) -> f32 {
        vector_orientation(&self.major)
    }
    
    /// Returns the orientation of the axis with the largest length in [-pi, pi].
    pub fn orientation_along_largest(&self) -> f32 {
        if self.major_scale() >= self.minor_scale() {
            vector_orientation(&self.major)
        } else {
            vector_orientation(&self.minor)
        }
    }

    // Lenght of the major axis. This equals the 'a' parameter for an axis-aligned ellipse.
    pub fn major_scale(&self) -> f32 {
        self.major.norm()
    }

    // Lenght of the minor axis. This equals the 'b' parameter for an axis-aligned ellipse.
    pub fn minor_scale(&self) -> f32 {
        self.minor.norm()
    }

    // Builds an ellipse from scales applied over the major and minor axes and
    // signed angle of the major axis (i.e. angle as returned by atan2).
    pub fn new(center : Vector2<f32>, a : f32, b : f32, mut theta : f32) -> Ellipse {
        // Transform [0,2*pi] to atan2-like coordinates (-pi, pi); clockwise rotations are negative.
        if theta > PI {
            theta = 2.*PI - theta;
        }
        let rot_major = Rotation2::new(theta);
        let major = a * (rot_major * Vector2::new(1., 0.));
        
        let rot_minor = if theta >= 0.0 {
        
            // If angle is in the positive half of the circle, set minor at + 90º quadrature
            Rotation2::new(theta + PI / 2.)
            
        } else {
        
            // If angle is in the negative half of the circle, set minor at -90º quadrature
            Rotation2::new(theta - PI / 2.)
            
        };
        let minor = b * (rot_minor * Vector2::new(1., 0.));
        Ellipse { center, major, minor }
    }

    pub fn axes(&self) -> EllipseAxes {
        let major = Vector2::new(self.center[0] + self.major[0], self.center[1] + self.major[1]);
        let minor = Vector2::new(self.center[0] + self.minor[0], self.center[1] + self.minor[1]);
        EllipseAxes { major, minor }
    }

    pub fn coords(&self, size: (usize, usize)) -> Option<EllipseCoords> {
    
        // let major_t = Vector2::new(self.center[0] + self.major[0], self.center[1] + self.major[1]);
        // let minor_t = Vector2::new(self.center[0] + self.minor[0], self.center[1] + self.minor[1]);
        let EllipseAxes { major, minor } = self.axes();
        
        if major[0] < 0.0 || major[0] > size.1 as f32 || major[1] < 0.0 || major[1] > size.0 as f32 {
            // println!("major = {:?}", major_t);
            return None;
        }
        
        if minor[0] < 0.0 || minor[0] > size.1 as f32 || minor[1] < 0.0 || minor[1] > size.0 as f32 {
            // println!("minor = {:?}", minor_t);
            return None;
        }
        
        let c = ((size.0 as f32 - self.center[1]).round() as usize, self.center[0].round() as usize);
        let dst_major = (
            (size.0 as f32 - (c.0 as f32 + self.major[1])).round() as usize,
            (c.1 as f32 + self.major[0]).round() as usize
        );
        let dst_minor = (
            (size.0 as f32 - (c.0 as f32 + self.minor[1])).round() as usize,
            (c.1 as f32 + self.minor[0]).round() as usize
        );
        Some(EllipseCoords { center : c, major : dst_major, minor : dst_minor })
    }

}

// Represents a triplet of pixel buffer coordinates (center, arrow point of major axis
// and arrow point of minor axis) that are close to an ellipse representation in the cartesian plane.
// To build this object, an image dimension must be informed. The EllipseCoord is only retrieved
// when the cartesian plane superimposed in the pixel buffer has valid 2D index bounds. y-coordinates
// are inverted to 2D coordinate relative to top-left representation.
/* Uses the ellipse (at the cartesian vector space) to index the image (in pixel space).
This returns a triplet of coordinates: One at the ellipse center, and other two at the
arrow points of the major and minor axes.  */
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EllipseCoords {
    pub center : (usize, usize),
    pub major : (usize, usize),
    pub minor : (usize, usize)
}

impl EllipseCoords {

    pub fn bounds(&self) -> Region {
        let left = self.major.1.min(self.minor.1);
        let right = self.major.1.max(self.minor.1);
        let top = self.major.0.min(self.minor.0);
        let bottom = self.major.0.max(self.minor.0);
        Region::new((top, left), (bottom.saturating_sub(top), right.saturating_sub(left)))
    }

}

/// Represents the elliptical axes after a translation is applied
/// through the center coordinate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EllipseAxes {
    pub major : Vector2<f32>,
    pub minor : Vector2<f32>
}

fn get_ofs(i : i32, eps : f32) -> Vector2<f32> {
    Vector2::new( (((i & 1)*2 - 1) as f32)*eps, (((i & 2) - 1) as f32)*eps)
}

fn sum_abs_diff(pts : &[Vector2<f32>], c : &Vector2<f32>) -> f32 {
    let mut s : f32 = 0.;
	for pt in pts.iter() {
		s += (pt[0] - c[0]).abs() + (pt[1] - c[1]).abs();
	}
	s
}

/*// Credit goes to the OpenCV implementation (imgproc::shapedescr) licensed under BSD.
fn fit_ellipse_direct(points : &[Vector2<f32>]) -> Result<Ellipse, &'static str> {

    let n = points.len();
    if n < 5 {
    	return Err("Too few points");
    }

    // Center vector
    let mut c = Vector2::new(0., 0.);
	for pt in points.iter() {
		c[0] += pt[0];
		c[1] += pt[1];
	}
	c[0] /= n as f32;
	c[1] /= n as f32;

    let s = sum_abs_diff(points, &c);
	let scale = 100. / s.max(std::f32::EPSILON);

	// A holds a tall matrix with 6 of the moments of the points (up to order 2).
	let mut A = DMatrix::zeros(n, 6);

    let mut eps : f32 = 0.;
	let mut iter = 0;

	let mut M = Matrix3::zeros();
	let mut TM = Matrix3::zeros();
	let mut Ts = 0.;

    while iter < 2 {
    	for i in 0..n {

            // Point2f p = is_float ? ptsf[i] : Point2f((float)ptsi[i].x, (float)ptsi[i].y);
    		let p = &points[i];

    		// Point2f delta = getOfs(i, eps);
    		let delta = get_ofs(i as i32, eps);

    		// double px = (p.x + delta.x - c.x)*scale, py = (p.y + delta.y - c.y)*scale;
    		let px = (p[0] + delta[0] - c[0])*scale;
    		let py = (p[1] + delta[1] - c[1])*scale;

    		A[(i,0)] = px*px;
            A[(i,1)] = px*py;
            A[(i,2)] = py*py;
            A[(i,3)] = px;
            A[(i,4)] = py;
            A[(i,5)] = 1.0;
    	}

    	/*
    	cv::mulTransposed( A, DM, true, noArray(), 1.0, -1 );
        DM *= (1.0/n);
        */

        // DM holds a 6x6 covariance of A.
    	let mut DM = A.clone().transpose() * &A;
    	DM.scale_mut(1. / n as f32);

    	TM[(0,0)] = DM[(0,5)]*DM[(3,5)]*DM[(4,4)] - DM[(0,5)]*DM[(3,4)]*DM[(4,5)] - DM[(0,4)]*DM[(3,5)]*DM[(5,4)] +
                  DM[(0,3)]*DM[(4,5)]*DM[(5,4)] + DM[(0,4)]*DM[(3,4)]*DM[(5,5)] - DM[(0,3)]*DM[(4,4)]*DM[(5,5)];
        TM[(0,1)] = DM[(1,5)]*DM[(3,5)]*DM[(4,4)] - DM[(1,5)]*DM[(3,4)]*DM[(4,5)] - DM[(1,4)]*DM[(3,5)]*DM[(5,4)] +
                  DM[(1,3)]*DM[(4,5)]*DM[(5,4)] + DM[(1,4)]*DM[(3,4)]*DM[(5,5)] - DM[(1,3)]*DM[(4,4)]*DM[(5,5)];
        TM[(0,2)] = DM[(2,5)]*DM[(3,5)]*DM[(4,4)] - DM[(2,5)]*DM[(3,4)]*DM[(4,5)] - DM[(2,4)]*DM[(3,5)]*DM[(5,4)] +
                  DM[(2,3)]*DM[(4,5)]*DM[(5,4)] + DM[(2,4)]*DM[(3,4)]*DM[(5,5)] - DM[(2,3)]*DM[(4,4)]*DM[(5,5)];
        TM[(1,0)] = DM[(0,5)]*DM[(3,3)]*DM[(4,5)] - DM[(0,5)]*DM[(3,5)]*DM[(4,3)] + DM[(0,4)]*DM[(3,5)]*DM[(5,3)] -
                  DM[(0,3)]*DM[(4,5)]*DM[(5,3)] - DM[(0,4)]*DM[(3,3)]*DM[(5,5)] + DM[(0,3)]*DM[(4,3)]*DM[(5,5)];
        TM[(1,1)] = DM[(1,5)]*DM[(3,3)]*DM[(4,5)] - DM[(1,5)]*DM[(3,5)]*DM[(4,3)] + DM[(1,4)]*DM[(3,5)]*DM[(5,3)] -
                  DM[(1,3)]*DM[(4,5)]*DM[(5,3)] - DM[(1,4)]*DM[(3,3)]*DM[(5,5)] + DM[(1,3)]*DM[(4,3)]*DM[(5,5)];
        TM[(1,2)] = DM[(2,5)]*DM[(3,3)]*DM[(4,5)] - DM[(2,5)]*DM[(3,5)]*DM[(4,3)] + DM[(2,4)]*DM[(3,5)]*DM[(5,3)] -
                  DM[(2,3)]*DM[(4,5)]*DM[(5,3)] - DM[(2,4)]*DM[(3,3)]*DM[(5,5)] + DM[(2,3)]*DM[(4,3)]*DM[(5,5)];
        TM[(2,0)] = DM[(0,5)]*DM[(3,4)]*DM[(4,3)] - DM[(0,5)]*DM[(3,3)]*DM[(4,4)] - DM[(0,4)]*DM[(3,4)]*DM[(5,3)] +
                  DM[(0,3)]*DM[(4,4)]*DM[(5,3)] + DM[(0,4)]*DM[(3,3)]*DM[(5,4)] - DM[(0,3)]*DM[(4,3)]*DM[(5,4)];
        TM[(2,1)] = DM[(1,5)]*DM[(3,4)]*DM[(4,3)] - DM[(1,5)]*DM[(3,3)]*DM[(4,4)] - DM[(1,4)]*DM[(3,4)]*DM[(5,3)] +
                  DM[(1,3)]*DM[(4,4)]*DM[(5,3)] + DM[(1,4)]*DM[(3,3)]*DM[(5,4)] - DM[(1,3)]*DM[(4,3)]*DM[(5,4)];
        TM[(2,2)] = DM[(2,5)]*DM[(3,4)]*DM[(4,3)] - DM[(2,5)]*DM[(3,3)]*DM[(4,4)] - DM[(2,4)]*DM[(3,4)]*DM[(5,3)] +
                  DM[(2,3)]*DM[(4,4)]*DM[(5,3)] + DM[(2,4)]*DM[(3,3)]*DM[(5,4)] - DM[(2,3)]*DM[(4,3)]*DM[(5,4)];

        Ts = (-(DM[(3,5)]*DM[(4,4)]*DM[(5,3)]) + DM[(3,4)]*DM[(4,5)]*DM[(5,3)] + DM[(3,5)]*DM[(4,3)]*DM[(5,4)] -
              DM[(3,3)]*DM[(4,5)]*DM[(5,4)]  - DM[(3,4)]*DM[(4,3)]*DM[(5,5)] + DM[(3,3)]*DM[(4,4)]*DM[(5,5)]);

        M[(0,0)] = (DM[(2,0)] + (DM[(2,3)]*TM[(0,0)] + DM[(2,4)]*TM[(1,0)] + DM[(2,5)]*TM[(2,0)])/Ts)/2.;
        M[(0,1)] = (DM[(2,1)] + (DM[(2,3)]*TM[(0,1)] + DM[(2,4)]*TM[(1,1)] + DM[(2,5)]*TM[(2,1)])/Ts)/2.;
        M[(0,2)] = (DM[(2,2)] + (DM[(2,3)]*TM[(0,2)] + DM[(2,4)]*TM[(1,2)] + DM[(2,5)]*TM[(2,2)])/Ts)/2.;
        M[(1,0)] = -DM[(1,0)] - (DM[(1,3)]*TM[(0,0)] + DM[(1,4)]*TM[(1,0)] + DM[(1,5)]*TM[(2,0)])/Ts;
        M[(1,1)] = -DM[(1,1)] - (DM[(1,3)]*TM[(0,1)] + DM[(1,4)]*TM[(1,1)] + DM[(1,5)]*TM[(2,1)])/Ts;
        M[(1,2)] = -DM[(1,2)] - (DM[(1,3)]*TM[(0,2)] + DM[(1,4)]*TM[(1,2)] + DM[(1,5)]*TM[(2,2)])/Ts;
        M[(2,0)] = (DM[(0,0)] + (DM[(0,3)]*TM[(0,0)] + DM[(0,4)]*TM[(1,0)] + DM[(0,5)]*TM[(2,0)])/Ts)/2.;
        M[(2,1)] = (DM[(0,1)] + (DM[(0,3)]*TM[(0,1)] + DM[(0,4)]*TM[(1,1)] + DM[(0,5)]*TM[(2,1)])/Ts)/2.;
        M[(2,2)] = (DM[(0,2)] + (DM[(0,3)]*TM[(0,2)] + DM[(0,4)]*TM[(1,2)] + DM[(0,5)]*TM[(2,2)])/Ts)/2.;

        /*double det = fabs(cv::determinant(M));
        if (fabs(det) > 1.0e-10)
            break;
        eps = (float)(s/(n*2)*1e-2);*/

        // TODO this determinant can be extracted from Eigen::new(.) call below, saving up its
        // next computation when iter < 2.
        let det = M.determinant().abs();
        if det > 1.0e-10 {
        	break;
        }
        eps = s / (n as f32 * 2.)*1.0e-2;
        iter += 1;
    }

    if iter >= 2 {
        return fit_ellipse_no_direct(points);
    }

    // This stores eigenvectors as rows into eVec matrix.
    // eigenNonSymmetric(M, eVal, eVec);
    // TODO verify if the ordering/transpose state of this decomposition is the same as opencv eigenNonSymmetric
    let eigen = nalgebra_lapack::Eigen::new(M, true, true).unwrap();
    let evecs = eigen.eigenvectors.as_ref().unwrap();

    // Select the eigen vector {a,b,c} which satisfies 4ac-b^2 > 0
    let mut cond = [0., 0., 0.];
    cond[0]=(4.0 * evecs[(0,0)] * evecs[(0,2)] - evecs[(0,1)] * evecs[(0,1)]);
    cond[1]=(4.0 * evecs[(1,0)] * evecs[(1,2)] - evecs[(1,1)] * evecs[(1,1)]);
    cond[2]=(4.0 * evecs[(2,0)] * evecs[(2,2)] - evecs[(2,1)] * evecs[(2,1)]);
    let i = if cond[0] < cond[1] {
        if cond[1] < cond[2] {
        	2
        } else {
        	1
        }
    } else {
        if cond[0] < cond[2] {
        	2
        } else {
        	0
        }
    };

    let mut norm = (evecs[(i,0)]*evecs[(i,0)] + evecs[(i,1)]*evecs[(i,1)] + evecs[(i,2)]*evecs[(i,2)]).sqrt();
    let evec_zero_neg = if evecs[(i,0)] < 0.0 { -1 } else { 1 };
    let evec_one_neg = if evecs[(i,1)] < 0.0 { -1 } else { 1 };
    let evec_two_neg = if evecs[(i,2)] < 0.0 { -1 } else { 1 };
    if evec_zero_neg * evec_one_neg * evec_two_neg <= 0 {
    	norm = -1.0*norm;
    }

    let mut pVec = Vector3::zeros();
    pVec[0] = evecs[(i,0)]/norm;
    pVec[1] = evecs[(i,1)]/norm;
    pVec[2] = evecs[(i,2)]/norm;

    // //  Q = (TM . pVec)/Ts; (but only fst col is used).
    let mut Q = Matrix3::zeros();
    Q[(0,0)] = (TM[(0,0)]*pVec[0] +TM[(0,1)]*pVec[1] +TM[(0,2)]*pVec[2] )/Ts;
    Q[(0,1)] = (TM[(1,0)]*pVec[0] +TM[(1,1)]*pVec[1] +TM[(1,2)]*pVec[2] )/Ts;
    Q[(0,2)] = (TM[(2,0)]*pVec[0] +TM[(2,1)]*pVec[1] +TM[(2,2)]*pVec[2] )/Ts;

	// We compute the ellipse properties in the shifted coordinates as doing so improves the numerical accuracy.
    let u1 = pVec[2]*Q[(0,0)]*Q[(0,0)] - pVec[1]*Q[(0,0)]*Q[(0,1)] + pVec[0]*Q[(0,1)]*Q[(0,1)] + pVec[1]*pVec[1]*Q[(0,2)];
    let u2 = pVec[0]*pVec[2]*Q[(0,2)];
    let l1 = (pVec[1]*pVec[1] + (pVec[0] - pVec[2])*(pVec[0] - pVec[2])).sqrt();
    let l2 = pVec[0] + pVec[2];
    let l3 = (pVec[1]*pVec[1] - 4.*pVec[0]*pVec[2]);
    let p1 = 2.*pVec[2]*Q[(0,0)] - pVec[1]*Q[(0,1)];
    let p2 = 2.*pVec[0]*Q[(0,1)] - pVec[1]*Q[(0,0)];

    if l3 == 0.0 {
        return fit_ellipse_no_direct(points);
    }

    let x0 = (p1 / l3 / scale) + c[0];
    let y0 = (p2 / l3 / scale) + c[1];
    let length_major = SQRT_2*((u1 - 4.0*u2)/((l1 - l2)*l3)).sqrt() / scale;
    let length_minor = SQRT_2*(-1.0*((u1 - 4.0*u2)/((l1 + l2)*l3))).sqrt() / scale;

    // Theta is the rotation of the major axis.
    let angle = if (pVec[1] == 0.) {
        if (pVec[0]  < pVec[2] ) {
            0.
        } else {
            PI / 2.
        }
    } else {
    	/*PI / 2. +*/ 0.5*pVec[1].atan2(pVec[0]  - pVec[2])
    };

    let center = Vector2::new(x0, y0);
    Ok(Ellipse::new(center, length_major, length_minor, angle))
}*/

fn populate_points(
    A : &mut DMatrixSliceMut<f32>, 
    b : &mut DVectorSliceMut<f32>, 
    points : &[Vector2<f32>], 
    c : &Vector2<f32>, 
    scale : f32
) {
	let n = points.len();
    for i in 0..n {
        let mut p = points[i].clone();
        p -= c;
        let px = p[0]*scale;
        let py = p[1]*scale;
        b[i] = 10000.0; // 1.0?
        A[(i, 0)] = -px * px; // A - C signs inverted as proposed by APP
        A[(i, 1)] = -py * py;
        A[(i, 2)] = -px * py;
        A[(i, 3)] = px;
        A[(i, 4)] = py;
    }
}

// use crate::prelude::Image;
// use crate::draw::*;

// Was 1e-8
const SVD_TOL : f32 = 1.0e-4;

// This was taken from the plain SVD::solve from nalgebra, since the lapack
// version does not have the solve method.
fn solve_lapack_svd<S>(
    svd : &nalgebra_lapack::SVD<f32, Dynamic, Dynamic>, 
    b : &Matrix<f32, Dynamic, U1, S>,
    eps : f32
) -> DVector<f32> 
where
    S : Storage<f32, Dynamic, U1>
{
    let mut ut_b = svd.u.ad_mul(b);
    for j in 0..ut_b.ncols() {
        let mut col = ut_b.column_mut(j);
        for i in 0..svd.singular_values.len() {
            let val = svd.singular_values[i].clone();
            if val > eps {
                col[i] = col[i].clone().unscale(val);
            } else {
                col[i] = 0.0;
            }
        }
    }
    svd.vt.ad_mul(&ut_b)
}

// Credit goes to the OpenCV implementation (imgproc::shapedescr) licensed under BSD.
fn fit_ellipse_no_direct(
    points : &mut [Vector2<f32>], 
    A : &mut DMatrix<f32>, 
    b : &mut DVector<f32>, 
    A3 : &mut DMatrix<f32>, 
    b3 : &mut DVector<f32>
) -> Result<Ellipse, &'static str> {
	let n = points.len();

	if n < 5 {
		return Err("Too few points");
	}
    
    // Center all points
    let mut c = Vector2::new(0., 0.);
    for pt in points.iter() {
    	c += pt;
    }
    c.scale_mut(1. / n as f32);

    // This is the same step as the direct method. Write a dedicated method for it.
    let s = sum_abs_diff(&points, &c);
	let scale = 100. / s.max(std::f32::EPSILON);

    assert!(A.nrows() == b.nrows());
    if n > A.nrows() {
        A.resize_vertically_mut(n, 0.0);
        b.resize_vertically_mut(n, 0.0);
    }
    let mut A = A.rows_mut(0, n);
    let mut b = b.rows_mut(0, n);
    
    // let mut A = DMatrix::zeros(n, 5);
    // let mut b = DVector::zeros(n);
    populate_points(&mut A, &mut b, &points, &c, scale);
    
    let mut A_svd = linalg::SVD::new(A.clone_owned(), true, true);
    // let mut A_svd = nalgebra_lapack::SVD::new(A.clone_owned()).ok_or("SVD failed")?;

    if(A_svd.singular_values[0]*std::f32::EPSILON > A_svd.singular_values[4]) {
        let eps = ( s / (n as f32 * 2.) * 1.0e-3);
        // let eps = ( s / (n as f32 * 2.) * 1.0e-4);
        for i in 0..n {
            let p = points[i] + get_ofs(i as i32, eps);
            points[i] = p;
        }
        populate_points(&mut A, &mut b, &points, &c, scale);
        
    	A_svd = linalg::SVD::new(A.clone_owned(), true, true);
    	// A_svd = nalgebra_lapack::SVD::new(A.clone_owned()).ok_or("SVD failed")?;
    }

    let gfp = A_svd.solve(&b, SVD_TOL).or(Err("SVD for A Failed"))?;
    // let gfp = solve_lapack_svd(&A_svd, &b, SVD_TOL);

    // now use general-form parameters A - E to find the ellipse center:
    // differentiate general form wrt x/y to get two equations for cx and cy
    let mut A2 = Matrix2::zeros();
    let mut b2 = Vector2::zeros();
    A2[(0, 0)] = 2. * gfp[0];
    A2[(0, 1)] = gfp[2];
    A2[(1, 0)] = gfp[2];
    A2[(1, 1)] = 2. * gfp[1];
    b2[0] = gfp[3];
    b2[1] = gfp[4];
    
    let A2_svd = linalg::SVD::new_unordered(A2, true, true);
    let rp = A2_svd.solve(&b2, SVD_TOL).or(Err("SVD for A2 Failed"))?;

    // re-fit for parameters A - C with those center coordinates
    if n > A3.nrows() {
        A3.resize_vertically_mut(n, 0.0);
        b3.resize_vertically_mut(n, 1.0);
    }
    let mut A3 = A3.rows_mut(0, n);
    let mut b3 = b3.rows_mut(0, n);

    for i in 0..n {
        let mut p = points[i].clone();
        p -= c;
        let px = p[0]*scale;
        let py = p[1]*scale;
        // b3[i] = 1.0;
        A3[(i, 0)] = (px - rp[0]) * (px - rp[0]);
        A3[(i, 1)] = (py - rp[1]) * (py - rp[1]);
        A3[(i, 2)] = (px - rp[0]) * (py - rp[1]);
    }

    // The old gfp buffer (nx1) was re-used here, but now only the first 3 entries.
    let x3 = linalg::SVD::new_unordered(A3.clone_owned(), true, true).solve(&b3, SVD_TOL).or(Err("SVD for A3 Failed"))?;
    // let x3 = solve_lapack_svd(&nalgebra_lapack::SVD::new(A3.clone_owned()).ok_or("SVD for A3 Failed")?, &b3, SVD_TOL);

    // Original impl stored angle in rp[4] because it was unused.
    let angle = -0.5 * x3[2].atan2(x3[1] - x3[0]); // convert from APP angle usage

    let min_eps : f32 = 1.0e-8;
    let t = if( x3[2].abs() > min_eps ) {
    	x3[2] / (-2.0 * angle).sin()
    } else {
    	// ellipse is rotated by an integer multiple of pi/2
        x3[1] - x3[0]
    };

    // Original impl stored length_major in rp[2] because it was unused.
    let mut major_unscaled = (x3[0] + x3[1] - t).abs();
    if major_unscaled > min_eps {
    	major_unscaled = (2.0 / major_unscaled).sqrt();
    }

    // Original impl stored length_minor in rp[3] because it was unused.
    let mut minor_unscaled = (x3[0] + x3[1] + t).abs();
    if minor_unscaled > min_eps {
    	minor_unscaled = (2.0 / minor_unscaled).sqrt();
    }

    let x0 = (rp[0] / scale ) + c[0];
    let y0 = (rp[1] / scale ) + c[1];
    let length_major = major_unscaled / scale;
    let length_minor = minor_unscaled / scale;
    // let angle = 90.0 + angle*180.0/PI;

    let center = Vector2::new(x0, y0);
    Ok(Ellipse::new(center, length_major, length_minor, angle))
}

/*// This is calculated via the cosine rule: (a.b)/(|a||b|)
// let angle_pt = (pt.clone() - &el.center ).angle(&el.major);
// let angle_pt = full_angle(&(pt.clone() - &el.center).normalize(), &el.major.clone().normalize());
let mut pt_norm = (Rotation2::new(theta).transpose() * (pt.clone() - &el.center)).normalize();
pt_norm[0] = pt[0] / a;
pt_norm[1] = pt[1] / b;
let arc_pt = pt_norm[1].atan2(pt_norm[0]);

// How does the angle of the point relative to the ellipse center maps
// to the canonical angle (if the ellipse were not rotated).

/*let arc_pt = if theta < 0.0 {
    theta + 2.*PI
} else {
    angle_pt - theta
};*/
// let arc_pt =

// println!("angle = {}", angle_pt);

// Closest circumference point at the same arc as the desired point.
// let closest = ellipse_circumference_point(&OrientedEllipse::from(el.clone()), angle_pt);
let closest = ellipse_circumference_point(&OrientedEllipse::from(el.clone()), arc_pt);

println!("point = {}; candidate = {}; arc point = {}", pt, closest, arc_pt);

img.draw(Mark::Dot((img.height() - (closest[1] as usize), closest[0] as usize), 2, 255));

// |x1 - x2| + |y1 - y2|
(closest - pt).abs().sum()*/

pub fn circumference_coord_error(el : &Ellipse, coord : (usize, usize), img_shape : (usize, usize)) -> Option<f32> {
    Some(abs_radial_deviation(el, &crate::shape::coord::coord_to_vec(coord, img_shape)?))
}

// Transform point to the elliptical coordinate frame, then calculates
// its radial position with respect to the ellipse circumference: If point
// is within ellipse, return positive value, 0.0 for point at the circumference,
// and negative value for point outside circumference. TODO revert that (pt_rad - 1.0)
pub fn radial_deviation(el : &Ellipse, pt : &Vector2<f32>) -> f32 {
    let theta = el.orientation();
    let a = el.major_scale();
    let b = el.minor_scale();

    // Align point to ellipse axis
    let aligned_pt = (Rotation2::new(-theta) * (pt.clone() - &el.center));

    // Radius from ellipse center to the current point.
    let pt_rad = ((aligned_pt[0] / a).powf(2.) + (aligned_pt[1] / b).powf(2.)).sqrt();
    (1. - pt_rad)
}

/// For a fitted point pt, returns its absolute deviation from the ellipse circumference.
pub fn abs_radial_deviation(el : &Ellipse, pt : &Vector2<f32>) -> f32 {
    radial_deviation(el, pt).abs()
}

use std::ops::Range;


pub fn coord_total_error(el : &Ellipse, coords : &[(usize, usize)], shape : (usize, usize)) -> f32 {
    coords.iter().fold(0.0, |e, c| e + circumference_coord_error(el, *c, shape).unwrap() )
}

pub fn total_error(el : &Ellipse, pts : &[Vector2<f32>]) -> f32 {
    pts.iter().fold(0.0, |e, pt| e + abs_radial_deviation(el, pt) )
}

/// Theta is the ellipse orientation; arc is the actual position of the
/// point in the circumference (as if it were the corresponding point in
/// the circle).
pub fn ellipse_circumference_point(
    el : &OrientedEllipse,
    arc : f32
) -> Vector2<f32> {

    // Generate a vector in the corresponding centered axis-aligned ellipse
    let x = el.aligned.major_scale * arc.cos();
    let y = el.aligned.minor_scale * arc.sin();
    let aligned_pt = Vector2::new(x, y);

    el.aligned.center.clone() + Rotation2::new(el.theta) * aligned_pt
}

// Generate the circumference points of an ellipse. To actually draw it,
// just draw Mark::Line between the generated points translated to coords.
pub fn generate_ellipse_points(
    el : &OrientedEllipse,
    n : usize
) -> Vec<Vector2<f32>> {
    let mut pts = Vec::new();
    for i in 0..n {
        // let arc = (2.0f32*PI / n as f32) * (i as f32);
        let arc = (2.0f32*PI / n as f32) * (i as f32);
        let pt = ellipse_circumference_point(el, arc);
        pts.push(pt);
    }
    pts
}

pub fn generate_ellipse_opt_coords(
    el : &OrientedEllipse,
    n : usize,
    shape : (usize, usize)
) -> Vec<Option<(usize, usize)>> {
    let mut pts = crate::shape::ellipse::generate_ellipse_points(el, n);
    pts.iter().map(move |pt| crate::shape::coord::point_to_coord(pt, shape) ).collect()
}

pub fn generate_ellipse_coords(
    el : &OrientedEllipse,
    n : usize,
    shape : (usize, usize)
) -> Vec<(usize, usize)> {
    let mut pts = crate::shape::ellipse::generate_ellipse_points(el, n);
    pts.iter().filter_map(move |pt| crate::shape::coord::point_to_coord(pt, shape) ).collect()
}

/*// cargo test --all-features --lib -- fit_ellipse_test --nocapture
#[test]
fn fit_ellipse_test() {

    // use crate::prelude::*;

    // For any non-zero orientation, the algorithms can return two equally valid orientations.

    let center = Vector2::new(256., 256.);
    let a = 200.0;
    let b = 100.0;
    // let orientation = 0.;
    // let orientation = 0.1;
    // let orientation = (-1.)*(PI / 4.);
    let orientation = (PI / 4.);
    let true_el = OrientedEllipse { aligned : AlignedEllipse { center, major_scale : a, minor_scale : b }, theta : orientation };

    let pts = generate_ellipse_points(&true_el, 100);

    let mut img = Image::new_constant(512, 512, 0);

    let el = fit_ellipse_no_direct(&pts[..]).unwrap();
    println!("{:?}", el);
    println!("ori = {}", el.orientation());
    println!("major = {}", el.major_scale());
    println!("minor = {}", el.minor_scale());
    println!("avg error = {}", total_error(&el, &pts) / pts.len() as f32);

    for pt in pts.iter() {
        if let Some(coord) = crate::shape::coord::point_to_coord(pt, img.shape()) {
            img.draw(Mark::Dot(coord, 4, 127));
            img.draw(Mark::EllipseArrows(el.coords(img.height()).unwrap()));
        }
    }
    img.show();

}*/

const MAX_BOUNDARY_ELEMENTS : usize = 5;

/* In center form, a conic C = (p-c)^T M (p-c) - z = 0 for z = c^T M c - w, with det(M) > 0 is
an ellipse with area pi/sqrt(det(A)) with A diagonal (factorization with principal axes of M)
For support with size 3, the samllest ellipse has rational form. For support size 5, the smallest
ellipse is the unique conic  through these points. For support size 4, the ellipse is not represented
*/

fn ellipse_contains(
    pts : &[Vector2<f32>],
    support : &BTreeSet<usize>,
    pt : usize
) -> bool {
    unimplemented!()
}

pub fn welzl_step(
    pts : &[Vector2<f32>],
    inliers : &mut BTreeSet<usize>,
    support : &mut BTreeSet<usize>,
    outliers : &mut BTreeSet<usize>
) {
    use rand::prelude::IteratorRandom;
    if outliers.is_empty() || support.len() == 5 {
        return;
    }
    let new_ix = *outliers.iter().choose(&mut rand::thread_rng()).unwrap();
    outliers.remove(&new_ix);
    welzl_step(pts, inliers, support, outliers);
    if ellipse_contains(pts, support, new_ix) {
        return;
    } else {
        inliers.insert(new_ix);
        welzl_step(pts, inliers, support, outliers);
    }
}

/* Algo. 2.2 of Gartner & Schonherr (1997) */
pub fn welzl_enclosing_ellipse(pts : &[Vector2<f32>]) -> Option<Ellipse> {
    if pts.len() < 3 {
        return None;
    }
    let mut inliers = BTreeSet::new();
    let mut outliers = BTreeSet::new();

    for i in 0..pts.len() {
        outliers.insert(i);
    }

    let mut support = init_enclosing_ellipse(pts);

    welzl_step(pts, &mut inliers, &mut support, &mut outliers);
    unimplemented!()
}

fn init_enclosing_ellipse(pts : &[Vector2<f32>]) -> BTreeSet<usize> {
    let mut largest_dist = (0, 1, 0.0);
    for i in 0..(pts.len()-1) {
        for j in (i+1)..pts.len() {
            let d = pts[i].metric_distance(&pts[j]);
            if d > largest_dist.2 {
                largest_dist = (i, j, d);
            }
        }
    }
    let mut best_proj = (0, 0.0);
    let axis = pts[largest_dist.0] - pts[largest_dist.1];
    for i in 0..pts.len() {
        if i != largest_dist.0 && i != largest_dist.1 {
            let v = pts[i] - pts[largest_dist.1];
            let proj_axis = axis.scale(v.dot(&axis));
            let normal = v - proj_axis;
            let magn = normal.magnitude();
            if magn > best_proj.1 {
                best_proj = (i, magn)
            }
        }
    }
    let mut support = BTreeSet::new();
    support.insert(largest_dist.0);
    support.insert(largest_dist.1);
    support.insert(best_proj.0);
    support
}
