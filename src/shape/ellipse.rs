use nalgebra::*;
use std::f32::consts::{SQRT_2, PI};
use nalgebra_lapack;
use serde::{Serialize, Deserialize};

/*
Axis-aligned parametrization of ellipse
(as function of coord): r = x^2/a^2 + y^2/b^2
(as function of angle: (x, y) = (a*cos(theta), b*sin(theta))

Increasing a stretches it along the horizontal axis; increasing b stretches it along the vertical axis.
If a = b we have the circle of radius r=a=b as a special case.
*/
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ellipse {
    pub center : Vector2<f32>,
    pub major : Vector2<f32>,
    pub minor : Vector2<f32>
}

impl Ellipse {

    // This was based on the Opencv "direct" ellipse fitting implementation
    // available at modules/imgproc/src/shapedescr.cpp
    pub fn estimate(pts : &[Vector2<f32>]) -> Result<Self, &'static str> {
        fit_ellipse_direct(pts)
    }

    // Returns the angle of the major axis.
    pub fn orientation(&self) -> f32 {
        // actually pi - this_result

        /*If this is the angle resulting from no_direct:
        CV_SWAP( box.size.width, box.size.height, tmp );
        box.angle = (float)(90 + rp[4]*180/CV_PI);
        }
        if( box.angle < -180 )
            box.angle += 360;
        if( box.angle > 360 )
            box.angle -= 360;*/
        (self.major.clone() - &self.center).angle(&Vector2::new(1., 0.))
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
    // angle of the major axis.
    pub fn new(center : Vector2<f32>, a : f32, b : f32, theta : f32) -> Ellipse {
        let major = a * (Rotation2::new(theta) * Vector2::new(1., 0.));
        let minor = b * (Rotation2::new(theta + PI / 2.) * Vector2::new(1., 0.));
        Ellipse { center, major, minor }
    }

    pub fn coords(&self, img_height : usize) -> Option<EllipseCoords> {
        let c = ((img_height as f32 - self.center[1]) as usize, self.center[0] as usize);
        let dst_major = (
            (img_height as f32 - (c.0 as f32 + self.major[1])) as usize,
            (c.1 as f32 + self.major[0]) as usize
        );
        let dst_minor = (
            (img_height as f32 - (c.0 as f32 + self.minor[1])) as usize,
            (c.1 as f32 + self.minor[0]) as usize
        );
        Some(EllipseCoords { center : c, major : dst_major, minor : dst_minor })
    }

}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EllipseCoords {
    pub center : (usize, usize),
    pub major : (usize, usize),
    pub minor : (usize, usize)
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
	println!("scale = {}", scale);

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

    println!("TM = {}", TM);
    println!("M = {}", M);

    if iter >= 2 {
        return fit_ellipse_no_direct(points);
    }

    // This stores eigenvectors as rows into eVec matrix.
    // eigenNonSymmetric(M, eVal, eVec);
    // TODO verify if the ordering/transpose state of this decomposition is the same as opencv eigenNonSymmetric
    let eigen = nalgebra_lapack::Eigen::new(M, true, true).unwrap();
    let evecs = eigen.eigenvectors.as_ref().unwrap();
    println!("evecs = {}", evecs);

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
    	PI / 2. + 0.5*pVec[1].atan2(pVec[0]  - pVec[2])
    };

    let center = Vector2::new(x0, y0);
    Ok(Ellipse::new(center, length_major, length_minor, angle))
}

fn populate_points(A : &mut DMatrix<f32>, b : &mut DVector<f32>, points_copy : &[Vector2<f32>], c : &Vector2<f32>, scale : f32) {
	let n = points_copy.len();
    for i in 0..n {
        let mut p = points_copy[i].clone();
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

fn fit_ellipse_no_direct(points : &[Vector2<f32>]) -> Result<Ellipse, &'static str> {
	let n = points.len();

	if n < 5 {
		return Err("Too few points");
	}

    // TODO rename to centered_points
    let mut points_copy : Vec<_> = points.iter().cloned().collect();
    let mut c = Vector2::new(0., 0.);
    for pt in points.iter() {
    	c += pt;
    }
    c.scale_mut(1. / n as f32);

    // This is the same step as the direct method. Write a dedicated method for it.
    /*let mut s = 0.;
    for p in points_copy.iter() {
    	let mut pt = p.clone();
    	pt -= c;
    	s += pt[0].abs() + pt[1].abs();
    }*/
    let s = sum_abs_diff(points, &c);
	let scale = 100. / s.max(std::f32::EPSILON);
	println!("scale = {}", scale);

	// W holds the singular values; u the left singular vectors; vt the right singular vectors.
    // SVDecomp(A, w, u, vt);

    let mut A = DMatrix::zeros(n, 5);
    let mut b = DVector::zeros(n);
    populate_points(&mut A, &mut b, &points_copy, &c, scale);
    let mut svd = linalg::SVD::new(A.clone(), true, true);

    if(svd.singular_values[0]*std::f32::EPSILON > svd.singular_values[4]) {
        let eps = ( s / (n as f32 * 2.) * 1.0e-3);
        for i in 0..n {
            let p = points_copy[i] + get_ofs(i as i32, eps);
            points_copy[i] = p;
        }
        populate_points(&mut A, &mut b, &points_copy, &c, scale);
    	svd = linalg::SVD::new(A, true, true);
    }

    // SVBackSubst(w, u, vt, b, x);
    // SVBackSubst either solves the system if it has exactly one solution OR gives the least squares
    // best solution if it has many solutions, so it is not strictly equivalent to solve.
    // The x vector (backed by gfp array) is written with the result of SVBackSubst (5x1)
    let gfp = svd.solve(&b, 10.0e-8).unwrap();

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
    let rp = linalg::SVD::new(A2, true, true).solve(&b2, 1.0e-8).unwrap();

    // re-fit for parameters A - C with those center coordinates
    let mut A3 = DMatrix::<f32>::zeros(n, 3);
    let mut b3 = DVector::<f32>::zeros(n);

    for i in 0..n {
        let mut p = points_copy[i].clone();
        p -= c;
        let px = p[0]*scale;
        let py = p[1]*scale;
        b3[i] = 1.0;
        A3[(i, 0)] = (px - rp[0]) * (px - rp[0]);
        A3[(i, 1)] = (py - rp[1]) * (py - rp[1]);
        A3[(i, 2)] = (px - rp[0]) * (py - rp[1]);
    }

    // The old gfp buffer (nx1) was re-used here, but now only the first 3 entries.
    let x3 = linalg::SVD::new(A3, true, true).solve(&b3, 1.0e-8).unwrap();

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

// Generate the circumference points of an ellipse. To actually draw it,
// just draw Mark::Line between the generated points translated to coords.
pub fn generate_ellipse_points(
    center : &Vector2<f32>,
    length_major : f32,
    length_minor : f32,
    theta : f32,
    n : usize
) -> Vec<Vector2<f32>> {
    let mut pts = Vec::new();
    for i in 0..n {
        let theta = (2.0f32*PI / n as f32) * (i as f32);
        let x = length_major*theta.cos();
        let y = lenght_minor*theta.sin();
        let pt = center.clone() + Rotation2::new(theta)*Vector2::new(x, y);
        pts.push(pt);
    }
    pts
}

// cargo test --all-features --lib -- fit_ellipse_test --nocapture
#[test]
fn fit_ellipse_test() {

    use crate::prelude::*;

    let center = Vector2::new(256., 256.);
    let a = 200.0;
    let b = 100.0;
    let orientation = 0.;

    // Both methods work for those
    // let orientation = (-1.)*(PI / 4.);
    // let orientation = (PI / 4.);

    let pts = generate_ellipse_points(&center, a, b, orientation, 100);

    let mut img = Image::new_constant(512, 512, 0);

    let el = fit_ellipse_direct(&pts[..]).unwrap();
    println!("{:?}", el);
    println!("ori = {}", el.orientation());
    println!("major = {}", el.major_scale());
    println!("minor = {}", el.minor_scale());

    for pt in pts.iter() {
        if let Some(coord) = crate::shape::coord::point_to_coord(pt, img.shape()) {
            img.draw(Mark::Dot(coord, 2, 127));
            img.draw(Mark::EllipseArrows(el.coords(img.height()).unwrap()));
        }
    }
    img.show();

}
