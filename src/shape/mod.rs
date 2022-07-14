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


