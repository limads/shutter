mod ellipse;

pub use ellipse::*;

pub mod coord {

    use nalgebra::Vector2;
    use num_traits::{AsPrimitive, Zero};
    use std::cmp::PartialOrd;

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


