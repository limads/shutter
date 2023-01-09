use crate::image::*;
use smallvec::SmallVec;
use std::convert::{AsRef, AsMut};

#[derive(Debug, Clone)]
pub struct Pyramid(SmallVec<[ImageBuf<u8>; 8]>);

pub struct PyramidStep<'a> {
    pub fine : &'a ImageBuf<u8>,
    pub coarse : &'a ImageBuf<u8>
}

pub struct PyramidStepMut<'a> {
    pub fine : &'a mut ImageBuf<u8>,
    pub coarse : &'a mut ImageBuf<u8>
}

impl Pyramid {

    pub fn step(&self, lvl : usize) -> PyramidStep {
        PyramidStep { coarse : &self.0[lvl-1], fine : &self.0[lvl] }
    }

    pub fn step_mut(&mut self, lvl : usize) -> PyramidStepMut {
        let (pyr_before, pyr_after) = self.0.split_at_mut(lvl);
        let fine = pyr_before.last_mut().unwrap();
        let coarse = pyr_after.first_mut().unwrap();
        PyramidStepMut { fine, coarse }
    }

    pub fn new_symmetric(height : usize, width : usize) -> Self {
        assert!(height.is_power_of_two() && width.is_power_of_two());
        Self(SmallVec::from_buf([
            ImageBuf::new_constant((height / 2).max(1), (width / 2).max(1), 0),
            ImageBuf::new_constant((height / 4).max(1), (width / 4).max(1), 0),
            ImageBuf::new_constant((height / 8).max(1), (width / 8).max(1), 0),
            ImageBuf::new_constant((height / 16).max(1), (width / 16).max(1), 0),
            ImageBuf::new_constant((height / 32).max(1), (width / 32).max(1), 0),
            ImageBuf::new_constant((height / 64).max(1), (width / 64).max(1), 0),
            ImageBuf::new_constant((height / 128).max(1), (width / 128).max(1), 0),
            ImageBuf::new_constant((height / 256).max(1), (width / 256).max(1), 0)
        ]))
    }

    // A horizontal pyramid preserves the original height and thins its width by 2 at each step.
    pub fn new_horizontal(height : usize, width : usize) -> Self {
        assert!(width.is_power_of_two());
        Self(SmallVec::from_buf([
            ImageBuf::new_constant(height, (width / 2).max(1), 0),
            ImageBuf::new_constant(height, (width / 4).max(1), 0),
            ImageBuf::new_constant(height, (width / 8).max(1), 0),
            ImageBuf::new_constant(height, (width / 16).max(1), 0),
            ImageBuf::new_constant(height, (width / 32).max(1), 0),
            ImageBuf::new_constant(height, (width / 64).max(1), 0),
            ImageBuf::new_constant(height, (width / 128).max(1), 0),
            ImageBuf::new_constant(height, (width / 256).max(1), 0)
        ]))
    }

    // A vertical pyramid preserves the original width and thins its height by 2 at each step.
    pub fn new_vertical(height : usize, width : usize) -> Self {
        assert!(height.is_power_of_two());
        Self(SmallVec::from_buf([
            ImageBuf::new_constant((height / 2).max(1), width, 0),
            ImageBuf::new_constant((height / 4).max(1), width, 0),
            ImageBuf::new_constant((height / 8).max(1), width, 0),
            ImageBuf::new_constant((height / 16).max(1), width, 0),
            ImageBuf::new_constant((height / 32).max(1), width, 0),
            ImageBuf::new_constant((height / 64).max(1), width, 0),
            ImageBuf::new_constant((height / 128).max(1), width, 0),
            ImageBuf::new_constant((height / 256).max(1), width, 0)
        ]))
    }

}

impl AsRef<[ImageBuf<u8>]> for Pyramid {

    fn as_ref(&self) -> &[ImageBuf<u8>] {
        &self.0[..]
    }

}

impl AsMut<[ImageBuf<u8>]> for Pyramid {

    fn as_mut(&mut self) -> &mut [ImageBuf<u8>] {
        &mut self.0[..]
    }

}


