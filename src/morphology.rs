use opencv::{core, imgproc};
use crate::image::*;
use nalgebra::Scalar;
use std::fmt::Debug;

#[derive(Clone, Debug)]
pub enum MorphShape {
    Rectangle,
    Cross,
    Ellipse
}

struct MorphKernel {
    iterations : i32,
    sz : i32,
    anchor : core::Point2i,
    border_val : core::Scalar,
    kernel : core::Mat
}

fn build_kernel(sz : usize, iterations : usize, shape : MorphShape) -> MorphKernel {
    let kernel_sz = (2*sz+1) as i32;
    let shape = match shape {
        MorphShape::Rectangle => imgproc::MORPH_RECT,
        MorphShape::Cross => imgproc::MORPH_CROSS,
        MorphShape::Ellipse => imgproc::MORPH_ELLIPSE
    };
    let kernel = imgproc::get_structuring_element(
        shape,
        core::Size2i { width : kernel_sz, height : kernel_sz },
        core::Point2i { x : sz as i32, y : sz as i32 }
    ).unwrap();
    let anchor = core::Point2i{x : -1, y : -1};
    let border_val = core::Scalar::from(1.);
    MorphKernel { iterations : iterations as i32, sz : sz as i32, anchor, border_val, kernel }
}

pub struct Erosion<P>
where
    P : Scalar + From<u8> + Debug + Copy + Default
{
    tgt : Image<P>,
    kernel : MorphKernel
}

impl<P> Erosion<P>
where
    P : Scalar + From<u8> + Debug + Copy + Default
{

    pub fn new(img_sz : (usize, usize), kernel_sz : usize, shape : MorphShape, iterations : usize) -> Self {
        let kernel = build_kernel(kernel_sz, iterations, shape);
        Self{ tgt : Image::new_constant(img_sz.0, img_sz.1, P::from(0)), kernel }
    }

    pub fn apply(&mut self, img : &Window<P>) -> Option<&Image<P>> {
        let src : core::Mat = img.clone().into();
        let mut dst : core::Mat = self.tgt.full_window_mut().into();
        unsafe {
            imgproc::erode(
                &src,
                &mut dst,
                &self.kernel.kernel,
                self.kernel.anchor,
                self.kernel.iterations,
                core::BORDER_CONSTANT,
                self.kernel.border_val
            ).ok()?;
        }
        Some(&self.tgt)
    }

    pub fn retrieve(&self) -> &Image<P> {
        &self.tgt
    }
}

pub struct Dilation<P>
where
    P : Scalar + From<u8> + Debug + Copy + Default
{
    tgt : Image<P>
}

impl<P> Dilation<P>
where
    P : Scalar + From<u8> + Debug + Copy + Default
{

    pub fn new(dim : (usize, usize)) -> Self {
        Self{ tgt : Image::new_constant(dim.0, dim.1, P::from(0)) }
    }

    pub fn apply(&mut self) -> Option<&Image<P>> {
        /*imgproc::dilate(
            src: &dyn ToInputArray,
            dst: &mut dyn ToOutputArray,
            kernel: &dyn ToInputArray,
            anchor: Point,
            iterations: i32,
            border_type: i32,
            border_value: Scalar
        ).ok()?*/
        unimplemented!()
    }

}

// TODO opening=erosion followed by dilation; closing=dilationn followed by erosion.

