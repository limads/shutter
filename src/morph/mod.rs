use opencv::{core, imgproc};
use crate::image::*;
use nalgebra::Scalar;
use std::fmt::Debug;
use num_traits::Zero;
use serde::{Serialize, Deserialize, de::DeserializeOwned};
use std::any::Any;
use std::iter::FromIterator;

// point kernel does nothing, really, since it will effectively ignore the neighborhood since it will never match the image.
pub const POINT_KERNEL : Window<'static, u8> = Window::from_static::<9, 3>(&[
    0, 0, 0,
    0, 255, 0,
    0, 0, 0
]);

pub const DIAG_KERNEL : Window<'static, u8> = Window::from_static::<9, 3>(&[
    255, 0, 0,
    0, 255, 0,
    0, 0, 255
]);

pub const ANTI_DIAG_KERNEL : Window<'static, u8> = Window::from_static::<9, 3>(&[
    0, 0, 255,
    0, 255, 0,
    255, 0, 0
]);

pub const HBAR_PAIR_KERNEL : Window<'static, u8> = Window::from_static::<9, 3>(&[
    255, 255, 255,
    0, 0, 0,
    255, 255, 255
]);

pub const VBAR_PAIR_KERNEL : Window<'static, u8> = Window::from_static::<9, 3>(&[
    255, 0, 255,
    255, 0, 255,
    255, 0, 255
]);

pub const HBAR_KERNEL : Window<'static, u8> = Window::from_static::<9, 3>(&[
    0, 0, 0,
    255, 255, 255,
    0, 0, 0
]);

pub const VBAR_KERNEL : Window<'static, u8> = Window::from_static::<9, 3>(&[
    0, 255, 0,
    0, 255, 0,
    0, 255, 0
]);

pub const BLOCK_KERNEL : Window<'static, u8> = Window::from_static::<9, 3>(&[
    255, 255, 255,
    255, 255, 255,
    255, 255, 255
]);

pub const BOX_KERNEL : Window<'static, u8> = Window::from_static::<9, 3>(&[
    255, 255, 255,
    255, 0, 255,
    255, 255, 255
]);

/*
Morphological operations consider a binary image as a set of (x, y) coordinates and a binary structuring element,
which is also a set of (x, y) elements. Dilation sets all the pixels that match the geometry of the structuring
element to foreground whenever there is any intersection between the foreground of the structuring element and
the image foreground. Erosion sets all pixels that match the geometry of the structutring element to background
whenever the intersection between foreground of the structuring element and foreground of image is not complete
(i.e. intersection = original structuring set). If the intersection is complete, the image is left untouched.
Opening is an erosion followed by a dilation; Closing is an dilation followed by an erosion.

For any center pixel c in the output:
(Erosion) c = 1 iff all 1-neighbors match structuring element. c = 0 otherwise.
Conversely, if any one of the pixels matching structuring element are background, all pixels matching
structuring element are set to background.

(Dilation) c = 1 iff at least one 1-neighbor match structuring element. c = 0 otherwise.
*/

#[derive(Clone, Debug)]
pub enum MorphShape {
    Rectangle,
    Cross,
    Ellipse
}

#[derive(Debug)]
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

#[derive(Debug)]
pub struct Erosion<P>
where
    P : Scalar + From<u8> + Debug + Copy + Default + Serialize + DeserializeOwned + Any
{
    tgt : Image<P>,
    kernel : MorphKernel
}

impl<P> Erosion<P>
where
    P : Scalar + From<u8> + Debug + Copy + Default + Zero + Serialize + DeserializeOwned + Any
{

    pub fn new(img_sz : (usize, usize), kernel_sz : usize, shape : MorphShape, iterations : usize) -> Self {
        let kernel = build_kernel(kernel_sz, iterations, shape);
        Self{ tgt : Image::new_constant(img_sz.0, img_sz.1, P::from(0)), kernel }
    }

    pub fn erode(&mut self, img : &Window<P>) -> Option<&Image<P>> {
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
    P : Scalar + From<u8> + Debug + Copy + Default + Serialize + DeserializeOwned + Any
{
    tgt : Image<P>,
    kernel : MorphKernel
}

impl<P> Dilation<P>
where
    P : Scalar + From<u8> + Debug + Copy + Default + Zero + Serialize + DeserializeOwned
{

    pub fn new(img_sz : (usize, usize), kernel_sz : usize, shape : MorphShape, iterations : usize) -> Self {
        let kernel = build_kernel(kernel_sz, iterations, shape);
        Self{ tgt : Image::new_constant(img_sz.0, img_sz.1, P::from(0)), kernel }
    }

    pub fn dilate(&mut self, img : &Window<P>) -> Option<&Image<P>> {
        let src : core::Mat = img.clone().into();
        let mut dst : core::Mat = self.tgt.full_window_mut().into();
        unsafe {
            imgproc::dilate(
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

#[cfg(feature="ipp")]
pub struct IppiMorph {
    spec : Vec<u8>,
    buf : Vec<u8>,
    kernel : Image<u8>,
    is_dilate : bool
}

#[cfg(feature="ipp")]
impl IppiMorph {

    pub fn new(kernel : Image<u8>, img_sz : (usize, usize), is_dilate : bool) -> Self {
        unsafe {
            let kernel_sz = kernel.full_window().shape();
            let img_roi_sz = crate::foreign::ipp::ippcv::IppiSizeL { width : img_sz.1 as i64, height : img_sz.0 as i64 };
            let kernel_roi_sz = crate::foreign::ipp::ippcv::IppiSizeL { width : kernel_sz.1 as i64, height : kernel_sz.0 as i64 };
            // let img_roi_sz : IppiSizeL = (img_sz.0 * img_sz.1) as i128;
            // let kernel_roi_sz = (kernel_sz.0 * kernel_sz.1) as i128;
            let num_channels = 1;
            let mut buf_sz = 0i64;
            let mut spec_sz = 0i64;
            let ans = if is_dilate {
                crate::foreign::ipp::ippcv::ippiDilateGetBufferSize_L(
                    img_roi_sz,
                    kernel_roi_sz,
                    crate::foreign::ipp::ippcv::IppDataType_ipp8u,
                    num_channels,
                    &mut buf_sz as *mut _
                )
            } else {
                crate::foreign::ipp::ippcv::ippiErodeGetBufferSize_L(
                    img_roi_sz,
                    kernel_roi_sz,
                    crate::foreign::ipp::ippcv::IppDataType_ipp8u,
                    num_channels,
                    &mut buf_sz as *mut _
                )
            };
            assert!(ans == 0);
            let ans = if is_dilate {
                crate::foreign::ipp::ippcv::ippiDilateGetSpecSize_L(
                    img_roi_sz,
                    kernel_roi_sz,
                    &mut spec_sz as *mut _
                )
            } else {
                crate::foreign::ipp::ippcv::ippiErodeGetSpecSize_L(
                    img_roi_sz,
                    kernel_roi_sz,
                    &mut spec_sz as *mut _
                )
            };
            assert!(ans == 0);
            let mut buf = Vec::from_iter((0..(buf_sz as usize)).map(|_| 0u8 ));
            let mut spec = Vec::from_iter((0..(spec_sz as usize)).map(|_| 0u8 ));
            let ans = if is_dilate {
                crate::foreign::ipp::ippcv::ippiDilateInit_L(
                    img_roi_sz,
                    kernel.full_window().as_ptr(),
                    kernel_roi_sz,
                    std::mem::transmute(spec.as_mut_ptr())
                )
            } else {
                crate::foreign::ipp::ippcv::ippiErodeInit_L(
                    img_roi_sz,
                    kernel.full_window().as_ptr(),
                    kernel_roi_sz,
                    std::mem::transmute(spec.as_mut_ptr())
                )
            };
            assert!(ans == 0);
            Self { spec, buf, kernel, is_dilate }
        }
    }

    pub fn apply(&mut self, src : &Window<'_, u8>, dst : &mut WindowMut<'_, u8>) {

        assert!(src.shape() == dst.shape());

        unsafe {
            let (src_step, src_sz) = crate::image::ipputils::step_and_size_for_window(src);
            let (dst_step, dst_sz) = crate::image::ipputils::step_and_size_for_window_mut(&dst);

            let src_roi = crate::foreign::ipp::ippcv::IppiSizeL { width : src.width() as i64, height : src.height() as i64 };
            let border_const_val = &0u8 as *const _;
            unsafe {
                let ans = if self.is_dilate {
                    crate::foreign::ipp::ippcv::ippiDilate_8u_C1R_L(
                        src.as_ptr(),
                        src_step as i64,
                        dst.as_mut_ptr(),
                        dst_step as i64,
                        src_roi,
                        crate::foreign::ipp::ippcv::_IppiBorderType_ippBorderRepl,
                        border_const_val,
                        std::mem::transmute(self.spec.as_ptr()),
                        self.buf.as_mut_ptr()
                    )
                } else {
                    crate::foreign::ipp::ippcv::ippiErode_8u_C1R_L(
                        src.as_ptr(),
                        src_step as i64,
                        dst.as_mut_ptr(),
                        dst_step as i64,
                        src_roi,
                        crate::foreign::ipp::ippcv::_IppiBorderType_ippBorderRepl,
                        border_const_val,
                        std::mem::transmute(self.spec.as_ptr()),
                        self.buf.as_mut_ptr()
                    )
                };
                assert!(ans == 0);
            }
        }
    }

}

/*pub struct Opening {

}

impl Opening {

    fn from(dil : Dilation, er : Erosion) -> Self { }

}

pub struct Closing { }

impl Closing {

    fn from(er : Erosion, dil : Dilation) -> Self { }

}*/

/*
morph_gradient(): dilate(img) - erode(img)
valley_detect(): closing(img) - img
boundary_detect(): img - erosion(img)
*/

/*pub trait Morphology {

}

impl<'a> Morphology for Window<'a, u8> {

    // pub fn dilate(&self, morph : &mut IppiMorph)

}

impl Morphology for Dilation {

}*/

// ippiMorphGetSpecSize_L(IppiSizeL roiSize, IppiSizeL maskSize, IppDataType dataType, int
// numChannels, IppSizeL* pSpecSize);

// TODO opening=erosion followed by dilation; closing=dilationn followed by erosion.

// IppStatus ippiDilate3x3_64f_C1R(const Ipp64f* pSrc, int srcStep, Ipp<datatype>* pDst,
// int dstStep, IppiSize roiSize );

// IppStatus ippiErode3x3_64f_C1R(const Ipp64f* pSrc, int srcStep, Ipp64f* pDst, int
// dstStep, IppiSize roiSize );
