#[cfg(feature="opencv")]
use opencv::{core, imgproc};

use crate::image::*;
use nalgebra::Scalar;
use std::fmt::Debug;
use num_traits::Zero;
use serde::{Serialize, Deserialize, de::DeserializeOwned};
use std::any::Any;
use std::iter::FromIterator;

// Gonzalez & Woods
pub fn boundary() {
    // A - (A (erode) B)
    // Where ("-" is set difference implemented by a.not(b).)
}

// Gonzalez & Woods
pub fn hole_fill() {
    // Start with A, an image with holes to be filled, and X0, an image
    // with a single white pixel anywhere within each hole. Let b be the 3x3 cross
    // SE. Then iterate:
    // X_k = (X_k-1 (dilate) B) \intersect A^C
    // Until X_k == X_k-1.
    // The set union of X_k and A contains all the filled holes and their boundaries.
}

// Gonzalez & Woods
pub fn connected_components() {
    // Let A be the set containing one or more connected components.
    // Let X0 be an image with a single white pixel within each connected component.
    // Let B be the 3x3 box SE for 8-neighborhood connectivity or
    // the 3x3 cross SE for 4-neighborhood connectivity.
    // Then iterate:
    // X_k = (X_k-1 (dilate) B) \intersect A
    // until X_k == X_k-1
}

pub fn hit_or_miss() {
    // a (hit-or-miss) b = a (*) b is defined as:
    // let b2 be the negation of b1 (b1 AND b2 == 0).
    // Then a (*) b = (a (erode) b1) - (a (dilate) \hat b2)
    // The resulting image contains points where b1 matches A and b2 matches A^C.
}

pub fn convex_hull() {
    // Let B1..B4 the the "c" structuring elements
    // Let X_k^i = (X_k-1 (hit-or-miss) B^i) (union) A for i=1..4
    // Let D_i be X_k^i at convergence (X_k^i == X_k)
    // Then C(A) = \union_i D^i is the convex hull of A.
}

pub fn thinning() {
    // The thinning operator (x) is:
    // A (x) B = A - (A (*) B) = A \intersect (A (*) B)^C
    // Let B^i be a rotated version of B^{i-1} in a sequence of SEs B^i ("helix" SEs).
    // Then A (x) {B} = (A (x) B^1) (x) B^2) ... (x) B^n)
    // Repeat until convergence.
}

pub fn thickening() {
    // The thickening operator (.) is:
    // A (.) B = A \union (A (*) B)
    // Thickening is just thinning the background:
    // Calculate thick(A) = thin(A^C)^C.
}

/*
Perhaps separate ses into mod neg { } and mod pos { }, where neg and pos
contain exactly the same SEs, but in inverted order.
*/

// Gonzalez & Woods:
// Erosion: A (erode) B = { z | (B)z \in A} (The set of all points z such that B translated by z is contained in A)
// or equivalently A (erode) B = { z | (B)z \intersect A^C = \empty } (the set of all points such that the
// intersection with complement of A is the empty set). Effect: Image details smaller than the structuring element
// are filtered away. Structuring elements are the smallest arrays padded with 0s such as to contain all coordinates
// of interest. A center point-like SE is the identity element for the erosion of the image.
// Dilation:
// A (dilate) B = { z | (\hat B)z \intersect A != \empty }, Where \hat B is the reflection of the structuring element
// B around its origin. This is the set resulting for all displacements z such that A and B share at least one element.
// Properties:
// (A (erode) B)^C = A^C (dilate) \hat B
// (A (dilate) B)^C = A^C (erode) \hat B
// if the SE is symmetric, \hat B = B.
// Structuring elements
pub mod se {

    /*const fn invert_array<const N : usize>(s : [N; u8]) -> [N; u8] {
        s.map(|el| 255 - el )
    }

    const fn invert_window

    const fn transpose_window*/

}

// Helix SEs are defined in clockwise fashion starting from horiz.

/*const fn transpose_array<const N : usize>(s : [N; u8]) -> [N; u8] {
    let mut out = s.clone();
    for i in 0..N.sqrt() {
        for j in 0..N.sqrt() {
            out[i*j + j] = s[j*i + i];
        }
    }
}*/

const fn negate_array<const N : usize>(arr : [u8; N]) -> [u8; N] {
    let mut b = [0; N];
    let mut i = 0;
    while i < N {
        b[i] = if arr[i] == 0 { 255 } else { 0 };
        i += 1;
    }
    b
}

pub const HELIX_HORIZ : Window<'static, u8> = Window::from_static::<9, 3>(&[
    0, 0, 0,
    255, 0, 255,
    0, 0, 0
]);

pub const HELIX_DIAG_BR : Window<'static, u8> = Window::from_static::<9, 3>(&[
    255, 0, 0,
    0, 0, 0,
    0, 0, 255
]);

// transpose(helix_horiz)
pub const HELIX_VERT : Window<'static, u8> = Window::from_static::<9, 3>(&[
    0, 255, 0,
    0, 0, 0,
    0, 255, 0
]);

// transpose(helix_diag_br)
pub const HELIX_DIAG_BL : Window<'static, u8> = Window::from_static::<9, 3>(&[
    0, 0, 255,
    0, 0, 0,
    255, 0, 0
]);

const HELIX_CLOCKWISE : [Window<'static, u8>; 4] = [
    HELIX_HORIZ,
    HELIX_DIAG_BR,
    HELIX_VERT,
    HELIX_DIAG_BL
];

const HELIX_COUNTER_CLOCKWISE : [Window<'static, u8>; 4] = [
    HELIX_HORIZ,
    HELIX_DIAG_BL,
    HELIX_VERT,
    HELIX_DIAG_BR
];

// point kernel does nothing, really, since it will effectively ignore the neighborhood since it will never match the image.

pub const C_UP_KERNEL : Window<'static, u8> = Window::from_static::<9, 3>(&[
    255, 0, 255,
    255, 0, 255,
    255, 255, 255
]);

pub const C_RIGHT_KERNEL : Window<'static, u8> = Window::from_static::<9, 3>(&[
    255, 255, 255,
    255, 0, 0,
    255, 255, 255
]);

pub const C_DOWN_KERNEL : Window<'static, u8> = Window::from_static::<9, 3>(&[
    255, 255, 255,
    255, 0, 255,
    255, 0, 255
]);

pub const C_LEFT_KERNEL : Window<'static, u8> = Window::from_static::<9, 3>(&[
    255, 255, 255,
    0, 0, 255,
    255, 255, 255
]);

const PT_ARR : [u8; 9] = [
    0, 0, 0,
    0, 255, 0,
    0, 0, 0
];

// point kernel does nothing, really, since it will effectively ignore the neighborhood since it will never match the image.
pub const POINT_KERNEL : Window<'static, u8> = Window::from_static::<9, 3>(&PT_ARR);

// The negative point kernel, however, is useful to remove salt noise from a binary image.
pub const NEG_POINT_KERNEL : Window<'static, u8> = Window::from_static::<9, 3>(
    &negate_array(PT_ARR)
);

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

const HBAR_ARR : [u8; 9] = [
    0, 0, 0,
    255, 255, 255,
    0, 0, 0
];

pub const HBAR_KERNEL : Window<'static, u8> = Window::from_static::<9, 3>(&HBAR_ARR);

pub const NEG_HBAR_KERNEL : Window<'static, u8> = Window::from_static::<9, 3>(&negate_array(HBAR_ARR));

pub const HBAR_5_KERNEL : Window<'static, u8> = Window::from_static::<25, 5>(&[
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    255, 255, 255, 255, 255,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
]);

pub const VBAR_5_KERNEL : Window<'static, u8> = Window::from_static::<25, 5>(&[
    0, 0, 255, 0, 0,
    0, 0, 255, 0, 0,
    0, 0, 255, 0, 0,
    0, 0, 255, 0, 0,
    0, 0, 255, 0, 0,
]);

const CROSS_5_ARR : [u8; 25] = [
    0, 0, 255, 0, 0,
    0, 0, 255, 0, 0,
    255, 255, 255, 255, 255,
    0, 0, 255, 0, 0,
    0, 0, 255, 0, 0,
];

pub const CROSS_5_KERNEL : Window<'static, u8> = Window::from_static::<25, 5>(&CROSS_5_ARR);

pub const NEG_CROSS_5_KERNEL : Window<'static, u8> = Window::from_static::<25, 5>(&negate_array(CROSS_5_ARR));

const VBAR_ARR : [u8; 9] = [
    0, 255, 0,
    0, 255, 0,
    0, 255, 0
];

pub const VBAR_KERNEL : Window<'static, u8> = Window::from_static::<9, 3>(&VBAR_ARR);

pub const NEG_VBAR_KERNEL : Window<'static, u8> = Window::from_static::<9, 3>(&negate_array(VBAR_ARR));

const CROSS_ARR : [u8; 9] = [
    0, 255, 0,
    255, 255, 255,
    0, 255, 0
];

const DIAMOND_ARR : [u8; 25] = [
    0, 0, 255, 0, 0,
    0, 255, 255, 255, 0,
    255, 255, 255, 255, 255,
    0, 255, 255, 255, 0,
    0, 0, 255, 0, 0,
];

pub const DIAMOND_KERNEL : Window<'static, u8> = Window::from_static::<25, 5>(&DIAMOND_ARR);

pub const NEG_DIAMOND_KERNEL : Window<'static, u8> = Window::from_static::<25, 5>(&negate_array(DIAMOND_ARR));

pub const CROSS_KERNEL : Window<'static, u8> = Window::from_static::<9, 3>(&CROSS_ARR);

pub const NEG_CROSS_KERNEL : Window<'static, u8> = Window::from_static::<9, 3>(&CROSS_ARR);

pub const BLOCK_KERNEL : Window<'static, u8> = Window::from_static::<9, 3>(&[
    255, 255, 255,
    255, 255, 255,
    255, 255, 255
]);

pub const BLOCK_5_KERNEL : Window<'static, u8> = Window::from_static::<25, 5>(&[
    255, 255, 255, 255, 255,
    255, 255, 255, 255, 255,
    255, 255, 255, 255, 255,
    255, 255, 255, 255, 255,
    255, 255, 255, 255, 255,
]);

pub const BLOCK_7_KERNEL : Window<'static, u8> = Window::from_static::<49, 7>(&[
    255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255,
]);

pub const BOX_KERNEL : Window<'static, u8> = Window::from_static::<9, 3>(&[
    255, 255, 255,
    255, 0, 255,
    255, 255, 255
]);

pub const DIAG_TL_BR_KERNEL : Window<'static, u8> = Window::from_static::<9, 3>(&[
    255, 0, 0,
    0, 255, 0,
    0, 0, 255
]);

pub const DIAG_TR_BL_KERNEL : Window<'static, u8> = Window::from_static::<9, 3>(&[
    0, 0, 255,
    0, 255, 0,
    255, 0, 0
]);

/*
Morphological operations consider a binary image as a set of (x, y) coordinates and a binary structuring element,
which is also a set of (x, y) elements. Dilation sets all the pixels that match the geometry of the structuring
element to foreground whenever there is any intersection between the foreground of the structuring element and
the image foreground. Erosion sets all pixels that match the geometry of the structutring element to background
whenever the intersection between foreground of the structuring element and foreground of image is not complete
(i.e. intersection = original structuring set to be maintained). If the intersection is complete, the image is left untouched.
Opening is an erosion followed by a dilation; Closing is an dilation followed by an erosion.

For any center pixel c in the output:
(Erosion) c = 1 iff all 1-neighbors match structuring element. c = 0 otherwise.
Conversely, if any one of the pixels matching structuring element are background, all pixels matching
structuring element are set to background.

(Dilation) c = 1 iff at least one 1-neighbor match structuring element. c = 0 otherwise.
*/

#[cfg(feature="opencv")]
#[derive(Clone, Debug)]
pub enum MorphShape {
    Rectangle,
    Cross,
    Ellipse
}

#[cfg(feature="opencv")]
#[derive(Debug)]
struct MorphKernel {
    iterations : i32,
    sz : i32,
    anchor : core::Point2i,
    border_val : core::Scalar,
    kernel : core::Mat
}

/*#[cfg(feature="opencv")]
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

#[cfg(feature="opencv")]
#[derive(Debug)]
pub struct Erosion<P>
where
    P : Pixel + Scalar + From<u8> + Debug + Copy + Default + Serialize + DeserializeOwned + Any
{
    tgt : ImageBuf<P>,
    kernel : MorphKernel
}

#[cfg(feature="opencv")]
impl<P> Erosion<P>
where
    P : Pixel + Scalar + From<u8> + Debug + Copy + Default + Zero + Serialize + DeserializeOwned + Any
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

#[cfg(feature="opencv")]
pub struct Dilation<P>
where
    P : Pixel + Scalar + From<u8> + Debug + Copy + Default + Serialize + DeserializeOwned + Any
{
    tgt : ImageBuf<P>,
    kernel : MorphKernel
}

#[cfg(feature="opencv")]
impl<P> Dilation<P>
where
    P : Pixel + Scalar + From<u8> + Debug + Copy + Default + Zero + Serialize + DeserializeOwned
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

}*/

#[derive(Debug, Clone)]
#[cfg(feature="ipp")]
pub struct IppiMorph {
    spec : Vec<u8>,
    buf : Vec<u8>,
    kernel : ImageBuf<u8>,
    is_dilate : bool
}

#[cfg(feature="ipp")]
impl IppiMorph {

    pub fn new(kernel : ImageBuf<u8>, img_sz : (usize, usize), is_dilate : bool) -> Self {
        unsafe {
            let kernel_sz = kernel.full_window().shape();
            let img_roi_sz = crate::foreign::ipp::ippcv::IppiSizeL { 
                width : img_sz.1 as i64, 
                height : img_sz.0 as i64 
            };
            let kernel_roi_sz = crate::foreign::ipp::ippcv::IppiSizeL { 
                width : kernel_sz.1 as i64, 
                height : kernel_sz.0 as i64 
            };
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

    pub fn apply<S, T>(&mut self, src : &Image<u8, S>, dst : &mut Image<u8, T>) 
    where
        S : Storage<u8>,
        T : StorageMut<u8>
    {

        assert!(src.shape() == dst.shape());

        unsafe {
            let (src_step, src_sz) = crate::image::ipputils::step_and_size_for_image(src);
            let (dst_step, dst_sz) = crate::image::ipputils::step_and_size_for_image(&dst);
            let src_roi = crate::foreign::ipp::ippcv::IppiSizeL { 
                width : src.width() as i64, 
                height : src.height() as i64 
            };
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

#[derive(Debug, Clone)]
#[cfg(feature="ipp")]
pub struct IppiGrayMorph {
    buf : Vec<u8>,
    kernel : ImageBuf<i32>,
    is_dilate : bool
}

#[cfg(feature="ipp")]
impl IppiGrayMorph {

    pub fn new(kernel : ImageBuf<i32>, img_sz : (usize, usize), is_dilate : bool) -> Self {
        assert!(kernel.height() % 2 == 1 && kernel.width() % 2 == 1);
        unsafe {
            let kernel_sz = kernel.full_window().shape();
            let img_roi_sz = crate::foreign::ipp::ippcv::IppiSize { 
                width : img_sz.1 as i32, 
                height : img_sz.0 as i32 
            };
            let kernel_roi_sz = crate::foreign::ipp::ippcv::IppiSize { 
                width : kernel_sz.1 as i32, 
                height : kernel_sz.0 as i32 
            };
            let mut spec_sz = 0i32;
            let ans = crate::foreign::ipp::ippcv::ippiMorphGrayGetSize_8u_C1R(
                img_roi_sz, 
                kernel.as_ptr(), 
                kernel_roi_sz, 
                &mut spec_sz
            );
            assert!(ans == 0);
            let mut buf : Vec<u8> = (0..(spec_sz as usize)).map(|_| 0u8 ).collect();
            let center_anchor = crate::foreign::ipp::ippcv::IppiPoint { 
                x : kernel.width() as i32 / 2 + 1,
                y : kernel.height() as i32 / 2 + 1
            };
            let ans = crate::foreign::ipp::ippcv::ippiMorphGrayInit_8u_C1R(
                buf.as_mut_ptr() as *mut _, 
                img_roi_sz,
                kernel.as_ptr(), 
                kernel_roi_sz, 
                center_anchor
            );
            assert!(ans == 0);
            Self { buf, kernel, is_dilate }
        }
    }

    pub fn apply<S, T>(&mut self, src : &Image<u8, S>, dst : &mut Image<u8, T>) 
    where
        S : Storage<u8>,
        T : StorageMut<u8>
    {
        assert!(src.shape() == dst.shape());
        unsafe {
            let (src_step, src_sz) = crate::image::ipputils::step_and_size_for_image(src);
            let (dst_step, dst_sz) = crate::image::ipputils::step_and_size_for_image(&dst);
            let src_roi = crate::foreign::ipp::ippcv::IppiSize { 
                width : src.width() as i32, 
                height : src.height() as i32 
            };
            let border_const_val = &0u8 as *const _;
            unsafe {
                let ans = if self.is_dilate {
                    crate::foreign::ipp::ippcv::ippiGrayDilateBorder_8u_C1R(
                        src.as_ptr(),
                        src_step as i32,
                        dst.as_mut_ptr(),
                        dst_step as i32,
                        src_roi,
                        crate::foreign::ipp::ippcv::_IppiBorderType_ippBorderRepl,
                        self.buf.as_mut_ptr() as *mut _
                    )
                } else {
                    crate::foreign::ipp::ippcv::ippiGrayErodeBorder_8u_C1R(
                        src.as_ptr(),
                        src_step as i32,
                        dst.as_mut_ptr(),
                        dst_step as i32,
                        src_roi,
                        crate::foreign::ipp::ippcv::_IppiBorderType_ippBorderRepl,
                        self.buf.as_mut_ptr() as *mut _
                    )
                };
                assert!(ans == 0);
            }
        }
    }

}

#[cfg(feature="ipp")]
pub struct IppiTopHat {

    spec : Vec<u8>,

    buffer : Vec<u8>

}

#[cfg(feature="ipp")]
impl IppiTopHat {

    pub fn new(height : usize, width : usize, mask : ImageBuf<u8>) -> Self {
        let roi_sz = crate::foreign::ipp::ippcv::IppiSizeL::from((height, width));
        let mask_sz = crate::foreign::ipp::ippcv::IppiSizeL::from(mask.size());
        let data_ty = crate::foreign::ipp::ippcv::IppDataType_ipp8u;
        let num_channels = 1;
        let mut spec_sz : i64 = 0;
        unsafe {
            crate::foreign::ipp::ippcv::ippiMorphGetSpecSize_L(
                roi_sz,
                mask_sz,
                data_ty,
                num_channels,
                &mut spec_sz as *mut _
            );
            assert!(spec_sz > 0);
            let mut spec = Vec::from_iter((0..(spec_sz as usize)).map(|_| 0u8 ));
            let spec_ptr : *mut crate::foreign::ipp::ippcv::IppiMorphAdvStateL = std::mem::transmute::<_, _>(spec.as_mut_ptr());
            let status = crate::foreign::ipp::ippcv::ippiMorphInit_L(
                roi_sz,
                mask.as_ptr(),
                mask_sz,
                data_ty,
                num_channels,
                spec_ptr
            );
            let mut buf_sz : i64 = 0;
            let ans = crate::foreign::ipp::ippcv::ippiMorphGetBufferSize_L(
                roi_sz,
                mask_sz,
                data_ty,
                num_channels,
                &mut buf_sz as *mut _
            );
            assert!(ans == 0);
            assert!(buf_sz > 0);
            let buffer = Vec::from_iter((0..(buf_sz as usize)).map(|_| 0u8 ));
            Self { spec, buffer }
        }
    }

    pub fn calculate<S, T>(&mut self, img : &Image<u8, S>, dst : &mut Image<u8, T>)
    where
        S : Storage<u8>,
        T : StorageMut<u8>
    {
        let border_val = 0u8;
        unsafe {
            let ans = crate::foreign::ipp::ippcv::ippiMorphTophat_8u_C1R_L(
                img.as_ptr(),
                img.byte_stride() as i64,
                dst.as_mut_ptr(),
                dst.byte_stride() as i64,
                crate::foreign::ipp::ippcv::IppiSizeL::from(img.size()),
                crate::foreign::ipp::ippcv::_IppiBorderType_ippBorderDefault,
                &border_val as *const _,
                self.spec.as_mut_ptr() as *mut _,
                self.buffer.as_mut_ptr()
            );
            assert!(ans == 0);
        }
    }

}

/*fn remove_salt(
    src_dst : &mut WindowMut<u8>,
    sum_dst : &mut WindowMut<u8>,
) {
    assert!(src_dst.width() / sum_dst.width() == 3 && src_dst.height() / sum_dst.height() == 3);
    crate::local::baseline_local_sum(src_dst.as_ref(), sum_dst);
    for i in 0..sum_dst.height() {
        for j in 0..sum_dst.width() {
            let center = (i * 3 + 1, j * 3 + 1);
            if sum_dst[(i, j)] == 1 {
                if center == 1 {
                    src_dst[center] = 0;
                } else {
                    let sum_top = sum_dst[(i-1, j)];
                    let sum_right = sum_dst[(i, j+1)];
                    let sum_bottom = sum_dst[(i+1, j)];
                    let sum_left = sum_dst[(i, j-1)];
                    if sum_top == 0 && sum_right == 0 && sum_bottom == 0 && sum_left == 0 {
                        src_dst.sub_window_mut((i*3, j*3), (3,3)).fill(0);
                    }
                    if  == 0 {

                    } else if  == 0 {

                    }
                }
            }
        }
    }
}*/

/*
/// Take a binary image and transform thick into thin lines and shrink dense shapes. Ignore borders.
/// After Davies (2005) alg. 2.15.
pub fn shrink(src : &Window<u8>, mut dst : WindowMut<u8>, win_side : usize) {
    assert!(win_side % 2 == 0);
    assert!(src.shape() == dst.shape());
    let center = (win_side / 2, win_side / 2);
    let noncentral_max_sum = ((win_side.pow(2) - 1)*255) as u64;
    for (mut d, s) in dst.windows_mut((win_side, win_side)).zip(src.windows((win_side, win_side))) {
        let sum = crate::stat::accum::<_, u64>(&s) - s[center] as u64;
        if sum < noncentral_max_sum {
            d[center] = 0;
        } else {
            d[center] = 255;
        }
    }
}

/// Take a binary image and transform thin into thick lines and stretch dense shapes. Ignore borders.
/// After Davies (2005) alg. 2.16.
pub fn expand(src : &Window<u8>, mut dst : WindowMut<u8>, win_side : usize) {
    assert!(win_side % 2 == 0);
    assert!(src.shape() == dst.shape());
    let center = (win_side / 2, win_side / 2);
    for (mut d, s) in dst.windows_mut((win_side, win_side)).zip(src.windows((win_side, win_side))) {
        let sum = crate::stat::accum::<_, u64>(&s) - s[center] as u64;
        if sum > 0 {
            d[center] = 255;
        } else {
            d[center] = 0;
        }
    }
}

*/

/*// Removes binary image fills, leaving only edges. The returned image
// is guaranteed to only contain edges which can have at most 2-pixel
// thickness, since the sum window edge might fall in the middle of a small
// object. Therefore objects with side of 4x4 pixels will not have their fill removed, but
// objects with side 5x5 pixels and larger will have the center pixel of their fill removed.
// The binary image is assumed to contain only ones and zeros. A second pass of this function
// over the window [1..n] at either or both directions is guaranteed to leave edges with 1 pixel only.
pub fn remove_fill(
    src_dst : &mut WindowMut<u8>,
    sum_dst : &mut WindowMut<u8>,
) {
    assert!(src_dst.width() / sum_dst.width() == 3 && src_dst.height() / sum_dst.height() == 3);
    crate::local::baseline_local_sum(src_dst.as_ref(), sum_dst);
    for i in 0..sum_dst.height() {
        for j in 0..sum_dst.width() {
            if sum_dst[(i, j)] == 9 {
                src_dst[(i * 3 + 1, j * 3 + 1)] = 0;
            } else {
                // The difference in sums to the neighbors or sum might suggest
                // if we need to re-evaluate them as well. If the sum allows an
                // object to be in the middle, evaluate it. Else pass.
            }
        }
    }
}*/

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

/* Noise removal routines. They would fit into shutter::local::logic, as would morphology ops */
// Davies 2.18
/*pub fn remove_salt_noise(src : &Window<u8>, mut dst : WindowMut<u8>, win_side : usize) {
    let value_is = 255;
    let neighbors_are = 0;
    conditional_set_when_neighbors_sum_equal(src, dst, win_side, value_is, neighbors_are, 0);
}

pub fn conditional_set_when_neighbors_sum_equal(
    src : &Window<u8>,
    mut dst : WindowMut<u8>,
    win_side : usize,
    cond_val : u8,
    eq_val : u64,
    true_val : u8
) -> usize {
    assert!(win_side % 2 == 0);
    assert!(src.shape() == dst.shape());
    let center = (win_side / 2, win_side / 2);
    let mut n_changes = 0;
    for (mut d, s) in dst.windows_mut((win_side, win_side)).zip(src.windows((win_side, win_side))) {
        if s[center] == 255 {
            let sum = crate::stat::accum::<_, u64>(&s) - s[center] as u64;
            if sum == 0 {
                d[center] = 0;
                n_changes += 1;
            } else {
                d[center] = s[center];
            }
        } else {
            d[center] = s[center];
        }
    }
    n_changes
}

// Davies 2.19
pub fn remove_pepper_noise(src : &Window<u8>, mut dst : WindowMut<u8>, win_side : usize) {
    let noncentral_max_sum = ((win_side.pow(2) - 1)*255) as u64;
    let value_is = 0;
    conditional_set_when_neighbors_sum_equal(src, dst, win_side, value_is, noncentral_max_sum, 255);
}

// Davies 2.20
pub fn remove_salt_and_pepper_noise(src : &Window<u8>, mut dst : WindowMut<u8>, win_side : usize) {
    assert!(win_side % 2 == 0);
    assert!(src.shape() == dst.shape());
    let center = (win_side / 2, win_side / 2);
    let noncentral_max_sum = ((win_side.pow(2) - 1)*255) as u64;
    for (mut d, s) in dst.windows_mut((win_side, win_side)).zip(src.windows((win_side, win_side))) {
        let sum = crate::stat::accum::<_, u64>(&s) - s[center] as u64;
        match s[center] {
            0 => {
                if sum == noncentral_max_sum {
                    d[center] = 255;
                } else {
                    d[center] = s[center];
                }
            },
            255 => {
                if sum == 0 {
                    d[center] = 0;
                } else {
                    d[center] = s[center];
                }
            },
            _ => { }
        }
        if s[center] == 255 {
            if sum == 0 {

            } else {
                d[center] = s[center];
            }
        } else {
            d[center] = s[center];
        }
    }
}

pub fn binary_edges(src : &Window<u8>, mut dst : WindowMut<u8>, win_side : usize) {
    let noncentral_max_sum = ((win_side.pow(2) - 1)*255) as u64;
    set_when_neighbors_sum_equal(src, dst, win_side, noncentral_max_sum, 0, Some(255));
}

pub fn set_when_neighbors_sum_less(
    src : &Window<u8>,
    mut dst : WindowMut<u8>,
    win_side : usize,
    less_val : u64,
    true_val : u8,
    false_val : Option<u8>
) -> usize {
    assert!(win_side % 2 == 0);
    assert!(src.shape() == dst.shape());
    let mut n_changes = 0;
    let center = (win_side / 2, win_side / 2);
    for (mut d, s) in dst.windows_mut((win_side, win_side)).zip(src.windows((win_side, win_side))) {
        let sum = crate::stat::accum::<_, u64>(&s) - s[center] as u64;
        if sum < less_val {
            d[center] = true_val;
            n_changes += 1;
        } else {
            d[center] = false_val.unwrap_or(src[center]);
        }
    }
    n_changes
}

pub fn set_when_neighbors_sum_greater(
    src : &Window<u8>,
    mut dst : WindowMut<u8>,
    win_side : usize,
    gt_val : u64,
    true_val : u8,
    false_val : Option<u8>
) -> usize {
    assert!(win_side % 2 == 0);
    assert!(src.shape() == dst.shape());
    let center = (win_side / 2, win_side / 2);
    let mut n_changes = 0;
    for (mut d, s) in dst.windows_mut((win_side, win_side)).zip(src.windows((win_side, win_side))) {
        let sum = crate::stat::accum::<_, u64>(&s) - s[center] as u64;
        if sum > gt_val {
            d[center] = true_val;
            n_changes += 1;
        } else {
            d[center] = false_val.unwrap_or(s[center]);
        }
    }
    n_changes
}

pub fn conditional_set_when_neighbors_sum_greater(
    src : &Window<u8>,
    mut dst : WindowMut<u8>,
    win_side : usize,
    cond_val : u8,
    gt_val : u64,
    true_val : u8,
    false_val : Option<u8>
) -> usize {
    assert!(win_side % 2 == 0);
    assert!(src.shape() == dst.shape());
    let center = (win_side / 2, win_side / 2);
    let mut n_changes = 0;
    for (mut d, s) in dst.windows_mut((win_side, win_side)).zip(src.windows((win_side, win_side))) {
        if s[center] == cond_val {
            let sum = crate::stat::accum::<_, u64>(&s) - s[center] as u64;
            if sum > gt_val {
                d[center] = true_val;
                n_changes += 1;
            } else {
                d[center] = false_val.unwrap_or(s[center]);
            }
        } else {
            d[center] = false_val.unwrap_or(s[center]);
        }
    }
    n_changes
}*/
/*
/// Keep only nonzero pixels when at least k in [1..8] neighboring nonzero pixels are also present,
/// irrespective of the position of those pixels.
pub fn supress_binary_speckles(win : &Window<'_, u8>, mut out : WindowMut<'_, u8>, min_count : usize) {

    assert!(win.shape() == out.shape());
    assert!(min_count >= 1 && min_count <= 8);

    for ((r, c), neigh) in win.labeled_neighborhoods() {
        if neigh.filter(|px| *px > 0 ).count() >= min_count {
            out[(r, c)] = win[(r, c)];
        } else {
            out[(r, c)] = 0;
        }
    }
}
*/
