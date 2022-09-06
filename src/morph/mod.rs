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
    1, 1, 1, 1, 1,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
]);

pub const VBAR_5_KERNEL : Window<'static, u8> = Window::from_static::<25, 5>(&[
    0, 0, 1, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 1, 0, 0,
]);


const CROSS_5_ARR : [u8; 25] = [
    0, 0, 1, 0, 0,
    0, 0, 1, 0, 0,
    1, 1, 1, 1, 1,
    0, 0, 1, 0, 0,
    0, 0, 1, 0, 0,
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

pub const CROSS_KERNEL : Window<'static, u8> = Window::from_static::<9, 3>(&CROSS_ARR);

pub const NEG_CROSS_KERNEL : Window<'static, u8> = Window::from_static::<9, 3>(&CROSS_ARR);

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

#[derive(Debug, Clone)]
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

// Removes binary image fills, leaving only edges. The returned image
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
