use nalgebra::*;
use ripple::signal::sampling::{self};
use std::ops::{Index, IndexMut, Mul, Add, AddAssign, MulAssign, SubAssign, Range, Div, Rem};
use simba::scalar::SubsetOf;
use std::fmt;
use std::fmt::Debug;
use simba::simd::{AutoSimd};
use std::convert::TryFrom;
use crate::feature::patch::{self, Patch};
use itertools::Itertools;
use crate::feature::patch::ColorMode;
use num_traits::Zero;
use serde::{Serialize, Deserialize, de::DeserializeOwned};
use std::any::Any;
use tempfile;
use std::fs;
use std::io::Write;
use num_traits::cast::FromPrimitive;
use num_traits::cast::AsPrimitive;
use num_traits::float::Float;
use std::borrow::{Borrow, BorrowMut};
use std::mem;
use crate::raster::*;
use crate::draw::*;
use crate::sparse::RunLength;
use num_traits::bounds::Bounded;

impl<'a> Window<'a, f32> {

    pub fn show(&self) {
        use crate::convert::*;
        use crate::point::*;
        let mut n = self.clone_owned();
        n.full_window_mut().abs_mut();
        normalize_max_inplace(n.as_mut());
        n.full_window_mut().scalar_mul(255.0);
        let dst : Image<u8> = n.convert(Conversion::Preserve);
        dst.show();
    }
    
}

impl Image<f32> {

    pub fn show(&self) {
        self.full_window().show();
    }
    
}

pub trait Pixel
where
    Self : Clone + Copy + Scalar + Debug + Zero + Bounded + Any + Default + 'static
{

    fn depth() -> Depth;

}

impl Pixel for u8 {

    fn depth() -> Depth {
        Depth::U8
    }

}

impl Pixel for u16 {

    fn depth() -> Depth {
        Depth::U16
    }

}

impl Pixel for i16 {

    fn depth() -> Depth {
        Depth::I16
    }

}

impl Pixel for i32 {

    fn depth() -> Depth {
        Depth::I32
    }

}

impl Pixel for f32 {

    fn depth() -> Depth {
        Depth::F32
    }

}

pub enum Depth {
    U8,
    U16,
    I16,
    I32,
    F32
}

pub trait UnsignedPixel where Self : Pixel { }

impl UnsignedPixel for u8 { }

impl UnsignedPixel for u16 { }

// impl UnsignedPixel for u32 { }

pub trait SignedPixel where Self : Pixel { }

impl SignedPixel for i16 { }

impl SignedPixel for i32 { }

// impl SignedPixel for i64 { }

pub trait FloatPixel where Self : Pixel { }

impl FloatPixel for f32 { }

// impl FloatPixel for f64 { }

// TODO make indexing operations checked at debug builds. They are segfaulting if the
// user passes an overflowing index, since the impl is using get_unchecked regardless for now.

#[cfg(feature="opencv")]
use opencv::core;

#[cfg(feature="opencv")]
use opencv::imgproc;

//#[cfg(feature="mkl")]
// mod fft;

/*#[cfg(feature="mkl")]
pub use fft::*;

#[cfg(feature="gsl")]
pub(crate) mod dwt;

#[cfg(feature="gsl")]
pub use dwt::*;*/

// #[cfg(feature="gsl")]
// mod interp;

// #[cfg(feature="gsl")]
// pub use interp::Interpolation2D;

// TODO image.equalize() (contrast equalization) image.strech() (constrast stretching)

use crate::io;

#[cfg(feature="ipp")]
pub(crate) mod ipputils;

#[cfg(feature="opencv")]
pub mod cvutils;

pub mod index;

pub(crate) mod iter;

/*pub trait Raster {

    fn width(&self) -> usize;

    fn height(&self) -> usize;

    /// Returns (height x width)
    fn dimensions(&self) -> (usize, usize) {
        (self.height(), self.width())
    }

    fn pixels(&self) -> impl Iterator<item=&u8>;

}*/

/// Owned digital image occupying a dynamically-allocated buffer. If you know your image content
/// at compile time, consider using Window::from_constant, which saves up the allocation.
/// Images are backed by Box<[N]>, because once a buffer is allocated, it cannot grow or
/// shrink like Vec<N>.
/// Fundamentally, an image differs from a matrix because
/// it is oriented row-wise in memory, while a matrix is oriented column-wise. Also, images are
/// bounded at a low and high end, because they are the product of a saturated digital quantization
/// process. But indexing, following OpenCV convention, happens from the top-left point, following
/// the matrix convention.
#[repr(C)]
#[derive(Debug, Clone, Default)]
pub struct Image<N> 
where
    N : Scalar + Clone + Copy + Any + Debug
{
    pub(crate) buf : Box<[N]>,
    width : usize,
    offset : (usize, usize),
    size : (usize, usize)
}

// Perhaps rename to GrayImage? (but float image is also gray).
pub type ByteImage = Image<u8>;

pub type FloatImage = Image<f32>;

impl<N> fmt::Display for Image<N>
where
    N : Scalar + Clone + Copy + Serialize + DeserializeOwned + Any + Default + num_traits::Zero
{

    fn fmt(&self, f : &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Image (Height = {}; Width = {})", self.height(), self.width())
    }

}

/*impl<N> Sub<Rhs=Image> for Image<N> {

    pub fn sub() -> {

    }
}*/

impl<N> Image<N>
where
    N : Scalar + Copy + Any {

    pub fn leak(self) {
        Box::leak(self.buf);
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.buf.len() / self.width, self.width)
    }

    pub fn area(&self) -> usize {
        self.buf.len()
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.buf.len() / self.width
    }

}

fn verify_size_and_alignment<A, B>() {
    assert!(mem::size_of::<A>() == mem::size_of::<B>());
    assert!(mem::align_of::<A>() == mem::align_of::<B>());
}

impl<'a, N> AsRef<Window<'a, N>> for Image<N>
where
    N : Scalar + Clone + Copy + Any + Debug
{

    fn as_ref(&self) -> &Window<'a, N> {
        verify_size_and_alignment::<Self, Window<'a, N>>();
        unsafe { mem::transmute(self) }
    }

}

impl<'a, N> AsRef<Window<'a, N>> for WindowMut<'a, N>
where
    N : Scalar + Clone + Copy + Any + Debug
{

    fn as_ref(&self) -> &Window<'a, N> {
        verify_size_and_alignment::<Self, Window<'a, N>>();
        unsafe { mem::transmute(self) }
    }

}

impl<'a, N> Borrow<Window<'a, N>> for Image<N>
where
    N : Scalar + Clone + Copy + Any + Debug
{

    fn borrow(&self) -> &Window<'a, N> {
        verify_size_and_alignment::<Self, Window<'a, N>>();
        unsafe { mem::transmute(self) }
    }

}

impl<'a, N> Borrow<WindowMut<'a, N>> for Image<N>
where
    N : Scalar + Clone + Copy + Any + Debug
{

    fn borrow(&self) -> &WindowMut<'a, N> {
        verify_size_and_alignment::<Self, Window<'a, N>>();
        unsafe { mem::transmute(self) }
    }
}

impl<'a, N> BorrowMut<WindowMut<'a, N>> for Image<N>
where
    N : Scalar + Clone + Copy + Any + Debug
{

    fn borrow_mut(&mut self) -> &mut WindowMut<'a, N> {
        verify_size_and_alignment::<Self, WindowMut<'a, N>>();
        unsafe { mem::transmute(self) }
    }

}

impl<'a, N> AsMut<WindowMut<'a, N>> for Image<N>
where
    N : Scalar + Clone + Copy + Any + Debug
{

    fn as_mut(&mut self) -> &mut WindowMut<'a, N> {
        verify_size_and_alignment::<Self, Window<'a, N>>();
        unsafe { mem::transmute(self) }
    }

}

/*// Not possible, since requires Borrow<WindowMut> as bound. But Borrow<WindowMut>
// conflicts with Borrow<Window>
impl<'a, N> BorrowMut<WindowMut<'a, N>> for Image<N>
where
    N : Scalar + Clone + Copy + Any + Debug
{

    fn borrow_mut(&mut self) -> &mut WindowMut<'a, N> {
        verify_size_and_alignment::<Self, Window<'a, N>>();
        unsafe { mem::transmute(self) }
    }

}*/

fn join_windows<N>(s : &[Window<'_, N>], horizontal : bool) -> Option<Image<N>>
where
    N : Scalar + Debug + Clone + Copy + Default + Zero
{
    if s.len() == 0 {
        return None;
    }
    let shape = s[0].shape();
    if !s.iter().skip(1).all(|w| w.shape() == shape ) {
        return None;
    }
    let width = if horizontal { shape.1 * s.len() } else { shape.1 };
    let height = if horizontal { shape.0 } else { shape.0 * s.len() };
    let n = width * height;
    let mut buf = Vec::with_capacity(n);
    unsafe { buf.set_len(n); }
    let mut img = Image {
        buf : buf.into_boxed_slice(),
        offset : (0, 0),
        width,
        size : (height, width)
    };
    for ix in 0..s.len() {
        let mut win_mut = if horizontal {
            img.window_mut((0, ix*shape.1), shape)
        } else {
            img.window_mut((ix*shape.0, 0), shape)
        };
        win_mut.unwrap().copy_from(&s[ix]);
    }
    Some(img)
}

pub trait PixelCopy<N>
where
    N : Scalar + Copy + Default + Zero + Copy + Any + Pixel
{

    // Copies the content of this image into the center of a larger buffer with size new_sz,
    // setting the remaining border entries to the pad value.
    fn padded_copy(&self, new_sz : (usize, usize), pad : N) -> Image<N>;

    /// Copies the content of this image into the center of a larger buffer with size
    /// new_sz, mirroring the left margin into the right margin (and vice-versa), and mirroring
    /// the top-margin into the bottom-margin (and vice-versa).
    fn wrapped_copy(&self, new_sz : (usize, usize)) -> Image<N>;

    /// Copies the content of this image into the center of a larger buffer with
    /// size new_sz, replicating the first value of each row to the left border;
    /// the last value of each row to the right border, the first value of each column
    /// to the top border, and the last value of each column to the bottom border.
    fn replicated_copy(&self, new_sz : (usize, usize)) -> Image<N>;

    // If true, copy from self. If not, copy from other.
    fn conditional_alt_copy(&self, mask : &Window<'_, u8>, false_alt : &Window<'_, N>) -> Image<N>;

    // If true at nonzero mask, copy from self. No nothing otherwise.
    fn conditional_copy(&self, mask : &Window<'_, u8>) -> Image<N>;

}

impl<N> PixelCopy<N> for Window<'_, N>
where
    N : Scalar + Copy + Default + Zero + Copy + Any + Pixel
{

    fn padded_copy(&self, new_sz : (usize, usize), pad : N) -> Image<N> {
        assert!(new_sz.0 > self.height() && new_sz.1 > self.width());
        let mut new = unsafe { Image::new_empty(new_sz.0, new_sz.1) };
        new.full_window_mut().padded_copy_from(self, pad);
        new
    }

    fn wrapped_copy(&self, new_sz : (usize, usize)) -> Image<N> {
        assert!(new_sz.0 > self.height() && new_sz.1 > self.width());
        let mut new = unsafe { Image::new_empty(new_sz.0, new_sz.1) };
        new.full_window_mut().wrapped_copy_from(self);
        new
    }

    fn replicated_copy(&self, new_sz : (usize, usize)) -> Image<N> {
        assert!(new_sz.0 > self.height() && new_sz.1 > self.width());
        let mut new = unsafe { Image::new_empty(new_sz.0, new_sz.1) };
        new.full_window_mut().replicated_copy_from(self);
        new
    }

    fn conditional_alt_copy(&self, mask : &Window<'_, u8>, alternative : &Window<'_, N>) -> Image<N> {
        let mut new = unsafe { Image::new_empty_like(self) };
        new.full_window_mut().conditional_alt_copy_from(mask, self, alternative);
        new
    }

    fn conditional_copy(&self, mask : &Window<'_, u8>) -> Image<N> {
        let mut new = unsafe { Image::new_empty_like(self) };
        new.full_window_mut().conditional_copy_from(mask, self);
        new
    }
}

impl<N> PixelCopy<N> for Image<N>
where
    N : Scalar + Copy + Default + Zero + Copy + Any + Pixel
{

    // Copies the content of this image into the center of a larger buffer with size new_sz,
    // setting the remaining border entries to the pad value.
    fn padded_copy(&self, new_sz : (usize, usize), pad : N) -> Image<N> {
        self.full_window().padded_copy(new_sz, pad)
    }

    /// Copies the content of this image into the center of a larger buffer with size
    /// new_sz, mirroring the left margin into the right margin (and vice-versa), and mirroring
    /// the top-margin into the bottom-margin (and vice-versa).
    fn wrapped_copy(&self, new_sz : (usize, usize)) -> Image<N> {
        self.full_window().wrapped_copy(new_sz)
    }

    /// Copies the content of this image into the center of a larger buffer with
    /// size new_sz, replicating the first value of each row to the left border;
    /// the last value of each row to the right border, the first value of each column
    /// to the top border, and the last value of each column to the bottom border.
    fn replicated_copy(&self, new_sz : (usize, usize)) -> Image<N> {
        self.full_window().replicated_copy(new_sz)
    }

    // If true, copy from self. If not, copy from other.
    fn conditional_alt_copy(&self, mask : &Window<'_, u8>, false_alt : &Window<'_, N>) -> Image<N> {
        self.full_window().conditional_alt_copy(mask, false_alt)
    }

    // If true at nonzero mask, copy from self. No nothing otherwise.
    fn conditional_copy(&self, mask : &Window<'_, u8>) -> Image<N> {
        self.full_window().conditional_copy(mask)
    }

}

impl<N> WindowMut<'_, N>
where
    N : Scalar + Copy + Default + Zero + Copy + Any
{

    // Copies elements into the center of self self from other, assuming dim(self) > dim(other), padding the
    // border elements with a constant user-supplied value.
    pub fn padded_copy_from(&mut self, other : &Window<'_, N>, pad : N) {
        verify_border_dims(other, self);
        let (bh, bw) = border_dims(other, self);

        #[cfg(feature="ipp")]
        unsafe {
            let (src_step, src_sz) = crate::image::ipputils::step_and_size_for_window(other);
            let (dst_step, dst_sz) = crate::image::ipputils::step_and_size_for_window_mut(&self);
            if self.pixel_is::<u8>() {
                //let ans = self.apply_to_sub((bh, bw), *other.size(), |mut dst| {
                let (dst_step, dst_sz) = crate::image::ipputils::step_and_size_for_window_mut(&self);
                let ans = crate::foreign::ipp::ippi::ippiCopyConstBorder_8u_C1R(
                    mem::transmute(other.as_ptr()),
                    src_step,
                    src_sz,
                    mem::transmute(self.as_mut_ptr()),
                    dst_step,
                    dst_sz,
                    bh as i32,
                    bw as i32,
                    *std::mem::transmute::<_, &u8>(&pad)
                );
                // });
                assert!(ans == 0);
                return;
            }
        }

        /*self.apply_to_sub_mut((0, 0), (100, 100), |mut win| {
            self.shape();
        });*/

        unimplemented!()
    }

    // Copies elements into the center of self self from other, assuming dim(self) > dim(other), wrapping the
    // border elements (i.e. border elements are copied by reflecting elements from the opposite border).
    pub fn wrapped_copy_from(&mut self, other : &Window<'_, N>) {
        verify_border_dims(other, self);

        let (bh, bw) = border_dims(other, self);

        #[cfg(feature="ipp")]
        unsafe {
            let (src_step, src_sz) = crate::image::ipputils::step_and_size_for_window(other);
            let (dst_step, dst_sz) = crate::image::ipputils::step_and_size_for_window_mut(&self);
            if self.pixel_is::<f32>() {
                //let ans = self.apply_to_sub((bh, bw), *other.size(), |mut dst| {

                let ans = crate::foreign::ipp::ippi::ippiCopyWrapBorder_32f_C1R(
                    mem::transmute(other.as_ptr()),
                    src_step,
                    src_sz,
                    mem::transmute(self.as_mut_ptr()),
                    dst_step,
                    dst_sz,
                    bh as i32,
                    bw as i32,
                );
                // });
                assert!(ans == 0);
                return;
            }
        }

        unimplemented!()
    }


    // Copies elements into the center of self self from other, assuming dim(self) > dim(other), padding the
    // border elements with the last element of other at each border pixel.
    pub fn replicated_copy_from(&mut self, other : &Window<'_, N>) {
        verify_border_dims(other, self);

        let (bh, bw) = border_dims(other, self);

         #[cfg(feature="ipp")]
        unsafe {
            let (src_step, src_sz) = crate::image::ipputils::step_and_size_for_window(other);
            let (dst_step, dst_sz) = crate::image::ipputils::step_and_size_for_window_mut(&self);
            let ans = if self.pixel_is::<u8>() {
                crate::foreign::ipp::ippi::ippiCopyReplicateBorder_8u_C1R(
                    mem::transmute(other.as_ptr()),
                    src_step,
                    src_sz,
                    mem::transmute(self.as_mut_ptr()),
                    dst_step,
                    dst_sz,
                    bh as i32,
                    bw as i32,
                )
            } else if self.pixel_is::<f32>() {
                crate::foreign::ipp::ippi::ippiCopyReplicateBorder_32f_C1R(
                    mem::transmute(other.as_ptr()),
                    src_step,
                    src_sz,
                    mem::transmute(self.as_mut_ptr()),
                    dst_step,
                    dst_sz,
                    bh as i32,
                    bw as i32,
                )
            } else {
                panic!("Invalid type");
            };
            assert!(ans == 0);
            return;
        }

        unimplemented!()
    }

    // Copies entries from true_other into self only where pixels at the mask are nonzero,
    // and from false_other into self when mask is zero.
    // self[px] = mask[px] == 0 ? false_other else true_other[px].
    pub fn conditional_alt_copy_from(&mut self, mask : &Window<'_, u8>, true_alt : &Window<'_, N>, false_alt : &Window<'_, N>) {
        // TODO remove unsafe (cannot borrow `*self` as mutable more than once at a time)
        unsafe { (&mut *(self as *mut WindowMut<'_, N>)).copy_from(false_alt) };
        self.conditional_copy_from(mask, true_alt);
    }

    // Copies entries from other into self only where pixels at the mask are nonzero.
    // self[px] = mask[px] == 0 ? nothing else other[px]
    pub fn conditional_copy_from(&mut self, mask : &Window<'_, u8>, other : &Window<'_, N>) {

        assert!(self.shape() == other.shape(), "Windows differ in shape");
        assert!(self.shape() == mask.shape(), "Windows differ in shape");

        #[cfg(feature="ipp")]
        unsafe {
            let (src_step, src_sz) = crate::image::ipputils::step_and_size_for_window(other);
            let (dst_step, dst_sz) = crate::image::ipputils::step_and_size_for_window_mut(&self);
            let (mask_step, mask_sz) = crate::image::ipputils::step_and_size_for_window(mask);

            if self.pixel_is::<u8>() {
                let ans = crate::foreign::ipp::ippi::ippiCopy_8u_C1MR(
                    mem::transmute(other.as_ptr()),
                    src_step,
                    mem::transmute(self.as_mut_ptr()),
                    dst_step,
                    src_sz,
                    mask.as_ptr(),
                    mask_step
                );
                assert!(ans == 0);
                return;
            }

            if self.pixel_is::<i16>() {
                let ans = crate::foreign::ipp::ippi::ippiCopy_16s_C1MR(
                    mem::transmute(other.as_ptr()),
                    src_step,
                    mem::transmute(self.as_mut_ptr()),
                    dst_step,
                    src_sz,
                    mask.as_ptr(),
                    mask_step
                );
                assert!(ans == 0);
                return;
            }

            if self.pixel_is::<i32>() {
                let ans = crate::foreign::ipp::ippi::ippiCopy_32s_C1MR(
                    mem::transmute(other.as_ptr()),
                    src_step,
                    mem::transmute(self.as_mut_ptr()),
                    dst_step,
                    src_sz,
                    mask.as_ptr(),
                    mask_step
                );
                assert!(ans == 0);
                return;
            }

            if self.pixel_is::<f32>() {
                let ans = crate::foreign::ipp::ippi::ippiCopy_32f_C1MR(
                    mem::transmute(other.as_ptr()),
                    src_step,
                    mem::transmute(self.as_mut_ptr()),
                    dst_step,
                    src_sz,
                    mask.as_ptr(),
                    mask_step
                );
                assert!(ans == 0);
                return;
            }
        }

        for ((mut d, s), m) in self.pixels_mut(1).zip(other.pixels(1)).zip(mask.pixels(1)) {
            if *m != 0 {
                *d = *s
            }
        }
    }

    // Copies elements into self from other, assuming same dims.
    pub fn copy_from(&mut self, other : & Window<'_, N>) {
        assert!(self.shape() == other.shape(), "Mutable windows differ in shape");

        #[cfg(feature="ipp")]
        unsafe {
            let (src_step, src_sz) = crate::image::ipputils::step_and_size_for_window(other);
            let (dst_step, dst_sz) = crate::image::ipputils::step_and_size_for_window_mut(&self);
            if self.pixel_is::<u8>() {
                let ans = crate::foreign::ipp::ippi::ippiCopy_8u_C1R(
                    mem::transmute(other.as_ptr()),
                    src_step,
                    mem::transmute(self.as_mut_ptr()),
                    dst_step,
                    src_sz
                );
                assert!(ans == 0);
                return;
            }

            if self.pixel_is::<i16>() {
                let ans = crate::foreign::ipp::ippi::ippiCopy_16s_C1R(
                    mem::transmute(other.as_ptr()),
                    src_step,
                    mem::transmute(self.as_mut_ptr()),
                    dst_step,
                    src_sz
                );
                assert!(ans == 0);
                return;
            }

            if self.pixel_is::<i32>() {
                let ans = crate::foreign::ipp::ippi::ippiCopy_32s_C1R(
                    mem::transmute(other.as_ptr()),
                    src_step,
                    mem::transmute(self.as_mut_ptr()),
                    dst_step,
                    src_sz
                );
                assert!(ans == 0);
                return;
            }

            if self.pixel_is::<f32>() {
                let ans = crate::foreign::ipp::ippi::ippiCopy_32f_C1R(
                    mem::transmute(other.as_ptr()),
                    src_step,
                    mem::transmute(self.as_mut_ptr()),
                    dst_step,
                    src_sz
                );
                assert!(ans == 0);
                return;
            }
        }

        self.rows_mut().zip(other.rows())
            .for_each(|(this, other)| this.copy_from_slice(other) );
    }

}

impl<N> Image<N>
where
    N : Scalar + Copy + Default + Zero + Copy + Any
{

    /* Mirroring ops are useful to represent image correlations via convolution:
    corr(a,b) = conv(a,mirror(b)) = conv(mirror(a), b) */
    pub fn mirror_vertically(&self) -> Image<N> {
        unimplemented!()
    }

    pub fn mirror_horizontally(&self) -> Image<N> {
        unimplemented!()
    }

    pub fn mirror(&self) -> Image<N> {
        unimplemented!()
    }

    pub fn transpose(&self) -> Image<N> {
        let mut dst = self.clone();
        dst.size = (self.size.1, self.size.0);
        dst.width = self.size.0;
        dst.transpose_from(&self.full_window());
        dst
    }

    pub fn transpose_from<'a>(&'a mut self, src : &Window<N>) {

        assert!(src.width() == self.height() && src.height() == self.width());

        #[cfg(feature="ipp")]
        unsafe {
            let (dst_step, dst_roi) = crate::image::ipputils::step_and_size_for_window_mut(&self.full_window_mut());
            let (src_step, src_roi) = crate::image::ipputils::step_and_size_for_window(src);
            if self.pixel_is::<u8>() {
                let ans = crate::foreign::ipp::ippi::ippiTranspose_8u_C1R(
                    mem::transmute(src.as_ptr()),
                    src_step,
                    mem::transmute(self.full_window_mut().as_mut_ptr()),
                    dst_step,
                    src_roi
                );
                assert!(ans == 0);
                return;
            }
        }

        /*use nalgebra::*;
        DMatrixSlice::from_slice(&src.buf[..], src.height(), src.width())
            .transpose_to(&mut DMatrixSliceMut::from_slice(&mut self.buf[..], src.width(), src.height()));*/
        unimplemented!()
    }

    pub fn transpose_mut(&mut self) {
        let original = self.clone();
        self.transpose_from(&original.full_window());
    }

    // Splits this image over S owning, memory-contiguous blocks. This does not
    // copy the underlying data and is useful to split the work across multiple threads
    // using ownership-based mechanisms (such as channels).
    pub fn split<const S : usize>(mut self) -> [Image<N>; S] {
        unimplemented!()
    }

    // Join S blocks of same size into an image again. This is useful to recover an image
    // after it has been split across different blocks with Image::split to process an image
    // in multithreaded algorithms. Passing images
    // whose buffers are not memory-contigous returns None, since the underlying vector
    // is built assuming zero-copy by just reclaiming ownership of the blocks, and this
    // cannot be done with images with non-contiguous memory blocks.
    pub fn concatenate(s : &[Window<'_, N>]) -> Option<Image<N>> {
        join_windows(s, true)
    }

    pub fn stack(s : &[Window<'_, N>]) -> Option<Image<N>> {
        join_windows(s, false)
    }

    pub unsafe fn unchecked_linear_index(&self, ix : usize) -> &N {
        self.buf.get_unchecked(ix)
    }

    pub unsafe fn unchecked_linear_index_mut(&mut self, ix : usize) -> &mut N {
        self.buf.get_unchecked_mut(ix)
    }

    pub fn linear_index(&self, ix : usize) -> &N {
        &self.buf[ix]
    }

    pub fn linear_index_mut(&mut self, ix : usize) -> &mut N {
        &mut self.buf[ix]
    }

    pub fn from_iter(iter : impl Iterator<Item=N>, width : usize) -> Self {
        let buf : Vec<N> = iter.collect();
        Self::from_vec(buf, width)
    }

    /*pub fn subsample_from(&mut self, content : &[N], ncols : usize, sample_n : usize) {
        assert!(ncols < content.len(), "ncols smaller than content length");
        let nrows = content.len() / ncols;
        let sparse_ncols = ncols / sample_n;
        let sparse_nrows = nrows / sample_n;
        self.width = sparse_ncols;
        if self.buf.len() != sparse_nrows * sparse_ncols {
            self.buf.clear();
            self.buf.extend((0..(sparse_nrows*sparse_ncols)).map(|_| N::zero() ));
        }
        for r in 0..sparse_nrows {
            for c in 0..sparse_ncols {
                self.buf[r*sparse_ncols + c] = content[r*sample_n*ncols + c*sample_n];
            }
        }
    }*/

    pub fn new_from_slice(source : &[N], width : usize) -> Self {
        let mut buf = Vec::with_capacity(source.len());
        let height = buf.len() / width;
        unsafe { buf.set_len(source.len()); }
        buf.copy_from_slice(&source);
        Self{ buf : buf.into_boxed_slice(), width, offset : (0, 0), size : (height, width) }
    }

    pub fn from_vec(buf : Vec<N>, width : usize) -> Self {
        //if buf.len() as f64 % ncols as f64 != 0.0 {
        //    panic!("Invalid image lenght");
        //}
        assert!(buf.len() % width == 0);
        let height = buf.len() / width;
        Self { buf : buf.into_boxed_slice(), width, offset : (0, 0), size : (height, width) }
    }

    pub fn from_rows<const R : usize, const C : usize>(pxs : [[N; C]; R]) -> Self {
        let mut buf : Vec<N> = Vec::with_capacity(R*C);
        for r in pxs {
            buf.extend(r.into_iter());
        }
        Self { buf : buf.into_boxed_slice(), width : C, offset : (0, 0), size : (R, C) }
    }

    pub fn from_cols<const R : usize, const C : usize>(pxs : [[N; C]; R]) -> Self {
        let mut img = Image::from_rows(pxs);
        img.transpose_mut();
        img
    }

    // Creates a new image with the same dimensions as other, but with values set from the given scalar.
    pub fn new_constant_like<'b, M>(other : &Window<'b, M>, value : N) -> Self
    where
        Window<'b, M> : Raster
    {
        Self::new_constant(other.height(), other.width(), value)
    }

    // Creates a new image with the same dimensions as other, with uninitialized values.
    // Make sure you write to the allocated buffer before reading from it, otherwise
    // the access is UB.
    pub unsafe fn new_empty_like<'b, M>(other : &Window<'b, M>) -> Self
    where
        Window<'b, M> : Raster
    {
        Self::new_empty(other.height(), other.width())
    }

    pub fn new_constant(height : usize, width : usize, value : N) -> Self
    {
        /*let mut buf = Vec::with_capacity(height * width);
        buf.extend((0..(height*width)).map(|_| value ));
        Self{ buf : buf.into_boxed_slice(), width, offset : (0, 0), size : (height, width) }*/
        let mut img = unsafe { Image::new_empty(height, width) };
        img.full_window_mut().fill(value);
        img
    }
    
    /// Initializes an image with allocated, but undefined content. Reading the contents
    /// of this image before writing to all entries is UB, which is why this funciton is marked
    /// as unsafe. Make sure your program copies content from another image, or set
    /// all its values with fill() before you attempt to read from it.
    pub unsafe fn new_empty(nrows : usize, ncols : usize) -> Self {
        assert_nonzero((nrows, ncols));
        let mut buf = Vec::<N>::with_capacity(nrows * ncols);
        buf.set_len(nrows * ncols);
        Self { buf : buf.into_boxed_slice(), width : ncols, offset : (0, 0), size : (nrows, ncols) }
    }

    /// Returns a borrowed view over the whole window. Same as self.as_ref(). But is
    /// convenient to have, since type inference for the AsRef impl might not be triggered
    /// or you need an owned version of the window
    pub fn full_window<'a>(&'a self) -> Window<'a, N> {
        self.window((0, 0), self.shape()).unwrap()
    }
    
    pub fn row(&self, ix : usize) -> Option<&[N]> {
        if ix >= self.height() {
            return None;
        }
        let start = ix*self.width;
        Some(&self.buf[start..(start+self.width)])
    }

    /// Returns a mutably borrowed view over the whole window. Note the current mutable reference
    /// to the window is invalidated when this view enters into scope. Same as self.as_mut(). But is
    /// convenient to have, since type inference for the AsMut impl might not be triggered, or you
    /// need an owned version of the window.
    pub fn full_window_mut<'a>(&'a mut self) -> WindowMut<'a, N> {
        let shape = self.shape();
        self.window_mut((0, 0), shape).unwrap()
    }
    
    pub fn window<'a>(&'a self, offset : (usize, usize), sz : (usize, usize)) -> Option<Window<'a, N>> {
        assert_nonzero(sz);
        let orig_sz = self.shape();
        if offset.0 + sz.0 <= orig_sz.0 && offset.1 + sz.1 <= orig_sz.1 {
            Some(Window {
                win : &self.buf[..],
                offset,
                width : self.width,
                win_sz : sz
            })
        } else {
            None
        }
    }
    
    pub fn window_mut<'a>(&'a mut self, offset : (usize, usize), sz : (usize, usize)) -> Option<WindowMut<'a, N>> {
        assert_nonzero(sz);
        let orig_sz = self.shape();
        if offset.0 + sz.0 <= orig_sz.0 && offset.1 + sz.1 <= orig_sz.1 {
            Some(WindowMut {
                win : &mut self.buf[..],
                offset,
                width : self.width,
                win_sz : sz
            })
        } else {
            None
        }
    }
    
    /*pub fn downsample(&mut self, src : &Window<N>) {
        assert!(src.is_full());
        let src_ncols = src.width;
        let dst_ncols = self.width;
        
        // TODO resize yields a nullpointer when allocating its data.
        #[cfg(feature="ipp")]
        unsafe {
            let dst_nrows = self.buf.len() / self.width;
            ipputils::resize(src.win, &mut self.buf);
        }

        panic!("Image::downsample requires that crate is compiled with opencv or ipp feature");

        // TODO use resize::resize for native Rust solution
    }*/
    
    pub fn copy_from(&mut self, other : &Image<N>) {

        // IppStatus ippiCopy_8uC1R(const Ipp<datatype>* pSrc, int srcStep, Ipp<datatype>* pDst,
        // int dstStep, IppiSize roiSize);
        // self.buf.copy_from_slice(&other.buf[..]);
        self.full_window_mut().copy_from(&other.full_window());
    }
    
    pub fn copy_from_slice(&mut self, slice : &[N]) {
        assert!(self.buf.len() == slice.len());
        self.buf.copy_from_slice(&slice);
    }

    pub fn as_slice(&self) -> &[N] {
        &self.buf[..]
    }
    
    pub fn as_mut_slice(&mut self) -> &mut [N] {
        &mut self.buf[..]
    }
    
    /*pub fn convert_from(&mut self, other : &Image<N>) {
        self.buf.as_slice_mut().copy_from(other.buf.as_slice());
    }*/
    
    pub fn windows_mut(&mut self, sz : (usize, usize)) -> impl Iterator<Item=WindowMut<'_, N>>
    where
        N : Mul<Output=N> + MulAssign
    {
        assert_nonzero(sz);
        self.full_window_mut().windows_mut(sz)
    }

    // pub fn center_sub_window(&self) -> Window<'_, N> {
    // }

    // Returns an iterator over the clockwise sub-window characterizing a pattern.
    // pub fn clockwise_sub_windows(&self) -> impl Iterator<Item=Window

    pub fn windows(&self, sz : (usize, usize)) -> impl Iterator<Item=Window<'_, N>>
    where
        N : Mul<Output=N> + MulAssign
    {
        assert_nonzero(sz);
        self.full_window().windows(sz)
    }
    
    /*pub fn iter(&self) -> impl Iterator<Item=&N> {
        let shape = self.shape();
        iterate_row_wise(&self.buf[..], (0, 0), shape, shape)
    }*/
    
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    // pub fn windows(&self) -> impl Iterator<Item=Window<'_, N>> {
    //    unimplemented!()
    // }
    
}

impl<N> Image<N>
    where N : Scalar + Copy + RealField + Copy + Any
{

    pub fn scale_by(&mut self, scalar : N)  {
        self.buf.iter_mut().for_each(|val| *val *= scalar);
    }
    
    pub fn unscale_by(&mut self, scalar : N)  {
        self.buf.iter_mut().for_each(|val| *val *= scalar);
    }
    
    /*pub fn iter(&self) -> impl Iterator<Item=&N> {
        self.full_window().clone().iter()
    }*/
    
}


impl<T> Raster for Image<T>
where
    T : Scalar + Copy
{

    type Slice = Box<[T]>;

    fn create(offset : (usize, usize), win_sz : (usize, usize), orig_sz : (usize, usize), win : Self::Slice) -> Self {
        assert!(offset == (0, 0));
        assert!(win_sz.0 * win_sz.1 == win.len());
        assert!(orig_sz == win_sz);
        Self {
            offset,
            size : win_sz,
            width : win_sz.1,
            buf : win
        }
    }

    fn offset(&self) -> &(usize, usize) {
        &self.offset
    }

    fn size(&self) -> &(usize, usize) {
        &self.size
    }

    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.size.0
    }

    fn original_width(&self) -> usize {
        self.width
    }

    fn original_height(&self) -> usize {
        self.buf.len() / self.width
    }

    fn original_size(&self) -> (usize, usize) {
        (self.original_height(), self.width)
    }

    unsafe fn original_slice(&mut self) -> Self::Slice {
        self.buf.clone()
    }

    unsafe fn essential_slice(&mut self) -> Self::Slice {
        self.buf.clone()
    }

}

impl Image<u8> {

    pub fn show(&self) {

        use std::process::Command;
        use tempfile;

        let mut tf = tempfile::NamedTempFile::new().unwrap();
        let png = crate::io::encode(self.clone()).unwrap();
        tf.write_all(&png).unwrap();
        let path = tf.path();
        let new_path = format!("{}.png", path.to_str().unwrap());
        fs::rename(path, new_path.clone()).unwrap();
        Command::new("eog")
            .args(&[&new_path])
            .output()
            .unwrap();
    }

    // Creates an image from a binary pattern. Bits start from the top-left
    // and progress clockwise wrt the center pixel.
    pub fn new_from_pattern(side : usize, center : bool, pattern : u8) -> Image<u8> {
        let mut pattern = crate::texture::pattern::to_binary(pattern);
        pattern.iter_mut().for_each(|p| *p *= 255 );
        let mut img = unsafe { Image::new_empty(side, side) };
        let mut wins = img.windows_mut((side / 3, side / 3));
        wins.next().unwrap().fill(pattern[0]);
        wins.next().unwrap().fill(pattern[1]);
        wins.next().unwrap().fill(pattern[2]);
        wins.next().unwrap().fill(pattern[7]);
        wins.next().unwrap().fill(if center { 255 } else { 0 });
        wins.next().unwrap().fill(pattern[3]);
        wins.next().unwrap().fill(pattern[6]);
        wins.next().unwrap().fill(pattern[5]);
        wins.next().unwrap().fill(pattern[4]);
        std::mem::drop(wins);
        img
    }

    /* TODO create new_from_pattern where Pattern is an Enum with checkerboard
    as one of its patterns */
    pub fn new_checkerboard(sz : usize, sq_sz : usize) -> Self {
        assert!(sz % sq_sz == 0);
        let mut img = Self::new_constant(sz, sz, 255);
        let n_sq = sz / sq_sz;
        for row in 0..n_sq {
            for col in 0..n_sq {
                if (row % 2 == 0 && col % 2 == 0) || (row % 2 != 0 && col % 2 != 0) {
                    img.window_mut((row*sq_sz, col*sq_sz), (sq_sz, sq_sz)).unwrap().fill(0);
                }
            }
        }
        img
    }

    /*pub fn draw(&mut self, mark : Mark) {
        self.full_window_mut().draw(mark);
    }*/

    /// Builds a binary image from a set of points.
    pub fn binary_from_points(nrow : usize, ncol : usize, pts : impl Iterator<Item=(usize, usize)>) -> Self {
        let mut img = Image::<u8>::new_constant(nrow, ncol, 0);
        for pt in pts {
            img[pt] = 255;
        }
        img
    }
}

impl Image<f32> {

    pub fn max(&self) -> ((usize, usize), f32) {
        let (mut max_ix, mut max) = ((0, 0), f32::NEG_INFINITY);
        for (lin_ix, px) in self.full_window().pixels(1).enumerate() {
            if *px > max {
                max_ix = index::coordinate_index(lin_ix, self.width);
                max = *px
            }
        }
        (max_ix, max)
    }
}

impl<N> Index<(usize, usize)> for Image<N> 
where
    N : Scalar + Copy + Any
{

    type Output = N;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        unsafe { self.buf.get_unchecked(index::linear_index(index, self.width)) }
    }
}

impl<N> IndexMut<(usize, usize)> for Image<N>
where
    N : Scalar + Copy + Default + Copy + Any
{
    
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        unsafe { self.buf.get_unchecked_mut(index::linear_index(index, self.width)) }
    }
    
}

/// Borrowed subset of an image. Referencing the whole source slice (instead of just its
/// portion of interest) might be useful to represent overlfowing operations (e.g. draw)
/// as long as the operation does not violate bounds of the original image. We just have
/// to be careful to not expose the rest of the image in the API.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Window<'a, N> 
//where
//    N : Scalar
{
    // Original image full slice. Might refer to an actual pre-allocated image
    // buffer slice; or a slice from an external source (which is why we don't
    // simply reference image here).
    pub(crate) win : &'a [N],

    // Original image dimensions (height, width). orig_sz.0 MUST be win.len() / orig_sz.1
    pub(crate) width : usize,

    // Window offset, with respect to the top-left point (row, col).
    pub(crate) offset : (usize, usize),
    
    // This window size.
    pub(crate) win_sz : (usize, usize),

}

pub struct WindowNeighborhood<'a, N>
where
    N : Scalar
{
    pub center : Window<'a, N>,
    pub left : Window<'a, N>,
    pub top : Window<'a, N>,
    pub right : Window<'a, N>,
    pub bottom : Window<'a, N>,
}

impl<'a, N> Window<'a, N> {

    /*pub const fn from_constant<const L : usize, const H : usize>(val : N) -> Window<'static, N> {
        let arr : [N; L] = [val; L];
        Window {
            win : &arr,
            width : (L / H),
            offset : (0, 0),
            win_sz : (H, L / H)
        }
    }*/

    pub const fn from_static<const S : usize, const W : usize>(array : &'static [N; S]) -> Window<'static, N> {

        if S % W != 0 {
            panic!("Invalid image dimensions");
        }

        Window {
            win : array,
            width : W,
            offset : (0, 0),
            win_sz : (S / W, W)
        }
    }

}

impl<N> Image<N>
where
    N : Scalar + Copy + Any + 'static + Pixel
{
    pub fn depth(&self) -> Depth {
        N::depth()
    }
}

impl<N> Image<N>
where
    N : Scalar + Copy + Any + 'static
{

    pub fn pixel_is<T>(&self) -> bool
    where
        T : 'static
    {
        (&self[(0usize, 0usize)] as &dyn Any).is::<T>()
    }
}

impl<'a, N> Window<'a, N>
where
    N : Scalar + Copy + Any + Pixel + 'static
{

    pub fn depth(&self) -> Depth {
        N::depth()
    }

}

impl<'a, N> Window<'a, N>
where
    N : Scalar + Copy + Any + 'static
{

    pub fn pixel_is<T>(&self) -> bool
    where
        T : 'static
    {
        (&self[(0usize, 0usize)] as &dyn Any).is::<T>()
    }
}

impl<'a, N> WindowMut<'a, N>
where
    N : Scalar + Copy + Any + Pixel + 'static
{

    pub fn depth(&self) -> Depth {
        N::depth()
    }

}

impl<'a, N> WindowMut<'a, N>
where
    N : Scalar + Copy + Any + 'static
{

    // Joins two mutable windows, as long as their slices are contiguous in memory.
    pub fn join(a : WindowMut<'a, N>, b : WindowMut<'a, N>) -> Option<WindowMut<'a, N>> {
        unimplemented!()
    }

    /// Splits this mutable window horizontally
    pub fn split_at(mut self, row : usize) -> (WindowMut<'a, N>, WindowMut<'a, N>) {
        assert!(self.win.len() % (self.width * row) == 0);
        let offset = self.offset;
        let win_sz = self.win_sz;
        let width = self.width;
        let (win1, win2) = self.win.split_at_mut(self.width * row);

        let w1 = WindowMut {
            win : win1,
            width,
            offset,
            win_sz : (row, win_sz.1),
        };
        let w2 = WindowMut {
            win : win2,
            width,
            offset : (offset.0 + row, offset.1),
            win_sz : (win_sz.0 - row, win_sz.1),
        };
        assert!(w1.win.len() == w1.win_sz.0 * w1.win_sz.1);
        assert!(w2.win.len() == w2.win_sz.0 * w2.win_sz.1);
        (w1, w2)
    }

    pub fn pixel_is<T>(&self) -> bool
    where
        T : 'static
    {
        (&self[(0usize, 0usize)] as &dyn Any).is::<T>()
    }
}

impl<'a, N> Window<'a, N>
where
    N : Scalar + Copy
{

    // pub fn original_size(&self) -> (usize, usize) {
    //    (self.win.len() / self.width, self.width)
    // }

    pub fn rect(&self) -> (usize, usize, usize, usize) {
        let off = self.offset();
        let sz = self.shape();
        (off.0, off.1, sz.0, sz.1)
    }

    pub fn center(&self) -> (usize, usize) {
        let rect = self.rect();
        (rect.0 + rect.2 / 2, rect.1 + rect.3 / 2)
    }

    pub fn get(&self, index : (usize, usize)) -> Option<&N> {
        if index.0 < self.height() && index.1 < self.width() {
            unsafe { Some(self.get_unchecked(index)) }
        } else {
            None
        }
    }

    pub unsafe fn get_unchecked(&self, index : (usize, usize)) -> &N {
        let off_ix = (self.offset.0 + index.0, self.offset.1 + index.1);
        let (limit_row, limit_col) = (self.offset.0 + self.win_sz.0, self.offset.1 + self.win_sz.1);
        unsafe { self.win.get_unchecked(index::linear_index(off_ix, self.original_size().1)) }
    }

    pub fn offset_ptr(&self) -> *const N {
        self.get((0, 0)).unwrap() as *const N
    }

    /*/// Converts this image from the range[0, 1] (floating-point) to the quantized range defined by max
    pub fn quantize<M : Scalar + >(&self, mut out : WindowMut<'_, M>)
    where
        N : Copy + Mul<Output=N> + num_traits::float::Float + num_traits::cast::AsPrimitive<M> + Scalar,
        M : num_traits::cast::AsPrimitive<N> + Copy + num_traits::Bounded
    {
        let max : N = M::max_value().as_();
        for i in 0..out.height() {
            for j in 0..out.width() {
                out[(i, j)] = (self[(i, j)] * max).min(max).as_();
            }
        }
    }*/

    /*/// Converts this image to the range [0, 1] (floating-point) by dividing by its maximum
    /// attainable value.
    pub fn smoothen<M : Scalar + >(&self, mut out : WindowMut<'_, M>)
    where
        N : num_traits::Bounded + Copy + num_traits::cast::AsPrimitive<M>,
        M : Copy + Div<Output=M>
    {
        let max : M = N::max_value().as_();
        for i in 0..out.height() {
            for j in 0..out.width() {
                out[(i, j)] = (self[(i, j)]).as_() / max;
            }
        }
    }*/

    /// The cartesian index is defined as (img.height() - pt[1], pt[0]).
    /// It indexes the image by imposing the cartesian analytical plane over it,
    /// an converting it as specified.
    pub fn cartesian_index<T>(&self, pt : Vector2<T>) -> &N
    where
        usize : From<T>,
        T : Copy,
        N : Mul<Output=N> + MulAssign + Copy
    {
        let ix = (self.height() - usize::from(pt[1]), usize::from(pt[0]));
        &self[ix]
    }

    pub fn offset(&self) -> (usize, usize) {
        self.offset
    }

    pub fn is_full(&'a self) -> bool {
        self.original_size() == self.win_sz
    }
    
    /*pub fn shape(&self) -> (usize, usize) {
        self.win_sz
    }

    pub fn width(&self) -> usize {
        self.shape().0
    }

    pub fn height(&self) -> usize {
        self.shape().1
    }*/

    /// Returns either the given sub window, or trim it to the window borders and return a smaller but also valid window
    pub fn largest_valid_sub_window(&'a self, offset : (usize, usize), dims : (usize, usize)) -> Option<Window<'a, N>> {
        let diff_h = offset.0 as i64 + dims.0 as i64 - self.height() as i64;
        let diff_w = offset.1 as i64 + dims.1 as i64 - self.width() as i64;
        let valid_dims = (dims.0 - diff_h.max(0) as usize, dims.1 - diff_w.max(0) as usize);
        self.sub_window(offset, valid_dims)
    }

    pub fn sub_window(&'a self, offset : (usize, usize), dims : (usize, usize)) -> Option<Window<'a, N>> {
        assert_nonzero(dims);
        let new_offset = (self.offset.0 + offset.0, self.offset.1 + offset.1);
        if new_offset.0 + dims.0 <= self.original_size().0 && new_offset.1 + dims.1 <= self.original_size().1 {
            Some(Self {
                win : self.win,
                offset : new_offset,
                width : self.width,
                // transposed : self.transposed,
                win_sz : dims
            })
        } else {
            // println!("Requested offset : {:?}; Requested dims : {:?}; Original image size : {:?}", offset, dims, self.original_size());
            None
        }
    }

}

fn shrink_to_divisor(mut n : usize, by : usize) -> Option<usize> {
    if n > by {
        while n % by != 0 {
            n -= 1;
        }
        Some(n)
    } else {
        None
    }
}

impl<'a, N> Window<'a, N>
where
    N : Scalar + Copy
{

    pub fn row(&self, ix : usize) -> Option<&[N]> {
        if ix >= self.win_sz.0 {
            return None;
        }
        let stride = self.original_size().1;
        let tl = self.offset.0 * stride + self.offset.1;
        let start = tl + ix*stride;
        Some(&self.win[start..(start+self.win_sz.1)])
    }

    pub fn linear_index(&self, ix : usize) -> &N {
        assert!(ix < self.width() * self.height());
        let (row, col) = (ix / self.width(), ix % self.width());
        unsafe { &*self.as_ptr().offset((self.width * row + col) as isize) as &N }
    }

    /// Iterate over windows of the given size. This iterator consumes the original window
    /// so that we can implement windows(.) for Image by using move semantics, without
    /// requiring the user to call full_windows(.).
    pub fn windows(&self, sz : (usize, usize)) -> impl Iterator<Item=Window<'a, N>> {
        assert_nonzero(sz);
        let (step_v, step_h) = sz;
        if sz.0 >= self.win_sz.0 || sz.1 >= self.win_sz.1 {
            panic!("Child window size bigger than parent window size");
        }
        if self.height() % sz.0 != 0 || self.width() % sz.1 != 0 {
            panic!("Image size should be a multiple of window size (Required window {:?} over parent window {:?})", sz, self.win_sz);
        }
        let offset = self.offset;
        WindowIterator::<'a, N> {
            source : self.clone(),
            size : sz,
            curr_pos : offset,
            step_v,
            step_h
        }
    }

    // Returns image corners with the given dimensions.
    pub fn corners(&'a self, height : usize, width : usize) -> Option<[Window<'a, N>; 4]> {
        let right = self.width() - width;
        let bottom = self.height() - height;
        let sz = (height, width);
        let tl = self.sub_window((0, 0), sz)?;
        let tr = self.sub_window((0, right), sz)?;
        let bl = self.sub_window((bottom, 0), sz)?;
        let br = self.sub_window((bottom, right), sz)?;
        Some([tl, tr, bl, br])
    }

    // Returns image sides (without corners or same dimension).
    pub fn sides(&'a self, height : usize, width : usize) -> Option<[Window<'a, N>; 4]> {
        let right = self.width() - width;
        let bottom = self.height() - height;
        let vert_sz = (height, self.width() - 2*width);
        let horiz_sz = (self.height() - 2*height, self.width());
        let top = self.sub_window((0, width), vert_sz)?;
        let right = self.sub_window((height, right), horiz_sz)?;
        let bottom = self.sub_window((bottom, width), vert_sz)?;
        let left = self.sub_window((height, 0), horiz_sz)?;
        Some([top, right, bottom, left])
    }

    pub fn shrink_to_subsample2(&'a self, row_by : usize, col_by : usize) -> Option<Window<'a, N>> {
        let height = shrink_to_divisor(self.height(), row_by)?;
        let width = shrink_to_divisor(self.width(), col_by)?;
        self.sub_window((0, 0), (height, width))
    }

    pub fn shrink_to_subsample(&'a self, by : usize) -> Option<Window<'a, N>> {
        let height = shrink_to_divisor(self.height(), by)?;
        let width = shrink_to_divisor(self.width(), by)?;
        self.sub_window((0, 0), (height, width))
    }

    pub fn byte_stride(&self) -> usize {
        std::mem::size_of::<N>() * self.width
    }

    pub fn as_ptr(&self) -> *const N {
        // self.win.as_ptr()
        &self[(0usize,0usize)] as *const _
    }

    pub fn full_slice(&self) -> &[N] {
        &self.win[..]
    }

    pub fn shape(&self) -> (usize, usize) {
        // self.win.shape()
        self.win_sz
    }

    pub fn area(&self) -> usize {
        self.width() * self.height()
    }

    pub fn width(&self) -> usize {
        // self.win.ncols()
        self.win_sz.1
    }

    pub fn height(&self) -> usize {
        // self.win.nrows()
        self.win_sz.0
    }

    // same as self.area
    pub fn len(&self) -> usize {
        self.win_sz.0 * self.win_sz.1
    }

    pub fn rows<'b>(&'b self) -> impl Iterator<Item=&'a [N]> + Clone + 'b {
        let stride = self.original_size().1;
        let tl = self.offset.0 * stride + self.offset.1;
        (0..self.win_sz.0).map(move |i| {
            let start = tl + i*stride;
            &self.win[start..(start+self.win_sz.1)]
        })
    }

    /// Returns iterator over (subsampled row index, subsampled col index, pixel color).
    /// Panics if L is an unsigend integer type that cannot represent one of the dimensions
    /// of the image precisely.
    pub fn labeled_pixels<L, E>(&'a self, px_spacing : usize) -> impl Iterator<Item=(L, L, N)> +'a + Clone
    where
        L : TryFrom<usize, Error=E> + Div<Output=L> + Mul<Output=L> + Rem<Output=L> + Clone + Copy + 'static,
        E : Debug,
        Range<L> : Iterator<Item=L>
    {
        let spacing = L::try_from(px_spacing).unwrap();
        let w = (L::try_from(self.width()).unwrap() / spacing );
        let h = (L::try_from(self.height()).unwrap() / spacing );
        let range = Range { start : L::try_from(0usize).unwrap(), end : (w*h) };
        range
            .zip(self.pixels(px_spacing))
            .map(move |(ix, px)| {
                let (r, c) = (ix / w, ix % w);
                (r, c, *px)
            })
    }

    pub fn pixels_across_line<'b>(&'b self, src : (usize, usize), dst : (usize, usize)) -> impl Iterator<Item=&N> + 'b {
        let (nrow, ncol) = self.shape();
        coords_across_line(src, dst, self.shape()).map(move |pt| &self[pt] )
    }

    /// Iterate over all image pixels if spacing=1; or over pixels spaced
    /// horizontally and verticallly by spacing. Iteration proceeds row-wise.
    pub fn pixels<'b>(&'b self, spacing : usize) -> impl Iterator<Item=&'a N> + Clone {
        assert!(spacing > 0, "Spacing should be at least one");
        assert!(self.width() % spacing == 0 && self.height() % spacing == 0, "Spacing should be integer divisor of width and height");
        iterate_row_wise(self.win, self.offset, self.win_sz, self.original_size(), spacing).step_by(spacing)
    }

    pub fn labeled_neighborhoods(&'a self) -> impl Iterator<Item=((usize, usize), PixelNeighborhood<N>)> + 'a {
        (0..(self.width()*self.height()))
            .map(move |ix| {
                let pos = (ix / self.width(), ix % self.width());
                let neigh = self.neighboring_pixels(pos).unwrap();
                (pos, neigh)
            })
    }

    pub fn neighboring_pixels(&'a self, pos : (usize, usize)) -> Option<PixelNeighborhood<N>> {

        if pos.0 > 0 && pos.1 > 0 && pos.0 < self.height() - 1 && pos.1 < self.width() - 1 {

            // Center
            let (tl, tr) = (top_left(pos), top_right(pos));
            let (bl, br) = (bottom_left(pos), bottom_right(pos));
            let (t, b) = (top(pos), bottom(pos));
            let (l, r) = (left(pos), right(pos));
            Some(PixelNeighborhood::Full([
                self[tl], self[t], self[tr],
                self[l], self[r],
                self[bl], self[b], self[br]
            ], 0))
        } else if pos.0 == 0 {

            // Top row

            if pos.1 == 0 {
                // Top row left corner
                let (b, br, r) = (bottom(pos), bottom_right(pos), right(pos));
                Some(PixelNeighborhood::Corner([self[b], self[br], self[r]], 0))
            } else if pos.1 == self.width()-1 {
                // Top right right corner
                let (l, bl, b) = (left(pos), bottom_left(pos), bottom(pos));
                Some(PixelNeighborhood::Corner([self[l], self[bl], self[b]], 0))
            } else {
                // Top row NOT corner
                let (l, r, bl, b, br) = (left(pos), right(pos), bottom_left(pos), bottom(pos), bottom_right(pos));
                Some(PixelNeighborhood::Edge([self[l], self[r], self[bl], self[b], self[br]], 0))
            }
        } else if pos.0 == self.height()-1 && pos.1 < self.width() {

            // Bottom row

            if pos.1 == 0 {
                // Bottom row left corner
                let (t, tr, r) = (top(pos), top_right(pos), right(pos));
                Some(PixelNeighborhood::Corner([self[t], self[tr], self[r]], 0))
            } else if pos.1 == self.width()-1 {
                // Bottom row right corner
                let (t, tl, l) = (top(pos), top_left(pos), left(pos));
                Some(PixelNeighborhood::Corner([self[t], self[tl], self[l]], 0))
            } else {
                // Bottom row NOT corner
                let (l, r, tl, t, tr) = (left(pos), right(pos), top_left(pos), top(pos), top_right(pos));
                Some(PixelNeighborhood::Edge([self[l], self[r], self[tl], self[t], self[tr]], 0))
            }

        } else if pos.1 == 0 && pos.0 < self.height() {

            //  Left column (except corner pixels, matched above)
            let (t, b, tr, r, br) = (top(pos), bottom(pos), top_right(pos), right(pos), bottom_right(pos));
            Some(PixelNeighborhood::Edge([self[t], self[b], self[tr], self[r], self[br]], 0))

        } else if pos.1 == self.width()-1 && pos.0 < self.height() {

            // Right column (except corner pixels, matched above)
            let (t, b, tl, l, bl) = (top(pos), bottom(pos), top_left(pos), left(pos), bottom_left(pos));
            Some(PixelNeighborhood::Edge([self[t], self[b], self[tl], self[l], self[bl]], 0))

        } else {
            // Outside area
            None
        }
    }



}

impl<'a, N> Window<'a, N>
where
    N : Scalar + Mul<Output=N> + MulAssign + Copy + Copy + Any
{

    /*pub unsafe fn get_unchecked_u16(&self, index : (u16, u16)) -> &N {
        let off_ix = (self.offset.0 + index.0, self.offset.1 + index.1);
        let (limit_row, limit_col) = (self.offset.0 + self.win_sz.0, self.offset.1 + self.win_sz.1);
        unsafe { self.win.get_unchecked(index::linear_index(off_ix, self.original_size().1)) }
    }*/

    // pub unsafe fn linear_index(&self, ix : usize) -> N {
    //    self.win.get_unchecked(ix)
    // }

    /// Returns a set of linear indices to the underlying slice that can
    /// be used to iterate over this window.
    pub fn linear_indices(&self, spacing : usize) -> Vec<usize> {
        unimplemented!()
    }

    pub fn subsampled_indices(&self, spacing : usize) -> Vec<(usize, usize)> {
        let mut ixs = Vec::new();
        for i in 0..(self.height() / spacing) {
            for j in 0..(self.width() / spacing) {
                ixs.push((i, j));
            }
        }
        ixs
    }

    pub fn slice_len(&self) -> usize {
        self.win.len()
    }

    /// Returns a minimum number of overlapping windows of the informed shape,
    /// that completely cover the current window.
    pub fn minimum_inner_windows(&'a self, shape : (usize, usize)) -> Vec<Window<'a, N>> {
        let mut sub_wins = Vec::new();

        sub_wins
    }

    pub fn four_distant_window_neighborhood(
        &'a self,
        center_tl : (usize, usize),
        win_sz : (usize, usize),
        neigh_ext : usize,
        dist : usize
    ) -> Option<WindowNeighborhood<'a, N>> {
        let outside_bounds = center_tl.0 < win_sz.0 + dist ||
            center_tl.0 > self.height().checked_sub(win_sz.0 + dist)? ||
            center_tl.1 > self.width().checked_sub(win_sz.1 + dist)? ||
            center_tl.1 < win_sz.1 + dist;
        if outside_bounds {
            return None;
        }
        let vert_sz = (neigh_ext, win_sz.1);
        let horiz_sz = (win_sz.0, neigh_ext);
        Some(WindowNeighborhood {
            center : self.sub_window(center_tl, win_sz)?,
            left : self.sub_window((center_tl.0, center_tl.1 - win_sz.1 - dist), horiz_sz)?,
            top : self.sub_window((center_tl.0 - win_sz.0 - dist, center_tl.1), vert_sz)?,
            right : self.sub_window((center_tl.0, center_tl.1 + win_sz.1 + dist), horiz_sz)?,
            bottom : self.sub_window((center_tl.0 + win_sz.0 + dist, center_tl.1), vert_sz)?
        })
    }

    // Get the four windows at top, left, bottom and right of a center window
    // identified by its top-left position, where top/bottom neighboring windows are
    // neigh_ext x win_sz.1 and left/right neighboring windows are win_sz.0 x win_sz.0
    pub fn four_window_neighborhood(
        &'a self,
        center_tl : (usize, usize),
        win_sz : (usize, usize),
        neigh_ext : usize
    ) -> Option<WindowNeighborhood<'a, N>> {
        let outside_bounds = center_tl.0 < win_sz.0 ||
            center_tl.0 > self.height().checked_sub(win_sz.0)? ||
            center_tl.1 > self.width().checked_sub(win_sz.1)? ||
            center_tl.1 < win_sz.1;
        if outside_bounds {
            return None;
        }
        let vert_sz = (neigh_ext, win_sz.1);
        let horiz_sz = (win_sz.0, neigh_ext);
        Some(WindowNeighborhood {
            center : self.sub_window(center_tl, win_sz)?,
            left : self.sub_window((center_tl.0, center_tl.1 - win_sz.1), horiz_sz)?,
            top : self.sub_window((center_tl.0 - win_sz.0, center_tl.1), vert_sz)?,
            right : self.sub_window((center_tl.0, center_tl.1 + win_sz.1), horiz_sz)?,
            bottom : self.sub_window((center_tl.0 + win_sz.0, center_tl.1), vert_sz)?
        })
    }

    pub fn sub_from_slice(
        src : &'a [N],
        original_width : usize,
        offset : (usize, usize),
        dims : (usize, usize)
    ) -> Option<Self> {
        let nrows = src.len() / original_width;
        if offset.0 + dims.0 <= nrows && offset.1 + dims.1 <= original_width {
            Some(Self {
                win : src,
                width : original_width,
                offset,
                win_sz : dims
            })
        } else {
            None
        }
    }

    /// Creates a window that cover the whole slice src, assuming it represents a square image.
    pub fn from_square_slice(src : &'a [N]) -> Option<Self> {
        Self::from_slice(src, (src.len() as f64).sqrt() as usize)
    }
    
    /*pub fn from_slice(
        src : &'a [N], 
        img_shape : (usize, usize), 
        offset : (usize, usize), 
        win_shape : (usize, usize), 
        transposed : bool
    ) {
    
    }*/
    
    /// Creates a window that cover the whole slice src. The slice is assumed to be in
    /// row-major order, but matrices are assumed to be 
    pub fn from_slice(src : &'a [N], ncols : usize) -> Option<Self> {
        if src.len() % ncols != 0 {
            return None;
        }
        let nrows = src.len() / ncols;
        Some(Self{
            // win : DMatrixSlice::from_slice_generic(src, Dynamic::new(nrows),
            // Dynamic::new(ncols)),
            win : src,
            offset : (0, 0),
            width : ncols,
            win_sz : (nrows, ncols),
            // transposed : true
        })
    }
    
    /*pub fn linear_index(&self, ix : usize) -> &N {
        let offset = self.original_size().1 * offset.0 + offset.1;
        let row = ix / self.original_size().1;
        unsafe{ self.win.get_unchecked(offset + ix) }
    }*/

    /// Splits this window into equally-sized subwindows, iterating row-wise over the blocks.
    pub fn equivalent_windows(self, num_rows : usize, num_cols : usize) -> impl Iterator<Item=Window<'a, N>> {
        assert!(self.height() % num_rows == 0 && self.width() % num_cols == 0);
        self.clone().windows((self.height() / num_rows, self.width() / num_cols))
    }

    pub fn column(&'a self, ix : usize) -> Option<impl Iterator<Item=N> + 'a > {
        if ix < self.width() {
            Some(self.rows().map(move |row| row[ix] ))
        } else {
            None
        }
    }

    /// Iterates over pairs of pixels within a row, carrying the column index of the left element at first position
    pub fn horizontal_pixel_pairs(&'a self, row : usize, comp_dist : usize) -> Option<impl Iterator<Item=(usize, (&'a N, &'a N))>> {
        Some(crate::raster::horizontal_row_iterator(self.row(row)?, comp_dist))
    }

    pub fn vertical_pixel_pairs(&'a self, col : usize, comp_dist : usize) -> Option<impl Iterator<Item=(usize, (&'a N, &'a N))>> {
        if col >= self.win_sz.1 {
            return None;
        }
        Some(crate::raster::vertical_col_iterator(self.rows(), comp_dist, col))
    }

    /// Iterate over one of the lower-triangular diagonals the image, starting at given row.
    /// If to_right is passed, iterate from the top-left to bottom-right corner. If not, iterate from the
    /// top-right to bottom-left corner.
    pub fn lower_to_right_diagonal_pixel_pairs(
        &'a self,
        row : usize,
        comp_dist : usize,
    ) -> Option<impl Iterator<Item=((usize, usize), (&'a N, &'a N))>> {
        if row < self.height() {
            Some(crate::raster::diagonal_right_row_iterator(self.rows(), comp_dist, (row, 0)))
        } else {
            None
        }
    }

    pub fn upper_to_right_diagonal_pixel_pairs(
        &'a self,
        col : usize,
        comp_dist : usize,
    ) -> Option<impl Iterator<Item=((usize, usize), (&'a N, &'a N))>> {
        if col < self.width() {
            Some(crate::raster::diagonal_right_row_iterator(self.rows(), comp_dist, (0, col)))
        } else {
            None
        }
    }

    pub fn lower_to_left_diagonal_pixel_pairs(
        &'a self,
        row : usize,
        comp_dist : usize,
    ) -> Option<impl Iterator<Item=((usize, usize), (&'a N, &'a N))>> {
        if row < self.height() {
            Some(crate::raster::diagonal_left_row_iterator(self.rows(), comp_dist, (row, self.width()-1)))
        } else {
            None
        }
    }

    pub fn upper_to_left_diagonal_pixel_pairs(
        &'a self,
        col : usize,
        comp_dist : usize,
    ) -> Option<impl Iterator<Item=((usize, usize), (&'a N, &'a N))>> {
        if col < self.width() {
            Some(crate::raster::diagonal_left_row_iterator(self.rows(), comp_dist, (0, col)))
        } else {
            None
        }
    }

    pub fn to_right_diagonal_pixel_pairs(
        &'a self,
        comp_dist : usize
    ) -> impl Iterator<Item=((usize, usize), (&'a N, &'a N))> {
        (0..self.height()).step_by(comp_dist)
            .map(move |r| self.lower_to_right_diagonal_pixel_pairs(r, comp_dist).unwrap() )
            .flatten()
            .chain((0..self.width()).step_by(comp_dist)
                .map(move |c| self.upper_to_right_diagonal_pixel_pairs(c, comp_dist).unwrap() )
                .flatten()
            )
    }

    pub fn to_left_diagonal_pixel_pairs(
        &'a self,
        comp_dist : usize
    ) -> impl Iterator<Item=((usize, usize), (&'a N, &'a N))> {
        (0..self.height()).step_by(comp_dist)
            .map(move |r| self.lower_to_left_diagonal_pixel_pairs(r, comp_dist).unwrap() )
            .flatten()
            .chain((0..self.width()).step_by(comp_dist)
                .map(move |c| self.upper_to_left_diagonal_pixel_pairs(c, comp_dist).unwrap() )
                .flatten()
            )
    }

    pub fn rect_pixels(&'a self, rect : (usize, usize, usize, usize)) -> impl Iterator<Item=N> + Clone + 'a {
        let row_iter = rect.0..(rect.0 + rect.2);
        let col_iter = rect.1..(rect.1 + rect.3);

        col_iter.clone().map(move |c| self[(rect.0, c)] )
            .chain(row_iter.clone().map(move |r| self[(r, rect.1 + rect.3 - 1 )] ) )
            .chain(col_iter.clone().rev().map(move |c| self[(rect.0 + rect.2 - 1, c)] ) )
            .chain(row_iter.clone().rev().map(move |r| self[(r, rect.1 )] ))
    }

    /// Iterate over image pixels, expanding from a given location, until any image border is found.
    /// Iteration happens clock-wise from the seed pixel. Indices are at the original image scale.
    pub fn expanding_pixels(
        &self,
        seed : (usize, usize),
        px_spacing : usize
    ) -> impl Iterator<Item=((usize, usize), &N)> + Clone {
        assert!(seed.0 < self.height() && seed.1 < self.width());
        let min_dist = seed.0.min(self.height() - seed.0).min(seed.1).min(self.width() - seed.1);
        (px_spacing..min_dist).map(move |abs_dist| {
            let left_col = seed.1 - abs_dist;
            let top_row = seed.0 - abs_dist;
            let right_col = seed.1 + abs_dist;
            let bottom_row = seed.0 + abs_dist;
            let row_range = (seed.0.saturating_sub(abs_dist-px_spacing)..((seed.0+abs_dist).min(self.height()))).step_by(px_spacing);
            let col_range = (seed.1.saturating_sub(abs_dist)..((seed.1+abs_dist+px_spacing).min(self.width()))).step_by(px_spacing);
            let top_iter = col_range.clone().map(move |c| ((top_row, c), &self[(top_row, c)] ) );
            let right_iter = row_range.clone().map(move |r| ((r, right_col), &self[(r, right_col)] ) );
            let bottom_iter = col_range.rev().map(move |c| ((bottom_row, c), &self[(bottom_row, c)] ) );
            let left_iter = row_range.rev().map(move |r| ((r, left_col), &self[(r, left_col)] ) );

                // Skip first element of each iterator because it is already contained in the last one.
                top_iter.chain(right_iter.skip(1)).chain(bottom_iter.skip(1)).chain(left_iter.skip(1))
        }).flatten()
    }

    // Returns the most representative k-colors of the image using K-means
    // pub fn colors(&self, px_spacing : usize, n_colors : usize) -> Vec<u8> {
    // }

    // Returns a thin vertical window over a given col. Prototype for impl Index<(Range<usize>, usize)>
    pub fn sub_col(&self, rows : Range<usize>, col : usize) -> Option<Window<'_, N>> {
        let height = rows.end - rows.start;
        self.sub_window((rows.start, col), (height, 1))
    }

    // Returns a thin horizontal window over a given row. Prototype for impl Index<(usize, Range<usize>)>
    pub fn sub_row(&self, row : usize, cols : Range<usize>) -> Option<Window<'_, N>> {
        let width = cols.end - cols.start;
        self.sub_window((row, cols.start), (1, width))
    }

    pub fn clone_owned(&self) -> Image<N>
    where
        N : Copy + Default + Zero
    {
        let mut buf = Vec::new();
        self.rows().for_each(|row| buf.extend(row.iter().cloned()) );
        Image::from_vec(buf, self.win_sz.1)
    }

    /*pub fn row_slices(&'a self) -> Vec<&'a [N]> {
        let mut rows = Vec::new();
        for r in (self.offset.0)..(self.offset.0+self.win_sz.0) {
            let begin = self.win_sz.1*r + self.offset.1;
            rows.push(&self.src[begin..begin+self.win_sz.1]);
        }
        rows
    }*/
    
}

/*pub struct PackedIterator<'a, T>
where
    T : Scalar + Debug + Copy
{
    win : Window<'a, T>,
    ix : usize,
    packed_per_row : usize,
    n_packed : usize
}

impl<'a> Iterator for PackedIterator<'a, u8> {

    type Item = AutoSimd<[u8; 32]>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ix == self.n_packed  {
            None
        } else {
            let row = self.win.row(self.ix / self.n_packed).unwrap();
            let packed = &row[self.ix % self.n_packed];
            self.ix += 1;
            Some(AutoSimd::try_from(packed).unwrap())
        }
    }
}*/

// Holds an array of neighboring pixels and the current index at the iterator.
pub enum PixelNeighborhood<N>
where
    N : Copy
{

    Corner([N; 3], usize),

    Edge([N; 5], usize),

    Full([N; 8], usize)

}

fn walk_neighborhood<N>(pxs : &[N], pos : &mut usize) -> Option<N>
where
    N : Copy
{
    let ans = pxs.get(*pos).copied();
    if ans.is_some() {
        *pos += 1;
    }
    ans
}

impl<N> Iterator for PixelNeighborhood<N>
where
    N : Copy
{

    type Item = N;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Corner(pxs, ref mut pos) => {
                walk_neighborhood(&pxs[..], pos)
            },
            Self::Edge(pxs, pos) => {
                walk_neighborhood(&pxs[..], pos)
            },
            Self::Full(pxs, pos) => {
                walk_neighborhood(&pxs[..], pos)
            },
        }
    }

}

fn top_left(pos : (usize, usize)) -> (usize, usize) {
    (pos.0 - 1, pos.1 - 1)
}

fn top_right(pos : (usize, usize)) -> (usize, usize) {
    (pos.0 - 1, pos.1 + 1)
}

fn bottom_left(pos : (usize, usize)) -> (usize, usize) {
    (pos.0 + 1, pos.1 - 1)
}

fn bottom_right(pos : (usize, usize)) -> (usize, usize) {
    (pos.0 + 1, pos.1 + 1)
}

fn top(pos : (usize, usize)) -> (usize, usize) {
    (pos.0 - 1, pos.1)
}

fn bottom(pos : (usize, usize)) -> (usize, usize) {
    (pos.0 + 1, pos.1)
}

fn left(pos : (usize, usize)) -> (usize, usize) {
    (pos.0, pos.1 - 1)
}

fn right(pos : (usize, usize)) -> (usize, usize) {
    (pos.0, pos.1 + 1)
}

pub fn labels<L, E>((height, width) : (usize, usize), px_spacing : usize) -> impl Iterator<Item=(L, L)> + Clone
where
    L : TryFrom<usize, Error=E> + Div<Output=L> + Mul<Output=L> + Rem<Output=L> + Clone + Copy + 'static,
    E : Debug,
    Range<L> : Iterator<Item=L>
{
    let spacing = L::try_from(px_spacing).unwrap();
    let w = (L::try_from(width).unwrap() / spacing );
    let h = (L::try_from(height).unwrap() / spacing );
    let range = Range { start : L::try_from(0usize).unwrap(), end : (w*h) };
    range.map(move |ix| (ix / w, ix % w) )
}

impl<'a> Window<'a, u8> {

    #[cfg(feature="opencv")]
    pub fn resize_mut(&self, other : WindowMut<'_, u8>) {
        use opencv::{core, imgproc};
        use opencv::prelude::MatTraitManual;
        let this_shape = self.shape();
        let other_shape = other.shape();
        let src : core::Mat = self.into();
        let mut dst : core::Mat = other.into();
        let dst_sz = dst.size().unwrap();
        imgproc::resize(&src, &mut dst, dst_sz, 0.0, 0.0, imgproc::INTER_NEAREST);
    }

    #[cfg(feature="opencv")]
    pub fn copy_scale_mut_within(&self, src : (usize, usize), src_sz : (usize, usize), dst : (usize, usize), dst_sz : (usize, usize)) {

        use opencv::{core, imgproc};

        let ratio_row = (dst_sz.0 / src_sz.0) as f64;
        let ratio_col = (dst_sz.1 / src_sz.1) as f64;
        if let Some(src) = self.sub_window(src, src_sz) {
            if let Some(dst) = self.sub_window(dst, dst_sz) {
                let mut src : core::Mat = src.into();
                let mut dst : core::Mat = dst.into();
                imgproc::resize(&src, &mut dst, core::Size2i::default(), ratio_col, ratio_row, imgproc::INTER_LINEAR);
            }
        }
    }

    /// Gets pos or the nearest pixel to it that satisfies a condition.
    pub fn nearest_matching(
        &self,
        seed : (usize, usize),
        px_spacing : usize,
        f : impl Fn(u8)->bool, max_dist : usize
    ) -> Option<(usize, usize)> {
        if f(self[seed]) {
            Some(seed)
        } else {
            self.expanding_pixels(seed, px_spacing)
                .take_while(|(pos, _)| {
                    let row_close_seed = ((pos.0 as i64 - seed.0 as i64).abs() as usize) < max_dist;
                    let col_close_seed = ((pos.1 as i64 - seed.1 as i64).abs() as usize) < max_dist;
                    row_close_seed && col_close_seed
                })
                .find(|(_, px)| f(**px) )
                .map(|(pos, _)| pos )
        }
    }

    pub fn color_at_labels(&'a self, ixs : impl Iterator<Item=(usize, usize)> + 'a) -> impl Iterator<Item=u8> + 'a {
        ixs.map(move |ix| self[ix] )
    }

    pub fn nonzero_pixels(&'a self, px_spacing : usize) -> impl Iterator<Item=&'a u8> +'a + Clone {
        self.pixels(px_spacing).filter(|px| **px > 0 )
    }

    pub fn masked_pixels(&'a self, mask : &'a Window<'_, u8>) -> impl Iterator<Item=&'a u8> +'a + Clone {
        mask.nonzero_labeled_pixels(1).map(move |(r, c, _)| &self[(r, c)] )
    }

    pub fn nonzero_labeled_pixels(&'a self, px_spacing : usize) -> impl Iterator<Item=(usize, usize, u8)> +'a + Clone {
        self.labeled_pixels(px_spacing).filter(|(_, _, px)| *px > 0 )
    }

    /// Iterate over pixels, as long as they are non-zero at the mask window.
    pub fn masked_labeled_pixels(&'a self, mask : &'a Window<'_, u8>) -> impl Iterator<Item=(usize, usize, u8)> +'a + Clone {
        mask.nonzero_labeled_pixels(1).map(move |(r, c, _)| (r, c, self[(r, c)]) )
    }

    /// Extract contiguous image regions of homogeneous color.
    pub fn patches(&self, px_spacing : usize) -> Vec<Patch> {
        /*let mut patches = Vec::new();
        color::full_color_patches(&mut patches, self, px_spacing as u16, ColorMode::Exact(0), color::ExpansionMode::Dense);
        patches*/
        unimplemented!()
    }

    /*pub fn binary_patches(&self, px_spacing : usize) -> Vec<BinaryPatch> {
        // TODO if we have a binary or a bit image with just a few classes,
        // there is no need for KMeans. Just use the allocations.
        // let label_img = segmentation::segment_colors_to_image(self, px_spacing, n_colors);
        color::binary_patches(self, px_spacing)
    }*/

    pub fn mean(&self, n_pxs : usize) -> Option<u8> {
        Some((self.shrink_to_subsample(n_pxs)?.pixels(n_pxs).map(|px| *px as u64 ).sum::<u64>() / (self.width() * self.height()) as u64) as u8)
    }

    /// If higher, returns binary image with all pixels > thresh set to 255 and others set to 0;
    /// If !higher, returns binary image with pixels < thresh set to 255 and others set to 0.
    pub fn threshold_mut(&self, dst : &mut Image<u8>, thresh : u8, higher : bool) {
        assert!(self.shape() == dst.shape());

        #[cfg(feature="opencv")]
        {
            // crate::threshold::threshold_window(self, dst, thresh as f64, 255.0, higher);
        }

        for (src, mut dst) in self.pixels(1).zip(dst.full_window_mut().pixels_mut(1)) {
            if (!higher && *src < thresh) || (higher && *src > thresh) {
                *dst = 255;
            } else {
                *dst = 0;
            }
        }
    }

    /*/// Packed iterator over contigous byte regions of the image within
    /// rows. Image horizontal dimension should be divisible by the
    pub fn iter_packed(&self) -> impl Iterator<Item=AutoSimd<[u8; 32]>> + 'a {

        // Substitute 32 for the size of packed SIMD value.
        let packed_per_row = self.width() / 32;

        let n_packed = packed_per_row * self.height();
        PackedIterator {
            win : self.clone(),
            ix : 0,
            packed_per_row,
            n_packed
        }
    }*/
}

#[test]
fn window_iter() {
    let img : Window<'_, u8> = Window::from_square_slice(&[
        1, 1, 1, 1, 0, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 0, 1, 1, 1, 1,
    ]).unwrap();

    for win in img.windows((4,4)) {
        println!("Outer: {:?}", win);
        for win_inner in win.windows((2,2)) {
            println!("\tInner : {:?}", win_inner);
        }
        println!("")
    }
}

impl<N> Index<(usize, usize)> for Window<'_, N>
where
    N : Scalar + Copy
{

    type Output = N;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let off_ix = (self.offset.0 + index.0, self.offset.1 + index.1);
        let (limit_row, limit_col) = (self.offset.0 + self.win_sz.0, self.offset.1 + self.win_sz.1);
        if off_ix.0 < limit_row && off_ix.1 < limit_col {
            unsafe { self.win.get_unchecked(index::linear_index(off_ix, self.original_size().1)) }
        } else {
            panic!("Invalid window index: {:?}", index);
        }
    }
}

impl<N> Index<(u16, u16)> for Window<'_, N>
where
    N : Scalar + Copy
{

    type Output = N;

    fn index(&self, index: (u16, u16)) -> &Self::Output {
        self.index((index.0 as usize, index.1 as usize))
    }

}

/*impl<N> Index<(Range<usize>, usize)> for Window<'_, N>
where
    N : Scalar
{

    type Output = Self;

    fn index(&self, index: (Range<usize>, usize)) -> &Self::Output {

    }

}*/

#[cfg(feature="opencv")]
impl<N> opencv::core::ToInputArray for Image<N>
where
    N : Scalar + Copy + Default + Zero + Any
{

    fn input_array(&self) -> opencv::Result<opencv::core::_InputArray> {
        let out : opencv::core::Mat = (&*self).into();
        out.input_array()
    }

}

#[cfg(feature="opencv")]
impl<N> opencv::core::ToOutputArray for Image<N>
where
    N : Scalar + Copy + Default + Zero + Any
{

    fn output_array(&mut self) -> opencv::Result<opencv::core::_OutputArray> {
        let mut out : opencv::core::Mat = (&mut *self).into();
        out.output_array()
    }

}

#[cfg(feature="opencv")]
impl<N> opencv::core::ToInputArray for Window<'_, N>
where
    N : Scalar + Copy + Default + Zero + Any
{

    fn input_array(&self) -> opencv::Result<opencv::core::_InputArray> {
        let out : opencv::core::Mat = (self.clone()).into();
        out.input_array()
    }

}

#[cfg(feature="opencv")]
impl<N> opencv::core::ToOutputArray for WindowMut<'_, N>
where
    N : Scalar + Copy + Default + Zero + Any
{

    fn output_array(&mut self) -> opencv::Result<opencv::core::_OutputArray> {
        let mut out : opencv::core::Mat = (self).into();
        out.output_array()
    }

}

#[cfg(feature="opencv")]
pub fn median_blur(win : &Window<'_, u8>, output : WindowMut<'_, u8>, kernel : usize) {

    use opencv::{imgproc, core};

    let input : core::Mat = win.clone().into();
    let mut out : core::Mat = output.into();
    imgproc::median_blur(&input, &mut out, kernel as i32).unwrap();

}

#[cfg(feature="opencv")]
impl<N> From<core::Mat> for Image<N>
where
    N : Scalar + Copy + Default + Zero + Any + opencv::core::DataType
{

    fn from(m : core::Mat) -> Image<N> {

        use opencv::prelude::MatTraitManual;
        use opencv::prelude::MatTrait;
        // assert!(m.is_contiguous().unwrap());

        let sz = m.size().unwrap();
        let h = sz.height as usize;
        let w = sz.width as usize;
        let mut img = Image::<N>::new_constant(h, w, N::zero());
        for i in 0..h {
            for j in 0..w {
                img[(i, j)] = *m.at_2d::<N>(i as i32, j as i32).unwrap();
            }
        }
        img
    }

}

/// TODO mark as unsafe impl
#[cfg(feature="opencv")]
impl<N> Into<core::Mat> for &Image<N>
where
    N : Scalar + Copy + Default + Zero + Any
{

    fn into(self) -> core::Mat {
        let sub_slice = None;
        let stride = self.width;
        unsafe{ cvutils::slice_to_mat(&self.buf[..], stride, sub_slice) }
        // self.full_window().into()
    }
}

#[cfg(feature="opencv")]
impl<N> Into<core::Mat> for &mut Image<N>
where
    N : Scalar + Copy + Default + Zero + Any
{

    fn into(self) -> core::Mat {
        let sub_slice = None;
        let stride = self.width;
        unsafe{ cvutils::slice_to_mat(&self.buf[..], stride, sub_slice) }
        // self.full_window_mut().into()
    }
}

/// TODO mark as unsafe impl
#[cfg(feature="opencv")]
impl<N> Into<core::Mat> for Window<'_, N>
where
    N : Scalar + Copy + Default
{

    fn into(self) -> core::Mat {
        let sub_slice = Some((self.offset, self.win_sz));
        let stride = self.original_size().1;
        unsafe{ cvutils::slice_to_mat(self.win, stride, sub_slice) }
    }
}

#[cfg(feature="opencv")]
impl<N> Into<core::Mat> for &Window<'_, N>
where
    N : Scalar + Copy + Default
{

    fn into(self) -> core::Mat {
        let sub_slice = Some((self.offset, self.win_sz));
        let stride = self.original_size().1;
        unsafe{ cvutils::slice_to_mat(self.win, stride, sub_slice) }
    }
}

/// TODO mark as unsafe impl
#[cfg(feature="opencv")]
impl<N> Into<core::Mat> for WindowMut<'_, N>
where
    N : Scalar + Copy + Default
{

    fn into(self) -> core::Mat {
        let sub_slice = Some((self.offset, self.win_sz));
        let stride = self.original_size().1;
        unsafe{ cvutils::slice_to_mat(self.win, stride, sub_slice) }
    }
}

#[cfg(feature="opencv")]
impl<N> Into<core::Mat> for &mut WindowMut<'_, N>
where
    N : Scalar + Copy + Default
{

    fn into(self) -> core::Mat {
        let sub_slice = Some((self.offset, self.win_sz));
        let stride = self.original_size().1;
        unsafe{ cvutils::slice_to_mat(self.win, stride, sub_slice) }
    }
}

/*impl TryFrom<rhai::Dynamic> for Mark {

    type Err = ();

    fn try_from(d : rhai::Dynamic) -> Result<Self, ()> {

    }

}*/

/// Wraps a mutable slice, offering access to it as if it were a mutable image buffer.
/// The consequence of this approach is that mutable windows sharing the same underlying image
/// rows cannot be represented simultaneously without violating aliasing rules. Mutable windows
/// sharing only columns (but at disjoint row intervals), however, can be safely represented
/// via split_at(.). This requires that sub_window_mut takes windows by-value. Iteration, however,
/// is possible by iter_windows_mut() and apply_to_sub/apply_to_sub_sequence, which work with raw pointers
/// internally (thus any closures or code segments working with the iterable items cannot access the original window for
/// the lifetime of the mutable iterator items in the public API).
#[repr(C)]
#[derive(Debug)]
pub struct WindowMut<'a, N> 
where
    N : Scalar + Copy
{

    // Original image full slice.
    pub(crate) win : &'a mut [N],
    
    // Original image size.
    pub(crate) width : usize,

    // Window offset, with respect to the top-left point (row, col).
    pub(crate) offset : (usize, usize),
    
    // This window size.
    pub(crate) win_sz : (usize, usize),

}

impl<'a, N> WindowMut<'a, N>
where
    N : Scalar + Copy
{

    /*// Since mutable references are not allowed in const fn, this isn't const as the window variant,
    // and therefore cannot be used to initialize a static variable
    pub fn from_static<const S : usize, const W : usize>(array : &'static mut [N; S]) -> WindowMut<'static, N> {

        if S % W != 0 {
            panic!("Invalid image dimensions");
        }

        WindowMut {
            win : array,
            width : W,
            offset : (0, 0),
            win_sz : (S / W, W)
        }
    }*/

    pub fn linear_index_mut(&mut self, ix : usize) -> &mut N {
        assert!(ix < self.width() * self.height());
        let (row, col) = (ix / self.width(), ix % self.width());
        unsafe { &mut *self.as_mut_ptr().offset((self.width * row + col) as isize) as &mut N }
    }

    pub fn as_mut_ptr(&mut self) -> *mut N {
        // self.win.as_ptr()
        &mut self[(0usize,0usize)] as *mut _
    }

    pub fn as_ptr(&self) -> *const N {
        // self.win.as_ptr()
        &self[(0usize,0usize)] as *const _
    }
}

impl<'a, N> WindowMut<'a, N>
where
    N : Scalar + Copy + Debug + Default
{

    /// Applies a closure to a subset of the mutable window. This effectively forces
    /// the user to innaugurate a new scope for which the lifetime of the original window
    /// is no longer valid, thus allowing applying an operation to different positions of
    /// a mutable window without violating aliasing rules.
    pub fn apply_to_sub<R>(&mut self, offset : (usize, usize), sz : (usize, usize), f : impl Fn(WindowMut<'_, N>)->R) -> R {
        assert_nonzero(sz);
        let ptr = self.win.as_mut_ptr();
        let len = self.win.len();
        let mut sub = unsafe { WindowMut::from_ptr(ptr, len, self.width, offset, sz).unwrap() };
        f(sub)
    }

    /// Calls the same closure with apply_to_sub by specifying a series of offsets and sizes. Elements can overlap
    /// without problem, since the operation is sequential.
    pub fn apply_to_sub_sequence<R>(&mut self, offs_szs : &[((usize, usize), (usize, usize))], f : impl Fn(WindowMut<'_, N>)->R + Clone) -> Vec<R> {
        let mut res = Vec::new();
        for (off, sz) in offs_szs.iter().copied() {
            res.push(self.apply_to_sub(off, sz, f.clone()));
        }
        res
    }

    pub fn area(&self) -> usize {
        self.width() * self.height()
    }

    pub fn byte_stride(&self) -> usize {
        std::mem::size_of::<N>() * self.width
    }

    pub fn clear(&'a mut self)
    where
        N : Zero + Pixel
    {
        self.pixels_mut(1).for_each(|px| *px = N::zero() );
    }

    // TODO rewrite this using safe rust.
    #[allow(mutable_transmutes)]
    pub fn clone_owned(&'a self) -> Image<N>
    where
        N : Copy + Default + Zero + 'static
    {
        let mut buf = Vec::new();
        let ncols = self.win_sz.1;
        unsafe {
            for row in std::mem::transmute::<_, &'a mut Self>(self).rows_mut() {
                buf.extend(row.iter().cloned())
            }
        }
        Image::from_vec(buf, ncols)
    }

    pub fn get_mut(mut self, index : (usize, usize)) -> Option<&'a mut N> {
        if index.0 < self.height() && index.1 < self.width() {
            unsafe { Some(self.get_unchecked_mut(index)) }
        } else {
            None
        }
    }

    pub unsafe fn get_unchecked_mut(mut self, index : (usize, usize)) -> &'a mut N {
        let off_ix = (self.offset.0 + index.0, self.offset.1 + index.1);
        let (limit_row, limit_col) = (self.offset.0 + self.win_sz.0, self.offset.1 + self.win_sz.1);
        unsafe { self.win.get_unchecked_mut(index::linear_index(off_ix, self.original_size().1)) }
    }

    pub fn offset_ptr_mut(mut self) -> *mut N {
        unsafe { self.get_unchecked_mut((0, 0)) as *mut N }
    }

    pub unsafe fn from_ptr(ptr : *mut N, len : usize, full_ncols : usize, offset : (usize, usize), sz : (usize, usize)) -> Option<Self> {
        let s = std::slice::from_raw_parts_mut(ptr, len);
        Self::sub_from_slice(s, full_ncols, offset, sz)
    }

    pub fn from_slice(src : &'a mut [N], ncols : usize) -> Option<Self> {
        if src.len() % ncols != 0 {
            return None;
        }
        /*let nrows = src.len() / ncols;
        Self {
            win : DMatrixSliceMut::from_slice_generic(src, Dynamic::new(nrows),
            Dynamic::new(ncols)),
            offset : (0, 0),
            orig_sz : (nrows, ncols)
        }*/
        let nrows = src.len() / ncols;
        Some(Self{
            // win : DMatrixSlice::from_slice_generic(src, Dynamic::new(nrows),
            // Dynamic::new(ncols)),
            win : src,
            offset : (0, 0),
            width : ncols,
            win_sz : (nrows, ncols),
        })
    }

    pub fn sub_from_slice(
        src : &'a mut [N],
        original_width : usize,
        offset : (usize, usize),
        dims : (usize, usize)
    ) -> Option<Self> {
        let nrows = src.len() / original_width;
        if offset.0 + dims.0 <= nrows && offset.1 + dims.1 <= original_width {
            Some(Self {
                win : src,
                offset,
                width : original_width,
                win_sz : dims
            })
        } else {
            None
        }
    }

    /// We might just as well make this take self by value, since the mutable reference to self will be
    /// invalidated by the borrow checker when we have the child.
    // pub fn sub_window_mut(&'a mut self, offset : (usize, usize), dims : (usize, usize)) -> Option<WindowMut<'a, N>> {
    pub fn sub_window_mut(mut self, offset : (usize, usize), dims : (usize, usize)) -> Option<WindowMut<'a, N>> {
        assert_nonzero(dims);
        let new_offset = (self.offset.0 + offset.0, self.offset.1 + offset.1);
        if new_offset.0 + dims.0 <= self.original_size().0 && new_offset.1 + dims.1 <= self.original_size().1 {
            Some(Self {
                win : self.win,
                offset : (self.offset.0 + offset.0, self.offset.1 + offset.1),
                width : self.width,
                win_sz : dims
            })
        } else {
            None
        }
    }

    // pub fn centered_sub_window_mut(mut self, center : (usize, usize))

    /// Creates a window that cover the whole slice src, assuming it represents a square image.
    pub fn from_square_slice(src : &'a mut [N]) -> Option<Self> {
        Self::from_slice(src, (src.len() as f64).sqrt() as usize)
    }

    pub fn windows_mut(mut self, sz : (usize, usize)) -> impl Iterator<Item=WindowMut<'a, N>>
    where
        N : Mul<Output=N> + MulAssign
    {
        assert_nonzero(sz);
        let (step_v, step_h) = sz;
        if sz.0 > self.win_sz.0 || sz.1 > self.win_sz.1 {
            panic!("Child window size bigger than parent window size");
        }
        if self.height() % sz.0 != 0 || self.width() % sz.1 != 0 {
            panic!("Image size should be a multiple of window size (Required window {:?} over parent window {:?})", sz, self.win_sz);
        }
        let offset = self.offset;
        WindowIteratorMut::<'a, N> {
            source : self,
            size : sz,
            curr_pos : offset,
            step_v,
            step_h
        }
    }

}

impl<'a> WindowMut<'a, u8> {

    // TODO also implement contrast_adjust_mut
    pub fn brightness_adjust_mut(&'a mut self, k : i16) {

        assert!(k <= 255 && k >= -255);
        let abs_k = k.abs() as u8;
        if k > 0 {
            self.pixels_mut(1).for_each(|px| *px = px.saturating_add(abs_k) );
        } else {
            self.pixels_mut(1).for_each(|px| *px = px.saturating_sub(abs_k) );
        }
    }

}

impl<'a> WindowMut<'a, u8> {

    /// Gamma-corrects, i.e. multiplies input by input^(1/gamma) and normalize.
    pub fn gamma_correct_inplace(&mut self, gamma : f32) {

        // Using gamma < 1.0 avoids saturating the image. Perhaps offer a version that
        // does just that, without normalization.
        assert!(gamma <= 1.0);
        for i in 0..self.win_sz.0 {
            for j in 0..self.win_sz.1 {
                self[(i,j)] = (self[(i,j)] as f32).powf(1. / gamma).max(0.0).min(255.0) as u8
            }
        }
    }

    /// For any pixel >= color, set it and its neighborhood to erase_color.
    pub fn erase_speckles(&mut self, color : u8, neigh : usize, erase_color: u8) {
        for i in 0..self.win_sz.0 {
            for j in 0..self.win_sz.1 {
                if self[(i, j)] >= color {
                    for ni in i.saturating_sub(neigh)..((i + neigh).min(self.win_sz.0-1)) {
                        for nj in j.saturating_sub(neigh)..((j + neigh).min(self.win_sz.1-1)) {
                            self[(ni, nj)] = erase_color;
                        }
                    }
                }
            }
        }
    }

    pub fn fill_with_byte(&'a mut self, byte : u8) {
        // self.rows_mut().for_each(|row| std::ptr::write_bytes(&mut row[0] as *mut _, byte, row.len()) );
        /*for ix in 0..self.win_sz.0 {
            let row = self.row_mut(ix);
            std::ptr::write_bytes(&mut row[0] as *mut _, byte, row.len());
        }*/
        unimplemented!()
    }

    pub fn patches(&self, px_spacing : usize) -> Vec<Patch> {
        /*let src_win = unsafe {
            Window {
                offset : (self.offset.0, self.offset.1),
                orig_sz : self.original_size(),
                win_sz : self.win_sz,
                win : std::slice::from_raw_parts(self.win.as_ptr(), self.win.len()),
            }
        };
        let mut patches = Vec::new();
        color::full_color_patches(&mut patches, &src_win, px_spacing as u16, ColorMode::Exact(0), color::ExpansionMode::Dense);
        patches*/
        unimplemented!()
    }

    pub fn draw_patch_contour(&mut self, patch : &Patch, color : u8) {
        let pxs = patch.outer_points::<usize>(crate::feature::patch::ExpansionMode::Contour);
        self.draw(Mark::Shape(pxs, false, color));
    }

    pub fn draw_patch_rect(&mut self, patch : &Patch, color : u8) {
        let rect = patch.outer_rect::<usize>();
        self.draw(Mark::Rect((rect.0, rect.1), (rect.2, rect.3), color));
    }

}

pub(crate) unsafe fn create_immutable<'a>(win : &'a WindowMut<'a, u8>) -> Window<'a, u8> {
    unsafe {
        Window {
            offset : (win.offset.0, win.offset.1),
            width : win.width,
            win_sz : win.win_sz,
            win : std::slice::from_raw_parts(win.win.as_ptr(), win.win.len()),
        }
    }
}

/*impl<N> WindowMut<'_, N>
where
    N : Scalar + Copy
{

    unsafe fn create_immutable_without_lifetime(&self, src : (usize, usize), dim : (usize, usize)) -> Window<'_, N> {
        Window {
            offset : (self.offset.0 + src.0, self.offset.1 + src.1),
            orig_sz : self.original_size(),
            win_sz : dim,
            win : std::slice::from_raw_parts(self.win.as_ptr(), self.win.len()),
        }
    }

    /// Creates a new mutable window, forgetting the lifetime of the previous window.
    unsafe fn create_mutable_without_lifetime(&mut self, src : (usize, usize), dim : (usize, usize)) -> WindowMut<'_, N> {
        WindowMut {
            offset : (self.offset.0 + src.0, self.offset.1 + src.1),
            orig_sz : self.original_size(),
            win_sz : dim,
            win : std::slice::from_raw_parts_mut(self.win.as_mut_ptr(), self.win.len()),
        }
    }
}*/

//fn step_row<'a, N>(r : &'a mut [N], spacing : usize) -> std::iter::Flatten<std::iter::StepBy<std::slice::IterMut<'a, N>>>
//where
//    N : 'a,
//    Flatten<std::iter::StepBy<std::slice::IterMut<'a, N>>> : IntoIterator<N>
//{
//    r.iter_mut().step_by(spacing).flatten()
    //r.iter_mut()
//}

// fn step<'a, N>(s : &'a [N], step : usize) -> std::iter::Flatten<std::iter::StepBy<std::slice::Iter<'a, N>>> {
//    s.iter().step_by(step)
// }

impl<'a, N> WindowMut<'a, N>
where
    N : Scalar + Copy + Default
{

    /*// Returns image corners with the given dimensions.
    pub fn corners_mut(mut self, height : usize, width : usize) -> Option<[WindowMut<'a, N>; 4]> {
        let right = self.width() - width;
        let bottom = self.height() - height;
        let sz = (height, width);

        self.apply_to_sub_mut((0, 0), (100, 100), |mut win| {
            self.shape();
        });

        unimplemented!();

        // let (top, rem) = self.split(height);
        // let (mid, bottom) = rem.split(rem.height() - height);

        /*let tl = self.sub_window((0, 0), sz)?;
        let tr = self.sub_window((0, right), sz)?;
        let bl = self.sub_window((bottom, 0), sz)?;
        let br = self.sub_window((bottom, right), sz)?;
        Some([tl, tr, bl, br])*/
    }

    // Returns image sides (without corners or same dimension).
    pub fn sides_mut(mut self, height : usize, width : usize) -> Option<[Window<'a, N>; 4]> {
        let right = self.width() - width;
        let bottom = self.height() - height;
        let vert_sz = (height, self.width() - 2*width);
        let horiz_sz = (self.height() - 2*height, self.width());
        let top = self.sub_window((0, width), vert_sz)?;
        let right = self.sub_window((height, right), horiz_sz)?;
        let bottom = self.sub_window((bottom, width), vert_sz)?;
        let left = self.sub_window((height, 0), horiz_sz)?;
        Some([top, right, bottom, left])
    }*/

    pub fn shape(&self) -> (usize, usize) {
        self.win_sz
    }

    pub fn orig_sz(&self) -> (usize, usize) {
        self.original_size()
    }

    pub fn full_slice(&'a self) -> &'a [N] {
        &self.win[..]
    }

    pub fn offset(&self) -> (usize, usize) {
        self.offset
    }

    pub fn width(&self) -> usize {
        self.shape().1
    }

    pub fn height(&self) -> usize {
        self.shape().0
    }

}

fn verify_border_dims<N>(src : &Window<N>, dst : &WindowMut<N>)
where
    N : Scalar + Copy + Debug + Default
{
    assert!(src.height() < dst.height());
    assert!(src.width() < dst.width());
    let diffw = dst.width() - src.width();
    let diffh = dst.height() - src.height();
    assert!(diffw % 2 == 0);
    assert!(diffh % 2 == 0);
    assert!(src.height() + diffh == dst.height());
    assert!(src.width() + diffw == dst.width());
}

fn border_dims<N>(src : &Window<N>, dst : &WindowMut<N>) -> (usize, usize)
where
    N : Scalar + Copy + Debug + Default
{
    let diffw = dst.width() - src.width();
    let diffh = dst.height() - src.height();
    (diffh / 2, diffw / 2)
}

impl<'a, N> WindowMut<'a, N>
where
    N : Scalar + Copy + Default
{

    /*/// Copies the content of the slice, assuming raster order.
    pub fn copy_from_slice<'b>(&'a mut self, other : &'b [N]) {
        assert!(self.area() == other.len());
        self.copy_from(&Window::from_slice(other, self.win_sz.1));
    }*/

    pub fn rows_mut<'b>(&'b mut self) -> impl Iterator<Item=&'a mut [N]> + 'b {
        let stride = self.original_size().1;
        let tl = self.offset.0 * stride + self.offset.1;
        (0..self.win_sz.0).map(move |i| {
            let start = tl + i*stride;
            let slice = &self.win[start..(start+self.win_sz.1)];
            unsafe { std::slice::from_raw_parts_mut(slice.as_ptr() as *mut _, slice.len()) }
        })
    }

    /*pub unsafe fn pixels_mut_ptr(&'a mut self, spacing : usize) -> impl Iterator<Item=*mut N> {
        self.pixels_mut(spacing).map(|px| px as *mut _ )
    }*/
    pub fn foreach_pixel<'b>(&'b mut self, spacing : usize, f : impl Fn(&mut N)) {
        for px in self.pixels_mut(spacing) {
            f(px);
        }
    }

    pub fn pixels_mut<'b>(&'b mut self, spacing : usize) -> impl Iterator<Item=&'a mut N> + 'b {
        self.rows_mut().step_by(spacing).map(move |r| r.iter_mut().step_by(spacing) ).flatten()
    }

    pub fn labeled_pixels_mut(&'a mut self, spacing : usize) -> impl Iterator<Item=(usize, usize, &'a mut N)> +'a {
        let w = self.width();
        self.pixels_mut(spacing)
            .enumerate()
            .map(move |(ix, px)| {
                let (r, c) = (ix / w, ix % w);
                (r, c, px)
            })
    }

    pub fn conditional_fill(&mut self, mask : &Window<u8>, color : N) {
        assert!(self.shape() == mask.shape());
        self.pixels_mut(1).zip(mask.pixels(1)).for_each(|(d, m)| if *m != 0 { *d = color } );
    }

    pub fn fill(&mut self, color : N) {

        // TODO use std::intrinsic::write_bytes?

        #[cfg(feature="ipp")]
        unsafe {
            let (step, sz) = crate::image::ipputils::step_and_size_for_window_mut(&self);
            if self.pixel_is::<f32>() {
                let ans = crate::foreign::ipp::ippi::ippiSet_32f_C1R(
                    *mem::transmute::<_, &f32>(&color),
                    mem::transmute(self.as_mut_ptr()),
                    step,
                    sz
                );
                assert!(ans == 0);
                return;
            }

            if self.pixel_is::<u8>() {
                let ans = crate::foreign::ipp::ippi::ippiSet_8u_C1R(
                    *mem::transmute::<_, &u8>(&color),
                    mem::transmute(self.as_mut_ptr()),
                    step,
                    sz
                );
                // println!("{}", ans);
                assert!(ans == 0);
                return;
            }
        }

        unsafe { mem::transmute::<_, &'a mut WindowMut<'a, N>>(self).pixels_mut(1).for_each(|px| *px = color ); }
    }

}

impl<'a, N> WindowMut<'a, N>
where
    N : Scalar + Copy + Mul<Output=N> + MulAssign + PartialOrd + Default + Pixel
{

    pub fn paint(&'a mut self, min : N, max : N, color : N) {
        self.pixels_mut(1).for_each(|px| if *px >= min && *px <= max { *px = color; } );
    }

    /*/// Applies closure to sub_window. Useful when you need to apply an operation
    /// iteratively to different regions of a window, and cannot do so because the mutable reference gets
    /// invalidated when a new sub-window is returned within a loop.
    pub fn apply_to_sub_window<'b, F>(&'b mut self, offset : (usize, usize), dim : (usize, usize), mut f : F)
    where
        F : FnMut(WindowMut<'b, N>)
    {
        let mut sub = unsafe { self.create_mutable_without_lifetime(offset, dim) };
        f(sub);
    }*/

    /// Analogous to slice::copy_within, copy a the sub_window (src, dim) to the sub_window (dst, dim).
    pub fn copy_within(&'a mut self, src : (usize, usize), dst : (usize, usize), dim : (usize, usize)) {
        use crate::feature::shape;
        assert!(!shape::rect_overlaps(&(src.0, src.1, dim.0, dim.1), &(dst.0, dst.1, dim.0, dim.1)), "copy_within: Windows overlap");
        let src_win = unsafe {
            Window {
                offset : (self.offset.0 + src.0, self.offset.1 + src.1),
                width : self.original_size().1,
                win_sz : dim,
                win : std::slice::from_raw_parts(self.win.as_ptr(), self.win.len()),
            }
        };

        // TODO not working since we made sub_window_mut take by value.
        // self.sub_window_mut(dst, dim).unwrap().copy_from(&src_win);
        unimplemented!()
    }

    pub fn row_mut(&'a mut self, i : usize) -> &'a mut [N] {
        let stride = self.original_size().1;
        let tl = self.offset.0 * stride + self.offset.1;
        let start = tl + i*stride;
        let slice = &self.win[start..(start+self.win_sz.1)];
        unsafe { std::slice::from_raw_parts_mut(slice.as_ptr() as *mut _, slice.len()) }
    }

}

impl<'a, N> WindowMut<'a, N>
where
    N : Scalar + Copy + MulAssign + AddAssign + Add<Output=N> + Mul<Output=N> + SubAssign + Field + SimdPartialOrd,
    f64 : SubsetOf<N>
{
    
    pub fn component_scale(&mut self, _other : &Window<N>) {
        // self.win.component_mul_mut(&other.win);
        unimplemented!()
    }

}

impl<N> Index<RunLength> for Window<'_, N>
where
    N : Scalar + Copy
{

    type Output = [N];

    fn index(&self, rl: RunLength) -> &Self::Output {
        assert!(rl.start.1 + rl.length <= self.width() && rl.start.0 < self.height());
        unsafe { std::slice::from_raw_parts(&self[rl.start] as *const _, rl.length) }
    }

}

impl<N> Index<RunLength> for WindowMut<'_, N>
where
    N : Scalar + Copy
{

    type Output = [N];

    fn index(&self, rl: RunLength) -> &Self::Output {
        assert!(rl.start.1 + rl.length <= self.width() && rl.start.0 < self.height());
        unsafe { std::slice::from_raw_parts(&self[rl.start] as *const _, rl.length) }
    }

}

impl<N> IndexMut<RunLength> for WindowMut<'_, N>
where
    N : Scalar + Copy
{

    fn index_mut(&mut self, rl: RunLength) -> &mut Self::Output {
        assert!(rl.start.1 + rl.length <= self.width() && rl.start.0 < self.height());
        unsafe { std::slice::from_raw_parts_mut(&mut self[rl.start] as *mut _, rl.length) }
    }

}

impl<N> Index<(usize, usize)> for WindowMut<'_, N>
where
    N : Scalar + Copy
{

    type Output = N;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        unsafe { self.win.get_unchecked(index::linear_index((self.offset.0 + index.0, self.offset.1 + index.1), self.original_size().1)) }
    }
}

impl<N> IndexMut<(usize, usize)> for WindowMut<'_, N> 
where
    N : Scalar + Copy
{
    
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        unsafe { self.win.get_unchecked_mut(index::linear_index((self.offset.0 + index.0, self.offset.1 + index.1), self.original_size().1)) }
    }

}

impl<N> Index<(u16, u16)> for WindowMut<'_, N>
where
    N : Scalar + Copy
{

    type Output = N;

    fn index(&self, index: (u16, u16)) -> &Self::Output {
        self.index((index.0 as u16, index.1 as u16))
    }
}

impl<N> IndexMut<(u16, u16)> for WindowMut<'_, N>
where
    N : Scalar + Copy
{

    fn index_mut(&mut self, index: (u16, u16)) -> &mut Self::Output {
        self.index_mut((index.0 as u16, index.1 as u16))
    }
    
}

/*impl<'a, N> AsRef<DMatrixSlice<'a, N>> for Window<'a, N>
where
    N : Scalar + Copy
{
    fn as_ref(&self) -> &DMatrixSlice<'a, N> {
        // &self.win
        unimplemented!()
    }
}*/

/*impl<'a, M, N> Downsample<Image<N>> for Window<'a, M> 
where
    M : Scalar + Copy,
    N : Scalar + From<M>
{

    fn downsample(&self, dst : &mut Image<N>) {
        // let (nrows, ncols) = dst.shape();
        /*let step_rows = self.win_sz.0 / dst.nrows;
        let step_cols = self.win_sz.1 / dst.ncols;
        assert!(step_rows == step_cols);
        sampling::slices::subsample_convert_window(
            self.src,
            self.offset,
            self.win_sz, 
            (dst.nrows, dst.ncols),
            step_rows,
            dst.buf.as_mut_slice().chunks_mut(dst.nrows)
        );*/
        unimplemented!()
    }
    
}*/

/*/// Data is assumed to live on the matrix in a column-order fashion, not row-ordered.
impl<N> From<DMatrix<N>> for Image<N> 
where
    N : Scalar + Copy
{
    fn from(buf : DMatrix<N>) -> Self {
        /*let (nrows, ncols) = s.shape();
        let data : Vec<N> = s.data.into();
        let buf = DVector::from_vec(data);
        Self{ buf, nrows, ncols }*/
        let ncols = buf.ncols();
        Self{ buf, ncols }
    }
}*/

/*impl<N> From<(Vec<N>, usize)> for Image<N> 
where
    N : Scalar
{
    fn from(s : (Vec<N>, usize)) -> Self {
        let (nrows, ncols) = (s.1, s.0.len() - s.1);
        Self{ buf : DVector::from_vec(s.0), nrows, ncols  }
    }
}*/

impl<N> AsRef<[N]> for Image<N>
where
    N : Scalar + Copy + Any
{
    fn as_ref(&self) -> &[N] {
        &self.buf[..]
    }
}

impl<N> AsMut<[N]> for Image<N> 
where
    N : Scalar + Copy + Any
{
    fn as_mut(&mut self) -> &mut [N] {
        &mut self.buf[..]
    }
}

impl<N> fmt::Display for Window<'_, N> 
where
    N : Scalar + Copy,
    f64 : From<N>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", io::build_pgm_string_from_slice(&self.win, self.original_size().1))
    }
}

impl<N> fmt::Display for WindowMut<'_, N> 
where
    N : Scalar + Copy + Default,
    f64 : From<N>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", io::build_pgm_string_from_slice(&self.win, self.original_size().1))
    }
}

/*impl<N> fmt::Display for Image<N>
where
    N : Scalar + Copy,
    f64 : From<N>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", io::build_pgm_string_from_slice(&self.buf[..], self.width))
    }
}*/

/*impl showable::Show for Window<'_, u8> {

    fn show(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // crate::io::to_html(&self).fmt(f)
        write!(f, "{}", crate::io::to_html(&self) )
    }

    fn modality(&self) -> showable::Modality {
        showable::Modality::XML
    }

}

impl showable::Show for Image<u8> {

    fn show(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // crate::io::to_html(&self.full_window()).fmt(f)
        write!(f, "{}", crate::io::to_html(&self.full_window()) )
    }

    fn modality(&self) -> showable::Modality {
        showable::Modality::XML
    }

}*/

/*#[cfg(feature="literate")]
impl<'a> literate::show::Stack<'a> for Window<'a, u8> {

    fn header(&'a self) -> Option<Box<dyn Display>> {
        None
    }

    fn stack(&'a self) -> Vec<&'a dyn Show> {
        vec![self as &dyn Show]
    }

}*/

#[test]
fn checkerboard() {
    let src : [u8; 4] = [0, 1, 1, 0];
    let mut converted : Image<f32> = Image::new_constant(4, 4, 0.0);
    let win = Window::from_square_slice(&src);
    //converted.convert_from_window(&win);
    // println!("{}", converted);
}

/*impl<N> AsRef<Vec<N>> for Image<N> 
where
    N : Scalar
{
    fn as_ref(&self) -> &Vec<N> {
        self.buf.data.as_vec()
    }
}

impl<N> AsMut<Vec<N>> for Image<N> 
where
    N : Scalar
{
    fn as_mut(&mut self) -> &mut Vec<N> {
        unsafe{ self.buf.data.as_vec_mut() }
    }
}*/

/*impl<N> AsRef<DVector<N>> for Image<N> 
where
    N : Scalar
{
    fn as_ref(&self) -> &DVector<N> {
        &self.buf
    }
}

impl<N> AsMut<DVector<N>> for Image<N> 
where
    N : Scalar
{
    fn as_mut(&mut self) -> &mut DVector<N> {
        &mut self.buf
    }
}*/

/*/// Result of thresholding an image. This structure carries a (row, col) index and
/// a scalar value for this index. Rename to Keypoints.
pub struct SparseImage {

}

/// Subset of a SparseImage
pub struct SparseWindow<'a> {

    /// Source sparse image
    source : &'a SparseImage,
    
    /// Which indices we will use from the source
    ixs : Vec<usize>
}*/

// TODO implement borrow::ToOwned

/*
// Local maxima of the image multiscale transformation
pub struct Keypoint { }

// Object characterized by a set of close keypoints where ordered pairs share close angles. 
pub struct Edge { }

// Low-dimensional approximation of an edge (in terms of lines and curves).
pub struct Shape { }

// Object characterized by a set of shapes.
pub struct Object { }
*/

/*
The match item at procedural macros has the syntax $name:token where token must be one of:
item — an item, like a function, struct, module, etc.
block — a block (i.e. a block of statements and/or an expression, surrounded by braces)
stmt — a statement
pat — a pattern
expr — an expression
ty — a type
ident — an identifier
path — a path (e.g., foo, ::std::mem::replace, transmute::<_, int>, …)
meta — a meta item; the things that go inside #[...] and #![...] attributes
tt — a single token tree
vis — a possibly empty Visibility qualifier

Repeated arguments are treated as (* matches zero or more; + matches zero or one):
macro_rules! add_as{
    ( $($a:expr), )=>{ {  0 $(+$a)* }    }
}

The token type that repeats is enclosed in $(), followed by a separator and a * or a +, indicating the number of times the token will repeat.
*/

/*macro_rules! sql {
 // macth like arm for macro
    ($a:expr) => {
    
    },
    ($a:expr, $b:expr)=>{
 // macro expand to this code
        {
// $a and $b will be templated using the value/variable provided to macro
            $a+$b
        }
    }
}*/

/*#[no_mangle]
pub extern "C" fn get_funcs() -> Box<Vec<interactive::AlienFunc>> {

    use interactive::AlienFunc;

    let mut funcs = Vec::new();
    funcs.push(AlienFunc::new("add_three", |a : i64| -> Result<i64, Box<rhai::EvalAltResult>> { Ok(a + 3) }));
    funcs.push(AlienFunc::new("add_four", |a : i64| -> Result<i64, Box<rhai::EvalAltResult>> { Ok(a + 4) }));
    funcs.push(AlienFunc::new("append_text_to_string", |a : String| -> Result<String, Box<rhai::EvalAltResult>> { Ok(format!("{}newtext", a)) }));

    Box::new(funcs)
}*/

/*#[derive(Serialize, Deserialize, Clone)]
pub struct MyStruct {
    field : [i64; 2]
}

impl Default for MyStruct {

    fn default() -> Self {
        Self { field : [0, 0] }
    }

}

impl std::iter::IntoIterator for MyStruct {

    type Item = i64;

    type IntoIter = Box<dyn Iterator<Item=i64>>;

    fn into_iter(self) -> Self::IntoIter {
        Box::new(vec![self.field[0].clone(), self.field[1].clone()].into_iter())
    }

}

fn convert_from_map(other : rhai::Dynamic) -> Result<MyStruct, Box<rhai::EvalAltResult>> {
    match other.try_cast::<rhai::Map>() {
        Some(map) => {
            match map.get("field") {
                Some(v) => {
                    match v.clone().try_cast::<rhai::Array>() {
                        Some(arr) => {
                            match (arr.get(0).and_then(|v| v.clone().try_cast::<i64>()), arr.get(1).and_then(|v| v.clone().try_cast::<i64>())) {
                                (Some(v1), Some(v2)) => {
                                    Ok(MyStruct { field : [v1, v2] })
                                },
                                _ => Err(Box::new("Invalid fields".into()))
                            }
                        },
                        None => Err(Box::new("Field is not array".into()))
                    }
                },
                None => Err(Box::new("Missing field".into()))
            }
        },
        None => Err(Box::new("Result is not a map".into()))
    }
}*/

/*impl deft::Interactive for MyStruct {

    #[export_name="register_MyStruct"]
    extern "C" fn interactive() -> Box<deft::TypeInfo> {
        deft::TypeInfo::builder::<Self>()
            .fallible("add_one", |a : i64| -> Result<i64, Box<rhai::EvalAltResult>> { Ok(a + 1) })
            .iterable()
            .initializable()
            .parseable()
            .indexable(|s : &mut Self, ix : i64| { Ok(s.field[ix as usize]) })
            .mutably_indexable(|s : &mut Self, ix : i64, val : i64| { s.field[ix as usize] = val; Ok(()) })
            .field("field", |s : &mut MyStruct| { Ok(vec![rhai::Dynamic::from(s.field[0]), rhai::Dynamic::from(s.field[1])]) })
            .convertible(|s : &mut Self, other : rhai::Dynamic| { convert_from_map(other) })
            .priority(0)
            .build()
    }

}*/

/*#[test]
fn deft_test() {
    let a = 1i64;
    let b = String::from("Hello");
    let c = &1i64;
    let d = MyStruct { field : [0, 1] };
    deft::repl!(a, b, c, d);
}

impl deft::Show for Image<u8> {

}

impl deft::Embed for Image<u8> {

    fn embed(&self) -> String {
        crate::io::to_html(&self.full_window())
    }

}

impl deft::Interactive for Image<u8> {

    fn short_name() -> &'static str {
        "Image"
    }

    #[export_name="register_Image"]
    extern "C" fn interactive() -> Box<deft::TypeInfo> {

        use rhai::{Dynamic, Array};
        use deft::ReplResult;

        deft::TypeInfo::builder::<Self>()
            .fallible("open",
            |img : &mut Self, path : rhai::ImmutableString| -> Result<Self, Box<rhai::EvalAltResult>> {
                let new_img = crate::io::decode_from_file(&path)
                    .map_err(|e| Box::new(rhai::EvalAltResult::from(format!("{}", e))) )?;
                Ok(new_img)
            })
            .fallible("shape", |s : &mut Self| -> ReplResult<Array> {
                Ok(vec![Dynamic::from(s.height() as i64), Dynamic::from(s.width() as i64) ])
            })
            .fallible("height", |s : &mut Self| -> ReplResult<i64> { Ok(s.height() as i64) })
            .fallible("width", |s : &mut Self| -> ReplResult<i64> { Ok(s.width() as i64) })
            .fallible("draw", |s : &mut Self, mark : rhai::Map| -> ReplResult<()> {
                /*println!("{:?}", s.shape());
                for mark in marks.iter() {
                    match mark.clone().try_cast::<Patch>() {
                        Some(patch) => {
                            s.full_window_mut().draw(Mark::Shape(patch.outer_points(crate::feature::patch::ExpansionMode::Contour), 255));
                        },
                        None => {
                            return Err("Mark is not patch".into());
                        }
                    }
                }*/

                let mark : Mark = deft::convert::deserialize_from_map(mark).unwrap();
                s.full_window_mut().draw(mark);
                Ok(())
            })
            .initializable()
            .showable()
            .embeddable()
            .priority(0)
            .build()
    }

}*/

/*impl interactive::Interactive for Image<u8> {

    // fn hosrt_name() -> &'static str {
    //    "Image"
    // }

    fn new() -> Self {
        unimplemented!()
    }

    fn embed(&self) -> Option<String> {
        Some(crate::io::to_html(&self.full_window()))
    }

    #[export_name="register_image"]
    extern "C" fn interactive(registry : Box<interactive::Registry<'_>>) -> Box<interactive::TypeInfo> {

        registry.add::<Self>()
            .fallible_method("cat", |a : rhai::ImmutableString| -> Result<String, Box<rhai::EvalAltResult>> { Ok(format!("{}hellofromclient", a)) })
            .fallible_method("add_one", |a : i64| -> Result<i64, Box<rhai::EvalAltResult>> { Ok(a + 1) })
            .register()

        /*use rhai;
        Self::prepare(engine);
        Self::display(engine);
        Self::serialization(engine);

        Self::new(
            engine,
            |img, map| {
                match (map.get("width").and_then(|w| w.as_int().ok() ), (map.get("height").and_then(|h| h.as_int().ok() ))) {
                    (Some(w), Some(h)) => {
                        if w > 0 && h > 0 {
                            *img = Image::new_constant(h as usize, w as usize, 0);
                            Ok(())
                        } else {
                            Err(Box::new(rhai::EvalAltResult::from("Arguments should be greater than zero")))
                        }
                    },
                    _ => {
                        Err(Box::new(rhai::EvalAltResult::from("Invalid fields")))
                    }
                }

            }
        );

        engine.register_result_fn(
            "open",
            |img : &mut Self, path : rhai::ImmutableString| -> Result<(), Box<rhai::EvalAltResult>> {
                let new_img = crate::io::decode_from_file(&path)
                    .map_err(|e| Box::new(rhai::EvalAltResult::from(format!("{}", e))) )?;
                *img = new_img;
                Ok(())
            }
        );

        engine.register_fn("add_two", |a : i64| -> i64 { a + 2 });
        engine.register_fn("add_two_float", |a : f64| -> f64 { a + 2. });
        engine.register_fn("append_text_client", |a : String| -> String { format!("{}newtext", a) });
        engine.register_fn("append_text_client_dyn", |a : rhai::Dynamic| -> rhai::Dynamic { rhai::Dynamic::from(format!("{}newtext", a.into_string().unwrap())) });

        use std::any::TypeId;
        let immutable_str_id = unsafe { std::mem::transmute::<u64, TypeId>(3264575275192760566) };

        engine.register_raw_fn(
            "append_text_client_raw",
            /*&[TypeId::of::<ImmutableString>()]*/ &[immutable_str_id],
            |ctx : rhai::NativeCallContext<'_>, args : &mut [&mut rhai::Dynamic]| -> Result<rhai::Dynamic, Box<rhai::EvalAltResult>> {
                Ok(rhai::Dynamic::from(format!("{}newtext", std::mem::take(args[0]).cast::<rhai::ImmutableString>())))
            }
        );

        println!("Type id of ImmutableString at shutter : {:?}", rhai::ImmutableString::from("").type_id() );
        println!("Type id of Dynamic at shutter : {:?}", rhai::Dynamic::from(1i16).type_id() );
        println!("Type id of Arc at client : {:?}", std::sync::Arc::new(1i16).type_id() );
        println!("ImmutableString at client: {:?}", rhai::ImmutableString::from(""));

        use smartstring;
        println!("SmartString at client: {:?}", smartstring::SmartString::<smartstring::LazyCompact>::new_const().type_id());

        Self::info()*/
    }

    // Perhaps we abstract certain details away,
    // and just require the registration of "associated" and "methods",
    // automatically taking care of the plumbing without exposing the engine
    // to the user.

}*/

#[cfg(feature="opencv")]
pub fn from_nalgebra3_vec(v : nalgebra::Vector3<f64>) -> opencv::core::Mat {

    use opencv::core;
    use opencv::prelude::MatTraitManual;
    use opencv::prelude::MatTrait;

    let mut mat = core::Mat::default();
    unsafe {
        mat.create_rows_cols(3, 1, core::CV_64F);
        for i in 0..3 {
            *mat.at_mut::<f64>(i).unwrap() = v[i as usize];
        }
    }

    mat
}

#[cfg(feature="opencv")]
pub fn to_nalgebra3_vec(m : opencv::core::Mat) -> nalgebra::Vector3<f64> {

    use opencv::core;
    use opencv::prelude::MatTraitManual;
    use opencv::prelude::MatTrait;

    let mut out = Vector3::zeros();
    unsafe {
        for i in 0..3 {
            out[i as usize] = *m.at_unchecked::<f64>(i).unwrap();
        }
    }
    out
}

#[cfg(feature="opencv")]
pub fn to_nalgebra3_mat(m : opencv::core::Mat) -> nalgebra::Matrix3<f64> {

    use opencv::core;
    use opencv::prelude::MatTraitManual;
    use opencv::prelude::MatTrait;

    let mut out = Matrix3::zeros();
    unsafe {
        for i in 0..3 {
            for j in 0..3 {
                out[(i as usize, j as usize)] = *m.at_2d_unchecked::<f64>(i, j).unwrap();
            }
        }
    }
    out
}

#[cfg(feature="opencv")]
pub fn from_nalgebra3(m : nalgebra::Matrix3<f64>) -> opencv::core::Mat {

    use opencv::core;
    use opencv::prelude::MatTraitManual;
    use opencv::prelude::MatTrait;

    let mut mat = core::Mat::default();
    unsafe {
        mat.create_rows_cols(3, 3, core::CV_64F);
        for i in 0..3 {
            for j in 0..3 {
                *mat.at_2d_unchecked_mut::<f64>(i, j).unwrap() = m[(i as usize, j as usize)];
            }
        }
    }

    mat
}

/*impl<'a, N> Borrow<Window<'a, N>> for Image<N>
where
    N : Scalar + Clone + Copy + Serialize + DeserializeOwned + Any + Zero + Default
{

    fn borrow(&self) -> &Window<'a, N> {
        &self.full_window()
    }

}*/

pub fn assert_nonzero(shape : (usize, usize)) {
    assert!(shape.0 >= 1 && shape.1 >= 1)
}

pub fn coords_across_line(
    src : (usize, usize),
    dst : (usize, usize),
    (nrow, ncol) : (usize, usize)
) -> Box<dyn Iterator<Item=(usize, usize)>> {
    if src.0 == dst.0 {
        Box::new((src.1.min(dst.1)..(src.1.max(dst.1))).map(move |c| (src.0, c) ))
    } else if src.1 == dst.1 {
        Box::new((src.0.min(dst.0)..(src.0.max(dst.0))).map(move |r| (r, src.1 ) ))
    } else {
        let (dist, theta) = index::index_distance(src, dst, nrow);
        let d_max = dist as usize;
        Box::new((0..d_max).map(move |i| {
            let x_incr = theta.cos() * i as f64;
            let y_incr = theta.sin() * i as f64;
            let x_pos = (src.1 as i32 + x_incr as i32) as usize;
            let y_pos = (src.0 as i32 - y_incr as i32) as usize;
            (y_pos, x_pos)
        }))
    }
}