use std::cmp::{Ord, PartialOrd, Eq, PartialEq};
use std::ops::*;
use crate::image::*;
use crate::stat;
use serde::{Serialize, de::DeserializeOwned};
use std::any::Any;
use std::fmt::Debug;
use num_traits::*;
use nalgebra::Scalar;
use std::mem;
use num_traits::Float;
// use crate::gray::Foreground;
use std::convert::{TryFrom, TryInto, AsRef, AsMut};
use nalgebra::Vector2;
use std::borrow::Borrow;

/*Organize as:

ops {

    unary {

        logic;

        gray;

        float;
    }

    binary {

        logic;

        gray;

        float;

    }

    scalar {

    }

}*/

/*
Perhaps the normalize_bound method could be part of a Bounded trait, for images
with an upper pixel bound.
*/

/*pub struct ShiftDiff {
    pub dh : ImageBuf<u8>,
    pub dw : ImageBuf<u8>
}

impl ShiftDiff {

    pub fn calculate(img : &Window<u8>, step : usize) -> Self {
        let red_shape_w = (img.height(), img.width()-step);
        let red_shape_h = (img.height()-step, img.width());
        let red_shape_both = (img.height()-step, img.width()-step);
        let mut img_cw = img.sub_window((0, step), red_shape_w).unwrap().clone_owned();
        let mut img_ch = img.sub_window((step, 0), red_shape_h).unwrap().clone_owned();

        let mut dh = img.clone_owned();
        let mut dw = img.clone_owned();

        dw.window_mut((0, 0), red_shape_w).unwrap().abs_diff_assign(img_cw.full_window());
        dh.window_mut((0, 0), red_shape_h).unwrap().abs_diff_assign(img_ch.full_window());
        dh.window_mut((img.height() - step, 0), (step, img.width())).unwrap().fill(0);
        dw.window_mut((0, img.width() - step), (img.height(), step)).unwrap().fill(0);

        // dw.full_window_mut().add_assign(dh.full_window());
        // let dwh = dw.window((0, 0), red_shape_both).unwrap().clone_owned();
        Self { dh, dw }
    }
}*/

/*
pub trait AddTo { }
pub trait MulTo { }
*/

impl<P, S> Image<P, S> 
where
    P : Pixel + PartialOrd,
    S : Storage<P>,
{

    // a < b ? 255 else 0
    pub fn is_smaller_to<T, U>(&self, b : &Image<P, T>, dst : &mut Image<P, U>) 
    where
        T : Storage<P>,
        U : StorageMut<P>
    {

        #[cfg(feature="ipp")]
        unsafe {
            if self.pixel_is::<u8>() {
                let (a_stride, a_roi) = crate::image::ipputils::step_and_size_for_image(self);
                let (b_stride, b_roi) = crate::image::ipputils::step_and_size_for_image(b);
                let (dst_stride, dst_roi) = crate::image::ipputils::step_and_size_for_image(dst);
                let ans = crate::foreign::ipp::ippi::ippiCompare_8u_C1R(
                    mem::transmute(self.as_ptr()),
                    a_stride,
                    mem::transmute(b.as_ptr()),
                    b_stride,
                    mem::transmute(dst.as_mut_ptr()),
                    dst_stride,
                    a_roi,
                    crate::foreign::ipp::ippi::IppCmpOp_ippCmpLess
                );
                assert!(ans == 0);
                return;
            }
        }

        unimplemented!()
    }
    
    // a > b ? a else b
    pub fn greater_to<T, U>(&self, b : &Image<P, T>, dst : &mut Image<P, U>) 
    where
        T : Storage<P>,
        U : StorageMut<P>
    {
        #[cfg(feature="ipp")]
        unsafe {
            let (a_stride, a_roi) = crate::image::ipputils::step_and_size_for_image(self);
            let (b_stride, b_roi) = crate::image::ipputils::step_and_size_for_image(b);
            let (dst_stride, dst_roi) = crate::image::ipputils::step_and_size_for_image(dst);
            if self.pixel_is::<u8>() {
                let ans = crate::foreign::ipp::ippi::ippiMaxEvery_8u_C1R(
                    mem::transmute(self.as_ptr()),
                    a_stride,
                    mem::transmute(b.as_ptr()),
                    b_stride,
                    mem::transmute(dst.as_mut_ptr()),
                    dst_stride,
                    a_roi
                );
                assert!(ans == 0);
                return;
            } else if self.pixel_is::<f32>() {
                let ans = crate::foreign::ipp::ippi::ippiMaxEvery_32f_C1R(
                    mem::transmute(self.as_ptr()),
                    a_stride,
                    mem::transmute(b.as_ptr()),
                    b_stride,
                    mem::transmute(dst.as_mut_ptr()),
                    dst_stride,
                    a_roi
                );
                assert!(ans == 0);
                return;
            }
        }
        unimplemented!()
    }
    
    // a < b ? a else b
    pub fn smaller_to<T, U>(&self, b : &Image<P, T>, dst : &mut Image<P, U>) 
    where
        T : Storage<P>,
        U : StorageMut<P>
    {

        #[cfg(feature="ipp")]
        unsafe {
            let (a_stride, a_roi) = crate::image::ipputils::step_and_size_for_image(self);
            let (b_stride, b_roi) = crate::image::ipputils::step_and_size_for_image(b);
            let (dst_stride, dst_roi) = crate::image::ipputils::step_and_size_for_image(dst);
            if self.pixel_is::<u8>() {
                let ans = crate::foreign::ipp::ippi::ippiMinEvery_8u_C1R(
                    mem::transmute(self.as_ptr()),
                    a_stride,
                    mem::transmute(b.as_ptr()),
                    b_stride,
                    mem::transmute(dst.as_mut_ptr()),
                    dst_stride,
                    a_roi
                );
                assert!(ans == 0);
                return;          
            } else if self.pixel_is::<f32>() {
                let ans = crate::foreign::ipp::ippi::ippiMinEvery_32f_C1R(
                    mem::transmute(self.as_ptr()),
                    a_stride,
                    mem::transmute(b.as_ptr()),
                    b_stride,
                    mem::transmute(dst.as_mut_ptr()),
                    dst_stride,
                    a_roi
                );
                assert!(ans == 0);
                return;
            }
        }

        unimplemented!()
    }


}

impl<P, S> Image<P, S> 
where
    P : Pixel + Add<Output=P> + Sub<Output=P> + Mul<Output=P> + Div<Output=P>,
    S : Storage<P>,
{

    #[cfg(feature="ipp")]
    pub fn div_to<T, U>(&self, rhs : &Image<P, T>, dst : &mut Image<P, U>)
    where
        T : Storage<P>,
        U : StorageMut<P>
    {
        let lhs = self;
        let (lhs_stride, lhs_roi) = crate::image::ipputils::step_and_size_for_image(lhs);
        let (rhs_stride, rhs_roi) = crate::image::ipputils::step_and_size_for_image(rhs);
        let (dst_stride, dst_roi) = crate::image::ipputils::step_and_size_for_image(dst);
        unsafe {
            if lhs.pixel_is::<f32>() {
                let ans = crate::foreign::ipp::ippi::ippiDiv_32f_C1R(
                    mem::transmute(lhs.as_ptr()),
                    lhs_stride,
                    mem::transmute(rhs.as_ptr()),
                    rhs_stride,
                    mem::transmute(dst.as_mut_ptr()),
                    dst_stride,
                    lhs_roi        
                );
                assert!(ans == 0);
                return;
            }
        }
        unimplemented!();
    }

    #[cfg(feature="ipp")]
    pub fn mul_to<T, U>(&self, rhs : &Image<P, T>, dst : &mut Image<P, U>)
        where
            T : Storage<P>,
            U : StorageMut<P>{
        let lhs = self;
        let (lhs_stride, lhs_roi) = crate::image::ipputils::step_and_size_for_image(lhs);
        let (rhs_stride, rhs_roi) = crate::image::ipputils::step_and_size_for_image(rhs);
        let (dst_stride, dst_roi) = crate::image::ipputils::step_and_size_for_image(dst);
        unsafe {
            if lhs.pixel_is::<f32>() {
                let ans = crate::foreign::ipp::ippi::ippiMul_32f_C1R(
                    mem::transmute(lhs.as_ptr()),
                    lhs_stride,
                    mem::transmute(rhs.as_ptr()),
                    rhs_stride,
                    mem::transmute(dst.as_mut_ptr()),
                    dst_stride,
                    lhs_roi        
                );
                assert!(ans == 0);
                return;
            }
        }
        unimplemented!();
    }

    #[cfg(feature="ipp")]
    pub fn add_to<T, U>(&self, rhs : &Image<P, T>, dst : &mut Image<P, U>)
        where
            T : Storage<P>,
            U : StorageMut<P>
    {
        let lhs = self;
        let (lhs_stride, lhs_roi) = crate::image::ipputils::step_and_size_for_image(lhs);
        let (rhs_stride, rhs_roi) = crate::image::ipputils::step_and_size_for_image(rhs);
        let (dst_stride, dst_roi) = crate::image::ipputils::step_and_size_for_image(dst);
        unsafe {
            if lhs.pixel_is::<f32>() {
                let ans = crate::foreign::ipp::ippi::ippiAdd_32f_C1R(
                    mem::transmute(lhs.as_ptr()),
                    lhs_stride,
                    mem::transmute(rhs.as_ptr()),
                    rhs_stride,
                    mem::transmute(dst.as_mut_ptr()),
                    dst_stride,
                    lhs_roi
                );
                assert!(ans == 0);
                return;
            }
        }
        unimplemented!();
    }

    #[cfg(feature="ipp")]
    pub fn sub_to<T, U>(&self, rhs : &Image<P, T>, dst : &mut Image<P, U>)
        where
            T : Storage<P>,
            U : StorageMut<P>
    {
        let lhs = self;
        let (lhs_stride, lhs_roi) = crate::image::ipputils::step_and_size_for_image(lhs);
        let (rhs_stride, rhs_roi) = crate::image::ipputils::step_and_size_for_image(rhs);
        let (dst_stride, dst_roi) = crate::image::ipputils::step_and_size_for_image(dst);
        let scale = 0;
        unsafe {
            if lhs.pixel_is::<u8>() {
                let ans = crate::foreign::ipp::ippi::ippiSub_8u_C1RSfs(
                    mem::transmute(lhs.as_ptr()),
                    lhs_stride,
                    mem::transmute(rhs.as_ptr()),
                    rhs_stride,
                    mem::transmute(dst.as_mut_ptr()),
                    dst_stride,
                    lhs_roi,
                    scale
                );
                assert!(ans == 0);
                return;
            }
            if lhs.pixel_is::<f32>() {
                let ans = crate::foreign::ipp::ippi::ippiSub_32f_C1R(
                    mem::transmute(lhs.as_ptr()),
                    lhs_stride,
                    mem::transmute(rhs.as_ptr()),
                    rhs_stride,
                    mem::transmute(dst.as_mut_ptr()),
                    dst_stride,
                    lhs_roi
                );
                assert!(ans == 0);
                return;
            }
        }
        unimplemented!()
    }
}

impl<P, S, T> AddAssign<&Image<P, T>> for Image<P, S> 
where
    P : Pixel + AddAssign,
    S : StorageMut<P>,
    T : Storage<P>

{
    fn add_assign(&mut self, rhs: &Image<P, T>) {

        assert!(self.shape() == rhs.shape());

        #[cfg(feature="ipp")]
        unsafe {

            let (src_dst_byte_stride, roi) = crate::image::ipputils::step_and_size_for_image(self);
            let rhs_byte_stride = crate::image::ipputils::byte_stride_for_image(&rhs);

            let scale_factor = 1;

            if self.pixel_is::<u8>() {
                let ans = crate::foreign::ipp::ippi::ippiAdd_8u_C1IRSfs(
                    mem::transmute(rhs.as_ptr()),
                    rhs_byte_stride,
                    mem::transmute(self.as_ptr()),
                    src_dst_byte_stride,
                    roi,
                    scale_factor
                );
                assert!(ans == 0);
                return;
            }

            if self.pixel_is::<f32>() {
                let ans = crate::foreign::ipp::ippi::ippiAdd_32f_C1IR(
                    mem::transmute(rhs.as_ptr()),
                    rhs_byte_stride,
                    mem::transmute(self.as_mut_ptr()),
                    src_dst_byte_stride,
                    roi
                );
                assert!(ans == 0);
                return;
            }
        }

        // Unsafe required because we cannot specify that Self : 'a using the trait signature.
        for (out, input) in self.pixels_mut(1).zip(rhs.pixels(1)) {
            *out += *input;
        }

    }

}

#[test]
fn sub_assign() {
    let mut a = ImageBuf::<u8>::new_constant(10, 10, 2);
    let mut b = ImageBuf::<u8>::new_constant(10, 10, 1);
    a -= &b;
    println!("{:?}", a[(0usize,0usize)]);
}

impl<P, S, T> SubAssign<&Image<P, T>> for Image<P, S> 
where
    P : Pixel + SubAssign,
    S : StorageMut<P>,
    T : Storage<P>,
{

    fn sub_assign(&mut self, rhs: &Image<P, T>) {

        assert!(self.shape() == rhs.shape());

        #[cfg(feature="ipp")]
        unsafe {

            let (src_dst_byte_stride, roi) = crate::image::ipputils::step_and_size_for_image(self);
            let rhs_byte_stride = crate::image::ipputils::byte_stride_for_image(&rhs);

            let scale_factor = 0;
            if self.pixel_is::<u8>() {
                let ans = crate::foreign::ipp::ippi::ippiSub_8u_C1IRSfs(
                    mem::transmute(rhs.as_ptr()),
                    rhs_byte_stride,
                    mem::transmute(self.as_ptr()),
                    src_dst_byte_stride,
                    roi,
                    scale_factor
                );
                assert!(ans == 0);
                return;
            }

            if self.pixel_is::<i16>() {
                let ans = crate::foreign::ipp::ippi::ippiSub_16s_C1IRSfs(
                    mem::transmute(rhs.as_ptr()),
                    rhs_byte_stride,
                    mem::transmute(self.as_ptr()),
                    src_dst_byte_stride,
                    roi,
                    scale_factor
                );
                assert!(ans == 0);
                return;
            }

            if self.pixel_is::<f32>() {
                let ans = crate::foreign::ipp::ippi::ippiSub_32f_C1IR(
                    mem::transmute(rhs.as_ptr()),
                    rhs_byte_stride,
                    mem::transmute(self.as_ptr()),
                    src_dst_byte_stride,
                    roi
                );
                assert!(ans == 0);
                return;
            }
        }

        //for (out, input) in self.pixels_mut(1).zip(rhs.pixels(1)) {
        //    *out -= *input;
        //}
        unimplemented!()

    }

}

impl<P, S, T> MulAssign<&Image<P, T>> for Image<P, S> 
where
    P : Pixel + MulAssign,
    S : StorageMut<P>,
    T : Storage<P>,
{
    fn mul_assign(&mut self, rhs: &Image<P, T>) {
        assert!(self.shape() == rhs.shape());
        #[cfg(feature="ipp")]
        unsafe {
            let (src_dst_byte_stride, roi) = crate::image::ipputils::step_and_size_for_image(self);
            let rhs_byte_stride = crate::image::ipputils::byte_stride_for_image(&rhs);
            let scale_factor = 1;
            if self.pixel_is::<u8>() {
                let ans = crate::foreign::ipp::ippi::ippiMul_8u_C1IRSfs(
                    mem::transmute(rhs.as_ptr()),
                    rhs_byte_stride,
                    mem::transmute(self.as_ptr()),
                    src_dst_byte_stride,
                    roi,
                    scale_factor
                );
                assert!(ans == 0);
                return;
            }

            if self.pixel_is::<f32>() {
                let ans = crate::foreign::ipp::ippi::ippiMul_32f_C1IR(
                    mem::transmute(rhs.as_ptr()),
                    rhs_byte_stride,
                    mem::transmute(self.as_ptr()),
                    src_dst_byte_stride,
                    roi
                );
                assert!(ans == 0);
                return;
            }
        }

        for (out, input) in self.pixels_mut(1).zip(rhs.pixels(1)) {
            *out *= *input;
        }
    }

}

impl<P, S, T> DivAssign<&Image<P, T>> for Image<P, S> 
where
    P : Pixel + DivAssign,
    S : StorageMut<P>,
    T : Storage<P>,
{
    fn div_assign(&mut self, rhs: &Image<P, T>) {
        assert!(self.shape() == rhs.shape());
        #[cfg(feature="ipp")]
        unsafe {
            let (src_dst_byte_stride, roi) = crate::image::ipputils::step_and_size_for_image(self);
            let rhs_byte_stride = crate::image::ipputils::byte_stride_for_image(&rhs);
            let scale_factor = 1;
            if self.pixel_is::<u8>() {
                let ans = crate::foreign::ipp::ippi::ippiDiv_8u_C1IRSfs(
                    mem::transmute(rhs.as_ptr()),
                    rhs_byte_stride,
                    mem::transmute(self.as_ptr()),
                    src_dst_byte_stride,
                    roi,
                    scale_factor
                );
                assert!(ans == 0);
                return;
            }

            if self.pixel_is::<f32>() {
                let ans = crate::foreign::ipp::ippi::ippiDiv_32f_C1IR(
                    mem::transmute(rhs.as_ptr()),
                    rhs_byte_stride,
                    mem::transmute(self.as_ptr()),
                    src_dst_byte_stride,
                    roi
                );
                assert!(ans == 0);
                return;
            }
        }

        for (out, input) in self.pixels_mut(1).zip(rhs.pixels(1)) {
            *out /= *input;
        }
    }

}

/*// Options: saturate, reflect
pub trait SignedToUnsignedConversion {

}

// Options: flatten, shrink
pub trait UnsignedToFloatConversion {

}

// Options: stretch, preserve
pub trait FloatToUnsignedConversion {

}*/

/*pub trait BinaryFloatOp<O> {

    // If self contains vertical filter gradients, and other horizontal filter gradients, then
    // atan_assign calculates atan2(self/other).
    // fn atan2_assign(&mut self, rhs : O);

    // Can also be implemented for integers.
    fn abs_diff_assign(&mut self, rhs : O);

    fn add_weighted_assign(&mut self, rhs : O, by : f32);

}

pub trait BinaryIntegerOp<O> {

    fn abs_diff_assign(&mut self, rhs : O);

}*/

impl<P, S> Image<P, S>
where
    P : UnsignedPixel,
    S : StorageMut<P>,
{

    pub fn abs_diff_assign<T>(&mut self, rhs : &Image<P, T>) 
    where
        T : Storage<P>
    {
        #[cfg(feature="ipp")]
        unsafe {
            let mut copy = self.clone_owned();
            let (copy_step, copy_roi) = crate::image::ipputils::step_and_size_for_image(&copy.full_window());
            let (rhs_step, rhs_roi) = crate::image::ipputils::step_and_size_for_image(&rhs);
            let (this_step, this_roi) = crate::image::ipputils::step_and_size_for_image(&self);
            if self.pixel_is::<u8>() {
                let ans = crate::foreign::ipp::ippcv::ippiAbsDiff_8u_C1R(
                    mem::transmute(copy.full_window().as_ptr()),
                    copy_step,
                    mem::transmute(rhs.as_ptr()),
                    rhs_step,
                    mem::transmute(self.as_mut_ptr()),
                    this_step,
                    mem::transmute(this_roi)
                );
                assert!(ans == 0);
                return;
            }
        }
        unimplemented!()
    }

}

/*impl<'a, N> BinaryFloatOp<Window<'a, N>> for WindowMut<'a, N>
where
    N : Div<Output=N> + Mul<Output=N> + num_traits::Float + AddAssign + Add<Output=N> + Debug + Scalar + Copy + Default + Any + Sized + From<f32>
{

    // Sets self to (1. - by)*self + by*other
    fn add_weighted_assign(&mut self, rhs : Window<'a, N>, by : f32) {

        assert!(self.shape() == rhs.shape());
        assert!(by >= 0. && by <= 1.);

        #[cfg(feature="ipp")]
        unsafe {
            let (rhs_step, rhs_roi) = crate::image::ipputils::step_and_size_for_image(&rhs);
            let (this_step, this_roi) = crate::image::ipputils::step_and_size_for_image(&self);
            if self.pixel_is::<f32>() {
                let ans = crate::foreign::ipp::ippcv::ippiAddWeighted_32f_C1IR(
                    mem::transmute(rhs.as_ptr()),
                    rhs_step,
                    mem::transmute(self.as_mut_ptr()),
                    this_step,
                    mem::transmute(rhs_roi),
                    by
                );
                assert!(ans == 0);
                return;
            }
        }

        let by_compl : N = From::from(1.0f32 - by);
        let by : N = From::from(by);
        unsafe {
            for (out, input) in mem::transmute::<_, &'a mut WindowMut<'a, N>>(self).pixels_mut(1).zip(rhs.pixels(1)) {
                *out = by_compl*(*out) + by*(*input);
            }
        }
    }

    fn abs_diff_assign(&mut self, rhs : Window<'a, N>) {

        #[cfg(feature="ipp")]
        unsafe {
            let mut copy = self.clone_owned();
            let (copy_step, copy_roi) = crate::image::ipputils::step_and_size_for_image(&copy.full_window());
            let (rhs_step, rhs_roi) = crate::image::ipputils::step_and_size_for_image(&rhs);
            let (this_step, this_roi) = crate::image::ipputils::step_and_size_for_image(&self);
            if self.pixel_is::<f32>() {
                let ans = crate::foreign::ipp::ippcv::ippiAbsDiff_32f_C1R(
                    mem::transmute(copy.full_window().as_ptr()),
                    copy_step,
                    mem::transmute(rhs.as_ptr()),
                    rhs_step,
                    mem::transmute(self.as_mut_ptr()),
                    this_step,
                    mem::transmute(this_roi)
                );
                assert!(ans == 0);
                return;
            }
        }

        unimplemented!()
    }

}*/

/*/// A bit image uses each byte to represent the pixel values of 8 contiguous
/// pixels. It is the cheapest dense image representation. (Contrast with ByteImage,
/// which uses a u8 to represent grayscale values; and BinaryImage, which wraps a ByteImage
/// to represent zero vs. nonzero elements.
struct BitImage {

    buf : Vec<u8>,

    width : usize
}

struct BinaryWindow {

}

impl BinaryWindow {

    pub fn local_sums(&self) -> Image<u8> {
        self.buf.iter().for_each(|px|  px.count_ones() );
        // leading_zeros
        // trailing_zeros
        // leading_ones
        // trailing_ones
        // swap_bytes
        // checked_add
    }

}*/

/*
For inplace operations, the trait cannot be implemented for Window, only WindowMut and Image (by &mut).
Perhaps refactor into BinaryOpInplace
pub trait BinaryOp {
    and
    or
    xor
    not
    scalar_and
    scalar_or
    scalar_xor
    and_mut
    or_mut
    xor_mut
    not_mut
    scalar_and_mut
    scalar_or_mut
    scalar_xor_mut
}
*/

impl<S> Image<u8, S>
where
    S : Storage<u8>
{

    // Equality is different in that we do not want to allocate a new buffer to accumulate individual pixel differences / compare tags.
    pub fn eq<T>(&self, other : &Image<u8, T>) -> bool
    where
        T : Storage<u8>
    {
        self.pixels(1).zip(other.pixels(1)).all(|(a, b)| *a == *b )
    }

    pub fn and_to<T, U>(&self, other : &Image<u8, T>, dst : &mut Image<u8, U>) 
    where
        T : Storage<u8>,
        U : StorageMut<u8>
    {

        #[cfg(feature="ipp")]
        unsafe {
            let (src_stride, src_roi) = crate::image::ipputils::step_and_size_for_image(self);
            let (other_stride, other_roi) = crate::image::ipputils::step_and_size_for_image(other);
            let (dst_stride, dst_roi) = crate::image::ipputils::step_and_size_for_image(&dst);
            let ans = crate::foreign::ipp::ippi::ippiAnd_8u_C1R(
                self.as_ptr(),
                src_stride,
                other.as_ptr(),
                other_stride,
                dst.as_mut_ptr(),
                dst_stride,
                dst_roi
            );
            assert!(ans == 0);
            return;
        }
        unimplemented!();
    }

    fn or_to<T, U>(&self, other : &Image<u8, T>, dst : &mut Image<u8, U>) 
    where
        T : Storage<u8>,
        U : StorageMut<u8>
    {

        #[cfg(feature="ipp")]
        unsafe {
            let (src_stride, src_roi) = crate::image::ipputils::step_and_size_for_image(self);
            let (other_stride, other_roi) = crate::image::ipputils::step_and_size_for_image(other);
            let (dst_stride, dst_roi) = crate::image::ipputils::step_and_size_for_image(&dst);
            let ans = crate::foreign::ipp::ippi::ippiOr_8u_C1R(
                self.as_ptr(),
                src_stride,
                other.as_ptr(),
                other_stride,
                dst.as_mut_ptr(),
                dst_stride,
                dst_roi
            );
            assert!(ans == 0);
            return;
        }
        unimplemented!();
    }

    pub fn xor_to<T, U>(&self, other : &Image<u8, T>, dst : &mut Image<u8, U>) 
    where
        T : Storage<u8>,
        U : StorageMut<u8>
    {

        #[cfg(feature="ipp")]
        unsafe {
            let (src_stride, src_roi) = crate::image::ipputils::step_and_size_for_image(self);
            let (other_stride, other_roi) = crate::image::ipputils::step_and_size_for_image(other);
            let (dst_stride, dst_roi) = crate::image::ipputils::step_and_size_for_image(&dst);
            let ans = crate::foreign::ipp::ippi::ippiXor_8u_C1R(
                self.as_ptr(),
                src_stride,
                other.as_ptr(),
                other_stride,
                dst.as_mut_ptr(),
                dst_stride,
                dst_roi
            );
            assert!(ans == 0);
            return;
        }
        unimplemented!();
    }

}

impl<S> Image<u8, S>
where
    S : StorageMut<u8>
{

    pub fn and_assign<T>(&mut self, other : &Image<u8, T>) 
    where
        T : Storage<u8>,
    {

        #[cfg(feature="ipp")]
        unsafe {
            let (this_stride, this_roi) = crate::image::ipputils::step_and_size_for_image(self);
            let (other_stride, other_roi) = crate::image::ipputils::step_and_size_for_image(other);
            let ans = crate::foreign::ipp::ippi::ippiAnd_8u_C1IR(
                other.as_ptr(),
                other_stride,
                self.as_mut_ptr(),
                this_stride,
                this_roi
            );
            assert!(ans == 0);
            return;
        }
        unimplemented!();
    }
    
    pub fn xor_assign<T>(&mut self, other : &Image<u8, T>) 
    where
        T : Storage<u8>,
    {

        #[cfg(feature="ipp")]
        unsafe {
            let (this_stride, this_roi) = crate::image::ipputils::step_and_size_for_image(self);
            let (other_stride, other_roi) = crate::image::ipputils::step_and_size_for_image(other);
            let ans = crate::foreign::ipp::ippi::ippiXor_8u_C1IR(
                other.as_ptr(),
                other_stride,
                self.as_mut_ptr(),
                this_stride,
                this_roi
            );
            assert!(ans == 0);
            return;
        }
        unimplemented!();
    }
    
    // The not op is analogous to the gray::invert and float::invert ops.
    pub fn not_mut(&mut self) {

        #[cfg(feature="ipp")]
        unsafe {
            let (dst_stride, dst_roi) = crate::image::ipputils::step_and_size_for_image(self);
            let ans = crate::foreign::ipp::ippi::ippiNot_8u_C1IR(
                self.as_mut_ptr(),
                dst_stride,
                dst_roi
            );
            assert!(ans == 0);
            return;
        }

        self.foreach_pixel(1, |px : &mut u8| { *px = if *px == 0 { 255 } else { 0 } });
    }
    
}

/*IppStatus ippiAnd_<mod> ( const Ipp<datatype>* pSrc1 , int src1Step , const Ipp<datatype>*
pSrc2 , int src2Step , Ipp<datatype>* pDst , int dstStep , IppiSize roiSize );

IppStatus ippiAndC_<mod> ( const Ipp<datatype>* pSrc , int srcStep , Ipp<datatype> value ,
Ipp<datatype>* pDst , int dstStep , IppiSize roiSize );

IppStatus ippiOr_<mod> ( const Ipp<datatype>* pSrc1 , int src1Step , const Ipp<datatype>*
pSrc2 , int src2Step , Ipp<datatype>* pDst , int dstStep , IppiSize roiSize );

IppStatus ippiXor_<mod> ( const Ipp<datatype>* pSrc1 , int src1Step , const Ipp<datatype>*
pSrc2 , int src2Step , Ipp<datatype>* pDst , int dstStep , IppiSize roiSize );

IppStatus ippiOrC_<mod> ( const Ipp<datatype>* pSrc , int srcStep , Ipp<datatype> value ,
Ipp<datatype>* pDst , int dstStep , IppiSize roiSize );

IppStatus ippiXor_<mod> ( const Ipp<datatype>* pSrc1 , int src1Step , const Ipp<datatype>*
pSrc2 , int src2Step , Ipp<datatype>* pDst , int dstStep , IppiSize roiSize );

IppStatus ippiXorC_<mod> ( const Ipp<datatype>* pSrc , int srcStep , const Ipp<datatype>
value[3] , Ipp<datatype>* pDst , int dstStep , IppiSize roiSize );

IppStatus ippiNot_<mod> ( const Ipp8u* pSrc , int srcStep , Ipp8u* pDst , int dstStep ,
IppiSize roiSize );*/


