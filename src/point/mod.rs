use std::cmp::{Ord, PartialOrd, Eq};
use std::ops::*;
use crate::image::*;
use crate::global;
use serde::{Serialize, de::DeserializeOwned};
use std::any::Any;
use std::fmt::Debug;
use num_traits::*;
use nalgebra::Scalar;
use std::mem;
use num_traits::Float;
use crate::gray::Foreground;

/*
Perhaps the normalize_bound method could be part of a Bounded trait, for images
with an upper pixel bound.
*/

// a < b ? a else b
pub fn pairwise_compare<N>(a : &Window<N>, b : &Window<N>, mut dst : WindowMut<N>, max : bool)
where
    N : Scalar + Copy + Debug + Clone + Default
{

    #[cfg(feature="ipp")]
    unsafe {
        let (a_stride, a_roi) = crate::image::ipputils::step_and_size_for_window(a);
        let (b_stride, b_roi) = crate::image::ipputils::step_and_size_for_window(b);
        let (dst_stride, dst_roi) = crate::image::ipputils::step_and_size_for_window_mut(&dst);

        if a.pixel_is::<u8>() {
            let ans = if max {
                crate::foreign::ipp::ippi::ippiMaxEvery_8u_C1R(
                    mem::transmute(a.as_ptr()),
                    a_stride,
                    mem::transmute(b.as_ptr()),
                    b_stride,
                    mem::transmute(dst.as_mut_ptr()),
                    dst_stride,
                    a_roi
                )
            } else {
                crate::foreign::ipp::ippi::ippiMinEvery_8u_C1R(
                    mem::transmute(a.as_ptr()),
                    a_stride,
                    mem::transmute(b.as_ptr()),
                    b_stride,
                    mem::transmute(dst.as_mut_ptr()),
                    dst_stride,
                    a_roi
                )
            };
            assert!(ans == 0);
            return;
        }

        if a.pixel_is::<f32>() {
            let ans = if max {
                crate::foreign::ipp::ippi::ippiMaxEvery_32f_C1R(
                    mem::transmute(a.as_ptr()),
                    a_stride,
                    mem::transmute(b.as_ptr()),
                    b_stride,
                    mem::transmute(dst.as_mut_ptr()),
                    dst_stride,
                    a_roi
                )
            } else {
                crate::foreign::ipp::ippi::ippiMinEvery_32f_C1R(
                    mem::transmute(a.as_ptr()),
                    a_stride,
                    mem::transmute(b.as_ptr()),
                    b_stride,
                    mem::transmute(dst.as_mut_ptr()),
                    dst_stride,
                    a_roi
                )
            };
            assert!(ans == 0);
            return;
        }
    }

    unimplemented!()
}

/*/// Stretch the image profile such that the gray levels make the best use of the dynamic range.
/// (Myler & Weeks, 1993). gray_max and gray_min are min and max intensities at the current image.
/// Stretching makes better use of the gray value space.
pub fn stretch() {
    (255. / (gray_max - gray_min )) * (px - gray_min)
}

/// Equalization makes the histogram more flat.
pub fn equalize(hist, src, dst) {
    // Based on Myler & Weeks, 1993:
    // (1) Calculate image CDF
    // (2) Order all available gray levels
    // (3) Divide CDF by total number of pixels (normalize by n at last bin). This will preserve order of gray levels.
    // (4) For all pixels: image[px] = new_cdf[old_order_of_px].
    // All pixels at last bin will be mapped to same value. All others will be brought closer by a ratio to it.
    // Basically transform the CDF to a ramp and the histogram to a uniform.
    for i in 0..256 {

        // Sum histogram entries up to ith index (calc cumulative)
        sum = 0.;
        for j in 0..(i+1) {
            sum += hist[j]
        }
        histeq[i] = (255*sum+0.5) as i32;
    }

    for r in 0..img.height() {
        for c in 0..img.width() {
            dst[(r, c)] = histeq[src[(r, c)] as usize];
        }
    }
}

*/

pub fn point_div_inplace<'a, N>(dst : &'a mut WindowMut<'a, N>, by : N)
where
    N : Scalar + Div<Output=N> + Copy + Any + Debug + Default
{
    dst.pixels_mut(1).for_each(|dst| *dst = *dst / by );
}

pub fn point_div_mut<'a, N>(win : &'a Window<'a, N>, by : N, dst : &'a mut WindowMut<'a, N>)
where
    N : Scalar + Div<Output=N> + Copy + Any + Debug + Default
{
    dst.pixels_mut(1).zip(win.pixels(1)).for_each(|(dst, src)| *dst = *src / by );
}

/// Normalizes the image relative to the max(infinity) norm. This limits values to [0., 1.]
pub fn normalize_max_mut<'a, N>(win : &'a Window<'a, N>, dst : &'a mut WindowMut<'a, N>) -> N
where
    N : Div<Output=N> + Copy + PartialOrd + Any + Debug + Default,
    u8 : AsPrimitive<N>
{
    let max = global::max(win);
    point_div_mut(win, max, dst);
    max
}

/// Normalizes the image, so that the sum of its pixels is 1.0. Useful for convolution filters,
/// which must preserve the original image norm so that convolution output does not overflow its integer
/// or float maximum.
pub fn normalize_unit_mut<'a, N>(win : &'a Window<'a, N>, dst : &'a mut WindowMut<'a, N>) -> N
where
    N : Div<Output=N> + Copy + PartialOrd + Serialize + DeserializeOwned + Any + Debug + Zero + From<f32> + Default,
    f32 : From<N>
{
    let sum = global::sum(win, 1);
    point_div_mut(win, sum, dst);
    sum
}

/// Normalizes the image relative to the max(infinity) norm. This limits values to [0., 1.]
/// This only really makes sense for float images, since integer bounded images (u8, i32)
/// will generate divisions at are either 0 or 1, effectively creating a dark image.
pub fn normalize_max_inplace<'a, N>(dst : &'a mut WindowMut<'a, N>) -> N
where
    N : Div<Output=N> + Copy + PartialOrd + Any + Debug + Default,
    u8 : AsPrimitive<N>
{
    unsafe {
        let max = global::max(mem::transmute(dst as *mut _));
        point_div_inplace(dst, max);
        max
    }
}

/// Normalizes the image, so that the sum of its pixels is 1.0. Useful for convolution filters,
/// which must preserve the original image norm so that convolution output does not overflow its integer
/// or float maximum.
pub fn normalize_unit_inplace<'a, N>(dst : &'a mut WindowMut<'a, N>) -> N
where
    N : Div<Output=N> + Copy + PartialOrd + Serialize + DeserializeOwned + Any + Debug + Zero + From<f32> + Default,
    f32 : From<N>
{
    unsafe {
        let sum = global::sum(mem::transmute(dst as *mut _), 1);
        point_div_inplace(dst, sum);
        sum
    }
}

pub trait PointOp<N> {

    // Calculate 1/x
    // invert_mut(.)

    // Calculate T::max(.) - x
    // fn complement_mut(.)

    // fn gamma_mut(.)

    /*IppStatus ippiSqr_<mod> ( const Ipp<datatype>* pSrc , int srcStep , Ipp<datatype>* pDst ,
    int dstStep , IppiSize roiSize , int scaleFactor );

    IppStatus ippiSqrt_<mod> ( const Ipp<datatype>* pSrc , int srcStep , Ipp<datatype>* pDst ,
    int dstStep , IppiSize roiSize , int scaleFactor );

    IppStatus ippiLn_<mod> ( const Ipp<datatype>* pSrc , int srcStep , Ipp<datatype>* pDst , int
    dstStep , IppiSize roiSize , int scaleFactor );

    IppStatus ippiExp_<mod> ( const Ipp<datatype>* pSrc , int srcStep , Ipp<datatype>* pDst ,
    int dstStep , IppiSize roiSize , int scaleFactor );*/

    fn abs_mut(&mut self);

    fn abs_diff_mut(&mut self, by : N);

    // Increase/decrease brightness (add positive or negative scalar). Perhaps call brighten_mut/brighten
    fn brightess_mut(&mut self, by : N);

    // Apply contrast enhancement (multiply by scalar > 1.0). Perhaps call enhance_contrast
    fn contrast_mut(&mut self, by : N);

    fn truncate_mut(&mut self, above : bool, val : N);

}

impl <N> PointOp<N> for WindowMut<'_, N>
where
    N : MulAssign + AddAssign + Debug + Scalar + Copy + Default + Any + Sized + PartialOrd + num_traits::Zero
{

    fn abs_mut(&mut self) {

        if self.pixel_is::<u8>() || self.pixel_is::<u64>() {
            return;
        }

        #[cfg(feature="ipp")]
        unsafe {

            let (byte_stride, roi) = crate::image::ipputils::step_and_size_for_window_mut(&self);

            if self.pixel_is::<f32>() {
                let ans = crate::foreign::ipp::ippi::ippiAbs_32f_C1IR(
                    mem::transmute(self.as_mut_ptr()),
                    byte_stride,
                    roi
                );
                assert!(ans == 0);
                return;
            }
        }

        unimplemented!()
    }

    fn abs_diff_mut(&mut self, by : N) {

        #[cfg(feature="ipp")]
        unsafe {
            let scale_factor = 1;
            let mut dst = self.clone_owned();
            let (byte_stride, roi) = crate::image::ipputils::step_and_size_for_window_mut(&self);
            let (dst_byte_stride, dst_roi) = crate::image::ipputils::step_and_size_for_window_mut(&dst.full_window_mut());
            if self.pixel_is::<f32>() {
                let ans = crate::foreign::ipp::ippcv::ippiAbsDiffC_32f_C1R(
                    mem::transmute(self.as_ptr()),
                    byte_stride,
                    mem::transmute(dst.full_window_mut().as_mut_ptr()),
                    dst_byte_stride,
                    mem::transmute(roi),
                    *mem::transmute::<_, &f32>(&by)
                );
                assert!(ans == 0);
                (&mut *(self as *mut WindowMut<'_, N>)).copy_from(&dst.full_window());
                return;
            }
        }

        unimplemented!()
    }

    fn brightess_mut(&mut self, by : N) {

        #[cfg(feature="ipp")]
        unsafe {
            let scale_factor = 1;
            let (byte_stride, roi) = crate::image::ipputils::step_and_size_for_window_mut(&self);

            if self.pixel_is::<u8>() {
                let ans = crate::foreign::ipp::ippi::ippiAddC_8u_C1IRSfs(
                    *mem::transmute::<_, &u8>(&by),
                    mem::transmute(self.as_mut_ptr()),
                    byte_stride,
                    roi,
                    scale_factor
                );
                assert!(ans == 0);
                return;
            }

            if self.pixel_is::<f32>() {
                let ans = crate::foreign::ipp::ippi::ippiAddC_32f_C1IR(
                    *mem::transmute::<_, &f32>(&by),
                    mem::transmute(self.as_mut_ptr()),
                    byte_stride,
                    roi
                );
                assert!(ans == 0);
                return;
            }
        }

        // self.pixels_mut(1).for_each(|p| *p += by );
        unimplemented!()
    }

    fn contrast_mut(&mut self, by : N) {

        #[cfg(feature="ipp")]
        unsafe {

            let (byte_stride, roi) = crate::image::ipputils::step_and_size_for_window_mut(&self);
            let scale_factor = 1;

            if self.pixel_is::<u8>() {
                let ans = crate::foreign::ipp::ippi::ippiMulC_8u_C1IRSfs(
                    *mem::transmute::<_, &u8>(&by),
                    mem::transmute(self.as_mut_ptr()),
                    byte_stride,
                    roi,
                    scale_factor
                );
                assert!(ans == 0);
                return;
            }

            if self.pixel_is::<i32>() {
                let ans = crate::foreign::ipp::ippi::ippiMulC_8u_C1IRSfs(
                    *mem::transmute::<_, &u8>(&by),
                    mem::transmute(self.as_mut_ptr()),
                    byte_stride,
                    roi,
                    scale_factor
                );
                assert!(ans == 0);
                return;
            }

            if self.pixel_is::<f32>() {
                let ans = crate::foreign::ipp::ippi::ippiMulC_32f_C1IR(
                    *mem::transmute::<_, &f32>(&by),
                    mem::transmute(self.as_mut_ptr()),
                    byte_stride,
                    roi
                );
                assert!(ans == 0);
                return;
            }
        }

        // self.pixels_mut(1).for_each(|p| *p *= by );
        unimplemented!()
    }

    fn truncate_mut(&mut self, above : bool, val : N) {
        unsafe {
            match above {
                true => {
                    mem::transmute::<_, &mut WindowMut<'_, N>>(self).pixels_mut(1).for_each(|px| if *px >= val { *px = N::zero(); });
                },
                false => {
                    mem::transmute::<_, &mut WindowMut<'_, N>>(self).pixels_mut(1).for_each(|px| if *px <= val { *px = N::zero(); });
                },
            }
        }
    }

}

impl<'a, N> AddAssign<Window<'a, N>> for WindowMut<'a, N>
where
    N : Scalar + Copy + Clone + Debug + AddAssign + Default + Any + 'static
{

    fn add_assign(&mut self, rhs: Window<'a, N>) {

        assert!(self.shape() == rhs.shape());

        #[cfg(feature="ipp")]
        unsafe {

            let (src_dst_byte_stride, roi) = crate::image::ipputils::step_and_size_for_window_mut(&self);
            let rhs_byte_stride = crate::image::ipputils::byte_stride_for_window(&rhs);

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
        unsafe {
            for (out, input) in mem::transmute::<_, &'a mut WindowMut<'a, N>>(self).pixels_mut(1).zip(rhs.pixels(1)) {
                *out += *input;
            }
        }

    }

}

impl<N> SubAssign<Window<'_, N>> for WindowMut<'_, N>
where
    N : Scalar + Copy + Clone + Debug + SubAssign + Default + Any + 'static
{

    fn sub_assign(&mut self, rhs: Window<'_, N>) {

        assert!(self.shape() == rhs.shape());

        #[cfg(feature="ipp")]
        unsafe {

            let (src_dst_byte_stride, roi) = crate::image::ipputils::step_and_size_for_window_mut(&self);
            let rhs_byte_stride = crate::image::ipputils::byte_stride_for_window(&rhs);

            let scale_factor = 1;
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

        // Unsafe required because we cannot specify that Self : 'a using the trait signature.
        unsafe {
            for (out, input) in mem::transmute::<_, &mut WindowMut<'_, N>>(self).pixels_mut(1).zip(rhs.pixels(1)) {
                *out -= *input;
            }
        }

    }

}

impl<N> MulAssign<Window<'_, N>> for WindowMut<'_, N>
where
    N : Scalar + MulAssign + Copy + Clone + Debug + AddAssign + Default + Any + 'static
{

    fn mul_assign(&mut self, rhs: Window<'_, N>) {
        assert!(self.shape() == rhs.shape());
        #[cfg(feature="ipp")]
        unsafe {
            let (src_dst_byte_stride, roi) = crate::image::ipputils::step_and_size_for_window_mut(&self);
            let rhs_byte_stride = crate::image::ipputils::byte_stride_for_window(&rhs);
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

        // Unsafe required because we cannot specify that Self : 'a using the trait signature.
        unsafe {
            for (out, input) in mem::transmute::<_, &'_ mut WindowMut<'_, N>>(self).pixels_mut(1).zip(rhs.pixels(1)) {
                *out *= *input;
            }
        }

    }

}

impl<N> DivAssign<Window<'_, N>> for WindowMut<'_, N>
where
    N : DivAssign + Scalar + Copy + Clone + Debug + AddAssign + Default + Any + 'static
{

    fn div_assign(&mut self, rhs: Window<'_, N>) {
        assert!(self.shape() == rhs.shape());
        #[cfg(feature="ipp")]
        unsafe {
            let (src_dst_byte_stride, roi) = crate::image::ipputils::step_and_size_for_window_mut(&self);
            let rhs_byte_stride = crate::image::ipputils::byte_stride_for_window(&rhs);
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

        // Unsafe required because we cannot specify that Self : 'a using the trait signature.
        unsafe {
            for (out, input) in mem::transmute::<_, &mut WindowMut<'_, N>>(self).pixels_mut(1).zip(rhs.pixels(1)) {
                *out /= *input;
            }
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

pub trait Pixel where Self : Scalar + Clone + Copy + Debug + Default + num_traits::Zero { }

impl Pixel for u8 { }

impl Pixel for u16 { }

impl Pixel for u32 { }

impl Pixel for i16 { }

impl Pixel for i32 { }

impl Pixel for i64 { }

impl Pixel for f32 { }

impl Pixel for f64 { }

pub trait UnsignedPixel where Self : Pixel { }

impl UnsignedPixel for u8 { }

impl UnsignedPixel for u16 { }

impl UnsignedPixel for u32 { }

pub trait SignedPixel where Self : Pixel { }

impl SignedPixel for i16 { }

impl SignedPixel for i32 { }

impl SignedPixel for i64 { }

pub trait FloatPixel where Self : Pixel { }

impl FloatPixel for f32 { }

impl FloatPixel for f64 { }

pub trait BinaryFloatOp<O> {

    // If self contains vertical filter gradients, and other horizontal filter gradients, then
    // atan_assign calculates atan2(self/other).
    fn atan2_assign(&mut self, rhs : O);

    // Can also be implemented for integers.
    fn abs_diff_assign(&mut self, rhs : O);

    fn add_weighted_assign(&mut self, rhs : O, by : f32);

}

pub trait BinaryIntegerOp<O> {

    fn abs_diff_assign(&mut self, rhs : O);

}

impl<'a, N> BinaryIntegerOp<Window<'a, N>> for WindowMut<'a, N>
where
    N : UnsignedPixel
{

    fn abs_diff_assign(&mut self, rhs : Window<'a, N>) {
        #[cfg(feature="ipp")]
        unsafe {
            let mut copy = self.clone_owned();
            let (copy_step, copy_roi) = crate::image::ipputils::step_and_size_for_window(&copy.full_window());
            let (rhs_step, rhs_roi) = crate::image::ipputils::step_and_size_for_window(&rhs);
            let (this_step, this_roi) = crate::image::ipputils::step_and_size_for_window_mut(&self);
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

impl<'a, N> BinaryFloatOp<Window<'a, N>> for WindowMut<'a, N>
where
    N : Div<Output=N> + Mul<Output=N> + num_traits::Float + AddAssign + Add<Output=N> + Debug + Scalar + Copy + Default + Any + Sized + From<f32>
{

    // Sets self to (1. - by)*self + by*other
    fn add_weighted_assign(&mut self, rhs : Window<'a, N>, by : f32) {

        assert!(self.shape() == rhs.shape());
        assert!(by >= 0. && by <= 1.);

        #[cfg(feature="ipp")]
        unsafe {
            let (rhs_step, rhs_roi) = crate::image::ipputils::step_and_size_for_window(&rhs);
            let (this_step, this_roi) = crate::image::ipputils::step_and_size_for_window_mut(&self);
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

    fn atan2_assign(&mut self, rhs : Window<'a, N>) {
        assert!(self.shape() == rhs.shape());
        unsafe {
            for (out, input) in mem::transmute::<_, &'a mut WindowMut<'a, N>>(self).pixels_mut(1).zip(rhs.pixels(1)) {
                *out = (*out).atan2(*input);
            }
        }
    }

    fn abs_diff_assign(&mut self, rhs : Window<'a, N>) {

        #[cfg(feature="ipp")]
        unsafe {
            let mut copy = self.clone_owned();
            let (copy_step, copy_roi) = crate::image::ipputils::step_and_size_for_window(&copy.full_window());
            let (rhs_step, rhs_roi) = crate::image::ipputils::step_and_size_for_window(&rhs);
            let (this_step, this_roi) = crate::image::ipputils::step_and_size_for_window_mut(&self);
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

}

