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

/*/// Stretch the image profile such that the gray levels make the best use of the dynamic range.
/// (Myler & Weeks, 1993). gray_max and gray_min are min and max intensities at the current image.
/// Stretching makes better use of the gray value space.
pub fn stretch() {
    (255. / (gray_max - gray_min )) * (px - gray_min)
}

/// Equalization makes the histogram more flat.
pub fn equalize() {
    // (1) Calculate image CDF
    // (2) Order all available gray levels
    // (3) Divide CDF by total number of pixels (normalize by n at last bin). This will preserve order of gray levels.
    // (4) For all pixels: image[px] = new_cdf[old_order_of_px].
    // All pixels at last bin will be mapped to same value. All others will be brought closer by a ratio to it.
    // Basically transform the CDF to a ramp and the histogram to a uniform.
*/

pub fn point_div_mut<'a, N>(win : &'a Window<'a, N>, by : N, dst : &'a mut WindowMut<'a, N>)
where
    N : Scalar + Div<Output=N> + Copy + Any + Debug + Default
{
    dst.pixels_mut(1).zip(win.pixels(1)).for_each(|(dst, src)| *dst = *src / by );
}

/// Normalizes the image relative to the max(infinity) norm. This limits values to [0., 1.]
pub fn normalize_max_mut<'a, N>(win : &'a Window<'a, N>, dst : &'a mut WindowMut<'a, N>)
where
    N : Div<Output=N> + Copy + Ord + Any + Debug + Default,
    u8 : AsPrimitive<N>
{
    let max = global::max(win);
    point_div_mut(win, max, dst);
}

/// Normalizes the image, so that the sum of its pixels is 1.0. Useful for convolution filters,
/// which must preserve the original image norm so that convolution output does not overflow its integer
/// or float maximum.
pub fn normalize_unit_mut<'a, N>(win : &'a Window<'a, N>, dst : &'a mut WindowMut<'a, N>)
where
    N : Div<Output=N> + Copy + PartialOrd + Serialize + DeserializeOwned + Any + Debug + Zero + From<f64> + Default,
    f64 : From<N>
{
    let sum = global::sum(win, 1);
    point_div_mut(win, sum, dst);
}

pub trait PointOp<N> {

    // Increase/decrease brightness (add positive or negative scalar)
    fn brightess_mut(&mut self, by : N);

    // Apply contrast enhancement (multiply by scalar > 1.0)
    fn contrast_mut(&mut self, by : N);

}

impl <'a, N> PointOp<N> for WindowMut<'a, N>
where
    N : MulAssign + AddAssign + Debug + Scalar + Copy + Default + Any + Sized
{

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
        }

        // self.pixels_mut(1).for_each(|p| *p *= by );
        unimplemented!()
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
                let ans = crate::foreign::ipp::ippi::ippiAdd_8u_C1RSfs(
                    mem::transmute(self.as_ptr()),
                    src_dst_byte_stride,
                    mem::transmute(rhs.as_ptr()),
                    rhs_byte_stride,
                    mem::transmute(self.as_mut_ptr()),
                    src_dst_byte_stride,
                    roi,
                    scale_factor
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

impl<'a, N> SubAssign<Window<'a, N>> for WindowMut<'a, N>
where
    N : Scalar + Copy + Clone + Debug + SubAssign + Default + Any + 'static
{

    fn sub_assign(&mut self, rhs: Window<'a, N>) {

        assert!(self.shape() == rhs.shape());

        #[cfg(feature="ipp")]
        unsafe {

            let (src_dst_byte_stride, roi) = crate::image::ipputils::step_and_size_for_window_mut(&self);
            let rhs_byte_stride = crate::image::ipputils::byte_stride_for_window(&rhs);

            let scale_factor = 1;
            if self.pixel_is::<u8>() {
                let ans = crate::foreign::ipp::ippi::ippiSub_8u_C1RSfs(
                    mem::transmute(self.as_ptr()),
                    src_dst_byte_stride,
                    mem::transmute(rhs.as_ptr()),
                    rhs_byte_stride,
                    mem::transmute(self.as_mut_ptr()),
                    src_dst_byte_stride,
                    roi,
                    scale_factor
                );
                assert!(ans == 0);
                return;
            }
        }

        // Unsafe required because we cannot specify that Self : 'a using the trait signature.
        unsafe {
            for (out, input) in mem::transmute::<_, &'a mut WindowMut<'a, N>>(self).pixels_mut(1).zip(rhs.pixels(1)) {
                *out -= *input;
            }
        }

    }

}

/*

Alternatively, match on the type and take the function pointer to the corresponding C function.

// Brighten
IppStatus ippiAddC_<mod> ( const Ipp<datatype>* pSrc , int srcStep , Ipp<datatype> value ,
Ipp<datatype>* pDst , int dstStep , IppiSize roiSize , int scaleFactor );

IppStatus ippiMul_<mod> ( const Ipp<datatype>* pSrc1 , int src1Step , const Ipp<datatype>*
pSrc2 , int src2Step , Ipp<datatype>* pDst , int dstStep , IppiSize roiSize , int
scaleFactor );

// Contrast-enhancement.
IppStatus ippiMulC_<mod> ( const Ipp<datatype>* pSrc , int srcStep , Ipp<datatype> value ,
Ipp<datatype>* pDst , int dstStep , IppiSize roiSize , int scaleFactor );

IppStatus ippiAbs_<mod> ( const Ipp<datatype>* pSrc , int srcStep , Ipp<datatype>* pDst ,
int dstStep , IppiSize roiSize );

IppStatus ippiAbsDiff_<mod> ( const Ipp<datatype>* pSrc1 , int src1Step , const
Ipp<datatype>* pSrc2 , int src2Step , Ipp<datatype>* pDst , int dstStep , IppiSize
roiSize );

IppStatus ippiSqr_<mod> ( const Ipp<datatype>* pSrc , int srcStep , Ipp<datatype>* pDst ,
int dstStep , IppiSize roiSize , int scaleFactor );

IppStatus ippiSqrt_<mod> ( const Ipp<datatype>* pSrc , int srcStep , Ipp<datatype>* pDst ,
int dstStep , IppiSize roiSize , int scaleFactor );

IppStatus ippiLn_<mod> ( const Ipp<datatype>* pSrc , int srcStep , Ipp<datatype>* pDst , int
dstStep , IppiSize roiSize , int scaleFactor );

IppStatus ippiExp_<mod> ( const Ipp<datatype>* pSrc , int srcStep , Ipp<datatype>* pDst ,
int dstStep , IppiSize roiSize , int scaleFactor );

IppStatus ippiRShiftC_<mod> ( const Ipp<datatype>* pSrc , int srcStep , Ipp32u value ,
Ipp<datatype>* pDst , int dstStep , IppiSize roiSize );

IppStatus ippiLShiftC_<mod> ( const Ipp<datatype>* pSrc , int srcStep , Ipp32u value ,
Ipp<datatype>* pDst , int dstStep , IppiSize roiSize );
*/
