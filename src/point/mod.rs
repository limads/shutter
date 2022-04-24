use std::cmp::{Ord, Eq};
use std::ops::Div;
use crate::image::*;
use crate::global;
use serde::{Serialize, de::DeserializeOwned};
use std::any::Any;
use std::fmt::Debug;

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
    N : Div<Output=N> + Copy + Serialize + DeserializeOwned + Any + Debug + Eq
{
    dst.pixels_mut(1).zip(win.pixels(1)).for_each(|(dst, src)| *dst = *src / by );
}

/// Normalizes the image relative to the max(infinity) norm. This limits values to [0., 1.]
pub fn normalize_max_mut<'a, N>(win : &'a Window<'a, N>, dst : &'a mut WindowMut<'a, N>)
where
    N : Div<Output=N> + Copy + Ord + Serialize + DeserializeOwned + Any + Debug
{
    let max = global::max(win);
    point_div_mut(win, max, dst);
}

/*IppStatus ippiAdd_<mod> ( const Ipp<datatype>* pSrc1 , int src1Step , const Ipp<datatype>*
pSrc2 , int src2Step , Ipp<datatype>* pDst , int dstStep , IppiSize roiSize , int
scaleFactor );

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
