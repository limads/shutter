use crate::image::*;
pub use ripple::conv::*;
use std::mem;
use std::any::Any;
use serde::{Serialize, Deserialize, de::DeserializeOwned};
use num_traits::Zero;
use nalgebra::Scalar;
use std::iter::FromIterator;

fn baseline_convolution<N>(input : &Window<'_, N>, filter : &Window<'_, N>, mut out : WindowMut<'_, N>)
where
    N : Scalar + Copy + Default + Zero + Copy + Serialize + DeserializeOwned +
        Any + std::ops::Mul<Output = N> + std::ops::AddAssign
{
    assert!(out.height() == input.height() - filter.height() + 1);
    assert!(out.width() == input.width() - filter.width() + 1);
    assert!(filter.width() % 2 == 1);
    assert!(filter.height() % 2 == 1);

    // Actually half the kernel size minus one, since size % 2 must be 1
    let half_kh = filter.height() / 2;
    let half_kw = filter.width() / 2;

    for (ix_ci, center_i) in (half_kh..(input.height() - half_kh)).enumerate() {
        for (ix_cj, center_j) in (half_kw..(input.width() - half_kw)).enumerate() {
            out[(ix_ci, ix_cj)] = N::zero();
            for (ix_ki, ki) in ((center_i - half_kh)..(center_i + half_kh + 1)).enumerate() {
                for (ix_kj, kj) in ((center_j - half_kw)..(center_j + half_kw + 1)).enumerate() {
                    out[(ix_ci, ix_cj)] += input[(ki, kj)] * filter[(ix_ki, ix_kj)];
                }
            }
        }
    }
}

#[cfg(feature="ipp")]
unsafe fn ipp_convolution_f32(img : &Image<f32>, kernel : &Image<f32>, out : &mut Image<f32>) {

    use crate::foreign::ipp::ippi::*;
    use std::os::raw::c_int;

    assert!(out.width() == img.width() - kernel.width() - 1);
    assert!(out.height() == img.height() - kernel.height() - 1);

    // With IppiROIShape_ippiROIFull  convolution with zero padding is applied (result is nrow_img + nrow_kenel - 1).
    // With IppiROIShape_ippiROIValid convolution without zero padding is applied (result is nrow_img - nrow_kernel + 1)
    let alg_ty = IppAlgType_ippAlgAuto + IppiROIShape_ippiROIValid;

    let img_sz = crate::image::ipputils::image_size(img);
    let kernel_sz = crate::image::ipputils::image_size(kernel);
    let mut buf_sz : c_int = 0;
    let status = ippiConvGetBufferSize(
        img_sz,
        kernel_sz,
        crate::foreign::ipp::ippcore::IppDataType_ipp32f,
        1,
        alg_ty.clone() as i32,
        &mut buf_sz
    );
    assert!(status == 0 && buf_sz > 0);
    let mut conv_buffer : Vec<u8> = Vec::from_iter((0..buf_sz).map(|_| 0u8 ) );

    let status = ippiConv_32f_C1R(
        &img.buf[0] as *const _,
        (mem::size_of::<f32>() * img.width()) as c_int,
        img_sz,
        &kernel.buf[0] as *const _,
        (mem::size_of::<f32>() * kernel.width()) as c_int,
        kernel_sz,
        &mut out.buf[0] as *mut _,
        (mem::size_of::<f32>() * out.width()) as c_int,
        alg_ty as i32,
        &mut conv_buffer[0] as *mut _
    );
    assert!(status == 0);
}

impl<N> Convolve for Image<N>
where
    N : Scalar + Copy + Default + Zero + Copy + Serialize + DeserializeOwned +
        Any + std::ops::Mul<Output = N> + std::ops::AddAssign
{

    type Output = Image<N>;

    fn convolve_mut(&self, filter : &Self, out : &mut Self) {

        #[cfg(feature="ipp")]
        {
            if (&self[(0, 0)] as &dyn Any).is::<f32>() {
                unsafe {
                    ipp_convolution_f32(mem::transmute(self), mem::transmute(filter), mem::transmute(out));
                }
                return;
            }

            if (&self[(0, 0)] as &dyn Any).is::<f64>() {

            }
        }

        #[cfg(feature="mkl")]
        {
            /*if (&self[0] as &dyn Any).is::<f32>() {
                // Dispatch to MKL impl
            }

            if (&self[0] as &dyn Any).is::<f64>() {
                // Dispatch to MKL impl
            }*/
        }

        #[cfg(feature="opencv")]
        #[cfg(not(feature="ipp"))]
        {
            use opencv;

            assert!(filter.height() % 2 != 0 && filter.width() % 2 != 0 );
            let input : opencv::core::Mat = self.into();
            let kernel : opencv::core::Mat = filter.into();
            let mut flip_kernel = kernel.clone();
            opencv::core::flip(&kernel, &mut flip_kernel, -1).unwrap();
            let delta = 0.0;
            let mut output : opencv::core::Mat = out.into();
            opencv::imgproc::filter_2d(
                &input,
                &mut output,
                cvutils::get_cv_type::<N>(),
                &flip_kernel,
                opencv::core::Point2i::new(0, 0),
                delta,
                opencv::core::BORDER_DEFAULT
            ).unwrap();
            return;
        }

        baseline_convolution(&self.full_window(), &filter.full_window(), out.full_window_mut());
    }
}

pub fn local_min(src : Window<'_, u8>, dst : WindowMut<'_, u8>, kernel_sz : usize) -> Image<u8> {
    unimplemented!()
}

pub fn local_max(src : Window<'_, u8>, dst : WindowMut<'_, u8>, kernel_sz : usize) -> Image<u8> {
    unimplemented!()
}

/* cv2::sepFilter2D */
/* cv2::Laplacian */

#[cfg(feature="opencv")]
pub fn sobel(img : &Window<u8>, dst : WindowMut<'_, i16>, sz : usize, dx : i32, dy : i32) {
    use opencv::core;
    let src : core::Mat = img.into();
    let mut dst : core::Mat = dst.into();
    opencv::imgproc::sobel(&src, &mut dst, core::CV_16S, dx, dy, sz as i32, 1.0, 0.0, core::BORDER_DEFAULT).unwrap();
}

/*IppStatus ippiFilterBilateral_<mod>(const Ipp<srcdatatype>* pSrc, int srcStep,
Ipp<dstdatatype>* pDst, int dstStep, IppiSize dstRoiSize, IppiBorderType borderType,
const Ipp<datatype> pBorderValue[1], const IppiFilterBilateralSpec* pSpec, Ipp8u*
pBuffer );

IppStatus ippiFilterBox_64f_C1R(const Ipp<datatype>* pSrc, int srcStep, Ipp<datatype>*
pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize, IppiPoint anchor );

IppStatus ippiFilterMedian_<mod>(const Ipp<datatype>* pSrc, int srcStep, Ipp<datatype>*
pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize, IppiPoint anchor, Ipp8u*
pBuffer );

IppStatus ippiFilterSeparable_<mod>(const Ipp<datatype>* pSrc, int srcStep,
Ipp<datatype>* pDst, int dstStep, IppiSize roiSize, IppiBorderType borderType,
Ipp<datatype> borderValue, const IppiFilterSeparableSpec* pSpec, Ipp8u* pBuffer );

IppStatus ippiConv_<mod>(const Ipp<datatype>* pSrc1, int src1Step, IppiSize src1Size,
const Ipp<datatype>* pSrc2, int src2Step, IppiSize src2Size, Ipp<datatype>* pDst, int
dstStep, int divisor, IppEnum algType, Ipp8u* pBuffer );

IppStatus ippiDeconvFFT_32f_C1R(const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int
dstStep, IppiSize roiSize, IppiDeconvFFTState_32f_C1R* pDeconvFFTState );

IppStatus ippiDeconvFFT_32f_C3R(const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int
dstStep, IppiSize roiSize, IppiDeconvFFTState_32f_C3R* pDeconvFFTState );

IppStatus ippiFilterSobel_<mod>(const Ipp<srcdatatype>* pSrc, int srcStep,
Ipp<dstdatatype>* pDst, int dstStep, IppiSize dstRoiSize, IppiMaskSize maskSize,
IppNormType normType, IppiBorderType borderType, Ipp<srcdatatype> borderValue, Ipp8u*
pBuffer );

IppStatus ippiFilterGaussian_<mod>(const Ipp<datatype>* pSrc, int srcStep,
Ipp<datatype>* pDst, int dstStep, IppiSize roiSize, IppiBorderType borderType, const
Ipp<datatype> borderValue[1], IppFilterGaussianSpec* pSpec, Ipp8u* pBuffer );

IppStatus ippiDCTFwd_<mod> (const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
const IppiDCTFwdSpec_32f* pDCTSpec, Ipp8u* pBuffer );

IppStatus ippiDCTInv_<mod> (const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
const IppiDCTInvSpec_32f* pDCTSpec, Ipp8u* pBuffer );

IppStatus ippiDCT8x8Fwd_<mod>(const Ipp<datatype>* pSrc, Ipp<datatype>* pDst );

IppStatus ippiDCT8x8Inv_2x2_16s_C1(const Ipp16s* pSrc, Ipp16s* pDst );

// imgproc::blur
// imgproc::median_blur
// imgproc::sobel

/* pub fn scharr(
    src: &dyn ToInputArray,
    dst: &mut dyn ToOutputArray,
    ddepth: i32,
    dx: i32,
    dy: i32,
    scale: f64,
    delta: f64,
    border_type: i32
) -> Result<()>
*/

// TODO use crate edge_detection::canny(
// TODO use image_conv::convolution(&img, filter, 1, PaddingType::UNIFORM(1)); with custom filters.
// mss_saliency = "1.0.6" for salient portion extraction.
*/

// Separability of Kernels (Szelisky, p. 116, after Perona, 1995). If SVD of the 2d kernel matrix has only
// the first singular value not zeroed, then the Kernel is separable into \sqrt \sigma_0 u_0 (vertical)
// and \sqrt \sigma_0 v_0^T (horizontal) kernels.




