use nalgebra::{Matrix2x3};
use std::ptr;
use crate::image::*;
use std::iter::FromIterator;
use std::fmt::Debug;
use std::mem;

pub fn perspective_to() {

    /*

    IppStatus ippiWarpPerspectiveNearest_<mod>(const Ipp<datatype>* pSrc, int srcStep,
    Ipp<datatype> pDst, int dstStep, IppiPoint dstRoiOffset, IppiSize dstRoiSize, const
    IppiWarpSpec* pSpec, Ipp8u* pBuffer);

    IppStatus ippiWarpPerspectiveCubic_<mod>(const Ipp<datatype>* pSrc, int srcStep,
    Ipp<datatype> pDst, int dstStep, IppiPoint dstRoiOffset, IppiSize dstRoiSize, const
    IppiWarpSpec* pSpec, Ipp8u* pBuffer);

    IppStatus ippiWarpPerspectiveLinear_<mod>(const Ipp<datatype>* pSrc, int srcStep,
    Ipp<datatype> pDst, int dstStep, IppiPoint dstRoiOffset, IppiSize dstRoiSize, const
    IppiWarpSpec* pSpec, Ipp8u* pBuffer);

    */


}

#[cfg(feature="ipp")]
pub fn affine_to<N>(w : &Window<N>, m : &Matrix2x3<f64>, dst : &mut WindowMut<N>) 
where
    N : Pixel + Debug + Copy,
    for<'a> &'a [N] : Storage<N>,
    for<'a> &'a mut [N] : StorageMut<N>,
{
    let (src_step, src_roi) = crate::image::ipputils::step_and_size_for_window(w);
    let (dst_step, dst_roi) = crate::image::ipputils::step_and_size_for_window_mut(&*dst);

    // IPP expects a row-oriented affine matrix, which
    // is why we transpose it here (nalgebra represents
    // matrices column-wise).
    let mt_coefs = [
        [m[(0, 0)], m[(0, 1)], m[(0, 2)]],
        [m[(1, 0)], m[(1, 1)], m[(1, 2)]]
    ];
    let ty = if w.pixel_is::<u8>() {
        crate::foreign::ipp::ippi::IppDataType_ipp8u
    } else if w.pixel_is::<f32>() {
        crate::foreign::ipp::ippi::IppDataType_ipp32f
    } else {
        panic!("Invalid warp type");
    };
    let interp = crate::foreign::ipp::ippi::IppiInterpolationType_ippNearest;
    let dir = crate::foreign::ipp::ippi::IppiWarpDirection_ippWarpForward;
    let border = crate::foreign::ipp::ippi::_IppiBorderType_ippBorderMirror;
    let n_channels = 1;
    let (mut spec_sz, mut init_buf_sz) : (i32, i32) = (0, 0);
    let border_val = 0.0;
    let smooth = 0;
    unsafe {
        let ans = crate::foreign::ipp::ippi::ippiWarpAffineGetSize(
            src_roi,
            dst_roi,
            ty,
            &mt_coefs as *const _,
            interp,
            dir,
            border,
            &mut spec_sz as *mut _,
            &mut init_buf_sz as *mut _,
        );
        assert!(ans == 0);
        let mut init_buf = Vec::from_iter((0..(init_buf_sz as usize)).map(|_| 0u8 ));
        let mut spec_bytes = Vec::from_iter((0..(spec_sz as usize)).map(|_| 0u8 ));
        let ans = crate::foreign::ipp::ippi::ippiWarpAffineNearestInit(
            src_roi,
            dst_roi,
            ty,
            &mt_coefs as *const _,
            dir,
            n_channels,
            border,
            &border_val,
            smooth,
            spec_bytes.as_mut_ptr() as *mut _
        );
        assert!(ans == 0);

        let mut buf_sz = 0;
        let ans = crate::foreign::ipp::ippi::ippiWarpGetBufferSize(
            spec_bytes.as_mut_ptr() as *mut _,
            src_roi,
            &mut buf_sz as *mut _
        );
        assert!(ans == 0);

        let mut buffer = Vec::from_iter((0..buf_sz).map(|_| 0u8 ));
        let offset = crate::foreign::ipp::ippi::IppiPoint { x : 0, y : 0 };
        let ans = if w.pixel_is::<u8>() {
            crate::foreign::ipp::ippi::ippiWarpAffineNearest_8u_C1R(
                mem::transmute(w.as_ptr()),
                src_step,
                mem::transmute(dst.as_mut_ptr()),
                dst_step,
                offset,
                dst_roi,
                spec_bytes.as_ptr() as *const _,
                buffer.as_mut_ptr()
            )
        } else if w.pixel_is::<f32>() {
            crate::foreign::ipp::ippi::ippiWarpAffineNearest_32f_C1R(
                mem::transmute(w.as_ptr()),
                src_step,
                mem::transmute(dst.as_mut_ptr()),
                dst_step,
                offset,
                dst_roi,
                spec_bytes.as_ptr() as *const _,
                buffer.as_mut_ptr()
            )
        } else {
            panic!("Invalid warp type");
        };
        assert!(ans == 0);
    }
}
