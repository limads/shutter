
/* SLIC (superpixel segmentation; Requires RGB or LAB images):
regionSize = 10 ;  // Increase superpixel size
regularizer = 10 ; //
segments = vl_slic(im, regionSize, regularizer) ; */

/// Foreground Gaussian mixture model for foreground/background extraction
pub struct FGMM {

}

impl FGMM {

    /// max_gauss: Maximal size of the Gaussian mixture components.
    pub fn new(max_gauss : usize) -> Self {
        let mut spec_sz : i32 = 0;
        let status = ippiFGMMGetBufferSize_8u_C3R(IppiSize roi, max_gauss as i32, &mut spec_sz as *mut _);
        check_status("Determining size of FGMM buffer", status);

        let model = ptr::null();
        let status = ippiFGMMInit_8u_C3R(IppiSize roi, max_gauss as i32, model,
            IppFGMMState_8u_C3R* pState );

            IppStatus ippiFGMMForeground_8u_C3R(const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int
            dstStep, IppiSize roi, IppFGMMState_8u_C3R* pState, IppFGMModel* pModel, double
            learning_rate );

            IppStatus ippiFGMMBackground_8u_C3R(Ipp8u* pDst, int dstStep, IppiSize roi,
            IppFGMMState_8u_C3R* pState );
    }
}
