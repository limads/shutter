pub struct MarrHildreth {

}

#[cfg(feature="vlfeat")]
pub struct SIFT {
    sift_filt : *const VlSiftFilt
}

#[cfg(feature="vlfeat")]
impl SIFT {

    pub fn new(
        height : usize, 
        width : usize, 
        n_octaves : usize, 
        n_lvls_per_octave : usize, 
        fst_oct_ix : Option<usize>
    ) -> Self {
        let o_min = fst_oct_ix.map(|o| o as i32).unwrap_or(-1);
        unsafe {
            let sift_filt :  *const VlSiftFilt = vl_sift_new(width as i32, height as i32, n_octaves as i32, n_lvls_per_octave as i32, o_min);
            Self { sift_filt }
        }
    }
    
    pub fn calc<S>(&mut self, img : &Image<f32, S>)
    where
        S : Image<f32, Storage<f32>>
    {
        unsafe {
            vl_sift_process_first_octave(self.sift_filt, img.as_ptr());
            let ans = vl_sift_process_next_octave(self.sift_filt);
            
            // Run detector to get keypoints
            vl_sift_detect(sift_filt);
            
            let x = 0.0;
            let y = 0.0;
            let sigma = 1.0;
            vl_sift_keypoint_init(self.sift_filt, keypoints, x, y, sigma);
            let mut angles : [f64; 4] = [0.0; 4];
            let keypoints : *const VlSiftKeypoint = vl_sift_get_keypoints(self.sift_filt);
            let n_keypoints = vl_sift_get_nkeypoints(self.sift_filt);
            vl_sift_calc_keypoint_orientations(self.sift_filt, &mut angles, keypoint);
            
            let desc : *mut vl_sift_pix;
            let mut angle0 : f64 = 0.;
            vl_sift_calc_keypoint_descriptor(self.sift_filt, descr, keypoint, angle0);
            vl_sift_calc_keypoint_orientations
            if ans == VL_ERR_EOF {
                break;
            }
        }
    }
    
}

#[cfg(feature="vlfeat")]
impl Drop for SIFT {

    fn drop(&mut self) {
        unsafe { vl_sift_delete(self.sift_filt); }
    }
    
}

pub struct SURF {

}

pub struct Harris {

}

/*IppStatus ippiHarrisCorner_8u32f_C1R(const Ipp8u* pSrc, int srcStep, Ipp32f* pDst, int
dstStep, IppiSize roiSize, IppiDifferentialKernel filterType, IppiMaskSize filterMask,
Ipp32u avgWndSize, float k, float scale, IppiBorderType borderType, Ipp8u borderValue,
Ipp8u* pBuffer);

IppStatus ippiHarrisCorner_32f_C1R(const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int
dstStep, IppiSize roiSize, IppiDifferentialKernel filterType, IppiMaskSize filterMask,
Ipp32u avgWndSize, float k, float scale, IppiBorderType borderType, Ipp32f borderValue,
Ipp8u* pBuffer);

IppStatus ippiHOGGetSize(const IppiHOGConfig* pConfig, int* pHOGSpecSize);
IppStatus ippiHOGInit(const IppiHOGConfig* pConfig, IppiHOGSpec* pHOGSpec);
IppStatus ippiHOGGetBufferSize(const IppiHOGSpec* pHOGSpec, IppiSize roiSize, int*
pBufferSize);
IppStatus ippiHOGGetDescriptorSize(const IppiHOGSpec* pHOGSpec, int*
pWinDescriptorSize);
IppStatus ippiHOG_<mod>(const Ipp<srcDatatype>* pSrc, int srcStep, IppiSize roiSize,
const IppiPoint* pLocation, int nLocations, Ipp32f* pDst, const IppiHOGSpec* pHOGSpec,
IppiBorderType borderID, Ipp<srcDatatype> borderValue, Ipp8u* pBuffer);

*/
