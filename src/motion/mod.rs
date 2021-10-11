/*IppStatus ippiFGMMForeground_8u_C3R(const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int
dstStep, IppiSize roi, IppFGMMState_8u_C3R* pState, IppFGMModel* pModel, double
learning_rate );

IppStatus ippiFGMMBackground_8u_C3R(Ipp8u* pDst, int dstStep, IppiSize roi,
IppFGMMState_8u_C3R* pState );

IppStatus ippiOpticalFlowPyrLK_<mod>(IppiPyramid* pPyr1, IppiPyramid* pPyr2, const
IppiPoint_32f* pPrev, IppiPoint_32f* pNext, Ipp8s* pStatus, Ipp32f* pError, int
numFeat, int winSize, int maxLev, int maxIter, Ipp32f threshold,
IppiOptFlowPyrLK_<mod>* pState );

/*// robust local optical flow (RLOF) (for color images)
pub struct SparseRLOF {

}

impl RLOF {
    calcOpticalFlowSparseRLOF()
}

pub struct Farneback{

}

//
pub struct GPC {

}

// Pyramidal Lucas-Kanade algorithm
pub struct SparsePyrLK {

}
    // Shi-Tomasi feature extractor
    goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);
    TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
    calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15,15), 2, criteria);
*/

/*fn optical_flow() {
    // or video::tracking
    video::SparsePyrLKOpticalFlow::create(
        win_size: Size,
        max_level: i32,
        crit: TermCriteria,
        flags: i32,
        min_eig_threshold: f64
    ) -> Result<Ptr<dyn SparsePyrLKOpticalFlow>>
}*/

 Point2d 	cv::phaseCorrelate (InputArray src1, InputArray src2, InputArray window=noArray(), double *response=0)
 	The function is used to detect translational shifts that occur between two images. More...

*/
