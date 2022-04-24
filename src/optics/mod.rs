/*IppStatus ippiUndistortRadial_<mod>(const Ipp<datatype>* pSrc, int srcStep,
Ipp<datatype>* pDst, int dstStep, IppiSize roiSize, Ipp32f fx, Ipp32f fy, Ipp32f cx,
Ipp32f cy, Ipp32f k1, Ipp32f k2, Ipp8u* pBuffer );*/

/// Calculate lens magnification (ratio of image to object size) from focal depth
/// From Harley & Weeks (1993)
fn magnification(focal_depth : f64, dist_obj : f64) -> f64 {
    focal_depth / (dist_obj - focal_depth)
}

fn depth_of_field(f_stop : f64, magnification : f64) -> f64 {
    ( 2. * f_stop * (magnification + 1.) ) / magnification.powf(2.)
}


