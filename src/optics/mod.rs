/*IppStatus ippiUndistortRadial_<mod>(const Ipp<datatype>* pSrc, int srcStep,
Ipp<datatype>* pDst, int dstStep, IppiSize roiSize, Ipp32f fx, Ipp32f fy, Ipp32f cx,
Ipp32f cy, Ipp32f k1, Ipp32f k2, Ipp8u* pBuffer );*/

/// Calculate lens magnification (ratio of image to object size) from focal depth
/// From Harley & Weeks (1993)
pub fn magnification(focal_depth : f64, dist_obj : f64) -> f64 {
    focal_depth / (dist_obj - focal_depth)
}

pub fn depth_of_field(f_stop : f64, magnification : f64) -> f64 {
    ( 2. * f_stop * (magnification + 1.) ) / magnification.powf(2.)
}

/// Calculates the focal length (lambda) using the thin lens formula,
/// by informing the distance between the focused object and lens and
/// the manufecturer-specified lens focal length (i.e. focal length for object
/// at infinity).
pub fn effective_focal_length(obj_dist : f32, lens_focal_len : f32) -> f32 {
    1. / ((1. / lens_focal_len) - (1. / obj_dist) )
}

/// Calculates the focal length given the radius of curvature of two lens surfaces
/// assuming the lens is embedded in air (approximation of refractive index).
pub fn lensmaker_focal_length(material_refr : f32, curv_1 : f32, curv_2 : f32) -> f32 {
    1. / (material_refr - 1.) * (1. / curv_1 - 1. / curv_2)
}


