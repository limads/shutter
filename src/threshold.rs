use crate::image::*;

/// Represents an interval over the 8-bit intensity level.
#[derive(Clone, Debug)]
pub struct Threshold {
    pub min : u8,
    pub max : u8
}

/// Ouptuts a 0-bit (0 OR 255) binary umage
#[cfg(feature="opencvlib")]
pub fn threshold_slice(src : &[u8], ncol : usize, dst : &mut [u8], thresh : f64, max_val : f64) {

    use opencv::{core, imgproc};
    use crate::image::cvutils::slice_to_mat;

    unsafe {
        let src = slice_to_mat(src, ncol, None);
        let mut dst = slice_to_mat(&dst, ncol, None);
        imgproc::threshold(
            &src,
            &mut dst, 
            thresh, 
            max_val,
            imgproc::THRESH_BINARY
        ).unwrap();
    }
}

#[cfg(feature="opencvlib")]
pub fn threshold_window(src : &Window<u8>, dst : &mut Image<u8>, thresh : f64, max_val : f64, higher : bool) {

    use opencv::{core, imgproc};
    use crate::image::cvutils::slice_to_mat;

    let src : core::Mat = src.clone().into();
    let mut dst : core::Mat = dst.full_window().into();
    let modality = if higher {
        imgproc::THRESH_BINARY
    } else {
        imgproc::THRESH_BINARY_INV
    };
    imgproc::threshold(
        &src,
        &mut dst,
        thresh,
        max_val,
        modality
    ).unwrap();
}

// IppStatus ippiThreshold_<mod> ( const Ipp<datatype>* pSrc , int srcStep , Ipp<datatype>*
// pDst , int dstStep , IppiSize roiSize , Ipp<datatype> threshold , IppCmpOp ippCmpOp );
