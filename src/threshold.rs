use opencv::{core, imgproc};
use crate::image::cvutils::slice_to_mat;
use crate::image::*;

/// Ouptuts a 0-bit (0 OR 255) binary umage
pub fn threshold_slice(src : &[u8], ncol : usize, dst : &mut [u8], thresh : f64, max_val : f64) {
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

pub fn threshold_window(src : &Window<u8>, dst : &mut Image<u8>, thresh : f64, max_val : f64, higher : bool) {
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


