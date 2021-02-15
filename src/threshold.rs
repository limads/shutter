use opencv::{core, imgproc};
use crate::image::cvutils::slice_to_mat;

/// Ouptuts a 0-bit (0 OR 255) binary umage
pub fn threshold(src : &[u8], ncol : usize, dst : &mut [u8], thresh : f64, max_val : f64) {
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


