use crate::image::*;
use std::default::Default;

/// Represents an interval over the 8-bit intensity level.
#[derive(Clone, Debug, Copy)]
pub struct Threshold {
    pub min : u8,
    pub max : u8
}

impl Default for Threshold {

    fn default() -> Self {
        Self { min : 0, max : 255 }
    }

}

#[cfg(feature="opencv")]
pub fn adaptive_threshold(img : &Window<'_, u8>, out : WindowMut<'_, u8>, block_sz : usize, gauss : bool, value : i16, below : bool) {

    assert!(block_sz % 2 == 1, "Block size must be an odd number");

    use opencv::{core, imgproc};

    let src : core::Mat = img.into();
    let mut dst : core::Mat = out.into();

    let mode = if below {
        imgproc::THRESH_BINARY_INV
    } else {
        imgproc::THRESH_BINARY
    };

    let local_weight = if gauss {
        imgproc::ADAPTIVE_THRESH_GAUSSIAN_C
    } else {
        imgproc::ADAPTIVE_THRESH_MEAN_C
    };
    imgproc::adaptive_threshold(
        &src,
        &mut dst,
        255.0,
        local_weight,
        mode,
        block_sz as i32,
        value as f64
    ).unwrap();

}

/// Ouptuts a 0-bit (0 OR 255) binary umage
#[cfg(feature="opencv")]
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

pub fn invert_colors_inplace(win : &mut WindowMut<'_, u8>) {
    // win.pixels_mut(1).for_each(|px| *px = 255 - *px );
    for i in 0..win.height() {
        for j in 0..win.width() {
            win[(i, j)] = 255 - win[(i, j)];
        }
    }
}

/// Sets all values below the given one to zero.
pub fn binarize_bytes_below(win : &Window<'_,u8>, out : &mut WindowMut<'_, u8>, val : u8) {
    assert!(win.width() % out.width() == 0 && win.height() % out.height() == 0 );
    assert!(win.width() / out.width() == win.height() / out.height());
    let scale = win.width() / out.width();
    for i in 0..out.height() {
        for j in 0..out.width() {
            if win[(i * scale, j * scale)] < val {
                out[(i, j)] = 0;
            }
        }
    }
}

/* Binarization sets all pixels below val to 0, all pixels at or above val to 1. If out is
smaller than self, every ith pixel of input is taken */
pub fn binarize_bytes_mut(win : &Window<'_, u8>, out : &mut WindowMut<'_, u8>, val : u8) {
    assert!(win.width() % out.width() == 0 && win.height() % out.height() == 0 );
    assert!(win.width() / out.width() == win.height() / out.height());
    let scale = win.width() / out.width();
    for i in 0..out.height() {
        for j in 0..out.width() {
            if win[(i * scale, j * scale)] < val {
                out[(i, j)] = 0;
            } else {
                out[(i, j)] = 1;
            }
        }
    }

    // Not working due to lifetime issue.
    /*for (mut px_out, px) in out.pixels_mut(1).zip(win.pixels(scale)) {
        if *px < val {
            *px_out = 0;
        } else {
            *px_out = 1;
        }
    }*/
}

#[derive(Debug, Clone, Copy)]
pub enum GlobalThreshold {
    Fixed(u8),
    Otsu
}

#[cfg(feature="opencv")]
pub fn global_threshold(src : &Window<u8>, mut dst : WindowMut<'_, u8>, thresh : GlobalThreshold, max_val : u8, higher : bool) {

    use opencv::{core, imgproc};
    use crate::image::cvutils::slice_to_mat;

    let src : core::Mat = src.clone().into();
    let mut dst : core::Mat = dst.into();
    let modality = match (higher, thresh) {
        (true, GlobalThreshold::Otsu) => {
            imgproc::THRESH_BINARY + imgproc::THRESH_OTSU
        },
        (true, GlobalThreshold::Fixed(_)) => {
            imgproc::THRESH_BINARY
        },
        (false, GlobalThreshold::Otsu) => {
            imgproc::THRESH_BINARY_INV + imgproc::THRESH_OTSU
        },
        (false, GlobalThreshold::Fixed(_)) => {
            imgproc::THRESH_BINARY_INV
        }
    };

    let thresh = match thresh {
        GlobalThreshold::Fixed(v) => v as f64,
        _ => 0.0
    };

    imgproc::threshold(
        &src,
        &mut dst,
        thresh,
        max_val as f64,
        modality
    ).unwrap();
}

// IppStatus ippiThreshold_<mod> ( const Ipp<datatype>* pSrc , int srcStep , Ipp<datatype>*
// pDst , int dstStep , IppiSize roiSize , Ipp<datatype> threshold , IppCmpOp ippCmpOp );
