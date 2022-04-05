use crate::image::*;
use std::default::Default;

/*pub trait Thresholding {

    fn threshold_mut(&mut self, win : WindowMut<'_, u8>);

    fn threshold_to(&mut self, win : &Window<'_, u8>, dst : &mut WindowMut<'_, u8>);

    fn threshold(&mut self, win : &Window<'_, u8>) -> Image<u8> {
        let mut img = Image::new_constant(win.height(), win.width(), 0);
        self.threshold_to(win, img.full_window_mut());
        img
    }

}

pub enum Foreground {
    Below(u8),
    Above(u8),
    Between(u8,u8)
}

pub struct FixedThresholding {
    for : Foreground,

}

impl FixedThresholding {

}

impl Thresholding for FixedThresholding {

}

impl Thresholding for OtsuThresholding {

}

impl Thresholding for BalancedThresholding {

}

*/

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

//pub fn positive_truncate_abs<'a>(src : &'a Window<'a, i16>, dst : &'a mut WindowMut<'a, u8>) {
//    dst.pixels_mut(1).zip(src.pixels(1)).for_each(move |(d, s)| *d = *s as u8 );
//}

pub fn truncate_abs<'a>(src : &'a Window<'a, i16>, dst : &'a mut WindowMut<'a, u8>) {
    dst.pixels_mut(1).zip(src.pixels(1)).for_each(move |(d, s)| *d = s.abs() as u8 );
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

/* Binarization sets all pixels below val to 0, all pixels at or above val to 'to' value. If out is
smaller than self, every ith pixel of input is taken */
pub fn binarize_bytes_mut(win : &Window<'_, u8>, out : &mut WindowMut<'_, u8>, val : u8, to : u8) {
    assert!(win.width() % out.width() == 0 && win.height() % out.height() == 0 );
    assert!(win.width() / out.width() == win.height() / out.height());
    let scale = win.width() / out.width();
    for i in 0..out.height() {
        for j in 0..out.width() {
            if win[(i * scale, j * scale)] < val {
                out[(i, j)] = 0;
            } else {
                out[(i, j)] = to;
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

use crate::feature::patch::ColorProfile;

/// Otsu's method determine the best discriminant for a bimodal intensity distribution.
/// It explores the fact that for 2 classes, minimizing intra-class variance is the same
/// as maximizing inter-class variance.
#[derive(Debug, Clone)]
pub struct Otsu {

}

pub fn partial_mean(min : usize, max : usize, probs : &[f32]) -> f32 {
    (min..max).map(|ix| ix as f32 * probs[ix] ).sum::<f32>()
}

impl Otsu {

    pub fn new() -> Self {
        Self { }
    }

    // From https://en.wikipedia.org/wiki/Otsu%27s_method
    pub fn estimate(&self, profile : &ColorProfile, step : usize) -> u8 {

        let num_pixels = profile.num_pxs();
        let mut max_th = 0;
        let mut max_inter_var = 0.;

        let probs = profile.probabilities();

        // Class probabilities (integrate histogram bins below and above current th)
        let mut prob_a = 0.;
        let mut prob_b = 0.;

        let mut mean_a = 0.;
        let mut mean_b = 0.;
        let mut mean_total = partial_mean(0, 256, &probs[..]);
        let mut inter_var = 0.;

        for th in (0..256).step_by(step) {

            prob_a = probs[0..th].iter().copied().sum::<f32>();
            prob_b = 1. - prob_a;

            mean_a = partial_mean(0, th, &probs[..]);
            mean_b = partial_mean(th, 256, &probs[..]);

            if prob_a == 0. || prob_b == 0. {
                continue;
            }

            inter_var = prob_a * prob_b * (mean_a - mean_b).powf(2.);
            if inter_var > max_inter_var {
                max_th = th;
                max_inter_var = inter_var;
            }
        }

        max_th as u8
    }

}

// From https://en.wikipedia.org/wiki/Balanced_histogram_thresholding. This is
// much faster than Otsu's method.
#[derive(Debug, Clone)]
pub struct BalancedHist {

    // Binds below this value at either end of the histogram won't count towards histogram equilibrium point
    // (but values below this that are not at extreme values will).
    min_count : usize,

}

impl BalancedHist {

    pub fn new(min_count : usize) -> Self {
        Self { min_count }
    }

    pub fn estimate(&self, profile : &ColorProfile, step : usize) -> u8 {
        let bins = profile.bins();

        // Those delimit the left region if the histogram follows a bimodal distribution.
        let mut left_limit = 0;
        let mut right_limit = 255;

        while bins[left_limit] < self.min_count && left_limit < (right_limit - step) {
            left_limit += step;
        }

        while bins[right_limit] < self.min_count && right_limit > left_limit {
            right_limit -= step;
        }

        // Take histogram expected value as first center guess.
        let mut center = partial_mean(0, 256, &profile.probabilities()[..]) as usize;

        // TODO propagate step to increment/decrement
        // for th in (start..end) /*.step_by(step)*/ {

        let mut weight_a = bins[..center].iter().copied().sum::<usize>();
        let mut weight_b = bins[center..].iter().copied().sum::<usize>();

        while left_limit < right_limit {

            if weight_a > weight_b {
                // Left part heavier (skew left limit towars histogram end)
                weight_a -= bins[left_limit];
                left_limit += step;
            } else {
                // Right part heavier (skew right limit towards histogram beginning)
                weight_b -= bins[right_limit];
                right_limit -= step;
            }

            // Calculate new center as an average of left and right limits
            let new_center = ((left_limit as f32 + right_limit as f32) / 2.) as usize;

            if new_center < center {

                // Move bin at center from left to right
                weight_a -= bins[center];
                weight_b += bins[center];
            } else {
                if new_center > center {

                    // Move bin at center from right to left
                    weight_a += bins[center];
                    weight_b -= bins[center];
                }

                // If new_center = center, do nothing
            }

            center = new_center;
        }
        center as u8
    }

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
