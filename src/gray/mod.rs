use crate::image::*;
use std::default::Default;
use crate::prelude::Raster;

pub fn copy_if_above_threshold(dst_orig : &Window<'_, u8>, orig : (usize, usize), win_dim : (usize, usize), threshold : i32, dst : &mut Image<u8>) {
    if crate::global::sum::<_, f32>(&dst_orig.sub_window(orig, win_dim).unwrap(), 1) as i32 > threshold {
        dst.window_mut(orig, win_dim).unwrap().copy_from(&dst_orig.sub_window(orig, win_dim).unwrap());
    }
}

pub fn zero_sparse_regions(dst : &mut Image<u8>, sum : &mut WindowMut<'_, i32>, win_dim : (usize, usize), min_pts : usize) {

    let sum_sz = (126 / win_dim.0, 126 / win_dim.1);

    crate::local::local_sum(dst.as_ref(), sum);

    let threshold = (255*min_pts) as i32;

    // Overwrite with disjoint windows (faster)
    println!("{:?}", sum);
    for i in 0..sum_sz.0 {
        for j in 0..sum_sz.1 {
            let origin = ((i*win_dim.0) as usize, (j*win_dim.1) as usize);
            if sum[(i, j)] <= threshold {
                dst.window_mut(origin, win_dim).unwrap().fill(0);
            } else {
                // Copying case the value is greater will overwrite the edges of a previous zeroed
                // window region, thus preserving large object edges, and leaving isolated
                // objects at zero. TODO this would happen only for a convolution-like sliding.
                // It is irrelevant for disjoint sliding.
                // dst.window_mut(origin, win_dim).unwrap().copy_from(&dst_orig.window(origin, win_dim).unwrap());
            }
        }
    }

    // Do another pass starting at half window height, this time copying the original values
    // from source when the sum is matched at the current sub-window but the previous window
    // at the side or top was not matched (or vice-versa). This is done to preserve
    // object edges at the transitions from a region that was matched to a region that was not.
    let dst_orig = dst.clone();
    for i in 0..(sum_sz.0-1) {
        for j in 0..(sum_sz.1-1) {

            // Identify this as a left-to-right transition
            let ltr_transition = (sum[(i, j)] > threshold && sum[(i, j+1)] <= threshold) ||
                (sum[(i, j)] <= threshold && sum[(i, j+1)] > threshold);
            if ltr_transition {
                let ltr_origin = ((i*win_dim.0) as usize, (j*win_dim.1 + win_dim.1 / 2) as usize);
                copy_if_above_threshold(dst_orig.as_ref(), ltr_origin, win_dim, threshold, dst);
            }

            // Identify this as a top-to-bottom transition
            let ttb_transition = (sum[(i, j)] > threshold && sum[(i+1, j)] <= threshold) ||
                (sum[(i, j)] <= threshold && sum[(i+1, j)] > threshold);
            if ttb_transition {
                let ttb_origin = ((i*win_dim.0 + win_dim.0 / 2) as usize, (j*win_dim.1) as usize);
                copy_if_above_threshold(dst_orig.as_ref(), ttb_origin, win_dim, threshold, dst);
            }

        }
    }

}

pub fn binary_coordinates(win : &Window<'_, u8>, coords : Option<Vec<(usize, usize)>>) -> Vec<(usize, usize)> {
    let mut coords = coords.unwrap_or(Vec::new());
    coords.clear();
    for i in 0..win.height() {
        for j in 0..win.width() {
            if win[(i, j)] != 0 {
                coords.push((i, j));
            }
        }
    }
    coords
}

pub fn count_range(win : &Window<'_, u8>, low : u8, high : u8) -> usize {

    #[cfg(feature="ipp")]
    unsafe {
        let (step, sz) = crate::image::ipputils::step_and_size_for_window(win);
        let mut count : i32 = 0;
        let ans = crate::foreign::ipp::ippi::ippiCountInRange_8u_C1R(
            win.as_ptr(),
            step,
            sz,
            &mut count as *mut _,
            low,
            high
        );
        assert!(ans == 0);
        return count as usize;
    }

    unimplemented!()
}

#[cfg(feature="opencv")]
pub fn equalize_inplace(win : &mut WindowMut<'_, u8>) {

    use opencv::{imgproc, core};

    let src : core::Mat = win.into();
    let mut dst : core::Mat = win.into();
    imgproc::equalize_hist(&src, &mut dst);
}

#[cfg(feature="opencv")]
pub fn equalize_mut(win : &Window<'_, u8>, dst : &mut WindowMut<'_, u8>) {

    use opencv::{imgproc, core};

    assert!(win.shape() == dst.shape());
    let src : core::Mat = win.into();
    let mut dst : core::Mat = dst.into();
    imgproc::equalize_hist(&src, &mut dst);
}

#[derive(Debug, Clone, Copy)]
pub enum Mask {

    // 3x3 mask
    Three,

    // 5x5 mask
    Five

}

pub enum Norm {
    L1,
    L2,
    Inf
}

#[cfg(feature="ipp")]
pub fn distance_transform(src : &Window<'_, u8>, dst : &mut WindowMut<'_, u8>, mask : Mask, norm : Norm) {

    assert!(src.shape() == dst.shape());

    let (src_byte_stride, roi) = crate::image::ipputils::step_and_size_for_window(src);
    let dst_byte_stride = crate::image::ipputils::byte_stride_for_window_mut(&dst);

    // 3x3 window uses only 2 first entries; 5x5 window uses all three entries.
    let mut metrics : [i32; 3] = [0, 0, 0];

    let mask_size = match mask {
        Mask::Three => 3,
        Mask::Five => 5
    };

    let norm_code = match norm {
        Norm::L1 => crate::foreign::ipp::ippi::_IppiNorm_ippiNormL1,
        Norm::L2 => crate::foreign::ipp::ippi::_IppiNorm_ippiNormL2,
        Norm::Inf => crate::foreign::ipp::ippi::_IppiNorm_ippiNormInf
    };
    unsafe {
        let ans = crate::foreign::ipp::ippcv::ippiGetDistanceTransformMask_32s(
            mask_size,
            norm_code,
            metrics.as_mut_ptr()
        );
        assert!(ans == 0);

        let ans = match mask {
            Mask::Three => {
                crate::foreign::ipp::ippcv::ippiDistanceTransform_3x3_8u_C1R(
                    src.as_ptr(),
                    src_byte_stride,
                    dst.as_mut_ptr(),
                    dst_byte_stride,
                    std::mem::transmute(roi),
                    metrics.as_mut_ptr()
                )
            },
            Mask::Five => {
                crate::foreign::ipp::ippcv::ippiDistanceTransform_5x5_8u_C1R(
                    src.as_ptr(),
                    src_byte_stride,
                    dst.as_mut_ptr(),
                    dst_byte_stride,
                    std::mem::transmute(roi),
                    metrics.as_mut_ptr()
                )
            }
        };
        assert!(ans == 0);
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Foreground {
    Below(u8),
    Above(u8),
    Between(u8, u8)
}

pub trait Threshold {

    fn threshold_to<'a>(&self, src : &'a Window<'a, u8>, out : &mut WindowMut<'a, u8>);

    fn threshold(&self, src : &Window<'_, u8>) -> Image<u8> {
        let mut img = Image::<u8>::new_constant(src.height(), src.width(), 0);
        self.threshold_to(src, &mut img.full_window_mut());
        img
    }

}

pub struct FixedThreshold(Foreground);

impl FixedThreshold {

    pub fn new(fg : Foreground) -> Self {
        Self(fg)
    }

}

impl Threshold for FixedThreshold {

    fn threshold_to<'a>(&self, src : &Window<'a, u8>, out : &mut WindowMut<'a, u8>) {

        assert!(src.shape() == out.shape());

        #[cfg(feature="ipp")]
        unsafe {

            let (src_byte_stride, roi) = crate::image::ipputils::step_and_size_for_window(src);
            let dst_byte_stride = crate::image::ipputils::byte_stride_for_window_mut(&out);

            match self.0 {
                Foreground::Below(v) => {
                    let ans = crate::foreign::ipp::ippi::ippiThreshold_LTVal_8u_C1R(
                        src.as_ptr(),
                        src_byte_stride,
                        out.as_mut_ptr(),
                        dst_byte_stride,
                        roi,
                        v.saturating_add(1),
                        255,
                    );
                    assert!(ans == 0);
                },
                Foreground::Above(v) => {
                    let ans = crate::foreign::ipp::ippi::ippiThreshold_GTVal_8u_C1R(
                        src.as_ptr(),
                        src_byte_stride,
                        out.as_mut_ptr(),
                        dst_byte_stride,
                        roi,
                        v.saturating_sub(1),
                        255
                    );
                    assert!(ans == 0);
                },
                Foreground::Between(a, b) => {
                    // IPP does not offer less-than-or-equal-to here.  We add and remove one from
                    // the user-supplied values to have the same effect (which will give wrong
                    // results for v == 0 and v == 255, since the sum/subtraction is saturating).
                    let less_than = b.saturating_add(1);
                    let greater_than = a.saturating_sub(1);

                    // Actually, this but rerversed (init image to zero; then set lt, gt to 255.
                    let ans = crate::foreign::ipp::ippi::ippiThreshold_LTValGTVal_8u_C1R(
                        src.as_ptr(),
                        src_byte_stride,
                        out.as_mut_ptr(),
                        dst_byte_stride,
                        roi,
                        greater_than,
                        255,
                        less_than,
                        255,
                    );
                    assert!(ans == 0);
                }
            }

            // The IPP calls above sets matching pixels to 255, leaving unmatching pixels at their original values.
            // Now, to produce a binary image, we set all pixels below 255 to zero inplace. TODO there is a problem
            // if the original pixels were originally at 255, since in this case they would be ambiguous with the values
            // set by the threshold op..
            let ans = crate::foreign::ipp::ippi::ippiThreshold_LTVal_8u_C1IR(
                out.as_mut_ptr(),
                dst_byte_stride,
                roi,
                255,
                0,
            );
            assert!(ans == 0);
            return;
        }

        /*#[cfg(feature="opencv")]
        {
            match self.0 {
                Foreground::Below(v) => {
                    // opencv_global_threshold(src, out, GlobalThreshold::Fixed(v), 255, false);
                },
                Foreground::Above(v) => {
                    // opencv_global_threshold(src, out, GlobalThreshold::Fixed(v), 255, true);
                },
                _ => {
                    unimplemented!()
                }
            }
            // return;
        }*/

        /*match self.0 {
            Foreground::Below(v) => {
                out.pixels_mut(1).zip(src.pixels(1))
                    .for_each(|(px_out, px_in)| if *px_in <= v { *px_out = 255 } else { *px_out = 0 });
            },
            Foreground::Above(v) => {
                out.pixels_mut(1).zip(src.pixels(1))
                    .for_each(|(px_out, px_in)| if *px_in >= v { *px_out = 255 } else { *px_out = 0 });
            },
            Foreground::Between(a, b) => {
                out.pixels_mut(1).zip(src.pixels(1))
                    .for_each(|(px_out, px_in)| if *px_in >= a && *px_in <= b { *px_out = 255 } else { *px_out = 0 });
            }
        }*/
    }

}

// pub struct GrayHistogram { // }

// pub struct FixedThresholding {
//    for : Foreground,
// }

// impl FixedThresholding {
// }

/*impl Thresholding for FixedThreshold {

}

impl Thresholding for OtsuThreshold {

}

impl Thresholding for BalancedThreshold {

}

*/

#[cfg(feature="opencv")]
pub fn opencv_adaptive_threshold(img : &Window<'_, u8>, out : WindowMut<'_, u8>, block_sz : usize, gauss : bool, value : i16, below : bool) {

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
pub fn opencv_threshold_slice(src : &[u8], ncol : usize, dst : &mut [u8], thresh : f64, max_val : f64) {

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

pub fn invert_colors(win : &Window<'_, u8>, mut dst : WindowMut<'_, u8>) {

    assert!(win.shape() == dst.shape());

    #[cfg(feature="ipp")]
    unsafe {
        let src_step = crate::image::ipputils::byte_stride_for_window(win);
        let dst_step = crate::image::ipputils::byte_stride_for_window_mut(&dst);
        let src_sz = crate::image::ipputils::window_size(win);
        let ans = crate::foreign::ipp::ippcv::ippiAbsDiffC_8u_C1R(
            win.as_ptr(),
            src_step,
            dst.as_mut_ptr(),
            dst_step,
            std::mem::transmute(src_sz),
            255
        );
        assert_eq!(ans, 0);
        return;
    }

    for i in 0..win.height() {
        for j in 0..win.width() {
            dst[(i, j)] = 255 - win[(i, j)];
        }
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
    min_count : u32,

}

fn relative_to_max(bins : &[u32]) -> Vec<f32> {
    let max = bins.iter().copied().max().unwrap() as f32;
    bins.iter().map(move |count| *count as f32 / max ).collect()
}

impl BalancedHist {

    pub fn new(min_count : u32) -> Self {
        Self { min_count }
    }

    pub fn estimate(&self, bins : &[u32], step : u32) -> u32 {
        // let bins = profile.bins();
        let probabilities = relative_to_max(bins);

        // Those delimit the left region if the histogram follows a bimodal distribution.
        let mut left_limit : u32 = 0;
        let mut right_limit : u32 = 255;

        while bins[left_limit as usize] < self.min_count && left_limit < (right_limit - step) {
            left_limit += step;
        }

        while bins[right_limit as usize] < self.min_count && right_limit > left_limit {
            right_limit -= step;
        }

        // Take histogram expected value as first center guess.
        let mut center = partial_mean(0, 256, &probabilities[..]) as usize;

        // TODO propagate step to increment/decrement
        // for th in (start..end) /*.step_by(step)*/ {

        let mut weight_a = bins[..center].iter().copied().sum::<u32>() as f32;
        let mut weight_b = bins[center..].iter().copied().sum::<u32>() as f32;

        while left_limit < right_limit {

            if weight_a > weight_b {
                // Left part heavier (skew left limit towars histogram end)
                weight_a -= bins[left_limit as usize] as f32;
                left_limit += step;
            } else {
                // Right part heavier (skew right limit towards histogram beginning)
                weight_b -= bins[right_limit as usize] as f32;
                right_limit -= step;
            }

            // Calculate new center as an average of left and right limits
            let new_center = ((left_limit as f32 + right_limit as f32) / 2.) as usize;

            if new_center < center {

                // Move bin at center from left to right
                weight_a -= bins[center] as f32;
                weight_b += bins[center] as f32;
            } else {
                if new_center > center {

                    // Move bin at center from right to left
                    weight_a += bins[center] as f32;
                    weight_b -= bins[center] as f32;
                }

                // If new_center = center, do nothing
            }

            center = new_center;
        }
        center as u32
    }

}

#[cfg(feature="opencv")]
#[derive(Debug, Clone, Copy)]
enum GlobalThreshold {
    Fixed(u8),
    Otsu
}

#[cfg(feature="opencv")]
fn opencv_global_threshold(src : &Window<u8>, mut dst : WindowMut<'_, u8>, thresh : GlobalThreshold, max_val : u8, higher : bool) {

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

/// Keep only nonzero pixels when at least k in [1..8] neighboring nonzero pixels are also present,
/// irrespective of the position of those pixels.
pub fn supress_binary_speckles(win : &Window<'_, u8>, mut out : WindowMut<'_, u8>, min_count : usize) {

    assert!(win.shape() == out.shape());
    assert!(min_count >= 1 && min_count <= 8);

    for ((r, c), neigh) in win.labeled_neighborhoods() {
        if neigh.filter(|px| *px > 0 ).count() >= min_count {
            out[(r, c)] = win[(r, c)];
        } else {
            out[(r, c)] = 0;
        }
    }
}

// IppStatus ippiThreshold_<mod> ( const Ipp<datatype>* pSrc , int srcStep , Ipp<datatype>*
// pDst , int dstStep , IppiSize roiSize , Ipp<datatype> threshold , IppCmpOp ippCmpOp );
