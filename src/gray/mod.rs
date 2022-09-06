use crate::image::*;
use std::default::Default;
use crate::prelude::Raster;
use std::mem;
use bayes::calc::*;
use std::collections::BTreeMap;
use std::ops::Sub;
use crate::hist::{GrayHistogram, GridHistogram};

#[derive(Debug, Clone, Copy)]
pub enum AdaptiveForeground {
    Above,
    Below,
    Inside,
    Outside
}

impl AdaptiveForeground {

    fn foreground(&self, v : u8) -> Foreground {
        match self {
            AdaptiveForeground::Above => Foreground::Above(v),
            AdaptiveForeground::Below => Foreground::Below(v),
            _ => panic!()
        }
    }

}

impl Window<'_, u8> {

    /// (1) Calculate partition values for each sub-window
    /// (Or the full window if win_sz is None).
    /// (3) Apply threshold using this partition value to each sub-window
    /// (or the full window if win_sz is None).
    pub fn adaptive_threshold_to(
        &self,
        alg : &mut impl AdadptiveThreshold,
        mut dst : WindowMut<u8>
    ) {
        let bv = alg.best_values(self);
        let mut i = 0;
        let sub_sz = alg.sub_sz();
        for (mut sub_dst, sub_src) in dst.windows_mut((sub_sz.0, sub_sz.1)).zip(self.windows((sub_sz.0, sub_sz.1))) {
            sub_src.threshold_to(bv[i], &mut sub_dst);
            i += 1;
        }
        assert!(i == bv.len());
    }

    // adaptive_truncate

}

pub trait AdadptiveThreshold {

    fn new(win_sz : (usize, usize), sub_sz : Option<(usize, usize)>, fg : AdaptiveForeground) -> Self;

    fn sub_sz(&self) -> (usize, usize);

    fn best_values(&mut self, w : &Window<u8>) -> Vec<Foreground>;

    // fn best_intervals(&mut self, w : &Window<u8>) -> Vec<(u8, u8)>;

}

#[derive(Debug, Clone)]
pub struct GridHistOp {
    grid : GridHistogram,
    fg : AdaptiveForeground,
    sub_sz : (usize, usize)
}

impl GridHistOp {

    fn sub_sz(&self) -> (usize, usize) {
        self.sub_sz
    }

    fn new(win_sz : (usize, usize), sub_sz : Option<(usize, usize)>, fg : AdaptiveForeground) -> Self {
        let hist_dim = if let Some(sub_sz) = sub_sz {
            (win_sz.0 / sub_sz.0, win_sz.1 / sub_sz.1)
        } else {
            (1,1)
        };
        Self { grid : GridHistogram::new(hist_dim.0, hist_dim.1), fg, sub_sz : sub_sz.unwrap_or(win_sz) }
    }

    fn best_value_with(&mut self, w : &Window<u8>, f : impl Fn(&GrayHistogram)->u8) -> Vec<Foreground> {
        let mut i = 0;
        let mut best_vals = Vec::new();
        for w in w.windows((self.sub_sz.0, self.sub_sz.1)) {
            self.grid.hists[i].update(&w);
            let bv = f(&self.grid.hists[i]);
            best_vals.push(self.fg.foreground(bv));
            i += 1;
        }
        assert!(best_vals.len() == self.grid.hists.len());
        best_vals
    }

}

#[derive(Debug, Clone)]
pub struct MeanThreshold(GridHistOp);

impl AdadptiveThreshold for MeanThreshold {

    fn sub_sz(&self) -> (usize, usize) {
        self.0.sub_sz
    }

    fn new(win_sz : (usize, usize), sub_sz : Option<(usize, usize)>, fg : AdaptiveForeground) -> Self {
        Self(GridHistOp::new(win_sz, sub_sz, fg))
    }

    fn best_values(&mut self, w : &Window<u8>) -> Vec<Foreground> {
        self.0.best_value_with(w, |h| h.mean() )
    }

}

#[derive(Debug, Clone)]
pub struct MedianThreshold(GridHistOp);

impl AdadptiveThreshold for MedianThreshold {

    fn sub_sz(&self) -> (usize, usize) {
        self.0.sub_sz
    }

    fn new(win_sz : (usize, usize), sub_sz : Option<(usize, usize)>, fg : AdaptiveForeground) -> Self {
        Self(GridHistOp::new(win_sz, sub_sz, fg))
    }

    fn best_values(&mut self, w : &Window<u8>) -> Vec<Foreground> {
        self.0.best_value_with(w, |h| h.median() )
    }

}

/*
IppStatus ippiThresholdAdaptiveBoxGetBufferSize(IppiSize roiSize, IppiSize maskSize,
IppDataType dataType, int numChannels, int* pBufferSize);

IppStatus ippiThresholdAdaptiveBox_8u_C1R(const Ipp8u* pSrc, int srcStep, Ipp8u* pDst,
int dstStep, IppiSize roiSize, IppiSize maskSize, Ipp32f delta, Ipp8u valGT, Ipp8u
valLE, IppiBorderType borderType, Ipp8u borderValue, Ipp8u* pBuffer);

IppStatus ippiThresholdAdaptiveGaussInit(IppiSize roiSize, IppiSize maskSize,
IppDataType dataType, int numChannels, Ipp32f sigma, IppiThresholdAdaptiveSpec* pSpec);

IppStatus ippiThresholdAdaptiveGauss_8u_C1R(const Ipp8u* pSrc, int srcStep, Ipp8u*
pDst, int dstStep, IppiSize roiSize, Ipp32f delta, Ipp8u valGT, Ipp8u valLE,
IppiBorderType borderType, Ipp8u borderValue, IppiThresholdAdaptiveSpec* pSpec, Ipp8u*
pBuffer);
*/

/*

IppStatus ippiReduceBitsGetBufferSize(IppChannels ippChan, IppiSize roiSize, int noise,
IppiDitherType dtype, int* pBufferSize);

IppStatus ippiReduceBits_8u_C1R(const Ipp<datatype>* pSrc, int srcStep, Ipp<datatype>*
pDst, int dstStep, IppiSize roiSize, int noise, IppiDitherType dtype, int levels,
Ipp8u* pBuffer);

IppStatus ippiLUT_8u_C1R(const Ipp<datatype>* pSrc, int srcStep, Ipp<datatype>* pDst,
int dstStep, IppiSize roiSize, IppiLUT_Spec* pSpec);

IppStatus ippiToneMapLinear_32f8u_C1R(const Ipp32f* pSrc, int srcStep, Ipp8u* pDst, int
dstStep, IppiSize roiSize);
*/

/*
If pixel >= upper, set bin_dst to 1. If pixel <= lower, set bin_dst to 0. If pixel
is >= lower and pixel <= upper, set it to 1 if it is connected to a 8-neighbor >= upper.
Set it to zero otherwise. Useful for edge detection.
pub fn histeresis_thresholding(img : &Image<u8>, bin_dst : Image<u8>, lower : u8, upper : u8) {

}*/

// JarvisJudiceNinke
pub struct JJNDithering {

}

// FloydSteinberg Dithering
pub struct FSDithering {

}

// TwoRowSierra Dithering
pub struct TRSDithering {

}

pub struct OrderedDithering {

}

pub struct RiemersmaDithering {

}

// Half-toning technique
// https://en.wikipedia.org/wiki/Error_diffusion
pub struct ErrorDiffusion {

}

pub struct ElserDifferenceMap {

}

/*
The binarization process can be throuht of as two steps:
(1) Partition - Identify one or more optimal gray value(s) in [0,255] that generates a thresholded image with the content of interest;
(2) Threshold - Use the partition(s) value at (1) to threshold the image according to a foreground rule.

Some partitioning algorithms assume a bimodal histogram for foreground rules (above, below);
others assume a unimodal histogram for foreground rules (inside, outside).
*/

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

    Exactly(u8),

    Below(u8),

    Above(u8),

    Inside(u8, u8),

    Outside(u8, u8)

}

pub trait Threshold
where
    Self : Raster
{

    // fn adaptive_threshold_to(&self, fg : Foreground, partition : &mut dyn Partition, out : &mut WindowMut<'a, u8>);

    // fn adaptive_threshold(&self, fg : Foreground, partition : &mut dyn Partition) -> Image<u8>;

    // Binarize an image, setting foreground pixels to u8::MAX and background pixels to u8::MIN
    fn threshold_to(&self, fg : Foreground, out : &mut WindowMut<u8>);

    // Allocating version of threshold_to
    fn threshold(&self, fg : Foreground) -> Image<u8> {
        let mut img = unsafe { Image::<u8>::new_empty(self.height(), self.width()) };
        self.threshold_to(fg, &mut img.full_window_mut());
        img
    }
}

pub trait Truncate
where
    Self : Raster
{

    // fn adaptive_truncate_to
    // fn adaptive_truncate

    // Truncate an image, setting foreground pixels to the desired value, and copying
    // unmatched pixels as they are.
    fn truncate_to(&self, fg : Foreground, out : &mut WindowMut<u8>, fg_val : u8);

    // Allocating version of truncate_to
    fn truncate(&self, fg : Foreground, fg_val : u8) -> Image<u8> {

        // Image cannot be created as empty here because the truncate operation leaves
        // unmatched pixels untouched, so we must guarantee there is valid data at
        // all pixels.
        let mut img = Image::<u8>::new_constant(self.height(), self.width(), 0);

        self.truncate_to(fg, &mut img.full_window_mut(), fg_val);
        img
    }

}

impl<'a> Threshold for Window<'a, u8> {

    fn threshold_to(&self, fg : Foreground, out : &mut WindowMut<u8>) {

        assert!(self.shape() == out.shape());

        #[cfg(feature="ipp")]
        unsafe { return ippi_full_threshold(fg, self, out); }
        baseline_threshold(self, fg, out);
    }

}

fn baseline_threshold(src : &Window<u8>, fg : Foreground, out : &mut WindowMut<u8>) {
    match fg {
        Foreground::Exactly(v) => {
            out.pixels_mut(1).zip(src.pixels(1))
                .for_each(|(px_out, px_in)| if *px_in == v { *px_out = 255 } else { *px_out = 0 });
        },
        Foreground::Below(v) => {
            out.pixels_mut(1).zip(src.pixels(1))
                .for_each(|(px_out, px_in)| if *px_in <= v { *px_out = 255 } else { *px_out = 0 });
        },
        Foreground::Above(v) => {
            out.pixels_mut(1).zip(src.pixels(1))
                .for_each(|(px_out, px_in)| if *px_in >= v { *px_out = 255 } else { *px_out = 0 });
        },
        Foreground::Inside(a, b) => {
            out.pixels_mut(1).zip(src.pixels(1))
                .for_each(|(px_out, px_in)| if *px_in >= a && *px_in <= b { *px_out = 255 } else { *px_out = 0 });
        },
        Foreground::Outside(a, b) => {
            out.pixels_mut(1).zip(src.pixels(1))
                .for_each(|(px_out, px_in)| if *px_in < a && *px_in > b { *px_out = 255 } else { *px_out = 0 });
        }
    }
}

fn baseline_truncate(src : &Window<u8>, fg : Foreground, out : &mut WindowMut<u8>, fg_val : u8) {
    match fg {
        Foreground::Exactly(v) => {
            out.pixels_mut(1).zip(src.pixels(1))
                .for_each(|(px_out, px_in)| if *px_in == v { *px_out = 255 } else { *px_out = *px_in });
        },
        Foreground::Below(v) => {
            out.pixels_mut(1).zip(src.pixels(1))
                .for_each(|(px_out, px_in)| if *px_in <= v { *px_out = 255 } else { *px_out = *px_in });
        },
        Foreground::Above(v) => {
            out.pixels_mut(1).zip(src.pixels(1))
                .for_each(|(px_out, px_in)| if *px_in >= v { *px_out = 255 } else { *px_out = *px_in });
        },
        Foreground::Inside(a, b) => {
            out.pixels_mut(1).zip(src.pixels(1))
                .for_each(|(px_out, px_in)| if *px_in >= a && *px_in <= b { *px_out = 255 } else { *px_out = *px_in });
        },
        Foreground::Outside(a, b) => {
            out.pixels_mut(1).zip(src.pixels(1))
                .for_each(|(px_out, px_in)| if *px_in < a && *px_in > b { *px_out = 255 } else { *px_out = *px_in });
        }
    }
}

impl<'a> Truncate for Window<'a, u8> {

    fn truncate_to(&self, fg : Foreground, out : &mut WindowMut<u8>, fg_val : u8) {
        #[cfg(feature="ipp")]
        unsafe { return ippi_truncate(fg, self, out, fg_val) };
        baseline_truncate(self, fg, out, fg_val);
    }

}

impl Threshold for Image<u8> {

    fn threshold_to(&self, fg : Foreground, out : &mut WindowMut<u8>) {
        self.full_window().threshold_to(fg, out);
    }

}

impl Truncate for Image<u8> {

    fn truncate_to<'a>(&self, fg : Foreground, out : &mut WindowMut<'a, u8>, fg_val : u8) {
        self.full_window().truncate_to(fg, out, fg_val);
    }

}

#[cfg(feature="ipp")]
unsafe fn ippi_truncate(
    fg : Foreground,
    src : &Window<u8>,
    out : &mut WindowMut<u8>,
    fg_val : u8
) {
    let (src_byte_stride, roi) = crate::image::ipputils::step_and_size_for_window(src);
    let dst_byte_stride = crate::image::ipputils::byte_stride_for_window_mut(&out);
    match fg {
        Foreground::Exactly(v) => {
            baseline_truncate(src, fg, out, fg_val);
        },
        Foreground::Below(v) => {
            let ans = crate::foreign::ipp::ippi::ippiThreshold_LTVal_8u_C1R(
                src.as_ptr(),
                src_byte_stride,
                out.as_mut_ptr(),
                dst_byte_stride,
                roi,
                v.saturating_add(1),
                fg_val,
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
                fg_val
            );
            assert!(ans == 0);
        },
        Foreground::Inside(a, b) => {
            baseline_truncate(src, fg, out, fg_val);
        },
        Foreground::Outside(a, b) => {
            let ans = crate::foreign::ipp::ippi::ippiThreshold_LTValGTVal_8u_C1R(
                src.as_ptr(),
                src_byte_stride,
                out.as_mut_ptr(),
                dst_byte_stride,
                roi,
                a,
                fg_val,
                b,
                fg_val,
            );
            assert!(ans == 0);
        }
    }
}

#[cfg(feature="ipp")]
unsafe fn ippi_full_threshold(
    fg : Foreground,
    src : &Window<u8>,
    out : &mut WindowMut<u8>
) {
    let (src_byte_stride, roi) = crate::image::ipputils::step_and_size_for_window(src);
    let dst_byte_stride = crate::image::ipputils::byte_stride_for_window_mut(&out);
    match fg {
        Foreground::Exactly(v) => {
            baseline_threshold(src, fg, out)
        },
        Foreground::Below(v) => {
            let ans = crate::foreign::ipp::ippi::ippiCompareC_8u_C1R(
                src.as_ptr(),
                src_byte_stride,
                v,
                out.as_mut_ptr(),
                dst_byte_stride,
                roi,
                crate::foreign::ipp::ippi::IppCmpOp_ippCmpLessEq
            );
            assert!(ans == 0);
        },
        Foreground::Above(v) => {
            let ans = crate::foreign::ipp::ippi::ippiCompareC_8u_C1R(
                src.as_ptr(),
                src_byte_stride,
                v.saturating_sub(1),
                out.as_mut_ptr(),
                dst_byte_stride,
                roi,
                crate::foreign::ipp::ippi::IppCmpOp_ippCmpGreaterEq
            );
            assert!(ans == 0);
        },
        Foreground::Inside(a, b) => {
            ippi_full_threshold(Foreground::Outside(a, b), src, out);
            crate::binary::inplace_not(out);
        },
        Foreground::Outside(a, b) => {
            ippi_full_threshold(Foreground::Below(a.saturating_sub(1)), src, out);
            ippi_full_threshold(Foreground::Above(b.saturating_add(1)), src, out);
        }
    }
}

/*impl Threshold for FixedThreshold {

    fn threshold_to<'a>(&self, src : &Window<'a, u8>, out : &mut WindowMut<'a, u8>) {

        assert!(src.shape() == out.shape());

        // IPP functions work by filling matched values and copying unmatched values.
        // This is simply a partial_threshold op. To do a full threshold, we must
        // make a second pass and take the output image and fill the non-matched pixels to
        // the given value. But even this does not work, because the original pixel
        // value might be the desired label output (255) in which case there is no way to
        // disambiguate a label 255 from an original image value 255 when the output is copied.
        out.fill(0);

        /* An alternative would be to fill a flat image with the desired threshold value,
        and use ippicompare, which compares two images and output 255 when the pixel value is true
        and 0 when the pixel value is false. */

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
        unimplemented!()
    }

}*/

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

/// Ouptuts a 0-bit (0 OR 255) binary image
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

pub fn invert_colors<N>(win : &Window<'_, N>, mut dst : WindowMut<'_, N>)
where
    N : UnsignedPixel + Sub<Output=N>
{

    assert!(win.shape() == dst.shape());

    #[cfg(feature="ipp")]
    unsafe {
        if win.pixel_is::<u8>() {
            let src_step = crate::image::ipputils::byte_stride_for_window(win);
            let dst_step = crate::image::ipputils::byte_stride_for_window_mut(&dst);
            let src_sz = crate::image::ipputils::window_size(win);
            let ans = crate::foreign::ipp::ippcv::ippiAbsDiffC_8u_C1R(
                mem::transmute(win.as_ptr()),
                src_step,
                mem::transmute(dst.as_mut_ptr()),
                dst_step,
                std::mem::transmute(src_sz),
                255
            );
            assert_eq!(ans, 0);
            return;
        }
    }

    let max = N::max_value();
    for i in 0..win.height() {
        for j in 0..win.width() {
            dst[(i, j)] = max - win[(i, j)];
        }
    }
}

pub fn invert_colors_inplace<N>(win : &mut WindowMut<'_, N>)
where
    N : UnsignedPixel + Sub<Output=N>
{
    // win.pixels_mut(1).for_each(|px| *px = 255 - *px );

    let max = N::max_value();
    for i in 0..win.height() {
        for j in 0..win.width() {
            win[(i, j)] = max - win[(i, j)];
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

pub fn partial_quantile(min : usize, max : usize, probs : &[f32], q : f32) -> f32 {
    assert!(q >= 0. && q <= 1.);
    let mut s = 0.0;
    let mut s_probs = 0.0;
    for ix in min..max {
        if s_probs >= q {
            return s;
        }
        s += ix as f32 * probs[ix];
        s_probs += probs[ix];
    }
    s
}

// Parker, 2011, p. 141
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

impl BalancedHist {

    pub fn new(min_count : u32) -> Self {
        Self { min_count }
    }

    pub fn estimate(&self, bins : &[u32], step : u32) -> u32 {
        // let bins = profile.bins();
        let probabilities = bayes::calc::counts_to_probs(bins);

        // Those delimit the left region if the histogram follows a bimodal distribution.
        let mut left_limit : u32 = 0;
        let mut right_limit : u32 = 255;

        while bins[left_limit as usize] < self.min_count && left_limit < (right_limit - step) {
            left_limit += step;
        }

        while bins[right_limit as usize] < self.min_count && right_limit > left_limit {
            right_limit -= step;
        }

        let mut center = partial_mean(0, 256, &probabilities[..]) as usize;

        // TODO propagate step to increment/decrement
        // for th in (start..end) /*.step_by(step)*/ {

        let mut weight_a = bins[..center].iter().copied().sum::<u32>() as f32;
        let mut weight_b = bins[center..].iter().copied().sum::<u32>() as f32;

        // If left_limit >= right limit, stop. We want to find the point [left, right]
        // when they join each other.
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

/// Pick the threshold that guarantees a certain proportion of pixels are below the threshold.
/// AKA p-tile methohd. This is a good method when you have an idea of the relative area
/// the foreground object should occupy, since the relative are in the image equals the
/// quantile over the histogram. This is a very fast method, since it does not even require
/// a full pass over the histogram.
pub struct QuantileHist {
    q : f32
}

impl QuantileHist {

    pub fn new(q : f32) -> Self {
        Self { q }
    }

    // TODO also, estimate_above, that sums bins from 255..0 until quantile is at 1-q.
    // This is faster if the limit is expected to be at the right end of the histogram.
    pub fn estimate(&self, bins : &[u32]) -> u8 {
        let probabilities = bayes::calc::counts_to_probs(bins);
        partial_quantile(0, 255, &probabilities, self.q) as u8
    }

}

/* Algorithms for finding optimal threshold values (partitions) over gray images with bimodal intensity distribution.
The partition can be calculated over the image or over the histogram directly. */
pub trait Partition {

    // Repeatedly calls partition (potentially using multiple threads) for each sub-window of the given size.
    // The returned binary tree maps window central pixels to the local partitions at each position.
    fn local_partitions<'a>(&'a mut self, img : &Window<u8>, sub_sz : (usize, usize)) -> &'a BTreeMap<(usize, usize), u8> {
        unimplemented!()
    }

    // Calls local_partitions, then for all pixels in the output image, produce an
    // optimal partition by interpolating the partition differences linearly, assuming
    // the partition at each sub-window applies to the center pixel, and the remaining pixel
    // partitoins are calculated by linear interpolation. The output image will give the
    // ideal partition for each pixel; and can be used as the input for Threshold::adaptive_threshold(.)
    // The returned image should be allocated by the partition at the first call of interpolated_partitions.
    fn interpolated_partitions<'a>(&'a mut self, img : &Window<u8>, sub_sz : (usize, usize)) -> &'a Window<'a, u8> {
        unimplemented!()
    }

    fn global_partition(&mut self, img : &Window<u8>) -> u8;

    fn global_partition_for_hist(&mut self, hist : &[u32]) -> u8;

}

// (1) Finds first mode of histogram m1
// (2) Finds second mode of histogram by minimizing (x-m1)^2*m1
// (3) Reports midpoint between m1 and m2 (Parker, 2011, p.139 ).
pub struct SquareDist {
    pub p1 : u8,
    pub p2 : u8,
    pub p1_val : u32,
    pub p2_val : u32,
    pub min : usize,
    pub max : usize
}

fn sqdist(i : usize, count : u32, mode : usize) -> f32 {
    (i.abs_diff(mode) as f32).powf(2.)*(count as f32)
}

impl SquareDist {

    pub fn new(min : usize, max : usize) -> Self {
        Self { p1 : 0, p2 : 0, p1_val : 0, p2_val : 0, min, max }
    }

    pub fn calculate(&mut self, hist : &[u32]) -> u8 {
        let mode_max = hist[self.min..self.max].iter().enumerate()
            .max_by(|a, b| a.1.cmp(&b.1) )
            .unwrap().0 + self.min;
        let snd_mode_max = hist[self.min..self.max].iter().enumerate()
            .max_by(|a, b| sqdist(a.0, *a.1, mode_max).total_cmp(&sqdist(b.0, *b.1, mode_max)) )
            .unwrap().0 + self.min;
        self.p1 = mode_max as u8;
        self.p2 = snd_mode_max as u8;
        self.p1_val = hist[mode_max];
        self.p2_val = hist[snd_mode_max];
        ((mode_max as f32 + snd_mode_max as f32) / 2.) as u8
    }

}

// Ideal partition point between two gausians. Vary the parameter t in a root-finding algorithm (eqsolver)
// until the solution is zero. If there are two solutions, return the unique one between m1 and m2.
// m1, m2 are the means; s1, s2 the variances; p1, p2 the relative probabilities. (perhaps there is
// an error in the equation; first term is 1/s2+1/s2 in the book, eq. 4.43 at Parker, 2011).
fn gaussian_partition(t : f32, m1 : f32, s1 : f32, m2 : f32, s2 : f32, p1 : f32, p2 : f32) -> f32 {
    (1. / s1 + 1. / s2)*t.powf(2.) + 2.*(m2/s2 - m1/s1)*t + 2.*(((p2*s1)/(p1*s2)).ln())
}

/* Calls f iteratively over the histogram until either the value does not change or 256 iterations
where completed without a stability point found */
fn histogram_steps<T, F : Fn(&[T], usize)->usize>(f : F, hist : &[T], init : usize) -> Option<u8> {
    let mut u1 = init;
    let mut u2 = init;
    let mut niter = 0;
    while niter < 256 {
        u2 = f(hist, u1);
        assert!(u2 <= 255);
        if u1 == u2 {
            return Some(u1 as u8);
        }
        niter += 1;
        u1 = u2;
    }
    None
}

// (Naive method)
// (1) Set initial threshold estimate as average gray-level intensity
// (2) Calculate average gray-level of pixels classified as fg and bg
// (3) Set new estimate as (fg+bg)/2
// (4) Repeat 2-3 until threshold stop changing. (Parker, 2011, p. 140).
//
// (Histogram-based method: (faster))
// (1) Find full image histogram average
// (2) Set kth threshold by tk_{i+1} = (sum_0^{tk-1} (i*h[i]) / (2*(sum_0^{tk-1} h[i] ) )) + ((sum_{tk+1}^N j*h[j]) / (2*(sum_{tk+1}^N h[j])))
pub struct IterativeSelection {

}

impl IterativeSelection {

    pub fn eval(h : &[u32]) -> Option<u8> {
        let cumul = bayes::calc::running::cumulative_sum(h.iter().copied()).collect::<Vec<_>>();
        let cumul_prod = bayes::calc::running::cumulative_sum(h.iter().enumerate().map(|(ix, u)| ix as u32 * (*u) )).collect::<Vec<_>>();
        let step = |h : &[u32], t : usize| -> usize {
            let sum_prod_below = cumul_prod[t] as f32;
            let sum_below = cumul[t] as f32;
            let sum_prod_above = (cumul_prod[255] - cumul_prod[t]) as f32;
            let sum_above = (cumul[255] - cumul[t]) as f32;
            (sum_prod_below / (2.*sum_below) + sum_prod_above / (2.*sum_above)) as usize
        };
        let avg = bayes::calc::running::mean(h.iter().map(|h| *h as f64 ), h.len());
        histogram_steps(step, h, avg as usize)
    }
}

// maximizes uniformity over the foreground class (Hw) and background class (Hb)
// The entropy over a class is Hw = -sum pi log(pi) where pi is the normalized histogram.
// Maximizing the sum Hw and Hb is the same as maximizing
// f(t) [Ht/HT] * (log(Pt) / log(max{p0..pt})) + [1 - Ht/HT]*(log(1-Pt)/log(max{pt+1..255}))
// where Ht is the iteration-dependent entropy over background pixels (up to t);
// HT is the total entropy (calculated once)
// and Pt is the cumulative probability up to gray level t
// The maximum is found by exhaustively evaluating over all candidate t's.
pub struct MaxEntropy {

}

impl MaxEntropy {

    pub fn eval(h : &[u32]) -> Option<u8> {
        let probs = bayes::calc::counts_to_probs(h);
        let cumul_probs = bayes::calc::running::cumulative_sum(probs.iter().copied()).collect::<Vec<_>>();
        let cumul_entropies = bayes::calc::cumulative_entropy(probs.iter().copied()).collect::<Vec<_>>();
        let total_entropy = cumul_entropies[255];
        let step = |h : &[u32], t : usize| -> usize {
            let entropy_ratio = cumul_entropies[t] / total_entropy;
            let max_prob_below = probs[..t].iter().copied().max_by(f32::total_cmp).unwrap().ln();
            let max_prob_above = probs[t..].iter().copied().max_by(f32::total_cmp).unwrap().ln();
            let prob_below = cumul_probs[t].ln();
            (entropy_ratio*(prob_below / max_prob_below) + (1. - entropy_ratio)*((1. - prob_below) / max_prob_above)) as usize
        };
        let avg = bayes::calc::running::mean(h.iter().map(|h| *h as f64 ), h.len());
        histogram_steps(step, h, avg as usize)
    }
}

// Parker (2011 p.149)
// Evaluate J(t) exhaustively over candidate t's, where J(t) is:
// J(t) = 1 + 2*( P1(t)*log(s1(t)) + P2(t)*log(s2(t)) ) - 2*(P1(t)*log(P1(t) + P2(t)*log(P2(t))
// where P1(t) is the cumulative probability up to t
// P2(t) the cumulative probability from P to 255
// m1(t) = sum_0^t i*h[i]/P1(t)
// m2(t) = sum_{t+1}^{255} i*h[i]/P2(t)
// s1(t) = sum_0^t (h[i]*(i - m1(t))^2)/P1(t)
// s2(t) = sum_{t+1}^{255} (h[i]*(i - m2(t))^2)/P2(t)
pub struct MinError {
    step : usize,
    start : usize,
    end : usize
}

impl MinError {

    pub fn new(start : usize, end : usize, step : usize) -> Self {
        Self { start, end, step }
    }

    fn eval(&mut self, h : &[u32]) -> Option<u8> {
        let probs = bayes::calc::counts_to_probs(h);
        let cumul_probs = bayes::calc::running::cumulative_sum(probs.iter().copied()).collect::<Vec<_>>();
        let mut unnorm_means = bayes::calc::running::cumulative_sum(
            probs.iter().enumerate().map(|(i, p)| i as f32 * (*p) as f32)
        ).collect::<Vec<_>>();
        let mut unnorm_vars = bayes::calc::running::cumulative_sum(
            probs.iter().enumerate().map(|(i, p)| (i as f32 - (unnorm_means[i] / cumul_probs[i])).powf(2.) * (*p) as f32)
        ).collect::<Vec<_>>();
        let mut js = Vec::new();
        for t in (self.start..self.end).step_by(self.step) {
            let p1 = cumul_probs[t];
            let p2 = cumul_probs[255]-cumul_probs[t];
            let m1 = unnorm_means[t] / p1;
            let m2 = (unnorm_means[255] - unnorm_means[t]) / p2;
            let s1 = unnorm_vars[t] / p1;
            let s2 = (unnorm_vars[255] - unnorm_vars[t]) / p2;
            let j = 1. + 2.*(p1*s1.ln() + p2*s2.ln()) - 2.*(p1*p1.ln() + p2*p2.ln());
            js.push((t, j));
        }
        Some(js.iter().min_by(|a, b| a.1.total_cmp(&b.1) ).unwrap().0 as u8)
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

