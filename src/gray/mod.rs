use crate::image::*;
use std::default::Default;
// use crate::prelude::Raster;
use std::mem;
use bayes::calc::*;
use std::collections::BTreeMap;
use std::ops::Sub;
use crate::hist::{GrayHistogram, GridHistogram, ColorProfile};

// Alt name to truncate towards extremes: clipping (but really to max and min). This
// might be a special case of conditional_fill (but take as argument a scalar foreground
// instead of a binary image). 
// Alt name to invert: negative.
// 

#[derive(Debug, Clone, Copy)]
pub enum Foreground {

    Exactly(u8),

    Below(u8),

    Above(u8),

    Inside(u8, u8),

    Outside(u8, u8)

}

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

#[cfg(feature="ipp")]
#[derive(Debug, Clone)]
pub struct ErodedHysteresisOutput
{
    pub out : ImageBuf<u8>,
    pub low : ImageBuf<u8>,
    pub high : ImageBuf<u8>,
    pub low_eroded : ImageBuf<u8>,
    pub high_eroded : ImageBuf<u8>,
    pub dilate : crate::morph::IppiMorph,
    pub erode : crate::morph::IppiMorph,
}

#[cfg(feature="ipp")]
impl ErodedHysteresisOutput {

    pub fn output(&self) -> &ImageBuf<u8> {
        &self.out
    }

    pub fn output_mut(&mut self) -> &mut ImageBuf<u8> {
        &mut self.out
    }

    pub fn new<S>(height : usize, width : usize, se : &Image<u8, S>) -> Self
    where
        S : Storage<u8>
    {
        let low = ImageBuf::<u8>::new_constant(height, width, 0);
        let out = low.clone();
        let high = low.clone();
        let low_eroded = low.clone();
        let high_eroded = high.clone();
        let dilate = crate::morph::IppiMorph::new(se.clone_owned(), (height, width), true);
        let erode = crate::morph::IppiMorph::new(se.clone_owned(), (height, width), false);
        Self { low, out, high, dilate, erode, low_eroded, high_eroded }
    }

}

#[cfg(feature="ipp")]
#[derive(Debug, Clone)]
pub struct HysteresisOutput
{
    pub low : ImageBuf<u8>,
    pub out : ImageBuf<u8>,
    pub high : ImageBuf<u8>,
    pub dilate : crate::morph::IppiMorph
}

#[cfg(feature="ipp")]
impl HysteresisOutput {

    pub fn output(&self) -> &ImageBuf<u8> {
        &self.out
    }
    
    pub fn output_mut(&mut self) -> &mut ImageBuf<u8> {
        &mut self.out
    }
    
    pub fn new<S>(height : usize, width : usize, se : &Image<u8, S>) -> Self 
    where
        S : Storage<u8>
    {
        let low = ImageBuf::<u8>::new_constant(height, width, 0);
        let out = low.clone();
        let high = low.clone();
        let dilate = crate::morph::IppiMorph::new(se.clone_owned(), (height, width), true);
        Self { low, out, high, dilate }
    }
    
}

impl ImageBuf<u8> {

    // After calling that, the pixels in the image are left
    // sorted up to i, while remaining pixels are left at an unspecified
    // order.
    pub fn nth_pixel_inplace(&mut self, p : f32) -> u8 {
        assert!(p >= 0. && p <= 1.0);
        let q = (self.area() as f32 * p) as usize;
        let (_, v, _) = self.as_mut_slice().select_nth_unstable(q);
        *v
    }

}

impl<S> Image<u8, S>
where
    S : StorageMut<u8>
{

    pub fn truncate_inplace(&mut self, fg : Foreground, fg_val : u8)
    {        
        #[cfg(feature="ipp")]
        unsafe { return ippi_truncate(fg, None, &mut self.full_window_mut(), fg_val) };
        
        unimplemented!()
    }
    
}

impl<S> Image<u8, S> 
where
    S : Storage<u8>
{

    /* Count pixels in the interval [low, high]. */
    pub fn count_pixels(&self, low : u8, high : u8) -> u32 {

        #[cfg(feature="ipp")]
        unsafe {
            let mut count : i32 = 0;
            let ans = crate::foreign::ipp::ippi::ippiCountInRange_8u_C1R(
                self.as_ptr(),
                self.byte_stride() as i32,
                self.size().into(),
                &mut count as *mut _,
                low,
                high
            );
            assert!(ans == 0);
            return count as u32;
        }

        let mut n = 0u32;
        for px in self.pixels(1) {
            if *px >= low && *px <= high {
                n += 1;
            }
        }
        n
    }

    pub fn iter_foreground<'a>(&'a self, fg : Foreground) -> Box<(dyn Iterator<Item=&'a u8> + 'a)> {
        match fg {
            Foreground::Exactly(px) => {
                Box::new(self.pixels(1).filter(move |p| **p == px ))
            },
            Foreground::Below(px) => {
                Box::new(self.pixels(1).filter(move |p| **p <= px ))
            },
            Foreground::Above(px) => {
                Box::new(self.pixels(1).filter(move |p| **p >= px ))
            },
            Foreground::Inside(a, b) => {
                Box::new(self.pixels(1).filter(move |p| **p >= a && **p <= b ))
            },
            Foreground::Outside(a, b) => {
                Box::new(self.pixels(1).filter(move |p| **p < a && **p > b ))
            }
        }
    }
    
    #[cfg(feature="ipp")]
    pub fn hysteresis_threshold_to(&self, fg_low : Foreground, fg_high : Foreground, out : &mut HysteresisOutput) 
    {
        self.threshold_to(fg_low, &mut out.low);
        self.threshold_to(fg_high, &mut out.high);
        out.dilate.apply(&out.low, &mut out.out);
        out.out.and_assign(&out.high);
    }

    #[cfg(feature="ipp")]
    pub fn eroded_hysteresis_threshold_to(&self, fg_low : Foreground, fg_high : Foreground, out : &mut ErodedHysteresisOutput)
    {
        self.threshold_to(fg_low, &mut out.low);
        self.threshold_to(fg_high, &mut out.high);
        out.erode.apply(&out.low, &mut out.low_eroded);
        out.erode.apply(&out.high, &mut out.high_eroded);
        out.dilate.apply(&out.low_eroded, &mut out.out);
        out.out.and_assign(&out.high_eroded);
    }
    
    pub fn local_area_to<T>(&self, dst : &mut Image<u8, T>)
    where
        T : StorageMut<u8>
    {
        assert!(self.height() % dst.height() == 0);
        assert!(self.width() % dst.width() == 0);
        let wins = self.windows((self.height() / dst.height(), self.width() / dst.width()));
        for (w, mut px) in wins.zip(dst.pixels_mut(1)) {
            *px = w.count_pixels(254, 255) as u8;
        }
    }

    /* Binarizes a bit image, setting all unmatched pixels to zero, and matched pixels
    to a foreground value. */
    pub fn threshold_to<T>(&self, fg : Foreground, out : &mut Image<u8, T>) 
    where
        T : StorageMut<u8>
    {
        assert!(self.shape() == out.shape());
        #[cfg(feature="ipp")]
        unsafe { return ippi_full_threshold(fg, &self.full_window(), &mut out.full_window_mut()); }
        
        // baseline_threshold(&self.full_window(), fg, &mut out.full_window_mut());
        panic!()
    }
    
    pub fn bit_threshold_to<T>(&self, fg : Foreground, out : &mut Image<u8, T>)
    where
        T : StorageMut<u8>
    {
        self.threshold_to(fg, out);
        byte_to_bit_inplace(out);
    }

    /*
    Sets all pixels matching foreground to the maximum or minimum value of the pixel,
    according to the following rule:
    If Foreground is below, sets matching pixels to their minimum value;
    If Foreground is above, sets matching pixels to their maximum value;
    If foreground is inside/outside, do nothing.
    Unlike threshold, that always attribute a binary value to all pixels, truncate
    leaves unmatched pixels untouched.
    */
    pub fn truncate_to<T>(&self, fg : Foreground, out : &mut Image<u8, T>, fg_val : u8) 
    where
        T : StorageMut<u8>
    {
        assert!(self.shape() == out.shape());
        
        #[cfg(feature="ipp")]
        unsafe { return ippi_truncate(fg, Some(&self.full_window()), &mut out.full_window_mut(), fg_val) };
        
        baseline_truncate(&self.full_window(), fg, &mut out.full_window_mut(), fg_val);
    }
    
    /// (1) Calculate partition values for each sub-window
    /// (Or the full window if win_sz is None).
    /// (3) Apply threshold using this partition value to each sub-window
    /// (or the full window if win_sz is None).
    pub fn adaptive_threshold_to<T>(
        &self,
        alg : &mut impl AdadptiveThreshold,
        dst : &mut Image<u8, T>
    ) where
        T : StorageMut<u8>
    {
        assert!(self.shape() == dst.shape());
        let bv = alg.best_values(&self.full_window());
        let mut i = 0;
        let sub_sz = alg.sub_sz();
        let win_iter = dst.windows_mut((sub_sz.0, sub_sz.1))
            .zip(self.windows((sub_sz.0, sub_sz.1)));
        for (mut sub_dst, sub_src) in win_iter {
            sub_src.threshold_to(bv[i], &mut sub_dst);
            i += 1;
        }
        assert!(i == bv.len());
    }
    
    // Allocating version of threshold_to
    pub fn threshold(&self, fg : Foreground) -> ImageBuf<u8> {
        let mut out = unsafe { ImageBuf::<u8>::new_empty(self.height(), self.width()) };
        self.threshold_to(fg, &mut out);
        out
    }
    
    pub fn truncate(&self, fg : Foreground, fg_val : u8) -> ImageBuf<u8> {
        // Image cannot be created as empty here because the truncate operation leaves
        // unmatched pixels untouched, so we must guarantee there is valid data at
        // all pixels.
        let mut out = ImageBuf::<u8>::new_constant(self.height(), self.width(), 0);
        self.truncate_to(fg, &mut out, fg_val);
        out
    }
    
    // Allocating version of adapting_threshold_to
    pub fn adaptive_threshold(&self, alg : &mut impl AdadptiveThreshold) -> ImageBuf<u8> {
        let mut out = unsafe { ImageBuf::<u8>::new_empty(self.height(), self.width()) };
        self.adaptive_threshold_to(alg, &mut out);
        out
    }
    
    // adaptive_truncate

}

pub trait AdadptiveThreshold {

    fn new(
        win_sz : (usize, usize), 
        sub_sz : Option<(usize, usize)>, 
        fg : AdaptiveForeground
    ) -> Self;

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

pub fn copy_if_above_threshold<S>(
    dst_orig : &Window<'_, u8>, 
    orig : (usize, usize), 
    win_dim : (usize, usize), 
    threshold : i32, 
    dst : &mut Image<u8, S>
) 
where 
    S : StorageMut<u8>,
    Box<[u8]> : StorageMut<u8>,
    for<'a> &'a [u8] : Storage<u8>,
    for<'a> &'a mut [u8] : StorageMut<u8>
{
    let sub = dst_orig.window(orig, win_dim).unwrap();
    if sub.sum::<f32>(1) as i32 > threshold {
        dst.window_mut(orig, win_dim).unwrap()
            .copy_from(&dst_orig.window(orig, win_dim).unwrap());
    }
}

/*pub fn zero_sparse_regions(
    dst : &mut Image<u8>, 
    sum : &mut WindowMut<'_, i32>, 
    win_dim : (usize, usize), 
    min_pts : usize
) {

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

}*/


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

/*pub trait Threshold
where
    Self : Raster
{

    // fn adaptive_threshold_to(&self, fg : Foreground, partition : &mut dyn Partition, out : &mut WindowMut<'a, u8>);

    // fn adaptive_threshold(&self, fg : Foreground, partition : &mut dyn Partition) -> Image<u8>;

    // Binarize an image, setting foreground pixels to u8::MAX and background pixels to u8::MIN
    fn threshold_to(&self, fg : Foreground, out : &mut WindowMut<u8>);

    
}*/

/*pub trait Truncate
where
    Self : Raster
{

    // fn adaptive_truncate_to
    // fn adaptive_truncate

    // Truncate an image, setting foreground pixels to the desired value, and copying
    // unmatched pixels as they are.
    fn truncate_to(&self, fg : Foreground, out : &mut WindowMut<u8>, fg_val : u8);

    // Allocating version of truncate_to
    

}*/

/*impl<'a> Threshold for Window<'a, u8> {
}*/

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

fn baseline_truncate_inplace(fg : Foreground, out : &mut WindowMut<u8>, fg_val : u8) {
    match fg {
        Foreground::Exactly(v) => {
            out.pixels_mut(1).for_each(|px_out| if *px_out == v { *px_out = fg_val } );
        },
        Foreground::Below(v) => {
            out.pixels_mut(1).for_each(|px_out| if *px_out <= v { *px_out = fg_val } );
        },
        Foreground::Above(v) => {
            out.pixels_mut(1).for_each(|px_out| if *px_out >= v { *px_out = fg_val } );
        },
        Foreground::Inside(a, b) => {
            out.pixels_mut(1).for_each(|px_out| if *px_out >= a && *px_out <= b { *px_out = fg_val } );
        },
        Foreground::Outside(a, b) => {
            out.pixels_mut(1).for_each(|px_out| if *px_out < a && *px_out > b { *px_out = fg_val } );
        }
    }
}

fn baseline_truncate(src : &Window<u8>, fg : Foreground, out : &mut WindowMut<u8>, fg_val : u8) {
    match fg {
        Foreground::Exactly(v) => {
            out.pixels_mut(1).zip(src.pixels(1))
                .for_each(|(px_out, px_in)| if *px_in == v { *px_out = fg_val } else { *px_out = *px_in });
        },
        Foreground::Below(v) => {
            out.pixels_mut(1).zip(src.pixels(1))
                .for_each(|(px_out, px_in)| if *px_in <= v { *px_out = fg_val } else { *px_out = *px_in });
        },
        Foreground::Above(v) => {
            out.pixels_mut(1).zip(src.pixels(1))
                .for_each(|(px_out, px_in)| if *px_in >= v { *px_out = fg_val } else { *px_out = *px_in });
        },
        Foreground::Inside(a, b) => {
            out.pixels_mut(1).zip(src.pixels(1))
                .for_each(|(px_out, px_in)| if *px_in >= a && *px_in <= b { *px_out = fg_val } else { *px_out = *px_in });
        },
        Foreground::Outside(a, b) => {
            out.pixels_mut(1).zip(src.pixels(1))
                .for_each(|(px_out, px_in)| if *px_in < a && *px_in > b { *px_out = fg_val } else { *px_out = *px_in });
        }
    }
}

/*impl<'a> Truncate for Window<'a, u8> {
 
}*/

/*impl Threshold for Image<u8> {

    fn threshold_to(&self, fg : Foreground, out : &mut WindowMut<u8>) {
        self.full_window().threshold_to(fg, out);
    }

}

impl Truncate for Image<u8> {

    fn truncate_to<'a>(&self, fg : Foreground, out : &mut WindowMut<'a, u8>, fg_val : u8) {
        self.full_window().truncate_to(fg, out, fg_val);
    }

}*/

impl<S> Image<i16, S>
where
    S : StorageMut<i16>
{

    pub fn truncate_negative_inplace(&mut self) {
        let threshold = 0i16;
        let fg_val = 0i16;
        unsafe {
            let ans = crate::foreign::ipp::ippi::ippiThreshold_LTVal_16s_C1IR(
                self.as_mut_ptr(),
                self.byte_stride() as i32,
                self.size().into(),
                threshold,
                fg_val,
            );
            assert!(ans == 0);
        }
    }

}

// If desired operation is inplace, pass None to source and Some(w) to src/dst.
// Else pass Some(src) and the destination.
#[cfg(feature="ipp")]
unsafe fn ippi_truncate(
    fg : Foreground,
    src : Option<&Window<u8>>,
    out : &mut WindowMut<u8>,
    fg_val : u8
) {
    let (src_byte_stride, roi) = if let Some(src) = src {
        crate::image::ipputils::step_and_size_for_window(src)
    } else {
        crate::image::ipputils::step_and_size_for_window_mut(&out)
    };
    let dst_byte_stride = crate::image::ipputils::byte_stride_for_window_mut(&out);
    match fg {
        Foreground::Exactly(v) => {
            if src.is_some() {
                baseline_truncate(src.unwrap(), fg, out, fg_val);
            } else {
                baseline_truncate_inplace(fg, out, fg_val);
            }
        },
        Foreground::Below(v) => {
            let ans = if let Some(src) = src {
                crate::foreign::ipp::ippi::ippiThreshold_LTVal_8u_C1R(
                    src.as_ptr(),
                    src_byte_stride,
                    out.as_mut_ptr(),
                    dst_byte_stride,
                    roi,
                    v.saturating_add(1),
                    fg_val,
                )
            } else {
                crate::foreign::ipp::ippi::ippiThreshold_LTVal_8u_C1IR(
                    out.as_mut_ptr(),
                    dst_byte_stride,
                    roi,
                    v.saturating_add(1),
                    fg_val,
                )
            };
            assert!(ans == 0);
        },
        Foreground::Above(v) => {
            let ans = if let Some(src) = src {
                crate::foreign::ipp::ippi::ippiThreshold_GTVal_8u_C1R(
                    src.as_ptr(),
                    src_byte_stride,
                    out.as_mut_ptr(),
                    dst_byte_stride,
                    roi,
                    v.saturating_sub(1),
                    fg_val
                )
            } else {
                crate::foreign::ipp::ippi::ippiThreshold_GTVal_8u_C1IR(
                    out.as_mut_ptr(),
                    dst_byte_stride,    
                    roi,
                    v.saturating_sub(1),
                    fg_val
                )
            };
            assert!(ans == 0);
        },
        Foreground::Inside(a, b) => {
            if src.is_some() {
                baseline_truncate(src.unwrap(), fg, out, fg_val);
            } else {
                baseline_truncate_inplace(fg, out, fg_val);
            }
        },
        Foreground::Outside(a, b) => {
            let ans = if let Some(src) = src {
                crate::foreign::ipp::ippi::ippiThreshold_LTValGTVal_8u_C1R(
                    src.as_ptr(),
                    src_byte_stride,
                    out.as_mut_ptr(),
                    dst_byte_stride,
                    roi,
                    a,
                    fg_val,
                    b,
                    fg_val,
                )
            } else {
                crate::foreign::ipp::ippi::ippiThreshold_LTValGTVal_8u_C1IR(
                    out.as_mut_ptr(),
                    dst_byte_stride,
                    roi,
                    a,
                    fg_val,
                    b,
                    fg_val,
                )
            };
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
            out.not_mut();
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

// use crate::feature::patch::ColorProfile;

#[cfg(feature="ipp")]
pub fn ipp_otsu<S>(img : &Image<u8, S>) -> u8
where
    S : Storage<u8>
{
    let mut thr : u8 = 0;
    unsafe {
        let status = crate::foreign::ipp::ippi::ippiComputeThreshold_Otsu_8u_C1R(
            img.as_ptr(),
            img.byte_stride() as i32,
            img.size().into(),
            &mut thr as *mut _
        );
        assert!(status == 0);
    }
    thr
}

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
    pub fn estimate(&self, hist : &ColorProfile, step : usize) -> u8 {

        let num_pixels = hist.num_pxs();
        let mut max_th = 0;
        let mut max_inter_var = 0.;

        let probs = hist.probabilities();

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

use std::ops::Range;

#[derive(Debug, Clone)]
pub struct MedianPartition {
    pub medians : Vec<usize>,
    pub bins : Vec<Range<usize>>
}

fn median_for_bin(acc : &[u32], bin : &Range<usize>) -> usize {
    let (bin_start, bin_end) = (bin.start, bin.end);
    let half_total = (acc[bin.end-1] - acc[bin.start]) / 2;
    let mut median = acc[bin.clone()].partition_point(|b| (b - acc[bin.start]) < half_total );
    median += bin_start;
    median
}

/// Recursively partition the image histogram by re-calculating the median values within bins.
/// Each resulting range is a valid color partition in [0,256] that is useful to quantize the color.
/// The median cut will distribute large intervals for sparsely-populated image regions; and short intervals
/// for densely-populated image regions. The result will have 2^n_partitions bins.
fn median_partitions<S>(w : &Image<u8, S>, n_partitions : usize, merge_diff : Option<usize>) -> MedianPartition 
where
    S : Storage<u8>
{
    
    let gh = GrayHistogram::calculate(&w.full_window());
    let acc = gh.accumulate();
    
    let mut bins = vec![Range { start : 0, end : 256 }];
    let mut medians = Vec::new();
    for _ in 0..n_partitions {
        medians.clear();
        for i in 0..bins.len() {
            let median = median_for_bin(acc.as_slice(), &bins[i]);
            let bin_end = bins[i].end;
            
            // Re-write first half at the same index.
            bins[i] = Range { start : bins[i].start, end : median };
            
            // Push second half to a new index.
            bins.push(Range { start : median, end : bin_end });
            
            medians.push(median);
        }
    }
    
    bins.sort_by(|a, b| a.start.cmp(&b.start) );
    medians.clear();
    for bin in &bins {
        medians.push(median_for_bin(acc.as_slice(), &bin));
    }
    
    if let Some(diff) = merge_diff {
        for i in (0..medians.len()-1).rev() {
            if medians[i].abs_diff(medians[i+1]) <= diff {
                let n1 = acc[bins[i].end-1] - acc[bins[i].start]; 
                let n2 = acc[bins[i+1].end-1] - acc[bins[i+1].start];
                let w1 = n1 as f32 / (n1+n2) as f32;
                let w2 = n2 as f32 / (n1+n2) as f32;
                medians[i] = (w1 * medians[i] as f32 + w2 * medians[i+1] as f32) as usize;
                bins[i].end = bins[i+1].end;
                bins.remove(i+1);
                medians.remove(i+1);
            }
        }
    }
    
    MedianPartition { bins, medians }
    
}

use itertools::Itertools;

pub trait Quantization {

    fn quantize_to<S, T>(&mut self, img : &Image<u8, S>, dst : &mut Image<u8, T>)
    where
        S : Storage<u8>,
        T : StorageMut<u8>;
        
}

pub struct MedianCutQuantization {
    n_parts : usize,
    merge_diff : Option<usize>
}

impl MedianCutQuantization {

    pub fn new(n_parts : usize, merge_diff : Option<usize>) -> Self {
        Self { n_parts, merge_diff }
    }
    
}

#[derive(Clone)]
#[cfg(feature="ipp")]
pub struct IppReduceBits {
    noise : i32,
    dither : crate::foreign::ipp::ippi::IppiDitherType,
    buf : Vec<u8>,
    levels : u32
}

#[cfg(feature="ipp")]
impl IppReduceBits {

    pub fn new(height : usize, width : usize, noise : Option<i32>, levels : u32) -> Self {
        // This is actually in ippcc
        if let Some(noise) = noise {
            assert!(noise >= 0 && noise <= 100);
        }
        let chan = crate::foreign::ipp::ippi::IppChannels_ippC1;
        let dither = if noise.is_some() {
            crate::foreign::ipp::ippi::IppiDitherType_ippDitherStucki
        } else {
            crate::foreign::ipp::ippi::IppiDitherType_ippDitherNone
        };
        let mut sz : i32 = 0;
        unsafe {
            let ans = crate::foreign::ipp::ippcc::ippiReduceBitsGetBufferSize(
                chan,
                std::mem::transmute(crate::foreign::ipp::ippi::IppiSize::from((height, width))),
                noise.unwrap_or(0),
                dither,
                &mut sz as *mut _
            );
            assert!(ans == 0);
            // assert!(sz > 0);
            let buf : Vec<_> = (0..sz).map(|_| 0u8 ).collect();
            Self { buf, noise : noise.unwrap_or(0), dither, levels }
        }
    }

    pub fn calculate<S, T>(&mut self, src : &Image<u8, S>, dst : &mut Image<u8, T>)
    where
        S : Storage<u8>,
        T : StorageMut<u8>
    {
        // This is actually in ippcc
        unsafe {
            let ans = crate::foreign::ipp::ippcc::ippiReduceBits_8u_C1R(
                src.as_ptr(),
                src.byte_stride() as i32,
                dst.as_mut_ptr(),
                dst.byte_stride() as i32,
                std::mem::transmute(crate::foreign::ipp::ippi::IppiSize::from(src.size())),
                self.noise,
                self.dither,
                self.levels as i32,
                self.buf.as_mut_ptr()
            );
            assert!(ans == 0);
        }
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
IppStatus ippiToneMapLinear_32f8u_C1R(const Ipp32f* pSrc, int srcStep, Ipp8u* pDst, int
dstStep, IppiSize roiSize);
*/

/*#[cfg(feature="ipp")]
pub struct IppiLUT {

}

#[cfg(feature="ipp")]
impl IppiLUT {

    pub fn new(sz : (usize, usize)) -> Self {
    
        let ans = ippiLUT_GetSize(IppiInterpolationType interpolation, IppDataType dataType,
IppChannels channels, IppiSize roiSize, const int nLevels[], int* pSpecSize);
        
        assert!(ans == 0);
        let ans = ippiLUT_Init_8u(
            IppiInterpolationType interpolation, IppChannels channels,
IppiSize roiSize, const Ipp32s* pValues[], const Ipp32s* pLevels[], int nLevels[],
IppiLUT_Spec* pSpec);
        assert!(ans == 0);
        Self { }
    }
    
    pub fn lookup<S, T>(&mut self, src : &Image<u8, S>, dst : &Image<u8, T>)
    where
        S : Storage<u8>,
        T : StorageMut<u8>
    {
        let ans = ippiLUT_8u_C1R(const Ipp<datatype>* pSrc, int srcStep, Ipp<datatype>* pDst,
        int dstStep, IppiSize roiSize, IppiLUT_Spec* pSpec);
        assert!(ans == 0);
    }
    
}*/

#[cfg(feature="ipp")]
pub unsafe fn ipp_lut_palette<P, Q, S, T>(src : &Image<P, S>, dst : &mut Image<Q, T>, table : &[Q])
where
    P : Pixel,
    Q : Pixel,
    S : Storage<P>,
    T : StorageMut<Q>
{
    let (src_step, sz) = crate::image::ipputils::step_and_size_for_image(src);
    let (dst_step, _) = crate::image::ipputils::step_and_size_for_image(dst);

    // assert!(tbl.len() == (2).pow(bit_sz));
    let bit_sz = (table.len() as f32).log(2.0) as i32;
    if src.pixel_is::<u8>() && dst.pixel_is::<u8>() {
        // let bit_sz = 8;
        let ans = crate::foreign::ipp::ippi::ippiLUTPalette_8u_C1R(
            std::mem::transmute(src.as_ptr()),
            src_step,
            std::mem::transmute(dst.as_mut_ptr()),
            dst_step,
            sz,
            std::mem::transmute(table.as_ptr()),
            bit_sz
        );
        assert!(ans == 0);
        return;
    }

    if src.pixel_is::<u16>() && dst.pixel_is::<u16>() {
        // let bit_sz = 16;
        let ans = crate::foreign::ipp::ippi::ippiLUTPalette_16u_C1R(
            std::mem::transmute(src.as_ptr()),
            src_step,
            std::mem::transmute(dst.as_mut_ptr()),
            dst_step,
            sz,
            std::mem::transmute(table.as_ptr()),
            bit_sz
        );
        assert!(ans == 0);
        return;
    }

    if src.pixel_is::<u8>() && dst.pixel_is::<u32>() {
        // let bit_sz = 32;
        let ans = crate::foreign::ipp::ippi::ippiLUTPalette_8u32u_C1R(
            std::mem::transmute(src.as_ptr()),
            src_step,
            std::mem::transmute(dst.as_mut_ptr()),
            dst_step,
            sz,
            std::mem::transmute(table.as_ptr()),
            bit_sz
        );
        assert!(ans == 0);
        return;
    }

    unimplemented!()
}

/// Represents a lookup index. Colors in a test image should be used as indices to the colors 
/// contained in the array.
#[derive(Debug, Clone)]
pub struct Lookup([u8; 256]);

impl Lookup {

    pub fn from_slice(s : &[u8]) -> Self {
        let mut arr : [u8; 256] = [0; 256];
        arr.copy_from_slice(s);
        Self(arr)
    }
    
    // ranges must be mutually exclusive and sorted. Each range will receive
    // the corresponding entry at value.
    pub fn from_ranges(ranges : &[Range<usize>], values : &[u8]) -> Self {
    
        println!("{:?}", (&ranges, &values));
        assert!(ranges.len() == values.len());
        assert!(ranges[0].start == 0);
        assert!(ranges.last().unwrap().end == 256);
        for i in 1..ranges.len() {
            assert!(ranges[i].start == ranges[i-1].end);
        }
        
        let mut dst : [u8; 256] = [0; 256];
        let mut curr_range = 0;
        for i in 0..256 {
            if i >= ranges[curr_range].end {
                curr_range += 1;
            }
            dst[i] = values[curr_range];            
        }
        Self(dst)
    }
    
    pub fn lookup_inplace<T>(&self, img : &mut Image<u8, T>)
    where
        T : StorageMut<u8>
    {
        for i in 0..img.height() {
            for j in 0..img.width() {
                img[(i, j)] = self[img[(i, j)]];
            }
        }
    }
    
    pub fn lookup_to<S, T>(&self, src : &Image<u8, S>, dst : &mut Image<u8, T>)
    where
        S : Storage<u8>,
        T : StorageMut<u8>
    {
    
        #[cfg(feature="ipp")]
        unsafe {
            return ipp_lut_palette(src, dst, &self.0[..]);
        }
        assert!(src.shape() == dst.shape());
        for i in 0..src.height() {
            for j in 0..src.width() {
                dst[(i, j)] = self[src[(i, j)]];
            }
        }
    }
    
}

use std::ops::Index;

impl Index<u8> for Lookup {

    type Output = u8;
    
    fn index(&self, ix : u8) -> &u8 {
        &self.0[ix as usize]
    }

}

impl Quantization for MedianCutQuantization {

    fn quantize_to<S, T>(&mut self, img : &Image<u8, S>, dst : &mut Image<u8, T>)
    where
        S : Storage<u8>,
        T : StorageMut<u8>
    {
        let MedianPartition { bins, mut medians } = median_partitions(img, self.n_parts, self.merge_diff);
        
        let mut median_bytes = Vec::new();
        for m in medians.drain(..) {
            median_bytes.push(m as u8);
        }
        let lookup = Lookup::from_ranges(&bins[..], &median_bytes[..]);
        assert!(img.shape() == dst.shape());
        lookup.lookup_to(img, dst);
    }
    
}

#[test]
fn test_median_cut() {
    let mut buf = crate::io::decode_from_file("/home/diego/Downloads/pinpoint.png").unwrap();
    let mut qt = MedianCutQuantization::new(4, None);
    qt.quantize_to(&buf.clone(), &mut buf);
    buf.show();
}

/// Represents an partition of the intensity space without any content-based criteria.
/// Bins are spaced uniformly at either a linear or log(intensity) scale.
pub struct UniformQuantization {
    log : bool
}

pub fn byte_to_bit<S, T>(byte : &Image<u8, S>, bit : &mut Image<u8, T>)
where
    S : Storage<u8>,
    T : StorageMut<u8>
{
    // Solution (1)
    // byte.scalar_sub_to(1, bit);
    // bit.scalar_xor_assign(255);

    // This can easily be done in-place too.
    // byte.scalar_sub_to(1, bit);
    // bit.bitwise_not_inplace();

    // This assumes saturation of 0 entries.
    // byte.scalar_sub_to(254, bit);

    unimplemented!()
}

pub fn byte_to_bit_inplace<T>(img : &mut Image<u8, T>)
where
    T : StorageMut<u8>
{
    // This assumes saturation of 0 entries.
    img.scalar_sub_mut(254);
}

// Output: px > (local avg - delta)
#[cfg(feature="ipp")]
#[derive(Debug, Clone)]
pub struct IppiThrAdaptBox {
    buffer : Vec<u8>,
    mask : (usize, usize),
    delta : f32
}

#[cfg(feature="ipp")]
impl IppiThrAdaptBox {

    pub fn new(height : usize, width : usize, mask_side : usize, delta : f32) -> Self {
        let data_ty = crate::foreign::ipp::ippi::IppDataType_ipp8u;
        let n_channels = 1;
        let mut buf_sz = 0;
        assert!(mask_side % 2 != 0);
        let mask = (mask_side, mask_side);
        unsafe {
            let status = crate::foreign::ipp::ippi::ippiThresholdAdaptiveBoxGetBufferSize(
                (height, width).into(),
                mask.into(),
                data_ty,
                n_channels,
                &mut buf_sz as *mut _
            );
            assert!(status == 0);
            assert!(buf_sz > 0);
            let mut buffer = Vec::with_capacity(buf_sz as usize);
            buffer.set_len(buf_sz as usize);
            Self { buffer, mask, delta }
        }
    }

    pub fn apply<S, T>(&mut self, img : &Image<u8, S>, dst : &mut Image<u8, T>)
    where
        S : Storage<u8>,
        T : StorageMut<u8>
    {
        let border_val : u8 = 0;
        // let border_ty = crate::foreign::ipp::ippi::_IppiBorderType_ippBorderConst;
        let border_ty = crate::foreign::ipp::ippi::_IppiBorderType_ippBorderRepl;
        let val_gt = 255;
        let val_le = 0;
        unsafe {
            let status = crate::foreign::ipp::ippi::ippiThresholdAdaptiveBox_8u_C1R(
                img.as_ptr(),
                img.byte_stride() as i32,
                dst.as_mut_ptr(),
                dst.byte_stride() as i32,
                img.size().into(),
                self.mask.into(),
                self.delta,
                val_gt,
                val_le,
                border_ty,
                border_val,
                self.buffer.as_mut_ptr()
            );
            assert!(status == 0);
        }
    }
}

#[cfg(feature="ipp")]
#[derive(Debug, Clone)]
pub struct IppiSAD {
    buffer : Vec<u8>
}

#[cfg(feature="ipp")]
impl IppiSAD {

    pub fn new(img_sz : (usize, usize), templ_sz : (usize, usize)) -> Self {
        let data_ty = crate::foreign::ipp::ippi::IppDataType_ipp8u;
        let num_channels = 1;
        let roi_shape = crate::foreign::ipp::ippi::IppiROIShape_ippiROIValid;
        unsafe {
            let mut buf_sz = 0;
            let status = crate::foreign::ipp::ippi::ippiSADGetBufferSize(
                img_sz.into(),
                templ_sz.into(),
                data_ty,
                num_channels,
                roi_shape,
                &mut buf_sz as *mut _
            );
            assert!(status == 0);
            // assert!(buf_sz > 0);
            let mut buffer = Vec::with_capacity(buf_sz as usize);
            buffer.set_len(buf_sz as usize);
            Self { buffer }
        }
    }

    pub fn calculate<R, S, T>(&mut self, src : &Image<u8, R>, template : &Image<u8, S>, dst : &mut Image<i32, T>)
    where
        R : Storage<u8>,
        S : Storage<u8>,
        T : StorageMut<i32>
    {
        let roi_shape = crate::foreign::ipp::ippi::IppiROIShape_ippiROIValid;
        let scale = 1;
        assert!(dst.height() == src.height() - template.height() + 1);
        assert!(dst.width() == src.width() - template.width() + 1);
        unsafe {
            let status = crate::foreign::ipp::ippi::ippiSAD_8u32s_C1RSfs(
                src.as_ptr(),
                src.byte_stride() as i32,
                src.size().into(),
                template.as_ptr(),
                template.byte_stride() as i32,
                template.size().into(),
                dst.as_mut_ptr(),
                dst.byte_stride() as i32,
                roi_shape,
                scale,
                self.buffer.as_mut_ptr()
            );
            assert!(status == 0);
        }
    }
}

// Histogram stretching/equalization.

// Important! Image cannot contain zero values (call scalar_add(1) first)
pub fn log_to<S, T>(src : &Image<u8, S>, dst : &mut Image<u8, T>)
where
    S : Storage<u8>,
    T : StorageMut<u8>
{
    let scale_factor = 1;
    unsafe {
        let status = crate::foreign::ipp::ippi::ippiLn_8u_C1RSfs(
            src.as_ptr(),
            src.byte_stride() as i32,
            dst.as_mut_ptr(),
            dst.byte_stride() as i32,
            src.size().into(),
            scale_factor
        );
        assert!(status == 0);
    }
}

// Moeslund (2012)
pub fn enhance_range() {
    // Maps range [low-high] to the whole integer domain of the image.
    // Truncate values below low to 0 and values above high to 255.
    // let ampl = 255 / (high - low);
    // a.scalar_mul_to(&mut dst, ampl);
    // dst.scalar_sub_mut(ampl * low);
}

// Moeslund (2012)
fn image_log() {
    // Brings detail from dark images
    // c = 255 / (log(1 + max))
    // img.scalar_mul_mut(c * log(1 + px))
}

// Moeslund (2012)
fn image_exp() {
    // Brings detail from bright images
    // c = 255 / (k^max - 1)
    // img.scalar_mul_mut(c * k^px - 1)
}

fn histogram_stretch() {
    // Call enhance_range with histogram minimum and maximum.
}

fn histogram_equalize() {
    // Divide cumulative histogram by total number of pixels
    // Multiply resulting ratio by 255
    // Use histogram as a LUT to lookup each y corresponding to each x.
}


