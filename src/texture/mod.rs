use crate::image::*;
use std::convert::{AsRef, AsMut};
use std::ops::BitAnd;
use std::ops::BitXor;
use std::collections::BTreeMap;

// Texture classification approaches (after Distante & Distante):
// Statistical: Calculate local histogram features (mean, variance, smoothness)
// Structural (geometrical): Match local regions to a set of pre-defined geometrical primitives
// Stochastic (probabilistic): Consider that texture is the realization of a random process.

// Reference: Cavalin & Oliveira (2017). A Review of Texture Classification Methods and Databases.

// Texture descriptor
pub trait Descriptor {

    fn contrast();

    fn energy();

    fn entropy();

    fn homogeneity();

    fn maxlikelihood();

    fn correlation();

    fn snd_momentum();

    fn trd_momentum();

}

use crate::image::*;

// Statistical approach calculate a vector of statistic from the local intensity histogram.
// statistical measures after Distante & Distante (2020).
pub mod stat {

    use super::*;

    // Smoothness tends to zero at homogeneous regions; And to 1.0 at rough
    // areas (Distante & Distante v.3 eq. 3.4). This is the complement of
    // homogeneity?
    pub fn smoothness(var : f64) -> f64 {
        1. - (1. / (1. + var))
    }

    // Maps Relative intensity histogram to global energy
    pub fn energy(s : &[f32]) -> f32 {
        s.iter().fold(0.0, |s, p| s + p.powf(2.0 ) )
    }

    // Maps relative intensity histogram to global entropy
    pub fn entropy(s : &[f32]) -> f32 {
        (-1.)*s.iter().fold(0.0, |s, p| s + (p*p.log(2.0)) )
    }

    // Both energy and entropy can be calculated for GLMC's as well.

    // The second-order statistics of an image (variance, energy, entropy) do
    // not capture spatial information, so dissimilar patterns with the same
    // statistics are indistinguishable (Distante & Distante, v. 3). But the
    // gray-level co-ocurrence matrix (GLCM) does capture spatial patterns.
    // This is a 2D histogram indexed by marginal intensity values (size 256^2)
    // that gives the relative frequency of an intensity pair (L1, L2) existing
    // at a distance (dx, dy). This is called axialglcm because it runs along the
    // image axis.
    pub struct AxialGLCM {
        dx : usize,
        dy : usize
    }

    // impl Descriptor for LineGLCM { }

    impl AxialGLCM {

        // Creates a new glcm for pixels at a horizontal distance dx and
        // vertical distance dy.
        pub fn new(dx : usize, dy : usize) -> Self {
            Self { dx, dy }
        }

        pub fn calculate(&mut self, w : Window<u8>) {

        }
    }

    // Define co-ocurrence in terms of (Radius, theta)
    pub struct OrientedGLCM { }

    impl OrientedGLCM {
        // Creates a new glcm for pixels spaced by radius dr and angular
        // differences 2*PI / int_theta. int_theta values are either 4
        // (cardinal direction) or 8 (extended direction) or 16 (complete direction).
        // (i.e. augment with northnorthest, north, northnorthwest).
        pub fn new(dr : usize, int_theta : usize) -> Self {
            Self { }
        }
    }

}

/* LBPs are one of the most compact ways to represent spatial information. A brighter speckle
is represented by the lbp 00000000; a darker speckle by the lbp 11111111. Patterns of two
horizontal brighter pixels by 0011000000; of two vertical brighter pixels by 001000100. The
LBP compress this information in an byte image for a 3x3 neighborhood or an u8 image for a 5x5
neighborhood, thus resulting in a description that has the same size as the original image. */

// Local binary patterns are calculated from sliding windows over the image. When
// a neighboring pixel >= center pixel, attribute 1 to the output window. Attribute 0
// otherwise. The descriptor is the resulting bit pattern. The radius over which
// the operation is applied can be varied. The histograrm of the output of this operation
// is the datum of interest, and can be used to classify textures.
pub struct SimpleLBP {

}

pub struct UniformLBP {

    // A LBP is uniform when were are at most 2 0->1 transitions in the circular binary pattern.

}

pub struct RobustLBP {

    // Transform 010->000 and 101->111 (remove noise); then apply uniform LBP.
}

pub struct MedianRobustLBP {

}

pub mod geom {

}

pub mod prob {

}

pub enum PixelDistance {

    Axial { dx : usize, dy : usize },

    Radial { r : usize, theta : usize }

}

/* Todo perhaps speed up sumdiffhist by only storing the range (min, max) for the
sum and diff histgram. then store the array (perhaps an arena-allocated vec) and the upper and
lower limits for each histogram. */
pub struct SumDiffHist {

    // Holds bins in the interval [0,512]
    pub sum : [u32; 512],

    // Hold bins in the interval [-256, 256]
    pub diff : [u32; 512]

}

impl SumDiffHist {

    pub fn distr(&self) -> SumDiffDistr {
        SumDiffDistr {
            sum : bayes::calc::counts_to_probs_static(&self.sum),
            diff : bayes::calc::counts_to_probs_static(&self.diff),
        }
    }

}

pub struct SumDiffDistr {

    pub sum : [f32; 512],

    pub diff : [f32; 512]

}

pub struct AxialDiffHist {

    dx : SumDiffHist,

    dy : SumDiffHist

}

pub struct RadialDiffHist {

    r : SumDiffHist,

    theta : SumDiffHist

}

/* Statistical measures after Parker (2011) p. 188 */
impl SumDiffDistr {

    pub fn mean(&self) -> f32 {
        0.5 * self.sum.iter().enumerate().fold(0.0, |s, (i, v)| s + i as f32 * (*v) )
    }

    pub fn contrast(&self) -> f32 {
        0.5 * self.diff.iter().enumerate().fold(0.0, |s, (i, v)| s + i.pow(2) as f32 * (*v) )
    }

    pub fn homogeneity(&self) -> f32 {
        self.diff.iter().enumerate().fold(0.0, |s, (i, v)| 1. / (1. + i.pow(2) as f32) * v )
    }

    pub fn entropy(&self) -> f32 {
        stat::entropy(&self.sum) + stat::entropy(&self.diff)
    }

    pub fn energy(&self) -> f32 {
        stat::energy(&self.sum) + stat::energy(&self.diff)
    }

}

/* Sum-difference histograms, after Unser (1986); Parker (2011) p. 186. */
fn sum_diff_hist(win : &Window<u8>, sub_sz : (usize, usize), dx : usize, dy : usize) -> BTreeMap<(usize, usize), AxialDiffHist> {
    let mut out = BTreeMap::new();
    for w in win.windows(sub_sz) {
        let mut dx_hist = SumDiffHist  { sum : [0; 512], diff : [0; 512] };
        let mut dy_hist = SumDiffHist  { sum : [0; 512], diff : [0; 512] };
        for i in 0..(w.height()-dy) {
            for j in (0..w.width()-dx) {
                let dx_sum = w[(i, j)] as u16 + w[(i+dy, j)] as u16;
                let dy_sum = w[(i, j)] as u16 + w[(i, j+dx)] as u16;
                let dx_diff = w[(i, j)] as i16 - w[(i+dy, j)] as i16;
                let dy_diff = w[(i, j)] as i16 - w[(i, j+dx)] as i16;
                dx_hist.sum[dx_sum as usize] += 1;
                dx_hist.diff[(256+dx_diff) as usize] += 1;
                dy_hist.sum[dy_sum as usize] += 1;
                dy_hist.diff[(256+dy_diff) as usize] += 1;
            }
        }
        out.insert(w.offset(), AxialDiffHist { dx : dx_hist, dy : dy_hist });
    }
    out
}

#[cfg(feature="ipp")]
pub struct IppLBP {
    dist : usize
}

// Maybe create type LBPImage/LBPWindow?

/* Returns a grayscale of the number of ones in a u8 resulting from a binary pattern. */
pub fn lbp_sum(src : &dyn AsRef<Window<u8>>, dst : &mut dyn AsMut<WindowMut<u8>>) {
    assert!(src.as_ref().shape() == dst.as_mut().shape());
    for (mut d, s) in dst.as_mut().pixels_mut(1).zip(src.as_ref().pixels(1)) {
        *d = 31u8 * (s.count_ones() as u8);
    }
}

/* Returns the number of 0->1 or 1->0 transitions in a binary pattern. */
pub fn lbp_diff_transitions(src : &dyn AsRef<Window<u8>>, dst : &mut dyn AsMut<WindowMut<u8>>) {
    assert!(src.as_ref().shape() == dst.as_mut().shape());
    for (mut d, s) in dst.as_mut().pixels_mut(1).zip(src.as_ref().pixels(1)) {
        *d = 31u8 * (s.bitxor(s.rotate_right(1)).count_ones() as u8);
    }
}

/* Returns the number of 0->0 or 1->1 transitions in a binary pattern. */
pub fn lbp_same_transitions(src : &dyn AsRef<Window<u8>>, dst : &mut dyn AsMut<WindowMut<u8>>) {
    assert!(src.as_ref().shape() == dst.as_mut().shape());
    for (mut d, s) in dst.as_mut().pixels_mut(1).zip(src.as_ref().pixels(1)) {
        *d = 31u8 * (s.bitxor(s.rotate_right(1)).count_zeros() as u8);
    }
}

/* Returns the number of elements that are equal to pattern */
pub fn lbp_compare_pattern(src : &dyn AsRef<Window<u8>>, pattern : u8, dst : &mut dyn AsMut<WindowMut<u8>>) {
    assert!(src.as_ref().shape() == dst.as_mut().shape());
    for (mut d, s) in dst.as_mut().pixels_mut(1).zip(src.as_ref().pixels(1)) {
        *d = 31u8 * (s.bitxor(pattern).count_zeros() as u8);
    }
}

// horizontal correlation: (1) shift image by one pixel to left; (2) Calculate u8::bitxor to original image
// vert. correlation: (1) shift image by one pixel to left; (2) Calculate u8::bitxor to original image.

#[test]
fn to_binary() {
    for i in 0u8..255 {
        assert!(pattern::from_binary(pattern::to_binary(i)) == i);
    }
}

// cargo test -- patterns
#[test]
fn patterns() {
    // Image::new_from_pattern(90, true, pattern::BRIGHT_SPECKLE).show();
    // Image::new_from_pattern(90, false, pattern::DARK_SPECKLE).show();
    // Image::new_from_pattern(90, false, pattern::BRIGHT_LEFT_VBAR).show();
    // Image::new_from_pattern(90, false, pattern::BRIGHT_RIGHT_VBAR).show();
    // Image::new_from_pattern(90, false, pattern::BRIGHT_TOP_HBAR).show();
    // Image::new_from_pattern(90, false, pattern::BRIGHT_BOTTOM_HBAR).show();
    // Image::new_from_pattern(90, false, pattern::TOP_LEFT_CORNER).show();
    // Image::new_from_pattern(90, false, pattern::TOP_RIGHT_CORNER).show();
    // Image::new_from_pattern(90, false, pattern::BOTTOM_LEFT_CORNER).show();
    // Image::new_from_pattern(90, false, pattern::BOTTOM_RIGHT_CORNER).show();
    // Image::new_from_pattern(90, true, pattern::DIAGONAL_RIGHT).show();
    // Image::new_from_pattern(90, true, pattern::DIAGONAL_LEFT).show();
}

pub mod pattern {

    pub const fn from_binary(bits : [u8; 8]) -> u8 {
        let mut byte = 0u8;
        let mut ix : usize = 7;
        loop {
            let bit = if bits[ix] == 0 { 0 } else { 1 };
            byte += (bit * (2 as i32).pow(7 - ix as u32)) as u8;
            ix -= 1;
            if ix == 0 {
                let bit = if bits[0] == 0 { 0 } else { 1 };
                byte += (bit * (2 as i32).pow(7 - ix as u32)) as u8;
                break;
            }
        }
        byte
    }

    pub const fn to_binary(mut byte : u8) -> [u8; 8] {
        let mut bits = [0; 8];
        let mut ix = 7;
        loop {
            bits[ix] = byte % 2;
            byte /= 2;
            ix -= 1;
            if ix == 0 {
                bits[ix] = byte % 2;
                break;
            }
        }
        bits
    }

    /* Any pattern can be built by bitwise operations between the following primitives.
    A typical image LBP distance to those primitive types (or combinations of them) can
    be used as feature vectors */

    pub const BRIGHT_SPECKLE : u8 = 0b00000000;

    pub const DARK_SPECKLE : u8 = 0b11111111;

    pub const LEFT_VBAR : u8 = 0b10000011;

    pub const RIGHT_VBAR : u8 = 0b00111000;

    pub const TOP_HBAR : u8 = 0b11100000;

    pub const BOTTOM_HBAR : u8 = 0b00001110;

    pub const TOP_LEFT_CORNER : u8 = 0b10000000;

    pub const TOP_RIGHT_CORNER : u8 = 0b00100000;

    pub const BOTTOM_RIGHT_CORNER : u8 = 0b00001000;

    pub const BOTTOM_LEFT_CORNER : u8 = 0b00000010;

    pub const DIAGONAL_RIGHT : u8 = 0b10001000;

    pub const DIAGONAL_LEFT : u8 = 0b00100010;

}

#[cfg(feature="ipp")]
impl IppLBP {

    pub fn new(dist : usize, shape : (usize, usize)) -> Self {
        assert!(dist == 3 || dist == 5);
        Self { dist }
    }

    // The output is the bit pattern resulting from the LBP application. Apply some LBP statistic
    // to get a desired result.
    pub fn apply(&mut self, src : &dyn AsRef<Window<u8>>, dst : &mut dyn AsMut<WindowMut<u8>>) {
        let (src_step, src_roi) = crate::image::ipputils::step_and_size_for_window(src.as_ref());
        let (dst_step, dst_roi) = crate::image::ipputils::step_and_size_for_window_mut(&*dst.as_mut());
        // mode=1 starts from top-left and proceedss clockwise.
        let mode = 1;
        let border_val = 0u8;
        unsafe {
            let ans = if self.dist == 3 {
                crate::foreign::ipp::ippi::ippiLBPImageMode3x3_8u_C1R(
                    src.as_ref().as_ptr(),
                    src_step,
                    dst.as_mut().as_mut_ptr(),
                    dst_step,
                    dst_roi,
                    mode,
                    crate::foreign::ipp::ippi::_IppiBorderType_ippBorderConst,
                    &border_val as *const _
                )
            } else if self.dist == 5 {
                crate::foreign::ipp::ippi::ippiLBPImageMode5x5_8u_C1R(
                    src.as_ref().as_ptr(),
                    src_step,
                    dst.as_mut().as_mut_ptr(),
                    dst_step,
                    dst_roi,
                    mode,
                    crate::foreign::ipp::ippi::_IppiBorderType_ippBorderConst,
                    &border_val as *const _
                )
            } else {
                panic!("Invalid dist");
            };
            assert!(ans == 0);
        }
    }

}