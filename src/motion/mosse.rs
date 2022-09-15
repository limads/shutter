// This was translated from the C++ version at https://github.com/opencv/opencv_contrib
// under tracking::mosseTracker originally licensed under Apache 2.0.

use crate::image::*;
// use nalgebra::*;
use ripple::fft::*;
use std::mem;
use crate::local::*;
use std::ops::{AddAssign, MulAssign};
// use crate::convert::*;
// use crate::global::*;
use nalgebra::{Complex, Vector2, Matrix2x3};

const EPS : f64 = 0.00001;

const RATE : f64 = 0.2;

// Peak signal/noise ratio. PSNR = 10*log10(R^2/MSE) where MSE is the mean square
// error between two arrays and R^2 is the maximum integer value depth (https://docs.rs/opencv/0.66.0/opencv/core/fn.psnr.html)

const PSR_THRESHOLD : f64 = 5.7;

pub struct MosseTracker {

    center : Vector2<f32>,

    center_coord : (usize, usize),

    fft : Fourier2D<f32>,

    win_sz : (usize, usize),

    han_win : ImageBuf<f32>,

    // Holds converted subwindow of current image
    image_sub : ImageBuf<f32>,

    image_sub_new : ImageBuf<f32>,

    // Holds FFT of subwindow of current image
    image_sub_freq : ImageBuf<Complex<f32>>,

    image_sub_new_freq : ImageBuf<Complex<f32>>,

    // Holds product of current and past FFT in frequency space.
    response_freq : ImageBuf<Complex<f32>>,

    // Holds product of current and past FFT in the original space.
    response_orig : ImageBuf<f32>,

    // Holds FFT of previous frame
    // h : Image<f32>,

    a : ImageBuf<Complex<f32>>,

    b : ImageBuf<Complex<f32>>,

    a_new : ImageBuf<Complex<f32>>,

    b_new : ImageBuf<Complex<f32>>,

    // Called H in original impl (holds fft of original frame)
    h_freq : ImageBuf<Complex<f32>>,

    // goal
    g_freq : ImageBuf<Complex<f32>>,

    // state

    re1 : ImageBuf<f32>,

    re2 : ImageBuf<f32>,

    im1 : ImageBuf<f32>,

    im2 : ImageBuf<f32>,

    sq_re2 : ImageBuf<f32>,

    sq_im2 : ImageBuf<f32>,

    sq_im1 : ImageBuf<f32>,

    denom : ImageBuf<f32>,

    prod_re1re2 : ImageBuf<f32>,

    prod_im1im2 : ImageBuf<f32>,

    numtr : ImageBuf<f32>,

    re_dst : ImageBuf<f32>,

    im_dst : ImageBuf<f32>,


}

fn complex_mul(a : &[Complex<f32>], b : &[Complex<f32>], dst : &mut [Complex<f32>]) {
    dst.iter_mut().zip(a.iter().zip(b.iter())).for_each(|(mut d, (a, b))| *d = (*a) * (*b) );
}

fn complex_mul_conj(a : &[Complex<f32>], b : &[Complex<f32>], dst : &mut [Complex<f32>]) {
    dst.iter_mut().zip(a.iter().zip(b.iter())).for_each(|(mut d, (a, b))| *d = (*a) * b.conj() );
}

fn pre_process(win : &mut WindowMut<f32>, han : &Window<f32>) {

    // log(window + 1.0f, window);
    for mut px in win.pixels_mut(1) {
        *px = (*px + 1.0).ln()
    }

    // normalize
    let (mean, mut stddev) = crate::global::mean_stddev(win.as_ref());
    stddev += EPS as f32;
    win.scalar_add(-mean);
    win.scalar_mul(1. / stddev);

    // Gaussain weighting
    win.mul_assign(han.clone());
}

impl MosseTracker {

    // store the result in self.h
    fn div_ffts(&mut self, src1 : &[Complex<f32>], src2 : &[Complex<f32>]) {

        let (mut re1, mut re2) = (mem::take(&mut self.re1), mem::take(&mut self.re2));
        let (mut im1, mut im2) = (mem::take(&mut self.im1), mem::take(&mut self.im2));
        // split into re and im per src
        cartesian_split_mut(src1, re1.as_mut_slice(), im1.as_mut_slice());
        cartesian_split_mut(src2, re2.as_mut_slice(), im2.as_mut_slice());

        // (Re2*Re2 + Im2*Im2) = denom
        let (mut sq_re2, mut sq_im2, mut denom) = (mem::take(&mut self.sq_re2), mem::take(&mut self.sq_im2), mem::take(&mut self.denom));
        let mut denom = mem::take(&mut self.denom);
        re2.mul_to(&mut sq_re2);
        im2.mul_to(&im2, &mut sq_im2);
        sq_re2.add_to(&sq_im2, &mut denom);

        // Note there is an error in the comment of the original impl
        // (it says re1*re2 + im1*im1) at the numerator
        // (Re1*Re2 + Im1*Im2)/(Re2*Re2 + Im2*Im2) = Re
        // TODO add eps to denom?
        let (mut prod_re1re2, mut prod_im1im2) = (mem::take(&mut self.prod_re1re2), mem::take(&mut self.prod_im1im2));
        let mut numtr = mem::take(&mut self.numtr);
        re1.mul_to(&re2, &mut prod_re1re2);
        im1.mul_to(&im2, &mut prod_im1im2);
        prod_re1re2.add_to(&prod_im1im2, &mut numtr);
        numtr.div_to(&denom, &mut self.re_dst);

        // (Im1*Re2 - Re1*Im2)/(Re2*Re2 + Im2*Im2) = Im
        let mut prod_im1re2 = prod_re1re2;
        let mut prod_re1im2 = prod_im1im2;
        im1.mul_to(&re2, &mut prod_im1re2);
        re1.mul_to(&im2, &mut prod_re1im2);
        prod_im1re2.add_to(&prod_re1im2, &mut numtr);
        numtr.scalar_mul(-1.);
        numtr.div_to(&denom, &mut self.im_dst);

        // Merge Re and Im back into a complex matrix (stored in the h field)
        cartesian_interleave_mut(self.re_dst.as_slice(), self.im_dst.as_slice(), self.h_freq.as_mut_slice());

        self.re1 = re1;
        self.re2 = re2;
        self.im1 = im1;
        self.im2 = im2;
        self.sq_re2 = sq_re2;
        self.sq_im2 = sq_im2;
        self.denom = denom;
        self.numtr = numtr;
        self.prod_re1re2 = prod_im1re2;
        self.prod_im1im2 = prod_re1im2;
    }

    // Note image is required to be memory-contiguous.
    fn correlate(&mut self, img_sub : &ImageBuf<f32>) -> (Vector2<f32>, f64) {

        let (mut image_sub_freq, mut response_freq) = (mem::take(&mut self.image_sub_freq), mem::take(&mut self.response_freq));
        let mut response_orig = mem::take(&mut self.response_orig);

        // filter in dft space
        self.fft.forward_mut(&img_sub.as_slice(), &mut image_sub_freq.as_mut_slice());

        complex_mul_conj(image_sub_freq.as_slice(), self.h_freq.as_slice(), response_freq.as_mut_slice());

        // TODO the result should be scaled back to the original units (original impl uses scale flag)
        self.fft.backward_mut(response_freq.as_slice(), &mut response_orig.as_mut_slice());

        // update center position
        let (_, max_data) = crate::local::min_max_idx(response_orig.as_ref(), false, true);
        let (max_row, max_col, max_val) = max_data.unwrap();

        let mut delta_xy = Vector2::zeros();
        delta_xy[0] = max_col as f32 - (self.win_sz.1 / 2) as f32;
        delta_xy[1] = max_row as f32 - (self.win_sz.0 / 2) as f32;

        // normalize response
        let (mean, stddev) = response_orig.mean_stddev();

        self.response_orig = response_orig;
        self.image_sub_freq = image_sub_freq;
        self.response_freq = response_freq;

        let psr = (max_val - mean) / (stddev+(EPS as f32));
        (delta_xy, psr as f64)
    }

    pub fn init(img : Window<u8>, pos : (usize, usize), size : (usize, usize)) -> Self {

        let mut fft = Fourier2D::<f32>::new(size.0, size.1).unwrap();

        // Get the center position
        let center = Vector2::new((pos.1 + size.1 / 2) as f32, (pos.0 + size.0 / 2) as f32);
        let center_coord = (center[1] as usize, center[0] as usize);

        let mut image_sub = Image::<f32>::new_constant(size.0, size.1, 0.);
        image_sub.convert_from(img.sub_window(pos, size).unwrap(), Conversion::Preserve);
        let han_win = hanning_window(size.0, size.1);

        // goal
        // This is originally a gaussian blur with standard deviation parameter 2.0.
        let mut g = Image::<f32>::new_constant(size.0, size.1, 0.);
        g[(size.0/2, size.1/2)] = 1.0;
        g.clone().convolve_mut(&blur::GAUSS, &mut g);
        let max_val = g.max();
        g.full_window_mut().scalar_mul(1. / max_val);
        let mut g_freq = unsafe { Image::<Complex<f32>>::new_empty(size.0, size.1) };
        fft.forward_mut(g.as_slice(), g_freq.as_mut_slice());

        // initial A,B and H
        let mut a = unsafe { Image::<Complex<f32>>::new_empty_like(&g.full_window()) };
        let mut b = unsafe { Image::<Complex<f32>>::new_empty_like(&g.full_window()) };
        let h_freq = unsafe { Image::<Complex<f32>>::new_empty_like(&g.full_window()) };
        let mut warp_dst = unsafe { Image::new_empty(size.0, size.1) };
        let mut a_i = unsafe { Image::<Complex<f32>>::new_empty(size.0, size.1) };
        let mut b_i = unsafe { Image::<Complex<f32>>::new_empty(size.0, size.1) };

        for i in 0..8 {
            let mut warped = image_sub.clone();
            rand_warp(image_sub.as_ref(), warped.as_mut());
            pre_process(warped.as_mut(), han_win.as_ref());
            fft.forward_mut(warped.as_slice(), warp_dst.as_mut_slice());
            complex_mul_conj(g_freq.as_slice(), warp_dst.as_slice(), a_i.as_mut_slice());
            complex_mul_conj(warp_dst.as_slice(), warp_dst.as_slice(), b_i.as_mut_slice());
            a.full_window_mut().add_assign(a_i.full_window());
            b.full_window_mut().add_assign(b_i.full_window());
        }

        let mut this = Self {
            center,
            center_coord,
            fft,
            han_win,
            win_sz : size,
            a_new : a.clone(),
            b_new : b.clone(),
            image_sub_new : image_sub.clone(),
            image_sub : image_sub.clone(),
            image_sub_freq : g_freq.clone(),
            image_sub_new_freq : g_freq.clone(),
            
            re1 : image_sub.clone(),

            re2 : image_sub.clone(),

            im1 : image_sub.clone(),

            im2 : image_sub.clone(),

            sq_re2 : image_sub.clone(),

            sq_im2 : image_sub.clone(),

            sq_im1 : image_sub.clone(),

            denom : image_sub.clone(),

            prod_re1re2 : image_sub.clone(),

            prod_im1im2 : image_sub.clone(),

            numtr : image_sub.clone(),

            re_dst : image_sub.clone(),

            im_dst : image_sub.clone(),

            response_orig : image_sub.clone(),

            response_freq : g_freq.clone(),
            g_freq,
            h_freq,
            a,
            b,
        };
        let (a, b) = (mem::take(&mut this.a), mem::take(&mut this.b));
        this.div_ffts(a.as_slice(), b.as_slice());
        this.a = a;
        this.b = b;
        this
    }

    pub fn update(&mut self, image : &Window<u8>) -> Option<(usize, usize, usize, usize)> {

        let (mut image_sub, mut image_sub_new) = (mem::take(&mut self.image_sub), mem::take(&mut self.image_sub_new));

        image_sub.convert_from(image.sub_window(self.center_coord, self.win_sz).unwrap(), Conversion::Preserve);
        pre_process(image_sub.as_mut(), self.han_win.as_ref());

        let (delta_xy, psr) = self.correlate(&image_sub);
        if psr < PSR_THRESHOLD {
            return None;
        }

        //update location
        // self.center[0] += self.delta_xy[0];
        // self.center[1] += self.delta_xy[1];
        self.center += delta_xy;
        if self.center[0] < 0.0 || self.center[1] < 0.0 {
            return None;
        }

        self.center_coord.0 = self.center[1] as usize;
        self.center_coord.1 = self.center[0] as usize;

        image_sub_new.convert_from(image.sub_window(self.center_coord, self.win_sz).unwrap(), Conversion::Preserve);
        pre_process(image_sub_new.as_mut(), self.han_win.as_ref());

        // new state for A and B

        let mut image_sub_new_freq = mem::take(&mut self.image_sub_new_freq);
        let (mut a_new, mut b_new) = (mem::take(&mut self.a_new), mem::take(&mut self.b_new));
        let (mut a, mut b) = (mem::take(&mut self.a), mem::take(&mut self.b));

        self.fft.forward_mut(image_sub_new.as_slice(), image_sub_new_freq.as_mut_slice());
        complex_mul_conj(self.g_freq.as_slice(), image_sub_new_freq.as_slice(), a_new.as_mut_slice());
        complex_mul_conj(image_sub_new_freq.as_slice(), image_sub_new_freq.as_slice(), b_new.as_mut_slice());

        // update A ,B, and H
        let rate = Complex { re : RATE as f32, im : 0.0 as f32 };
        let rate_compl = Complex { re : (1. - RATE) as f32, im : 0.0 as f32 };
        a_new.full_window_mut().scalar_mul(rate);
        b_new.full_window_mut().scalar_mul(rate);
        a.full_window_mut().scalar_mul(rate_compl);
        b.full_window_mut().scalar_mul(rate_compl);
        a.full_window_mut().add_assign(a_new.full_window());
        b.full_window_mut().add_assign(b_new.full_window());
        self.div_ffts(a.as_slice(), b.as_slice());

        // return tracked rect
        let pos = (self.center_coord.0 - self.win_sz.0/2, self.center_coord.1 - self.win_sz.1/2);
        let sz = (self.center_coord.0+self.win_sz.0/2 - pos.0, self.center_coord.1+self.win_sz.1/2 - pos.1);

        self.image_sub = image_sub;
        self.image_sub_new = image_sub_new;
        self.image_sub_new_freq = image_sub_new_freq;
        self.a = a;
        self.b = b;
        self.a_new = a_new;
        self.b_new = b_new;

        Some((pos.0, pos.1, sz.0, sz.1))
    }

}

fn random_bounded() -> f64 {
    let mut u : f64 = rand::random();
    let bound=0.1;
    -bound + 2.*bound*u
}

fn hanning_window(height : usize, width : usize) -> ImageBuf<f32> {
    let mut dst = unsafe { Image::new_empty(height, width) };
    let coeff0 = 2.0 * std::f32::consts::PI / (width - 1) as f32;
    let coeff1 = 2.0 * std::f32::consts::PI / (height - 1) as f32;
    let mut wc = Vec::with_capacity(width);
    for j in 0..width {
        wc.push(0.5 * (1.0 - (coeff0 * j as f32).cos() ));
    }

    for i in 0..height {
        let wr = 0.5 * (1.0 - (coeff1 * i as f32).cos() );
        for j in 0..width {
            dst[(i, j)] = (wr * wc[j]);
        }
    }

    dst.full_window_mut().pixels_mut(1).for_each(|px| *px = px.sqrt() );
    dst
}

fn rand_warp(a : &Window<f32>, warped : &mut WindowMut<f32>) {

    let ang = random_bounded();
    let (c, s) = (ang.cos(), ang.sin());

    // affine warp matrix
    let mut w = Matrix2x3::zeros();
    w[(0, 0)] = c + random_bounded();
    w[(0, 1)] = -s + random_bounded();
    w[(1, 0)] = random_bounded();
    w[(1, 1)] = c + random_bounded();

    // random translation
    let mut center_warp = Vector2::zeros();
    center_warp[0] = (a.width()/2) as f64;
    center_warp[1] = (a.height()/2) as f64;
    let fst_cols = w.columns(0, 2).clone_owned();
    w.column_mut(2).copy_from(&(center_warp.clone() - (fst_cols * &center_warp)));

    // It is important that the reflection mode for the border is used here.
    crate::warp::affine_to(a, &w, warped);
}