use std::any::Any;
use crate::image::*;
use num_traits::Zero;
use nalgebra::*;
use std::mem;
use std::ops::{Add, Shl};

#[cfg(feature="ipp")]
fn ipp_integral(win : &Window<u8>, dst : &mut WindowMut<i32>) {
    assert!(win.width() == dst.width() - 1);
    assert!(win.height() == dst.height() - 1);
    unsafe {
        let (src_step, src_sz) = crate::image::ipputils::step_and_size_for_window(win);
        let (dst_step, dst_sz) = crate::image::ipputils::step_and_size_for_window_mut(&dst);

        // Usually a negative value, which allows for larger images to be processed without
        // an integer overflow.
        let offset_val : i32 = 0;
        let ans = crate::foreign::ipp::ippcv::ippiIntegral_8u32s_C1R(
            win.as_ptr(),
            src_step,
            mem::transmute(dst.as_mut_ptr()),
            dst_step,
            std::mem::transmute(src_sz),
            offset_val
        );
        assert!(ans == 0, "Error processing integral image: {}", ans);
        return;
    }
}

pub struct Integral<T>(Image<T>)
where
    T : Scalar + Clone + Copy + Any;

impl<T> Integral<T>
where
    T : Scalar + Clone + Copy + Any + Zero + From<u8> + Default + std::ops::AddAssign
{

    /* Integral sums the rectangle (0, 0) (i, j) above the pixel (i, j).*/
    pub fn update(&mut self, win : &Window<'_, u8>) {
        #[cfg(feature="ipp")]
        unsafe {
            if (&T::default() as &dyn Any).is::<i32>() {
                return ipp_integral(win, mem::transmute(&mut self.0.full_window_mut()));
            }
        }

        // TODO  'attempt to add with overflow'
        let mut dst = &mut self.0;
        dst[(0, 0)] = T::from(win[(0 as usize, 0 as usize)]);
        unsafe {
            for ix in 1..dst.len() {
                let prev = *dst.unchecked_linear_index(ix-1);
                *dst.unchecked_linear_index_mut(ix) += prev;
            }
        }
    }

    pub fn calculate(win : &Window<'_, u8>) -> Self {
        // TODO Make sure IppiIntegral overwrites all pixels.
        // The first pixel of the integral image equals the first pixel of the original
        // image. All other pixels are the sum of the previous pixels up to the current pixel.
        let mut dst = unsafe { Self(Image::<T>::new_empty(win.height() + 1, win.width() + 1)) };
        dst.update(win);
        dst
    }

}

impl<T> AsRef<Image<T>> for Integral<T> 
where
    T : Scalar + Clone + Copy + Any + Zero + From<u8>
{

    fn as_ref(&self) -> &Image<T> {
        &self.0
    }

}

impl<'a, T> AsRef<Window<'a, T>> for Integral<T>
where
    T : Scalar + Clone + Copy + Any + Zero + From<u8>
{

    fn as_ref(&self) -> &Window<'a, T> {
        self.0.as_ref()
    }

}

impl<'a, T> AsMut<WindowMut<'a, T>> for Integral<T>
where
    T : Scalar + Clone + Copy + Any + Zero + From<u8>
{

    fn as_mut(&mut self) -> &mut WindowMut<'a , T> {
        self.0.as_mut()
    }

}

pub struct Accumulated(Image<i32>);

impl Accumulated {

    pub fn calculate(win : &Window<u8>) -> Self {
        let mut dst = unsafe { Self(Image::<i32>::new_empty(win.height(), win.width())) };
        dst.update(win);
        dst
    }

    // This sets pixel (i,j) to the sum of all pixels before it. It is
    // calculated using IPP by re-using the IppIntegral, but interpreting
    // the image as a 1D buffer (which is why it requires a reference to an
    // owned buffer - this trick cannot be applied to image views).
    pub fn update(&mut self, w : &Window<u8>) {
        baseline_accumulate(self.0.as_mut(), w);
    }

    pub fn update_vectorized(&mut self, w : &Window<i32>) {
        if w.width() % 8 == 0 && self.0.width() == w.width() {
            for (mut d, r) in self.0.full_window_mut().rows_mut().zip(w.rows()) {
                unsafe { vectorized_cumulative_sum(r, d) };
            }
        } else {
            panic!()
        }
    }

}

fn baseline_accumulate(dst : &mut WindowMut<i32>, src : &Window<u8>) {
    assert!(dst.shape() == src.shape());
    let mut s = 0;
    for (mut d, px) in dst.pixels_mut(1).zip(src.pixels(1)) {
        s += *px as i32;
        *d = s;
    }
}

impl<'a> AsRef<Window<'a, i32>> for Accumulated
{

    fn as_ref(&self) -> &Window<'a, i32> {
        self.0.as_ref()
    }

}

impl<'a> AsMut<WindowMut<'a, i32>> for Accumulated
{

    fn as_mut(&mut self) -> &mut WindowMut<'a , i32> {
        self.0.as_mut()
    }

}

// cargo test -- foo --nocapture
#[test]
fn foo() {

    use std::arch::x86_64::*;

    unsafe {
        let mut s : __m128i = _mm_setzero_si128();
        println!("{:?}", s);
    }
}

// cargo test -- prefix_sum --nocapture
#[test]
fn prefix_sum() {

    let a : [i32; 16] = [1,2,3,4,5,6,7,8,9,10,11,12,13,15,15,16];
    let mut s : [i32; 16] = [0; 16];

    // Never transmute the array pointer like *mut i32 to __mm128 pointer, transmute the dereferenced array to __mm128
    // let mut ptr = s.as_mut_ptr() as *mut [i32; 4];
    // unsafe { println!("{:?}", mem::transmute::<_, [i32; 4]>(int::local_prefix4(mem::transmute(*ptr)))); };
    // Using types from wide, however, pointer cast will work.

    let mut s2 = s.clone();
    unsafe {
        vectorized_cumulative_sum(&a[..], &mut s[..]);
        baseline_cumulative_sum(&a[..], &mut s2[..]);
        println!("{:?}", s);
        println!("{:?}", s2);
        assert!(s == s2);
    }
}

// rustc -C target-feature=+sse2
pub mod int {

    use std::arch::x86_64::*;
    use std::mem;

    pub unsafe fn local_prefix4(mut x : __m128i) -> __m128i {
        // x = 1, 2, 3, 4
        x = _mm_add_epi32(x, _mm_slli_si128(x, 4));
        // x = 1, 2, 3, 4
        //   + 0, 1, 2, 3
        //   = 1, 3, 5, 7
        x = _mm_add_epi32(x, _mm_slli_si128(x, 8));
        // x = 1, 3, 5, 7
        //   + 0, 0, 1, 3
        //   = 1, 3, 6, 10
        return x;
    }

    // This does the first step (local sums in 8-lane i32) */
    // #[target_feature(enable="sse2")]
    // #[cfg(target_feature = "avx")]
    pub unsafe fn local_prefix8(p : *mut i32) {
        println!("{:p}", p);
        println!("{:p}", p.offset(7));
        println!("a");
        if !is_x86_feature_detected!( "sse2" ) {
            panic!("SSE2 not detected");
        }

        let mut x : __m256i = _mm256_load_si256(p as *const _);
        println!("done");
        x = _mm256_add_epi32(x, _mm256_slli_si256(x, 4));
        x = _mm256_add_epi32(x, _mm256_slli_si256(x, 8));
        _mm256_store_si256(p as *mut _, x);
        // }
        // unimplemented!()
    }

    // This does the second step.
    unsafe fn accumulate(p : *mut i32, s : __m128i) -> __m128i {
        let d : __m128i = mem::transmute(_mm_broadcast_ss(mem::transmute::<_, &f32>(p.offset(3))));
        let mut x : __m128i = _mm_load_si128(p as *const _);
        x = _mm_add_epi32(s, x);
        _mm_store_si128(p as *mut _, x);
        return _mm_add_epi32(s, d);
    }

    // This calls both steps
    pub unsafe fn prefix(a : &mut [i32]) {
        for i in (0..a.len()).step_by(8) {
            local_prefix8(&mut a[i] as *mut i32);
        }

        let mut s : __m128i = _mm_setzero_si128();
        for i in (4..a.len()).step_by(4) {
            s = accumulate(&mut a[i] as *mut i32, s);
        }
    }
}
/*/*
_mm_shuffle_ps can be used for integer vectors as *pA = _mm_shuffle_epi32(_mm_unpacklo_epi32(*pA, _mm_shuffle_epi32(*pB, 0xe)), 0xd8);
OR
*pA = _mm_blend_epi16(*pA, *pB, 0xf0);
https://stackoverflow.com/questions/26983569/implications-of-using-mm-shuffle-ps-on-integer-vector
*/

use wide::*;

// _mm_castps_si128 cast [f32; 4] into [i32; 4]
// _mm_castsi128_ps cast [i32; 4] into [f32; 4]
// _mm_slli_si128 Shifts first [i32; 4] argment to left by N bytes while shifting in zeros.
fn scan_SSE(mut x : wide::i32x4) -> wide::i32x4 {
    x = x + x.rotate_left(i32x4::splat(4));
    x = x + x.rotate_left(i32x4::splat(8));
    x
}

// local_prefix_wide implements that using simd.
fn cumulative_sum(a : &mut [i32]) {
    let n = a.len();
    for l in 0..(n.log(10)) {
        let m = 1 << l;
        for i in m..n {
            a[i] += a[i - m];
        }
    }
}

unsafe fn local_prefix_wide4(p : &mut [i32; 4]) {
    let mut x = wide::i32x4::from(*p);
    // rotate_lanes_left(4)
    println!("x = {:?}", x);
    let xs4 = x.rotate_left(i32x4::splat(4));
    println!("xs4 = {:?}", xs4);
    x += xs4;
    let xs8 = x.rotate_left(i32x4::splat(8));
    println!("xs8 = {:?}", xs8);
    x += xs8;
    *p = x.into();
}*/

fn baseline_cumulative_sum<T>(a : &[T], dst : &mut [T])
where
    T : Copy + Add<Output=T>
{
    assert!(a.len() == dst.len());
    dst[0] = a[0];
    for i in 1..a.len() {
        dst[i] = dst[i-1] + a[i];
    }
}

/* Based on https://en.algorithmica.org/hpc/algorithms/prefix/
(1) Do local prefix sums within a vector in parallel (each sub-vector has the size of the SIMD lane)
(2) Use SIMD to add the previous vector to the current vector */
unsafe fn vectorized_cumulative_sum(a : &[i32], dst : &mut [i32]) {
    assert!(a.len() == dst.len());
    assert!(a.len() % 8 == 0);
    for i in (0..a.len()).step_by(8) {
        local_prefix_wide8(a[i..].as_ptr() as *const _, dst[i..].as_mut_ptr() as *mut _);
    }

    let mut s = wide::i32x4::ZERO;
    for i in (0..a.len()).step_by(4) {
        s = accumulate_wide(mem::transmute(&mut dst[i]), s);
    }
}

// This does the same thing as scan_SSE before.
unsafe fn local_prefix_wide8(p : *const [i32; 8], s : *mut [i32; 8]) {
    use std::arch::x86_64::*;
    let mut x = wide::i32x8::from(*p);
    x += mem::transmute::<_, wide::i32x8>(_mm256_slli_si256(mem::transmute(x), 4));
    x += mem::transmute::<_, wide::i32x8>(_mm256_slli_si256(mem::transmute(x), 8));
    *s = x.into();
}

unsafe fn accumulate_wide(dst : &mut [i32; 4], s : wide::i32x4) -> wide::i32x4 {
    let d : wide::i32x4 = mem::transmute(_mm_broadcast_ss(mem::transmute::<_, &f32>(&dst[3])));
    let mut x : wide::i32x4 = (*dst).into();
    x += s;
    *dst = x.into();
    s + d
}

/*unsafe fn vectorized_u8_cumulative_sum(a : &[u8], dst : &mut [u8]) {
    assert!(a.len() == dst.len());
    assert!(a.len() % 16 == 0);
    for i in (0..a.len()).step_by(16) {
        local_u8_prefix_wide8(a[i..].as_ptr() as *const _, dst[i..].as_mut_ptr() as *mut _);
    }
    let mut s = wide::u8x16::ZERO;
    for i in (0..a.len()).step_by(16) {
        s = accumulate_u8_wide(mem::transmute(&mut dst[i]), s);
    }
}
// This does the same thing as scan_SSE before.
unsafe fn local_u8_prefix_wide8(p : *const [u8; 16], s : *mut [u8; 16]) {
    use std::arch::x86_64::*;
    let mut x = wide::u8x16::from(*p);
    x += mem::transmute::<_, wide::u8x16>(_mm_slli_si128(mem::transmute(x), 4));
    x += mem::transmute::<_, wide::u8x16>(_mm_slli_si128(mem::transmute(x), 8));
    *s = x.into();
}
unsafe fn accumulate_u8_wide(dst : &mut [u8; 16], s : wide::u8x16) -> wide::u8x16 {
    let d : wide::u8x16 = mem::transmute(_mm_broadcast_ss(mem::transmute::<_, &f32>(&dst[3])));
    let mut x : wide::u8x16 = (*dst).into();
    x += s;
    *dst = x.into();
    s + d
}*/

use std::arch::x86_64::*;

// Mimics the _MM_SHUFFLE(z, y, x, w) macro, which is currently unstable
// https://shybovycha.github.io/2017/02/21/speeding-up-algorithms-with-sse.html
// Given those definitions, the call m3 = _mm_shuffle_ps(m1, m2, _MM_SHUFFLE(z, y, x, w))
// is equal to the formula m3 = (m2(z) << 6) | (m2(y) << 4) | (m1(x) << 2) | m1(w).
const fn mm_shuffle_mask(z : i32, y : i32, x : i32, w : i32) -> i32 {
    (z << 6) | (y << 4) | (x << 2) | w
}

unsafe fn packed_slice<T, const N : usize>(a : &[T]) -> &[[T;N]] {
    let n = a.len();
    assert!(n >= N);
    assert!(n % N == 0);
    std::mem::transmute::<_, &[[T; N]]>(std::slice::from_raw_parts(&a[0] as *const T, n / N))
}

unsafe fn packed_slice_mut<T, const N : usize>(a : &mut [T]) -> &mut [[T; N]] {
    let n = a.len();
    assert!(n >= N);
    assert!(n % N == 0);
    std::mem::transmute::<_, &mut [[T; N]]>(std::slice::from_raw_parts_mut(&mut a[0] as *mut T, n / N))
}

/*// Based on https://stackoverflow.com/questions/19494114/parallel-prefix-cumulative-sum-with-sse
mod sse {

    use core::arch::x86_64::*;

    unsafe fn scan_SSE(mut x : __m128) -> __m128 {
        x = _mm_add_ps(x, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(x), 4)));
        x = _mm_add_ps(x, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(x), 8)));
        x
    }

    // Store the prefix sum of a into the s vector (assumed of same length and length divisible by 4).
    pub unsafe fn prefix_sum_SSE(a : &[f32], s : &mut [f32]) {
        assert!(a.len() == s.len());
        assert!(a.len() % 4 == 0);
        let mut offset : __m128 = _mm_setzero_ps();
        for i in (0..a.len()).step_by(4) {

            // TODO segfault here
            let x : __m128 = _mm_load_ps(&a[i]);

            let out : __m128 = scan_SSE(x);

            let out = _mm_add_ps(out, offset);
            _mm_store_ps(&mut s[i] as *mut _, out);
            println!("here2");
            offset = _mm_shuffle_ps(out, out, super::mm_shuffle_mask(3, 3, 3, 3));
        }
    }

}

mod avx {

    /*use core::arch::x86_64::*;

     fn scan_AVX(x : __m256) -> __m256 {
        let (t0, t1) : (__m256, __m256) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        //shift1_AVX + add
        let t0 = _mm256_permute_ps(x, _MM_SHUFFLE(2, 1, 0, 3));
        let t1 = _mm256_permute2f128_ps(t0, t0, 41);
        let x = _mm256_add_ps(x, _mm256_blend_ps(t0, t1, 0x11));
        //shift2_AVX + add
        let t0 = _mm256_permute_ps(x, _MM_SHUFFLE(1, 0, 3, 2));
        let t1 = _mm256_permute2f128_ps(t0, t0, 41);
        let x = _mm256_add_ps(x, _mm256_blend_ps(t0, t1, 0x33));
        //shift3_AVX + add
        let x = _mm256_add_ps(x,_mm256_permute2f128_ps(x, x, 41));
        x
    }

    void prefix_sum_AVX(float *a, float *s, const int n) {
        let offset : __m256 = _mm256_setzero_ps();
        for (int i = 0; i < n; i += 8) {
            __m256 x = _mm256_loadu_ps(&a[i]);
            __m256 out = scan_AVX(x);
            out = _mm256_add_ps(out, offset);
            _mm256_storeu_ps(&s[i], out);
            //broadcast last element
            __m256 t0 = _mm256_permute2f128_ps(out, out, 0x11);
            offset = _mm256_permute_ps(t0, 0xff);
        }
    }*/

}*/

