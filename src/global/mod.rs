use std::cmp::{PartialOrd, Ord, Ordering};
use crate::foreign::ipp;
use crate::image::*;
use serde::{Serialize, de::DeserializeOwned};
use std::any::Any;
use std::fmt::Debug;
use std::ops::*;
use num_traits::Zero;
use std::ops::Add;
use nalgebra::Scalar;
use num_traits::AsPrimitive;
use std::mem;

/// Calculates the global maximum and minimum of the image.
/// Then sets all pixels close to the minimum to zero.
pub fn supress_close_to_min_mut(w : &mut WindowMut<u8>) {
    let (min, max) = min_max(&w);
    for px in w.pixels_mut(1) {
        if *px - min < max - *px {
            *px = 0;
        }
    }
}

pub fn baseline_sum<U>(s : &Window<U>) -> U 
where
    U : Pixel + Add<Output=U>
{
    s.pixels(1).fold(U::zero(), |s, p| s + *p )
}

#[cfg(feature="ipp")]
pub fn min_max<N>(w : &dyn AsRef<Window<N>>) -> (N, N)
where
    N : Pixel,
    u8 : AsPrimitive<N>,
    i16 : AsPrimitive<N>,
    u16 : AsPrimitive<N>,
    f32 : AsPrimitive<N>
{
    let w = w.as_ref();
    let (step, roi) = crate::image::ipputils::step_and_size_for_window(w);
    unsafe {
        match w.depth() {
            Depth::U8 => {
                let (mut min, mut max) = (0u8, 0u8);
                let ans = crate::foreign::ipp::ippi::ippiMinMax_8u_C1R(
                    mem::transmute(w.as_ptr()),
                    step,
                    roi,
                    &mut min as *mut _,
                    &mut max as *mut _
                );
                assert!(ans == 0);
                return (min.as_(), max.as_());
            },
            Depth::U16 => {
                let (mut min, mut max) = (0u16, 0u16);
                let ans = crate::foreign::ipp::ippi::ippiMinMax_16u_C1R(
                    mem::transmute(w.as_ptr()),
                    step,
                    roi,
                    &mut min as *mut _,
                    &mut max as *mut _
                );
                assert!(ans == 0);
                return (min.as_(), max.as_());
            },
            Depth::I16 => {
                let (mut min, mut max) = (0i16, 0i16);
                let ans = crate::foreign::ipp::ippi::ippiMinMax_16s_C1R(
                    mem::transmute(w.as_ptr()),
                    step,
                    roi,
                    &mut min as *mut _,
                    &mut max as *mut _
                );
                assert!(ans == 0);
                return (min.as_(), max.as_());
            },
            Depth::F32 => {
                let (mut min, mut max) = (0.0f32, 0.0f32);
                let ans = crate::foreign::ipp::ippi::ippiMinMax_32f_C1R(
                    mem::transmute(w.as_ptr()),
                    step,
                    roi,
                    &mut min as *mut _,
                    &mut max as *mut _
                );
                assert!(ans == 0);
                return (min.as_(), max.as_());
            },
            _ => panic!("Invalid depth")
        }
    }
}

#[cfg(feature="ipp")]
pub fn mean_stddev(w : &Window<f32>) -> (f32, f32) {
    let (stride, roi) = crate::image::ipputils::step_and_size_for_window(w);
    unsafe {
        let (mut mean, mut stddev) : (f64, f64) = (0. ,0.);
        let ans = crate::foreign::ipp::ippcv::ippiMean_StdDev_32f_C1R(
            w.as_ptr(),
            stride,
            mem::transmute(roi),
            &mut mean as *mut _,
            &mut stddev as *mut _
        );
        assert!(ans == 0);
        (mean as f32, stddev as f32)
    }
}

// TODO rename this module to fold

// a < b ? 255 else 0
pub fn is_pairwise_min(a : &Window<'_, u8>, b : &Window<'_, u8>, mut dst : WindowMut<'_, u8>) {

    #[cfg(feature="ipp")]
    unsafe {
        let (a_stride, a_roi) = crate::image::ipputils::step_and_size_for_window(a);
        let (b_stride, b_roi) = crate::image::ipputils::step_and_size_for_window(b);
        let (dst_stride, dst_roi) = crate::image::ipputils::step_and_size_for_window_mut(&dst);
        let ans = crate::foreign::ipp::ippi::ippiCompare_8u_C1R(
            a.as_ptr(),
            a_stride,
            b.as_ptr(),
            b_stride,
            dst.as_mut_ptr(),
            dst_stride,
            a_roi,
            crate::foreign::ipp::ippi::IppCmpOp_ippCmpLess
        );
        assert!(ans == 0);
        return;
    }

    unimplemented!()
}

#[cfg(feature="ipp")]
pub fn max_f32(win : &Window<f32>) -> f32 {
    if win.pixel_is::<f32>() {
        let (step, roi) = crate::image::ipputils::step_and_size_for_window(win);
        let mut max : f32 = 0.;
        unsafe {
            let ans = crate::foreign::ipp::ippi::ippiMax_32f_C1R(
                std::mem::transmute(win.as_ptr()), 
                step, 
                roi, 
                &mut max as *mut _
            );
            assert!(ans == 0);
            return max;
        }
    }
    unimplemented!()
}

pub fn max<N>(win : &Window<'_, N>) -> N
where
    N : PartialOrd + Any + Clone + Copy + Debug,
    u8 : AsPrimitive<N>
{

    #[cfg(feature="ipp")]
    unsafe {
        let (byte_stride, roi) = crate::image::ipputils::step_and_size_for_window(win);
        if win.pixel_is::<u8>() {
            let mut max : u8 = 0;
            let ans = crate::foreign::ipp::ippi::ippiMax_8u_C1R(std::mem::transmute(win.as_ptr()), byte_stride, roi, &mut max as *mut _);
            assert!(ans == 0);
            let out : N = max.as_();
            return out;
        }
    }

    *(win.pixels(1).max_by(|a, b| a.partial_cmp(&b).unwrap_or(Ordering::Equal) ).unwrap())
}

pub fn mean<N, S>(win : &Window<'_, N>, n_pxs : usize) -> Option<S>
where
    N : Scalar + Any + Clone + Copy + Debug + Zero,
    S : Zero + From<f32> + From<N> + Add<Output=S> + Div<Output=S>,
    f32 : From<N>
{

    // TODO ipp version is ignoring subsampling parameter

    #[cfg(feature="ipp")]
    {
        let s = sum::<N, S>(win, n_pxs);
        return Some(s / S::from(win.area() as f32));
    }

    let sum_f = win.shrink_to_subsample(n_pxs)?.pixels(n_pxs).map(|px| f32::from(*px) ).sum::<f32>();
    let avg = S::from(sum_f / (win.area() as f32));
    Some(avg)
}

pub fn accum<N, S>(win : &Window<'_, N>) -> S
where
    N : Scalar + Any + Clone + Copy + Debug + Zero,
    S : Zero + From<N> + Add<Output=S>
{
    win.pixels(1).fold(S::zero(), |s, px| s + S::from(*px) )
}

pub fn sum<N, S>(win : &Window<'_, N>, n_pxs : usize) -> S
where
    N : Scalar + Any + Clone + Copy + Debug + Zero,
    S : Zero + From<f32> + From<N> + Add<Output=S>
{

    #[cfg(feature="ipp")]
    unsafe {

        let (step, roi) = crate::image::ipputils::step_and_size_for_window(win);
        let mut sum : f64 = 0.;

        if win.pixel_is::<u8>() {
            let ans = crate::foreign::ipp::ippi::ippiSum_8u_C1R(
                std::mem::transmute(win.as_ptr()),
                step,
                roi,
                &mut sum as *mut _
            );
            assert!(ans == 0);
            return S::from(sum as f32);
        }
    }

    win.pixels(n_pxs).fold(S::zero(), |s, px| s + S::from(*px) )
}

pub fn min<N>(win : &Window<'_, N>) -> N
where
    N : Ord + Any + Clone + Copy + Debug
{
    *(win.pixels(1).min().unwrap())
}

pub fn dot_prod_to<N>(a : &Window<N>, b : &Window<u8>, dst : &mut WindowMut<N>)
where
    N : Scalar + Debug + Copy
{

    // use std::string::String;
    // unsafe { crate::foreign:ipp::ippi::ippiDotProd_32f64f_C1R(src1, src2) }
    // let ans = crate::foreign::ipp::ippi::ippiDotProd_( , int src1Step , const
    // Ipp<srcDatatype>* pSrc2 , int src2Step , IppiSize roiSize , Ipp64f* pDp );*/
}

