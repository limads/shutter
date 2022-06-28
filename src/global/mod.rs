use std::cmp::{PartialOrd, Ord, Ordering};
use crate::image::*;
use serde::{Serialize, de::DeserializeOwned};
use std::any::Any;
use std::fmt::Debug;
use std::ops::*;
use num_traits::Zero;
use std::ops::Add;
use nalgebra::Scalar;
use num_traits::AsPrimitive;

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

    // TODO ignoring n_pxs parameter

    #[cfg(feature="ipp")]
    unsafe {

        let (byte_stride, roi) = crate::image::ipputils::step_and_size_for_window(win);
        let mut sum : f64 = 0.;

        if win.pixel_is::<u8>() {
            let ans = crate::foreign::ipp::ippi::ippiSum_8u_C1R(std::mem::transmute(win.as_ptr()), byte_stride, roi, &mut sum as *mut _);
            assert!(ans == 0);
            return S::from(sum as f32);
        }
    }

    win.pixels(1).fold(S::zero(), |s, px| s + S::from(*px) )
}

pub fn min<N>(win : &Window<'_, N>) -> N
where
    N : Ord + Any + Clone + Copy + Debug
{
    *(win.pixels(1).min().unwrap())
}

/*IppStatus ippiDotProd_<mod> ( const Ipp<srcDatatype>* pSrc1 , int src1Step , const
Ipp<srcDatatype>* pSrc2 , int src2Step , IppiSize roiSize , Ipp64f* pDp );*/

