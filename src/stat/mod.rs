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

#[cfg(feature="ipp")]
pub fn dot_prod<S, T>(a : &Image<u8, S>, b : &Image<u8, T>) -> f64
where
    S : Storage<u8>,
    T : Storage<u8>
{
    let (a_stride, a_roi) = crate::image::ipputils::step_and_size_for_image(a);
    let (b_stride, b_roi) = crate::image::ipputils::step_and_size_for_image(b);
    let mut dst : f64 = 0.0;
    unsafe {
        let ans = crate::foreign::ipp::ippi::ippiDotProd_8u64f_C1R(
            a.as_ptr(),
            a_stride,
            b.as_ptr(),
            b_stride,
            a_roi,
            &mut dst
        );
        assert!(ans == 0);
        dst
    }
}

// Calculates the global maximum and minimum of the image.
// Then sets all pixels close to the minimum to zero.
/*pub fn supress_close_to_min_mut(w : &mut WindowMut<u8>) {
    let (min, max) = min_max(&w);
    for px in w.pixels_mut(1) {
        if *px - min < max - *px {
            *px = 0;
        }
    }
}*/

/*#[cfg(feature="ipp")]
pub fn max_f32(win : &Window<f32>) -> f32 {
    if win.pixel_is::<f32>() {
        let (step, roi) = crate::image::ipputils::step_and_size_for_image(win);
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
}*/

impl<P, S> Image<P, S> 
where
    P : Pixel + PartialOrd,
    S : Storage<P>,
    Box<[P]> : Storage<P>,
    for<'a> &'a [P] : Storage<P>
{

    /*
    impl ImageBuf<f32> {
        pub fn imax(&self) -> ((usize, usize), f32) {
            let (mut max_ix, mut max) = ((0, 0), f32::NEG_INFINITY);
            for (lin_ix, px) in self.full_window().pixels(1).enumerate() {
                if *px > max {
                    max_ix = index::coordinate_index(lin_ix, self.width);
                    max = *px
                }
            }
            (max_ix, max)
        }
    }
    */
    
    pub fn max(&self) -> P 
    where
        f32 : AsPrimitive<P>,
        u8 : AsPrimitive<P>
    {

        #[cfg(feature="ipp")]
        unsafe {
            let (byte_stride, roi) = crate::image::ipputils::step_and_size_for_image(self);
            if self.pixel_is::<u8>() {
                let mut max : u8 = 0;
                let ans = crate::foreign::ipp::ippi::ippiMax_8u_C1R(
                    std::mem::transmute(self.as_ptr()), 
                    byte_stride, 
                    roi, 
                    &mut max as *mut _
                );
                assert!(ans == 0);
                let out : P = max.as_();
                return out;
            } else if self.pixel_is::<f32>() {
                let mut max : f32 = 0.0;
                let ans = crate::foreign::ipp::ippi::ippiMax_32f_C1R(
                    std::mem::transmute(self.as_ptr()), 
                    byte_stride, 
                    roi, 
                    &mut max as *mut _
                );
                assert!(ans == 0);
                let out : P = max.as_();
                return out;
            }
        }

        let max = self.pixels(1).max_by(|a, b| 
            a.partial_cmp(&b).unwrap_or(Ordering::Equal) 
        ).unwrap();
        *max
    }
    
    pub fn min(&self) -> P {
        let min = self.pixels(1).min_by(|a, b| 
            a.partial_cmp(&b).unwrap_or(Ordering::Equal) 
        ).unwrap();
        *min
    }
    
    #[cfg(feature="ipp")]
    pub fn min_max(&self) -> (P, P)
    where
        u8 : AsPrimitive<P>,
        i16 : AsPrimitive<P>,
        u16 : AsPrimitive<P>,
        f32 : AsPrimitive<P>
    {
        let (step, roi) = crate::image::ipputils::step_and_size_for_image(self);
        unsafe {
            // No option for i32.
            match self.depth() {
                Depth::U8 => {
                    let (mut min, mut max) = (0u8, 0u8);
                    let ans = crate::foreign::ipp::ippi::ippiMinMax_8u_C1R(
                        mem::transmute(self.as_ptr()),
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
                        mem::transmute(self.as_ptr()),
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
                        mem::transmute(self.as_ptr()),
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
                        mem::transmute(self.as_ptr()),
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


}

impl<S> Image<u8, S>
where
    S : Storage<u8>
{

    pub fn masked_sum<T>(&self, mask : &Image<u8, T>) -> u32
    where
        T : Storage<u8>
    {
        assert!(self.shape() == mask.shape());
        // assert!(self.width() % 16 == 0);
        /*let mut sum = wide::u8x16::ZERO;
        for (pxs, ms) in self.packed_pixels().zip(mask.packed_pixels()) {

        }*/
        let mut sum = 0;
        for (px, m) in self.pixels(1).zip(mask.pixels(1)) {
            if *m != 0 {
                sum += *px as u32;
            }
        }
        sum
    }

}

impl<P, S> Image<P, S>
where
    P : Pixel + Add<Output=P>,
    S : Storage<P>,
    //Box<[P]> : Storage<P>,
    //for<'a> &'a [P] : Storage<P>
{

    /*
     pub fn mean(&self, n_pxs : usize) -> Option<u8> {
        Some((self.shrink_to_subsample(n_pxs)?.pixels(n_pxs).map(|px| *px as u64 ).sum::<u64>() / (self.width() * self.height()) as u64) as u8)
    }
    */

    pub fn mean<A>(&self, n_pxs : usize) -> Option<A>
    where
        A : Pixel + From<P>,
        A : From<f32> + Add<Output=A> + Div<Output=A>,
        f32 : From<P>
    {

        // TODO ipp version is ignoring subsampling parameter

        #[cfg(feature="ipp")]
        {
            let s = self.sum::<A>(n_pxs);
            return Some(s / A::from(self.area() as f32));
        }

        let sum_f = self.shrink_to_subsample(n_pxs)?.pixels(n_pxs)
            .map(|px| f32::from(*px) )
            .sum::<f32>();
        let avg = A::from(sum_f / (self.area() as f32));
        Some(avg)
    }
    
    #[cfg(feature="ipp")]
    pub fn masked_mean_stddev<T>(&self, mask : &Image<P, T>) -> (f32, f32)
    where
        T : Storage<P>
    {
        let (stride, roi) = crate::image::ipputils::step_and_size_for_image(self);
        unsafe {
            if self.pixel_is::<u8>() {
                let (mut mean, mut stddev) : (f64, f64) = (0. ,0.);
                let ans = crate::foreign::ipp::ippcv::ippiMean_StdDev_8u_C1MR(
                    mem::transmute(self.as_ptr()),
                    self.byte_stride() as i32,
                    mem::transmute(mask.as_ptr()),
                    mask.byte_stride() as i32,
                    self.size().into(),
                    &mut mean as *mut _,
                    &mut stddev as *mut _
                );
                assert!(ans == 0);
                return (mean as f32, stddev as f32);
            }
        }
        unimplemented!()
    }

    #[cfg(feature="ipp")]
    pub fn mean_stddev(&self) -> (f32, f32) {
        let (stride, roi) = crate::image::ipputils::step_and_size_for_image(self);
        unsafe {

            if self.pixel_is::<u8>() {
                let (mut mean, mut stddev) : (f64, f64) = (0. ,0.);
                let ans = crate::foreign::ipp::ippcv::ippiMean_StdDev_8u_C1R(
                    mem::transmute(self.as_ptr()),
                    stride,
                    mem::transmute(roi),
                    &mut mean as *mut _,
                    &mut stddev as *mut _
                );
                assert!(ans == 0);
                return (mean as f32, stddev as f32);
            }

            if self.pixel_is::<f32>() {
                let (mut mean, mut stddev) : (f64, f64) = (0. ,0.);
                let ans = crate::foreign::ipp::ippcv::ippiMean_StdDev_32f_C1R(
                    mem::transmute(self.as_ptr()),
                    stride,
                    mem::transmute(roi),
                    &mut mean as *mut _,
                    &mut stddev as *mut _
                );
                assert!(ans == 0);
                return (mean as f32, stddev as f32);
            }
        }
        unimplemented!()
    }

    pub fn accum<A>(&self) -> A
    where
        A : From<P>,
        A : Zero + From<P> + Add<Output=A> + Pixel
    {
        self.pixels(1).fold(A::zero(), |s, px| s + A::from(*px) )
    }

    pub fn sum<A>(&self, n_pxs : usize) -> A
    where
        A : Pixel + From<f32> + From<P> + Add<Output=A>
    {

        #[cfg(feature="ipp")]
        unsafe {

            let (step, roi) = crate::image::ipputils::step_and_size_for_image(self);
            let mut sum : f64 = 0.;

            if self.pixel_is::<u8>() {
                let ans = crate::foreign::ipp::ippi::ippiSum_8u_C1R(
                    std::mem::transmute(self.as_ptr()),
                    step,
                    roi,
                    &mut sum as *mut _
                );
                assert!(ans == 0);
                return A::from(sum as f32);
            } else if self.pixel_is::<i16>() {
                let ans = crate::foreign::ipp::ippi::ippiSum_16s_C1R(
                    std::mem::transmute(self.as_ptr()),
                    step,
                    roi,
                    &mut sum as *mut _
                );
                assert!(ans == 0);
                return A::from(sum as f32);
            }
        }

        self.pixels(n_pxs).fold(A::zero(), |s, px| s + A::from(*px) )
    }
    
    // Accumulates pixels without conversion.
    pub fn baseline_sum(s : &Image<P, S>) -> P {
        s.pixels(1).fold(P::zero(), |s, p| s + *p )
    }
    
}

/*pub fn dot_prod_to<N>(a : &Window<N>, b : &Window<u8>, dst : &mut WindowMut<N>)
where
    N : Scalar + Debug + Copy
{

    /*// use std::string::String;
    // unsafe { crate::foreign:ipp::ippi::ippiDotProd_32f64f_C1R(src1, src2) }
    // let ans = crate::foreign::ipp::ippi::ippiDotProd_( , int src1Step , const
    // Ipp<srcDatatype>* pSrc2 , int src2Step , IppiSize roiSize , Ipp64f* pDp );*/
}*/



