use crate::image::*;
use std::mem::transmute;
use std::ops::Add;

impl<P, S> Image<P, S>
where
    S : Storage<P>,
    P : Pixel + num_traits::Zero + Add + num_traits::ToPrimitive
{

    pub fn norm_l2(&self) -> f32 {
        unsafe {
            if self.pixel_is::<u8>() {
                let mut norm = 0.0;
                let ans = crate::foreign::ipp::ippi::ippiNorm_L2_8u_C1R(
                    transmute(self.as_ptr()),
                    self.byte_stride() as i32,
                    self.size().into(),
                    &mut norm
                );
                assert!(ans == 0);
                norm as f32
            } else if self.pixel_is::<i16>() {
                let mut norm = 0.0;
                let ans = crate::foreign::ipp::ippi::ippiNorm_L2_16s_C1R(
                    transmute(self.as_ptr()),
                    self.byte_stride() as i32,
                    self.size().into(),
                    &mut norm
                );
                assert!(ans == 0);
                norm as f32
            } else {
                // Only for P : signed
                /*self.pixels(1).map(|px| px.abs() )
                    .fold(P::zero(), |s, p| s + p )
                    .to_f32().unwrap()*/
                unimplemented!()
            }
        }
    }

    pub fn norm_l1(&self) -> f32 {
        unsafe {
            if self.pixel_is::<u8>() {
                let mut norm = 0.0;
                let ans = crate::foreign::ipp::ippi::ippiNorm_L1_8u_C1R(
                    transmute(self.as_ptr()),
                    self.byte_stride() as i32,
                    self.size().into(),
                    &mut norm
                );
                assert!(ans == 0);
                norm as f32
            } else if self.pixel_is::<i16>() {
                let mut norm = 0.0;
                let ans = crate::foreign::ipp::ippi::ippiNorm_L1_16s_C1R(
                    transmute(self.as_ptr()),
                    self.byte_stride() as i32,
                    self.size().into(),
                    &mut norm
                );
                assert!(ans == 0);
                norm as f32
            } else {
                // Only for P : signed
                /*self.pixels(1).map(|px| px.abs() )
                    .fold(P::zero(), |s, p| s + p )
                    .to_f32().unwrap()*/
                unimplemented!()
            }
        }
    }

    pub fn norm_l1_masked<T : Storage<u8>>(&self, mask : &Image<u8, T>) -> f32 {
        assert!(self.size() == mask.size());
        let mut norm = 0.0;
        let ans = unsafe {
            if self.pixel_is::<u8>() {
                crate::foreign::ipp::ippcv::ippiNorm_L1_8u_C1MR(
                    transmute(self.as_ptr()),
                    self.byte_stride() as i32,
                    mask.as_ptr(),
                    mask.byte_stride() as i32,
                    self.size().into(),
                    &mut norm
                )
            } else {
                unimplemented!()
            }
        };
        assert!(ans == 0);
        norm as f32
    }

    pub fn diff_l1<T : Storage<P>>(&self, other : &Image<P, T>) -> f32 {
        let mut diff : f64 = 0.0;
        unsafe {
            if self.pixel_is::<i16>() {
                let ans = crate::foreign::ipp::ippi::ippiNormDiff_L1_16s_C1R(
                    transmute(self.as_ptr()),
                    self.byte_stride() as i32,
                    transmute(other.as_ptr()),
                    other.byte_stride() as i32,
                    self.size().into(),
                    &mut diff
                );
                assert!(ans == 0);
            } else {
                unimplemented!()
            }
        }
        diff as f32
    }

    pub fn diff_l2<T : Storage<P>>(&self, other : &Image<P, T>) -> f32 {
        let mut diff : f64 = 0.0;
        unsafe {
            if self.pixel_is::<i16>() {
                let ans = crate::foreign::ipp::ippi::ippiNormDiff_L2_16s_C1R(
                    transmute(self.as_ptr()),
                    self.byte_stride() as i32,
                    transmute(other.as_ptr()),
                    other.byte_stride() as i32,
                    self.size().into(),
                    &mut diff
                );
                assert!(ans == 0);
            } else {
                unimplemented!()
            }
        }
        diff as f32
    }


}


