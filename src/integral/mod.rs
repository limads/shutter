use std::any::Any;
use crate::image::*;
use num_traits::Zero;
use nalgebra::*;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::mem;

pub struct Integral<T>(Image<T>)
where
    T : Scalar + Clone + Copy + Serialize + DeserializeOwned + Any;

impl<T> Integral<T>
where
    T : Scalar + Clone + Copy + Serialize + DeserializeOwned + Any + Zero + From<u8> + Default + std::ops::AddAssign
{

    pub fn calculate(win : &Window<'_, u8>) -> Self {

        let mut dst = Image::<T>::new_constant(win.height(), win.width(), T::zero());

        #[cfg(feature="ipp")]
        unsafe {
            if (&T::default() as &dyn Any).is::<i32>() {
                let (src_step, src_sz) = crate::image::ipputils::step_and_size_for_window(win);
                let (dst_step, dst_sz) = crate::image::ipputils::step_and_size_for_window_mut(&dst.full_window_mut());
                let offset : i32 = 0;
                let ans = crate::foreign::ipp::ippcv::ippiIntegral_8u32s_C1R(
                    win.as_ptr(),
                    src_step,
                    mem::transmute(dst.full_window_mut().as_mut_ptr()),
                    dst_step,
                    std::mem::transmute(src_sz),
                    offset
                );
                assert!(ans == 0);
                return Integral(dst);
            }
        }

        dst[(0, 0)] = T::from(win[(0 as usize, 0 as usize)]);
        unsafe {
            for ix in 1..dst.len() {
                let prev = *dst.unchecked_linear_index(ix-1);
                *dst.unchecked_linear_index_mut(ix) += prev;
            }
        }
        Self(dst)
    }

}

impl<T> AsRef<Image<T>> for Integral<T> 
where
    T : Scalar + Clone + Copy + Serialize + DeserializeOwned + Any + Zero + From<u8>
{

    fn as_ref(&self) -> &Image<T> {
        &self.0
    }

}


