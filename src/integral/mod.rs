/*IppStatus ippiIntegral_8u32s_C1R(const Ipp8u* pSrc, int srcStep, Ipp32s* pDst, int
dstStep, IppiSize srcRoiSize, Ipp32s val );*/
use std::any::Any;
use crate::image::*;
use num_traits::Zero;
use nalgebra::*;
use serde::de::DeserializeOwned;
use serde::Serialize;

pub struct Integral<T>(Image<T>)
where
    T : Scalar + Clone + Copy + Serialize + DeserializeOwned + Any;

impl<T> Integral<T>
where
    T : Scalar + Clone + Copy + Serialize + DeserializeOwned + Any + Zero + From<u8> + Default + std::ops::AddAssign
{

    pub fn calculate(win : &Window<'_, u8>) -> Self {
        let mut img = Image::<T>::new_constant(win.height(), win.width(), T::zero());
        img[(0, 0)] = T::from(win[(0 as usize, 0 as usize)]);
        unsafe {
            for ix in 1..img.len() {
                let prev = *img.unchecked_linear_index(ix-1);
                *img.unchecked_linear_index_mut(ix) += prev;
            }
        }
        Self(img)
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


