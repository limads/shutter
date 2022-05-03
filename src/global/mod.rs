use std::cmp::{PartialOrd, Ord, Ordering};
use crate::image::*;
use serde::{Serialize, de::DeserializeOwned};
use std::any::Any;
use std::fmt::Debug;
use std::ops::*;
use num_traits::Zero;
use std::ops::Add;
use nalgebra::Scalar;

pub fn max<N>(win : &Window<'_, N>) -> N
where
    N : PartialOrd + Any + Clone + Copy + Debug
{
    *(win.pixels(1).max_by(|a, b| a.partial_cmp(&b).unwrap_or(Ordering::Equal) ).unwrap())
}

pub fn sum<N>(win : &Window<'_, N>) -> N
where
    N : Scalar + Any + Clone + Copy + Debug + Zero + Add<Output=N>
{
    win.pixels(1).fold(N::zero(), |s, px| s + *px )
}

pub fn min<N>(win : &Window<'_, N>) -> N
where
    N : Ord + Any + Clone + Copy + Debug
{
    *(win.pixels(1).min().unwrap())
}

/*IppStatus ippiDotProd_<mod> ( const Ipp<srcDatatype>* pSrc1 , int src1Step , const
Ipp<srcDatatype>* pSrc2 , int src2Step , IppiSize roiSize , Ipp64f* pDp );*/

