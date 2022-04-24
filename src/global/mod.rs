use std::cmp::Ord;
use crate::image::*;
use serde::{Serialize, de::DeserializeOwned};
use std::any::Any;
use std::fmt::Debug;

pub fn max<N>(win : &Window<'_, N>) -> N
where
    N : Ord + Serialize + DeserializeOwned + Any + Clone + Copy + Debug
{
    *(win.pixels(1).max().unwrap())
}

pub fn min<N>(win : &Window<'_, N>) -> N
where
    N : Ord + Serialize + DeserializeOwned + Any + Clone + Copy + Debug
{
    *(win.pixels(1).min().unwrap())
}

/*IppStatus ippiDotProd_<mod> ( const Ipp<srcDatatype>* pSrc1 , int src1Step , const
Ipp<srcDatatype>* pSrc2 , int src2Step , IppiSize roiSize , Ipp64f* pDp );*/

