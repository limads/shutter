use crate::image::Image;
use std::ops::Index;
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy)]
pub enum RGB {
    Red,
    Green,
    Blue,
}

#[derive(Debug, Clone, Copy)]
pub enum YCbCr {
    Y,
    Cb,
    Cr
}

#[derive(Debug, Clone, Copy)]
pub enum YUV {
    Y,
    U,
    V
}

/// Represents a color image in its planar representation (channels can be treated as separate gray images
/// with the same dimensions, and pixels of the same channel are memory-contiguous).
pub struct ColorImage<P> {

    chs : [Image<u8>; 3],

    ty : PhantomData<P>

}

impl<P> AsRef<[Image<u8>; 3]> for ColorImage<P> {

    fn as_ref(&self) -> &[Image<u8>; 3] {
        &self.chs
    }

}

impl<P> AsMut<[Image<u8>; 3]> for ColorImage<P> {

    fn as_mut(&mut self) -> &mut [Image<u8>; 3] {
        &mut self.chs
    }

}

pub type RGBImage = ColorImage<RGB>;

impl Index<RGB> for RGBImage {

    type Output = Image<u8>;

    fn index(&self, ix : RGB) -> &Self::Output {
        match ix {
            RGB::Red => &self.chs[0],
            RGB::Green => &self.chs[1],
            RGB::Blue => &self.chs[2],
        }
    }

}

pub type YCbCrImage = ColorImage<YCbCr>;

impl Index<YCbCr> for YCbCrImage {

    type Output = Image<u8>;

    fn index(&self, ix : YCbCr) -> &Self::Output {
        match ix {
            YCbCr::Y => &self.chs[0],
            YCbCr::Cb => &self.chs[1],
            YCbCr::Cr => &self.chs[2],
        }
    }

}

pub type YUVImage = ColorImage<YUV>;

impl Index<YUV> for YUVImage {

    type Output = Image<u8>;

    fn index(&self, ix : YUV) -> &Self::Output {
        match ix {
            YUV::Y => &self.chs[0],
            YUV::U => &self.chs[1],
            YUV::V => &self.chs[2],
        }
    }

}
