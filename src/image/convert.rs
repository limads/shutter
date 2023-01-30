use crate::image::*;
use std::convert::{AsRef, AsMut, TryFrom};
use num_traits::Zero;
use nalgebra::Scalar;
use std::ops::Div;
use std::fmt::Debug;
use std::str::FromStr;
use std::borrow::Borrow;
use std::ops::Mul;
use num_traits::Bounded;
use num_traits::cast::{AsPrimitive, ToPrimitive};
use num_traits::ops::inv::Inv;
use num_traits::One;
use std::any::Any;
use std::mem;
use num_traits::Signed;
use std::ops::Sub;

pub use base::*;

pub use mul::*;

pub use abs::*;

/*
Integer <-> Float conversion is done by the mul_convert and div_convert
methods. mul_convert maps [-1.0-1.0] or [0.0-1.0] to [Integer::Min-Integer::MAX].

There is also norm_max_convert that maps integers to floats in [0.0-1.0].

Unsigned Integer <-> Signed integer conversion is done by the abs_convert methods.

Signed integer <-> Float conversion is one by the convert_to methods.
*/

impl<P, S> Image<P, S>
where
    P : Pixel,
    S : Storage<P>
{

    pub fn convert<Q, T>(&self) -> ImageBuf<Q>
    where
        Q : Pixel,
        Self : Convert<ImageBuf<Q>>
    {
        let mut dst = unsafe { ImageBuf::<Q>::new_empty_like(self) };
        self.convert_to(&mut dst);
        dst
    }
    
    pub fn abs_convert<Q, T>(&self) -> ImageBuf<Q>
    where
        Q : Pixel,
        Self : AbsConvert<ImageBuf<Q>>
    {
        let mut dst = unsafe { ImageBuf::<Q>::new_empty_like(self) };
        self.abs_convert_to(&mut dst);
        dst
    }
    
    pub fn mul_convert<Q, T>(&self) -> ImageBuf<Q>
    where
        Q : Pixel,
        Self : MulConvert<ImageBuf<Q>>
    {
        let mut dst = unsafe { ImageBuf::<Q>::new_empty_like(self) };
        self.mul_convert_to(&mut dst);
        dst
    }
    
    pub fn div_convert<Q, T>(&self) -> ImageBuf<Q>
    where
        Q : Pixel,
        Self : DivConvert<ImageBuf<Q>>
    {
        let mut dst = unsafe { ImageBuf::<Q>::new_empty_like(self) };
        self.div_convert_to(&mut dst);
        dst
    }
    
    pub fn abs_mul_convert<Q, T>(&self) -> ImageBuf<Q>
    where
        Q : Pixel,
        Self : AbsMulConvert<ImageBuf<Q>>
    {
        let mut dst = unsafe { ImageBuf::<Q>::new_empty_like(self) };
        self.abs_mul_convert_to(&mut dst);
        dst
    }
    
    pub fn abs_div_convert<Q, T>(&self) -> ImageBuf<Q>
    where
        Q : Pixel,
        Self : AbsDivConvert<ImageBuf<Q>>
    {
        let mut dst = unsafe { ImageBuf::<Q>::new_empty_like(self) };
        self.abs_div_convert_to(&mut dst);
        dst
    }
    
    pub fn norm_max_convert<Q, T>(&self) -> ImageBuf<Q>
    where
        Q : Pixel,
        Self : NormMaxConvert<ImageBuf<Q>>
    {
        let mut dst = unsafe { ImageBuf::<Q>::new_empty_like(self) };
        self.norm_max_convert_to(&mut dst);
        dst
    }
    
    pub fn norm_min_max_convert<Q, T>(&self) -> ImageBuf<Q>
    where
        Q : Pixel,
        Self : NormMinMaxConvert<ImageBuf<Q>>
    {
        let mut dst = unsafe { ImageBuf::<Q>::new_empty_like(self) };
        self.norm_min_max_convert_to(&mut dst);
        dst
    }
    
}

/* Truncating/saturating bidirectional integer to float conversion 
or unidirectional unsigned integer to signed integer conversion, or unidirectional
unsigned integer to float conversion. Those methods warp some of the possible standard library
From/Into implementations, since some conversions here are lossless. */
mod base {

    use super::*;
    
    pub trait Convert<T> {
        
        fn convert_to(&self, dst : &mut T);
        
    }

    /* Truncating unsigned to unsigned conversion */
    impl<S, T> Convert<Image<u8, T>> for Image<u16, S>
    where
        S : Storage<u16>,
        T : StorageMut<u8>,
    {

        fn convert_to(&self, dst : &mut Image<u8, T>) {
            convert_to(self, dst);
        }

    }
    impl<S, T> Convert<Image<u8, T>> for Image<u32, S>
    where
        S : Storage<u32>,
        T : StorageMut<u8>,
    {

        fn convert_to(&self, dst : &mut Image<u8, T>) {
            convert_to(self, dst);
        }

    }
    impl<S, T> Convert<Image<u16, T>> for Image<u32, S>
    where
        S : Storage<u32>,
        T : StorageMut<u16>,
    {

        fn convert_to(&self, dst : &mut Image<u16, T>) {
            convert_to(self, dst);
        }

    }

    /* Preserving unsigned to unsigned conversion */
    impl<S, T> Convert<Image<u16, T>> for Image<u8, S>
    where
        S : Storage<u8>,
        T : StorageMut<u16>,
    {

        fn convert_to(&self, dst : &mut Image<u16, T>) {
            convert_to(self, dst);
        }

    }
    impl<S, T> Convert<Image<u32, T>> for Image<u8, S>
    where
        S : Storage<u8>,
        T : StorageMut<u32>,
    {

        fn convert_to(&self, dst : &mut Image<u32, T>) {
            convert_to(self, dst);
        }

    }
    impl<S, T> Convert<Image<u32, T>> for Image<u16, S>
    where
        S : Storage<u16>,
        T : StorageMut<u32>,
    {

        fn convert_to(&self, dst : &mut Image<u32, T>) {
            convert_to(self, dst);
        }

    }

    /* Bidirectional float to signed implementations */
    
    // f32 -> i32
    impl<S, T> Convert<Image<f32, T>> for Image<i32, S> 
    where
        S : Storage<i32>,
        T : StorageMut<f32>,
    {

        fn convert_to(&self, dst : &mut Image<f32, T>) {
            convert_to(self, dst);
        }
        
    }

    // i32 -> f32
    impl<S, T> Convert<Image<i32, T>> for Image<f32, S> 
    where
        S : Storage<f32>,
        T : StorageMut<i32>,
    {

        fn convert_to(&self, dst : &mut Image<i32, T>) {
            convert_to(self, dst);
        }
        
    }
    
    // f32 -> i16
    impl<S, T> Convert<Image<f32, T>> for Image<i16, S> 
    where
        S : Storage<i16>,
        T : StorageMut<f32>,
    {

        fn convert_to(&self, dst : &mut Image<f32, T>) {
            convert_to(self, dst);
        }
        
    }

    impl<S, T> Convert<Image<u8, T>> for Image<i16, S>
    where
        S : Storage<i16>,
        T : StorageMut<u8>,
    {

        fn convert_to(&self, dst : &mut Image<u8, T>) {
            convert_to(self, dst);
        }
    }

    // i16 -> f32
    impl<S, T> Convert<Image<i16, T>> for Image<f32, S> 
    where
        S : Storage<f32>,
        T : StorageMut<i16>,
    {

        fn convert_to(&self, dst : &mut Image<i16, T>) {
            convert_to(self, dst);
        }
        
    }
    
    /* Unidirectional unsigned to signed */ 
    
    // u8 -> i32
    impl<S, T> Convert<Image<i32, T>> for Image<u8, S> 
    where
        S : Storage<u8>,
        T : StorageMut<i32>,
    {

        fn convert_to(&self, dst : &mut Image<i32, T>) {
            convert_to(self, dst);
        }
        
    }
    
    // i16 -> u8
    impl<S, T> Convert<Image<i16, T>> for Image<u8, S> 
    where
        S : Storage<u8>,
        T : StorageMut<i16>,
    {

        fn convert_to(&self, dst : &mut Image<i16, T>) {
            convert_to(self, dst);
        }
        
    }

    /* Unidirectional unsigned to float */ 
    
    // u8 -> f32
    impl<S, T> Convert<Image<f32, T>> for Image<u8, S> 
    where
        S : Storage<u8>,
        T : StorageMut<f32>,
    {

        fn convert_to(&self, dst : &mut Image<f32, T>) {
            convert_to(self, dst);
        }
        
    }

    fn convert_to<P, Q, S, T>(a : &Image<P, S>, b : &mut Image<Q, T>) 
    where
        P : Pixel + ToPrimitive + PartialOrd +  AsPrimitive<Q>,
        Q : Pixel + ToPrimitive + PartialOrd +  AsPrimitive<P>,
        S : Storage<P>,
        T : StorageMut<Q>,
        u8 : AsPrimitive<Q> + AsPrimitive<P>,
        f32 : AsPrimitive<Q> + AsPrimitive<P>,
        i16 : AsPrimitive<P>,
        u16 : AsPrimitive<P>
    {
        assert!(a.same_size(b));
        #[cfg(feature="ipp")]
        unsafe { 
            if ippi_convert::<P, Q, S, T>(a, b, Conversion::Preserve) {
                return;
            } 
        }
        b.pixels_mut(1).zip(a.pixels(1)).for_each(|(dst, src)| *dst = src.as_() );
        // unimplemented!()
    }

}

/* Saturating unidirectional signed integer to unsigned integer conversion */
mod abs {

    use super::*;
        
    pub trait AbsConvert<T> {

        fn abs_convert_to(&self, dst : &mut T);
        
    }

    // i32 -> u8
    impl<S, T> AbsConvert<Image<u8, T>> for Image<i32, S> 
    where
        S : Storage<i32>,
        T : StorageMut<u8>,
    {

        fn abs_convert_to(&self, dst : &mut Image<u8, T>) {
            abs_convert_to(self, dst);
        }
        
    }
    
    // i16 -> u8
    impl<S, T> AbsConvert<Image<u8, T>> for Image<i16, S> 
    where
        S : Storage<i16>,
        T : StorageMut<u8>,
    {

        fn abs_convert_to(&self, dst : &mut Image<u8, T>) {
            abs_convert_to(self, dst);
        }
        
    }
    
    fn abs_convert_to<P, Q, S, T>(a : &Image<P, S>, b : &mut Image<Q, T>) 
    where
        P : Pixel + num_traits::Signed,
        Q : Pixel,
        S : Storage<P>,
        T : StorageMut<Q>,
        Q : AsPrimitive<P>,
        P : AsPrimitive<Q>
    {
        assert!(a.same_size(b));
        b.pixels_mut(1).zip(a.pixels(1)).for_each(|(dst, src)| *dst = src.abs().as_() );
    }

}

mod div {

    use super::*;
        
    pub trait DivConvert<T> {

        fn div_convert_to(&self, dst : &mut T);
        
    }

    // Converts [0, u8::MAX] to [0.0,1.0]
    impl<S, T> DivConvert<Image<f32, T>> for Image<u8, S> 
    where
        S : Storage<u8>,
        T : StorageMut<f32>,
    {

        fn div_convert_to(&self, dst : &mut Image<f32, T>) {
            div_convert_to(self, dst);
        }
        
    }
    
    // Converts [i32::MIN, i32::MAX] to [-1.0, 1.0]
    impl<S, T> DivConvert<Image<f32, T>> for Image<i32, S> 
    where
        S : Storage<i32>,
        T : StorageMut<f32>,
    {

        fn div_convert_to(&self, dst : &mut Image<f32, T>) {
            div_convert_to(self, dst);
        }
        
    }
    
    fn div_convert_to<P, Q, S, T>(a : &Image<P, S>, b : &mut Image<Q, T>) 
    where
        P : Pixel + ToPrimitive + PartialOrd +  AsPrimitive<Q> + Bounded,
        Q : Pixel + ToPrimitive + PartialOrd +  AsPrimitive<P> + Div<Output=Q>,
        S : Storage<P>,
        T : StorageMut<Q>,
        u8 : AsPrimitive<Q> + AsPrimitive<P>,
        f32 : AsPrimitive<Q> + AsPrimitive<P>,
        i16 : AsPrimitive<P>,
        u16 : AsPrimitive<P>
    {
        assert!(a.same_size(b));
        #[cfg(feature="ipp")]
        unsafe { 
            if ippi_convert::<P, Q, S, T>(a, b, Conversion::Scale) {
                return;
            } 
        }
        let bound : Q = P::max_value().as_();
        b.pixels_mut(1).zip(a.pixels(1)).for_each(|(dst, src)| *dst = src.as_() / bound );
    }
}

mod mul {

    use super::*;
    
    pub trait MulConvert<T> {

        fn mul_convert_to(&self, dst : &mut T);
        
    }

    // Converts 0.0..1.0 to 0..u8::MAX (saturating floats outside that range).
    impl<S, T> MulConvert<Image<u8, T>> for Image<f32, S> 
    where
        S : Storage<f32>,
        T : StorageMut<u8>,
    {

        fn mul_convert_to(&self, dst : &mut Image<u8, T>) {
            mul_convert_to(self, dst);
        }
        
    }
    
    // Converts -1.0..1.0 to i32::MIN..i32::MAX (saturating floats outside that range).
    impl<S, T> MulConvert<Image<i32, T>> for Image<f32, S> 
    where
        S : Storage<f32>,
        T : StorageMut<i32>,
    {

        fn mul_convert_to(&self, dst : &mut Image<i32, T>) {
            mul_convert_to(self, dst);
        }
        
    }
    
    fn mul_convert_to<P, Q, S, T>(a : &Image<P, S>, b : &mut Image<Q, T>) 
    where
        P : Pixel + ToPrimitive + PartialOrd +  AsPrimitive<Q> + Mul<Output=P>,
        Q : Pixel + ToPrimitive + PartialOrd +  AsPrimitive<P> + Bounded,
        S : Storage<P>,
        T : StorageMut<Q>,
        u8 : AsPrimitive<Q> + AsPrimitive<P>,
        f32 : AsPrimitive<Q> + AsPrimitive<P>,
        i16 : AsPrimitive<P>,
        u16 : AsPrimitive<P>
    {
        assert!(a.same_size(b));
        #[cfg(feature="ipp")]
        unsafe { 
            if ippi_convert::<P, Q, S, T>(a, b, Conversion::Scale) {
                return;
            } 
        }
        let bound : P = Q::max_value().as_();
        b.pixels_mut(1).zip(a.pixels(1)).for_each(|(dst, src)| *dst = (*src * bound).as_() );
    }
    
}

mod abs_mul {

    use super::*;
    
    pub trait AbsMulConvert<T> {

        fn abs_mul_convert_to(&self, dst : &mut T);
        
    }

    // Converts -1.0..1.0 to 0..i32::MAX (taking abs value first).
    impl<S, T> AbsMulConvert<Image<i32, T>> for Image<f32, S> 
    where
        S : Storage<f32>,
        T : StorageMut<i32>,
    {

        fn abs_mul_convert_to(&self, dst : &mut Image<i32, T>) {
            abs_mul_convert_to(self, dst);
        }
        
    }
    
    fn abs_mul_convert_to<P, Q, S, T>(a : &Image<P, S>, b : &mut Image<Q, T>) 
    where
        P : Pixel + Signed + Mul<Output=P>,
        Q : Pixel + Bounded,
        S : Storage<P>,
        T : StorageMut<Q>,
        Q : AsPrimitive<P>,
        P : AsPrimitive<Q>
    {
        assert!(a.same_size(b));
        let bound : P = Q::max_value().as_();
        b.pixels_mut(1).zip(a.pixels(1)).for_each(|(dst, src)| *dst = (src.abs() * bound).as_() );
    }
    
}

mod abs_div {
    
    use super::*;
        
    pub trait AbsDivConvert<T> {

        fn abs_div_convert_to(&self, dst : &mut T);
        
    }

    // Converts i32::MIN..i32::MAX to 0.0..1.0, taking the absolute value first.
    impl<S, T> AbsDivConvert<Image<f32, T>> for Image<i32, S> 
    where
        S : Storage<i32>,
        T : StorageMut<f32>,
    {

        fn abs_div_convert_to(&self, dst : &mut Image<f32, T>) {
            abs_div_convert_to(self, dst);
        }
        
    }
    
    fn abs_div_convert_to<P, Q, S, T>(a : &Image<P, S>, b : &mut Image<Q, T>) 
    where
        P : Pixel + Bounded,
        Q : Pixel + Div<Output=Q>,
        S : Storage<P>,
        T : StorageMut<Q>,
        Q : AsPrimitive<P>,
        P : AsPrimitive<Q>
    {
        assert!(a.same_size(b));
        let bound : Q = P::max_value().as_();
        b.pixels_mut(1).zip(a.pixels(1)).for_each(|(dst, src)| *dst = src.as_() / bound );
    }
}

/* Bidirectional unsigned to float conversion by normalization:
For unsigned -> float map [0, max] to [0.0, 1.0] 
For float -> unsigned map [0.0, 1.0] to [0..UMAX], saturating if necessary */ 
mod norm_max {

    use super::*;
    
    pub trait NormMaxConvert<T> {

        fn norm_max_convert_to(&self, dst : &mut T);
        
    }

    impl<S, T> NormMaxConvert<Image<f32, T>> for Image<u8, S> 
    where
        S : Storage<u8>,
        T : StorageMut<f32>,
    {

        fn norm_max_convert_to(&self, dst : &mut Image<f32, T>) {
            norm_max_convert_to(self, dst);
        }
        
    }
    
    impl<S, T> NormMaxConvert<Image<u8, T>> for Image<f32, S> 
    where
        S : Storage<f32>,
        T : StorageMut<u8>,
    {

        fn norm_max_convert_to(&self, dst : &mut Image<u8, T>) {
            norm_max_convert_to(self, dst);
        }
        
    }
    
    fn norm_max_convert_to<P, Q, S, T>(a : &Image<P, S>, b : &mut Image<Q, T>) 
    where
        P : Pixel + ToPrimitive + PartialOrd +  AsPrimitive<Q> + Bounded,
        Q : Pixel + ToPrimitive + PartialOrd +  AsPrimitive<P> + Div<Output=Q>,
        S : Storage<P>,
        T : StorageMut<Q>,
        u8 : AsPrimitive<Q> + AsPrimitive<P>,
        f32 : AsPrimitive<Q> + AsPrimitive<P>,
        i16 : AsPrimitive<P>,
        u16 : AsPrimitive<P>
    {
        assert!(a.same_size(b));
        #[cfg(feature="ipp")]
        unsafe { 
            if ippi_convert::<P, Q, S, T>(a, b, Conversion::NormalizeMax) {
                return;
            } 
        }
        let max : Q = a.max().as_();
        b.pixels_mut(1).zip(a.pixels(1)).for_each(|(dst, src)| *dst = src.as_() / max );
    }
    
}

// Bidirectional signed or unsigned conversion by minimum and maximum normalization
// For unsigned -> float map [min, max] to [0.0, 1.0] (or the converse)
// For signed -> float map [min, max] to [-1.0, 1.0] (or the converse)
mod norm_min_max {

    use super::*;
    
    pub trait NormMinMaxConvert<T> {

        fn norm_min_max_convert_to(&self, dst : &mut T);
        
    }

    impl<S, T> NormMinMaxConvert<Image<f32, T>> for Image<u8, S> 
    where
        S : Storage<u8>,
        T : StorageMut<f32>,
    {

        fn norm_min_max_convert_to(&self, dst : &mut Image<f32, T>) {
            norm_min_max_convert_to(self, dst);
        }
        
    }
    
    impl<S, T> NormMinMaxConvert<Image<u8, T>> for Image<f32, S> 
    where
        S : Storage<f32>,
        T : StorageMut<u8>,
    {

        fn norm_min_max_convert_to(&self, dst : &mut Image<u8, T>) {
            norm_min_max_convert_to(self, dst);
        }
        
    }
    
    fn norm_min_max_convert_to<P, Q, S, T>(a : &Image<P, S>, b : &mut Image<Q, T>) 
    where
        P : Pixel + ToPrimitive + PartialOrd +  AsPrimitive<Q> + Bounded,
        Q : Pixel + ToPrimitive + PartialOrd +  AsPrimitive<P> + Div<Output=Q> + Sub<Output=Q>,
        S : Storage<P>,
        T : StorageMut<Q>,
        u8 : AsPrimitive<Q> + AsPrimitive<P>,
        f32 : AsPrimitive<Q> + AsPrimitive<P>,
        i16: num_traits::AsPrimitive<P>,
        u16: num_traits::AsPrimitive<P>
    {
        assert!(a.same_size(b));
        #[cfg(feature="ipp")]
        unsafe { 
            if ippi_convert::<P, Q, S, T>(a, b, Conversion::NormalizeMinMax) {
                return;
            }
        }
        let (min, max) = a.min_max();
        let min : Q = min.as_();
        let max : Q = max.as_();
        b.pixels_mut(1).zip(a.pixels(1)).for_each(|(dst, src)| *dst = (src.as_() - min) / max );
    }
    
}

pub use base::*;

pub use abs::*;

pub use mul::*;

pub use div::*;

pub use abs_mul::*;

pub use abs_div::*;

pub use norm_max::*;

pub use norm_min_max::*;

/*/* Signed to float conversions */
impl<S> Image<i32, S>
where
    S : Storage<i32>,
{

    pub fn convert_to<Q, T>(&self, dst : &mut Image<f32, T>) 
    where
        i32 : AsPrimitive<f32>,
        f32 : AsPrimitive<i32>,
        T : StorageMut<f32>
    {
        convert_to(self, dst)
    }
}

/* Float to signed conversions */
impl<S> Image<f32, S>
where
    S : Storage<f32>,
{

    pub fn convert_to<Q, T>(&self, dst : &mut Image<Q, T>) 
    where
        f32 : AsPrimitive<i32>,
        i32 : AsPrimitive<f32>,
        T : StorageMut<i32>
    {
        convert_to(self, dst)
    }
    
}*/

/*
convert: signed integer <-> float. Saturates and truncates if necessary.
abs_convert: signed integer -> unsigned integer. Saturates if necessary.
mul_convert: float -> integer. Maps -1.0 to 1.0 to a signed integer domain and 0.0 to 1.0 to an unsigned integer domain, saturating if necessary.
abs_mul_convert: float -> integer. Maps -1.0 to 1.0 to a signed or unsigned integer domain on [0..IMAX], always taking the absolute value of the FP first.
div_convert: integer -> float. Signed integers map to [-1.0, 1.0] and unsigned integers to [0.0, 1.0]
abs_div_convert: integer -> float. Both signed and unsigned integers map to [0.0, 1.0], always taking the absolute value of signed integers first.
norm_max_convert: unsigned integer -> float. Maps [0..MAX] to [0..1], where the maximum value of the image is used to define the scale.
norm_min_max_convert: signed or unsigned integer -> float. Maps[IMIN..IMAX] to [-1.0..1.0] where the minimum and maximum values of the image are used to define the scale.
norm_sum_convert: unsigned interger -> float. Maps unsigned [0..MAX] to [0..1], where the sum of the values of the image is used to define the scale.
abs_norm_max_convert signed integer -> float. Same as the unsigned version, but takes absolute values first.
abs_norm_sum_convert: signed integer -> float. Same as unsigned version, but takes absolute values first.
*/

/*pub fn into_normalized_owned(w : &Window<f32>) -> ImageBuf<u8> {
    let mut n = w.clone_owned();
    n.full_window_mut().abs_mut();
    normalize_max_inplace(n.as_mut());
    n.full_window_mut().scalar_mul(255.0);
    let dst : ImageBuf<u8> = n.convert(Conversion::Preserve);
    dst
}

// TODO absolute value convert -> call convert(abs(M)); then only the
// bit depth must be taken into account.

impl From<ImageBuf<u8>> for ImageBuf<f32> {

    fn from(img : ImageBuf<u8>) -> ImageBuf<f32> {
        #[cfg(feature="opencv")]
        {
            use opencv::prelude::MatTrait;
            let mut out = Image::<f32>::new_constant(img.height(), img.width(), 0.);
            let m : opencv::core::Mat = (&img).into();
            m.convert_to(&mut out, opencv::core::CV_32F, 1.0, 0.0);
            return out;
        }
        unimplemented!()
    }

}

impl TryFrom<ImageBuf<f32>> for ImageBuf<u8> {

    type Error = ();

    fn try_from(img : ImageBuf<f32>) -> Result<ImageBuf<u8>, ()> {
        unimplemented!()
    }

}

impl FromStr for Conversion {

    type Err = ();

    fn from_str(s : &str) -> Result<Self, ()> {
        match s {
            "preserve" => Ok(Self::Preserve),
            "stretch" => Ok(Self::Stretch),
            "shrink" => Ok(Self::Shrink),
            _ => Err(())
        }
    }

}

fn baseline_conversion<'a, 'b, N, M>(to : &'a mut WindowMut<'a, N>, from : &'b Window<'b, M>)
where
    N : Scalar + Default + Debug + Clone + Copy + num_traits::Zero + Default,
    M : Scalar + Default + num_traits::cast::AsPrimitive<N> + Debug + Clone + Copy
{

    use num_traits::cast::AsPrimitive;

    // TODO do saturarting version from_px.min(DST_MAX).max(DST_MIN)
    to.pixels_mut(1).zip(from.pixels(1)).for_each(move |(to_px, from_px)| *to_px = from_px.as_() );
}

/// Multiplies from by scale factor M then converts the result to N.
fn baseline_scale_then_convert<'a, 'b, N, M>(to : &'a mut WindowMut<'a, N>, from : &'b Window<'b, M>, scale : M)
where
    N : Scalar + Default + Debug + Clone + Copy + num_traits::Zero + Default,
    M : Scalar + Default + num_traits::cast::AsPrimitive<N> + Debug + Clone + Copy + Mul<Output=M>
{

    use num_traits::cast::AsPrimitive;

    to.pixels_mut(1).zip(from.pixels(1)).for_each(move |(to_px, from_px)| *to_px = (*from_px * scale).as_() );
}

/// Multiplies from by scale factor M then converts the result to N.
fn baseline_scale_by_float_then_convert<'a, 'b, N, M>(to : &'a mut WindowMut<'a, N>, from : &'b Window<'b, M>, scale : f64)
where
    N : Scalar + Default + Debug + Clone + Copy + num_traits::Zero + Default,
    M : Scalar + Default + num_traits::cast::AsPrimitive<N> + Debug + Clone + Copy + Mul<Output=M> + ToPrimitive,
    f64 : AsPrimitive<N>
{

    use num_traits::cast::AsPrimitive;

    to.pixels_mut(1).zip(from.pixels(1)).for_each(move |(to_px, from_px)| *to_px = (from_px.to_f64().unwrap() * scale).as_() );
}

fn baseline_convert_then_scale<'a, 'b, N, M>(to : &'a mut WindowMut<'a, N>, from : &'b Window<'b, M>, scale : N)
where
    N : Scalar + Default + Debug + Clone + Copy + num_traits::Zero + Default + Mul<Output=N>,
    M : Scalar + Default + num_traits::cast::AsPrimitive<N> + Debug + Clone + Copy
{

    use num_traits::cast::AsPrimitive;

    to.pixels_mut(1).zip(from.pixels(1)).for_each(move |(to_px, from_px)| *to_px = from_px.as_() * scale );
}

/*fn baseline_relative_conversion<N, M>(to : &mut ImageBuf<N>, from : &Window<M>)
where
    N : Scalar + Default + Debug + DeserializeOwned + Serialize + Clone + Copy + num_traits::Zero,
    M : Scalar + Default + num_traits::cast::AsPrimitive<N> + Debug + DeserializeOwned + Serialize + Clone + Copy
{

    use num_traits::cast::AsPrimitive;

    /*to.full_window_mut().pixels_mut(1).zip(from.pixels(1))
        .for_each(|(to_px, from_px)| {
            let mut rel : f32 = from_px.as_();
            rel /= M::max();
            *to_px = rel * N::max();
        });*/
    unimplemented!()
}*/

/*pub fn convert_from_scaling<M>(&mut self, other : &Window<M>, scale : f32, offset : f32)
where
    M : Scalar + Default + num_traits::cast::AsPrimitive<N> +
{
    // IppStatus ippiScaleC_<mod>_C1R(const Ipp<srcDatatype>* pSrc, int srcStep, Ipp64f mVal,
    // Ipp64f aVal, Ipp<dstDatatype>* pDst, int dstStep, IppiSize roiSize, IppHintAlgorithm
    // hint);
    unimplemented!()
}

// Elects the highest value of other to be the Self::MAX, such as to avoid overflowing.
pub fn convert_from_shrinking<M>(&'a mut self, other : &Window<'_, M>)
where
    N : Zero,
    M : Scalar + Default + num_traits::cast::AsPrimitive<N> + PartialOrd + Div<Output=M> + Zero
{
    // Alternatively, choose IppHintAlgorithm_ippAlgHintFast IppHintAlgorithm_ippAlgHintAccurate
    // IppStatus ippiScale_16s8u_C1R(const Ipp<srcDatatype>* pSrc, int srcStep, Ipp<dstDatatype>*
    //    pDst, int dstStep, IppiSize roiSize, IppHintAlgorithm_ippAlgHintNone);

    let max = crate::stat::max(other);
    scaled_baseline_conversion(self, other, max);
}

// TODO make this generic like impl AsRef<Window<M>>, and make self carry a field Window corresponding
// to the full window to work as the AsRef implementation, so the user can pass images here as well.
pub fn convert_from_stretching<M>(&mut self, other : &Window<M>)
where
    M : Scalar + Default + num_traits::cast::AsPrimitive<N>
{
    // This maps srcmin..srcmax at the integer scale of src to the integer scale of destination
    // IppStatus ippiScale_8u16s_C1R(const Ipp<srcDatatype>* pSrc, int srcStep, Ipp<dstDatatype>* pDst, int dstStep, IppiSize roiSize);

    // This maps srcmin..srcmax at the integer scale of src to a user-defined floating point scale of destination
    // IppStatus ippiScale_8u32f_C1R(const Ipp8u* pSrc, int srcStep, Ipp32f* pDst, int dstStep, IppiSize roiSize, Ipp32f vMin, Ipp32f vMax);

    // For a full user-defined scale (saturating), where mval is the coefficient and aval is the offset:
    // IppStatus ippiScaleC_<mod>_C1R(const Ipp<srcDatatype>* pSrc, int srcStep, Ipp64f mVal,
    // Ipp64f aVal, Ipp<dstDatatype>* pDst, int dstStep, IppiSize roiSize, IppHintAlgorithm
    // hint);

    unimplemented!()
}*/

pub trait Convert<'a, S, N, M> {

    fn convert_from(&'a mut self, other : S, conv : Conversion);

}

impl<'a, 'b, W, N, M> Convert<'a, W, N, M> for WindowMut<'a, N>
where
    W : Borrow<Window<'b, M>>,
    N : Zero + Copy + Scalar + Default + Bounded + AsPrimitive<M> + Mul<Output=N> + Div<Output=N> + One + Any + ToPrimitive + PartialOrd,
    u8 : AsPrimitive<M>,
    M : Scalar + Default + num_traits::cast::AsPrimitive<N> + Zero + Bounded + Mul<Output=M> + Div<Output=M> + One + Any + ToPrimitive + PartialOrd,
    f64 : AsPrimitive<N>
{

    // TODO make this generic like impl AsRef<Window<M>>, and make self carry a field Window corresponding
    // to the full window to work as the AsRef implementation, so the user can pass images here as well.
    // This converts the pixel values without any type of scaling.
    fn convert_from(&'a mut self, other : W, conv : Conversion) {

        assert!(self.size() == other.size());

        #[cfg(feature="ipp")]
        unsafe {
            ippi_convert(other.borrow(), self, conv);
            return;
        }

        match conv {
            Conversion::Stretch => {
                println!("stretching conversion");
                let this_max : M = N::max_value().as_();
                baseline_scale_then_convert(self, other.borrow(), this_max);
            },
            Conversion::Shrink => {
                println!("shrinking conversion");
                let other_max : N = M::max_value().as_();
                let other_max_inv = N::one() / other_max;
                baseline_convert_then_scale(self, other.borrow(), other_max_inv);
            },
            Conversion::Preserve => {

                // TODO account for unequal strides.
                /*#[cfg(feature="ipp")]
                {
                    assert!(other.is_full());
                    unsafe { ipputils::convert(other.win, &mut self.win[..], ncols); }
                    return;
                }*/

                #[cfg(feature="opencv")]
                unsafe {
                    let ncols = self.original_size().1;
                    crate::image::cvutils::convert(
                        other.borrow().full_slice(),
                        unsafe { self.original_slice() },
                        other.original_size().1,
                        Some((*other.offset(), *other.size())),
                        ncols,
                        None
                    );
                    return;
                }

                baseline_conversion(self, other.borrow());
            },
            Conversion::Normalize => {
                // Multiplying by 'by' leaves at [0,1]. Then multiply again by maximum of destination domain.
                // Equals (px/max)*dst_domain
                let img_max = crate::stat::max(other.borrow()).to_f64().unwrap();
                let by = (1. / img_max) * N::max_value().to_f64().unwrap();
                baseline_scale_by_float_then_convert(self, other.borrow(), by);
            }
        }
    }

}

impl<'a, 'b, W, N, M> Convert<'a, W, N, M> for ImageBuf<N>
where
    ImageBuf<N> : AsMut<WindowMut<'a, N>>,
    W : Borrow<Window<'b, M>> + Raster,
    N : Zero + Copy + Scalar + Default + Bounded + AsPrimitive<M> + Mul<Output=N> + Div<Output=N> + One + Any + ToPrimitive + PartialOrd,
    u8 : AsPrimitive<M>,
    M : Scalar + Default + AsPrimitive<N> + Zero + Bounded + Mul<Output=M> + Div<Output=M> + One + Any + ToPrimitive + PartialOrd,
    f64 : AsPrimitive<N>
{

    fn convert_from(&'a mut self, other : W, conv : Conversion) {
        self.as_mut().convert_from(other, conv);
    }

}

pub trait ConvertInto<'a, N>
where
    N : Copy + Scalar + Clone + Debug
{

    fn convert_to(&self, conv : Conversion, other : &mut WindowMut<'a, N>);

    fn convert(&self, conv : Conversion) -> ImageBuf<N>;

}

impl<'a, M, N> ConvertInto<'a, N> for Window<'a, M>
where

    WindowMut<'a, N> : Convert<'a, Window<'a, M>, N, M>,
    N : Zero + Copy + Scalar + Default + Bounded + AsPrimitive<M> + Mul<Output=N> + Div<Output=N> + One + Any + ToPrimitive + PartialOrd,
    u8 : AsPrimitive<M>,
    M : Scalar + Default + num_traits::cast::AsPrimitive<N> + Zero + Bounded + Mul<Output=M> + Div<Output=M> + One + Any + ToPrimitive + PartialOrd,
{

    fn convert_to(&self, conv : Conversion, other : &mut WindowMut<'a, N>) {
        unsafe { mem::transmute::<_, &'a mut WindowMut<'a, N>>(other).convert_from(self.clone(), conv); }
    }

    fn convert(&self, conv : Conversion) -> ImageBuf<N> {
        let mut out = unsafe { Image::<N>::new_empty(self.height(), self.width()) };
        unsafe { self.convert_to(conv, mem::transmute::<_, &'a mut WindowMut<'a, N>>(&mut out.full_window_mut())) };
        out
    }

}

impl<'a, M, N> ConvertInto<'a, N> for ImageBuf<M>
where
    WindowMut<'a, N> : Convert<'a, Window<'a, M>, N, M>,
    N : Zero + Copy + Scalar + Default + Bounded + AsPrimitive<M> + Mul<Output=N> + Div<Output=N> + One + Any + ToPrimitive + PartialOrd,
    u8 : AsPrimitive<M>,
    M : Scalar + Default + num_traits::cast::AsPrimitive<N> + Zero + Bounded + Mul<Output=M> + Div<Output=M> + One + Any + ToPrimitive + PartialOrd,
{

    fn convert_to(&self, conv : Conversion, other : &mut WindowMut<'a, N>) {
        let other = unsafe {  mem::transmute::<_, &'a mut WindowMut<'a, N>>(other) };
        unsafe { other.convert_from(mem::transmute::<_, Window<'a, M>>(self.full_window()), conv) };
    }

    fn convert(&self, conv : Conversion) -> ImageBuf<N> {
        let mut out = unsafe { Image::<N>::new_empty(self.height(), self.width()) };
        unsafe { self.convert_to(conv, mem::transmute::<_, &'a mut WindowMut<'a, N>>(&mut out.full_window_mut())) };
        out
    }

}*/

// IppiScale: p′ = dst_Min + k*(p - src_Min); where k = (dst_Max - dst_Min)/(src_Max - src_Min)
// For floating point to integer conversions, a range must be specified. For integer data, the
// limits of the datatype itself is used.

// IppiConvert only supports:
// Unsigned to Unsigned of higher depth
// Unsigned or signed to signed of higher depth
// All to float.
// Float to integer
// For other conversions, must use IppiScale.

enum Conversion {

    // Literal conversion, preserving pixel values. Panics if numeric domains are extrapolated.
    Preserve,
    
    Abs,

    // Flatten

    // Takes minimum and maximum over souce pixels, mapping those limits to the destination
    // numeric boundaries. This means at least one pixel at destination image will have
    // value 0. an at least one other pixel will have value N::max.
    // Normalize,

    /// Maps float in [-1, 1] to signed integer domain and float in [0,1] to unsigned integer
    /// domain [0, 1] (and vice-versa).
    Scale,

    // Divides all pixels of source by the maximum value at the image, 
    // producing an image at [0.0,1.0].
    // Then scale this ratio to the destination type maximum.
    NormalizeMax,
    
    NormalizeMinMax


}

// Panics if conversion was attempted but failed, returns false if conversion
// is not implemented in IPP.
#[cfg(feature="ipp")]
unsafe fn ippi_convert<P, Q, S, T>(
    src : &Image<P, S>, 
    dst : &mut Image<Q, T>, 
    conv : Conversion
) -> bool 
where
    P : Pixel + ToPrimitive + PartialOrd,
    Q : Pixel + ToPrimitive + PartialOrd,
    u8 : AsPrimitive<P>,
    u8 : AsPrimitive<Q>,
    S : Storage<P>,
    T : StorageMut<Q>,
    f32 : AsPrimitive<P>,
    i16 : AsPrimitive<P>,
    u16 : AsPrimitive<P>
{

    use std::ffi;
    use crate::image::ipputils;
    use crate::foreign::ipp::ippi::ippiConvert_32f8u_C1R;
    use crate::foreign::ipp::ippi::ippiScale_32f8u_C1R;
    use crate::foreign::ipp::ippi::ippiScale_8u32f_C1R;
    use crate::foreign::ipp::ippi::ippiConvert_8u32f_C1R;
    use crate::foreign::ipp::ippi::ippiConvert_16s8u_C1R;
    use crate::foreign::ipp::ippi::ippiConvert_8u16s_C1R;
    use crate::foreign::ipp::ippi::ippiScale_16s8u_C1R;
    use crate::foreign::ipp::ippi::ippiScale_8u16s_C1R;

    // Foi ROI/Step info, check page 27 of IPPI manual.
    // 8-bit to 16-bit
    // srcStep and dstStep are distances in bytes between starting points of consecutive lines in source
    // and destination images. So actual distance is dist_px * sizeof(data). If using ROIs, passed pointers
    // should refer to the ROI start, NOT image start. roiSize is always in pixels.
    // assert!(src.len() == dst.len());

    assert!(src.shape() == dst.shape());
    // let size = IppiSize{ width : ncol as i32, height : (src.len() / ncol) as i32 };
    let size = ipputils::image_size(src);
    let mut status : Option<i32> = None;

    let src_ptr = src.as_ptr() as *const ffi::c_void;
    let dst_ptr = dst.as_mut_ptr() as *mut ffi::c_void;

    let src_step = ipputils::byte_stride_for_image(src);
    let dst_step = ipputils::byte_stride_for_image(dst);

    if src.pixel_is::<u8>() {
        if dst.pixel_is::<f32>() {
            match conv {
                Conversion::Scale => {
                    let min = 0.0;
                    let max = 1.0;
                    status = Some(ippiScale_8u32f_C1R(
                        src_ptr as *const u8,
                        src_step,
                        dst_ptr as *mut f32,
                        dst_step,
                        size,
                        min,
                        max
                    ));
                },
                Conversion::Preserve => {
                     status = Some(ippiConvert_8u32f_C1R(
                        src_ptr as *const u8,
                        src_step,
                        dst_ptr as *mut f32,
                        dst_step,
                        size
                    ));
                },
                Conversion::NormalizeMax => {
                    let max = src.max().to_f64().unwrap();
                    let scale = (1. / max) as f64;
                    let offset = 0.0;
                    status = Some(crate::foreign::ipp::ippi::ippiScaleC_8u32f_C1R(
                        src_ptr as *const u8,
                        src_step,
                        scale,
                        offset,
                        dst_ptr as *mut f32,
                        dst_step,
                        size,
                        crate::foreign::ipp::ippi::IppHintAlgorithm_ippAlgHintFast
                    ));
                },
                Conversion::NormalizeMinMax => {
                    let (min, max) = src.min_max();
                    let offset = min.to_f64().unwrap();
                    let max = max.to_f64().unwrap();
                    let scale = (1. / max) as f64;
                    status = Some(crate::foreign::ipp::ippi::ippiScaleC_8u32f_C1R(
                        src_ptr as *const u8,
                        src_step,
                        scale,
                        offset,
                        dst_ptr as *mut f32,
                        dst_step,
                        size,
                        crate::foreign::ipp::ippi::IppHintAlgorithm_ippAlgHintFast
                    ));
                },
                _ => {
                    return false;
                }
            }
        } else if dst.pixel_is::<i32>() {
            match conv {
                Conversion::Preserve => {
                    status = Some(crate::foreign::ipp::ippi::ippiConvert_8u32s_C1R(
                        src_ptr as *const u8,
                        src_step,
                        dst_ptr as *mut i32,
                        dst_step,
                        size
                    ));
                },
                _ => {
                    return false;
                }
            }
        } else if dst.pixel_is::<i16>() {
            match conv {
                Conversion::Scale => {
                    status = Some(ippiScale_8u16s_C1R(
                        src_ptr as *const u8,
                        src_step,
                        dst_ptr as *mut i16,
                        dst_step,
                        size
                    ));
                },
                Conversion::Preserve => {
                     status = Some(ippiConvert_8u16s_C1R(
                        src_ptr as *const u8,
                        src_step,
                        dst_ptr as *mut i16,
                        dst_step,
                        size
                    ));
                },
                _ => {
                    return false;
                }
            }
        } else if dst.pixel_is::<u16>() {
            match conv {
                Conversion::Preserve => {
                     status = Some(crate::foreign::ipp::ippi::ippiConvert_8u16u_C1R(
                        src_ptr as *const u8,
                        src_step,
                        dst_ptr as *mut u16,
                        dst_step,
                        size
                    ));
                },
                _ => {
                    return false;
                }
            }
        } else {
            return false;
        }
    }

    if src.pixel_is::<u16>() {
        if dst.pixel_is::<u8>() {
            match conv {
                Conversion::Preserve => {
                     status = Some(crate::foreign::ipp::ippi::ippiConvert_16u8u_C1R(
                        src_ptr as *const u16,
                        src_step,
                        dst_ptr as *mut u8,
                        dst_step,
                        size
                    ));
                },
                _ => { return false; }
            }
        }
    }

    if src.pixel_is::<i16>() {
        if dst.pixel_is::<u8>() {
            match conv {
                Conversion::Scale => {
                    status = Some(ippiScale_16s8u_C1R(
                        src_ptr as *const i16,
                        src_step,
                        dst_ptr as *mut u8,
                        dst_step,
                        size,
                        crate::foreign::ipp::ippi::IppHintAlgorithm_ippAlgHintNone
                    ));
                },
                Conversion::Preserve => {
                     status = Some(ippiConvert_16s8u_C1R(
                        src_ptr as *const i16,
                        src_step,
                        dst_ptr as *mut u8,
                        dst_step,
                        size
                    ));
                },
                _ => {
                    return false;
                }
            }
        } else if dst.pixel_is::<i32>() {
            match conv {
                Conversion::Preserve => {
                    status = Some(crate::foreign::ipp::ippi::ippiConvert_16s32s_C1R(
                        src_ptr as *const i16,
                        src_step,
                        dst_ptr as *mut i32,
                        dst_step,
                        size
                    ));
                },
                _ => {
                    return false;
                }
            }
        } else {
            return false;
        }
    }

    if src.pixel_is::<f32>() {
        if dst.pixel_is::<u8>() {
            match conv {
                Conversion::Scale => {
                    let min = 0.0;
                    let max = 1.0;
                    status = Some(ippiScale_32f8u_C1R(
                        src_ptr as *const f32,
                        src_step,
                        dst_ptr as *mut u8,
                        dst_step,
                        size,
                        min,
                        max
                    ));
                },
                Conversion::Preserve => {
                     status = Some(ippiConvert_32f8u_C1R(
                        src_ptr as *const f32,
                        src_step,
                        dst_ptr as *mut u8,
                        dst_step,
                        size,
                        crate::foreign::ipp::ippcore::IppRoundMode_ippRndNear
                    ));
                },
                Conversion::NormalizeMax => {
                    let max = src.max().to_f64().unwrap();
                    let scale = (1. / max) * u8::max_value() as f64;
                    let offset = 0.0;
                    status = Some(crate::foreign::ipp::ippi::ippiScaleC_32f8u_C1R(
                        src_ptr as *const f32,
                        src_step,
                        scale,
                        offset,
                        dst_ptr as *mut u8,
                        dst_step,
                        size,
                        crate::foreign::ipp::ippi::IppHintAlgorithm_ippAlgHintFast
                    ));
                },
                Conversion::NormalizeMinMax => {
                    let (min, max) = src.min_max();
                    let offset = min.to_f64().unwrap() * u8::max_value() as f64;
                    let max = max.to_f64().unwrap() * u8::max_value() as f64;
                    let scale = (1. / max) as f64;
                    status = Some(crate::foreign::ipp::ippi::ippiScaleC_32f8u_C1R(
                        src_ptr as *const f32,
                        src_step,
                        scale,
                        offset,
                        dst_ptr as *mut u8,
                        dst_step,
                        size,
                        crate::foreign::ipp::ippi::IppHintAlgorithm_ippAlgHintFast
                    ));
                },
                _ => {
                    return false;
                }
            }
        } else if dst.pixel_is::<i32>() {
            match conv {
                Conversion::Preserve => {
                    let offset = 0.0;
                    let scale = 1.0;
                    status = Some(crate::foreign::ipp::ippi::ippiScaleC_32f32s_C1R(
                        src_ptr as *const f32,
                        src_step,
                        scale,
                        offset,
                        dst_ptr as *mut i32,
                        dst_step,
                        size,
                        crate::foreign::ipp::ippi::IppHintAlgorithm_ippAlgHintFast
                    ));
                },
                _ => {
                    return false;
                }
            }
        } else {
            return false;
        }
    }

    if src.pixel_is::<i32>() {
        if dst.pixel_is::<u8>() {
            let offset = 0.0;
            match conv {
                Conversion::NormalizeMax => {
                    let max = src.max().to_f64().unwrap();
                    let scale = (1. / max) * u8::max_value() as f64;
                    status = Some(crate::foreign::ipp::ippi::ippiScaleC_32s8u_C1R(
                        src_ptr as *const i32,
                        src_step,
                        scale,
                        offset,
                        dst_ptr as *mut u8,
                        dst_step,
                        size,
                        crate::foreign::ipp::ippi::IppHintAlgorithm_ippAlgHintFast
                    ));
                },
                Conversion::Preserve => {
                    let scale = u8::max_value() as f64;
                    status = Some(crate::foreign::ipp::ippi::ippiScaleC_32s8u_C1R(
                        src_ptr as *const i32,
                        src_step,
                        scale,
                        offset,
                        dst_ptr as *mut u8,
                        dst_step,
                        size,
                        crate::foreign::ipp::ippi::IppHintAlgorithm_ippAlgHintFast
                    ));
                },
                _ => {
                    return false;
                }
            }
        } else if dst.pixel_is::<i16>() {
            match conv {
                Conversion::Preserve => {
                    let scale = 0;
                    status = Some(crate::foreign::ipp::ippi::ippiConvert_32s16s_C1RSfs(
                        src_ptr as *const i32,
                        src_step,
                        dst_ptr as *mut i16,
                        dst_step,
                        size,
                        crate::foreign::ipp::ippcore::IppRoundMode_ippRndNear,
                        scale
                    ));
                },
                _ => unimplemented!()
            }
        } else if dst.pixel_is::<f32>() {
            status = Some(crate::foreign::ipp::ippi::ippiConvert_32s32f_C1R(
                src_ptr as *const i32,
                src_step,
                dst_ptr as *mut f32,
                dst_step,
                size
            ));
        } else {
            return false;
        }
    }

    if src.pixel_is::<u32>() {
        if dst.pixel_is::<u16>() {
            let scale = 0;
            match conv {
                Conversion::Preserve => {
                    status = Some(crate::foreign::ipp::ippi::ippiConvert_32u16u_C1RSfs(
                        src_ptr as *const u32,
                        src_step,
                        dst_ptr as *mut u16,
                        dst_step,
                        size,
                        crate::foreign::ipp::ippcore::IppRoundMode_ippRndNear,
                        scale
                    ));
                },
                _ => { return false; }
            }
        } else {
            return false;
        }
    }

    match status {
        Some(status) => ipputils::check_status("Conversion", status),
        None => panic!("Invalid conversion type")
    }
    
    true

}

// Conversion between unsigned integer types
// increase_depth
// reduce_depth

// Conversion between signed/unsigned integer types
// increase_depth_abs
// reduce_depth_abs

// Converstion integer <-> float
// ratio_to_scalar (uses random scalar)
// ratio_to_bound (uses integer bounds)
// ratio_to_max (uses maximum at current image)
// ratio_to_min_max (uses minimum and maximum at current image)*/


