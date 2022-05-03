use crate::image::*;
use std::convert::{AsRef, AsMut, TryFrom};
use num_traits::Zero;
use nalgebra::Scalar;
use std::ops::Div;
use std::fmt::Debug;
use crate::raster::*;
use std::str::FromStr;

impl From<Image<u8>> for Image<f32> {

    fn from(img : Image<u8>) -> Image<f32> {
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

impl TryFrom<Image<f32>> for Image<u8> {

    type Error = ();

    fn try_from(img : Image<f32>) -> Result<Image<u8>, ()> {
        unimplemented!()
    }

}

pub enum Conversion {

    // Literal conversion, preserving pixel values. Panics if numeric domains are extrapolated.
    Preserve,

    // Maps ratios of pixels to scalar numeric boundaries (e.g. u8::MIN and u8::MAX) to the destination
    // numeric boundaries. Negative values are calculated relative to negative boundaries; Positive
    // values are calculated relative to positive boundaries.
    Flatten,

    // Takes minimum and maximum over souce pixels, mapping those limits to the destination
    // numeric boundaries. This means at least one pixel at destination image will have
    // value 0. an at least one other pixel will have value N::max.
    Stretch,

    // Convert with custom user-defined scaling factor and offset factor.
    Linearize { offset : f32, scale : f32 }

}

impl FromStr for Conversion {

    type Err = ();

    fn from_str(s : &str) -> Result<Self, ()> {
        match s {
            "preserve" => Ok(Self::Preserve),
            "flatten" => Ok(Self::Flatten),
            "stretch" => Ok(Self::Stretch),
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

    to.pixels_mut(1).zip(from.pixels(1)).for_each(move |(to_px, from_px)| *to_px = from_px.as_() );
}

fn scaled_baseline_conversion<'a, 'b, N, M>(to : &'a mut WindowMut<'a, N>, from : &'b Window<'b, M>, scale : M)
where
    N : Scalar + Default + Debug + Clone + Copy + num_traits::Zero + Default,
    M : Scalar + Default + num_traits::cast::AsPrimitive<N> + Debug + Clone + Copy + Div<Output=M>
{

    use num_traits::cast::AsPrimitive;

    to.pixels_mut(1).zip(from.pixels(1)).for_each(move |(to_px, from_px)| *to_px = (*from_px / scale).as_() );
}

/*fn baseline_relative_conversion<N, M>(to : &mut Image<N>, from : &Window<M>)
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

    let max = crate::global::max(other);
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
    W : AsRef<Window<'b, M>>,
    N : Zero + Copy + Scalar + Default,
    M : Scalar + Default + num_traits::cast::AsPrimitive<N> + Zero
{

    // TODO make this generic like impl AsRef<Window<M>>, and make self carry a field Window corresponding
    // to the full window to work as the AsRef implementation, so the user can pass images here as well.
    // This converts the pixel values without any type of scaling.
    fn convert_from(&'a mut self, other : W, conv : Conversion) {
        let ncols = self.original_size().1;

        // TODO account for unequal strides.
        /*#[cfg(feature="ipp")]
        {
            assert!(other.is_full());
            unsafe { ipputils::convert(other.win, &mut self.win[..], ncols); }
            return;
        }*/

        #[cfg(feature="opencv")]
        unsafe {
            crate::image::cvutils::convert(
                other.as_ref().win,
                unsafe { self.original_slice() },
                other.original_size().1,
                Some((*other.offset(), *other.size())),
                ncols,
                None
            );
            return;
        }

        baseline_conversion(self, other.as_ref());
    }

}

impl<'a, 'b, W, N, M> Convert<'a, W, N, M> for Image<N>
where
    Image<N> : AsMut<WindowMut<'a, N>>,
    W : AsRef<Window<'b, M>>,
    N : Zero + Copy + Scalar + Default,
    M : Scalar + Default + num_traits::cast::AsPrimitive<N> + Zero
{

    fn convert_from(&'a mut self, other : W, conv : Conversion) {
        self.as_mut().convert_from(&other, conv);
    }

}

