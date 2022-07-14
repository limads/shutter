use crate::image::*;
use std::convert::{AsRef, AsMut, TryFrom};
use num_traits::Zero;
use nalgebra::Scalar;
use std::ops::Div;
use std::fmt::Debug;
use crate::raster::*;
use std::str::FromStr;
use std::borrow::Borrow;
use std::ops::Mul;
use num_traits::Bounded;
use num_traits::cast::{AsPrimitive, ToPrimitive};
use num_traits::ops::inv::Inv;
use num_traits::One;
use std::any::Any;
use std::mem;

// TODO absolute value convert -> call convert(abs(M)); then only the
// bit depth must be taken into account.

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

    // Flatten

    // Takes minimum and maximum over souce pixels, mapping those limits to the destination
    // numeric boundaries. This means at least one pixel at destination image will have
    // value 0. an at least one other pixel will have value N::max.
    // Normalize,

    /// Inverse of shrink, using ratios on (-1, 1) to map to numeric boundaries of the destination.
    Stretch,

    // Maps ratios of pixels to scalar numeric boundaries of the source (e.g. u8::MIN and u8::MAX) to the destination
    // numeric boundaries. Negative values are calculated relative to negative boundaries; Positive
    // values are calculated relative to positive boundaries.
    Shrink,

    // Divides all pixels of source by the maximum value at the image, producing an image at [0.0,1.0].
    // Then scale this ratio to the destination type maximum.
    Normalize

    // Convert with custom user-defined scaling factor and offset factor.
    // Linearize { offset : f32, scale : f32 }

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
    W : Borrow<Window<'b, M>> + Raster,
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
                let img_max = crate::global::max(other.borrow()).to_f64().unwrap();
                let by = (1. / img_max) * N::max_value().to_f64().unwrap();
                baseline_scale_by_float_then_convert(self, other.borrow(), by);
            }
        }
    }

}

impl<'a, 'b, W, N, M> Convert<'a, W, N, M> for Image<N>
where
    Image<N> : AsMut<WindowMut<'a, N>>,
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

    fn convert_into(&self, conv : Conversion, other : &mut WindowMut<'a, N>);

    fn convert_owned(&self, conv : Conversion) -> Image<N>;

}

impl<'a, M, N> ConvertInto<'a, N> for Window<'a, M>
where

    WindowMut<'a, N> : Convert<'a, Window<'a, M>, N, M>,
    N : Zero + Copy + Scalar + Default + Bounded + AsPrimitive<M> + Mul<Output=N> + Div<Output=N> + One + Any + ToPrimitive + PartialOrd,
    u8 : AsPrimitive<M>,
    M : Scalar + Default + num_traits::cast::AsPrimitive<N> + Zero + Bounded + Mul<Output=M> + Div<Output=M> + One + Any + ToPrimitive + PartialOrd,
{

    fn convert_into(&self, conv : Conversion, other : &mut WindowMut<'a, N>) {
        unsafe { mem::transmute::<_, &'a mut WindowMut<'a, N>>(other).convert_from(self.clone(), conv); }
    }

    fn convert_owned(&self, conv : Conversion) -> Image<N> {
        let mut out = Image::<N>::new_constant(self.height(), self.width(), N::zero());
        unsafe { self.convert_into(conv, mem::transmute::<_, &'a mut WindowMut<'a, N>>(&mut out.full_window_mut())) };
        out
    }

}

impl<'a, M, N> ConvertInto<'a, N> for Image<M>
where
    WindowMut<'a, N> : Convert<'a, Window<'a, M>, N, M>,
    N : Zero + Copy + Scalar + Default + Bounded + AsPrimitive<M> + Mul<Output=N> + Div<Output=N> + One + Any + ToPrimitive + PartialOrd,
    u8 : AsPrimitive<M>,
    M : Scalar + Default + num_traits::cast::AsPrimitive<N> + Zero + Bounded + Mul<Output=M> + Div<Output=M> + One + Any + ToPrimitive + PartialOrd,
{

    fn convert_into(&self, conv : Conversion, other : &mut WindowMut<'a, N>) {
        let other = unsafe {  mem::transmute::<_, &'a mut WindowMut<'a, N>>(other) };
        unsafe { other.convert_from(mem::transmute::<_, Window<'a, M>>(self.full_window()), conv) };
    }

    fn convert_owned(&self, conv : Conversion) -> Image<N> {
        let mut out = Image::<N>::new_constant(self.height(), self.width(), N::zero());
        unsafe { self.convert_into(conv, mem::transmute::<_, &'a mut WindowMut<'a, N>>(&mut out.full_window_mut())) };
        out
    }

}

// IppiScale: p′ = dst_Min + k*(p - src_Min); where k = (dst_Max - dst_Min)/(src_Max - src_Min)
// For floating point to integer conversions, a range must be specified. For integer data, the
// limits of the datatype itself is used.

#[cfg(feature="ipp")]
pub unsafe fn ippi_convert<S,T>(src : &Window<'_, S>, dst : &mut WindowMut<'_, T>, conv : Conversion)
where
    S : Scalar + Any + Copy + ToPrimitive + PartialOrd,
    T : Scalar + Copy + Default + Any + ToPrimitive + PartialOrd,
    u8 : AsPrimitive<S>
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
    let size = ipputils::window_size(src);
    let mut status : Option<i32> = None;

    let src_ptr = src.as_ptr() as *const ffi::c_void;
    let dst_ptr = dst.as_mut_ptr() as *mut ffi::c_void;

    let src_step = ipputils::byte_stride_for_window(&src);
    let dst_step = ipputils::byte_stride_for_window_mut(&dst);

    // u8->f32
    if src.pixel_is::<u8>() && dst.pixel_is::<f32>() {
        match conv {
            Conversion::Shrink => {
                status = Some(ippiScale_8u32f_C1R(
                    src_ptr as *const u8,
                    src_step,
                    dst_ptr as *mut f32,
                    dst_step,
                    size,
                    0.0,
                    1.0
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
            _ => panic!("Invalid conversion")
        }
    }

    // u8->i16
    if src.pixel_is::<u8>() && dst.pixel_is::<i16>() {
        match conv {
            Conversion::Stretch => {
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
            _ => panic!("Invalid conversion")
        }
    }

    // i16->u8
    if src.pixel_is::<i16>() && dst.pixel_is::<u8>() {
        match conv {
            Conversion::Shrink => {
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
            _ => panic!("Invalid conversion")
        }
    }

    // f32->u8
    if src.pixel_is::<f32>() && dst.pixel_is::<u8>() {
        match conv {
            Conversion::Stretch => {
                status = Some(ippiScale_32f8u_C1R(
                    src_ptr as *const f32,
                    src_step,
                    dst_ptr as *mut u8,
                    dst_step,
                    size,
                    0.0,
                    1.0
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
            _ => panic!("Invalid conversion")
        }
    }

    // u8 -> i32
    if src.pixel_is::<u8>() && dst.pixel_is::<i32>() {
        unimplemented!()
    }

    // i32 -> u8
    if src.pixel_is::<i32>() && dst.pixel_is::<u8>() {
        match conv {
            Conversion::Normalize => {
                let max = crate::global::max(src).to_f64().unwrap();
                let scale = (1. / max) * u8::max_value() as f64;
                let offset = 0.0;
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
            _ => panic!("Invalid conversion")
        }
    }

    match status {
        Some(status) => ipputils::check_status("Conversion", status),
        None => panic!("Invalid conversion type")
    }

}

