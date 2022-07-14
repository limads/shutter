use crate::image::ByteImage;
use std::mem;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr::NonNull;
use serde_json;
use crate::image::Image;
use crate::image::WindowMut;
use crate::feature::patch::Patch;
use crate::image::*;
use crate::draw::*;
use crate::convert::*;
use std::str::FromStr;
use std::fmt::Debug;
use nalgebra::Scalar;
use num_traits::Zero;
use std::ops::Mul;
use num_traits::Bounded;
use num_traits::AsPrimitive;
use num_traits::ops::inv::Inv;
use num_traits::One;
use std::ops::Div;
use crate::local::*;
use num_traits::{ToPrimitive};

/*#[no_mangle]
pub extern "C" fn convolve_f32(img : &[f32], img_ncol : i64, kernel : &[f32], kernel_ncol : i64, out : &[f32]) -> i64 {
    use crate::local::*;
    0
}*/

#[no_mangle]
pub extern "C" fn peaks(
    hist : &[i64],
    width : i64,
    height : i64,
    left_peaks : &mut [i64],
    valleys : &mut [i64],
    right_peaks : &mut [i64],
    num_found : &mut i64
) -> i64 {

    if 256 % width != 0 {
        return -1;
    }

    let mut ans = crate::hist::peaks_and_valleys(hist, width as usize, height, i64::MAX, 0);
    *num_found = ans.len() as i64;
    for (ix, (left, valley, right)) in ans.drain(..).enumerate() {
        if let (Some(l), Some(v), Some(r)) = (left_peaks.get_mut(ix), valleys.get_mut(ix), right_peaks.get_mut(ix)) {
            *l = left as i64;
            *v = valley as i64;
            *r = right as i64;
        } else {
            return -1;
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn histogram(img : &[u8], width : i64, smooth : i64, hist_dst : &mut [i64]) -> i64 {
    if hist_dst.len() == 256 {
        if let Some(win) = Window::from_slice(img, width as usize) {
            let mut hist = crate::hist::GrayHistogram::calculate(&win);
            if smooth > 0 {
                crate::hist::smoothen(&mut hist, smooth as usize);
            }
            for ix in 0..256 {
                hist_dst[ix] = hist.as_slice()[ix] as i64;
            }
            0
        } else {
            -1
        }
    } else {
        -1
    }
}

static NOT_POSITIVE : i64 = 1;

fn all_positive(s : &[i64]) -> i64 {
    if s.iter().all(|s| *s > 0 ) {
        0
    } else {
        NOT_POSITIVE
    }
}

#[no_mangle]
pub extern "C" fn convolve_f32(
    input : &[f32],
    input_width : i64,
    filter : &[f32],
    filter_width : i64,
    output : &mut [f32],
    mode : &str
) -> i64 {
    if let Some(src) = Window::from_slice(input, input_width as usize) {
        if let Some(kernel) = Window::from_slice(filter, filter_width as usize) {
            if mode == "linear" {
                let (_, ncol) = crate::local::linear_conv_sz(src.shape(), kernel.shape());
                if let Some(mut out) = WindowMut::from_slice(output, ncol) {
                    src.convolve_mut(&kernel, Convolution::Linear, &mut out);
                    return 0;
                }
            } else if mode == "pad" {

            } else if mode == "repeat" {

            } else if mode == "interpolate" {

            }
        }
    }

    -1
}

fn convert_slice_pair<T, U>(input : &[T], width : i64, output : &mut [U], conv : Conversion)
where
    T : Scalar + Copy + Debug + Clone + Zero + Default + Mul<Output=T> + AsPrimitive<U> + std::ops::MulAssign + Bounded + Div<Output=T> + One + PartialOrd + ToPrimitive,
    U : Scalar + Copy + Debug + Clone  + Zero + Default + Bounded + AsPrimitive<T> + Mul<Output=U> + One + Div<Output=U> + PartialOrd + ToPrimitive,
    u8 : AsPrimitive<U>,
    u8 : AsPrimitive<T>,
    f64 : AsPrimitive<U>
{
    let mut dst = WindowMut::from_slice(output, width as usize).unwrap();
    let src = Window::from_slice(input, width as usize).unwrap();
    dst.convert_from(src, conv);
}

#[no_mangle]
pub extern "C" fn convert_u8_f32(input : &[u8], width : i64, output : &mut [f32], mode : &str) -> i64 {
    if let Ok(conv) = Conversion::from_str(mode) {
        convert_slice_pair(input, width, output, conv);
        0
    } else {
        -1
    }
}

#[no_mangle]
pub extern "C" fn convert_f32_u8(input : &[f32], width : i64, output : &mut [u8], mode : &str) -> i64 {
    if let Ok(conv) = Conversion::from_str(mode) {
        convert_slice_pair(input, width, output, conv);
        0
    } else {
        -1
    }
}

#[no_mangle]
pub extern "C" fn encode_rgb(input : &[u8], output : &mut [u8]) -> i64 {

    println!("Input ptr: {:?}; Output ptr: {:?}", input.as_ptr(), output.as_ptr());
    println!("Input length: {}; Output length: {}", input.len(), output.len());

    assert!(input.len() * 3 == output.len());

    if output.len() == 3*input.len() {
        for ix in 0..output.len() {
            output[ix] = input[(ix / 3)];
        }
        0
    } else {
        -1
    }
}

#[no_mangle]
pub extern "C" fn fit_circle(xs : &[f64], ys : &[f64], center : &mut [f64], radius : &mut f64) -> i64 {
    use nalgebra::Vector2;
    let pts : Vec<Vector2<f64>> = xs.iter().zip(ys.iter()).map(|(x, y)| Vector2::new(*x, *y) ).collect();
    match crate::feature::shape::CircleFit::calculate_from_points(&pts[..]) {
        Some(circ) => {
            center[0] = circ.center[0];
            center[1] = circ.center[1];
            *radius = circ.radius;
            0
        },
        None => -1
    }
}

#[no_mangle]
pub extern "C" fn image_draw<'a>(buf : &'a mut [u8], width : i64, mark : &str) -> i64 {
    let mark : Result<crate::draw::Mark, _> = serde_json::from_str(mark);
    if let Ok(mark) = mark {
        if let Some(mut win) = WindowMut::from_slice(buf, width as usize) {
            win.draw(mark);
            0
        } else {
            -1
        }
    } else {
        -1
    }
}

#[no_mangle]
pub extern "C" fn image_size(path : &str, dst : &mut [i64]) -> i64 {
    if let Some(sz) = crate::io::file_dimensions(path) {
        dst[0] = sz.0 as i64;
        dst[1] = sz.1 as i64;
        println!("{:?}", dst);
        0
    } else {
        -1
    }
}

#[no_mangle]
pub extern "C" fn image_open(path : &str, dst : &mut [u8]) -> i64 {
    if let Ok(img) = crate::io::decode_from_file(path) {
        if img.height() * img.width() == dst.len() {
            dst.copy_from_slice(img.as_ref());
            0
        } else {
            -1
        }
    } else {
        -1
    }
}

/*#[no_mangle]
pub unsafe extern "C" fn image_new(rows : usize, cols : usize, intensity : u8) -> *mut ByteImage {
    let img = ByteImage::new_constant(rows, cols, intensity);
    Box::into_raw(Box::new(img))
}*/

/*#[no_mangle]
pub unsafe extern "C" fn image_free(img : *mut ByteImage) {
    if img.is_null() {
        return;
    }
    Box::from_raw(img);
}*/

#[no_mangle]
pub extern "C" fn image_show(buf : &[u8], width : i64) -> i64 {
    let img = Image::new_from_slice(buf, width as usize);
    img.show();
    0
}

#[no_mangle]
pub extern "C" fn segment_all(
    buf : &[u8],
    width : i64,
    spacing : i64,
    margin : i64,
    out : &mut [u8],
    out_len : &mut i64
) -> i64 {
    let win = Window::from_slice(buf, width as usize).unwrap();
    let mut segmenter = crate::feature::patch::raster::RasterSegmenter::new(win.shape(), spacing as usize);
    let patches : Vec<Patch> = segmenter.segment_all(&win, margin as u8, crate::feature::patch::ExpansionMode::Contour).iter().cloned().collect();
    let patches_str = serde_json::to_string(&patches).unwrap();
    if patches_str.len() <= out.len() {
        *out_len = patches_str.len() as i64;
        out[0..patches_str.len()].copy_from_slice(&patches_str.as_bytes());
        0
    } else {
        -1
    }

}

#[no_mangle]
pub extern "C" fn draw_patches<'a>(buf : &mut [u8], width : i64, patches : &str, color : i64) -> i64 {
    let mut win = WindowMut::from_slice(buf, width as usize).unwrap();
    let patches : Vec<Patch> = serde_json::from_str(&patches).unwrap();
    for patch in &patches {
        win.draw_patch_contour(patch, color as u8);
    }
    0
}

/*#[no_mangle]
pub unsafe extern "C" fn image_open(path : *const c_char) -> Option<NonNull<ByteImage>> {
    let mut img = crate::io::decode_from_file(CStr::from_ptr(path).to_str().ok()?).ok()?;
    Some(NonNull::new_unchecked(Box::into_raw(Box::new(img))))
}

#[no_mangle]
pub unsafe extern "C" fn image_new(rows : usize, cols : usize, intensity : u8) -> *mut ByteImage {
    let img = ByteImage::new_constant(rows, cols, intensity);
    Box::into_raw(Box::new(img))
}

#[no_mangle]
pub unsafe extern "C" fn image_free(img : *mut ByteImage) {
    if img.is_null() {
        return;
    }
    Box::from_raw(img);
}

#[no_mangle]
pub unsafe extern "C" fn image_show(img : *const ByteImage) {
    if img.is_null() {
        return;
    }
    if let Some(img) = img.as_ref() {
        img.show();
    }
}*/


