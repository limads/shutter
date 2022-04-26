use crate::image::ByteImage;
use std::mem;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr::NonNull;
use serde_json;
use crate::image::Image;
use crate::image::WindowMut;
use crate::feature::patch::Patch;
use crate::image::Window;

/*#[no_mangle]
pub extern "C" fn convolve_f32(img : &[f32], img_ncol : i64, kernel : &[f32], kernel_ncol : i64, out : &[f32]) -> i64 {
    use crate::local::*;
    0
}*/

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
pub extern "C" fn image_draw(buf : &mut [u8], width : i64, mark : &str) -> i64 {
    let mark : Result<crate::image::Mark, _> = serde_json::from_str(mark);
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
pub extern "C" fn draw_patches(buf : &mut [u8], width : i64, patches : &str, color : i64) -> i64 {
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


