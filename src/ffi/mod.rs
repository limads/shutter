use crate::image::ByteImage;
use std::mem;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::ptr::NonNull;

#[no_mangle]
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
}


