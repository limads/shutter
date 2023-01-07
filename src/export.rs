use crate::image::*;

/*#[no_mangle]
extern "C" fn open(path : &str, dst : &mut [u8]) -> i64 {
    match crate::io::decode_from_file(path) {
        Ok(buf) => {
            match ImageMut::from_slice(dst, buf.width() as usize) {
                Some(mut dst) => {
                    dst.copy_from(&buf);
                    0
                },
                None => {
                    -1
                }
            }
        },
        Err(_) => {
            -1
        }
    }
}

#[no_mangle]
extern "C" fn show(buf : &[u8], width : i64) -> i64 {
    if let Some(img) = ImageRef::from_slice(buf, width as usize) {
        img.clone_owned().show();
        0
    } else {
        -1
    }
}*/

