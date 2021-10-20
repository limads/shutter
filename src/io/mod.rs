use crate::image::Window;
use base64;

mod png;

mod pgm;

pub use png::*;

pub use pgm::*;

pub fn to_html(win : &Window<'_, u8>) -> String {
    let img = win.clone_owned();
    let png = encode(img).unwrap();
    format!("<img src='data:image/png;base64,{}' />", base64::encode(png))
}


