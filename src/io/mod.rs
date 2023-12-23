use crate::image::Window;
use base64;
use crate::image::{Image, WindowMut};
use crate::image::ImageBuf;

mod png;

mod pgm;

pub use png::*;

pub use pgm::*;

pub struct ImageGrid {

    img : ImageBuf<u8>,

    nrow : usize,

    ncol : usize,

    sub_shape : (usize, usize)

}

impl ImageGrid {

    pub fn image_mut(&mut self) -> &mut ImageBuf<u8> {
        &mut self.img
    }

    pub fn image(&self) -> &ImageBuf<u8> {
        &self.img
    }

    pub fn from_paths(paths : &[impl AsRef<str>], width : usize) -> Result<Self, &'static str> {
        if paths.len() % width != 0 {
            return Err("Invalid grid width");
        }
        let mut imgs = Vec::new();
        let mut sub_shape = (0, 0);
        for path in paths {
            let new = decode_from_file(path.as_ref())?;
            sub_shape = new.shape();
            imgs.push(new);
        }
        let mut rows = Vec::new();
        for set in imgs.chunks(width) {
            let set_wins : Vec<_> = set.iter().map(|w| w.full_window() ).collect();
            rows.push(Image::concatenate(&set_wins).unwrap());
        }
        let set_cols : Vec<_> = rows.iter().map(|w| w.full_window() ).collect();
        Ok(ImageGrid {
            img : ImageBuf::stack(&set_cols).unwrap(),
            nrow : paths.len() / width,
            ncol : width,
            sub_shape
        })
    }

    pub fn windows(&self) -> impl Iterator<Item=Window<'_, u8>> {
        self.img.windows(self.sub_shape)
    }

    //pub fn windows_mut(&mut self) -> impl Iterator<Item=WindowMut<'_, u8>> {
        // self.img.windows_mut(self.sub_shape)
    //    unimplemented!()
    // }

}

pub fn to_html(win : &Window<'_, u8>) -> String {
    let img = win.clone_owned();
    let png = encode(&img).unwrap();
    format!("<img src='data:image/png;base64,{}' />", base64::encode(png))
}


