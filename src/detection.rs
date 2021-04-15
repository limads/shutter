use opencv::{core, objdetect::{self, CascadeClassifier}};
use crate::image::{Image, Window, WindowMut};
use opencv::prelude::CascadeClassifierTrait;

pub struct Classifier {
    casc : CascadeClassifier
}

impl Classifier {

    pub fn new(filename : &str) -> Self {
        Self { casc : CascadeClassifier::new(filename).unwrap() }
    }

    /// Return (y, x, nrow, ncol)
    pub fn detect(&mut self, img : WindowMut<u8>) -> Result<Vec <(usize, usize, usize, usize)>, String> {
        let (height, width) = img.shape();
        let mat_img : core::Mat = img.into();
        let scale_factor = 1.5;
        let min_neighbors = 3;
        let min_size = core::Size { width : 92, height : 92 };
        let max_size = core::Size { width : 180, height : 180 };
        let mut objs = core::Vector::new();
        let flags = 0;
        self.casc.detect_multi_scale(
            &mat_img,
            &mut objs,
            scale_factor,
            min_neighbors,
            flags,
            min_size,
            max_size
        ).map_err(|e| format!("{}", e))?;
        let mut out = Vec::new();
        for rect in objs.iter() {
            if rect.y > 0 && rect.x > 0 && rect.x + rect.width < width as i32 && rect.y + rect.height < height as i32 {
                out.push((rect.y as usize, rect.x as usize, rect.width as usize, rect.height as usize));
            }
        }
        Ok(out)
    }
}

