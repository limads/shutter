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
    pub fn detect(&mut self, img : WindowMut<u8>) -> Vec <(usize, usize, usize, usize)> {
        let mut objs = core::Vector::new();
        let img : core::Mat = img.into();
        let scale_factor = 1.5;
        let min_neighbors = 3;
        let min_size = core::Size { width : 30, height : 30 };
        let max_size = core::Size { width : 180, height : 180 };
        let flags = 0;
        self.casc.detect_multi_scale(
            &img,
            &mut objs,
            scale_factor,
            min_neighbors,
            flags,
            min_size,
            max_size
        ).unwrap();
        let mut out = Vec::new();
        for rect in objs.iter() {
            out.push((rect.y as usize, rect.x as usize, rect.width as usize, rect.height as usize));
        }
        out
    }
}

