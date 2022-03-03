use opencv::imgproc;
use crate::image::*;
use crate::threshold;
use opencv::core;
use opencv::prelude::RotatedRectTrait;

/// Contour extraction from Suzuki (1995)
pub struct Contour {
    bin : Image<u8>,
    thresh : f64,
    max_val : f64,
    out : Vec<(usize, usize)>
}

impl Contour {

    pub fn new(dims : (usize, usize), thresh : f64, max_val : f64) -> Self {
        Self { bin : Image::new_constant(dims.0, dims.1, 0), thresh, max_val, out : Vec::new() }
    }

    pub fn set_threshold(&mut self, threshold : f64) {
        self.thresh = threshold;
    }

    pub fn get_threshold(&self) -> f64 {
        self.thresh
    }

    pub fn find(&mut self, img : &Window<u8>) -> Option<&[(usize, usize)]> {
        assert!(img.height() == self.bin.height() && img.width() == self.bin.width());

        // Binarize image
        // threshold::threshold_window(&img, &mut self.bin, self.thresh, self.max_val, true);

        // Extract contours
        let mut out_vec : core::Vector<core::Point2i> = core::Vector::new();
        let bin : core::Mat = self.bin.full_window().into();
        let ans = imgproc::find_contours(
            &bin,
            &mut out_vec,
            imgproc::RETR_EXTERNAL,
            imgproc::CHAIN_APPROX_NONE,
            core::Point2i::new(0, 0)
        );
        if let Err(e) = ans {
            println!("{}", e);
            return None;
        }

        // Collect contour points
        self.out.clear();
        if out_vec.len() == 0 {
            return None;
        }
        for i in 0..out_vec.len() {
            if let Ok(pt) = out_vec.get(i) {
                self.out.push((pt.y as usize, pt.x as usize));
            } else {
                println!("Invalid vector indexing");
            }
        }

        Some(&self.out[..])
    }

}
