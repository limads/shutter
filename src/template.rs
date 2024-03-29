use std::collections::HashMap;
use crate::image::*;
use nalgebra::*;
use std::fmt::Debug;
use std::any::Any;
use num_traits::Zero;
use std::iter::FromIterator;

#[cfg(feature="ipp")]
pub fn abs_diff_template(src : &Window<'_, u8>, templ : &Window<'_, u8>, mut dst : WindowMut<'_, i32>) {

    assert!(dst.height() == src.height() - templ.height() + 1);
    assert!(dst.width() == src.width() - templ.width() + 1);

    let (src_byte_stride, src_roi) = crate::image::ipputils::step_and_size_for_window(src);
    let (templ_byte_stride, templ_roi) = crate::image::ipputils::step_and_size_for_window(templ);
    let (dst_byte_stride, dst_roi) = crate::image::ipputils::step_and_size_for_window_mut(&dst);

    unsafe {
        let mut buf_sz : i32 = 0;
        let n_channels = 1;
        let roi_ty = crate::foreign::ipp::ippi::IppiROIShape_ippiROIValid;
        let ans = crate::foreign::ipp::ippi::ippiSADGetBufferSize(
            src_roi,
            templ_roi,
            crate::foreign::ipp::ippi::IppDataType_ipp8u,
            n_channels,
            roi_ty,
            &mut buf_sz as *mut _
        );
        assert!(ans == 0, "Error: {}", ans);
        // assert!(buf_sz > 0);

        // The scale factor takes the floating point calc result and multiplies by 2^(-scale_factor)
        // (i.e. divide result by 2) so that the result is not saturated when converted back to integers.
        // sf = 1 means the actual reasult will be result*2.
        // "The integer square root operation ippiSqr (without scaling) for the input value 2 gives the result equal to 1 instead of 1.414.
        // Scaling of the internally computed output value with the factor scaleFactor = -3 gives the result 11,
        // and permits the more precise value to be restored as 11*2^-3 = 1.375."
        let scale_factor = 0;

        let mut buf = Vec::from_iter((0..buf_sz).map(|_| 0u8 ));
        let ans = crate::foreign::ipp::ippi::ippiSAD_8u32s_C1RSfs(
            src.as_ptr(),
            src_byte_stride,
            src_roi,
            templ.as_ptr(),
            templ_byte_stride,
            templ_roi,
            dst.as_mut_ptr(),
            dst_byte_stride,
            roi_ty,
            scale_factor,
            buf.as_mut_ptr()
        );
        assert!(ans == 0);
    }
}

pub struct TemplateSearch<T>
where
    T : Scalar + Debug + Copy + Default + Any
{
    map : Image<f32>,
    template : Image<T>,
    region_sz : (usize, usize)
}

impl<T> TemplateSearch<T>
where
    T : Scalar + Debug + Copy + Default + Zero + Any
{

    /// Returns local maxima over regions with the given size.
    pub fn new(sz : (usize, usize), region_sz : (usize, usize), template : Image<T>) -> Self {
        Self { map : Image::new_constant(sz.0, sz.1, 0.0), region_sz, template }
    }

    pub fn set_template(&mut self, template : Image<T>) {
        self.template = template;
    }

    #[cfg(feature="opencv")]
    pub fn search_local(&mut self, src : &Image<T>) -> (usize, usize) {
        self.calculate_match(src);
        search_maximum(&self.map.full_window())
    }

    fn calculate_match(&mut self, src : &Image<T>) {
        let map_width = self.map.width();
        let template_width = self.template.width();

        #[cfg(feature="opencv")]
        unsafe {
            template_match(
                src.as_ref(),
                self.map.as_mut(),
                map_width,
                self.template.as_mut(),
                template_width
            );
            return;
        }
        unimplemented!()
    }

    #[cfg(feature="opencv")]
    /// Returns (region index, maximum) collection
    pub fn search_global(&mut self, src : &Image<T>) -> HashMap<(usize, usize), (usize, usize)> {
        self.calculate_match(src);
        let mut maxima = HashMap::new();
        for i in 0..(self.map.height() / self.region_sz.0) {
            for j in 0..(self.map.width() / self.region_sz.1) {
                let calc_this = maxima.get(&(i,j)).is_none();
                if calc_this {
                    let offset = (i*self.region_sz.0, j*self.region_sz.1);
                    let max_ix = search_maximum(&self.map.window(offset, self.region_sz).unwrap());
                    maxima.insert((i, j), max_ix);
                }
            }
        }
        maxima
    }

}

#[cfg(feature="opencv")]
unsafe fn template_match<T>(src : &[T], result : &mut [f32], src_ncol : usize, template : &mut [T], template_ncol : usize)
where
    T : Scalar + Debug + Copy + Default
{

    use opencv::{core, imgproc};
    use crate::image::cvutils::slice_to_mat;

    let any_t = (&src[0] as &dyn Any);
    assert!(any_t.is::<u8>() || any_t.is::<f32>());
    assert!(src.len() == result.len());
    let mut src = slice_to_mat(src, src_ncol, None);
    let mut template = slice_to_mat(template, template_ncol, None);
    let mut dst = slice_to_mat(result, src_ncol, None);

    // Write match correlation map
    imgproc::match_template(
        &mut src,
        &mut template,
        &mut dst,
        imgproc::TM_CCORR_NORMED, 
        &core::no_array().unwrap()
    ).unwrap();
}

#[cfg(feature="opencv")]
fn search_minimum<T>(win : &Window<T>) -> (usize, usize)
where
    T : Scalar + Debug + Copy + Copy + Default
{
    search_minimum_maximum(win).1
}

#[cfg(feature="opencv")]
fn search_maximum<T>(win : &Window<T>) -> (usize, usize)
where
    T : Scalar + Debug + Copy + Copy + Default
{
    search_minimum_maximum(win).0
}

#[cfg(feature="opencv")]
fn search_minimum_maximum<T>(win : &Window<T>) -> ((usize, usize), (usize, usize))
where
    T : Scalar + Debug + Copy + Default
{

    use opencv::{core, imgproc};

    let mut min_loc = core::Point2i { x : 0, y : 0 };
    let mut max_loc = core::Point2i { x : 0, y : 0 };
    let mut min_val = 0.0;
    let mut max_val = 0.0;
    
    let mut m : core::Mat = win.clone().into();

    // Then find local maxima and minima (If used TM_CCORR, get local maximum; for TM_CCOEFF use minimum).
    core::min_max_loc(
        &mut m,
        &mut min_val,
        &mut max_val,
        &mut min_loc,
        &mut max_loc,
        &core::no_array().unwrap()
    ).unwrap();

    ((min_loc.y as usize, min_loc.x as usize), (max_loc.y as usize, max_loc.x as usize))
}




