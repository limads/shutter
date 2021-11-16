use nalgebra::*;
use nalgebra::storage::*;
use std::iter::Iterator;
use std::fmt::Display;
use std::convert::TryFrom;
use std::ops::Range;
use opencv::{core, imgproc};
use opencv::prelude::MatTrait;
use opencv::prelude::MatTraitManual;
use std::ffi;
use std::mem;
use std::any::Any;

pub fn get_cv_type<T>() -> i32 
where
    T : Scalar + Default
{
    let t : T = Default::default();
    let any_t = &t as &dyn Any;
    
    if any_t.is::<u8>() {
        core::CV_8UC1
    } else {
        if any_t.is::<i16>() {
            core::CV_16SC1
        } else {
            if any_t.is::<f32>() {
                core::CV_32FC1
            } else {
                if any_t.is::<f64>() {
                    core::CV_64FC1
                } else {
                    panic!("Invalid matrix type");        
                }
            }
        }
    }
}

pub unsafe fn dmatrix_to_mat<T>(d : &DMatrix<T>) -> core::Mat 
where
    T : Scalar + Default
{
    slice_to_mat(d.as_slice(), d.ncols(), None)
}

pub unsafe fn dvector_to_mat<T>(d : &DVector<T>) -> core::Mat 
where
    T : Scalar + Default
{
    slice_to_mat(d.as_slice(), 1, None)
}

/// Converts a Rust slice carrying nalgebra::Scalar elements into an OpenCV mat 
/// (which does not own the data). ncol carries the full
/// image number of columns; offset carries the slice offset; size carries the window size. By
/// converting a slice to Mat, we are unsafely allowing an immutable value to be mutated and aliased.
pub unsafe fn slice_to_mat<T>(
    slice : &[T], 
    stride : usize,
    opt_subslice : Option<((usize, usize), (usize, usize))>,
) -> core::Mat 
where
    T : Scalar + Default
{
    let (nrow, ncol) = if let Some((_, sz)) = opt_subslice {
        sz
    } else {
        (slice.len() / stride, stride)
    };
    
    // Stride size in raw bytes
    let byte_stride = if opt_subslice.is_some() {
        stride * mem::size_of::<T>()
    } else {
        core::Mat_AUTO_STEP
    };
    
    // Trim start and end of the slice IF user informed a subslice
    let effective_slice : &[T] = if let Some((offset, _)) = opt_subslice {
        let fst_ix = stride*offset.0 + offset.1;
        &slice[fst_ix..]
    } else {
        slice
    };
    
    let cv_type = get_cv_type::<T>();
    
    core::Mat::new_rows_cols_with_data(
        nrow as i32, 
        ncol as i32, 
        cv_type, 
        effective_slice.as_ptr() as *mut ffi::c_void,
        byte_stride
    ).unwrap()
}

// TODO for color conversion: core::cvtColor(src, dst, RGB2GRAY)

pub unsafe fn convert<T, U>(
    src : &[T], 
    dst : &mut [U], 
    src_stride : usize, 
    src_opt_subsample : Option<((usize, usize), (usize, usize))>,
    dst_stride : usize,
    dst_opt_subsample : Option<((usize, usize), (usize, usize))>
) 
where
    T : Scalar + Default,
    U : Scalar + Default
{
    let src = slice_to_mat(src, src_stride, src_opt_subsample);
    let mut dst = slice_to_mat(&dst, dst_stride, dst_opt_subsample);
    let scale = 1.0;
    let offset = 0.0;
    src.convert_to(&mut dst, get_cv_type::<U>(), scale, offset).unwrap();
}

pub unsafe fn resize<T>(
    src : &[T], 
    dst : &mut [T], 
    src_stride : usize, 
    src_opt_subsample : Option<((usize, usize), (usize, usize))>,
    dst_stride : usize,
    dst_opt_subsample : Option<((usize, usize), (usize, usize))>
) 
where
    T : Scalar + Default
{
    let src = slice_to_mat(src, src_stride, src_opt_subsample);
    let mut dst = slice_to_mat(&dst, dst_stride, dst_opt_subsample);
    let dst_sz = dst.size().unwrap();
    imgproc::resize(&src, &mut dst, dst_sz, 0.0, 0.0, imgproc::INTER_NEAREST);
}

pub unsafe fn write_text(dst : &mut [u8], ncol : usize, tl_pos : (usize, usize), text : &str, color : u8) {
    let mut dst_mat = slice_to_mat(&dst, ncol, None);
    let scale = 0.5;
    let weight = 1;
    let face = imgproc::FONT_HERSHEY_SIMPLEX; //imgproc::FONT_HERSHEY_PLAIN,
    let thickness = imgproc::LINE_8;
    // Gets size if text were to fit inside single line
    let sz = imgproc::get_text_size(text, face, scale, thickness, &mut 0).unwrap();
    for (line_ix, line) in text.split("\n").enumerate() {
        imgproc::put_text(
            &mut dst_mat,
            line,
            core::Point{ x : tl_pos.1 as i32, y : tl_pos.0 as i32 + (line_ix as i32 * sz.height) },
            face,
            scale,
            core::Scalar::all(color.into()),
            weight,
            thickness,
            false
        ).unwrap()
    }
}

pub unsafe fn draw_line(dst : &mut [u8], ncol : usize, from : (usize, usize), to : (usize, usize), color : u8) {
    let nrow = dst.len() / ncol;
    if from.0 > nrow || from.1 > ncol {
        // println!("Line origin outside bounds");
        return;
    }
    if to.0 > nrow || to.1 > ncol {
        // println!("Line destination outside bounds");
        return;
    }
    let mut dst_mat = slice_to_mat(&dst, ncol, None);
    let line_ty = imgproc::LINE_8;
    imgproc::line(
        &mut dst_mat, 
        core::Point{ x : from.1 as i32, y : from.0 as i32 }, 
        core::Point{ x : to.1 as i32, y : to.0 as i32 }, 
        core::Scalar::all(color.into()),
        1, 
        line_ty,
        0
    ).unwrap()
}

pub unsafe fn draw_circle(dst : &mut [u8], ncol : usize, center : (usize, usize), radius : usize, color : u8) {
    let nrow = dst.len() / ncol;
    if center.0 + radius > nrow || center.1 + radius > ncol {
        // println!("Circle outside bounds");
        return;
    }
    let mut dst_mat = slice_to_mat(&dst, ncol, None);
    let thickness = 1;
    let line_ty = imgproc::LINE_8;
    let shift = 0;
    imgproc::circle(
        &mut dst_mat,
        core::Point2i{ x : center.1 as i32, y : center.0 as i32 },
        radius as i32,
        core::Scalar::all(color.into()),
        thickness,
        line_ty,
        shift
    ).unwrap();
}


