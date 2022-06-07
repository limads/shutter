use crate::foreign::ipp::{ippcore, ippi::*};
use std::mem;
use std::ptr;
use std::ffi;
use crate::foreign::ipp::ipps;
use nalgebra::Scalar;
use crate::foreign::ipp::ippcore::{ippMalloc};
use std::any::Any;
use crate::image::*;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::iter::FromIterator;
use crate::raster::*;

// Allocates a buffer from a function that writes how many bytes should be allocated.
pub fn allocate_buffer_with<F>(f : F) -> Vec<u8>
where
    F : Fn(*mut i32)->IppStatus
{
    let mut buf_sz : i32 = 0;
    let status = f(&mut buf_sz as *mut _);
    assert!(status == 0);
    assert!(buf_sz > 0);
    Vec::from_iter((0..buf_sz).map(|_| 0u8 ) )
}

pub fn check_status(action : &str, status : i32) {
    if status as u32 == ippcore::ippStsNoErr {
        return;
    }
    let err_msg : &'static str = match status {
        ippcore::ippStsNullPtrErr => "Null pointer",
        ippcore::ippStsNumChannelsErr => "Wrong number of channels",
        ippcore::ippStsAnchorErr => "Anchor error",
        ippcore::ippStsSizeErr => "Size error",
        ippcore::ippStsStepErr => "Step error",
        ippcore::ippStsContextMatchErr => "Context match error",
        ippcore::ippStsMemAllocErr => "Memory allocation error",
        ippcore::ippStsBadArgErr => "Bad argument",
        _ => "Unknown error"
    };
    panic!("IPPS Error\tAction: {}\tCode: {}\tMessage: {}", action, status, err_msg);
}

pub fn step_and_size_for_window_mut<N>(win : &WindowMut<'_, N>) -> (i32, crate::foreign::ipp::ippi::IppiSize)
where
    N : Scalar + Clone + Copy + Any
{
    let win_row_bytes = row_size_bytes::<N>(win.width);
    (win_row_bytes, window_mut_size(win))
}

// Replace by raster implementor
pub fn step_and_size_for_window<N>(win : &Window<'_, N>) -> (i32, crate::foreign::ipp::ippi::IppiSize)
where
    N : Scalar + Clone + Copy + Any
{
    let win_row_bytes = row_size_bytes::<N>(win.original_width());
    /*let pad_front = win.offset.1 * mem::size_of::<N>();
    let pad_back = (win.orig_sz.1 - (win.offset.1 + win.width())) * mem::size_of::<N>();
    (pad_front + win_row_bytes + pad_back, window_size(win))*/
    (win_row_bytes, window_size(win))
}

pub fn byte_stride_for_window<N>(win : &Window<'_, N>) -> i32
where
    N : Scalar + Clone + Copy
{
    (mem::size_of::<N>() * win.original_width()) as i32
}

pub fn byte_stride_for_window_mut<N>(win : &WindowMut<'_, N>) -> i32
where
N : Scalar + Clone + Copy
{
    (mem::size_of::<N>() * win.original_width()) as i32
}

pub fn step_and_size_for_image<N>(img : &Image<N>) -> (i32, crate::foreign::ipp::ippi::IppiSize)
where
    N : Scalar + Clone + Copy + Any
{
    // For image, img.width() == window.width
    (row_size_bytes::<N>(img.width()), image_size(img))
}

pub fn window_size<N>(win : &Window<'_, N>) -> crate::foreign::ipp::ippi::IppiSize
where
    N : Scalar + Clone + Copy + Any
{
    crate::foreign::ipp::ippi::IppiSize { width : win.width() as i32, height : win.height() as i32 }
}

pub fn window_mut_size<N>(win : &WindowMut<'_, N>) -> crate::foreign::ipp::ippi::IppiSize
where
    N : Scalar + Clone + Copy + Any
{
    crate::foreign::ipp::ippi::IppiSize { width : win.width() as i32, height : win.height() as i32 }
}

pub fn image_size<N>(img : &Image<N>) -> crate::foreign::ipp::ippi::IppiSize
where
    N : Scalar + Clone + Copy + Any
{
    crate::foreign::ipp::ippi::IppiSize { width : img.width() as i32, height : img.height() as i32 }
}

pub fn slice_size_bytes<T>(s : &[T]) -> i32 {
    (s.len() * mem::size_of::<T>()) as i32
}

pub fn row_size_bytes<T>(ncol : usize) -> i32 {
    (ncol * mem::size_of::<T>()) as i32
}

pub struct IppiResize {
    spec_bytes : Vec<u8>,
    init_buf_bytes : Vec<u8>,
    work_buf_bytes : Vec<u8>
}

impl IppiResize {

    unsafe fn spec_ptr(&self) -> *mut IppiResizeSpec_32f {
        mem::transmute::<_, _>(&self.spec_bytes[0])
    }

    fn buf_ptr(&mut self) -> *mut u8 {
        self.work_buf_bytes.as_mut_ptr()
    }

    unsafe fn new<T>(src_size : IppiSize, dst_size : IppiSize) -> Self
    where
        T : Scalar + Default
    {
        let mut status_get_size : Option<i32> = None;
        let mut status_init : Option<i32> = None;
        let mut status_get_buf_size : Option<i32> = None;

        // For resize: First calculate required size:
        let mut spec_size : i32 = 0;
        let mut init_buf_size : i32 = 0;
        let antialiasing = 0;

        let t : T = Default::default();
        let is_u8 = (&t as &dyn Any).is::<u8>();
        let is_f32 = (&t as &dyn Any).is::<f32>();
        let interp_ty = IppiInterpolationType_ippNearest;
        if is_u8 {
            let status_code = ippiResizeGetSize_8u(
                src_size,
                dst_size,
                interp_ty,
                antialiasing,
                &mut spec_size,
                &mut init_buf_size
            );
            check_status("Resize get size", status_code);
            status_get_size = Some(status_code);
        } else if is_f32 {
            let status_code = ippiResizeGetSize_32f(
                src_size,
                dst_size,
                interp_ty,
                antialiasing,
                &mut spec_size,
                &mut init_buf_size
            );
            check_status("Resize get size", status_code);
            status_get_size = Some(status_code);
        } else {
            panic!("Image expected to be u8 or f32");
        }

        // Allocating an initialization buffer with init_buf_size is only required for
        // lanczos and cubic filters. For nearest and linear, we can do without it, since
        // ippiresizenearestinit and ippiresizelinearinit do not take an initialization
        // buffer argument.

        // Then initialize structure using the out parameters:
        // IppiResizeSpec_<T> has only types for T = 32f or T = 64f
        let spec_bytes = Vec::from_iter((0..(spec_size as usize)).map(|_| 0u8 ));
        let init_buf_bytes = Vec::from_iter((0..(init_buf_size as usize)).map(|_| 0u8 ));
        let spec : *mut IppiResizeSpec_32f = mem::transmute::<_, _>(&spec_bytes[0]);
        let init_buf : *mut u8 = mem::transmute::<_, _>(&init_buf_bytes[0]);

        // mem::forget(spec_bytes);
        // mem::forget(init_buf_bytes);

        if is_u8 {
            let status_code = ippiResizeNearestInit_8u(src_size, dst_size, spec);
            check_status("Resize init", status_code);
            status_init = Some(status_code);
        } else if is_f32 {
            let status_code = ippiResizeNearestInit_32f(src_size, dst_size, spec);
            check_status("Resize init", status_code);
            status_init = Some(status_code);
        } else {
            panic!("Image expected to be u8 or f32");
        }

        // Get buffer size for the current spec
        let n_channels = 1;
        let mut work_buf_sz = 0;

        if is_u8 {
            let status_code = ippiResizeGetBufferSize_8u(spec, dst_size, n_channels, &mut work_buf_sz as *mut _);
            check_status("Allocate resize buffer", status_code);
            assert!(work_buf_sz > 0);
            status_get_buf_size = Some(status_code);
        } else if is_f32 {
            let status_code = ippiResizeGetBufferSize_32f(spec, dst_size, n_channels, &mut work_buf_sz as *mut _);
            check_status("Allocate resize buffer", status_code);
            assert!(work_buf_sz > 0);
            status_get_buf_size = Some(status_code);
        } else {
            panic!("Expected u8");
        }

        let work_buf_bytes = Vec::from_iter((0..(work_buf_sz as usize)).map(|_| 0u8 ));
        if status_get_size.is_some() && status_init.is_some() && status_get_buf_size.is_some() {
            IppiResize { spec_bytes, init_buf_bytes, work_buf_bytes }
        } else {
            panic!("Invalid resize type");
        }
    }

}

pub unsafe fn resize<T>(src : &Window<'_, T>, dst : &mut WindowMut<'_, T>)
where
    T : Scalar + Copy + Default
{
    let mut status : Option<i32> = None;
    let (src_size, dst_size) = (window_size(src), window_mut_size(&dst));
    let (src_step, dst_step) = (byte_stride_for_window(src), byte_stride_for_window_mut(&dst));
    let (src_ptr, dst_ptr) = (src.as_ptr() as *const ffi::c_void, dst.as_mut_ptr() as *mut ffi::c_void);
    let (is_u8, is_f32) = (src.pixel_is::<u8>(), src.pixel_is::<f32>());
    let pt = IppiPoint{ x : 0, y : 0 };
    if is_u8 {
        let mut resize = IppiResize::new::<u8>(src_size, dst_size);
        let status_code = ippiResizeNearest_8u_C1R(
            src_ptr as *const u8,
            src_step,
            dst_ptr as *mut u8,
            dst_step,
            pt,
            dst_size,
            resize.spec_ptr(),
            resize.buf_ptr()
        );
        status = Some(status_code);
    } else if is_f32 {
        let mut resize = IppiResize::new::<f32>(src_size, dst_size);
        let status_code = crate::foreign::ipp::ippi::ippiResizeNearest_32f_C1R(
            src_ptr as *const f32,
            src_step,
            dst_ptr as *mut f32,
            dst_step,
            pt,
            dst_size,
            resize.spec_ptr(),
            resize.buf_ptr()
        );
        status = Some(status_code);
    } else {
        panic!("Expected u8 or f32 image for resize op");
    }
    match status {
        Some(status) => check_status("Resize", status),
        None => panic!("Invalid resize type")
    }
}


