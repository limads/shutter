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
    N : Scalar + Clone + Copy + Serialize + DeserializeOwned + Any
{
    let win_row_bytes = row_size_bytes::<N>(win.orig_sz.1);
    (win_row_bytes, window_size(win))
}

pub fn step_and_size_for_window<N>(win : &Window<'_, N>) -> (i32, crate::foreign::ipp::ippi::IppiSize)
where
    N : Scalar + Clone + Copy + Serialize + DeserializeOwned + Any
{
    let win_row_bytes = row_size_bytes::<N>(win.orig_sz.1);
    /*let pad_front = win.offset.1 * mem::size_of::<N>();
    let pad_back = (win.orig_sz.1 - (win.offset.1 + win.width())) * mem::size_of::<N>();
    (pad_front + win_row_bytes + pad_back, window_size(win))*/
    (win_row_bytes, window_size(win))
}

pub fn step_and_size_for_image<N>(img : &Image<N>) -> (i32, crate::foreign::ipp::ippi::IppiSize)
where
    N : Scalar + Clone + Copy + Serialize + DeserializeOwned + Any
{
    (row_size_bytes::<N>(img.width()), image_size(img))
}

pub fn window_size<N>(win : &Window<'_, N>) -> crate::foreign::ipp::ippi::IppiSize
where
    N : Scalar + Clone + Copy + Serialize + DeserializeOwned + Any
{
    crate::foreign::ipp::ippi::IppiSize { width : win.width() as i32, height : win.height() as i32 }
}

pub fn window_mut_size<N>(win : &WindowMut<'_, N>) -> crate::foreign::ipp::ippi::IppiSize
where
    N : Scalar + Clone + Copy + Serialize + DeserializeOwned + Any
{
    crate::foreign::ipp::ippi::IppiSize { width : win.width() as i32, height : win.height() as i32 }
}

pub fn image_size<N>(img : &Image<N>) -> crate::foreign::ipp::ippi::IppiSize
where
    N : Scalar + Clone + Copy + Serialize + DeserializeOwned + Any
{
    crate::foreign::ipp::ippi::IppiSize { width : img.width() as i32, height : img.height() as i32 }
}

pub fn slice_size_bytes<T>(s : &[T]) -> i32 {
    (s.len() * mem::size_of::<T>()) as i32
}

pub fn row_size_bytes<T>(ncol : usize) -> i32 {
    (ncol * mem::size_of::<T>()) as i32
}

pub unsafe fn convert<S,T>(src : &[S], dst : &mut [T], ncol : usize)
where
    S : Scalar + Any,
    T : Scalar
{
    // Foi ROI/Step info, check page 27 of IPPI manual.
    // 8-bit to 16-bit
    // srcStep and dstStep are distances in bytes between starting points of consecutive lines in source
    // and destination images. So actual distance is dist_px * sizeof(data). If using ROIs, passed pointers
    // should refer to the ROI start, NOT image start. roiSize is always in pixels.
    assert!(src.len() == dst.len());
    let size = IppiSize{ width : ncol as i32, height : (src.len() / ncol) as i32 };
    let mut status : Option<i32> = None;

    let src_ptr = src.as_ptr() as *const ffi::c_void;
    let dst_ptr = dst.as_mut_ptr() as *mut ffi::c_void;

    if (&src[0] as &dyn Any).is::<u8>() && (&dst[0] as &dyn Any).is::<f32>() {
         status = Some(ippiConvert_8u32f_C1R(
            src_ptr as *const u8,
            row_size_bytes::<u8>(ncol),
            dst_ptr as *mut f32,
            row_size_bytes::<f32>(ncol),
            size
        ));
    }

    if (&src[0] as &dyn Any).is::<f32>() && (&dst[0] as &dyn Any).is::<u8>() {
         status = Some(ippiConvert_32f8u_C1R(
            src_ptr as *const f32,
            row_size_bytes::<f32>(ncol),
            dst_ptr as *mut u8,
            row_size_bytes::<u8>(ncol),
            size,
            crate::foreign::ipp::ippcore::IppRoundMode_ippRndNear
        ));
    }

    match status {
        Some(status) => check_status("Conversion", status),
        None => panic!("Invalid conversion type")
    }

}

unsafe fn init_resize_state<T>(src_size : IppiSize, dst_size : IppiSize) -> (*mut IppiResizeSpec_32f, *mut ffi::c_void)
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

    if is_u8 {
        let status_code = ippiResizeGetSize_8u(
            src_size,
            dst_size,
            IppiInterpolationType_ippNearest,
            antialiasing,
            &mut spec_size,
            &mut init_buf_size
        );
        check_status("Resize get size", status_code);
        status_get_size = Some(status_code);
    }

    // Then initialize structure using the out parameters:
    // IppiResizeSpec_<T> has only types for T = 32f or T = 64f
    let spec : *mut IppiResizeSpec_32f = ptr::null_mut();

    if is_u8 {
        let status_code = ippiResizeNearestInit_8u(src_size, dst_size, spec);
        check_status("Resize init", status_code);
        status_init = Some(status_code);
    }

    // Get buffer size for the current spec
    let n_channels = 1;
    let mut buf_sz = 0;

    let mut buf_ptr : *mut ffi::c_void = ptr::null_mut();
    if is_u8 {
        let status_code = ippiResizeGetBufferSize_8u(spec, dst_size, n_channels, &mut buf_sz as *mut _);
        check_status("Allocate resize buffer", status_code);
        buf_ptr = ipps::ippsMalloc_8u(buf_sz) as *mut ffi::c_void;
        status_get_buf_size = Some(status_code);
    }

    if status_get_size.is_some() && status_init.is_some() && status_get_buf_size.is_some() {
        (spec, buf_ptr)
    } else {
        panic!("Invalid resize type");
    }
}

pub unsafe fn resize<T>(src : &[T], dst : &mut [T], src_dim : (usize, usize), dst_dim : (usize, usize))
where
    T : Scalar
{
    let mut status : Option<i32> = None;
    let src_size = IppiSize{ width : src_dim.1 as i32, height : src_dim.0 as i32 };
    let dst_size = IppiSize{ width : dst_dim.1 as i32, height : dst_dim.0 as i32 };

    let dst_offset = 0;
    if (&src[0] as &dyn Any).is::<u8>() {
        let (spec, buf_ptr) = init_resize_state::<u8>(src_size, dst_size);
        let status_code = ippiResizeNearest_8u_C1R(
            (src.as_ptr() as *const ffi::c_void) as *const u8,
            row_size_bytes::<u8>(src_dim.1),
            (dst.as_mut_ptr() as *mut ffi::c_void) as *mut u8,
            row_size_bytes::<u8>(dst_dim.1),
            IppiPoint{ x : 0, y : 0 },
            dst_size,
            spec as *const _,
            buf_ptr as *mut u8
        );
        ipps::ippsFree(buf_ptr);
        status = Some(status_code);
    }

    match status {
        Some(status) => check_status("Resize", status),
        None => panic!("Invalid resize type")
    }
}


