use crate::foreign::ipp::{ippcore, ippi::*};
use std::mem;
use std::ptr;
use std::ffi;
use crate::foreign::ipp::ipps;
use nalgebra::Scalar;
use crate::foreign::ipp::ippcore::{ippMalloc};

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

pub fn slice_size_bytes<T>(s : &[T]) -> i32 {
    (s.len() * mem::size_of::<T>()) as i32
}

pub fn row_size_bytes<T>(ncol : usize) -> i32 {
    (ncol * mem::size_of::<T>()) as i32
}

pub unsafe fn convert<S,T>(src : &[S], dst : &mut [T], ncol : usize) 
where  
    S : Scalar,
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
    
    if S::is::<u8>() && T::is::<f32>() {
         let status_code = ippiConvert_8u32f_C1R(
            (src.as_ptr() as *const ffi::c_void) as *const u8,
            row_size_bytes::<u8>(ncol),
            (dst.as_mut_ptr() as *mut ffi::c_void) as *mut f32,
            row_size_bytes::<f32>(ncol), 
            size
        );
        status = Some(status_code);
    } 
    
    match status {
        Some(status) => check_status("Conversion", status),
        None => panic!("Invalid conversion type")  
    }
    
}

unsafe fn init_resize_state<T>(src_size : IppiSize, dst_size : IppiSize) -> (*mut IppiResizeSpec_32f, *mut ffi::c_void) 
where
    T : Scalar
{
    let mut status_get_size : Option<i32> = None;
    let mut status_init : Option<i32> = None;
    let mut status_get_buf_size : Option<i32> = None;
    
    // For resize: First calculate required size:
    let mut spec_size : i32 = 0;
    let mut init_buf_size : i32 = 0;
    let antialiasing = 0;
    
    if T::is::<u8>() {
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
    
    if T::is::<u8>() {
        let status_code = ippiResizeNearestInit_8u(src_size, dst_size, spec);
        check_status("Resize init", status_code);
        status_init = Some(status_code);
    }
    
    // Get buffer size for the current spec
    let n_channels = 1;
    let mut buf_sz = 0;
    
    let mut buf_ptr : *mut ffi::c_void = ptr::null_mut();
    if T::is::<u8>() {
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
    if T::is::<u8>() {
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

