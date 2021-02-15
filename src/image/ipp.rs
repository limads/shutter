// use crate::foreign::ipp::ippcore::*;
// use crate::foreign::ipp::ippvm::*;
use volta::foreign::ipp::ippi::*;
use std::mem;
use std::ptr;
use std::ffi;
use volta::foreign::ipp::ipps;

pub fn slice_size_bytes<T>(s : &[T]) -> i32 {
    (s.len() * mem::size_of::<T>()) as i32
}

pub fn row_size_bytes<T>(ncol : usize) -> i32 {
    (ncol * mem::size_of::<T>()) as i32
}

unsafe fn convert(src : &[u8], dst : &mut [f32], ncol : usize) {
    // Foi ROI/Step info, check page 27 of IPPI manual.
    // 8-bit to 16-bit
    // srcStep and dstStep are distances in bytes between starting points of consecutive lines in source
    // and destination images. So actual distance is dist_px * sizeof(data). If using ROIs, passed pointers
    // should refer to the ROI start, NOT image start. roiSize is always in pixels.
    assert!(src.len() == dst.len());
    let status = ippiConvert_8u32f_C1R(
        src.as_ptr(), 
        row_size_bytes::<u8>(ncol),
        dst.as_mut_ptr(),
        row_size_bytes::<f32>(ncol), 
        IppiSize{ width : ncol as i32, height : (src.len() / ncol) as i32 } 
    );
}

unsafe fn resize(src : &[u8], dst : &mut [u8], src_dim : (usize, usize), dst_dim : (usize, usize)) {
    // For resize: First calculate required size:
    let mut spec_size : i32 = 0;
    let mut init_buf_size : i32 = 0;
    let antialiasing = 0;
    let src_size = IppiSize{ width : src_dim.1 as i32, height : src_dim.0 as i32 };
    let dst_size = IppiSize{ width : dst_dim.1 as i32, height : dst_dim.0 as i32 };
    let status = ippiResizeGetSize_8u(
        src_size, 
        dst_size,
        IppiInterpolationType_ippNearest,
        antialiasing,
        &mut spec_size,                 
        &mut init_buf_size                    
    );
    
    // Then initialize structure using the out parameters:
    // IppiResizeSpec_<T> has only types for T = 32f or T = 64f
    let spec : *mut IppiResizeSpec_32f = ptr::null_mut();
    let status = ippiResizeNearestInit_8u(src_size, dst_size, spec);
    
    // Get buffer size for the current spec
    let n_channels = 1;
    let mut buf_sz = 0;
    let status = ippiResizeGetBufferSize_8u(spec, dst_size, n_channels, &mut buf_sz as *mut _);
    let buf_ptr = ipps::ippsMalloc_8u(buf_sz);
    let dst_offset = 0;
    ippiResizeNearest_8u_C1R(
        src.as_ptr(), 
        row_size_bytes::<u8>(src_dim.1),
        dst.as_mut_ptr(), 
        row_size_bytes::<u8>(dst_dim.1), 
        IppiPoint{ x : 0, y : 0 }, 
        dst_size, 
        spec as *const _, 
        buf_ptr
    );
    ipps::ippsFree(buf_ptr as *mut ffi::c_void);
}

