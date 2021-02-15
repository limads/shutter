use volta::foreign::ipp::ippi::*;
use super::super::ipp::row_size_bytes;
use volta::foreign::ipp::ippcore::{self, ippMalloc};
use volta::signal::dwt;

/*pub struct Wavelet2D<N> {
    levels : Vec<WaveletLevel<N>>
}

pub struct WaveletLevel<N> {
    src : Image<N>,
    dst : Image<N>
}*/

fn check_status(action : &str, status : i32) {
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
        _ => "Unknown error"
    };
    panic!("IPPS Error\tAction: {}\tCode: {}\tMessage: {}", action, status, err_msg);
}

/// For a filter of size k, if anchor = 0, the full image should be padded with k-1 pixels
/// to the left, right, top and bottom portions. The left and top borders will have the same
/// size; the right and bottom borders will also have the same size. The function return the
/// value pair corresponding to those borders. For all anchor values, the allocated image size
/// will be the same (the left/top + bottom/right should sum to k-1), but the border size will
/// be different. For anchor = 0 left/top borders will be k-1 and the right/bottom borders
/// will be 0. For increasing anchor size, the top/left value will decrease and the right/bottom
/// borders will decrease. 
fn border_size(
    src_width : usize, 
    src_height : usize, 
    filt_len_low : usize, 
    filt_len_high : usize, 
    anchor_low : usize, 
    anchor_high : usize
) -> (usize, usize) {
    let left_border_low = filt_len_low - 1 - anchor_low;
    let left_border_high = filt_len_high - 1 - anchor_high;
    let right_border_low = anchor_low;
    let right_border_high = anchor_high;
    let left_top_border = left_border_low.max(left_border_high);
    let right_bottom_border = right_border_low.max(right_border_high);
    (left_top_border, right_bottom_border)
}

/// Returns extended required image size, as (nrow, ncol)
fn extended_image_size(
    src_width : usize, 
    src_height : usize, 
    filt_len_low : usize, 
    filt_len_high : usize, 
    anchor_low : usize, 
    anchor_high : usize
) -> (usize, usize) {
    // Required full image size:
    let (left_top_border, right_bottom_border) = border_size(src_width, src_height, filt_len_low, filt_len_high, anchor_low, anchor_high);
    let src_width_with_borders = src_width + left_top_border + right_bottom_border;
    let src_height_with_borders = src_height + left_top_border + right_bottom_border;
    (src_width_with_borders, src_height_with_borders)
}

unsafe fn build_filters(taps_low : &[f32], taps_high :&[f32]) -> (*mut IppiWTFwdSpec_32f_C1R, *mut u8) {
    assert!(taps_low.len() == 4 && taps_high.len() == 4);
    let num_channels = 1;
    let len_low = taps_low.len() as i32;
    let len_high = taps_high.len() as i32;

    // Anchor : Left-most filter position wrt image first row or first column (>0)
    let anchor_low = 0;
    let anchor_high = 0;
    
    let mut spec_sz = 0;
    let mut buf_sz = 0;
    let get_sz_status = ippiWTFwdGetSize_32f(
        num_channels, 
        len_low, 
        anchor_low, 
        len_high,
        anchor_high, 
        &mut spec_sz as *mut _, 
        &mut buf_sz as *mut _
    );
    check_status("get fwd size", get_sz_status);
    
    let fwd_spec = ippMalloc(spec_sz) as *mut IppiWTFwdSpec_32f_C1R;
    let init_status = ippiWTFwdInit_32f_C1R(
        fwd_spec, 
        taps_low.as_ptr(),
        len_low, 
        anchor_low, 
        taps_high.as_ptr(), 
        len_high, 
        anchor_high
    );
    check_status("init forward", init_status);
    
    let buf = ippMalloc(buf_sz) as *mut u8;
    (fwd_spec, buf)
}

/// Note: src slice should point to inner image region, to account for DWT borders.
/// approx/detail_x/detail_y/detail_xy: Destination slices of size (nrow / 2, ncol / 2) each
/// src: Padded Source slice of size (tl+nrow,tl+ncol)
/// ncol: ROI number of columns (assumed to be equal to number of rows)
unsafe fn apply_filters(
    spec : *const IppiWTFwdSpec_32f_C1R,
    buf : *mut u8,              
    src : &[f32],               
    ncol : usize,               
    approx : &mut [f32],
    detail_x : &mut [f32],
    detail_y : &mut [f32],
    detail_xy : &mut [f32]
) {
    // Bottom-left border will be zero if anchor is zero.
    assert!(ncol % 2 == 0);
    let filt_len = 4;
    let anchor = 0;
    let (tl_border, br_border) = border_size(ncol, ncol, filt_len, filt_len, anchor, anchor);
    assert!(br_border == 0);
    let px_stride = ncol + tl_border;
    
    // Total padding applied to the top-left of the original buffer slice: Since the top and
    // left borders are the same, count (ncol + left) border (top) times.
    let tl_pad = tl_border*px_stride + tl_border;
    println!("TL pad: {}", tl_pad);
    println!("TL border: {}", tl_border);
    let roi_slice = &src[tl_pad..];
    
    let src_step = row_size_bytes::<f32>(px_stride);
    let x_step = row_size_bytes::<f32>(ncol / 2);
    let y_step = x_step;
    let xy_step = x_step;
    let approx_step = x_step;
    
    let dst_roi = IppiSize{ width : (ncol / 2) as i32, height : (ncol / 2) as i32 };
    let fwd_status = ippiWTFwd_32f_C1R (
        roi_slice.as_ptr(), 
        src_step,
        approx.as_mut_ptr(), 
        approx_step, 
        detail_x.as_mut_ptr(), 
        x_step, 
        detail_y.as_mut_ptr(), 
        y_step,
        detail_xy.as_mut_ptr(), 
        xy_step, 
        dst_roi, 
        spec, 
        buf 
    );
    check_status("apply forward", fwd_status);
}

#[test]
fn test_extended_image_size() {
    /// Anchor should be smaller than filter size. In all cases, a 64x64 image is extended to 67 x 67 (filter len - 1)
    let anchors : [usize; 4] = [0, 1, 2, 3];
    let filt_len_low = 4;
    let filt_len_high = 4;
    for anchor in &anchors {
        let (left_top, right_bottom) = border_size(4, 4, filt_len_low, filt_len_high, *anchor, *anchor);
        println!("Left and top borders = {:?}; Right and bottom borders = {:?}", left_top, right_bottom);
        let extended_sz = extended_image_size(4, 4, filt_len_low, filt_len_high, *anchor, *anchor);
        println!("Anchor = {:?}; Extended size = {:?}", (anchor, anchor), extended_sz);
    }    
}

// cargo test --all-features image_dwt -- --nocapture
#[test]
fn image_dwt() {
    // A 4x4 image should be padded to a 7x7 image (filter len-1), which has 49 entries.
    // If we assume anchor=0, the padding is in the first 3 rows and columns.
    //                  | Effective image starts here 
    //                  V     
    let img : [f32; 49] = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // <-- Effective image starts here
        0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    
    // Output destination will be a 2x2 image
    let mut approx : [f32; 4] = [0.0; 4];
    let mut detail_x = approx.clone();
    let mut detail_y = approx.clone();
    let mut detail_xy = approx.clone();
    
    unsafe {
        let (spec, buf) = build_filters(&dwt::DAUB4_LOW[..], &dwt::DAUB4_HIGH[..]);
        apply_filters(
            spec, 
            buf, 
            &img[..], 
            4, 
            &mut approx[..], 
            &mut detail_x[..], 
            &mut detail_y[..], 
            &mut detail_xy[..]
        );
    }
    println!("Approx = {:?}", approx);
    println!("Detail X = {:?}", detail_x);
    println!("Detail Y = {:?}", detail_y);
    println!("Detail XY = {:?}", detail_xy);
    
    /*
    Approx = [0.0, 0.0, 0.40400636, 1.291266]
    Detail X = [0.0, 0.0, -0.10825317, -0.34599364]
    Detail Y = [0.0, 0.0, -0.10825317, 0.40400636]
    Detail XY = [0.0, 0.0, 0.02900635, -0.10825318]
    */
}
