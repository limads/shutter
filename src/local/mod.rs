use crate::image::*;
use std::mem;
use std::any::Any;
use serde::{Serialize, Deserialize, de::DeserializeOwned};
use num_traits::{Zero, AsPrimitive};
use nalgebra::Scalar;
use std::iter::FromIterator;
pub use ripple::conv::*;
use std::fmt::Debug;
use std::ops::Add;

#[test]
pub fn sum_row() {
    let side = 4;
    let mut img = ImageBuf::<u8>::new_constant(side, side, 1);
    let mut dst = ImageBuf::<f32>::new_constant(4, 4, 0.);
    let (src_step, src_roi) = crate::image::ipputils::step_and_size_for_image(&img);
    let (dst_step, dst_roi) = crate::image::ipputils::step_and_size_for_image(&img);
    let mask_sz = 4 as i32;
    let anchor = 0;
    let ans = unsafe {
        crate::foreign::ipp::ippi::ippiSumWindowRow_8u32f_C1R(
            img.as_ptr(),
            src_step,
            dst.as_mut_ptr(),
            dst_step,
            dst_roi,
            mask_sz,
            anchor
        )
    };
    assert!(ans == 0);
    let mask_sz = 16;
    for i in 0..4usize {
        println!("Row {}", i);
        for j in 0..4usize {
            println!("{}", dst[(i, j)]);
        }
    }
}

/*/*
The filling routines assumes 3x3 neighborhoods at object edges must have a given
number of positive entries for the overall shape to be convex. The center pixel
of the local 3x3 window is a potential concavity, that is filled whenever the local sum
is too big.
*/
// After Davies (2005) Alg. 6.9. Makes shapes in a binary imageg convex.
// (Ignore corners, but faster)
pub fn fill_without_corners(src : &Window<u8>, mut dst : WindowMut<u8>) {
    let win_sz = 3;
    let gt_val = 3;
    let cond_val = 0;
    let mut n_changed = usize::MAX;
    dst.copy_from(src);
    while n_changed > 0 {
        let mut dst = unsafe { WindowMut::sub_from_slice(dst.original_slice(), dst.original_width(), dst.offset(), dst.shape()).unwrap() };
        n_changed = crate::morph::conditional_set_when_neighbors_sum_greater(
            src,
            dst,
            win_sz,
            cond_val,
            gt_val,
            255,
            None
        );
    }
}*/

fn bool_to_u64(b : bool) -> u64 {
    if b { 255 } else { 0 }
}

/*// After Davies (2005) Alg. 6.10. Makes shapes in a binary imageg convex.
// (Account for corners, but more costly). The notation of the author is:
// A0 is the middle element; A1-A8 are the neighbors element in a clockwise
// fashion around A0 starting from top-left.
pub fn fill_with_corners(src : &Window<u8>, mut dst : WindowMut<u8>) 
where
    u64 : Pixel,
    u8 : Pixel
{
    assert!(src.shape() == dst.shape());
    let center = (1usize,1usize);
    let mut n_changed = usize::MAX;
    dst.copy_from(src);
    while n_changed > 0 {
        n_changed = 0;
        let mut dst = unsafe { WindowMut::sub_from_slice(dst.original_slice(), dst.original_width(), dst.offset(), dst.shape()).unwrap() };
        for (mut d, s) in dst.windows_mut((3, 3)).zip(src.windows((3, 3))) {
            if s[center] == 0 {
                let q1 = bool_to_u64(*s.linear_index(0) == 255 && *s.linear_index(1) != 255 && *s.linear_index(2) == 255);
                let q2 = bool_to_u64(*s.linear_index(2) == 255 && *s.linear_index(5) != 255 && *s.linear_index(8) == 255);
                let q3 = bool_to_u64(*s.linear_index(8) == 255 && *s.linear_index(7) != 255 && *s.linear_index(6) == 255);
                let q4 = bool_to_u64(*s.linear_index(6) == 255 && *s.linear_index(3) != 255 && *s.linear_index(0) == 255);
                let sum = s.accum::<u64>() - s[center] as u64 + q1 + q2 + q3 + q4;
                if sum > 3 {
                    d[center] = 255;
                    n_changed += 1;
                } else {
                    // Already solved by a first copy from at the start.
                    // d[center] = s[center];
                }
            } else {
                // Already solved by a first copy from at the start
                // d[center] = s[center];
            }
        }
    }
}*/

const MEDIAN_POS_3 : u64 = 5;

pub fn median_filter3(src : &Window<u8>, mut dst : WindowMut<u8>) {
    let mut hist : [u8; 256] = [0; 256];
    for (d, s) in dst.windows_mut((3,3)).zip(src.windows((3,3))) {
        local_median_filtering(s, d, &mut hist);
    }
}

// After Davies (2005) Fig. 3.4. This is a single step of the median filter.
fn local_median_filtering(win : Window<u8>, mut dst : WindowMut<u8>, hist : &mut [u8; 256])  {

    // Clear histogram
    hist.iter_mut().for_each(|h| *h = 0 );

    /* Iteration assumes that in a 3x3 window, the linear indices 0..(N^2-1)
    excluding the central (N^2-1)/2 index index the neighborhood, while
    the central (N^2-1)/2 index the window center. */

    // Iter neighbors
    for m in [0, 1, 2, 3, 5, 6, 7, 8] {

        // Update histogram
        hist[*win.linear_index(m) as usize] += 1;

        // Walk from start towards end of the histogram until the median is found.
        let mut median : u8 = 0;
        let mut sum : u64  = 0;
        while sum < MEDIAN_POS_3 {
            sum += hist[median as usize] as u64;
            median += 1;
        }
        *(dst.linear_index_mut(5)) = median.saturating_sub(1);
    }
}

/*
Based on https://rosettacode.org/wiki/Median_filter
void del_pixels(image im, int row, int col, int size, color_histo_t *h)
{
	int i;
	rgb_t *pix;

	if (col < 0 || col >= im->w) return;
	for (i = row - size; i <= row + size && i < im->h; i++) {
		if (i < 0) continue;
		pix = im->pix[i] + col;
		h->r[pix->r]--;
		h->g[pix->g]--;
		h->b[pix->b]--;
		h->n--;
	}
}

void add_pixels(image im, int row, int col, int size, color_histo_t *h)
{
	int i;
	rgb_t *pix;

	if (col < 0 || col >= im->w) return;
	for (i = row - size; i <= row + size && i < im->h; i++) {
		if (i < 0) continue;
		pix = im->pix[i] + col;
		h->r[pix->r]++;
		h->g[pix->g]++;
		h->b[pix->b]++;
		h->n++;
	}
}

void init_histo(image im, int row, int size, color_histo_t*h)
{
	int j;

	memset(h, 0, sizeof(color_histo_t));

	for (j = 0; j < size && j < im->w; j++)
		add_pixels(im, row, j, size, h);
}

int median(const int *x, int n)
{
	int i;
	for (n /= 2, i = 0; i < 256 && (n -= x[i]) > 0; i++);
	return i;
}

void median_color(rgb_t *pix, const color_histo_t *h)
{
	pix->r = median(h->r, h->n);
	pix->g = median(h->g, h->n);
	pix->b = median(h->b, h->n);
}

image median_filter(image in, int size)
{
	int row, col;
	image out = img_new(in->w, in->h);
	color_histo_t h;

	for (row = 0; row < in->h; row ++) {
		for (col = 0; col < in->w; col++) {
			if (!col) init_histo(in, row, size, &h);
			else {
				del_pixels(in, row, col - size, size, &h);
				add_pixels(in, row, col + size, size, &h);
			}
			median_color(out->pix[row] + col, &h);
		}
	}

	return out;
}
*/

pub fn baseline_local_sum<P>(src : &Window<P>, dst : &mut WindowMut<P>)
where
    P : Pixel + Add<Output=P>
{
    assert!(src.height() % dst.height() == 0);
    assert!(src.width() % dst.width() == 0);
    let local_win_sz = (src.height() / dst.height(), src.width() / dst.width());
    for i in 0..dst.height() {
        for j in 0..dst.width() {
            let local = src.sub_window(
                (i*local_win_sz.0, j*local_win_sz.1),
                local_win_sz
            ).unwrap();
            dst[(i, j)] = local.accum::<P>();
        }
    }
}

#[test]
fn local_sum_test() {
    let mut img = ImageBuf::<u8>::new_constant(32, 32, 1);
    let mut dst = ImageBuf::<i32>::new_constant(4, 4, 0);
    local_sum(&img.full_window(), &mut dst.full_window_mut());
    println!("{:?}", dst[(2usize,2usize)]);
    dst.scalar_div_mut(4);
    println!("{:?}", dst[(2usize,2usize)]);
}

pub fn local_sum(src : &Window<'_, u8>, dst : &mut WindowMut<'_, i32>) {

    assert!(src.height() % dst.height() == 0);
    assert!(src.width() % dst.width() == 0);

    let local_win_sz = (src.height() / dst.height(), src.width() / dst.width());

    #[cfg(feature="ipp")]
    unsafe {
        let (src_step, src_sz) = crate::image::ipputils::step_and_size_for_window(src);
        let (dst_step, dst_sz) = crate::image::ipputils::step_and_size_for_window_mut(&dst);
        let mask_sz = crate::foreign::ipp::ippi::IppiSize {
            width : local_win_sz.1 as i32,
            height : local_win_sz.0 as i32
        };
        let mut num_channels = 1;
        let mut buf_sz : i32 = 0;
        let ans = crate::foreign::ipp::ippi::ippiSumWindowGetBufferSize(
            dst_sz,
            mask_sz,
            crate::foreign::ipp::ippi::IppDataType_ipp8u,
            num_channels,
            &mut buf_sz as *mut _
        );
        assert!(ans == 0);
        let mut buffer = Vec::from_iter((0..buf_sz).map(|_| 0u8 ));
        println!("Buffer allocated");
        let border_val = 0u8;
        let ans = crate::foreign::ipp::ippi::ippiSumWindow_8u32s_C1R(
            src.as_ptr(),
            src_step,
            dst.as_mut_ptr(),
            dst_step,
            dst_sz,
            mask_sz,
            crate::foreign::ipp::ippi::_IppiBorderType_ippBorderConst,
            &border_val,
            buffer.as_mut_ptr()
        );
        assert!(ans == 0);
        return;
    }
    
    
    /*for i in 0..dst.height() {
        for j in 0..dst.width() {
            let off = (i*local_win_sz.0, j*local_win_sz.1);
            dst[(i, j)] = src.sub_window(off, local_win_sz).unwrap().sum::<f32>(1) as i32;
        }
    }*/
    unimplemented!()

}

// Implements median filter by histogram (Huang & Yang, 1979)
fn histogram_median_filter(src : &Window<'_, u8>, mut dst : WindowMut<'_, u8>, win_sz : (usize, usize)) {

    // (1) Calculate the median by building a histogram and sorting pixels
    // and getting the middle one at first sub-window. Store number of pixels less than the median.

    // (2) Update histogram by moving one column to the left. Update the count as well. Now the count
    // means number of pixels less than the median of the previous window.

    // (3) Starting from the median of the previous window, move up/down histogram of bins if
    // the count is not greater/greater than #pixels in the window / 2 and update the count
    // of #less median until the median bin is reached.
}

// This implements min-pool or max-pool.
pub fn block_min_or_max<N>(
    win : &Window<'_, N>,
    min_dst : Option<&mut WindowMut<'_, N>>,
    max_dst : Option<&mut WindowMut<'_, N>>
) where
    N : Pixel,
    for<'a> &'a [N] : Storage<N>,
    for<'a> &'a mut [N] : StorageMut<N>,
{

    let mut block_sz = (0, 0);
    if let Some(dst) = &min_dst {
        assert!(win.width() % dst.width() == 0);
        assert!(win.height() % dst.height() == 0);
        block_sz = (win.height() / dst.height(), win.width() / dst.width());
    }
    if let Some(dst) = &max_dst {
        assert!(win.width() % dst.width() == 0);
        assert!(win.height() % dst.height() == 0);
        if block_sz.0 == 0 {
            block_sz = (win.height() / dst.height(), win.width() / dst.width());
        } else {
            assert!(win.height() / dst.height() == block_sz.0 && win.width() / dst.width() == block_sz.1);
        }
    }
    assert!(block_sz.0 > 0 && block_sz.1 > 0);

    #[cfg(feature="ipp")]
    unsafe {
        let (src_step, src_sz) = crate::image::ipputils::step_and_size_for_window(win);
        let (dst_step, dst_sz) = if let Some(dst) = &min_dst {
            crate::image::ipputils::step_and_size_for_window_mut(dst)
        } else {
            crate::image::ipputils::step_and_size_for_window_mut(max_dst.as_ref().unwrap())
        };

        // If pdstmin or pdstmax are null, the corresponding result is not calculated.
        let (ptr_dst_min, ptr_dst_max) : (*mut N, *mut N) = match (min_dst, max_dst) {
            (Some(min), Some(max)) => (min.as_mut_ptr(), max.as_mut_ptr()),
            (Some(min), None) => (min.as_mut_ptr(), std::ptr::null_mut()),
            (None, Some(max)) => (std::ptr::null_mut(), max.as_mut_ptr()),
            (None, None) => panic!("Either minimum or maximum should be Some(win)")
        };
        /*let mut other = dst.clone_owned();
        let (ptr_dst_min, ptr_dst_max) : (*mut u8, *mut u8) = match is_maximum {
            true => (other.full_window_mut().as_mut_ptr(), dst.as_mut_ptr()),
            false => (dst.as_mut_ptr(), other.full_window_mut().as_mut_ptr())
        };*/

        // println!("{:?}", block_sz);

        let block_size = crate::foreign::ipp::ippi::IppiSize  { width : block_sz.1 as i32, height : block_sz.0 as i32 };
        let src_ptr = win.as_ptr();
        let ans = if win.pixel_is::<u8>() {
            let mut global_min = 0u8;
            let mut global_max = 0u8;
            crate::foreign::ipp::ippi::ippiBlockMinMax_8u_C1R(
                mem::transmute(src_ptr),
                src_step,
                src_sz,
                mem::transmute(ptr_dst_min),
                dst_step,
                mem::transmute(ptr_dst_max),
                dst_step,
                block_size,
                &mut global_min as *mut _,
                &mut global_max as *mut _
            )
        } else if win.pixel_is::<f32>() {
            let mut global_min = 0.0f32;
            let mut global_max = 0.0f32;
            crate::foreign::ipp::ippi::ippiBlockMinMax_32f_C1R(
                mem::transmute(src_ptr),
                src_step,
                src_sz,
                mem::transmute(ptr_dst_min),
                dst_step,
                mem::transmute(ptr_dst_max),
                dst_step,
                block_size,
                &mut global_min as *mut _,
                &mut global_max as *mut _
            )
        } else {
            unimplemented!()
        };
        assert!(ans == 0);

        // println!("{} {}", global_min, global_max);
        return;
    }

    unimplemented!()

}

// This returns the minimum and maximum values and indices.
// Can be applied block-wise to get indices at many points.
pub fn min_max_idx<N>(
    win : &Window<N>,
    min : bool,
    max : bool
) -> (Option<(usize, usize, N)>, Option<(usize, usize, N)>)
where
    N : Pixel + AsPrimitive<f32>,
    f32 : AsPrimitive<N>,
    for<'a> &'a [N] : Storage<N>,
    for<'a> &'a mut [N] : StorageMut<N>,
{

    #[cfg(feature="ipp")]
    unsafe {

        use crate::foreign::ipp::ippcv::IppiPoint;

        let (step, sz) = crate::image::ipputils::step_and_size_for_window(win);
        if min && max {
            let (mut min, mut max) : (f32, f32) = (0., 0.);
            let (mut min_ix, mut max_ix) : (IppiPoint, IppiPoint) = (
                IppiPoint { x : 0, y : 0 },
                IppiPoint { x : 0, y : 0 }
            );
            let ans = if win.pixel_is::<u8>() {
                crate::foreign::ipp::ippcv::ippiMinMaxIndx_8u_C1R(
                    mem::transmute(win.as_ptr()),
                    step,
                    mem::transmute(sz),
                    &mut min as *mut _,
                    &mut max as *mut _,
                    &mut min_ix as *mut _,
                    &mut max_ix as *mut _
                )
            } else if win.pixel_is::<f32>() {
                crate::foreign::ipp::ippcv::ippiMinMaxIndx_32f_C1R(
                    mem::transmute(win.as_ptr()),
                    step,
                    mem::transmute(sz),
                    &mut min as *mut _,
                    &mut max as *mut _,
                    &mut min_ix as *mut _,
                    &mut max_ix as *mut _
                )
            } else {
                panic!("Invalid type");
            };
            assert!(ans == 0);
            return (
                Some((min_ix.y as usize, min_ix.x as usize, min.as_())),
                Some((max_ix.y as usize, max_ix.x as usize, max.as_()))
            );
        } else if min {
            let mut min_x : i32 = 0;
            let mut min_y : i32 = 0;
            if win.pixel_is::<u8>() {
                let mut min : u8 = 0;
                let ans = crate::foreign::ipp::ippi::ippiMinIndx_8u_C1R(
                    mem::transmute(win.as_ptr()),
                    step,
                    mem::transmute(sz),
                    &mut min as *mut _,
                    &mut min_x as *mut _,
                    &mut min_y as *mut _
                );
                assert!(ans == 0);
                let min_f : f32 = min.as_();
                return (Some((min_y as usize, min_x as usize, min_f.as_())), None);
            } else if win.pixel_is::<f32>() {
                let mut min : f32 = 0.;
                let ans = crate::foreign::ipp::ippi::ippiMinIndx_32f_C1R(
                    mem::transmute(win.as_ptr()),
                    step,
                    mem::transmute(sz),
                    &mut min as *mut _,
                    &mut min_x as *mut _,
                    &mut min_y as *mut _
                );
                assert!(ans == 0);
                return (Some((min_y as usize, min_x as usize, min.as_())), None);
            } else {
                panic!("Invalid type");
            }
        } else if max {
            let mut max_x : i32 = 0;
            let mut max_y : i32 = 0;
            if win.pixel_is::<u8>() {
                let mut max : u8 = 0;
                let ans = crate::foreign::ipp::ippi::ippiMaxIndx_8u_C1R(
                    mem::transmute(win.as_ptr()),
                    step,
                    mem::transmute(sz),
                    &mut max as *mut _,
                    &mut max_x as *mut _,
                    &mut max_y as *mut _
                );
                assert!(ans == 0);
                let max_f : f32 = max.as_();
                return (None, Some((max_y as usize, max_x as usize, max_f.as_())));
            } else if win.pixel_is::<f32>() {
                let mut max : f32 = 0.;
                let ans = crate::foreign::ipp::ippi::ippiMaxIndx_32f_C1R(
                    mem::transmute(win.as_ptr()),
                    step,
                    mem::transmute(sz),
                    &mut max as *mut _,
                    &mut max_x as *mut _,
                    &mut max_y as *mut _
                );
                assert!(ans == 0);
                return (None, Some((max_y as usize, max_x as usize, max.as_())));
            } else {
                panic!("Invalid type");
            }
        }
    }

    unimplemented!()
}

pub fn find_peaks(
    win : &Window<'_, i32>, 
    threshold : i32, 
    max_peaks : usize
) -> Vec<(usize, usize)> {

    #[cfg(feature="ipp")]
    unsafe {

        use crate::foreign::ipp::ippi::IppiPoint;

        let (step, sz) = crate::image::ipputils::step_and_size_for_window(win);
        let mut pts : Vec<IppiPoint> = (0..max_peaks).map(|_| IppiPoint { x : 0, y : 0} ).collect();
        let mut n_peaks = 0;

        // This uses max-norm withh 8-neighborhood. Alternatively use L1 norm for absolute-max norm
        // at 4-neighborhood.
        let norm = crate::foreign::ipp::ippi::_IppiNorm_ippiNormInf;
        let border = 1;

        let mut buf_sz = 0;
        let ans = crate::foreign::ipp::ippcv::ippiFindPeaks3x3GetBufferSize_32s_C1R(
            win.width() as i32,
            &mut buf_sz as *mut _
        );
        assert!(ans == 0);

        let mut buffer = Vec::from_iter((0..buf_sz).map(|_| 0u8 ));
        let ans = crate::foreign::ipp::ippcv::ippiFindPeaks3x3_32s_C1R(
            win.as_ptr(),
            step,
            mem::transmute(sz),
            threshold,
            mem::transmute(pts.as_mut_ptr()),
            max_peaks as i32,
            &mut n_peaks as *mut _,
            mem::transmute(norm),
            border,
            buffer.as_mut_ptr()
        );
        assert!(ans == 0);

        let out_pts : Vec<_> = pts.iter()
            .take(n_peaks as usize)
            .map(|pt| (pt.y as usize, pt.x as usize) )
            .collect();
        return out_pts;
    }

    unimplemented!()
}

/// Pool operations are nonlinear local image transformation that replace each k x k region of an
/// image by a statistic stored at a corresponding downsampled version of the image.
pub trait Pool {

}

pub struct MinPool {

}

pub struct MaxPool {

}

pub struct MedianPool {

}

pub struct AvgPool {

}

pub fn local_min(src : Window<'_, u8>, dst : WindowMut<'_, u8>, kernel_sz : usize) -> ImageBuf<u8> {
    unimplemented!()
}

pub fn local_max(src : Window<'_, u8>, dst : WindowMut<'_, u8>, kernel_sz : usize) -> ImageBuf<u8> {
    unimplemented!()
}

/* cv2::sepFilter2D */
/* cv2::Laplacian */

#[cfg(feature="opencv")]
pub fn sobel(img : &Window<u8>, dst : WindowMut<'_, i16>, sz : usize, dx : i32, dy : i32) {
    use opencv::core;
    let src : core::Mat = img.into();
    let mut dst : core::Mat = dst.into();
    opencv::imgproc::sobel(&src, &mut dst, core::CV_16S, dx, dy, sz as i32, 1.0, 0.0, core::BORDER_DEFAULT).unwrap();
}

pub fn median_filter(src : &Window<u8>, dst : &mut WindowMut<u8>, mask_sz : usize) {

    assert_eq!(src.shape(), dst.shape());

    #[cfg(feature="ipp")]
    unsafe {
        let (src_step, roi_sz) = crate::image::ipputils::step_and_size_for_window(src);
        let (dst_step, _) = crate::image::ipputils::step_and_size_for_window_mut(&dst);
        let mask_sz = crate::foreign::ipp::ippi::IppiSize { width : mask_sz as i32, height : mask_sz as i32 };
        let mut buf_sz : i32 = 0;
        let num_channels = 1;
        let ans = crate::foreign::ipp::ippi::ippiFilterMedianBorderGetBufferSize(
            roi_sz,
            mask_sz,
            crate::foreign::ipp::ippi::IppDataType_ipp8u,
            num_channels,
            &mut buf_sz as *mut _
        );
        assert!(ans == 0);
        let mut buffer = Vec::<u8>::with_capacity(buf_sz as usize);
        buffer.set_len(buf_sz as usize);

        let anchor = crate::foreign::ipp::ippi::IppiPoint { x : mask_sz.width/2, y : mask_sz.height/2 };
        let border_val = 0;
        let border_ty = crate::foreign::ipp::ippi::_IppiBorderType_ippBorderConst;
        let ans = crate::foreign::ipp::ippi::ippiFilterMedianBorder_8u_C1R(
            src.as_ptr(),
            src_step,
            dst.as_mut_ptr(),
            dst_step,
            roi_sz,
            mask_sz,
            border_ty,
            border_val,
            buffer.as_mut_ptr()
        );
        assert!(ans == 0);
        return;
    }

    unimplemented!()
}

/*IppStatus ippiFilterBilateral_<mod>(const Ipp<srcdatatype>* pSrc, int srcStep,
Ipp<dstdatatype>* pDst, int dstStep, IppiSize dstRoiSize, IppiBorderType borderType,
const Ipp<datatype> pBorderValue[1], const IppiFilterBilateralSpec* pSpec, Ipp8u*
pBuffer );

IppStatus ippiFilterBox_64f_C1R(const Ipp<datatype>* pSrc, int srcStep, Ipp<datatype>*
pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize, IppiPoint anchor );

IppStatus ippiFilterSeparable_<mod>(const Ipp<datatype>* pSrc, int srcStep,
Ipp<datatype>* pDst, int dstStep, IppiSize roiSize, IppiBorderType borderType,
Ipp<datatype> borderValue, const IppiFilterSeparableSpec* pSpec, Ipp8u* pBuffer );

IppStatus ippiConv_<mod>(const Ipp<datatype>* pSrc1, int src1Step, IppiSize src1Size,
const Ipp<datatype>* pSrc2, int src2Step, IppiSize src2Size, Ipp<datatype>* pDst, int
dstStep, int divisor, IppEnum algType, Ipp8u* pBuffer );

IppStatus ippiDeconvFFT_32f_C1R(const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int
dstStep, IppiSize roiSize, IppiDeconvFFTState_32f_C1R* pDeconvFFTState );

IppStatus ippiDeconvFFT_32f_C3R(const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int
dstStep, IppiSize roiSize, IppiDeconvFFTState_32f_C3R* pDeconvFFTState );

IppStatus ippiFilterSobel_<mod>(const Ipp<srcdatatype>* pSrc, int srcStep,
Ipp<dstdatatype>* pDst, int dstStep, IppiSize dstRoiSize, IppiMaskSize maskSize,
IppNormType normType, IppiBorderType borderType, Ipp<srcdatatype> borderValue, Ipp8u*
pBuffer );

IppStatus ippiFilterGaussian_<mod>(const Ipp<datatype>* pSrc, int srcStep,
Ipp<datatype>* pDst, int dstStep, IppiSize roiSize, IppiBorderType borderType, const
Ipp<datatype> borderValue[1], IppFilterGaussianSpec* pSpec, Ipp8u* pBuffer );

IppStatus ippiDCTFwd_<mod> (const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
const IppiDCTFwdSpec_32f* pDCTSpec, Ipp8u* pBuffer );

IppStatus ippiDCTInv_<mod> (const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
const IppiDCTInvSpec_32f* pDCTSpec, Ipp8u* pBuffer );

IppStatus ippiDCT8x8Fwd_<mod>(const Ipp<datatype>* pSrc, Ipp<datatype>* pDst );

IppStatus ippiDCT8x8Inv_2x2_16s_C1(const Ipp16s* pSrc, Ipp16s* pDst );

// imgproc::blur
// imgproc::median_blur
// imgproc::sobel

/* pub fn scharr(
    src: &dyn ToInputArray,
    dst: &mut dyn ToOutputArray,
    ddepth: i32,
    dx: i32,
    dy: i32,
    scale: f64,
    delta: f64,
    border_type: i32
) -> Result<()>
*/

// TODO use crate edge_detection::canny(
// TODO use image_conv::convolution(&img, filter, 1, PaddingType::UNIFORM(1)); with custom filters.
// mss_saliency = "1.0.6" for salient portion extraction.
*/

// Separability of Kernels (Szelisky, p. 116, after Perona, 1995). If SVD of the 2d kernel matrix has only
// the first singular value not zeroed, then the Kernel is separable into \sqrt \sigma_0 u_0 (vertical)
// and \sqrt \sigma_0 v_0^T (horizontal) kernels.

/*
Median filter can be considered as misnomer, because it does not imply convolution.
Therefore a better API would be calling it "median pooling", for the pooling parameter
being 1 actually means no pooling is applied (image size is preserved).
*/
