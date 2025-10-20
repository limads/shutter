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

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum Axis {
    Vertical,
    Horizontal,
    Both
}

#[cfg(feature="ipp")]
impl<S : Storage<u8>> Image<u8, S> {

    pub fn filter_sobel(&self, axs : Axis, mask_sz : usize) -> ImageBuf<i16> {
        let mut dst = ImageBuf::<i16>::new_constant(self.height(), self.width(), 0);
        match axs {
            Axis::Both => {
                let mut sobel = IppFilterSobel::new((self.height(), self.width()), mask_sz);
                sobel.apply(self, &mut dst);
            },
            Axis::Vertical => {
                let mut sobel = IppFilterSobelVert::new((self.height(), self.width()), mask_sz);
                sobel.apply(self, &mut dst);
            },
            Axis::Horizontal => {
                let mut sobel = IppFilterSobelHoriz::new((self.height(), self.width()), mask_sz);
                sobel.apply(self, &mut dst);
            }
        }
        dst
    }

}

#[cfg(feature="ipp")]
impl<S : Storage<u8>> Image<u8, S> {

    pub fn filter_maximum(&self, height : usize, width : usize) -> ImageBuf<u8> {
        let mut dst = ImageBuf::new_constant_like(&self, 0);
        crate::local::IppiFilterMinMax::new(self.height(), self.width(), (height, width), false)
            .apply(&self, &mut dst);
        dst
    }

    pub fn filter_minimum(&self, height : usize, width : usize) -> ImageBuf<u8> {
        let mut dst = ImageBuf::new_constant_like(&self, 0);
        crate::local::IppiFilterMinMax::new(self.height(), self.width(), (height, width), true)
            .apply(&self, &mut dst);
        dst
    }

    pub fn filter_box(&self, height : usize, width : usize) -> ImageBuf<u8> {
        let mut dst = ImageBuf::new_constant_like(&self, 0);
        crate::local::IppiFilterBox::new(self.height(), self.width(), (height, width))
            .apply(&self, &mut dst);
        dst
    }

    pub fn filter_sobel_abs(&self, axis : Axis, mask_sz : usize) -> ImageBuf<u8> {
        let mut sobel_abs = self.clone_owned();
        let sobel = self.filter_sobel(axis, mask_sz);
        sobel.abs_convert_to(&mut sobel_abs);
        sobel_abs
    }

    pub fn filter_weiner(&self) -> ImageBuf<u8> {
        // let mut sobel =
        unimplemented!()
    }

    pub fn filter_median(&self) -> ImageBuf<u8> {
        unimplemented!()
    }

}

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
        hist[*win.linear_index(m).unwrap() as usize] += 1;

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

#[cfg(feature="ipp")]
#[derive(Debug, Clone)]
pub struct IppSumWindow {
    buf : Vec<u8>,
    mask_sz : crate::foreign::ipp::ippi::IppiSize,
    dst_sz : crate::foreign::ipp::ippi::IppiSize,
}

#[cfg(feature="ipp")]
impl IppSumWindow {

    pub fn new(
        src_step : usize,
        src_sz : (usize, usize),
        dst_step : usize,
        dst_sz : (usize, usize)
    ) -> Self {
        let local_win_sz = (src_sz.0 / dst_sz.0, src_sz.1 / dst_sz.1);
        let (src_step, src_sz) = crate::image::ipputils::step_and_size_for_tuple::<u8>(src_step, src_sz);
        let (dst_step, dst_sz) = crate::image::ipputils::step_and_size_for_tuple::<i32>(dst_step, dst_sz);
        let mask_sz = crate::foreign::ipp::ippi::IppiSize {
            width : local_win_sz.1 as i32,
            height : local_win_sz.0 as i32
        };
        let mut num_channels = 1;
        let mut buf_sz : i32 = 0;
        unsafe {
            let ans = crate::foreign::ipp::ippi::ippiSumWindowGetBufferSize(
                dst_sz.clone(),
                mask_sz.clone(),
                crate::foreign::ipp::ippi::IppDataType_ipp8u,
                num_channels,
                &mut buf_sz as *mut _
            );
            assert!(ans == 0);
            assert!(buf_sz > 0);
            let mut buf = Vec::from_iter((0..buf_sz).map(|_| 0u8 ));
            Self { buf, mask_sz, dst_sz }
        }
    }

    pub fn calculate<S, T>(&mut self, src : &Image<u8,S>, dst : &mut Image<i32, T>)
    where
        S : Storage<u8>,
        T : StorageMut<i32>
    {
        let border_val = 0u8;
        let (src_step, src_sz) = crate::image::ipputils::step_and_size_for_image(&src);
        let (dst_step, dst_sz) = crate::image::ipputils::step_and_size_for_image(&*dst);
        assert!(dst_sz.width == self.dst_sz.width && dst_sz.height == self.dst_sz.height);
        unsafe {
            let ans = crate::foreign::ipp::ippi::ippiSumWindow_8u32s_C1R(
                src.as_ptr(),
                src_step,
                dst.as_mut_ptr(),
                dst_step,
                dst_sz,
                self.mask_sz,
                crate::foreign::ipp::ippi::_IppiBorderType_ippBorderConst,
                &border_val,
                self.buf.as_mut_ptr()
            );
            assert!(ans == 0);
            return;
        }
    }
    
}

impl<'a> Window<'a, u8> {

    pub fn vectorized_local_max_to(&self, dst: &mut WindowMut<u8>) {
        assert!(self.width() % dst.width() == 0);
        assert!(self.height() % dst.height() == 0);
        let block_sz = (self.height() / dst.height(), self.width() / dst.width());
        if self.width() % 32 == 0 {
            for (dst_px, w) in dst.pixels_mut(1).zip(self.windows((block_sz.0, block_sz.1))) {
                let mut maximum = wide::u8x32::splat(0);
                for pxs in w.packed_pixels_32().unwrap() {
                    maximum = maximum.max(pxs);
                }
                *dst_px = *maximum.to_array().iter().max_by(|a, b| a.cmp(&b) ).unwrap();
            }
            return;
        }
        unimplemented!()
    }

    pub fn vectorized_local_min_to(&self, dst: &mut WindowMut<u8>) {
        assert!(self.width() % dst.width() == 0);
        assert!(self.height() % dst.height() == 0);
        let block_sz = (self.height() / dst.height(), self.width() / dst.width());
        if self.width() % 32 == 0 {
            for (dst_px, w) in dst.pixels_mut(1).zip(self.windows((block_sz.0, block_sz.1))) {
                let mut minimum = wide::u8x32::splat(255);
                for pxs in w.packed_pixels_32().unwrap() {
                    minimum = minimum.min(pxs);
                }
                *dst_px = *minimum.to_array().iter().min_by(|a, b| a.cmp(&b) ).unwrap();
            }
            return;
        }
        unimplemented!()
    }

}

pub fn local_sum(src : &Window<'_, u8>, dst : &mut WindowMut<'_, i32>) {

    assert!(src.height() % dst.height() == 0);
    assert!(src.width() % dst.width() == 0);

    #[cfg(feature="ipp")]
    unsafe {
        let mut sw = IppSumWindow::new(src.original_width(), src.size(), dst.original_width(), dst.size());
        sw.calculate(src, dst);
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

impl<N, S> Image<N, S>
where
    S : Storage<N>,
    N : Pixel
{

    // Local (block-wise minima)
    pub fn local_minima_to<T : StorageMut<N>>(&self, min : &mut Image<N, T>) {
        block_min_or_max(&self.full_window(), Some(&mut min.full_window_mut()), None);
    }

    // Local (block-wise maxima)
    pub fn local_maxima_to<T : StorageMut<N>>(&self, max : &mut Image<N, T>) {
        block_min_or_max(&self.full_window(), None, Some(&mut max.full_window_mut()));
    }

    // Local (block-wise minima and maxima).
    pub fn local_extrema_to<T : StorageMut<N>, U : StorageMut<N>>(
        &self,
        min : &mut Image<N, T>,
        max : &mut Image<N, U>
    ) {
        block_min_or_max(&self.full_window(), Some(&mut min.full_window_mut()), Some(&mut max.full_window_mut()));
    }

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

pub fn masked_min_max_idx<N, S, T>(
    win : &Image<N, S>,
    mask : &Image<u8, T>,
    // min : bool,
    // max : bool
) -> (Option<(usize, usize, N)>, Option<(usize, usize, N)>)
where
    N : Pixel + AsPrimitive<f32>,
    f32 : AsPrimitive<N>,
    S : Storage<N>,
    T : Storage<u8>
{
    #[cfg(feature="ipp")]
    unsafe {
        use crate::foreign::ipp::ippcv::IppiPoint;
        let (step, sz) = crate::image::ipputils::step_and_size_for_image(win);
        // if min && max {
        let (mut min, mut max) : (f32, f32) = (0., 0.);
        let (mut min_ix, mut max_ix) : (IppiPoint, IppiPoint) = (
            IppiPoint { x : 0, y : 0 },
            IppiPoint { x : 0, y : 0 }
        );
        if win.pixel_is::<u8>() {
            let ans = crate::foreign::ipp::ippcv::ippiMinMaxIndx_8u_C1MR(
                mem::transmute(win.as_ptr()),
                win.byte_stride() as i32,
                mask.as_ptr(),
                mask.byte_stride() as i32,
                mem::transmute(sz),
                &mut min as *mut _,
                &mut max as *mut _,
                &mut min_ix as *mut _,
                &mut max_ix as *mut _
            );
            // assert!(ans == 0, "{}", ans);
            return (
                Some((min_ix.y as usize, min_ix.x as usize, min.as_())),
                Some((max_ix.y as usize, max_ix.x as usize, max.as_()))
            );
        } else {
            unimplemented!()
        }
        /*} else if min {
            let mut min_x : i32 = 0;
            let mut min_y : i32 = 0;
            if win.pixel_is::<u8>() {
                let mut min : u8 = 0;
                let ans = crate::foreign::ipp::ippi::ippiMinIndx_8u_C1MR(
                    mem::transmute(win.as_ptr()),
                    step,
                    mask.as_ptr(),
                    mask.byte_stride() as i32,
                    mem::transmute(sz),
                    &mut min as *mut _,
                    &mut min_x as *mut _,
                    &mut min_y as *mut _
                );
                assert!(ans == 0);
                let min_f : f32 = min.as_();
                return (Some((min_y as usize, min_x as usize, min_f.as_())), None);
            } else {
                unimplemented!()
            }
        } else if max {
            let mut max_x : i32 = 0;
            let mut max_y : i32 = 0;
            if win.pixel_is::<u8>() {
                let mut max : u8 = 0;
                let ans = crate::foreign::ipp::ippi::ippiMaxIndx_8u_C1MR(
                    mem::transmute(win.as_ptr()),
                    step,
                    mask.as_ptr(),
                    mask.byte_stride() as i32,
                    mem::transmute(sz),
                    &mut max as *mut _,
                    &mut max_x as *mut _,
                    &mut max_y as *mut _
                );
                assert!(ans == 0);
                let max_f : f32 = max.as_();
                return (None, Some((max_y as usize, max_x as usize, max_f.as_())));
            } else {
                unimplemented!()
            }
        } else {
            panic!("Invalid operation");
        }*/
    }
}

impl<N, S> Image<N, S>
where
    N : Pixel + num_traits::Bounded + num_traits::Zero + PartialOrd,
    f32 : AsPrimitive<N>,
    S : Storage<N>,
    N : AsPrimitive<f32>
{

    pub fn indexed_minimum(&self) -> ((usize, usize), N) {
        let (Some((r, c, val)), _) = min_max_idx(&self.full_window(), true, false) else { panic!() };
        ((r, c), val)
    }

    pub fn indexed_maximum(&self) -> ((usize, usize), N) {
        let (_, Some((r, c, val))) = min_max_idx(&self.full_window(), false, true) else { panic!() };
        ((r, c), val)
    }

    pub fn masked_indexed_extrema<T:Storage<u8>>(&self, mask : &Image<u8, T>) -> (((usize, usize), N), ((usize, usize), N)) {
        let (Some((rmin, cmin, valmin)), Some((rmax, cmax, valmax))) = masked_min_max_idx(&self.full_window(), mask) else {
            panic!()
        };
        (((rmin, cmin), valmin), ((rmax, cmax), valmax))
    }

    pub fn indexed_extrema(&self) -> (((usize, usize), N), ((usize, usize), N)) {
        let (Some((rmin, cmin, valmin)), Some((rmax, cmax, valmax))) = min_max_idx(&self.full_window(), true, true) else {
            panic!()
        };
        (((rmin, cmin), valmin), ((rmax, cmax), valmax))
    }
}

fn baseline_min_max_idx<N>(win : &Window<N>) -> (Option<(usize, usize, N)>, Option<(usize, usize, N)>)
where
    N : Pixel + num_traits::Bounded + num_traits::Zero + PartialOrd
{
    let mut i = 0;
    let mut min = N::max_value();
    let mut max = N::min_value();
    let mut min_ix = 0;
    let mut max_ix = 0;
    for px in win.pixels(1) {
        if *px < min {
            min = *px;
            min_ix = i;
        }
        if *px > max {
            max = *px;
            max_ix = i;
        }
        i += 1;
    }
    let (nr, nc) = win.size();
    (
        Some((min_ix / nr, min_ix % nc, min)),
        Some((max_ix / nr, max_ix % nc, max))
    )
}

// This returns the minimum and maximum values and indices.
// Can be applied block-wise to get indices at many points.
pub fn min_max_idx<N>(
    win : &Window<N>,
    min : bool,
    max : bool
) -> (Option<(usize, usize, N)>, Option<(usize, usize, N)>)
where
    N : Pixel + AsPrimitive<f32> + PartialOrd,
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
                return baseline_min_max_idx(win);
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
                return baseline_min_max_idx(win);
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
                return baseline_min_max_idx(win);
            }
        } else {
            panic!("Either min or max must be informed")
        }
    }
    baseline_min_max_idx(win)
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

#[cfg(feature="ipp")]
#[derive(Debug, Clone)]
pub struct IppiFilterBoxF32 {
    buf : Vec<u8>,
    mask_sz : (usize, usize)
}

#[cfg(feature="ipp")]
impl IppiFilterBoxF32 {

    pub fn new(height : usize, width : usize, mask_sz : (usize, usize)) -> Self {
        let dst_size = crate::foreign::ipp::ippi::IppiSize::from((height, width));
        let mask_size = crate::foreign::ipp::ippi::IppiSize::from(mask_sz);
        let num_channels = 1;
        let data_ty = crate::foreign::ipp::ippi::IppDataType_ipp32f;
        let mut buf_sz = 0;
        unsafe {
            let ans = crate::foreign::ipp::ippi::ippiFilterBoxBorderGetBufferSize(
                dst_size,
                mask_size,
                data_ty,
                num_channels,
                &mut buf_sz as *mut _
            );
            assert!(buf_sz > 0);
            assert!(ans == 0);
            let mut buf = Vec::<u8>::with_capacity(buf_sz as usize);
            buf.set_len(buf_sz as usize);
            Self { buf, mask_sz }
        }
    }

    pub fn apply<S, T>(&mut self, src : &Image<f32, S>, dst : &mut Image<f32, T>)
    where
        S : Storage<f32>,
        T : StorageMut<f32>
    {
        let border_val : f32 = 0.0;
        // let border_ty = crate::foreign::ipp::ippi::_IppiBorderType_ippBorderConst;
        let border_ty = crate::foreign::ipp::ippi::_IppiBorderType_ippBorderRepl;
        unsafe {
            let ans = crate::foreign::ipp::ippi::ippiFilterBoxBorder_32f_C1R(
                src.as_ptr(),
                src.byte_stride() as i32,
                dst.as_mut_ptr(),
                dst.byte_stride() as i32,
                dst.size().into(),
                self.mask_sz.into(),
                border_ty,
                &border_val as *const _,
                self.buf.as_mut_ptr() as *mut _
            );
            assert!(ans == 0);
        }
    }

}

#[cfg(feature="ipp")]
#[derive(Debug, Clone)]
pub struct IppiFilterBox {
    buf : Vec<u8>,
    mask_sz : (usize, usize)
}

#[cfg(feature="ipp")]
impl IppiFilterBox {

    pub fn new(height : usize, width : usize, mask_sz : (usize, usize)) -> Self {
        let dst_size = crate::foreign::ipp::ippi::IppiSize::from((height, width));
        let mask_size = crate::foreign::ipp::ippi::IppiSize::from(mask_sz);
        let num_channels = 1;
        let data_ty = crate::foreign::ipp::ippi::IppDataType_ipp8u;
        let mut buf_sz = 0;
        unsafe {
            let ans = crate::foreign::ipp::ippi::ippiFilterBoxBorderGetBufferSize(
                dst_size,
                mask_size,
                data_ty,
                num_channels,
                &mut buf_sz as *mut _
            );
            assert!(buf_sz > 0);
            assert!(ans == 0);
            let mut buf = Vec::<u8>::with_capacity(buf_sz as usize);
            buf.set_len(buf_sz as usize);
            Self { buf, mask_sz }
        }
    }

    pub fn apply<S, T>(&mut self, src : &Image<u8, S>, dst : &mut Image<u8, T>)
    where
        S : Storage<u8>,
        T : StorageMut<u8>
    {
        let border_val : u8 = 0;
        // let border_ty = crate::foreign::ipp::ippi::_IppiBorderType_ippBorderConst;
        let border_ty = crate::foreign::ipp::ippi::_IppiBorderType_ippBorderRepl;
        unsafe {
            let ans = crate::foreign::ipp::ippi::ippiFilterBoxBorder_8u_C1R(
                src.as_ptr(),
                src.byte_stride() as i32,
                dst.as_mut_ptr(),
                dst.byte_stride() as i32,
                dst.size().into(),
                self.mask_sz.into(),
                border_ty,
                &border_val as *const _,
                self.buf.as_mut_ptr() as *mut _
            );
            assert!(ans == 0);
        }
    }

}

#[cfg(feature="ipp")]
#[derive(Debug, Clone)]
pub struct IppiFilterMinMax {
    buf : Vec<u8>,
    mask_sz : (usize, usize),
    is_min : bool
}

#[cfg(feature="ipp")]
impl IppiFilterMinMax {

    pub fn new(height : usize, width : usize, mask_sz : (usize, usize), is_min : bool) -> Self {
        let dst_size = crate::foreign::ipp::ippi::IppiSize::from((height, width));
        let mask_size = crate::foreign::ipp::ippi::IppiSize::from(mask_sz);
        let num_channels = 1;
        let data_ty = crate::foreign::ipp::ippi::IppDataType_ipp8u;
        let mut buf_sz = 0;
        unsafe {
            let ans = if is_min {
                crate::foreign::ipp::ippi::ippiFilterMinBorderGetBufferSize(
                    dst_size,
                    mask_size,
                    data_ty,
                    num_channels,
                    &mut buf_sz as *mut _
                )
            } else {
                crate::foreign::ipp::ippi::ippiFilterMaxBorderGetBufferSize(
                    dst_size,
                    mask_size,
                    data_ty,
                    num_channels,
                    &mut buf_sz as *mut _
                )
            };
            assert!(buf_sz > 0);
            assert!(ans == 0);
            let mut buf = Vec::<u8>::with_capacity(buf_sz as usize);
            buf.set_len(buf_sz as usize);
            Self { buf, mask_sz, is_min }
        }
    }

    pub fn apply<S, T>(&mut self, src : &Image<u8, S>, dst : &mut Image<u8, T>)
    where
        S : Storage<u8>,
        T : StorageMut<u8>
    {
        let border_val : u8 = 0;
        // let border_ty = crate::foreign::ipp::ippi::_IppiBorderType_ippBorderConst;
        let border_ty = crate::foreign::ipp::ippi::_IppiBorderType_ippBorderRepl;
        unsafe {
            let ans = if self.is_min {
                crate::foreign::ipp::ippi::ippiFilterMinBorder_8u_C1R(
                    src.as_ptr(),
                    src.byte_stride() as i32,
                    dst.as_mut_ptr(),
                    dst.byte_stride() as i32,
                    dst.size().into(),
                    self.mask_sz.into(),
                    border_ty,
                    border_val,
                    self.buf.as_mut_ptr() as *mut _
                )
            } else {
                crate::foreign::ipp::ippi::ippiFilterMaxBorder_8u_C1R(
                    src.as_ptr(),
                    src.byte_stride() as i32,
                    dst.as_mut_ptr(),
                    dst.byte_stride() as i32,
                    dst.size().into(),
                    self.mask_sz.into(),
                    border_ty,
                    border_val,
                    self.buf.as_mut_ptr() as *mut _
                )
            };
            assert!(ans == 0);
        }
    }

}

#[cfg(feature="ipp")]
pub fn local_min_filter(src : &Window<'_, u8>, dst : &mut WindowMut<'_, u8>, kernel_sz : (usize, usize)) {
    IppiFilterMinMax::new(src.height(), src.width(), kernel_sz, true).apply(src, dst);
}

#[cfg(feature="ipp")]
pub fn local_max_filter(src : &Window<'_, u8>, dst : &mut WindowMut<'_, u8>, kernel_sz : (usize, usize)) {
    IppiFilterMinMax::new(src.height(), src.width(), kernel_sz, false).apply(src, dst);
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

#[cfg(feature="ipp")]
#[derive(Debug, Clone)]
pub struct IppFilterMedian {
    buffer : Vec<u8>,
    mask_sz : (usize, usize)
}

#[cfg(feature="ipp")]
impl IppFilterMedian {

    pub fn new(height : usize, width : usize, mask_sz : (usize, usize)) -> Self {
        let mut buf_sz : i32 = 0;
        let num_channels = 1;
        unsafe {
            let ans = crate::foreign::ipp::ippi::ippiFilterMedianBorderGetBufferSize(
                (height, width).into(),
                mask_sz.into(),
                crate::foreign::ipp::ippi::IppDataType_ipp8u,
                num_channels,
                &mut buf_sz as *mut _
            );
            assert!(ans == 0);
            let mut buffer = Vec::<u8>::with_capacity(buf_sz as usize);
            buffer.set_len(buf_sz as usize);
            Self { buffer, mask_sz }
        }
    }

    pub fn apply<S, T>(&mut self, src : &Image<u8, S>, dst : &mut Image<u8, T>)
    where
        S : Storage<u8>,
        T : StorageMut<u8>
    {
        assert_eq!(src.shape(), dst.shape());
        let anchor = crate::foreign::ipp::ippi::IppiPoint { x : (self.mask_sz.1 as i32)/2, y : (self.mask_sz.0 as i32) / 2 };
        let border_val = 0;
        // let border_ty = crate::foreign::ipp::ippi::_IppiBorderType_ippBorderConst;
        let border_ty = crate::foreign::ipp::ippi::_IppiBorderType_ippBorderRepl;
        unsafe {
            let ans = crate::foreign::ipp::ippi::ippiFilterMedianBorder_8u_C1R(
                src.as_ptr(),
                src.byte_stride() as i32,
                dst.as_mut_ptr(),
                dst.byte_stride() as i32,
                src.size().into(),
                self.mask_sz.into(),
                border_ty,
                border_val,
                self.buffer.as_mut_ptr()
            );
            assert!(ans == 0);
        }
    }

}

pub fn median_filter(src : &Window<u8>, dst : &mut WindowMut<u8>, mask_sz : usize) {

    /*assert_eq!(src.shape(), dst.shape());

    #[cfg(feature="ipp")]
    unsafe {
        let (src_step, roi_sz) = crate::image::ipputils::step_and_size_for_window(src);
        let (dst_step, _) = crate::image::ipputils::step_and_size_for_window_mut(&dst);
        let mask_sz = crate::foreign::ipp::ippi::IppiSize { width : mask_sz as i32, height : mask_sz as i32 };
        return;
    }*/

    unimplemented!()
}

#[derive(Clone, Debug)]
#[cfg(feature="ipp")]
pub struct IppiFilterGauss {
    spec : Vec<u8>,
    buffer : Vec<u8>
}

#[cfg(feature="ipp")]
impl IppiFilterGauss {

    // range_sq: Smooth based on pixel intensity difference
    // geom_sq : Smooth based on distance between pixels.
    pub fn new(height : usize, width : usize, kernel_side : usize, sigma : f32) -> Self {
        let mut buf_sz : i32 = 0;
        let mut spec_sz : i32 = 0;
        let num_channels = 1;
        let data_ty = crate::foreign::ipp::ippi::IppDataType_ipp8u;
        unsafe {
            let ans = crate::foreign::ipp::ippcv::ippiFilterGaussianGetBufferSize(
                (height, width).into(),
                kernel_side as u32,
                data_ty,
                num_channels,
                &mut spec_sz as *mut _,
                &mut buf_sz as *mut _,
            );
            assert!(ans == 0);
            assert!(spec_sz > 0);
            assert!(buf_sz > 0);
            let mut buffer = Vec::<u8>::with_capacity(buf_sz as usize);
            buffer.set_len(buf_sz as usize);
            let mut spec = Vec::<u8>::with_capacity(spec_sz as usize);
            spec.set_len(spec_sz as usize);
            let border_val : u8 = 0;
            // let border_ty = crate::foreign::ipp::ippi::_IppiBorderType_ippBorderConst;
            let border_ty = crate::foreign::ipp::ippi::_IppiBorderType_ippBorderRepl;
            let ans = crate::foreign::ipp::ippcv::ippiFilterGaussianInit(
                (height, width).into(),
                kernel_side as u32,
                sigma,
                border_ty,
                data_ty,
                num_channels,
                spec.as_mut_ptr() as *mut _,
                buffer.as_mut_ptr() as *mut _
            );
            assert!(ans == 0);
            Self { spec, buffer }
        }
    }

    pub fn apply<S, T>(&mut self, src : &Image<u8, S>, dst : &mut Image<u8, T>)
    where
        S : Storage<u8>,
        T : StorageMut<u8>
    {
        unsafe {
            let border_val : u8 = 0;
            let border_ty = crate::foreign::ipp::ippcv::_IppiBorderType_ippBorderConst;
            let ans = crate::foreign::ipp::ippcv::ippiFilterGaussianBorder_8u_C1R(
                src.as_ptr(),
                src.byte_stride() as i32,
                dst.as_mut_ptr(),
                dst.byte_stride() as i32,
                src.size().into(),
                border_val,
                self.spec.as_mut_ptr() as *mut _,
                self.buffer.as_mut_ptr() as *mut _
            );
            assert!(ans == 0);
        }
    }

}

#[derive(Clone, Debug)]
#[cfg(feature="ipp")]
pub struct IppiFilterGaussF32 {
    spec : Vec<u8>,
    buffer : Vec<u8>
}

#[cfg(feature="ipp")]
impl IppiFilterGaussF32 {

    // range_sq: Smooth based on pixel intensity difference
    // geom_sq : Smooth based on distance between pixels.
    pub fn new(height : usize, width : usize, kernel_side : usize, sigma : f32) -> Self {
        let mut buf_sz : i32 = 0;
        let mut spec_sz : i32 = 0;
        let num_channels = 1;
        let data_ty = crate::foreign::ipp::ippi::IppDataType_ipp32f;
        unsafe {
            let ans = crate::foreign::ipp::ippcv::ippiFilterGaussianGetBufferSize(
                (height, width).into(),
                kernel_side as u32,
                data_ty,
                num_channels,
                &mut spec_sz as *mut _,
                &mut buf_sz as *mut _,
            );
            assert!(ans == 0);
            assert!(spec_sz > 0);
            assert!(buf_sz > 0);
            let mut buffer = Vec::<u8>::with_capacity(buf_sz as usize);
            buffer.set_len(buf_sz as usize);
            let mut spec = Vec::<u8>::with_capacity(spec_sz as usize);
            spec.set_len(spec_sz as usize);
            let border_val : u8 = 0;
            // let border_ty = crate::foreign::ipp::ippi::_IppiBorderType_ippBorderConst;
            let border_ty = crate::foreign::ipp::ippi::_IppiBorderType_ippBorderRepl;
            let ans = crate::foreign::ipp::ippcv::ippiFilterGaussianInit(
                (height, width).into(),
                kernel_side as u32,
                sigma,
                border_ty,
                data_ty,
                num_channels,
                spec.as_mut_ptr() as *mut _,
                buffer.as_mut_ptr() as *mut _
            );
            assert!(ans == 0);
            Self { spec, buffer }
        }
    }

    pub fn apply<S, T>(&mut self, src : &Image<f32, S>, dst : &mut Image<f32, T>)
    where
        S : Storage<f32>,
        T : StorageMut<f32>
    {
        unsafe {
            let border_val : f32 = 0.0;
            let border_ty = crate::foreign::ipp::ippcv::_IppiBorderType_ippBorderConst;
            let ans = crate::foreign::ipp::ippcv::ippiFilterGaussianBorder_32f_C1R(
                src.as_ptr(),
                src.byte_stride() as i32,
                dst.as_mut_ptr(),
                dst.byte_stride() as i32,
                src.size().into(),
                border_val,
                self.spec.as_mut_ptr() as *mut _,
                self.buffer.as_mut_ptr() as *mut _
            );
            assert!(ans == 0);
        }
    }

}

#[derive(Debug, Clone)]
#[cfg(feature="ipp")]
pub struct IppiFilterBilateral {
    spec : Vec<u8>,
    buffer : Vec<u8>
}

#[cfg(feature="ipp")]
impl IppiFilterBilateral {

    // range_sq: Smooth based on pixel intensity difference
    // geom_sq : Smooth based on distance between pixels.
    pub fn new(height : usize, width : usize, kernel_side : usize, range_sq : f64, geom_sq : f64) -> Self {
        // let filt_ty = crate::foreign::ipp::ippi::IppiFilterBilateralType_ippiFilterBilateralGaussFast;
        let filt_ty = crate::foreign::ipp::ippi::IppiFilterBilateralType_ippiFilterBilateralGauss;
        let mut buf_sz : i32 = 0;
        let mut spec_sz : i32 = 0;
        let num_channels = 1;
        let data_ty = crate::foreign::ipp::ippi::IppDataType_ipp8u;
        let dst_size = crate::foreign::ipp::ippi::IppiSize::from((height, width));
        let dist_method = crate::foreign::ipp::ippi::IppiDistanceMethodType_ippDistNormL1;
        unsafe {
            let ans = crate::foreign::ipp::ippi::ippiFilterBilateralGetBufferSize(
                filt_ty,
                dst_size,
                kernel_side as i32,
                data_ty,
                num_channels,
                dist_method,
                &mut spec_sz as *mut _,
                &mut buf_sz as *mut _,
            );
            assert!(ans == 0);
            assert!(spec_sz > 0);
            assert!(buf_sz > 0);
            let mut buffer = Vec::<u8>::with_capacity(buf_sz as usize);
            buffer.set_len(buf_sz as usize);
            let mut spec = Vec::<u8>::with_capacity(spec_sz as usize);
            spec.set_len(spec_sz as usize);
            let ans = crate::foreign::ipp::ippi::ippiFilterBilateralInit(
                filt_ty,
                dst_size,
                kernel_side as i32,
                data_ty,
                num_channels,
                dist_method,
                range_sq,
                geom_sq,
                spec.as_mut_ptr() as *mut _
            );
            assert!(ans == 0);
            Self { spec, buffer }
        }
    }

    pub fn apply<S, T>(&mut self, src : &Image<u8, S>, dst : &mut Image<u8, T>)
    where
        S : Storage<u8>,
        T : StorageMut<u8>
    {
        unsafe {
            let border_val : u8 = 0;
            let border_ty = crate::foreign::ipp::ippi::_IppiBorderType_ippBorderConst;
            let ans = crate::foreign::ipp::ippi::ippiFilterBilateral_8u_C1R(
                src.as_ptr(),
                src.byte_stride() as i32,
                dst.as_mut_ptr(),
                dst.byte_stride() as i32,
                src.size().into(),
                border_ty,
                &border_val as *const _,
                self.spec.as_mut_ptr() as *mut _,
                self.buffer.as_mut_ptr() as *mut _
            );
            assert!(ans == 0);
        }
    }

}

pub fn bilateral_filter(src : &Window<u8>, dst : &mut WindowMut<u8>, mask_sz : usize) {
    IppiFilterBilateral::new(src.height(), src.width(), mask_sz, 1.0, 1.0).apply(src, dst);
}

#[derive(Clone, Debug)]
pub struct IppFilterSobel {
    buf : Vec<u8>,
    mask : crate::foreign::ipp::ippi::IppiMaskSize,
    norm : crate::foreign::ipp::ippi::IppNormType
}

impl IppFilterSobel {

    pub fn new(sz : (usize, usize), mask_sz : usize) -> Self {
        let mask = if mask_sz == 3 {
            crate::foreign::ipp::ippi::_IppiMaskSize_ippMskSize3x3
        } else {
            crate::foreign::ipp::ippi::_IppiMaskSize_ippMskSize5x5
        };
        let num_channels = 1;
        let norm = crate::foreign::ipp::ippi::IppNormType_ippNormL1;
        unsafe {
            let mut buf_sz = 0;
            let ans = crate::foreign::ipp::ippi::ippiFilterSobelGetBufferSize(
                sz.into(),
                mask,
                norm,
                crate::foreign::ipp::ippi::IppDataType_ipp8u,
                crate::foreign::ipp::ippi::IppDataType_ipp16s,
                num_channels,
                &mut buf_sz
            );
            assert!(ans == 0);
            let mut buf : Vec<u8> = (0..(buf_sz as usize)).map(|_| 0u8 ).collect();
            Self { buf, mask, norm }
        }
    }

    pub fn apply(
        &mut self,
        src : &Image<u8, impl Storage<u8>>,
        dst : &mut Image<i16, impl StorageMut<i16>>
    ) {
        unsafe {
            let border_val = 0;
            let ans = crate::foreign::ipp::ippi::ippiFilterSobel_8u16s_C1R(
                src.as_ptr(),
                src.byte_stride() as i32,
                dst.as_mut_ptr(),
                dst.byte_stride() as i32,
                src.size().into(),
                self.mask,
                self.norm,
                crate::foreign::ipp::ippi::_IppiBorderType_ippBorderConst,
                border_val,
                self.buf.as_mut_ptr()
            );
            assert!(ans == 0);
        }
    }

}


#[derive(Clone, Debug)]
pub struct IppFilterSobelHoriz {
    buf : Vec<u8>,
    mask : crate::foreign::ipp::ippi::IppiMaskSize,
    norm : crate::foreign::ipp::ippi::IppNormType
}

impl IppFilterSobelHoriz {

    pub fn new(sz : (usize, usize), mask_sz : usize) -> Self {
        let mask = if mask_sz == 3 {
            crate::foreign::ipp::ippi::_IppiMaskSize_ippMskSize3x3
        } else {
            crate::foreign::ipp::ippi::_IppiMaskSize_ippMskSize5x5
        };
        let num_channels = 1;
        let norm = crate::foreign::ipp::ippi::IppNormType_ippNormL1;
        unsafe {
            let mut buf_sz = 0;
            let ans = crate::foreign::ipp::ippi::ippiFilterSobelHorizBorderGetBufferSize(
                sz.into(),
                mask,
                // norm,
                crate::foreign::ipp::ippi::IppDataType_ipp8u,
                crate::foreign::ipp::ippi::IppDataType_ipp16s,
                num_channels,
                &mut buf_sz
            );
            assert!(ans == 0);
            let mut buf : Vec<u8> = (0..(buf_sz as usize)).map(|_| 0u8 ).collect();
            Self { buf, mask, norm }
        }
    }

    pub fn apply(
        &mut self,
        src : &Image<u8, impl Storage<u8>>,
        dst : &mut Image<i16, impl StorageMut<i16>>
    ) {
        unsafe {
            let border_val = 0;
            let ans = crate::foreign::ipp::ippi::ippiFilterSobelHorizBorder_8u16s_C1R(
                src.as_ptr(),
                src.byte_stride() as i32,
                dst.as_mut_ptr(),
                dst.byte_stride() as i32,
                src.size().into(),
                self.mask,
                // self.norm,
                crate::foreign::ipp::ippi::_IppiBorderType_ippBorderConst,
                border_val,
                self.buf.as_mut_ptr()
            );
            assert!(ans == 0);
        }
    }

}

#[derive(Clone, Debug)]
pub struct IppFilterSobelVert {
    buf : Vec<u8>,
    mask : crate::foreign::ipp::ippi::IppiMaskSize,
    norm : crate::foreign::ipp::ippi::IppNormType
}

impl IppFilterSobelVert {

    pub fn new(sz : (usize, usize), mask_sz : usize) -> Self {
        let mask = if mask_sz == 3 {
            crate::foreign::ipp::ippi::_IppiMaskSize_ippMskSize3x3
        } else {
            crate::foreign::ipp::ippi::_IppiMaskSize_ippMskSize5x5
        };
        let num_channels = 1;
        let norm = crate::foreign::ipp::ippi::IppNormType_ippNormL1;
        unsafe {
            let mut buf_sz = 0;
            let ans = crate::foreign::ipp::ippi::ippiFilterSobelVertBorderGetBufferSize(
                sz.into(),
                mask,
                // norm,
                crate::foreign::ipp::ippi::IppDataType_ipp8u,
                crate::foreign::ipp::ippi::IppDataType_ipp16s,
                num_channels,
                &mut buf_sz
            );
            assert!(ans == 0);
            let mut buf : Vec<u8> = (0..(buf_sz as usize)).map(|_| 0u8 ).collect();
            Self { buf, mask, norm }
        }
    }

    pub fn apply(
        &mut self,
        src : &Image<u8, impl Storage<u8>>,
        dst : &mut Image<i16, impl StorageMut<i16>>
    ) {
        unsafe {
            let border_val = 0;
            let ans = crate::foreign::ipp::ippi::ippiFilterSobelVertBorder_8u16s_C1R(
                src.as_ptr(),
                src.byte_stride() as i32,
                dst.as_mut_ptr(),
                dst.byte_stride() as i32,
                src.size().into(),
                self.mask,
                // self.norm,
                crate::foreign::ipp::ippi::_IppiBorderType_ippBorderConst,
                border_val,
                self.buf.as_mut_ptr()
            );
            assert!(ans == 0);
        }
    }

}

#[derive(Clone, Debug)]
pub struct IppFilterPrewittHoriz {
    buf : Vec<u8>,
    mask : crate::foreign::ipp::ippi::IppiMaskSize,
    norm : crate::foreign::ipp::ippi::IppNormType
}

impl IppFilterPrewittHoriz {

    pub fn new(sz : (usize, usize), mask_sz : usize) -> Self {
        let mask = if mask_sz == 3 {
            crate::foreign::ipp::ippi::_IppiMaskSize_ippMskSize3x3
        } else {
            crate::foreign::ipp::ippi::_IppiMaskSize_ippMskSize5x5
        };
        let num_channels = 1;
        let norm = crate::foreign::ipp::ippi::IppNormType_ippNormL1;
        unsafe {
            let mut buf_sz = 0;
            let ans = crate::foreign::ipp::ippi::ippiFilterPrewittHorizBorderGetBufferSize(
                sz.into(),
                mask,
                crate::foreign::ipp::ippi::IppDataType_ipp8u,
                crate::foreign::ipp::ippi::IppDataType_ipp16s,
                num_channels,
                &mut buf_sz
            );
            assert!(ans == 0);
            let mut buf : Vec<u8> = (0..(buf_sz as usize)).map(|_| 0u8 ).collect();
            Self { buf, mask, norm }
        }
    }

    pub fn apply(
        &mut self,
        src : &Image<u8, impl Storage<u8>>,
        dst : &mut Image<i16, impl StorageMut<i16>>
    ) {
        unsafe {
            let border_val = 0;
            let ans = crate::foreign::ipp::ippi::ippiFilterPrewittHorizBorder_8u16s_C1R(
                src.as_ptr(),
                src.byte_stride() as i32,
                dst.as_mut_ptr(),
                dst.byte_stride() as i32,
                src.size().into(),
                self.mask,
                crate::foreign::ipp::ippi::_IppiBorderType_ippBorderConst,
                border_val,
                self.buf.as_mut_ptr()
            );
            assert!(ans == 0);
        }
    }

}

#[derive(Clone, Debug)]
pub struct IppFilterPrewittVert {
    buf : Vec<u8>,
    mask : crate::foreign::ipp::ippi::IppiMaskSize,
    norm : crate::foreign::ipp::ippi::IppNormType
}

impl IppFilterPrewittVert {

    pub fn new(sz : (usize, usize), mask_sz : usize) -> Self {
        let mask = if mask_sz == 3 {
            crate::foreign::ipp::ippi::_IppiMaskSize_ippMskSize3x3
        } else {
            crate::foreign::ipp::ippi::_IppiMaskSize_ippMskSize5x5
        };
        let num_channels = 1;
        let norm = crate::foreign::ipp::ippi::IppNormType_ippNormL1;
        unsafe {
            let mut buf_sz = 0;
            let ans = crate::foreign::ipp::ippi::ippiFilterPrewittVertBorderGetBufferSize(
                sz.into(),
                mask,
                crate::foreign::ipp::ippi::IppDataType_ipp8u,
                crate::foreign::ipp::ippi::IppDataType_ipp16s,
                num_channels,
                &mut buf_sz
            );
            assert!(ans == 0);
            let mut buf : Vec<u8> = (0..(buf_sz as usize)).map(|_| 0u8 ).collect();
            Self { buf, mask, norm }
        }
    }

    pub fn apply(
        &mut self,
        src : &Image<u8, impl Storage<u8>>,
        dst : &mut Image<i16, impl StorageMut<i16>>
    ) {
        unsafe {
            let border_val = 0;
            let ans = crate::foreign::ipp::ippi::ippiFilterPrewittVertBorder_8u16s_C1R(
                src.as_ptr(),
                src.byte_stride() as i32,
                dst.as_mut_ptr(),
                dst.byte_stride() as i32,
                src.size().into(),
                self.mask,
                crate::foreign::ipp::ippi::_IppiBorderType_ippBorderConst,
                border_val,
                self.buf.as_mut_ptr()
            );
            assert!(ans == 0);
        }
    }

}

#[derive(Clone, Debug)]
pub struct IppFilterRobertsHoriz {
    buf : Vec<u8>,
    mask : crate::foreign::ipp::ippi::IppiMaskSize,
    norm : crate::foreign::ipp::ippi::IppNormType
}

impl IppFilterRobertsHoriz {

    pub fn new(sz : (usize, usize), mask_sz : usize) -> Self {
        let mask = if mask_sz == 3 {
            crate::foreign::ipp::ippi::_IppiMaskSize_ippMskSize3x3
        } else {
            crate::foreign::ipp::ippi::_IppiMaskSize_ippMskSize5x5
        };
        let num_channels = 1;
        let norm = crate::foreign::ipp::ippi::IppNormType_ippNormL1;
        unsafe {
            let mut buf_sz = 0;
            let ans = crate::foreign::ipp::ippi::ippiFilterRobertsDownBorderGetBufferSize(
                sz.into(),
                mask,
                crate::foreign::ipp::ippi::IppDataType_ipp8u,
                crate::foreign::ipp::ippi::IppDataType_ipp16s,
                num_channels,
                &mut buf_sz
            );
            assert!(ans == 0);
            let mut buf : Vec<u8> = (0..(buf_sz as usize)).map(|_| 0u8 ).collect();
            Self { buf, mask, norm }
        }
    }

    pub fn apply(
        &mut self,
        src : &Image<u8, impl Storage<u8>>,
        dst : &mut Image<i16, impl StorageMut<i16>>
    ) {
        unsafe {
            let border_val = 0;
            let ans = crate::foreign::ipp::ippi::ippiFilterRobertsDownBorder_8u16s_C1R(
                src.as_ptr(),
                src.byte_stride() as i32,
                dst.as_mut_ptr(),
                dst.byte_stride() as i32,
                src.size().into(),
                self.mask,
                crate::foreign::ipp::ippi::_IppiBorderType_ippBorderConst,
                border_val,
                self.buf.as_mut_ptr()
            );
            assert!(ans == 0);
        }
    }

}

#[derive(Clone, Debug)]
pub struct IppFilterRobertsVert {
    buf : Vec<u8>,
    mask : crate::foreign::ipp::ippi::IppiMaskSize,
    norm : crate::foreign::ipp::ippi::IppNormType
}

impl IppFilterRobertsVert {

    pub fn new(sz : (usize, usize), mask_sz : usize) -> Self {
        let mask = if mask_sz == 3 {
            crate::foreign::ipp::ippi::_IppiMaskSize_ippMskSize3x3
        } else {
            crate::foreign::ipp::ippi::_IppiMaskSize_ippMskSize5x5
        };
        let num_channels = 1;
        let norm = crate::foreign::ipp::ippi::IppNormType_ippNormL1;
        unsafe {
            let mut buf_sz = 0;
            let ans = crate::foreign::ipp::ippi::ippiFilterRobertsUpBorderGetBufferSize(
                sz.into(),
                mask,
                crate::foreign::ipp::ippi::IppDataType_ipp8u,
                crate::foreign::ipp::ippi::IppDataType_ipp16s,
                num_channels,
                &mut buf_sz
            );
            assert!(ans == 0);
            let mut buf : Vec<u8> = (0..(buf_sz as usize)).map(|_| 0u8 ).collect();
            Self { buf, mask, norm }
        }
    }

    pub fn apply(
        &mut self,
        src : &Image<u8, impl Storage<u8>>,
        dst : &mut Image<i16, impl StorageMut<i16>>
    ) {
        unsafe {
            let border_val = 0;
            let ans = crate::foreign::ipp::ippi::ippiFilterRobertsUpBorder_8u16s_C1R(
                src.as_ptr(),
                src.byte_stride() as i32,
                dst.as_mut_ptr(),
                dst.byte_stride() as i32,
                src.size().into(),
                self.mask,
                crate::foreign::ipp::ippi::_IppiBorderType_ippBorderConst,
                border_val,
                self.buf.as_mut_ptr()
            );
            assert!(ans == 0);
        }
    }

}

// Separability of Kernels (Szelisky, p. 116, after Perona, 1995). If SVD of the 2d kernel matrix has only
// the first singular value not zeroed, then the Kernel is separable into \sqrt \sigma_0 u_0 (vertical)
// and \sqrt \sigma_0 v_0^T (horizontal) kernels.

/*
Median filter can be considered as misnomer, because it does not imply convolution.
Therefore a better API would be calling it "median pooling", for the pooling parameter
being 1 actually means no pooling is applied (image size is preserved).
*/

// Laplace filter
#[cfg(feature="ipp")]
#[derive(Debug, Clone)]
pub struct IppFilterLaplace {
    buffer : Vec<u8>,
    mask_sz : usize,
}

#[cfg(feature="ipp")]
impl IppFilterLaplace {

    pub fn new(height : usize, width : usize, mask_sz : usize) -> Self {
        let mut buf_sz : i32 = 0;
        let num_channels = 1;
        assert!(mask_sz == 3 || mask_sz == 5);
        let mask = if mask_sz == 3 {
            crate::foreign::ipp::ippi::_IppiMaskSize_ippMskSize3x3
        } else {
            crate::foreign::ipp::ippi::_IppiMaskSize_ippMskSize5x5
        };
        unsafe {
            let ans = crate::foreign::ipp::ippi::ippiFilterLaplaceBorderGetBufferSize(
                (height, width).into(),
                mask,
                crate::foreign::ipp::ippi::IppDataType_ipp8u,
                crate::foreign::ipp::ippi::IppDataType_ipp8u,
                num_channels,
                &mut buf_sz as *mut _
            );
            assert!(ans == 0);
            let mut buffer = Vec::<u8>::with_capacity(buf_sz as usize);
            buffer.set_len(buf_sz as usize);
            Self { buffer, mask_sz }
        }
    }

    pub fn apply<S, T>(&mut self, src : &Image<u8, S>, dst : &mut Image<u8, T>)
    where
        S : Storage<u8>,
        T : StorageMut<u8>
    {
        assert_eq!(src.shape(), dst.shape());
        // let anchor = crate::foreign::ipp::ippi::IppiPoint { x : (self.mask_sz.1 as i32)/2, y : (self.mask_sz.0 as i32) / 2 };
        let border_val = 0;
        // let border_ty = crate::foreign::ipp::ippi::_IppiBorderType_ippBorderConst;
        let border_ty = crate::foreign::ipp::ippi::_IppiBorderType_ippBorderRepl;
        let mask = if self.mask_sz == 3 {
            crate::foreign::ipp::ippi::_IppiMaskSize_ippMskSize3x3
        } else {
            crate::foreign::ipp::ippi::_IppiMaskSize_ippMskSize5x5
        };
        unsafe {
            let ans = crate::foreign::ipp::ippi::ippiFilterLaplaceBorder_8u_C1R(
                src.as_ptr(),
                src.byte_stride() as i32,
                dst.as_mut_ptr(),
                dst.byte_stride() as i32,
                src.size().into(),
                mask,
                border_ty,
                border_val,
                self.buffer.as_mut_ptr()
            );
            assert!(ans == 0);
        }
    }

}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Fraction {
    Polyphase12,
    Polyphase35,
    Polyphase23,
    Polyphase710,
    Polyphase34
}

#[cfg(feature="ipp")]
unsafe fn decimate_filter<S, T>(
    src : &Image<u8, S>,
    dst : &mut Image<u8, T>,
    fr : Fraction,
    byrow : bool
) where
    S : Storage<u8>,
    T : StorageMut<u8>
{
    let inner = src.window((4, 4), (src.height()-4, src.width()-4)).unwrap();
    let frac = match fr {
        Fraction::Polyphase12 => crate::foreign::ipp::ippi::IppiFraction_ippPolyphase_1_2,
        Fraction::Polyphase35 => crate::foreign::ipp::ippi::IppiFraction_ippPolyphase_1_2,
        Fraction::Polyphase23 => crate::foreign::ipp::ippi::IppiFraction_ippPolyphase_1_2,
        Fraction::Polyphase710 => crate::foreign::ipp::ippi::IppiFraction_ippPolyphase_1_2,
        Fraction::Polyphase34 => crate::foreign::ipp::ippi::IppiFraction_ippPolyphase_1_2
    };
    let (dst_len, src_len) = match fr {
        Fraction::Polyphase12 => (1, 2),
        Fraction::Polyphase35 => (3, 5),
        Fraction::Polyphase23 => (2, 3),
        Fraction::Polyphase710 => (7, 10),
        Fraction::Polyphase34 => (3, 4)
    };
    let ans = if byrow {
        assert!(inner.height() % src_len == 0);
        assert!(dst.height() % dst_len == 0);
        assert!(dst.height() % inner.height() == 0);
        crate::foreign::ipp::ippi::ippiDecimateFilterRow_8u_C1R(
            inner.as_ptr(),
            src.byte_stride() as i32,
            src.size().into(),
            dst.as_mut_ptr(),
            dst.byte_stride() as i32,
            frac
        )
    } else {
        assert!(inner.width() % src_len == 0);
        assert!(dst.width() % dst_len == 0);
        assert!(dst.width() % inner.width() == 0);
        crate::foreign::ipp::ippi::ippiDecimateFilterColumn_8u_C1R(
            src.as_ptr(),
            src.byte_stride() as i32,
            src.size().into(),
            dst.as_mut_ptr(),
            dst.byte_stride() as i32,
            frac
        )
    };
    assert!(ans == 0);
}

/* Wiener (noise-removal, edge-preserving) filter. The power
is assumed to be constant power and additive. */
#[cfg(feature="ipp")]
pub struct IppWiener {
    buf : Vec<u8>,
    sz : (usize, usize),
    mask_sz : (usize, usize),
    noise : f32
}

#[cfg(feature="ipp")]
impl IppWiener {

    pub fn new(height : usize, width : usize, mask_sz : (usize, usize), noise : f32) -> Self {
        assert!(noise >= 0.0 && noise <= 1.0);
        let sz = (height, width);
        let n_channels = 1;
        let mut buf_sz : i32 = 0;
        unsafe {
            let ans = crate::foreign::ipp::ippi::ippiFilterWienerGetBufferSize(
                sz.into(),
                mask_sz.into(),
                n_channels,
                &mut buf_sz as *mut _
            );
            assert!(ans == 0);
            let mut buf : Vec<_> = (0..(buf_sz as usize)).map(|_| 0u8 ).collect();
            Self {
                buf,
                sz,
                mask_sz,
                noise
            }
        }
    }

    pub fn apply<S, T>(&mut self, src : &Image<u8, S>, dst : &mut Image<u8, T>)
    where
        S : Storage<u8>,
        T : StorageMut<u8>
    {
        let anchor = crate::foreign::ipp::ippi::IppiPoint { x : 0, y : 0 };
        unsafe {
            let ans = crate::foreign::ipp::ippi::ippiFilterWiener_8u_C1R(
                src.as_ptr(),
                src.byte_stride() as i32,
                dst.as_mut_ptr(),
                dst.byte_stride() as i32,
                dst.size().into(),
                self.mask_sz.into(),
                anchor,
                &mut self.noise as *mut _,
                self.buf.as_mut_ptr()
            );
            assert!(ans == 0);
        }
    }
}

#[derive(Clone, Debug)]
pub struct MaxPoolTree {
    niter : usize,
    pool_sz : (usize, usize),
    pub imgs : Vec<ImageBuf<u8>>
}

fn increment_original<const N : usize, S : Storage<u8>>(
    pos : (usize, usize),
    pool_sz : (usize, usize),
    img : &Image<u8,S>,
    v : u8
) -> [Option<(usize, usize)>;N] {
    let mut n = 0;
    let mut ixs = [None; N];
    let off = (pos.0*pool_sz.0, pos.1*pool_sz.1);
    for i in off.0..(off.0+pool_sz.0) {
        for j in off.1..(off.1+pool_sz.1) {
            if img[(i, j)] == v {
                ixs[n] = Some((i, j));
                n += 1;
                if n == N {
                    return ixs;
                }
            }
        }
    }
    ixs
}

impl MaxPoolTree {

    // Returns the point and the first maximum found, ignoring the others.
    pub fn next_original_unique<S : Storage<u8>>(
        &mut self,
        img : &Image<u8,S>
    ) -> ((usize, usize), u8) {
        let ans = self.next_original::<1,_>(img);
        (ans.0[0].unwrap(), ans.1)
    }

    /* Returns the next N original indices, all contained in the next relevant local region,
    whose value equals the next maximum value. If N is chosen to be pool_sz*pool_sz, then all
    pixels with value equal to the next maximum value are returned. If N is 1, then only the
    first pixel in raster order matching the maximum value is chosen, and any others having
    the same value are ignored. Since the tree does not store the original image, you must
    pass the same image used during the update(.) call, or else the result will be meaningless. */
    pub fn next_original<const N : usize, S : Storage<u8>>(
        &mut self,
        img : &Image<u8,S>
    ) -> ([Option<(usize, usize)>;N], u8) {
        let (pos, v) = self.next();
        let ixs = increment_original(pos, self.pool_sz, img, v);
        // assert!(ixs[0].is_some());
        (ixs,v)
    }

    pub fn next(&mut self) -> ((usize, usize), u8) {
        let ((r, c), v) = self.imgs.last().unwrap().indexed_maximum();
        if self.imgs.len() == 1 {
            self.imgs[0][(r, c)] = 0;
            return ((r, c), v);
        }
        let curr_max = search_recursive(
            &self.imgs[..(self.imgs.len()-1)],
            (r, c),
            self.pool_sz,
            |img| img.indexed_maximum()
        );
        self.imgs[0][curr_max.0] = 0;
        propagate_recursive(&mut self.imgs, curr_max.0, self.pool_sz, |img| img.indexed_maximum() );
        // assert_integrity(&self.imgs[..]);
        curr_max
    }

    pub fn update<S : Storage<u8>>(&mut self, img : &Image<u8, S>) {
        pool_cascade(img, &mut self.imgs, |src, mut dst| src.local_maxima_to(&mut dst) );
        // assert_integrity(&self.imgs[..]);
    }

    pub fn new(
        img_sz : (usize, usize),
        pool_sz : (usize, usize),
        niter : usize
    ) -> Result<Self, Box<dyn std::error::Error>> {
        is_pool_valid(img_sz, pool_sz, niter)?;
        let imgs = create_cascade(img_sz, pool_sz, niter);
        Ok(Self {
            pool_sz,
            niter,
            imgs
        })
    }

}

/* A spatial tree representing recursive calls to local minima. Each node
of the tree represents the minimum pixel of a pool_sz region at the next
level. This is relevant for iterated search of global minima pixels, where
previously selected pixels are not considered further. The tree can
easily be generalized to a MaxPoolTree as well (use indexed_local_maximum) and
reset selected pixels to 0. */
#[derive(Clone, Debug)]
pub struct MinPoolTree {
    niter : usize,
    pool_sz : (usize, usize),
    pub imgs : Vec<ImageBuf<u8>>
}

impl MinPoolTree {

    // Returns the point and the first maximum found, ignoring the others.
    pub fn next_original_unique<S : Storage<u8>>(
        &mut self,
        img : &Image<u8,S>
    ) -> ((usize, usize), u8) {
        let ans = self.next_original::<1,_>(img);
        (ans.0[0].unwrap(), ans.1)
    }

    /* Returns the next N original indices, all contained in the next relevant local region,
    whose value equals the next maximum value. If N is chosen to be pool_sz*pool_sz, then all
    pixels with value equal to the next maximum value are returned. If N is 1, then only the
    first pixel in raster order matching the maximum value is chosen, and any others having
    the same value are ignored. Since the tree does not store the original image, you must
    pass the same image used during the update(.) call, or else the result will be meaningless. */
    pub fn next_original<const N : usize, S : Storage<u8>>(
        &mut self,
        img : &Image<u8,S>
    ) -> ([Option<(usize, usize)>;N], u8) {
        let (pos, v) = self.next();
        let ixs = increment_original(pos, self.pool_sz, img, v);
        // assert!(ixs[0].is_some());
        (ixs,v)
    }

    pub fn next(&mut self) -> ((usize, usize), u8) {
        let ((r, c), v) = self.imgs.last().unwrap().indexed_minimum();
        if self.imgs.len() == 1 {
            self.imgs[0][(r, c)] = 255;
            return ((r, c), v);
        }
        let curr_min = search_recursive(
            &self.imgs[..(self.imgs.len()-1)],
            (r, c),
            self.pool_sz,
            |img| img.indexed_minimum()
        );
        self.imgs[0][curr_min.0] = 255;
        propagate_recursive(&mut self.imgs, curr_min.0, self.pool_sz, |img| img.indexed_minimum() );
        // assert_integrity(&self.imgs[..]);
        curr_min
    }

    pub fn update<S : Storage<u8>>(&mut self, img : &Image<u8, S>) {
        pool_cascade(img, &mut self.imgs, |src, mut dst| src.local_minima_to(&mut dst) );
        // assert_integrity(&self.imgs[..]);
    }

    pub fn new(
        img_sz : (usize, usize),
        pool_sz : (usize, usize),
        niter : usize
    ) -> Result<Self, Box<dyn std::error::Error>> {
        is_pool_valid(img_sz, pool_sz, niter)?;
        let imgs = create_cascade(img_sz, pool_sz, niter);
        Ok(Self {
            pool_sz,
            niter,
            imgs
        })
    }

}

fn is_pool_valid(
    img_sz : (usize, usize),
    pool_sz : (usize, usize),
    niter : usize
) -> Result<(), Box<dyn std::error::Error>> {
    if niter == 0 {
        Err(format!("Invalid number of iterations"))?;
    }
    if img_sz.0 % pool_sz.0 != 0 {
        Err(format!("Height of {} not divisible by pool size {} at iteration {}", img_sz.0, pool_sz.0, niter))?;
    }
    if img_sz.1 % pool_sz.1 != 0 {
        Err(format!("Width of {} not divisible by pool size {} at iteration {}", img_sz.1, pool_sz.1, niter))?;
    }
    if niter == 1 {
        Ok(())
    } else {
        is_pool_valid((img_sz.0 / pool_sz.0, img_sz.1 / pool_sz.1), pool_sz, niter - 1)
    }
}

// ix is the position of the minimum at the previous level.
// The imgs slice contains the current search at its last position.
// The search keeps a size of 4x4
fn search_recursive<F>(
    imgs : &[ImageBuf<u8>],
    ix : (usize, usize),
    pool_sz : (usize, usize),
    op : F
) -> ((usize, usize), u8)
where
    F : Fn(ImageRef<u8>)->((usize, usize), u8)
{
    let last_img = imgs.last().unwrap();
    let off = (ix.0 * pool_sz.0, ix.1 * pool_sz.1);

    // let ((r, c), min) = last_img.window(off, pool_sz).unwrap().indexed_minimum();
    let ((r, c), min) = op(last_img.window(off, pool_sz).unwrap());

    let off_ix = (r + off.0, c + off.1);
    if imgs.len() > 1 {
        search_recursive(&imgs[..(imgs.len()-1)], off_ix, pool_sz, op)
    } else {
        (off_ix, min)
    }
}

// Updates upward are *always* required.
fn propagate_recursive<F>(
    imgs : &mut [ImageBuf<u8>],
    ix : (usize, usize),
    pool_sz : (usize, usize),
    op : F
) where
    F : Fn(ImageRef<u8>)->((usize, usize), u8)
{
    let fst_img = &imgs[0];
    let off = (ix.0 - ix.0 % pool_sz.0, ix.1 - ix.1 % pool_sz.1);
    // let mut curr_min = fst_img.window(off, pool_sz).unwrap().indexed_minimum();
    let mut curr_extr = op(fst_img.window(off, pool_sz).unwrap());
    curr_extr.0.0 += off.0;
    curr_extr.0.1 += off.1;
    let next_off = (curr_extr.0.0 / pool_sz.0, curr_extr.0.1 / pool_sz.1);
    imgs[1][next_off] = curr_extr.1;
    if imgs.len() > 2 {
        propagate_recursive(&mut imgs[1..], (curr_extr.0.0 / pool_sz.0, curr_extr.0.1 / pool_sz.1), pool_sz, op);
    }
}

fn create_cascade(img_dims : (usize, usize), pool_sz : (usize, usize), niter : usize) -> Vec<ImageBuf<u8>> {
    let mut imgs = Vec::with_capacity(niter);
    for i in 0..niter {
        imgs.push(ImageBuf::<u8>::new_constant(
            img_dims.0 / pool_sz.0.pow((i+1) as u32),
            img_dims.1 / pool_sz.1.pow((i+1) as u32),
            0
        ));
    }
    imgs
}

fn pool_cascade<S, F>(src : &Image<u8,S>, imgs : &mut [ImageBuf<u8>], op : F)
where
    F : Fn(&ImageRef<u8>, &mut ImageBuf<u8>),
    S : Storage<u8>
{
    // src.local_minima_to(&mut imgs[0]);
    op(&src.full_window(), &mut imgs[0]);
    for i in 0..(imgs.len()-1) {
        let new_src = std::mem::take(&mut imgs[i]);
        // new_src.local_minima_to(&mut imgs[i+1]);
        op(&new_src.full_window(), &mut imgs[i+1]);
        imgs[i] = new_src;
    }
}

fn assert_integrity(imgs : &[ImageBuf<u8>]) {
    for i in 0..(imgs.len()-1) {
        for (w1, (r, c, w2)) in imgs[i].windows((imgs[i].height()/imgs[i+1].height(),imgs[i].width()/imgs[i+1].width()))
            .zip(imgs[i+1].labeled_pixels::<usize, _>(1))
        {
            let min = w1.min_max().0;
        }
    }
}

