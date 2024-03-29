use std::ops::*;
use crate::image::*;
use std::mem;

impl<P, S> Image<P, S>
where
    S : StorageMut<P>,
    P : Pixel + MulAssign + DivAssign,
    //Box<[P]> : StorageMut<P>,
    //for<'a> &'a [P] : Storage<P>,
    //for<'a> &'a mut [P] : StorageMut<P>,
{

    fn abs_mut(&mut self) {

        if self.pixel_is::<u8>() || self.pixel_is::<u64>() {
            return;
        }

        #[cfg(feature="ipp")]
        unsafe {

            let (byte_stride, roi) = crate::image::ipputils::step_and_size_for_image(self);

            if self.pixel_is::<f32>() {
                let ans = crate::foreign::ipp::ippi::ippiAbs_32f_C1IR(
                    mem::transmute(self.as_mut_ptr()),
                    byte_stride,
                    roi
                );
                assert!(ans == 0);
                return;
            }

            if self.pixel_is::<i16>() {
                let ans = crate::foreign::ipp::ippi::ippiAbs_16s_C1IR(
                    mem::transmute(self.as_mut_ptr()),
                    byte_stride,
                    roi
                );
                assert!(ans == 0);
                return;
            }
        }

        unimplemented!()
    }

    fn scalar_abs_diff_mut(&mut self, by : P) {

        #[cfg(feature="ipp")]
        unsafe {
            let scale_factor = 1;
            let mut dst = self.clone_owned();
            let (byte_stride, roi) = crate::image::ipputils::step_and_size_for_image(self);
            let (dst_byte_stride, dst_roi) = crate::image::ipputils::step_and_size_for_image(&dst.full_window_mut());
            if self.pixel_is::<f32>() {
                let ans = crate::foreign::ipp::ippcv::ippiAbsDiffC_32f_C1R(
                    mem::transmute(self.as_ptr()),
                    byte_stride,
                    mem::transmute(dst.full_window_mut().as_mut_ptr()),
                    dst_byte_stride,
                    mem::transmute(roi),
                    *mem::transmute::<_, &f32>(&by)
                );
                assert!(ans == 0);
                self.copy_from(&dst.full_window());
                return;
            }
        }

        unimplemented!()
    }

    fn scalar_add_mut(&mut self, by : P) {

        #[cfg(feature="ipp")]
        unsafe {
            let scale_factor = 0;
            let (byte_stride, roi) = crate::image::ipputils::step_and_size_for_image(self);

            if self.pixel_is::<u8>() {
                let ans = crate::foreign::ipp::ippi::ippiAddC_8u_C1IRSfs(
                    *mem::transmute::<_, &u8>(&by),
                    mem::transmute(self.as_mut_ptr()),
                    byte_stride,
                    roi,
                    scale_factor
                );
                assert!(ans == 0);
                return;
            }

            if self.pixel_is::<f32>() {
                let ans = crate::foreign::ipp::ippi::ippiAddC_32f_C1IR(
                    *mem::transmute::<_, &f32>(&by),
                    mem::transmute(self.as_mut_ptr()),
                    byte_stride,
                    roi
                );
                assert!(ans == 0);
                return;
            }
        }

        // unsafe { mem::transmute::<_, &'static mut WindowMut<'static, N>>(self).pixels_mut(1).for_each(|p| *p = p.saturating_add(&by) ); }
        // unsafe { mem::transmute::<_, &'static mut WindowMut<'static, N>>(self).pixels_mut(1).for_each(|p| *p += by );
        unimplemented!()
    }

    fn scalar_mul_mut(&mut self, by : P) {

        #[cfg(feature="ipp")]
        unsafe {

            let (byte_stride, roi) = crate::image::ipputils::step_and_size_for_image(self);
            let scale_factor = 1;

            if self.pixel_is::<u8>() {
                let ans = crate::foreign::ipp::ippi::ippiMulC_8u_C1IRSfs(
                    *mem::transmute::<_, &u8>(&by),
                    mem::transmute(self.as_mut_ptr()),
                    byte_stride,
                    roi,
                    scale_factor
                );
                assert!(ans == 0);
                return;
            }

            if self.pixel_is::<i32>() {
                // does not exist
                /*let ans = crate::foreign::ipp::ippi::ippiMulC_32s_C1IRSfs(
                    *mem::transmute::<_, &i32>(&by),
                    mem::transmute(self.as_mut_ptr()),
                    byte_stride,
                    roi,
                    scale_factor
                );
                assert!(ans == 0);
                return;*/
            }

            if self.pixel_is::<f32>() {
                let ans = crate::foreign::ipp::ippi::ippiMulC_32f_C1IR(
                    *mem::transmute::<_, &f32>(&by),
                    mem::transmute(self.as_mut_ptr()),
                    byte_stride,
                    roi
                );
                assert!(ans == 0);
                return;
            }
        }

        self.pixels_mut(1).for_each(|p| *p *= by );
    }
    
    fn scalar_div_mut(&mut self, by : P) {

        /*#[cfg(feature="ipp")]
        unsafe {

            let (byte_stride, roi) = crate::image::ipputils::step_and_size_for_image(&self);
            let scale_factor = 1;

            if self.pixel_is::<u8>() {
                let ans = crate::foreign::ipp::ippi::ippiDivC_8u_C1IRSfs(
                    *mem::transmute::<_, &u8>(&by),
                    mem::transmute(self.as_mut_ptr()),
                    byte_stride,
                    roi,
                    scale_factor
                );
                assert!(ans == 0);
                return;
            }

            if self.pixel_is::<i32>() {
                // does not exist
                /*let ans = crate::foreign::ipp::ippi::ippiMulC_32s_C1IRSfs(
                    *mem::transmute::<_, &i32>(&by),
                    mem::transmute(self.as_mut_ptr()),
                    byte_stride,
                    roi,
                    scale_factor
                );
                assert!(ans == 0);
                return;*/
            }

            if self.pixel_is::<f32>() {
                let ans = crate::foreign::ipp::ippi::ippiDivC_32f_C1IR(
                    *mem::transmute::<_, &f32>(&by),
                    mem::transmute(self.as_mut_ptr()),
                    byte_stride,
                    roi
                );
                assert!(ans == 0);
                return;
            }
        }*/

        self.pixels_mut(1).for_each(|p| *p /= by );
    }

    /*fn truncate_mut(&mut self, above : bool, val : N) {
        unsafe {
            match above {
                true => {
                    mem::transmute::<_, &mut WindowMut<'_, N>>(self).pixels_mut(1).for_each(|px| if *px >= val { *px = N::zero(); });
                },
                false => {
                    mem::transmute::<_, &mut WindowMut<'_, N>>(self).pixels_mut(1).for_each(|px| if *px <= val { *px = N::zero(); });
                },
            }
        }
    }*/

}

impl<P, S> Image<P, S>
where
    S : Storage<P>,
    P : FloatPixel + MulAssign + DivAssign,
    //Box<[P]> : StorageMut<P>,
    //for<'a> &'a [P] : Storage<P>,
    //for<'a> &'a mut [P] : StorageMut<P>,
{

    fn atan2_to<T>(&self, dst : &mut Image<P, T>) 
    where
        T : StorageMut<P>
    {
        assert!(self.shape() == dst.shape());
        for (out, input) in dst.pixels_mut(1).zip(self.pixels(1)) {
            *out = (*out).atan2(*input);
        }
    }
    
}

impl<P, S> Image<P, S>
where
    S : Storage<P>,
    P : Pixel + MulAssign + DivAssign + Div<Output=P>,
    //Box<[P]> : StorageMut<P>,
    //for<'a> &'a [P] : Storage<P>,
    //for<'a> &'a mut [P] : StorageMut<P>,
{

    pub fn scalar_div_to<T>(&self, by : P, dst : &mut Image<P, T>)
    where
        T : StorageMut<P>
    {
        dst.pixels_mut(1).zip(self.pixels(1)).for_each(|(dst, src)| *dst = *src / by );
    }

}

/*// Maps the domain of an integer image to u8 domain, setting zero the image minimum
// and 256 as the integer maximum, using only integer division.
#[cfg(feature="ipp")]
pub fn integer_normalize_max_min<S, T>(win : &Image<S>, dst : &mut Image<T>) -> (T, N)
where
    N : crate::image::Pixel + Sub<Output=N> + Mul<Output=N> + Div<Output=N> + Debug + AsPrimitive<u8>,
    u8 : AsPrimitive<N>,
    i16 : AsPrimitive<N>,
    u16 : AsPrimitive<N>,
    f32 : AsPrimitive<N>
{
    let (min, max) = crate::stat::min_max(win);
    let max_u8 : N = (255u8).as_();
    let win = win.as_ref();
    for (mut d, px) in dst.pixels_mut(1).zip(win.pixels(1)) {
        *d = (((*px - min) * max_u8) / max).as_();
    }
    (min, max)
}*/

// Normalizes the image relative to the max(infinity) norm. This limits values to [0., 1.]
/*pub fn normalize_max_mut<'a, N>(win : &'a Window<'a, N>, dst : &'a mut WindowMut<'a, N>) -> N
where
    N : Div<Output=N> + Copy + PartialOrd + Any + Debug + Default,
    u8 : AsPrimitive<N>
{
    let max = stat::max(win);
    point_div_mut(win, max, dst);
    max
}

// Normalizes the image, so that the sum of its pixels is 1.0. Useful for convolution filters,
// which must preserve the original image norm so that convolution output does not overflow its integer
/// or float maximum.
pub fn normalize_unit_mut<'a, N>(win : &'a Window<'a, N>, dst : &'a mut WindowMut<'a, N>) -> N
where
    N : Div<Output=N> + Copy + PartialOrd + Serialize + DeserializeOwned + Any + Debug + Zero + From<f32> + Default,
    f32 : From<N>
{
    let sum = stat::sum(win, 1);
    point_div_mut(win, sum, dst);
    sum
}

// Normalizes the image relative to the max(infinity) norm. This limits values to [0., 1.]
// This only really makes sense for float images, since integer bounded images (u8, i32)
// will generate divisions at are either 0 or 1, effectively creating a dark image.
pub fn normalize_max_inplace<'a, N>(dst : &'a mut WindowMut<'a, N>) -> N
where
    N : Div<Output=N> + Copy + PartialOrd + Any + Debug + Default,
    u8 : AsPrimitive<N>
{
    unsafe {
        let max = stat::max(mem::transmute(dst as *mut _));
        point_div_inplace(dst, max);
        max
    }
}

/// Normalizes the image, so that the sum of its pixels is 1.0. Useful for convolution filters,
/// which must preserve the original image norm so that convolution output does not overflow its integer
/// or float maximum.
pub fn normalize_unit_inplace<'a, N>(dst : &'a mut WindowMut<'a, N>) -> N
where
    N : Div<Output=N> + Copy + PartialOrd + Serialize + DeserializeOwned + Any + Debug + Zero + From<f32> + Default,
    f32 : From<N>
{
    unsafe {
        let sum = stat::sum(mem::transmute(dst as *mut _), 1);
        point_div_inplace(dst, sum);
        sum
    }
}*/

/*pub fn normalize_ratio_inplace<'a, N>(dst : &mut WindowMut<N>)
where
    N : UnsignedPixel + PartialOrd,
    f32 : AsPrimitive<N>,
    N : AsPrimitive<f32>,
    u8 : AsPrimitive<N>
{
    let mut max : f32 = crate::stat::max(dst.as_ref()).as_();
    max += std::f32::EPSILON;
    let bound_val : f32 = N::max_value().as_();
    for mut px in dst.pixels_mut(1) {
        let pxf : f32 = px.as_();
        *px = ((pxf / max) * bound_val).as_();
    }
}*/

/*IppStatus ippiSqr_<mod> ( const Ipp<datatype>* pSrc , int srcStep , Ipp<datatype>* pDst ,
    int dstStep , IppiSize roiSize , int scaleFactor );

IppStatus ippiSqrt_<mod> ( const Ipp<datatype>* pSrc , int srcStep , Ipp<datatype>* pDst ,
int dstStep , IppiSize roiSize , int scaleFactor );

IppStatus ippiLn_<mod> ( const Ipp<datatype>* pSrc , int srcStep , Ipp<datatype>* pDst , int
dstStep , IppiSize roiSize , int scaleFactor );

IppStatus ippiExp_<mod> ( const Ipp<datatype>* pSrc , int srcStep , Ipp<datatype>* pDst ,
int dstStep , IppiSize roiSize , int scaleFactor );*/

/*/// Stretch the image profile such that the gray levels make the best use of the dynamic range.
/// (Myler & Weeks, 1993). gray_max and gray_min are min and max intensities at the current image.
/// Stretching makes better use of the gray value space.
pub fn stretch() {
    (255. / (gray_max - gray_min )) * (px - gray_min)
}

/// Equalization makes the histogram more flat.
pub fn equalize(hist, src, dst) {
    // Based on Myler & Weeks, 1993:
    // (1) Calculate image CDF
    // (2) Order all available gray levels
    // (3) Divide CDF by total number of pixels (normalize by n at last bin). This will preserve order of gray levels.
    // (4) For all pixels: image[px] = new_cdf[old_order_of_px].
    // All pixels at last bin will be mapped to same value. All others will be brought closer by a ratio to it.
    // Basically transform the CDF to a ramp and the histogram to a uniform.
    for i in 0..256 {

        // Sum histogram entries up to ith index (calc cumulative)
        sum = 0.;
        for j in 0..(i+1) {
            sum += hist[j]
        }
        histeq[i] = (255*sum+0.5) as i32;
    }

    for r in 0..img.height() {
        for c in 0..img.width() {
            dst[(r, c)] = histeq[src[(r, c)] as usize];
        }
    }
}

*/
/*/// Converts this image from the range[0, 1] (floating-point) to the quantized range defined by max
    pub fn quantize<M : Scalar + >(&self, mut out : WindowMut<'_, M>)
    where
        N : Copy + Mul<Output=N> + num_traits::float::Float + num_traits::cast::AsPrimitive<M> + Scalar,
        M : num_traits::cast::AsPrimitive<N> + Copy + num_traits::Bounded
    {
        let max : N = M::max_value().as_();
        for i in 0..out.height() {
            for j in 0..out.width() {
                out[(i, j)] = (self[(i, j)] * max).min(max).as_();
            }
        }
    }*/

    /*/// Converts this image to the range [0, 1] (floating-point) by dividing by its maximum
    /// attainable value.
    pub fn smoothen<M : Scalar + >(&self, mut out : WindowMut<'_, M>)
    where
        N : num_traits::Bounded + Copy + num_traits::cast::AsPrimitive<M>,
        M : Copy + Div<Output=M>
    {
        let max : M = N::max_value().as_();
        for i in 0..out.height() {
            for j in 0..out.width() {
                out[(i, j)] = (self[(i, j)]).as_() / max;
            }
        }
    }*/

/*pub fn conditional_set_when_neighbors_sum_less(
    src : &Window<u8>,
    mut dst : WindowMut<u8>,
    win_side : usize,
    cond_val : u8,
    less_val : u64,
    true_val : u8,
    false_val : Option<u8>
) -> usize {
    assert!(win_side % 2 == 0);
    assert!(src.shape() == dst.shape());
    let center = (win_side / 2, win_side / 2);
    let mut n_changes = 0;
    for (mut d, s) in dst.windows_mut((win_side, win_side)).zip(src.windows((win_side, win_side))) {
        if s[center] == cond_val {
            let sum = crate::stat::accum::<_, u64>(&s) - s[center] as u64;
            if sum < less_val {
                d[center] = true_val;
                n_changes += 1;
            } else {
                d[center] = false_val.unwrap_or(s[center]);
            }
        } else {
            d[center] = false_val.unwrap_or(s[center]);
        }
    }
    n_changes
}

/// Take a binary image and sets shape borders to 1, and non-borders to zero.
/// After Davies (2005) alg. 2.17
pub fn set_when_neighbors_sum_equal(
    src : &Window<u8>,
    mut dst : WindowMut<u8>,
    win_side : usize,
    eq_val : u64,
    true_val : u8,
    false_val : Option<u8>
) -> usize {
    assert!(win_side % 2 == 0);
    assert!(src.shape() == dst.shape());
    let center = (win_side / 2, win_side / 2);
    let mut n_changes = 0;
    for (mut d, s) in dst.windows_mut((win_side, win_side)).zip(src.windows((win_side, win_side))) {
        let sum = crate::stat::accum::<_, u64>(&s) - s[center] as u64;
        if sum == eq_val {
            d[center] = true_val;
            n_changes += 1;
        } else {
            d[center] = false_val.unwrap_or(s[center]);
        }
    }
    n_changes
}*/


/*impl<S> Image<S> 
where
    S : PixelsMut
{

    // TODO also implement contrast_adjust_mut
    pub fn brightness_adjust_mut<'a>(&'a mut self, k : i16) {

        assert!(k <= 255 && k >= -255);
        let abs_k = k.abs() as u8;
        if k > 0 {
            self.pixels_mut(1).for_each(|px| *px = px.saturating_add(abs_k) );
        } else {
            self.pixels_mut(1).for_each(|px| *px = px.saturating_sub(abs_k) );
        }
    }

}*/

