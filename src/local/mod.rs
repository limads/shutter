use crate::image::*;
use std::mem;
use std::any::Any;
use serde::{Serialize, Deserialize, de::DeserializeOwned};
use num_traits::Zero;
use nalgebra::Scalar;
use std::iter::FromIterator;
pub use ripple::conv::*;

pub fn local_sum(src : &Window<'_, u8>, dst : &mut WindowMut<'_, i32>) {

    assert!(src.height() % dst.height() == 0);
    assert!(src.width() % dst.width() == 0);

    let local_win_sz = (src.height() / dst.height(), src.width() / dst.width());

    /*#[cfg(feature="ipp")]
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
    }*/
    for i in 0..dst.height() {
        for j in 0..dst.width() {
            dst[(i, j)] = crate::global::sum::<_, f32>(&src.sub_window((i*local_win_sz.0, j*local_win_sz.1), local_win_sz).unwrap(), 1) as i32;
        }
    }

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
pub fn block_min_or_max<N>(win : &Window<'_, N>, dst : &mut WindowMut<'_, N>, is_maximum : bool)
where
    N : Scalar + Copy + Default + Zero + Copy
{

    assert!(win.width() % dst.width() == 0);
    assert!(win.height() % dst.height() == 0);

    let block_sz = (win.height() / dst.height(), win.width() / dst.width());
    // let block_sz = (win.height() / num_blocks.0, win.width() / num_blocks.1);

    #[cfg(feature="ipp")]
    unsafe {
        let (src_step, src_sz) = crate::image::ipputils::step_and_size_for_window(win);
        let (dst_step, dst_sz) = crate::image::ipputils::step_and_size_for_window_mut(&dst);

        // If pdstmin or pdstmax are null, the corresponding result is not calculated.
        let (ptr_dst_min, ptr_dst_max) : (*mut N, *mut N) = match is_maximum {
            true => (std::ptr::null_mut(), dst.as_mut_ptr()),
            false => (dst.as_mut_ptr(), std::ptr::null_mut())
        };
        /*let mut other = dst.clone_owned();
        let (ptr_dst_min, ptr_dst_max) : (*mut u8, *mut u8) = match is_maximum {
            true => (other.full_window_mut().as_mut_ptr(), dst.as_mut_ptr()),
            false => (dst.as_mut_ptr(), other.full_window_mut().as_mut_ptr())
        };*/

        println!("{:?}", block_sz);

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

// This returns the minimum and maximum values and indices. Can be applied block-wise to get indices at many points.
pub fn min_max_idx(
    win : &Window<'_, u8>,
    min : bool,
    max : bool
) -> (Option<(usize, usize, u8)>, Option<(usize, usize, u8)>) {

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
            let ans = crate::foreign::ipp::ippcv::ippiMinMaxIndx_8u_C1R(
                win.as_ptr(),
                step,
                mem::transmute(sz),
                &mut min as *mut _,
                &mut max as *mut _,
                &mut min_ix as *mut _,
                &mut max_ix as *mut _
            );
            assert!(ans == 0);
            return (
                Some((min_ix.y as usize, min_ix.x as usize, min as u8)),
                Some((max_ix.y as usize, max_ix.x as usize, max as u8))
            );
        } else if min {
            let mut min : u8 = 0;
            let mut min_x : i32 = 0;
            let mut min_y : i32 = 0;
            let ans = crate::foreign::ipp::ippi::ippiMinIndx_8u_C1R(
                win.as_ptr(),
                step,
                mem::transmute(sz),
                &mut min as *mut _,
                &mut min_x as *mut _,
                &mut min_y as *mut _
            );
            assert!(ans == 0);
            return (Some((min_y as usize, min_x as usize, min)), None);
        } else if max {
            let mut max : u8 = 0;
            let mut max_x : i32 = 0;
            let mut max_y : i32 = 0;
            let ans = crate::foreign::ipp::ippi::ippiMaxIndx_8u_C1R(
                win.as_ptr(),
                step,
                mem::transmute(sz),
                &mut max as *mut _,
                &mut max_x as *mut _,
                &mut max_y as *mut _
            );
            assert!(ans == 0);
            return (None, Some((max_y as usize, max_x as usize, max)));
        }
    }

    unimplemented!()
}

pub fn find_peaks(win : &Window<'_, i32>, threshold : i32, max_peaks : usize) -> Vec<(usize, usize)> {

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

/// Implements separable convolution, representing the filter as a one-row filter.
pub trait ConvolveSep {

    type Output;

    type OwnedOutput;

    fn convolve_sep_mut(&self, filter : &Self, conv : Convolution, out : &mut Self::Output);

    fn convolve_sep(&self, filter : &Self, conv : Convolution) -> Self::OwnedOutput;

}

/// Common difference (edge-detection) filters
pub mod edge {

    use crate::image::Window;
    use std::f32::consts::SQRT_2;

    const NEG_SQRT_2 : f32 = -1.41421356237309504880168872420969808f32;

    // pub const ROBERTS_HORIZ : Window<'static, f32> = Window::from_static::<4, 2>(&[1., 0., 0., -1.]);

    // pub const ROBERTS_VERT : Window<'static, f32> = Window::from_static::<4, 2>(&[0., 1., -1., 0.]);

    // 3x3 directional filters
    pub const SOBEL_HORIZ : Window<'static, f32> = Window::from_static::<9, 3>(&[
        -1., 0., 1.,
        -2., 0., 2.,
        -1., 0., 1.
    ]);

    pub const SOBEL_VERT : Window<'static, f32> = Window::from_static::<9, 3>(&[
        1., 2., 1.,
        0., 0., 0.,
        -1., -2., -1.
    ]);

    pub const PREWIT_HORIZ : Window<'static, f32> = Window::from_static::<9, 3>(&[
        -1., 0., 1.,
        -1., 0., 1.,
        -1., 0., 1.
    ]);

    pub const PREWIT_VERT : Window<'static, f32> = Window::from_static::<9, 3>(&[
        1., 1., 1.,
        0., 0., 0.,
        -1., -1., -1.
    ]);

    pub const FREI_HORIZ : Window<'static, f32> = Window::from_static::<9, 3>(&[
        -1., 0., 1.,
        NEG_SQRT_2, 0., SQRT_2,
        -1., 0., 1.
    ]);

    pub const FREI_VERT : Window<'static, f32> = Window::from_static::<9, 3>(&[
        1., SQRT_2, 1.,
        0., 0., 0.,
        -1., NEG_SQRT_2, -1.
    ]);

    // Symmetric laplace filter
    pub const LAPLACE : Window<'static, f32> = Window::from_static::<9, 3>(&[
        0., 1., 0.,
        1., -4., 1.,
        0., 1., 0.
    ]);

    // Symmetric Laplacian of Gaussian (LoG) filter
    pub const LAPLAC_OF_GAUSS : Window<'static, f32> = Window::from_static::<25, 5>(&[
        0., 0., 1., 0., 0.,
        0., 1., 2., 1., 0.,
        1., 2., -16., 2., 1.,
        0., 1., 2., 1., 0.,
        0., 0., 1., 0., 0.
    ]);

    // TODO difference of gaussian 5x5 with DOG_21 with var(2) - var(1); DOG_31 with var(3) - var(1) and so on.

}

/// Common weighted averaging (blur) filters
pub mod blur {

    use crate::image::Window;

    const FRAC_1_9 : f32 = 0.111111111;

    // TODO build all scaled versions of gauss_3 and gauss_5.
    // const FRAC_1_

    pub const UNSCALED_BOX_5 : Window<'static, f32> = Window::from_static::<25, 5>(&[
        0., 0., 0., 0., 0.,
        0., 1., 1., 1., 0.,
        0., 1., 1., 1., 0.,
        0., 1., 1., 1., 0.,
        0., 0., 0., 0., 0.
    ]);

    pub const BOX_5 : Window<'static, f32> = Window::from_static::<25, 5>(&[
        0., 0., 0., 0., 0.,
        0., FRAC_1_9, FRAC_1_9, FRAC_1_9, 0.,
        0., FRAC_1_9, FRAC_1_9, FRAC_1_9, 0.,
        0., FRAC_1_9, FRAC_1_9, FRAC_1_9, 0.,
        0., 0., 0., 0., 0.
    ]);

    pub const UNSCALED_GAUSS_5 : Window<'static, f32> = Window::from_static::<25, 5>(&[
        1., 4., 6., 4., 1.,
        4., 16., 24., 16., 4.,
        6., 24., 36., 24., 6.,
        4., 16., 24., 16., 4.,
        1., 4., 6., 4., 1.
    ]);
}

pub fn convolution_buffer<N>(img_sz : (usize, usize), kernel_sz : (usize, usize)) -> Image<N>
where
    N : Scalar + Clone + Zero + Serialize + DeserializeOwned + Copy + Default
{
    let (nrow, ncol) = linear_conv_sz(img_sz, kernel_sz);
    Image::new_constant(nrow, ncol, N::zero())
}

pub fn linear_conv_sz(img_sz : (usize, usize), kernel_sz : (usize, usize)) -> (usize, usize) {
    let nrow = img_sz.0 - kernel_sz.0 + 1;
    let ncol = img_sz.1 - kernel_sz.1 + 1;
    (nrow, ncol)
}

fn padded_baseline_convolution<'a, N>(input : &Window<'_, N>, filter : &Window<'_, N>, out : &'a mut WindowMut<'a, N>)
where
    N : Scalar + Copy + Default + Zero + Copy + Serialize + DeserializeOwned +
    Any + std::ops::Mul<Output = N> + std::ops::AddAssign
{
    let mut extended_in = Image::new_constant(
        input.height() + filter.height() / 2 + 1,
        input.width() + filter.width() / 2 + 1,
        N::zero()
    );
    let mut extended_out = extended_in.clone();
    extended_in.window_mut(((filter.height() / 2), (filter.width() / 2)), out.shape()).unwrap().copy_from(&input);
    baseline_convolution(extended_in.as_ref(), filter, &mut extended_out.full_window_mut());
    out.copy_from(&extended_out.window(((filter.height() / 2), (filter.width() / 2)), out.shape()).unwrap());
}

/*fn extended_baseline_convolution<N>(input : &Window<'_, N>, filter : &Window<'_, N>, out : &mut WindowMut<'_, N>)
where
    N : Scalar + Copy + Default + Zero + Copy + Serialize + DeserializeOwned +
    Any + std::ops::Mul<Output = N> + std::ops::AddAssign
{
    /*let mut extended_in = Image::new_constant(
        input.height() + filter.height() / 2 + 1,
        input.width() + filter.width() / 2 + 1,
        N::zero()
    );
    let mut extended_out = extended_in.clone();
    extened_in.sub_window_mut((filter.height() / 2), (filter.width() / 2), out.shape()).unwrap().copy_from(&input);
    baseline_convolution(extended_in.as_ref(), filter.as_ref(), extended_out.as_mut());
    out.copy_from(&extended_out.sub_window().unwrap());*/
}*/

fn baseline_convolution<N>(input : &Window<'_, N>, filter : &Window<'_, N>, out : &mut WindowMut<'_, N>)
where
    N : Scalar + Copy + Default + Zero + Copy + Serialize + DeserializeOwned +
        Any + std::ops::Mul<Output = N> + std::ops::AddAssign
{
    // assert!(out.height() == input.height() - filter.height() + 1);
    // assert!(out.width() == input.width() - filter.width() + 1);
    // assert!(filter.width() % 2 == 1);
    // assert!(filter.height() % 2 == 1);

    // Actually half the kernel size minus one, since size % 2 must be 1
    let half_kh = filter.height() / 2;
    let half_kw = filter.width() / 2;

    for (ix_ci, center_i) in (half_kh..(input.height() - half_kh)).enumerate() {
        for (ix_cj, center_j) in (half_kw..(input.width() - half_kw)).enumerate() {
            out[(ix_ci, ix_cj)] = N::zero();
            for (ix_ki, ki) in ((center_i - half_kh)..(center_i + half_kh + 1)).enumerate() {
                for (ix_kj, kj) in ((center_j - half_kw)..(center_j + half_kw + 1)).enumerate() {
                    out[(ix_ci, ix_cj)] += input[(ki, kj)] * filter[(ix_ki, ix_kj)];
                }
            }
        }
    }
}

#[cfg(feature="ipp")]
unsafe fn ipp_sep_convolution_f32(
    img : &Window<f32>,
    row_kernel : &Window<f32>,
    col_kernel : &Window<f32>,
    out : &mut WindowMut<f32>
) {

    use crate::foreign::ipp::ippi::*;

    // assert!(out.width() == img.width() - kernel.width() + 1);
    // assert!(out.height() == img.height() - kernel.height() + 1);

    // Separable convolution is implemented with border only

    assert!(row_kernel.height() == 1 || row_kernel.width() == 1);
    assert!(col_kernel.height() == 1 || col_kernel.width() == 1);
    assert!(row_kernel.width() == col_kernel.width());
    assert!(row_kernel.height() == col_kernel.height());

    let kernel_dim = row_kernel.width().max(row_kernel.height());
    let kernel_sz = crate::foreign::ipp::ippi::IppiSize { width : kernel_dim as i32, height : kernel_dim as i32  };

    let img_sz = crate::image::ipputils::window_size(img);

    let img_stride = crate::image::ipputils::byte_stride_for_window(img);
    // let kernel_stride = crate::image::ipputils::byte_stride_for_window(kernel);
    let out_stride = crate::image::ipputils::byte_stride_for_window_mut(out);

    let src_ty = crate::foreign::ipp::ippcore::IppDataType_ipp32f;
    let kernel_ty = src_ty;
    let n_channels = 1;

    let mut buf_sz = 0;
    let ans = crate::foreign::ipp::ippcv::ippiFilterSeparableGetBufferSize(
        mem::transmute(img_sz),
        mem::transmute(kernel_sz),
        src_ty,
        kernel_ty,
        n_channels,
        &mut buf_sz as *mut _
    );
    assert!(ans == 0);

    let mut spec_sz = 0;
    let ans = crate::foreign::ipp::ippcv::ippiFilterSeparableGetSpecSize(
        mem::transmute(kernel_sz),
        src_ty,
        n_channels,
        &mut spec_sz as *mut _
    );
    assert!(ans == 0);

    let mut buffer : Vec<u8> = Vec::from_iter((0..buf_sz).map(|_| 0u8 ) );
    let mut spec : Vec<u8> = Vec::from_iter((0..spec_sz).map(|_| 0u8 ) );

    let num_channels = 1;
    let ans = crate::foreign::ipp::ippcv::ippiFilterSeparableInit_32f(
        row_kernel.as_ptr(),
        col_kernel.as_ptr(),
        mem::transmute(kernel_sz),
        crate::foreign::ipp::ippcore::IppDataType_ipp32f,
        num_channels,
        mem::transmute(spec.as_mut_ptr())
    );
    assert!(ans == 0);

    let border_const_val = 0.0f32;
    let ans = crate::foreign::ipp::ippcv::ippiFilterSeparable_32f_C1R(
        img.as_ptr(),
        img_stride,
        out.as_mut_ptr(),
        out_stride,
        mem::transmute(img_sz),
        crate::foreign::ipp::ippcv::_IppiBorderType_ippBorderRepl,
        border_const_val,
        mem::transmute(spec.as_ptr()),
        buffer.as_mut_ptr()
    );
    assert!(ans == 0);
}

#[cfg(feature="ipp")]
unsafe fn ipp_convolution_f32(img : &Window<f32>, kernel : &Window<f32>, out : &mut WindowMut<f32>) {

    use crate::foreign::ipp::ippi::*;
    use std::os::raw::c_int;

    assert!(out.width() == img.width() - kernel.width() + 1);
    assert!(out.height() == img.height() - kernel.height() + 1);

    // With IppiROIShape_ippiROIFull  convolution with zero padding is applied (result is nrow_img + nrow_kenel - 1).
    // With IppiROIShape_ippiROIValid convolution without zero padding is applied (result is nrow_img - nrow_kernel + 1)
    let alg_ty = (IppAlgType_ippAlgAuto + IppiROIShape_ippiROIValid) as i32;

    let img_sz = crate::image::ipputils::window_size(img);
    let kernel_sz = crate::image::ipputils::window_size(kernel);

    let img_stride = crate::image::ipputils::byte_stride_for_window(img);
    let kernel_stride = crate::image::ipputils::byte_stride_for_window(kernel);
    let out_stride = crate::image::ipputils::byte_stride_for_window_mut(out);

    let num_channels = 1;
    let mut buf_sz : c_int = 0;
    let status = ippiConvGetBufferSize(
        img_sz,
        kernel_sz,
        crate::foreign::ipp::ippcore::IppDataType_ipp32f,
        num_channels,
        alg_ty.clone() as i32,
        &mut buf_sz
    );
    assert!(status == 0 && buf_sz > 0);
    let mut conv_buffer : Vec<u8> = Vec::from_iter((0..buf_sz).map(|_| 0u8 ) );

    let status = ippiConv_32f_C1R(
        img.as_ptr(),
        img_stride,
        img_sz,
        kernel.as_ptr(),
        kernel_stride,
        kernel_sz,
        out.as_mut_ptr(),
        out_stride,
        alg_ty,
        conv_buffer.as_mut_ptr()
    );
    assert!(status == 0);
}

// Waiting GAT stabilization
/*impl<N> Convolve for Image<N>
where
    N : Scalar + Copy + Default + Zero + Copy + Serialize + DeserializeOwned +
        Any + std::ops::Mul<Output = N> + std::ops::AddAssign
{

}*/

impl<'a, N> Convolve for Window<'a, N>
where
N : Scalar + Copy + Default + Zero + Copy + Serialize + DeserializeOwned +
    Any + std::ops::Mul<Output = N> + std::ops::AddAssign
{

    type Output = WindowMut<'a, N>;

    type OwnedOutput = Image<N>;

    fn convolve_mut(&self, filter : &Self, conv : Convolution, out : &mut Self::Output) {

        #[cfg(feature="ipp")]
        unsafe {
            if self.pixel_is::<f32>() {
                ipp_convolution_f32(mem::transmute(self), mem::transmute(filter), mem::transmute(out));
                return;
            }

            if (&self[(0usize, 0usize)] as &dyn Any).is::<f64>() {

            }
        }

        #[cfg(feature="mkl")]
        {
            /*if (&self[0] as &dyn Any).is::<f32>() {
                // Dispatch to MKL impl
            }

            if (&self[0] as &dyn Any).is::<f64>() {
                // Dispatch to MKL impl
            }*/
        }

        #[cfg(feature="opencv")]
        {
            use opencv;

            // println!("Processing with opencv");

            assert!(filter.height() % 2 != 0 && filter.width() % 2 != 0 );
            let input : opencv::core::Mat = self.into();
            let kernel : opencv::core::Mat = filter.into();
            let mut flip_kernel = kernel.clone();
            opencv::core::flip(&kernel, &mut flip_kernel, -1).unwrap();
            let delta = 0.0;
            let mut output : opencv::core::Mat = out.into();
            opencv::imgproc::filter_2d(
                &input,
                &mut output,
                cvutils::get_cv_type::<N>(),
                &flip_kernel,
                opencv::core::Point2i::new(0, 0),
                delta,
                opencv::core::BORDER_DEFAULT
            ).unwrap();
            return;
        }

        // println!("Processing with baseline");
        baseline_convolution(&self, &filter, out);
    }

    fn convolve(&self, filter : &Self, conv : Convolution) -> Image<N> {
        let (height, width) = linear_conv_sz(self.shape(), filter.shape());
        let mut out = Image::new_constant(height, width, N::zero());
        self.convolve_mut(filter, conv, &mut out.full_window_mut());
        out
    }

}

pub fn local_min(src : Window<'_, u8>, dst : WindowMut<'_, u8>, kernel_sz : usize) -> Image<u8> {
    unimplemented!()
}

pub fn local_max(src : Window<'_, u8>, dst : WindowMut<'_, u8>, kernel_sz : usize) -> Image<u8> {
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

/*IppStatus ippiFilterBilateral_<mod>(const Ipp<srcdatatype>* pSrc, int srcStep,
Ipp<dstdatatype>* pDst, int dstStep, IppiSize dstRoiSize, IppiBorderType borderType,
const Ipp<datatype> pBorderValue[1], const IppiFilterBilateralSpec* pSpec, Ipp8u*
pBuffer );

IppStatus ippiFilterBox_64f_C1R(const Ipp<datatype>* pSrc, int srcStep, Ipp<datatype>*
pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize, IppiPoint anchor );

IppStatus ippiFilterMedian_<mod>(const Ipp<datatype>* pSrc, int srcStep, Ipp<datatype>*
pDst, int dstStep, IppiSize dstRoiSize, IppiSize maskSize, IppiPoint anchor, Ipp8u*
pBuffer );

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




