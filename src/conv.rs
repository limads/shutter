use crate::image::*;
use std::mem;
use std::any::Any;
use serde::{Serialize, Deserialize, de::DeserializeOwned};
use num_traits::{Zero, AsPrimitive};
use nalgebra::Scalar;
use std::iter::FromIterator;
pub use ripple::conv::*;
use std::fmt::Debug;
// use crate::raster::Raster;
use std::ops::Add;

// Ipp only has separable convolution for f32 and i16 types; and
// conventional convolution for f32, i16 and u8 types.

// Before convolving an u8 image, be sure to divide the image the filter
// denominator scale factor, then convolve with the unscaled filter to
// get the result (they are prefixed with UNSCALED_). This guarantees
// the output will be in the domain of the original image and won't be saturated.s

#[derive(Debug, Clone)]
pub struct SepFilter<T>
where
    T : Pixel
{
    pub col : ImageBuf<T>,
    pub row : ImageBuf<T>
}

impl<T> SepFilter<T>
where
    T : Pixel + Scalar + Copy + Debug + Default + Zero,
    f32 : AsPrimitive<T>
{

    // If the rank of the matrix is 1, the filter is separable via SVD.
    pub fn calculate(win : Window<f32>) -> Option<Self> {
        let (h, w) = win.shape();
        let m = nalgebra::DMatrix::from_iterator(w, h, win.pixels(1).copied()).transpose();
        let svd = nalgebra::linalg::SVD::new(m, true, true);
        if svd.rank(10.0e-8) == 1 {
            let sroot = svd.singular_values[0].sqrt();
            let mut col = ImageBuf::<T>::from_iter(svd.u.as_ref().unwrap().column(0).iter()
                .map(|c| { let out : T = (c*sroot).as_(); out }), 1);
            let mut row = ImageBuf::<T>::from_iter(svd.v_t.as_ref().unwrap().row(0).iter()
                .map(|r| { let out : T = (r*sroot).as_(); out }), w);
            Some(Self { row, col })
        } else {
            None
        }
    }

}

/// Implements separable convolution, if the filter can be represented as the outer product
/// of a row and column (which themselves are represented as windows). After Szelisky (2010):
/// If K = vh^T, K is separable if the first singular value of K is non-zero, and
/// sqrt(sigma_0) u0 and sqrt(sigma_0) v0^T (first left and right singular vectors weighted
/// by first singular value) are the separate components of the filter. Separable convolutions
/// reduces the number of operations from K^2 multiply-adds per pixel to 2K multiply-adds per pixel.
pub trait ConvolveSep<O, M> {

    // Separable convolution is with border only input.shape() == output.shape()
    fn convolve_sep_to(&self, filter_vert : &Self, filter_horiz : &Self, out : &mut M);

    // Separable convolution is with border only input.shape() == output.shape()
    fn convolve_sep(&self, filter_vert : &Self, filter_horiz : &Self) -> O;

}

impl<P, S, T> ConvolveSep<ImageBuf<P>, Image<P, T>> for Image<P, S>
where
    P : Pixel,
    S : Storage<P>,
    T : StorageMut<P>
{

    fn convolve_sep_to(
        &self, 
        filter_vert : &Self, 
        filter_horiz : &Self, 
        out : &mut Image<P, T>
    ) {

        #[cfg(feature="ipp")]
        unsafe {
            if self.pixel_is::<f32>() {
                ipp_sep_convolution_float(
                    mem::transmute(&self.full_window()),
                    mem::transmute(&filter_vert.full_window()),
                    mem::transmute(&filter_horiz.full_window()),
                    mem::transmute(&mut out.full_window_mut())
                );
                return;
            } else if self.pixel_is::<i16>() {
                ipp_sep_convolution_integer(
                    mem::transmute(&self.full_window()),
                    mem::transmute(&filter_vert.full_window()),
                    mem::transmute(&filter_horiz.full_window()),
                    mem::transmute(&mut out.full_window_mut())
                );
                return;
            } else if self.pixel_is::<i32>() {
                /*ipp_sep_convolution_integer(
                    self,
                    filter_vert,
                    filter_horiz,
                    out
                );
                return;*/
                unimplemented!()
            }
        }

        unimplemented!();
    }

    fn convolve_sep(
        &self, 
        filter_vert : &Self, 
        filter_horiz : &Self, 
    ) -> ImageBuf<P> {
        let mut out = unsafe { Image::new_empty_like(self) };
        self.convolve_sep_to(
            filter_vert, 
            filter_horiz,
            &mut out.full_window_mut()
        );
        out
    }

}

const FRAC_NEG_2_4 : f32 = -0.5;

const FRAC_4_4 : f32 = 1.0;

const FRAC_2_4 : f32 = 0.5;

const FRAC_1_4 : f32 = 0.25;

const FRAC_1_2 : f32 = 0.5;

const FRAC_NEG_2_2 : f32 = -1.;

const FRAC_1_16 : f32 = 0.0625;

const FRAC_2_16 : f32 = 0.125;

const FRAC_4_16 : f32 = 0.25;

const FRAC_6_16 : f32 = 0.375;

const FRAC_1_256 : f32 = 0.00390625;

const FRAC_4_256 : f32 = 0.015625;

const FRAC_6_256 : f32 = 0.0234375;

const FRAC_16_256 : f32 = 0.0625;

const FRAC_24_256 : f32 = 0.09375;

const FRAC_36_256 : f32 = 0.140625;

const FRAC_1_3 : f32 = 0.333333333;

const FRAC_1_9 : f32 = 0.111111111;

const FRAC_1_25 : f32 = 0.04;

/// Separable convolution filters useful to calculate texture energy.
/// Any outer product combination of the filters here is possible to produce
/// an texture energy output image. The energy of the local
/// texture region can be calculated by sum_i sum_j abs(M_ij) (summing
/// over absolute value of pixels inside a local output window).
/// The output of multiple combinations of such filters can be used
/// for texture classification (Laws, 1980; Parker, 2011, p. 191).
pub mod energy {

    use super::*;

    pub const E5_SEP : Window<'static, f32> = Window::from_static::<5, 5>(&[
        -1., -2., 0., 2., 1.
    ]);

    pub const L5_SEP : Window<'static, f32> = Window::from_static::<5, 5>(&[
        1., 4., 6., 4., 1.
    ]);

    pub const R5_SEP : Window<'static, f32> = Window::from_static::<5, 5>(&[
        1., -4., 6., -4., 1.
    ]);

}

/// Common difference (edge-detection) filters
pub mod edge {

    use crate::image::Window;
    use std::f32::consts::SQRT_2;
    use super::*;

    const NEG_SQRT_2 : f32 = -1.41421356237309504880168872420969808;

    // pub const ROBERTS_HORIZ : Window<'static, f32> = Window::from_static::<4, 2>(&[1., 0., 0., -1.]);

    // pub const ROBERTS_VERT : Window<'static, f32> = Window::from_static::<4, 2>(&[0., 1., -1., 0.]);

    // 3x3 directional filters

    // The sobel operator must be normalized by 1/8 (sum of absolute values of nonzero entries).
    // Perhaps call this "scaled" and "unscaled" sobel.
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

    pub const SOBEL_HORIZ_SEP_ROW : Window<'static, f32> = Window::from_static::<3, 3>(&[
        -0.5, 0., 0.5
    ]);
    
    pub const SOBEL_HORIZ_SEP_COL : Window<'static, f32> = Window::from_static::<3, 1>(&[
        -0.5, 0., 0.5
    ]);
    
    pub const SOBEL_VERT_SEP_ROW : Window<'static, f32> = Window::from_static::<3, 3>(&[
        -0.5, 0., 0.5
    ]);
    
    pub const SOBEL_VERT_SEP_COL : Window<'static, f32> = Window::from_static::<3, 1>(&[
        -0.5, 0., 0.5
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

    // Bilinear filter (Szelisky (p.116))
    pub const BILINEAR : Window<'static, f32> = Window::from_static::<9, 3>(&[
        FRAC_1_16, FRAC_2_16, FRAC_1_16,
        FRAC_2_16, FRAC_4_16, FRAC_2_16,
        FRAC_1_16, FRAC_2_16, FRAC_1_16
    ]);

    pub const BILINEAR_SEP : Window<'static, f32> = Window::from_static::<3, 3>(&[
        FRAC_1_4,  FRAC_2_4, FRAC_1_4
    ]);

    pub const BILINEAR_SEP_INT_COL : Window<'static, i16> = Window::from_static::<3, 1>(&[
        1, 2, 1
    ]);

    pub const BILINEAR_SEP_INT_ROW : Window<'static, i16> = Window::from_static::<3, 3>(&[
        1,  2, 1
    ]);

    // Corner filter (Szelisky, p. 116)
    pub const CORNER : Window<'static, f32> = Window::from_static::<9, 3>(&[
        FRAC_1_4, FRAC_NEG_2_4, FRAC_1_4,
        FRAC_NEG_2_4, FRAC_4_4, FRAC_NEG_2_4,
        FRAC_1_4, FRAC_NEG_2_4, FRAC_1_4
    ]);

    pub const CORNER_SEP : Window<'static, f32> = Window::from_static::<3, 3>(&[
        FRAC_1_2, FRAC_NEG_2_2, FRAC_1_2
    ]);

    pub const SHARPEN_COL : Window<'static, i16> = Window::from_static::<5, 1>(&[-1, -1, 8, -1, -1]);

    pub const SHARPEN_ROW : Window<'static, i16> = Window::from_static::<5, 5>(&[-1, -1, 8, -1, -1]);
    // This also yields a nice edge-enhancing effect
    // &Window::from_static::<7,1>(&[-2,1,0,8,0,1,-2]),
    // &Window::from_static::<7,7>(&[-2,1,0,8,0,1,-2]),

    pub const SOBEL_HORIZ_COL_INT : Window<'static, i16> = Window::from_static::<3, 1>(&[1, 2, 1]);

    pub const SOBEL_HORIZ_ROW_INT : Window<'static, i16> = Window::from_static::<3, 3>(&[1, 0, -1]);

    pub const SOBEL_HORIZ_COL_I32 : Window<'static, i32> = Window::from_static::<3, 1>(&[1, 2, 1]);

    pub const SOBEL_HORIZ_ROW_I32 : Window<'static, i32> = Window::from_static::<3, 3>(&[1, 0, -1]);

    pub const SOBEL_VERT_COL_INT : Window<'static, i16> = Window::from_static::<3, 1>(&[1, 0, -1]);

    pub const SOBEL_VERT_ROW_INT : Window<'static, i16> = Window::from_static::<3, 3>(&[1, 2, 1]);

    // TODO difference of gaussian 5x5 with DOG_21 with var(2) - var(1); DOG_31 with var(3) - var(1) and so on.

}

/// Common weighted averaging (blur) filters
pub mod blur {

    use crate::image::Window;
    use super::*;

    pub const BOX3_SEP : Window<'static, f32> = Window::from_static::<3, 3>(&[
        FRAC_1_3, FRAC_1_3, FRAC_1_3
    ]);

    pub const BOX5_SEP : Window<'static, f32> = Window::from_static::<5, 5>(&[
        1., 1., 1., 1., 1.
    ]);

    pub const BOX3 : Window<'static, f32> = Window::from_static::<9, 3>(&[
        FRAC_1_9, FRAC_1_9, FRAC_1_9,
        FRAC_1_9, FRAC_1_9, FRAC_1_9,
        FRAC_1_9, FRAC_1_9, FRAC_1_9,
    ]);

    // Divide your input by 9 before applying convolution.
    pub const UNSCALED_BOX3_U16 : Window<'static, i16> = Window::from_static::<9, 3>(&[
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
    ]);

    pub const UNSCALED_BOX3_U8 : Window<'static, u8> = Window::from_static::<9, 3>(&[
        1, 1, 1,
        1, 1, 1,
        1, 1, 1,
    ]);

    pub const BOX5 : Window<'static, f32> = Window::from_static::<25, 5>(&[
        FRAC_1_25,FRAC_1_25,FRAC_1_25,FRAC_1_25,FRAC_1_25,
        FRAC_1_25,FRAC_1_25,FRAC_1_25,FRAC_1_25,FRAC_1_25,
        FRAC_1_25,FRAC_1_25,FRAC_1_25,FRAC_1_25,FRAC_1_25,
        FRAC_1_25,FRAC_1_25,FRAC_1_25,FRAC_1_25,FRAC_1_25,
        FRAC_1_25,FRAC_1_25,FRAC_1_25,FRAC_1_25,FRAC_1_25
    ]);

    // Divide your output by 25 after applying convolution.
    pub const UNSCALED_BOX5_I16 : Window<'static, i16> = Window::from_static::<25, 5>(&[
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
    ]);

    // Divide your output by 25 after applying convolution.
    pub const UNSCALED_BOX5_U8 : Window<'static, u8> = Window::from_static::<25, 5>(&[
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
    ]);

    // Divide your output by 256 after applying convolution.
    pub const UNSCALED_GAUSS_I16 : Window<'static, i16> = Window::from_static::<25, 5>(&[
        1, 4,  6,  4,  1,
        4, 16, 24, 16, 4,
        6, 24, 36, 24, 6,
        4, 16, 24, 16, 4,
        1, 4,  6,  4,  1,
    ]);

    // Divide your output by 256 after applying convolution.
    pub const UNSCALED_GAUSS_U8 : Window<'static, u8> = Window::from_static::<25, 5>(&[
        1, 4,  6,  4,  1,
        4, 16, 24, 16, 4,
        6, 24, 36, 24, 6,
        4, 16, 24, 16, 4,
        1, 4,  6,  4,  1,
    ]);

    // Gaussian filter (Szelisky, p. 116).
    pub const GAUSS : Window<'static, f32> = Window::from_static::<25, 5>(&[
        FRAC_1_256, FRAC_4_256, FRAC_6_256, FRAC_4_256, FRAC_1_256,
        FRAC_4_256, FRAC_16_256, FRAC_24_256, FRAC_16_256, FRAC_4_256,
        FRAC_6_256, FRAC_24_256, FRAC_36_256, FRAC_24_256, FRAC_6_256,
        FRAC_4_256, FRAC_16_256, FRAC_24_256, FRAC_16_256, FRAC_4_256,
        FRAC_1_256, FRAC_4_256, FRAC_6_256, FRAC_4_256, FRAC_1_256
    ]);

    pub const GAUSS_SEP : Window<'static, f32> = Window::from_static::<5, 5>(&[
        FRAC_1_16, FRAC_4_16, FRAC_6_16, FRAC_4_16, FRAC_1_16
    ]);

}

pub fn convolution_buffer<N>(img_sz : (usize, usize), kernel_sz : (usize, usize)) -> ImageBuf<N>
where
    N : Pixel
{
    let (nrow, ncol) = linear_conv_sz(img_sz, kernel_sz);
    Image::new_constant(nrow, ncol, N::zero())
}

pub fn linear_conv_sz(img_sz : (usize, usize), kernel_sz : (usize, usize)) -> (usize, usize) {
    assert!(img_sz.0 > kernel_sz.0 && img_sz.1 > kernel_sz.1, "Image size = {:?}; Kernel size = {:?}", img_sz, kernel_sz);
    let nrow = img_sz.0 - kernel_sz.0 + 1;
    let ncol = img_sz.1 - kernel_sz.1 + 1;
    (nrow, ncol)
}

fn padded_baseline_convolution<'a, N>(input : &Window<'_, N>, filter : &Window<'_, N>, out : &'a mut WindowMut<'a, N>)
where
    N : Pixel + Scalar + Copy + Default + Zero + Copy + Serialize + DeserializeOwned +
    Any + std::ops::Mul<Output = N> + std::ops::AddAssign
{
    let mut extended_in = ImageBuf::<N>::new_constant(
        input.height() + filter.height() / 2 + 1,
        input.width() + filter.width() / 2 + 1,
        N::zero()
    );
    let mut extended_out = extended_in.clone();
    extended_in.window_mut(((filter.height() / 2), (filter.width() / 2)), out.shape()).unwrap().copy_from(&input);
    baseline_convolution(
        &extended_in.full_window(), 
        filter, 
        &mut extended_out.full_window_mut()
    );
    let ext_win = extended_out.window(
        ((filter.height() / 2), (filter.width() / 2)), 
        out.shape()
    ).unwrap();
    out.copy_from(&ext_win);
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

fn baseline_convolution<N>(
    input : &Window<'_, N>, 
    filter : &Window<'_, N>, 
    out : &mut WindowMut<'_, N>
) where
    // for<'a>&[N]
    N : Pixel + std::ops::Mul<Output = N> + std::ops::AddAssign
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
struct ConvSepParams {
    img_sz : crate::foreign::ipp::ippi::IppiSize,
    img_stride : i32,
    out_stride : i32,
    kernel_sz : crate::foreign::ipp::ippi::IppiSize,
    src_ty : i32,
    border_ty : u32,
    num_channels : i32,
    buffer : Vec<u8>,
    spec : Vec<u8>
}

#[cfg(feature="ipp")]
impl ConvSepParams {

    unsafe fn evaluate<T>(
        img : &Window<T>,
        row_kernel : &Window<T>,
        col_kernel : &Window<T>,
        out : &mut WindowMut<T>
    ) -> Self
    where
        T : Pixel + Scalar + Copy + Debug + Default,
        for<'a> &'a [T] : Storage<T>,
        for<'a> &'a mut [T] : StorageMut<T>,
    {
        use crate::foreign::ipp::ippi::*;

        // Separable convolution is with border only.
        assert!(img.shape() == out.shape());

        assert!(row_kernel.height() == 1);
        assert!(col_kernel.width() == 1);
        assert!(row_kernel.width() == col_kernel.height());
        // assert!(row_kernel.height() * row_kernel.width() == col_kernel.height() * col_kernel.width());

        let kernel_dim = row_kernel.width().max(row_kernel.height());
        let kernel_sz = crate::foreign::ipp::ippi::IppiSize { width : kernel_dim as i32, height : kernel_dim as i32  };

        let img_sz = crate::image::ipputils::window_size(img);

        let img_stride = crate::image::ipputils::byte_stride_for_window(img);
        // let kernel_stride = crate::image::ipputils::byte_stride_for_window(kernel);
        let out_stride = crate::image::ipputils::byte_stride_for_window_mut(out);

        let src_ty = if img.pixel_is::<f32>() {
            crate::foreign::ipp::ippcore::IppDataType_ipp32f
        } else if img.pixel_is::<i16>() {
            crate::foreign::ipp::ippcore::IppDataType_ipp16s
        } else if img.pixel_is::<i32>() {
            crate::foreign::ipp::ippcore::IppDataType_ipp32s
        } else {
            panic!("Invalid data type")
        };

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

        let border_ty = crate::foreign::ipp::ippcv::_IppiBorderType_ippBorderRepl;
        let num_channels = 1;
        Self {
            img_sz,
            img_stride,
            out_stride,
            kernel_sz,
            src_ty,
            border_ty,
            num_channels,
            buffer,
            spec
        }
    }

}

#[cfg(feature="ipp")]
unsafe fn ipp_sep_convolution_integer(
    img : &Window<i16>,
    col_kernel : &Window<i16>,
    row_kernel : &Window<i16>,
    out : &mut WindowMut<i16>
) {
    use crate::foreign::ipp::ippi::*;

    // Since the full kernel is the external product of two vecs, the sum over
    // the kernel is the sum over the products.
    // let divisor=1;
    let divisor = (col_kernel.sum::<f32>(1) as i32).max(1);
    // let divisor = (col_kernel.pixels(1).zip(row_kernel.pixels(1)).fold(0, |s, (a, b)| s+a.abs()*b.abs()) as i32).max(1);

    let mut params = ConvSepParams::evaluate(img, row_kernel, col_kernel, out);

    let ans = if img.pixel_is::<i16>() {
        let scale_factor = 1i32;
        crate::foreign::ipp::ippcv::ippiFilterSeparableInit_16s(
            row_kernel.as_ptr(),
            col_kernel.as_ptr(),
            mem::transmute(params.kernel_sz),
            divisor,
            scale_factor,
            params.src_ty,
            params.num_channels,
            mem::transmute(params.spec.as_mut_ptr())
        )
    } else if img.pixel_is::<i32>() {
        /*let scale_factor = 1i32;
        crate::foreign::ipp::ippcv::ippiFilterSeparableInit_32s(
            mem::transmute(row_kernel.as_ptr()),
            mem::transmute(col_kernel.as_ptr()),
            mem::transmute(params.kernel_sz),
            divisor,
            scale_factor,
            params.src_ty,
            params.num_channels,
            mem::transmute(params.spec.as_mut_ptr())
        )*/
        unimplemented!()
    } else {
        panic!("Invalid pixel type");
    };
    assert!(ans == 0);
    let ans = if img.pixel_is::<i16>() {
        let border_const_val = 0i16;
        crate::foreign::ipp::ippcv::ippiFilterSeparable_16s_C1R(
            mem::transmute(img.as_ptr()),
            params.img_stride,
            mem::transmute(out.as_mut_ptr()),
            params.out_stride,
            mem::transmute(params.img_sz),
            params.border_ty,
            border_const_val,
            mem::transmute(params.spec.as_ptr()),
            params.buffer.as_mut_ptr()
        )
    } else if img.pixel_is::<i32>() {
        /*let border_const_val = 0i32;
        crate::foreign::ipp::ippcv::ippiFilterSeparable_32s_C1R(
            mem::transmute(img.as_ptr()),
            params.img_stride,
            mem::transmute(out.as_mut_ptr()),
            params.out_stride,
            mem::transmute(params.img_sz),
            params.border_ty,
            border_const_val,
            mem::transmute(params.spec.as_ptr()),
            params.buffer.as_mut_ptr()
        )*/
        unimplemented!()
    } else {
        panic!("Invalid pixel type");
    };
    assert!(ans == 0);
}

#[cfg(feature="ipp")]
unsafe fn ipp_sep_convolution_float(
    img : &Window<f32>,
    col_kernel : &Window<f32>,
    row_kernel : &Window<f32>,
    out : &mut WindowMut<f32>
) {

    use crate::foreign::ipp::ippi::*;
    let mut params = ConvSepParams::evaluate(img, row_kernel, col_kernel, out);

    let ans = crate::foreign::ipp::ippcv::ippiFilterSeparableInit_32f(
        row_kernel.as_ptr(),
        col_kernel.as_ptr(),
        mem::transmute(params.kernel_sz),
        params.src_ty,
        params.num_channels,
        mem::transmute(params.spec.as_mut_ptr())
    );
    assert!(ans == 0);

    let border_const_val = 0.0f32;
    let ans = crate::foreign::ipp::ippcv::ippiFilterSeparable_32f_C1R(
        img.as_ptr(),
        params.img_stride,
        out.as_mut_ptr(),
        params.out_stride,
        mem::transmute(params.img_sz),
        params.border_ty,
        border_const_val,
        mem::transmute(params.spec.as_ptr()),
        params.buffer.as_mut_ptr()
    );
    assert!(ans == 0);
}

#[cfg(feature="ipp")]
pub struct ConvParams {
    alg_ty : i32,
    img_stride : i32,
    kernel_stride : i32,
    out_stride : i32,
    dtype : i32,
    conv_buffer : Vec<u8>,
    img_sz : crate::foreign::ipp::ippi::IppiSize,
    kernel_sz : crate::foreign::ipp::ippi::IppiSize
}

#[cfg(feature="ipp")]
impl ConvParams {

    unsafe fn evaluate<T>(img : &Window<T>, kernel : &Window<T>, out : &mut WindowMut<T>) -> Self
    where
        T : Pixel + Scalar + Debug + Copy,
        for<'a> &'a [T] : Storage<T>,
        for<'a> &'a mut [T] : StorageMut<T>,
    {

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
        let dtype = if img.pixel_is::<f32>() {
            crate::foreign::ipp::ippcore::IppDataType_ipp32f
        } else if img.pixel_is::<u8>() {
            crate::foreign::ipp::ippcore::IppDataType_ipp8u
        } else if img.pixel_is::<i16>() {
            crate::foreign::ipp::ippcore::IppDataType_ipp16s
        } else {
            panic!("Invalid type")
        };
        let mut buf_sz : c_int = 0;
        let num_channels = 1;
        let status = ippiConvGetBufferSize(
            img_sz,
            kernel_sz,
            dtype,
            num_channels,
            alg_ty.clone() as i32,
            &mut buf_sz
        );
        assert!(status == 0 && buf_sz > 0);
        let mut conv_buffer : Vec<u8> = Vec::from_iter((0..buf_sz).map(|_| 0u8 ) );
        Self {
            alg_ty,
            img_stride,
            kernel_stride,
            dtype,
            out_stride,
            img_sz,
            kernel_sz,
            conv_buffer
        }
    }

}

#[cfg(feature="ipp")]
unsafe fn ipp_convolution_i16(img : &Window<i16>, kernel : &Window<i16>, out : &mut WindowMut<i16>) {

    use crate::foreign::ipp::ippi::*;
    use std::os::raw::c_int;

    let mut params = ConvParams::evaluate(img, kernel, out);
    // let divisor = (crate::global::sum::<_, f64>(kernel, 1) as i32).max(1);
    let divisor = 1;

    let status = ippiConv_16s_C1R(
        img.as_ptr(),
        params.img_stride,
        params.img_sz,
        kernel.as_ptr(),
        params.kernel_stride,
        params.kernel_sz,
        out.as_mut_ptr(),
        params.out_stride,
        divisor,
        params.alg_ty,
        params.conv_buffer.as_mut_ptr()
    );
    assert!(status == 0);
}

#[cfg(feature="ipp")]
unsafe fn ipp_convolution_u8(img : &Window<u8>, kernel : &Window<u8>, out : &mut WindowMut<u8>) {

    use crate::foreign::ipp::ippi::*;
    use std::os::raw::c_int;

    let mut params = ConvParams::evaluate(img, kernel, out);

    // In the worst case that the image content is u8::MAX and the kernel
    // is u8::MAX, the result will be divided by u8::MAX*kernel_sz, thus
    // making the output u8::MAX.
    let divisor = (kernel.sum::<f32>(1) as i32).max(1);
    // let divisor = 1;

    let status = ippiConv_8u_C1R(
        img.as_ptr(),
        params.img_stride,
        params.img_sz,
        kernel.as_ptr(),
        params.kernel_stride,
        params.kernel_sz,
        out.as_mut_ptr(),
        params.out_stride,
        divisor,
        params.alg_ty,
        params.conv_buffer.as_mut_ptr()
    );
    assert!(status == 0);
}

#[cfg(feature="ipp")]
unsafe fn ipp_convolution_f32(img : &Window<f32>, kernel : &Window<f32>, out : &mut WindowMut<f32>) {

    use crate::foreign::ipp::ippi::*;
    use std::os::raw::c_int;

    let mut params = ConvParams::evaluate(img, kernel, out);
    let status = ippiConv_32f_C1R(
        img.as_ptr(),
        params.img_stride,
        params.img_sz,
        kernel.as_ptr(),
        params.kernel_stride,
        params.kernel_sz,
        out.as_mut_ptr(),
        params.out_stride,
        params.alg_ty,
        params.conv_buffer.as_mut_ptr()
    );
    assert!(status == 0);
}

/*#[cfg(feature="ipp")]
unsafe fn ipp_convolution_i32(img : &Window<i32>, kernel : &Window<i32>, out : &mut WindowMut<i32>) {

    use crate::foreign::ipp::ippi::*;
    use std::os::raw::c_int;

    let mut params = ConvParams::evaluate(img, kernel, out);

    // In the worst case that the image content is u8::MAX and the kernel
    // is u8::MAX, the result will be divided by u8::MAX*kernel_sz, thus
    // making the output u8::MAX.
    let divisor = (crate::global::sum::<_, f64>(kernel, 1) as i32).max(1);

    let status = ippiConv_32s_C1R(
        img.as_ptr(),
        params.img_stride,
        params.img_sz,
        kernel.as_ptr(),
        params.kernel_stride,
        params.kernel_sz,
        out.as_mut_ptr(),
        params.out_stride,
        divisor,
        params.alg_ty,
        params.conv_buffer.as_mut_ptr()
    );
    assert!(status == 0);
}*/

// Waiting GAT stabilization
/*impl<N> Convolve for Image<N>
where
    N : Scalar + Copy + Default + Zero + Copy + Serialize + DeserializeOwned +
        Any + std::ops::Mul<Output = N> + std::ops::AddAssign
{

}*/

impl<P, S, T> Convolve<ImageBuf<P>, Image<P, T>> for Image<P, S>
where
    S : Storage<P>,
    T : StorageMut<P>,
    P : Pixel + std::ops::Mul<Output = P> + std::ops::AddAssign
{

    // Convolution is whithout border (require out.width == self.width-kernel.width+1
    // and out.height == self.height-kernel.height+1). To keep the original size, use
    // convolve_padded_mut, which will write the result to the center of the buffer.
    fn convolve_padded_mut(&self, filter : &Self, out : &mut Image<P, T>) {
        assert!(self.shape() == out.shape());
        let shape = (self.height()-filter.height()+1, self.width()-filter.width()+ 1);
        let mut out_sub = out.sub_window_mut(
            (filter.height() / 2, filter.width() / 2), 
            shape
        ).unwrap();
        self.convolve_mut(
            filter,
            &mut out_sub
        );
    }

    fn convolve_mut(&self, filter : &Self, out : &mut Image<P, T>) {

        #[cfg(feature="ipp")]
        unsafe {
            if self.pixel_is::<f32>() {
                ipp_convolution_f32(mem::transmute(self), mem::transmute(filter), mem::transmute(out));
                return;
            }

            if self.pixel_is::<u8>() {
                ipp_convolution_u8(
                    mem::transmute(self), 
                    mem::transmute(filter), 
                    mem::transmute(out)
                );
                return;
            }

            if self.pixel_is::<i16>() {
                ipp_convolution_i16(mem::transmute(self), mem::transmute(filter), mem::transmute(out));
                return;
            }

            /*if self.pixel_is::<i32>() {
                ipp_convolution_i32(mem::transmute(self), mem::transmute(filter), mem::transmute(out));
                return;
            }*/
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

        /*#[cfg(feature="opencv")]
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
        }*/

        // println!("Processing with baseline");
        /*baseline_convolution(
            &self.full_window(), 
            &filter.full_window(), 
            &mut out.full_window_mut()
        );*/
        unimplemented!()
    }

    fn convolve(&self, filter : &Self) -> ImageBuf<P> {
        let (height, width) = linear_conv_sz(self.shape(), filter.shape());
        let mut out = unsafe { Image::new_empty(height, width) };
        self.convolve_mut(filter, &mut out.full_window_mut());
        out
    }

    fn convolve_padded(&self, filter : &Self) -> ImageBuf<P> {
        let (height, width) = linear_conv_sz(self.shape(), filter.shape());
        let mut out = unsafe { Image::new_empty(self.height(), self.width()) };
        let mut out_sub = out.window_mut(
            (filter.height() / 2, filter.width() / 2), 
            (height, width)
        ).unwrap();
        self.convolve_mut(
            filter, 
            &mut out_sub);
        out
    }

}

/*

IppStatus ippiFilterBilateralInit(IppiFilterBilateralType filter, IppiSize dstRoiSize,
int kernelWidthHeight, IppDataType dataType, int numChannels, IppiDistanceMethodType
distMethod, Ipp64f valSquareSigma, Ipp64f posSquareSigma, IppiFilterBilateralSpec*
pSpec);

IppStatus ippiFilterBilateralBorderInit(IppiFilterBilateralType filter, IppiSize
dstRoiSize, int radius, IppDataType dataType, int numChannels, IppiDistanceMethodType
distMethod, Ipp32f valSquareSigma, Ipp32f posSquareSigma, IppiFilterBilateralSpec*
pSpec);

ippiFilterBilateralBorder_<mod>(const

IppStatus ippiFilterBoxBorder_<mod>(const Ipp<datatype>* pSrc, int srcStep,
Ipp<datatype>* pDst, int dstStep, IppiSize roiSize, IppiSize maskSize, IppiBorderType
border, const Ipp<datatype>* borderValue, Ipp8u* pBuffer);

IppStatus ippiFilterGaussianBorder_<mod>(const Ipp<datatype>* pSrc, int srcStep,
Ipp<datatype>* pDst, int dstStep, IppiSize roiSize, Ipp<datatype> borderValue,
IppFilterGaussianSpec* pSpec, Ipp8u* pBuffer);

IppStatus ippiDecimateFilterRow_8u_C1R(const Ipp8u* pSrc, int srcStep, IppiSize
srcRoiSize, Ipp8u* pDst, int dstStep, IppiFraction fraction);

IppStatus ippiDecimateFilterColumn_8u_C1R(const Ipp8u* pSrc, int srcStep, IppiSize
srcRoiSize, Ipp8u* pDst, int dstStep, IppiFraction fraction);

*/
