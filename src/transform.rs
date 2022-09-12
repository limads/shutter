use crate::image::*;

pub struct Distance {

}

pub struct Polar {
    pub magn : ImageBuf<f32>,
    pub phase : ImageBuf<f32>
}

pub struct Haar {
    
}

/* Contains integral image, where pixel (i, j) is the sum of the rectanlge a <= i, b <= i */
pub struct Integral {

}

/* Contains cumulative sums over rows and cumulative sums over cols, such
that the last column and/or row contains the image profile */
pub enum Profile<P> {
    Rows(ImageBuf<P>),
    Cols(ImageBuf<P>),
    Both { rows : ImageBuf<P>, cols : ImageBuf<P> }
}

#[derive(Debug, Clone, Copy)]
pub enum Mask {

    // 3x3 mask
    Three,

    // 5x5 mask
    Five

}

pub enum Norm {
    L1,
    L2,
    Inf
}

#[cfg(feature="ipp")]
pub fn distance_transform(src : &Window<'_, u8>, dst : &mut WindowMut<'_, u8>, mask : Mask, norm : Norm) {

    assert!(src.shape() == dst.shape());

    let (src_byte_stride, roi) = crate::image::ipputils::step_and_size_for_window(src);
    let dst_byte_stride = crate::image::ipputils::byte_stride_for_window_mut(&dst);

    // 3x3 window uses only 2 first entries; 5x5 window uses all three entries.
    let mut metrics : [i32; 3] = [0, 0, 0];

    let mask_size = match mask {
        Mask::Three => 3,
        Mask::Five => 5
    };

    let norm_code = match norm {
        Norm::L1 => crate::foreign::ipp::ippi::_IppiNorm_ippiNormL1,
        Norm::L2 => crate::foreign::ipp::ippi::_IppiNorm_ippiNormL2,
        Norm::Inf => crate::foreign::ipp::ippi::_IppiNorm_ippiNormInf
    };
    unsafe {
        let ans = crate::foreign::ipp::ippcv::ippiGetDistanceTransformMask_32s(
            mask_size,
            norm_code,
            metrics.as_mut_ptr()
        );
        assert!(ans == 0);

        let ans = match mask {
            Mask::Three => {
                crate::foreign::ipp::ippcv::ippiDistanceTransform_3x3_8u_C1R(
                    src.as_ptr(),
                    src_byte_stride,
                    dst.as_mut_ptr(),
                    dst_byte_stride,
                    std::mem::transmute(roi),
                    metrics.as_mut_ptr()
                )
            },
            Mask::Five => {
                crate::foreign::ipp::ippcv::ippiDistanceTransform_5x5_8u_C1R(
                    src.as_ptr(),
                    src_byte_stride,
                    dst.as_mut_ptr(),
                    dst_byte_stride,
                    std::mem::transmute(roi),
                    metrics.as_mut_ptr()
                )
            }
        };
        assert!(ans == 0);
    }
}

