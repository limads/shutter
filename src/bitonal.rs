use crate::image::*;
use std::convert::AsRef;

#[derive(Debug, Clone)]
pub struct BitonalImage(ImageBuf<u8>);

impl AsRef<ImageBuf<u8>> for BitonalImage {

    fn as_ref(&self) -> &ImageBuf<u8> {
        &self.0
    }

}

impl AsMut<ImageBuf<u8>> for BitonalImage {

    fn as_mut(&mut self) -> &mut ImageBuf<u8> {
        &mut self.0
    }

}

impl BitonalImage {

    pub fn new(height : usize, width : usize) -> Self {
        Self(ImageBuf::<u8>::new_constant(height, width, 0))
    }

    pub fn update_from_binary<S>(&mut self, img : &Image<u8, S>)
    where
        S : Storage<u8>
    {
        self.update_from_gray(img, 0)
    }

    // Values smaller than below_thr are encoded as bit=1
    pub fn update_from_gray_inverted<S>(&mut self, img : &Image<u8, S>, below_thr : u8)
    where
        S : Storage<u8>
    {
        self.update_from_gray(img, below_thr);

        // This not is bitwise, so updating from below threshold is the
        // same as updating from above threshold then negating the
        // bitonal image.
        self.0.not_mut();
    }

    // Values greater than above_thr are encoded as bit=1
    pub fn update_from_gray<S>(&mut self, img : &Image<u8, S>, above_thr : u8)
    where
        S : Storage<u8>
    {
        assert!(img.width() % 8 == 0);
        assert!(self.0.height() == img.height());
        assert!(self.0.width() == img.width() / 8);

        let dst_bit_offset = 0;
        unsafe {
            let ans = crate::foreign::ipp::ippi::ippiGrayToBin_8u1u_C1R(
                img.as_ptr(),
                img.byte_stride() as i32,
                self.0.as_mut_ptr(),
                self.0.byte_stride() as i32,
                dst_bit_offset,
                crate::foreign::ipp::ippi::IppiSize::from(img.size()),
                above_thr
            );
            assert!(ans == 0);
        }
    }

    pub fn to_binary<T>(&self, dst : &mut Image<u8, T>)
    where
        T : StorageMut<u8>
    {
        assert!(dst.width() % 8 == 0);
        assert!(self.0.height() == dst.height());
        assert!(self.0.width() == dst.width() / 8);
        let src_bit_offset = 0;
        let low_val : u8 = 0;
        let high_val : u8 = 255;
        unsafe {
            let ans = crate::foreign::ipp::ippi::ippiBinToGray_1u8u_C1R(
                self.0.as_ptr(),
                self.0.byte_stride() as i32,
                src_bit_offset,
                dst.as_mut_ptr(),
                dst.byte_stride() as i32,
                dst.size().into(),
                low_val,
                high_val
            );
            assert!(ans == 0);
        }
    }

}


