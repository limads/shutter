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

    pub fn update<S>(&mut self, img : &Image<u8, S>)
    where
        S : Storage<u8>
    {
        assert!(self.0.height() == img.height());
        assert!(self.0.width() == img.width() / 8);

        // Values greater than that are encoded as bit=1
        let threshold : u8 = 0;
        let dst_bit_offset = 0;
        unsafe {
            let ans = crate::foreign::ipp::ippi::ippiGrayToBin_8u1u_C1R(
                img.as_ptr(),
                img.byte_stride() as i32,
                self.0.as_mut_ptr(),
                self.0.byte_stride() as i32,
                dst_bit_offset,
                crate::foreign::ipp::ippi::IppiSize::from(img.size()),
                threshold
            );
            assert!(ans == 0);
        }
    }

}


