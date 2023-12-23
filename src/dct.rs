use crate::image::*;
use std::mem::transmute;

impl<S> Image<i16, S>
where
    S : Storage<i16>
{

    pub fn dct_fwd_local_to<T>(&self, dst : &mut Image<i16, T>)
    where
        T : StorageMut<i16>
    {
        assert!(self.height() == dst.height() && self.width() == dst.width());
        assert!(self.height() % 8 == 0 && self.width() % 8 == 0);
        unsafe {
            for (src_block, mut dst_block) in self.windows((8,8)).zip(dst.windows_mut(((8,8)))) {
                let ans = crate::foreign::ipp::ippi::ippiDCT8x8Fwd_16s_C1(
                    src_block.as_ptr(),
                    dst_block.as_mut_ptr()
                );
                assert!(ans == 0);
            }
        }
    }

    pub fn dct_inv_local_to<T>(&self, dst : &mut Image<i16, T>)
    where
        T : StorageMut<i16>
    {
        assert!(self.height() == dst.height() && self.width() == dst.width());
        assert!(self.height() % 8 == 0 && self.width() % 8 == 0);
        unsafe {
            for (src_block, mut dst_block) in self.windows((8,8)).zip(dst.windows_mut((8,8))) {
                let ans = crate::foreign::ipp::ippi::ippiDCT8x8Inv_16s_C1(
                    src_block.as_ptr(),
                    dst_block.as_mut_ptr()
                );
                assert!(ans == 0);
            }
        }
    }

}

#[cfg(feature="ipp")]
#[derive(Clone)]
pub struct DCT {
    is_fwd : bool,
    sz : (usize, usize),
    buf : Vec<u8>,
    spec : Vec<u8>,
    init : Vec<u8>
}

impl DCT {

    pub fn new(sz : (usize, usize), is_fwd : bool) -> Self {
        let (mut sz_spec, mut sz_init, mut sz_buf) = (0i32, 0i32, 0i32);
        unsafe {
            let ans = if is_fwd {
                crate::foreign::ipp::ippi::ippiDCTFwdGetSize_32f(
                    sz.into(),
                    &mut sz_spec,
                    &mut sz_init,
                    &mut sz_buf
                )
            } else {
                crate::foreign::ipp::ippi::ippiDCTInvGetSize_32f (
                    sz.into(),
                    &mut sz_spec,
                    &mut sz_init,
                    &mut sz_buf
                )
            };
            assert!(ans == 0);

            // This buffer is used by the init function
            let mut init : Vec<_> = (0..(sz_init as usize)).map(|_| 0u8 ).collect();

            // The spec and buffers used by the fwd/inv functions.
            let mut spec : Vec<_> = (0..(sz_spec as usize)).map(|_| 0u8 ).collect();
            let mut buf : Vec<_> = (0..(sz_buf as usize)).map(|_| 0u8 ).collect();

            let ans = if is_fwd {
                crate::foreign::ipp::ippi::ippiDCTFwdInit_32f(
                    transmute(spec.as_mut_ptr()),
                    sz.into(),
                    init.as_mut_ptr()
                )
            } else {
                crate::foreign::ipp::ippi::ippiDCTInvInit_32f(
                    transmute(spec.as_mut_ptr()),
                    sz.into(),
                    init.as_mut_ptr()
                )
            };
            assert!(ans == 0);
            Self { sz, is_fwd, spec, buf, init }
        }
    }

    // pub fn apply_scaled<S, T>(&mut self, src : &Image<u8, S>, dst : &mut Image<u8, T>) -> f32 {
    // }

    pub fn apply<S, T>(&mut self, src : &Image<f32, S>, dst : &mut Image<f32, T>)
    where
        S : Storage<f32>,
        T : StorageMut<f32>
    {
        assert!(src.size() == dst.size());
        assert!(src.size() == self.sz);
        unsafe {
            let ans = if self.is_fwd {
                crate::foreign::ipp::ippi::ippiDCTFwd_32f_C1R(
                    src.as_ptr(),
                    src.byte_stride() as i32,
                    dst.as_mut_ptr(),
                    dst.byte_stride() as i32,
                    transmute(self.spec.as_mut_ptr()),
                    self.buf.as_mut_ptr()
                )
            } else {
                crate::foreign::ipp::ippi::ippiDCTInv_32f_C1R(
                    src.as_ptr(),
                    src.byte_stride() as i32,
                    dst.as_mut_ptr(),
                    dst.byte_stride() as i32,
                    transmute(self.spec.as_mut_ptr()),
                    self.buf.as_mut_ptr()
                )
            };
            assert!(ans == 0);
        }
    }

}



