use crate::image::*;

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

// IppStatus ippiDCTFwd_<mod> (const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
// const IppiDCTFwdSpec_32f* pDCTSpec, Ipp8u* pBuffer );

// IppStatus ippiDCTInv_<mod> (const Ipp32f* pSrc, int srcStep, Ipp32f* pDst, int dstStep,
// const IppiDCTInvSpec_32f* pDCTSpec, Ipp8u* pBuffer );

// IppStatus ippiDCT8x8Inv_2x2_16s_C1(const Ipp16s* pSrc, Ipp16s* pDst );

