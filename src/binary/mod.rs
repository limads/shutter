// Threshold should come here, as well as all bitwise-logical operations.

use crate::image::*;

pub trait BinaryOp<'a> {

    fn and(&'a self, other : &Window<'a, u8>, dst : &'a mut WindowMut<'a, u8>);

    fn or(&'a self, other : &Window<'a, u8>, dst : &'a mut WindowMut<'a, u8>);

    fn xor(&'a self, other : &Window<'a, u8>, dst : &'a mut WindowMut<'a, u8>);

}

impl<'a> BinaryOp<'a> for Window<'a, u8> {

    fn and(&'a self, other : &Window<'a, u8>, dst : &'a mut WindowMut<'a, u8>) {

        #[cfg(feature="ipp")]
        unsafe {
            let (src_stride, src_roi) = crate::image::ipputils::step_and_size_for_window(self);
            let (other_stride, other_roi) = crate::image::ipputils::step_and_size_for_window(other);
            let (dst_stride, dst_roi) = crate::image::ipputils::step_and_size_for_window_mut(&dst);
            let ans = crate::foreign::ipp::ippi::ippiAnd_8u_C1R(
                self.as_ptr(),
                src_stride,
                other.as_ptr(),
                other_stride,
                dst.as_mut_ptr(),
                dst_stride,
                dst_roi
            );
            assert!(ans == 0);
            return;
        }
        unimplemented!();
    }

    fn or(&'a self, other : &Window<'a, u8>, dst : &'a mut WindowMut<'a, u8>) {

        #[cfg(feature="ipp")]
        unsafe {
            let (src_stride, src_roi) = crate::image::ipputils::step_and_size_for_window(self);
            let (other_stride, other_roi) = crate::image::ipputils::step_and_size_for_window(other);
            let (dst_stride, dst_roi) = crate::image::ipputils::step_and_size_for_window_mut(&dst);
            let ans = crate::foreign::ipp::ippi::ippiOr_8u_C1R(
                self.as_ptr(),
                src_stride,
                other.as_ptr(),
                other_stride,
                dst.as_mut_ptr(),
                dst_stride,
                dst_roi
            );
            assert!(ans == 0);
            return;
        }
        unimplemented!();
    }

    fn xor(&'a self, other : &Window<'a, u8>, dst : &'a mut WindowMut<'a, u8>) {

        #[cfg(feature="ipp")]
        unsafe {
            let (src_stride, src_roi) = crate::image::ipputils::step_and_size_for_window(self);
            let (other_stride, other_roi) = crate::image::ipputils::step_and_size_for_window(other);
            let (dst_stride, dst_roi) = crate::image::ipputils::step_and_size_for_window_mut(&dst);
            let ans = crate::foreign::ipp::ippi::ippiXor_8u_C1R(
                self.as_ptr(),
                src_stride,
                other.as_ptr(),
                other_stride,
                dst.as_mut_ptr(),
                dst_stride,
                dst_roi
            );
            assert!(ans == 0);
            return;
        }
        unimplemented!();
    }

}

/*IppStatus ippiAnd_<mod> ( const Ipp<datatype>* pSrc1 , int src1Step , const Ipp<datatype>*
pSrc2 , int src2Step , Ipp<datatype>* pDst , int dstStep , IppiSize roiSize );

IppStatus ippiAndC_<mod> ( const Ipp<datatype>* pSrc , int srcStep , Ipp<datatype> value ,
Ipp<datatype>* pDst , int dstStep , IppiSize roiSize );

IppStatus ippiOr_<mod> ( const Ipp<datatype>* pSrc1 , int src1Step , const Ipp<datatype>*
pSrc2 , int src2Step , Ipp<datatype>* pDst , int dstStep , IppiSize roiSize );

IppStatus ippiXor_<mod> ( const Ipp<datatype>* pSrc1 , int src1Step , const Ipp<datatype>*
pSrc2 , int src2Step , Ipp<datatype>* pDst , int dstStep , IppiSize roiSize );

IppStatus ippiOrC_<mod> ( const Ipp<datatype>* pSrc , int srcStep , Ipp<datatype> value ,
Ipp<datatype>* pDst , int dstStep , IppiSize roiSize );

IppStatus ippiXor_<mod> ( const Ipp<datatype>* pSrc1 , int src1Step , const Ipp<datatype>*
pSrc2 , int src2Step , Ipp<datatype>* pDst , int dstStep , IppiSize roiSize );

IppStatus ippiXorC_<mod> ( const Ipp<datatype>* pSrc , int srcStep , const Ipp<datatype>
value[3] , Ipp<datatype>* pDst , int dstStep , IppiSize roiSize );

IppStatus ippiNot_<mod> ( const Ipp8u* pSrc , int srcStep , Ipp8u* pDst , int dstStep ,
IppiSize roiSize );*/



