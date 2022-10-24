use crate::image::*;
// use nalgebra::*;
use ripple::signal::sampling::{self};
use std::ops::{Index, IndexMut, Mul, Add, AddAssign, MulAssign, SubAssign, Range, Div, Rem};
use simba::scalar::SubsetOf;
use std::fmt;
use std::fmt::Debug;
use simba::simd::{AutoSimd};
use std::convert::TryFrom;
// use crate::feature::patch::{self, Patch};
use itertools::Itertools;
// use crate::feature::patch::ColorMode;
use num_traits::Zero;
use serde::{Serialize, Deserialize, de::DeserializeOwned};
use std::any::Any;
use tempfile;
use std::fs;
use std::io::Write;
use num_traits::cast::FromPrimitive;
use num_traits::cast::AsPrimitive;
use num_traits::float::Float;
use std::borrow::{Borrow, BorrowMut};
use std::mem;
// use crate::raster::*;
// use crate::draw::*;
// use crate::sparse::RunLength;
use num_traits::bounds::Bounded;
use std::clone::Clone;
use std::any::{TypeId};
use std::marker::PhantomData;

impl<P> Clone for ImageBuf<P> 
where
    P : Pixel
{
    fn clone(&self) -> Self {
        Self { 
            offset : self.offset,
            sz : self.sz,
            width : self.width,
            slice : self.slice.clone(),
            _px : PhantomData
        }
    }

    fn clone_from(&mut self, source: &Self) { 
        self.offset = source.offset;
        self.sz = source.sz;
        self.width = source.width;
        self.slice = source.slice.clone();
    }
    
}

impl<'a, P> Clone for ImageRef<'a, P> 
where
    P : Pixel
{
    fn clone(&self) -> Self {
        Self { 
            offset : self.offset,
            sz : self.sz,
            width : self.width,
            slice : self.slice.clone(),
            _px : PhantomData
        }
    }

    fn clone_from(&mut self, source: &Self) { 
        self.offset = source.offset;
        self.sz = source.sz;
        self.width = source.width;
        self.slice = source.slice.clone();
    }
    
}

impl<'a, P, S> Image<P, S>
where
    P : Pixel,
    S : Storage<P>,
    //Box<[P]> : StorageMut<P>,
    //&'a mut [P] : StorageMut<P>,
    //&'a [P] : Storage<P>
{

    pub fn clone_owned(&self) -> ImageBuf<P>
    where
        P : Copy + Default + Zero
    {
        let mut buf = Vec::new();
        self.rows().for_each(|row| buf.extend(row.iter().cloned()) );
        ImageBuf::from_vec(buf, self.sz.1)
    }
    
}

impl<P, S> Image<P, S> 
where
    P : Pixel,
    S : StorageMut<P>
{

    // Copies elements into self from other, assuming same dims.
    pub fn copy_from<T>(&mut self, other : &Image<P, T>) 
    where
        T : Storage<P>
    {
        assert!(self.same_size(other));
        // TODO Assert pixel depth is same.

        #[cfg(feature="ipp")]
        unsafe {
            let (src_step, src_sz) = crate::image::ipputils::step_and_size_for_image(other);
            let (dst_step, dst_sz) = crate::image::ipputils::step_and_size_for_image(self);
            if self.pixel_is::<u8>() {
                let ans = crate::foreign::ipp::ippi::ippiCopy_8u_C1R(
                    mem::transmute(other.as_ptr()),
                    src_step,
                    mem::transmute(self.as_mut_ptr()),
                    dst_step,
                    src_sz
                );
                assert!(ans == 0);
                return;
            }

            if self.pixel_is::<i16>() {
                let ans = crate::foreign::ipp::ippi::ippiCopy_16s_C1R(
                    mem::transmute(other.as_ptr()),
                    src_step,
                    mem::transmute(self.as_mut_ptr()),
                    dst_step,
                    src_sz
                );
                assert!(ans == 0);
                return;
            }

            if self.pixel_is::<i32>() {
                let ans = crate::foreign::ipp::ippi::ippiCopy_32s_C1R(
                    mem::transmute(other.as_ptr()),
                    src_step,
                    mem::transmute(self.as_mut_ptr()),
                    dst_step,
                    src_sz
                );
                assert!(ans == 0);
                return;
            }

            if self.pixel_is::<f32>() {
                let ans = crate::foreign::ipp::ippi::ippiCopy_32f_C1R(
                    mem::transmute(other.as_ptr()),
                    src_step,
                    mem::transmute(self.as_mut_ptr()),
                    dst_step,
                    src_sz
                );
                assert!(ans == 0);
                return;
            }
        }

        //self.rows_mut().zip(other.rows())
        //    .for_each(|(this, other)| this.copy_from_slice(other) );
        unimplemented!()
    }
}

impl<'a, P, S> Image<P, S>
where
    P : Pixel,
    S : StorageMut<P>,
    // &'a mut [P]: StorageMut<P>,
    // &'a [P]: Storage<P>,
{

    /*/// Copies the content of the slice, assuming raster order.
    pub fn copy_from_slice<'b>(&'a mut self, other : &'b [N]) {
        assert!(self.area() == other.len());
        self.copy_from(&Window::from_slice(other, self.sz.1));
    }*/

     pub fn copy_from_slice(&mut self, slice : &[P]) {
        // assert!(self.slice.len() == slice.len());
        // self.slice.as_mut().copy_from_slice(&slice);
        // Use rows_mut here.
        unimplemented!()
    }
    
    // Copies elements into the center of self self from other, assuming dim(self) > dim(other), padding the
    // border elements with a constant user-supplied value.
    pub fn padded_copy_from<T>(&mut self, other : &Image<P, T>, pad : P) 
    where
        T : Storage<P>
    {
        verify_border_dims(other, self);
        let (bh, bw) = border_dims(other, self);

        #[cfg(feature="ipp")]
        unsafe {
            let (src_step, src_sz) = crate::image::ipputils::step_and_size_for_image(other);
            let (dst_step, dst_sz) = crate::image::ipputils::step_and_size_for_image(&self);
            if self.pixel_is::<u8>() {
                //let ans = self.apply_to_sub((bh, bw), *other.size(), |mut dst| {
                let (dst_step, dst_sz) = crate::image::ipputils::step_and_size_for_image(&self);
                let ans = crate::foreign::ipp::ippi::ippiCopyConstBorder_8u_C1R(
                    mem::transmute(other.as_ptr()),
                    src_step,
                    src_sz,
                    mem::transmute(self.as_mut_ptr()),
                    dst_step,
                    dst_sz,
                    bh as i32,
                    bw as i32,
                    *std::mem::transmute::<_, &u8>(&pad)
                );
                // });
                assert!(ans == 0);
                return;
            }
        }

        /*self.apply_to_sub_mut((0, 0), (100, 100), |mut win| {
            self.shape();
        });*/

        unimplemented!()
    }

    // Copies elements into the center of self self from other, assuming dim(self) > dim(other), wrapping the
    // border elements (i.e. border elements are copied by reflecting elements from the opposite border).
    pub fn wrapped_copy_from<T>(&mut self, other : &Image<P, T>) 
    where
        T : Storage<P>
    {
        verify_border_dims(other, self);

        let (bh, bw) = border_dims(other, self);

        #[cfg(feature="ipp")]
        unsafe {
            let (src_step, src_sz) = crate::image::ipputils::step_and_size_for_image(other);
            let (dst_step, dst_sz) = crate::image::ipputils::step_and_size_for_image(self);
            if self.pixel_is::<f32>() {
                //let ans = self.apply_to_sub((bh, bw), *other.size(), |mut dst| {

                let ans = crate::foreign::ipp::ippi::ippiCopyWrapBorder_32f_C1R(
                    mem::transmute(other.as_ptr()),
                    src_step,
                    src_sz,
                    mem::transmute(self.as_mut_ptr()),
                    dst_step,
                    dst_sz,
                    bh as i32,
                    bw as i32,
                );
                // });
                assert!(ans == 0);
                return;
            }
        }

        unimplemented!()
    }

    // Copies elements into the center of self self from other, 
    // assuming dim(self) > dim(other), padding the
    // border elements with the last element of other at each border pixel.
    pub fn replicated_copy_from<T>(&mut self, other : &Image<P, T>) 
    where
        T : Storage<P>
    {
        verify_border_dims(other, self);

        let (bh, bw) = border_dims(other, self);

         #[cfg(feature="ipp")]
        unsafe {
            let (src_step, src_sz) = crate::image::ipputils::step_and_size_for_image(other);
            let (dst_step, dst_sz) = crate::image::ipputils::step_and_size_for_image(self);
            let ans = if self.pixel_is::<u8>() {
                crate::foreign::ipp::ippi::ippiCopyReplicateBorder_8u_C1R(
                    mem::transmute(other.as_ptr()),
                    src_step,
                    src_sz,
                    mem::transmute(self.as_mut_ptr()),
                    dst_step,
                    dst_sz,
                    bh as i32,
                    bw as i32,
                )
            } else if self.pixel_is::<f32>() {
                crate::foreign::ipp::ippi::ippiCopyReplicateBorder_32f_C1R(
                    mem::transmute(other.as_ptr()),
                    src_step,
                    src_sz,
                    mem::transmute(self.as_mut_ptr()),
                    dst_step,
                    dst_sz,
                    bh as i32,
                    bw as i32,
                )
            } else {
                panic!("Invalid type");
            };
            assert!(ans == 0);
            return;
        }

        unimplemented!()
    }

    // Copies entries from true_other into self only where pixels at the mask are nonzero,
    // and from false_other into self when mask is zero.
    // self[px] = mask[px] == 0 ? false_other else true_other[px].
    pub fn alternated_copy_from<T, U>(
        &mut self, 
        mask : &Window<'_, u8>, 
        true_alt : &Image<P, T>, 
        false_alt : &Image<P, U>
    ) where
        T : Storage<P>,
        U : Storage<P>
    {
        self.copy_from(false_alt);
        self.conditional_copy_from(mask, true_alt);
    }

    // Copies entries from other into self only where pixels at the mask are nonzero.
    // self[px] = mask[px] == 0 ? nothing else other[px]
    pub fn conditional_copy_from<R, T>(
        &mut self, 
        mask : &Image<u8, R>, 
        other : &Image<P, T>
    ) where
        R : Storage<u8>,
        T : Storage<P>,
    {
        // TODO Assert pixel depth is same.
        assert!(self.shape() == other.shape(), "Windows differ in shape");
        assert!(self.shape() == mask.shape(), "Windows differ in shape");

        #[cfg(feature="ipp")]
        unsafe {
            let (src_step, src_sz) = crate::image::ipputils::step_and_size_for_image(other);
            let (dst_step, dst_sz) = crate::image::ipputils::step_and_size_for_image(self);
            let (mask_step, mask_sz) = crate::image::ipputils::step_and_size_for_image(mask);

            if self.pixel_is::<u8>() {
                let ans = crate::foreign::ipp::ippi::ippiCopy_8u_C1MR(
                    mem::transmute(other.as_ptr()),
                    src_step,
                    mem::transmute(self.as_mut_ptr()),
                    dst_step,
                    src_sz,
                    mask.as_ptr(),
                    mask_step
                );
                assert!(ans == 0);
                return;
            }

            if self.pixel_is::<i16>() {
                let ans = crate::foreign::ipp::ippi::ippiCopy_16s_C1MR(
                    mem::transmute(other.as_ptr()),
                    src_step,
                    mem::transmute(self.as_mut_ptr()),
                    dst_step,
                    src_sz,
                    mask.as_ptr(),
                    mask_step
                );
                assert!(ans == 0);
                return;
            }

            if self.pixel_is::<i32>() {
                let ans = crate::foreign::ipp::ippi::ippiCopy_32s_C1MR(
                    mem::transmute(other.as_ptr()),
                    src_step,
                    mem::transmute(self.as_mut_ptr()),
                    dst_step,
                    src_sz,
                    mask.as_ptr(),
                    mask_step
                );
                assert!(ans == 0);
                return;
            }

            if self.pixel_is::<f32>() {
                let ans = crate::foreign::ipp::ippi::ippiCopy_32f_C1MR(
                    mem::transmute(other.as_ptr()),
                    src_step,
                    mem::transmute(self.as_mut_ptr()),
                    dst_step,
                    src_sz,
                    mask.as_ptr(),
                    mask_step
                );
                assert!(ans == 0);
                return;
            }
        }

        /*for ((mut d, s), m) in self.pixels_mut(1).zip(other.pixels(1)).zip(mask.pixels(1)) {
            if *m != 0 {
                *d = *s
            }
        }*/
        unimplemented!()
    }

}

impl<'a, P, S> Image<P, S>
where
    P : Pixel,
    S : Storage<P>,
    // Box<[P]> : StorageMut<P>,
    // &'a mut [P]: StorageMut<P>,
    // &'a [P]: Storage<P>
{

    // Copies the content of this image into the center of a larger buffer with size new_sz,
    // setting the remaining border entries to the pad value.
    pub fn padded_copy(&self, new_sz : (usize, usize), pad : P) -> ImageBuf<P> {
        assert!(new_sz.0 > self.height() && new_sz.1 > self.width());
        let mut new = unsafe { ImageBuf::new_empty(new_sz.0, new_sz.1) };
        new.padded_copy_from(self, pad);
        new
    }

    /// Copies the content of this image into the center of a larger buffer with size
    /// new_sz, mirroring the left margin into the right margin (and vice-versa), and mirroring
    /// the top-margin into the bottom-margin (and vice-versa).
    pub fn wrapped_copy(&self, new_sz : (usize, usize)) -> ImageBuf<P> {
        assert!(new_sz.0 > self.height() && new_sz.1 > self.width());
        let mut new = unsafe { ImageBuf::new_empty(new_sz.0, new_sz.1) };
        new.wrapped_copy_from(self);
        new
    }

    /// Copies the content of this image into the center of a larger buffer with
    /// size new_sz, replicating the first value of each row to the left border;
    /// the last value of each row to the right border, the first value of each column
    /// to the top border, and the last value of each column to the bottom border.
    pub fn replicated_copy(&self, new_sz : (usize, usize)) -> ImageBuf<P> {
        assert!(new_sz.0 > self.height() && new_sz.1 > self.width());
        let mut new = unsafe { ImageBuf::new_empty(new_sz.0, new_sz.1) };
        new.replicated_copy_from(self);
        new
    }

    // If true, copy from self. If not, copy from other.
    pub fn alternated_copy(
        &self, 
        mask : &Window<'_, u8>, 
        alternative : &Window<'a, P>
    ) -> ImageBuf<P> {
        let mut new = unsafe { ImageBuf::<P>::new_empty_like(self) };
        new.alternated_copy_from(mask, self, alternative);
        new
    }

    // If true at nonzero mask, copy from self. No nothing otherwise.
    pub fn conditional_copy(&self, mask : &Window<'_, u8>) -> ImageBuf<P> {
        let mut new = unsafe { ImageBuf::<P>::new_empty_like(self) };
        new.conditional_copy_from(mask, self);
        new
    }
}

