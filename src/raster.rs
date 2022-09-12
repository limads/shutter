use crate::image::*;
use nalgebra::Scalar;
use std::fmt::Debug;

/* Encapsulates indexing and iteration over image windows. A "raster" is simply an owned or borrowed slice of a primitive
pixel type (such as u8, i16, f32) that can be interpreted as a set of contiguous raster lines (pixel rows).
Each row is not necessarily contiguous over memory with the next row;
but all pixels within the same row are contiguous over memory. RasterRef and RasterMut are
subtraits that apply to borrowed an mutably borrowed slices respectively, and allows different
patterns of iterations and indexing that preserve aliasing rules for the mutability of the underlying slice. */
pub trait Raster {

    type Slice;

    fn create(offset : (usize, usize), win_sz : (usize, usize), orig_sz : (usize, usize), win : Self::Slice) -> Self;

    fn offset(&self) -> &(usize, usize);

    fn size(&self) -> &(usize, usize);

    fn width(&self) -> usize;

    fn height(&self) -> usize;

    fn original_size(&self) -> (usize, usize);

    fn original_width(&self) -> usize;

    fn original_height(&self) -> usize;

    // This takes the implementor because for mutable windows,
    // the only way to safely give the slice is to let go of the
    // current object. This is unsafe because the returned slice
    // will be aliased with the original content.
    unsafe fn original_slice(&mut self) -> Self::Slice;

    // Returns the subsection of the original slice that is
    // effectively used to represent this window. slice.as_ptr()
    // will be the same as self.ptr(). This is unsafe because the returned slice
    // will be aliased with the original content.
    unsafe fn essential_slice(&mut self) -> Self::Slice;

}

impl<'a, T> Raster for Window<'a, T>
where
    T : Scalar + Copy
{

    type Slice = &'a [T];

    fn create(offset : (usize, usize), win_sz : (usize, usize), orig_sz : (usize, usize), win : Self::Slice) -> Self {
        
    }

    fn offset(&self) -> &(usize, usize) {
        &self.offset
    }

    fn size(&self) -> &(usize, usize) {
        &self.win_sz
    }

    fn width(&self) -> usize {
        self.win_sz.1
    }

    fn height(&self) -> usize {
        self.win_sz.0
    }

    fn original_width(&self) -> usize {
        self.width
    }

    fn original_height(&self) -> usize {
        self.win.len() / self.width
    }

    fn original_size(&self) -> (usize, usize) {
        (self.original_height(), self.width)
    }

    unsafe fn original_slice(&mut self) -> Self::Slice {
        self.win
    }

    unsafe fn essential_slice(&mut self) -> Self::Slice {
        std::slice::from_raw_parts(self.as_ptr(), self.original_width() * self.height())
    }

}

impl<'a, T> Raster for WindowMut<'a, T>
where
    T : Scalar + Copy //+ Serialize + DeserializeOwned + Any + Zero + From<u8>
{

    type Slice = &'a mut [T];

    fn create(offset : (usize, usize), win_sz : (usize, usize), orig_sz : (usize, usize), win : Self::Slice) -> Self {
        WindowMut { offset, win_sz, width : orig_sz.1, win }
    }

    fn offset(&self) -> &(usize, usize) {
        &self.offset
    }

    fn size(&self) -> &(usize, usize) {
        &self.win_sz
    }

    fn width(&self) -> usize {
        self.win_sz.1
    }

    fn height(&self) -> usize {
        self.win_sz.0
    }

    unsafe fn original_slice(&mut self) -> Self::Slice {
        std::slice::from_raw_parts_mut(self.win.as_mut_ptr(), self.win.len())
    }

    unsafe fn essential_slice(&mut self) -> Self::Slice {
        std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.original_width() * self.height())
    }

}

pub trait RasterRef {

}

impl<'a, N> RasterRef for Window<'a, N>
where
    N : Scalar + Copy + Debug
{

}

pub trait RasterMut {

}

impl <'a, N> RasterMut for WindowMut<'a, N>
where
    N : Scalar + Copy + Debug
{

}

