use nalgebra::Scalar;
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
use nalgebra::Vector2;
use nalgebra::Complex;
use std::borrow::ToOwned;

#[cfg(feature="opencv")]
use nalgebra::{Matrix3, Vector3};

// use crate::io;

#[cfg(feature="ipp")]
pub mod ipputils;

#[cfg(feature="opencv")]
pub mod cvutils;

#[cfg(feature="opencv")]
use opencv::core;

#[cfg(feature="opencv")]
use opencv::core::Mat;

pub mod index;

mod iter;

pub use iter::*;

mod copy;

pub use copy::*;

mod convert;

pub use convert::*;

/// Represents an integer offset from the first top-left pixel, using the convention 
/// (row offset, column offset)
pub type Coord = (usize, usize);

/// Represents an integer image size, using the convention
/// (image height, image width)
pub type Size = (usize, usize);

/// Represents a rectangle via its size and offset.
pub type Rect = (Coord, Size);

// Represents a circle via its center and radius.
// pub type Circle = (Coord, usize);

/* Perhaps use those type wrappers over slice<T> and box<[T]> */
pub struct Borrowed<'a, T>(&'a [T]);

pub struct BorrowedMut<'a, T>(&'a mut [T]);

pub struct Boxed<T>(Box<[T]>);

/* Owned image buffer, generic over the pixel type */
/// Owned digital image occupying a dynamically-allocated buffer. If you know your image content
/// at compile time, consider using Window::from_constant, which saves up the allocation.
/// Images are backed by Box<[N]>, because once a buffer is allocated, it cannot grow or
/// shrink like Vec<N>.
/// Fundamentally, an image differs from a matrix because
/// it is oriented row-wise in memory, while a matrix is oriented column-wise. Also, images are
/// bounded at a low and high end, because they are the product of a saturated digital quantization
/// process. But indexing, following OpenCV convention, happens from the top-left point, following
/// the matrix convention.
pub type ImageBuf<P> = Image<P, Box<[P]>>;

/* Immutably borrowed image, generic over the pixel type */
/// Borrowed subset of an image. Referencing the whole source slice (instead of just its
/// portion of interest) might be useful to represent overlfowing operations (e.g. draw)
/// as long as the operation does not violate bounds of the original image. We just have
/// to be careful to not expose the rest of the image in the API.
pub type ImageRef<'a, P> = Image<P, &'a [P]>;

/* Mutably borrowed image, generic over the pixel type. */
pub type ImageMut<'a, P> = Image<P, &'a mut [P]>;

pub type Window<'a, N> = ImageRef<'a, N>;

pub type WindowMut<'a, N> = ImageMut<'a, N>;

// Alternatives:
// ImageCut (mutable)
// ImageView (immutable)
// ImageBorrow
// ImageFrame
// Frame
// ImagePiece
// ImageSelection

pub trait Pixel 
where
    Self : Clone + Copy + Scalar + Debug + Zero + Any + Bounded + Default + 'static
    // Bounded is implemented for all except Complex<f32>
{

    fn depth() -> Depth;
    
}

impl Pixel for u8 { 
    
    fn depth() -> Depth {
        Depth::U8
    }
    
}

impl Pixel for u16 { 

    fn depth() -> Depth {
        Depth::U16
    }
    
}

impl Pixel for i8 { 

    fn depth() -> Depth {
        Depth::I8
    }
    
}

impl Pixel for i16 { 
    
    fn depth() -> Depth {
        Depth::I16
    }
    
}

impl Pixel for i32 { 

    fn depth() -> Depth {
        Depth::I32
    }
    
}

impl Pixel for f32 { 

    fn depth() -> Depth {
        Depth::F32
    }
    
}

impl Pixel for u32 {

    fn depth() -> Depth {
        Depth::U32
    }

}

/*impl Pixel for Complex<f32> {

    fn depth() -> Depth {
        unimplemented!()
    }
    
}*/

// impl FloatPixel for Complex<f32> { }

/*pub trait BinaryPixel {

    type False;
    
}

impl BinaryPixel for u8 { 

    type False = 0;
     
}

impl BinaryPixel for bool {

    type False = false;
    
}*/

/* Represents an image buffer agnostic to ownership status of the
pixel buffer. The pixel buffer should always implement Pixels. 
If the buffer is mutable (&mut [P]) or AsMut<[P]>, it will also
implement PixelsMut. */
pub struct Image<P, S> {
    offset : Coord,
    sz : Size,
    width : usize,
    slice : S,
    _px : PhantomData<P>
}

impl<P, S> fmt::Debug for Image<P, S>
where
    S : Debug
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        write!(f, "Image {{ {:?} {:?} }}", self.offset, self.sz)
    }
}

/*impl<N> fmt::Display for ImageBuf<N>
where
    N : Scalar + Clone + Copy + Serialize + DeserializeOwned + Any + Default + num_traits::Zero
{

    fn fmt(&self, f : &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Image (Height = {}; Width = {})", self.height(), self.width())
    }

}*/

impl<P> Default for ImageBuf<P> 
where
    P : Pixel
{

    fn default() -> Self {
        Image { 
            offset : (0, 0), 
            sz : (0, 0), 
            width : 0, 
            slice : Vec::new().into_boxed_slice(),
            _px : PhantomData 
        }
    }
    
}

/* The most generic implementations, not requiring any trait bounds over the storage
and/or pixels. */
impl<P, S> Image<P, S> {

    pub fn from_components(
        offset : (usize, usize), 
        sz : (usize, usize), 
        orig_sz : (usize, usize), 
        slice : S
    )-> Self {
        Image { offset, sz, width : orig_sz.1, slice, _px : PhantomData }
    }
    
    pub fn width(&self) -> usize {
        self.sz.1
    }
    
    pub fn height(&self) -> usize {
        self.sz.0
    }
    
    pub fn offset(&self) -> Coord {
        self.offset
    }
    
    pub fn size(&self) -> Size {
        self.sz
    }
    
    // If self has a size that differ from other only by a positive integer scale factor,
    // (i.e. self is bigger than or equal to other) return it (over both dimensions).
    pub fn size_scale<Q, T>(&self, other : &Image<Q, T>) -> Option<(usize, usize)> {
        if self.height() >= other.height() && self.width() >= other.width() {
            if self.height() % other.height() == 0 && self.width() % other.width() == 0 {
                return Some((self.height() / other.height(), self.width() / other.width()));
            }
        }
        None
    }
    
    // Verifies if those two images have sizes that are equal
    pub fn same_size<Q, T>(&self, other : &Image<Q, T>) -> bool {
        self.size() == other.size()
    }
    
    pub fn shape(&self) -> Size {
        self.sz
    }
    
    pub fn original_width(&self) -> usize {
        self.width
    }
    
    // Returns a set of linear indices to the underlying slice that can
    // be used to iterate over this window.
    // pub fn linear_indices(&self, spacing : usize) -> Vec<usize> {
    //    unimplemented!()
    // }

    pub fn subsampled_indices(&self, spacing : usize) -> Vec<(usize, usize)> {
        let mut ixs = Vec::new();
        for i in 0..(self.height() / spacing) {
            for j in 0..(self.width() / spacing) {
                ixs.push((i, j));
            }
        }
        ixs
    }
    
}

#[derive(Debug, Clone, Copy)]
pub enum Flip {
    Horizontal,
    Vertical,
    Both
}

impl<P, S> Image<P, S>
where
    S : Storage<P>,
    P : Pixel
{
    
    pub fn is_empty(&self) -> bool {
        self.slice.as_ref().is_empty()
    }

    pub fn flip_to<T>(&self, flip : Flip, dst : &mut Image<P, T>)
    where
        T : StorageMut<P>
    {
        #[cfg(feature="ipp")]
        unsafe {
            if self.pixel_is::<u8>() {
                let flip_op = match flip {
                    Flip::Vertical => crate::foreign::ipp::ippi::IppiAxis_ippAxsVertical,
                    Flip::Horizontal => crate::foreign::ipp::ippi::IppiAxis_ippAxsHorizontal,
                    Flip::Both => crate::foreign::ipp::ippi::IppiAxis_ippAxsBoth
                };
                let ans = crate::foreign::ipp::ippi::ippiMirror_8u_C1R(
                    mem::transmute(self.as_ptr()),
                    self.byte_stride() as i32,
                    mem::transmute(dst.as_mut_ptr()),
                    dst.byte_stride() as i32,
                    self.size().into(),
                    flip_op
                );
                assert!(ans == 0);
                return;
            }
        }
        unimplemented!()
    }

    pub fn get(&self, index : (usize, usize)) -> Option<&P> {
        if index.0 < self.height() && index.1 < self.width() {
            unsafe { Some(self.get_unchecked(index)) }
        } else {
            None
        }
    }
    
    pub unsafe fn get_unchecked(&self, index : (usize, usize)) -> &P {
        // let off_ix = (self.offset.0 + index.0, self.offset.1 + index.1);
        // let (limit_row, limit_col) = (self.offset.0 + self.sz.0, self.offset.1 + self.sz.1);
        self.slice.as_ref().get_unchecked(index::linear_index(index, self.width))
    }

    pub unsafe fn unchecked_linear_index(&self, ix : usize) -> &P {
        self.slice.as_ref().get_unchecked(ix)
    }

    // The linear index directly indexes the underlying slice (but it might)
    // index elements outside the effective region.
    pub fn linear_index(&self, ix : usize) -> &P {
        // assert!(ix < self.width() * self.height());
        // let (row, col) = (ix / self.width(), ix % self.width());
        // unsafe { &*self.as_ptr().offset((self.width * row + col) as isize) as &P }
        &self.slice.as_ref()[ix]
    }

    /// Returns a borrowed view over the whole window. Same as self.as_ref(). But is
    /// convenient to have, since type inference for the AsRef impl might not be triggered
    /// or you need an owned version of the window
    pub fn full_window(&self) -> Window<P> {
        self.window((0, 0), self.size()).unwrap()
    }

    pub fn sub_window(
        &self, 
        offset : (usize, usize), 
        sz : (usize, usize)
    ) -> Option<Window<P>> {
        self.window(offset, sz)
    }
    
    pub unsafe fn window_unchecked(
        &self,
        offset : (usize, usize),
        dims : (usize, usize)
    ) -> Window<P> {
        let new_offset = (self.offset.0 + offset.0, self.offset.1 + offset.1);
        Image {
            slice : index::sub_slice(self.slice.as_ref(), offset, dims, self.width),
            offset : offset,
            width : self.width,
            sz : dims,
            _px : PhantomData
        }
    }

    pub fn region(
        &self,
        region : &crate::shape::Region
    ) -> Option<ImageRef<P>> {
        let (y, x, h, w) = region.to_rect_tuple();
        self.window((y, x), (h, w))
    }

    pub fn area_(
        &self,
        area : &crate::shape::Area
    ) -> Option<ImageRef<P>> {
        let (y, x, h, w) = area.region(self.sz)?.to_rect_tuple();
        self.window((y, x), (h, w))
    }

    /* Gets window centered at the given point. If outside bounds, gets
    the window closest to this center also satisfying the bounds. The only
    way this fails is if the dims are smaller than the parent window dims. */
    pub fn centered_bounded_window(
        &self,
        center : (usize, usize),
        dims : (usize, usize)
    ) -> Option<Window<P>> {
        if dims.0 < self.height() || dims.1 < self.width() {
            return None;
        }
        let half_dims = (dims.0 / 2, dims.1 / 2);
        let tl = (center.0.saturating_sub(half_dims.0), center.1.saturating_sub(half_dims.1));
        let br = (tl.0 + dims.0, tl.1 + dims.1);
        let off_tl = (
            tl.0.saturating_sub(br.0.saturating_sub(self.height())),
            tl.1.saturating_sub(br.1.saturating_sub(self.width()))
        );
        self.window(off_tl, dims)
    }

    pub fn window(
        &self, 
        offset : (usize, usize), 
        dims : (usize, usize)
    ) -> Option<Window<P>> {
        assert_nonzero(dims);
        let new_offset = (self.offset.0 + offset.0, self.offset.1 + offset.1);
        // println!("Window = {:?}", (offset, dims));
        // if new_offset.0 + dims.0 <= self.original_size().0 &&
        //    new_offset.1 + dims.1 <= self.original_size().1
        if offset.0 + dims.0 <= self.height() && offset.1 + dims.1 <= self.width()
        {
            Some(Image {
                slice : index::sub_slice(self.slice.as_ref(), offset, dims, self.width),
                offset : new_offset,
                width : self.width,
                sz : dims,
                _px : PhantomData
            })
        } else {
            None
        }
    }
    
    // For an owned image, will return self.size. For a
    // window, returns the size of the original image from
    // which it is a view.
    pub fn original_size(&self) -> Size {
        (self.original_height(), self.original_width())
    }
    
    pub fn original_height(&self) -> usize {
        self.slice.as_ref().len() / self.width
    }

    // Works both for ImageRef and &ImageBuf
    pub fn slice(&self) -> &[P] {
        self.slice.as_ref()
    }
    
    // pub fn essential_slice(&mut self) -> &[P] {
    //    unsafe { std::slice::from_raw_parts(self.as_ptr(), self.original_width() * self.height()) }
    // }
    
    pub fn depth(&self) -> Depth {
        P::depth()
    }
    
    pub fn pixel_is<T : 'static>(&self) -> bool {
        TypeId::of::<T>() == TypeId::of::<P>()
    }

    /// The cartesian index is defined as (img.height() - pt[1], pt[0]).
    /// It indexes the image by imposing the cartesian analytical plane over it,
    /// an converting it as specified.
    pub fn cartesian_index<T>(&self, pt : Vector2<T>) -> &P
    where
        usize : From<T>,
        T : Copy,
        P : Mul<Output=P> + MulAssign + Copy
    {
        let ix = (self.height() - usize::from(pt[1]), usize::from(pt[0]));
        &self[ix]
    }
    
    pub fn as_ptr(&self) -> *const P {
        // self.slice.as_ptr()
        &self[(0usize,0usize)] as *const _
    }
    
}

impl<P, S> Image<P, S>
where
    S : UnsafeStorage<P>,
    P : Pixel
{

    pub unsafe fn raw_ptr(&self) -> *const P {
        // self.slice.as_pointer().as_ptr()
        unimplemented!()
    }

}

impl<P, S> Image<P, S>
where
    S : UnsafeStorageMut<P>,
    P : Pixel
{

    pub unsafe fn raw_mut_ptr(&mut self) -> *mut P {
        // self.slice.as_mut_pointer().as_mut_ptr()
        unimplemented!()
    }

}

/*impl<N, S> Borrow<ImageRef<N>> for Image<S>
where
    S : Storage<N>
{

    fn borrow(&self) -> &Borrowed {
    
    }
    
}*/

use either::Either;

pub enum MutableImage<'a, N> {
    Borrowed(ImageMut<'a, N>),
    Owned(ImageBuf<N>)
}

impl<'a, N> MutableImage<'a, N> 
where
    N : Pixel
{

    pub fn from_like<S, T>(mut dst : Option<&'a mut Image<N, T>>, s : &Image<N, S>) -> Self 
    where
        S : Storage<N>,
        T : StorageMut<N> + 'a 
    {
        match dst {
            Some(mut d) => MutableImage::from(Either::Left(d.full_window_mut())),
            None => MutableImage::from(Either::Right(s.shape()))
        }
    }
    
    pub fn from(mut img_or_size : Either<ImageMut<'a, N>, Size>) -> Self
    {
        match img_or_size {
            Either::Left(mut img) => MutableImage::Borrowed(img),
            Either::Right(shape) => MutableImage::Owned(unsafe { Image::new_empty(shape.0, shape.1) })
        }
    }
    
    pub fn full_window_mut(&'a mut self) -> ImageMut<'a, N> {
        match self {
            MutableImage::Borrowed(m) => m.full_window_mut(),
            MutableImage::Owned(buf) => buf.full_window_mut()
        }
    }
    
    pub fn window_mut(&'a mut self, off : Coord, sz : Size) -> Option<ImageMut<'a, N>> {
        match self {
            MutableImage::Borrowed(m) => m.window_mut(off, sz),
            MutableImage::Owned(buf) => buf.window_mut(off, sz)
        }
    }
    

}

pub enum ImmutableImage<'a, N> {
    Borrowed(ImageRef<'a, N>),
    Owned(ImageBuf<N>)
}

/*impl<N, S> ToOwned for Image<N, S>
where
    S : Storage<N>
{

    type Owned = ImageBuf<N>;

    fn to_owned(&self) -> ImageBuf<N> {
        self.clone_owned()
    }

    fn clone_into(&self, target: &mut Self::Owned) { 
        unimplemented!()
    }
    
}*/

impl<P, S> Image<P, S> 
where
    S : StorageMut<P>,
    // P : Pixel
{

    // Works both for ImageMut and &mut ImageBuf
    pub fn slice_mut(&mut self) -> &mut [P] {
        self.slice.as_mut()
    }
    
    // pub fn essential_slice_mut(&mut self) -> &mut [P] {
    //    unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.original_width() * self.height()) }
    // }
    
    pub unsafe fn unchecked_linear_index_mut(&mut self, ix : usize) -> &mut P {
        self.slice.as_mut().get_unchecked_mut(ix)
    }
    
    pub fn linear_index_mut(&mut self, ix : usize) -> &mut P {
        // assert!(ix < self.width() * self.height());
        // let (row, col) = (ix / self.width(), ix % self.width());
        // unsafe { &mut *self.as_mut_ptr().offset((self.width * row + col) as isize) as &mut P }
        unsafe { &mut *self.as_mut_ptr().offset(ix as isize) as &mut P }
    }

    pub fn as_mut_ptr(&mut self) -> *mut P {
        // self.slice.as_ptr()
        &mut self[(0usize,0usize)] as *mut _
    }

}

/*// Type alias variant
pub type ImageBuf<T> = base::Image<Box<[T]>>;
pub type Image<'a, T> = base::Image<&'a [T]>;
pub type ImageMut<'a, T> = base::Image<&'a [T]>;

// But not for ImageBuf/ImageMut
impl Copy for Image<'a, T>;

// Default generic argument variant.
pub struct Image<'a, S = &'a [T]> { }*/

/*impl<'a> Window<'a, f32> {

    pub fn show(&self) {
        use crate::convert::*;
        use crate::point::*;
        let dst = into_normalized_owned(self);
        dst.show();
    }
    
}*/

/*impl ImageBuf<f32> {

    pub fn show(&self) {
        self.full_window().show();
    }
    
}*/

mod storage {

    use super::*;
    
    pub trait Storage<P>
    where
        Self : AsRef<[P]>
    {

    }
    
    impl<'a, T> Storage<T> for &'a [T] where T : Pixel { }
    
    impl<'a, T> Storage<T> for &'a mut [T] where T : Pixel { }
    
    impl<T> Storage<T> for Box<[T]> where T : Pixel { }

    pub trait UnsafeStorage<T> {

        fn as_pointer(&self) -> *const [T];

    }

    impl<T> UnsafeStorage<T> for *const [T] where T : Pixel {

        fn as_pointer(&self) -> *const [T] {
            *self
        }

    }

}

mod storage_mut {

    use super::*;
    
    pub trait StorageMut<P>
    where
        // P : Pixel,
        Self : Storage<P> + AsMut<[P]>,
    {

    }
    
    impl<'a, T> StorageMut<T> for &'a mut [T] where T : Pixel,
    &'a mut [T] : Storage<T> { }
    
    impl<'a, T> StorageMut<T> for Box<[T]> where T : Pixel,
    Box<[T]> : Storage<T> { }

    pub trait UnsafeStorageMut<T> {

        fn as_mut_pointer(&mut self) -> *mut [T];

    }

    impl<T> UnsafeStorageMut<T> for *mut [T] where T : Pixel {

        fn as_mut_pointer(&mut self) -> *mut [T] {
            *self
        }

    }

}

pub trait SignedPixel where Self : Pixel + num_traits::sign::Signed { }

impl SignedPixel for i16 { }

impl SignedPixel for i32 { }
    
pub trait UnsignedPixel where Self : Pixel + num_traits::sign::Unsigned { }

impl UnsignedPixel for u8 { }

impl UnsignedPixel for u16 { }
        
pub trait FloatPixel where Self : Pixel + num_traits::Float { }

impl FloatPixel for f32 { }

pub enum Depth {
    U8,
    I8,
    U16,
    I16,
    I32,
    U32,
    F32
}

pub use storage::*;

pub use storage_mut::*;

// TODO make indexing operations checked at debug builds. They are segfaulting if the
// user passes an overflowing index, since the impl is using get_unchecked regardless for now.

// TODO image.equalize() (contrast equalization) image.strech() (constrast stretching)

// Perhaps rename to GrayImage? (but float image is also gray).
// pub type ByteImage = Image<u8>;
// pub type FloatImage = Image<f32>;

fn verify_size_and_alignment<A, B>() {
    assert!(mem::size_of::<A>() == mem::size_of::<B>());
    assert!(mem::align_of::<A>() == mem::align_of::<B>());
}

fn join_windows<N>(s : &[Window<N>], horizontal : bool) -> Option<ImageBuf<N>>
where
    N : Pixel,
    // for<'a> &'a mut [N] : StorageMut<N>,
    for<'a> &'a [N] : Storage<N>,
    for<'a> &'a mut [N] : StorageMut<N>,
    // Box<[N]> : Storage<N>,
    Box<[N]> : StorageMut<N>
{
    if s.len() == 0 {
        return None;
    }
    let shape = s[0].shape();
    if !s.iter().skip(1).all(|w| w.shape() == shape ) {
        return None;
    }
    let width = if horizontal { shape.1 * s.len() } else { shape.1 };
    let height = if horizontal { shape.0 } else { shape.0 * s.len() };
    let n = width * height;
    let mut buf = Vec::with_capacity(n);
    unsafe { buf.set_len(n); }
    let mut img = ImageBuf {
        slice : buf.into_boxed_slice(),
        offset : (0, 0),
        width,
        sz : (height, width),
        _px : PhantomData
    };
    for ix in 0..s.len() {
        if horizontal {
            img.window_mut((0, ix*shape.1), shape).unwrap().copy_from(&s[ix]);
        } else {
            img.window_mut((ix*shape.0, 0), shape).unwrap().copy_from(&s[ix]);
        }
    }
    Some(img)
}

impl<N> ImageBuf<N> 
where
    N : Pixel 
{
    
}

impl<N> ImageBuf<N>
where
    N : Pixel,
    Box<[N]> : Storage<N>
{

    pub fn to_boxed_slice(self) -> Box<[N]> {
        self.slice
    }

    pub fn to_vec(self) -> Vec<N> {
        self.slice.into()
    }

}

impl<N> ImageBuf<N>
where
    N : Pixel,
    Box<[N]> : StorageMut<N>,
    //&'a [N] : Storage<N>,
    //&'a mut [N] : StorageMut<N>,
    //for<'b> &'b mut [N] : StorageMut<N>,
    //for<'b> &'b [N] : Storage<N>
{

    pub fn as_slice(&self) -> &[N] {
        &self.slice[..]
    }

    pub fn as_mut_slice(&mut self) -> &mut [N] {
        &mut self.slice[..]
    }

    pub fn leak(self) {
        Box::leak(self.slice);
    }

    /* Mirroring ops are useful to represent image correlations via convolution:
    corr(a,b) = conv(a,mirror(b)) = conv(mirror(a), b) */
    pub fn mirror_vertically(&self) -> ImageBuf<N> {
        unimplemented!()
    }

    pub fn mirror_horizontally(&self) -> ImageBuf<N> {
        unimplemented!()
    }

    pub fn mirror(&self) -> ImageBuf<N> {
        unimplemented!()
    }

    pub fn transpose(&self) -> ImageBuf<N> {
        /*let mut dst = self.clone();
        dst.sz = (self.sz.1, self.sz.0);
        dst.width = self.sz.0;
        dst.transpose_from(&self.full_window());
        dst*/
        unimplemented!()
    }

    pub fn transpose_from<'a>(&mut self, src : &'a Window<'a, N>) 
    where
        Box<[N]> : Storage<N>
    {

        assert!(src.width() == self.height() && src.height() == self.width());

        /*#[cfg(feature="ipp")]
        unsafe {
            let (dst_step, dst_roi) = crate::image::ipputils::step_and_size_for_image(
                &self.full_window_mut()
            );
            let (src_step, src_roi) = crate::image::ipputils::step_and_size_for_image(src);
            if self.pixel_is::<u8>() {
                let ans = crate::foreign::ipp::ippi::ippiTranspose_8u_C1R(
                    mem::transmute(src.as_ptr()),
                    src_step,
                    mem::transmute(self.full_window_mut().as_mut_ptr()),
                    dst_step,
                    src_roi
                );
                assert!(ans == 0);
                return;
            }
        }*/

        /*use nalgebra::*;
        DMatrixSlice::from_slice(
            &src.buf[..], src.height(), src.width()
        )
            .transpose_to(
                &mut DMatrixSliceMut::from_slice(&mut self.slice[..], src.width(), src.height())
            );*/
            
        unimplemented!()
    }

    pub fn transpose_mut(&mut self) {
        // let original = self.clone();
        // self.transpose_from(&original.full_window());
        unimplemented!()
    }

    // Splits this image over S owning, memory-contiguous blocks. This does not
    // copy the underlying data and is useful to split the work across multiple threads
    // using ownership-based mechanisms (such as channels).
    pub fn split<const S : usize>(mut self) -> [ImageBuf<N>; S] {
        unimplemented!()
    }

    // Join S blocks of same size into an image again. This is useful to recover an image
    // after it has been split across different blocks with ImageBuf::split to process an image
    // in multithreaded algorithms. Passing images
    // whose buffers are not memory-contigous returns None, since the underlying vector
    // is built assuming zero-copy by just reclaiming ownership of the blocks, and this
    // cannot be done with images with non-contiguous memory blocks.
    pub fn concatenate(s : &[Window<'_, N>]) -> Option<ImageBuf<N>> 
    //where
    //for<'b> &'b [N] : Storage<N>,
    //for<'b> &'b mut [N] : StorageMut<N>
    {
        // join_windows(s, true)
        unimplemented!()
    }

    pub fn stack(s : &[Window<'_, N>]) -> Option<ImageBuf<N>> 
    // where
    // for<'b> &'b [N] : Storage<N>,
    // for<'b> &'b mut [N] : StorageMut<N>,
    {
        // join_windows(s, false)
        unimplemented!()
    }

    /*pub fn linear_index(&self, ix : usize) -> &N {
        &self.slice[ix]
    }

    pub fn linear_index_mut(&mut self, ix : usize) -> &mut N {
        &mut self.slice[ix]
    }*/

    pub fn from_iter(iter : impl Iterator<Item=N>, width : usize) -> Self {
        let slice : Vec<N> = iter.collect();
        Self::from_vec(slice, width)
    }

    /*pub fn subsample_from(&mut self, content : &[N], ncols : usize, sample_n : usize) {
        assert!(ncols < content.len(), "ncols smaller than content length");
        let nrows = content.len() / ncols;
        let sparse_ncols = ncols / sample_n;
        let sparse_nrows = nrows / sample_n;
        self.width = sparse_ncols;
        if self.slice.len() != sparse_nrows * sparse_ncols {
            self.slice.clear();
            self.slice.extend((0..(sparse_nrows*sparse_ncols)).map(|_| N::zero() ));
        }
        for r in 0..sparse_nrows {
            for c in 0..sparse_ncols {
                self.slice[r*sparse_ncols + c] = content[r*sample_n*ncols + c*sample_n];
            }
        }
    }*/

    pub fn new_random(height : usize, width : usize) -> Self
    where
        rand::distributions::Standard : rand::distributions::Distribution<N>
    {
        let mut img = ImageBuf::<N>::new_constant(height, width, N::zero());
        img.fill_random();
        img
    }

    pub fn new_from_slice(source : &[N], width : usize) -> Self {
        let mut buf = Vec::with_capacity(source.len());
        let height = buf.len() / width;
        unsafe { buf.set_len(source.len()); }
        buf.copy_from_slice(&source);
        Self{ 
            slice : buf.into_boxed_slice(), 
            width, 
            offset : (0, 0), 
            sz : (height, width),
            _px : PhantomData
        }
    }

    pub fn from_vec(slice : Vec<N>, width : usize) -> Self {
        //if buf.len() as f64 % ncols as f64 != 0.0 {
        //    panic!("Invalid image lenght");
        //}
        assert!(slice.len() % width == 0);
        let height = slice.len() / width;
        Self { 
            slice : slice.into_boxed_slice(), 
            width, 
            offset : (0, 0), 
            sz : (height, width), 
            _px : PhantomData 
        }
    }

    pub fn new_unallocated() -> Self {
        Self { 
            slice : Vec::new().into_boxed_slice(), 
            width : 0, 
            offset : (0, 0), 
            sz : (0, 0),
            _px : PhantomData
        }
    }
    
    pub fn from_rows<const R : usize, const C : usize>(pxs : [[N; C]; R]) -> Self {
        let mut slice : Vec<N> = Vec::with_capacity(R*C);
        for r in pxs {
            slice.extend(r.into_iter());
        }
        Self { 
            slice : slice.into_boxed_slice(), 
            width : C, 
            offset : (0, 0), 
            sz : (R, C),
            _px : PhantomData
        }
    }

    pub fn from_cols<const R : usize, const C : usize>(pxs : [[N; C]; R]) -> Self {
        let mut img = ImageBuf::from_rows(pxs);
        img.transpose_mut();
        img
    }

    // Creates a new image with the same dimensions as other, but with values set from the given scalar.
    pub fn new_constant_like<'b, Q, T>(other : &Image<Q, T>, value : N) -> Self
    where
        Box<[N]> : StorageMut<N>
    {
        Self::new_constant(other.height(), other.width(), value)
    }

    // Creates a new image with the same dimensions as other, with uninitialized values.
    // Make sure you write to the allocated buffer before reading from it, otherwise
    // the access is UB.
    pub unsafe fn new_empty_like<'b, T, Q>(other : &Image<Q, T>) -> Self
    //where
    //    Window<'b, M> : Raster
    {
        Self::new_empty(other.height(), other.width())
    }

    pub fn new_constant(height : usize, width : usize, value : N) -> Self
    where
        Box<[N]> : StorageMut<N>
    {
        /*let mut buf = Vec::with_capacity(height * width);
        buf.extend((0..(height*width)).map(|_| value ));
        Self{ slice : buf.into_boxed_slice(), width, offset : (0, 0), sz : (height, width) }*/
        let mut img = unsafe { ImageBuf::new_empty(height, width) };
        img.fill(value);
        img
    }
    
    /// Initializes an image with allocated, but undefined content. Reading the contents
    /// of this image before writing to all entries is UB, which is why this funciton is marked
    /// as unsafe. Make sure your program copies content from another image, or set
    /// all its values with fill() before you attempt to read from it.
    pub unsafe fn new_empty(nrows : usize, ncols : usize) -> Self {
        assert_nonzero((nrows, ncols));
        let mut buf = Vec::<N>::with_capacity(nrows * ncols);
        buf.set_len(nrows * ncols);
        Self { 
            slice : buf.into_boxed_slice(), 
            width : ncols, 
            offset : (0, 0), 
            sz : (nrows, ncols),
            _px : PhantomData
        }
    }

    /*pub fn downsample(&mut self, src : &Window<N>) {
        assert!(src.is_full());
        let src_ncols = src.width;
        let dst_ncols = self.width;
        
        // TODO resize yields a nullpointer when allocating its data.
        #[cfg(feature="ipp")]
        unsafe {
            let dst_nrows = self.slice.len() / self.width;
            ipputils::resize(src.win, &mut self.slice);
        }

        panic!("ImageBuf::downsample requires that crate is compiled with opencv or ipp feature");

        // TODO use resize::resize for native Rust solution
    }*/
    
    // pub fn copy_from(&mut self, other : &ImageBuf<N>) {

        // IppStatus ippiCopy_8uC1R(const Ipp<datatype>* pSrc, int srcStep, Ipp<datatype>* pDst,
        // int dstStep, IppiSize roiSize);
        // self.slice.copy_from_slice(&other.buf[..]);
    //    self.full_window_mut().copy_from(&other.full_window());
    // }
    
    /*pub fn convert_from(&mut self, other : &Image<N>) {
        self.slice.as_slice_mut().copy_from(other.buf.as_slice());
    }*/
    
    /*pub fn windows_mut(&mut self, sz : (usize, usize)) -> impl Iterator<Item=WindowMut<'_, S::Pixel>>
    where
        S::Pixel : Mul<Output=S::Pixel> + MulAssign
    {
        assert_nonzero(sz);
        self.full_window_mut().windows_mut(sz)
    }*/

    // pub fn center_sub_window(&self) -> Window<'_, N> {
    // }

    // Returns an iterator over the clockwise sub-window characterizing a pattern.
    // pub fn clockwise_sub_windows(&self) -> impl Iterator<Item=Window

    /*pub fn windows(&self, sz : (usize, usize)) -> impl Iterator<Item=Window<'_, S::Pixel>>
    where
        S::Pixel : Mul<Output=S::Pixel> + MulAssign
    {
        assert_nonzero(sz);
        self.full_window().windows(sz)
    }*/
    
    /*pub fn iter(&self) -> impl Iterator<Item=&N> {
        let shape = self.shape();
        iterate_row_wise(&self.slice[..], (0, 0), shape, shape)
    }*/
    
    //pub fn len(&self) -> usize {
    //    self.slice.len()
    //}

    // pub fn windows(&self) -> impl Iterator<Item=Window<'_, N>> {
    //    unimplemented!()
    // }
    
}

impl ImageBuf<u8> {

    pub fn show(&self) {

        use std::process::Command;
        use tempfile;

        let mut tf = tempfile::NamedTempFile::new().unwrap();
        let png = crate::io::encode(self.clone()).unwrap();
        tf.write_all(&png).unwrap();
        let path = tf.path();
        let new_path = format!("{}.png", path.to_str().unwrap());
        fs::rename(path, new_path.clone()).unwrap();
        Command::new("eog")
            .args(&[&new_path])
            .output()
            .unwrap();
    }

    /*// Creates an image from a binary pattern. Bits start from the top-left
    // and progress clockwise wrt the center pixel.
    pub fn new_from_pattern(side : usize, center : bool, pattern : u8) -> Image<u8> {
        let mut pattern = crate::texture::pattern::to_binary(pattern);
        pattern.iter_mut().for_each(|p| *p *= 255 );
        let mut img = unsafe { ImageBuf::new_empty(side, side) };
        let mut wins = img.windows_mut((side / 3, side / 3));
        wins.next().unwrap().fill(pattern[0]);
        wins.next().unwrap().fill(pattern[1]);
        wins.next().unwrap().fill(pattern[2]);
        wins.next().unwrap().fill(pattern[7]);
        wins.next().unwrap().fill(if center { 255 } else { 0 });
        wins.next().unwrap().fill(pattern[3]);
        wins.next().unwrap().fill(pattern[6]);
        wins.next().unwrap().fill(pattern[5]);
        wins.next().unwrap().fill(pattern[4]);
        std::mem::drop(wins);
        img
    }*/

    /* TODO create new_from_pattern where Pattern is an Enum with checkerboard
    as one of its patterns */
    pub fn new_checkerboard(sz : usize, sq_sz : usize) -> Self {
        assert!(sz % sq_sz == 0);
        let mut img = Self::new_constant(sz, sz, 255);
        let n_sq = sz / sq_sz;
        for row in 0..n_sq {
            for col in 0..n_sq {
                if (row % 2 == 0 && col % 2 == 0) || (row % 2 != 0 && col % 2 != 0) {
                    img.window_mut((row*sq_sz, col*sq_sz), (sq_sz, sq_sz)).unwrap().fill(0);
                }
            }
        }
        img
    }

    /*pub fn draw(&mut self, mark : Mark) {
        self.full_window_mut().draw(mark);
    }*/

    /// Builds a binary image from a set of points.
    pub fn binary_from_points(nrow : usize, ncol : usize, pts : impl Iterator<Item=(usize, usize)>) -> Self {
        let mut img = ImageBuf::<u8>::new_constant(nrow, ncol, 0);
        for pt in pts {
            img[pt] = 255;
        }
        img
    }
}

impl<'a, N> WindowMut<'a, N>
where
    N : Scalar + Copy + Any + 'static
{

    // Joins two mutable windows, as long as their slices are contiguous in memory.
    pub fn join(a : WindowMut<'a, N>, b : WindowMut<'a, N>) -> Option<WindowMut<'a, N>> {
        unimplemented!()
    }

    /// Splits this mutable window horizontally
    pub fn split_at(mut self, row : usize) -> (WindowMut<'a, N>, WindowMut<'a, N>) {
        assert!(self.slice.as_ref().len() % (self.width * row) == 0);
        let offset = self.offset;
        let sz = self.sz;
        let width = self.width;
        let (win1, win2) = self.slice.as_mut().split_at_mut(self.width * row);
        let w1 = WindowMut {
            slice : win1,
            width,
            offset,
            sz : (row, sz.1),
            _px : PhantomData
        };
        let w2 = WindowMut {
            slice : win2,
            width,
            offset : (offset.0 + row, offset.1),
            sz : (sz.0 - row, sz.1),
            _px : PhantomData
        };
        // assert!(w1.win.len() == w1.sz.0 * w1.sz.1);
        // assert!(w2.win.len() == w2.sz.0 * w2.sz.1);
        (w1, w2)
    }

}

/*impl<'a, N> Window<'a, N>
where
    N : Scalar + Copy
{

    pub fn rect(&self) -> (usize, usize, usize, usize) {
        let off = self.offset();
        let sz = self.shape();
        (off.0, off.1, sz.0, sz.1)
    }

    pub fn center(&self) -> (usize, usize) {
        let rect = self.rect();
        (rect.0 + rect.2 / 2, rect.1 + rect.3 / 2)
    }

    /// Returns either the given sub window, or trim it to the window borders and return a smaller but also valid window
    pub fn largest_valid_sub_window(&'a self, offset : (usize, usize), dims : (usize, usize)) -> Option<Window<'a, N>> {
        let diff_h = offset.0 as i64 + dims.0 as i64 - self.height() as i64;
        let diff_w = offset.1 as i64 + dims.1 as i64 - self.width() as i64;
        let valid_dims = (dims.0 - diff_h.max(0) as usize, dims.1 - diff_w.max(0) as usize);
        self.sub_window(offset, valid_dims)
    }

}*/

fn shrink_to_divisor(mut n : usize, by : usize) -> Option<usize> {
    if n > by {
        while n % by != 0 {
            n -= 1;
        }
        Some(n)
    } else {
        None
    }
}

impl<P, S> Image<P, S>
where
    S : Storage<P>,
    P : Pixel
{

    pub unsafe fn row_unchecked(&self, ix : usize) -> &[P] {
        // let tl = self.offset.0 * self.width + self.offset.1;
        let start = ix*self.width;
        &self.slice.as_ref().get_unchecked(start..(start+self.sz.1))
    }

    pub fn row(&self, ix : usize) -> Option<&[P]> {
        if ix >= self.sz.0 {
            return None;
        }
        // let stride = self.original_size().1;
        // let tl = self.offset.0 * stride + self.offset.1;
        let start = ix*self.width;
        Some(&self.slice.as_ref()[start..(start+self.sz.1)])
    }

    // Returns image corners with the given dimensions.
    pub fn corners<'a>(
        &'a self, 
        height : usize, 
        width : usize
    ) -> Option<[Window<'a, P>; 4]> {
        let right = self.width() - width;
        let bottom = self.height() - height;
        let sz = (height, width);
        let tl = self.window((0, 0), sz)?;
        let tr = self.window((0, right), sz)?;
        let bl = self.window((bottom, 0), sz)?;
        let br = self.window((bottom, right), sz)?;
        Some([tl, tr, bl, br])
    }

    // Returns image sides (without corners or same dimension).
    pub fn side<'a>(&'a self, height : usize, width : usize) -> Option<[Window<'a, P>; 4]> {
        let right = self.width() - width;
        let bottom = self.height() - height;
        let vert_sz = (height, self.width() - 2*width);
        let horiz_sz = (self.height() - 2*height, self.width());
        let top = self.window((0, width), vert_sz)?;
        let right = self.window((height, right), horiz_sz)?;
        let bottom = self.window((bottom, width), vert_sz)?;
        let left = self.window((height, 0), horiz_sz)?;
        Some([top, right, bottom, left])
    }

    pub fn shrink_to_subsample2<'a>(
        &'a self, 
        row_by : usize, 
        col_by : usize
    ) -> Option<Window<'a, P>> {
        let height = shrink_to_divisor(self.height(), row_by)?;
        let width = shrink_to_divisor(self.width(), col_by)?;
        self.window((0, 0), (height, width))
    }

    pub fn shrink_to_subsample<'a>(&'a self, by : usize) -> Option<Window<'a, P>> {
        let height = shrink_to_divisor(self.height(), by)?;
        let width = shrink_to_divisor(self.width(), by)?;
        self.window((0, 0), (height, width))
    }

    pub fn stride(&self) -> usize {
        self.width
    }

    pub fn byte_stride(&self) -> usize {
        std::mem::size_of::<P>() * self.width
    }

    pub fn area(&self) -> usize {
        self.width() * self.height()
    }
    
}

impl<'a, P> ImageRef<'a, P> {

    pub unsafe fn from_ptr(
        ptr : *const P, 
        len : usize, 
        full_ncols : usize, 
        offset : (usize, usize), 
        sz : (usize, usize)
    ) -> Option<Self> {
        let s = std::slice::from_raw_parts(ptr, len);
        Self::sub_from_slice(s, full_ncols, offset, sz)
    }
    
    pub fn sub_from_slice(
        src : &'a [P],
        original_width : usize,
        offset : (usize, usize),
        dims : (usize, usize)
    ) -> Option<Self> {
        let nrows = src.len() / original_width;
        if offset.0 + dims.0 <= nrows && offset.1 + dims.1 <= original_width {
            Some(Self {
                slice : index::sub_slice(src, offset, dims, original_width),
                width : original_width,
                offset,
                sz : dims,
                _px : PhantomData
            })
        } else {
            None
        }
    }

    /// Creates a window that cover the whole slice src, assuming it represents a square image.
    pub fn from_square_slice(src : &'a [P]) -> Option<Self> {
        Self::from_slice(src, (src.len() as f64).sqrt() as usize)
    }
    
    /// Creates a window that cover the whole slice src. The slice is assumed to be in
    /// row-major order, but matrices are assumed to be 
    pub fn from_slice(src : &'a [P], ncols : usize) -> Option<Self> {
        if src.len() % ncols != 0 {
            return None;
        }
        let nrows = src.len() / ncols;
        Some(Self{
            // slice : DMatrixSlice::from_slice_generic(src, Dynamic::new(nrows),
            // Dynamic::new(ncols)),
            slice : src,
            offset : (0, 0),
            width : ncols,
            sz : (nrows, ncols),
            _px : PhantomData
            // transposed : true
        })
    }
}

/*impl<S> Image<S>
where
    S::Pixel : Scalar + Mul<Output=S::Pixel> + MulAssign + Copy + Copy + Any,
    S : Pixels
{
    
    // Returns a thin vertical window over a given col. Prototype for impl Index<(Range<usize>, usize)>
    pub fn sub_col(&self, rows : Range<usize>, col : usize) -> Option<Window<'_, S::Pixel>> {
        let height = rows.end - rows.start;
        self.window((rows.start, col), (height, 1))
    }

    // Returns a thin horizontal window over a given row. Prototype for impl Index<(usize, Range<usize>)>
    pub fn sub_row(&self, row : usize, cols : Range<usize>) -> Option<Window<'_, S::Pixel>> {
        let width = cols.end - cols.start;
        self.window((row, cols.start), (1, width))
    }

    
}*/

/*pub struct PackedIterator<'a, T>
where
    T : Scalar + Debug + Copy
{
    slice : Window<'a, T>,
    ix : usize,
    packed_per_row : usize,
    n_packed : usize
}

impl<'a> Iterator for PackedIterator<'a, u8> {

    type Item = AutoSimd<[u8; 32]>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.ix == self.n_packed  {
            None
        } else {
            let row = self.slice.row(self.ix / self.n_packed).unwrap();
            let packed = &row[self.ix % self.n_packed];
            self.ix += 1;
            Some(AutoSimd::try_from(packed).unwrap())
        }
    }
}*/

impl<'a> Window<'a, u8> {

    #[cfg(feature="opencv")]
    pub fn resize_mut(&self, other : WindowMut<'_, u8>) {
        use opencv::prelude::*;
        use opencv::{core, imgproc};
        let this_shape = self.shape();
        let other_shape = other.shape();
        let src : core::Mat = self.into();
        let mut dst : core::Mat = other.into();
        let dst_sz = dst.size().unwrap();
        imgproc::resize(&src, &mut dst, dst_sz, 0.0, 0.0, imgproc::INTER_NEAREST);
    }

    #[cfg(feature="opencv")]
    pub fn copy_scale_mut_within(
        &self, 
        src : (usize, usize), 
        src_sz : (usize, usize), 
        dst : (usize, usize), 
        dst_sz : (usize, usize)
    ) {

        use opencv::{core, imgproc};
        use opencv::prelude::*; 
        let ratio_row = (dst_sz.0 / src_sz.0) as f64;
        let ratio_col = (dst_sz.1 / src_sz.1) as f64;
        if let Some(src) = self.sub_window(src, src_sz) {
            if let Some(dst) = self.sub_window(dst, dst_sz) {
                let mut src : core::Mat = src.into();
                let mut dst : core::Mat = dst.into();
                imgproc::resize(&src, &mut dst, core::Size2i::default(), ratio_col, ratio_row, imgproc::INTER_LINEAR);
            }
        }
    }

    /// Gets pos or the nearest pixel to it that satisfies a condition.
    pub fn nearest_matching(
        &self,
        seed : (usize, usize),
        px_spacing : usize,
        f : impl Fn(u8)->bool, max_dist : usize
    ) -> Option<(usize, usize)> {
        if f(self[seed]) {
            Some(seed)
        } else {
            self.expanding_pixels(seed, px_spacing)
                .take_while(|(pos, _)| {
                    let row_close_seed = ((pos.0 as i64 - seed.0 as i64).abs() as usize) < max_dist;
                    let col_close_seed = ((pos.1 as i64 - seed.1 as i64).abs() as usize) < max_dist;
                    row_close_seed && col_close_seed
                })
                .find(|(_, px)| f(**px) )
                .map(|(pos, _)| pos )
        }
    }

    pub fn color_at_labels(&'a self, ixs : impl Iterator<Item=(usize, usize)> + 'a) -> impl Iterator<Item=u8> + 'a {
        ixs.map(move |ix| self[ix] )
    }

    pub fn nonzero_pixels(&'a self, px_spacing : usize) -> impl Iterator<Item=&'a u8> +'a + Clone {
        self.pixels(px_spacing).filter(|px| **px != 0 )
    }

    pub fn masked_pixels(&'a self, mask : &'a Window<'_, u8>) -> impl Iterator<Item=&'a u8> +'a + Clone {
        mask.nonzero_labeled_pixels(1).map(move |(r, c, _)| &self[(r, c)] )
    }

    pub fn nonzero_labeled_pixels(&'a self, px_spacing : usize) -> impl Iterator<Item=(usize, usize, u8)> +'a + Clone {
        self.labeled_pixels(px_spacing).filter(|(_, _, px)| *px > 0 )
    }

    /// Iterate over pixels, as long as they are non-zero at the mask window.
    pub fn masked_labeled_pixels(&'a self, mask : &'a Window<'_, u8>) -> impl Iterator<Item=(usize, usize, u8)> +'a + Clone {
        mask.nonzero_labeled_pixels(1).map(move |(r, c, _)| (r, c, self[(r, c)]) )
    }

    /*/// Extract contiguous image regions of homogeneous color.
    pub fn patches(&self, px_spacing : usize) -> Vec<Patch> {
        /*let mut patches = Vec::new();
        color::full_color_patches(&mut patches, self, px_spacing as u16, ColorMode::Exact(0), color::ExpansionMode::Dense);
        patches*/
        unimplemented!()
    }*/

    /*pub fn binary_patches(&self, px_spacing : usize) -> Vec<BinaryPatch> {
        // TODO if we have a binary or a bit image with just a few classes,
        // there is no need for KMeans. Just use the allocations.
        // let label_img = segmentation::segment_colors_to_image(self, px_spacing, n_colors);
        color::binary_patches(self, px_spacing)
    }*/

    /// If higher, returns binary image with all pixels > thresh set to 255 and others set to 0;
    /// If !higher, returns binary image with pixels < thresh set to 255 and others set to 0.
    pub fn threshold_mut(&self, dst : &mut ImageBuf<u8>, thresh : u8, higher : bool) {
        
        assert!(self.shape() == dst.shape());

        #[cfg(feature="opencv")]
        {
            // crate::threshold::threshold_window(self, dst, thresh as f64, 255.0, higher);
        }

        for (src, mut dst) in self.pixels(1).zip(dst.full_window_mut().pixels_mut(1)) {
            if (!higher && *src < thresh) || (higher && *src > thresh) {
                *dst = 255;
            } else {
                *dst = 0;
            }
        }
    }

    /*/// Packed iterator over contigous byte regions of the image within
    /// rows. Image horizontal dimension should be divisible by the
    pub fn iter_packed(&self) -> impl Iterator<Item=AutoSimd<[u8; 32]>> + 'a {

        // Substitute 32 for the size of packed SIMD value.
        let packed_per_row = self.width() / 32;

        let n_packed = packed_per_row * self.height();
        PackedIterator {
            slice : self.clone(),
            ix : 0,
            packed_per_row,
            n_packed
        }
    }*/
}

#[test]
fn window_iter() {
    let img : Window<'_, u8> = Window::from_square_slice(&[
        1, 1, 1, 1, 0, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 0, 1, 1, 1, 1,
    ]).unwrap();

    for win in img.windows((4,4)) {
        println!("Outer: {:?}", win);
        for win_inner in win.windows((2,2)) {
            println!("\tInner : {:?}", win_inner);
        }
        println!("")
    }
}

/*impl<N> Index<(usize, usize)> for Window<'_, N>
where
    N : Scalar + Copy
{

    type Output = N;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let off_ix = (self.offset.0 + index.0, self.offset.1 + index.1);
        let (limit_row, limit_col) = (self.offset.0 + self.sz.0, self.offset.1 + self.sz.1);
        if off_ix.0 < limit_row && off_ix.1 < limit_col {
            unsafe { self.slice.get_unchecked(index::linear_index(off_ix, self.original_size().1)) }
        } else {
            panic!("Invalid window index: {:?}", index);
        }
    }
}

impl<N> Index<(u16, u16)> for Window<'_, N>
where
    N : Scalar + Copy
{

    type Output = N;

    fn index(&self, index: (u16, u16)) -> &Self::Output {
        self.index((index.0 as usize, index.1 as usize))
    }

}*/

/*impl<N> Index<(Range<usize>, usize)> for Window<'_, N>
where
    N : Scalar
{

    type Output = Self;

    fn index(&self, index: (Range<usize>, usize)) -> &Self::Output {

    }

}*/

#[cfg(feature="opencv")]
impl<N> opencv::core::ToInputArray for ImageBuf<N>
where
    N : Pixel + Scalar + Copy + Default + Zero + Any
{

    fn input_array(&self) -> opencv::Result<opencv::core::_InputArray> {
        use opencv::prelude::*;
        let out : opencv::core::Mat = (&*self).into();
        out.input_array()
    }

}

#[cfg(feature="opencv")]
impl<N> opencv::core::ToOutputArray for ImageBuf<N>
where
    N : Pixel + Scalar + Copy + Default + Zero + Any
{

    fn output_array(&mut self) -> opencv::Result<opencv::core::_OutputArray> {
        let mut out : opencv::core::Mat = (&mut *self).into();
        out.output_array()
    }

}

#[cfg(feature="opencv")]
impl<N> opencv::core::ToInputArray for Window<'_, N>
where
    N : Pixel + Scalar + Copy + Default + Zero + Any
{

    fn input_array(&self) -> opencv::Result<opencv::core::_InputArray> {
        let out : opencv::core::Mat = (self.clone()).into();
        out.input_array()
    }

}

#[cfg(feature="opencv")]
impl<N> opencv::core::ToOutputArray for WindowMut<'_, N>
where
    N : Pixel + Scalar + Copy + Default + Zero + Any
{

    fn output_array(&mut self) -> opencv::Result<opencv::core::_OutputArray> {
        let mut out : opencv::core::Mat = (self).into();
        out.output_array()
    }

}

#[cfg(feature="opencv")]
pub fn median_blur(slice : &Window<'_, u8>, output : WindowMut<'_, u8>, kernel : usize) {

    use opencv::{imgproc, core};

    let input : core::Mat = slice.clone().into();
    let mut out : core::Mat = output.into();
    imgproc::median_blur(&input, &mut out, kernel as i32).unwrap();

}

#[cfg(feature="opencv")]
impl<N> From<core::Mat> for ImageBuf<N>
where
    N : Pixel + Scalar + Copy + Default + Zero + Any + opencv::core::DataType
{

    fn from(m : core::Mat) -> ImageBuf<N> {

        use opencv::prelude::*;
        // assert!(m.is_contiguous().unwrap());

        let sz = m.size().unwrap();
        let h = sz.height as usize;
        let w = sz.width as usize;
        let mut img = ImageBuf::<N>::new_constant(h, w, N::zero());
        for i in 0..h {
            for j in 0..w {
                img[(i, j)] = *m.at_2d::<N>(i as i32, j as i32).unwrap();
            }
        }
        img
    }

}

/// TODO mark as unsafe impl
#[cfg(feature="opencv")]
impl<N> Into<core::Mat> for &ImageBuf<N>
where
    N : Pixel + Scalar + Copy + Default + Zero + Any
{

    fn into(self) -> core::Mat {
        let sub_slice = None;
        let stride = self.width;
        unsafe{ cvutils::slice_to_mat(&self.slice[..], stride, sub_slice) }
        // self.full_window().into()
    }
}

#[cfg(feature="opencv")]
impl<N> Into<core::Mat> for &mut ImageBuf<N>
where
    N : Pixel + Scalar + Copy + Default + Zero + Any
{

    fn into(self) -> core::Mat {
        let sub_slice = None;
        let stride = self.width;
        unsafe{ cvutils::slice_to_mat(&self.slice[..], stride, sub_slice) }
        // self.full_window_mut().into()
    }
}

/// TODO mark as unsafe impl
#[cfg(feature="opencv")]
impl<N> Into<core::Mat> for Window<'_, N>
where
    N : Pixel + Scalar + Copy + Default
{

    fn into(self) -> core::Mat {
        let sub_slice = Some((self.offset, self.sz));
        let stride = self.original_size().1;
        unsafe{ cvutils::slice_to_mat(self.slice, stride, sub_slice) }
    }
}

#[cfg(feature="opencv")]
impl<N> Into<core::Mat> for &Window<'_, N>
where
    N : Pixel + Scalar + Copy + Default
{

    fn into(self) -> core::Mat {
        let sub_slice = Some((self.offset, self.sz));
        let stride = self.original_size().1;
        unsafe{ cvutils::slice_to_mat(self.slice, stride, sub_slice) }
    }
}

/// TODO mark as unsafe impl
#[cfg(feature="opencv")]
impl<N> Into<core::Mat> for WindowMut<'_, N>
where
    N : Pixel + Scalar + Copy + Default
{

    fn into(self) -> core::Mat {
        let sub_slice = Some((self.offset, self.sz));
        let stride = self.original_size().1;
        unsafe{ cvutils::slice_to_mat(self.slice, stride, sub_slice) }
    }
}

#[cfg(feature="opencv")]
impl<N> Into<core::Mat> for &mut WindowMut<'_, N>
where
    N : Pixel + Scalar + Copy + Default
{

    fn into(self) -> core::Mat {
        let sub_slice = Some((self.offset, self.sz));
        let stride = self.original_size().1;
        unsafe{ cvutils::slice_to_mat(self.slice, stride, sub_slice) }
    }
}

/*impl TryFrom<rhai::Dynamic> for Mark {

    type Err = ();

    fn try_from(d : rhai::Dynamic) -> Result<Self, ()> {

    }

}*/

/*/// Wraps a mutable slice, offering access to it as if it were a mutable image buffer.
/// The consequence of this approach is that mutable windows sharing the same underlying image
/// rows cannot be represented simultaneously without violating aliasing rules. Mutable windows
/// sharing only columns (but at disjoint row intervals), however, can be safely represented
/// via split_at(.). This requires that sub_window_mut takes windows by-value. Iteration, however,
/// is possible by iter_windows_mut() and apply_to_sub/apply_to_sub_sequence, which work with raw pointers
/// internally (thus any closures or code segments working with the iterable items cannot access the original window for
/// the lifetime of the mutable iterator items in the public API).
#[repr(C)]
#[derive(Debug)]
pub struct WindowMut<'a, N> 
where
    N : Scalar + Copy
{

    // Original image full slice.
    pub(crate) slice : &'a mut [N],
    
    // Original image size.
    pub(crate) width : usize,

    // Window offset, with respect to the top-left point (row, col).
    pub(crate) offset : (usize, usize),
    
    // This window size.
    pub(crate) sz : (usize, usize),

}*/

/*impl<'a, N> WindowMut<'a, N>
where
    N : Scalar + Copy
{

    /*// Since mutable references are not allowed in const fn, this isn't const as the window variant,
    // and therefore cannot be used to initialize a static variable
    pub fn from_static<const S : usize, const W : usize>(array : &'static mut [N; S]) -> WindowMut<'static, N> {

        if S % W != 0 {
            panic!("Invalid image dimensions");
        }

        WindowMut {
            slice : array,
            width : W,
            offset : (0, 0),
            sz : (S / W, W)
        }
    }*/

    
}*/

impl<'a, P> ImageMut<'a, P> 
where
    &'a mut [P]: Storage<P>,
    P : Pixel
{

     /// We might just as well make this take self by value, since the mutable reference to self will be
    /// invalidated by the borrow checker when we have the child.
    // pub fn sub_window_mut(&'a mut self, offset : (usize, usize), dims : (usize, usize)) -> Option<WindowMut<'a, N>> {
    
    // Require pointer to parent slice, NOT top-left offset element.
    pub unsafe fn from_ptr(
        ptr : *mut P, 
        len : usize, 
        full_ncols : usize, 
        offset : (usize, usize), 
        sz : (usize, usize)
    ) -> Option<Self> {
        let s = std::slice::from_raw_parts_mut(ptr, len);
        Self::sub_from_slice(s, full_ncols, offset, sz)
    }

    /// Creates a window that cover the whole slice src, assuming it represents a square image.
    pub fn from_square_slice(src : &'a mut [P]) -> Option<Self> {
        Self::from_slice(src, (src.len() as f64).sqrt() as usize)
    }

    pub fn from_slice(src : &'a mut [P], ncols : usize) -> Option<Self> {
        if src.len() % ncols != 0 {
            return None;
        }
        /*let nrows = src.len() / ncols;
        Self {
            slice : DMatrixSliceMut::from_slice_generic(src, Dynamic::new(nrows),
            Dynamic::new(ncols)),
            offset : (0, 0),
            orig_sz : (nrows, ncols)
        }*/
        let nrows = src.len() / ncols;
        Some(Self{
            // slice : DMatrixSlice::from_slice_generic(src, Dynamic::new(nrows),
            // Dynamic::new(ncols)),
            slice : src,
            offset : (0, 0),
            width : ncols,
            sz : (nrows, ncols),
            _px : PhantomData
        })
    }

    pub fn sub_from_slice(
        src : &'a mut [P],
        original_width : usize,
        offset : (usize, usize),
        dims : (usize, usize)
    ) -> Option<Self> {
        let nrows = src.len() / original_width;
        if offset.0 + dims.0 <= nrows && offset.1 + dims.1 <= original_width {
            Some(Self {
                slice : index::sub_slice_mut(src, offset, dims, original_width),
                offset,
                width : original_width,
                sz : dims,
                _px : PhantomData
            })
        } else {
            None
        }
    }
    
}

impl<P, S> Image<P, S>
where
    P : Pixel,
    S : StorageMut<P>,
{

    /// Returns a mutably borrowed view over the whole window. Note the current mutable reference
    /// to the window is invalidated when this view enters into scope. Same as self.as_mut(). But is
    /// convenient to have, since type inference for the AsMut impl might not be triggered, or you
    /// need an owned version of the window.
    pub fn full_window_mut(&mut self) -> WindowMut<P> {
        let shape = self.size();
        self.window_mut((0, 0), shape).unwrap()
    }
   
   pub fn sub_window_mut(
        &mut self, 
        offset : (usize, usize), 
        sz : (usize, usize)
    ) -> Option<WindowMut<P>> {
        self.window_mut(offset, sz)
    }
    
    // If any of the rects overlap, the returned vector is empty.
    // This cannot possibly return mutable windows because the
    // undarlying buffer mihght overlap.
    pub fn disjoint_windows(
        &self,
        offsets : &[(usize, usize)],
        sizes : &[(usize, usize)]
    ) -> Vec<ImageRef<P>> {
        assert!(offsets.len() == sizes.len());
        for i in 0..offsets.len() {
            for j in (i+1)..offsets.len() {
                let r1 = (offsets[i].0, offsets[i].1, sizes[i].0, sizes[i].1);
                let r2 = (offsets[j].0, offsets[j].1, sizes[j].0, sizes[j].1);
                // if crate::shape::rect_overlaps(&r1, &r2) {
                //    return Vec::new();
                // }
            }
        }
        let mut wins = Vec::new();
        let ptr = self.slice.as_ref().as_ptr();
        let len = self.slice.as_ref().len();
        unsafe {
            for (off, sz) in offsets.iter().zip(sizes.iter()) {
                if let Some(sub) = ImageRef::from_ptr(ptr, len, self.width, *off, *sz) {
                    wins.push(sub);
                } else {
                    return Vec::new();
                }
            }
        }
        wins
    }
    
    pub fn area_mut(
        &mut self,
        area : &crate::shape::Area
    ) -> Option<ImageMut<P>> {
        let (y, x, h, w) = area.region(self.sz)?.to_rect_tuple();
        self.window_mut((y, x), (h, w))
    }

    pub fn region_mut(
        &mut self,
        region : &crate::shape::Region
    ) -> Option<ImageMut<P>> {
        let (y, x, h, w) = region.to_rect_tuple();
        self.window_mut((y, x), (h, w))
    }

    pub fn window_mut(
        &mut self, 
        offset : (usize, usize), 
        dims : (usize, usize)
    ) -> Option<WindowMut<P>> {
        assert_nonzero(dims);
        // println!("Window Mut = {:?}", (offset, dims));
        let new_offset = (self.offset.0 + offset.0, self.offset.1 + offset.1);
        // if new_offset.0 + dims.0 <= self.original_size().0
        //    && new_offset.1 + dims.1 <= self.original_size().1
        if offset.0 + dims.0 <= self.height() && offset.1 + dims.1 <= self.width() {
            Some(Image {
                slice : index::sub_slice_mut(self.slice.as_mut(), offset, dims, self.width),
                offset : new_offset,
                width : self.width,
                sz : dims,
                _px : PhantomData
            })
        } else {
            None
        }
    }
    
    /*pub fn window_mut(
        &mut self, 
        offset : (usize, usize), 
        sz : (usize, usize)
    ) -> Option<WindowMut<P>> {
        assert_nonzero(sz);
        let orig_sz = self.shape();
        if offset.0 + sz.0 <= orig_sz.0 && offset.1 + sz.1 <= orig_sz.1 {
            Some(WindowMut {
                slice : &mut self.slice.as_mut()[..],
                offset,
                width : self.width,
                sz : sz,
                _px : PhantomData
            })
        } else {
            None
        }
    }*/

}

impl<'a, P, S> Image<P, S>
where
    P : Pixel,
    S : StorageMut<P>,
    &'a mut [P] : StorageMut<P>
{

    pub fn fill_random(&mut self)
    where
        rand::distributions::Standard : rand::distributions::Distribution<P>
    {
        self.pixels_mut(1).for_each(|px| *px = rand::random() );
    }

    /// Applies a closure to a subset of the mutable window. This effectively forces
    /// the user to innaugurate a new scope for which the lifetime of the original window
    /// is no longer valid, thus allowing applying an operation to different positions of
    /// a mutable window without violating aliasing rules.
    pub fn apply_to_sub<R>(
        &mut self, 
        offset : (usize, usize), 
        sz : (usize, usize), 
        f : impl Fn(WindowMut<'_, P>)->R
    ) -> R {
        assert_nonzero(sz);
        let ptr = self.slice.as_mut().as_mut_ptr();
        let len = self.slice.as_mut().len();
        let mut sub = unsafe { WindowMut::from_ptr(ptr, len, self.width, offset, sz).unwrap() };
        f(sub)
    }

    /// Calls the same closure with apply_to_sub by specifying a series of offsets and sizes. Elements can overlap
    /// without problem, since the operation is sequential.
    pub fn apply_to_sub_sequence<R>(
        &mut self, 
        offs_szs : &[((usize, usize), (usize, usize))], 
        f : impl Fn(WindowMut<'_, P>)->R + Clone
    ) -> Vec<R> {
        let mut res = Vec::new();
        for (off, sz) in offs_szs.iter().copied() {
            res.push(self.apply_to_sub(off, sz, f.clone()));
        }
        res
    }

    // pub fn area(&self) -> usize {
    //    self.width() * self.height()
    // }

    //pub fn byte_stride(&self) -> usize {
    //    std::mem::size_of::<S::Pixel>() * self.width
    //}

    pub fn clear(&'a mut self)
    where
        P : Zero 
    {
        self.pixels_mut(1).for_each(|px| *px = P::zero() );
    }

    // TODO rewrite this using safe rust.
    /*#[allow(mutable_transmutes)]
    pub fn clone_owned<'a>(&'a self) -> ImageBuf<S::Pixel>
    where
        S::Pixel : Copy + Default + Zero + 'static
    {
        let mut buf = Vec::new();
        let ncols = self.sz.1;
        unsafe {
            for row in std::mem::transmute::<_, &'a mut Self>(self).rows_mut() {
                buf.extend(row.iter().cloned())
            }
        }
        ImageBuf::from_vec(buf, ncols)
    }*/

    pub fn get_mut(&'a mut self, index : (usize, usize)) -> Option<&'a mut P> 
    where
        S : 'a 
    {
        /*if index.0 < self.height() && index.1 < self.width() {
            unsafe { Some(self.get_unchecked_mut(index)) }
        } else {
            None
        }*/
        self.slice.as_mut().get_mut(index::linear_index(index, self.width))
    }

    pub unsafe fn get_unchecked_mut(&'a mut self, index : (usize, usize)) -> &'a mut P 
    where
        S : 'a 
    {
        /*let off_ix = (self.offset.0 + index.0, self.offset.1 + index.1);
        let (limit_row, limit_col) = (self.offset.0 + self.sz.0, self.offset.1 + self.sz.1);
        let lin_ix = index::linear_index(off_ix, self.original_size().1);
        unsafe { 
            self.slice.as_mut().get_unchecked_mut(lin_ix) 
        }*/
        let w = self.width;
        unsafe { self.slice.as_mut().get_unchecked_mut(index::linear_index(index, w)) }
    }

    // pub fn offset_ptr_mut(mut self) -> *mut S::Pixel {
    //    unsafe { self.get_unchecked_mut((0, 0)) as *mut N }
    // }

    // pub fn centered_sub_window_mut(mut self, center : (usize, usize))

}

impl<P, S> Image<P, S> 
where
    S : StorageMut<P>
{

    /*/// Gamma-corrects, i.e. multiplies input by input^(1/gamma) and normalize.
    pub fn gamma_correct_inplace(&mut self, gamma : f32) {

        // Using gamma < 1.0 avoids saturating the image. Perhaps offer a version that
        // does just that, without normalization.
        assert!(gamma <= 1.0);
        for i in 0..self.sz.0 {
            for j in 0..self.sz.1 {
                self[(i,j)] = (self[(i,j)] as f32).powf(1. / gamma).max(0.0).min(255.0) as u8
            }
        }
    }

    /// For any pixel >= color, set it and its neighborhood to erase_color.
    pub fn erase_speckles(&mut self, color : u8, neigh : usize, erase_color: u8) {
        for i in 0..self.sz.0 {
            for j in 0..self.sz.1 {
                if self[(i, j)] >= color {
                    for ni in i.saturating_sub(neigh)..((i + neigh).min(self.sz.0-1)) {
                        for nj in j.saturating_sub(neigh)..((j + neigh).min(self.sz.1-1)) {
                            self[(ni, nj)] = erase_color;
                        }
                    }
                }
            }
        }
    }*/

    //pub fn fill_with_byte(&'a mut self, byte : u8) {
        // self.rows_mut().for_each(|row| std::ptr::write_bytes(&mut row[0] as *mut _, byte, row.len()) );
        /*for ix in 0..self.sz.0 {
            let row = self.row_mut(ix);
            std::ptr::write_bytes(&mut row[0] as *mut _, byte, row.len());
        }*/
    //    unimplemented!()
    // }

    /*pub fn patches(&self, px_spacing : usize) -> Vec<Patch> {
        /*let src_win = unsafe {
            Window {
                offset : (self.offset.0, self.offset.1),
                orig_sz : self.original_size(),
                sz : self.sz,
                slice : std::slice::from_raw_parts(self.slice.as_ptr(), self.slice.len()),
            }
        };
        let mut patches = Vec::new();
        color::full_color_patches(&mut patches, &src_win, px_spacing as u16, ColorMode::Exact(0), color::ExpansionMode::Dense);
        patches*/
        unimplemented!()
    }*/

    /*pub fn draw_patch_contour(&mut self, patch : &Patch, color : u8) {
        let pxs = patch.outer_points::<usize>(crate::feature::patch::ExpansionMode::Contour);
        self.draw(Mark::Shape(pxs, false, color));
    }

    pub fn draw_patch_rect(&mut self, patch : &Patch, color : u8) {
        let rect = patch.outer_rect::<usize>();
        self.draw(Mark::Rect((rect.0, rect.1), (rect.2, rect.3), color));
    }*/

}

/*pub(crate) unsafe fn create_immutable<'a>(win : &'a WindowMut<'a, u8>) -> Window<'a, u8> {
    unsafe {
        Window {
            offset : (win.offset.0, win.offset.1),
            width : win.width,
            sz : win.sz,
            slice : std::slice::from_raw_parts(win.win.as_ptr(), win.win.len()),
        }
    }
}*/

/*impl<N> WindowMut<'_, N>
where
    N : Scalar + Copy
{

    unsafe fn create_immutable_without_lifetime(&self, src : (usize, usize), dim : (usize, usize)) -> Window<'_, N> {
        Window {
            offset : (self.offset.0 + src.0, self.offset.1 + src.1),
            orig_sz : self.original_size(),
            sz : dim,
            slice : std::slice::from_raw_parts(self.slice.as_ptr(), self.slice.len()),
        }
    }

    /// Creates a new mutable window, forgetting the lifetime of the previous window.
    unsafe fn create_mutable_without_lifetime(&mut self, src : (usize, usize), dim : (usize, usize)) -> WindowMut<'_, N> {
        WindowMut {
            offset : (self.offset.0 + src.0, self.offset.1 + src.1),
            orig_sz : self.original_size(),
            sz : dim,
            slice : std::slice::from_raw_parts_mut(self.slice.as_mut_ptr(), self.slice.len()),
        }
    }
}*/

//fn step_row<'a, N>(r : &'a mut [N], spacing : usize) -> std::iter::Flatten<std::iter::StepBy<std::slice::IterMut<'a, N>>>
//where
//    N : 'a,
//    Flatten<std::iter::StepBy<std::slice::IterMut<'a, N>>> : IntoIterator<N>
//{
//    r.iter_mut().step_by(spacing).flatten()
    //r.iter_mut()
//}

// fn step<'a, N>(s : &'a [N], step : usize) -> std::iter::Flatten<std::iter::StepBy<std::slice::Iter<'a, N>>> {
//    s.iter().step_by(step)
// }

/*impl<'a, N> WindowMut<'a, N>
where
    N : Scalar + Copy + Default
{

    /*// Returns image corners with the given dimensions.
    pub fn corners_mut(mut self, height : usize, width : usize) -> Option<[WindowMut<'a, N>; 4]> {
        let right = self.width() - width;
        let bottom = self.height() - height;
        let sz = (height, width);

        self.apply_to_sub_mut((0, 0), (100, 100), |mut win| {
            self.shape();
        });

        unimplemented!();

        // let (top, rem) = self.split(height);
        // let (mid, bottom) = rem.split(rem.height() - height);

        /*let tl = self.sub_window((0, 0), sz)?;
        let tr = self.sub_window((0, right), sz)?;
        let bl = self.sub_window((bottom, 0), sz)?;
        let br = self.sub_window((bottom, right), sz)?;
        Some([tl, tr, bl, br])*/
    }

    // Returns image sides (without corners or same dimension).
    pub fn sides_mut(mut self, height : usize, width : usize) -> Option<[Window<'a, N>; 4]> {
        let right = self.width() - width;
        let bottom = self.height() - height;
        let vert_sz = (height, self.width() - 2*width);
        let horiz_sz = (self.height() - 2*height, self.width());
        let top = self.sub_window((0, width), vert_sz)?;
        let right = self.sub_window((height, right), horiz_sz)?;
        let bottom = self.sub_window((bottom, width), vert_sz)?;
        let left = self.sub_window((height, 0), horiz_sz)?;
        Some([top, right, bottom, left])
    }*/

    //pub fn shape(&self) -> (usize, usize) {
    //    self.sz
    //}

    pub fn orig_sz(&self) -> (usize, usize) {
        self.original_size()
    }

    pub fn full_slice(&'a self) -> &'a [N] {
        &self.slice[..]
    }

    /*pub fn offset(&self) -> (usize, usize) {
        self.offset
    }

    pub fn width(&self) -> usize {
        self.shape().1
    }

    pub fn height(&self) -> usize {
        self.shape().0
    }*/

}*/

pub(crate) fn verify_border_dims<P, Q, S, T>(src : &Image<P, S>, dst : &Image<Q, T>)
// where
//    S : Storage<P>,
//    T : Storage<Q>
{
    assert!(src.height() < dst.height());
    assert!(src.width() < dst.width());
    let diffw = dst.width() - src.width();
    let diffh = dst.height() - src.height();
    assert!(diffw % 2 == 0);
    assert!(diffh % 2 == 0);
    assert!(src.height() + diffh == dst.height());
    assert!(src.width() + diffw == dst.width());
}

pub(crate) fn border_dims<S, T, P, Q>(src : &Image<P, S>, dst : &Image<Q, T>) -> (usize, usize)
// where
//    S : Storage<P>,
//    T : Storage<Q>
{
    let diffw = dst.width() - src.width();
    let diffh = dst.height() - src.height();
    (diffh / 2, diffw / 2)
}

impl<'a, P, S> Image<P, S>
where
    P : Pixel,
    S : StorageMut<P>,
{

    // Fills all nonzero bytes of self with the given intensity value, leaving
    // zeroed values untouched.
    pub fn fill_nonzero(&mut self, color : P) {
        let zero = P::zero();
        self.pixels_mut(1).for_each(|px| if *px != zero { *px = color });
    }
    
    pub fn fill(&mut self, color : P) {

        // TODO use std::intrinsic::write_bytes?

        #[cfg(feature="ipp")]
        unsafe {
            let (step, sz) = crate::image::ipputils::step_and_size_for_image(self);
            if self.pixel_is::<f32>() {
                let ans = crate::foreign::ipp::ippi::ippiSet_32f_C1R(
                    *mem::transmute::<_, &f32>(&color),
                    mem::transmute(self.as_mut_ptr()),
                    step,
                    sz
                );
                assert!(ans == 0);
                return;
            }

            if self.pixel_is::<u8>() {
                let ans = crate::foreign::ipp::ippi::ippiSet_8u_C1R(
                    *mem::transmute::<_, &u8>(&color),
                    mem::transmute(self.as_mut_ptr()),
                    step,
                    sz
                );
                // println!("{}", ans);
                assert!(ans == 0);
                return;
            }
        }

        self.pixels_mut(1).for_each(|px| *px = color );
    }

}

impl<'a, P, S> Image<P, S>
where
    P : Pixel,
    S : StorageMut<P>,
    &'a [P] : Storage<P>
{

    /*/// Applies closure to sub_window. Useful when you need to apply an operation
    /// iteratively to different regions of a window, and cannot do so because the mutable reference gets
    /// invalidated when a new sub-window is returned within a loop.
    pub fn apply_to_sub_window<'b, F>(&'b mut self, offset : (usize, usize), dim : (usize, usize), mut f : F)
    where
        F : FnMut(WindowMut<'b, N>)
    {
        let mut sub = unsafe { self.create_mutable_without_lifetime(offset, dim) };
        f(sub);
    }*/

    /*/// Analogous to slice::copy_within, copy a the sub_window (src, dim) to the sub_window (dst, dim).
    pub fn copy_within<'a>(
        &'a mut self, 
        src : (usize, usize), 
        dst : (usize, usize), 
        dim : (usize, usize)
    ) {
        // use crate::feature::shape;
        // assert!(!shape::rect_overlaps(&(src.0, src.1, dim.0, dim.1), &(dst.0, dst.1, dim.0, dim.1)), "copy_within: Windows overlap");//
        let src_win = unsafe {
            Window {
                offset : (self.offset.0 + src.0, self.offset.1 + src.1),
                width : self.original_size().1,
                sz : dim,
                slice : std::slice::from_raw_parts(self.slice.as_ref().as_ptr(), self.slice.as_ref().len()),
            }
        };

        // TODO not working since we made sub_window_mut take by value.
        // self.sub_window_mut(dst, dim).unwrap().copy_from(&src_win);
        unimplemented!()
    }*/

    pub fn row_mut(&'a mut self, ix : usize) -> Option<&'a mut [P]> {
        /*let stride = self.original_size().1;
        let tl = self.offset.0 * stride + self.offset.1;
        let start = tl + i*stride;
        let slice = &self.slice.as_ref()[start..(start+self.sz.1)];
        unsafe { std::slice::from_raw_parts_mut(slice.as_ptr() as *mut _, slice.len()) }*/
        if ix >= self.sz.0 {
            return None;
        }
        let start = ix*self.width;
        Some(&mut self.slice.as_mut()[start..(start+self.sz.1)])
    }

    pub unsafe fn row_mut_unchecked(&mut self, ix : usize) -> &mut [P] {
        let start = ix*self.width;
        self.slice.as_mut().get_unchecked_mut(start..(start+self.sz.1))
    }

}

/*impl<N> Index<RunLength> for Window<'_, N>
where
    N : Scalar + Copy
{

    type Output = [N];

    fn index(&self, rl: RunLength) -> &Self::Output {
        assert!(rl.start.1 + rl.length <= self.width() && rl.start.0 < self.height());
        unsafe { std::slice::from_raw_parts(&self[rl.start] as *const _, rl.length) }
    }

}

impl<N> Index<RunLength> for WindowMut<'_, N>
where
    N : Scalar + Copy
{

    type Output = [N];

    fn index(&self, rl: RunLength) -> &Self::Output {
        assert!(rl.start.1 + rl.length <= self.width() && rl.start.0 < self.height());
        unsafe { std::slice::from_raw_parts(&self[rl.start] as *const _, rl.length) }
    }

}

impl<N> IndexMut<RunLength> for WindowMut<'_, N>
where
    N : Scalar + Copy
{

    fn index_mut(&mut self, rl: RunLength) -> &mut Self::Output {
        assert!(rl.start.1 + rl.length <= self.width() && rl.start.0 < self.height());
        unsafe { std::slice::from_raw_parts_mut(&mut self[rl.start] as *mut _, rl.length) }
    }

}*/

/*impl<N> Index<(usize, usize)> for WindowMut<'_, N>
where
    N : Scalar + Copy
{

    type Output = N;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        unsafe { self.slice.get_unchecked(index::linear_index((self.offset.0 + index.0, self.offset.1 + index.1), self.original_size().1)) }
    }
}

impl<N> IndexMut<(usize, usize)> for WindowMut<'_, N> 
where
    N : Scalar + Copy
{
    
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        unsafe { self.slice.get_unchecked_mut(index::linear_index((self.offset.0 + index.0, self.offset.1 + index.1), self.original_size().1)) }
    }

}

impl<N> Index<(u16, u16)> for WindowMut<'_, N>
where
    N : Scalar + Copy
{

    type Output = N;

    fn index(&self, index: (u16, u16)) -> &Self::Output {
        self.index((index.0 as u16, index.1 as u16))
    }
}

impl<N> IndexMut<(u16, u16)> for WindowMut<'_, N>
where
    N : Scalar + Copy
{

    fn index_mut(&mut self, index: (u16, u16)) -> &mut Self::Output {
        self.index_mut((index.0 as u16, index.1 as u16))
    }
    
}*/

/*impl<'a, N> AsRef<DMatrixSlice<'a, N>> for Window<'a, N>
where
    N : Scalar + Copy
{
    fn as_ref(&self) -> &DMatrixSlice<'a, N> {
        // &self.slice
        unimplemented!()
    }
}*/

/*impl<'a, M, N> Downsample<Image<N>> for Window<'a, M> 
where
    M : Scalar + Copy,
    N : Scalar + From<M>
{

    fn downsample(&self, dst : &mut Image<N>) {
        // let (nrows, ncols) = dst.shape();
        /*let step_rows = self.sz.0 / dst.nrows;
        let step_cols = self.sz.1 / dst.ncols;
        assert!(step_rows == step_cols);
        sampling::slices::subsample_convert_window(
            self.src,
            self.offset,
            self.sz, 
            (dst.nrows, dst.ncols),
            step_rows,
            dst.buf.as_mut_slice().chunks_mut(dst.nrows)
        );*/
        unimplemented!()
    }
    
}*/

/*/// Data is assumed to live on the matrix in a column-order fashion, not row-ordered.
impl<N> From<DMatrix<N>> for Image<N> 
where
    N : Scalar + Copy
{
    fn from(slice : DMatrix<N>) -> Self {
        /*let (nrows, ncols) = s.shape();
        let data : Vec<N> = s.data.into();
        let buf = DVector::from_vec(data);
        Self{ buf, nrows, ncols }*/
        let ncols = buf.ncols();
        Self{ buf, ncols }
    }
}*/

/*impl<N> From<(Vec<N>, usize)> for Image<N> 
where
    N : Scalar
{
    fn from(s : (Vec<N>, usize)) -> Self {
        let (nrows, ncols) = (s.1, s.0.len() - s.1);
        Self{ slice : DVector::from_vec(s.0), nrows, ncols  }
    }
}*/

/*impl<N> AsRef<[N]> for Image<N>
where
    N : Scalar + Copy + Any
{
    fn as_ref(&self) -> &[N] {
        &self.slice[..]
    }
}

impl<N> AsMut<[N]> for Image<N> 
where
    N : Scalar + Copy + Any
{
    fn as_mut(&mut self) -> &mut [N] {
        &mut self.slice[..]
    }
}*/

/*impl<N> fmt::Display for Window<'_, N> 
where
    N : Scalar + Copy,
    f64 : From<N>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", io::build_pgm_string_from_slice(&self.slice, self.original_size().1))
    }
}

impl<N> fmt::Display for WindowMut<'_, N> 
where
    N : Scalar + Copy + Default,
    f64 : From<N>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", io::build_pgm_string_from_slice(&self.slice, self.original_size().1))
    }
}*/

/*impl<N> fmt::Display for Image<N>
where
    N : Scalar + Copy,
    f64 : From<N>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", io::build_pgm_string_from_slice(&self.slice[..], self.width))
    }
}*/

/*impl showable::Show for Window<'_, u8> {

    fn show(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // crate::io::to_html(&self).fmt(f)
        write!(f, "{}", crate::io::to_html(&self) )
    }

    fn modality(&self) -> showable::Modality {
        showable::Modality::XML
    }

}

impl showable::Show for Image<u8> {

    fn show(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // crate::io::to_html(&self.full_window()).fmt(f)
        write!(f, "{}", crate::io::to_html(&self.full_window()) )
    }

    fn modality(&self) -> showable::Modality {
        showable::Modality::XML
    }

}*/

/*#[cfg(feature="literate")]
impl<'a> literate::show::Stack<'a> for Window<'a, u8> {

    fn header(&'a self) -> Option<Box<dyn Display>> {
        None
    }

    fn stack(&'a self) -> Vec<&'a dyn Show> {
        vec![self as &dyn Show]
    }

}*/

/*#[test]
fn checkerboard() {
    let src : [u8; 4] = [0, 1, 1, 0];
    let mut converted : Image<f32> = ImageBuf::new_constant(4, 4, 0.0);
    let win = Window::from_square_slice(&src);
    //converted.convert_from_window(&win);
    // println!("{}", converted);
}*/

/*impl<N> AsRef<Vec<N>> for Image<N> 
where
    N : Scalar
{
    fn as_ref(&self) -> &Vec<N> {
        self.slice.data.as_vec()
    }
}

impl<N> AsMut<Vec<N>> for Image<N> 
where
    N : Scalar
{
    fn as_mut(&mut self) -> &mut Vec<N> {
        unsafe{ self.slice.data.as_vec_mut() }
    }
}*/

/*impl<N> AsRef<DVector<N>> for Image<N> 
where
    N : Scalar
{
    fn as_ref(&self) -> &DVector<N> {
        &self.slice
    }
}

impl<N> AsMut<DVector<N>> for Image<N> 
where
    N : Scalar
{
    fn as_mut(&mut self) -> &mut DVector<N> {
        &mut self.slice
    }
}*/

/*/// Result of thresholding an image. This structure carries a (row, col) index and
/// a scalar value for this index. Rename to Keypoints.
pub struct SparseImage {

}

/// Subset of a SparseImage
pub struct SparseWindow<'a> {

    /// Source sparse image
    source : &'a SparseImage,
    
    /// Which indices we will use from the source
    ixs : Vec<usize>
}*/

// TODO implement borrow::ToOwned

/*
// Local maxima of the image multiscale transformation
pub struct Keypoint { }

// Object characterized by a set of close keypoints where ordered pairs share close angles. 
pub struct Edge { }

// Low-dimensional approximation of an edge (in terms of lines and curves).
pub struct Shape { }

// Object characterized by a set of shapes.
pub struct Object { }
*/

/*
The match item at procedural macros has the syntax $name:token where token must be one of:
item — an item, like a function, struct, module, etc.
block — a block (i.e. a block of statements and/or an expression, surrounded by braces)
stmt — a statement
pat — a pattern
expr — an expression
ty — a type
ident — an identifier
path — a path (e.g., foo, ::std::mem::replace, transmute::<_, int>, …)
meta — a meta item; the things that go inside #[...] and #![...] attributes
tt — a single token tree
vis — a possibly empty Visibility qualifier

Repeated arguments are treated as (* matches zero or more; + matches zero or one):
macro_rules! add_as{
    ( $($a:expr), )=>{ {  0 $(+$a)* }    }
}

The token type that repeats is enclosed in $(), followed by a separator and a * or a +, indicating the number of times the token will repeat.
*/

/*macro_rules! sql {
 // macth like arm for macro
    ($a:expr) => {
    
    },
    ($a:expr, $b:expr)=>{
 // macro expand to this code
        {
// $a and $b will be templated using the value/variable provided to macro
            $a+$b
        }
    }
}*/

/*#[no_mangle]
pub extern "C" fn get_funcs() -> Box<Vec<interactive::AlienFunc>> {

    use interactive::AlienFunc;

    let mut funcs = Vec::new();
    funcs.push(AlienFunc::new("add_three", |a : i64| -> Result<i64, Box<rhai::EvalAltResult>> { Ok(a + 3) }));
    funcs.push(AlienFunc::new("add_four", |a : i64| -> Result<i64, Box<rhai::EvalAltResult>> { Ok(a + 4) }));
    funcs.push(AlienFunc::new("append_text_to_string", |a : String| -> Result<String, Box<rhai::EvalAltResult>> { Ok(format!("{}newtext", a)) }));

    Box::new(funcs)
}*/

/*#[derive(Serialize, Deserialize, Clone)]
pub struct MyStruct {
    field : [i64; 2]
}

impl Default for MyStruct {

    fn default() -> Self {
        Self { field : [0, 0] }
    }

}

impl std::iter::IntoIterator for MyStruct {

    type Item = i64;

    type IntoIter = Box<dyn Iterator<Item=i64>>;

    fn into_iter(self) -> Self::IntoIter {
        Box::new(vec![self.field[0].clone(), self.field[1].clone()].into_iter())
    }

}

fn convert_from_map(other : rhai::Dynamic) -> Result<MyStruct, Box<rhai::EvalAltResult>> {
    match other.try_cast::<rhai::Map>() {
        Some(map) => {
            match map.get("field") {
                Some(v) => {
                    match v.clone().try_cast::<rhai::Array>() {
                        Some(arr) => {
                            match (arr.get(0).and_then(|v| v.clone().try_cast::<i64>()), arr.get(1).and_then(|v| v.clone().try_cast::<i64>())) {
                                (Some(v1), Some(v2)) => {
                                    Ok(MyStruct { field : [v1, v2] })
                                },
                                _ => Err(Box::new("Invalid fields".into()))
                            }
                        },
                        None => Err(Box::new("Field is not array".into()))
                    }
                },
                None => Err(Box::new("Missing field".into()))
            }
        },
        None => Err(Box::new("Result is not a map".into()))
    }
}*/

/*impl deft::Interactive for MyStruct {

    #[export_name="register_MyStruct"]
    extern "C" fn interactive() -> Box<deft::TypeInfo> {
        deft::TypeInfo::builder::<Self>()
            .fallible("add_one", |a : i64| -> Result<i64, Box<rhai::EvalAltResult>> { Ok(a + 1) })
            .iterable()
            .initializable()
            .parseable()
            .indexable(|s : &mut Self, ix : i64| { Ok(s.field[ix as usize]) })
            .mutably_indexable(|s : &mut Self, ix : i64, val : i64| { s.field[ix as usize] = val; Ok(()) })
            .field("field", |s : &mut MyStruct| { Ok(vec![rhai::Dynamic::from(s.field[0]), rhai::Dynamic::from(s.field[1])]) })
            .convertible(|s : &mut Self, other : rhai::Dynamic| { convert_from_map(other) })
            .priority(0)
            .build()
    }

}*/

/*#[test]
fn deft_test() {
    let a = 1i64;
    let b = String::from("Hello");
    let c = &1i64;
    let d = MyStruct { field : [0, 1] };
    deft::repl!(a, b, c, d);
}

impl deft::Show for Image<u8> {

}

impl deft::Embed for Image<u8> {

    fn embed(&self) -> String {
        crate::io::to_html(&self.full_window())
    }

}

impl deft::Interactive for Image<u8> {

    fn short_name() -> &'static str {
        "Image"
    }

    #[export_name="register_Image"]
    extern "C" fn interactive() -> Box<deft::TypeInfo> {

        use rhai::{Dynamic, Array};
        use deft::ReplResult;

        deft::TypeInfo::builder::<Self>()
            .fallible("open",
            |img : &mut Self, path : rhai::ImmutableString| -> Result<Self, Box<rhai::EvalAltResult>> {
                let new_img = crate::io::decode_from_file(&path)
                    .map_err(|e| Box::new(rhai::EvalAltResult::from(format!("{}", e))) )?;
                Ok(new_img)
            })
            .fallible("shape", |s : &mut Self| -> ReplResult<Array> {
                Ok(vec![Dynamic::from(s.height() as i64), Dynamic::from(s.width() as i64) ])
            })
            .fallible("height", |s : &mut Self| -> ReplResult<i64> { Ok(s.height() as i64) })
            .fallible("width", |s : &mut Self| -> ReplResult<i64> { Ok(s.width() as i64) })
            .fallible("draw", |s : &mut Self, mark : rhai::Map| -> ReplResult<()> {
                /*println!("{:?}", s.shape());
                for mark in marks.iter() {
                    match mark.clone().try_cast::<Patch>() {
                        Some(patch) => {
                            s.full_window_mut().draw(Mark::Shape(patch.outer_points(crate::feature::patch::ExpansionMode::Contour), 255));
                        },
                        None => {
                            return Err("Mark is not patch".into());
                        }
                    }
                }*/

                let mark : Mark = deft::convert::deserialize_from_map(mark).unwrap();
                s.full_window_mut().draw(mark);
                Ok(())
            })
            .initializable()
            .showable()
            .embeddable()
            .priority(0)
            .build()
    }

}*/

/*impl interactive::Interactive for Image<u8> {

    // fn hosrt_name() -> &'static str {
    //    "Image"
    // }

    fn new() -> Self {
        unimplemented!()
    }

    fn embed(&self) -> Option<String> {
        Some(crate::io::to_html(&self.full_window()))
    }

    #[export_name="register_image"]
    extern "C" fn interactive(registry : Box<interactive::Registry<'_>>) -> Box<interactive::TypeInfo> {

        registry.add::<Self>()
            .fallible_method("cat", |a : rhai::ImmutableString| -> Result<String, Box<rhai::EvalAltResult>> { Ok(format!("{}hellofromclient", a)) })
            .fallible_method("add_one", |a : i64| -> Result<i64, Box<rhai::EvalAltResult>> { Ok(a + 1) })
            .register()

        /*use rhai;
        Self::prepare(engine);
        Self::display(engine);
        Self::serialization(engine);

        Self::new(
            engine,
            |img, map| {
                match (map.get("width").and_then(|w| w.as_int().ok() ), (map.get("height").and_then(|h| h.as_int().ok() ))) {
                    (Some(w), Some(h)) => {
                        if w > 0 && h > 0 {
                            *img = ImageBuf::new_constant(h as usize, w as usize, 0);
                            Ok(())
                        } else {
                            Err(Box::new(rhai::EvalAltResult::from("Arguments should be greater than zero")))
                        }
                    },
                    _ => {
                        Err(Box::new(rhai::EvalAltResult::from("Invalid fields")))
                    }
                }

            }
        );

        engine.register_result_fn(
            "open",
            |img : &mut Self, path : rhai::ImmutableString| -> Result<(), Box<rhai::EvalAltResult>> {
                let new_img = crate::io::decode_from_file(&path)
                    .map_err(|e| Box::new(rhai::EvalAltResult::from(format!("{}", e))) )?;
                *img = new_img;
                Ok(())
            }
        );

        engine.register_fn("add_two", |a : i64| -> i64 { a + 2 });
        engine.register_fn("add_two_float", |a : f64| -> f64 { a + 2. });
        engine.register_fn("append_text_client", |a : String| -> String { format!("{}newtext", a) });
        engine.register_fn("append_text_client_dyn", |a : rhai::Dynamic| -> rhai::Dynamic { rhai::Dynamic::from(format!("{}newtext", a.into_string().unwrap())) });

        use std::any::TypeId;
        let immutable_str_id = unsafe { std::mem::transmute::<u64, TypeId>(3264575275192760566) };

        engine.register_raw_fn(
            "append_text_client_raw",
            /*&[TypeId::of::<ImmutableString>()]*/ &[immutable_str_id],
            |ctx : rhai::NativeCallContext<'_>, args : &mut [&mut rhai::Dynamic]| -> Result<rhai::Dynamic, Box<rhai::EvalAltResult>> {
                Ok(rhai::Dynamic::from(format!("{}newtext", std::mem::take(args[0]).cast::<rhai::ImmutableString>())))
            }
        );

        println!("Type id of ImmutableString at shutter : {:?}", rhai::ImmutableString::from("").type_id() );
        println!("Type id of Dynamic at shutter : {:?}", rhai::Dynamic::from(1i16).type_id() );
        println!("Type id of Arc at client : {:?}", std::sync::Arc::new(1i16).type_id() );
        println!("ImmutableString at client: {:?}", rhai::ImmutableString::from(""));

        use smartstring;
        println!("SmartString at client: {:?}", smartstring::SmartString::<smartstring::LazyCompact>::new_const().type_id());

        Self::info()*/
    }

    // Perhaps we abstract certain details away,
    // and just require the registration of "associated" and "methods",
    // automatically taking care of the plumbing without exposing the engine
    // to the user.

}*/

#[cfg(feature="opencv")]
pub fn from_nalgebra3_vec(v : nalgebra::Vector3<f64>) -> opencv::core::Mat {

    use opencv::core;
    use opencv::prelude::MatTraitManual;
    use opencv::prelude::MatTrait;

    let mut mat = core::Mat::default();
    unsafe {
        mat.create_rows_cols(3, 1, core::CV_64F);
        for i in 0..3 {
            *mat.at_mut::<f64>(i).unwrap() = v[i as usize];
        }
    }

    mat
}

#[cfg(feature="opencv")]
pub fn to_nalgebra3_vec(m : opencv::core::Mat) -> nalgebra::Vector3<f64> {

    use opencv::core;
    use opencv::prelude::*;

    let mut out = Vector3::zeros();
    unsafe {
        for i in 0..3 {
            out[i as usize] = *m.at_unchecked::<f64>(i).unwrap();
        }
    }
    out
}

#[cfg(feature="opencv")]
pub fn to_nalgebra3_mat(m : opencv::core::Mat) -> nalgebra::Matrix3<f64> {

    use opencv::core;
    use opencv::prelude::*;

    let mut out = Matrix3::zeros();
    unsafe {
        for i in 0..3 {
            for j in 0..3 {
                out[(i as usize, j as usize)] = *m.at_2d_unchecked::<f64>(i, j).unwrap();
            }
        }
    }
    out
}

#[cfg(feature="opencv")]
pub fn from_nalgebra3(m : nalgebra::Matrix3<f64>) -> opencv::core::Mat {

    use opencv::core;
    use opencv::prelude::*;

    let mut mat = core::Mat::default();
    unsafe {
        mat.create_rows_cols(3, 3, core::CV_64F);
        for i in 0..3 {
            for j in 0..3 {
                *mat.at_2d_unchecked_mut::<f64>(i, j).unwrap() = m[(i as usize, j as usize)];
            }
        }
    }

    mat
}

/*impl<'a, N> Borrow<Window<'a, N>> for Image<N>
where
    N : Scalar + Clone + Copy + Serialize + DeserializeOwned + Any + Zero + Default
{

    fn borrow(&self) -> &Window<'a, N> {
        &self.full_window()
    }

}*/

pub fn assert_nonzero(shape : (usize, usize)) {
    assert!(shape.0 >= 1 && shape.1 >= 1)
}

pub fn coords_across_line(
    src : (usize, usize),
    dst : (usize, usize),
    (nrow, ncol) : (usize, usize)
) -> Box<dyn Iterator<Item=(usize, usize)>> {
    if src.0 == dst.0 {
        Box::new((src.1.min(dst.1)..(src.1.max(dst.1))).map(move |c| (src.0, c) ))
    } else if src.1 == dst.1 {
        Box::new((src.0.min(dst.0)..(src.0.max(dst.0))).map(move |r| (r, src.1 ) ))
    } else {
        let (dist, theta) = index::index_distance(src, dst, nrow);
        let d_max = dist as usize;
        Box::new((0..d_max).map(move |i| {
            let x_incr = theta.cos() * i as f64;
            let y_incr = theta.sin() * i as f64;
            let x_pos = (src.1 as i32 + x_incr as i32) as usize;
            let y_pos = (src.0 as i32 - y_incr as i32) as usize;
            (y_pos, x_pos)
        }))
    }
}

impl ImageBuf<u8> {

    pub fn filter_max(&self, height : usize, width : usize) -> Self {
        let mut dst = ImageBuf::new_constant_like(&self, 0);
        crate::local::IppiFilterMinMax::new(self.height(), self.width(), (height, width), false)
            .apply(&self, &mut dst);
        dst
    }

    pub fn filter_box(&self, height : usize, width : usize) -> Self {
        let mut dst = ImageBuf::new_constant_like(&self, 0);
        crate::local::IppiFilterBox::new(self.height(), self.width(), (height, width))
            .apply(&self, &mut dst);
        dst
    }

}

impl rhai::CustomType for ImageBuf<u8> {

    fn build(mut builder: rhai::TypeBuilder<'_, Self>) {

        type Result<T> = std::result::Result<T, Box<rhai::EvalAltResult>>;

        builder.with_name("image")
            .with_fn("window", |img : &mut Self, y : i64, x : i64, height : i64, width : i64| -> Result<Self> {
                let win = img.window((y as usize, x as usize), (height as usize, width as usize))
                    .ok_or("Invalid bounds")?;
                Ok(win.clone_owned())
            })
            .with_fn("show", |img : &mut Self| img.show() )
            .with_fn("paste", |img : &mut Self, other : Self, y : i64, x : i64| -> Result<Self> {
                let dst = img.paste(&other, (y as usize, x as usize))
                    .ok_or("Invalid bounds")?;
                Ok(dst)
            })
            .with_fn("filter_max", |img : &mut Self, height : i64, width : i64| -> Self {
                img.filter_max(height as usize, width as usize)
            })
            .with_fn("filter_box", |img : &mut Self, height : i64, width : i64| -> Self {
                img.filter_box(height as usize, width as usize)
            })
            .with_fn("scalar_add", |img : &mut Self, by : i64| -> Self {
                img.scalar_add(by as u8)
            })
            .with_fn("scalar_sub", |img : &mut Self, by : i64| -> Self {
                img.scalar_sub(by as u8)
            })
            .with_fn("scalar_mul", |img : &mut Self, by : i64| -> Self {
                img.scalar_mul(by as u8)
            })
            .with_fn("scalar_div", |img : &mut Self, by : i64| -> Self {
                img.scalar_div(by as u8)
            });
    }

}

