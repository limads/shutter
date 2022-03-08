use nalgebra::*;
use ripple::signal::sampling::{self};
use std::ops::{Index, IndexMut, Mul, Add, AddAssign, MulAssign, SubAssign, Range, Div, Rem};
use simba::scalar::SubsetOf;
use std::fmt;
use std::fmt::Debug;
use simba::simd::{AutoSimd};
use std::convert::TryFrom;
use crate::feature::patch::{self, Patch};
use itertools::Itertools;
use crate::feature::patch::ColorMode;
use num_traits::Zero;
use serde::{Serialize, Deserialize, de::DeserializeOwned};
use std::any::Any;
use tempfile;
use std::fs;
use std::io::Write;

#[cfg(feature="opencv")]
use opencv::core;

#[cfg(feature="opencv")]
use opencv::imgproc;

//#[cfg(feature="mkl")]
// mod fft;

/*#[cfg(feature="mkl")]
pub use fft::*;

#[cfg(feature="gsl")]
pub(crate) mod dwt;

#[cfg(feature="gsl")]
pub use dwt::*;*/

// #[cfg(feature="gsl")]
// mod interp;

// #[cfg(feature="gsl")]
// pub use interp::Interpolation2D;

use crate::io;

#[cfg(feature="ipp")]
pub mod ipp;

#[cfg(feature="opencv")]
pub mod cvutils;

pub mod index;

pub mod draw;

pub(crate) mod iter;

/*pub trait Raster {

    fn width(&self) -> usize;

    fn height(&self) -> usize;

    /// Returns (height x width)
    fn dimensions(&self) -> (usize, usize) {
        (self.height(), self.width())
    }

    fn pixels(&self) -> impl Iterator<item=&u8>;

}*/

/// Digital image, represented row-wise. Fundamentally, an image differs from a matrix because
/// it is oriented row-wise in memory, while a matrix is oriented column-wise. Also, images are
/// bounded at a low and high end, because they are the product of a saturated digital quantization
/// process. But indexing, following OpenCV convention, happens from the top-left point, following
/// the matrix convention.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(bound = "")]
#[repr(C)]
pub struct Image<N> 
where
    N : Scalar + Clone + Copy + Serialize + DeserializeOwned + Any
{
    buf : Vec<N>,
    ncols : usize
}

pub type ByteImage = Image<u8>;

pub type FloatImage = Image<f32>;

impl<N> fmt::Display for Image<N>
where
    N : Scalar + Clone + Copy + Serialize + DeserializeOwned + Any + Default + num_traits::Zero
{

    fn fmt(&self, f : &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Image (Height = {}; Width = {})", self.height(), self.width())
    }

}

impl From<Image<u8>> for Image<f32> {

    fn from(img : Image<u8>) -> Image<f32> {
        #[cfg(feature="opencv")]
        {
            use opencv::prelude::MatTrait;
            let mut out = Image::<f32>::new_constant(img.height(), img.width(), 0.);
            let m : opencv::core::Mat = (&img).into();
            m.convert_to(&mut out, opencv::core::CV_32F, 1.0, 0.0);
            return out;
        }
        unimplemented!()
    }

}

impl TryFrom<Image<f32>> for Image<u8> {

    type Error = ();

    fn try_from(img : Image<f32>) -> Result<Image<u8>, ()> {
        unimplemented!()
    }

}

/*impl<N> Sub<Rhs=Image> for Image<N> {

    pub fn sub() -> {

    }
}*/

impl<N> Image<N>
where
    N : Scalar + Copy + Default + Zero + Copy + Serialize + DeserializeOwned + Any
{

    pub unsafe fn unchecked_linear_index(&self, ix : usize) -> &N {
        self.buf.get_unchecked(ix)
    }

    pub unsafe fn unchecked_linear_index_mut(&mut self, ix : usize) -> &mut N {
        self.buf.get_unchecked_mut(ix)
    }

    pub fn linear_index(&self, ix : usize) -> &N {
        &self.buf[ix]
    }

    pub fn linear_index_mut(&mut self, ix : usize) -> &mut N {
        &mut self.buf[ix]
    }

    pub fn from_iter(iter : impl Iterator<Item=N>, ncols : usize) -> Self {
        let buf : Vec<N> = iter.collect();
        Self::from_vec(buf, ncols)
    }

    pub fn subsample_from(&mut self, content : &[N], ncols : usize, sample_n : usize) {
        assert!(ncols < content.len(), "ncols smaller than content length");
        let nrows = content.len() / ncols;
        let sparse_ncols = ncols / sample_n;
        let sparse_nrows = nrows / sample_n;
        self.ncols = sparse_ncols;
        if self.buf.len() != sparse_nrows * sparse_ncols {
            self.buf.clear();
            self.buf.extend((0..(sparse_nrows*sparse_ncols)).map(|_| N::zero() ));
        }
        for r in 0..sparse_nrows {
            for c in 0..sparse_ncols {
                self.buf[r*sparse_ncols + c] = content[r*sample_n*ncols + c*sample_n];
            }
        }
    }

    #[cfg(feature="opencv")]
    pub fn equalize_inplace(&mut self) {
        // assert!(self.shape() == dst.shape());
        let src : core::Mat = self.full_window().into();
        let mut dst : core::Mat = self.full_window_mut().into();
        imgproc::equalize_hist(&src, &mut dst);
    }

    #[cfg(feature="opencv")]
    pub fn equalize_mut(&mut self, dst : &mut Image<N>) {
        assert!(self.shape() == dst.shape());
        let src : core::Mat = self.full_window().into();
        let mut dst : core::Mat = dst.full_window_mut().into();
        imgproc::equalize_hist(&src, &mut dst);
    }

    pub fn new_from_slice(source : &[N], ncols : usize) -> Self {
        let mut buf = Vec::with_capacity(source.len());
        unsafe { buf.set_len(source.len()); }
        buf.copy_from_slice(&source);
        Self{ buf, ncols }
    }

    pub fn from_vec(buf : Vec<N>, ncols : usize) -> Self {
        //if buf.len() as f64 % ncols as f64 != 0.0 {
        //    panic!("Invalid image lenght");
        //}
        assert!(buf.len() % ncols == 0);
        Self { buf, ncols }
    }

    pub fn new_constant(nrows : usize, ncols : usize, value : N) -> Self {
        let mut buf = Vec::with_capacity(nrows * ncols);
        buf.extend((0..(nrows*ncols)).map(|_| value ));
        Self{ buf, ncols }
    }
    
    pub fn shape(&self) -> (usize, usize) {
        (self.buf.len() / self.ncols, self.ncols)
    }
    
    pub fn width(&self) -> usize {
        self.ncols
    }

    pub fn height(&self) -> usize {
        self.buf.len() / self.ncols
    }

    pub fn full_window<'a>(&'a self) -> Window<'a, N> {
        self.window((0, 0), self.shape()).unwrap()
    }
    
    pub fn full_window_mut<'a>(&'a mut self) -> WindowMut<'a, N> {
        let shape = self.shape();
        self.window_mut((0, 0), shape).unwrap()
    }
    
    pub fn window<'a>(&'a self, offset : (usize, usize), sz : (usize, usize)) -> Option<Window<'a, N>> {
        let orig_sz = self.shape();
        if offset.0 + sz.0 <= orig_sz.0 && offset.1 + sz.1 <= orig_sz.1 {
            Some(Window {
                win : &self.buf[..],
                offset,
                orig_sz,
                win_sz : sz
            })
        } else {
            None
        }
    }
    
    pub fn window_mut<'a>(&'a mut self, offset : (usize, usize), sz : (usize, usize)) -> Option<WindowMut<'a, N>> {
        let orig_sz = self.shape();
        if offset.0 + sz.0 <= orig_sz.0 && offset.1 + sz.1 <= orig_sz.1 {
            Some(WindowMut {
                win : &mut self.buf[..],
                offset,
                orig_sz,
                win_sz : sz
            })
        } else {
            None
        }
    }
    
    pub fn downsample(&mut self, src : &Window<N>) {
        assert!(src.is_full());
        let src_ncols = src.orig_sz.1;
        let dst_ncols = self.ncols;
        
        #[cfg(feature="opencv")]
        unsafe {
            cvutils::resize(
                src.win,
                &mut self.buf[..],
                src_ncols, 
                None,
                dst_ncols,
                None
            );
            return;
        }
        
        #[cfg(feature="ipp")]
        unsafe {
            let dst_nrows = self.buf.len() / self.ncols;
            ipp::resize(src.win, &mut self.buf, src.orig_sz, (dst_nrows, dst_ncols));
        }
        
        panic!("Image::downsample requires that crate is compiled with opencv or ipp feature");

        // TODO use resize::resize for native Rust solution
    }
    
    // TODO call this downsample_convert, and leave alias as a second enum argument:
    // AntiAliasing::On OR AntiAliasing::Off. Disabling antialiasing calls this implementation
    // that just iterates over the second buffer; enabling it calls for more costly operations.
    pub fn downsample_aliased<M>(&mut self, src : &Window<M>) 
    where
        M : Scalar + Copy,
        N : Scalar + From<M>
    {
        let (nrows, ncols) = self.shape();
        let step_rows = src.win_sz.0 / nrows;
        let step_cols = src.win_sz.1 / ncols;
        assert!(step_rows == step_cols);
        sampling::slices::subsample_convert_with_offset(
            src.win,
            src.offset,
            src.win_sz, 
            (nrows, ncols),
            step_rows,
            self.buf.chunks_mut(nrows)
        );
    }
    
    /*
    // TODO
    IppStatus ippiCopy_<mod> ( const Ipp<datatype>* pSrc , int srcStep , Ipp<datatype>* pDst ,
    int dstStep , IppiSize roiSize );
    */
    pub fn copy_from(&mut self, other : &Image<N>) {
        self.buf.copy_from_slice(other.buf.as_slice());
    }
    
    pub fn copy_from_slice(&mut self, slice : &[N]) {
        assert!(self.buf.len() == slice.len());
        self.buf.copy_from_slice(&slice);
    }

    pub fn as_slice(&self) -> &[N] {
        &self.buf[..]
    }
    
    pub fn as_mut_slice(&mut self) -> &mut [N] {
        &mut self.buf[..]
    }
    
    /*pub fn convert_from(&mut self, other : &Image<N>) {
        self.buf.as_slice_mut().copy_from(other.buf.as_slice());
    }*/
    
    pub fn windows_mut(&mut self, sz : (usize, usize)) -> impl Iterator<Item=WindowMut<'_, N>>
    where
        N : Mul<Output=N> + MulAssign
    {
        self.full_window_mut().windows_mut(sz)
    }

    pub fn windows(&self, sz : (usize, usize)) -> impl Iterator<Item=Window<'_, N>>
    where
        N : Mul<Output=N> + MulAssign
    {
        self.full_window().windows(sz)
    }
    
    // TODO make this generic like impl AsRef<Window<M>>, and make self carry a field Window corresponding
    // to the full window to work as the AsRef implementation, so the user can pass images here as well.
    pub fn convert<M>(&mut self, other : &Window<M>) 
    where
        M : Scalar + Default
    {
        let ncols = self.ncols;
        
        #[cfg(feature="opencv")]
        unsafe {
            cvutils::convert(
                other.win, 
                &mut self.buf[..], 
                other.orig_sz.1, 
                Some((other.offset, other.win_sz)),
                ncols,
                None
            );
            return;
        }
        
        #[cfg(feature="ipp")]
        {
            assert!(other.is_full());
            unsafe { ipp::convert(other.win, &mut self.buf[..], ncols); }
            return;
        }
        
        panic!("Either opencv or ipp feature should be enabled for image conversion");
    }
    
    /*pub fn iter(&self) -> impl Iterator<Item=&N> {
        let shape = self.shape();
        iterate_row_wise(&self.buf[..], (0, 0), shape, shape)
    }*/
    
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    // pub fn windows(&self) -> impl Iterator<Item=Window<'_, N>> {
    //    unimplemented!()
    // }
    
}

impl<N> Image<N>
    where N : Scalar + Copy + RealField + Copy + Serialize + DeserializeOwned + Any
{

    pub fn scale_by(&mut self, scalar : N)  {
        self.buf.iter_mut().for_each(|val| *val *= scalar);
    }
    
    pub fn unscale_by(&mut self, scalar : N)  {
        self.buf.iter_mut().for_each(|val| *val *= scalar);
    }
    
    /*pub fn iter(&self) -> impl Iterator<Item=&N> {
        self.full_window().clone().iter()
    }*/
    
}

impl Image<u8> {

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

    pub fn draw(&mut self, mark : Mark) {
        self.full_window_mut().draw(mark);
    }

    /// Builds a binary image from a set of points.
    pub fn binary_from_points(nrow : usize, ncol : usize, pts : impl Iterator<Item=(usize, usize)>) -> Self {
        let mut img = Image::<u8>::new_constant(nrow, ncol, 0);
        for pt in pts {
            img[pt] = 255;
        }
        img
    }
}

impl Image<f32> {

    pub fn max(&self) -> ((usize, usize), f32) {
        let (mut max_ix, mut max) = ((0, 0), f32::NEG_INFINITY);
        for (lin_ix, px) in self.full_window().pixels(1).enumerate() {
            if *px > max {
                max_ix = index::coordinate_index(lin_ix, self.ncols);
                max = *px
            }
        }
        (max_ix, max)
    }
}

impl<N> Index<(usize, usize)> for Image<N> 
where
    N : Scalar + Copy + Serialize + DeserializeOwned + Any
{

    type Output = N;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        unsafe { self.buf.get_unchecked(index::linear_index(index, self.ncols)) }
    }
}

impl<N> IndexMut<(usize, usize)> for Image<N>
where
    N : Scalar + Copy + Default + Copy + Serialize + DeserializeOwned + Any
{
    
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        unsafe { self.buf.get_unchecked_mut(index::linear_index(index, self.ncols)) }
    }
    
}

/// Borrowed subset of an image. Referencing the whole source slice (instead of just its
/// portion of interest) might be useful to represent overlfowing operations (e.g. draw)
/// as long as the operation does not violate bounds of the original image. We just have
/// to be careful to not expose the rest of the image in the API.
#[derive(Debug, Clone)]
pub struct Window<'a, N> 
where
    N : Scalar
{
    // Window offset, with respect to the top-left point (row, col).
    offset : (usize, usize),
    
    // Original image dimensions (height, width). orig_sz.0 MUST be win.len() / orig_sz.1
    orig_sz : (usize, usize),
    
    // This window size.
    win_sz : (usize, usize),
    
    // Original image full slice. Might refer to an actual pre-allocated image
    // buffer slice; or a slice from an external source (which is why we don't
    // simply reference image here).
    win : &'a [N],
    
    // Stack-allocated arrays which keep the slices for the chunks(.) method.
    // chunks : ([&[N]; 2], [&[N]; 4], [&[N]; 8], [&[N]; 16]);
    
    // TODO remove
    // transposed : bool
}

pub struct Neighborhood<'a, N>
where
    N : Scalar
{
    pub center : Window<'a, N>,
    pub left : Window<'a, N>,
    pub top : Window<'a, N>,
    pub right : Window<'a, N>,
    pub bottom : Window<'a, N>,
}

impl<'a, N> Window<'a, N>
where
    N : Scalar
{

    /// The cartesian index is defined as (img.height() - pt[1], pt[0]).
    /// It indexes the image by imposing the cartesian analytical plane over it,
    /// an converting it as specified.
    pub fn cartesian_index<T>(&self, pt : Vector2<T>) -> &N
    where
        usize : From<T>,
        T : Copy,
        N : Mul<Output=N> + MulAssign + Copy + Serialize + DeserializeOwned
    {
        let ix = (self.height() - usize::from(pt[1]), usize::from(pt[0]));
        &self[ix]
    }

    pub fn offset(&self) -> (usize, usize) {
        self.offset
    }

    pub fn is_full(&'a self) -> bool {
        self.orig_sz == self.win_sz
    }
    

    /*pub fn shape(&self) -> (usize, usize) {
        self.win_sz
    }

    pub fn width(&self) -> usize {
        self.shape().0
    }

    pub fn height(&self) -> usize {
        self.shape().1
    }*/

    pub fn sub_window(&'a self, offset : (usize, usize), dims : (usize, usize)) -> Option<Window<'a, N>> {
        let new_offset = (self.offset.0 + offset.0, self.offset.1 + offset.1);
        if new_offset.0 + dims.0 <= self.orig_sz.0 && new_offset.1 + dims.1 <= self.orig_sz.1 {
            Some(Self {
                win : self.win,
                offset : new_offset,
                orig_sz : self.orig_sz,
                // transposed : self.transposed,
                win_sz : dims
            })
        } else {
            // println!("Requested offset : {:?}; Requested dims : {:?}; Original image size : {:?}", offset, dims, self.orig_sz);
            None
        }
    }

}

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


impl<'a, N> Window<'a, N>
where
    N : Scalar + Copy
{

    pub fn rows(&self) -> impl Iterator<Item=&[N]> + Clone {
        let stride = self.orig_sz.1;
        let tl = self.offset.0 * stride + self.offset.1;
        (0..self.win_sz.0).map(move |i| {
            let start = tl + i*stride;
            &self.win[start..(start+self.win_sz.1)]
        })
    }

}

impl<'a, N> Window<'a, N>
where
    N : Scalar + Mul<Output=N> + MulAssign + Copy + Copy + Serialize + DeserializeOwned + Any
{

    pub unsafe fn get_unchecked(&self, index : (usize, usize)) -> &N {
        let off_ix = (self.offset.0 + index.0, self.offset.1 + index.1);
        let (limit_row, limit_col) = (self.offset.0 + self.win_sz.0, self.offset.1 + self.win_sz.1);
        unsafe { self.win.get_unchecked(index::linear_index(off_ix, self.orig_sz.1)) }
    }

    /*pub unsafe fn get_unchecked_u16(&self, index : (u16, u16)) -> &N {
        let off_ix = (self.offset.0 + index.0, self.offset.1 + index.1);
        let (limit_row, limit_col) = (self.offset.0 + self.win_sz.0, self.offset.1 + self.win_sz.1);
        unsafe { self.win.get_unchecked(index::linear_index(off_ix, self.orig_sz.1)) }
    }*/

    // pub unsafe fn linear_index(&self, ix : usize) -> N {
    //    self.win.get_unchecked(ix)
    // }

    /// Returns a set of linear indices to the underlying slice that can
    /// be used to iterate over this window.
    pub fn linear_indices(&self, spacing : usize) -> Vec<usize> {
        unimplemented!()
    }

    pub fn subsampled_indices(&self, spacing : usize) -> Vec<(usize, usize)> {
        let mut ixs = Vec::new();
        for i in 0..(self.height() / spacing) {
            for j in 0..(self.width() / spacing) {
                ixs.push((i, j));
            }
        }
        ixs
    }

    pub fn as_ptr(&self) -> *const N {
        self.win.as_ptr()
    }

    pub fn slice_len(&self) -> usize {
        self.win.len()
    }

    /// Returns a minimum number of overlapping windows of the informed shape,
    /// that completely cover the current window.
    pub fn minimum_inner_windows(&'a self, shape : (usize, usize)) -> Vec<Window<'a, N>> {
        let mut sub_wins = Vec::new();

        sub_wins
    }

    pub fn far_thin_neighborhood(&'a self, center_tl : (usize, usize), win_sz : (usize, usize), dist : usize) -> Option<Neighborhood<'a, N>> {
        let outside_bounds = center_tl.0 < win_sz.0 + dist ||
            center_tl.0 > self.height().checked_sub(win_sz.0 + dist)? ||
            center_tl.1 > self.width().checked_sub(win_sz.1 + dist)? ||
            center_tl.1 < win_sz.1 + dist;
        if outside_bounds {
            return None;
        }
        Some(Neighborhood {
            center : self.sub_window(center_tl, win_sz)?,
            left : self.sub_window((center_tl.0, center_tl.1 - win_sz.1 - dist), win_sz)?,
            top : self.sub_window((center_tl.0 - win_sz.0 - dist, center_tl.1), win_sz)?,
            right : self.sub_window((center_tl.0, center_tl.1 + win_sz.1 + dist), win_sz)?,
            bottom : self.sub_window((center_tl.0 + win_sz.0 + dist, center_tl.1), win_sz)?
        })
    }

    pub fn thin_neighborhood(&'a self, center_tl : (usize, usize), win_sz : (usize, usize)) -> Option<Neighborhood<'a, N>> {
        let outside_bounds = center_tl.0 < win_sz.0 ||
            center_tl.0 > self.height().checked_sub(win_sz.0)? ||
            center_tl.1 > self.width().checked_sub(win_sz.1)? ||
            center_tl.1 < win_sz.1;
        if outside_bounds {
            return None;
        }
        Some(Neighborhood {
            center : self.sub_window(center_tl, win_sz)?,
            left : self.sub_window((center_tl.0, center_tl.1 - win_sz.1), win_sz)?,
            top : self.sub_window((center_tl.0 - win_sz.0, center_tl.1), win_sz)?,
            right : self.sub_window((center_tl.0, center_tl.1 + win_sz.1), win_sz)?,
            bottom : self.sub_window((center_tl.0 + win_sz.0, center_tl.1), win_sz)?
        })
    }

    pub fn sub_from_slice(
        src : &'a [N],
        full_ncols : usize,
        offset : (usize, usize),
        dims : (usize, usize)
    ) -> Option<Self> {
        let nrows = src.len() / full_ncols;
        if offset.0 + dims.0 <= nrows && offset.1 + dims.1 <= full_ncols {
            Some(Self {
                win : src,
                offset,
                orig_sz : (nrows, full_ncols),
                win_sz : dims
            })
        } else {
            None
        }
    }

    pub fn shrink_to_subsample(&'a self, by : usize) -> Option<Window<'a, N>> {
        let height = shrink_to_divisor(self.height(), by)?;
        let width = shrink_to_divisor(self.width(), by)?;
        self.sub_window((0, 0), (height, width))
    }

    /// Creates a window that cover the whole slice src, assuming it represents a square image.
    pub fn from_square_slice(src : &'a [N]) -> Option<Self> {
        Self::from_slice(src, (src.len() as f64).sqrt() as usize)
    }
    
    /*pub fn from_slice(
        src : &'a [N], 
        img_shape : (usize, usize), 
        offset : (usize, usize), 
        win_shape : (usize, usize), 
        transposed : bool
    ) {
    
    }*/
    
    /// Creates a window that cover the whole slice src. The slice is assumed to be in
    /// row-major order, but matrices are assumed to be 
    pub fn from_slice(src : &'a [N], ncols : usize) -> Option<Self> {
        if src.len() % ncols != 0 {
            return None;
        }
        let nrows = src.len() / ncols;
        Some(Self{
            // win : DMatrixSlice::from_slice_generic(src, Dynamic::new(nrows),
            // Dynamic::new(ncols)),
            win : src,
            offset : (0, 0),
            orig_sz : (nrows, ncols),
            win_sz : (nrows, ncols),
            // transposed : true
        })
    }
    
    pub fn shape(&self) -> (usize, usize) {
        // self.win.shape()
        self.win_sz
    }
    
    pub fn width(&self) -> usize {
        // self.win.ncols()
        self.win_sz.1
    }
    
    pub fn height(&self) -> usize {
        // self.win.nrows()
        self.win_sz.0
    }
    
    pub fn len(&self) -> usize {
        self.win_sz.0 * self.win_sz.1
    }

    /*pub fn linear_index(&self, ix : usize) -> &N {
        let offset = self.orig_sz.1 * offset.0 + offset.1;
        let row = ix / self.orig_sz.1;
        unsafe{ self.win.get_unchecked(offset + ix) }
    }*/

    /// Iterate over windows of the given size. This iterator consumes the original window
    /// so that we can implement windows(.) for Image by using move semantics, without
    /// requiring the user to call full_windows(.).
    pub fn windows(self, sz : (usize, usize)) -> impl Iterator<Item=Window<'a, N>> {
        let (step_v, step_h) = sz;
        if sz.0 >= self.win_sz.0 || sz.1 >= self.win_sz.1 {
            panic!("Child window size bigger than parent window size");
        }
        if self.height() % sz.0 != 0 || self.width() % sz.1 != 0 {
            panic!("Image size should be a multiple of window size (Required window {:?} over parent window {:?})", sz, self.win_sz);
        }
        let offset = self.offset;
        WindowIterator::<'a, N> {
            source : self,
            size : sz,
            curr_pos : offset,
            step_v,
            step_h
        }
    }
    
    pub fn row(&self, ix : usize) -> Option<&[N]> {
        if ix >= self.win_sz.0 {
            return None;
        }
        let stride = self.orig_sz.1;
        let tl = self.offset.0 * stride + self.offset.1;
        let start = tl + ix*stride;
        Some(&self.win[start..(start+self.win_sz.1)])
    }

    pub fn column(&'a self, ix : usize) -> Option<impl Iterator<Item=N> + 'a > {
        if ix < self.width() {
            Some(self.rows().map(move |row| row[ix] ))
        } else {
            None
        }
    }

    /// Iterates over pairs of pixels within a row, carrying the column index of the left element at first position
    pub fn horizontal_pixel_pairs(&'a self, row : usize, comp_dist : usize) -> Option<impl Iterator<Item=(usize, (&'a N, &'a N))>> {
        Some(iter::horizontal_row_iterator(self.row(row)?, comp_dist))
    }

    pub fn vertical_pixel_pairs(&'a self, col : usize, comp_dist : usize) -> Option<impl Iterator<Item=(usize, (&'a N, &'a N))>> {
        if col >= self.win_sz.1 {
            return None;
        }
        Some(iter::vertical_col_iterator(self.rows(), comp_dist, col))
    }

    /// Iterate over one of the lower-triangular diagonals the image, starting at given row.
    /// If to_right is passed, iterate from the top-left to bottom-right corner. If not, iterate from the
    /// top-right to bottom-left corner.
    pub fn lower_to_right_diagonal_pixel_pairs(
        &'a self,
        row : usize,
        comp_dist : usize,
    ) -> Option<impl Iterator<Item=((usize, usize), (&'a N, &'a N))>> {
        if row < self.height() {
            Some(iter::diagonal_right_row_iterator(self.rows(), comp_dist, (row, 0)))
        } else {
            None
        }
    }

    pub fn upper_to_right_diagonal_pixel_pairs(
        &'a self,
        col : usize,
        comp_dist : usize,
    ) -> Option<impl Iterator<Item=((usize, usize), (&'a N, &'a N))>> {
        if col < self.width() {
            Some(iter::diagonal_right_row_iterator(self.rows(), comp_dist, (0, col)))
        } else {
            None
        }
    }

    pub fn lower_to_left_diagonal_pixel_pairs(
        &'a self,
        row : usize,
        comp_dist : usize,
    ) -> Option<impl Iterator<Item=((usize, usize), (&'a N, &'a N))>> {
        if row < self.height() {
            Some(iter::diagonal_left_row_iterator(self.rows(), comp_dist, (row, self.width()-1)))
        } else {
            None
        }
    }

    pub fn upper_to_left_diagonal_pixel_pairs(
        &'a self,
        col : usize,
        comp_dist : usize,
    ) -> Option<impl Iterator<Item=((usize, usize), (&'a N, &'a N))>> {
        if col < self.width() {
            Some(iter::diagonal_left_row_iterator(self.rows(), comp_dist, (0, col)))
        } else {
            None
        }
    }

    pub fn to_right_diagonal_pixel_pairs(
        &'a self,
        comp_dist : usize
    ) -> impl Iterator<Item=((usize, usize), (&'a N, &'a N))> {
        (0..self.height()).step_by(comp_dist)
            .map(move |r| self.lower_to_right_diagonal_pixel_pairs(r, comp_dist).unwrap() )
            .flatten()
            .chain((0..self.width()).step_by(comp_dist)
                .map(move |c| self.upper_to_right_diagonal_pixel_pairs(c, comp_dist).unwrap() )
                .flatten()
            )
    }

    pub fn to_left_diagonal_pixel_pairs(
        &'a self,
        comp_dist : usize
    ) -> impl Iterator<Item=((usize, usize), (&'a N, &'a N))> {
        (0..self.height()).step_by(comp_dist)
            .map(move |r| self.lower_to_left_diagonal_pixel_pairs(r, comp_dist).unwrap() )
            .flatten()
            .chain((0..self.width()).step_by(comp_dist)
                .map(move |c| self.upper_to_left_diagonal_pixel_pairs(c, comp_dist).unwrap() )
                .flatten()
            )
    }

    /// Iterate over all image pixels if spacing=1; or over pixels spaced
    /// horizontally and verticallly by spacing. Iteration proceeds row-wise.
    pub fn pixels(&self, spacing : usize) -> impl Iterator<Item=&N> + Clone {
        assert!(spacing > 0, "Spacing should be at least one");
        assert!(self.width() % spacing == 0 && self.height() % spacing == 0, "Spacing should be integer divisor of width and height");
        iterate_row_wise(self.win, self.offset, self.win_sz, self.orig_sz, spacing).step_by(spacing)
    }

    pub fn rect_pixels(&'a self, rect : (usize, usize, usize, usize)) -> impl Iterator<Item=N> + Clone + 'a {
        let row_iter = rect.0..(rect.0 + rect.2);
        let col_iter = rect.1..(rect.1 + rect.3);

        col_iter.clone().map(move |c| self[(rect.0, c)] )
            .chain(row_iter.clone().map(move |r| self[(r, rect.1 + rect.3 - 1 )] ) )
            .chain(col_iter.clone().rev().map(move |c| self[(rect.0 + rect.2 - 1, c)] ) )
            .chain(row_iter.clone().rev().map(move |r| self[(r, rect.1 )] ))
    }

    /// Iterate over image pixels, expanding from a given location, until any image border is found.
    /// Iteration happens clock-wise from the seed pixel. Indices are at the original image scale.
    pub fn expanding_pixels(
        &self,
        seed : (usize, usize),
        px_spacing : usize
    ) -> impl Iterator<Item=((usize, usize), &N)> + Clone {
        assert!(seed.0 < self.height() && seed.1 < self.width());
        let min_dist = seed.0.min(self.height() - seed.0).min(seed.1).min(self.width() - seed.1);
        (px_spacing..min_dist).map(move |abs_dist| {
            let left_col = seed.1 - abs_dist;
            let top_row = seed.0 - abs_dist;
            let right_col = seed.1 + abs_dist;
            let bottom_row = seed.0 + abs_dist;
            let row_range = (seed.0.saturating_sub(abs_dist-px_spacing)..((seed.0+abs_dist).min(self.height()))).step_by(px_spacing);
            let col_range = (seed.1.saturating_sub(abs_dist)..((seed.1+abs_dist+px_spacing).min(self.width()))).step_by(px_spacing);
            let top_iter = col_range.clone().map(move |c| ((top_row, c), &self[(top_row, c)] ) );
            let right_iter = row_range.clone().map(move |r| ((r, right_col), &self[(r, right_col)] ) );
            let bottom_iter = col_range.rev().map(move |c| ((bottom_row, c), &self[(bottom_row, c)] ) );
            let left_iter = row_range.rev().map(move |r| ((r, left_col), &self[(r, left_col)] ) );

                // Skip first element of each iterator because it is already contained in the last one.
                top_iter.chain(right_iter.skip(1)).chain(bottom_iter.skip(1)).chain(left_iter.skip(1))
        }).flatten()
    }

    // Returns the most representative k-colors of the image using K-means
    // pub fn colors(&self, px_spacing : usize, n_colors : usize) -> Vec<u8> {
    // }

    // Returns a thin vertical window over a given col. Prototype for impl Index<(Range<usize>, usize)>
    pub fn sub_col(&self, rows : Range<usize>, col : usize) -> Option<Window<'_, N>> {
        let height = rows.end - rows.start;
        self.sub_window((rows.start, col), (height, 1))
    }

    // Returns a thin horizontal window over a given row. Prototype for impl Index<(usize, Range<usize>)>
    pub fn sub_row(&self, row : usize, cols : Range<usize>) -> Option<Window<'_, N>> {
        let width = cols.end - cols.start;
        self.sub_window((row, cols.start), (1, width))
    }

    pub fn clone_owned(&self) -> Image<N>
    where
        N : Copy + Default + Zero
    {
        let mut buf = Vec::new();
        self.rows().for_each(|row| buf.extend(row.iter().cloned()) );
        Image::from_vec(buf, self.win_sz.1)
    }

    /*pub fn row_slices(&'a self) -> Vec<&'a [N]> {
        let mut rows = Vec::new();
        for r in (self.offset.0)..(self.offset.0+self.win_sz.0) {
            let begin = self.win_sz.1*r + self.offset.1;
            rows.push(&self.src[begin..begin+self.win_sz.1]);
        }
        rows
    }*/
    
}

/*pub struct PackedIterator<'a, T>
where
    T : Scalar + Debug + Copy
{
    win : Window<'a, T>,
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
            let row = self.win.row(self.ix / self.n_packed).unwrap();
            let packed = &row[self.ix % self.n_packed];
            self.ix += 1;
            Some(AutoSimd::try_from(packed).unwrap())
        }
    }
}*/

pub fn labels<L, E>((height, width) : (usize, usize), px_spacing : usize) -> impl Iterator<Item=(L, L)> + Clone
where
    L : TryFrom<usize, Error=E> + Div<Output=L> + Mul<Output=L> + Rem<Output=L> + Clone + Copy + 'static,
    E : Debug,
    Range<L> : Iterator<Item=L>
{
    let spacing = L::try_from(px_spacing).unwrap();
    let w = (L::try_from(width).unwrap() / spacing );
    let h = (L::try_from(height).unwrap() / spacing );
    let range = Range { start : L::try_from(0usize).unwrap(), end : (w*h) };
    range.map(move |ix| (ix / w, ix % w) )
}

impl<'a> Window<'a, u8> {

    #[cfg(feature="opencv")]
    pub fn resize_mut(&self, other : WindowMut<'_, u8>) {
        use opencv::{core, imgproc};
        use opencv::prelude::MatTraitManual;
        let this_shape = self.shape();
        let other_shape = other.shape();
        let src : core::Mat = self.into();
        let mut dst : core::Mat = other.into();
        let dst_sz = dst.size().unwrap();

        println!("{:?} {:?} {:?} {:?}", this_shape, other_shape, src.size().unwrap(), dst_sz);
        imgproc::resize(&src, &mut dst, dst_sz, 0.0, 0.0, imgproc::INTER_NEAREST);
    }

    #[cfg(feature="opencv")]
    pub fn copy_scale_mut_within(&self, src : (usize, usize), src_sz : (usize, usize), dst : (usize, usize), dst_sz : (usize, usize)) {

        use opencv::{core, imgproc};

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
    pub fn nearest_matching(&self, pos : (usize, usize), px_spacing : usize, f : impl Fn(u8)->bool) -> Option<(usize, usize)> {
        if f(self[pos]) {
            Some(pos)
        } else {
            self.expanding_pixels(pos, px_spacing).find(|(_, px)| f(**px) ).map(|(pos, _)| pos )
        }
    }

    pub fn color_at_labels(&'a self, ixs : impl Iterator<Item=(usize, usize)> + 'a) -> impl Iterator<Item=u8> + 'a {
        ixs.map(move |ix| self[ix] )
    }

    pub fn nonzero_pixels(&'a self, px_spacing : usize) -> impl Iterator<Item=&'a u8> +'a + Clone {
        self.pixels(px_spacing).filter(|px| **px > 0 )
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

    /// Returns iterator over (subsampled row index, subsampled col index, pixel color).
    /// Panics if L is an unsigend integer type that cannot represent one of the dimensions
    /// of the image precisely.
    pub fn labeled_pixels<L, E>(&'a self, px_spacing : usize) -> impl Iterator<Item=(L, L, u8)> +'a + Clone
    where
        L : TryFrom<usize, Error=E> + Div<Output=L> + Mul<Output=L> + Rem<Output=L> + Clone + Copy + 'static,
        E : Debug,
        Range<L> : Iterator<Item=L>
    {
        let spacing = L::try_from(px_spacing).unwrap();
        let w = (L::try_from(self.width()).unwrap() / spacing );
        let h = (L::try_from(self.height()).unwrap() / spacing );
        let range = Range { start : L::try_from(0usize).unwrap(), end : (w*h) };
        range
            .zip(self.pixels(px_spacing))
            .map(move |(ix, px)| {
                let (r, c) = (ix / w, ix % w);
                (r, c, *px)
            })
    }

    /// Extract contiguous image regions of homogeneous color.
    pub fn patches(&self, px_spacing : usize) -> Vec<Patch> {
        /*let mut patches = Vec::new();
        color::full_color_patches(&mut patches, self, px_spacing as u16, ColorMode::Exact(0), color::ExpansionMode::Dense);
        patches*/
        unimplemented!()
    }

    /*pub fn binary_patches(&self, px_spacing : usize) -> Vec<BinaryPatch> {
        // TODO if we have a binary or a bit image with just a few classes,
        // there is no need for KMeans. Just use the allocations.
        // let label_img = segmentation::segment_colors_to_image(self, px_spacing, n_colors);
        color::binary_patches(self, px_spacing)
    }*/

    pub fn mean(&self, n_pxs : usize) -> Option<u8> {
        Some((self.shrink_to_subsample(n_pxs)?.pixels(n_pxs).map(|px| *px as u64 ).sum::<u64>() / (self.width() * self.height()) as u64) as u8)
    }

    /// If higher, returns binary image with all pixels > thresh set to 255 and others set to 0;
    /// If !higher, returns binary image with pixels < thresh set to 255 and others set to 0.
    pub fn threshold_mut(&self, dst : &mut Image<u8>, thresh : u8, higher : bool) {
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
            win : self.clone(),
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

impl<N> Index<(usize, usize)> for Window<'_, N>
where
    N : Scalar
{

    type Output = N;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let off_ix = (self.offset.0 + index.0, self.offset.1 + index.1);
        let (limit_row, limit_col) = (self.offset.0 + self.win_sz.0, self.offset.1 + self.win_sz.1);
        if off_ix.0 < limit_row && off_ix.1 < limit_col {
            unsafe { self.win.get_unchecked(index::linear_index(off_ix, self.orig_sz.1)) }
        } else {
            panic!("Invalid window index: {:?}", index);
        }
    }
}

impl<N> Index<(u16, u16)> for Window<'_, N>
where
    N : Scalar
{

    type Output = N;

    fn index(&self, index: (u16, u16)) -> &Self::Output {
        self.index((index.0 as usize, index.1 as usize))
    }

}

/*impl<N> Index<(Range<usize>, usize)> for Window<'_, N>
where
    N : Scalar
{

    type Output = Self;

    fn index(&self, index: (Range<usize>, usize)) -> &Self::Output {

    }

}*/

pub fn iterate_row_wise<N>(
    src : &[N], 
    offset : (usize, usize), 
    win_sz : (usize, usize), 
    orig_sz : (usize, usize),
    row_spacing : usize
) -> impl Iterator<Item=&N> + Clone {
    let start = orig_sz.1 * offset.0 + offset.1;
    (0..win_sz.0).step_by(row_spacing).map(move |i| unsafe {
        let row_offset = start + i*orig_sz.1;
        src.get_unchecked(row_offset..(row_offset+win_sz.1))
    }).flatten()
}

trait BorrowedRegion {

    type Slice;

    fn create(offset : (usize, usize), win_sz : (usize, usize), orig_sz : (usize, usize), win : Self::Slice) -> Self;

    fn offset(&self) -> &(usize, usize);

    fn size(&self) -> &(usize, usize);

    fn original_size(&self) -> &(usize, usize);

    // This takes the implementor because for mutable windows,
    // the only way to safely give the slice is to let go of the
    // current object.
    unsafe fn original_slice(&mut self) -> Self::Slice;

}

impl<'a, T> BorrowedRegion for Window<'a, T>
where
    T : Scalar + Copy
{

    type Slice = &'a [T];

    fn create(offset : (usize, usize), win_sz : (usize, usize), orig_sz : (usize, usize), win : Self::Slice) -> Self {
        Window { offset, win_sz, orig_sz, win }
    }

    fn offset(&self) -> &(usize, usize) {
        &self.offset
    }

    fn size(&self) -> &(usize, usize) {
        &self.win_sz
    }

    fn original_size(&self) -> &(usize, usize) {
        &self.orig_sz
    }

    unsafe fn original_slice(&mut self) -> Self::Slice {
        self.win
    }

}

impl<'a, T> BorrowedRegion for WindowMut<'a, T>
where
    T : Scalar + Copy //+ Serialize + DeserializeOwned + Any + Zero + From<u8>
{

    type Slice = &'a mut [T];

    fn create(offset : (usize, usize), win_sz : (usize, usize), orig_sz : (usize, usize), win : Self::Slice) -> Self {
        WindowMut { offset, win_sz, orig_sz, win }
    }

    fn offset(&self) -> &(usize, usize) {
        &self.offset
    }

    fn size(&self) -> &(usize, usize) {
        &self.win_sz
    }

    fn original_size(&self) -> &(usize, usize) {
        &self.orig_sz
    }

    unsafe fn original_slice(&mut self) -> Self::Slice {
        std::slice::from_raw_parts_mut(self.win.as_mut_ptr(), self.win.len())
    }

}

pub struct WindowIterator<'a, N>
where
    N : Scalar,
{
    source : Window<'a, N>,
    
    // This child window size
    size : (usize, usize),

    // Index the most ancestral window possible.
    curr_pos : (usize, usize),

    /// Vertical increment. Either U1 or Dynamic.
    step_v : usize,

    /// Horizontal increment. Either U1 or Dynamic.
    step_h : usize,

}

impl<'a, N> Iterator for WindowIterator<'a, N>
where
    N : Scalar + Copy //+ Clone + Copy + Serialize + Zero + From<u8>
{

    type Item = Window<'a, N>;

    fn next(&mut self) -> Option<Self::Item> {
        /*let within_horiz = self.curr_pos.0  + self.size.0 <= (self.source.offset.0 + self.source.win_sz.0);
        let within_vert = self.curr_pos.1 + self.size.1 <= (self.source.offset.1 + self.source.win_sz.1);
        let within_bounds = within_horiz && within_vert;
        let win = if within_bounds {
            Some(Window { 
                offset : self.curr_pos,
                win_sz : self.size,
                orig_sz : self.source.orig_sz,
                win : &self.source.win
            })
        } else {
            None
        };
        self.curr_pos.1 += self.step_h;
        if self.curr_pos.1 + self.size.1 > (self.source.offset.1 + self.source.win_sz.1) {
            self.curr_pos.1 = self.source.offset.1;
            self.curr_pos.0 += self.step_v;
        }
        win*/
        iterate_windows(&mut self.source, &mut self.curr_pos, self.size, (self.step_h, self.step_v))
    }

}

pub struct WindowIteratorMut<'a, N>
where
    N : Scalar + Copy,
{

    source : WindowMut<'a, N>,

    // This child window size
    size : (usize, usize),

    // Index the most ancestral window possible.
    curr_pos : (usize, usize),

    /// Vertical increment. Either U1 or Dynamic.
    step_v : usize,

    /// Horizontal increment. Either U1 or Dynamic.
    step_h : usize,

}

impl<'a, N> Iterator for WindowIteratorMut<'a, N>
where
    N : Scalar + Copy
{

    type Item = WindowMut<'a, N>;

    fn next(&mut self) -> Option<Self::Item> {
        iterate_windows(&mut self.source, &mut self.curr_pos, self.size, (self.step_h, self.step_v))
    }

}

fn iterate_windows<R, S>(
    s : &mut R,
    curr_pos : &mut (usize, usize),
    size : (usize, usize),
    (step_h, step_v) : (usize, usize)
) -> Option<R>
where
    R : BorrowedRegion<Slice=S>
{
    let (offset, win_sz) = (*s.offset(), *s.size());
    let within_horiz = curr_pos.0  + size.0 <= (offset.0 + win_sz.0);
    let within_vert = curr_pos.1 + size.1 <= (offset.1 + win_sz.1);
    let within_bounds = within_horiz && within_vert;
    let win = if within_bounds {
        Some(BorrowedRegion::create(*curr_pos, size, *s.original_size(), unsafe { s.original_slice() }))
    } else {
        None
    };
    curr_pos.1 += step_h;
    if curr_pos.1 + size.1 > (offset.1 + win_sz.1) {
        curr_pos.1 = offset.1;
        curr_pos.0 += step_v;
    }
    win
}

#[cfg(feature="opencv")]
impl<N> opencv::core::ToInputArray for Image<N>
where
    N : Scalar + Copy + Default + Zero + Serialize + DeserializeOwned + Any
{

    fn input_array(&self) -> opencv::Result<opencv::core::_InputArray> {
        let out : opencv::core::Mat = (&*self).into();
        out.input_array()
    }

}

#[cfg(feature="opencv")]
impl<N> opencv::core::ToOutputArray for Image<N>
where
    N : Scalar + Copy + Default + Zero + Serialize + DeserializeOwned + Any
{

    fn output_array(&mut self) -> opencv::Result<opencv::core::_OutputArray> {
        let mut out : opencv::core::Mat = (&mut *self).into();
        out.output_array()
    }

}

#[cfg(feature="opencv")]
impl<N> opencv::core::ToInputArray for Window<'_, N>
where
    N : Scalar + Copy + Default + Zero + Serialize + DeserializeOwned + Any
{

    fn input_array(&self) -> opencv::Result<opencv::core::_InputArray> {
        let out : opencv::core::Mat = (self.clone()).into();
        out.input_array()
    }

}

#[cfg(feature="opencv")]
impl<N> opencv::core::ToOutputArray for WindowMut<'_, N>
where
    N : Scalar + Copy + Default + Zero + Serialize + DeserializeOwned + Any
{

    fn output_array(&mut self) -> opencv::Result<opencv::core::_OutputArray> {
        let mut out : opencv::core::Mat = (self).into();
        out.output_array()
    }

}

#[cfg(feature="opencv")]
pub fn median_blur(win : &Window<'_, u8>, output : WindowMut<'_, u8>, kernel : usize) {

    use opencv::{imgproc, core};

    let input : core::Mat = win.clone().into();
    let mut out : core::Mat = output.into();
    imgproc::median_blur(&input, &mut out, kernel as i32).unwrap();

}

#[cfg(feature="opencv")]
impl<N> From<core::Mat> for Image<N>
where
    N : Scalar + Copy + Default + Zero + Serialize + DeserializeOwned + Any + opencv::core::DataType
{

    fn from(m : core::Mat) -> Image<N> {

        use opencv::prelude::MatTraitManual;
        use opencv::prelude::MatTrait;
        // assert!(m.is_contiguous().unwrap());

        let sz = m.size().unwrap();
        let h = sz.height as usize;
        let w = sz.width as usize;
        let mut img = Image::<N>::new_constant(h, w, N::zero());
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
impl<N> Into<core::Mat> for &Image<N>
where
    N : Scalar + Copy + Default + Zero + Serialize + DeserializeOwned + Any
{

    fn into(self) -> core::Mat {
        let sub_slice = None;
        let stride = self.ncols;
        unsafe{ cvutils::slice_to_mat(&self.buf[..], stride, sub_slice) }
        // self.full_window().into()
    }
}

#[cfg(feature="opencv")]
impl<N> Into<core::Mat> for &mut Image<N>
where
    N : Scalar + Copy + Default + Zero + Serialize + DeserializeOwned + Any
{

    fn into(self) -> core::Mat {
        let sub_slice = None;
        let stride = self.ncols;
        unsafe{ cvutils::slice_to_mat(&self.buf[..], stride, sub_slice) }
        // self.full_window_mut().into()
    }
}

/// TODO mark as unsafe impl
#[cfg(feature="opencv")]
impl<N> Into<core::Mat> for Window<'_, N>
where
    N : Scalar + Copy + Default
{

    fn into(self) -> core::Mat {
        let sub_slice = Some((self.offset, self.win_sz));
        let stride = self.orig_sz.1;
        unsafe{ cvutils::slice_to_mat(self.win, stride, sub_slice) }
    }
}

#[cfg(feature="opencv")]
impl<N> Into<core::Mat> for &Window<'_, N>
where
    N : Scalar + Copy + Default
{

    fn into(self) -> core::Mat {
        let sub_slice = Some((self.offset, self.win_sz));
        let stride = self.orig_sz.1;
        unsafe{ cvutils::slice_to_mat(self.win, stride, sub_slice) }
    }
}

/// TODO mark as unsafe impl
#[cfg(feature="opencv")]
impl<N> Into<core::Mat> for WindowMut<'_, N>
where
    N : Scalar + Copy + Default
{

    fn into(self) -> core::Mat {
        let sub_slice = Some((self.offset, self.win_sz));
        let stride = self.orig_sz.1;
        unsafe{ cvutils::slice_to_mat(self.win, stride, sub_slice) }
    }
}

#[cfg(feature="opencv")]
impl<N> Into<core::Mat> for &mut WindowMut<'_, N>
where
    N : Scalar + Copy + Default
{

    fn into(self) -> core::Mat {
        let sub_slice = Some((self.offset, self.win_sz));
        let stride = self.orig_sz.1;
        unsafe{ cvutils::slice_to_mat(self.win, stride, sub_slice) }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Mark {

    // Position, square lenght and color
    Cross((usize, usize), usize, u8),
    
    // Position, square lenght and color
    Corner((usize, usize), usize, u8),
    
    // Start and end positions and color
    Line((usize, usize), (usize, usize), u8),
    
    // Position, digit value, digit size and color
    Digit((usize, usize), usize, usize, u8),

    // Position, label, digit value, size and color
    // Label((usize, usize), &'static str, usize, u8),

    // Center, radius and color
    Circle((usize, usize), usize, u8),

    /// A dense circle
    Dot((usize, usize), usize, u8),

    /// TL pos, size and color
    Rect((usize, usize), (usize, usize), u8),

    /// Arbitrary shape
    Shape(Vec<(usize, usize)>, u8),

    Text((usize, usize), String, u8),

    Arrow((usize, usize), (usize, usize), usize, u8)
    
}

/*impl TryFrom<rhai::Dynamic> for Mark {

    type Err = ();

    fn try_from(d : rhai::Dynamic) -> Result<Self, ()> {

    }

}*/

#[derive(Debug)]
pub struct WindowMut<'a, N> 
where
    N : Scalar + Copy
{
    // Window offset, with respect to the top-left point (row, col).
    offset : (usize, usize),
    
    // Original image size.
    orig_sz : (usize, usize),
    
    // This window size.
    win_sz : (usize, usize),
    
    // Original image full slice.
    win : &'a mut [N],
}

impl<'a, N> WindowMut<'a, N>
where
    N : Scalar + Copy + Debug
{

    pub unsafe fn from_ptr(ptr : *mut N, len : usize, full_ncols : usize, offset : (usize, usize), sz : (usize, usize)) -> Option<Self> {
        let s = std::slice::from_raw_parts_mut(ptr, len);
        Self::sub_from_slice(s, full_ncols, offset, sz)
    }

    pub fn from_slice(src : &'a mut [N], ncols : usize) -> Option<Self> {
        if src.len() % ncols != 0 {
            return None;
        }
        /*let nrows = src.len() / ncols;
        Self {
            win : DMatrixSliceMut::from_slice_generic(src, Dynamic::new(nrows),
            Dynamic::new(ncols)),
            offset : (0, 0),
            orig_sz : (nrows, ncols)
        }*/
        let nrows = src.len() / ncols;
        Some(Self{
            // win : DMatrixSlice::from_slice_generic(src, Dynamic::new(nrows),
            // Dynamic::new(ncols)),
            win : src,
            offset : (0, 0),
            orig_sz : (nrows, ncols),
            win_sz : (nrows, ncols),
        })
    }

    pub fn sub_from_slice(
        src : &'a mut [N],
        full_ncols : usize,
        offset : (usize, usize),
        dims : (usize, usize)
    ) -> Option<Self> {
        let nrows = src.len() / full_ncols;
        if offset.0 + dims.0 <= nrows && offset.1 + dims.1 <= full_ncols {
            Some(Self {
                win : src,
                offset,
                orig_sz : (nrows, full_ncols),
                win_sz : dims
            })
        } else {
            None
        }
    }

    /// We might just as well make this take self by value, since the mutable reference to self will be
    /// invalidated by the borrow checker when we have the child.
    pub fn sub_window_mut(&'a mut self, offset : (usize, usize), dims : (usize, usize)) -> Option<WindowMut<'a, N>> {
        let new_offset = (self.offset.0 + offset.0, self.offset.1 + offset.1);
        if new_offset.0 + dims.0 <= self.orig_sz.0 && new_offset.1 + dims.1 <= self.orig_sz.1 {
            Some(Self {
                win : self.win,
                offset : (self.offset.0 + offset.0, self.offset.1 + offset.1),
                orig_sz : self.orig_sz,
                win_sz : dims
            })
        } else {
            None
        }
    }

    /// Creates a window that cover the whole slice src, assuming it represents a square image.
    pub fn from_square_slice(src : &'a mut [N]) -> Option<Self> {
        Self::from_slice(src, (src.len() as f64).sqrt() as usize)
    }

    pub fn windows_mut(mut self, sz : (usize, usize)) -> impl Iterator<Item=WindowMut<'a, N>>
    where
        N : Mul<Output=N> + MulAssign
    {
        let (step_v, step_h) = sz;
        if sz.0 >= self.win_sz.0 || sz.1 >= self.win_sz.1 {
            panic!("Child window size bigger than parent window size");
        }
        if self.height() % sz.0 != 0 || self.width() % sz.1 != 0 {
            panic!("Image size should be a multiple of window size (Required window {:?} over parent window {:?})", sz, self.win_sz);
        }
        let offset = self.offset;
        WindowIteratorMut::<'a, N> {
            source : self,
            size : sz,
            curr_pos : offset,
            step_v,
            step_h
        }
    }

}

impl<'a> WindowMut<'a, u8> {

    // TODO also implement contrast_adjust_mut
    pub fn brightness_adjust_mut(&'a mut self, k : i16) {

        assert!(k <= 255 && k >= -255);
        let abs_k = k.abs() as u8;
        if k > 0 {
            self.pixels_mut(1).for_each(|px| *px = px.saturating_add(abs_k) );
        } else {
            self.pixels_mut(1).for_each(|px| *px = px.saturating_sub(abs_k) );
        }
    }

}

impl WindowMut<'_, u8> {

    /// Gamma-corrects, i.e. multiplies input by input^(1/gamma) and normalize.
    pub fn gamma_correct_inplace(&mut self, gamma : f32) {

        // Using gamma < 1.0 avoids saturating the image. Perhaps offer a version that
        // does just that, without normalization.
        assert!(gamma <= 1.0);
        for i in 0..self.win_sz.0 {
            for j in 0..self.win_sz.1 {
                self[(i,j)] = (self[(i,j)] as f32).powf(1. / gamma).max(0.0).min(255.0) as u8
            }
        }
    }

    /// For any pixel >= color, set it and its neighborhood to erase_color.
    pub fn erase_speckles(&mut self, color : u8, neigh : usize, erase_color: u8) {
        for i in 0..self.win_sz.0 {
            for j in 0..self.win_sz.1 {
                if self[(i, j)] >= color {
                    for ni in i.saturating_sub(neigh)..((i + neigh).min(self.win_sz.0-1)) {
                        for nj in j.saturating_sub(neigh)..((j + neigh).min(self.win_sz.1-1)) {
                            self[(ni, nj)] = erase_color;
                        }
                    }
                }
            }
        }
    }

    pub fn fill_with_byte<'a>(&'a mut self, byte : u8) {
        // self.rows_mut().for_each(|row| std::ptr::write_bytes(&mut row[0] as *mut _, byte, row.len()) );
        /*for ix in 0..self.win_sz.0 {
            let row = self.row_mut(ix);
            std::ptr::write_bytes(&mut row[0] as *mut _, byte, row.len());
        }*/
    }

    pub fn patches(&self, px_spacing : usize) -> Vec<Patch> {
        /*let src_win = unsafe {
            Window {
                offset : (self.offset.0, self.offset.1),
                orig_sz : self.orig_sz,
                win_sz : self.win_sz,
                win : std::slice::from_raw_parts(self.win.as_ptr(), self.win.len()),
            }
        };
        let mut patches = Vec::new();
        color::full_color_patches(&mut patches, &src_win, px_spacing as u16, ColorMode::Exact(0), color::ExpansionMode::Dense);
        patches*/
        unimplemented!()
    }

    pub fn draw(&mut self, mark : Mark) {
        /*let slice_ptr = self.win.data.as_mut_slice().as_mut_ptr();
        let ptr_offset = slice_ptr as u64 - (self.orig_sz.0*(self.offset.1 - 1)) as u64 - self.offset.0 as u64;
        let orig_ptr = ptr_offset as *mut u8;
        let orig_slice = unsafe { std::slice::from_raw_parts_mut(orig_ptr, self.orig_sz.0 * self.orig_sz.1) };*/
        match mark {
            Mark::Cross(pos, sz, col) => {
                let cross_pos = (self.offset.0 + pos.0, self.offset.1 + pos.1);
                draw::draw_cross(
                    self.win,
                    self.orig_sz,
                    cross_pos,
                    col,
                    sz
                );
            },
            Mark::Corner(pos, sz, col) => {
                let center_pos = (self.offset.0 + pos.0, self.offset.1 + pos.1);
                draw::draw_corners(
                    self.win,
                    self.orig_sz,
                    center_pos,
                    col,
                    sz
                );
            },
            Mark::Line(src, dst, color) => {
                let src_pos = (self.offset.0 + src.0, self.offset.1 + src.1);
                let dst_pos = (self.offset.0 + dst.0, self.offset.1 + dst.1);
                
                #[cfg(feature="opencv")]
                unsafe {
                    cvutils::draw_line(self.win, self.orig_sz.1, src_pos, dst_pos, color);
                    return;
                }
                
                draw::draw_line(
                    self.win,
                    self.orig_sz,
                    src_pos,
                    dst_pos,
                    color
                );
            },
            Mark::Rect(tl, sz, color) => {
                let tr = (tl.0, tl.1 + sz.1);
                let br = (tl.0 + sz.0, tl.1 + sz.1);
                let bl = (tl.0 + sz.0, tl.1);
                self.draw(Mark::Line(tl, tr, color));
                self.draw(Mark::Line(tr, br, color));
                self.draw(Mark::Line(br, bl, color));
                self.draw(Mark::Line(bl, tl, color));
            },
            Mark::Digit(pos, val, sz, color) => {
                let tl_pos = (self.offset.0 + pos.0, self.offset.1 + pos.1);

                #[cfg(feature="opencv")]
                unsafe {
                    cvutils::write_text(self.win, self.orig_sz.1, tl_pos, &val.to_string()[..], color);
                    return;
                }
                
                draw::draw_digit_native(self.win, self.orig_sz.1, tl_pos, val, sz, color);
            },
            /*Mark::Label(pos, msg, sz, color) => {
                let tl_pos = (self.offset.0 + pos.0, self.offset.1 + pos.1);

                #[cfg(feature="opencv")]
                unsafe {
                    cvutils::write_text(self.win, self.orig_sz.1, tl_pos, msg, color);
                    return;
                }

                panic!("Label draw require 'opencv' feature");
            },*/
            Mark::Circle(pos, radius, color) => {
                let center_pos = (self.offset.0 + pos.0, self.offset.1 + pos.1);

                #[cfg(feature="opencv")]
                unsafe {
                    cvutils::draw_circle(self.win, self.orig_sz.1, center_pos, radius, color);
                    return;
                }

                panic!("Circle draw require 'opencv' feature");
            },
            Mark::Dot(pos, radius, color) => {
                for i in 0..self.height() {
                    for j in 0..self.width() {
                        if crate::feature::shape::point_euclidian((i, j), pos) <= radius as f32 {
                            self[(i, j)] = color;
                        }
                    }
                }
            },
            Mark::Shape(pts, col) => {
                let n = pts.len();
                if n < 2 {
                    return;
                }
                for (p1, p2) in pts.iter().take(n-1).zip(pts.iter().skip(1)) {
                    self.draw(Mark::Line(*p1, *p2, col));
                }

                // Close point?
                // if crate::feature::shape::point_euclidian(pts[0], pts[pts.len()-1]) < 32.0 {
                //    self.draw(Mark::Line(pts[0], pts[pts.len()-1], col));
                // }
            },
            Mark::Text(tl_pos, txt, color) => {

                #[cfg(feature="opencv")]
                {
                    unsafe { 
                        cvutils::write_text(
                            self.win, 
                            self.orig_sz.1, 
                            (self.offset.0 + tl_pos.0, self.offset.1 + tl_pos.1),
                            &txt[..], 
                            color
                        ); 
                    }
                    return;
                }

                println!("Warning: Text drawing require opencv feature");
            },
            Mark::Arrow(from, to, thickness, color) => {

                #[cfg(feature="opencv")]
                {
                    let mut out : opencv::core::Mat = self.into();
                    opencv::imgproc::arrowed_line(
                        &mut out,
                        opencv::core::Point2i::new(from.1 as i32, from.0 as i32),
                        opencv::core::Point2i::new(to.1 as i32, to.0 as i32),
                        opencv::core::Scalar::from(color as f64),
                        thickness as i32,
                        opencv::imgproc::LINE_8,
                        0,
                        0.1
                    );
                    return;
                }

                println!("Warning: Arrow drawing require opencv feature");
            }
        }
    }

}

pub(crate) unsafe fn create_immutable<'a>(win : &'a WindowMut<'a, u8>) -> Window<'a, u8> {
    unsafe {
        Window {
            offset : (win.offset.0, win.offset.1),
            orig_sz : win.orig_sz,
            win_sz : win.win_sz,
            win : std::slice::from_raw_parts(win.win.as_ptr(), win.win.len()),
        }
    }
}

/*impl<N> WindowMut<'_, N>
where
    N : Scalar + Copy
{

    unsafe fn create_immutable_without_lifetime(&self, src : (usize, usize), dim : (usize, usize)) -> Window<'_, N> {
        Window {
            offset : (self.offset.0 + src.0, self.offset.1 + src.1),
            orig_sz : self.orig_sz,
            win_sz : dim,
            win : std::slice::from_raw_parts(self.win.as_ptr(), self.win.len()),
        }
    }

    /// Creates a new mutable window, forgetting the lifetime of the previous window.
    unsafe fn create_mutable_without_lifetime(&mut self, src : (usize, usize), dim : (usize, usize)) -> WindowMut<'_, N> {
        WindowMut {
            offset : (self.offset.0 + src.0, self.offset.1 + src.1),
            orig_sz : self.orig_sz,
            win_sz : dim,
            win : std::slice::from_raw_parts_mut(self.win.as_mut_ptr(), self.win.len()),
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

impl<'a, N> WindowMut<'a, N>
where
    N : Scalar + Copy
{

    pub fn shape(&self) -> (usize, usize) {
        self.win_sz
    }

    pub fn orig_sz(&self) -> (usize, usize) {
        self.orig_sz
    }

    pub fn full_slice(&'a self) -> &'a [N] {
        &self.win[..]
    }

    pub fn offset(&self) -> (usize, usize) {
        self.offset
    }

    pub fn width(&self) -> usize {
        self.shape().1
    }

    pub fn height(&self) -> usize {
        self.shape().0
    }

}

impl<'a, N> WindowMut<'a, N>
where
    N : Scalar + Copy
{

    pub fn rows_mut(&'a mut self) -> impl Iterator<Item=&'a mut [N]> {
        let stride = self.orig_sz.1;
        let tl = self.offset.0 * stride + self.offset.1;
        (0..self.win_sz.0).map(move |i| {
            let start = tl + i*stride;
            let slice = &self.win[start..(start+self.win_sz.1)];
            unsafe { std::slice::from_raw_parts_mut(slice.as_ptr() as *mut _, slice.len()) }
        })
    }

}

impl<'a, N> WindowMut<'a, N>
where
    N : Scalar + Copy + Mul<Output=N> + MulAssign + PartialOrd + Serialize + DeserializeOwned
{

    pub fn fill(&'a mut self, color : N) {
        self.pixels_mut(1).for_each(|px| *px = color );
    }

    pub fn paint(&'a mut self, min : N, max : N, color : N) {
        self.pixels_mut(1).for_each(|px| if *px >= min && *px <= max { *px = color; } );
    }

    pub fn pixels_mut(&'a mut self, spacing : usize) -> impl Iterator<Item=&'a mut N> {
        self.rows_mut().step_by(spacing).map(move |r| r.iter_mut().step_by(spacing) ).flatten()
    }

    pub fn labeled_pixels_mut(&'a mut self, spacing : usize) -> impl Iterator<Item=(usize, usize, &'a mut N)> +'a {
        let w = self.width();
        self.pixels_mut(spacing)
            .enumerate()
            .map(move |(ix, px)| {
                let (r, c) = (ix / w, ix % w);
                (r, c, px)
            })
    }

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

    /// Analogous to slice::copy_within, copy a the sub_window (src, dim) to the sub_window (dst, dim).
    pub fn copy_within(&'a mut self, src : (usize, usize), dst : (usize, usize), dim : (usize, usize)) {
        use crate::feature::shape;
        assert!(!shape::rect_overlaps(&(src.0, src.1, dim.0, dim.1), &(dst.0, dst.1, dim.0, dim.1)), "copy_within: Windows overlap");
        let src_win = unsafe {
            Window {
                offset : (self.offset.0 + src.0, self.offset.1 + src.1),
                orig_sz : self.orig_sz,
                win_sz : dim,
                win : std::slice::from_raw_parts(self.win.as_ptr(), self.win.len()),
            }
        };
        self.sub_window_mut(dst, dim).unwrap().copy_from(&src_win);
    }

    pub fn row_mut(&'a mut self, i : usize) -> &'a mut [N] {
        let stride = self.orig_sz.1;
        let tl = self.offset.0 * stride + self.offset.1;
        let start = tl + i*stride;
        let slice = &self.win[start..(start+self.win_sz.1)];
        unsafe { std::slice::from_raw_parts_mut(slice.as_ptr() as *mut _, slice.len()) }
    }

    pub fn copy_from(&'a mut self, other : &Window<N>) {
        assert!(self.shape() == other.shape(), "Mutable windows differ in shape");
        self.rows_mut().zip(other.rows()).for_each(|(this, other)| this.copy_from_slice(other) );
    }

}

impl<'a, N> WindowMut<'a, N>
where
    N : Scalar + Copy + MulAssign + AddAssign + Add<Output=N> + Mul<Output=N> + SubAssign + Field + SimdPartialOrd,
    f64 : SubsetOf<N>
{
    
    pub fn component_scale(&mut self, _other : &Window<N>) {
        // self.win.component_mul_mut(&other.win);
        unimplemented!()
    }

}

impl<N> Index<(usize, usize)> for WindowMut<'_, N> 
where
    N : Scalar + Copy
{

    type Output = N;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        unsafe { self.win.get_unchecked(index::linear_index((self.offset.0 + index.0, self.offset.1 + index.1), self.orig_sz.1)) }
    }
}

impl<N> IndexMut<(usize, usize)> for WindowMut<'_, N> 
where
    N : Scalar + Copy
{
    
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        unsafe { self.win.get_unchecked_mut(index::linear_index((self.offset.0 + index.0, self.offset.1 + index.1), self.orig_sz.1)) }
    }
    
}

/*impl<'a, N> AsRef<DMatrixSlice<'a, N>> for Window<'a, N>
where
    N : Scalar + Copy
{
    fn as_ref(&self) -> &DMatrixSlice<'a, N> {
        // &self.win
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
        /*let step_rows = self.win_sz.0 / dst.nrows;
        let step_cols = self.win_sz.1 / dst.ncols;
        assert!(step_rows == step_cols);
        sampling::slices::subsample_convert_window(
            self.src,
            self.offset,
            self.win_sz, 
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
    fn from(buf : DMatrix<N>) -> Self {
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
        Self{ buf : DVector::from_vec(s.0), nrows, ncols  }
    }
}*/

impl<N> ripple::filter::Convolve for Image<N>
where
    N : Scalar + Copy + Default + Zero + Copy + Serialize + DeserializeOwned + Any + std::ops::Mul<Output = N> + std::ops::AddAssign
{

    fn convolve_mut(&self, filter : &Self, out : &mut Self) {

        {
            assert!(out.height() == self.height() - filter.height() + 1);
            assert!(out.width() == self.width() - filter.width() + 1);
            assert!(filter.width() % 2 == 1);
            assert!(filter.height() % 2 == 1);
            let half_kh = filter.height() / 2;
            let half_kw = filter.width() / 2;

            for (ix_ci, center_i) in (half_kh..(self.height() - half_kh)).enumerate() {
                for (ix_cj, center_j) in (half_kw..(self.width() - half_kw)).enumerate() {
                    out[(ix_ci, ix_cj)] = N::zero();
                    for (ix_ki, ki) in ((center_i - half_kh)..(center_i + half_kh + 1)).enumerate() {
                        for (ix_kj, kj) in ((center_j - half_kw)..(center_j + half_kw + 1)).enumerate() {
                            out[(ix_ci, ix_cj)] += self[(ki, kj)] * filter[(ix_ki, ix_kj)];
                        }
                    }
                }
            }
            return;
        }

        #[cfg(feature="opencv")]
        {
            use opencv;

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
        }

        unimplemented!()
    }
}

impl<N> AsRef<[N]> for Image<N>
where
    N : Scalar + Copy + Serialize + DeserializeOwned + Any
{
    fn as_ref(&self) -> &[N] {
        &self.buf[..]
    }
}

impl<N> AsMut<[N]> for Image<N> 
where
    N : Scalar + Copy + Serialize + DeserializeOwned + Any
{
    fn as_mut(&mut self) -> &mut [N] {
        &mut self.buf[..]
    }
}

impl<N> fmt::Display for Window<'_, N> 
where
    N : Scalar + Copy,
    f64 : From<N>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", io::build_pgm_string_from_slice(&self.win, self.orig_sz.1))
    }
}

impl<N> fmt::Display for WindowMut<'_, N> 
where
    N : Scalar + Copy,
    f64 : From<N>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", io::build_pgm_string_from_slice(&self.win, self.orig_sz.1))
    }
}

/*impl<N> fmt::Display for Image<N>
where
    N : Scalar + Copy,
    f64 : From<N>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", io::build_pgm_string_from_slice(&self.buf[..], self.ncols))
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

#[test]
fn checkerboard() {
    let src : [u8; 4] = [0, 1, 1, 0];
    let mut converted : Image<f32> = Image::new_constant(4, 4, 0.0);
    let win = Window::from_square_slice(&src);
    //converted.convert_from_window(&win);
    // println!("{}", converted);
}

/*impl<N> AsRef<Vec<N>> for Image<N> 
where
    N : Scalar
{
    fn as_ref(&self) -> &Vec<N> {
        self.buf.data.as_vec()
    }
}

impl<N> AsMut<Vec<N>> for Image<N> 
where
    N : Scalar
{
    fn as_mut(&mut self) -> &mut Vec<N> {
        unsafe{ self.buf.data.as_vec_mut() }
    }
}*/

/*impl<N> AsRef<DVector<N>> for Image<N> 
where
    N : Scalar
{
    fn as_ref(&self) -> &DVector<N> {
        &self.buf
    }
}

impl<N> AsMut<DVector<N>> for Image<N> 
where
    N : Scalar
{
    fn as_mut(&mut self) -> &mut DVector<N> {
        &mut self.buf
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
                            *img = Image::new_constant(h as usize, w as usize, 0);
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
    use opencv::prelude::MatTraitManual;
    use opencv::prelude::MatTrait;

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
    use opencv::prelude::MatTraitManual;
    use opencv::prelude::MatTrait;

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
    use opencv::prelude::MatTraitManual;
    use opencv::prelude::MatTrait;

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
