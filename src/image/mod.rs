use nalgebra::*;
use volta::signal::sampling::{self};
use std::ops::{Index, IndexMut, Mul, Add, AddAssign, MulAssign, SubAssign};
use simba::scalar::SubsetOf;
use std::fmt;
use std::fmt::Debug;

#[cfg(feature="opencvlib")]
use opencv::core;

#[cfg(feature="mkl")]
mod fft;

#[cfg(feature="mkl")]
pub use fft::*;

#[cfg(feature="gsl")]
pub(crate) mod dwt;

#[cfg(feature="gsl")]
pub use dwt::*;

#[cfg(feature="gsl")]
mod interp;

#[cfg(feature="gsl")]
pub use interp::Interpolation2D;

pub mod pgm;

#[cfg(feature="ipp")]
pub mod ipp;

#[cfg(feature="opencvlib")]
pub mod cvutils;

pub mod index;

pub mod draw;

/// Digital image, represented row-wise. Fundamentally, an image differs from a matrix because
/// it is oriented row-wise in memory, while a matrix is oriented column-wise. Also, images are
/// bounded at a low and high end, because they are the product of a saturated digital quantization
/// process. But indexing, following OpenCV convention, happens from the top-left point, following
/// the matrix convention.
#[derive(Debug, Clone)]
pub struct Image<N> 
where
    N : Scalar
{
    buf : Vec<N>,
    ncols : usize
}

impl<N> Image<N>
where
    N : Scalar + Copy
{

    pub fn new_from_slice(source : &[N], ncols : usize) -> Self {
        let mut buf = Vec::with_capacity(source.len());
        buf.copy_from_slice(&source);
        Self{ buf, ncols }
    }

    pub fn new_constant(nrows : usize, ncols : usize, value : N) -> Self {
        let mut buf = Vec::with_capacity(nrows * ncols);
        buf.extend((0..(nrows*ncols)).map(|_| value ));
        Self{ buf, ncols }
    }
    
    pub fn shape(&self) -> (usize, usize) {
        (self.buf.len() / self.ncols, self.ncols)
    }
    
    pub fn full_window<'a>(&'a self) -> Window<'a, N> {
        self.window((0, 0), self.shape())
    }
    
    pub fn full_window_mut<'a>(&'a mut self) -> WindowMut<'a, N> {
        let shape = self.shape();
        self.window_mut((0, 0), shape)
    }
    
    pub fn window<'a>(&'a self, offset : (usize, usize), sz : (usize, usize)) -> Window<'a, N> {
        Window { win : self.buf.as_slice(), orig_sz : self.shape(), offset, win_sz : sz }
    }
    
    pub fn window_mut<'a>(&'a mut self, offset : (usize, usize), win_sz : (usize, usize)) -> WindowMut<'a, N> {
        let shape = self.shape();
        WindowMut { win : &mut self.buf[..], orig_sz : shape, offset, win_sz }
    }
    
    pub fn downsample(&mut self, src : &Window<N>) {
        assert!(src.is_full());
        let src_ncols = src.orig_sz.1;
        let dst_ncols = self.ncols;
        
        #[cfg(feature="opencvlib")]
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
    
    pub fn width(&self) -> usize {
        self.ncols
    }
    
    pub fn height(&self) -> usize {
        self.buf.len() / self.ncols
    }
    
    pub fn copy_from(&mut self, other : &Image<N>) {
        self.buf.copy_from_slice(other.buf.as_slice());
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
        M : Scalar
    {
        let ncols = self.ncols;
        
        #[cfg(feature="opencvlib")]
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
        
        panic!("Either opencvlib or ipp feature should be enabled for image conversion");
    }
    
    pub fn iter(&self) -> impl Iterator<Item=&N> {
        let shape = self.shape();
        iterate_row_wise(&self.buf[..], (0, 0), shape, shape)
    }
    
    pub fn len(&self) -> usize {
        self.buf.len()
    }

    // pub fn windows(&self) -> impl Iterator<Item=Window<'_, N>> {
    //    unimplemented!()
    // }
    
}

impl<N> Image<N>
    where N : Scalar + Copy + RealField 
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

    pub fn draw(&mut self, mark : Mark) {
        self.full_window_mut().draw(mark);
    }
}

impl Image<f32> {

    pub fn max(&self) -> ((usize, usize), f32) {
        let (mut max_ix, mut max) = ((0, 0), f32::NEG_INFINITY);
        for (lin_ix, px) in self.iter().enumerate() {
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
    N : Scalar
{

    type Output = N;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.buf[index::linear_index(index, self.ncols)]
    }
}

/*impl<N> IndexMut<(usize, usize)> for Image<N> 
where
    N : Scalar
{
    
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        &mut self.buf[index]
    }
    
}*/

/*impl<N> AsRef<DMatrix<N>> for Image<N> 
where
    N : Scalar
{
    fn as_ref(&self) -> &DMatrix<N> {
        &self.buf
    }
}

impl<N> AsMut<DMatrix<N>> for Image<N> 
where
    N : Scalar
{
    fn as_mut(&mut self) -> &mut DMatrix<N> {
        &mut self.buf
    }
}*/

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
    
    // Original image dimensions (height, width).
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

impl<'a, N> Window<'a, N>
where
    N : Scalar
{
    pub fn is_full(&'a self) -> bool {
        self.orig_sz == self.win_sz
    }
}
    
impl<'a, N> Window<'a, N> 
where
    N : Scalar + Mul<Output=N> + MulAssign
{

    /// Creates a window that cover the whole slice src, assuming it represents a square image.
    pub fn from_square_slice(src : &'a [N]) -> Self {
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
    
    pub fn sub_window(&'a self, offset : (usize, usize), dims : (usize, usize)) -> Window<'a, N> {
        Self { 
            win : self.win, 
            offset : (self.offset.0 + offset.0, self.offset.1 + offset.1), 
            orig_sz : self.orig_sz, 
            // transposed : self.transposed,
            win_sz : dims
        }
    }
    
    /// Creates a window that cover the whole slice src. The slice is assumed to be in
    /// row-major order, but matrices are assumed to be 
    pub fn from_slice(src : &'a [N], ncols : usize) -> Self {
        let nrows = src.len() / ncols;
        Self{ 
            // win : DMatrixSlice::from_slice_generic(src, Dynamic::new(nrows),
            // Dynamic::new(ncols)),
            win : src,
            offset : (0, 0),
            orig_sz : (nrows, ncols),
            win_sz : (nrows, ncols),
            // transposed : true
        }
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
    
    pub fn rows(&self) -> impl Iterator<Item=&[N]> {
        let stride = self.orig_sz.1;
        let tl = self.offset.0 * stride + self.offset.1;
        (0..self.win_sz.0).map(move |i| {
            let start = tl + i*stride;
            &self.win[start..(start+self.win_sz.1)]
        })
    }

    pub fn iter(&self) -> impl Iterator<Item=&N> {
        iterate_row_wise(self.win, self.offset, self.win_sz, self.orig_sz)
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
    ]);

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
            &self.win[index::linear_index(off_ix, self.orig_sz.1)]
        } else {
            panic!("Invalid window index: {:?}", index);
        }
    }
}

pub fn iterate_row_wise<N>(
    src : &[N], 
    offset : (usize, usize), 
    win_sz : (usize, usize), 
    orig_sz : (usize, usize)
) -> impl Iterator<Item=&N> {
    let start = orig_sz.1 * offset.0 + offset.1;
    (0..win_sz.0).map(move |i| {
        let row_offset = start + i*orig_sz.1;
        &src[row_offset..(row_offset+win_sz.1)]
    }).flatten()
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
    N : Scalar
{

    type Item = Window<'a, N>;

    fn next(&mut self) -> Option<Self::Item> {
        let within_horiz = self.curr_pos.0  + self.size.0 <= (self.source.offset.0 + self.source.win_sz.0);
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
        win
    }

}

#[cfg(feature="opencvlib")]
impl<N> Into<core::Mat> for Window<'_, N> 
where
    N : Scalar + Copy
{

    fn into(self) -> core::Mat {
        let sub_slice = Some((self.offset, self.win_sz));
        let stride = self.orig_sz.1;
        unsafe{ cvutils::slice_to_mat(self.win, stride, sub_slice) }
    }
}

#[cfg(feature="opencvlib")]
impl<N> Into<core::Mat> for WindowMut<'_, N>
where
    N : Scalar + Copy
{

    fn into(self) -> core::Mat {
        let sub_slice = Some((self.offset, self.win_sz));
        let stride = self.orig_sz.1;
        unsafe{ cvutils::slice_to_mat(self.win, stride, sub_slice) }
    }
}

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
    Label((usize, usize), &'static str, usize, u8),

    // Center, radius and color
    Circle((usize, usize), usize, u8)
    
}

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

    pub fn from_slice(src : &'a mut [N], ncols : usize) -> Self {
        /*let nrows = src.len() / ncols;
        Self {
            win : DMatrixSliceMut::from_slice_generic(src, Dynamic::new(nrows),
            Dynamic::new(ncols)),
            offset : (0, 0),
            orig_sz : (nrows, ncols)
        }*/
        let nrows = src.len() / ncols;
        Self{
            // win : DMatrixSlice::from_slice_generic(src, Dynamic::new(nrows),
            // Dynamic::new(ncols)),
            win : src,
            offset : (0, 0),
            orig_sz : (nrows, ncols),
            win_sz : (nrows, ncols),
        }
    }

    pub fn sub_window(&'a mut self, offset : (usize, usize), dims : (usize, usize)) -> WindowMut<'a, N> {
        /*let offset = (self.offset.0 + offset.0, self.offset.1 + offset.1);
        let orig_sz = self.orig_sz;
        Self{ win : self.win.slice_mut(offset, dims), offset, orig_sz }*/
        Self {
            win : self.win,
            offset : (self.offset.0 + offset.0, self.offset.1 + offset.1),
            orig_sz : self.orig_sz,
            // transposed : self.transposed,
            win_sz : dims
        }
    }

    /// Creates a window that cover the whole slice src, assuming it represents a square image.
    pub fn from_square_slice(src : &'a mut [N]) -> Self {
        Self::from_slice(src, (src.len() as f64).sqrt() as usize)
    }

}

impl WindowMut<'_, u8> {

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
                
                #[cfg(feature="opencvlib")]
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
            Mark::Digit(pos, val, sz, color) => {
                let tl_pos = (self.offset.0 + pos.0, self.offset.1 + pos.1);

                #[cfg(feature="opencvlib")]
                unsafe {
                    cvutils::write_text(self.win, self.orig_sz.1, tl_pos, &val.to_string()[..], color);
                    return;
                }
                
                draw::draw_digit_native(self.win, self.orig_sz.1, tl_pos, val, sz, color);
            },
            Mark::Label(pos, msg, sz, color) => {
                let tl_pos = (self.offset.0 + pos.0, self.offset.1 + pos.1);

                #[cfg(feature="opencvlib")]
                unsafe {
                    cvutils::write_text(self.win, self.orig_sz.1, tl_pos, msg, color);
                    return;
                }

                panic!("Label draw require 'opencvlib' feature");
            },
            Mark::Circle(pos, radius, color) => {
                let center_pos = (self.offset.0 + pos.0, self.offset.1 + pos.1);

                #[cfg(feature="opencvlib")]
                unsafe {
                    cvutils::draw_circle(self.win, self.orig_sz.1, center_pos, radius, color);
                    return;
                }

                panic!("Circle draw require 'opencvlib' feature");
            }
        }
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

    fn index(&self, _index: (usize, usize)) -> &Self::Output {
        // &self.win[index]
        unimplemented!()
    }
}

impl<N> IndexMut<(usize, usize)> for WindowMut<'_, N> 
where
    N : Scalar + Copy
{
    
    fn index_mut(&mut self, _index: (usize, usize)) -> &mut Self::Output {
        //&mut self.win[index]
        unimplemented!()
    }
    
}

impl<'a, N> AsRef<DMatrixSlice<'a, N>> for Window<'a, N> 
where
    N : Scalar + Copy
{
    fn as_ref(&self) -> &DMatrixSlice<'a, N> {
        // &self.win
        unimplemented!()
    }
}

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

impl<N> AsRef<[N]> for Image<N> 
where
    N : Scalar
{
    fn as_ref(&self) -> &[N] {
        &self.buf[..]
    }
}

impl<N> AsMut<[N]> for Image<N> 
where
    N : Scalar
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
        write!(f, "{}", pgm::build_pgm_string_from_slice(&self.win, self.orig_sz.1))
    }
}

impl<N> fmt::Display for WindowMut<'_, N> 
where
    N : Scalar + Copy,
    f64 : From<N>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", pgm::build_pgm_string_from_slice(&self.win, self.orig_sz.1))
    }
}

impl<N> fmt::Display for Image<N> 
where
    N : Scalar + Copy,
    f64 : From<N>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", pgm::build_pgm_string_from_slice(&self.buf[..], self.ncols))
    }
}

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
