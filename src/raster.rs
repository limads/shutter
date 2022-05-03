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
    // current object.
    unsafe fn original_slice(&mut self) -> Self::Slice;

}

impl<'a, T> Raster for Window<'a, T>
where
    T : Scalar + Copy
{

    type Slice = &'a [T];

    fn create(offset : (usize, usize), win_sz : (usize, usize), orig_sz : (usize, usize), win : Self::Slice) -> Self {
        Window { offset, win_sz, width : orig_sz.1, win }
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
        (self.win.len() / self.width, self.width)
    }

    unsafe fn original_slice(&mut self) -> Self::Slice {
        self.win
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

    fn original_size(&self) -> (usize, usize) {
        (self.win.len() / self.width, self.width)
    }

    fn original_width(&self) -> usize {
        self.width
    }

    fn original_height(&self) -> usize {
        self.win.len() / self.width
    }

    unsafe fn original_slice(&mut self) -> Self::Slice {
        std::slice::from_raw_parts_mut(self.win.as_mut_ptr(), self.win.len())
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

pub struct WindowIterator<'a, N>
where
    N : Scalar,
{
    pub(crate) source : Window<'a, N>,

    // This child window size
    pub(crate) size : (usize, usize),

    // Index the most ancestral window possible.
    pub(crate) curr_pos : (usize, usize),

    /// Vertical increment. Either U1 or Dynamic.
    pub(crate) step_v : usize,

    /// Horizontal increment. Either U1 or Dynamic.
    pub(crate) step_h : usize,

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

    pub(crate) source : WindowMut<'a, N>,

    // This child window size
    pub(crate) size : (usize, usize),

    // Index the most ancestral window possible.
    pub(crate) curr_pos : (usize, usize),

    /// Vertical increment. Either U1 or Dynamic.
    pub(crate) step_v : usize,

    /// Horizontal increment. Either U1 or Dynamic.
    pub(crate) step_h : usize,

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
    R : Raster<Slice=S>
{
    let (offset, win_sz) = (*s.offset(), *s.size());
    let within_horiz = curr_pos.0  + size.0 <= (offset.0 + win_sz.0);
    let within_vert = curr_pos.1 + size.1 <= (offset.1 + win_sz.1);
    let within_bounds = within_horiz && within_vert;
    let win = if within_bounds {
        Some(Raster::create(*curr_pos, size, s.original_size(), unsafe { s.original_slice() }))
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

/// Iterates over pairs of pixels within a column, carrying row index of the top element at first position.
pub fn vertical_col_iterator<'a, N>(
    rows : impl Iterator<Item=&'a [N]>+Clone+ 'a,
    comp_dist : usize,
    col : usize
) -> impl Iterator<Item=(usize, (&'a N, &'a N))> + 'a
where
    N : Copy + 'a
{
    let n = rows.clone().count();
    rows.clone().take(n - comp_dist)
        .map(move |row| &row[col] )
        .zip(rows.skip(comp_dist).map(move |row| unsafe { row.get_unchecked(col) } ))
        .enumerate()
}

/// Iterates over pairs of pixels within a row, carrying the column index of the left element at first position
pub fn horizontal_row_iterator<'a, N>(
    row : &'a [N],
    comp_dist : usize
) -> impl Iterator<Item=(usize, (&'a N, &'a N))>+'a {
    unsafe {
        row[0..(row.len().saturating_sub(comp_dist))].iter()
            .zip(row.get_unchecked(comp_dist..row.len()).iter())
            .enumerate()
    }
}

/// Iterates over pairs of pixels, carrying row index at first position for the informed column.
pub fn vertical_row_iterator<'a, N>(
    rows : impl Iterator<Item=&'a [N]> + Clone + 'a,
    comp_dist : usize,
    col : usize
) -> impl Iterator<Item=(usize, (&'a N, &'a N))>+'a
where
    N : Copy + 'a
{
    let lower_rows = rows.clone().step_by(comp_dist);
    let upper_rows = rows.skip(comp_dist).step_by(comp_dist);
    lower_rows.zip(upper_rows)
        .enumerate()
        .map(move |(row_ix, (lower, upper))| unsafe { (comp_dist*row_ix, (lower.get_unchecked(col), upper.get_unchecked(col))) })
}

/// Iterates over a diagonal, starting at given (row, col) and going from top-left to bottom-right
pub fn diagonal_right_row_iterator<'a, N>(
    rows : impl Iterator<Item=&'a [N]> + Clone + 'a,
    comp_dist : usize,
    start : (usize, usize)
) -> impl Iterator<Item=((usize, usize), (&'a N, &'a N))>+'a
where
    N : Copy + 'a
{
    let nrows = rows.clone().count();
    let ncols = rows.clone().next().unwrap().len();
    let take_n = (nrows.saturating_sub(start.0) / comp_dist).min(ncols.saturating_sub(start.1) / comp_dist).saturating_sub(1);
    let lower_rows = rows.clone().skip(start.0).step_by(comp_dist);
    let upper_rows = rows.clone().skip(start.0+comp_dist).step_by(comp_dist);
    lower_rows.zip(upper_rows)
        .enumerate()
        .take(take_n)
        .map(move |(ix, (row1, row2))| unsafe {
            let px_ix = (start.0 + comp_dist*ix, start.1 + comp_dist*ix);
            // println!("{:?}", (start, nrows, ncols, take_n, px_ix));
            (px_ix, (row1.get_unchecked(px_ix.1), row2.get_unchecked(px_ix.1 + comp_dist)))
         })
}

// Iterates over a diagonal, starting at given (row, col) and going from top-right to bottom-left
pub fn diagonal_left_row_iterator<'a, N>(
    rows : impl Iterator<Item=&'a [N]> + Clone + 'a,
    comp_dist : usize,
    start : (usize, usize)
) -> impl Iterator<Item=((usize, usize), (&'a N, &'a N))>+'a
where
    N : Copy + 'a
{
    let nrows = rows.clone().count();
    let ncols = rows.clone().next().unwrap().len();
    let take_n = (nrows.saturating_sub(start.0) / comp_dist).min(start.1 / comp_dist).saturating_sub(1);
    let lower_rows = rows.clone().skip(start.0).step_by(comp_dist);
    let upper_rows = rows.skip(start.0+comp_dist).step_by(comp_dist);
    lower_rows.zip(upper_rows)
        .enumerate()
        .take(take_n)
        .map(move |(ix, (row1, row2))| unsafe {
            let px_ix = (start.0 + comp_dist*ix, start.1 - comp_dist*ix);
            // println!("{:?}", (start, nrows, ncols, take_n, ncols, px_ix));
            (px_ix, (row1.get_unchecked(px_ix.1), row2.get_unchecked(px_ix.1 - comp_dist)))
         })
}

#[test]
fn diag() {
    use crate::image::Image;
    let mut img = Image::new_constant(10, 10, 1);

    for r in 0..10 {
        for c in 0..10 {
            img[(r, c)] = r;
        }
    }

    println!("Diagonal right:");
    diagonal_right_row_iterator(img.full_window().rows(), 2, (0, 0))
        .for_each(|(ix, px)| println!("{:?} : {:?}", ix, px) );

    println!("Diagonal left:");
    diagonal_left_row_iterator(img.full_window().rows(), 2, (0, 9))
        .for_each(|(ix, px)| println!("{:?} : {:?}", ix, px) );
}

