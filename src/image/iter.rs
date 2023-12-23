use super::*;
use tuples::TupleTranspose;

impl<S> Image<u8, S>
where
S : Storage<u8>
{

    /*pub fn lanes(&self) -> impl Iterator<Item=wide::u8x16> {
        assert!(self.width() % 16 == 0);
        // TODO use std::slice::array_chunks when stable.
        self.rows().map(|r| r.chunks(16).map(|c| unsafe { wide::u8x16::new(std::mem::transmute(c[0]) ) } ) ).flatten()
    }*/

}

impl<P, S> Image<P, S>
where
    S : Storage<P>,
    P : Pixel
{

    /* Returns a sequence of rows of this image from top to bottom. */
    pub fn rows<'a>(&'a self) -> impl Iterator<Item=&'a [P]> + Clone + 'a {
        // let tl = self.offset.0 * self.width + self.offset.1;
        (0..self.sz.0).map(move |i| {
            let start = i*self.width;
            unsafe { self.slice.as_ref().get_unchecked(start..(start+self.sz.1)) }
        })
    }

    /* Returns the sequence of row sub-slices of size sz contained in this image, in raster
    order. Panics if image width is not divisible by sz */
    pub fn row_chunks<'a>(&'a self, sz : usize) -> impl Iterator<Item=&'a [P]> + Clone + 'a {
        assert!(self.sz.1 % sz == 0);
        self.rows().map(move |r| r.chunks(sz) ).flatten()
    }

    /// Returns iterator over (subsampled row index, subsampled col index, pixel color).
    /// Panics if L is an unsigend integer type that cannot represent one of the dimensions
    /// of the image precisely.
    pub fn labeled_pixels<'a, L, E>(
        &'a self, 
        px_spacing : usize
    ) -> impl Iterator<Item=(L, L, P)> +'a + Clone
    where
        L : TryFrom<usize, Error=E> + Div<Output=L> + 
            Mul<Output=L> + Rem<Output=L> + Clone + Copy + 'static,
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

    pub fn pixels_across_line<'b>(
        &'b self, 
        src : (usize, usize), 
        dst : (usize, usize)
    ) -> impl Iterator<Item=&P> + 'b {
        let (nrow, ncol) = self.shape();
        coords_across_line(src, dst, self.shape()).map(move |pt| &self[pt] )
    }
    
    /// Iterate over windows of the given size. This iterator consumes the original window
    /// so that we can implement windows(.) for Image by using move semantics, without
    /// requiring the user to call full_windows(.).
    pub fn windows(&self, sz : (usize, usize)) -> impl Iterator<Item=Window<P>> {
        assert_nonzero(sz);
        // let (step_v, step_h) = sz;
        if sz.0 > self.sz.0 || sz.1 > self.sz.1 {
            panic!("Child window size bigger than parent window size");
        }
        if self.height() % sz.0 != 0 || self.width() % sz.1 != 0 {
            panic!("Image size should be a multiple of window size (Required window {:?} over parent window {:?})", sz, self.sz);
        }
        iter::WindowIterator::<P> {
            source : self.window((0, 0), self.size()).unwrap(),
            size : sz,
            curr_pos : Some((0, 0)),
        }
    }
    
    /// Iterate over all image pixels if spacing=1; or over pixels spaced
    /// horizontally and verticallly by spacing. Iteration proceeds row-wise.
    pub fn pixels<'a>(&'a self, spacing : usize) -> impl Iterator<Item=&'a P> + Clone {
        assert!(spacing > 0, "Spacing should be at least one");
        assert!(
            self.width() % spacing == 0 && self.height() % spacing == 0, 
            "Spacing should be integer divisor of width and height"
        );
        iter::iterate_row_wise(
            self.slice.as_ref(), 
            self.sz, 
            self.original_size(), 
            spacing
        ).step_by(spacing)
    }

    pub fn labeled_neighborhoods<'a>(&'a self) -> impl Iterator<Item=((usize, usize), PixelNeighborhood<P>)> + 'a {
        (0..(self.width()*self.height()))
            .map(move |ix| {
                let pos = (ix / self.width(), ix % self.width());
                let neigh = self.neighboring_pixels(pos).unwrap();
                (pos, neigh)
            })
    }

    pub fn neighboring_pixels<'a>(
        &'a self, 
        pos : (usize, usize)
    ) -> Option<PixelNeighborhood<P>> {

        if pos.0 > 0 && pos.1 > 0 && pos.0 < self.height() - 1 && pos.1 < self.width() - 1 {

            // Center
            let (tl, tr) = (top_left(pos), top_right(pos));
            let (bl, br) = (bottom_left(pos), bottom_right(pos));
            let (t, b) = (top(pos), bottom(pos));
            let (l, r) = (left(pos), right(pos));
            Some(PixelNeighborhood::Full([
                self[tl], self[t], self[tr],
                self[l], self[r],
                self[bl], self[b], self[br]
            ], 0))
        } else if pos.0 == 0 {

            // Top row

            if pos.1 == 0 {
                // Top row left corner
                let (b, br, r) = (bottom(pos), bottom_right(pos), right(pos));
                Some(PixelNeighborhood::Corner([self[b], self[br], self[r]], 0))
            } else if pos.1 == self.width()-1 {
                // Top right right corner
                let (l, bl, b) = (left(pos), bottom_left(pos), bottom(pos));
                Some(PixelNeighborhood::Corner([self[l], self[bl], self[b]], 0))
            } else {
                // Top row NOT corner
                let (l, r, bl, b, br) = (
                    left(pos), 
                    right(pos), 
                    bottom_left(pos), 
                    bottom(pos), 
                    bottom_right(pos)
                );
                Some(PixelNeighborhood::Edge([self[l], self[r], self[bl], self[b], self[br]], 0))
            }
        } else if pos.0 == self.height()-1 && pos.1 < self.width() {

            // Bottom row

            if pos.1 == 0 {
                // Bottom row left corner
                let (t, tr, r) = (top(pos), top_right(pos), right(pos));
                Some(PixelNeighborhood::Corner([self[t], self[tr], self[r]], 0))
            } else if pos.1 == self.width()-1 {
                // Bottom row right corner
                let (t, tl, l) = (top(pos), top_left(pos), left(pos));
                Some(PixelNeighborhood::Corner([self[t], self[tl], self[l]], 0))
            } else {
                // Bottom row NOT corner
                let (l, r, tl, t, tr) = (
                    left(pos), 
                    right(pos), 
                    top_left(pos), 
                    top(pos), 
                    top_right(pos)
                );
                Some(PixelNeighborhood::Edge([self[l], self[r], self[tl], self[t], self[tr]], 0))
            }

        } else if pos.1 == 0 && pos.0 < self.height() {

            //  Left column (except corner pixels, matched above)
            let (t, b, tr, r, br) = (
                top(pos), 
                bottom(pos), 
                top_right(pos), 
                right(pos), 
                bottom_right(pos)
            );
            Some(PixelNeighborhood::Edge([self[t], self[b], self[tr], self[r], self[br]], 0))

        } else if pos.1 == self.width()-1 && pos.0 < self.height() {

            // Right column (except corner pixels, matched above)
            let (t, b, tl, l, bl) = (
                top(pos), 
                bottom(pos), 
                top_left(pos), 
                left(pos), 
                bottom_left(pos)
            );
            Some(PixelNeighborhood::Edge([self[t], self[b], self[tl], self[l], self[bl]], 0))

        } else {
            // Outside area
            None
        }
    }
    
    /// Returns a minimum number of overlapping windows of the informed shape,
    /// that completely cover the current window.
    pub fn minimum_inner_windows<'a>(&'a self, shape : (usize, usize)) -> Vec<Window<'a, P>> {
        let mut sub_wins = Vec::new();
        sub_wins
    }

    pub fn four_distant_window_neighborhood<'a>(
        &'a self,
        center_tl : (usize, usize),
        sz : (usize, usize),
        neigh_ext : usize,
        dist : usize
    ) -> Option<WindowNeighborhood<'a, P>> {
        let outside_bounds = center_tl.0 < sz.0 + dist ||
            center_tl.0 > self.height().checked_sub(sz.0 + dist)? ||
            center_tl.1 > self.width().checked_sub(sz.1 + dist)? ||
            center_tl.1 < sz.1 + dist;
        if outside_bounds {
            return None;
        }
        let vert_sz = (neigh_ext, sz.1);
        let horiz_sz = (sz.0, neigh_ext);
        Some(WindowNeighborhood {
            center : self.window(center_tl, sz)?,
            left : self.window((center_tl.0, center_tl.1 - sz.1 - dist), horiz_sz)?,
            top : self.window((center_tl.0 - sz.0 - dist, center_tl.1), vert_sz)?,
            right : self.window((center_tl.0, center_tl.1 + sz.1 + dist), horiz_sz)?,
            bottom : self.window((center_tl.0 + sz.0 + dist, center_tl.1), vert_sz)?
        })
    }

    /*pub fn complement_windows<'a>(
        &'a self,
        tl : (usize, usize),
        sz : (usize, usize)
    ) -> Option<WindowNeighborhood<'a, P>> {
        Some(WindowNeighborhood {
            center : self.window(center_tl, sz)?,
            left : self.window((center_tl.0, center_tl.1 - sz.1), horiz_sz)?,
            top : self.window((center_tl.0 - sz.0, center_tl.1), vert_sz)?,
            right : self.window((center_tl.0, center_tl.1 + sz.1), horiz_sz)?,
            bottom : self.window((center_tl.0 + sz.0, center_tl.1), vert_sz)?
        })
    }*/

    // Get the four windows at top, left, bottom and right of a center window
    // identified by its top-left position, where top/bottom neighboring windows are
    // neigh_ext x sz.1 and left/right neighboring windows are sz.0 x sz.0
    pub fn four_window_neighborhood<'a>(
        &'a self,
        center_tl : (usize, usize),
        sz : (usize, usize),
        neigh_ext : usize
    ) -> Option<WindowNeighborhood<'a, P>> {
        let outside_bounds = center_tl.0 < sz.0 ||
            center_tl.0 > self.height().checked_sub(sz.0)? ||
            center_tl.1 > self.width().checked_sub(sz.1)? ||
            center_tl.1 < sz.1;
        if outside_bounds {
            return None;
        }
        let vert_sz = (neigh_ext, sz.1);
        let horiz_sz = (sz.0, neigh_ext);
        Some(WindowNeighborhood {
            center : self.window(center_tl, sz)?,
            left : self.window((center_tl.0, center_tl.1 - sz.1), horiz_sz)?,
            top : self.window((center_tl.0 - sz.0, center_tl.1), vert_sz)?,
            right : self.window((center_tl.0, center_tl.1 + sz.1), horiz_sz)?,
            bottom : self.window((center_tl.0 + sz.0, center_tl.1), vert_sz)?
        })
    }
    
    // Splits this window into equally-sized subwindows, iterating row-wise over the blocks.
    // Same as self.windows(.) but here decide on how many windows are desired instead of the
    // sub window size.
    pub fn equivalent_windows<'a>(
        &'a self, 
        num_rows : usize, 
        num_cols : usize
    ) -> impl Iterator<Item=Window<'a, P>> 
    {
        assert!(self.height() % num_rows == 0 && self.width() % num_cols == 0);
        self.windows((self.height() / num_rows, self.width() / num_cols))
    }

    pub fn column<'a>(&'a self, ix : usize) -> Option<impl Iterator<Item=P> + 'a > {
        if ix < self.width() {
            Some(self.rows().map(move |row| row[ix] ))
        } else {
            None
        }
    }

    /// Iterates over pairs of pixels within a row, carrying the column index of the left element at first position
    pub fn horizontal_pixel_pairs<'a>(&'a self, row : usize, comp_dist : usize) -> Option<impl Iterator<Item=(usize, (&'a P, &'a P))>> {
        Some(iter::horizontal_row_iterator(self.row(row)?, comp_dist))
    }

    pub fn vertical_pixel_pairs<'a>(&'a self, col : usize, comp_dist : usize) -> Option<impl Iterator<Item=(usize, (&'a P, &'a P))>> {
        if col >= self.sz.1 {
            return None;
        }
        Some(iter::vertical_col_iterator(self.rows(), comp_dist, col))
    }

    /// Iterate over one of the lower-triangular diagonals the image, starting at given row.
    /// If to_right is passed, iterate from the top-left to bottom-right corner. If not, iterate from the
    /// top-right to bottom-left corner.
    pub fn lower_to_right_diagonal_pixel_pairs<'a>(
        &'a self,
        row : usize,
        comp_dist : usize,
    ) -> Option<impl Iterator<Item=((usize, usize), (&'a P, &'a P))>> {
        if row < self.height() {
            Some(iter::diagonal_right_row_iterator(self.rows(), comp_dist, (row, 0)))
        } else {
            None
        }
    }

    pub fn upper_to_right_diagonal_pixel_pairs<'a>(
        &'a self,
        col : usize,
        comp_dist : usize,
    ) -> Option<impl Iterator<Item=((usize, usize), (&'a P, &'a P))>> {
        if col < self.width() {
            Some(iter::diagonal_right_row_iterator(self.rows(), comp_dist, (0, col)))
        } else {
            None
        }
    }

    pub fn lower_to_left_diagonal_pixel_pairs<'a>(
        &'a self,
        row : usize,
        comp_dist : usize,
    ) -> Option<impl Iterator<Item=((usize, usize), (&'a P, &'a P))>> {
        if row < self.height() {
            Some(iter::diagonal_left_row_iterator(self.rows(), comp_dist, (row, self.width()-1)))
        } else {
            None
        }
    }

    pub fn upper_to_left_diagonal_pixel_pairs<'a>(
        &'a self,
        col : usize,
        comp_dist : usize,
    ) -> Option<impl Iterator<Item=((usize, usize), (&'a P, &'a P))>> {
        if col < self.width() {
            Some(iter::diagonal_left_row_iterator(self.rows(), comp_dist, (0, col)))
        } else {
            None
        }
    }

    pub fn to_right_diagonal_pixel_pairs<'a>(
        &'a self,
        comp_dist : usize
    ) -> impl Iterator<Item=((usize, usize), (&'a P, &'a P))> {
        (0..self.height()).step_by(comp_dist)
            .map(move |r| self.lower_to_right_diagonal_pixel_pairs(r, comp_dist).unwrap() )
            .flatten()
            .chain((0..self.width()).step_by(comp_dist)
                .map(move |c| self.upper_to_right_diagonal_pixel_pairs(c, comp_dist).unwrap() )
                .flatten()
            )
    }

    pub fn to_left_diagonal_pixel_pairs<'a>(
        &'a self,
        comp_dist : usize
    ) -> impl Iterator<Item=((usize, usize), (&'a P, &'a P))> {
        (0..self.height()).step_by(comp_dist)
            .map(move |r| self.lower_to_left_diagonal_pixel_pairs(r, comp_dist).unwrap() )
            .flatten()
            .chain((0..self.width()).step_by(comp_dist)
                .map(move |c| self.upper_to_left_diagonal_pixel_pairs(c, comp_dist).unwrap() )
                .flatten()
            )
    }

    pub fn rect_pixels<'a>(&'a self, rect : (usize, usize, usize, usize)) -> impl Iterator<Item=P> + Clone + 'a {
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
    ) -> impl Iterator<Item=((usize, usize), &P)> + Clone {
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
    
    // Returns an iterator over (radius, pixel value) that approximates a circle of max_radius
    // using the 8 cardinal directions (45 degrees apart). Returns (radius, pixel). Panics if
    // center is outside the admissible area.
    pub fn simplified_radial_pixels<'a>(
        &'a self, 
        center : (usize, usize),
        step : usize,
        max_radius : usize
    ) -> impl Iterator<Item=(usize, &'a P)> + 'a {
    
        let center_iter = std::iter::once((0, &self[center]));
        let radial_iter = (step..(max_radius+1))
            .step_by(step)
            .map(move |rad| {
                
                let opt_l = center.1.checked_sub(rad);
                let opt_t = center.0.checked_sub(rad);
                let cand_b = center.0 + rad;
                let cand_r = center.1 + rad;
                let opt_b = (cand_b < self.height()).then_some(cand_b);
                let opt_r = (cand_r < self.width()).then_some(cand_r);
                let t = (opt_t, Some(center.1)).transpose();
                let r = (Some(center.0), opt_r).transpose();
                let b = (opt_b, Some(center.1)).transpose();
                let l = (Some(center.0), opt_l).transpose();
                
                // cos(pi/4) = sin(pi/4) ~ 0.7, so the diagonal (x, y) increment should be ~70% of axial lines
                // to preserve the radius.
                let rad_diag = (rad as f32 * 0.707107) as usize;
                let opt_diag_t = center.0.checked_sub(rad_diag);
                let opt_diag_l = center.1.checked_sub(rad_diag);
                let cand_diag_b = center.0 + rad_diag;
                let cand_diag_r = center.1 + rad_diag;
                let opt_diag_b = (cand_diag_b < self.height()).then_some(cand_diag_b);
                let opt_diag_r = (cand_diag_r < self.width()).then_some(cand_diag_r);
                let tl = (opt_diag_t, opt_diag_l).transpose();
                let tr = (opt_diag_t, opt_diag_r).transpose();
                let br = (opt_diag_b, opt_diag_r).transpose();
                let bl = (opt_diag_b, opt_diag_l).transpose();
                
                tl.into_iter().chain(t.into_iter()).chain(tr.into_iter()).chain(r.into_iter())
                    .chain(br.into_iter()).chain(b.into_iter()).chain(bl.into_iter()).chain(l.into_iter())
                    .map(move |c| (rad, &self[c]) ) 
           }).flatten();
       center_iter.chain(radial_iter)
    }

    // Returns the most representative k-colors of the image using K-means
    // pub fn colors(&self, px_spacing : usize, n_colors : usize) -> Vec<u8> {
    // }

    
}

impl<P, S> Image<P, S>
where
    P : Pixel,
    S : StorageMut<P>
{

    pub fn windows_mut<'a>(
        &'a mut self,
        sz : (usize, usize)
    ) -> impl Iterator<Item=WindowMut<'a, P>>
    where
        Self : 'a,
        P : Mul<Output=P> + MulAssign + 'a
    {
        assert_nonzero(sz);
        // let (step_v, step_h) = sz;
        if sz.0 > self.sz.0 || sz.1 > self.sz.1 {
            panic!("Child window size bigger than parent window size");
        }
        if self.height() % sz.0 != 0 || self.width() % sz.1 != 0 {
            panic!("Image size should be a multiple of window size (Required window {:?} over parent window {:?})", sz, self.sz);
        }
        iter::WindowIteratorMut::<'a, P> {
            source : self.window_mut((0, 0), self.size()).unwrap(),
            size : sz,
            curr_pos : Some((0, 0)),
        }
    }

    pub fn rows_mut<'a, 'b>(&'b mut self) -> impl Iterator<Item=&'a mut [P]> + 'b {
        // let tl = self.offset.0 * self.width + self.offset.1;
        (0..self.sz.0).map(move |i| {
            let start = i*self.width;
            unsafe {
                let mut slice = &mut self.slice.as_mut().get_unchecked_mut(start..(start+self.sz.1));
                std::slice::from_raw_parts_mut(slice.as_mut_ptr() as *mut _, slice.len())
            }
        })
    }

    pub fn row_chunks_mut<'b>(&'b mut self, sz : usize) -> impl Iterator<Item=&'b mut [P]> + 'b {
        assert!(self.sz.1 % sz == 0);
        self.rows_mut().map(move |r| r.chunks_mut(sz) ).flatten()
    }

    /*pub unsafe fn pixels_mut_ptr(&'a mut self, spacing : usize) -> impl Iterator<Item=*mut N> {
        self.pixels_mut(spacing).map(|px| px as *mut _ )
    }*/
    pub fn foreach_pixel<'b>(&'b mut self, spacing : usize, f : impl Fn(&mut P)) {
        for px in self.pixels_mut(spacing) {
            f(px);
        }
    }

    /*pub fn border_mut<'a>(
        &'a mut self
    ) -> Box<dyn Iterator<Item=&'a mut P> + 'a>
    {
        let (h, w) = self.size();
        let stride = self.stride();
        let lst_row = stride*(h - 1);
        let iter = (0..w)
            .chain((0..h).map(move |r| stride*r + w ) )
            .chain((0..w).map(move |c| lst_row + c ))
            .chain((0..h).map(move |r| stride*r ))
            .map(move |lin_ix| self.linear_index_mut(lin_ix) );
        Box::new(iter)
    }*/
    pub fn apply_to_border(
        mut self,
        mut f : impl FnMut(&mut P)
    ) {
        let (h, w) = self.size();
        let stride = self.stride();
        let lst_row = stride*(h - 1);
        for lin_ix in (0..w)
            .chain((0..h).map(move |r| stride*r + w ) )
            .chain((0..w).map(move |c| lst_row + c ))
            .chain((0..h).map(move |r| stride*r )) {
            f(self.linear_index_mut(lin_ix));
        }
    }

    pub fn pixels_mut<'a, 'b>(
        &'b mut self, 
        spacing : usize
    ) -> impl Iterator<Item=&'a mut P> + 'b {
        self.rows_mut().step_by(spacing).map(move |r| r.iter_mut().step_by(spacing) ).flatten()
    }

    pub fn labeled_pixels_mut<'a>(
        &'a mut self, 
        spacing : usize
    ) -> impl Iterator<Item=(usize, usize, &'a mut P)> +'a {
        let w = self.width();
        self.pixels_mut(spacing)
            .enumerate()
            .map(move |(ix, px)| {
                let (r, c) = (ix / w, ix % w);
                (r, c, px)
            })
    }

    pub fn conditional_fill(&mut self, mask : &Window<u8>, color : P) {
        assert!(self.shape() == mask.shape());

        if self.pixel_is::<u8>() {
            unsafe {
                let ans = crate::foreign::ipp::ippi::ippiSet_8u_C1MR(
                    *std::mem::transmute::<_,&u8>(&color),
                    std::mem::transmute(self.as_mut_ptr()),
                    self.byte_stride() as i32,
                    self.size().into(),
                    mask.as_ptr(),
                    mask.byte_stride() as i32
                );
                assert!(ans == 0);
                return;
            }
        }

        self.pixels_mut(1).zip(mask.pixels(1)).for_each(|(d, m)| if *m != 0 { *d = color } );
    }

    
}

impl<'a, P> Window<'a, P> 
where
    &'a [P] : Storage<P>,
    P : Pixel
{

    /// Behaves like split_windows, except that it takes the window by value.
    /// Useful for recursion algorithms, that cannot assume the lifetime of
    /// 'window is the same as its slice.
    pub fn split_equivalent_windows(
        self, 
        num_rows : usize, 
        num_cols : usize
    ) -> impl Iterator<Item=Window<'a, P>> 
    {
        assert!(self.height() % num_rows == 0 && self.width() % num_cols == 0);
        self.clone().split_windows((self.height() / num_rows, self.width() / num_cols))
    }
    
    /// Behaves like windows, except that it takes the window by value.
    /// Useful for recursion algorithms, that cannot assume the lifetime of
    /// 'window is the same as its slice.
    pub fn split_windows(self, sz : (usize, usize)) -> impl Iterator<Item=Window<'a, P>> 
    {
        assert_nonzero(sz);
        // let (step_v, step_h) = sz;
        if sz.0 >= self.sz.0 || sz.1 >= self.sz.1 {
            panic!("Child window size bigger than parent window size");
        }
        if self.height() % sz.0 != 0 || self.width() % sz.1 != 0 {
            panic!("Image size should be a multiple of window size (Required window {:?} over parent window {:?})", sz, self.sz);
        }
        let offset = self.offset;
        iter::WindowIterator::<P> {
            source : self,
            size : sz,
            curr_pos : Some((0, 0)),
        }
    }
    
}

pub struct WindowIterator<'a, N>
where
    N : Scalar,
{
    pub(crate) source : Window<'a, N>,

    // This child window size
    pub(crate) size : (usize, usize),

    // Index the parent window.
    pub(crate) curr_pos : Option<(usize, usize)>,


}

impl<'a, N> Iterator for WindowIterator<'a, N>
where
    N : Scalar + Copy //+ Clone + Copy + Serialize + Zero + From<u8>
{

    type Item = Window<'a, N>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut old_offset = self.curr_pos?;
        self.curr_pos = iterate_next_pos(
            &self.source, 
            old_offset,
            self.size, 
        );
        let src_offset = self.source.offset();
        Some(Window {
            offset : (src_offset.0 + old_offset.0, src_offset.1 + old_offset.1),
            sz : self.size, 
            width : self.source.width, 
            slice : index::sub_slice(self.source.slice.as_ref(), old_offset, self.size, self.source.width),
            _px : PhantomData
        })
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
    pub(crate) curr_pos : Option<(usize, usize)>,

}

impl<'a, N> Iterator for WindowIteratorMut<'a, N>
where
    N : Scalar + Copy + Default
{

    type Item = WindowMut<'a, N>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut old_offset = self.curr_pos?;
        self.curr_pos = iterate_next_pos(
            &self.source, 
            old_offset,
            self.size
        );
        let src_offset = self.source.offset();
        let src_sub = index::sub_slice_mut(self.source.slice.as_mut(), old_offset, self.size, self.source.width);
        Some(WindowMut {
            offset : (src_offset.0 + old_offset.0, src_offset.1 + old_offset.1),
            sz : self.size, 
            width : self.source.width, 
            slice : unsafe { std::slice::from_raw_parts_mut(src_sub.as_mut_ptr(), src_sub.len()) },
            _px : PhantomData
        })
    }

}

/*#[test]
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
}*/

pub fn iterate_row_wise<N>(
    src : &[N],
    // offset : (usize, usize),
    win_sz : (usize, usize),
    orig_sz : (usize, usize),
    row_spacing : usize
) -> impl Iterator<Item=&N> + Clone {
    // let start = orig_sz.1 * offset.0 + offset.1;
    (0..win_sz.0).step_by(row_spacing).map(move |i| unsafe {
        let row_offset = /*start +*/ i*orig_sz.1;
        src.get_unchecked(row_offset..(row_offset+win_sz.1))
    }).flatten()
}

fn iterate_next_pos<P, S>(
    s : &Image<P, S>,
    mut curr_pos : (usize, usize),
    sub_size : (usize, usize)
) -> Option<(usize, usize)>
{
    let parent_sz = s.size();
    let within_horiz = curr_pos.1 + sub_size.1 < parent_sz.1;
    if within_horiz {
        curr_pos.1 += sub_size.1;
        Some(curr_pos)
    } else {
        let within_vert = curr_pos.0 + sub_size.0 < parent_sz.0;
        curr_pos.0 += sub_size.0;
        curr_pos.1 = 0;
        if within_vert {
            Some(curr_pos)
        } else {
            None
        }
    }
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

// Holds an array of neighboring pixels and the current index at the iterator.
pub enum PixelNeighborhood<N>
where
    N : Copy
{

    Corner([N; 3], usize),

    Edge([N; 5], usize),

    Full([N; 8], usize)

}

fn walk_neighborhood<N>(pxs : &[N], pos : &mut usize) -> Option<N>
where
    N : Copy
{
    let ans = pxs.get(*pos).copied();
    if ans.is_some() {
        *pos += 1;
    }
    ans
}

impl<N> Iterator for PixelNeighborhood<N>
where
    N : Copy
{

    type Item = N;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Corner(pxs, ref mut pos) => {
                walk_neighborhood(&pxs[..], pos)
            },
            Self::Edge(pxs, pos) => {
                walk_neighborhood(&pxs[..], pos)
            },
            Self::Full(pxs, pos) => {
                walk_neighborhood(&pxs[..], pos)
            },
        }
    }

}

fn top_left(pos : (usize, usize)) -> (usize, usize) {
    (pos.0 - 1, pos.1 - 1)
}

fn top_right(pos : (usize, usize)) -> (usize, usize) {
    (pos.0 - 1, pos.1 + 1)
}

fn bottom_left(pos : (usize, usize)) -> (usize, usize) {
    (pos.0 + 1, pos.1 - 1)
}

fn bottom_right(pos : (usize, usize)) -> (usize, usize) {
    (pos.0 + 1, pos.1 + 1)
}

fn top(pos : (usize, usize)) -> (usize, usize) {
    (pos.0 - 1, pos.1)
}

fn bottom(pos : (usize, usize)) -> (usize, usize) {
    (pos.0 + 1, pos.1)
}

fn left(pos : (usize, usize)) -> (usize, usize) {
    (pos.0, pos.1 - 1)
}

fn right(pos : (usize, usize)) -> (usize, usize) {
    (pos.0, pos.1 + 1)
}

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

pub struct WindowNeighborhood<'a, N>
where
    N : Scalar
{
    pub center : Window<'a, N>,
    pub left : Window<'a, N>,
    pub top : Window<'a, N>,
    pub right : Window<'a, N>,
    pub bottom : Window<'a, N>,
}

#[test]
fn window_iterator() {
    let v : Vec<_> = (0..1024).map(|i| i as u8 ).collect();
    let img = ImageBuf::from_vec(v, 32);
    let w = img.window((15, 15), (16, 16)).unwrap();
    for w in w.windows((4, 4)) {
        println!("{:?}", w.offset());
    }
}

// cargo test --lib -- window_iterator2 --nocapture
#[test]
fn window_iterator2() {
    let img = ImageBuf::new_constant(128, 128, 0);
    for w in img.windows((16, 128)) {
        println!("{:?} {:?}", w.offset(), w.size());
    }
}
