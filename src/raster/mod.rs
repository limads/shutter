use crate::image::*;
use nalgebra::Scalar;
use std::cmp::Ordering;
use std::convert::AsRef;
use std::collections::HashMap;

// TODO create struct Boundary, to represent a contour that is not necessarily ordered
// (which cannot necessarily be drawn by tracing the positions). If you aim to just calcualte statistics
// such as central moments or view the segment limits, the boundary gives the same
// results as the contour without the computational cost of re-ordering the array.
// Every contour is also a boundary, so impl From<Contour> for Boundary is a simple vector move.

#[derive(Debug, Clone)]
pub struct Contour(Vec<(usize, usize)>);

impl Contour {

    // Since the RasterSegmenter organizes the image contour
    //
    //    r1 l1
    //   l2   r2
    //  l3      r3
    //
    // as l3 l2 l1 r1 r2 r3, we can iterate over the inner pixels without furter
    // calculations by just iterating over this natural order from the middle
    // to the extremities.
    pub fn inner_pixels<'a>(&'a self, win : &'a Window<'a, u8>) -> impl Iterator<Item=u8> +'a {
        let n = self.0.len();
        assert!(n % 2 == 0);
        (0..(n/2+1)).rev().zip((n/2+1)..n)
            .map(move |(left, right)| { assert!(self.0[left].0 == self.0[right].0); win.row(self.0[left].0).unwrap()[self.0[left].1..self.0[right].1].iter().copied() })
            .flatten()
    }

}

impl AsRef<[(usize, usize)]> for Contour {

    fn as_ref(&self) -> &[(usize, usize)] {
        &self.0
    }

}

/// Holds a run-length encoding (RLE) of an image. Each inner vector at index i
/// represents row i of the image. The elements of the inner vectors represent
/// the start column, end column and a dominant pixel color of the raster scan,
/// whose value depend on the rule adopted: the first rule used the color of the first
/// pixel; the central rule the color of the center-most pixel; the mean rule the
/// running average of the pixel color values (at a slightly higher computational cost).
#[derive(Debug, Clone)]
pub struct Raster(Vec<Vec<(usize, usize, u8)>>);

fn raster_overlap(a : (usize, usize), b : (usize, usize)) -> bool {
    /*(a.0 <= b.0 && a.1 >= b.0) ||
        (a.0 <= b.1 && a.1 >= b.1) ||
        (a.0 >= b.0 && a.1 <= b.1) ||
        (a.0 <= b.0 && a.1 >= b.1)*/
    !((a.0 < b.0 && a.1 < b.1) || (a.0 > b.0 && a.1 > b.1))
}

#[derive(Debug, Clone)]
pub struct OpenContour {
    pxs : Vec<(usize, usize)>,
    fst_col_last_row : usize,
    lst_col_last_row : usize,
    last_row : usize,
    col : u8
}

fn innaugurate_contour(contours : &mut Vec<OpenContour>, row : usize, c1 : usize, c2 : usize, color : u8) {
    let mut pxs = Vec::new();
    pxs.push((row, c1));
    if c1 != c2 {
        pxs.push((row, c2));
    }
    contours.push(OpenContour {
        fst_col_last_row : pxs[0].1,
        lst_col_last_row : pxs.last().unwrap().1,
        col : color,
        pxs,
        last_row : row
    });
}

fn overlap_open_contour(open : &OpenContour, this_c1 : usize, this_c2 : usize, this_color : u8, abs_diff : u8) -> Ordering {
    let overlaps = raster_overlap((open.fst_col_last_row, open.lst_col_last_row), (this_c1, this_c2));
    if overlaps {
        Ordering::Equal
    } else {

        // Compare by first. Just as same as comparing by last.
        open.fst_col_last_row.cmp(&this_c1)
    }
}

impl Raster {

    pub fn new(height : usize) -> Self {
        Raster((0..height).map(|_| Vec::<(usize, usize, u8)>::new() ).collect())
    }

    pub fn segments(&self) -> &[Vec<(usize, usize, u8)>] {
        &self.0
    }

    /*pub fn calculate_limits(&mut self, win : &Window<'_, u8>, low : u8, high : u8) {
        assert!(win.height() == self.0.len());
        self.0.iter_mut().for_each(|row| row.clear() );
        for (r, c, px) in win.labeled_pixels::<usize, _>(1) {
            if px > low && px < high {
                if self.0[r].len() == 0 {
                    self.0[r].push((c, c, px));
                }

                } else {
                    if (px as i16 - self.0[r].last().unwrap().2 as i16).abs() <= abs_diff as i16 {
                        self.0[r].last_mut().unwrap().1 += 1;

                        // TODO perhaps set color to the mean to improve future matches.
                        // Also keep a pixel sum for each raster to calculate the mean more easily.
                    } else {
                        self.0[r].push((c, c, px));
                    }
                }
            }
        }
    }*/

    // The mean rule updates the color of each raster segment to the current running average,
    // and matches any further pixels to this value. We could also calculate a running variance.
    // This makes for more robust matches, since it avoids blurred pixels at the edges of dominating
    // the color of the patch. A center-most pixel rule could also be adopted.
    pub fn calculate(&mut self, win : &Window<'_, u8>, mean : bool, abs_diff : u8) {
        assert!(win.height() == self.0.len());
        self.0.iter_mut().for_each(|row| row.clear() );

        let mut sum : f32 = 0.;
        for (r, c, px) in win.labeled_pixels::<usize, _>(1) {
            if c == 0 {
                self.0[r].push((c, c, px));
                sum = px as f32;
            } else {

                let color_match = (px as i16 - self.0[r].last().unwrap().2 as i16).abs() <= abs_diff as i16;
                if color_match {

                    self.0[r].last_mut().unwrap().1 += 1;

                    // Update color with the average. If mean=false, will use color of first pixel.
                    // An alternative would be to use color of the central pixel, to avoid edge
                    // pixels determining the patch color.

                    sum += px as f32;
                    let n = (self.0[r].last().unwrap().1 - self.0[r].last().unwrap().0 + 1) as f32;
                    if mean {
                        self.0[r].last_mut().unwrap().2 = (sum / n) as u8;
                    }

                } else {
                    self.0[r].push((c, c, px));
                    sum = px as f32;
                }
            }
        }

        assert!(self.0.len() == win.height());
        for row in self.0.iter() {
            assert!(row.len() >= 1);
        }
    }

    pub fn contours(&self, abs_diff : u8) -> (Vec<Contour>, Vec<u8>) {

        // Holds (row, col, color)
        let mut open_contours : Vec<OpenContour> = Vec::new();

        // Holds Contour
        let mut closed_contours : Vec<Contour> = Vec::new();
        let mut closed_colors : Vec<u8> = Vec::new();

        // Innaugurate contours for first line.
        for (c1, c2, col) in &self.0[0] {
            innaugurate_contour(&mut open_contours, 0, *c1, *c2, *col);
        }

        open_contours.sort_by(|a, b| a.fst_col_last_row.cmp(&b.fst_col_last_row) );

        let mut matched_contours : Vec<usize> = Vec::new();
        let mut new_contours : Vec<OpenContour> = Vec::new();

        // Carries index of opened patch and the new (start, end) columns.
        let mut editions : HashMap<usize, (usize, usize)> = HashMap::new();

        for r in 1..(self.0.len()) {

            // Iterate over this patch.
            for (this_c1, this_c2, this_color) in &self.0[r] {

                assert!(open_contours.iter().all(|c| c.last_row == r-1 || c.last_row == r ));

                // Take any of the overlaps
                let res_prev_ix = open_contours.binary_search_by(|open| {
                    overlap_open_contour(open, *this_c1, *this_c2, *this_color, abs_diff)
                });

                if let Ok(cand_ix) = res_prev_ix {

                    // The binary search returns one of the many possible overlaps. We search within
                    // all possible overlaps for the left-most one that also matches the color.
                    let mut prev_ix = cand_ix;

                    // Start from right-most overlap
                    while let Some(open) = open_contours.get(prev_ix+1) {
                        if raster_overlap((open.fst_col_last_row, open.lst_col_last_row), (*this_c1, *this_c2)) {
                            prev_ix +=1 ;
                        } else {
                            break;
                        }
                    }

                    // Iterate backward to get the first overlap that also matches color.
                    let mut search_back = 0;
                    let mut best_offset = 0;
                    let mut matched_color = false;
                    while let Some(prev) = prev_ix.checked_sub(search_back) {
                        if let Some(open) = open_contours.get(prev) {
                            if raster_overlap((open.fst_col_last_row, open.lst_col_last_row), (*this_c1, *this_c2)) {
                                let color_match = (open_contours[prev].col as i16 - *this_color as i16).abs() <= abs_diff as i16;
                                if color_match {
                                    best_offset = search_back;
                                    matched_color = true;
                                }
                                search_back += 1;
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                    prev_ix = prev_ix - best_offset;

                    // Case this top segment has been matched to a previous bottom segment
                    // (which will happen when we have a larger top raster segment and two smaller
                    // bottom segments), continue the top segment only with the first match, and
                    // innaugurate a new segment for the current segment.
                    if matched_color {
                        if editions.contains_key(&prev_ix) {
                            innaugurate_contour(&mut new_contours, r, *this_c1, *this_c2, *this_color);
                        } else {
                            editions.insert(prev_ix, (*this_c1, *this_c2));
                            matched_contours.push(prev_ix);
                        }
                    } else {
                        innaugurate_contour(&mut new_contours, r, *this_c1, *this_c2, *this_color);
                    }
                } else {

                    println!("{:?} had no overlaps", (r, this_c1, this_c2));

                    // Innaugurate new contour. Insert at another vector so they won't be matched now
                    // and mess with the indices of OpenContours.
                    innaugurate_contour(&mut new_contours, r, *this_c1, *this_c2, *this_color);

                }
            }

            for (ix, (c1, c2)) in editions.drain() {

                // TODO just push pixels to the end of vector, and then
                // at the end of contour tracing
                // swap odd entries at 0..n/2 to n/2..n to avoid moving
                // so much data.

                // Insert new left col at beginning to preserve contour order
                open_contours[ix].pxs.insert(0, (r, c1));

                // Insert new right col at end to preserve contour order
                open_contours[ix].pxs.push((r, c2));

                open_contours[ix].last_row = r;
                open_contours[ix].fst_col_last_row = c1;
                open_contours[ix].lst_col_last_row = c2;
            }

            // Close all contours that weren't matched at this row.
            let unmatched_contours = (0..open_contours.len())
                .filter(|ix| matched_contours.iter().find(|c| *c == ix ).is_none() );

            // Reverse the iterator and remove from the end to the start so
            // as to not mess up with the indices.
            for ix in unmatched_contours.rev() {
                let open = open_contours.remove(ix);
                closed_colors.push(open.col);
                closed_contours.push(Contour(open.pxs));
            }

            for contour in new_contours.drain(..) {
                open_contours.push(contour);
            }

            open_contours.sort_by(|a, b| a.fst_col_last_row.cmp(&b.fst_col_last_row) );

            /*// Verify if there are any "holes" generated by the bottom raster segments
            // at the bottom line when they match the top segments. We should create new segments
            // to fill the holes in those cases.
            if let Some(fst) = open_contours.first() {

            }
            for (open_left, open_right)
            if let Some(lst) = open_contours.last() {

            }

            open_contours.sort_by(|a, b| a.fst_col_last_row.cmp(&b.fst_col_last_row) );*/

            matched_contours.clear();
        }

        // Push remaining contours after all rows are finished
        for open in open_contours {
            closed_colors.push(open.col);
            closed_contours.push(Contour(open.pxs));
        }

        // TODO rearrange pixel order here.

        (closed_contours, closed_colors)
    }

}

// use packed_simd::*;

fn assert_dims(dims : &[(usize, usize)]) {
    let s1 = dims[0];
    assert!(dims.iter().all(|s| *s == s1 ))
}

fn assert_multiple(vals : &[usize], by : usize) {
    assert!(vals.iter().all(|v| v % by == 0 ))
}

fn assign_row_iter<'a, N>(
    out : &'a mut WindowMut<'a, N>, 
    a : &'a Window<'a, N>, 
    b : &'a Window<'a, N>
) -> impl Iterator<Item=(&'a mut [N], (&'a [N], &'a [N]))> 
    where N : Scalar + Copy
{
    out.rows_mut().zip(a.rows().zip(b.rows()))
}

/*fn simd_iter(out : &mut WindowMut<'_, u8>, a : &Window<'_, u8>, b : &Window<'_, u8>) -> impl Iterator<Item=(&mut u8x4, &u8x4, &u8x4)> {
    let row_iter = assign_row_iter(out, a, b);
    row_iter.map(|out, (a, b)| {
        (
            &mut out.chunks_exact_mut(4).map(u8x4::from_slice_unaligned),
            &a.chunks_exact(4).map(u8x4::from_slice_unaligned),
            &b.chunks_exact(4).map(u8x4::from_slice_unaligned)
        )
    }).flatten()
}

pub fn sub_mut_u8(out : &mut WindowMut<'_, u8>, a : &Window<'_, u8>, b : &Window<'_, u8>) {
    assert_dims(&[a.shape(), b.shape(), out.shape()]);
    assert_multiple(&[out.width(), a.width(), b.width()], 4);
    for (out_v, a_v, b_v) in simd_iter(out, a, b) {
        *out_v = a_v - b_v
    }
}*/

pub fn sub_mut_u8<'a>(out : &'a mut WindowMut<'a, u8>, a : &'a Window<'a, u8>, b : &'a Window<'a, u8>) {
    assert_dims(&[a.shape(), b.shape(), out.shape()]);
    assert_multiple(&[out.width(), a.width(), b.width()], 4);
    for (out, (a, b)) in assign_row_iter(out, a, b) {
        for (px_out, (px_a, px_b)) in out.iter_mut().zip(a.iter().zip(b.iter())) {
            *px_out = px_a - px_b;
        }
    }
}

// Draws each raster with its color, and two black points delimiting each raster.
fn draw_segments(raster : &Raster, mut img : WindowMut<'_, u8>) {

    use crate::image::Mark;

    for (r, segs) in raster.segments().iter().enumerate() {
        for s in segs {
            img.draw(crate::image::Mark::Line((r, s.0), (r, s.1), s.2));
            img[(r,s.0)] = 0;
            img[(r,s.1)] = 0;
        }
    }
}
