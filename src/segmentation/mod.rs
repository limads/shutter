// use either::Either;
use std::cmp::{Eq, PartialEq};
use itertools::Itertools;
use crate::image::{Image, Window, WindowMut};
use bayes::fit::{cluster::KMeans, cluster::KMeansSettings, Estimator};
use crate::shape::*;
use std::fmt;
use std::collections::HashMap;
use std::mem;
use std::default::Default;

// #[cfg(feature="opencvlib")]
// pub mod fgmm;

// #[cfg(feature="opencvlib")]
// pub mod mser;

/// The most general patch is a set of pixel positions with a homogeneous color
/// and a scale that was used for extraction. The patch is assumed to be
/// homonegeous within a pixel spacing given by the scale field.
#[derive(Clone, Debug, Default)]
pub struct Patch {
    // Instead of storing all pixels, we can store a rect and only
    // the outer pixels. The rect is just characterized by a top-left
    // corner and an extension. The rect extension is increased
    // any time we have a set of inserted pixels that comptetely border
    // either its bottom or right borders.
    pub pxs : Vec<(usize, usize)>,

    // Outer rect, at patch scale
    pub outer_rect : (usize, usize, usize, usize),
    pub color : u8,
    pub scale : usize
}

impl Patch {

    /// Starts a new patch, optionally with a pixel buffer to be recycled.
    pub fn new(pt : (usize, usize), color : u8, scale : usize, pxs : Option<Vec<(usize, usize)>>) -> Self {
        let pxs = if let Some(mut pxs) = pxs {
            pxs.clear();
            pxs.push(pt);
            pxs
        } else {
            let mut pxs = Vec::with_capacity(16);
            pxs.push(pt);
            pxs
        };
        Self {
            pxs,
            outer_rect : (pt.0, pt.1, 1, 1),
            color,
            scale
        }
    }

    /// Outer rect, at image scale
    pub fn outer_rect(&self) -> (usize, usize, usize, usize) {
        (
            self.outer_rect.0 * self.scale,
            self.outer_rect.1 * self.scale,
            self.outer_rect.2 * self.scale,
            self.outer_rect.3 * self.scale
        )
    }

    // Use short-circuit to only iterate over pixels for verification when absolutely required.
    // Note: Outer rect right border should equal position of new pixel, since outer rect starts with
    // size 1.
    pub fn pixel_is_below(&self, (r, c) : (usize, usize)) -> bool {
        // println!("rect={}; r = {}", self.outer_rect.0 + self.outer_rect.2, r);
        let rect_r = self.outer_rect.0 + self.outer_rect.2;

        // First is true when first pixel at row is being inserted; second after.
        let is_below = (rect_r == r || rect_r == r+1);

        // assert!(self.pxs.is_sorted_by(|a, b| a.0.partial_cmp(&b.0 )));

        r >= 1 &&
            is_below &&
            self.outer_rect.1 <= c &&
            self.outer_rect.1 + self.outer_rect.3 >= c &&
            self.pxs.iter().rev().take_while(|px| px.0 >= r-1 ).any(|px| px.0 == r-1 && px.1 == c )
    }

    pub fn pixel_is_right(&self, (r, c) : (usize, usize)) -> bool {
        // println!("rect={}; c= {}", self.outer_rect.0 + self.outer_rect.2, c);
        let rect_c = self.outer_rect.1 + self.outer_rect.3;

        // First is true when first pixel at col is being inserted; second after.
        let is_right = (rect_c == c || rect_c == c+1);
        c >= 1 &&
            is_right &&
            self.outer_rect.0 <= r &&
            self.outer_rect.0 + self.outer_rect.2 >= r &&
            self.pxs.iter().rev().any(|px|  px.0 == r && px.1 == c-1 )
    }

    pub fn expand(&mut self, pts : &[(usize, usize)]) {
        for pt in pts.iter() {
            self.pxs.push(*pt);
            if pt.0 < self.outer_rect.0 {
                self.outer_rect.0 = pt.0;
                self.outer_rect.2 = (pt.0 - self.outer_rect.0)+1;
            }
            if pt.1 < self.outer_rect.1 {
                self.outer_rect.1 = pt.1;
                self.outer_rect.3 = (pt.1 - self.outer_rect.1)+1;
            }
            let new_h = (pt.0 - self.outer_rect.0)+1;
            let new_w = (pt.1 - self.outer_rect.1)+1;
            if new_h > self.outer_rect.2 {
                self.outer_rect.2 = new_h;
            }
            if new_w > self.outer_rect.3 {
                self.outer_rect.3 = new_w;
            }

            // When adding more than one point, we are merging two
            // patches. In this case, we must guarantee that row order
            // is preserved. When adding a single point, we are in
            // raster order insertion. Even if we are merging left patch
            // to right and left has one element, then left is guranteed
            // to be in row order to top, since they will be different.
            if pts.len() > 1 {
                self.pxs.sort_unstable_by(|a, b| a.0.cmp(&b.0) );
            }

            //println!("")
        }
    }

    pub fn same_color(&self, other : &Self) -> bool {
        self.color == other.color
    }

    /// Maps rows to a set of columns
    pub fn group_rows(&self) -> HashMap<usize, Vec<usize>> {
        let mut row_pxs = HashMap::new();
        for (row, pxs) in self.pxs.iter().group_by(|px| px.0 ).into_iter() {
            row_pxs.insert(row, pxs.map(|px| px.1 ).collect::<Vec<_>>());
        }
        row_pxs
    }

    pub fn num_regions(&self) -> usize {
        let row_pxs = self.group_rows();
        let mut n_regions = 0;
        for (_, cols) in row_pxs {
            n_regions += cols.len();
        }
        n_regions
    }

    // pub fn area(&self) -> usize {
    //    self.num_regions() * self.scale.pow(2)
    // }

    pub fn polygon(&self) -> Option<Polygon> {
        let mut row_pxs = self.group_rows();
        let mut sorted_keys = row_pxs.iter().map(|(k, _)| k ).collect::<Vec<_>>();
        if sorted_keys.len() < 3 {
            return None;
        }
        sorted_keys.sort();
        let n = sorted_keys.len();
        let mut pts : Vec<(usize, usize)> = Vec::new();

        // Points with "top" part of the patch
        let fst_row = sorted_keys[0];
        for col in row_pxs[fst_row].iter() {
            pts.push((*fst_row * self.scale, *col * self.scale));
        }

        // Points with "right" part of the patch
        for row in sorted_keys[1..n-1].iter() {
            pts.push((**row * self.scale, *row_pxs[row].last().unwrap() * self.scale));
        }

        // Points with "bottom" part of the patch
        let last_row = sorted_keys.last().unwrap();
        for col in row_pxs[last_row].iter().rev() {
            pts.push((**last_row * self.scale, *col * self.scale));
        }

        // Points with "left" part of the patch
        for row in sorted_keys[1..n-1].iter().rev() {
            pts.push((**row * self.scale, *row_pxs[row].first().unwrap() * self.scale));
        }

        Some(Polygon::from(pts))
    }

}

/// Unlike a Polygon, which has a precise mathematical description as a set of delimiting points,
/// a patch is an amorphous set of pixels known to have a certain color. To occupy a reasonable size,
/// patches are usually calculated by subsampling the image, and verifying the pixels closest to one of a few colors.
/// Neighborhoods might be superimposed to one-another. If image is subsampled by 2, scale will be two, and
/// all pixel coordinates at neighborhood should be multiplied by this value. Neighborhoods are each a 3x3 image
/// region, in raster order, each satisfying merges_with(ix-1) (left neighborhood) and/or merges_with(ix-ncol) (top neighborhood)
#[derive(Clone, Debug)]
pub struct BinaryPatch {

    // pub win : &'a Window<u8>,
    pub neighborhoods : Vec<Neighborhood>,

    pub color : u8,

    pub scale : u8
}

impl BinaryPatch {

    pub fn inner_polygon(&self) -> Polygon {
        unimplemented!()
    }

    pub fn outer_polygon(&self) -> Polygon {
        unimplemented!()
    }

    /// If the neighborhoods composing this patch are are at scale k,
    /// verifies if they can be simplified to have fewer neighborhoods
    /// at scale k+1.
    pub fn simplify(&mut self) {

    }

}

/// Local neighborhood, representing equality state between a center pixel and
/// its spacing=1 neighbors.
#[derive(Clone, Debug)]
pub struct Neighborhood {

    pub center : (usize, usize),

    pub color : u8,

    /// Whether outer pixels of the patch are equal to center starting from top-left and going row-wise,
    /// ignoring the center pixel:
    /// | 0 | 1 | 2 |
    /// | 3 | X | 4 |
    /// | 5 | 6 | 7 |
    pub pattern : [bool; 8]

}

fn symbol(has_px : bool) -> &'static str {
    if has_px {
        "X"
    } else {
        " "
    }
}

impl fmt::Display for Neighborhood {

    fn fmt(&self, f : &mut fmt::Formatter) -> Result<(), fmt::Error> {
        let mut layout = String::new();
        layout += &format!("|{}|{}|{}|\n", symbol(self.pattern[0]), symbol(self.pattern[1]), symbol(self.pattern[2]));
        layout += &format!("|{}|{}|{}|\n", symbol(self.pattern[3]), "X", symbol(self.pattern[4]));
        layout += &format!("|{}|{}|{}|\n", symbol(self.pattern[5]), symbol(self.pattern[6]), symbol(self.pattern[7]));
        write!(f, "{}", layout)
    }
}

impl Neighborhood {

    pub fn any_left(&self) -> bool {
        self.pattern[0] || self.pattern[3] || self.pattern[5]
    }

    pub fn all_left(&self) -> bool {
        self.pattern[0] && self.pattern[3] && self.pattern[5]
    }

    pub fn any_top(&self) -> bool {
        self.pattern[0] || self.pattern[1] || self.pattern[2]
    }

    pub fn all_top(&self) -> bool {
        self.pattern[0] && self.pattern[1] && self.pattern[2]
    }

    pub fn any_right(&self) -> bool {
        self.pattern[2] || self.pattern[4] || self.pattern[7]
    }

    pub fn any_bottom(&self) -> bool {
        self.pattern[5] || self.pattern[6] || self.pattern[7]
    }

    pub fn all_bottom(&self) -> bool {
        self.pattern[5] && self.pattern[6] && self.pattern[7]
    }

    pub fn any_horizontal_center(&self) -> bool {
        self.pattern[3] || self.pattern[4]
    }

    pub fn all_horizontal_center(&self) -> bool {
        self.pattern[3] && self.pattern[4]
    }

    pub fn any_vertical_center(&self) -> bool {
        self.pattern[1] || self.pattern[6]
    }

    pub fn all_vertical_center(&self) -> bool {
        self.pattern[1] && self.pattern[6]
    }

    pub fn all_right(&self) -> bool {
        self.pattern[2] && self.pattern[4] && self.pattern[7]
    }

    pub fn horizontal_extension(&self) -> u8 {
        if self.all_top() || self.all_horizontal_center() || self.all_bottom() {
            3
        } else {
            let top_2 = (self.pattern[0] && self.pattern[1]) || (self.pattern[1] && self.pattern[2]);
            let center_2 = self.pattern[3] || self.pattern[4];
            let bottom_2 = (self.pattern[5] && self.pattern[6]) || (self.pattern[6] && self.pattern[7]);
            if top_2 || center_2 || bottom_2 {
                2
            } else {
                1
            }
        }
    }

    pub fn vertical_extension(&self) -> u8 {
        if self.all_left() || self.all_vertical_center() || self.all_right() {
            3
        } else {
            let left_2 = (self.pattern[0] && self.pattern[3]) || (self.pattern[3] && self.pattern[5]);
            let center_2 = self.pattern[1] || self.pattern[6];
            let right_2 = (self.pattern[2] && self.pattern[4]) || (self.pattern[4] && self.pattern[7]);
            if left_2 || center_2 || right_2 {
                2
            } else {
                1
            }
        }
    }

    pub fn left_border(&self) -> [bool; 3] {
        [self.pattern[0], self.pattern[3], self.pattern[5]]
    }

    pub fn top_border(&self) -> [bool; 3] {
        [self.pattern[0], self.pattern[1], self.pattern[2]]
    }

    pub fn right_border(&self) -> [bool; 3] {
        [self.pattern[2], self.pattern[4], self.pattern[7]]
    }

    pub fn bottom_border(&self) -> [bool; 3] {
        [self.pattern[5], self.pattern[6], self.pattern[7]]
    }

    pub fn merges_left(&self, other : &Self) -> bool {
        let lb = self.left_border();
        let border_iter = lb.iter().zip(other.right_border());
        self.color == other.color &&
            self.left_border().iter().any(|l| *l ) &&
            border_iter.clone().all(|(l, r)| *l && r )
    }

    pub fn merges_top(&self, other : &Self) -> bool {
        let tb = self.top_border();
        let border_iter = tb.iter().zip(other.bottom_border());
        self.color == other.color &&
            self.top_border().iter().any(|l| *l ) &&
            border_iter.clone().all(|(t, b)| *t && b )
    }

}

pub struct PatchSegmentation {
    px_spacing : usize,
    patches : Vec<Patch>
}

impl PatchSegmentation {

    pub fn new(px_spacing : usize) -> Self {
        Self { patches : Vec::new(), px_spacing }
    }

    pub fn segment<'a>(&'a mut self, win : &WindowMut<'_, u8>) -> &'a [Patch] {
        /*let src_win = unsafe {
            Window {
                offset : win.offset(),
                orig_sz : win.orig_sz(),
                win_sz : win.shape(),
                win : std::slice::from_raw_parts(win.full_slice().as_ptr(), win.full_slice().len()),
            }
        };*/
        let src_win = unsafe { crate::image::create_immutable(&win) };
        color_patches(&mut self.patches, &src_win, self.px_spacing);
        &self.patches[..]
    }

}

/// During pixel insertion in a patch, row raster order is preserved, but column raster order is not.
pub fn color_patches(patches : &mut Vec<Patch>, win : &Window<'_, u8>, px_spacing : usize) {

    // Recycle previous pixel vectors to avoid reallocations
    let mut prev_pxs : Vec<Vec<(usize, usize)>> = patches
        .iter_mut()
        .map(|mut patch| mem::take(&mut patch.pxs ) )
        .collect();
    patches.clear();

    let (ncol, nrow) = (win.width() / px_spacing, win.height() / px_spacing);

    // Maps each column to a patch index
    // let mut prev_row_patch_ixs : HashMap<usize, usize> = HashMap::with_capacity(ncol);
    let mut prev_row_patch_ixs : Vec<usize> = Vec::with_capacity(ncol);
    let mut left_patch_ix : Option<usize> = None;

    for c in (0..ncol) {
        //prev_row_patch_ixs.insert(c, 0);
        prev_row_patch_ixs.push(0);
    }

    for (ix, px) in win.pixels(px_spacing).enumerate() {
        let (r, c) = (ix / ncol, ix % ncol);
        let color = win[(r*px_spacing, c*px_spacing)];

        let might_merge_top = if r >= 1 {
            color == win[((r-1)*px_spacing, c*px_spacing)]
        } else {
            false
        };
        let might_merge_left = if c >= 1 {
            color == win[(r*px_spacing, (c-1)*px_spacing)]
        } else {
            false
        };

        if c == 0 {
            left_patch_ix = None;
        }

        // println!("{},{}", r, c);
        // println!("{:?}", patches);

        /*// Get patch that contains the pixel above the current pixel
        // let top_patch_ix = patches.iter().position(|patch| patch.pxs.iter().any(|px| r >= 1 && px.0 == r-1 && px.1 == c ) );
        let top_patch_ix = patches.iter().rev()
            .position(|patch| patch.pixel_is_below((r, c)) )
            .map(|inv_pos| patches.len() - 1 - inv_pos);*/
        let top_patch_ix = if r >= 1 { Some(prev_row_patch_ixs[c]) } else { None };
        println!("{},{}", r, c);
        // Get patch that contains the pixel to the left of current pixel
        // let left_patch_ix = patches.iter().position(|patch| patch.pxs.iter().any(|px| c >= 1 && px.0 == r && px.1 == c-1 ) );
        /*let left_patch_ix = patches.iter().rev()
            .position(|patch| patch.pixel_is_right((r, c)) )
            .map(|inv_pos| patches.len() - 1 - inv_pos);*/
        // let left_patch_ix =

        // Verifies if patches have the same color
        /*let merges_top = might_merge_top && if let Some(top) = top_patch_ix.and_then(|ix| patches.get(ix) ) {
            top.color == color
        } else {
            false
        };
        let merges_left = might_merge_left && if let Some(left) = left_patch_ix.and_then(|ix| patches.get(ix) ) {
            left.color == color
        } else {
            false
        };*/
        let merges_top = might_merge_top && top_patch_ix.is_some();
        let merges_left = might_merge_left && left_patch_ix.is_some();

        match (merges_left, merges_top) {
            (true, true) => {

                let top_differs_left = top_patch_ix.unwrap() != left_patch_ix.unwrap();
                if top_differs_left {
                    // Merge two patches
                    let left_patch = mem::take(&mut patches[left_patch_ix.unwrap()]);
                    // let left_pxs = .pxs.iter().cloned().collect::<Vec<_>>();
                    patches[top_patch_ix.unwrap()].expand(&left_patch.pxs);
                }

                // Push new pixel
                patches[top_patch_ix.unwrap()].expand(&[(r, c)]);
                //*(prev_row_patch_ixs.get_mut(&c).unwrap()) = top_patch_ix.unwrap();
                prev_row_patch_ixs[c] = top_patch_ix.unwrap();
                left_patch_ix = Some(top_patch_ix.unwrap());

                if top_differs_left {
                    // Remove old left patch (now merged with top).
                    patches.swap_remove(left_patch_ix.unwrap());
                    for mut ix in prev_row_patch_ixs.iter_mut() {
                        if *ix == left_patch_ix.unwrap() {
                            *ix = top_patch_ix.unwrap();
                        }
                    }
                }
            },
            (true, false) => {
                patches[left_patch_ix.unwrap()].expand(&[(r, c)]);
                //*(prev_row_patch_ixs.get_mut(&c).unwrap()) = left_patch_ix.unwrap();
                prev_row_patch_ixs[c] = left_patch_ix.unwrap();
                // Left patch is left unchanged
            }
            (false, true) => {
            
                // TODO panicking here
                patches[top_patch_ix.unwrap()].expand(&[(r, c)]);
                //*(prev_row_patch_ixs.get_mut(&c).unwrap()) = top_patch_ix.unwrap();
                prev_row_patch_ixs[c] = top_patch_ix.unwrap();
                left_patch_ix = Some(top_patch_ix.unwrap());
            },
            (false, false) => {
                let opt_pxs = match prev_pxs.len() {
                    0 => None,
                    1 => Some(prev_pxs.remove(0)),
                    _ => Some(prev_pxs.swap_remove(0))
                };
                // println!("Inserted new patch starting from {},{}", r, c);
                patches.push(Patch::new((r, c), color, px_spacing, opt_pxs));
                //*(prev_row_patch_ixs.get_mut(&c).unwrap()) = patches.len() - 1;
                prev_row_patch_ixs[c] = patches.len() - 1;
                left_patch_ix = Some(patches.len() - 1);
            }
        }

        // println!("{:?}", patches.last().unwrap());
    }

}

#[test]
fn test_patches() {
    let check = Image::<u8>::new_checkerboard(16, 4);
    println!("{}", check);
    for patch in check.full_window().patches(1).iter() {
        println!("{:?}", patch);
        println!("{:?}", patch.polygon());
    }
}

/// Returns the color patches of a given labeled window (such as retrurned by segment_colors).
pub fn binary_patches(label_win : &Window<'_, u8>, px_spacing : usize) -> Vec<BinaryPatch> {
    let mut curr_patch = 0;
    let mut neighborhoods : Vec<(Neighborhood, usize)> = Vec::new();
    let (ncol, nrow) = (label_win.width() / px_spacing, label_win.height() / px_spacing);
    assert!(px_spacing % 2 == 0, "Pixel spacing should be an even number");

    // Establish 4-neighborhoods by iterating over non-border allocations.
    for (ix_row, row) in (1..nrow-1).step_by(3).enumerate() {
        for (ix_col, col) in (1..ncol-1).step_by(3).enumerate() {
            let neighborhood = extract_neighborhood(label_win, (row, col));

            let merges_top = if row > 1 {
                let top_neighbor = &neighborhoods[ix_row*ncol + ix_col - ncol].0;
                neighborhood.merges_top(top_neighbor)
            } else {
                false
            };

            let merges_left = if col > 1 {
                let left_neighbor = &neighborhoods[ix_row*ncol + ix_col - 1].0;
                neighborhood.merges_left(left_neighbor)
            } else {
                false
            };

            match (merges_left, merges_top) {
                (true, true) => {
                    let left_neighbor_ix = neighborhoods[ix_row*ncol + ix_col - 1].1;
                    let top_neighbor_ix = neighborhoods[ix_row*ncol + ix_col - ncol].1;

                    // If both are true, and left and top are not merged, this means this
                    // neighborhood is a link element to the top and left patches. Attribute
                    // top neighbor (older) index to all patches matched to left neighbor index.
                    if left_neighbor_ix != top_neighbor_ix {
                        neighborhoods.iter_mut().for_each(|(_, patch_ix)| {
                            if *patch_ix == left_neighbor_ix {
                                *patch_ix = top_neighbor_ix;
                            }
                        });
                    }
                    neighborhoods.push((neighborhood, top_neighbor_ix));
                },
                (true, false) => {
                    let left_neighbor_ix = neighborhoods[ix_row*ncol + ix_col - 1].1;
                    neighborhoods.push((neighborhood, left_neighbor_ix));
                },
                (false, true) => {
                    let top_neighbor_ix = neighborhoods[ix_row*ncol + ix_col - ncol].1;
                    neighborhoods.push((neighborhood, top_neighbor_ix));
                },
                (false, false) => {
                    neighborhoods.push((neighborhood, curr_patch));
                    curr_patch += 1;
                }
            }
        }
    }

    let mut patches = Vec::new();
    let distinct_patch_indices = neighborhoods.iter().map(|(_, ix)| ix ).unique();

    for ix in distinct_patch_indices {

        let curr_neighborhoods : Vec<Neighborhood> = neighborhoods.iter()
            .filter(|(_, n_ix)| n_ix == ix )
            .map(|(n, _)| n )
            .cloned()
            .collect();
        let color = curr_neighborhoods[0].color;

        patches.push(BinaryPatch {
            neighborhoods : curr_neighborhoods,
            scale : px_spacing as u8,
            color
        });
    }

    patches
}

/// Transform each cluster mean to a valid 8-bit color value.
pub fn extract_colors(km : &KMeans) -> Vec<u8> {
    km.means()
        .map(|m| m[0].max(0.0).min(255.0) as u8 )
        .collect::<Vec<_>>()
}

/// Returns an image, with each pixel attributed to its closest K-means color pixel
/// according to a given subsampling given by px_spacing. Also return the allocations,
/// which are the indices of the color vector each pixel in raster order belongs to.
/// (1) Call k-means for image 1
/// (2) For images 2..n:
///     (2.1). Find closest mean to each pixel
///     (2.2). Modify pixels to have this mean value.
pub fn segment_colors(win : &Window<'_, u8>, px_spacing : usize, n_colors : usize) -> KMeans {
    let km = KMeans::estimate(
        win.pixels(px_spacing).map(|d| [*d as f64] ),
        KMeansSettings { n_cluster : n_colors, max_iter : 1000 }
    ).unwrap();

    km
}

pub fn segment_colors_to_image(win : &Window<'_, u8>, px_spacing : usize, n_colors : usize) -> Image<u8> {
    let km = segment_colors(win, px_spacing, n_colors);
    let colors = extract_colors(&km);
    let ncol = win.width() / px_spacing;
    Image::from_vec(
        km.allocations().iter().map(|alloc| colors[*alloc] ).collect(),
        ncol
    )
}

/// colors : Sequence of representative colors returned by segment_colors ->extract_colors. Overwrites
/// all pixels of win with the closest color in the vector to each pixel.
pub fn write_segmented_colors_to_window<'a>(win : &'a mut WindowMut<'a, u8>, colors : &'a [u8]) {
    for px in win.pixels_mut(1) {
        let mut min_dist : (usize, u8) = (0, u8::MAX);
        for (ix_col, color) in colors.iter().enumerate() {
            let dist = ((*px as i16) - (*color as i16)).abs() as u8;
            if dist < min_dist.1 {
                min_dist = (ix_col, dist);
            }
        }
        *px = colors[min_dist.0];
    }
}

/*pub fn write_patches_to_window<'a>(win : &'a mut WindowMut<'a, u8>, patches : &'a [Patch]) {
    for patch in patches.iter() {
        assert!(patch.scale % 2 == 0, "Patch scale should be an even number");
        for neigh in patch.neighborhoods.iter() {
            let scaled_center = (neigh.center.0 * (patch.scale as usize), neigh.center.1 * (patch.scale as usize));
            let scaled_dim = (3*(patch.scale as usize), 3*(patch.scale as usize));
            let scaled_offset = (scaled_center.0 - scaled_dim.0 / 2, scaled_center.1 - scaled_dim.1 / 2);

            // This assumes the neighborhood is completely filled with the center color, ignoring neighborhood pattern.
            // *win` was mutably borrowed here in the previous iteration of the loop
            win.apply_to_sub_window(scaled_offset, scaled_dim, move |mut local : WindowMut<'_, u8>| { local.fill(neigh.color); } );
            // win.pixels_mut(2);
        }
    }
}*/

/// Verifies equality of neighbor (spacing=1 only) pixels to pixel centered at (row, col).
pub fn extract_neighborhood(label_img : &Window<'_, u8>, (row, col) : (usize, usize)) -> Neighborhood {
    let center = label_img[(row, col)];
    let up = label_img[(row - 1, col)];
    let down = label_img[(row + 1, col)];
    let left = label_img[(row, col -1)];
    let right = label_img[(row, col + 1)];
    let up_left = label_img[(row - 1, col - 1)];
    let up_right = label_img[(row - 1, col + 1)];
    let down_left = label_img[(row + 1, col - 1)];
    let down_right = label_img[(row + 1, col + 1)];
    let pattern = [
        up_left == center,
        up == center,
        up_right == center,
        left == center,
        right == center,
        down_left == center,
        down == center,
        down_left == center
    ];
    Neighborhood { color : center, pattern, center : (row, col) }
}

/*/// Verifies equality of neighbor (spacing=2 only) pixels to pixel centered at (row, col).
pub fn extract_extended_neighborhood(label_img : &Window<'_, u8>, (row, col) : (usize, usize)) -> Option<Extended> {
    let center = label_img[(row, col)];
    let up = label_img[(row - 2, col)];
    let down = label_img[(row + 2, col)];
    let left = label_img[(row, col - 2)];
    let right = label_img[(row, col + 2)];

    // Verify if pixels at 3x3 cross are equal.
    if [up, down, left, right].iter().all(|px| *px == center ) {

        // Verify if borders at 3x3 (inner) box are all equal.
        let top_border = label_img.sub_row(row - 2, (col-1)..(col+2)).unwrap().pixels().all(|px| *px == center );
        let bottom_border = label_img.sub_row(row + 2, (col-1)..(col+2)).unwrap().pixels().all(|px| *px == center );
        let left_border = label_img.sub_col((row - 1)..(row+2), col-2).unwrap().pixels().all(|px| *px == center );
        let right_border = label_img.sub_col((row - 1)..(row+2), col+2).unwrap().pixels().all(|px| *px == center );

        // Verify if corners at 5x5 box are all equal.
        let top_left = label_img[(row-2, col-2)] == center;
        let top_right = label_img[(row-2, col+2)] == center;
        let bottom_left = label_img[(row+2, col-2)] == center;
        let bottom_right = label_img[(row+2, col+2)] == center;

        if top_border && left_border && right_border && bottom_border {
            if top_left && top_right && bottom_left && bottom_right {
                Some(Extended::TwentyFour)
            } else {
                Some(Extended::Twenty)
            }
        } else {
            Some(Extended::Twelve)
        }
    } else {
        None
    }
}*/

/*let kind = match local {
    Local::Eight => {
        match segmentation::extract_extended_neighborhood(label_win, (row, col)) {
            Some(ext) => Either::Right(ext),
            None => Either::Left(local)
        }
    },
    local => Either::Left(local)
 };*/

// TODO examine kmeans_colors crate


