// use either::Either;
use std::cmp::{Eq, PartialEq};
use itertools::Itertools;
use crate::image::{Image, Window, WindowMut};
use bayes::fit::{cluster::KMeans, cluster::KMeansSettings, Estimator};
use crate::shape::*;
use std::fmt;
use std::collections::HashMap;

// #[cfg(feature="opencvlib")]
// pub mod fgmm;

// #[cfg(feature="opencvlib")]
// pub mod mser;

/// The most general patch is a set of pixel positions with a homogeneous color
/// and a scale that was used for extraction. The patch is assumed to be
/// homonegeous within a pixel spacing given by the scale field.
#[derive(Clone, Debug)]
pub struct Patch {
    pub pxs : Vec<(usize, usize)>,
    pub color : u8,
    pub scale : usize
}

impl Patch {

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

    pub fn polygon(&self) -> Polygon {
        let mut row_pxs = self.group_rows();
        let mut sorted_keys = row_pxs.iter().map(|(k, _)| k ).collect::<Vec<_>>();
        sorted_keys.sort();
        let n = sorted_keys.len();
        let mut pts : Vec<(usize, usize)> = Vec::new();

        // Points with "top" part of the patch
        let fst_row = sorted_keys[0];
        for col in row_pxs[fst_row].iter() {
            pts.push((*fst_row, *col));
        }

        // Points with "right" part of the patch
        for row in sorted_keys[1..n-1].iter() {
            pts.push((**row, *row_pxs[row].last().unwrap()));
        }

        // Points with "bottom" part of the patch
        let last_row = sorted_keys.last().unwrap();
        for col in row_pxs[last_row].iter().rev() {
            pts.push((**last_row, *col));
        }

        // Points with "left" part of the patch
        for row in sorted_keys[1..n-1].iter().rev() {
            pts.push((**row, *row_pxs[row].first().unwrap()));
        }
        Polygon::from(pts)
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

pub fn color_patches(win : &Window<'_, u8>, px_spacing : usize) -> Vec<Patch> {
    let mut patches : Vec<Patch> = Vec::new();
    let (ncol, nrow) = (win.width() / px_spacing, win.height() / px_spacing);
    for (ix, px) in win.pixels(px_spacing).enumerate() {
        let (r, c) = (ix / ncol, ix % ncol);
        let color = win[(r*px_spacing, c*px_spacing)];

        // Get patch that contains the pixel above the current pixel
        let top_patch_ix = patches.iter().position(|patch| patch.pxs.iter().any(|px| r >= 1 && px.0 == r-1 && px.1 == c ) );

        // Get patch that contains the pixel to the left of current pixel
        let left_patch_ix = patches.iter().position(|patch| patch.pxs.iter().any(|px| c >= 1 && px.0 == r && px.1 == c-1 ) );

        // Verifies if patches have the same color
        let merges_top = if let Some(top) = top_patch_ix.and_then(|ix| patches.get(ix)) {
            top.color == color
        } else {
            false
        };
        let merges_left = if let Some(left) = left_patch_ix.and_then(|ix| patches.get(ix)) {
            left.color == color
        } else {
            false
        };

        match (merges_left, merges_top) {
            (true, true) => {

                let top_differs_left = top_patch_ix != left_patch_ix;
                if top_differs_left {
                    // Merge two patches
                    let left_pxs = patches[left_patch_ix.unwrap()].pxs.iter().cloned().collect::<Vec<_>>();
                    patches[top_patch_ix.unwrap()].pxs.extend(left_pxs);
                }

                // Push new pixel
                patches[top_patch_ix.unwrap()].pxs.push((r, c));

                if top_differs_left {
                    // Remove old left patch (now merged with top).
                    patches.remove(left_patch_ix.unwrap());
                }
            },
            (true, false) => {
                patches[left_patch_ix.unwrap()].pxs.push((r, c));
            }
            (false, true) => {
                patches[top_patch_ix.unwrap()].pxs.push((r, c));
            },
            (false, false) => {
                patches.push(Patch { color, pxs : vec![(r, c)], scale : px_spacing });
            }
        }
    }
    patches
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


