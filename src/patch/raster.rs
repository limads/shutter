use super::*;

#[derive(Debug, Clone)]
pub struct RasterSegmenter {

    px_spacing : usize,

    patches : Vec<Patch>,

    n_patches : usize,

    prev_row_mask : Vec<bool>,

    // Holds how many pixels are there for each column for the given patch
    // n_matches_col : Vec<u16>,

    // Holds how many pixels are there for each row for the given patch
    // n_matches_row : Vec<u16>,

    search : PatchSearch,

    win_sz : (usize, usize),

    labels : Vec<(usize, usize)>
}

impl RasterSegmenter {

    pub fn new(win_sz : (usize, usize), px_spacing : usize) -> Self {
        let mut prev_row_mask : Vec<bool> = Vec::with_capacity(win_sz.1 / px_spacing as usize);
        for c in (0..(win_sz.1 / px_spacing as usize)) {
            prev_row_mask.push(false);
        }
        let search = PatchSearch::new(win_sz, px_spacing);
        let labels : Vec<(usize, usize)> = crate::image::labels(win_sz, px_spacing).collect();
        Self { patches : Vec::with_capacity(16), px_spacing, n_patches : 0, prev_row_mask, search, win_sz, /*n_matches_row, n_matches_col*/labels }
    }

    // TODO use color mode to EXCLUDE a given color. Then, the struct should also hold a top
    // row with the marker telling whether ANY of the admissible colors were found.
    pub fn segment_all<'a>(&'a mut self, win : &Window<'_, u8>, margin : u8, exp_mode : ExpansionMode) -> &'a [Patch] {
        // let src_win = unsafe { crate::image::create_immutable(&win) };
        assert!(win.shape() == self.win_sz);
        assert!(win.shape().1 / self.px_spacing == self.search.prev_row_patch_ixs.len());
        let n_patches = match exp_mode {
            ExpansionMode::Rect => {
                full_color_patches(
                    &mut self.search,
                    &mut self.patches,
                    &win,
                    self.px_spacing.try_into().unwrap(),
                    margin,
                    Patch::add_to_right_rect,
                    Patch::add_to_bottom_rect,
                )
            },
            ExpansionMode::Contour => {
                let mut n_patches = full_color_patches(
                    &mut self.search,
                    &mut self.patches,
                    &win,
                    self.px_spacing.try_into().unwrap(),
                    margin,
                    Patch::add_to_right_contour,
                    Patch::add_to_bottom_contour,
                );
                filter_contours(&mut n_patches, &mut self.patches, &win);
                n_patches
            },
            ExpansionMode::Dense => {
                full_color_patches(
                    &mut self.search,
                    &mut self.patches,
                    &win,
                    self.px_spacing.try_into().unwrap(),
                    margin,
                    Patch::add_to_right_dense,
                    Patch::add_to_bottom_dense,
                )
            }
        };
        self.n_patches = n_patches;
        &self.patches[0..self.n_patches]
    }

    /// Returns only segments matching the given color.
    pub fn segment_single_color<'a, F>(&'a mut self, win : &Window<'_, u8>, comp : F, exp_mode : ExpansionMode) -> &'a [Patch]
    where
        F : Fn(u8)->bool
    {
        assert!(win.shape() == self.win_sz);
        assert!(win.shape().1 / self.px_spacing == self.search.prev_row_patch_ixs.len());
        assert!(win.width() / self.px_spacing == self.prev_row_mask.len());
        let n_patches = unsafe {
            match exp_mode {
                ExpansionMode::Rect => {
                    single_color_patches(
                        &mut self.search,
                        &mut self.prev_row_mask,
                        &mut self.patches,
                        &win,
                        &self.labels,
                        self.px_spacing.try_into().unwrap(),
                        comp,
                        Patch::add_to_right_rect,
                        Patch::add_to_bottom_rect
                    )
                },
                ExpansionMode::Contour => {
                    let mut n_patches = single_color_patches(
                        &mut self.search,
                        &mut self.prev_row_mask,
                        &mut self.patches,
                        &win,
                        &self.labels,
                        self.px_spacing.try_into().unwrap(),
                        comp,
                        Patch::add_to_right_contour,
                        Patch::add_to_bottom_contour
                    );
                    filter_contours(&mut n_patches, &mut self.patches, &win);
                    n_patches
                },
                ExpansionMode::Dense => {
                    single_color_patches(
                        &mut self.search,
                        &mut self.prev_row_mask,
                        &mut self.patches,
                        &win,
                        &self.labels,
                        self.px_spacing.try_into().unwrap(),
                        comp,
                        Patch::add_to_right_dense,
                        Patch::add_to_bottom_dense
                    )
                }
            }
        };

        // Since rect expansion just expands the rect, now insert the border pixels and calculate area.
        if exp_mode == ExpansionMode::Rect {
            for mut patch in self.patches[0..n_patches].iter_mut() {
                let rect = patch.outer_rect::<usize>();
                patch.pxs.clear();
                for r in [rect.0, rect.0 + rect.2] {
                    for c in rect.1..(rect.1 + rect.3) {
                        patch.pxs.push((r as u16, c as u16));
                    }
                }
                for c in [rect.1, rect.1 + rect.3] {
                    for r in rect.0..(rect.0 + rect.2) {
                        patch.pxs.push((r as u16, c as u16));
                    }
                }
                patch.area = rect.2 * rect.3;
            }
        }

        self.n_patches = n_patches;
        assert!(self.patches[0..self.n_patches].iter().all(|patch| patch.pxs.len() > 0 ));
        &self.patches[0..self.n_patches]
    }

    pub fn patches(&self) -> &[Patch] {
        &self.patches[0..self.n_patches]
    }

    /// Segments the image into separate nonoverlapping colors. The color to which a
    /// patch is attributed is stored in patch.color as the midpoint of each interval.
    pub fn segment_grouped_colors<'a>(&'a self, win : &'a Window<'_, u8>, colors : &'a [(u8, u8)], exp_mode : ExpansionMode) -> &'a [Patch] {

        // Verify colors are disjoint.
        assert!(colors.iter().all(|c| c.0 < c.1 ));
        for (ix, c1) in colors.iter().enumerate() {
            for c2 in colors[ix+1..].iter() {
                assert!((c1.0 < c2.0 && c1.1 < c2.1) || (c2.0 < c1.0 && c2.1 < c1.1));
            }
        }

        panic!()

        // &self.patches[0..self.n_patches]
    }

}

/*impl deft::Interactive for RasterSegmenter {

    #[export_name="register_RasterSegmenter"]
    extern "C" fn interactive() -> Box<deft::TypeInfo> {

        use deft::ReplResult;
        use rhai::{Array, Dynamic};

        deft::TypeInfo::builder::<Self>()
            .fallible("RasterSegmenter", |h : i64, w : i64, spacing : i64| -> ReplResult<Self> {
                Ok(Self::new((h as usize, w as usize), spacing as usize))
            })
            .fallible("segment_all", |s : &mut Self, w : Image<u8>, margin : i64| -> ReplResult<Array> {
                let patches = s.segment_all(&w.full_window(), margin as u8, ExpansionMode::Contour)
                    .iter()
                    .map(|patch| Dynamic::from(patch.clone()) )
                    .collect::<Vec<_>>();
                Ok(patches)
            })
            .fallible("segment_single", |s : &mut Self, w : Image<u8>, below : i64| -> ReplResult<Array> {
                let patches = s.segment_single_color(&w.full_window(), move |px| { px <= below as u8 }, ExpansionMode::Contour)
                    .iter()
                    .map(|patch| Dynamic::from(patch.clone()) )
                    .collect::<Vec<_>>();
                Ok(patches)
            })
            .priority(1)
            .build()
    }

}*/

/// Keeps state of a patch search
#[derive(Debug, Clone)]
struct PatchSearch {
    prev_row_patch_ixs : Vec<usize>,
    left_patch_ix : usize,
    top_patch_ix : usize,
    merges_left : bool,
    merges_top : bool,
    nrow : usize,
    ncol : usize
}

impl PatchSearch {

    fn new((height, width) : (usize, usize), px_spacing : usize) -> Self {
        let (ncol, nrow) = (width / px_spacing, height / px_spacing);
        let mut prev_row_patch_ixs : Vec<usize> = Vec::with_capacity(ncol);
        for c in (0..ncol) {
            prev_row_patch_ixs.push(0);
        }
        let (left_patch_ix, top_patch_ix) : (usize, usize) = (0, 0);
        Self { prev_row_patch_ixs, left_patch_ix, top_patch_ix, nrow, ncol, merges_left : false, merges_top : false }
    }

}

/*#[derive(Debug, Clone, Copy)]
pub struct Strip {
    extremes : [u16; 2],
    complete : bool,
}

#[derive(Debug, Clone)]
pub struct Contour {
    row_start : usize,
    rows : Vec<Strip>,
    col_start : usize,
    cols : Vec<Strip>
}*/

/*/// Represents a row or column of a raster contour. If it is incomplete,
/// a single pixel was pushed for the given row/column. If it is complete,
/// holds the index of the pxs vector where the last element of the row/column
/// was held (since in this case it will simply be substituted).
#[derive(Debug, Copy, Clone)]
enum ContourStrip {
    Complete(usize),
    Incomplete
}

struct PatchContourState {
    n_matches_row : HashMap<usize, ContourStrip>,
    n_matches_col : HashMap<usize, ContourStrip>,
}

/// Takes full pixel vector AFTER they are merged.
fn merge_dimension(
    this : &mut HashMap<usize, ContourStrip>,
    other : &mut HashMap<usize, ContourStrip>,
    new_pxs : &[(u16, u16)],
    is_row : bool,
    old_top_pxs_len : usize,
    curr_px_ix : usize
) {

    // When two patches are merged, the pixels of other are inserted in their natural
    // order to the pixels of self. So we must take care to re-index everything.
    for (strip_ix, mut strip) in other.iter_mut() {

        match strip {
            ContourStrip::Complete(ref mut pxs_ix) => {
                *pxs_ix += old_top_pxs_len;
            },
            _ => { }
        }

        match (this.get(stip_ix), stip) {
            (Some(ContourStrip::Incomplete), ContourStrip::Incomplete) => {
                *self.n_matches_row.get_mut(r).unwrap() = ContourStrip::Complete(pxs_ix);
                None
            },
            (Some(ContourStrip::Incomplete), ContourStrip::Complete(other_pos)) => {

            },
            (Some(ContourStrip::Complete(this_pos)), ContourStrip::Incomplete) => {

            },
            (Some(ContourStrip::Complete(this_pos), ContourStrip::Complete(other_pos))) => {
                Some(pos)
            },
            (None, strip) => {
                this.insert(stip_ix, strip);
                None
            }
        }
    }
}

impl PatchContourState {

    pub fn new() -> Self {
        Self {
            n_matches_row : HashMap::with_hasher(nohash_hasher::NoHashHasher),
            n_matches_col : HashMap::with_hasher(nohash_hasher::NoHashHasher),
        }
    }

    pub fn merge(&mut self, other : PatchContourState, new_pxs : &[(u16, u16)]) {
        merge_dimension(&mut self.n_matches_row, &other.n_matches_row, new_pxs, true);
        merge_dimension(&mut self.n_matches_col, &other.n_matches_col, new_pxs, false);
    }

    pub fn clear(&mut self) {
        self.n_matches_row.clear();
        self.n_matches_col.clear();
    }

    pub fn update_row(&mut self, r : usize, pxs_ix : usize) -> Option<usize> {
        update_contour_state(&mut self.n_matches_row, r, pxs_ix)
    }

    pub fn update_row_and_col(&mut self, r : usize, c : usize, pxs_ix : usize) {
        update_contour_state(&mut self.n_matches_row, r, pxs_ix);
        update_contour_state(&mut self.n_matches_col, c, pxs_ix);
    }

    pub fn update_col(&mut self, c : usize, pxs_ix : usize) -> Option<usize> {
        update_contour_state(&mut self.n_matches_col, c, pxs_ix)
    }

}*/

/// Search the image for disjoint color patches of a single user-specified color. If tol
/// is informed, any pixel witin patch+- tol is considered. If not, only pixels with strictly
/// the desired color are returned. Unsafe is required because we use get_unchecked in the hot
/// loop that iterate over pixels.
unsafe fn single_color_patches<F, R, B>(
    search : &mut PatchSearch,
    prev_row_mask : &mut Vec<bool>,
    patches : &mut Vec<Patch>,
    win : &Window<'_, u8>,
    labels : &[(usize, usize)],
    px_spacing : u16,
    comp : F,
    add_to_right : R,
    add_to_bottom : B
) -> usize
where
    F : Fn(u8)->bool,
    R : Fn(&mut Patch, (u16, u16)) + Copy,
    B : Fn(&mut Patch, (u16, u16)) + Copy
{
    let mut n_patches = 0;
    let mut last_matching_col = None;
    let mut color_match = false;
    let subsampled_ncols = win.width() / px_spacing as usize;
    let last_col = subsampled_ncols - 1;
    for (r, c, px_color) in win.labeled_pixels::<usize, _>(px_spacing as usize) {
    // for ((r, c), px_color) in labels.iter().zip(win.pixels(px_spacing as usize)) {

        if comp(px_color) {
            if r >= 1 && *prev_row_mask.get_unchecked(c) {
                search.merges_top = true;
                search.top_patch_ix = *search.prev_row_patch_ixs.get_unchecked(c);
            } else {
                search.merges_top = false;
            }
            search.merges_left = if let Some(last_c) = last_matching_col {
                c > 0 && c - last_c == 1
            } else {
                false
            };

            append_or_update_patch(
                patches,
                search,
                &mut n_patches,
                win,
                r,
                c,
                px_color,
                px_spacing,
                add_to_right,
                add_to_bottom
            );
            if c < last_col  {
                last_matching_col = Some(c);
            } else {
                last_matching_col = None;
            }
            *prev_row_mask.get_unchecked_mut(c) = true;

        } else {
            *prev_row_mask.get_unchecked_mut(c) = false;
            if c == last_col {
                last_matching_col = None;
            }
        }
    }

    n_patches
}

/// During pixel insertion in a patch, row raster order is preserved, but column raster order is not.
/// Returns up to which index of patches the new data is valid. We keep patches of a previous iteration
/// so the pixel vectors within patches do not get reallocated. In the public API, we use this quantity
/// to limit the index of the patch slice only to the points generated by the current iteration.
/// Patches are defined by the next color being within **margin** from the last pixel color.
fn full_color_patches<R, B>(
    search : &mut PatchSearch,
    patches : &mut Vec<Patch>,
    win : &Window<'_, u8>,
    px_spacing : u16,
    margin : u8,
    add_to_right : R,
    add_to_bottom : B
) -> usize
where
    R : Fn(&mut Patch, (u16, u16)) + Copy,
    B : Fn(&mut Patch, (u16, u16)) + Copy
{

    // It is working, just update using the new merges_left/merges_top logic used at single_color_patches.
    let mut n_patches = 0;
    for (r, c, color) in win.labeled_pixels::<usize, _>(px_spacing as usize) {

        search.merges_top = if r >= 1 {
            ((c as i16 - win[(r-1, c)] as i16).abs() as u8) < margin
        } else {
            false
        };

        search.merges_left = if c >= 1 {
            ((c as i16 - win[(r, c-1)] as i16).abs() as u8) < margin
        } else {
            false
        };

        /*if c == 0 {
            // search.left_patch_ix = None;
            search.merges_left = false;
        } else {
            search.merges_left = might_merge_left;
        }*/

        // search.top_patch_ix = if r >= 1 {
        //    Some(search.prev_row_patch_ixs[c])
        // } else {
        //    None
        // };

        // let merges_left = might_merge_left && search.left_patch_ix.is_some();
        // let merges_top = might_merge_top && search.top_patch_ix.is_some();
        append_or_update_patch(patches, search, &mut n_patches, win, r, c, color, px_spacing, add_to_right, add_to_bottom);
    }

    n_patches
}

fn filter_contours(n_patches : &mut usize, patches : &mut Vec<Patch>, win : &Window<'_, u8>) {

    // A contour should have at least three pixel points. Remove the ones
    // without this many points.
    let mut ix = 0;
    while ix < *n_patches {
        if patches[ix].pxs.len() < 3 {
            patches.swap(ix, *n_patches-1);
            *n_patches -= 1;
        } else {
            ix += 1;
        }
    }

    for mut patch in patches[0..*n_patches].iter_mut() {
        close_contour(patch, win);
    }

}

// TODO review patch color strategy. For now, the patch color
// is the color that inaugurated the patch via its top-left pixel
// at the raster search strategy. This contrasts with the growth
// strategy where the patch color is the seed position color.
fn append_or_update_patch<R, B>(
    patches : &mut Vec<Patch>,
    search : &mut PatchSearch,
    n_patches : &mut usize,
    win : &Window<'_, u8>,
    r : usize,
    c : usize,
    color : u8,
    px_spacing : u16,
    add_to_right : R,
    add_to_bottom : B
) where
    R : Fn(&mut Patch, (u16, u16)) + Copy,
    B : Fn(&mut Patch, (u16, u16)) + Copy
{
    let pos = (r as u16, c as u16);
    match (search.merges_left, search.merges_top) {
        (true, true) => {

            let top_differs_left = search.top_patch_ix != search.left_patch_ix;
            if top_differs_left {
                // This condition will only happen when the last row of the left patch
                // is either at the last row of the top patch (where this last row has only
                // a few pixels that matched bottom) or one-past the last row of the top patch
                // (if this past row is sharp).
                merge_left_to_top_patch(patches, search, n_patches);

                // Push new pixel to top patch
                add_to_bottom(&mut patches[search.top_patch_ix], pos);
                search.prev_row_patch_ixs[c] = search.top_patch_ix;
                search.left_patch_ix = search.top_patch_ix;
            } else {
                // Push new pixel to left patch. We could also push to the bottom,
                // and stay at a valid state, but pushing to the left is cheaper.
                add_to_right(&mut patches[search.left_patch_ix], pos);
                search.prev_row_patch_ixs[c] = search.left_patch_ix;
                // search.left_patch_ix = Some(search.top_patch_ix.unwrap());
            }
        },
        (true, false) => {
            add_to_right(&mut patches[search.left_patch_ix], pos);
            search.prev_row_patch_ixs[c] = search.left_patch_ix;
        },
        (false, true) => {
            add_to_bottom(&mut patches[search.top_patch_ix], pos);
            search.prev_row_patch_ixs[c] = search.top_patch_ix;
            search.left_patch_ix = search.top_patch_ix;
        },
        (false, false) => {
            add_patch(patches, search, n_patches, win, /*r, c,*/pos, color, px_spacing);
        }
    }

    // Irrespective of whether a patch was added or updated, it should have
    // at least one pixel at the last valid position of n_patches.
    assert!(patches[*n_patches-1].pxs.len() > 0);
}

/// Since we might be recycling the patches Vec<_> from a previous iteration,
/// either update one of the junk patches to hold the new patch, or push
/// this new patch to the vector. In either case, n_patches (which hold the
/// valid patch slice size) will be incremented by one.
fn add_patch(
    patches : &mut Vec<Patch>,
    search : &mut PatchSearch,
    n_patches : &mut usize,
    win : &Window<'_, u8>,
    // r : usize,
    // c : usize,
    pos : (u16, u16),
    color : u8,
    px_spacing : u16
) {
    if *n_patches < patches.len() {
        patches[*n_patches].pxs.clear();
        patches[*n_patches].pxs.push(pos);
        patches[*n_patches].outer_rect = (pos.0, pos.1, 1, 1);
        patches[*n_patches].color = color;
        patches[*n_patches].scale = px_spacing;
        patches[*n_patches].img_height = win.height();
        patches[*n_patches].area += 1;
    } else {
        patches.push(Patch::new(pos, color, px_spacing, win.height()));
    }
    *n_patches += 1;
    search.prev_row_patch_ixs[pos.1 as usize] = *n_patches - 1;
    search.left_patch_ix = *n_patches - 1;
    // search.contour_state.add_new(r, c, 0);
}

fn merge_left_to_top_patch(patches : &mut Vec<Patch>, search : &mut PatchSearch, n_patches : &mut usize) {

    // This method can't be called when the left and top patches are the same.
    assert!(search.left_patch_ix != search.top_patch_ix);

    let left_patch = mem::take(&mut patches[search.left_patch_ix]);

    // let old_top_len = patches[search.top_patch_ix.unwrap()].pxs.len();
    patches[search.top_patch_ix].merge(left_patch);

    // search.contour_state.merge(search.top_patch_ix.unwrap(), search.left_patch_ix.unwrap(), &patches[search.top_patch_ix.unwrap()].pxs[..], old_top_len);

    // Remove old left patch (now merged with top).
    patches.remove(search.left_patch_ix);

    if search.top_patch_ix > search.left_patch_ix {
        search.top_patch_ix -= 1;
    }

    // } else {
    //    panic!()
    // }

    // Update indices of previous row patches, since the remove(.) invalidated
    // every index >= left patch.
    for mut ix in search.prev_row_patch_ixs.iter_mut() {
        if *ix == search.left_patch_ix {
            *ix = search.top_patch_ix;
        } else {
            if *ix > search.left_patch_ix {
                *ix -= 1;
            }
        }
    }

    // Account for removed patch.
    *n_patches -= 1;
}

/*#[test]
fn test_patches() {
    let check = Image::<u8>::new_checkerboard(16, 4);
    println!("{}", check);
    for patch in check.full_window().patches(1).iter() {
        println!("{:?}", patch);
        // println!("{:?}", patch.polygon());
    }
}*/

/*#[test]
fn test_raster() {

    use deft::Show;
    use crate::image::Mark;

    let mut img = crate::io::decode_from_file("/home/diego/Downloads/full_bright_image2.png").unwrap();
    let mut raster = RasterSegmenter::new((img.height(), img.width()), 1);
    let patches = raster.segment_single_color(&img.full_window(), |b| b < 30, ExpansionMode::Contour);
    for patch in patches {
        img.full_window_mut().draw(Mark::Shape(patch.outer_points(ExpansionMode::Contour), 255));
    }
    // img.full_window_mut().draw(Mark::Rect((1,1), (10, 10), 255));
    img.show();
    // crate::io::encode_to_file(img, "/home/diego/Downloads/index-drawn.png");
}*/


