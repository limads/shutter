use crate::image::*;
use petgraph::graph::UnGraph;
use itertools::Itertools;
use std::ops::Range;
use parry2d::utils::Interval;
use petgraph::unionfind::UnionFind;
use std::mem;
use petgraph::graph::NodeIndex;
use crate::graph::SplitGraph;
use std::collections::BTreeMap;
use petgraph::prelude::*;
use petgraph::visit::*;
use std::convert::{AsRef, AsMut};
use std::cmp::Ordering;
use num_traits::bounds::Bounded;
use nalgebra::{Point2, Vector2};
use std::ops::AddAssign;
use crate::gray::Foreground;
use crate::local::IppSumWindow;
use smallvec::SmallVec;
use rangetools::Rangetools;
use crate::bitonal::*;
use crate::pyr::*;

#[derive(Debug, Clone)]
pub struct PyrBitonalEncoder {
    pyr : Pyramid,
    bit_img : BitonalImage,
    strip_height : usize,
    levels : usize
}

impl PyrBitonalEncoder {

    pub fn new(height : usize, width : usize, levels : usize) -> Self {
        assert!(levels > 0 && levels <= 5);
        assert!(height / (2usize).pow(levels as u32) > 1);
        let strip_height = height / (2usize).pow(levels as u32);
        // assert!(strip_height == 16);
        Self { bit_img : BitonalImage::new(height, width / 8), pyr : Pyramid::new_vertical(height, width / 8), strip_height, levels }
    }

    fn update(&mut self) -> RunLengthCode {
        let bit : &ImageBuf<u8> = self.bit_img.as_ref();
        let (h, w) = self.bit_img.as_ref().size();
        let sh = self.strip_height;
        let mut tops = bit.windows((sh, w)).step_by(2);
        let win_bottom = bit.window((sh, 0), (h - sh, w)).unwrap();
        let mut bottoms = win_bottom.windows((sh, w)).step_by(2);
        let mut dst_win = &mut self.pyr.as_mut()[0];

        for ((top, bottom), mut dst) in tops.zip(bottoms).zip(dst_win.windows_mut((sh, w))) {
            top.or_to(&bottom, &mut dst);
        }

        // Use open range since first level has already been calculated from original image.
        for lvl in 1..self.levels {
            let PyramidStepMut { mut coarse, fine } = self.pyr.step_mut(lvl);
            let h = fine.height();
            let mut tops = fine.windows((sh, w)).step_by(2);
            let win_bottom = fine.window((sh, 0), (h - sh, w)).unwrap();
            let mut bottoms = win_bottom.windows((sh, w)).step_by(2);
            // let mut niter = 0;
            for ((top, bottom), mut dst) in tops.zip(bottoms).zip(coarse.windows_mut((sh, w))) {
                top.or_to(&bottom, &mut dst);
                // niter += 1;
            }
        }

        let num_rles = (2usize).pow(self.levels as u32);
        let mut accum_rles : Vec<Vec<RunLength>> = (0..num_rles).map(|_| Default::default() ).collect();
        let mut curr_rles : Vec<Option<RunLength>> = (0..num_rles).map(|_| Default::default() ).collect();

        let pyr = &self.pyr.as_ref()[..self.levels];
        let last_pyr = pyr.last().unwrap();

        for r in 0..last_pyr.height() {
            for c in 0..last_pyr.width() {
                lossy_encode_bitonal_pyr(
                    pyr,
                    self.bit_img.as_ref(),
                    self.levels-1,
                    (r, c),
                    self.bit_img.as_ref().width()-1,
                    self.strip_height,
                    &mut accum_rles,
                    &mut curr_rles,
                );
            }
            for (ix, curr) in curr_rles.iter_mut().enumerate() {
                if let Some(last) = curr.take() {
                    accum_rles[ix].push(last);
                }
            }
        }
        let (mut fst_accum, mut rem_accum) = accum_rles.split_at_mut(1);
        for mut rem in rem_accum.iter_mut() {
            fst_accum[0].extend(rem.drain(..));
        }
        RunLengthCode::from_unsorted(std::mem::take(&mut fst_accum[0]))
    }

    // Calculate from binary
    pub fn calculate<S>(&mut self, img : &Image<u8, S>) -> RunLengthCode
    where
        S : Storage<u8>
    {
        self.bit_img.update_from_binary(&img);
        self.update()
    }

    pub fn calculate_from_gray<S>(&mut self, img : &Image<u8, S>, thr_above : u8) -> RunLengthCode
    where
        S : Storage<u8>
    {
        self.bit_img.update_from_gray(&img, thr_above);
        self.update()
    }

    pub fn calculate_from_gray_inverted<S>(&mut self, img : &Image<u8, S>, thr_below : u8) -> RunLengthCode
    where
        S : Storage<u8>
    {
        self.bit_img.update_from_gray_inverted(&img, thr_below);
        self.update()
    }

}

const TOP_PYR_IXS_2 : [usize; 2048] = build_top_pyramid_indices(2);
const BOTTOM_PYR_IXS_2 : [usize; 2048] = build_bottom_pyramid_indices(2);

const TOP_PYR_IXS_4 : [usize; 2048] = build_top_pyramid_indices(4);
const BOTTOM_PYR_IXS_4 : [usize; 2048] = build_bottom_pyramid_indices(4);

const TOP_PYR_IXS_8 : [usize; 2048] = build_top_pyramid_indices(8);
const BOTTOM_PYR_IXS_8 : [usize; 2048] = build_bottom_pyramid_indices(8);

const TOP_PYR_IXS_16 : [usize; 2048] = build_top_pyramid_indices(16);
const BOTTOM_PYR_IXS_16 : [usize; 2048] = build_bottom_pyramid_indices(16);

const TOP_PYR_IXS_32 : [usize; 2048] = build_top_pyramid_indices(32);
const BOTTOM_PYR_IXS_32 : [usize; 2048] = build_bottom_pyramid_indices(32);

const TOP_PYR_IXS_64 : [usize; 2048] = build_top_pyramid_indices(64);
const BOTTOM_PYR_IXS_64 : [usize; 2048] = build_bottom_pyramid_indices(64);

const fn build_top_pyramid_indices(strip_height : usize) -> [usize; 2048] {
    let mut dst = [0; 2048];
    let mut ix = 0;
    while ix < 2048 {
        dst[ix] = top_index_at_next_level(ix, strip_height);
        ix += 1;
    }
    dst
}

const fn build_bottom_pyramid_indices(strip_height : usize) -> [usize; 2048] {
    let mut dst = [0; 2048];
    let mut ix = 0;
    while ix < 2048 {
        dst[ix] = bottom_index_at_next_level(ix, strip_height);
        ix += 1;
    }
    dst
}

const fn top_index_at_next_level(ix : usize, strip_height : usize) -> usize {
    (ix / strip_height) * (2*strip_height) + ix % strip_height
}

const fn bottom_index_at_next_level(ix : usize, strip_height : usize) -> usize {
    top_index_at_next_level(ix, strip_height) + strip_height
}

// the strip height is always the same; it is just the number of
// strips that double at each pyramid level. The strip height is
// a function of the desired number of levels.
pub fn lossy_encode_bitonal_pyr(
    pyr : &[ImageBuf<u8>],
    coarse : &ImageBuf<u8>,
    level : usize,
    ix : (usize, usize),
    max_ix : usize,
    strip_height : usize,
    accum_rles : &mut [Vec<RunLength>],
    curr_rles : &mut [Option<RunLength>]
) {
    if pyr[level][ix] != 0 {

        let top_ix = ((ix.0 / strip_height) * (2*strip_height) + ix.0 % strip_height, ix.1);
        let bottom_ix = (top_ix.0 + strip_height, ix.1);

        // The indices can be read from a constant, but there isn't a lot of difference in performance..
        // let top_ix = (unsafe { *TOP_PYR_IXS_16.get_unchecked(ix.0) }, ix.1);
        // let bottom_ix = (unsafe { *BOTTOM_PYR_IXS_16.get_unchecked(ix.0) }, ix.1);

        if level > 0 {
            let prev_lvl = level-1;
            lossy_encode_bitonal_pyr(
                pyr,
                coarse,
                prev_lvl,
                top_ix,
                max_ix,
                strip_height,
                accum_rles,
                curr_rles
            );
            lossy_encode_bitonal_pyr(
                pyr,
                coarse,
                prev_lvl,
                bottom_ix,
                max_ix,
                strip_height,
                accum_rles,
                curr_rles
            );
        } else {
            // println!("lvl = {:?}, ix = {:?}; top = {:?}, bottom = {:?}", level, ix, top_ix, bottom_ix);
            // println!("len = {:?}", accum_rles.len());

            let top_pos = top_ix.0 / strip_height;
            let bottom_pos = bottom_ix.0 / strip_height;
            // let bottom_pos = top_pos + 1;

            // println!("{:?} {:?}", top_pos, bottom_pos);
            lossy_encode_bitonal_pixel(
                top_ix.0,
                top_ix.1,
                max_ix,
                unsafe { coarse.row_unchecked(top_ix.0) },
                &mut curr_rles[top_pos],
                &mut accum_rles[top_pos]
            );
            lossy_encode_bitonal_pixel(
                bottom_ix.0,
                bottom_ix.1,
                max_ix,
                unsafe { coarse.row_unchecked(bottom_ix.0) },
                &mut curr_rles[bottom_pos],
                &mut accum_rles[bottom_pos]
            );
        }
    }
}

#[derive(Clone)]
pub struct BitonalEncoder {
    bit_img : BitonalImage,
    lut : RLEOpLut
}

impl BitonalEncoder {

    pub fn bitonal_image(&self) -> &BitonalImage {
        &self.bit_img
    }

    pub fn new(height : usize, width : usize) -> Self {
        Self { bit_img : BitonalImage::new(height, width / 8), lut : populate_rle_lut() }
    }

    fn update(&mut self) -> RunLengthCode {
        let bitonal : &ImageBuf<u8> = self.bit_img.as_ref();
        let mut rles = Vec::with_capacity(bitonal.height());
        let mut rows = Vec::with_capacity(bitonal.height());

        for r in 0..bitonal.height() {
            let len_before = rles.len();
            // lossy_encode_bitonal_row(unsafe { bitonal.row_unchecked(r) }, r, &mut rles);
            lossless_encode_bitonal_row(&self.lut[..], unsafe { bitonal.row_unchecked(r) }, r, &mut rles);
            let len_after = rles.len();
            if len_after > len_before {
                rows.push(Range { start : len_before, end : len_after });
            }
        }
        RunLengthCode { rles, rows }
    }

    pub fn calculate<S>(&mut self, img : &Image<u8, S>) -> RunLengthCode
    where
        S : Storage<u8>
    {
        self.bit_img.update_from_binary(&img);
        self.update()
    }

    pub fn calculate_from_binary<S>(&mut self, img : &Image<u8, S>) -> RunLengthCode
    where
        S : Storage<u8>
    {
        self.calculate_from_gray(img, 0)
    }

    pub fn calculate_from_gray<S>(&mut self, img : &Image<u8, S>, above_thr : u8) -> RunLengthCode
    where
        S : Storage<u8>
    {
        self.bit_img.update_from_gray(&img, above_thr);
        self.update()
    }

    pub fn calculate_from_gray_inverted<S>(&mut self, img : &Image<u8, S>, thr_below : u8) -> RunLengthCode
    where
        S : Storage<u8>
    {
        self.bit_img.update_from_gray_inverted(&img, thr_below);
        self.update()
    }

}

fn add_one(
    rles : &mut Vec<RunLength>,
    open : &mut bool,
    opened : bool,
    row : usize,
    col : usize,
    length : usize
) {
    rles.push(RunLength { start : (row, col), length });
    *open = opened;
}

fn add_two(
    rles : &mut Vec<RunLength>,
    open : &mut bool,
    opened : bool,
    row : usize,
    col1 : usize,
    length1 : usize,
    col2 : usize,
    length2 : usize
) {
    rles.push(RunLength { start : (row, col1), length : length1 });
    rles.push(RunLength { start : (row, col2), length : length2 });
    *open = opened;
}

fn add_three(
    rles : &mut Vec<RunLength>,
    open : &mut bool,
    opened : bool,
    row : usize,
    col1 : usize,
    length1 : usize,
    col2 : usize,
    length2 : usize,
    col3 : usize,
    length3 : usize
) {
    rles.push(RunLength { start : (row, col1), length : length1 });
    rles.push(RunLength { start : (row, col2), length : length2 });
    rles.push(RunLength { start : (row, col3), length : length3 });
    *open = opened;
}

fn add_four(
    rles : &mut Vec<RunLength>,
    open : &mut bool,
    opened : bool,
    row : usize,
    col1 : usize,
    length1 : usize,
    col2 : usize,
    length2 : usize,
    col3 : usize,
    length3 : usize,
    col4 : usize,
    length4 : usize
) {
    rles.push(RunLength { start : (row, col1), length : length1 });
    rles.push(RunLength { start : (row, col2), length : length2 });
    rles.push(RunLength { start : (row, col3), length : length3 });
    rles.push(RunLength { start : (row, col4), length : length4 });
    *open = opened;
}

type FPtrDyn = &'static (dyn Fn(&mut Vec<RunLength>, &mut bool, usize, usize) + Send + Sync + 'static);

type RLEOpLut = [FPtrDyn; 256];

fn populate_rle_lut() -> RLEOpLut {
    let mut ops = [(&|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
    }) as FPtrDyn; 256];
    ops[0b0] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        *open = false;
    }) as FPtrDyn;
    ops[0b1] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, false, row, col, 1)
    }) as FPtrDyn;
    ops[0b10] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, false, row, col + 1, 1)
    }) as FPtrDyn;
    ops[0b11] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, false, row, col , 2)
    }) as FPtrDyn;
    ops[0b100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, false, row, col + 2, 1)
    }) as FPtrDyn;
    ops[0b101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 1, col + 2, 1)
    }) as FPtrDyn;
    ops[0b110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, false, row, col + 1, 2)
    }) as FPtrDyn;
    ops[0b111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, false, row, col, 3)
    }) as FPtrDyn;
    ops[0b1000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, false, row, col + 3, 1)
    }) as FPtrDyn;
    ops[0b1001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 1, col + 3, 1)
    }) as FPtrDyn;
    ops[0b1010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 1, 1, col + 3, 1)
    }) as FPtrDyn;
    ops[0b1011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 2, col + 3, 1)
    }) as FPtrDyn;
    ops[0b1100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, false, row, col + 2, 2)
    }) as FPtrDyn;
    ops[0b1101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 1, col + 2, 2)
    }) as FPtrDyn;
    ops[0b1110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, false, row, col + 1, 3)
    }) as FPtrDyn;
    ops[0b1111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, false, row, col, 4)
    }) as FPtrDyn;
    ops[0b10000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, false, row, col + 4, 1)
    }) as FPtrDyn;
    ops[0b10001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 1, col + 4, 1)
    }) as FPtrDyn;
    ops[0b10010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 1, 1, col + 4, 1)
    }) as FPtrDyn;
    ops[0b10011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 2, col + 4, 1)
    }) as FPtrDyn;
    ops[0b10100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 2, 1, col + 4, 1)
    }) as FPtrDyn;
    ops[0b10101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, false, row, col, 1, col + 2, 1, col + 4, 1)
    }) as FPtrDyn;
    ops[0b10110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 1, 2, col + 4, 1)
    }) as FPtrDyn;
    ops[0b10111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 3, col + 4, 1)
    }) as FPtrDyn;
    ops[0b11000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, false, row, col + 3, 2)
    }) as FPtrDyn;
    ops[0b11001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 1, col + 3, 2)
    }) as FPtrDyn;
    ops[0b11010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 1, 1, col + 3, 2)
    }) as FPtrDyn;
    ops[0b11011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 2, col + 3, 2)
    }) as FPtrDyn;
    ops[0b11100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, false, row, col + 2, 3)
    }) as FPtrDyn;
    ops[0b11101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 1, col + 2, 3)
    }) as FPtrDyn;
    ops[0b11110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, false, row, col + 1, 4)
    }) as FPtrDyn;
    ops[0b11111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, false, row, col, 5)
    }) as FPtrDyn;
    ops[0b100000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, false, row, col + 5, 1)
    }) as FPtrDyn;
    ops[0b100001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 1, col + 5, 1)
    }) as FPtrDyn;
    ops[0b100010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 1, 1, col + 5, 1)
    }) as FPtrDyn;
    ops[0b100011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 2, col + 5, 1)
    }) as FPtrDyn;
    ops[0b100100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 2, 1, col + 5, 1)
    }) as FPtrDyn;
    ops[0b100101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, false, row, col, 1, col + 2, 1, col + 5, 1)
    }) as FPtrDyn;
    ops[0b100110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 1, 2, col + 5, 1)
    }) as FPtrDyn;
    ops[0b100111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 3, col + 5, 1)
    }) as FPtrDyn;
    ops[0b101000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 3, 1, col + 5, 1)
    }) as FPtrDyn;
    ops[0b101001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, false, row, col, 1, col + 3, 1, col + 5, 1)
    }) as FPtrDyn;
    ops[0b101010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, false, row, col + 1, 1, col + 3, 1, col + 5, 1)
    }) as FPtrDyn;
    ops[0b101011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, false, row, col, 2, col + 3, 1, col + 5, 1)
    }) as FPtrDyn;
    ops[0b101100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 2, 2, col + 5, 1)
    }) as FPtrDyn;
    ops[0b101101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, false, row, col, 1, col + 2, 2, col + 5, 1)
    }) as FPtrDyn;
    ops[0b101110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 1, 3, col + 5, 1)
    }) as FPtrDyn;
    ops[0b101111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 4, col + 5, 1)
    }) as FPtrDyn;
    ops[0b110000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, false, row, col + 4, 2)
    }) as FPtrDyn;
    ops[0b110001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 1, col + 4, 2)
    }) as FPtrDyn;
    ops[0b110010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 1, 1, col + 4, 2)
    }) as FPtrDyn;
    ops[0b110011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 2, col + 4, 2)
    }) as FPtrDyn;
    ops[0b110100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 2, 1, col + 4, 2)
    }) as FPtrDyn;
    ops[0b110101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, false, row, col, 1, col + 2, 1, col + 4, 2)
    }) as FPtrDyn;
    ops[0b110110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 1, 2, col + 4, 2)
    }) as FPtrDyn;
    ops[0b110111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 3, col + 4, 2)
    }) as FPtrDyn;
    ops[0b111000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, false, row, col + 3, 3)
    }) as FPtrDyn;
    ops[0b111001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 1, col + 3, 3)
    }) as FPtrDyn;
    ops[0b111010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 1, 1, col + 3, 3)
    }) as FPtrDyn;
    ops[0b111011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 2, col + 3, 3)
    }) as FPtrDyn;
    ops[0b111100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, false, row, col + 2, 4)
    }) as FPtrDyn;
    ops[0b111101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 1, col + 2, 4)
    }) as FPtrDyn;
    ops[0b111110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, false, row, col + 1, 5)
    }) as FPtrDyn;
    ops[0b111111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, false, row, col, 6)
    }) as FPtrDyn;
    ops[0b1000000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, false, row, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1000001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 1, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1000010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 1, 1, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1000011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 2, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1000100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 2, 1, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1000101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, false, row, col, 1, col + 2, 1, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1000110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 1, 2, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1000111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 3, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1001000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 3, 1, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1001001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, false, row, col, 1, col + 3, 1, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1001010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, false, row, col + 1, 1, col + 3, 1, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1001011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, false, row, col, 2, col + 3, 1, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1001100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 2, 2, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1001101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, false, row, col, 1, col + 2, 2, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1001110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 1, 3, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1001111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 4, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1010000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 4, 1, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1010001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, false, row, col, 1, col + 4, 1, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1010010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, false, row, col + 1, 1, col + 4, 1, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1010011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, false, row, col, 2, col + 4, 1, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1010100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, false, row, col + 2, 1, col + 4, 1, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1010101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_four(rles, open, false, row, col, 1, col + 2, 1, col + 4, 1, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1010110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, false, row, col + 1, 2, col + 4, 1, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1010111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, false, row, col, 3, col + 4, 1, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1011000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 3, 2, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1011001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, false, row, col, 1, col + 3, 2, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1011010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, false, row, col + 1, 1, col + 3, 2, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1011011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, false, row, col, 2, col + 3, 2, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1011100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 2, 3, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1011101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, false, row, col, 1, col + 2, 3, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1011110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 1, 4, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1011111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 5, col + 6, 1)
    }) as FPtrDyn;
    ops[0b1100000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, false, row, col + 5, 2)
    }) as FPtrDyn;
    ops[0b1100001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 1, col + 5, 2)
    }) as FPtrDyn;
    ops[0b1100010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 1, 1, col + 5, 2)
    }) as FPtrDyn;
    ops[0b1100011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 2, col + 5, 2)
    }) as FPtrDyn;
    ops[0b1100100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 2, 1, col + 5, 2)
    }) as FPtrDyn;
    ops[0b1100101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, false, row, col, 1, col + 2, 1, col + 5, 2)
    }) as FPtrDyn;
    ops[0b1100110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 1, 2, col + 5, 2)
    }) as FPtrDyn;
    ops[0b1100111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 3, col + 5, 2)
    }) as FPtrDyn;
    ops[0b1101000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 3, 1, col + 5, 2)
    }) as FPtrDyn;
    ops[0b1101001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, false, row, col, 1, col + 3, 1, col + 5, 2)
    }) as FPtrDyn;
    ops[0b1101010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, false, row, col + 1, 1, col + 3, 1, col + 5, 2)
    }) as FPtrDyn;
    ops[0b1101011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, false, row, col, 2, col + 3, 1, col + 5, 2)
    }) as FPtrDyn;
    ops[0b1101100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 2, 2, col + 5, 2)
    }) as FPtrDyn;
    ops[0b1101101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, false, row, col, 1, col + 2, 2, col + 5, 2)
    }) as FPtrDyn;
    ops[0b1101110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 1, 3, col + 5, 2)
    }) as FPtrDyn;
    ops[0b1101111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 4, col + 5, 1)
    }) as FPtrDyn;
    ops[0b1110000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, false, row, col + 4, 3)
    }) as FPtrDyn;
    ops[0b1110001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 1, col + 4, 3)
    }) as FPtrDyn;
    ops[0b1110010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 1, 1, col + 4, 3)
    }) as FPtrDyn;
    ops[0b1110011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 2, col + 4, 3)
    }) as FPtrDyn;
    ops[0b1110100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 2, 1, col + 4, 3)
    }) as FPtrDyn;
    ops[0b1110101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, false, row, col, 1, col + 2, 1, col + 4, 3)
    }) as FPtrDyn;
    ops[0b1110110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 1, 2, col + 4, 3)
    }) as FPtrDyn;
    ops[0b1110111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 3, col + 4, 3)
    }) as FPtrDyn;
    ops[0b1111000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, false, row, col + 3, 4)
    }) as FPtrDyn;
    ops[0b1111001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 1, col + 3, 4)
    }) as FPtrDyn;
    ops[0b1111010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col + 1, 1, col + 3, 4)
    }) as FPtrDyn;
    ops[0b1111011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 2, col + 3, 4)
    }) as FPtrDyn;
    ops[0b1111100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, false, row, col + 2, 5)
    }) as FPtrDyn;
    ops[0b1111101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, false, row, col, 1, col + 2, 5)
    }) as FPtrDyn;
    ops[0b1111110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, false, row, col + 1, 6)
    }) as FPtrDyn;
    ops[0b1111111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, false, row, col, 7)
    }) as FPtrDyn;
    ops[0b10000000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, true, row, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10000001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10000010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 1, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10000011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col, 2, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10000100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 2, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10000101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 1, col + 2, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10000110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 1, 2, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10000111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col, 3, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10001000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 3, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10001001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 1, col + 3, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10001010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col + 1, 1, col + 3, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10001011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 2, col + 3, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10001100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 2, 2, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10001101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 1, col + 2, 2, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10001110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 1, 3, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10001111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col, 4, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10010000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 4, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10010001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 1, col + 4, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10010010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col + 1, 1, col + 4, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10010011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 2, col + 4, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10010100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col + 2, 1, col + 4, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10010101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_four(rles, open, true, row, col, 1, col + 2, 1, col + 4, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10010110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col + 1, 2, col + 4, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10010111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 3, col + 4, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10011000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 3, 2, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10011001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 1, col + 3, 2, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10011010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col + 1, 1, col + 3, 2, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10011011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 2, col + 3, 2, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10011100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 2, 3, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10011101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 1, col + 2, 3, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10011110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 1, 4, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10011111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col, 5, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10100000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 5, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10100001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 1, col + 5, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10100010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col + 1, 1, col + 5, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10100011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 2, col + 5, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10100100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col + 2, 1, col + 5, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10100101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_four(rles, open, true, row, col, 1, col + 2, 1, col + 5, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10100110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col + 1, 2, col + 5, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10100111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 3, col + 5, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10101000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col + 3, 1, col + 5, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10101001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_four(rles, open, true, row, col, 1, col + 3, 1, col + 5, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10101010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_four(rles, open, true, row, col + 1, 1, col + 3, 1, col + 5, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10101011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_four(rles, open, true, row, col, 2, col + 3, 1, col + 5, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10101100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col + 2, 2, col + 5, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10101101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_four(rles, open, true, row, col, 1, col + 2, 2, col + 5, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10101110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col + 1, 3, col + 5, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10101111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 4, col + 5, 1, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10110000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 4, 2, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10110001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 1, col + 4, 2, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10110010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col + 1, 1, col + 4, 2, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10110011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 2, col + 4, 2, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10110100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col + 2, 1, col + 4, 2, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10110101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_four(rles, open, true, row, col, 1, col + 2, 1, col + 4, 2, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10110110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col + 1, 2, col + 4, 2, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10110111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 3, col + 4, 2, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10111000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 3, 3, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10111001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 1, col + 3, 3, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10111010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col + 1, 1, col + 3, 3, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10111011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 2, col + 3, 3, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10111100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 2, 4, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10111101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 1, col + 2, 4, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10111110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 1, 5, col + 7, 1)
    }) as FPtrDyn;
    ops[0b10111111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col, 6, col + 7, 1)
    }) as FPtrDyn;
    ops[0b11000000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, true, row, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11000001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col, 1, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11000010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 1, 1, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11000011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col, 2, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11000100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 2, 1, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11000101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 1, col + 2, 1, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11000110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 1, 2, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11000111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col, 3, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11001000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 3, 1, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11001001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 1, col + 3, 1, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11001010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col + 1, 1, col + 3, 1, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11001011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 2, col + 3, 1, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11001100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 2, 2, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11001101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 1, col + 2, 2, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11001110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 1, 3, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11001111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col, 4, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11010000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 4, 1, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11010001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 1, col + 4, 1, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11010010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col + 1, 1, col + 4, 1, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11010011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 2, col + 4, 1, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11010100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col + 2, 1, col + 4, 1, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11010101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_four(rles, open, true, row, col, 1, col + 2, 1, col + 4, 1, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11010110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col + 1, 2, col + 4, 1, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11010111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 3, col + 4, 1, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11011000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 3, 2, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11011001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 1, col + 3, 2, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11011010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col + 1, 1, col + 3, 2, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11011011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 2, col + 3, 2, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11011100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 2, 3, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11011101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 1, col + 2, 3, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11011110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 1, 4, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11011111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col, 5, col + 6, 2)
    }) as FPtrDyn;
    ops[0b11100000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, true, row, col + 5, 3)
    }) as FPtrDyn;
    ops[0b11100001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col, 1, col + 5, 3)
    }) as FPtrDyn;
    ops[0b11100010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 1, 1, col + 5, 3)
    }) as FPtrDyn;
    ops[0b11100011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col, 2, col + 5, 3)
    }) as FPtrDyn;
    ops[0b11100100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 2, 1, col + 5, 3)
    }) as FPtrDyn;
    ops[0b11100101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 1, col + 2, 1, col + 5, 3)
    }) as FPtrDyn;
    ops[0b11100110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 1, 2, col + 5, 3)
    }) as FPtrDyn;
    ops[0b11100111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col, 3, col + 5, 3)
    }) as FPtrDyn;
    ops[0b11101000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 3, 1, col + 5, 3)
    }) as FPtrDyn;
    ops[0b11101001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 1, col + 3, 1, col + 5, 3)
    }) as FPtrDyn;
    ops[0b11101010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col + 1, 1, col + 3, 1, col + 5, 3)
    }) as FPtrDyn;
    ops[0b11101011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 2, col + 3, 1, col + 5, 3)
    }) as FPtrDyn;
    ops[0b11101100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 2, 2, col + 5, 3)
    }) as FPtrDyn;
    ops[0b11101101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 1, col + 2, 2, col + 5, 3)
    }) as FPtrDyn;
    ops[0b11101110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 1, 3, col + 5, 3)
    }) as FPtrDyn;
    ops[0b11101111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col, 4, col + 5, 3)
    }) as FPtrDyn;
    ops[0b11110000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, true, row, col + 4, 4)
    }) as FPtrDyn;
    ops[0b11110001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col, 1, col + 4, 4)
    }) as FPtrDyn;
    ops[0b11110010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 1, 1, col + 4, 4)
    }) as FPtrDyn;
    ops[0b11110011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col, 2, col + 4, 4)
    }) as FPtrDyn;
    ops[0b11110100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 2, 1, col + 4, 4)
    }) as FPtrDyn;
    ops[0b11110101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_three(rles, open, true, row, col, 1, col + 2, 1, col + 4, 4)
    }) as FPtrDyn;
    ops[0b11110110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 1, 2, col + 4, 4)
    }) as FPtrDyn;
    ops[0b11110111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col, 3, col + 4, 4)
    }) as FPtrDyn;
    ops[0b11111000] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, true, row, col + 3, 5)
    }) as FPtrDyn;
    ops[0b11111001] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col, 1, col + 3, 5)
    }) as FPtrDyn;
    ops[0b11111010] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col + 1, 1, col + 3, 5)
    }) as FPtrDyn;
    ops[0b11111011] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col, 2, col + 3, 5)
    }) as FPtrDyn;
    ops[0b11111100] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, true, row, col + 2, 6)
    }) as FPtrDyn;
    ops[0b11111101] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_two(rles, open, true, row, col, 1, col + 2, 6)
    }) as FPtrDyn;
    ops[0b11111110] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, true, row, col + 1, 7)
    }) as FPtrDyn;
    ops[0b11111111] = &(|rles : &mut Vec<RunLength>, open : &mut bool, row : usize, col : usize| {
        add_one(rles, open, true, row, col, 8)
    }) as FPtrDyn;

    ops
}

/* A good characteristic of the bitonal encoder is that no computation
is performed on the non-match branch (which should improve performance
for sparser binary images). The current RLE is pushed at either a neighboring
match with zero trailing zeros or the current RLE at leading zeros, or the
last column. The short-circuit leads to only one of those conditions being
evaluated. The representation is "lossy" by omitting isolated positive pixels,
and never proposing positive pixels where there should be none, so that
connectivity analysis of the RunLengths never join shapes that wouldn't
be joined in the lossless representation (alhtough it might separate them). */
#[inline(always)]
fn lossy_encode_bitonal_pixel(
    row_ix : usize,
    col_ix : usize,
    max_ix : usize,
    row : &[u8],
    curr_rle : &mut Option<RunLength>,
    rles : &mut Vec<RunLength>,
) {
    let px = unsafe { row.get_unchecked(col_ix) };
    if *px > 0 {
        if let Some(rle) = curr_rle.as_mut() {

            // Note: At this point, it is known that column>0 (or curr_rle would be empty) or
            // trailing_zeros is 0 (because trailing_zeros of previous column > 0
            // empties the curr_rle variable) So falling here means the two RLEs are connected.
            let length_incr = 8 - px.leading_zeros() as usize;

            // Ignore non-contiguous RLEs. The lossy representation
            // will always be zero when not contiguous, not messing with connectivity
            // analysis.
            let is_contiguous = length_incr == px.count_ones() as usize;
            if !is_contiguous {
                rles.push(rle.clone());
                *curr_rle = None;
                return;
            }

            rle.length += length_incr;

            // The last condition is never evaluated at last column due to short circuit,
            // so there is no risk of a buffer overflow.
            let must_push = length_incr < 8 ||
                col_ix == max_ix ||
                row[col_ix+1].trailing_zeros() > 0;
            if must_push {
                rles.push(rle.clone());
                *curr_rle = None;
            }
        } else {
            let trailing = px.trailing_zeros() as usize;
            let leading = px.leading_zeros() as usize;

            let is_contiguous = 8 - leading - trailing == px.count_ones() as usize;
            if !is_contiguous {
                return;
            }

            let col_start = col_ix * 8 + trailing;
            let length = 8 - trailing - leading;
            let new_rle = RunLength { start : (row_ix, col_start), length };

            let must_push = leading > 0 ||
                col_ix == max_ix ||
                row[col_ix+1].trailing_zeros() > 0;
            if must_push {
                // RLE starts and ends here: Push it instead of setting as current.
                rles.push(new_rle);
            } else {
                // RLE might continue
                *curr_rle = Some(new_rle);
            }
        }
    }
}

#[inline(always)]
fn lossless_encode_bitonal_pixel(
    lut : &[FPtrDyn],
    row_ix : usize,
    col_ix : usize,
    row : &[u8],
    open : &mut bool,
    rles : &mut Vec<RunLength>,
) {
    unsafe {
        let px = row.get_unchecked(col_ix);
        if *open {
            let ones = px.trailing_ones() as usize;
            rles.last_mut().unwrap().length += ones;
            if ones < 8 {
                (&*lut[(px >> ones) as usize])(rles, open, row_ix, col_ix*8 + ones);
            }
        } else {
            (&*lut[*px as usize])(rles, open, row_ix, col_ix*8);
        }
    }
}

/* Note the encoding is in reverse order: Empty spaces at the front of the RunLength
is examined with trailing_zeros, and empty spaces at the end of the RunLength are examined
with leading_zeros. This representation is lossy because it misses zeroed bits inside each
8-bit sequence to gain performance by only evaluating the trailing and leading bits
within each packed region. If the objects of interest are solid and have no holes (i.e are convex),
the representation is complete. */
fn lossy_encode_bitonal_row(row : &[u8], row_ix : usize, rles : &mut Vec<RunLength>) {
    let mut curr_rle : Option<RunLength> = None;
    let max_ix = row.len()-1;
    for col_ix in 0..row.len() {
        lossy_encode_bitonal_pixel(row_ix, col_ix, max_ix, row, &mut curr_rle, rles);
    }
    // assert!(curr_rle.is_none());
}

fn lossless_encode_bitonal_row(lut : &[FPtrDyn], row : &[u8], row_ix : usize, rles : &mut Vec<RunLength>) {
    let mut open = false;
    for col_ix in 0..row.len() {
        lossless_encode_bitonal_pixel(lut, row_ix, col_ix, row, &mut open, rles);
    }
}

// cargo test --lib -- bitonal_encoder --nocapture
#[test]
fn bitonal_encoder() {

    use crate::draw::*;
    use crate::util::Timer;

    let mut img = ImageBuf::<u8>::new_constant(128, 128, 0);
    img.draw(Mark::Dot((64, 64), 32), 255);
    // img.window_mut((64 - 16, 64 - 16), (32, 32)).unwrap().fill(0);
    /*img.window_mut((10, 10), (20, 20)).unwrap().fill(255);
    img.window_mut((10, 35), (20, 20)).unwrap().fill(255);
    img.window_mut((10, 100), (20, 20)).unwrap().fill(255);
    img.window_mut((100, 10), (20, 20)).unwrap().fill(255);*/

    /*let mut t = Timer::start();
    let mut enc = ChainEncoder::new(img.height(), img.width());
    enc.encode(&img, None);
    t.time("chain code");*/

    // let mut enc = BitEncoder::new(img.height(), img.width());
    let mut enc = BitonalEncoder::new(img.height(), img.width());
    // let mut enc = PyrBitonalEncoder::new(img.height(), img.width(), 3);

    let mut t = Timer::start();
    let rles = enc.calculate(&img);
    // assert!(rles.rles.iter().any(|r| r.start.0 == 79 ));
    // println!("{:?}", rles);

    // enc.edge_xor_img.show();
    t.time("encode");
    for rle in &rles.rles {
        for pt in rle.points() {
            img[pt] = 127;
        }
    }
    t.time("draw");
    img.show();

}

/*
pub struct LogicFold { }

pub struct SumFold { }
*/

/*pub struct BitChainEncoder {
    enc : BitEncoder,
    edge_xor_img : ImageBuf<u8>,
    shift_right : ImageBuf<u8>,
    shift_down_left : ImageBuf<u8>,
    shift_down : ImageBuf<u8>,
    shift_down_right : ImageBuf<u8>
}

impl BitChainEncoder {

    pub fn new(height : usize, width : usize) -> Self {
        let enc = BitEncoder::new(height, width);
        let edge_xor_img = ImageBuf::<u8>::new_constant(height, width / 8, 0);
        let shift_right = edge_xor_img.clone();
        let shift_down_left = edge_xor_img.clone();
        let shift_down = edge_xor_img.clone();
        let shift_down_right = edge_xor_img.clone();
        Self { enc, edge_xor_img, shift_right, shift_down_left, shift_down, shift_down_right }
    }

    pub fn calculate<S>(&mut self, img : &Image<u8, S>)
    where
        S : Storage<u8>
    {
        self.enc.fold(img);
        binary_edges(&self.enc.index_sum, &mut self.edeg_xor_img);
        // (edge_xor_img) >> 1 to right AND edge_xor_img
        // Also repeat with others
        // Build Vec<OpenChain>
        //
    }

}*/

/* Folding bit RunLength, Point and Chain encoders beat the performance
of full raster scan encoders by making more heavy use of vectorized
registers (holding u8x16) to process and exclude regions of the image
without white pixels. We divide the image into 8 sectors, which are
folded into a thin image of dimensions h x (w/8). Each byte of this
image encode which sectors contain relevant information (with the first
bit for the first sector and so on up to the eighth bit for the last sector).
Instead of looking at individual pixels in each section, we can exclude 8 pixels
at a time for all pixels 0b00. If the pixel is nonzero, we evaluate sectors
by excluding the trailing zeros of the pattern, and proceed up the point where
the pixel is 0b00, looking at the full 8 pixels only in the cases where all 8
sectors are white. */
#[derive(Debug, Clone)]
pub struct BitEncoder {

    indices : ImageBuf<u8>,

    masked_indices : ImageBuf<u8>,

    index_sum : ImageBuf<u8>,

    size : (usize, usize)

}

/* Those patterns [0..=7] that are folded with a bitwise OR across the 8 image sections. */
const BIT_INDICES : [u8; 8] = [0b00000001, 0b00000010, 0b00000100, 0b00001000, 0b00010000, 0b00100000, 0b01000000, 0b10000000];

/* Those patterns [0..=7] are used to probe the index via a bitwise OR */
// The bit probe at position 8 isn't really used, but it is required if trailing_zeros() return 0 to avoid a buffer overflow.
const BIT_PROBES : [u8; 9] = [0b11111111, 0b11111110, 0b11111100, 0b11111000, 0b11110000, 0b11100000, 0b11000000, 0b10000000, 0b00000000];

impl BitEncoder {

    pub fn new(height : usize, width : usize) -> Self {
        assert!(width % 8 == 0);
        let strip_width = width / 8;
        let mut indices = ImageBuf::new_constant(height, width, 0);
        for i in 0..8 {
            indices.window_mut((0, i*strip_width), (height, strip_width)).unwrap().fill(BIT_INDICES[i]);
        }
        Self {
            indices,
            masked_indices : ImageBuf::new_constant(height, width, 0),
            index_sum : ImageBuf::new_constant(height, strip_width, 0),
            size : (height, width)
        }
    }

    fn fold<S>(&mut self, img : &Image<u8, S>)
    where
        S : Storage<u8>
    {
        assert!(img.size() == self.size);
        img.mul_to(&self.indices, &mut self.masked_indices);
        fold_or_image(&self.masked_indices, &mut self.index_sum);
    }

    // img is assumed to be a binary image with 1's at hits.
    pub fn calculate<S>(&mut self, img : &Image<u8, S>) -> RunLengthCode
    where
        S : Storage<u8>
    {
        self.fold(img);
        let strip_width = img.width() / 8;
        encode_from_index_sum(&self.index_sum, strip_width)
    }

}

// cargo test --lib -- distinct_sum --nocapture
#[test]
fn distinct_sum() {
    /*let root_vals = [1, 2, 16, 17];
    let mut sums = Vec::new();
    let mut prev = Vec::new();
    for lvl in 0..4 {
        sums.clear();
        if lvl == 0 {
            sums.push(root_vals[0]);
            sums.push(root_vals[1]);
            sums.push(root_vals[0] + root_vals[1]);
            sums.push(root_vals[2]);
            sums.push(root_vals[3]);
            sums.push(root_vals[2] + root_vals[3]);
        } else {
            for i in 0..=(prev.len() / 3) {
                sums.push(prev[i]);
                sums.push(prev[i+1]);
                sums.push(prev[i] + prev[i+1]);
            }
        }
        println!("{:?}", sums);
        prev = sums.clone();
    }*/
}

pub struct ScaledBitEncoder {
    enc : BitEncoder,
    or_img : ImageBuf<u8>,
    small_index_sum : ImageBuf<u8>,
    scale : usize
}

impl ScaledBitEncoder {

    // scale should be a power of two.
    pub fn new(height : usize, width : usize, scale : usize) -> Self {
        assert!(scale.next_power_of_two() == scale);
        Self {
            enc : BitEncoder::new(height, width),
            or_img : ImageBuf::<u8>::new_constant(height, width / 8, 0),
            small_index_sum : ImageBuf::<u8>::new_constant(height / scale, width / scale, 0),
            scale
        }
    }

    pub fn calculate<S>(&mut self, img : &Image<u8, S>) -> RunLengthCode
    where
        S : Storage<u8>
    {
        use ripple::resample::Resample;
        self.enc.fold(img);
        let mut red_sz = self.enc.index_sum.size();
        red_sz.0 -= 1;
        red_sz.1 -= 1;
        let shift_right = self.enc.index_sum.window((0, 1), red_sz).unwrap();
        let shift_down = self.enc.index_sum.window((1, 0), red_sz).unwrap();
        let original = self.enc.index_sum.window((0, 0), red_sz).unwrap();
        let mut or_img = self.or_img.window_mut((0, 0), red_sz).unwrap();
        original.or_to(&shift_right, &mut or_img);
        or_img.or_assign(&shift_down);
        or_img.downsample_to(&mut self.small_index_sum, ripple::resample::Downsample::Aliased);
        let strip_width = (img.width() / 8) / self.scale;
        encode_from_index_sum(&self.small_index_sum, strip_width)
    }

}

#[derive(Debug, Clone)]
pub struct EdgeBitEncoder {
    enc : BitEncoder,
    edge_xor_img : ImageBuf<u8>,
}

impl EdgeBitEncoder {

    // scale should be a power of two.
    pub fn new(height : usize, width : usize) -> Self {
        Self {
            enc : BitEncoder::new(height, width),
            edge_xor_img : ImageBuf::<u8>::new_constant(height, width / 8, 0)
        }
    }

    /*pub fn calculate2<S>(&mut self, img : &Image<u8, S>) -> RunLengthCode
    where
        S : Storage<u8>
    {
        let mut dst = unsafe { ImageBuf::<u8>::new_constant(img.height(), img.width(), 0) };
        let mut sz = img.shape();
        sz.1 -= 1;
        img.window((0, 0), sz).unwrap().xor_to(&img.window((0, 1), sz).unwrap(), &mut dst.window_mut((0, 0), sz).unwrap());
        // dst.scalar_mul_mut(255);
        dst.show();
        self.enc.fold(&dst);
        let strip_width = (img.width() / 8);
        encode_from_edges(&self.enc.index_sum, &self.edge_xor_img, strip_width)
    }*/

    pub fn calculate<S>(&mut self, img : &Image<u8, S>) -> RunLengthCode
    where
        S : Storage<u8>
    {
        self.enc.fold(img);
        binary_edges(&self.enc.index_sum, &mut self.edge_xor_img);
        let strip_width = (img.width() / 8);
        encode_from_edges(&self.enc.index_sum, &self.edge_xor_img, strip_width)
    }

}

fn binary_edges<S, T>(index_sum : &Image<u8, S>, edge_xor_img : &mut Image<u8, T>)
where
    S : Storage<u8>,
    T : StorageMut<u8>
{
    assert!(index_sum.size() == edge_xor_img.size());

    let mut edge_ixs_sz = edge_xor_img.size();
    edge_ixs_sz.1 -= 1;

    let orig = index_sum.window((0, 0), edge_ixs_sz).unwrap();
    let shifted = index_sum.window((0, 1), edge_ixs_sz).unwrap();
    orig.xor_to(&shifted, &mut edge_xor_img.window_mut((0, 0), edge_ixs_sz).unwrap());

    // This calculates the xor of the last column of each section with the
    // first column of the next section using a bit shift of the first column
    // (thus aligning the first column of the next sector with the last column
    // of the previous sector).
    let (h, w) = index_sum.size();
    let fst_col = index_sum.window((0, 0), (h,1)).unwrap();
    let last_col = index_sum.window((0, w-1), (h,1)).unwrap();
    let mut last_xor_col = edge_xor_img.window_mut((0, w-1), (h, 1)).unwrap();
    fst_col.shr_to(1, &mut last_xor_col);
    last_xor_col.xor_assign(&last_col);
}

use std::ops::BitOr;

fn column_range(bytes : &[u8]) -> Range<usize> {
    let row_byte = bytes.iter().fold(0, |bytes, b| bytes.bitor(b) );
    Range { start : row_byte.leading_zeros() as usize, end : row_byte.trailing_zeros() as usize }
}

fn encode_from_index_sum<S>(index_sum : &Image<u8, S>, strip_width : usize) -> RunLengthCode
where
    S : Storage<u8>
{
    let mut rles : Vec<RunLength> = Default::default();
    let mut rows : Vec<Range<usize>> = Default::default();
    let mut rle_front : [Option<RunLength>; 8] = [None; 8];

    let strip_offsets : Vec<_> = (0..8).map(|i| strip_width*i ).collect();

    let height = index_sum.height();

    /*// Fold sections that must be evaluated across all offsets, to get bounds for the actual iterator.
    let mut section_evals : Box<dyn Iterator<Item=Range<usize>>> = match index_sum.width() % 16 == 0 {
        true => {
            let iter = (0..height).map(|r| {
                let packed_row : wide::u8x16 = index_sum.row(r).unwrap()
                    .chunks(16)
                    .fold(wide::u8x16::ZERO, |bytes, b| bytes.bitor(wide::u8x16::from(b)) );
                column_range(&packed_row.as_array_ref()[..])
            });
            Box::new(iter)
        },
        false => {

            // Default: Eval all columns.
            // Box::new((0..height).map(|_| Range { 0..index_sum.width() } ))

            let iter = (0..height).map(|r| column_range(index_sum.row(r).unwrap()) );
            Box::new(iter)
        }
    };*/

    // let mut pxs = self.index_sum.pixels(1);

    for r in (0..height) {
        let n_before = rles.len();
        let row = unsafe { index_sum.row_unchecked(r) };

        for c in 0..index_sum.width() {
        // for c in col_evals.next().unwrap() {
            // let bit = self.index_sum[(r, c)];
            // let bit = unsafe { self.index_sum.get_unchecked((r, c)) };
            // let bit = pxs.next().unwrap();
            let bit = unsafe { row.get_unchecked(c) };

            // let mut i = 0;
            let mut i = bit.trailing_zeros() as usize;
            // if i == 8 { continue; }

            // This bitwise op selects which ones of the 8 sections
            // should be examined (there is one bit for each).
            let mut bit_and_ix = bit & BIT_PROBES[i];
            while bit_and_ix > 0 {

                // All are valid options here:
                // if bit_and_ix & (1 << i) == (1 << i) {
                // if (bit_and_ix >> i) % 2 == 1 {
                // if (bit_and_ix >> i).trailing_zeros() == 0 {
                if bit_and_ix & BIT_INDICES[i] == BIT_INDICES[i] {
                    let col = strip_offsets[i] + c;
                    if let Some(mut rle) = rle_front[i].as_mut() {
                        if (col - rle.start.1) - rle.length == 0 {
                            rle.length += 1;
                        } else {
                            rles.push(rle.clone());
                            *rle = RunLength { start : (r, col), length : 1 };
                        }
                    } else {
                        rle_front[i] = Some(RunLength { start : (r, col), length : 1 });
                    }
                }

                i += 1;
                bit_and_ix = bit & BIT_PROBES[i];
            }
        }
        close_rles(&mut rles, &mut rle_front);
        organize_rles_at_last_row(&mut rles, &mut rows, n_before);
    }
    // timer.time("encoding");
    RunLengthCode {
        rles,
        rows
    }
}

fn close_rles(rles : &mut Vec<RunLength>, rle_front : &mut [Option<RunLength>; 8]) {
    for i in 0..8 {
        if let Some(rle) = rle_front[i].take() {
            rles.push(rle);
        }
    }
}

fn organize_rles_at_last_row(rles : &mut Vec<RunLength>, rows : &mut Vec<Range<usize>>, n_before : usize) {
    let mut n_after = rles.len();
    if n_after > n_before {
        rles[n_before..n_after].sort_by(|a, b| a.start.1.cmp(&b.start.1) );
        for i in (n_before..(n_after - 1)).rev() {
            if rles[i].start.1 + rles[i].length == rles[i+1].start.1 {
                rles[i].length += rles[i+1].length;
                rles.remove(i+1);
            }
        }
        n_after = rles.len();
        rows.push(Range { start : n_before, end : n_after });
    }
}

fn encode_from_edges<S, T>(index_sum : &Image<u8, S>, edge_index_sum : &Image<u8, T>, strip_width : usize) -> RunLengthCode
where
    S : Storage<u8>,
    T : Storage<u8>,
{
    let mut rles : Vec<RunLength> = Default::default();
    let mut rows : Vec<Range<usize>> = Default::default();
    let mut cols : [SmallVec<[usize; 32]>; 8] = Default::default();
    let strip_offsets : Vec<_> = (0..8).map(|i| strip_width*i ).collect();
    let height = edge_index_sum.height();
    let lst_col = index_sum.width()-1;

    for r in (0..height) {
        let row = unsafe { edge_index_sum.row_unchecked(r) };
        let mut ncols = 0;
        for c in 0..edge_index_sum.width() {

            let bit = unsafe { row.get_unchecked(c) };
            let mut i = bit.trailing_zeros() as usize;
            let mut bit_and_ix = bit & BIT_PROBES[i];

            while bit_and_ix > 0 {
                if bit_and_ix & BIT_INDICES[i] == BIT_INDICES[i] {
                    let col = strip_offsets[i] + c;
                    cols[i].push(col);
                    ncols += 1;
                }
                i += 1;
                bit_and_ix = bit & BIT_PROBES[i];
            }
        }

        if ncols >= 1 {
            let fst_white = index_sum[(r, 0)] & 1 == 1;
            let lst_white = index_sum[(r, lst_col)] & 0b10000000 == 0b10000000;
            collect_rles(&mut cols, &mut rles, &mut rows, fst_white, lst_white, r, strip_width);
        }
    }
    // timer.time("encoding");
    RunLengthCode {
        rles,
        rows
    }
}

fn collect_rles(
    cols : &mut [SmallVec<[usize; 32]>; 8],
    rles : &mut Vec<RunLength>,
    rows : &mut Vec<Range<usize>>,
    fst_white : bool,
    lst_white : bool,
    row : usize,
    strip_width : usize
) {
    let n_before = rles.len();
    // if cols.iter().map(|cs| cs.len() ).sum::<usize>() == 0 {
    //    return;
    // }

    // let mut cols_iter = cols.iter_mut().map(|cs| cs.drain(..) ).flatten();
    let mut cols_iter = cols.iter_mut()
        .filter_map(|cs| if cs.len() > 0 { Some(cs.drain(..)) } else { None } )
        .flatten();

    if fst_white {
        if let Some(c) = cols_iter.next() {
            let fst = RunLength { start : (row, 0), length : c  };
            rles.push(fst);
        }
    }

    loop {
        match (cols_iter.next(), cols_iter.next()) {
            (Some(c1), Some(c2)) => {
                rles.push(RunLength { start : (row, c1), length : c2 - c1 });
            },
            (Some(c1), None) => {
                // It is undecidable if this should extend to end or be a lenght-1 encoding.
                // unless we look at the original image.
                if lst_white {
                    rles.push(RunLength { start : (row, c1), length : strip_width*8 - c1 });
                } else {
                    rles.push(RunLength { start : (row, c1), length : 1 });
                }
            },
            _ => break
        }
    }

    let n_after = rles.len();
    rows.push(Range { start : n_before, end : n_after });
    // cols.iter_mut().for_each(|cs| cs.clear() );
}

fn fold_or_image<S, T>(w : &Image<u8, S>, dst : &mut Image<u8, T>)
where
    S : Storage<u8>,
    T : StorageMut<u8>
{
    assert!(w.width() % 8 == 0);
    let strip_w = w.width() / 8;
    assert!(strip_w == dst.width());
    assert!(w.height() == dst.height());
    let fold_sz = (w.height(), strip_w);
    dst.copy_from(&w.window((0, 0), fold_sz).unwrap());
    for i in 1..8 {
        dst.or_assign(&w.window((0, strip_w*i), fold_sz).unwrap());
    }
}

// cargo test --lib -- bit_encoding --nocapture
#[test]
fn bit_encoding() {

    use crate::draw::*;
    use crate::util::Timer;

    let mut img = ImageBuf::<u8>::new_constant(128, 128, 0);
    img.draw(Mark::Dot((64, 64), 32), 1);

    /*let mut t = Timer::start();
    let mut enc = ChainEncoder::new(img.height(), img.width());
    enc.encode(&img, None);
    t.time("chain code");*/

    // let mut enc = BitEncoder::new(img.height(), img.width());
    let mut enc = EdgeBitEncoder::new(img.height(), img.width());

    let mut t = Timer::start();
    let rles = enc.calculate(&img);
    // enc.edge_xor_img.show();
    t.time("encode");
    for rle in &rles.rles {
        for pt in rle.points() {
            img[pt] = 255;
        }
    }
    t.time("draw");

    let _ = rles.split();
    t.time("split");

    let _ = rles.graph();
    t.time("graph");

    let mut img2 = ImageBuf::<u8>::new_constant(128, 128, 0);
    img2.draw(Mark::Dot((64, 64), 32), 255);
    RunLengthCode::encode(&img2.full_window());
    t.time("RLE");

    img.show();
}

const fn sum_pattern(offset : u8) -> [u8; 16] {
    let mut arr = [0; 16];
    let mut i = 0;
    while i < 16 {
        arr[i] = i as u8 + 1 + offset;
        i += 1;
    }
    arr
}

const fn sum_arrays(a : [u8; 16], b : [u8; 16]) -> [u8; 16] {
    // This can't be a loop in const context (Exceeded interpreter step limit error).
    [
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
        a[3] + b[3],
        a[4] + b[4],
        a[5] + b[5],
        a[6] + b[6],
        a[7] + b[7],
        a[8] + b[8],
        a[9] + b[9],
        a[10] + b[10],
        a[11] + b[11],
        a[12] + b[12],
        a[13] + b[13],
        a[14] + b[14],
        a[15] + b[15],
    ]
}

const ONE_PATTERN : [[u8; 16]; 4] = [
    sum_pattern(0),
    sum_pattern(16),
    sum_pattern(32),
    sum_pattern(48)
];

const TWO_PATTERN : [[u8; 16]; 6] = [
    sum_arrays(ONE_PATTERN[0], ONE_PATTERN[1]),
    sum_arrays(ONE_PATTERN[0], ONE_PATTERN[2]),
    sum_arrays(ONE_PATTERN[0], ONE_PATTERN[3]),
    sum_arrays(ONE_PATTERN[1], ONE_PATTERN[2]),
    sum_arrays(ONE_PATTERN[1], ONE_PATTERN[3]),
    sum_arrays(ONE_PATTERN[2], ONE_PATTERN[3]),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OneFoldOrder {
    Fst,
    Snd,
    Thd,
    Fth
}

const fn get_one_fold_order(patt_ix : usize) -> OneFoldOrder {
    match patt_ix {
        0 => OneFoldOrder::Fst,
        1 => OneFoldOrder::Snd,
        2 => OneFoldOrder::Thd,
        3 => OneFoldOrder::Fth,
        _ => panic!("Invalid order")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TwoFoldOrder {
    FstSnd,
    FstThd,
    FstFth,
    SndThd,
    SndFth,
    ThdFth
}

const fn get_two_fold_order(patt_ix : usize) -> TwoFoldOrder {
    match patt_ix {
        0 => TwoFoldOrder::FstSnd,
        1 => TwoFoldOrder::FstThd,
        2 => TwoFoldOrder::FstFth,
        3 => TwoFoldOrder::SndThd,
        4 => TwoFoldOrder::SndFth,
        5 => TwoFoldOrder::ThdFth,
        _ => panic!("Invalid order")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreeFoldOrder {
    FstSndThd,
    FstSndFth,
    FstThdFth,
    SndThdFth
}

const fn get_three_fold_order(patt_ix : usize) -> ThreeFoldOrder {
    match patt_ix {
        0 => ThreeFoldOrder::FstSndThd,
        1 => ThreeFoldOrder::FstSndFth,
        2 => ThreeFoldOrder::FstThdFth,
        3 => ThreeFoldOrder::SndThdFth,
        _ => panic!("Invalid fold order")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fold {
    Empty,
    Undecidable,
    One(usize, OneFoldOrder),
    Two([usize; 2], TwoFoldOrder),
    Three([usize; 3], ThreeFoldOrder),
    Four([usize; 4])
}

const fn get_fold(n_patts : usize, patt_ix : usize, pos : usize) -> Fold {
    match (n_patts, patt_ix) {
        (1, 0) => Fold::One(pos, OneFoldOrder::Fst),                                    // 0
        (1, 1) => Fold::One(16+pos, OneFoldOrder::Snd),                                 // 1
        (1, 2) => Fold::One(32+pos, OneFoldOrder::Thd),                                 // 2
        (1, 3) => Fold::One(48+pos, OneFoldOrder::Fth),                                 // 3
        (2, 0) => Fold::Two([pos, 16+pos], TwoFoldOrder::FstSnd),                       // 0,1
        (2, 1) => Fold::Two([pos, 32+pos], TwoFoldOrder::FstThd),                       // 0,2
        (2, 2) => Fold::Two([pos, 48+pos], TwoFoldOrder::FstFth),                       // 0,3
        (2, 3) => Fold::Two([16+pos, 32+pos], TwoFoldOrder::SndThd),                    // 1,2
        (2, 4) => Fold::Two([16+pos, 48+pos], TwoFoldOrder::SndFth),                    // 1,3
        (2, 5) => Fold::Two([32+pos, 48+pos], TwoFoldOrder::ThdFth),                    // 2,3
        (3, 0) => Fold::Three([pos, 16+pos, 32+pos], ThreeFoldOrder::FstSndThd),        // 0, 1, 2
        (3, 1) => Fold::Three([pos, 16+pos, 48+pos], ThreeFoldOrder::FstSndFth),        // 0, 1, 3
        (3, 2) => Fold::Three([pos, 32+pos, 48+pos], ThreeFoldOrder::FstThdFth),        // 0, 2, 3
        (3, 3) => Fold::Three([16+pos, 32+pos, 48+pos], ThreeFoldOrder::SndThdFth),     // 1, 2, 3
        (4, 0) => Fold::Four([pos, 16+pos, 32+pos, 48+pos]),                            // 0, 1, 2, 3
        _ => panic!("Invalid fold")
    }
}

/*const TWO_PATTERN_FOLDS : [Fold; 6] = [
    Fold::Two(0, 1),
    Fold::Two(0, 2),
    Fold::Two(0, 3),
    Fold::Two(1, 2),
    Fold::Two(1, 3),
    Fold::Two(2, 3)
];*/

const THREE_PATTERN : [[u8; 16]; 4] = [
    sum_arrays(sum_arrays(ONE_PATTERN[0], ONE_PATTERN[1]), ONE_PATTERN[2]),
    sum_arrays(sum_arrays(ONE_PATTERN[0], ONE_PATTERN[1]), ONE_PATTERN[3]),
    sum_arrays(sum_arrays(ONE_PATTERN[0], ONE_PATTERN[2]), ONE_PATTERN[3]),
    sum_arrays(sum_arrays(ONE_PATTERN[1], ONE_PATTERN[2]), ONE_PATTERN[3])
];

/*const THREE_PATTERN_FOLDS : [Fold; 4] = [
    Fold::Three(0, 1, 2),
    Fold::Three(0, 1, 3),
    Fold::Three(0, 2, 3),
    Fold::Three(1, 2, 3)
];*/

const FOUR_PATTERN : [[u8; 16];1] = [sum_arrays(sum_arrays(sum_arrays(ONE_PATTERN[0], ONE_PATTERN[1]), ONE_PATTERN[2]), ONE_PATTERN[3])];

// cargo test --lib -- sum_conflicts --nocapture
#[test]
fn sum_conflicts() {
    let conflicts_two_two = [
        (2, 3, 0),
        (2, 3, 1),
        (2, 3, 2),
        (2, 3, 3),
        (2, 3, 4),
        (2, 3, 5),
        (2, 3, 6),
        (2, 3, 7),
        (2, 3, 8),
        (2, 3, 9),
        (2, 3, 10),
        (2, 3, 11),
        (2, 3, 12),
        (2, 3, 13),
        (2, 3, 14),
        (2, 3, 15),
    ];
    let conflicts_three_three : [(usize, usize, usize); 0] = [

    ];
    let conflicts_two_three = [
        (4, 0, 15),
        (5, 1, 15),
    ];
    let conflicts_one_two = [
        (2, 0, 15),
        (3, 1, 15)
    ];
    let conflicts_one_four : [(usize, usize, usize); 0] = [

    ];
    let conflicts_two_four : [(usize, usize, usize); 0] = [

    ];
    let conflicts_three_four : [(usize, usize, usize); 0] = [

    ];
    for i in 0..16 {
        for ix1 in 0..5 {
            for ix2 in (ix1+1)..6 {
                if TWO_PATTERN[ix1][i] == TWO_PATTERN[ix2][i] && !conflicts_two_two.iter().any(|c| c.0 == ix1 && c.1 == ix2 && c.2 == i ) {
                    panic!("Conflict (Two at {}, Two at {}) at pos {}", ix1, ix2, i);
                }
            }
        }

        for ix1 in 0..3 {
            for ix2 in (ix1+1)..4 {
                if THREE_PATTERN[ix1][i] == THREE_PATTERN[ix2][i] && !conflicts_three_three.iter().any(|c| c.0 == ix1 && c.1 == ix2 && c.2 == i ) {
                    panic!("Conflict (Three at {}, Three at {}) at pos {}", ix1, ix2, i);
                }
            }
        }

        for (ix1, one) in ONE_PATTERN.iter().enumerate() {
            for (ix2, two) in TWO_PATTERN.iter().enumerate() {
                if one[i] == two[i] && !conflicts_one_two.iter().any(|c| c.0 == ix1 && c.1 == ix2 && c.2 == i ) {
                    panic!("Conflict (One at {}, Two at {}) at pos {}", ix1, ix2, i);
                }
            }
        }

        for (ix1, two) in TWO_PATTERN.iter().enumerate() {
            for (ix2, three) in THREE_PATTERN.iter().enumerate() {
                if two[i] == three[i] && !conflicts_two_three.iter().any(|c| c.0 == ix1 && c.1 == ix2 && c.2 == i ) {
                    panic!("Conflict (Two at {}, Three at {}) at pos {}", ix1, ix2, i);
                }
            }
        }

        for (ix1, one) in ONE_PATTERN.iter().enumerate() {
            if one[i] == FOUR_PATTERN[0][i] && !conflicts_one_four.iter().any(|c| c.0 == ix1 && c.1 == 0 && c.2 == i ) {
                panic!("Conflict (One at {}, Four) at pos {}", ix1, i);
            }
        }
        for (ix1, two) in TWO_PATTERN.iter().enumerate() {
            if two[i] == FOUR_PATTERN[0][i] && !conflicts_two_four.iter().any(|c| c.0 == ix1 && c.1 == 0 && c.2 == i ) {
                panic!("Conflict (Two at {}, Four) at pos {}", ix1, i);
            }
        }
        for (ix1, three) in THREE_PATTERN.iter().enumerate() {
            if three[i] == FOUR_PATTERN[0][i] && !conflicts_three_four.iter().any(|c| c.0 == ix1 && c.1 == 0 && c.2 == i ) {
                panic!("Conflict (Three at {}, Four) at pos {}", ix1, i);
            }
        }
    }

    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    struct CandSum(usize, u8);

    impl std::cmp::PartialOrd for CandSum {
        fn partial_cmp(&self, other : &Self) -> Option<std::cmp::Ordering> {
            Some(self.0.cmp(&other.0).then(self.1.cmp(&other.1)))
        }
    }
    impl std::cmp::Ord for CandSum {
        fn cmp(&self, other : &Self) -> std::cmp::Ordering {
            self.0.cmp(&other.0).then(self.1.cmp(&other.1))
        }
    }

    /* Fold = 1 simply use the sum value as the index. Fold=4 use
    (sum_value-3*16, sum_value-2*16, sum_value-1*16, sum_value) as the indices.
    Fold=2 use the (sum_value-N*16, sum_value-M*16) as the indices. Fold=3
    use the (sum_value-N*16, sum_value-M*16, sum_value-P*16) indices. */
    let mut pos_map : BTreeMap<CandSum, Vec<Fold>> = BTreeMap::new();
    for pos in 0..16 {
        for sum in 1..=(FOUR_PATTERN[0].iter().max().copied().unwrap()) {
            for (i, pat) in ONE_PATTERN.iter().enumerate() {
                if sum == pat[pos] {
                    pos_map.entry(CandSum(pos, sum)).or_insert(Vec::new()).push(get_fold(1, i, pos));
                }
            }
            for (i, pat) in TWO_PATTERN.iter().enumerate() {
                if sum == pat[pos] {
                    pos_map.entry(CandSum(pos, sum)).or_insert(Vec::new()).push(get_fold(2, i, pos));
                }
            }
            for (i, pat) in THREE_PATTERN.iter().enumerate() {
                if sum == pat[pos] {
                    pos_map.entry(CandSum(pos, sum)).or_insert(Vec::new()).push(get_fold(3, i, pos));
                }
            }
            if sum == FOUR_PATTERN[0][pos] {
                pos_map.entry(CandSum(pos, sum)).or_insert(Vec::new()).push(get_fold(4, 0, pos));
            }
        }
    }

    let mut mins = BTreeMap::new();
    let mut maxs = BTreeMap::new();
    let mut lens = BTreeMap::new();

    for (s, p) in pos_map {
        println!("{:?}: {:?}", s, p);
        let mut min = mins.entry(s.0).or_insert(s.1);
        let mut max = maxs.entry(s.0).or_insert(s.1);
        let mut len = lens.entry(s.0).or_insert(0);
        if s.1 < *min {
            *min = s.1;
        }
        if s.1 > *max {
            *max = s.1;
        }
        *len += 1;
    }

    for ix in 0..16 {
        println!("{} : (min = {}; max = {}; len = {}", ix, mins[&ix], maxs[&ix], lens[&ix]);
    }

    // println!("{:?}", FOLD_LUT_0);
}

// Those LUTs contain the minimum possible size for the sum at each entry*/
const FOLD_LUT_0 : [Fold; 101] = build_fold_lut::<101>(0, 1, 100);
const FOLD_LUT_1 : [Fold; 105] = build_fold_lut::<105>(1, 2, 104);
const FOLD_LUT_2 : [Fold; 109] = build_fold_lut::<109>(2, 3, 108);
const FOLD_LUT_3 : [Fold; 113] = build_fold_lut::<113>(3, 4, 112);
const FOLD_LUT_4 : [Fold; 117] = build_fold_lut::<117>(4, 5, 116);
const FOLD_LUT_5 : [Fold; 121] = build_fold_lut::<121>(5, 6, 120);
const FOLD_LUT_6 : [Fold; 125] = build_fold_lut::<125>(6, 7, 124);
const FOLD_LUT_7 : [Fold; 129] = build_fold_lut::<129>(7, 8, 128);
const FOLD_LUT_8 : [Fold; 133] = build_fold_lut::<133>(8, 9, 132);
const FOLD_LUT_9 : [Fold; 137] = build_fold_lut::<137>(9, 10, 136);
const FOLD_LUT_10 : [Fold; 141] = build_fold_lut::<141>(10, 11, 140);
const FOLD_LUT_11 : [Fold; 145] = build_fold_lut::<145>(11, 12, 144);
const FOLD_LUT_12 : [Fold; 149] = build_fold_lut::<149>(12, 13, 148);
const FOLD_LUT_13 : [Fold; 153] = build_fold_lut::<153>(13, 14, 152);
const FOLD_LUT_14 : [Fold; 157] = build_fold_lut::<157>(14, 15, 156);
const FOLD_LUT_15 : [Fold; 161] = build_fold_lut::<161>(15, 16, 160);

// All those LUTs contain the maximum possible size
// (so they might be packed into an array with the same
// homogeneous element type). It is best to leave the upper
// limit at u8::MAX (even if not used in practice) or else
// a wrong input might lead to a buffer overflow (instead),
// we get a wrong result.
const FOLD_LUTS : [[Fold; 256]; 16] = [
    build_fold_lut::<256>(0, 1, 100),
    build_fold_lut::<256>(1, 2, 104),
    build_fold_lut::<256>(2, 3, 108),
    build_fold_lut::<256>(3, 4, 112),
    build_fold_lut::<256>(4, 5, 116),
    build_fold_lut::<256>(5, 6, 120),
    build_fold_lut::<256>(6, 7, 124),
    build_fold_lut::<256>(7, 8, 128),
    build_fold_lut::<256>(8, 9, 132),
    build_fold_lut::<256>(9, 10, 136),
    build_fold_lut::<256>(10, 11, 140),
    build_fold_lut::<256>(11, 12, 144),
    build_fold_lut::<256>(12, 13, 148),
    build_fold_lut::<256>(13, 14, 152),
    build_fold_lut::<256>(14, 15, 156),
    build_fold_lut::<256>(15, 16, 160)
];

const fn next_fold<const M : usize>(
    pos : usize,
    sum : usize,
    i : usize,
    pattern : [[u8; 16]; M],
    prev : Fold,
    n_patts : usize
) -> Option<Fold> {
    if sum == pattern[i][pos] as usize {
        match prev {
            Fold::Undecidable => {
                Some(get_fold(n_patts, i, pos))
            },
            _ => {
                Some(Fold::Undecidable)
            }
        }
    } else {
        None
    }
}

const fn build_fold_lut<const N : usize>(pos : usize, min_sum : usize, max_sum : usize) -> [Fold; N] {
    let mut folds : [Fold; N] = [Fold::Undecidable; N];
    folds[0] = Fold::Empty;
    let mut sum = min_sum;
    while sum <= max_sum {
        let mut i = 0;
        while i < ONE_PATTERN.len() {
            if let Some(fold) = next_fold(pos, sum, i, ONE_PATTERN, folds[sum], 1) {
                folds[sum] = fold;
            }
            i += 1;
        }
        i = 0;
        while i < TWO_PATTERN.len() {
            if let Some(fold) = next_fold(pos, sum, i, TWO_PATTERN, folds[sum], 2) {
                folds[sum] = fold;
            }
            i += 1;
        }
        i = 0;
        while i < THREE_PATTERN.len() {
            if let Some(fold) = next_fold(pos, sum, i, THREE_PATTERN, folds[sum], 3) {
                folds[sum] = fold;
            }
            i += 1;
        }
        i = 0;
        while i < FOUR_PATTERN.len() {
            if let Some(fold) = next_fold(pos, sum, i, FOUR_PATTERN, folds[sum], 4) {
                folds[sum] = fold;
            }
            i += 1;
        }
        sum += 1;
    }
    folds
}

/*CandSum(0, 1): [One(0)]
CandSum(0, 17): [One(16)]
CandSum(0, 18): [Two(0, 16)]
CandSum(0, 33): [One(32)]
CandSum(0, 34): [Two(0, 32)]
CandSum(0, 49): [One(48)]
CandSum(0, 50): [Two(0, 48), Two(16, 32)]
CandSum(0, 51): [Three(0, 16, 32)]
CandSum(0, 66): [Two(16, 48)]
CandSum(0, 67): [Three(0, 16, 48)]
CandSum(0, 82): [Two(32, 48)]
CandSum(0, 83): [Three(0, 32, 48)]
CandSum(0, 99): [Three(16, 32, 48)]*/

// Folds an image of dimension h x 64 into an image of dimension h x 16
fn fold_image<S, T>(w : &Image<u8, S>, dst : &mut Image<u8, T>)
where
    S : Storage<u8>,
    T : StorageMut<u8>
{
    assert!(w.width() % 16 == 0);
    assert!(dst.width() == 16);
    assert!(w.height() == dst.height());
    let fold_sz = (w.height(), 16);
    dst.copy_from(&w.window((0, 0), fold_sz).unwrap());
    for i in 1..(w.width() / 16) {
        dst.add_assign(&w.window((0, 16*i), fold_sz).unwrap());
    }
}

#[derive(Debug, Clone, Default)]
struct Section {
    fst : Vec<usize>,
    snd : Vec<usize>,
    thd : Vec<usize>,
    fth : Vec<usize>,
}

impl Section {

    fn clear(&mut self) {
        self.fst.clear();
        self.snd.clear();
        self.thd.clear();
        self.fth.clear();
    }

}

#[derive(Debug, Clone, Default)]
pub struct RasterPoints {

    // All row indices (same dimensionality as cols).
    pub rows : Vec<usize>,

    // Each group is an individual line (with connectivity of points within lines unspecified).
    pub groups : Vec<Range<usize>>,

    // All column indices
    pub cols : Vec<usize>,

    sec : Section

}

impl RasterPoints {

    pub fn code(&self) -> RunLengthCode {
        assert!(self.rows.len() == self.groups.len());
        let mut rles = Vec::new();
        let mut rle_rows = Vec::new();

        for r in 0..self.rows.len() {
            let n_before = rles.len();
            let row = self.rows[r];
            let group = &self.groups[r];
            let mut curr_rle = Some(RunLength { start : (row, self.cols[group.start]), length : 1 });
            for i in (group.start+1)..group.end {
                if let Some(ref mut rle) = curr_rle {
                    if self.cols[i].abs_diff(self.cols[i-1]) == 1 {
                        rle.length += 1;
                    } else {
                        rles.push(rle.clone());
                        curr_rle = None;
                    }
                } else {
                    curr_rle = Some(RunLength { start : (row, self.cols[i]), length : 1 });
                }
            }
            if let Some(rle) = curr_rle {
                rles.push(rle);
            }
            let n_after = rles.len();
            if n_after > n_before {
                rle_rows.push(Range { start : n_before, end : n_after});
            }
        }
        RunLengthCode {
            rles,
            rows : rle_rows
        }
    }

}

// cargo test --lib -- fold_encoding --nocapture
#[test]
fn fold_encoding() {

    use crate::draw::*;

    // let mut img = ImageBuf::<u8>::new_constant(64, 64, 0);
    // img.draw(Mark::Dot((32, 32), 8), 1);

    let mut img = ImageBuf::<u8>::new_constant(128, 128, 0);
    img.draw(Mark::Dot((64, 64), 32), 1);

    let mut enc = FoldEncoder::new(img.height(), img.width());
    let rles = enc.encode(&img);

    for i in 0..enc.pts.rows.len() {
        let gr = enc.pts.groups[i].clone();
        let r = enc.pts.rows[i];
        for c in &enc.pts.cols[gr] {
            img[(r, *c)] = 255;
        }
    }

    /*for rle in &rles.rles {
        for pt in rle.points() {
            img[pt] = 255;
        }
    }*/

    img.show();
}

pub struct FoldEncoder {
    fold : Folded,
    indices : ImageBuf<u8>,
    masked_indices : ImageBuf<u8>,
    pts : RasterPoints
}

impl FoldEncoder {

    pub fn new(height : usize, width : usize) -> Self {
        let mut indices = ImageBuf::<u8>::new_constant(height, width, 0);
        let ixs = (1..=64).collect::<Vec<_>>();
        for r in 0..height {
            let mut row = unsafe { indices.row_mut_unchecked(r) };
            for i in 0..(width / 64) {
                row[(i*64)..((i+1)*64)].copy_from_slice(&ixs[..]);
            }
        }
        let masked_indices = indices.clone();
        Self {
            fold : Folded::new(height, width),
            pts : RasterPoints::default(),
            indices,
            masked_indices
        }
    }

    pub fn encode<S>(&mut self, w : &Image<u8, S>) -> RunLengthCode
    where
        S : Storage<u8>
    {
        w.mul_to(&self.indices, &mut self.masked_indices);

        // #[cfg(feature="ipp")]
        // self.pts.update_folded2(&self.masked_indices, &mut self.fold);

        // #[cfg(not(feature="ipp"))]
        self.pts.update_folded(&self.masked_indices, &mut self.fold);

        self.pts.code()
    }

}

#[derive(Debug, Clone)]
struct Folded {

    // Contains sum of horizontal image fold (always with h x 16 size)
    sum : ImageBuf<u8>,

    // Contains concatenated folded regions, with dimension h x (n_folds*16) = h x (w / 4)
    dst : ImageBuf<u8>,

    #[cfg(feature="ipp")]
    sw_rows : IppSumWindow,

    #[cfg(feature="ipp")]
    sw_cols : IppSumWindow,

    sum_rows : ImageBuf<u8>,

    sum_cols : ImageBuf<u8>,

    one_rows : ImageBuf<u8>,

    one_cols : ImageBuf<u8>,

}

impl Folded {

    fn new(height : usize, width : usize) -> Self {
        let sum = ImageBuf::<u8>::new_constant(height, 16, 0);
        let dst = ImageBuf::<u8>::new_constant(height, width / 4, 0);
        let sum_rows = ImageBuf::<u8>::new_constant(height, 1, 0);
        let sum_cols = ImageBuf::<u8>::new_constant(1, 16, 0);
        let one_cols = ImageBuf::<u8>::new_constant(height, 1, 1);
        let one_rows = ImageBuf::<u8>::new_constant(1, 16, 1);

        #[cfg(feature="ipp")]
        {
            let src_sz = (height, 16);
            let sw_rows = crate::local::IppSumWindow::new(
                16,
                src_sz,
                1,
                (height, 1)
            );
            let sw_cols = crate::local::IppSumWindow::new(
                16,
                src_sz,
                16,
                (1, 16)
            );
            Self { sum, dst, sum_rows, sum_cols, sw_cols, sw_rows, one_cols, one_rows }
        }

        #[cfg(not(feature="ipp"))]
        Self { sum, dst, sum_rows, sum_cols }
    }

    fn add_to_sum(&mut self) {
        let height = self.sum.height();
        let fst_strip = self.dst.window((0, 0), (height, 16)).unwrap();
        let rem_sz = (height, self.dst.width()-16);
        let rem_strips = self.dst.window((0, 16), rem_sz).unwrap();
        self.sum.copy_from(&fst_strip);
        for strip in rem_strips.windows((height, 16)) {
            self.sum.add_assign(&strip);
        }
    }

    #[cfg(feature="ipp")]
    fn sum_profile(&mut self) {
        // self.sw_rows.calculate(&self.sum, &mut self.sum_rows);
        // self.sw_cols.calculate(&self.sum, &mut self.sum_cols);
        let h = self.sum.height();
        for c in 0..16usize {
            self.sum_cols[(0, c)] = self.sum.window((0usize, c), (h, 1)).unwrap().sum::<f32>(1) as u8;
        }
        for r in 0..h {
            self.sum_rows[(r,0)] = self.sum.window((r, 0), (1usize, 16usize)).unwrap().sum::<f32>(1) as u8;
        }
    }

}

const BASE_OFFSET : [usize; 4] = [0, 16, 32, 48];

pub struct FoldRowIter {
    col_evals : Vec<usize>,
    chunks_per_row : usize,
    col_offsets : Vec<usize>
}

impl FoldRowIter {

    pub fn new(width : usize) -> Self {
        let col_evals : Vec<_> = (0..16).collect();
        let chunks_per_row = width / 64;
        let col_offsets : Vec<_> = (0..chunks_per_row).map(|i| i * 64 ).collect();
        Self {
            col_evals,
            chunks_per_row,
            col_offsets
        }
    }

}

impl RasterPoints {

    fn clear(&mut self) {
        self.rows.clear();
        self.groups.clear();
        self.cols.clear();
    }

    fn verify<S>(&self, w : &Image<u8, S>, folded : &Folded)
    where
        S: Storage<u8>
    {
        assert!(w.width() % 64 == 0);
        assert!(w.height() == folded.dst.height());
        assert!(w.width() / 4 == folded.dst.width());
    }

    // This not only sums the folds, but removes from analysis
    // empty rows and columns.
    #[cfg(feature="ipp")]
    fn update_folded2<S>(&mut self, w : &Image<u8, S>, folded : &mut Folded)
    where
        S: Storage<u8>
    {
        self.verify(w, &*folded);
        self.clear();

        // No point in folding an image with the minimum size.
        if w.width() == 64 {
            self.update_flat(w, folded);
            return;
        }

        let mut timer = crate::util::Timer::start();

        self.fold_all(w, folded);
        timer.time("fold");

        // Add up concatenated regions.
        folded.add_to_sum();
        timer.time("sum");

        folded.sum_profile();
        timer.time("sum profile");

        let mut strip_rows = w.row_chunks(64);
        let mut dst_rows = folded.dst.row_chunks(16);
        let FoldRowIter { mut col_evals, chunks_per_row, col_offsets} = FoldRowIter::new(w.width());

        let valid_cols : Vec<_> = folded.sum_cols.pixels(1)
            .enumerate()
            .filter_map(|(pos, px)| if *px > 0 { Some(pos) } else { None })
            .collect();
        // assert!(valid_cols.len() <= 16);
        // println!("{:?}", valid_cols);
        // let valid_cols = col_evals.clone();
        for r in 0..w.height() {

            // println!("{}", folded.sum_rows[(r, 0)]);
            if folded.sum_rows[(r, 0)] == 0 {
                for _ in 0..chunks_per_row {
                    let _ = strip_rows.next().unwrap();
                    let _ = dst_rows.next().unwrap();
                }
                continue;
            }

            col_evals.clear();
            let sum_row = unsafe { folded.sum.row_unchecked(r) };
            for i in &valid_cols {
                if sum_row[*i] > 0 {
                    col_evals.push(*i);
                }
            }
            let n_before = self.cols.len();
            for i in 0..chunks_per_row {
                self.update_row_segment(strip_rows.next().unwrap(), dst_rows.next().unwrap(), &col_evals[..], /*r*/ col_offsets[i]);
            }
            self.push_row(r, n_before);
        }

        timer.time("collect");
    }

    /* Updates by restricting the evaluated pixels to those that are nonzero
    at a final image. */
    fn update_folded<S>(&mut self, w : &Image<u8, S>, folded : &mut Folded)
    where
        S: Storage<u8>
    {
        self.verify(w, &*folded);
        self.clear();

        // No point in folding an image with the minimum size.
        if w.width() == 64 {
            self.update_flat(w, folded);
            return;
        }

        let mut timer = crate::util::Timer::start();

        self.fold_all(w, folded);
        timer.time("fold");

        // Add up concatenated regions.
        folded.add_to_sum();
        timer.time("sum");

        let mut strip_rows = w.row_chunks(64);
        let mut dst_rows = folded.dst.row_chunks(16);
        let FoldRowIter { mut col_evals, chunks_per_row, col_offsets} = FoldRowIter::new(w.width());
        for r in 0..w.height() {
            col_evals.clear();
            let sum_row = unsafe { folded.sum.row_unchecked(r) };
            for (i, px) in sum_row.iter().enumerate() {
                if *px > 0 {
                    col_evals.push(i);
                }
            }
            let n_before = self.cols.len();
            for i in 0..chunks_per_row {
                let strip_slice = strip_rows.next().unwrap();
                let dst_slice = dst_rows.next().unwrap();
                self.update_row_segment(strip_slice, dst_slice, &col_evals[..], col_offsets[i]);
            }
            self.push_row(r, n_before);
        }

        timer.time("collect");
    }

    fn push_row(&mut self, r : usize, n_before : usize) {
        let n_after = self.cols.len();
        if n_after > n_before {
            self.rows.push(r);
            self.groups.push(Range { start : n_before, end : n_after });
        }
    }

    fn fold_all<S>(&mut self, w : &Image<u8, S>, folded : &mut Folded)
    where
        S: Storage<u8>
    {
        let strips = w.windows((w.height(), 64));
        let dsts = folded.dst.windows_mut((w.height(), 16));
        for (strip, mut dst) in strips.zip(dsts) {
            fold_image(&strip, &mut dst);
        }
    }

    /* Updates by examining all pixels between (0..16) at each image fold. */
    fn update_flat<S>(&mut self, w : &Image<u8, S>, folded : &mut Folded)
    where
        S: Storage<u8>,
    {
        self.verify(w, &*folded);
        self.clear();
        self.fold_all(w, folded);

        // It is best to iterate over rows then over strips for each row, since
        // in this case the pixels at the same row will be contiguous.

        let mut strip_rows = w.row_chunks(64);
        let mut dst_rows = folded.dst.row_chunks(16);
        let FoldRowIter { col_evals, chunks_per_row, col_offsets} = FoldRowIter::new(w.width());
        for r in 0..w.height() {
            let n_before = self.cols.len();
            for i in 0..chunks_per_row {
                self.update_row_segment(strip_rows.next().unwrap(), dst_rows.next().unwrap(), &col_evals[..], col_offsets[i]);
            }
            self.push_row(r, n_before);
        }
    }

    // W is assumed to be a bit binary image of dimensions r x 64, while
    // dst is assumed to be a destination image with any content of
    // dimensions r x 16.
    fn update_row_segment(
        &mut self,
        orig_row : &[u8],
        fold_row : &[u8],
        col_evals : &[usize],
        offset : usize
    ) {
        self.sec.clear();
        let n_before = self.cols.len();
        for c in col_evals {
            match unsafe { FOLD_LUTS.get_unchecked(*c).get_unchecked(fold_row[*c] as usize) } {
                Fold::Empty => { },
                Fold::Undecidable => {
                    for offset in &BASE_OFFSET[..] {
                        let oc = *offset + *c;
                        if orig_row[oc] > 0 {
                            self.cols.push(oc);
                        }
                    }
                },
                Fold::One(pt, ord) => {
                    match ord {
                        OneFoldOrder::Fst => self.sec.fst.push(*pt),
                        OneFoldOrder::Snd => self.sec.snd.push(*pt),
                        OneFoldOrder::Thd => self.sec.thd.push(*pt),
                        OneFoldOrder::Fth => self.sec.fth.push(*pt),
                    }
                },
                Fold::Two(pts, ord) => {
                    match ord {
                        TwoFoldOrder::FstSnd => { self.sec.fst.push(pts[0]); self.sec.snd.push(pts[1]); },
                        TwoFoldOrder::FstThd => { self.sec.fst.push(pts[0]); self.sec.thd.push(pts[1]); },
                        TwoFoldOrder::FstFth => { self.sec.fst.push(pts[0]); self.sec.fth.push(pts[1]); },
                        TwoFoldOrder::SndThd => { self.sec.snd.push(pts[0]); self.sec.thd.push(pts[1]); },
                        TwoFoldOrder::SndFth => { self.sec.snd.push(pts[0]); self.sec.fth.push(pts[1]); },
                        TwoFoldOrder::ThdFth => { self.sec.thd.push(pts[0]); self.sec.fth.push(pts[1]); },
                    }
                },
                Fold::Three(pts, ord) => {
                    match ord {
                        ThreeFoldOrder::FstSndThd => {
                            self.sec.fst.push(pts[0]);
                            self.sec.snd.push(pts[1]);
                            self.sec.thd.push(pts[2]);
                        },
                        ThreeFoldOrder::FstSndFth => {
                            self.sec.fst.push(pts[0]);
                            self.sec.snd.push(pts[1]);
                            self.sec.fth.push(pts[2]);
                        },
                        ThreeFoldOrder::FstThdFth => {
                            self.sec.fst.push(pts[0]);
                            self.sec.thd.push(pts[1]);
                            self.sec.fth.push(pts[2]);
                        },
                        ThreeFoldOrder::SndThdFth => {
                            self.sec.snd.push(pts[0]);
                            self.sec.thd.push(pts[1]);
                            self.sec.fth.push(pts[2]);
                        }
                    }
                },
                Fold::Four(pts) => {
                    self.sec.fst.push(pts[0]);
                    self.sec.snd.push(pts[1]);
                    self.sec.thd.push(pts[2]);
                    self.sec.fth.push(pts[3]);
                }
            }
        }
        let mut cols = std::mem::take(&mut self.cols);
        cols.extend_from_slice(&self.sec.fst[..]);
        cols.extend_from_slice(&self.sec.snd[..]);
        cols.extend_from_slice(&self.sec.thd[..]);
        cols.extend_from_slice(&self.sec.fth[..]);
        self.cols = cols;

        let n_after = self.cols.len();
        if n_after > n_before {
            let range = Range { start : n_before, end : n_after };
            offset_cols(&mut self.cols[range.clone()], offset);
        }
    }

}

fn offset_vectorized(cols : &mut [usize], col_offset : usize) {
    let mut ptr : *mut wide::u64x4 = cols.as_mut_ptr() as *mut wide::u64x4;
    let mut cols_v : &mut [wide::u64x4] = unsafe { std::slice::from_raw_parts_mut(ptr, cols.len() / 4) };
    let col_offset_v = wide::u64x4::splat(col_offset as u64);
    cols_v.iter_mut().for_each(|c| *c += col_offset_v );
}

fn offset_scalar(cols : &mut [usize], col_offset : usize) {
    cols.iter_mut().for_each(|c| *c += col_offset );
}

fn offset_cols(cols : &mut [usize], col_offset : usize) {
    if col_offset > 0 {
        let sz = crate::util::closest_divisor(cols.len(), 4);
        match sz.cmp(&4) {
            Ordering::Less => {
                offset_scalar(cols, col_offset);
            },
            Ordering::Equal => {
                offset_vectorized(cols, col_offset);
            },
            Ordering::Greater => {
                let (cols1, cols2) = cols.split_at_mut(sz);
                offset_vectorized(cols1, col_offset);
                offset_scalar(cols2, col_offset);
            }
        }
    }
}

/*fn offset_cols(cols : &mut [usize], col_offset : usize) {
    if col_offset > 0 {
        cols.iter_mut().for_each(|c| *c += col_offset );
    }
}*/

// Contains a sparse image code. This implements Encoding
// by dispatching to the encoding implementation of each
// variant, so users can use the Encoding methods (points, etc)
// Irrespective of what encoding is being used.
pub enum ImageCode {
    Point(PointCode),
    RunLength(RunLengthCode),
    Chain(ChainCode),
    Pattern(PatternCode)
}

/* Represents a binary image by a set of foreground pixel coordinates. Useful when
objects are expected to be speckle-like. For dense objects, consider using RunLengthEncoding
instead, since PointEncoding is an encoding that is potentially memory-heavy. But if objects are
speckle-like, then the RunLength encoding would mostly carry a start and length=1 that wouldn't
be very informative, so PointEncoding would be a best option. */
#[derive(Default)]
pub struct PointCode {
    pub pts :  Vec<(usize, usize)>,
    pub rows : Vec<Range<usize>>
}

impl Encoding for PointCode {

    fn points<'a>(&'a self) -> PointIter<'a> {
        PointIter(Box::new(self.pts.clone().into_iter()))
    }

    fn encode_distinct<S>(img : &Image<u8, S>) -> BTreeMap<u8, Self> {
        unimplemented!()
    }

    fn encode_from<S>(&mut self, img : &Image<u8, S>) {
        unimplemented!()
    }
    
    // Write the labels back to the image.
    fn decode_distinct_to(labels : &BTreeMap<u8, Self>, img : &dyn AsMut<WindowMut<u8>>) {
        unimplemented!()
    }

}

// Assume pts is sorted by rows and by cols within rows.
fn rows_for_sorted_points(pts : &[(usize, usize)], rows : &mut Vec<Range<usize>>) {
    rows.clear();
    let mut n = 0;
    for (_, mut pts) in &pts.iter().group_by(|pt| pt.0 ) {
        let row_len = pts.count();
        rows.push(Range { start : n, end : n+row_len } );
        n += row_len;
    }
}

impl PointCode {

    // Builds a PointEncoding from a set of points (assume neither order or uniqueness
    // of the underlying array).
    pub fn from_points(mut pts : Vec<(usize, usize)>) -> Self {
        pts.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)) );
        pts.dedup();
        let mut rows = Vec::new();
        rows_for_sorted_points(&pts[..], &mut rows);
        Self { pts, rows }
    }

    /// Returns coordinates of dense set of points that are nonzero.
    pub fn encode(win : &Window<'_, u8>) -> Self {
        let mut pts = Vec::new();
        let mut rows : Vec<Range<usize>> = Vec::new();
        pts.clear();
        for i in 0..win.height() {
            let mut found_at_row = 0;
            for j in 0..win.width() {
                if win[(i, j)] != 0 {
                    found_at_row += 1;
                    pts.push((i, j));
                }
            }
            if found_at_row > 0 {
                let last = rows.last().map(|l| l.end ).unwrap_or(0);
                rows.push(Range { start : last, end : last + found_at_row });
            }
        }
        Self { pts, rows }
    }

    pub fn connectivity_graph(&self, neigh : Neighborhood) -> PointGraph {
        let mut uf = UnionFind::new(self.pts.len());
        let mut graph = UnGraph::new_undirected();
        let mut found_pts = Vec::new();
        for (row_ix, row) in self.rows.iter().enumerate() {
            for pt_ix_at_row in 0..row.len() {
                let pt_ix = row.start + pt_ix_at_row;
                let graph_ix = graph.add_node(self.pts[pt_ix]);

                let this_pt_col = self.pts[pt_ix].1;
                let this_pt_row = self.pts[pt_ix].0;

                // Link to element(s) to left
                let mut left_offset = 1;
                if let Some(left_ix) = pt_ix_at_row.checked_sub(1) {
                    if this_pt_col - self.pts[row.clone()][left_ix].1 == 1 {
                        let left_ix = NodeIndex::new(graph_ix.index()-1);
                        graph.add_edge(graph_ix, left_ix, ());
                        uf.union(graph_ix, left_ix);
                    }
                }

                // Link to any element to top as long as it is below a given distance. Worst
                // case scenario, we have points in all rows above the point (but leave
                // early if not).
                let mut top_row_offset = 1;
                if let Some(top_row_ix) = row_ix.checked_sub(1) {

                    let top_row = &self.rows[top_row_ix];

                    // (Leave early if not)
                    if this_pt_row - self.pts[top_row.clone()][0].0 == 1 {

                        // TODO move this match to return a function pointer.
                        let res_pt = match neigh {
                            Neighborhood::Immediate => {
                                self.pts[top_row.clone()].binary_search_by(|pt| {
                                    pt.1.cmp(&this_pt_col)
                                })
                            },
                            Neighborhood::Extended => {
                                self.pts[top_row.clone()].binary_search_by(|pt| {
                                    if this_pt_col.abs_diff(pt.1) <= 1 {
                                        Ordering::Equal
                                    } else {
                                        pt.1.cmp(&this_pt_col)
                                    }
                                })
                            }
                        };
                        if let Ok(found) = res_pt {
                            match neigh {
                                Neighborhood::Immediate => {
                                    let top_ix = NodeIndex::new(top_row.start + found);
                                    graph.add_edge(graph_ix, top_ix, ());
                                    uf.union(graph_ix, top_ix);
                                },
                                Neighborhood::Extended => {
                                    found_pts.clear();
                                    found_pts.push(found);

                                    // Search for neighboring element when the binary search returns middle OR extreme element.
                                    if found > 0 && self.pts[top_row.start + found - 1].1.abs_diff(this_pt_col) <= 1 {
                                        found_pts.push(found - 1);
                                    }
                                    if found < top_row.len()-1 && self.pts[top_row.start + found + 1].1.abs_diff(this_pt_col) <= 1 {
                                        found_pts.push(found + 1);
                                    }

                                    // Search for two next elements when the binary search returns an extreme element
                                    if found > 1 && self.pts[top_row.start + found - 2].1.abs_diff(this_pt_col) <= 1 {
                                        found_pts.push(found - 2);
                                    }
                                    if top_row.len() >= 2 && found < top_row.len()-2 && self.pts[top_row.start + found + 2].1.abs_diff(this_pt_col) <= 1 {
                                        found_pts.push(found + 2);
                                    }

                                    for pt_ix in &found_pts {
                                        let top_ix = NodeIndex::new(top_row.start + pt_ix);
                                        graph.add_edge(graph_ix, top_ix, ());
                                        uf.union(graph_ix, top_ix);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        PointGraph { uf, graph }
    }

    pub fn proximity_graph(&self, max_dist : usize) -> PointGraph {
        let mut uf = UnionFind::new(self.pts.len());
        let mut graph = UnGraph::new_undirected();
        for (row_ix, row) in self.rows.iter().enumerate() {
            for pt_ix_at_row in 0..row.len() {
                let pt_ix = row.start + pt_ix_at_row;
                let graph_ix = graph.add_node(self.pts[pt_ix]);

                let this_pt_col = self.pts[pt_ix].1;
                let this_pt_row = self.pts[pt_ix].0;

                // Link to element(s) to left
                let mut left_offset = 1;
                while let Some(left_ix) = pt_ix_at_row.checked_sub(left_offset) {
                    if this_pt_col - self.pts[row.clone()][left_ix].1 <= max_dist {
                        // let left_ix = NodeIndex::new(graph_ix.index()-1);
                        let left_ix = NodeIndex::new(graph_ix.index()-left_offset);
                        graph.add_edge(graph_ix, left_ix, ());
                        uf.union(graph_ix, left_ix);
                    } else {
                        break;
                    }
                    left_offset += 1;
                }

                // Link to any element to top as long as it is below a given distance. Worst
                // case scenario, we have points in all rows above the point (but leave
                // early if not).
                let mut top_row_offset = 1;
                while top_row_offset <= max_dist && row_ix.checked_sub(top_row_offset).is_some() {

                    let top_row = &self.rows[row_ix - top_row_offset];

                    // (Leave early if not)
                    if this_pt_row - self.pts[top_row.clone()][0].0 > max_dist {
                        break;
                    }

                    let res_pt = self.pts[top_row.clone()].binary_search_by(|pt| {
                        if this_pt_col.abs_diff(pt.1) <= max_dist {
                            Ordering::Equal
                        } else {
                            pt.1.cmp(&this_pt_col)
                        }
                    });

                    if let Ok(mut start) = res_pt {
                        // Iterate over all points near this one.
                        let mut end = start+1;
                        loop {
                            if start > 0 && self.pts[top_row.start + start - 1].1.abs_diff(this_pt_col) <= max_dist {
                                start -= 1;
                            } else {
                                break;
                            }
                        }
                        loop {
                            if end < top_row.len()-1 && self.pts[top_row.start + end + 1].1.abs_diff(this_pt_col) <= max_dist {
                                end += 1;
                            } else {
                                break;
                            }
                        }
                        for pt_ix in start..end {
                            let top_ix = NodeIndex::new(top_row.start + pt_ix);
                            graph.add_edge(graph_ix, top_ix, ());
                            uf.union(graph_ix, top_ix);
                        }
                    }
                    top_row_offset += 1;
                }
            }
        }
        PointGraph { uf, graph }
    }

}

// Establish edges between points that are close together.
#[derive(Debug, Clone)]
pub struct PointGraph {

    pub graph : UnGraph<(usize, usize), ()>,

    pub uf : UnionFind<NodeIndex<u32>>

}

impl PointGraph {

    pub fn new() -> Self {
        Self {
            graph : UnGraph::new_undirected(),
            uf : UnionFind::new(0)
        }
    }
    
    pub fn enclosing_rects(&self) -> BTreeMap<NodeIndex<u32>, (usize, usize, usize, usize)> {
        let mut rects = BTreeMap::new();
        for (ix, pts) in self.groups() {
            let min_y = pts.iter().min_by(|a, b| a.0.cmp(&b.0) ).copied().unwrap().0;
            let min_x = pts.iter().min_by(|a, b| a.1.cmp(&b.1) ).copied().unwrap().1;
            let max_y = pts.iter().max_by(|a, b| a.0.cmp(&b.0) ).copied().unwrap().0;
            let max_x = pts.iter().max_by(|a, b| a.1.cmp(&b.1) ).copied().unwrap().1;
            rects.insert(ix, (min_y, min_x, (max_y - min_y)+1, (max_x - min_x)+1));
        }
        rects
    }

    pub fn groups(&self) -> BTreeMap<NodeIndex<u32>, Vec<(usize, usize)>> {
        let mut grs = BTreeMap::new();
        for ix in self.graph.node_indices() {
            let parent = self.uf.find(ix);
            grs.entry(parent).or_insert(Vec::new()).push(self.graph[ix]);
        }
        grs
    }

}

type BumpVecOpen<'a> = bumpalo::collections::Vec<'a, OpenChain<'a>>;

type BumpVecDir<'a> = bumpalo::collections::Vec<'a, Direction>;

#[derive(Clone)]
struct OpenChain<'a> {

    start : (usize, usize),

    end : (usize, usize),

    directions : BumpVecDir<'a>

}

fn is_close(a : (usize, usize), b : (usize, usize)) -> bool {
    a.1.abs_diff(b.1) <= 1 && a.0.abs_diff(b.0) <= 1
}

pub fn draw_connected_chain_random(chain : &ChainCode, img : &mut WindowMut<u8>) {
    img.fill(0);
    let codes = chain.split_by_proximity(1);
    for (_, code) in codes {
        let r : f64 = rand::random();
        for ch in code.chains() {
            let color = (10.0 + r * 245.0) as u8;
            for pt in ch.trajectory() {
                img[pt] = color;
            }
        }
    }
}

pub fn draw_chain_random(chain : &ChainCode, img : &mut WindowMut<u8>) {
    img.fill(0);
    for ch in chain.chains() {
        let r : f64 = rand::random();
        let color = (10.0 + r * 245.0) as u8;
        for pt in ch.trajectory() {
            img[pt] = color;
        }
    }
}

/*// cargo test -- chain --nocapture
#[test]
fn chain() {

    use crate::draw::Draw;
    use crate::draw::Mark;
    
    let mut img = Image::new_constant(32, 32, 0);
    
    for i in 8..16 {
        img[(i, 16)] = 255;
    }
    
    println!("Vertical: {:?}", ChainCode::encode(&img.full_window()) );
    
    img.full_window_mut().fill(0);
    for i in 8..16 {
        img[(16, i)] = 255;
    }
    println!("Horizontal: {:?}", ChainCode::encode(&img.full_window()) );
    
    img.full_window_mut().fill(0);
    for i in 8..16 {
        img[(i, i)] = 255;
    }
    println!("Diagonal right: {:?}", ChainCode::encode(&img.full_window()) );
    
    img.full_window_mut().fill(0);
    for i in 8..16 {
        img[(i, 32 - i)] = 255;
    }
    println!("Diagonal left: {:?}", ChainCode::encode(&img.full_window()) );
    
    img.full_window_mut().fill(0);
    img.draw(Mark::Rect((8, 8), (16, 16), 255));
    img.show();
    let enc = ChainCode::encode(&img.full_window());
    println!("Rect: {:?}", enc);
    draw_chain_random(&enc, img.as_mut());
    img.show();
    
    img.full_window_mut().fill(0);
    img.draw(Mark::Circle((16, 16), 8, 255));
    img.show();
    let enc = ChainCode::encode(&img.full_window());
    println!("Circle: {:?}", enc );
    draw_chain_random(&enc, img.as_mut());
    img.show();
    
}*/

#[derive(Debug)]
pub struct ChainEncoder {

    bump : bumpalo::Bump,
    
    size : (usize, usize)
    
}

impl Clone for ChainEncoder {

    fn clone(&self) -> Self {
        ChainEncoder::new(self.size.0, self.size.1)
    }

}

impl ChainEncoder {

    pub fn new(height : usize, width : usize) -> Self {
        let arena_sz = mem::size_of::<Direction>() * height * width;
        let bump = bumpalo::Bump::with_capacity(arena_sz);
        Self { bump, size : (height, width) }
    }
    
    pub fn encode<S>(&self, bin_img : &Image<u8, S>, mut old : Option<ChainCode>) -> ChainCode
    where
        S : Storage<u8>
    {
        assert!(bin_img.height() == self.size.0 && bin_img.width() == self.size.1);
        let mut old = old.unwrap_or(ChainCode::new_empty());
        old.clear();
        let ChainCode { mut starts, mut ends, mut ranges, mut directions } = old;
        // let mut starts = Vec::new();
        // let mut ends = Vec::new();
        // let mut ranges : Vec<Range<usize>> = Vec::new();
        // let mut directions = Vec::new();
        let mut open_chains = BumpVecOpen::new_in(&self.bump);
        let mut row_chains = BumpVecOpen::new_in(&self.bump);

        // Holds indices of open_chains that matched the current row.
        let mut open_matched = bumpalo::collections::Vec::new_in(&self.bump);

        for (row_ix, row) in bin_img.rows().enumerate() {

            let mut past = row[0];
            if row[0] != 0 {
                // Inaugurate a chain when first pixel is nonzero.
                row_chains.push(OpenChain {
                    start : (row_ix, 0),
                    end : (row_ix, 0),
                    directions : BumpVecDir::new_in(&self.bump)
                });
            }

            for col_ix in 1..row.len() {
                match (past != 0, row[col_ix] != 0) {
                    (true, true) => {
                        // If left and this matched, just grow the last chain.
                        let last = row_chains.last_mut().unwrap();
                        last.end.1 += 1;
                        last.directions.push(Direction::East);
                    },
                    (false, true) => {
                        // If transition to new matched pixel, innaugurate a new chain.
                        row_chains.push(OpenChain { 
                            start : (row_ix, col_ix), 
                            end : (row_ix, col_ix), 
                            directions : BumpVecDir::new_in(&self.bump)
                        });
                    },
                    (true, false) => {
                        // Positive->negative transitions can be ignored, since
                        // chains need to be kept open.
                    },
                    (false, false) => {
                        // Negative->negative transitions also can be ignored.
                    }
                }
                past = row[col_ix];
            }

            // row_chains now contains a set of horizontal chains accumulated over
            // the current row; open_chains all the previous chains. Combine the two.
            open_matched.clear();
            let n_before = open_chains.len();
            for mut row_chain in row_chains.drain(..) {
                let mut matched = None;
                for i in 0..n_before {

                    // Guarantees matched elements are not re-used.
                    if open_matched.binary_search(&i).is_ok() {
                        continue;
                    }

                    if is_close(open_chains[i].end, row_chain.start) {

                        // Extend by appending element (and going left-to-right if required).
                        let end = open_chains[i].end;
                        open_chains[i].directions.push(Direction::compare_below(end, row_chain.start));
                        open_chains[i].directions.extend(row_chain.directions.drain(..));
                        open_chains[i].end = row_chain.end;

                        matched = Some(i);
                        break;

                    // This second condition will only match for row chains that have at least
                    // two elements, since single-element ones would have already matched at the start
                    // in the first condition.
                    } else if is_close(open_chains[i].end, row_chain.end) {

                        // Extend now by going right-to-left through this row chain (Set all row_chain directions to west).
                        row_chain.directions.iter_mut().for_each(|d| d.reverse() );
                        let end = open_chains[i].end;
                        open_chains[i].directions.push(Direction::compare_below(end, row_chain.end));
                        open_chains[i].directions.extend(row_chain.directions.drain(..));
                        open_chains[i].end = row_chain.start;

                        matched = Some(i);
                        break;

                    // The two conditions below will trigger only for chains that have
                    // two or more elements (since connecting, single-element chains would
                    // have already matched the two previous conditions). The conditions will
                    // also match only when the previous open_chain[i] started and ended at the same
                    // row above (since any chains started at a row before will return false for is_close).
                    // The conditions below produce "snake-like" patterns for dense objects.
                    } else if is_close(open_chains[i].start, row_chain.start) {

                        // Revert chain at previous row (as if it started at the end) so it
                        // can be a part of the next chain.

                        open_chains[i].directions.iter_mut().for_each(|d| d.reverse() );

                        // The starts will only be equal to next start when the chain above is present on a single row.

                        // Pushing to the end of row_chains and reversing has the same effect as pushing
                        // to the start of open_chains[i], but there is no risk of re-allocating the vector.
                        let start = open_chains[i].start;
                        open_chains[i].directions.push(Direction::compare_below(start, row_chain.start));

                        // "snake turns right"
                        open_chains[i].directions.extend(row_chain.directions.drain(..));
                        open_chains[i].start = open_chains[i].end;
                        open_chains[i].end = row_chain.end;

                        matched = Some(i);
                        break;

                    } else if is_close(open_chains[i].start, row_chain.end) {

                        // The start will only be equal to next end when the chain above is present on a single row.
                        open_chains[i].directions.iter_mut().for_each(|d| d.reverse() );
                        row_chain.directions.iter_mut().for_each(|d| d.reverse() );
                        let start = open_chains[i].start;
                        open_chains[i].directions.push(Direction::compare_below(start, row_chain.end));

                        // "snake turns left"
                        open_chains[i].directions.extend(row_chain.directions.drain(..));
                        open_chains[i].start = open_chains[i].end;
                        open_chains[i].end = row_chain.start;

                        matched = Some(i);
                        break;
                    }
                }
                if let Some(matched_ix) = matched {
                    open_matched.push(matched_ix);
                } else {
                    // To be considered for next row iteration. This item should
                    // not be considered further at this iteration.
                    open_chains.push(row_chain);
                }
            }

            // Close unmatched chains, by iterating up to the size of open_chains before row
            // was evaluated.
            for i in (0..n_before).rev() {
                if open_matched.binary_search(&i).is_err() {
                    let closed = open_chains.swap_remove(i);
                    starts.push(closed.start);
                    ends.push(closed.end);

                    // Elements with a single element will yield a open range (start = end)
                    let start = ranges.last().map(|r| r.end ).unwrap_or(0);
                    ranges.push(Range { start, end : start + closed.directions.len() });
                    directions.extend(closed.directions);
                }
            }
        }

        ChainCode {
            starts,
            directions,
            ranges,
            ends
        }
    }
}

impl ChainCode {

    pub fn new_empty() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.starts.clear();
        self.ends.clear();
        self.ranges.clear();
        self.directions.clear();
    }

    pub fn retain_larger(&mut self, min_sz : usize) {
        for i in (0..self.starts.len()).rev() {
            if self.ranges[i].end - self.ranges[i].start < min_sz {
                // Note the directions Vector isn't actually cleaned
                // so as to not invalidate the indices of the ranges.
                // But there is no way to access the removed elements
                // from the public API.
                self.ranges.remove(i);
                self.starts.remove(i);
                self.ends.remove(i);
            }
        }
    }

    pub fn count_points(&self) -> usize {
        self.chains().fold(0, |count, chain| count + chain.len() )
    }

    pub fn len(&self) -> usize {
        self.starts.len()
    }

    pub fn get(&self, ix : usize) -> Option<Chain> {
        let start = self.starts.get(ix)?;
        let end = self.ends.get(ix)?;
        let traj = &self.directions[self.ranges.get(ix)?.clone()];
        Some(Chain { start : *start, end : *end, traj })
    }

    unsafe fn get_unchecked(&self, ix : usize) -> Chain {
        let start = self.starts.get_unchecked(ix);
        let end = self.ends.get_unchecked(ix);
        let traj = &self.directions[self.ranges.get_unchecked(ix).clone()];
        Chain { start : *start, end : *end, traj }
    }

    pub fn iter(&self) -> ChainIter {
        ChainIter {
            enc : self,
            curr : 0,
            end : self.starts.len()
        }
    }

    pub fn iter_segment(&self, range : Range<usize>) -> ChainIter {
        assert!(range.end >= range.start);
        assert!(range.start <= self.starts.len());
        assert!(range.end <= self.starts.len());
        ChainIter {
            enc : self,
            curr : range.start,
            end : range.end
        }
    }

    pub fn chains<'a>(&'a self) -> ChainIter<'a> {
        self.iter()
    }

    fn chain_union_to_btree(&self, unions : UnionFind<usize>) -> BTreeMap<usize, ChainCode> {
        let mut dst : BTreeMap<usize, ChainCode> = BTreeMap::new();
        for i in 0..self.len() {
            let ch = self.get(i).unwrap();
            let repr = unions.find(i);
            if let Some(enc) = dst.get_mut(&repr) {
                enc.starts.push(ch.start);
                enc.ends.push(ch.end);
                let n = enc.directions.len();
                enc.directions.extend(ch.traj.iter().copied());
                enc.ranges.push(Range { start : n, end : enc.directions.len() });
            } else {
                let enc = ChainCode {
                    starts : vec![ch.start],
                    ends : vec![ch.end],
                    directions : ch.traj.iter().copied().collect(),
                    ranges : vec![Range { start : 0, end : ch.traj.len() }]
                };
                dst.insert(repr, enc);
            }
        }
        dst
    }

    pub fn split_by_proximity(&self, by : usize) -> BTreeMap<usize, ChainCode> {
        let unions = self.proximity_unions(by);
        self.chain_union_to_btree(unions)
    }

    pub fn split(&self) -> BTreeMap<usize, ChainCode> {
        let unions = self.unions();
        self.chain_union_to_btree(unions)
    }

    pub fn proximity_unions(&self, dist : usize) -> UnionFind<usize> {
        let mut uf = UnionFind::new(self.starts.len());
        /*for (i, c1) in self.chains().enumerate() {
            for (j, c2) in self.chains().enumerate() {
                if i != j && c1.close_to(&c2, dist).is_some() {
                    uf.union(i, j);
                }
            }
        }*/

        if self.starts.len() < 2 {
            return uf;
        }
        for i in 0..(self.starts.len()-1) {
            for j in i..self.starts.len() {
                let c1 = unsafe { self.get_unchecked(i) };
                let c2 = unsafe { self.get_unchecked(j) };
                if c1.is_close_to(&c2, dist) {
                    uf.union(i, j);
                }
            }
        }
        uf
    }

    /* Can also build an undirected graph using link as edges */
    pub fn unions(&self) -> UnionFind<usize> {
        let mut uf = UnionFind::new(self.starts.len());
        for (i, c1) in self.chains().enumerate() {
            for (j, c2) in self.chains().enumerate() {
                if i != j && c1.link(&c2).is_some() {
                    uf.union(i, j);
                }
            }
        }
        uf
    }

    pub fn encode<S>(bin_img : &Image<u8, S>) -> ChainCode
    where
        S : Storage<u8>
    {
        ChainEncoder::new(bin_img.height(), bin_img.width()).encode(bin_img, None)
    }

}

impl Encoding for ChainCode {

    fn points<'a>(&'a self) -> PointIter<'a> {
        PointIter(Box::new(
            self.chains().map(|chain| chain.trajectory() ).flatten()
        ))
    }

    fn encode_distinct<S>(img : &Image<u8, S>) -> BTreeMap<u8, Self> {
        unimplemented!()
    }

    fn encode_from<S>(&mut self, img : &Image<u8, S>) {
        unimplemented!()
    }
    
    // Write the labels back to the image.
    fn decode_distinct_to(labels : &BTreeMap<u8, Self>, img : &dyn AsMut<WindowMut<u8>>) {
        unimplemented!()
    }

}

/// Wraps any iterator over (usize, usize)
pub struct PointIter<'a>(Box<dyn Iterator<Item=(usize, usize)> + 'a>);

impl<'a> Iterator for PointIter<'a> {

    type Item=(usize, usize);
    
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
    
}

pub trait Encoding
where
    Self : Default
{

    fn points<'a>(&'a self) -> PointIter<'a>;

    fn cartesian_points<'a>(&'a self, img_size : Size) -> Box<dyn Iterator<Item=Vector2<f32>> + 'a> {
        let h = img_size.0 as f32;
        Box::new(self.points().map(move |pt| Vector2::new(pt.1 as f32, h - pt.0 as f32) ))
    }
    
    // If the argument is a labeled image, where distinct matched labels are values greater than or
    // equal to 1, and unmatched pixels are represented by zero, then encode_distinct
    // should return a set of distinct encodings that map to each label, as if each labeled
    // pixel were a binary image encoded as "this label against all others".
    fn encode_distinct<S>(img : &Image<u8, S>) -> BTreeMap<u8, Self>;

    fn encode_from<S>(&mut self, img : &Image<u8, S>);

    fn encode<S>(img : &Image<u8, S>) -> Self {
        let mut encoding : Self = Default::default();
        encoding.encode_from(img);
        encoding
    }

    // Tries to decode points into the image. If size is insufficient, return false
    // and stops.
    fn decode_to<T>(&self, img : &mut Image<u8, T>) -> bool 
    where
        T : StorageMut<u8>
    {
        img.fill(0);
        for pt in self.points().0 {
            if pt.0 < img.height() && pt.1 < img.width() {
                img[pt] = 255;
            } else {
                return false;
            }
        }
        true
    }

    fn decode(&self, shape : (usize, usize)) -> Option<ImageBuf<u8>> 
    where
        Box<[u8]> : StorageMut<u8>
    {
        // Assumes image must always be filled with zeros at the start of the call of decode_to
        let mut img = unsafe { Image::new_empty(shape.0, shape.1) };
        if self.decode_to(&mut img) {
            Some(img)
        } else {
            None
        }
    }

    // Write the labels back to the image.
    fn decode_distinct_to(labels : &BTreeMap<u8, Self>, img : &dyn AsMut<WindowMut<u8>>);

}

/* Chain encodings are mostly useful to encode binary images resulting from edge operators.
dense objects will be represented by snake-like chain patterns (pattern repeats forward and
backward across even and odd rows, so for dense images it is best to stick with RunLengthEncoding).
But for edge images, representing the (start, end) pair and a set of directions is much more
efficient, since each pixel is represented by a single byte tagging the direction of change,
instead of a full usize pair. The points can be recovered from any resolution desired by
calling trajectory(.) and sparse_trajectory(.). */
#[derive(Debug, Clone, Default)]
pub struct ChainCode {
    starts : Vec<(usize, usize)>,
    ends : Vec<(usize, usize)>,
    directions : Vec<Direction>,
    ranges : Vec<Range<usize>>
}

pub struct ChainIter<'a> {
    enc : &'a ChainCode,
    curr : usize,
    end : usize
}

impl<'a> Iterator for ChainIter<'a> {

    type Item = Chain<'a>;

    fn next(&mut self) -> Option<Chain<'a>> {
        if self.curr >= self.end {
            return None;
        }
        let ch = unsafe { self.enc.get_unchecked(self.curr) };
        self.curr += 1;
        Some(ch)
    }

}

/** Represents an edge using an 8-direction chain code.
Useful to represent binary images resulting from edge and contour operators. **/
#[derive(Debug, Clone, Copy)]
pub struct Chain<'a> {
    pub start : (usize, usize),
    pub end : (usize, usize),
    pub traj : &'a [Direction]
}

fn update_vec(pt : &mut Point2<f32>, d : Direction) {
    // *pt += &d.offset_vec();
}

fn update_point(pt : &mut (usize, usize), d : Direction) {
    let off = d.offset();
    pt.0 = (pt.0 as i32 + off.0) as usize;
    pt.1 = (pt.1 as i32 + off.1) as usize;
}

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum Link {
    StartToStart,
    StartToEnd,
    EndToStart,
    EndToEnd,
    Ambiguous,
    Closed,
}

fn is_point_close(a : (usize, usize), b: (usize, usize), by : usize) -> bool {
    a.0.abs_diff(b.0) <= by && a.1.abs_diff(b.1) <= by
}

impl<'a> Chain<'a> {

    pub fn len(&self) -> usize {
        // Sum 1 to count the start coordinate.
        self.traj.len() + 1
    }
    
    // Given a contiguous sequence of indices, split this chain in two by 
    // removing the directions at those indices.
    pub fn split(&self, range : Range<usize>) -> Option<(Chain, Chain)> {
        let mut traj = self.trajectory();
        let n_end_fst = range.start.checked_sub(1)?;
        let end_fst = traj.nth(n_end_fst)?;
        for _ in range.clone() {
            let _ = traj.next();
        }
        let start_snd = traj.next()?;
        Some((
            Chain { start : self.start, end : end_fst, traj : &self.traj[..=n_end_fst] },
            Chain { start : start_snd, end : self.end, traj : &self.traj[range.end..] }
        ))
    }

    pub fn is_close_to(&self, other : &Self, by : usize) -> bool {
        let start_start = is_point_close(self.start, other.start, by);
        let start_end = is_point_close(self.start, other.end, by);
        let end_start = is_point_close(self.end, other.start, by);
        let end_end = is_point_close(self.end, other.end, by);
        start_start || start_end || end_start || end_end
    }

    pub fn close_to(&self, other : &Self, by : usize) -> Option<Link> {
        let start_start = is_point_close(self.start, other.start, by);
        let start_end = is_point_close(self.start, other.end, by);
        let end_start = is_point_close(self.end, other.start, by);
        let end_end = is_point_close(self.end, other.end, by);
        if (start_start && end_end) || (start_end && end_start) {
            Some(Link::Closed)
        } else {
            if start_start && (!start_end && !end_start && !end_end) {
                Some(Link::StartToStart)
            } else if start_end && (!start_start && !end_start && !end_end) {
                Some(Link::StartToEnd)
            } else if end_start && (!start_start && !start_end && !end_end) {
                Some(Link::EndToStart)
            } else if end_end && (!start_start && !start_end && !end_start) {
                Some(Link::EndToEnd)
            } else if start_start || start_end || end_start || end_end {
                Some(Link::Ambiguous)
            } else {
                None
            }
        }
    }
    
    pub fn link(&self, other : &Self) -> Option<Link> {
        self.close_to(other, 1)
    }
    
    /* Returns the full set of points this chain represents. The method is lightweight and
    does not allocate. */
    pub fn trajectory(&self) -> impl Iterator<Item=(usize, usize)> + 'a {
        std::iter::once(self.start)
            .chain(self.traj.iter().scan(self.start, move |pt, d| {
                update_point(pt, *d);
                Some(*pt)
            }))
            
            // Already accounted for at the last direction step.
            // .chain(std::iter::once(self.end))
    }

    /* Returns every nth point this chain represents. The method is lightweight and
    does not allocate. */
    pub fn sparse_trajectory(&'a self, step : usize) -> impl Iterator<Item=(usize, usize)> + 'a {
        std::iter::once(self.start)
            .chain((0..self.traj.len()).step_by(step).skip(1).scan(self.start, move |pt, i| {

                // Accumulate previous points (excluding the previous step)
                for j in (i-step-1)..(i+1) {
                    update_point(pt, self.traj[j]);
                }

                Some(*pt)
            }))
    }

}

#[repr(u8)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Neighborhood {

    /// Four-neighborhood (top, below, left, right pixels) (Aka. Von Neumann neighborhood),
    /// characterized by CardinalDirection
    Immediate,

    /// Eight-neighborhood (all elements at immediate neighborhood plus
    /// top-left, top-right, bottom-left and bottom-right pixels) (Aka. Moore neighborhood),
    /// characterized by Direction
    Extended

}

/// Represents one of the four cardinal directions in a 4-neighborhood pixel connectivity graph
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum CardinalDirection {
    North,
    East,
    South,
    West
}

/// Represents one of the eight directions in a 8-neighborhood pixel connectivity graph
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum Direction {
    North,
    NorthEast,
    East,
    SouthEast,
    South,
    SouthWest,
    West,
    NorthWest
}

impl Direction {

    pub fn compare_below(above : (usize, usize), below : (usize, usize)) -> Direction {
        assert!(above.0 + 1 == below.0 && above.1.abs_diff(below.1) <= 1);
        if below.1 == above.1 {
            Direction::South
        } else if below.1 + 1 == above.1 {
            Direction::SouthWest
        } else if above.1 + 1 == below.1 {
            Direction::SouthEast
        } else {
            panic!("Invalid direction")
        }
    }

    pub fn reverse(&mut self) {
        match *self {
            Direction::North => { *self = Direction::South },
            Direction::NorthEast => { *self = Direction::SouthWest },
            Direction::East => { *self = Direction::West },
            Direction::SouthEast => { *self = Direction::NorthWest },
            Direction::South => { *self = Direction::North },
            Direction::SouthWest => { *self = Direction::NorthEast },
            Direction::West => { *self = Direction::East },
            Direction::NorthWest => { *self = Direction::SouthEast },
        }
    }

    pub fn offset(&self) -> (i32, i32) {
        match self {
            Direction::North => (-1, 0),
            Direction::NorthEast => (-1, 1),
            Direction::East => (0, 1),
            Direction::SouthEast => (1, 1),
            Direction::South => (1, 0),
            Direction::SouthWest => (1, -1),
            Direction::West => (0, -1),
            Direction::NorthWest => (-1, -1)
        }
    }
    
    pub fn offset_vec(&self) -> Point2<f32> {
        match self {
            Direction::North => Point2::new(0., 1.),
            Direction::NorthEast => Point2::new(1., 1.),
            Direction::East => Point2::new(1., 0.),
            Direction::SouthEast => Point2::new(1., -1.),
            Direction::South => Point2::new(0., -1.),
            Direction::SouthWest => Point2::new(-1., -1.),
            Direction::West => Point2::new(-1., 0.),
            Direction::NorthWest => Point2::new(-1., 1.)
        }
    }

}

/// Represents a set of homogeneous pixels. Useful to represent binary images resulting
/// in dense regions (patches), such as the ones generated by thresholding operations.
/// A lightweight (copy) struct representing a horizontal sequence of homogeneous
/// pixels by the coordinate of the first pixel and the length of the sequence.
/// The RunLength is the basis for a sparse representation of a binary image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunLength {
    pub start : (usize, usize),
    pub length : usize
}

impl RunLength {

    pub fn contains(&self, pt : (usize, usize)) -> bool {
        pt.0 == self.start.0 && pt.1 >= self.start.1 && pt.1 < self.start.1 + self.length
    }

    pub fn bounds(&self) -> ((usize, usize), (usize, usize)) {
        (self.start, self.end())
    }

    pub fn points<'a>(&'a self) -> impl Iterator<Item=(usize, usize)> + 'a {
        ((self.start.1)..(self.start.1 + self.length)).map(move |c| (self.start.0, c) )
    }

    pub fn middle_point(&self) -> (usize, usize) {
        (self.start.0, self.start.1 + self.length / 2 )
    }
    
    // TODO rename to inner_points
    pub fn middle_points<'a>(&'a self) -> impl Iterator<Item=(usize, usize)> + 'a {
        ((self.start.1+1)..(self.end().1.saturating_sub(1))).map(move |c| (self.start.0, c) )
    }

    pub fn start_col(&self) -> usize {
        self.start.1
    }

    pub fn end_col(&self) -> usize {
        self.start.1 + self.length-1
    }

    pub fn end(&self) -> (usize, usize) {
        (self.start.0, self.end_col())
    }

    pub fn as_rect(&self) -> (usize, usize, usize, usize) {
        (self.start.0, self.start.1, 1, self.length)
    }

    /// Represents a closed interval covered by this run-length
    pub fn interval(&self) -> Interval<usize> {
        Interval(self.start.1, self.start.1+self.length-1)
    }

    pub fn range(&self) -> Range<usize> {
        Range { start : self.start.1, end : self.start.1+self.length }
    }

    pub fn intersect(&self, other : &Self) -> Option<RunLength> {
        self.intersect_interval(other.interval())
    }

    /// Returns the subset of Self that intersects vertically with the given interval
    pub fn intersect_interval(&self, intv : Interval<usize>) -> Option<RunLength> {
        self.interval().intersect(intv)
            .map(|intv| RunLength { start : (self.start.0, intv.0), length : intv.1-intv.0+1 } )
    }

    // It never makes sense to check for contact of RLEs at the same row,
    // because two neighboring RLEs should have been merged into one. So
    // contacts only test elements in different rows.
    pub fn contacts(&self, other : &Self) -> bool {
        (self.start.0 as i32 - other.start.0 as i32).abs() == 1 &&
            self.interval_overlap(other)
    }

    // Panics if Other > self.
    pub fn interval_overlap(&self, other : &Self) -> bool {
        // crate::region::raster::interval_overlap(self.interval(), other.interval())
        self.interval().intersect(other.interval()).is_some()
    }

}

pub fn strict_nonzero_bound_window<'a>(w : &'a Window<'a, u8>) -> Window<'a, u8> {
    let vert_win = vertical_nonzero_bound_window(w);
    let (min_r, mut min_c) = vert_win.offset();
    let shape = vert_win.shape();
    let (max_r, mut max_c) = (min_r + shape.0, min_c + shape.1);
    let sub_height = max_r - min_r;
    let left_win = w.sub_window((min_r, 0), (sub_height, min_c)).unwrap();
    let right_win = w.sub_window((min_r, max_c), (sub_height, w.width() - max_c)).unwrap();
    let mut r = min_r;
    while r < max_r {
        for c in 0..min_c {
            if w[(r, c)] == 255 && c < min_c {
                min_c = c;
                break;
            }
        }
        for c in (max_c..w.width()).rev() {
            if w[(r, c)] == 255 && c > max_c {
                max_c = c;
                break;
            }
        }
        r += 1;
    }
    let shape = (max_r - min_r, max_c - min_c);
    w.sub_window((min_r, min_c), shape).unwrap()
}

pub fn vertical_nonzero_bound_at_integral<'a>(w : &'a Window<'a, i32>) -> Window<'a, i32> {
    let mut tl = (0, 0);
    let mut br = (w.height()-1, w.width()-1);
    for r in 0..w.height() {
        for c in 0..w.width() {
            if w[(r, c)] != 0 {
                tl = (r, c);
                break;
            }
        }
    }
    let last = w[(w.height()-1, w.width()-1)];
    for r in (0..w.height()).rev() {
        for c in (0..w.width()).rev() {
            if w[(r, c)] != last {
                br = (r, c);
                break;
            }
        }
    }
    let shape = (br.0 - tl.0, br.1 - tl.1);
    w.sub_window(tl, shape).unwrap()
}

// Returns the subwindow of w that contains the smallest region containing
// the totality of the nonzero pixels of w (assumed binary). This can speed up
// encoding for sparse binary images, since the algorithm does not iterate over
// all pixels. Rather, it uses the row of the minimum column and the column of
// the minimum row to ignore full image regions at further iterations. If window
// contains no nozero pixels, return the full window. This verifies only the vertical dimension,
// if the strict minimal bound is desired, use the strict variant.
pub fn vertical_nonzero_bound_window<'a>(w : &'a Window<'a, u8>) -> Window<'a, u8> {

    assert!(w.height() > 1);

    let mut min_r = 0;
    let mut max_r = w.height() - 1;
    let mut min_c = 0;
    let mut max_c = w.width() - 1;

    // Iterate from bottom to top
    let mut r = 1;
    let mut c = 0;
    while r < w.height() - 1 {

        // Iterate from left-to-right
        while c < w.width() {
            if w[(r, c)] == 255 {
                min_c = c;
                max_c = c;
                min_r = r;
                break;
            }
            c += 1;
        }

        // If first nonzero pixel found here, iterate from right-to-left
        if min_r == r {
            c = w.width();
            while c > min_c {
                if w[(r, c)] == 255 {
                    max_c = c;
                    break;
                }
                c -= 1;
            }
            break;
        }

        r += 1;
    }

    // Iterate from top to bottom
    r = w.height() - 2;
    c = 0;
    while r > min_r {
        // Iterate from left-to-right
        while c < w.width() {
            if w[(r, c)] == 255 {
                if c < min_c {
                    min_c = c;
                }
                if c > max_c {
                    max_c = c;
                }
                max_r = r;
                break;
            }
            c += 1;
        }

        // If first nonzero pixel found here, iterate from right-to-left
        if max_r == r {
            c = w.width();
            while c > max_c {
                if w[(r, c)] == 255 {
                    if c > max_c {
                        max_c = c;
                    }
                    break;
                }
                c -= 1;
            }
            break;
        }

        r -= 1;
    }

    let shape = (max_r - min_r, max_c - min_c);
    w.sub_window((min_r, min_c), shape).unwrap()
}

// A compact representation for a set of coordinates. Can be used
// to represent the foreground pixels of a binary image. Each RunLength
// represents a set of horizontally contiguous points by the first point
// in the sequence and the number of points in the sequence. To speed up
// search, also carry a set of ranges that signals RunLengths that
// are part of the same row. RunLengths are always sorted by rows, then
// by columns within rows.
// RunLength encodings are sorted by starting points within rows
// and also by rows, so to determine if a pixel is positive can be
// done by bisection.
#[derive(Debug, Clone, Default)]
pub struct RunLengthCode {
    pub rles : Vec<RunLength>,
    pub rows : Vec<Range<usize>>
}

pub struct RunLengthIter<'a> {
    enc : &'a RunLengthCode,
    curr : usize
}

impl Encoding for RunLengthCode {

    fn points<'a>(&'a self) -> PointIter<'a> {
        PointIter(Box::new(self.rles.iter().map(move |rle| rle.points() ).flatten()))
    }

    fn encode_distinct<T>(img : &Image<u8, T>) -> BTreeMap<u8, Self> {
        unimplemented!()
    }

    fn encode_from<T>(&mut self, img : &Image<u8, T>) {
        unimplemented!()
    }

    // Write the labels back to the image.
    fn decode_distinct_to(labels : &BTreeMap<u8, Self>, img : &dyn AsMut<WindowMut<u8>>) {
        unimplemented!()
    }

}

/* Given a (sorted) RLE vector, extract its row vector. */
fn rle_rows_from_sorted(rles : &[RunLength]) -> Vec<Range<usize>> {
    let mut rows = Vec::new();
    let mut start = 0;
    for (r, mut s) in &rles.iter().group_by(|v| v.start.0 ) {
        let row_len = s.count();
        rows.push(Range { start, end : start + row_len });
        start += row_len;
    }
    rows
}

// Recursively search for regions in a row of an accumulated binary image
// that might contain RunLengths, by just comparing the differences of the
// boundary pixels over a row segment.
fn calculate_valid_col_ranges(ranges : &mut Vec<Range<usize>>, w : &Window<i32>, r : usize, a : usize, b : usize, min_sz : usize) {

    // min_sz cannot be close to 2, because two pixels can be equal and be a white->dark transition.
    assert!(min_sz >= 4);

    assert!(b > a);

    // assert!((b - a) % 2 == 0, "a = {}; b = {}", a, b);

    // If left and right pixels are equal in the accumulated image,
    // this is a dark region at the binary image.
    if w[(r, b)] > w[(r, a)] {
        if b - a > min_sz {

            // Since a transition between foreground pixel to background pixel
            // always happen for i == j, it is critical that the valid range
            // contains the first pixel repetition.

            let half = ((a as f32 + b as f32) / 2.0).ceil() as usize;
            assert!((half - a) + (b - half) == b - a);

            calculate_valid_col_ranges(ranges, w, r, a, half, min_sz);
            calculate_valid_col_ranges(ranges, w, r, half+1, b, min_sz);
        } else {
            ranges.push(Range { start : a, end : b });
        }
    }
    // If w[a] and w[b] are equal, ignore the region for further processing
}

fn calculate_valid_row_ranges(ranges : &mut Vec<Range<usize>>, w : &Window<i32>, a : usize, b : usize, min_sz : usize) {
    assert!(b > a);
    // If top left and bottom right pixels in the row rergion are equal in the accumulated image,
    // this is a dark region at the binary image.
    if w[(a, 0)] < w[(b, w.width()-1)] {
        if b - a > min_sz {
            let half = ((a as f32 + b as f32) / 2.0).ceil() as usize;
            assert!((half - a) + (b - half) == b - a);
            calculate_valid_row_ranges(ranges, w, a, half, min_sz);
            calculate_valid_row_ranges(ranges, w, half+1, b, min_sz);
        } else {
            ranges.push(Range { start : a, end : b });
        }
    }
    // If w[a] and w[b] are equal, ignore the region for further processing
}

fn merge_col_ranges(ranges : &mut Vec<Range<usize>>) {
    if ranges.len() < 2 {
        return;
    }

    // Holds a "lead" range index (one that might be extended with zero, one or more "part" ranges
    let mut lead = 0;

    // Holds a "part" range index, that might either merge to the lead range before it if they are
    // contiguous, or become its own lead range if it is not contiguous to the past lead range.
    let mut part = 1;

    while part < ranges.len() {
        if ranges[lead].end + 1 == ranges[part].start {

            // Extend lead range
            ranges[lead].end = ranges[part].end;

            // Invalidate this part end
            ranges[part].end = ranges[part].start;

        } else {
            lead = part;
        }
        part += 1;
    }

    // Not strictly necessary (since start==end range iterators yield None, so
    // there is no harm in keeping them in the valid range iterator), but added
    // here for clarity.
    ranges.retain(|r| r.end > r.start );
}

// Perform a search for one or two RunLenghts recursively starting from the extremities. From the
// structure of the IntegralImage, we can know if we have one or more RunLengths in a pair of pixel
// boundaries by verifying their difference in pixel value against their horizontal distance (if they
// are equal, we have a single RunLenght and can stop iteration. If not, we iterate until we find the
// end of the first left runlength and the start of the last right runlenght, then call this recursively
// starting after the end of the left runlenght and before the start of the right runlength). This represents
// huge computational savings for sparse binary images, since whole rows or large segments of rows can be
// ignored during the search.
fn find_at_row_segment(rles : &mut Vec<RunLength>, w : &Window<i32>, r : usize, mut a : usize, mut b : usize) {
    assert!(b >= a, "a = {}; b = {}", a, b);
    let mut left_px = w[(r, a)];
    let mut right_px = w[(r, b)];
    let d = right_px - left_px;
    // println!("d = {d}");
    if d == 0 {
        return;
    } else {

        // TODO differentiate an "even" start from an "odd" start, since
        // we do not know if we will start at the beginning or end of the RLE.

        // Skip flat region to right (dark binary segment after any RLEs)
        /*while w[(r, b-1)] == right_px {
            b -= 1;
        }
        b -= 1;*/

        // Note the first white->dark transition repeats the last
        // white pixel value, which is why we evaluate one-past b.
        while w[(r, b)] - w[(r, b-1)] == 0 {
            b -= 1;
        }

        // Skip flat region to left (dark binary segment before any RLEs)
        /*while w[(r, a+1)] == left_px {
            a += 1;
        }*/
        while w[(r, a+1)] - w[(r, a)] == 0 {
            a += 1;
        }
        assert!(b >= a + 1);

        right_px = w[(r, b)];
        left_px = w[(r, a)];
        assert!(b >= a, "{}, {}", a, b);
        println!("Diff a = {}, b = {}", a, b);
        if (right_px - left_px) / 255 == (b - a) as i32 {
            if r == 0 {
                println!("Unique RLE found: {:?} between {:?}", RunLength { start : (r, a), length : b - a + 1 }, (a, b));
            }
            rles.push(RunLength { start : (r, a), length : b - a });
        } else {
            let mut left_length = 1;
            let mut right_length = 1;
            while w[(r, a+left_length+1)] - w[(r, a+left_length)] == 255 {
                left_length += 1;
            }

            // This iterator goes one-past the right length because the first dark
            // pixel repeats the last while pixel accumulated value.
            while w[(r, b-right_length-1)] - w[(r, b-right_length-2)] == 255 {
                right_length += 1;
            }

            // We add an offset of 1 to all RunLenghts because the accumulated image
            // is shifted by one relative to the original image (since the first source
            // pixel is equal to the first accumulated pixel).
            let left = RunLength { start : (r, a + 1), length : left_length };
            let right = RunLength { start : (r, b - right_length + 1), length : right_length };
            if r == 0 {
                println!("RLE pair found between {}-{} : {:?} {:?}",
                    a,
                    b,
                    left,
                    right
                );
            }
            rles.push(left);
            rles.push(right);

            // There should be at least one dark pixel between the two recently-inserted RLEs
            // (since it would have been considered a sigle RLE in the if clause otherwise).
            let new_a = a+left_length+1;
            let new_b = b-right_length-1;
            assert!(new_a < new_b, "a : {}, b : {}", new_a, new_b);
            find_at_row_segment(rles, w, r, new_a, new_b);
        }
    }
}

#[test]
fn rle_distinct() {

    use crate::draw::*;
    use crate::gray::MedianCutQuantization;
    use crate::gray::Quantization;
    
    let mut buf = crate::io::decode_from_file("/home/diego/Downloads/pinpoint.png").unwrap();
    let mut qt = MedianCutQuantization::new(4, None);
    qt.quantize_to(&buf.clone(), &mut buf);
    let rles = RunLengthCode::encode_distinct(&buf, None);
    
    let keys : Vec<u8> = rles.keys().cloned().collect();
    let mut rects_b = BTreeMap::new();
    for key in &keys {
        let graph = rles[key].graph();
        let rects : Vec<_> = graph.outer_rects(None).values().cloned().collect();
        rects_b.insert(key.clone(), rects);
    }
    
    /*for k1 in &keys {
        for k2 in &keys {
            if k1 == k2 {
                continue;
            }
            for i in 0..rects_b[k1].len() {
                for j in (0..rects_b[k2].len()).rev() {
                    if crate::shape::rect_contacts3(&rects_b[k1][i], &rects_b[k2][j]) && k1.abs_diff(*k2) <= 8 {
                        rects_b.get_mut(k1).unwrap()[i] = crate::shape::rect_enclosing_pair(&rects_b[k1][i], &rects_b[k2][j]);
                        rects_b.get_mut(&k2).unwrap().remove(j);
                    }
                }
            }
        }
    }*/
    
    for (_, rects) in rects_b.iter() {
        let mut buf = buf.clone();
        for r in rects {
            buf.draw(Mark::Rect((r.0, r.1), (r.2, r.3)), 255);
        }
        buf.show();
    }
}

fn find_row_rles(
    valid_col_ranges : &mut Vec<Range<usize>>,
    rles : &mut Vec<RunLength>,
    rows : &mut Vec<Range<usize>>,
    w : &Window<i32>,
    r : usize,
    min_col_range : Option<usize>
) {
    calculate_valid_col_ranges(valid_col_ranges, w, r, 0, w.width() - 1, min_col_range.unwrap_or(MIN_VALID_INTEGRAL_COL_RANGE));
    valid_col_ranges.sort_by(|a, b| a.start.cmp(&b.start) );
    merge_col_ranges(valid_col_ranges);
    for range in valid_col_ranges.drain(..) {
        let n_before = rles.len();
        find_at_row_segment(rles, w, r, range.start, range.end);
        let n_after = rles.len();
        rles[n_before..n_after].sort_by(|a, b| a.start.1.cmp(&b.start.1) );
        rows.push(Range { start : n_before, end : n_after });
    }
}

/*// cargo test --all-features -- accumulated_rle --nocapture
#[test]
fn accumulated_rle() {
    let mut img = Image::new_checkerboard(100, 10);
    img[(0, 2)] = 255;
    img[(0, 3)] = 255;
    img[(0, 4)] = 255;

    img[(0, 7)] = 255;
    img[(0, 8)] = 255;

    img.show();
    println!("{:?}", img.row(0));

    let acc = crate::integral::Accumulated::calculate(img.as_ref());
    let iw : &Window<i32> = acc.as_ref();
    println!("{:?}", iw.row(0));
    // for i in 0..5 {
    //    println!("row {}: {:?}", i, iw.row(i).unwrap());
    // }
    // println!("Integral = {:?}", iw);
    // println!("{:?}", iw.row(1));
    let rle = RunLengthEncoding::calculate_from_accumulated(&acc, None, None);
    // println!("{:?}", rle);
}*/

pub const MIN_VALID_INTEGRAL_COL_RANGE : usize = 8;

pub const MIN_VALID_INTEGRAL_ROW_RANGE : usize = 8;

// Assume pts is sorted by rows and by cols within rows.
fn rows_for_sorted_rles(rles : &[RunLength], rows : &mut Vec<Range<usize>>) {
    rows.clear();
    let mut n = 0;
    for (_, mut rles) in &rles.iter().group_by(|r| r.start.0 ) {
        let row_len = rles.count();
        rows.push(Range { start : n, end : n+row_len } );
        n += row_len;
    }
}

fn push_left(
    rles : &mut Vec<RunLength>,
    rows : &mut Vec<Range<usize>>,
    pt : (usize, usize),
    rle : RunLength,
    // row_start_pos : usize,
    row_pos : usize
) -> bool {
    if pt.1 < rle.start.1 {
        rles.insert(rows[row_pos].start, RunLength { start : pt, length : 1 });
        rows[row_pos].end += 1;
        // if row_pos < rows.len()-1 {
        shift_by(rows, row_pos, 1);
        // }
        true
    } else {
        false
    }
}

fn push_right(
    rles : &mut Vec<RunLength>,
    rows : &mut Vec<Range<usize>>,
    pt : (usize, usize),
    rle : RunLength,
    // row_start_pos : usize,
    row_pos : usize
) -> bool {
    if pt.1 > rle.range().end {
        rles.insert(rows[row_pos].end, RunLength { start : pt, length : 1 });
        rows[row_pos].end += 1;
        // if row_pos < rows.len()-1 {
        shift_by(rows, row_pos, 1);
        // }
        true
    } else {
        false
    }
}

fn shift_by(all_rows : &mut [Range<usize>], row_pos : usize, val : i32) {
    if row_pos < all_rows.len()-1 {
        let right_rows = &mut all_rows[(row_pos+1)..];
        let val_abs = val.abs() as usize;
        if val > 0 {
            for row in right_rows {
                row.start += val_abs;
                row.end += val_abs;
            }
        } else {
            for row in right_rows {
                row.start -= val_abs;
                row.end -= val_abs;
            }
        }
    }
}

fn merges_left(rle : &mut RunLength, pt : (usize, usize)) -> bool {
    if rle.start.1.checked_sub(pt.1) == Some(1) {
        rle.start.1 -= 1;
        rle.length += 1;
        true
    } else {
        false
    }
}

fn merges_right(rle : &mut RunLength, pt : (usize, usize)) -> bool {
    if pt.1 == rle.range().end {
        rle.length += 1;
        true
    } else {
        false
    }
}

fn merges_left_or_right(rle : &mut RunLength, pt : (usize, usize)) -> bool {
    if merges_left(rle, pt) {
        true
    } else {
        merges_right(rle, pt)
    }
}

// cargo test --lib -- rle_include --nocapture
#[test]
fn rle_include() {
    let pts = [
        vec![(16, 19), (16, 20), (17, 14), (17, 15), (17, 17), (17, 19), (17, 20), (18, 14), (18, 15)]
        //vec![(0,1), (0,2), (0,3)],
        //vec![(0,3), (0,2), (0,1)],
        //vec![(0,1), (0,3), (0,2)],
        //vec![(0,1), (1,1), (2, 1)],
        //vec![(2,0), (1,0), (0,0)],
        //vec![(0,1), (0,2), (0,5), (0,6), (0,7), (0,3), (0,4)],
    ];
    for pt_set in pts {
        let mut rle = RunLengthCode::default();
        pt_set.iter().for_each(|pt| { rle.include(*pt); });
        println!("{:?}", rle);
    }

}

impl RunLengthCode {

    // Adds a single point to this RLE (if not already present).
    // Either increment an existing RLE or innaugurate a new one.
    // Panics if point was already present on RLE.
    pub fn include(&mut self, pt : (usize, usize)) {
        match self.rows.binary_search_by(|r| self.rles[r.clone()][0].   start.0.cmp(&pt.0) ) {
            Ok(row_pos) => {
                let fst_rle_pos = self.rows[row_pos].start;
                match self.rows[row_pos].len() {
                    1 => {
                        let fst_rle = self.rles[fst_rle_pos].clone();
                        if !merges_left_or_right(&mut self.rles[fst_rle_pos], pt) {
                            let pushed_left = push_left(
                                &mut self.rles,
                                &mut self.rows,
                                pt,
                                fst_rle,
                                row_pos
                            );
                            if !pushed_left {
                                push_right(
                                    &mut self.rles,
                                    &mut self.rows,
                                    pt,
                                    fst_rle,
                                    row_pos
                                );
                            }
                        }
                    },
                    2.. => {
                        let row_end_pos = self.rows[row_pos].end - 1;
                        let merged_or_pushed_start = if merges_left(&mut self.rles[fst_rle_pos], pt) {
                            true
                        } else {
                            let fst_rle = self.rles[fst_rle_pos].clone();
                            push_left(
                                &mut self.rles,
                                &mut self.rows,
                                pt,
                                fst_rle,
                                row_pos
                            )
                        };
                        let merged_or_pushed_end = if merged_or_pushed_start {
                            false
                        } else {
                            let lst_rle = self.rles[row_end_pos].clone();
                            if merges_right(&mut self.rles[row_end_pos], pt) {
                                true
                            } else {
                                push_right(
                                    &mut self.rles,
                                    &mut self.rows,
                                    pt,
                                    lst_rle,
                                    row_pos
                                )
                            }
                        };
                        if !(merged_or_pushed_start || merged_or_pushed_end) {
                            let part = self.rles[self.rows[row_pos].clone()]
                                .partition_point(|rle| rle.start.1 < pt.1 && rle.range().end <= pt.1 );
                            let pos_left = fst_rle_pos + part - 1;
                            let pos_right = fst_rle_pos + part;
                            let merges_left = pt.1 == self.rles[pos_left].range().end;
                            let merges_right = self.rles[pos_right].start.1.checked_sub(pt.1) == Some(1);
                            if merges_left && merges_right {
                                let length_right = self.rles[pos_right].length;
                                self.rles[pos_left].length += (length_right + 1);
                                self.rles.remove(pos_right);
                                self.rows[row_pos].end -= 1;
                                //if row_pos < self.rows.len()-1 {
                                shift_by(&mut self.rows, row_pos, -1);
                                // }
                            } else if merges_left {
                                self.rles[pos_left].length += 1;
                            } else if merges_right {
                                self.rles[pos_right].start.1 -= 1;
                                self.rles[pos_right].length += 1;
                            } else if pt.1 > self.rles[pos_left].range().end && pt.1 < self.rles[pos_right].start.1 {
                                self.rles.insert(pos_right, RunLength { start : pt, length : 1 });
                                self.rows[row_pos].end += 1;
                                //if row_pos < self.rows.len()-1 {
                                shift_by(&mut self.rows, row_pos, 1);
                                //}
                            }
                        }
                    },
                    _ => { }
                }
            },
            Err(row_pos) => {
                if row_pos == 0 {
                    self.rows.insert(row_pos, Range { start : 0, end : 1 });
                    self.rles.insert(0, RunLength { start : pt, length : 1});
                    //if row_pos < self.rows.len()-1 {
                         shift_by(&mut self.rows, row_pos, 1);
                    // }
                } else {
                    let rle_pos = self.rows[row_pos-1].end;
                    self.rows.insert(row_pos, Range { start : rle_pos, end : rle_pos + 1});
                    self.rles.insert(rle_pos, RunLength { start : pt, length : 1 });
                    //if row_pos < self.rows.len()-1 {
                    shift_by(&mut self.rows, row_pos, 1);
                    // }
                }
            }
        }
        verify_rle_state(&self);
    }

    pub fn contains(&self, pt : (usize, usize)) -> bool {
        match self.rows.binary_search_by(|r| self.rles[r.clone()][0].start.0.cmp(&pt.0) ) {
            Ok(row) => {
                self.rles[self.rows[row].clone()]
                    .binary_search_by(|rle| if rle.contains(pt) { Ordering::Equal } else { rle.start.1.cmp(&pt.1) } )
                    .is_ok()
            },
            Err(_) => false
        }
    }

    // Unity measures the average number of run lenghs per row in a connected RLC.
    // Solidity 1.0 means a single RunLength per row, signalling a fully connected object
    // without holes. Greater values means objects full of holes.
    pub fn unity(&self) -> f32 {
        let mut s = 0.0;
        for row in &self.rows {
            s += (row.end - row.start) as f32;
        }
        s /= self.rows.len() as f32;
        s
    }

    /** Distance between last and first row of this RLE. This is
    calculated in constant time. RLEs might not be connected.
    When this RLE is connected, this is the height of the outer
    rect. Returns None when RLE is empty. This equals self.rows.len()
    when the RLE is connected. **/
    pub fn height(&self) -> Option<usize> {
        Some(self.rles.last()?.start.0 - self.rles.first()?.start.0 + 1)
    }

    // Returns the width of the longest RunLength. This serves as a lower
    // bound for the width of the outer rect.
    pub fn best_width(&self) -> usize {
        let mut max_w = 0;
        for rl in &self.rles {
            if rl.length > max_w {
                max_w = rl.length;
            }
        }
        max_w
    }

    pub fn region(&self) -> Option<crate::shape::Region> {
        self.outer_rect().as_ref().map(|r| crate::shape::Region::from_rect_tuple(r) )
    }

    pub fn outer_rect(&self) -> Option<(usize, usize, usize, usize)> {
        let mut rect = self.rles.first()?.as_rect();
        if self.rles.len() > 1 {
            for rl in &self.rles[1..] {
                extend_rect_with_sorted_rle(&mut rect, rl);
            }
        }
        Some(rect)
    }

    pub fn from_sorted(rles : Vec<RunLength>) -> Self {
        let rows = rle_rows_from_sorted(&rles);
        Self { rles, rows }
    }

    pub fn from_unsorted(mut rles : Vec<RunLength>) -> Self {
        rles.sort_by(|a, b| a.start.0.cmp(&b.start.0).then(a.start.1.cmp(&b.start.1)) );
        Self::from_sorted(rles)
    }

    /* Transpose this RunLenght, so that its decoding is equivalent to the decoding of the transposed image */
    pub fn transpose(mut self) -> Self {
        if self.rows.len() == 0 {
            return Default::default();
        }
        self.rles.iter_mut().for_each(|r| r.start = (r.start.1, r.start.0) );
        self.rles.sort_by(|a, b| 
            a.start.0.cmp(&b.start.0).then(a.start.1.cmp(&b.start.1)) 
        );
        rows_for_sorted_rles(&self.rles[..], &mut self.rows);
        // self
        // todo use connectivity to decide new runlenght size
        unimplemented!()
    }
    
    pub fn preserve_larger(&self, min_len : usize) -> RunLengthCode {
        let mut new_rles = self.rles.clone();
        new_rles.retain(|rle| rle.length >= min_len );
        let mut new_rows : Vec<Range<usize>> = Vec::new();
        rows_for_sorted_rles(&new_rles[..], &mut new_rows);
        let new_rle = Self { rles : new_rles, rows : new_rows };
        verify_rle_state(&self);
        new_rle
    }

    pub fn iter(&self) -> RunLengthIter {
        RunLengthIter {
            enc : self,
            curr : 0
        }
    }

    // pub fn split(&self) -> crate::graph::SplitGraph {
    //    crate::graph::group_weights(&self.graph, &self.uf)
    // }

    /*/// Returns the complement of this RunLength.
    pub fn gaps(&self) -> RunLengthEncoding {
        let mut neg_rles = Vec::new();
        let mut neg_rows = Vec::new();
        let mut ix = 0;
        for r in 0..self.rows[self.rows.len()-1] {
            while r < rles[curr_ix].0 {
                neg_rles.push(full_row);

            }
            // Push RLEs at this row
        }
    }*/

    // Builds a RunLength encoding from offsets (useful when a filter
    // was applied to img.windows(.) and only a few windows were maintained.
    // offsets is assumed to be a non-overlapping set of matching windows of the
    // given shape.
    pub fn from_offsets(offsets : &[(usize, usize)], shape : (usize, usize)) -> Self {

        let mut offsets : Vec<_> = offsets.to_owned();
        offsets.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)) );

        // For a start, do only the first row of each window.
        let mut rles = Vec::new();
        let mut curr : Option<RunLength> = None;
        for off in offsets {
            if let Some(mut this_curr) = curr.take() {
                if off.0 > this_curr.start.0 {
                    rles.push(this_curr);
                    curr = Some(RunLength { start : off, length : shape.1 });
                } else if off.0 == this_curr.start.0 {
                    if let Some(dist) = off.1.checked_sub(this_curr.start.1+this_curr.length) {
                        if dist == shape.1 {
                            this_curr.length += shape.1;
                            curr = Some(this_curr);
                        } else {
                            rles.push(this_curr);
                            curr = Some(RunLength { start : off, length : shape.1 });
                        }
                    } else {
                        // println!("{:?}", (this_curr, off));
                        panic!("Offset element in non-increasing col order found");
                    }
                } else {
                    // off.0 < curr.0
                    panic!("Offset element in non-increasing row order found");
                }
            } else {
                curr = Some(RunLength { start : off, length : shape.1 });
            }
        }

        if let Some(last_curr) = curr.take() {
            rles.push(last_curr);
        }

        // Now, reproduce all RLEs for as many rows as the window has.
        if shape.0 > 1 {
            let cloned_rles = rles.clone();
            for extra_row in 1..shape.1 {
                let row_below = cloned_rles.iter()
                    .map(|r| RunLength { start : (r.start.0 + extra_row, r.start.1), length : r.length });
                rles.extend(row_below);
            }
        }
        let rlc = RunLengthCode::from_unsorted(rles);

        verify_rle_state(&rlc);

        rlc
    }

    pub fn calculate_from_accumulated(
        w : &dyn AsRef<Window<i32>>,
        min_valid_row_range : Option<usize>,
        min_valid_col_range : Option<usize>
    ) -> Self {
        let mut r = Self::default();
        r.update_from_accumulated(w, min_valid_row_range, min_valid_col_range);
        r
    }

    // Similar to prune, but leaves the extremities untouched and examine
    // only the middle of the RunLenghth, possibly splitting them into
    // two or more RunLengths depending on whether there are new unmatched
    // pixels separating them.
    // pub fn split(&mut self, w : &dyn AsRef<Window<u8>>) {
    // }

    // Similar to expand, but shortens unmatched RunLength regions (or remove
    // them entirely), by just examining their extremities.
    pub fn prune(&mut self, w : &dyn AsRef<Window<u8>>) {

    }

    // Prunes, then expands the current RunLenghs.
    pub fn prune_and_expand(&mut self, w : &dyn AsRef<Window<u8>>) {
        self.prune(w);
        self.expand(w);
    }

    /* Updates the current RLE by iterating only over pixels *outside* the RLE, but
    connected to it.
    This is useful for an iterative thresholding strategy: If the desired foreground
    is more extreme than the current one, we know the RLE should contain at least the
    current pixels, therefore it is not necessary to iterate over them anymore. For
    this strategy to work, always start from a more conservative estimate, and increment
    towards a more extreme value. This only examines pixels that are either at the start,
    end, top or bottom of existing RLEs, or pixels neihboring the new matched regions recursively.
    Pixel rows outside these neighborhoods are ignored. */
    pub fn expand(&mut self, w : &dyn AsRef<Window<u8>>) {
        let mut r = 0;
    }

    /* pixel values 255 and 0 cannot be used as a labels. This algorithm first calculates a binary
    image with continuities in labels set to 255 and discontinuities between labels set to zero.
    It verifies the actual color labels only once per RLE. It does not represent the discontinuities
    between label patches, nor does it represents RLEs with size smaller than 3. Optionally receive
    an shif_diff buff of the same size as labels to save up on an extra image allocation. */
    pub fn encode_distinct<S>(
        labels : &Image<u8, S>, 
        shift_diff : Option<&mut ImageMut<u8>>
    ) -> BTreeMap<u8, RunLengthCode>
    where
        S : Storage<u8>
    {
        let sz_reduced = (labels.height(), labels.width()-1);
        let mut shift_diff_m = MutableImage::from_like(shift_diff, labels);
        let mut shift_diff = shift_diff_m.window_mut((0, 0), sz_reduced).unwrap();
        
        let left_labels = labels.window((0, 0), sz_reduced).unwrap();
        let right_labels = labels.window((0, 1), sz_reduced).unwrap();
        shift_diff.copy_from(&left_labels);
        shift_diff.abs_diff_assign(&right_labels);
        shift_diff.truncate_inplace(Foreground::Exactly(0), 255);
        shift_diff.truncate_inplace(Foreground::Below(255), 0);
        let mut global_rle = RunLengthCode::calculate(&shift_diff);
        let mut color_rles : BTreeMap<u8, RunLengthCode> = BTreeMap::new();
        
        for rl in global_rle.rles.drain(..) {
            if rl.length < 3 {
                continue;
            }
            let color = labels[rl.middle_point()];
            assert!(color != 0);
            assert!(color != 255);
            if let Some(code) = color_rles.get_mut(&color) {
                code.rles.push(rl.clone());
            } else {
                let mut rles = vec![rl];
                color_rles.insert(color, RunLengthCode { rles, rows : Vec::new() });
            }
        }
        
        for (_, rle) in color_rles.iter_mut() {
            let mut n = 0;
            let mut rows = Vec::new();
            for row_group in &rle.rles.iter().group_by(|r| r.start.0 ) {
                let end = n + row_group.1.count();
                rows.push(Range { start : n, end });
                n = end;
            }
            rle.rows = rows;
        }
        
        color_rles
    }
    
    // Find RLEs by bisecting over running sum over rows only (each new row
    // start at zero, and goes up to 256 at the final column if we are at a u8 image).
    // Although slower than bisecting over a full accumulated image, since there is still
    // a linear search over rows, this can operate over u8 images directly if the
    // binary image uses 1 to represent foreground values. This fails if any rows
    // saturate their sum (last pixel is 256), since it is not possible to determine
    // if there are any RLEs after the saturated column. The performance gain here
    // explores the fact that running sums of u8 images can execute vectorized
    // instructions containing more pixels, and there isn't the cost of conversion
    // if you start with a u8 image already.
    pub fn update_from_accumulated_rows(
        &mut self,
        w : &dyn AsRef<Window<i32>>,
        min_col_range : Option<usize>
    ) -> bool {
        let w = w.as_ref();
        self.rles.clear();
        self.rows.clear();
        let mut valid_col_ranges = Vec::new();
        for r in 0..w.height() {
            if w[(r, w.width()-1)] == i32::max_value() {
                return false;
            }
            find_row_rles(&mut valid_col_ranges, &mut self.rles, &mut self.rows, w, r, min_col_range);
        }
        verify_rle_state(&self);
        true
    }

    /* A RunLength encoding can usually be calculated much faster by bisection
    over the rows and columns of an accumulator image (containing the running sum of
    pixels in raster order). At each iteration, check if the pixels of a region boundary
    are equal. If they are, ignore the row region. This can be repeated
    recursively over non-examined regions. When a region does not have equal
    boundary pixels, it must be examined to determine the n RunLengths.
    If the difference between pixels happen to be equal to the number of columns
    times the non-zero constant value, we know there is a single RunLength there.
    This might mean a performance gain because while calculating the accumulated image
    is a vectorized operation, pointwise evaluating the RunLenghts for flat images is not.
    min_valid_range is the row length at which bisection stops, and can be chosen based on
    the expected object width (smaller values mean more bisection iterations, which might
    increase performance if objects are smaller than the desired range, since it reduces
    the number of linear pixel iterations). A defaut value of 8 is used if no value is informed.
    The image can also be bisected over whole row ranges, by passing the min_row_range argument. */
    pub fn update_from_accumulated(
        &mut self,
        w : &dyn AsRef<Window<i32>>,
        min_row_range : Option<usize>,
        min_col_range : Option<usize>
    ) {
        let w = w.as_ref();
        self.rles.clear();
        self.rows.clear();
        let mut valid_rows = Vec::new();
        calculate_valid_row_ranges(&mut valid_rows, w, 0, w.height() - 1, min_row_range.unwrap_or(MIN_VALID_INTEGRAL_ROW_RANGE));
        valid_rows.sort_by(|a, b| a.start.cmp(&b.start) );
        let mut valid_col_ranges = Vec::new();
        for r in valid_rows.drain(..).flatten() {
            find_row_rles(&mut valid_col_ranges, &mut self.rles, &mut self.rows, w, r, min_col_range);
        }
        verify_rle_state(&self);
    }

    pub fn calculate<S>(w : &Image<u8, S>) -> Self 
    where
        S : Storage<u8>
    {
        let mut rle = Self::default();
        rle.update(w);
        rle
    }

    /*
    Builds a runlength from a dense binary image. Pixels that are set to zero are
    not encoded in the RunLenght; pixels that are non-zero are considered part
    of the RunLength.

    Run-length encoding is trivially parallelizable by splitting an image with n
    rows into 4 images with dimensions n/4 x m, running it separately on the images,
    then polling the results back together.
    */
    pub fn update<S>(&mut self, w : &Image<u8, S>) 
    where
        S : Storage<u8>
    {
        self.rles.clear();
        self.rows.clear();
        let mut last_rle : Option<RunLength> = None;
        for r in 0..w.height() {

            let mut curr_range = Range { start : self.rles.len(), end : self.rles.len() };

            update_range(&w.full_window(), r, 0..w.width(), &mut last_rle, &mut curr_range, &mut self.rles);

            // Push any remaining RunLenghts that haven't ended before the last column.
            if let Some(rle) = last_rle.take() {
                self.rles.push(rle);
                curr_range.end += 1;
            }

            // If at least one RunLength has been pushed, add this range to the
            // row count. If no RunLenghts have been pushed, ignore this range.
            if curr_range.end > curr_range.start {
                self.rows.push(curr_range);
            }
        }

        verify_rle_state(&self);
    }

    pub fn new() -> Self {
        Default::default()
    }

    pub fn encode(w : &Window<u8>) -> Self {
        let mut encoding = Self::default();
        encoding.update(w);
        encoding
    }

    // Assuming this RLE is connected, splits this RLE vertically at rows
    // that have a small horizontal overlap, based on a percentage of the largest row
    // that is covered by its overlap with the smallest row. If no rows satisfy the
    // condition, returns a vector with the unique RunLength unchanged.
    pub fn split_vertically(mut self, ratio : f32) -> SmallVec<[RunLengthCode;4]> {
        use crate::shape::HalfOpen;
        let nrows = self.rows.len();
        let mut split_points : SmallVec<[usize; 4]> = SmallVec::new();
        let row_pair_iter = self.rows[0..(nrows-1)].iter()
            .zip(self.rows[1..nrows].iter());
        for (ix, (row_above, row_below)) in row_pair_iter.enumerate() {
            let start_above = self.rles[row_above.clone()].first().unwrap().start.1;
            let start_below = self.rles[row_below.clone()].first().unwrap().start.1;
            let end_above = self.rles[row_above.clone()].last().unwrap().end_col();
            let end_below = self.rles[row_below.clone()].last().unwrap().end_col();
            let range_above = Range { start : start_above, end : end_above };
            let range_below = Range { start : start_below, end : end_below };
            if let Some(overlap) = range_above.overlap(&range_below) {
                let overlap_len = overlap.end - overlap.start;
                let max_len = (range_above.end - range_above.start).max(range_below.end - range_below.start);
                if (overlap_len as f32 / max_len as f32) < ratio {
                    split_points.push(ix + 1);
                }
            }
        }
        let mut dst = SmallVec::new();
        for pt in split_points.drain(..).rev() {
            let mut offset = 0;
            let mut new_rows = Vec::new();
            for len in self.rows[pt..].iter().map(|row| row.end - row.start ) {
                new_rows.push(Range { start : offset, end : offset + len });
                offset += len;
            }
            let new_rles = self.rles.split_off(offset);
            self.rows.truncate(pt-1);
            dst.push(RunLengthCode { rles : new_rles, rows : new_rows });
        }
        dst.push(self);
        dst
    }

    pub fn row_mass(&self) -> Vec<usize> {
        let mut mass = Vec::with_capacity(self.rows.len());
        for r in &self.rows {
            let row_mass = self.rles[r.clone()].iter().fold(0, |m, r| m + r.length );
            mass.push(row_mass);
        }
        mass
    }

    pub fn mass(&self) -> usize {
        self.rles.iter().fold(0, |m, r| m + r.length )
    }

    // Split this RLE into others based on 4-connectivity
    // of the pixels inside them.
    pub fn split(&self) -> BTreeMap<usize, RunLengthCode> {
        let nrows = self.rows.len();
        let mut uf = UnionFind::new(self.rles.len());
        if self.rows.len() == 0 {
            return BTreeMap::new();
        }
        let mut last_matched_above : Option<&RunLength> = None;
        let row_pair_iter = self.rows[0..(nrows-1)].iter()
            .zip(self.rows[1..nrows].iter());

        for (row_above, row_below) in row_pair_iter {
            let row_vert_sep = self.rles[row_below.start].start.0 - self.rles[row_above.start].start.0;

            // If vert sep > 1, there is no way any RLEs in this row pair
            // are connected. skip to next pair.
            if row_vert_sep == 1 {

                // Iterate over cartesian product of top row RLEs and bottom row RLEs
                for (local_ix_below, rl_below) in self.rles[row_below.clone()].iter().enumerate() {

                    // Iterate over top RLEs overlapping with this bottom RLE.
                    // (since RLEs are ordered, there is no
                    // need to check the overlap of intermediate elements:
                    // only the start and end matching RLEs).
                    let iter_above = self.rles[row_above.clone()].iter().enumerate();
                    for (local_ix_above, above) in iter_above {

                        if above.end_col() >= rl_below.start.1 {
                            if above.start.1 <= rl_below.end_col() {
                                uf.union(row_above.start + local_ix_above, row_below.start + local_ix_below);
                            } else {
                                // If above start col > below end col, there is
                                // no point in iterating in the top row for this RLE further.
                                break;
                            }
                        }
                        /*if rl_below.range().intersects(above.range()) {
                            uf.union(row_above.start + local_ix_above, row_below.start + local_ix_below);
                        }*/

                    }
                }
            }
        }

        let mut rlcs = BTreeMap::new();

        // This linear iteration guarantees the RLEs are still in raster order
        // within each connected subset in the connectivity BTreeMap.
        for i in 0..self.rles.len() {
            let parent_ix = uf.find(i);
            rlcs.entry(parent_ix).or_insert(RunLengthCode::default()).rles.push(self.rles[i].clone());
        }

        for (_, rlc) in rlcs.iter_mut() {
            let mut rows = Vec::new();
            let mut offset = 0;
            for (_, gr) in &rlc.rles.iter().group_by(|r| r.start.0 ) {
                let n = gr.count();
                let new_offset = offset + n;
                rows.push(Range { start : offset, end : new_offset });
                offset = new_offset;
            }
            rlc.rows = rows;
        }
        rlcs
    }

    pub fn graph(&self) -> RunLengthGraph {
        let nrows = self.rows.len();

        let mut graph = UnGraph::<RunLength, /*Interval<usize>*/ () >::new_undirected();

        let mut uf = UnionFind::new(self.rles.len());
        if self.rows.len() == 0 {
            return RunLengthGraph { graph, uf };
        }

        // let mut last_matched_above : Option<&RunLength> = None;
        let mut curr_ixs = Vec::new();
        let mut past_ixs = Vec::new();

        // Populate graph initially with first row
        for rl in &self.rles[self.rows[0].clone()] {
            past_ixs.push(graph.add_node(*rl));
        }

        // println!("Intersection = {:?}", Interval(1usize,1usize).intersect(Interval(1usize,1usize)));

        let row_pair_iter = self.rows[0..(nrows-1)].iter()
            .zip(self.rows[1..nrows].iter());
        for (row_above, row_below) in row_pair_iter {

            // TODO verify if row_above and row_below are actually connected to
            // avoid the more costly iteration over the row below.

            for rl_below in &self.rles[row_below.clone()] {

                // Add current bottom RLE
                let below_ix = graph.add_node(*rl_below);
                curr_ixs.push(below_ix);

                /* Decide on the RunLength connectivity strategy: If the strict overlap
                is desired, we are working with 4-neighborhood; If diagonal linking (8-neighborhood)
                is desired, the diagonally-connecting intervals will not share an overlap */

                // Iterate overr overlapping top RLEs (since they are ordered, there is no
                // need to check the overlap of intermediate elements:
                // only the start and end matching RLEs).
                let iter_above = self.rles[row_above.clone()].iter()
                    .enumerate()

                    // use this for diagonally-linking RLEs (CANNOT have intersections as weights)
                    // .skip_while(|(_, above)| (above.start.1+above.length+1 ) < rl_below.start.1 )
                    // .take_while(|(_, above)| above.start.1 < (rl_below.start.1+rl_below.length+1 ) )

                    // Use this for strictly overlapping RLEs (intersections can be weights).
                    .skip_while(|(_, above)| (above.start.1+above.length-1) < rl_below.start.1 )
                    .take_while(|(_, above)| above.start.1 <= (rl_below.start.1+rl_below.length-1) )

                    // Return this to use RLE intervals as edes
                    // .map(|(ix, above)| (ix, rl_below.intersect(&above).unwrap().interval() ) );

                    // Return this for edges without information.
                    .map(|(ix, _above)| ix );

                // Add edges to top RLEs
                // for (above_ix, intv) in iter_above {
                for above_ix in iter_above {
                    graph.add_edge(below_ix, past_ixs[above_ix], /*intv*/ () );
                    uf.union(below_ix, past_ixs[above_ix]);
                }
            }
            mem::swap(&mut past_ixs, &mut curr_ixs);
            curr_ixs.clear();
        }

        RunLengthGraph { graph, uf }
    }

    /// Returns all points from the first and last row
    /// of points. If the first row is the same as the
    /// last row, returns all points of this row.
    pub fn extremal_points(&self) -> Vec<(usize, usize)> {
        let mut pts = Vec::new();
        if let Some(top_row) = self.rows.first() {
            for rle in &self.rles[top_row.clone()] {
                pts.extend(rle.points());
            }
            if let Some(bottom_row) = self.rows.last() {
                if bottom_row != top_row {
                    for rle in &self.rles[bottom_row.clone()] {
                        pts.extend(rle.points());
                    }
                }
            }
        }
        pts
    }

    // Behaves like lateral_points, but ignore rows that have a gap greater than tol.
    pub fn lateral_points_with_tol(&self, tol : usize) -> Vec<(usize, usize)> {
        let mut pts = Vec::with_capacity(self.rows.len() * 2);
        'outer : for row in &self.rows {
            if row.end - row.start == 1 {
                pts.push(self.rles[row.start].start);
                pts.push(self.rles[row.start].end());
            } else {

                // Gap tolerance check: Skip to next row when gap > tol
                let rles = &self.rles[row.clone()];
                'inner : for (left, right) in rles.iter().take(rles.len()-1).zip(rles.iter().skip(1)) {
                    if right.start_col() - left.end_col() > tol {
                        continue 'outer;
                    }
                }

                if let (Some(fst), Some(lst)) = (self.rles[row.clone()].first(), self.rles[row.clone()].last()) {
                    pts.push(fst.start);
                    pts.push(lst.end());
                }
            }
        }
        pts
    }

    // Gets lateral points, ignoring rows with more than a single RLE.
    // This guarantees points are connected.
    pub fn solid_lateral_points(&self) -> Vec<(usize, usize)> {
        let mut pts = Vec::with_capacity(self.rows.len() * 2);
        for row in &self.rows {
            if row.end - row.start == 1 {
                pts.push(self.rles[row.start].start);
                pts.push(self.rles[row.start].end());
            }
        }
        pts
    }

    // If there is more than one RLE per row, get the lateral points of the largest RLE at the row,
    // ignoring the remaining ones. This makes sure the lateral points are connected, BUT parts
    // of a connected object might be ignored.
    pub fn maximal_lateral_points(&self) -> Vec<(usize, usize)> {
        let mut pts = Vec::with_capacity(self.rows.len() * 2);
        for row in &self.rows {
            if let Some(largest) = self.rles[row.clone()].iter().max_by(|a, b| a.length.cmp(&b.length) ) {
                pts.push(largest.start);
                pts.push(largest.end());
            }
        }
        pts
    }

    /// Returns the RLE lateral points (left points occupy even indices,
    /// right points occupy odd indices). If each row contains more than
    /// one RLE, returns the most extreme points at different RLEs. This
    /// means the (start, end) points might not be connected.
    pub fn lateral_points(&self) -> Vec<(usize, usize)> {
        let mut pts = Vec::with_capacity(self.rows.len() * 2);
        for row in &self.rows {
            if let (Some(fst), Some(lst)) = (self.rles[row.clone()].first(), self.rles[row.clone()].last()) {
                pts.push(fst.start);
                pts.push(lst.end());
            }
        }
        pts
    }

    pub fn central_point(&self) -> Option<(usize, usize)> {
        let nr = self.rles.len();
        if nr == 0 {
            return None;
        }
        let mut r = 0;
        let mut c = 0;
        for row in &self.rows {
            if let (Some(fst), Some(lst)) = (self.rles[row.clone()].first(), self.rles[row.clone()].last()) {
                r += fst.start.0;
                c += fst.start.1;
                c += lst.end().1;
            }
        }
        Some((r / nr, c / (2*nr)))
    }

    // Total area represented by the RunLength (sum of all lengths).
    // Calculated in linear time.
    pub fn area(&self) -> usize {
        self.rles.iter().fold(0, |area, rle| area + rle.length )
    }

    // Measures how RLE length varies away from center.
    pub fn convexity(&self) -> Option<f32> {
        let half_f = self.height()? / 2;
        let mut cvx = 0.0;
        for (i, r) in self.rows.iter().enumerate() {
            let row = &self.rles[r.clone()];
            if let (Some(fst), Some(lst)) = (row.first(), row.last()) {
                let row_len = lst.end().1 - fst.start.1;
                cvx += ((row_len * i.abs_diff(half_f)) as f32).ln();
            }
        }
        cvx /= self.height()? as f32;
        Some(cvx)
    }

    pub fn hole_area(&self) -> usize {
        let mut area = 0;
        for r in &self.rows {
            let row = &self.rles[r.clone()];
            for (left, right) in row.iter().take(row.len()-1).zip(row.iter().skip(1)) {
                area += right.start.1 - left.end().1;
            }
        }
        area
    }

    pub fn empty_outer_area(&self, outer_rect : &(usize, usize, usize, usize)) -> usize {
        let last_col = outer_rect.1 + outer_rect.3;
        let mut area = 0;
        for r in &self.rows {
            let row = &self.rles[r.clone()];
            if let Some(fst) = row.first() {
                area += fst.start.1 - outer_rect.1;
            }
            if let Some(lst) = row.last() {
                area += last_col - lst.end().1;
            }
        }
        area
    }

    // Returns the complement of this runlength (with the same number of rows as original.)
    // The rows not encoded at original RLE are ignored (i.e. it is assumed to be an connected RLE).
    pub fn connected_complement(&self, outer_rect : &(usize, usize, usize, usize)) -> RunLengthCode {
        let mut rows = Vec::new();
        let mut rles = Vec::new();
        let tr_row = outer_rect.1 + outer_rect.3;
        for r in &self.rows {
            let n_before = rles.len();
            let curr_rles = &self.rles[r.clone()];
            if let Some(fst) = curr_rles.first() {
                if fst.start.1 > outer_rect.1 {
                    rles.push(RunLength { start : (fst.start.0, 0), length : fst.start.1 - outer_rect.1 });
                }
            }
            for (left, right) in curr_rles.iter().zip(curr_rles.iter().skip(1)) {
                rles.push(RunLength { start : left.end(), length : right.start.1 - left.end().1 });
            }
            if let Some(lst) = curr_rles.last() {
                let tr_rl = lst.start.1+lst.length;
                if tr_rl  < tr_row {
                    rles.push(RunLength { start : (lst.start.0, lst.start.1), length : tr_row - tr_rl });
                }
            }
            rows.push(Range { start : n_before, end : n_before + rles.len() });
        }
        RunLengthCode { rows, rles }
    }

    // pub fn hole_area(&self) -> usize {
    //
    // }

}

// cargo test --lib -- rle_split_vert --nocapture
#[test]
fn rle_split_vert() {
    let mut buf = ImageBuf::<u8>::new_constant(128, 128, 0);
    buf.window_mut((32,32),(32,32)).unwrap().fill(255);
    buf.window_mut((64,32),(32,8)).unwrap().fill(255);
    let mut rle = RunLengthCode::encode(&buf.full_window());
    let splits = rle.split_vertically(0.5);
    for s in splits {
        println!("{:?}", s);
    }
    buf.show();
}

// cargo test --lib -- rle_split --nocapture
#[test]
fn rle_split() {

    use crate::draw::*;
    let mut buf = ImageBuf::<u8>::new_constant(128, 128, 0);
    buf.draw(Mark::Dot((32, 32), 16), 255);
    buf.draw(Mark::Dot((100, 100), 20), 255);
    buf.draw(Mark::Dot((100, 50), 10), 255);
    let mut rle = RunLengthCode::encode(&buf.full_window());

    let mut colors : Vec<_> = (50..255).step_by(40).collect();
    let mut i = 0;
    for (_, rle) in rle.split() {
        for pt in rle.points() {
            buf[pt] = colors[i];
        }
        i += 1;
    }

    buf.show();

}

fn update_range(
    w : &Window<u8>,
    r : usize,
    cols : Range<usize>,
    last_rle : &mut Option<RunLength>,
    curr_range : &mut Range<usize>,
    rles : &mut Vec<RunLength>
) {
    for c in cols {
        if w[(r, c)] == 0 {
            // End current run-length (if any) when match is zero.
            if let Some(rle) = last_rle.take() {
                rles.push(rle);
                curr_range.end += 1;
            }

        } else {
            if let Some(mut rle) = last_rle.as_mut() {
                // Extend current run-length (if any) for a nonzero match.
                rle.length += 1;
            } else {
                // Innaugurate a new run-length if there isn't one already.
                *last_rle = Some(RunLength { start : (r, c), length : 1 });
            }
        }
    }
}

fn verify_rle_state(rle : &RunLengthCode) {

    // Verify each row index at least one valid RLE
    assert!(rle.rows.iter().all(|r| r.end - r.start >= 1 ), "{:?}", rle);

    // Verify the summed length of sub-indices equals the total length.
    assert!(rle.rles.len() == rle.rows.iter()
        .fold(0, |total, r| total + rle.rles[r.clone()].len() ), "{:?}", rle );

    // Verify end of previous range (open) equals start of next range (closed).
    assert!(rle.rows.iter()
        .zip(rle.rows.iter().skip(1)).all(|(a, b)| a.end == b.start ), "{:?}", rle);

}

/*pub fn draw_distinct(graph : &RunLengthGraph, bin : &mut WindowMut<u8>) {
    let split = graph.graph.split();
    for r in split.ranges {
        let v : f64 = rand::random();
        for node in &split.indices[r] {
            for pt in graph.graph[*node].middle_points() {
                bin[pt] = (v*255.) as u8;
            }
        }
    }
}*/

/*pub fn draw_distinct(img : &mut WindowMut<u8>, graph : &UnGraph<RunLength, /*Interval<usize>*/ ()>, split : &SplitGraph) {
    for range in &split.ranges  {
        let r : f64 = rand::random();
        let color : u8 = 64 + ((256. - 64.)*r) as u8;
        for ix in split.indices[range.clone()].iter() {
            img[graph[*ix]].iter_mut().for_each(|px| *px = color );
        }
    }
}*/

/// A graph of connected RunLength elements is much cheaper to represent than a graph of connected pixels,
/// and encode the same set of spatial relationships by making horizontal pixel
/// connections implicit (all pixels represented by a RunLength are connected; horizontal
/// pixels in the same row but different RunLengths are disconnected; vertical RunLengths
/// are connected by graph edges). The weights can represent start column and end column of the
/// relative overlap. RLEs stay ordered in the graph: A graph index B > graph Index A means
/// row(B) >= row(A), and col(B) > col(A). graph.node_indices() therefore also represent a
/// raster order.
#[derive(Debug, Clone)]
pub struct RunLengthGraph {

    // Edges carry the intersection of the RunLenghts.
    pub graph : UnGraph<RunLength, /*Interval<usize>*/ ()>,

    pub uf : UnionFind<NodeIndex>

}

fn extend_extremities(ps : &mut Vec<(usize, usize)>, i : usize) {
    let diff = ps[i].1 as i32 - ps[i-2].1 as i32;

    // Left entries
    if i % 2 == 0 {
        if diff >= 2 {
            let a = ps[i-2].1+1;
            let b = ps[i].1.saturating_sub(1);
            let r = ps[i].0;
            ps.extend((a..b).map(|c| (r, c) ));
        }
        if diff <= -2 {
            let a = ps[i].1+1;
            let b = ps[i-2].1.saturating_sub(1);
            let r = ps[i-2].0;
            ps.extend((a..b).map(|c| (r, c) ));
        }

    // Right entries
    } else {
        if diff >= 2 {
            let a = ps[i].1+1;
            let b = ps[i-2].1.saturating_sub(1);
            let r = ps[i-2].0;
            ps.extend((a..b).map(|c| (r, c) ));
        }
        if diff <= -2 {
            let a = ps[i-2].1+1;
            let b = ps[i].1.saturating_sub(1);
            let r = ps[i].0;
            ps.extend((a..b).map(|c| (r, c) ));
        }
    }
}

/*fn push_rect(
    rect : &mut (usize, usize, usize, usize),
    out : &mut BTreeMap<NodeIndex<u32>, Vec<(usize, usize, usize, usize)>>,
    graph : &UnGraph<RunLength, Interval<usize>>,
    b : NodeIndex<u32>,
    parent_ix : NodeIndex<u32>
) {
    println!("pushed");
    out.entry(parent_ix).or_insert(Vec::new()).push(*rect);
    *rect = graph[b].as_rect();
}*/

/*fn add_finished(
    finished : &mut Vec<NodeIndex<u32>>,
    rects : &mut Vec<(usize, usize, usize, usize)>,
    graph : &UnGraph<RunLength, Interval<usize>>,
) {

    finished.clear();
}*/

// If RLEs are in raster order, the TL of the rect is defined by the first
// rle.as_rect(). The following RLEs might affect the rect only by changes
// made through this function.
fn extend_rect_with_sorted_rle(r: &mut (usize, usize, usize, usize), rle : &RunLength) {
    let intv = rle.interval();

    // Update left
    if intv.0 < r.1 {
        r.3 = (r.1 + r.3) - intv.0;
        r.1 = intv.0;
    }

    // Update width
    let dist_left = intv.1 - r.1;
    if dist_left > r.3 {
        r.3 = dist_left;
    }

    // Update height
    r.2 = (rle.start.0 - r.0);
}

impl RunLengthGraph {

    // Since the graph is in raster order, the index of all labels in the union find distinct
    // from the previous in a serial iterator over the union find candidate top-left
    // element in a new group. Sorting and taking the distinct of those candidates gives
    // the first top-left element. The first index of each group can be used as a handle to start
    // a depth or breadth search over the graph restricted to the current group. Consider using
    // splitgraph::first_nodes instead to avoid calling into_labeling more than once.
    pub fn first_nodes(&self) -> Vec<NodeIndex> {
        let labels = self.uf.clone().into_labeling();
        let mut fst_nodes = Vec::new();
        if labels.len() == 0 {
            return fst_nodes;
        }
        fst_nodes.push(NodeIndex::new(0));
        for (pair_ix, (prev, curr)) in labels.iter().zip(labels.iter().skip(1)).enumerate() {
            if *curr != *prev {
                fst_nodes.push(NodeIndex::new(pair_ix+1));
            }
        }
        fst_nodes.sort();
        fst_nodes.dedup();
        fst_nodes
    }

    // Returns external points in unspecified order.
    pub fn outer_points(
        &self,
        pts : Option<BTreeMap<NodeIndex, Vec<(usize, usize)>>>
    ) -> BTreeMap<NodeIndex, Vec<(usize, usize)>> {
        let mut pts = pts.unwrap_or(BTreeMap::new());
        pts.clear();

        let n = self.graph.raw_nodes().len();
        if n == 0 {
            return pts;
        }

        for ix in self.graph.node_indices() {
            let parent = self.uf.find(ix);
            let rle = &self.graph[ix];
            if let Some(mut this_pts) = pts.get_mut(&parent) {
                let mut last = this_pts.last_mut().unwrap();
                if last.0 == rle.start.0 {
                    // Last is from current row. Just extend right point.
                    last.1 = rle.end().1;

                } else {
                    // Last is from previous row. Add two new points.
                    this_pts.push(rle.start);
                    this_pts.push(rle.end());
                }

            } else {
                pts.insert(parent, vec![rle.start, rle.end()]);
            }
        }

        // Extends top and bottom row with middle points
        for (_, ps) in &mut pts {

            // println!("{:?}", ps);

            // Fill bottom and top rows
            let n_ps = ps.len();
            let fst_row = ps[0].0;
            let fst_col_fst_row = (ps[0].1+1);
            let lst_col_fst_row = (ps[1].1.saturating_sub(1));
            let fst_col_lst_row = (ps[n_ps-2].1+1);
            let lst_col_lst_row = (ps[n_ps-1].1.saturating_sub(1));
            // assert!(ps[0].0 == ps[1].0);
            // assert!(ps[n-2].0 == ps[n-1].0);
            ps.extend((fst_col_fst_row..lst_col_fst_row).map(|c| (fst_row, c) ));
            if n_ps >= 4 {
                let lst_row = ps[n_ps-2].0;
                ps.extend((fst_col_lst_row..lst_col_lst_row).map(|c| (lst_row, c) ));
            }
        }

        // Extends all contiguous intervals that have a small overlap (large gaps)
        for (_, ps) in &mut pts {
            // Iterate over even (left) entries
            for i in (2..ps.len()).step_by(2) {
                extend_extremities(ps, i);
            }

            // Iterate over odd (right) entries
            for i in (3..ps.len()).step_by(2) {
                extend_extremities(ps, i);
            }
        }

        pts
    }

    /// Return a single outer rect per object.
    pub fn outer_rects(
        &self,
        rects : Option<BTreeMap<NodeIndex, (usize, usize, usize, usize)>>
    ) -> BTreeMap<NodeIndex, (usize, usize, usize, usize)> {
        let mut rects = rects.unwrap_or(BTreeMap::new());
        rects.clear();

        for ix in self.graph.node_indices() {
            let parent = self.uf.find(ix);
            let rle = &self.graph[ix];
            if let Some(mut r) = rects.get_mut(&parent) {

                // Since raster order is preserved as we iterate over graph indices,
                // only update the sides and the bottom regions of rect.
                extend_rect_with_sorted_rle(&mut r, &rle);
            } else {
                rects.insert(parent, rle.as_rect());
            }
        }

        // let mut labels = self.uf.clone().into_labeling().iter().unique().count();
        // assert!(rects.len() == labels, "Has {} rects but {} parent labels", rects.len(), labels);

        rects
    }

    /// Returns several rects that are contained in grouped  RunLengths. The rects are found
    /// by a single depth-first search at the run-length encoding graph. The rects are minimally-overlapping
    /// and will have maximum dimension given by the desired arguments, but some of them might be smaller.
    /// It is important to give a maximum height that is much smaller than the expected object size, or
    /// else if the object geometry is too far away from a rect (think a blob or circle) the rects will be
    /// too thin to cover a good proportion of the object area. The maximum width, however, can be whatever
    /// is best for your application. Passing usize::MAX to max_width will make output rects cover the whole object
    /// width when possible. Returned rects are in an unspecified order. The covered proportion of the object
    /// can be found by rects.fold(0., |s, r| s + r.area() ) / rle.area(), which is useful when choosing appropriate
    //// max_height and max_width parameters.
    pub fn inner_rects(
        &self,
        max_height : usize,
        max_width : usize
    ) -> BTreeMap<NodeIndex<u32>, Vec<(usize, usize, usize, usize)>> {
        let split = SplitGraph::new(&self.graph, &self.uf);
        let mut out = BTreeMap::new();
        let mut finished = Vec::new();
        for group in split.ranges.iter() {

            // Used to start the DFS
            let fst_el = split.indices[group.start];

            // Do a depth-first search at the RLE graph. Vertically-neighboring RLES
            // will be grouped together, so the inner rects can be found by simply
            // iterating over neighboring entries if we order the finish event indices into
            // a vector and look for contiguous RLEs. All rects found at this step have the
            // maximum possible width.
            finished.clear();
            petgraph::visit::depth_first_search(&self.graph, [fst_el], |event| {
                match event {
                    DfsEvent::Finish(ix, _) => {
                        finished.push(ix);
                    },
                    _ => { }
                }
            });
            if finished.len() < 2 {
                continue;
            }

            // Rect currently being grown.
            let mut rect : Option<(usize, usize, usize, usize)> = Some(self.graph[finished[0]].as_rect());
            let n = finished.len();
            let mut prev_is_below : Option<bool> = None;
            let parent = self.uf.find(split.indices[group.start]);

            // Holds all rects for a given parent.
            let mut rects = out.entry(parent).or_insert(Vec::new());

            for i in 0..(n-1) {
                let a = finished[i];
                let b = finished[i+1];
                let row_abs_diff = self.graph[b].start.0.abs_diff(self.graph[a].start.0);
                let b_is_contiguous_a = row_abs_diff == 1;
                if b_is_contiguous_a {
                    let mut r = rect.take().unwrap_or(self.graph[a].as_rect());
                    let rect_intv = Interval(r.1, r.1 + r.3);
                    let b_intv = self.graph[b].interval();
                    if let Some(intc) = b_intv.intersect(rect_intv) {
                        let next_is_below = self.graph[b].start.0 > self.graph[a].start.0;
                        if next_is_below && (prev_is_below.is_none() || prev_is_below == Some(true)) {
                            grow_bottom(&mut r, &intc);
                        } else if !next_is_below && (prev_is_below.is_none() || prev_is_below == Some(false)) {
                            grow_top(&mut r, &intc);
                        } else {
                            rects.push(r);
                        };
                        prev_is_below = Some(next_is_below);
                        if r.2 >= max_height {
                            rects.push(r);
                        } else {
                            rect = Some(r);
                        }
                    }
                } else {
                    if let Some(r) = rect.take() {
                        if r.2 > 1 {
                            rects.push(r);
                        }
                    }
                }
            }
            if let Some(r) = rect {
                if r.2 > 1 {
                    rects.push(r);
                }
            }
        }

        // Now, split all found rects respecting the desired maximum width.
        if max_width < usize::MAX {
            for (_, rects) in &mut out {
                let n = rects.len();
                for i in 0..n {
                    if rects[i].3 > max_width {
                        let mut divisor = 2;
                        let mut w = rects[i].3 / divisor;
                        while rects[i].3 / divisor > max_width {
                            divisor += 1;
                            w = rects[i].3 / divisor;
                        }
                        for ix in 1..divisor {
                            let r = (rects[i].0, rects[i].1 + ix*w, rects[i].2, w);
                            rects.push(r);
                        }
                        rects[i].3 = w;
                    }
                }
            }
        }

        out
    }

    pub fn split(&self) -> crate::graph::SplitGraph {
        crate::graph::SplitGraph::new(&self.graph, &self.uf)
    }

}

fn shrink_sides(
    r : &mut (usize, usize, usize, usize),
    intc : &Interval<usize>
) {
    if intc.0 > r.1 {
        r.1 = intc.0;
        r.3 = (r.1 + r.3) - intc.0;
    }

    // Shrink right end of rect. Here, we can split the remainder into another rect.
    if intc.1 < (r.1+r.3) {
        r.3 = intc.1 - r.1;
    }
}

fn grow_bottom(
    r : &mut (usize, usize, usize, usize), intc : &Interval<usize>
) {
    shrink_sides(r, intc);
    r.2 += 1;
    // rems
}

fn grow_top(
    r : &mut (usize, usize, usize, usize), intc : &Interval<usize>
) {
    shrink_sides(r, intc);
    r.0 -= 1;
    r.2 += 1;
}

/*// cargo test --all-features --lib -- inner_rects --nocapture
#[test]
fn inner_rects() {
    use crate::draw::*;
    for path in ["/home/diego/Downloads/pattern-hole.png", "/home/diego/Downloads/pattern.png"] {
        let mut img = crate::io::decode_from_file(path).unwrap();
        let rle = RunLengthEncoding::calculate(img.as_ref());
        let graph = rle.graph();
        let rects = graph.inner_rects(16, 16);
        for (_, rs) in rects {
            for r in rs {
                img.draw(Mark::Rect((r.0, r.1), (r.2, r.3), 255));
            }
        }
        img.show();
    }
}*/

pub enum Knot {

    // Elements are certain to be connected.
    Tight,

    // Elements might be connected, pending an evaluation of
    // the 8-neighborhood.
    Loose

}

// Represents a binary image in terms of rects. The smallest rects are pixels;
// the largest rect is the full image. Always use the largest possible rect to
// represent an object.
pub struct RectEncoder {

    sum_dst : ImageBuf<u8>,

    // Indices of the local sum image.
    graph : UnGraph<(usize, usize), Knot>

}

pub enum RegionCompleteness {

    // A dense region (rect).
    Complete((usize, usize, usize, usize)),

    // A 8-neighborhood anchored at the top-left with some missing pixels.
    Incomplete((usize, usize), Pattern)

}

pub struct RegionEncoding {

    graph : UnGraph<RegionCompleteness, ()>

}

fn update_graph(
    graph : &mut UnGraph<(usize, usize), Knot>,
    last_left : &mut NodeIndex,
    last_tops : &mut [NodeIndex],
    s : &ImageBuf<u8>,
    i : usize,
    j : usize,
    can_be_tight : bool
) {
    let curr_ix = graph.add_node((i, j));
    if i >= 1 {
        if can_be_tight && s[(i-1, j)] >= 7 {
            graph.add_edge(curr_ix, last_tops[j], Knot::Tight);
        } else if s[(i-1, j)] > 0 {
            graph.add_edge(curr_ix, last_tops[j], Knot::Loose);
        }
    }

    if j >= 1 {
        if can_be_tight && s[(i, j-1)] >= 7 {
            graph.add_edge(curr_ix, *last_left, Knot::Tight);
        } else if s[(i, j-1)] > 0 {
            graph.add_edge(curr_ix, *last_left, Knot::Loose);
        }
    }
    *last_left = curr_ix;
    last_tops[j] = curr_ix;
}

impl RectEncoder {

    pub fn new(sz : (usize, usize)) -> Self {
        assert!(sz.0 % 3 == 0 && sz.1 % 3 == 0);
        Self {
            sum_dst : Image::new_constant(sz.0 / 3, sz.1 / 3, 0),
            graph : UnGraph::new_undirected()
        }
    }

    pub fn encode(&mut self, img : &ImageBuf<u8>) {
        crate::local::baseline_local_sum(
            &img.full_window(),
            &mut self.sum_dst.full_window_mut()
        );
        self.graph.clear();
        let s = &self.sum_dst;
        let (sum_height, sum_width) = self.sum_dst.shape();
        let mut last_tops : Vec<_> = (0..sum_width).map(|_| NodeIndex::from(0) ).collect();
        let mut last_left = NodeIndex::from(0);
        for i in 0..sum_height {
            for j in 0..sum_width {
                match s[(i,j)] {
                    0 => {  },
                    1..=6 => {
                        update_graph(&mut self.graph, &mut last_left, &mut last_tops, s, i, j, false);
                    },
                    7.. => {
                        update_graph(&mut self.graph, &mut last_left, &mut last_tops, s, i, j, true);
                    }
                }
            }
        }
    }

}

const FULL_WHITE_PATTERN : Pattern = Pattern {
    center : true,
    neigh : 0b11111111
};

/*const FULL_DARK_PATTERN : Pattern = {
    center : false,
    neigh : 0b00000000
};*/

/// Represents the binary pattern of the 8-neighborhood and the
/// center pixel on/off status. The connectedness of two such neighorhoods
/// can be determined via a single match expression.
#[derive(Debug, Clone, Copy)]
pub struct Pattern {
    pub center : bool,
    pub neigh : u8
}

pub struct PatternCode {

    // 8-bit neighborhood and center patterns
    patterns : Vec<Pattern>,

    // Scale ranges, starting at the original image scale and multiplying
    // by 3 at each iteration.
    scales : Vec<Range<usize>>,

    // Top left position that anchors each pattern (in the original image scale)
    positions : Vec<(usize, usize)>,
}

pub struct PatternEncoder {
    sums : Vec<ImageBuf<u8>>,
    binaries : Vec<ImageBuf<u8>>
}

const PX_EVALUATED : u8 = 10;

fn eval_to_next_binary_or_pattern(
    patterns : &mut Vec<Pattern>,
    scale_ranges : &mut Vec<Range<usize>>,
    positions : &mut Vec<(usize, usize)>,
    next_bin : &mut WindowMut<u8>,
    curr_bin : &Window<u8>,
    curr_sum : &Window<u8>,
) {
    assert!(curr_sum.width() * 3 == curr_bin.width());
    assert!(next_bin.width() == curr_sum.width());
    for i in 0..curr_sum.height() {
        for j in 0..curr_sum.width() {
            match curr_sum[(i, j)] {
                0 => {
                    // Write to be evaluated at the binary image of next level.
                    next_bin[(i, j)] = 0;
                },
                9 => {
                    // Write to be evaluated at the binary image of next level.
                    next_bin[(i, j)] = 1;
                },
                s => {
                    let offset = (i*3, j*3);
                    if s <= 8 {
                        // This means all iterations in the neighborhood were
                        // hit by 0 or 1 (sum=0 or sum=9). Therefore, evaluate
                        // the full pattern at the current level.
                        next_bin[(i, j)] = PX_EVALUATED;
                        patterns.push(eval_pattern(curr_bin, offset, s));
                        positions.push(offset);
                    } else {
                        // This means that at least one pixel received the PX_EVALUATED
                        // constant (10), therefore the remaining pixels that matched 1
                        // must be inserted at the  previous level as a full white pattern.
                        for k in 0..3 {
                            for l in 0..3 {
                                if curr_bin[(offset.0+k, offset.1+l)] == 1 {
                                    if let Some(mut r) = scale_ranges.last_mut() {
                                        patterns.push(FULL_WHITE_PATTERN);
                                        let last = patterns.len()-1;
                                        patterns.swap(r.end, last);
                                        positions.push(offset);
                                        positions.swap(r.end, last);
                                        r.end += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn px_at_pos(w : &Window<u8>, tl : (usize, usize), pos : u8) -> u8 {
    w[offset_at_pos(tl, pos)]
}

fn white_at_pos(pos : u8) -> u8 {
    /*match pos {
        0 => 0b10000000,
        1 => 0b01000000,
        2 => 0b00100000,
        3 => 0b00010000,
        4 => 0b00001000,
        5 => 0b00000100,
        6 => 0b00000010,
        7 => 0b00000001,
        _ => panic!()
    }*/
    0b10000000u8 >> pos
}

fn black_at_pos(pos : u8) -> u8 {
    /*match pos {
        0 => 0b01111111,
        1 => 0b10111111,
        2 => 0b11011111,
        3 => 0b11101111,
        4 => 0b11110111,
        5 => 0b11111011,
        6 => 0b11111101,
        7 => 0b11111110,
        _ => panic!()
    }*/
    (0b01111111u8).rotate_right(pos as u32)
}

fn linear_offset_at_pos(linear_tl : usize, width : usize, pos : u8) -> usize {
    match pos {
        0 => linear_tl,
        1 => linear_tl+1,
        2 => linear_tl+2,
        3 => linear_tl+width+2,
        4 => linear_tl+2*width+2,
        5 => linear_tl+2*width+1,
        6 => linear_tl+2*width,
        7 => linear_tl+width,
        _ => panic!()
    }
}

fn offset_at_pos(tl : (usize, usize), pos : u8) -> (usize, usize) {
    match pos {
        0 => tl,
        1 => (tl.0, tl.1+1),
        2 => (tl.0, tl.1+2),
        3 => (tl.0+1, tl.1+2),
        4 => (tl.0+2, tl.1+2),
        5 => (tl.0+2, tl.1+1),
        6 => (tl.0+2, tl.1),
        7 => (tl.0+1, tl.1),
        _ => panic!()
    }
}

fn eval_next_white(w : &Window<u8>, tl : (usize, usize), pos : u8, neigh : u8, sum : u8) -> u8 {
    if sum == 0 {
        neigh
    } else {
        if px_at_pos(w, tl, pos) == 1 {
            eval_next_white(w, tl, pos+1, neigh | white_at_pos(pos), sum-1)
        } else {
            eval_next_white(w, tl, pos+1, neigh, sum)
        }
    }
}

fn eval_next_black(w : &Window<u8>, tl : (usize, usize), pos : u8, neigh : u8, sum : u8) -> u8 {
    if sum == 0 {
        neigh
    } else {
        if px_at_pos(w, tl, pos) == 0 {
            eval_next_black(w, tl, pos+1, neigh | black_at_pos(pos), sum-1)
        } else {
            eval_next_black(w, tl, pos+1, neigh, sum)
        }
    }
}

fn eval_pattern(w : &Window<u8>, tl : (usize, usize), sum : u8) -> Pattern {
    let center = w[(tl.0+1, tl.1+1)] == 1;
    if center {
        if sum <= 4 {
            Pattern { center, neigh : eval_next_white(w, tl, 0, 0b00000000, sum - 1) }
        } else {
            Pattern { center, neigh : eval_next_black(w, tl, 0, 0b11111111, 8 - sum - 1) }
        }
    } else {
        if sum <= 4 {
            Pattern { center, neigh : eval_next_white(w, tl, 0, 0b00000000, sum) }
        } else {
            Pattern { center, neigh : eval_next_black(w, tl, 0, 0b11111111, 8 - sum) }
        }
    }
}

impl PatternEncoder {

    pub fn new(mut shape : (usize, usize)) -> Self {
        assert!(shape.0 % 3 == 0 && shape.1 % 3 == 0);
        let mut sums = Vec::new();
        while shape.0 >= 3 {
            shape.0 /= 3;
            shape.1 /= 3;
            sums.push(Image::new_constant(shape.0, shape.1, 0));
        }
        // The first binary is actually the received image, so we
        // need one less binary than sum images.
        let binaries = sums.clone();
        Self { sums, binaries }
    }
    
    pub fn encode(&mut self, img : &Window<u8>) -> PatternCode {
        let mut patterns = Vec::new();
        let mut scales = Vec::new();
        let mut positions = Vec::new();
        for lvl in 0..self.sums.len() {

            // Calculate sum over previous binary image (the 0th sum
            // is the original image).
            if lvl == 0 {
                crate::local::baseline_local_sum(
                    &img,
                    &mut self.sums[0].full_window_mut()
                );
            } else {
                crate::local::baseline_local_sum(
                    &self.binaries[lvl-1].full_window(),
                    &mut self.sums[lvl].full_window_mut()
                );
            }

            let mut next_binary = mem::take(&mut self.binaries[lvl]);
            let curr_binary = if lvl == 0 {
                img.clone()
            } else {
                self.binaries[lvl-1].full_window()
            };

            let size_before = patterns.len();
            eval_to_next_binary_or_pattern(
                &mut patterns,
                &mut scales,
                &mut positions,
                &mut next_binary.full_window_mut(),

                &curr_binary,
                &self.sums[lvl].full_window()
            );
            self.binaries[lvl] = next_binary;

            let scale_range = Range { start : size_before, end : patterns.len() };
            scales.push(scale_range);

            // Empty scale range migt mean ALL patterns might be upgraded to
            // the next level, and does not mean level is empty.
            /*if scale_range.is_empty() {
                // This means all pixels were 0 at this level, so there
                // is no point in iterating on any remaining coarser scales.
                return;
            } else {
                scales.push(scale_range);
            }*/

        }
        PatternCode { patterns, scales, positions }
    }

}

#[test]
fn shl() {
    // println!("{:#08b} {:#08b}", ONE_TL, ONE_T);
}

/*const fn shr_by<const N : u8>(a : u8) -> u8 {
    a >> N
}

const fn shl_by<const N : u8>(a : u8) -> u8 {
    a << N
}

const fn bitor(a : u8, b : u8) -> u8 {
    a | b
}

pub const ZERO : u8 = 0b00000000;

pub const ONE_TL : u8 = 0b10000000;
pub const ONE_T : u8 = ONE_TL.rotate_right(1);
pub const ONE_TR : u8 = ONE_TL.rotate_right(2);
pub const ONE_R : u8 = ONE_TL.rotate_right(3);
pub const ONE_BR : u8 = ONE_TL.rotate_right(4);
pub const ONE_B : u8 = ONE_TL.rotate_right(5);
pub const ONE_BL : u8 = ONE_TL.rotate_right(6);
pub const ONE_L : u8 = ONE_TL.rotate_right(7);

// Two values close together.
pub const TWO_TL_T : u8 = 0b11000000;
pub const TWO_T_TR : u8 = TWO_TL_T.rotate_right(1);
pub const TWO_TR_R : u8 = TWO_TL_T.rotate_right(2);
pub const TWO_R_BR : u8 = TWO_TL_T.rotate_right(3);
pub const TWO_BR_B : u8 = TWO_TL_T.rotate_right(4);
pub const TWO_B_BL : u8 = TWO_TL_T.rotate_right(5);
pub const TWO_BL_L : u8 = TWO_TL_T.rotate_right(6);
pub const TWO_L_TL : u8 = TWO_TL_T.rotate_right(7);

// Two values separated by one 0.
pub const TWO_S1_TL_TR : u8 = 0b10100000;
pub const TWO_S1_T_R : u8 = TWO_S1_TL_TR.rotate_right(1);
pub const TWO_S1_TR_BR : u8 = TWO_S1_TL_TR.rotate_right(2);
pub const TWO_S1_R_B : u8 = TWO_S1_TL_TR.rotate_right(3);
pub const TWO_S1_BR_B : u8 = TWO_S1_TL_TR.rotate_right(4);
pub const TWO_S1_B_BL : u8 = TWO_S1_TL_TR.rotate_right(5);
pub const TWO_S1_BL_L : u8 = TWO_S1_TL_TR.rotate_right(6);
pub const TWO_S1_L_TL : u8 = TWO_S1_TL_TR.rotate_right(7);*/

// The values w/ 5 are the bit neg of 3; values w/7 are bitneg of 2
// values with/ 8 is unique. 3 and 4 are the hard ones.

/*// Two values separated by two 0s.
pub const TWO_S2_TL_TR : u8 = 0b100100000;
pub const TWO_S2_T_R : u8 = TWO_S1_TL_TR.rotate_right(1);
pub const TWO_S2_TR_BR : u8 = TWO_S1_TL_TR.rotate_right(2);
pub const TWO_S2_R_B : u8 = TWO_S1_TL_TR.rotate_right(3);
pub const TWO_S2_BR_B : u8 = TWO_S1_TL_TR.rotate_right(4);
pub const TWO_S2_B_BL : u8 = TWO_S1_TL_TR.rotate_right(5);
pub const TWO_S2_BL_L : u8 = TWO_S1_TL_TR.rotate_right(6);
pub const TWO_S2_L_TL : u8 = TWO_S1_TL_TR.rotate_right(7);*/










