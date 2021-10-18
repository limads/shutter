use crate::image::Window;
use std::iter::FromIterator;
use std::collections::HashMap;
use std::ops::Range;

/*IppStatus ippiHistogram_<mod>(const Ipp<dataType>* pSrc, int srcStep, IppiSize roiSize,
Ipp32u* pHist, const IppiHistogramSpec* pSpec, Ipp8u* pBuffer );*/

pub trait ColorHistogram {

}

/// Only hold color values for pixels present in the image.
pub struct SparseHistogram(HashMap<u8, usize>);

impl SparseHistogram {

    pub fn calculate(win : &Window<'_, u8>, spacing : usize) -> Self {
        let mut hist : HashMap<u8, usize> = HashMap::new();
        for px in win.pixels(spacing) {
            if let Some(mut px) = hist.get_mut(px) {
                *px += 1;
            } else {
                hist.insert(*px, 1);
            }
        }
        Self(hist)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ColorMode {
    pub color : u8,
    pub n_pxs : usize
}

/// Calculates the intensity histogram of the image read.
/// Always return a 256-sized vector of bit values.
#[derive(Debug, Clone)]
pub struct DenseHistogram([usize; 256]);

impl DenseHistogram {

    pub fn calculate(win : &Window<'_, u8>, spacing : usize) -> Self {
        let mut hist = [0; 256];
        for px in win.pixels(spacing) {
            hist[*px as usize] += 1;
        }
        Self(hist)
    }

    /// Returns the histogram modes. The vector contains at most [required] modes,
    /// but might contain fewer than than if there is no pixel that satisfies
    /// the spacing constraints.
    pub fn modes(&self, required : usize, min_space : usize) -> Vec<ColorMode> {
        let mut modes = Vec::new();
        next_mode(&mut modes, &self.0[..], required, min_space);
        modes.drain(..).map(|ix| ColorMode { n_pxs : self.0[ix], color : ix as u8 }).collect()
    }
}

fn find_mode_at_range(
    max : &mut (usize, usize),
    found : &mut bool,
    vals : &[usize],
    range : &Range<usize>,
    min_space : usize
) {
    if let Some((ix, m)) = vals[range.clone()].iter().enumerate().max_by(|a, b| a.1.cmp(&b.1) ) {
        // TODO iterate only over region within [start+min_space, end-min_space].
        if *m > max.1 && (range.start == 0 || ix > min_space) && (range.end == 256 || (range.end - range.start).saturating_sub(ix) > min_space) {
            *max = (range.start + ix, *m);
            *found = true;
        }
    }
}

fn next_mode(modes : &mut Vec<usize>, vals : &[usize], mut required : usize, min_space : usize) {
    let mut max = (0, usize::MIN);
    let n = modes.len();
    let mut found_mode = false;
    match n {
        0 => {
            find_mode_at_range(&mut max, &mut found_mode, vals, &(0..256), min_space);
        },
        1 => {
            find_mode_at_range(&mut max, &mut found_mode, vals, &(0..modes[0]), min_space);
            find_mode_at_range(&mut max, &mut found_mode, vals, &(modes[0].saturating_add(1)..256), min_space);
        },
        _ => {
            let fst_range_arr = [0..modes[0]];
            let last_range_arr = [modes[n-1].saturating_add(1)..256];
            let fst_range = fst_range_arr.into_iter().cloned();
            let found_pairs = modes.iter().take(n-1)
                .zip(modes.iter().skip(1))
                .map(|(a, b)| (a.saturating_add(1)..*b) );
            let last_range = last_range_arr.into_iter().cloned();
            for range in fst_range.chain(found_pairs).chain(last_range) {
                find_mode_at_range(&mut max, &mut found_mode, vals, &range, min_space);
            }
        }
    }
    if found_mode {
        modes.push(max.0);
        modes.sort();
    } else {
        required -= 1;
    }
    if modes.len() < required {
        next_mode(modes, vals, required, min_space);
    }
}

#[test]
fn find_modes() {
    use crate::image::Image;
    let a = Image::new_checkerboard(8, 2);
    let hist = DenseHistogram::calculate(&a.full_window(), 1);
    println!("{:?}", a);
    println!("{:?}", hist);
    println!("{:?}", hist.modes(2, 2));
}

pub struct ColorStats {
    pub avg : u8,
    pub absdev : u8,
    pub min : u8,
    pub max : u8
}

impl ColorStats {

    pub fn calculate(win : &Window<'_, u8>, spacing : usize) -> Option<Self> {
        let mut avg : u64 = 0;

        let mut min = u8::MAX;
        let mut max = u8::MIN;
        let mut n_px = 0;
        for px in win.pixels(spacing) {
            avg += *px as u64;
            n_px += 1;
            if *px < min {
                min = *px;
            }
            if *px > max {
                max = *px;
            }
        }
        if n_px >= 1 {
            let avg = (avg / n_px as u64) as u8;

            let mut absdev : u64 = 0;
            for px in win.pixels(spacing) {
                absdev += (*px as i16 - avg as i16).abs() as u64;
            }
            let absdev = (absdev / n_px) as u8;
            Some(ColorStats { avg, absdev, min, max })
        } else {
            None
        }
    }

}
