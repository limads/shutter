use crate::image::*;
use std::iter::FromIterator;
use std::mem;
use std::cmp::{Ord, Ordering};
use num_traits::AsPrimitive;
use std::ops::Range;
use bayes::calc::rank::Rank;
use bayes::calc::running::Accumulated;

// For a histogram marginalized over two conditionals, the mean of
// the marginal is the weighted average over the mean of the conditionals.
// simply taking the histogram mean assumes the weights of the averages are the
// same. If the weights are not the same, the mean will be biased towards the
// conditional with more weight.

// The computational cost of calculatking m x n histograms over the image
// is the same as calculating the full image histogram, the only difference
// is that the memory cost is much higher.
#[derive(Debug, Clone)]
pub struct GridHistogram {
    pub hists : Vec<GrayHistogram>,
    pub rows : Vec<Range<usize>>,
}

impl GridHistogram {

    pub fn new(nrow : usize, ncol : usize) -> Self {
        let mut rows = Vec::new();
        let mut hists = Vec::new();
        let mut n = 0;
        for r in 0..nrow {
            rows.push(Range { start : n, end : n + ncol });
            n += ncol;
            for c in 0..ncol {
                hists.push(GrayHistogram::new());
            }
        }
        Self { hists, rows }
    }

    pub fn update(&mut self, w : &Window<u8>) {
        let ncol = self.rows[0].end - self.rows[0].start;
        let nrow = self.rows.len();
        let mut n = 0;
        for sub_w in w.windows((w.height() / nrow, w.width() / ncol)) {
            self.hists[n].update(&sub_w);
            n += 1;
        }
    }

    pub fn calculate(w : &Window<u8>, nrow : usize, ncol : usize) -> Self {
        let mut grid = Self::new(nrow, ncol);
        grid.update(w);
        grid
    }

    pub fn at(&self, pos : (usize, usize)) -> Option<&GrayHistogram> {
        let ncol = self.rows[0].end - self.rows[0].start;
        self.hists.get(pos.0 * ncol + pos.1)
    }

    pub fn at_mut(&mut self, pos : (usize, usize)) -> Option<&mut GrayHistogram> {
        let ncol = self.rows[0].end - self.rows[0].start;
        self.hists.get_mut(pos.0 * ncol + pos.1)
    }

}

#[derive(Debug, Clone)]
pub struct GrayHistogram([u32; 256]);

pub fn smoothen(hist : &mut GrayHistogram, interval : usize) {
    assert!(interval % 2 == 0);
    let half_int = (interval / 2);
    let coef = 1. / interval as f64;
    for i in (half_int..(256-half_int)) {
        let mut accum = 0.;
        for j in (i-half_int)..(i+half_int) {
            accum += coef * hist.0[j] as f64;
        }
        hist.0[i] = accum as u32;
    }
}

fn is_strict_maximum<T>(s : &[T], pos : usize, window : usize) -> bool
where
    T : Ord
{

    // Cannot establish window
    if pos < 1 {
        return false;
    }

    let mut left = pos-1;
    let mut right = pos+1;

    // Not local maximum
    if s[pos] < s[left] || s[pos] < s[right] {
        return false;
    }

    while right - left <= window && left > 0 && right < s.len()-1 {
        if s[left] < s[left-1] || s[right] < s[right+1] {
            return false;
        }
        left -= 1;
        right += 1;
    }

    true
}

pub fn peaks<T>(s : &[T], min_width : usize, min_peak_height : T) -> Vec<usize>
where
    T : Ord + Copy
{
    assert!(s.len() % min_width == 0);

    // The nice property for the algorithm that looks for local maxima at the peak minimum width is
    // that we can guarantee peaks located two chunks apart will always be valid candidates,
    // so we can remove peaks one-chunk apart if they are two close by compairing
    // only the contiguous pairs.

    let labeled : Vec<(usize, T)> = s.iter().copied().enumerate().collect();
    let mut peaks : Vec<_> = labeled
        .chunks(min_width)
        .map(|s| s.iter().max_by(|a, b| a.1.cmp(&b.1) ).copied().unwrap() )
        .filter(|p| p.1 > min_peak_height )
        .collect();
    if peaks.len() == 0 {
        return Vec::new();
    }

    let mut ix = peaks.len()-1;
    while ix > 0 {
        if peaks[ix].0 - peaks[ix-1].0 < min_width {
            if peaks[ix].1 > peaks[ix-1].1 {
                peaks.swap(ix, ix-1);
            }
            peaks.remove(ix);
            ix = ix.saturating_sub(2);
        } else {
            ix -= 1;
        }
    }

    peaks.drain(..).map(|p| p.0 ).collect()
}

// Returns triplets (lim_min, peak, lim_max) given the result of peaks_and_valleys.
pub fn delimited_peaks(peaks_valleys : &[(usize, usize, usize)]) -> Vec<(usize, usize, usize)> {
    let mut out = Vec::new();
    match peaks_valleys.len() {
        0 => { },
        1 => {
            out.push((0, peaks_valleys[0].0, peaks_valleys[0].1));
            out.push((peaks_valleys[0].1, peaks_valleys[0].2, 255));
        },
        n => {
            out.push((0, peaks_valleys[0].0, peaks_valleys[0].1));
            for ix in 0..peaks_valleys.len() {
                if ix > 0 {
                    out.push((peaks_valleys[ix-1].2, peaks_valleys[ix].0, peaks_valleys[ix].1));
                }
                if ix < n-1 {
                    out.push((peaks_valleys[ix].1, peaks_valleys[ix].2, peaks_valleys[ix+1].0));
                }
            }
            out.push((peaks_valleys[n-1].1, peaks_valleys[n-1].2, 255));
        }
    }
    out
}

// Return triplets (peak, valley, peak). Returns empty vector for N<2 peaks.
pub fn peaks_and_valleys<T>(
    s : &[T],
    min_width : usize,
    min_peak_height : T,
    max_valley_height : T,
    min_peak_valley_diff : i64
) -> Vec<(usize, usize, usize)>
where
    T : Ord + Copy,
    i64 : From<T>
{
    let peaks = peaks(s, min_width, min_peak_height);
    if peaks.len() < 2 {
        return Vec::new();
    }

    let mut triplets = Vec::new();
    let mut curr_mins : Vec<(usize, T)> = Vec::new();
    for (p1, p2) in peaks[0..(peaks.len()-1)].iter().zip(peaks[1..peaks.len()].iter()) {

        // Take minimum value. If there are multiple minimums w/ same value,
        // take the one closest to the midpoint between p1 and p2.
        curr_mins.clear();
        curr_mins.push((0, s[*p1]));
        for (local_ix, val) in s[(*p1+1)..*p2].iter().enumerate() {
            if *val < curr_mins[0].1 && *val <= max_valley_height &&
                (i64::from(*val) - i64::from(s[*p1])).abs() < min_peak_valley_diff &&
                (i64::from(*val) - i64::from(s[*p2])).abs() < min_peak_valley_diff
            {
                curr_mins.clear();
                curr_mins.push((local_ix + 1, *val));
            } else if *val == curr_mins[0].1 {
                curr_mins.push((local_ix + 1, *val));
            }
        }
        let half_len = ((p2 - p1) / 2) as f32;
        let min = curr_mins.iter()
            .min_by(|a, b| (a.0 as f32 - half_len).abs().partial_cmp(&(b.0 as f32 - half_len).abs()).unwrap_or(Ordering::Equal) )
            .copied()
            .unwrap();
        let valley = *p1 + min.0;
        triplets.push((*p1, valley, *p2));
    }
    triplets
}

impl GrayHistogram {

    pub fn accumulate(&self) -> Vec<u32> {
        bayes::calc::running::cumulative_sum(self.0.iter().copied()).collect()
    }

    pub fn new() -> Self {
        Self([0; 256])
    }

    pub fn mean(&self) -> u8 {
        bayes::approx::mean_for_hist(&self.0) as u8
    }

    pub fn median(&self) -> u8 {
        let acc = self.accumulate();
        bayes::approx::median_for_accum_hist(&acc).val as u8
    }

    /*pub fn local_maxima(&self, interval_len : usize) -> Vec<(usize, u32)> {
        self.0.chunks(interval_len)
            .enumerate()
            .map(|(chunk_ix, chunk)| chunk.iter().enumerate().max_by(|a, b| a.1.cmp(&b.1).unwrap() )
            .map(|(chunk_ix, (ix, val))| (chunk_ix*interval_len+ix, val) )
            .collect()
    }*/

    // pub fn valleys(&self, interval_len : usize, same_thr : i32) -> Vec<usize> {
    // }

    pub fn show(&self, shape : (usize, usize)) {
        let mut img = Image::new_constant(shape.0, shape.1, 0);
        self.draw(&mut img.full_window_mut(), 255);
        img.show();
    }

    pub fn draw(&self, win : &mut WindowMut<'_, u8>, color : u8) {
        assert!(win.width() % 256 == 0);
        assert!(win.width() >= 256);
        let max = self.0.iter().copied().max().unwrap() as f32;
        let col_w = win.width() / 256;
        for ix in 0..256 {
            let h = ((self.0[ix] as f32 / max) * win.height() as f32) as usize;
            let h_compl = win.height() - h;
            if h > 0 {
                win.apply_to_sub((h_compl, ix*col_w), (h, col_w), |mut w| { w.fill(color); } );
            }
        }
    }

    pub fn as_slice(&self) -> &[u32] {
        &self.0[..]
    }

    pub fn as_mut_slice(&mut self) -> &mut [u32] {
        &mut self.0[..]
    }

    pub fn iter_as<'a, T>(&'a self) -> impl Iterator<Item=T> + 'a
    where
        u32 : AsPrimitive<T>,
        T : Copy + 'static
    {
        self.0.iter().map(move |u| { let v : T = u.as_(); v })
    }

    pub fn update(&mut self, win : &Window<'_, u8>) {

        // Maybe IPP already does this step?
        self.0.iter_mut().for_each(|bin| *bin = 0 );

        #[cfg(feature="ipp")]
        unsafe {
            let mut n_levels = 256;
            let n_channels = 1;
            let (step, sz) = crate::image::ipputils::step_and_size_for_window(&win);
            let dtype = crate::foreign::ipp::ippi::IppDataType_ipp8u;
            let uniform_step = 1;
            let mut spec_sz = 0;
            let mut buf_sz = 0;
            let ans = crate::foreign::ipp::ippi::ippiHistogramGetBufferSize(
                dtype,
                sz,
                &n_levels as *const _,
                n_channels,
                uniform_step,
                &mut spec_sz as *mut _,
                &mut buf_sz as *mut _
            );
            assert!(ans == 0);
            let mut spec = Vec::from_iter((0..spec_sz).map(|_| 0u8 ));
            let mut hist_buffer = Vec::from_iter((0..buf_sz).map(|_| 0u8 ));

            let mut lower_lvl : f32 = 0.0;
            let mut upper_lvl : f32 = 255.0;
            let ans = crate::foreign::ipp::ippi::ippiHistogramUniformInit(
                dtype,
                &mut lower_lvl as *mut _,
                &mut upper_lvl as *mut _,
                &mut n_levels as *mut _,
                n_channels,
                mem::transmute(spec.as_mut_ptr())
            );
            assert!(ans == 0);

            let ans = crate::foreign::ipp::ippi::ippiHistogram_8u_C1R(
                win.as_ptr(),
                step,
                sz,
                self.0.as_mut_ptr(),
                mem::transmute(spec.as_ptr()),
                hist_buffer.as_mut_ptr()
            );
            assert!(ans == 0);
            // let ans = crate::foreign::ipp::ippi::ippiHistogramGetLevels(spec.as_ptr(), )
            return;
        }

        unimplemented!()
    }

    pub fn calculate(win : &Window<'_, u8>) -> Self {
        // TODO Actually, hist is n_levels - 1 = 255, since position ix means a count of pixels w/ intensity <= px.
        let mut hist = Self([0; 256]);
        hist.update(win);
        hist
   }

}


