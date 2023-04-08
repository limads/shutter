use crate::image::*;
use std::iter::FromIterator;
use std::mem;
use std::cmp::{Ord, Ordering};
use num_traits::AsPrimitive;
use std::ops::Range;
use bayes::calc::rank::Rank;
use bayes::calc::running::Accumulated;

// TODO perhaps use space = "0.18.0"

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
pub struct GrayHistogram {
    hist : [u32; 256],
    
    /* A nice property of an image histogram is that the sum of its values is simply the
    product of image dimensions, therefore total and partial probability calculations can be done
    in a single pass. */
    sum : u32
    
}

pub fn smoothen(hist : &mut GrayHistogram, interval : usize) {
    assert!(interval % 2 == 0);
    let half_int = (interval / 2);
    let coef = 1. / interval as f64;
    for i in (half_int..(256-half_int)) {
        let mut accum = 0.;
        for j in (i-half_int)..(i+half_int) {
            accum += coef * hist.hist[j] as f64;
        }
        hist.hist[i] = accum as u32;
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

    pub fn probabilities(&self) -> Vec<f64> {
        let sum = self.sum as f64;
        self.hist.iter().map(|f| *f as f64 / sum ).collect()
    }
    
    pub fn accumulate_inplace(&mut self) {
        for i in 1..256 {
            self.hist[i] += self.hist[i-1];
            // assert!(self.hist[i] >= self.hist[i-1]);
        }
        self.sum = self.hist[255];
        // assert!(self.hist[255] == self.sum, "{} vs. {}", self.hist[255], self.sum);
    }

    pub fn quantile_when_accumulated(&self, q : f32) -> u8 {
        let target = (self.hist[255] as f32 * q) as u32;
        self.hist.partition_point(|px| *px < target ) as u8

        /*for i in 0..256 {
            if self.hist[i] >= target {
                return i as u8;
            }
        }
        255 as u8*/
    }

    pub fn accumulate(&self) -> Vec<u32> {
        bayes::calc::running::cumulative_sum(self.hist.iter().copied()).collect()
    }
    
    // Accumulates histogram values until a given probability is reached. Iterates
    // the histogram only up to the desired point.
    pub fn quantile(&self, q : f64) -> u8 {
        let sum = self.sum as f64;
        let mut p = 0.0;
        for ix in 0..=255 {
            p += self.hist[ix] as f64 / sum;
            if p >= q {
                return ix as u8;
            }
        }
        255
    }

    pub fn new() -> Self {
        Self { hist : [0; 256], sum : 0 }
    }

    pub fn mean(&self) -> u8 {
        bayes::approx::mean_for_hist(&self.hist) as u8
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
        let max = self.hist.iter().copied().max().unwrap() as f32;
        let col_w = win.width() / 256;
        for ix in 0..256 {
            let h = ((self.hist[ix] as f32 / max) * win.height() as f32) as usize;
            let h_compl = win.height() - h;
            if h > 0 {
                win.apply_to_sub((h_compl, ix*col_w), (h, col_w), |mut w| { w.fill(color); } );
            }
        }
    }

    pub fn as_slice(&self) -> &[u32] {
        &self.hist[..]
    }

    pub fn as_mut_slice(&mut self) -> &mut [u32] {
        &mut self.hist[..]
    }

    pub fn iter_as<'a, T>(&'a self) -> impl Iterator<Item=T> + 'a
    where
        u32 : AsPrimitive<T>,
        T : Copy + 'static
    {
        self.hist.iter().map(move |u| { let v : T = u.as_(); v })
    }

    pub fn update(&mut self, win : &Window<'_, u8>) {

        self.sum = (win.height() * win.width()) as u32;

        #[cfg(feature="ipp")]
        unsafe {
            let mut h = IppHistogram::new(win.height(), win.width());
            h.update(win);
            self.hist.copy_from_slice(&h.hist[..]);

            // assert!(ans == 0);
            // let ans = crate::foreign::ipp::ippi::ippiHistogramGetLevels(spec.as_ptr(), )
            return;
        }

        unimplemented!()
    }

    pub fn calculate(win : &Window<'_, u8>) -> Self {
        // TODO Actually, hist is n_levels - 1 = 255, since position ix means a count of pixels w/ intensity <= px.
        let mut hist = Self { hist : [0; 256], sum : 0 };
        hist.update(win);
        hist
   }

}

#[derive(Clone, Debug)]
#[cfg(feature="ipp")]
pub struct IppHistogram {
    spec : Vec<u8>,
    pub hist : [u32; 256],
    pub hist_buffer : Vec<u8>,
    sum : u32
}

#[cfg(feature="ipp")]
impl IppHistogram {

    pub fn nonzero(&self) -> Vec<usize> {
        let mut nz = Vec::with_capacity(32);
        for i in 0..256 {
            if self.hist[i] > 0 {
                nz.push(i);
            }
        }
        nz
    }

    pub fn as_slice(&self) -> &[u32] {
        &self.hist[..]
    }

    pub fn draw(&self, win : &mut WindowMut<'_, u8>, color : u8) {
        assert!(win.width() % 256 == 0);
        assert!(win.width() >= 256);
        let max = self.hist.iter().copied().max().unwrap() as f32;
        let col_w = win.width() / 256;
        for ix in 0..256 {
            let h = ((self.hist[ix] as f32 / max) * win.height() as f32) as usize;
            let h_compl = win.height() - h;
            if h > 0 {
                win.apply_to_sub((h_compl, ix*col_w), (h, col_w), |mut w| { w.fill(color); } );
            }
        }
    }

    pub fn show(&self, shape : (usize, usize)) {
        let mut img = Image::new_constant(shape.0, shape.1, 0);
        self.draw(&mut img.full_window_mut(), 255);
        img.show();
    }

    pub fn inverse_quantile(&self, q : f64) -> u8 {
        let sum = self.sum as f64;
        let mut p = 1.0;
        for ix in (0..=255).rev() {
            p -= self.hist[ix] as f64 / sum;
            if p <= q {
                return ix as u8;
            }
        }
        0
    }

    pub fn quantile(&self, q : f64) -> u8 {
        let sum = self.sum as f64;
        let mut p = 0.0;
        for ix in 0..=255 {
            p += self.hist[ix] as f64 / sum;
            if p >= q {
                return ix as u8;
            }
        }
        255
    }

    pub fn accumulate_inplace(&mut self) {
        for i in 1..256 {
            self.hist[i] += self.hist[i-1];
        }
        self.sum = self.hist[255];
    }

    pub fn quantile_when_accumulated(&self, q : f32) -> u8 {
        let target = (self.hist[255] as f32 * q) as u32;
        self.hist.partition_point(|px| *px < target ) as u8
    }

    pub fn calculate(img : &Image<u8, impl Storage<u8>>) -> Self {
        let mut gh = Self::new(img.height(), img.width());
        gh.update(img);
        gh
    }

    pub fn new(height : usize, width : usize) -> Self {

        let mut n_levels = 256;
        let n_channels = 1;
        let dtype = crate::foreign::ipp::ippi::IppDataType_ipp8u;
        let uniform_step = 1;
        let mut spec_sz = 0;
        let mut buf_sz = 0;
        let sz = crate::foreign::ipp::ippi::IppiSize { width : width as i32, height : height as i32 };
        unsafe {
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
            Self { spec, hist : [0; 256], hist_buffer, sum : (width * height) as u32 }
        }
    }

    pub fn update<S>(&mut self, win : &Image<u8, S>)
    where
        S : Storage<u8>
    {
        // TODO Maybe IPP already does this step?
        // self.hist.iter_mut().for_each(|bin| *bin = 0 );
        self.hist.fill(0);

        unsafe {
            let (step, sz) = crate::image::ipputils::step_and_size_for_image(win);
            let ans = crate::foreign::ipp::ippi::ippiHistogram_8u_C1R(
                win.as_ptr(),
                step,
                sz,
                self.hist.as_mut_ptr(),
                mem::transmute(self.spec.as_ptr()),
                self.hist_buffer.as_mut_ptr()
            );
            assert!(ans == 0);
        }
    }
}

/// Calculates the intensity histogram of the image read.
/// Always return a 256-sized vector of bit values. The second
/// field contains the total number of pixels at the image.
#[derive(Debug, Clone)]
pub struct ColorProfile([usize; 256], usize);

impl Default for ColorProfile {

    fn default() -> Self {
        ColorProfile([0; 256], 0)
    }

}

impl ColorProfile {

    pub fn num_pxs(&self) -> usize {
        self.1
    }

    pub fn calculate_from_pixels<'a>(px_iter : impl Iterator<Item=&'a u8>) -> Self {
        let mut hist = [0; 256];
        let mut n_pxs = 0;
        for px in px_iter {
            hist[*px as usize] += 1;
            n_pxs += 1;
        }
        Self(hist, n_pxs)
    }

    pub fn calculate(win : &Window<'_, u8>, spacing : usize) -> Self {
        Self::calculate_from_pixels(win.pixels(spacing))
    }

    pub fn quantiles(&self, qs : &[f64]) -> Vec<u8> {
        use bayes::calc::running;
        running::quantiles(&mut self.0.iter().cloned(), self.1, qs)
            .iter()
            .map(|qs| qs.0 as u8 )
            .collect()
    }

    /// Returns the histogram modes. The vector contains at most [required] modes,
    /// but might contain fewer than than if there is no pixel that satisfies
    /// the spacing constraints.
    pub fn modes(&self, required : usize, min_space : usize, limits : Option<(usize, usize)>) -> Vec<Mode> {
        let mut modes = Vec::new();
        next_mode(&mut modes, &self.0[..], required, min_space, limits);
        modes.drain(..).map(|ix| Mode { n_pxs : self.0[ix], color : ix as u8 }).collect()
    }

    /// Assumes each pixel comes from a normal density. Calculates the
    /// quality of the normal fir for each pixel attribution.
    pub fn normal_fit_quality(&self, modes : &[Mode], discrs : &[usize]) -> f64 {

        /*let mut joint_lp = 0.0;
        assert!(discrs.len() == modes.len() - 1);

        for (ix, m) in modes.iter().enumerate() {
            let low = if ix == 0 {
                0
            } else {
                m - discr[ix]
            };

            let high = if ix == modes.len() {
                255
            } else {
                m + discr[ix+1]
            };

            let width = (high - low) as f64 / 2.;
            let stddev = width / 1.96;
            let norm = Normal::prior(m as f64, stddev.powf(2.));
            for px in [low..high] {
                joint_lp += norm.log_prob(px as f64) * self.0[px] as f64;
            }
        }

        joint_lp*/
        unimplemented!()
    }

    /// Returns vec of modes and their discriminants.
    pub fn unsupervised_modes(&self, win_sz : usize, min_width : usize, min_rel : f32) -> (Vec<Mode>, Vec<(usize, usize)>) {
        let mut modes = Vec::new();
        let mut discrs = Vec::new();
        let n_pxs = self.0.iter().sum::<usize>();
        // println!("{}", n_pxs);
        bump_modes(&mut modes, &mut discrs, &self.0[..], 0..256, win_sz, min_width, min_rel, n_pxs);
        (modes, discrs)
    }

    pub fn unsupervised_modes_ordered(&self, win_sz : usize, min_dist : usize, min_rel : f32) -> Vec<Mode> {
        let mut vals : Vec<(usize, f32)> = self.0.iter()
            .map(|v| *v as f32 / self.1 as f32 )
            .enumerate()
            .collect();
        vals.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal) );
        let mut modes = Vec::new();
        for (ix, v) in vals.iter() {

            if *ix >= 1 && vals[0..(*ix-1)].iter().any(|prev_v| ((prev_v.0 as i64 - *ix as i64).abs() as usize) < min_dist ) {
                continue;
            }

            if *v >= min_rel {
                modes.push(Mode { color : *ix as u8, n_pxs : (v * self.1 as f32) as usize });
            } else {
                break;
            }

        }
        modes
    }

    /// Finds the k-1 smallest values between each pair of the k modes. Might contain
    /// fewer discriminants if fewer modes are found.
    pub fn discriminants(&self, modes : &[Mode]) -> Vec<usize> {
        modes.iter().take(modes.len()-1).zip(modes.iter().skip(1))
            .filter_map(|(m1, m2)| {
                if let Some(m) = self.0[(m1.color as usize)..(m2.color as usize)].iter().enumerate().min_by(|a, b| a.1.cmp(&b.1)) {
                    Some(m.0 + m1.color as usize)
                } else {
                    None
                }
            }).collect()
    }

    pub fn iter_bins<'a>(&'a self) -> impl Iterator<Item=(u8, usize)> + Clone + 'a {
        self.0.iter().enumerate().map(|(ix, n)| (ix as u8, *n) )
    }

    pub fn bins(&self) -> &[usize; 256] {
        &self.0
    }

    pub fn dense_regions(&self, min_count_per_px : usize, min_dist : f64, min_cluster_sz : usize) -> Vec<(u8, u8)> {
    
        use bayes::fit::cluster::SpatialClustering;
        
        let pts : Vec<[f64; 1]> = self.0.iter()
            .enumerate()
            .filter(|(_, n)| **n > min_count_per_px )
            .map(|(ix, n)| vec![ix, *n] ).flatten().map(|pt| [pt as f64] ).collect::<Vec<_>>();
        let mut clusts = SpatialClustering::cluster_linear(&pts[..], min_dist, min_cluster_sz);
        let mut limits = Vec::new();
        for (_, mut clust) in clusts.clusters.iter_mut() {
            clust.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(Ordering::Equal) );
            limits.push((clust[0][0] as u8, clust[clust.len()-1][0] as u8));
        }
        limits
    }

    pub fn probabilities(&self) -> Vec<f32> {
        let max = self.1 as f32;
        self.0.iter().map(move |count| *count as f32 / max ).collect()
    }

}

#[derive(Debug, Clone, Copy)]
pub struct Mode {
    pub color : u8,
    pub n_pxs : usize
}

fn bump_modes(
    modes : &mut Vec<Mode>,
    discrs : &mut Vec<(usize, usize)>,
    vals : &[usize],
    range : Range<usize>,
    win_sz : usize,
    min_width : usize,
    min_rel : f32,
    n_pxs : usize
) {
    if range.end <= range.start {
        return;
    }
    let mut max = (0, usize::MIN);
    let mut found_mode = false;
    find_mode_at_range(&mut max, &mut found_mode, &vals[..], &range);
    if found_mode {
        let mode = max.0;
        if (vals[mode] as f32 / n_pxs as f32) < min_rel {
            return;
        }
        match bump_limits(vals, mode, win_sz, min_width, range.clone()) {
            (Some(low), Some(high)) => {
                modes.push(Mode { color : mode as u8, n_pxs : vals[mode] });
                discrs.push((low, high));
                bump_modes(modes, discrs, vals, range.start..(low.saturating_sub(min_width/2)), win_sz, min_width, min_rel, n_pxs);
                bump_modes(modes, discrs, vals, (high + min_width/2)..range.end, win_sz, min_width, min_rel, n_pxs);
            },
            (Some(low), None) => {
                modes.push(Mode { color : mode as u8, n_pxs : vals[mode] });
                discrs.push((low, range.end));
                let new_range = range.start..(low.saturating_sub(min_width/2));
                if new_range.end - new_range.start > min_width {
                    bump_modes(modes, discrs, vals, new_range, win_sz, min_width, min_rel, n_pxs);
                }
            },
            (None, Some(high)) => {
                modes.push(Mode { color : mode as u8, n_pxs : vals[mode] });
                discrs.push((range.start, high));
                let new_range = (high + min_width/2)..range.end;
                if new_range.end - new_range.start > min_width {
                    bump_modes(modes, discrs, vals, new_range, win_sz, min_width, min_rel, n_pxs);
                }
            }
            (None, None) => {
                modes.push(Mode { color : mode as u8, n_pxs : vals[mode] });
                discrs.push((range.start, range.end));
            }
        }
    }
}

fn bump_min_limit(vals : &[usize], mode : usize, win_sz : usize, min_width : usize, range : &Range<usize>) -> Option<usize> {
    let mut min_val = mode.checked_sub(min_width / 2)?;
    if min_val <= range.start {
        return None;
    }
    while min_val > range.start + win_sz {
        if vals[(min_val-win_sz)..min_val].iter().sum::<usize>() as f32 / win_sz as f32 >= vals[min_val] as f32 /* * 0.95*/ {
            break;
        }
        min_val -= 1;
    }
    Some(min_val)
}

fn bump_max_limit(vals : &[usize], mode : usize, win_sz : usize, min_width : usize, range : &Range<usize>) -> Option<usize> {
    let mut max_val = mode + min_width / 2;
    if max_val >= range.end {
        return None;
    }
    while max_val < range.end.checked_sub(win_sz)? {
        if vals[max_val..(max_val+win_sz)].iter().sum::<usize>() as f32 / win_sz as f32 >= vals[max_val] as f32 /* * 0.95*/ {
            break;
        }
        max_val += 1;
    }
    Some(max_val)
}

fn bump_limits(vals : &[usize], mode : usize, win_sz : usize, min_width : usize, range : Range<usize>) -> (Option<usize>, Option<usize>) {
    let min_val = bump_min_limit(vals, mode, win_sz, min_width, &range);
    let max_val = bump_max_limit(vals, mode, win_sz, min_width, &range);
    (min_val, max_val)
}

fn find_mode_at_range(
    max : &mut (usize, usize),
    found : &mut bool,
    vals : &[usize],
    range : &Range<usize>,
    // min_space : usize
) {
    // println!("Iteration start");
    // println!("{:?}", (&range, &max, &found, &min_space));
    let range_len = range.end - range.start;
    if let Some((ix, max_val)) = vals[range.clone()].iter().enumerate().max_by(|a, b| a.1.cmp(&b.1) ) {
        // let far_from_left = ix > min_space;
        // let far_from_right = range_len - ix > min_space;
        /*let far = (range.start == 0 && range.end == 256) ||
            (range.start == 0 && far_from_right) ||
            (range.end == 256 && far_from_left) ||
            (far_from_left && far_from_right);*/
        // println!("Far = {:?}", far);
        if *max_val > max.1 {
            *max = (range.start + ix, *max_val);
            *found = true;
        }
    } else {
        // println!("No max val for {:?}", range);
    }

    // println!("Iteration end");
    // println!("{:?}", (&range, &max, &found, &min_space));
    // println!("\n\n");
}

fn next_mode(
    modes : &mut Vec<usize>,
    vals : &[usize],
    mut required : usize,
    min_space : usize,
    limits : Option<(usize, usize)>
) {
    let mut max = (0, usize::MIN);
    let n = modes.len();
    let mut found_mode = false;
    if let Some(lims) = limits {
        assert!(lims.1 > lims.0 && lims.1 <= 255);
    }
    let (lim_low, lim_high) = limits.unwrap_or((0, 256));
    match n {
        0 => {
            find_mode_at_range(&mut max, &mut found_mode, vals, &(lim_low..lim_high), /*min_space*/);
        },
        1 => {
            find_mode_at_range(&mut max, &mut found_mode, vals, &(lim_low..modes[0].saturating_sub(min_space)), /*min_space*/);
            modes.sort();
            find_mode_at_range(&mut max, &mut found_mode, vals, &((modes[0]+min_space+1).min(lim_high)..lim_high), /*min_space*/);
            modes.sort();
        },
        _ => {
            // Just arrays with a single range each. Any impl Iterator here that yields
            // a single element would work, since we will chain it below.
            let fst_range_arr = [0..modes[0].saturating_sub(min_space)];
            let last_range_arr = [(modes[n-1]+min_space+1).min(lim_high)..lim_high];
            let fst_range = fst_range_arr.into_iter().cloned();
            let found_pairs = modes.iter().take(n-1)
                .zip(modes.iter().skip(1))
                .map(|(a, b)| ( (a+min_space+1).min(b.saturating_sub(min_space))..(b.saturating_sub(min_space)) ));
            let last_range = last_range_arr.into_iter().cloned();

            for range in fst_range.chain(found_pairs).chain(last_range) {
                find_mode_at_range(&mut max, &mut found_mode, vals, &range, /*min_space*/ );
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
        next_mode(modes, vals, required, min_space, limits);
    }
}

#[test]
fn find_modes() {
    use crate::image::Image;
    let a = Image::new_checkerboard(8, 2);
    let hist = ColorProfile::calculate(&a.full_window(), 1);
    //println!("{:?}", a);
    //println!("{:?}", hist);
    //println!("{:?}", hist.modes(2, 2));
}



