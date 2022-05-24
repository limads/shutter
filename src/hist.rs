use crate::image::*;
use std::iter::FromIterator;
use std::mem;
use std::cmp::Ord;

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

pub fn peaks<T>(s : &[T], min_width : usize, min_height : T) -> Vec<usize>
where
    T : Ord + Copy
{
    assert!(s.len() % min_width == 0);

    // The nice property about looking for local maxima at the peak minimum width is
    // that we can guarantee peaks located two chunks apart will always be valid candidates,
    // so we can remove peaks one-chunk apart if they are two close by only compairing
    // only the contiguous pairs.

    let labeled : Vec<(usize, T)> = s.iter().copied().enumerate().collect();
    let mut peaks : Vec<_> = labeled
        .chunks(min_width)
        .map(|s| s.iter().max_by(|a, b| a.1.cmp(&b.1) ).copied().unwrap() )
        .filter(|p| p.1 > min_height )
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

    for ix in (0..peaks.len()).rev() {
        if !is_strict_maximum(&s[..], peaks[ix].0, min_width) {
            peaks.remove(ix);
        }
    }

    peaks.drain(..).map(|p| p.0 ).collect()
}

// Return triplets (peak, valley, peak)
pub fn peaks_and_valleys<T>(s : &[T], min_width : usize, min_height : T) -> Vec<(usize, usize, usize)>
where
    T : Ord + Copy
{
    let peaks = peaks(s, min_width, min_height);
    let mut triplets = Vec::new();
    for (p1, p2) in peaks[0..(peaks.len()-1)].iter().zip(peaks[1..peaks.len()].iter()) {
        let min = s[*p1..*p2].iter().enumerate().min_by(|a, b| a.1.cmp(&b.1) ).unwrap();
        triplets.push((*p1, *p1 + min.0, *p2));
    }
    triplets
}

impl GrayHistogram {

    pub fn as_slice(&self) -> &[u32] {
        &self.0[..]
    }

    pub fn update(&mut self, win : &Window<'_, u8>) {

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


