use crate::image::Window;
use std::iter::FromIterator;
use std::collections::HashMap;
use std::ops::Range;
use crate::image::WindowMut;
// use crate::segmentation;
use bayes::fit::{cluster::KMeans, cluster::KMeansSettings, Estimator};

/*IppStatus ippiHistogram_<mod>(const Ipp<dataType>* pSrc, int srcStep, IppiSize roiSize,
Ipp32u* pHist, const IppiHistogramSpec* pSpec, Ipp8u* pBuffer );*/

mod segmentation;

pub use segmentation::*;

#[derive(Debug, Clone, Copy)]
pub struct ColorCluster {
    pub color : u8,
    pub n_px : usize,
    pub min : u8,
    pub max : u8
}

impl ColorCluster {

    pub fn belong(&self, px : u8) -> bool {
        px >= self.min && px <= self.max
    }

}

pub struct ColorClustering {
    pub colors : Vec<ColorCluster>
}

impl ColorClustering {

    pub fn calculate_from_pixels<'a>(pxs : impl Iterator<Item=&'a u8> + 'a + Clone, n_colors : usize, hist_init : bool) -> Self {
        let km = segment_colors(pxs.clone(), n_colors, hist_init);
        let means = segmentation::extract_mean_colors(&km);
        //let px_iter = win.pixels(px_spacing).map(|d| [*d as f64] );
        let px_iter = pxs.map(|d| [*d as f64] );
        let extremes = segmentation::extract_extreme_colors(&km, px_iter);
        let n_pxs = (0..n_colors).map(|c| km.count_allocations(c) ).collect::<Vec<_>>();
        let mut colors = means.iter().zip(extremes.iter().zip(n_pxs.iter()))
            .map(|(color, ((min, max), n_px))| ColorCluster { color : *color, min : *min, max : *max, n_px : *n_px })
            .collect::<Vec<_>>();
        colors.sort_by(|c1, c2| c1.color.cmp(&c2.color) );
        Self { colors }
    }

    pub fn calculate(win : &Window<'_, u8>, px_spacing : usize, n_colors : usize, hist_init : bool) -> Self {
        Self::calculate_from_pixels(win.pixels(px_spacing), n_colors, hist_init)
    }

    /// Paints each pixel with the assigned cluster average. Paint black unassigned clusters.
    pub fn paint<'a>(&'a self, win : &'a mut WindowMut<'a, u8>) {
        for mut px in win.pixels_mut(1) {
            let mut painted = false;
            for c in self.colors.iter() {
                if c.belong(*px) {
                    *px = c.color;
                    painted = true;
                }
            }
            if !painted {
                *px = 0;
            }
        }
    }

}

/// Returns an image, with each pixel attributed to its closest K-means color pixel
/// according to a given subsampling given by px_spacing. Also return the allocations,
/// which are the indices of the color vector each pixel in raster order belongs to.
/// (1) Call k-means for image 1
/// (2) For images 2..n:
///     (2.1). Find closest mean to each pixel
///     (2.2). Modify pixels to have this mean value.
pub fn segment_colors<'a>(pxs : impl Iterator<Item=&'a u8> + 'a + Clone, n_colors : usize, hist_init : bool) -> KMeans {
    let allocations = if hist_init {
        let hist = DenseHistogram::calculate_from_pixels(pxs.clone());
        let modes = hist.modes(n_colors, ((256 / n_colors) / 4) );
        if modes.len() == n_colors {
            let mut allocs : Vec<usize> = pxs.clone()
                .map(|px| {
                    modes.iter()
                        .enumerate()
                        .min_by(|m1, m2| (*px as i16 - m1.1.color as i16).abs().cmp(&(*px as i16 - m2.1.color as i16).abs()) ).unwrap().0
                }).collect();
            Some(allocs)
        } else {
            None
        }
    } else {
        None
    };
    let km = KMeans::estimate(
        // win.pixels(px_spacing).map(|d| [*d as f64] ),
        pxs.map(|d| [*d as f64] ),
        KMeansSettings { n_cluster : n_colors, max_iter : 1000, allocations }
    ).unwrap();

    km
}

/*pub fn segment_colors_to_image(win : &Window<'_, u8>, px_spacing : usize, n_colors : usize) -> Image<u8> {
    let km = segment_colors(win, px_spacing, n_colors, true);
    let colors = extract_mean_colors(&km);
    let ncol = win.width() / px_spacing;
    Image::from_vec(
        km.allocations().iter().map(|alloc| colors[*alloc] ).collect(),
        ncol
    )
}*/

pub trait ColorHistogram {

}

/// Only hold color values for pixels present in the image.
pub struct SparseHistogram(HashMap<u8, usize>);

impl SparseHistogram {

    pub fn calculate_from_pixels<'a>(px_iter : impl Iterator<Item=&'a u8>) -> Self {
        let mut hist : HashMap<u8, usize> = HashMap::new();
        for px in px_iter {
            if let Some(mut px) = hist.get_mut(px) {
                *px += 1;
            } else {
                hist.insert(*px, 1);
            }
        }
        Self(hist)
    }

    pub fn calculate(win : &Window<'_, u8>, spacing : usize) -> Self {
        Self::calculate_from_pixels(win.pixels(spacing))
    }
}

impl From<SparseHistogram> for DenseHistogram {

    fn from(sparse : SparseHistogram) -> Self {
        unimplemented!()
    }

}

#[derive(Debug, Clone, Copy)]
pub struct Mode {
    pub color : u8,
    pub n_pxs : usize
}

/// Calculates the intensity histogram of the image read.
/// Always return a 256-sized vector of bit values.
#[derive(Debug, Clone)]
pub struct DenseHistogram([usize; 256]);

impl DenseHistogram {

    pub fn calculate_from_pixels<'a>(px_iter : impl Iterator<Item=&'a u8>) -> Self {
        let mut hist = [0; 256];
        for px in px_iter {
            hist[*px as usize] += 1;
        }
        Self(hist)
    }

    pub fn calculate(win : &Window<'_, u8>, spacing : usize) -> Self {
        Self::calculate_from_pixels(win.pixels(spacing))
    }

    /// Returns the histogram modes. The vector contains at most [required] modes,
    /// but might contain fewer than than if there is no pixel that satisfies
    /// the spacing constraints.
    pub fn modes(&self, required : usize, min_space : usize) -> Vec<Mode> {
        let mut modes = Vec::new();
        next_mode(&mut modes, &self.0[..], required, min_space);
        modes.drain(..).map(|ix| Mode { n_pxs : self.0[ix], color : ix as u8 }).collect()
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

    pub fn bins<'a>(&'a self) -> impl Iterator<Item=(u8, usize)> + 'a {
        self.0.iter().enumerate().map(|(ix, n)| (ix as u8, *n) )
    }
}

fn find_mode_at_range(
    max : &mut (usize, usize),
    found : &mut bool,
    vals : &[usize],
    range : &Range<usize>,
    min_space : usize
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

fn next_mode(modes : &mut Vec<usize>, vals : &[usize], mut required : usize, min_space : usize) {
    let mut max = (0, usize::MIN);
    let n = modes.len();
    let mut found_mode = false;
    match n {
        0 => {
            find_mode_at_range(&mut max, &mut found_mode, vals, &(0..256), min_space);
        },
        1 => {
            find_mode_at_range(&mut max, &mut found_mode, vals, &(0..modes[0].saturating_sub(min_space)), min_space);
            modes.sort();
            find_mode_at_range(&mut max, &mut found_mode, vals, &(modes[0].saturating_add(min_space+1)..256), min_space);
            modes.sort();
        },
        _ => {
            // Just arrays with a single range each. Any impl Iterator here that yields
            // a single element would work, since we will chain it below.
            let fst_range_arr = [0..modes[0].saturating_sub(min_space)];
            let last_range_arr = [modes[n-1].saturating_add(min_space+1).min(256)..256];
            let fst_range = fst_range_arr.into_iter().cloned();
            let found_pairs = modes.iter().take(n-1)
                .zip(modes.iter().skip(1))
                .map(|(a, b)| (a.saturating_add(min_space+1).min(b.saturating_sub(min_space))..(b.saturating_sub(min_space)) ));
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
