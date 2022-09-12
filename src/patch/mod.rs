use crate::image::Window;
use std::iter::FromIterator;
use std::collections::HashMap;
use std::ops::Range;
use crate::image::WindowMut;
// use crate::segmentation;
use bayes::fit::{cluster::KMeans, cluster::KMeansSettings, Estimator};
use bayes::fit::cluster::SpatialClustering;
use std::cmp::Ordering;

pub mod seed;

pub mod raster;

pub mod density;

pub mod ray;

pub mod pattern;

pub mod edge;

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

#[derive(Clone, Debug)]
pub struct ColorClustering {
    pub colors : Vec<ColorCluster>,
    pub hist : Option<ColorProfile>
}

impl ColorClustering {

    pub fn calculate_from_pixels<'a>(pxs : impl Iterator<Item=&'a u8> + 'a + Clone, n_colors : usize, hist_init : bool) -> Self {
        let (km, hist) = segment_colors(pxs.clone(), n_colors, hist_init);
        let means = segmentation::extract_mean_colors(&km);
        //let px_iter = win.pixels(px_spacing).map(|d| [*d as f64] );
        let px_iter = pxs.map(|d| [*d as f64] );
        let extremes = segmentation::extract_extreme_colors(&km, px_iter);
        let n_pxs = (0..n_colors).map(|c| km.count_allocations(c) ).collect::<Vec<_>>();
        let mut colors = means.iter().zip(extremes.iter().zip(n_pxs.iter()))
            .map(|(color, ((min, max), n_px))| ColorCluster { color : *color, min : *min, max : *max, n_px : *n_px })
            .collect::<Vec<_>>();
        colors.sort_by(|c1, c2| c1.color.cmp(&c2.color) );
        Self { colors, hist }
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
pub fn segment_colors<'a>(pxs : impl Iterator<Item=&'a u8> + 'a + Clone, n_colors : usize, hist_init : bool) -> (KMeans, Option<ColorProfile>) {
    let hist = if hist_init {
        Some(ColorProfile::calculate_from_pixels(pxs.clone()))
    } else {
        None
    };
    let allocations = if let Some(hist) = hist.as_ref() {
        let modes = hist.modes(n_colors, ((256 / n_colors) / 4), None);
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

    (km, hist)
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

/*impl From<SparseHistogram> for ColorProfile {

    fn from(sparse : SparseHistogram) -> Self {
        unimplemented!()
    }

}*/

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
