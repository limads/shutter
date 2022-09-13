use crate::image::Window;
use super::Patch;
use std::cmp::{PartialEq, Eq};
use crate::shape;

// TODO rename to RadialSegmenter

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Inward,
    Outward
}

/// Similar to the seed segmenter, the ray segmenter starts from
/// a center pixel and propagates outward. Unlike the seed segmenter,
/// which propagates in a fan comparing all pixels upward and downard,
/// the ray compares only the next pixel in a ray of angle theta distant
/// from other rays.
pub struct RaySegmenter {

}

fn step_inward(px : (usize, usize), theta : f64, radius_incr : usize) -> (f64, f64) {
    (
        (px.0 as f64 - theta.sin() * (-1.*(radius_incr as f64) - 1.)).floor(),
        (px.1 as f64 + theta.cos() * (-1.*(radius_incr as f64) - 1.)).floor()
    )
}

fn step_outward(px : (usize, usize), theta : f64, radius_incr : usize) -> (f64, f64) {
    (
        (px.0 as f64 - theta.sin() * (radius_incr as f64 + 1.)).ceil(),
        (px.1 as f64 + theta.cos() * (radius_incr as f64 + 1.)).ceil()
    )
}

impl RaySegmenter {

    pub fn new() -> Self {
        Self{}
    }

    // If points appear close together, try choosing a bigger initial radius. This happens
    // because for a small seed ring, discretization errors might lead to a lot of overlapping rays.
    pub fn segment(
        &mut self,
        win : &Window<'_, u8>,
        center : (usize, usize),
        fst_step_radius : usize,
        radius_incr : usize,
        theta : f64,
        byte_delta : u8,
        direction : Direction
    ) -> Option<Patch> {
        assert!(theta > 0.0 && theta < 2. * std::f64::consts::PI);
        assert!(radius_incr > 0 && fst_step_radius > 0);

        let n_radii = ((2. * std::f64::consts::PI) / theta) as usize;

        let step_fn = match direction {
            Direction::Inward => step_inward,
            Direction::Outward => step_outward
        };

        // Holds pixel index, values and their angles
        let mut circ_pxs : Vec<(usize, (usize, usize), f64)> = Vec::new();
        let mut final_pxs : Vec<(usize, (u16, u16))> = Vec::new();
        for ix in 0..n_radii {
            let this_theta = (ix as f64 * theta);
            let px = (
                center.0 as f64 - (this_theta.sin()*(fst_step_radius as f64)),
                center.1 as f64 + (this_theta.cos()*(fst_step_radius as f64))
            );

            if px.0 > 0. && px.1 > 0. && px.0 < (win.height() as f64 - 1.0) && px.1 < (win.width() as f64 - 1.0) {

                let px_u = (px.0 as usize, px.1 as usize);

                // Avoids redundant ray pixel sources, independent of how close the user informed them.
                // If the user informs a small fst_step_radius, there will be a lot of overlapping pixels.
                // If many calls are done, the start values can be kept in a cache, and only
                // translated to a different center at the next call to segment.

                // TODO verify if seed ring pixel also satisfies patch color. Perhaps the ring
                // is mis-located relative to a patch, in which case each ray will have a different
                // color.
                if circ_pxs.iter().find(|px| px.1 == px_u ).is_none() {
                    circ_pxs.push((ix, px_u, this_theta));
                }
            }
        }

        // Holds indices of pixels that should be removed at this iteration
        let mut remove_ixs : Vec<usize> = Vec::new();
        let mut n_expansions = 0;
        while circ_pxs.len() >= 1 {

            for (ix, px, theta) in circ_pxs.iter_mut() {
                let next_px = step_fn(*px, *theta, radius_incr);
                let within_bounds = next_px.0 > 0. && next_px.1 > 0. && next_px.0 < (win.height() as f64 - 1.0) && next_px.1 < (win.width() as f64 - 1.0);

                // For outward mode: remove the pixel when it reaches the border
                if !within_bounds {
                    remove_ixs.push(*ix);
                    final_pxs.push((*ix, (px.0 as u16, px.1 as u16)));
                    continue;
                }

                let next_px_u = (next_px.0 as usize, next_px.1 as usize);

                // For inward mode only: Stop when the pixel reaches the center.
                if direction == Direction::Inward && shape::point_euclidian(next_px_u, center) <= 1. {
                    remove_ixs.push(*ix);
                    final_pxs.push((*ix, (px.0 as u16, px.1 as u16)));
                    continue;
                }

                if ((win[*px] as i16 - win[next_px_u] as i16).abs() as u8) < byte_delta {
                    *px = next_px_u;
                } else {
                    remove_ixs.push(*ix);
                    final_pxs.push((*ix, (px.0 as u16, px.1 as u16)));
                }
            }

            for rem_ix in remove_ixs.drain(..) {
                if let Ok(curr_ix) = circ_pxs.binary_search_by(|(ix, _, _)| ix.cmp(&rem_ix) ) {
                    circ_pxs.remove(curr_ix);
                }
            }

            if circ_pxs.len() >= 1 {
                n_expansions += 1;
            }

        }

        if n_expansions >= 1 {
            final_pxs.sort_by(|a, b| a.0.cmp(&b.0) );
            let pt = final_pxs[0].1;
            let pxs = final_pxs.drain(..).map(|(_, px)| px ).collect::<Vec<_>>();
            let mut patch = Patch {
                pxs,
                outer_rect : (pt.0, pt.1, 1, 1),
                color : 0,
                scale : 1,
                img_height : win.height(),
                area : 0
            };
            patch.expand_rect(&patch.pxs.clone()[..]);
            // super::close_contour(&mut patch, win);
            Some(patch)
        } else {
            None
        }
    }

}


