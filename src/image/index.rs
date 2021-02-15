use nalgebra::*;
use nalgebra::storage::*;
use std::iter::Iterator;
use std::fmt::Display;
use std::convert::TryFrom;
use std::ops::Range;
use super::*;
use std::ffi;
use std::mem;

/// Returns position relative to the current linear index in the image with the given number of columns.
pub fn coordinate_index(lin_ix : usize, ncol : usize) -> (usize, usize) {
    (lin_ix / ncol, lin_ix % ncol)
}

pub fn linear_index(coord_ix : (usize, usize), ncol : usize) -> usize {
    coord_ix.0 * ncol + coord_ix.1
}

/// Translate a matrix index (row, col) into planar coordinates (x, y),
/// keeping the x value but flipping the y value.
pub fn index_to_plane_coord(ix : (usize, usize), nrow : usize) -> (f64, f64) {
    let x = ix.1 as f64;
    let y = (nrow - ix.0) as f64; 
    (x, y)
}

/// Assuming the index (row, col) can be interpreted as a graphical coordinate,
/// returns the distance and angle between the source index (taken to be the (0,0) origin in the
/// cartesian plane) and the destination index as (dist, angle). Angles are in radians, and
/// count positive from 0º-180º (0 rad - 3.14 rad) and negative from 180º-360º (3.14 rad - 0 rad).
/// This is as if the positive radian value has been "reflected" over the lower half of the trig circle. 
pub fn index_distance(src : (usize, usize), dst : (usize, usize), nrow : usize) -> (f64, f64) {
    let (src_x, src_y) = index_to_plane_coord(src, nrow);
    let (dst_x, dst_y) = index_to_plane_coord(dst, nrow);
    
    // "Base" and "Height" might be negative here depending on the right triangle orientation
    let base = dst_x - src_x;
    
    // Convert from matrix index to graphical plane
    let height = dst_y - src_y;
    
    // We can use abs here because reflecting the triangle over the vertical or horizontal axis
    // does not change its hypotenuse.
    let dist = base.abs().hypot(height.abs());
    let theta = height.atan2(base);
    
    (dist, theta)
}

#[test]
fn ix_dist() {
    let img_sz = (10, 10);
    
    // Cross
    println!("Dist = 3; 0º: {:?}", index_distance((4, 4), (4, 7), img_sz.0));
    println!("Dist = 3; 90º: {:?}", index_distance((4, 4), (1, 4), img_sz.0));
    println!("Dist = 3; 180º: {:?}", index_distance((4, 4), (4, 1), img_sz.0)); 
    println!("Dist = 3; 360º: {:?}", index_distance((4, 4), (7, 4), img_sz.0));
    
    // X
    println!("Dist = 4.24; 45º: {:?}", index_distance((4, 4), (1, 7), img_sz.0));
    println!("Dist = 4.24; 135º: {:?}", index_distance((4, 4), (1, 1), img_sz.0));
    println!("Dist = 4.24; 225º: {:?}", index_distance((4, 4), (7, 1), img_sz.0)); 
    println!("Dist = 4.24; 315º: {:?}", index_distance((4, 4), (7, 7), img_sz.0));
     
}
