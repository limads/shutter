use nalgebra::*;
use nalgebra::storage::*;
use std::iter::Iterator;
use std::fmt::Display;
use std::convert::TryFrom;
use std::ops::Range;
use super::*;
use std::ffi;
use std::mem;

// Wait for stabilization of f64 const fn here
// const fn half_pi() -> f64 { std::f64::consts::PI / 2.0 }
// const HALF_PI : f64 = half_pi();
// const HALF_PI : f64 = 1.570796327;

pub fn draw_square(
    mut frame : DMatrix<u8>,
    offset: (usize, usize),
    size: (usize, usize)
) -> DMatrix<u8> {
    for r in offset.0..offset.0 + size.0 {
        for c in offset.1..offset.1 + size.1 {
            frame[(r, c)] = 255;
        }
    }
    frame
}

// Returns pairs (rows (y); cols (x))
// center of shape (rows (y), cols (x))
// frame dims
// symmetric size of shape
fn shape_bounds_if_valid(
    center : (usize, usize),
    dims : (usize, usize),
    size : usize
) -> Option<(Range<usize>, Range<usize>)> {
    let int_range = (-1)*(size as i32 / 2)..(size as i32 / 2);
    let y_start = usize::try_from(center.0 as i32 + int_range.start);
    let y_end = usize::try_from(center.0 as i32 + int_range.end);
    let x_start = usize::try_from(center.1 as i32 + int_range.start);
    let x_end = usize::try_from(center.1 as i32 + int_range.end);
    match (y_start, y_end, x_start, x_end) {
        (Ok(y0), Ok(y1), Ok(x0), Ok(x1)) => {
            if y0 < dims.0 && y1 <= dims.0 && y0 < dims.1 && y1 <= dims.1 {
                Some((y0..(y1 + 1), x0..(x1 + 1)))
            } else {
                None
            }
        },
        _ => None
    }
}

pub fn draw_cross(
    buf : &mut [u8],
    image_dims : (usize, usize),
    center: (usize, usize),
    color : u8,
    size : usize
) {
    let mut pts : Vec<_> = Vec::new();
    match shape_bounds_if_valid(center, image_dims, size) {
        Some((yr, xr)) => {
            for y in yr {
                pts.push((y, center.1, color));
            }
            for x in xr {
                pts.push((center.0, x , color));
            }
            mark_image(buf, pts, image_dims.1);
        },
        None => { }
    }
}

/// Overwrite pixels in the buffer by informing (x, y, intensity) triples.
pub fn mark_image(
    buf : &mut [u8],
    pts : Vec<(usize, usize, u8)>,
    col_stride : usize
) {
    for (y, x, val) in pts {
        if let Some(p) = buf.get_mut(col_stride * y + x) {
            //println!("Prev: {}", p);
            *p = val;
            //println!("Curr: {}", p);
        } else {
            println!("Out of bounds marking");
        }
    }
}

pub fn draw_corners(
    buf : &mut [u8],
    image_dims : (usize, usize),
    center: (usize, usize),
    color : u8,
    size : usize
) {
    let mut pts : Vec<_> = Vec::new();
    match shape_bounds_if_valid(center, image_dims, size) {
        Some((mut yr, mut xr)) => {
            let ymin = yr.next().unwrap();
            let ymax = yr.last().unwrap();
            let xmin = xr.next().unwrap();
            let xmax = xr.last().unwrap();
            for (i, x) in [xmin, xmin + 1, xmax - 1, xmax].iter().enumerate() {
                for (j, y) in [ymin, ymin + 1, ymax - 1, ymax].iter().enumerate() {
                    if i == 0 || j == 0 || i == 3 || j == 3 {
                        pts.push((*y, *x, color));
                    }
                }
            }
            /*pts.push((xmin, ymin, color));
            pts.push((xmin, ymax, color));
            pts.push((xmax, ymin, color));
            pts.push((xmax, ymax, color));*/
            mark_image(buf, pts, image_dims.1);
        },
        None => { }
    }
}

/* Draws a straight line on the image carried by buf, assumed to be of dimensions
img_dims. When drawing, src is taken to be the reference point with respect to which
calculations are happening. The units are the same as the matrix index. */ 
pub fn draw_line(
    buf : &mut [u8],
    img_dims : (usize, usize),
    src : (usize, usize),
    dst : (usize, usize),
    color : u8 
) {
    let (nrow, _) = img_dims;
    let (dist, theta) = index::index_distance(src, dst, nrow);
    // println!("dist = {}; theta = {}", dist, theta);
    let d_max = dist as usize;
    // let reflect_x = if theta.abs() > HALF_PI { -1.0 } else { 1.0 };
    // let reflect_y = if theta < 0.0 { -1.0 } else { 1.0 };
    // println!("Reflect x : {}; Reflect y: {}", reflect_x, reflect_y);
    for i in 0..d_max {
        let x_incr = theta.cos() * i as f64;
        let y_incr = theta.sin() * i as f64;
        let x_pos = (src.1 as i32 + x_incr as i32) as usize;
        let y_pos = (src.0 as i32 - y_incr as i32) as usize;
        // println!("y index incr = {} (double)", (y_incr*reflect));
        // println!("y index incr = {} (usize)", (y_incr*reflect) as usize);
        // println!("x_incr = {}; y_incr = {}; x_pos = {}; y_pos = {}", x_incr, y_incr, x_pos, y_pos);
        buf[y_pos*nrow + x_pos] = color;
    }
}

#[test]
fn test_line() {

    let mut buf : [u8; 100] = [0; 100];
    
    // Drawn an X starting the trace from the upper portion
    draw_line(&mut buf, (10, 10), (2,2), (9, 9), 255);
    draw_line(&mut buf, (10, 10), (2,9), (9, 2), 255);
    
    let img = Image::new_from_slice(&buf, 10);
    println!("{}", img);
    
    // assert!(buf[3*10 + 3] == 1);
    // assert!(buf[4*10 + 4] == 1);
}

pub struct Digit {
    top_row : bool,
    mid_row : bool,
    bottom_row : bool,
    tl_col : bool,
    tr_col : bool,
    bl_col : bool,
    br_col : bool,
}

impl Digit {

    fn try_new(d : usize) -> Option<Self> {
        match d {
            0 => Some(Self{
                top_row : true,
                mid_row : false,
                bottom_row : true,
                tl_col : true,
                tr_col : true,
                bl_col : true,
                br_col : true
            }),
            1 => Some(Self{
                top_row : false,
                mid_row : false,
                bottom_row : false,
                tl_col : true,
                tr_col : false,
                bl_col : true,
                br_col : false
            }),
            2 => Some(Self{
                top_row : true,
                mid_row : true,
                bottom_row : true,
                tl_col : false,
                tr_col : true,
                bl_col : true,
                br_col : false
            }),
            3 => Some(Self{
                top_row : true,
                mid_row : true,
                bottom_row : true,
                tl_col : false,
                tr_col : true,
                bl_col : false,
                br_col : true
            }),
            4 => Some(Self {
                top_row : false,
                mid_row : true,
                bottom_row : false,
                tl_col : false,
                tr_col : true,
                bl_col : false,
                br_col : true
            }),
            5 => Some(Self {
                top_row : true,
                mid_row : true,
                bottom_row : true,
                tl_col : true,
                tr_col : false,
                bl_col : false,
                br_col : true
            }),
            6 => Some(Self {
                top_row : true,
                mid_row : true,
                bottom_row : true,
                tl_col : true,
                tr_col : false,
                bl_col : true,
                br_col : true
            }),
            7 => Some(Self {
                top_row : true,
                mid_row : false,
                bottom_row : false,
                tl_col : false,
                tr_col : true,
                bl_col : false,
                br_col : true
            }),
            8 => Some(Self {
                top_row : true,
                mid_row : true,
                bottom_row : true,
                tl_col : true,
                tr_col : true,
                bl_col : true,
                br_col : true
            }),
            9 => Some(Self {
                top_row : true,
                mid_row : true,
                bottom_row : true,
                tl_col : true,
                tr_col : true,
                bl_col : false,
                br_col : true
            }),
            _ => None
        }
    }

    fn to_points(&self, dim : usize) -> Vec<(usize, usize)> {
        let top_row : Vec<_> = (0..(dim / 2)).map(|c| (0, c as usize)).collect();
        let mid_row : Vec<_> = top_row.iter().map(|(r, c)| ((dim / 2) as usize, *c) ).collect();
        let bottom_row : Vec<_> = mid_row.iter().map(|(_, c)| ((dim - 1 as usize), *c) ).collect();
        let tl_col : Vec<_> = (0..(dim / 2)).map(|r| (r as usize, 0)).collect();
        let bl_col : Vec<_> = tl_col.iter().map(|(r, _)| ((r + dim/2) as usize, 0) ).collect();
        let tr_col : Vec<_> = (0..(dim/2)).map(|r| (r, (dim/2-1) as usize) ).collect();
        let br_col : Vec<_> = tr_col.iter().map(|(r, c)| ( (r + dim/2) as usize, (dim/2-1) as usize) ).collect();
        let mut digits = Vec::new();
        if self.top_row {
            digits.extend(top_row.iter());
        }
        if self.mid_row {
            digits.extend(mid_row.iter());
        }
        if self.bottom_row {
            digits.extend(bottom_row.iter());
        }
        if self.tl_col {
            digits.extend(tl_col.iter());
        }
        if self.tr_col {
            digits.extend(tr_col.iter());
        }
        if self.br_col {
            digits.extend(br_col.iter());
        }
        digits
    }

}

pub fn draw_digit(
    buf : &mut [u8], 
    digit : usize, 
    offset : (usize, usize), 
    size : usize, 
    col_stride : usize
) {
    if let Some(dig) = Digit::try_new(digit) {
        let pts : Vec<(usize, usize, u8)> = dig.to_points(size).iter()
            .map(|(r, c)| (*r + offset.0, *c + offset.1, 255 as u8)).collect();
        mark_image(buf, pts, col_stride);
    } else {
        println!("Could not convert digit {}", digit)
    }
}

pub fn draw_digit_native(slice : &mut [u8], ncols : usize, tl_pos : (usize, usize), val : usize, sz : usize, col : u8) {
    let mut digits : Vec<usize> = Vec::new();
    let mut curr_val = val;
    while curr_val > 10 {
        let rem = val % 10;
        if rem == 0 {
            digits.push(1);
        }
        digits.push(rem);
        curr_val /= 10;
    }
    if curr_val == 10 {
        digits.push(1);
    }
    digits.push(curr_val);
    for (ix, digit) in digits.iter().enumerate() {
        let space = if ix == 0 { 0 } else { 1 };
        let row = tl_pos.0;
        let col = tl_pos.1 + ix * sz + space;
        draw_digit(
            slice, 
            *digit, 
            (row, col), 
            sz, 
            ncols
        );
    }
}
