use nalgebra::*;
use std::iter::Iterator;
use std::convert::{TryFrom, AsMut};
use std::ops::Range;
use crate::image::*;
use crate::raster::*;
use serde::{Serialize, Deserialize};

pub trait Draw {

    fn draw(&mut self, mark : Mark);

}

impl<'a> Draw for WindowMut<'a, u8> {

    fn draw(&mut self, mark : Mark) {
        /*let slice_ptr = self.win.data.as_mut_slice().as_mut_ptr();
        let ptr_offset = slice_ptr as u64 - (self.original_size().0*(self.offset.1 - 1)) as u64 - self.offset.0 as u64;
        let orig_ptr = ptr_offset as *mut u8;
        let orig_slice = unsafe { std::slice::from_raw_parts_mut(orig_ptr, self.original_size().0 * self.original_size().1) };*/
        match mark {
            Mark::Cross(pos, sz, col) => {
                let cross_pos = (self.offset().0 + pos.0, self.offset().1 + pos.1);
                draw_cross(
                    unsafe { self.original_slice() },
                    self.original_size(),
                    cross_pos,
                    col,
                    sz
                );
            },
            Mark::Corner(pos, sz, col) => {
                let center_pos = (self.offset().0 + pos.0, self.offset().1 + pos.1);
                draw_corners(
                    unsafe { self.original_slice() },
                    self.original_size(),
                    center_pos,
                    col,
                    sz
                );
            },
            Mark::Line(src, dst, color) => {
                let src_pos = (self.offset().0 + src.0, self.offset().1 + src.1);
                let dst_pos = (self.offset().0 + dst.0, self.offset().1 + dst.1);

                #[cfg(feature="opencv")]
                unsafe {
                    cvutils::draw_line(self.win, self.original_size().1, src_pos, dst_pos, color);
                    return;
                }

                draw_line(
                    unsafe { self.original_slice() },
                    self.original_size(),
                    src_pos,
                    dst_pos,
                    color
                );
            },
            Mark::Rect(tl, sz, color) => {
                let tr = (tl.0, tl.1 + sz.1);
                let br = (tl.0 + sz.0, tl.1 + sz.1);
                let bl = (tl.0 + sz.0, tl.1);
                self.draw(Mark::Line(tl, tr, color));
                self.draw(Mark::Line(tr, br, color));
                self.draw(Mark::Line(br, bl, color));
                self.draw(Mark::Line(bl, tl, color));
            },
            Mark::Digit(pos, val, sz, color) => {
                let tl_pos = (self.offset().0 + pos.0, self.offset().1 + pos.1);

                #[cfg(feature="opencv")]
                unsafe {
                    cvutils::write_text(self.win, self.original_size().1, tl_pos, &val.to_string()[..], color);
                    return;
                }

                draw_digit_native(unsafe { self.original_slice() }, self.original_size().1, tl_pos, val, sz, color);
            },
            /*Mark::Label(pos, msg, sz, color) => {
                let tl_pos = (self.offset.0 + pos.0, self.offset.1 + pos.1);

                #[cfg(feature="opencv")]
                unsafe {
                    cvutils::write_text(self.win, self.original_size().1, tl_pos, msg, color);
                    return;
                }

                panic!("Label draw require 'opencv' feature");
            },*/
            Mark::Circle(pos, radius, color) => {
                let center_pos = (self.offset().0 + pos.0, self.offset().1 + pos.1);

                #[cfg(feature="opencv")]
                unsafe {
                    crate::image::cvutils::draw_circle(self.win, self.original_size().1, center_pos, radius, color);
                    return;
                }

                panic!("Circle draw require 'opencv' feature");
            },
            Mark::Dot(pos, radius, color) => {
                for i in 0..self.height() {
                    for j in 0..self.width() {
                        if crate::feature::shape::point_euclidian((i, j), pos) <= radius as f32 {
                            self[(i, j)] = color;
                        }
                    }
                }
            },
            Mark::Shape(pts, close, col) => {
                let n = pts.len();
                if n < 2 {
                    return;
                }
                for (p1, p2) in pts.iter().take(n-1).zip(pts.iter().skip(1)) {
                    self.draw(Mark::Line(*p1, *p2, col));
                }

                if close {
                    self.draw(Mark::Line(pts[0], pts[pts.len()-1], col));
                }
            },
            Mark::Text(tl_pos, txt, color) => {

                #[cfg(feature="opencv")]
                {
                    unsafe {
                        crate::image::cvutils::write_text(
                            self.win,
                            self.original_size().1,
                            (self.offset.0 + tl_pos.0, self.offset.1 + tl_pos.1),
                            &txt[..],
                            color
                        );
                    }
                    return;
                }

                println!("Warning: Text drawing require opencv feature");
            },
            Mark::Arrow(from, to, thickness, color) => {

                #[cfg(feature="opencv")]
                {
                    let mut out : opencv::core::Mat = self.into();
                    opencv::imgproc::arrowed_line(
                        &mut out,
                        opencv::core::Point2i::new(from.1 as i32, from.0 as i32),
                        opencv::core::Point2i::new(to.1 as i32, to.0 as i32),
                        opencv::core::Scalar::from(color as f64),
                        thickness as i32,
                        opencv::imgproc::LINE_8,
                        0,
                        0.1
                    );
                    return;
                }

                println!("Warning: Arrow drawing require opencv feature");
            }
        }
    }

}

impl Draw for Image<u8> {

    fn draw(&mut self, mark : Mark) {
        let win : &mut WindowMut<'_, u8> = self.as_mut();
        win.draw(mark);
    }

}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Mark {

    // Position, square lenght and color
    Cross((usize, usize), usize, u8),

    // Position, square lenght and color
    Corner((usize, usize), usize, u8),

    // Start and end positions and color
    Line((usize, usize), (usize, usize), u8),

    // Position, digit value, digit size and color
    Digit((usize, usize), usize, usize, u8),

    // Position, label, digit value, size and color
    // Label((usize, usize), &'static str, usize, u8),

    // Center, radius and color
    Circle((usize, usize), usize, u8),

    /// A dense circle
    Dot((usize, usize), usize, u8),

    /// TL pos, size and color
    Rect((usize, usize), (usize, usize), u8),

    /// Arbitrary shape coordinates; whether to close it; and color
    Shape(Vec<(usize, usize)>, bool, u8),

    Text((usize, usize), String, u8),

    Arrow((usize, usize), (usize, usize), usize, u8)

}

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
            /*println!(
                "Warning: Out of bounds marking. Requested coordinate: ({}, {}); But image is {} x {}",
                y,
                x,
                buf.len() / col_stride,
                col_stride
            );*/
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
    let (nrow, ncol) = img_dims;

    // Draw straight horizontal line (if applicable)
    if src.0 == dst.0 {
        for c in src.1.min(dst.1)..(src.1.max(dst.1)) {
            buf[src.0*ncol + c] = color;
        }
        return;
    }

    // Draw straight vertical line (if applicable)
    if src.1 == dst.1 {
        for r in src.0.min(dst.0)..(src.0.max(dst.0)) {
            buf[r*ncol + src.1] = color;
        }
        return;
    }

    // Draw non-straight line. A bit more costly, since we must calculate the
    // inclination angle, and fill all pixels across the diagonal.
    let (dist, theta) = index::index_distance(src, dst, nrow);
    let d_max = dist as usize;
    for i in 0..d_max {
        let x_incr = theta.cos() * i as f64;
        let y_incr = theta.sin() * i as f64;
        let x_pos = (src.1 as i32 + x_incr as i32) as usize;
        let y_pos = (src.0 as i32 - y_incr as i32) as usize;
        buf[y_pos*ncol + x_pos] = color;
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
        let mid_row : Vec<_> = top_row.iter().map(|(_r, c)| ((dim / 2) as usize, *c) ).collect();
        let bottom_row : Vec<_> = mid_row.iter().map(|(_, c)| ((dim - 1 as usize), *c) ).collect();
        let tl_col : Vec<_> = (0..(dim / 2)).map(|r| (r as usize, 0)).collect();
        let bl_col : Vec<_> = tl_col.iter().map(|(r, _)| ((r + dim/2) as usize, 0) ).collect();
        let tr_col : Vec<_> = (0..(dim/2)).map(|r| (r, (dim/2-1) as usize) ).collect();
        let br_col : Vec<_> = tr_col.iter().map(|(r, _c)| ( (r + dim/2) as usize, (dim/2-1) as usize) ).collect();

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
        if self.bl_col {
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

pub fn draw_digit_native(slice : &mut [u8], ncols : usize, tl_pos : (usize, usize), val : usize, sz : usize, _col : u8) {
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
