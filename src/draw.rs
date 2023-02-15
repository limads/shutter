use nalgebra::DMatrix;
use std::iter::{Iterator, IntoIterator};
use std::convert::{TryFrom, AsMut};
use std::ops::Range;
use crate::image::*;
use serde::{Serialize, Deserialize};
use crate::shape::*;
use crate::shape::ellipse::EllipseCoords;
use crate::shape::ellipse::OrientedEllipse;
use nalgebra::Vector2;

pub const DARK : u8 = 0;

pub const DARK_GRAY : u8 = 65;

pub const GRAY : u8 = 127;

pub const LIGHT_GRAY : u8 = 191;

pub const WHITE : u8 = 255;

const TWO_PI : f64 = 6.283185307;

impl<S> Image<u8, S> 
where
    S : StorageMut<u8>
{

    // Color is decided adaptively based on the average intensity of a local window
    // around the target pixels: Color = (255 - avg_px_col). The window size is chosen to
    // minimally enclose the shape, and the same shape is always drawn with the same
    // color (decided as the inversion of the average window color).
    pub fn draw_contrasting(&mut self, mark : Mark) {
        let (off, sz) = mark.enclosing_rect();
        if let Some(sub) = self.window(off, sz) {
            let mean = sub.mean::<f32>(1).unwrap() as u8;
            let inv_color = 255 - mean;
            self.draw(mark, inv_color);
        }
    }

    pub fn draw_many_contrasting(&mut self, marks : impl IntoIterator<Item=Mark>) {
        for mark in marks.into_iter() {
            self.draw_contrasting(mark);
        }
    }
    
    pub fn draw_many(&mut self, marks : impl IntoIterator<Item=Mark>, color : u8) {
        for mark in marks.into_iter() {
            self.draw(mark, color);
        }
    }

    pub fn draw(&mut self, mark : Mark, color : u8) {
        match mark {
            Mark::Cross(pos, sz) => {
                // let cross_pos = (self.offset().0 + pos.0, self.offset().1 + pos.1);
                let orig_sz = self.original_size();
                draw_cross(
                    self.slice_mut(),
                    orig_sz,
                    pos,
                    color,
                    sz
                );
            },
            Mark::Corner(pos, sz) => {
                // let center_pos = (self.offset().0 + pos.0, self.offset().1 + pos.1);
                let orig_sz = self.original_size();
                draw_corners(
                    self.slice_mut(),
                    orig_sz,
                    pos,
                    color,
                    sz
                );
            },
            Mark::Line(src, dst) => {
                // let src_pos = (self.offset().0 + src.0, self.offset().1 + src.1);
                // let dst_pos = (self.offset().0 + dst.0, self.offset().1 + dst.1);

                #[cfg(feature="opencv")]
                unsafe {
                    let orig_sz = self.original_size();
                    cvutils::draw_line(
                        self.slice_mut(), 
                        orig_sz.1, 
                        src,
                        dst,
                        color
                    );
                    return;
                }
                let orig_sz = self.original_size();
                draw_line(
                    self.slice_mut(),
                    orig_sz,
                    src,
                    dst,
                    color
                );
            },
            Mark::Rect(tl, sz) => {
                let tr = (tl.0, tl.1 + sz.1);
                let br = (tl.0 + sz.0, tl.1 + sz.1);
                let bl = (tl.0 + sz.0, tl.1);
                self.draw(Mark::Line(tl, tr), color);
                self.draw(Mark::Line(tr, br), color);
                self.draw(Mark::Line(br, bl), color);
                self.draw(Mark::Line(bl, tl), color);
            },
            Mark::Digit(pos, val, sz) => {
                // let tl_pos = (self.offset().0 + pos.0, self.offset().1 + pos.1);
                
                #[cfg(feature="opencv")]
                unsafe {
                    let m_sz = self.original_size();
                    cvutils::write_text(
                        self.slice_mut(),
                        m_sz.1, 
                        pos,
                        &val.to_string()[..], 
                        color
                    );
                    return;
                }
                
                let orig_w = self.original_size().1;
                draw_digit_native(
                    self.slice_mut(), 
                    orig_w, 
                    pos,
                    val, 
                    sz, 
                    color
                );
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
            Mark::Circle(pos, radius) => {
                // let center_pos = (self.offset().0 + pos.0, self.offset().1 + pos.1);

                #[cfg(feature="opencv")]
                unsafe {
                    let orig_sz = self.original_size();
                    crate::image::cvutils::draw_circle(
                        self.slice_mut(), 
                        orig_sz.1, 
                        pos,
                        radius, 
                        color
                    );
                    return;
                }

                let n_points = (TWO_PI * radius as f64) as usize + 4;
                for ix in 0..=n_points {
                    let theta = (ix as f64 / n_points as f64) * TWO_PI;
                    let x = pos.1 as f64 + theta.cos() * radius as f64;
                    let y = (self.height() - pos.0) as f64 + theta.sin() * radius as f64;
                    if x >= 0.0 && x < self.width() as f64 {
                        if y >= 0.0 && y < self.height() as f64 {
                            let i = x as usize;
                            let j = self.height() - y as usize;
                            self[(i, j)] = color;
                        }
                    }
                }
            },
            Mark::Dot(pos, radius) => {
                for i in pos.0.saturating_sub(radius)..(pos.0 + radius).min(self.width()) {
                    for j in pos.1.saturating_sub(radius)..(pos.1 + radius).min(self.height()) {
                        if crate::shape::point_euclidian((i, j), pos) <= radius as f32 {
                            self[(i, j)] = color;
                        }
                    }
                }
            },
            Mark::Shape(pts, close) => {
                let n = pts.len();
                if n < 2 {
                    return;
                }
                for (p1, p2) in pts.iter().take(n-1).zip(pts.iter().skip(1)) {
                    self.draw(Mark::Line(*p1, *p2), color);
                }

                if close {
                    self.draw(Mark::Line(pts[0], pts[pts.len()-1]), color);
                }
            },
            Mark::Text(pos, txt) => {

                #[cfg(feature="opencv")]
                {
                    let sz = self.original_size();
                    let off = self.offset();
                    // let tl_pos =  (off.0 + tl_pos.0, off.1 + tl_pos.1);
                    unsafe {
                        crate::image::cvutils::write_text(
                            self.slice_mut(),
                            sz.1,
                            pos,
                            &txt[..],
                            color
                        );
                    }
                    return;
                }

                draw_label(self, &txt, pos);
            },
            Mark::Arrow(from, to, thickness) => {

                #[cfg(feature="opencv")]
                {
                    /*let mut out : opencv::core::Mat = self.into();
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
                    return;*/
                    unimplemented!()
                }

                println!("Warning: Arrow drawing require opencv feature");
            },
            Mark::EllipseArrows(coords) => {
                self.draw(Mark::Arrow(coords.center, coords.major, 1), color);
                self.draw(Mark::Arrow(coords.center, coords.minor, 1), color);
            },
            Mark::FilledEllipse(el) => {
                for r in 0..self.height() {
                    for c in 0..self.width() {
                        let pt = Vector2::new(c as f32, (self.height() - r) as f32);
                        if crate::shape::ellipse::contains(&el, &pt) {
                            self[(r, c)] = color;
                        }
                    }
                }
            },
            Mark::Ellipse(el, n) => {
                let pts = crate::shape::ellipse::generate_ellipse_points(&el, n);
                for n in 0..(pts.len()-1) {
                    let a = crate::shape::coord::point_to_coord(&pts[n], self.shape());
                    let b = crate::shape::coord::point_to_coord(&pts[n+1], self.shape());
                    if let (Some(coord_a), Some(coord_b)) = (a, b) {
                        self.draw(Mark::Line(coord_a, coord_b), color);
                    }
                }
            }
        }
    }

}

#[derive(Clone, Debug)]
pub enum Mark {

    // Position, square lenght and color
    Cross((usize, usize), usize),

    // Position, square lenght and color
    Corner((usize, usize), usize),

    // Start and end positions and color
    Line((usize, usize), (usize, usize)),

    // Position, digit value, digit size and color
    Digit((usize, usize), usize, usize),

    // Position, label, digit value, size and color
    // Label((usize, usize), &'static str, usize, u8),

    // Center, radius and color
    Circle((usize, usize), usize),

    /// A dense circle
    Dot((usize, usize), usize),

    /// TL pos, size and color
    Rect((usize, usize), (usize, usize)),

    /// Arbitrary shape coordinates; whether to close it; and color
    Shape(Vec<(usize, usize)>, bool),

    Text((usize, usize), String),

    Arrow((usize, usize), (usize, usize), usize),

    EllipseArrows(EllipseCoords),

    FilledEllipse(crate::shape::ellipse::Ellipse),

    Ellipse(OrientedEllipse, usize)

}

impl Mark {

    pub fn enclosing_rect(&self) -> (Coord, Size) {
        /*match self {
        Self::Cross(center, sz, u8),

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

        Arrow((usize, usize), (usize, usize), usize, u8),

        EllipseArrows(EllipseCoords),

        Ellipse(OrientedEllipse, usize, u8)*/
        unimplemented!()
    }
    
}

impl From<(usize, usize, usize, usize)> for Mark {

    fn from(r : (usize, usize, usize, usize)) -> Self {
        Mark::Rect((r.0, r.1), (r.2, r.3))
    }

}

// Wait for stabilization of f64 const fn here
// const fn half_pi() -> f64 { std::f64::consts::PI / 2.0 }
// const HALF_PI : f64 = half_pi();
// const HALF_PI : f64 = 1.570796327;

/*pub fn draw_square(
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
}*/

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

pub fn mark_window_with_color(win : &mut WindowMut<u8>, pts : &[(usize, usize)], color : u8) {
    let orig_w = win.original_width();
    mark_slice_with_color(
        win.slice_mut(),
        pts, 
        color, 
        orig_w
    );
}

/// Overwrite pixels in the buffer by informing (x, y, intensity) triples.
pub fn mark_slice_with_color(
    buf : &mut [u8],
    pts : &[(usize, usize)],
    val : u8,
    col_stride : usize
) {
    for (y, x) in pts {
        if let Some(p) = buf.get_mut(col_stride * y + x) {
            *p = val;
        }
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
            *p = val;
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

    if src.0 >= img_dims.0 || dst.0 >= img_dims.0 {
        return;
    }

    if src.1 >= img_dims.1 || dst.1 >= img_dims.1 {
        return;
    }

    // Draw straight horizontal line (if applicable)
    if src.0 == dst.0 {
        for c in src.1.min(dst.1)..=(src.1.max(dst.1)) {
            buf[src.0*ncol + c] = color;
        }
        return;
    }

    // Draw straight vertical line (if applicable)
    if src.1 == dst.1 {
        for r in src.0.min(dst.0)..=(src.0.max(dst.0)) {
            buf[r*ncol + src.1] = color;
        }
        return;
    }

    // Draw non-straight line. A bit more costly, since we must calculate the
    // inclination angle, and fill all pixels across the diagonal.
    let (dist, theta) = index::index_distance(src, dst, nrow);
    let d_max = dist as usize;
    for i in 0..=d_max {
        let x_incr = theta.cos() * i as f64;
        let y_incr = theta.sin() * i as f64;
        let x_pos = (src.1 as i32 + x_incr as i32) as usize;
        let y_pos = (src.0 as i32 - y_incr as i32) as usize;
        buf[y_pos*ncol + x_pos] = color;
    }
}

/*#[test]
fn test_line() {

    let mut buf : [u8; 100] = [0; 100];

    // Drawn an X starting the trace from the upper portion
    draw_line(&mut buf, (10, 10), (2,2), (9, 9), 255);
    draw_line(&mut buf, (10, 10), (2,9), (9, 2), 255);

    let img = Image::new_from_slice(&buf, 10);
    println!("{}", img);

    // assert!(buf[3*10 + 3] == 1);
    // assert!(buf[4*10 + 4] == 1);
}*/

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

// Based on https://github.com/dhepper/font8x8
pub fn write_char(mut c : char, rows : &mut [&mut [u8]]) {
    if c as u8 >= 128 {
        c = '?';
    }
    let bitmap = FONT_8_BASIC[c as usize];
    let mut set : u8 = 0;
    for i in 0..8 {
        for j in 0..8 {
            set = bitmap[i] & 1 << j;
            rows[i][j] = if bitmap[i] & 1 << j == 0 { 0 } else { 255 };
        }
    }
}

pub fn draw_label<S>(img : &mut Image<u8, S>, msg : &str, tl : (usize, usize))
where
    S : StorageMut<u8>
{
    let mut sub_w = img.window_mut(tl, (img.height() - tl.0, img.width() - tl.1)).unwrap();
    let mut curr_line = 0;
    let mut curr_col = 0;
    for c in msg.chars() {
        if c == '\n' {
            curr_line += 1;
            curr_col = 0;
        } else {
            if let Some(mut w) = sub_w.window_mut((curr_line*8, curr_col*8), (8, 8)) {
                write_char(c, &mut w.rows_mut().collect::<Vec<_>>()[..]);
            }
            curr_col += 1;
        }
    }
}

#[test]
fn write_image() {
    let mut img = Image::new_constant(8, 8*7, 0u8);
    for (c, mut win) in "message".chars().zip(img.windows_mut((8, 8))) {
        write_char(c, &mut win.rows_mut().collect::<Vec<_>>()[..]);
    }
    img.show();
}

// Based on https://github.com/dhepper/font8x8
const FONT_8_BASIC : [[u8; 8]; 128] = [
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+0000 (nul)
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+0001
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+0002
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+0003
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+0004
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+0005
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+0006
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+0007
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+0008
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+0009
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+000A
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+000B
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+000C
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+000D
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+000E
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+000F
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+0010
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+0011
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+0012
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+0013
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+0014
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+0015
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+0016
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+0017
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+0018
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+0019
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+001A
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+001B
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+001C
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+001D
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+001E
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+001F
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+0020 (space)
    [ 0x18, 0x3C, 0x3C, 0x18, 0x18, 0x00, 0x18, 0x00],   // U+0021 (!)
    [ 0x36, 0x36, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+0022 (")
    [ 0x36, 0x36, 0x7F, 0x36, 0x7F, 0x36, 0x36, 0x00],   // U+0023 (#)
    [ 0x0C, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x0C, 0x00],   // U+0024 ($)
    [ 0x00, 0x63, 0x33, 0x18, 0x0C, 0x66, 0x63, 0x00],   // U+0025 (%)
    [ 0x1C, 0x36, 0x1C, 0x6E, 0x3B, 0x33, 0x6E, 0x00],   // U+0026 (&)
    [ 0x06, 0x06, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+0027 (')
    [ 0x18, 0x0C, 0x06, 0x06, 0x06, 0x0C, 0x18, 0x00],   // U+0028 (()
    [ 0x06, 0x0C, 0x18, 0x18, 0x18, 0x0C, 0x06, 0x00],   // U+0029 ())
    [ 0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00],   // U+002A (*)
    [ 0x00, 0x0C, 0x0C, 0x3F, 0x0C, 0x0C, 0x00, 0x00],   // U+002B (+)
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x06],   // U+002C (,)
    [ 0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00, 0x00],   // U+002D (-)
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x00],   // U+002E (.)
    [ 0x60, 0x30, 0x18, 0x0C, 0x06, 0x03, 0x01, 0x00],   // U+002F (/)
    [ 0x3E, 0x63, 0x73, 0x7B, 0x6F, 0x67, 0x3E, 0x00],   // U+0030 (0)
    [ 0x0C, 0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x3F, 0x00],   // U+0031 (1)
    [ 0x1E, 0x33, 0x30, 0x1C, 0x06, 0x33, 0x3F, 0x00],   // U+0032 (2)
    [ 0x1E, 0x33, 0x30, 0x1C, 0x30, 0x33, 0x1E, 0x00],   // U+0033 (3)
    [ 0x38, 0x3C, 0x36, 0x33, 0x7F, 0x30, 0x78, 0x00],   // U+0034 (4)
    [ 0x3F, 0x03, 0x1F, 0x30, 0x30, 0x33, 0x1E, 0x00],   // U+0035 (5)
    [ 0x1C, 0x06, 0x03, 0x1F, 0x33, 0x33, 0x1E, 0x00],   // U+0036 (6)
    [ 0x3F, 0x33, 0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x00],   // U+0037 (7)
    [ 0x1E, 0x33, 0x33, 0x1E, 0x33, 0x33, 0x1E, 0x00],   // U+0038 (8)
    [ 0x1E, 0x33, 0x33, 0x3E, 0x30, 0x18, 0x0E, 0x00],   // U+0039 (9)
    [ 0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x00],   // U+003A (:)
    [ 0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x06],   // U+003B (;)
    [ 0x18, 0x0C, 0x06, 0x03, 0x06, 0x0C, 0x18, 0x00],   // U+003C (<)
    [ 0x00, 0x00, 0x3F, 0x00, 0x00, 0x3F, 0x00, 0x00],   // U+003D (=)
    [ 0x06, 0x0C, 0x18, 0x30, 0x18, 0x0C, 0x06, 0x00],   // U+003E (>)
    [ 0x1E, 0x33, 0x30, 0x18, 0x0C, 0x00, 0x0C, 0x00],   // U+003F (?)
    [ 0x3E, 0x63, 0x7B, 0x7B, 0x7B, 0x03, 0x1E, 0x00],   // U+0040 (@)
    [ 0x0C, 0x1E, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x00],   // U+0041 (A)
    [ 0x3F, 0x66, 0x66, 0x3E, 0x66, 0x66, 0x3F, 0x00],   // U+0042 (B)
    [ 0x3C, 0x66, 0x03, 0x03, 0x03, 0x66, 0x3C, 0x00],   // U+0043 (C)
    [ 0x1F, 0x36, 0x66, 0x66, 0x66, 0x36, 0x1F, 0x00],   // U+0044 (D)
    [ 0x7F, 0x46, 0x16, 0x1E, 0x16, 0x46, 0x7F, 0x00],   // U+0045 (E)
    [ 0x7F, 0x46, 0x16, 0x1E, 0x16, 0x06, 0x0F, 0x00],   // U+0046 (F)
    [ 0x3C, 0x66, 0x03, 0x03, 0x73, 0x66, 0x7C, 0x00],   // U+0047 (G)
    [ 0x33, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x33, 0x00],   // U+0048 (H)
    [ 0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00],   // U+0049 (I)
    [ 0x78, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E, 0x00],   // U+004A (J)
    [ 0x67, 0x66, 0x36, 0x1E, 0x36, 0x66, 0x67, 0x00],   // U+004B (K)
    [ 0x0F, 0x06, 0x06, 0x06, 0x46, 0x66, 0x7F, 0x00],   // U+004C (L)
    [ 0x63, 0x77, 0x7F, 0x7F, 0x6B, 0x63, 0x63, 0x00],   // U+004D (M)
    [ 0x63, 0x67, 0x6F, 0x7B, 0x73, 0x63, 0x63, 0x00],   // U+004E (N)
    [ 0x1C, 0x36, 0x63, 0x63, 0x63, 0x36, 0x1C, 0x00],   // U+004F (O)
    [ 0x3F, 0x66, 0x66, 0x3E, 0x06, 0x06, 0x0F, 0x00],   // U+0050 (P)
    [ 0x1E, 0x33, 0x33, 0x33, 0x3B, 0x1E, 0x38, 0x00],   // U+0051 (Q)
    [ 0x3F, 0x66, 0x66, 0x3E, 0x36, 0x66, 0x67, 0x00],   // U+0052 (R)
    [ 0x1E, 0x33, 0x07, 0x0E, 0x38, 0x33, 0x1E, 0x00],   // U+0053 (S)
    [ 0x3F, 0x2D, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00],   // U+0054 (T)
    [ 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x3F, 0x00],   // U+0055 (U)
    [ 0x33, 0x33, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00],   // U+0056 (V)
    [ 0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00],   // U+0057 (W)
    [ 0x63, 0x63, 0x36, 0x1C, 0x1C, 0x36, 0x63, 0x00],   // U+0058 (X)
    [ 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x0C, 0x1E, 0x00],   // U+0059 (Y)
    [ 0x7F, 0x63, 0x31, 0x18, 0x4C, 0x66, 0x7F, 0x00],   // U+005A (Z)
    [ 0x1E, 0x06, 0x06, 0x06, 0x06, 0x06, 0x1E, 0x00],   // U+005B ([)
    [ 0x03, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x40, 0x00],   // U+005C (\)
    [ 0x1E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x1E, 0x00],   // U+005D (])
    [ 0x08, 0x1C, 0x36, 0x63, 0x00, 0x00, 0x00, 0x00],   // U+005E (^)
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF],   // U+005F (_)
    [ 0x0C, 0x0C, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+0060 (`)
    [ 0x00, 0x00, 0x1E, 0x30, 0x3E, 0x33, 0x6E, 0x00],   // U+0061 (a)
    [ 0x07, 0x06, 0x06, 0x3E, 0x66, 0x66, 0x3B, 0x00],   // U+0062 (b)
    [ 0x00, 0x00, 0x1E, 0x33, 0x03, 0x33, 0x1E, 0x00],   // U+0063 (c)
    [ 0x38, 0x30, 0x30, 0x3e, 0x33, 0x33, 0x6E, 0x00],   // U+0064 (d)
    [ 0x00, 0x00, 0x1E, 0x33, 0x3f, 0x03, 0x1E, 0x00],   // U+0065 (e)
    [ 0x1C, 0x36, 0x06, 0x0f, 0x06, 0x06, 0x0F, 0x00],   // U+0066 (f)
    [ 0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x1F],   // U+0067 (g)
    [ 0x07, 0x06, 0x36, 0x6E, 0x66, 0x66, 0x67, 0x00],   // U+0068 (h)
    [ 0x0C, 0x00, 0x0E, 0x0C, 0x0C, 0x0C, 0x1E, 0x00],   // U+0069 (i)
    [ 0x30, 0x00, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E],   // U+006A (j)
    [ 0x07, 0x06, 0x66, 0x36, 0x1E, 0x36, 0x67, 0x00],   // U+006B (k)
    [ 0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00],   // U+006C (l)
    [ 0x00, 0x00, 0x33, 0x7F, 0x7F, 0x6B, 0x63, 0x00],   // U+006D (m)
    [ 0x00, 0x00, 0x1F, 0x33, 0x33, 0x33, 0x33, 0x00],   // U+006E (n)
    [ 0x00, 0x00, 0x1E, 0x33, 0x33, 0x33, 0x1E, 0x00],   // U+006F (o)
    [ 0x00, 0x00, 0x3B, 0x66, 0x66, 0x3E, 0x06, 0x0F],   // U+0070 (p)
    [ 0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x78],   // U+0071 (q)
    [ 0x00, 0x00, 0x3B, 0x6E, 0x66, 0x06, 0x0F, 0x00],   // U+0072 (r)
    [ 0x00, 0x00, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x00],   // U+0073 (s)
    [ 0x08, 0x0C, 0x3E, 0x0C, 0x0C, 0x2C, 0x18, 0x00],   // U+0074 (t)
    [ 0x00, 0x00, 0x33, 0x33, 0x33, 0x33, 0x6E, 0x00],   // U+0075 (u)
    [ 0x00, 0x00, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00],   // U+0076 (v)
    [ 0x00, 0x00, 0x63, 0x6B, 0x7F, 0x7F, 0x36, 0x00],   // U+0077 (w)
    [ 0x00, 0x00, 0x63, 0x36, 0x1C, 0x36, 0x63, 0x00],   // U+0078 (x)
    [ 0x00, 0x00, 0x33, 0x33, 0x33, 0x3E, 0x30, 0x1F],   // U+0079 (y)
    [ 0x00, 0x00, 0x3F, 0x19, 0x0C, 0x26, 0x3F, 0x00],   // U+007A (z)
    [ 0x38, 0x0C, 0x0C, 0x07, 0x0C, 0x0C, 0x38, 0x00],   // U+007B ({)
    [ 0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18, 0x00],   // U+007C (|)
    [ 0x07, 0x0C, 0x0C, 0x38, 0x0C, 0x0C, 0x07, 0x00],   // U+007D (})
    [ 0x6E, 0x3B, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],   // U+007E (~)
    [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]    // U+007F
];

/// Draw an axis-aligned ellipse (After Davies (2005) Fig. 6.6). The generated
/// points can go through a pure rotation matrix to generate the generic affine-projection
/// ellipse.
pub fn ellipse_draw(win : &mut WindowMut<u8>, a : f32, b : f32, color : u8) {
    for i in 0..360 {
        let theta = i as f32 * 2. * (std::f32::consts::PI / 360.);
        let x = a * theta.sin();
        let y = b * theta.cos();
        let h = win.height();
        win[((h as f32 - y) as usize, x as usize)] = color;
    }
}

// Based on https://rosettacode.org/wiki/Bitmap/Bresenham's_line_algorithms
mod bresenham {
    struct Point {
        x: i32,
        y: i32
    }

    fn get_coordinates(x1: i32, y1: i32, x2: i32, y2: i32) -> Vec<Point> {
        let mut coordinates: Vec<Point> = vec![];
        let dx:i32 = i32::abs(x2 - x1);
        let dy:i32 = i32::abs(y2 - y1);
        let sx:i32 = { if x1 < x2 { 1 } else { -1 } };
        let sy:i32 = { if y1 < y2 { 1 } else { -1 } };

        let mut error:i32 = (if dx > dy  { dx } else { -dy }) / 2 ;
        let mut current_x:i32 = x1;
        let mut current_y:i32 = y1;
        loop {
            coordinates.push(Point { x : current_x, y: current_y });

            if current_x == x2 && current_y == y2 { break; }

            let error2:i32 = error;

            if error2 > -dx {
                error -= dy;
                current_x += sx;
            }
            if error2 < dy {
                error += dx;
                current_y += sy;
            }
        }
        coordinates
    }

    fn draw_line(line: std::vec::Vec<Point>, width: i32, height: i32) {
        for col in 0..height {
            for row in 0..width {
                let is_point_in_line = line.iter().any(| point| point.x == row && point.y == col);
                match is_point_in_line {
                    true => print!("@"),
                    _ => print!(".")
                };
            }
            print!("\n");
        }
    }

    /*
    // Based on https://rosettacode.org/wiki/Bitmap/B%C3%A9zier_curves/Quadratic
    void quad_bezier(
            image img,
            unsigned int x1, unsigned int y1,
            unsigned int x2, unsigned int y2,
            unsigned int x3, unsigned int y3,
            color_component r,
            color_component g,
            color_component b )
    {
        unsigned int i;
        double pts[N_SEG+1][2];
        for (i=0; i <= N_SEG; ++i)
        {
            double t = (double)i / (double)N_SEG;
            double a = pow((1.0 - t), 2.0);
            double b = 2.0 * t * (1.0 - t);
            double c = pow(t, 2.0);
            double x = a * x1 + b * x2 + c * x3;
            double y = a * y1 + b * y2 + c * y3;
            pts[i][0] = x;
            pts[i][1] = y;
        }

    #if 0
        /* draw only points */
        for (i=0; i <= N_SEG; ++i)
        {
            plot( pts[i][0],
                  pts[i][1] );
        }
    #else
        /* draw segments */
        for (i=0; i < N_SEG; ++i)
        {
            int j = i + 1;
            line( pts[i][0], pts[i][1],
                  pts[j][0], pts[j][1] );
        }
    #endif
    }
    #undef plot
    #undef line

    // Based on https://rosettacode.org/wiki/Bitmap/B%C3%A9zier_curves
    void cubic_bezier(
           	image img,
            unsigned int x1, unsigned int y1,
            unsigned int x2, unsigned int y2,
            unsigned int x3, unsigned int y3,
            unsigned int x4, unsigned int y4,
            color_component r,
            color_component g,
            color_component b )
    {
        unsigned int i;
        double pts[N_SEG+1][2];
        for (i=0; i <= N_SEG; ++i)
        {
            double t = (double)i / (double)N_SEG;

            double a = pow((1.0 - t), 3.0);
            double b = 3.0 * t * pow((1.0 - t), 2.0);
            double c = 3.0 * pow(t, 2.0) * (1.0 - t);
            double d = pow(t, 3.0);

            double x = a * x1 + b * x2 + c * x3 + d * x4;
            double y = a * y1 + b * y2 + c * y3 + d * y4;
            pts[i][0] = x;
            pts[i][1] = y;
        }

    #if 0
        /* draw only points */
        for (i=0; i <= N_SEG; ++i)
        {
            plot( pts[i][0],
                  pts[i][1] );
        }
    #else
        /* draw segments */
        for (i=0; i < N_SEG; ++i)
        {
            int j = i + 1;
	    line( pts[i][0], pts[i][1],
                  pts[j][0], pts[j][1] );
        }
    #endif
    }
    #undef plot
    #undef line
    */

    // Based on https://rosettacode.org/wiki/Bitmap/Midpoint_circle_algorithm
    /*void raster_circle(
        image img,
        unsigned int x0,
        unsigned int y0,
        unsigned int radius,
        color_component r,
        color_component g,
        color_component b )
{
    int f = 1 - radius;
    int ddF_x = 0;
    int ddF_y = -2 * radius;
    int x = 0;
    int y = radius;

    plot(x0, y0 + radius);
    plot(x0, y0 - radius);
    plot(x0 + radius, y0);
    plot(x0 - radius, y0);

    while(x < y)
    {
        if(f >= 0)
        {
            y--;
            ddF_y += 2;
            f += ddF_y;
        }
        x++;
        ddF_x += 2;
        f += ddF_x + 1;
        plot(x0 + x, y0 + y);
        plot(x0 - x, y0 + y);
        plot(x0 + x, y0 - y);
        plot(x0 - x, y0 - y);
        plot(x0 + y, y0 + x);
        plot(x0 - y, y0 + x);
        plot(x0 + y, y0 - x);
        plot(x0 - y, y0 - x);
    }
}*/

}

// Based on https://rosettacode.org/wiki/Xiaolin_Wu%27s_line_algorithm
// Draw antialiased lines
/*
inline void _dla_changebrightness(rgb_color_p from,
				  rgb_color_p to, float br)
{
  if ( br > 1.0 ) br = 1.0;
  /* linear... Maybe something more complex could give better look */
  to->red = br * (float)from->red;
  to->green = br * (float)from->green;
  to->blue = br * (float)from->blue;
}

#define plot_(X,Y,D) do{ rgb_color f_;				\
  f_.red = r; f_.green = g; f_.blue = b;			\
  _dla_plot(img, (X), (Y), &f_, (D)) ; }while(0)

inline void _dla_plot(image img, int x, int y, rgb_color_p col, float br)
{
  rgb_color oc;
  _dla_changebrightness(col, &oc, br);
  put_pixel_clip(img, x, y, oc.red, oc.green, oc.blue);
}

#define ipart_(X) ((int)(X))
#define round_(X) ((int)(((double)(X))+0.5))
#define fpart_(X) (((double)(X))-(double)ipart_(X))
#define rfpart_(X) (1.0-fpart_(X))

#define swap_(a, b) do{ __typeof__(a) tmp;  tmp = a; a = b; b = tmp; }while(0)
void draw_line_antialias(
  image img,
  unsigned int x1, unsigned int y1,
  unsigned int x2, unsigned int y2,
  color_component r,
  color_component g,
  color_component b )
{
  double dx = (double)x2 - (double)x1;
  double dy = (double)y2 - (double)y1;
  if ( fabs(dx) > fabs(dy) ) {
    if ( x2 < x1 ) {
      swap_(x1, x2);
      swap_(y1, y2);
    }
    double gradient = dy / dx;
    double xend = round_(x1);
    double yend = y1 + gradient*(xend - x1);
    double xgap = rfpart_(x1 + 0.5);
    int xpxl1 = xend;
    int ypxl1 = ipart_(yend);
    plot_(xpxl1, ypxl1, rfpart_(yend)*xgap);
    plot_(xpxl1, ypxl1+1, fpart_(yend)*xgap);
    double intery = yend + gradient;

    xend = round_(x2);
    yend = y2 + gradient*(xend - x2);
    xgap = fpart_(x2+0.5);
    int xpxl2 = xend;
    int ypxl2 = ipart_(yend);
    plot_(xpxl2, ypxl2, rfpart_(yend) * xgap);
    plot_(xpxl2, ypxl2 + 1, fpart_(yend) * xgap);

    int x;
    for(x=xpxl1+1; x < xpxl2; x++) {
      plot_(x, ipart_(intery), rfpart_(intery));
      plot_(x, ipart_(intery) + 1, fpart_(intery));
      intery += gradient;
    }
  } else {
    if ( y2 < y1 ) {
      swap_(x1, x2);
      swap_(y1, y2);
    }
    double gradient = dx / dy;
    double yend = round_(y1);
    double xend = x1 + gradient*(yend - y1);
    double ygap = rfpart_(y1 + 0.5);
    int ypxl1 = yend;
    int xpxl1 = ipart_(xend);
    plot_(xpxl1, ypxl1, rfpart_(xend)*ygap);
    plot_(xpxl1 + 1, ypxl1, fpart_(xend)*ygap);
    double interx = xend + gradient;

    yend = round_(y2);
    xend = x2 + gradient*(yend - y2);
    ygap = fpart_(y2+0.5);
    int ypxl2 = yend;
    int xpxl2 = ipart_(xend);
    plot_(xpxl2, ypxl2, rfpart_(xend) * ygap);
    plot_(xpxl2 + 1, ypxl2, fpart_(xend) * ygap);

    int y;
    for(y=ypxl1+1; y < ypxl2; y++) {
      plot_(ipart_(interx), y, rfpart_(interx));
      plot_(ipart_(interx) + 1, y, fpart_(interx));
      interx += gradient;
    }
  }
}
#undef swap_
#undef plot_
#undef ipart_
#undef fpart_
#undef round_
#undef rfpart_
*/
