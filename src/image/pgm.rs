// extern crate gtk;
// extern crate gio;
// extern crate gdk;

use nalgebra::*;
use nalgebra::storage::*;
use std::iter::Iterator;
use std::fmt::Display;
use std::convert::TryFrom;
use std::ops::Range;
use super::*;
use std::ffi;
use std::mem;

// To get images over [-1, 1], transform to -256, 1/280
// To get images over [0, 1], transform to -128, 1/128

/*pub fn load_images<'a>(
    srcs : impl Iterator<Item=&'a str>,
    dims : (usize, usize)
) -> Result<Vec<DMatrix<f64>>, String> {
    let mut imgs = Vec::new();
    for src in srcs {
        let mut opts = SamplingOptions::new_standard(dims);
        opts.set_depth(-128.0, 1. / 128.0);
        let mut img : DMatrix<f64> = DMatrix::from_element(dims.0, dims.1, 0.0);
        img.load::<PngBuffer>(src, Some(opts.clone()))
            .map_err(|e| { format!("{}", e) })?;
        imgs.push(img);
    }
    Ok(imgs)
}

pub fn save_images<'a>(
    imgs : Vec<DMatrix<f64>>,
    dsts : Vec<String>,
    dims : (usize, usize)
) -> Result<(), String> {
    if imgs.len() != dsts.len() {
        return Err(format!("Informed {} names but only {} matrices", dsts.len(), imgs.len()) );
    }
    for (img, dst) in imgs.iter().zip(dsts.iter()) {
        let mut opts = SamplingOptions::new_standard(dims);
        opts.set_region((0, 0), img.shape());
        opts.set_depth(1.0, 128.0);
        img.save::<PngBuffer>(dst, Some(opts))
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

pub fn image_filter<F, E>(
    f : F,
    src : &str,
    dst : &str,
    dims : (usize, usize)
) -> Result<(), String>
    where
        F : Fn(&DMatrix<f64>)->Result<DMatrix<f64>, E>,
        E : Display
{
    let mut opts = SamplingOptions::new_standard(dims);
    opts.set_depth(-128.0, 1. / 128.0);
    let mut img : DMatrix<f64> = DMatrix::from_element(dims.0, dims.1, 0.0);
    img.load::<PngBuffer>(src, Some(opts.clone()))
        .map_err(|e| { format!("{}", e) })?;
    match f(&img) {
        Ok(img) => {
            let mut opts = opts.clone();
            opts.set_region((0, 0), img.shape());
            opts.set_depth(1.0, 128.0);
            img.save::<PngBuffer>(dst, Some(opts)).map_err(|e| e.to_string())?;
        },
        Err(e) => {
            return Err(format!("{}", e))
        }
    }
    Ok(())
}*/

pub fn build_pgm_string_from_slice<N>(m : &[N], ncol : usize) -> String 
where
    f64 : From<N>,
    N : Scalar + Copy
{
    let mut pgm = format!("P2\n");
    pgm += &format!("{} {}\n", m.len() / ncol, ncol);
    pgm += &format!("255\n");
    for px in m.iter() {
        pgm += &format!("{}\n", (1. + f64::from(*px) * 127.).max(0.0).min(255.0) as u8);
    }
    pgm
}

pub fn build_pgm_string<N, S>(m : &Matrix<N, Dynamic, Dynamic, S>) -> String
where
    f64 : From<N>,
    N : Scalar + Copy,
    S : Storage<N, Dynamic, Dynamic>
{
    let mut pgm = format!("P2\n");
    pgm += &format!("{} {}\n", m.nrows(), m.ncols());
    pgm += &format!("255\n");
    for px in m.iter() {
        pgm += &format!("{}\n", (1. + f64::from(*px) * 127.).max(0.0).min(255.0) as u8);
    }
    pgm
}

/// Reference: https://en.wikipedia.org/wiki/Netpbm#File_formats
pub fn print_pgm<N, S>(m : &Matrix<N, Dynamic, Dynamic, S>) 
where
    f64 : From<N>,
    N : Scalar + Copy,
    S : Storage<N, Dynamic, Dynamic>
{
    println!("{}", pgm::build_pgm_string(&m));
}
