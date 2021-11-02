use std::io::BufWriter;
use ::image::codecs::png;
use crate::image::Image;
use ::image::{ImageDecoder, ColorType};
use std::fs::File;
use std::io::{Read, Write};
use std::iter::FromIterator;

pub fn encode_to_file(dec : Image<u8>, path : &str) -> Result<(),&'static str> {
    let mut f = File::create(path)
        .map_err(|_| { "Error creating file." })?;
    let buff : Vec<u8> = encode(dec)?;
    f.write(&buff[..]).map_err(
        |_| { "Error writing to file." })?;
    Ok(())
}

pub fn decode_from_file(
    path : &str
) -> Result<Image<u8>, &'static str> {
    let mut buffer = Vec::new();
    let mut f = File::open(path)
        .map_err(|_| { "Error opening file" })?;
    let _ = f.read_to_end(&mut buffer)
        .map_err(|_| { "Error reading buffer" })?;
    decode(buffer)
}

pub fn decode(
    data : Vec<u8>
) -> Result<Image<u8>, &'static str> {
    let dec = png::PngDecoder::new(&data[..])
        .map_err(|_| { "Error building decoder" })?;
    let color_type = dec.color_type();
    if color_type != ColorType::L8 {
        return Err("Image import error: Unsupported color type. Convert image to L8 first.");
    }
    let nrows = dec.dimensions().0 as usize;
    let ncols = dec.dimensions().1 as usize;
    let mut buf = Vec::<u8>::from_iter((0..(nrows * ncols)).map(|_| 0 ));
    let img = dec.read_image(&mut buf[..])
        .map_err(|_| { "Error reading image" })?;
    /*match color_type {
        ColorType::L8 => {
            for byte in img.iter() {
                //buf.push( N::from(*byte)*scale/N::from(255.0)+offset );
                buf.push(*byte);
            }
        },
        _ => {
            return Err("Unsupported color type");
        }
    };*/
    Ok(Image::from_vec(buf, ncols))
}

pub fn encode(
    dec : Image<u8>
) -> Result<Vec<u8>, &'static str> {
    let mut dst : Vec<u8> = Vec::new();
    {
        let ref mut writ = BufWriter::new(&mut dst);
        let encoder = png::PngEncoder::new(writ);
        let nrows = dec.len() / dec.width();
        encoder.encode(
            dec.as_slice(),
            dec.width() as u32,
            nrows as u32,
            image::ColorType::L8
        ).map_err(|e|{ println!("{:?}", e); "Could not encode image" } )?;
    }
    Ok(dst)
}

