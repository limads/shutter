use std::io::BufWriter;
use ::image::codecs::png;
use crate::image::ImageBuf;
use ::image::{ImageDecoder, ColorType, ExtendedColorType};
use std::fs::File;
use std::io::{Read, Write};
use std::iter::FromIterator;

pub fn file_dimensions(path : &str) -> Option<(usize, usize)> {
    let mut f = File::open(path).ok()?;
    let dec = png::PngDecoder::new(f).ok()?;
    let dims = dec.dimensions();
    if dims.0 > 0 && dims.1 > 0 {
        Some((dims.0 as usize, dims.1 as usize))
    } else {
        None
    }
}

pub fn encode_to_file(dec : ImageBuf<u8>, path : &str) -> Result<(),&'static str> {
    let mut f = File::create(path)
        .map_err(|_| { "Error creating file." })?;
    let buff : Vec<u8> = encode(dec)?;
    f.write(&buff[..]).map_err(
        |_| { "Error writing to file." })?;
    Ok(())
}

pub fn decode_from_file(
    path : &str
) -> Result<ImageBuf<u8>, &'static str> {
    let mut buffer = Vec::new();
    let mut f = File::open(path)
        .map_err(|_| { "Error opening file" })?;
    let _ = f.read_to_end(&mut buffer)
        .map_err(|_| { "Error reading buffer" })?;
    decode(buffer)
}

pub fn decode(
    data : Vec<u8>
) -> Result<ImageBuf<u8>, &'static str> {

    use ::image::buffer::ConvertBuffer;

    let dec = png::PngDecoder::new(&data[..])
        .map_err(|_| { "Error building decoder" })?;
    // let color_type = dec.color_type();
    // if color_type != ColorType::L8 {
    //    return Err("Image import error: Unsupported color type : {}. Convert image to L8 first.");
    // }

    let nrows = dec.dimensions().1 as usize;
    let ncols = dec.dimensions().0 as usize;
    match dec.original_color_type() {
        ExtendedColorType::L8 => {
            let mut buf = Vec::<u8>::from_iter((0..(nrows * ncols)).map(|_| 0 ));
            let img = dec.read_image(&mut buf[..]).map_err(|_| { "Error reading image" })?;
            Ok(ImageBuf::from_vec(buf, ncols))
        },
        ExtendedColorType::Rgb8 => {
            let mut buf = Vec::<u8>::from_iter((0..(nrows * ncols * 3)).map(|_| 0 ));
            let img = dec.read_image(&mut buf[..]).map_err(|_| { "Error reading image" })?;
            let rgbimg = image::RgbImage::from_raw(ncols as u32, nrows as u32, buf).unwrap();
            let gray_image: image::GrayImage = rgbimg.convert();
            Ok(ImageBuf::from_vec(gray_image.into_raw(), ncols))
        },
        _ => {
            Err("Image import error: Unsupported color type. Convert image to L8/RGB/RGBA first.")
        }
    }

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

}

pub fn encode(
    dec : ImageBuf<u8>
) -> Result<Vec<u8>, &'static str> {
    let mut dst : Vec<u8> = Vec::new();
    {
        let ref mut writ = BufWriter::new(&mut dst);
        let encoder = png::PngEncoder::new(writ);
        encoder.encode(
            dec.slice(),
            dec.width() as u32,
            dec.height() as u32,
            image::ColorType::L8
        ).map_err(|e|{ println!("{:?}", e); "Could not encode image" } )?;
    }
    Ok(dst)
}

