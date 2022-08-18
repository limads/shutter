// Based on https://rosettacode.org/wiki/Hough_transform

#[allow(clippy::cast_precision_loss)]
#[allow(clippy::clippy::cast_possible_truncation)]
fn hough(image: &ImageGray8, out_width: usize, out_height: usize) -> ImageGray8 {
    let in_width = image.width;
    let in_height = image.height;

    // Allocate accumulation buffer
    let out_height = ((out_height / 2) * 2) as usize;
    let mut accum = ImageGray8 {
        width: out_width,
        height: out_height,
        data: repeat(255).take(out_width * out_height).collect(),
    };

    // Transform extents
    let rmax = (in_width as f64).hypot(in_height as f64);
    let dr = rmax / (out_height / 2) as f64;
    let dth = std::f64::consts::PI / out_width as f64;

    // Process input image in raster order
    for y in 0..in_height {
        for x in 0..in_width {
            let in_idx = y * in_width + x;
            let col = image.data[in_idx];
            if col == 255 {
                continue;
            }

            // Project into rho,theta space
            for jtx in 0..out_width {
                let th = dth * (jtx as f64);
                let r = (x as f64) * (th.cos()) + (y as f64) * (th.sin());

                let iry = out_height as i64 / 2 - (r / (dr as f64) + 0.5).floor() as i64;
                #[allow(clippy::clippy::cast_sign_loss)]
                let out_idx = (jtx as i64 + iry * out_width as i64) as usize;
                let col = accum.data[out_idx];
                if col > 0 {
                    accum.data[out_idx] = col - 1;
                }
            }
        }
    }
    accum
}