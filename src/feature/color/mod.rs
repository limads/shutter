use crate::image::Window;

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
