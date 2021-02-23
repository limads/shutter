// use nalgebra::*;
// use super::image::*;

/// An image frame can be economically represented by
/// a Vec<Patch> where patch is a monochromatic color
/// blob represented by its (row, col) center and average color.
/// The reproduction of the image requires a single iteration
/// through its (row, col) indices. At each (row, col), take
/// the closest Patch in the Vec<Patch> collection. To build
/// the collection, one must define a threshold of u8 pixel
/// difference that define the maximum pixel horizontal
/// discrepancy for which a new patch is added to the collection.
#[derive(Copy, Clone, Debug)]
pub struct Patch {

    /// Patch center, as fractional coordinates (row, col)
    center : (f64, f64),

    /// Number of pixels aggregated
    n : usize,

    /// Average pixel color of the patch.
    color : u8,

    // Rectangular distance away from the patch
    // center that fully encloses all its pixels.
    extension : (usize, usize),

    last_pos : (usize, usize)
}

impl Patch {

    pub fn new(center : (usize, usize), color : u8) -> Self {
        let center = (center.0 as f64, center.1 as f64);
        let color = color;
        let n = 1;
        let extension = (0, 0);
        let last_pos = (0, 0);
        Self{ center, n, color, extension, last_pos }
    }

    // Patches are always extended from the top-left to the
    // bottom-right. Panic if a value is informed smaller than the
    // current patch center over both dimensions.
    pub fn increment(&mut self, pos: (usize, usize), color : u8) {
        //println!("center = {:?}; pos = {:?}", self.center, pos);
        assert!(pos.0 >= self.center.0 as usize || pos.1 >= self.center.1 as usize, "New pixel smaller than current center");
        assert!(pos.0 >= self.last_pos.0 || pos.1 >= self.last_pos.1 as usize, "New position smaller than previous position");
        self.n += 1;
        let n = self.n as f64;
        let dist_y = pos.0 as f64 - self.center.0;
        let dist_x = pos.1 as f64 - self.center.1;
        self.center.0 = (1./n) * dist_y + ((n-1.)/n) * self.center.0;
        self.center.1 = (1./n) * dist_x + ((n-1.)/n) * self.center.1;
        self.color = ((1./n)*color as f64 + (self.color as f64)*(n-1.)/n) as u8;
        if dist_y > self.extension.0 as f64 {
            self.extension.0 = dist_y as usize;
        }
        if dist_x > self.extension.1 as f64 {
            self.extension.1 = dist_x as usize
        }
        self.last_pos = pos;
    }

    /// Returns the smallest rectangle that fully
    /// encloses the current patch.
    pub fn enclosing_rect(&self) -> ((usize, usize), (usize, usize)) {
        //assert!(self.center.0 - self.extension.0 as f64 > 0.0);
        //assert!(self.center.1 - self.extension.1 as f64 > 0.0);
        let tl_y = (self.center.0 - self.extension.0 as f64).max(0.0);
        let tl_x = (self.center.1 - self.extension.1 as f64).max(0.0);
        //println!("tl: {}", );
        //println!("tr: {}", );
        let tl_yu = (tl_y as usize).min(self.last_pos.0);
        let tl_xu = (tl_x as usize).min(self.last_pos.1);
        let h = 2 * self.extension.0;
        let w = 2 * self.extension.1;
        //assert!(h <= self.last_pos.0);
        //assert!(w <= self.last_pos.1);
        ((tl_yu, tl_xu), (h, w))
    }

    /// Given a new frame at img, iterate only over the nearby external
    /// boundary pixels of the enclosing rectangle and decide if the color,
    /// center or dimensions of the enclosing rectangle should be changed
    /// (if the patch expanded in any direction). Also, iterate over the
    /// internal boundary pixels. If any changes are detected, iterate
    /// depper into the segment and verify if those pixels should also be updated.
    pub fn update(&mut self, _img : &[u8]) {
        unimplemented!()
    }

    pub fn color(&self) -> u8 {
        self.color
    }

    pub fn center(&self) -> (f64, f64) {
        self.center
    }

    pub fn dist(&self, coord : (usize, usize)) -> f64 {
        ((self.center.0 - coord.0 as f64).powf(2.) +
            (self.center.1 - coord.1 as f64).powf(2.)).sqrt()
    }

}

#[derive(Clone, Debug)]
pub struct PatchGroup {

    patches : Vec<Patch>,

    threshold : u8,

    /// Dense frame dimension (rows, cols) from which this group was calculated
    dim : (usize, usize)
}

impl PatchGroup {

    pub fn segment(img : &[u8], ncols : usize, threshold : u8) -> Self {
        let mut patches = Vec::new();
        for (i, px) in img.iter().enumerate() {
            let r = i / ncols;
            let c = i % ncols;
            if i == 0 {
                patches.push(Patch::new((r, c), *px));
            } else {
                let last_patch = &mut patches.last_mut().unwrap();
                if last_patch.color() - px < threshold {
                    last_patch.increment((r, c), *px);
                } else {
                    let mut any_close = false;
                    if r >= 2 && c >= 2 {
                        for prev_patch in patches.iter_mut().rev() {
                            let close_row = prev_patch.last_pos.0 - r <= 2;
                            let close_col = prev_patch.last_pos.1 - c <= 2;
                            if (close_row || close_col) && prev_patch.color() - px < threshold {
                                if !any_close {
                                    any_close = true;
                                    prev_patch.increment((r, c), *px);
                                }
                            }
                        }
                    }
                    if !any_close {
                        patches.push(Patch::new((r, c), *px));
                    }
                }
            }
        }
        Self{ patches, threshold, dim : (img.len() / ncols, ncols) }
    }

    /*pub fn reconstruct(&self) -> Image<'_> {
        //let mut img = DMatrix::zeros(self.dim.0, self.dim.1);
        let mut img_cont = vec![0_u8; self.dim.0 * self.dim.1];
        for r in 0..self.dim.0 {
            for c in 0..self.dim.1 {
                //let dist = ((r as f64).powf(2.) + (c as f64).powf(2.)).sqrt();
                let mut closest_patch = self.patches[0];
                let mut closest_dist = closest_patch.dist((r, c));
                for patch in self.patches.iter().skip(1) {
                    let dist = patch.dist((r, c));
                    if dist < closest_dist {
                        closest_patch = *patch;
                        closest_dist = dist;
                    }
                }
                //img[(r, c)] = closest_patch.color();
                img_cont[r*self.dim.1 + c] = closest_patch.color();
            }
        }
        //img.transpose()

        Image::from_vec(img_cont, self.dim.1)
    }*/

    pub fn patches(&self) -> &[Patch] {
        &self.patches[..]
    }

}

#[test]
fn patch() {

    /*use super::*;
    use super::image::*;

    let data : Vec<u8> = vec![
        255, 255, 255, 255, 255, 255, 255, 255,
        0, 0, 0, 0, 0, 0, 0, 0
    ];
    let image = Image::from_vec(data, 4);
    let group = PatchGroup::segment(image.pixels(), 4, 127);
    println!("{:?}", image);
    println!("{:?}", group);
    println!("{:?}", group.reconstruct());
    for p in group.patches().iter() {
        println!("{:?}", p.enclosing_rect());
    }*/
}


