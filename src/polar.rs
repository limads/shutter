use std::rc::Rc;
use crate::image::*;
use nalgebra::DMatrix;

/* Rows represent increasing radii from 0.0 up to PolarMap::max_radius from top to bottom
(increased resolution with taller polar image);  Columns represent increasing angle between
[-pi, pi] from left to right (Increased resolution with wider polar image) */
pub struct PolarImage {

    polar : ImageBuf<u8>,

    map : PolarMap

}

impl PolarImage {

    // Returns a "swept" version of the polar image, that ignores
    // all holes.
    pub fn swept(&self) -> ImageBuf<u8> {
        let mut swept = ImageBuf::new_constant(self.polar.height(), self.polar.width(), 0);
        // let mut curr_ixs : Vec<usize> = (0..self.polar.height()).map(|_| 0 ).collect();
        for r in 0..self.polar.height() {
            let mut col_ix = 0;
            for c in 0..self.polar.width() {
                let px = &self.polar[(r, c)];
                if *px > 0 {
                    swept[(r, col_ix)] = *px;
                    col_ix += 1;
                }
            }
        }
        swept
    }

    pub fn image_mut(&mut self) -> &mut ImageBuf<u8> {
        &mut self.polar
    }

    pub fn image_ref(&self) -> &ImageBuf<u8> {
        &self.polar
    }

    pub fn empty_centered(height : usize, width : usize) -> Self {
        let polar = ImageBuf::<u8>::new_constant(height, width, 0);
        let c = (height / 2, width / 2);
        Self { polar, map : PolarMap::new((height, width), c) }
    }

    pub fn map(&self) -> PolarMap {
        self.map.clone()
    }

    // Returns the subregion of the polar image that is complete
    // (i.e. is a valid mapping to the polar image). This is equivalent
    // to the circle inscribed in the original cartesian window. The
    // returned image does not have artifacts due to pixels not being
    // represented in the polar image (i.e regions that are not in the
    // rect inscribed in the outer circle).
    pub fn complete_inscribed_region(&self) -> Window<u8> {

        // Get the smallest projection (x or y) of the 45º vector (the farthest representable
        // pixel in the circumscribed circle).
        let min_dim = std::f32::consts::FRAC_PI_4.sin().min(std::f32::consts::FRAC_PI_4.cos());

        // By taking the rows up to this minimum dimension, all pixels at this radius are represented
        // in the polar image.
        let h = (min_dim * self.polar.height() as f32) as usize;
        self.polar.window((0, 0), (h, self.polar.width())).unwrap()
    }

    pub fn from_centered<S>(w : &Image<u8, S>) -> Self
    where
        S : Storage<u8>
    {
        let map = PolarMap::new(w.size(), (w.height() / 2, w.width() / 2));
        Self::from(w, map)
    }

    pub fn from<S>(w : &Image<u8, S>, map : PolarMap) -> Self
    where
        S : Storage<u8>
    {
        let polar = ImageBuf::<u8>::new_constant(w.height(), w.width(), 0);
        let mut pi = Self { polar, map };
        pi.update(w);
        pi
    }

    // TODO write this into an inverse map with the linear index y*width + x.
    pub fn to_cartesian(&self, cart : &mut Image<u8, impl StorageMut<u8>>) {
        let rad_delta = self.map.max_rad / self.polar.height() as f32;
        // let angle_delta = std::f32::consts::PI / self.polar.width() as f32 / 2.0;
        let half_w = self.polar.width() as f32 / 2.0;
        let angle_delta = (2.0*std::f32::consts::PI) / self.polar.width() as f32;
        for r in 0..self.polar.height() {
            for theta in 0..self.polar.width() {
                let theta_center = -std::f32::consts::PI + theta as f32 * angle_delta;
                // let theta_center = theta as f32 * angle_delta;
                let rad = r as f32 * rad_delta;
                let y = cart.height() - (self.map.center.0 as f32 + theta_center.sin() * rad) as usize;
                let x = (self.map.center.1 as f32 + theta_center.cos() * rad) as usize;
                if let Some(mut px) = cart.get_mut((y, x)) {
                    *px = self.polar[(r, theta)];
                }
            }
        }
    }

    pub fn update<S>(&mut self, w : &Image<u8, S>)
    where
        S : Storage<u8>
    {
        for i in 0..w.height() {
            for j in 0..w.width() {
                let lin_polar = cartesian_to_polar(&self.map.lut, &(i, j), &w.size());
                self.polar.as_mut_slice()[lin_polar] = w[(i, j)];
            }
        }
    }

    pub fn update_from_windows(&mut self, ws : &[Window<'_, u8>]) {
        let (orig_h, orig_w) = ws[0].original_size();
        for w in ws {
            let off = w.offset();
            for i in 0..w.height() {
                for j in 0..w.width() {
                    let ix_off = (off.0 + i, off.1 + j);
                    let lin_polar = cartesian_to_polar(&self.map.lut, &ix_off, &(orig_h, orig_w));
                    self.polar.as_mut_slice()[lin_polar] = w[(i, j)];
                }
            }
        }
    }

    // Generate polar image from subset of windows of a parent window. The
    // windows share a center in the parent window given as the second argument.
    pub fn from_windows(ws : &[Window<'_, u8>], map : PolarMap) -> Self {
        let sz = ws[0].original_size();
        let polar = ImageBuf::<u8>::new_constant(sz.0, sz.1, 0);
        let mut pi = Self { polar, map };
        pi.update_from_windows(ws);
        pi
    }

    pub fn from_windows_centered<S>(ws : &[Window<'_, u8>]) -> Self {
        let sz = ws[0].original_size();
        let map = PolarMap::new(sz, (sz.0 / 2, sz.1 / 2));
        Self::from_windows(ws, map)
    }

}

fn cartesian_to_polar(
    polar_lut : &[usize],
    cart_coord : &(usize, usize),
    cart_dims : &(usize, usize)
) -> usize {
    let lin_ix = cart_coord.0*cart_dims.1 + cart_coord.1;
    let pl_ix = polar_lut[lin_ix];
    pl_ix
}

// Holds (linear) polar coordinate of all (linearized) cartesian indices.
// Assume a sequence of row-wise cartesian indices. The returned LUT gives
// a linear index of the polar image buffer corresponding to this cartesian index.
// Returns maximum radius as second argument. Polar maps can be expensive to calculate
// for large images, so it is recommended to re-use them as much as possible if
// polar images share dimensions and center.
#[derive(Clone, Debug)]
pub struct PolarMap {

    // If ix is a linear index of the cartesian image, lut[ix]
    // is a linear index of the polar image.
    lut : Rc<[usize]>,

    center : (usize, usize),

    max_rad : f32

}

impl PolarMap {

    fn new(size : (usize, usize), center : (usize, usize)) -> Self {
        let mut mtx_r = DMatrix::<f32>::zeros(size.0, size.1);
        let mut mtx_theta = DMatrix::<f32>::zeros(size.0, size.1);

        let mut max_radius = 1.0;

        // This maps the max radius to the corners, without the conditional below.
        // let mut max_radius = sqrt(2) * height.max(width)

        for i in 0..size.0 {
            for j in 0..size.1 {
                let dx = (j as f32 - center.1 as f32);
                let dy = (i as f32 - center.0 as f32);
                let r = dx.hypot(dy);
                let y = dy / r;
                let x =  dx / r;
                let theta = y.atan2(x);
                mtx_r[(i, j)] = r;
                mtx_theta[(i, j)] = theta;
                if r > max_radius {
                    max_radius = r;
                }
            }
        }

        let mut lut = Vec::new();
        for i in 0..size.0 {
            for j in 0..size.1 {

                // First column is radius zero. Columns increase in normalized radius
                // up to maximum radius.
                let norm_radius = mtx_r[(i, j)] / max_radius;
                let r_ix = (norm_radius*((size.0-1) as f32)) as usize;
                assert!(r_ix < size.0);

                // Map (-pi, pi) to [0..ncols]. Zero angle is then ncols/2, -pi is 0 and pi is ncols.
                let norm_theta = (mtx_theta[(i, j)] + std::f32::consts::PI) / (2.0*std::f32::consts::PI);
                let theta_ix = (norm_theta*((size.1-1) as f32)) as usize;
                assert!(theta_ix < size.1);

                // lut.push((r_ix, theta_ix));
                lut.push(r_ix*size.1 + theta_ix);
            }
        }

        Self { lut : lut.into(), center, max_rad : max_radius }
    }

}

/*// cargo test --lib --message-format short -- polar --nocapture
#[test]
fn polar() {
    let mut buf = ImageBuf::<u8>::new_constant(64, 64, 0);
    buf.draw(Mark::Circle((32, 32), 20), 255);
    buf.show();
    let polar = PolarImage::from_centered(&buf);
    polar.polar.show();
}*/

// The pupil/iris manifests in a polar image centered at the pupil as a dark strip
// on top of the image followed by a less dark strip after it.

