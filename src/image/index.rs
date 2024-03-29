// use nalgebra::*;

use super::*;
use std::ops::{Sub, Add, Mul};
use num_traits::ToPrimitive;
use crate::code::RunLength;

impl<P, S> Index<RunLength> for Image<P, S>
where
    S : Storage<P>,
    P : Pixel
{

    type Output = [P];

    fn index(&self, rl: RunLength) -> &Self::Output {
        assert!(rl.start.1 + rl.length <= self.width() && rl.start.0 < self.height());
        &self.row(rl.start.0).unwrap()[rl.range()]
    }

}

impl<P, S> IndexMut<RunLength> for Image<P, S>
where
    S : StorageMut<P>,
    P : Pixel
{

    fn index_mut(&mut self, rl: RunLength) -> &mut Self::Output {
        assert!(rl.start.1 + rl.length <= self.width() && rl.start.0 < self.height());
        &mut self.row_mut(rl.start.0).unwrap()[rl.range()]
    }

}

impl<P, S> Index<(usize, usize)> for Image<P, S>
//where
//    N : Scalar + Copy + Any
where
    S : Storage<P>
{

    type Output = P;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        // let local_ix = (index.0 + self.offset.0, index.1 + self.offset.1);
        // unsafe { self.slice.as_ref().get_unchecked(index::linear_index(local_ix, self.width)) }
        &self.slice.as_ref()[index::linear_index(index, self.width)]
    }
}

impl<P, S> IndexMut<(usize, usize)> for Image<P, S>
// where
//     N : Scalar + Copy + Default + Copy + Any
where
    S : StorageMut<P>
{

    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        // let local_ix = (index.0 + self.offset.0, index.1 + self.offset.1);
        // unsafe { self.slice.as_mut().get_unchecked_mut(index::linear_index(local_ix, self.width)) }
        &mut self.slice.as_mut()[index::linear_index(index, self.width)]
    }

}

impl<P, S> Index<(u16, u16)> for Image<P, S>
//where
//    N : Scalar + Copy + Any
where
    S : Storage<P>
{

    type Output = P;

    fn index(&self, index: (u16, u16)) -> &Self::Output {
        &self[(index.0 as usize, index.1 as usize)]
    }
}

impl<P, S> IndexMut<(u16, u16)> for Image<P, S>
// where
//     N : Scalar + Copy + Default + Copy + Any
where
    S : StorageMut<P>
{

    fn index_mut(&mut self, index: (u16, u16)) -> &mut Self::Output {
        &mut self[(index.0 as usize, index.1 as usize)]
    }

}

impl<'a, N> Window<'a, N> {

    pub const fn from_static<const S : usize, const W : usize>(
        array : &'static [N; S]
    ) -> Window<'static, N> {

        if S % W != 0 {
            panic!("Invalid image dimensions");
        }

        Window {
            slice : array,
            width : W,
            offset : (0, 0),
            sz : (S / W, W),
            _px : PhantomData
        }
    }

}

pub fn sub_range(offset : Coord, size : Size, original_width : usize) -> Range<usize> {
    let start = linear_index(offset, original_width);
    let end = start + (size.0 - 1)*original_width + size.1;
    Range { start, end }
}

pub fn sub_slice<S>(slice : &[S], offset : Coord, size : Size, original_width : usize) -> &[S] {
    let range = sub_range(offset, size, original_width);
    &slice[range]
}

pub fn sub_slice_mut<S>(slice : &mut [S], offset : Coord, size : Size, original_width : usize) -> &mut [S] {
    let range = sub_range(offset, size, original_width);
    &mut slice[range]
}

/// Returns position relative to the current linear index in the image with the given number of columns.
pub fn coordinate_index(lin_ix : usize, ncol : usize) -> (usize, usize) {
    (lin_ix / ncol, lin_ix % ncol)
}

// pub fn sub_index(linear_ix : N, size : (N, N), ncol : N) -> N {
//    linear_ix / ncol + linear_ix
// }

/*pub fn linear_index<N>(coord_ix : (N, N), ncol : N) -> N
where
    N : Mul<Output=N> + Add<Output=N>
{
    coord_ix.0 * ncol + coord_ix.1
}*/

pub fn linear_index(coord_ix : (usize, usize), ncol : usize) -> usize {
    // coord_ix.0 * ncol + coord_ix.1
    coord_ix.0.wrapping_mul(ncol).wrapping_add(coord_ix.1)
}

/// Translate a matrix index (row, col) into planar coordinates (x, y),
/// keeping the x value but flipping the y value.
pub fn index_to_plane_coord<N>(ix : (N, N), nrow : N) -> (f64, f64)
where
    N : Sub<Output=N> + Copy + ToPrimitive
{
    let x : f64 = (ix.1).to_f64().unwrap();
    let y : f64 = (nrow - ix.0).to_f64().unwrap();
    (x, y)
}

/// Assuming the index (row, col) can be interpreted as a graphical coordinate,
/// returns the distance and angle between the source index (taken to be the (0,0) origin in the
/// cartesian plane) and the destination index as (dist, angle). Angles are in radians, and
/// count positive from 0º-180º (0 rad - 3.14 rad) and negative from 180º-360º (3.14 rad - 0 rad).
/// This is as if the positive radian value has been "reflected" over the lower half of the trig circle. 
/// TODO create version with just the distance calculation, which will just reflect the y coordinate.
pub fn index_distance<N>(src : (N, N), dst : (N, N), nrow : N) -> (f64, f64)
where
    N : Sub<Output=N> + Copy + ToPrimitive
{
    let (src_x, src_y) = index_to_plane_coord(src, nrow);
    let (dst_x, dst_y) = index_to_plane_coord(dst, nrow);
    
    // "Base" and "Height" might be negative here depending on the right triangle orientation
    let base = dst_x - src_x;
    
    // Convert from matrix index to graphical plane
    let height = dst_y - src_y;
    
    // We can use abs here because reflecting the triangle over the vertical or horizontal axis
    // does not change its hypotenuse lenght.
    let dist = base.abs().hypot(height.abs());

    let theta = height.atan2(base);
    
    (dist, theta)
}

#[test]
fn ix_dist() {
    let img_sz = (10, 10);
    
    // Cross
    println!("Dist = 3; 0º: {:?}", index_distance((4, 4), (4, 7), img_sz.0));
    println!("Dist = 3; 90º: {:?}", index_distance((4, 4), (1, 4), img_sz.0));
    println!("Dist = 3; 180º: {:?}", index_distance((4, 4), (4, 1), img_sz.0)); 
    println!("Dist = 3; 360º: {:?}", index_distance((4, 4), (7, 4), img_sz.0));
    
    // X
    println!("Dist = 4.24; 45º: {:?}", index_distance((4, 4), (1, 7), img_sz.0));
    println!("Dist = 4.24; 135º: {:?}", index_distance((4, 4), (1, 1), img_sz.0));
    println!("Dist = 4.24; 225º: {:?}", index_distance((4, 4), (7, 1), img_sz.0)); 
    println!("Dist = 4.24; 315º: {:?}", index_distance((4, 4), (7, 7), img_sz.0));
     
}

