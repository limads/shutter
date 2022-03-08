use crate::image::*;
use nalgebra::Scalar;

// use packed_simd::*;

fn assert_dims(dims : &[(usize, usize)]) {
    let s1 = dims[0];
    assert!(dims.iter().all(|s| *s == s1 ))
}

fn assert_multiple(vals : &[usize], by : usize) {
    assert!(vals.iter().all(|v| v % by == 0 ))
}

fn assign_row_iter<'a, N>(
    out : &'a mut WindowMut<'a, N>, 
    a : &'a Window<'a, N>, 
    b : &'a Window<'a, N>
) -> impl Iterator<Item=(&'a mut [N], (&'a [N], &'a [N]))> 
    where N : Scalar + Copy
{
    out.rows_mut().zip(a.rows().zip(b.rows()))
}

/*fn simd_iter(out : &mut WindowMut<'_, u8>, a : &Window<'_, u8>, b : &Window<'_, u8>) -> impl Iterator<Item=(&mut u8x4, &u8x4, &u8x4)> {
    let row_iter = assign_row_iter(out, a, b);
    row_iter.map(|out, (a, b)| {
        (
            &mut out.chunks_exact_mut(4).map(u8x4::from_slice_unaligned),
            &a.chunks_exact(4).map(u8x4::from_slice_unaligned),
            &b.chunks_exact(4).map(u8x4::from_slice_unaligned)
        )
    }).flatten()
}

pub fn sub_mut_u8(out : &mut WindowMut<'_, u8>, a : &Window<'_, u8>, b : &Window<'_, u8>) {
    assert_dims(&[a.shape(), b.shape(), out.shape()]);
    assert_multiple(&[out.width(), a.width(), b.width()], 4);
    for (out_v, a_v, b_v) in simd_iter(out, a, b) {
        *out_v = a_v - b_v
    }
}*/

pub fn sub_mut_u8<'a>(out : &'a mut WindowMut<'a, u8>, a : &'a Window<'a, u8>, b : &'a Window<'a, u8>) {
    assert_dims(&[a.shape(), b.shape(), out.shape()]);
    assert_multiple(&[out.width(), a.width(), b.width()], 4);
    for (out, (a, b)) in assign_row_iter(out, a, b) {
        for (px_out, (px_a, px_b)) in out.iter_mut().zip(a.iter().zip(b.iter())) {
            *px_out = px_a - px_b;
        }
    }
}
