use nalgebra::*;
use nalgebra::base::storage::Storage;
use std::convert::TryInto;
use crate::image::*;
use super::*;
use crate::image::dwt::*;

#[derive(Clone, Copy)]
pub(crate) enum DWTFilter {
    Vertical,   // HL
    Horizontal, // LH
    Both        // HH
}

/// Base structure for pyramid iterators. The full generic type will keep either
/// a &'a [N] or &'a mut [N] to the full slice of the pyramid. Any concrete instantiations
/// will implement iterator to yield Window/WindowMut depending on the underlying mutability
/// of the reference.
pub(crate) struct DWTIteratorBase<S> {
    max_lvl : usize,
    curr_lvl : usize,
    region : DWTFilter,
    full : S
}

impl<'a, N> DWTIteratorBase<&'a ImagePyramid<N>> 
where
    N : Scalar + Copy
{

    /*pub fn new_ref<'a, C>(
        full : &'a Matrix<f64, Dynamic, C, VecStorage<f64, Dynamic, C>>
    ) -> DWTIteratorBase<&'a Matrix<f64, Dynamic, C, VecStorage<f64, Dynamic, C>>>
        where
            C : Dim
    {
        if C::try_to_usize().is_none() {
            assert!(full.nrows() == full.ncols());
        }
        assert!( (full.nrows() as f64).log2().fract() == 0.0 );
        let max_lvl = ((full.nrows() as f32).log2() - 1.) as usize;
        DWTIteratorBase::<&'a Matrix<f64, Dynamic, C, VecStorage<f64, Dynamic, C>>> {
            max_lvl,
            curr_lvl : 0,
            region : DWTFilter::Vertical,
            full
        }
    }*/
    
    pub fn new_ref(
        full : &'a ImagePyramid<N>
    ) -> DWTIteratorBase<&'a ImagePyramid<N>>
        where
            N : Scalar + Copy
    {
        let max_lvl = calc_max_level(full.as_ref());
        DWTIteratorBase::<&'a ImagePyramid<N>> {
            max_lvl,
            curr_lvl : 0,
            region : DWTFilter::Vertical,
            full
        }
    }

    /*pub fn new_mut<'a, C>(
        full : &'a mut Matrix<f64, Dynamic, C, VecStorage<f64, Dynamic, C>>
    ) -> DWTIteratorBase<&'a mut Matrix<f64, Dynamic, C, VecStorage<f64, Dynamic, C>>>
        where
            C : Dim
    {
        let max_lvl = ((full.nrows() as f32).log2() - 1.) as usize;
        DWTIteratorBase::<&'a mut Matrix<f64, Dynamic, C, VecStorage<f64, Dynamic, C>>>{
            max_lvl,
            curr_lvl : 0,
            region : DWTFilter::Vertical,
            full
        }
    }*/
    /*pub fn new_mut<'a, N>(
        full : &'a mut [N]
    ) -> DWTIteratorBase<&'a mut [N]>
        where
            N : Scalar + Copy
    {
        let max_lvl = calc_max_level(&full);
        DWTIteratorBase::<&'a mut [N]> {
            max_lvl,
            curr_lvl : 0,
            region : DWTFilter::Vertical,
            full
        }
    }*/

    // Resolve iteration over 1D vectors
    /*pub fn update_1d(&mut self) -> Option<()> {
        if self.curr_lvl == self.max_lvl + 1 {
            return None;
        }
        self.curr_lvl += 1;
        Some(())
    }*/

    // Resolve iteration over 2D images
    // pub fn update_2d(&mut self) -> Option<()> {        
    // }
}

pub(crate) type DWTIterator2D<'a, N> = DWTIteratorBase<&'a ImagePyramid<N>>;

impl<'a, N> DWTIterator2D<'a, N> 
where
    N : Scalar + Copy
{

    pub fn new(full : &'a ImagePyramid<N>) -> Self {
        let max_lvl = calc_max_level(full.as_ref());
        DWTIterator2D{ max_lvl, curr_lvl : 0, region : DWTFilter::Vertical, full }
    }

}

impl<'a, N> Iterator for DWTIterator2D<'a, N> 
where
    N : Scalar + Copy
{

    type Item = ImageLevel<'a, N>;

    fn next(&mut self) -> Option<ImageLevel<'a, N>> {
        let ans = get_level_slice_2d(self.full, self.curr_lvl, self.region)?;
        if self.curr_lvl == self.max_lvl + 1 {
            return None;
        }
        if self.curr_lvl == 0 {
            self.curr_lvl += 1;
        } else {
            let (new_region, new_lvl) = match self.region {
                DWTFilter::Vertical => {
                    (DWTFilter::Both, self.curr_lvl)
                },
                DWTFilter::Both => {
                    (DWTFilter::Horizontal, self.curr_lvl)
                },
                DWTFilter::Horizontal => {
                    (DWTFilter::Vertical, self.curr_lvl + 1)
                }
                };
            self.region = new_region;
            self.curr_lvl = new_lvl;
        }
        Some(ans)
    }

}

/*pub(crate) type DWTIteratorMut2D<'a, N> = DWTIteratorBase<&'a mut ImagePyramid<N>>;

impl<'a, N> DWTIteratorMut2D<'a, N> 
where
    N : Scalar + Copy
{

    pub fn new(full : &'a mut ImagePyramid<N>) -> Self
        where Self : 'a,
        N : Scalar + Copy
    {
        let max_lvl = calc_max_level(full.as_ref());
        DWTIteratorMut2D{ max_lvl, curr_lvl : 0, region : DWTFilter::Vertical, full }
    }

}

impl<'a, N> Iterator for DWTIteratorMut2D<'a, N> 
where
    N : Scalar + Copy
{

    type Item = ImageLevelMut<'a, N>;

    fn next(&mut self) -> Option<ImageLevelMut<'a, N>>
        where Self : 'a
    {
        let ans = get_level_slice_2d(self.full, self.curr_lvl, self.region)?;
        let strides = ans.strides();
        let shape = ans.shape();
        let ptr : *mut N = ans.as_slice().ptr() as *mut _;
        self.update_2d()?;
        unsafe {
            /*let storage = SliceStorageMut::from_raw_parts(
                ptr,
                (Dynamic::from_usize(shape.0), Dynamic::from_usize(shape.1)),
                (U1::from_usize(strides.0), Dynamic::from_usize(strides.1))
            );
            let slice_mut = DMatrixSliceMut::from_data(storage);
            Some(slice_mut)*/
        }
    }
}*/

/*pub(crate) type DWTIterator1D<'a> = DWTIteratorBase<&'a DVector<f64>>;

impl<'a> Iterator for DWTIterator1D<'a> {

    type Item = DVectorSlice<'a, f64>;

    fn next(&mut self) -> Option<DVectorSlice<'a, f64>>
        where Self : 'a
    {
        let slice = get_level_slice_1d(&self.full, self.curr_lvl)?;
        self.update_1d();
        Some(slice)
    }

}

pub(crate) type DWTIteratorMut1D<'a> = DWTIteratorBase<&'a mut DVector<f64>>;

impl<'a> Iterator for DWTIteratorMut1D<'a> {

    type Item = DVectorSliceMut<'a, f64>;

    fn next(&mut self) -> Option<DVectorSliceMut<'a, f64>>
        where Self : 'a
    {
        let slice = get_level_slice_1d(&self.full, self.curr_lvl)?;
        let shape = slice.shape();
        let ptr : *mut f64 = slice.data.ptr() as *mut _;
        self.update_1d()?;
        unsafe {
            let storage = SliceStorageMut::from_raw_parts(
                ptr,
                (Dynamic::from_usize(shape.0), U1::from_usize(1)),
                (U1::from_usize(1), Dynamic::from_usize(shape.1))
            );
            let slice_mut = DVectorSliceMut::from_data(storage);
            Some(slice_mut)
        }
    }
}
*/

/// Define an (offset, size) tuple pair for the given DWT decomposition level.
fn define_level_bounds(lvl : usize, region : DWTFilter) -> ((usize, usize), (usize, usize)) {
    let lvl_pow = (2 as i32).pow(lvl.try_into().unwrap()) as usize;
    let region_offset = match region {
        DWTFilter::Both => (lvl_pow, lvl_pow),
        DWTFilter::Vertical => (lvl_pow as usize, 0),
        DWTFilter::Horizontal => (0, lvl_pow as usize)
    };
    match lvl {
        0 => ((0, 0), (2, 2)),
        _ => (region_offset, (lvl_pow, lvl_pow))
    }
}

/*fn get_level_slice_2d<'a>(
    m : &'a DMatrix<f64>,
    lvl : usize,
    region : DWTFilter
) -> Option<DMatrixSlice<'a, f64>>
{
    let (off, bounds) = define_level_bounds(lvl, region);
    if bounds.0 > m.nrows() / 2 {
        return None;
    }
    Some(m.slice(off, bounds))
}*/

/*fn get_level_slice_2d<'a>(
    m : &'a DMatrix<f64>,
    lvl : usize,
    region : DWTFilter
) -> Option<DMatrixSlice<'a, f64>>
{
    let (off, bounds) = define_level_bounds(lvl, region);
    if bounds.0 > m.nrows() / 2 {
        return None;
    }
    Some(m.slice(off, bounds))
}*/

/// Calculates the DWT iterator maximum level based on a slice.
pub fn calc_max_level<N>(full : &[N]) -> usize {
    let side = (full.len() as f32).sqrt() as usize;
    assert!(super::is_valid_dwt_len(side));
    let max_lvl = ((side as f32).log2() - 1.) as usize;
    max_lvl
}

fn get_level_slice_2d<'a, N>(
    m : &'a ImagePyramid<N>,
    lvl : usize,
    region : DWTFilter
) -> Option<ImageLevel<'a, N>>
where
    N : Scalar + Copy
{
    let (off, bounds) = define_level_bounds(lvl, region);
    let img : &'a Image<_> = m.as_ref(); 
    let nrows = img.height();
    if bounds.0 > nrows / 2 {
        return None;
    }
    Some(ImageLevel::<'a, N>::from(img.window(off, bounds).unwrap()))
}

#[test]
fn dwt_iter_1d() {
    let d = DVector::from_row_slice(
        &[0., 0., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3.]
    );
    let mut iter = DWTIterator1D::new_ref(&d);
    while let Some(s) = iter.next() {
        println!("{}", s);
    }
}

#[test]
fn dwt_iter_2d() {
    let d = DMatrix::from_row_slice(16, 16,
        &[0., 0., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3.,
          0., 0., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3.,
          1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3.,
          1., 1., 1., 1., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3.,
          2., 2., 2., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3.,
          2., 2., 2., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3.,
          2., 2., 2., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3.,
          2., 2., 2., 2., 2., 2., 2., 2., 3., 3., 3., 3., 3., 3., 3., 3.,
          3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
          3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
          3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
          3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
          3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
          3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
          3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
          3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.,
        ]);
    let mut iter = DWTIterator2D::new_ref(&d);
    while let Some(s) = iter.next() {
        println!("{}", s);
    }
}

#[test]
fn dwt_iter_64() {
    let m = DMatrix::zeros(64,64);
    let mut iter = DWTIterator2D::new_ref(&m);
    while let Some(s) = iter.next() {
        println!("{}", s);
    }
}
