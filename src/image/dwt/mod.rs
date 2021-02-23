use nalgebra::*;
use iter::*;
use nalgebra::storage::*;
use gsl::*;
use volta::signal::*;
use crate::image::*;
use volta::signal::dwt::gsl::*;

#[cfg(feature="ipp")]
pub mod ipp;

pub mod iter;

use iter::DWTIteratorBase;

/// Two-dimensional wavelet decomposition
pub struct Wavelet2D {
    plan : DWTPlan
}

/// Output of a wavelet decomposition. Imgage pyramids are indexed by a (scale, x, y) triplet.
#[derive(Clone, Debug)]
pub struct ImagePyramid<N> 
where
    N : Scalar + Copy
{
    pyr : Image<N>
}

impl<N> ImagePyramid<N> 
where
    N : Scalar + Copy
{

    pub fn len(&self) -> usize {
        self.pyr.len()
    }
}

impl<N> ImagePyramid<N> 
where
    N : Scalar + Copy
{

    pub fn new_constant(n : usize, value : N) -> Self {
        Self{ pyr : Image::new_constant(n, n, value) }
    }
    
    pub fn levels<'a>(&'a self) -> impl Iterator<Item=ImageLevel<'a, N>> {
        DWTIteratorBase::<&'a ImagePyramid<N>>::new_ref(&self)
    }
    
    /*pub fn levels_mut<'a>(&'a mut self) -> impl Iterator<Item=DMatrixSliceMut<'a, f64>> {
        DWTIteratorBase::<&'a mut DMatrix<f64>>::new_mut(&mut self.pyr)
    }*/
}

impl<N> AsRef<[N]> for ImagePyramid<N> 
where
    N : Scalar + Copy
{
    fn as_ref(&self) -> &[N] {
        self.pyr.as_slice()
    }
}

impl<N> AsMut<[N]> for ImagePyramid<N> 
where
    N : Scalar + Copy
{
    fn as_mut(&mut self) -> &mut [N] {
        self.pyr.as_mut_slice()
    }
}

pub struct ImageLevel<'a, N> 
where
    N : Scalar + Copy
{
    win : Window<'a, N>
}

impl<'a, N> ImageLevel<'a, N> 
where
    N : Scalar + Copy + Mul<Output=N> + MulAssign
{
    pub fn windows(&self, sz : (usize, usize)) -> impl Iterator<Item=Window<'a, N>> {
        self.win.clone().windows(sz)
    }
}

impl<'a, N> Index<(usize, usize)> for ImageLevel<'a, N> 
where
    N : Scalar + Copy
{

    type Output = N;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.win[index]
    }
}

impl<'a, N> From<Window<'a, N>> for ImageLevel<'a, N> 
where
    N : Scalar + Copy
{

    fn from(win : Window<'a, N>) -> Self {
        Self{ win }
    }
}

pub struct ImageLevelMut<'a, N> 
where
    N : Scalar + Copy
{
    win : WindowMut<'a, N>
}

impl<'a, N> From<WindowMut<'a, N>> for ImageLevelMut<'a, N> 
where
    N : Scalar + Copy
{

    fn from(win : WindowMut<'a, N>) -> Self {
        Self{ win }
    }
}

impl<N> AsRef<Image<N>> for ImagePyramid<N> 
where
    N : Scalar + Copy
{
    fn as_ref(&self) -> &Image<N> {
        &self.pyr
    }
}

/*impl<N> From<DMatrix<N>> for ImagePyramid<N> 
where
    N : Scalar
{
    fn from(s : DMatrix<N>) -> Self {
        Self{ pyr : s }
    }
}

impl<N> AsRef<DMatrix<N>> for ImagePyramid<N> 
where
    N : Scalar
{
    fn as_ref(&self) -> &DMatrix<N> {
        &self.pyr
    }
}*/

/*impl<N> From<Vec<N>> for Pyramid<N> 
where
    N : Scalar
{
    fn from(s : Vec<N>) -> Self {
        Self{ buf : DVector::from_vec(s) }
    }
}*/

impl Wavelet2D {

    pub fn new(basis : Basis, sz : usize) -> Result<Self, &'static str> {
        Ok(Self { plan : DWTPlan::new(basis, (sz, sz) )? })
    }
    
    pub fn forward_mut(&self, src : &Image<f64>, dst : &mut ImagePyramid<f64>) {
        self.plan.apply_forward(src.as_ref(), dst.as_mut())
            .map_err(|e| panic!("{}", e) );
    }
    
    pub fn forward(&self, src : &Image<f64>) -> ImagePyramid<f64> {
        let (nrows, ncols) = self.plan.shape();
        let mut dst = ImagePyramid::new_constant(nrows, 0.0);
        self.forward_mut(src, &mut dst);
        dst
    }
    
    pub fn backward_mut(&self, src : &ImagePyramid<f64>, dst : &mut Image<f64>) {
        self.plan.apply_backward(src.as_ref(), dst.as_mut())
            .map_err(|e| panic!("{}", e) );
    }
    
    pub fn backward(&self, src : &ImagePyramid<f64>) -> Image<f64> {
        let (nrows, ncols) = self.plan.shape();
        let mut dst = Image::new_constant(nrows, ncols, 0.0);
        self.backward_mut(src, &mut dst);
        dst
    }
}

/*impl Forward<Image<f64>> for Wavelet2D {
    
    type Output = Image<f64>;
    
    fn forward_mut(&self, src : &Image<f64>, dst : &mut Self::Output) {
        self.plan.apply_forward(src.as_ref(), dst.as_mut())
            .map_err(|e| panic!("{}", e) );
    }
    
    fn forward(&self, src : &Image<f64>) -> Self::Output {
        let (nrows, ncols) = self.plan.shape();
        let mut dst = Image::new_constant(nrows, ncols, 0.0);
        self.forward_mut(src, &mut dst);
        dst
    }
}

impl Backward<Image<f64>> for Wavelet2D {
    
    type Output = Image<f64>;
    
    fn backward_mut(&self, src : &Image<f64>, dst : &mut Self::Output) {
        self.plan.apply_backward(src.as_ref(), dst.as_mut())
            .map_err(|e| panic!("{}", e) );
    }
    
    fn backward(&self, src : &Image<f64>) -> Self::Output {
        let (nrows, ncols) = self.plan.shape();
        let mut dst = Image::new_constant(nrows, ncols, 0.0);
        self.backward_mut(src, &mut dst);
        dst
    }
}*/

/*
If the interest is exclusively on image reconstruction from low spatial frequencies,
see opencv::imgproc::{pyr_down, pyr_up} as an alternative. If the interest is on the coefficients
themselves (e.g. keypoint extraction) then the DWT is the way to go.

*/
