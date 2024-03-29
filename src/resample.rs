/*pub trait Resample {

    type Output;

    type OwnedOutput;

    fn downsample_to(&self, out : &mut Self::Output, down : Downsample, by : usize);

    fn upsample_to(&self, out : &mut Self::Output, up : Upsample, by : usize);

    fn downsample(&self, down : Downsample, by : usize) -> Self::OwnedOutput;

    fn upsample(&self, up : Upsample, by : usize) -> Self::OwnedOutput;

}*/

use nalgebra::Scalar;
use crate::image::*;
pub use ripple::resample::*;
use std::fmt::Debug;
use std::any::Any;
use crate::image::ipputils::Resize;

impl<N, S, T> Resample<ImageBuf<N>, Image<N, T>> for Image<N, S>
where
    N : Pixel,
    S : Storage<N>,
    T : StorageMut<N>
{

    fn downsample_to(&self, out : &mut Image<N, T>, down : Downsample) {

        assert!(self.height() % out.height() == 0 && self.width() % out.width() == 0);

        /*#[cfg(feature="opencv")]
        unsafe {
            cvutils::resize(
                self.win,
                out.original_slice()[..],
                src_ncols,
                None,
                dst_ncols,
                None
            );
            return;
        }*/

        #[cfg(feature="ipp")]
        unsafe {
            crate::image::ipputils::resize(&self.full_window(), &mut out.full_window_mut(), Resize::Nearest);
            return;
        }

        unimplemented!()
    }

    fn upsample_to(&self, out : &mut Image<N, T>, up : Upsample) {

        #[cfg(feature="ipp")]
        unsafe {
            crate::image::ipputils::resize(&self.full_window(), &mut out.full_window_mut(), Resize::Nearest);
            return;
        }

        unimplemented!()
    }

    fn downsample(&self, down : Downsample, by : usize) -> ImageBuf<N> {
        assert!(self.height() % by == 0 && self.width() % by == 0);
        let mut downsampled = unsafe { ImageBuf::<N>::new_empty(self.height() / by, self.width() / by) };
        self.downsample_to(&mut downsampled.full_window_mut(), down);
        downsampled
    }

    fn upsample(&self, up : Upsample, by : usize) -> ImageBuf<N> {
        // assert!(self.height() % by == 0 && self.width() % by == 0);
        let mut upsampled = unsafe { ImageBuf::<N>::new_empty(self.height() * by, self.width() * by) };
        self.upsample_to(&mut upsampled.full_window_mut(), up);
        upsampled
    }
}

/*
Establish the distinction between resample (target #samples is an integer multiple or divisor of original #samples)
and resize (target does not necessarily follow those restrictions). For the case where resize does satisfy that,
just dispatch to resample. If not, call custom function.
*/

/*// TODO call this downsample_convert, and leave alias as a second enum argument:
// AntiAliasing::On OR AntiAliasing::Off. Disabling antialiasing calls this implementation
// that just iterates over the second buffer; enabling it calls for more costly operations.
pub fn downsample_aliased<M>(&mut self, src : &Window<M>)
where
    M : Scalar + Copy,
    N : Scalar + From<M>
{
    let (nrows, ncols) = self.shape();
    let step_rows = src.win_sz.0 / nrows;
    let step_cols = src.win_sz.1 / ncols;
    assert!(step_rows == step_cols);
    sampling::slices::subsample_convert_with_offset(
        src.win,
        src.offset,
        src.win_sz,
        (nrows, ncols),
        step_rows,
        self.buf.chunks_mut(nrows)
    );
}*/
    
