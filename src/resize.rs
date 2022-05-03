
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
    
