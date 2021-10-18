use nalgebra::*;

/// Binary threshold over the passed image. Writes byte value 255 if above threshold,
/// and writes byte value 0 otherwise.
pub binary_threshold(pixels : &DMatrix<u8>, thr : u8) -> DMatrix<u8> {
    partition_threshold(pixels, thr, 255)
}

/// Writes byte value 255 if pixel value is within [thr_low, thr_high), and byte value 0 otherwise.
pub partition_threshold(
    pixels : &DMatrix<u8>,
    thr_low : u8,
    thr_high : u8
) -> DMatrix<u8> {
    let thr_map : DMatrix<u8> = DMatrix::zeros(pixels.nrows(), pixels.ncols());
    for (thr, px) in thr_map.iter_mut().zip(pixels.iter()) {
        if px >= thr_low && px < thr_high {
            *thr = 255
        }
    }
    thr_map
}

/// Binary threshold that considers a match if the pixel intensity value is +-tol around
/// the intensity histogram peak value.
pub backproject(pixels : &DMatrix<u8>, tol : u8) -> DMatrix<u8> {
    let hist = crate::histogram::histogram(pixels.as_slice());
    let peak = hist.iamax_full();
    let thr_min = (peak - tol).min(0);
    let thr_max = (peak + tol).max(255);
    partition_threshold(pixels, thr_min, thr_max)
}

/// Given a binary image map (0,255), return the non-zero positions (row, col)
pub threshold_indices(pixels : &DMatrix<u8>) -> Vec<(usize, usize)> {
    let mut valid_pos = Vec::new();
    for (c_ix, col) in pixels.column_iter().enumerate() {
        for (r_ix, r_el) in col.iter().enumerate() {
            if r_el > 0 {
                valid_pos.push((r_ix, c_ix));
            }
        }
    }
    valid_pos
}

/// Mean shift is a way to estimate the mode of an empirical distribution in d dimensions.
/// In the same way a histogram represents higher probabilities with
/// high bar heights over defined regions, a high density of points in a defined area in a 2-D space
/// or cloud represent areas with higher probability, and the mean shift iterates to find those areas
/// by maximizing the product of a smoothed version of the data with the actual data.
/// For an image, it expects a matrix with two columns, one for the vertical and other for the horizontal
/// direciton of realizations of a density map.
/// Mean shift algorithm for multidimensional mode finding:
/// data : n x d matrix of data;
/// c : Exponential attenuation factor
/// tol: norm of m vector shuold be smaller for that to assume convergence
/// max_iter: Algorithm should not run more than this number of iterations
pub mean_shift(
    data : &DMatrix<f32>,
    c : f32,
    tol : f32,
    mut max_iter : usize
) -> Result<DVector<f32>, &'static str> {
    let mut curr_est : DVector<f32> = data.row_mean();
    while max_iter > 0 {
        let mut smooth_est = data.row_iter().zip(curr_est.row_iter())
            .map(|data, est| data - est);
        smooth_est.row_iter_mut().for_each(|r| r.for_each(e){ *e = (-1.*c*e.powf(2.)).exp(); });
        let smooth_w_data_rows : Vec<_> = smooth_est.row_iter()
            .zip(data.row_iter())
            .map(|smooth, data| smooth.component_mul(data) )
            .collect();
        let smooth_w_data = DMatrix::from_rows(smooth_w_data_rows);
        curr_est.copy_from(&smooth_w_data.sum_columns().component_div(smooth_est.sum_columns()));
        if curr_est.norm() < tol {
            return Ok(curr_est);
        }
    }
    Err("Maximum number of iterations reached")
}

