use super::*;

/// Extracts patches from image based on dense, homogeneous regions.
/// Searches the 3d space or (row, col, color) for regions that are
/// very clustered together, therefore mostly ignoring regions that are
/// close but have non-homogeneous color, depending on the min_dist
/// and min_cluster_sz parameters chosen.
pub fn patches_from_dense_regions(
    win : &Window<'_, u8>,
    scale : usize,
    min_dist : f64,
    min_cluster_sz : usize,
    mut mode : ColorMode
) -> Vec<Patch> {
    let pxs : Vec<[f64; 3]> = win.labeled_pixels::<usize, _>(scale)
        .filter(|(_, _, px)| mode.matches(*px) )
        .map(|(r, c, color)| [r as f64, c as f64, color as f64] )
        .collect::<Vec<_>>();
    let clust = SpatialClustering::cluster_linear(&pxs, min_dist, min_cluster_sz);
    let mut patches = Vec::new();
    for (_, clust) in clust.clusters.iter() {
        let mut color = (clust.iter().map(|[_, _, c]| c ).sum::<f64>() / clust.len() as f64) as u8;
        let outer_rect = (0, 0, 0, 0);
        let pxs : Vec<_> = clust.iter().map(|[r, c, _]| (*r as u16, *c as u16) ).collect();
        let mut patch = Patch {
            outer_rect,
            color,
            scale : scale.try_into().unwrap(),
            img_height : win.height(),
            area : clust.len(),
            pxs
        };
        let mut row_pxs = patch.group_rows();
        let min_row = row_pxs.keys().min().unwrap();
        let max_row = row_pxs.keys().max().unwrap();

        let mut col_pxs = patch.group_cols();
        let min_col = col_pxs.keys().min().unwrap();
        let max_col = col_pxs.keys().max().unwrap();

        patch.outer_rect = (*min_row, *min_col, *max_row - *min_row, *max_col - *min_col);
        patches.push(patch);
    }

    patches
}

