use super::*;

#[derive(Clone, Debug)]
pub struct DensitySegmenter {
    px_spacing : usize,
    win_sz : (usize, usize),
    color_weight : f64
}

impl Default for DensitySegmenter {

    fn default() -> Self {
        Self { px_spacing : 1, win_sz : (1, 1), color_weight : 1. }
    }

}

impl DensitySegmenter {

    pub fn new(win_sz : (usize, usize), px_spacing : usize) -> Self {
        Self { px_spacing, win_sz, color_weight : 1. }
    }

    pub fn set_px_spacing(&mut self, spacing : usize) {
        self.px_spacing = spacing;
    }

    pub fn set_dims(&mut self, dims : (usize, usize)) {
        self.win_sz = dims;
    }

    pub fn set_color_weight(&mut self, weight : f64) {
        self.color_weight = weight;
    }

    /// Extracts patches from image based on dense, homogeneous regions.
    /// Searches the 3d space or (row, col, color) for regions that are
    /// very clustered together, therefore mostly ignoring regions that are
    /// close but have non-homogeneous color, depending on the max_dist
    /// and min_cluster_sz parameters chosen. If you want to segment all colors,
    /// just pass |px| { true }, and the pixels won't be filtered.
    pub fn segment<F>(
        &mut self,
        win : &Window<'_, u8>,
        max_dist : f64,
        min_cluster_sz : usize,
        comp : F
    ) -> Vec<Patch>
    where
        F : Fn(u8)-> bool
    {

        assert!(win.shape() == self.win_sz);
        let (nrows, ncols) = (self.win_sz.0 as f64, self.win_sz.1 as f64);

        // A typical pixel distance meaning "near" is between 1 and \sqrt(2). To make the
        // color live in a distance near this scale, we must apply the square root transform,
        // or similar colors such as 240 and 250 will be too far away compared with the distances
        // and they will dominate the calculation. By applying sqrt, colors will live on a scale
        // between 0.0 - 16.0, not too far from the pixel scale.
        let pxs : Vec<[f64; 3]> = win.labeled_pixels::<usize, _>(self.px_spacing)
            .filter(|(_, _, px)| comp(*px) )
            // .map(|(r, c, color)| [r as f64 / nrows, c as f64 / ncols, self.color_weight * (color as f64 / 255.)] )
            // .map(|(r, c, color)| [r as f64, c as f64, color as f64] )
            .map(|(r, c, color)| [r as f64, c as f64, self.color_weight*(color as f64).sqrt()] )
            .collect::<Vec<_>>();

        // Or use cluster_linear to work without r-tree indexing of the clusters.
        // let clust = SpatialClustering::cluster_indexed(&pxs, max_dist, min_cluster_sz);
        let clust = SpatialClustering::cluster_linear(&pxs, max_dist, min_cluster_sz);

        let mut patches = Vec::new();
        for (_, clust) in clust.clusters.iter() {

            let mut avg_color = 0.;
            let mut n_pxs = 0.;
            let mut pxs = Vec::new();
            for [r, c, color] in clust.iter() {
                pxs.push((*r as u16, *c as u16));
                avg_color += color;
                n_pxs += 1.;
            }
            avg_color /= n_pxs;

            let mut patch = Patch {
                outer_rect : (0, 0, 0, 0),
                color : avg_color as u8,
                scale : self.px_spacing.try_into().unwrap(),
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

}

/*impl deft::Interactive for DensitySegmenter {

    #[export_name="register_DensitySegmenter"]
    extern "C" fn interactive() -> Box<deft::TypeInfo> {

        use deft::ReplResult;
        use rhai::{Array, Dynamic};

        deft::TypeInfo::builder::<Self>()
            .initializable()
            .fallible("set_px_spacing", |s : &mut Self, spacing : i64| -> ReplResult<()> {
                s.set_px_spacing(spacing as usize);
                Ok(())
            })
            .fallible("set_dims", |s : &mut Self, dims : rhai::Array| -> ReplResult<()> {
                s.set_dims((dims[0].clone().cast::<i64>() as usize, dims[1].clone().cast::<i64>() as usize));
                Ok(())
            })
            .fallible("set_color_weight", |s : &mut Self, weight : f64| -> ReplResult<()> {
                s.set_color_weight(weight);
                Ok(())
            })
            .fallible("segment", |s : &mut Self, w : Image<u8>, max_dist : f64, min_cluster_sz : i64| -> ReplResult<Array> {
                let patches = s.segment(&w.full_window(), max_dist, min_cluster_sz as usize, |_| true )
                    .iter()
                    .map(|patch| Dynamic::from(patch.clone()) )
                    .collect::<Vec<_>>();
                Ok(patches)
            })
            .priority(1)
            .build()
    }

}*/


