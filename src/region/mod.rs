use crate::image::*;
use std::collections::{VecDeque, HashMap};
use std::iter::FromIterator;
use std::mem;
use std::convert::AsRef;
use std::fmt::Debug;
use nalgebra::Scalar;

pub mod raster;

/* pub trait Region {

    fn center()

    fn width()

    fn height()

    fn bounding_rect()

    fn inner_rect()

    fn bounding_circle()

    fn inner_circle()

    fn approximate_ellipse()

    fn approximate_circle()

    fn central_moments()

    fn normalized_moments()

    fn isotropic_moments()

    fn overlaps(self, other)

    fn contacts(self, other)

    fn distance(self, other)

    fn orientation(self, other)

    fn group(&self, others : &[Region])

    fn inner_pixels(&self, win : &Window<'_, u8>)

}

impl Region for Boundary { }

impl Region for Contour { }

impl Region for Patch { }

impl Region for Rectangle { }

impl Region for Circle { }

impl Region for Ellipse { }

impl Region for Patch {

}

impl Region for Boundary {

}

impl Region for Contour {

}
*/

// Searches for pixels that are likely to be near the center of homogeneous
// regions, since all their neighbors have approximately the same color as the pixel.
// Pivotal pixels are good candidates for seeds for Filler algorithms.
pub fn pivotal_pixels() -> Vec<(usize, usize)> {

    let mut piv_pxs = Vec::new();

    piv_pxs
}

/// Holds an array with all the pixel coordinates. It is recommended to use patches only
/// on small and/or coarse images, or images for which you expect the regions of interest
/// to be small. Otherwise, use labelling algorithms (that generate a image of same dimension,
/// but using only one numeric value per pixel, instead of two coordinate values).
pub struct Patch {

}

impl Patch {

    /// Extract a dense pixel coordinate vector from a labeled window.
    pub fn extract(labels : &Window<'_, u8>) -> Vec<Self> {
        unimplemented!()
    }

}

pub struct Boundary {

}

impl Boundary {

    /// Extract a dense pixel coordinate vector from a labeled window.
    pub fn extract(labels : &Window<'_, u8>) -> Vec<Self> {
        unimplemented!()
    }
}

pub struct Contour {

}

impl Contour {

    /// Extract a dense pixel coordinate vector from a labeled window.
    pub fn extract(labels : &Window<'_, u8>) -> Vec<Self> {
        unimplemented!()
    }
}

/// Fillers build a dense and distinct region representation from a seed.
/// fill_regions writers the distinct image labels into another region
/// of same dimension. fill_patches actually holds all the pixel coordinates
/// for distinct regions.
pub trait Filler {

    fn fill_region();

    fn fill_patch();

}

/// Tracers build a single contour or boundary representation from a seed.
/// Boundaries are unordered pixel coordiantes. Contours are ordered pixel
/// coordinates. Tracers can extract contours from generic images, sidestepping
/// the labelling step. But contours can always be extracted from label images
/// via Region::extract.
pub trait Tracer {

    fn trace_contour();

    fn trace_boundary();

}

/// Segmenters use fillers and/or tracers to separate distinct image regions.
/// Labels images using fillers or other strategies. (i.e. attribute unique integer labels of same numeric type as the image
/// for foreground regions that are disconnected; connected foreground images have the
/// same label). Segmenters can output a labeled image via segment_labels or output a set of Region
/// implementors (patches, boundaries and contours).
pub trait Segmenter {

    fn segment_labels();

    fn segment_boundaries();

    fn segment_contours();

}

// Labeling of a binary image specified by Myler & Weeks (p. 75). Joined regions marked as foreground
// with size > max_clust_dist will be marked as new regions, even if they are connected.
// The image will be labeled with the cluster index label. The algorithm will fail if more than
// 254 clusters are found, and will stop iterating in this case. This can be used for binary images
// over edges, using a distance that considers both orientation and magnitude.
fn rough_binary_region_labelling(mut win : WindowMut<'_, u8>, max_clust_dist : f32) {

    // Clusters are identified by the index of the last pixel added to them.
    // Membership to the cluster is defined as distance of new candidate pixel
    // to the position of this last pixel.
    let mut last_clust_px = Vec::new();

    for r in 0..win.height() {
        for c in 0..win.width() {

            if win[(r, c)] == 0 {
                continue;
            }

            // Start first cluster at first nonzero element
            if last_clust_px.len() == 0 {

                // Curr_cluster is still zero.
                last_clust_px.push((r, c));

                // Mark pixel as cluster 1
                win[(r, c)] = last_clust_px.len() as u8;

            } else {
                let mut new_cluster : bool = true;
                let mut k = 0;

                // Test for membership in all known clusters by comparing to last pixel
                for k in 0..last_clust_px.len() {
                    let dist_k = crate::feature::shape::point_euclidian((r, c), last_clust_px[k]);
                    if dist_k < max_clust_dist {
                        // Label image to the old cluster (add one since cluster label is the size
                        // of the last_clust_px vector when the first pixel for it was found)
                        win[(r, c)] = (k + 1) as u8;
                        new_cluster = false;
                        break;
                    }
                }

                if new_cluster {
                    last_clust_px.push((r, c));
                    win[(r, c)] = last_clust_px.len() as u8;
                }

                if last_clust_px.len() == 254 {
                    return;
                }

            }
        }
    }
}

/*
Marks pixels belonging to segment boundaries. The contours can then be extracted by
verifying which pixels equals the boundary value.
IppStatus ippiBoundSegments_8u_C1IR(Ipp8u* pMarker, int markerStep, IppiSize roiSize,
Ipp8u val, IppiNorm norm);
IppStatus ippiBoundSegments_16u_C1IR(Ipp16u* pMarker, int markerStep, IppiSize roiSize,
Ipp16u val, IppiNorm norm);
*/

// Grows with the watershed algorithm. Labels contains a gray image with 0 at the pixels that are
// yet to be labeled; and non-zero distinct labels at the individual desired seed values. The image
// is modified in-place by filling the zero labels with the desired values.
// Watershed segmentation is preferable for images with local minimums, for example, gradient images
#[cfg(feature="ipp")]
pub unsafe fn ipp_watershed(mut labels : WindowMut<'_, u8>, win : &Window<'_, u8>) {

    use crate::foreign::ipp::ippi::*;
    use crate::foreign::ipp::ippcv;
    use crate::image::ipputils;

    assert!(labels.width() == win.width() && labels.height() == win.height());
    let (src_step, src_sz) = ipputils::step_and_size_for_window(win);
    let (label_step, label_sz)  = ipputils::step_and_size_for_window(win);

    let mut buffer = ipputils::allocate_buffer_with(|buf_sz|
        ippcv::ippiSegmentWatershedGetBufferSize_8u_C1R(mem::transmute(src_sz), buf_sz)
    );

    // Alt: IPP_SEGMENT_DISTANCE + IPP_SEGMENT_BORDER_8
    let flags = IPP_SEGMENT_QUEUE + IPP_SEGMENT_BORDER_4;

    // alt norms: ippiNormInf, ippiNormL1, IppiNormL2, IppiNormFM
    let norm = _IppiNorm_ippiNormInf;
    let status = ippcv::ippiSegmentWatershed_8u_C1IR(
        win.offset_ptr(),
        src_step,
        labels.offset_ptr_mut(),
        label_step,
        mem::transmute(src_sz),
        norm,
        flags as i32,
        &mut buffer[0] as *mut _
    );
    assert!(status == 0);
}

// Grows to the least-gradient direction
#[cfg(feature="ipp")]
pub unsafe fn ipp_segment_gradient(mut labels : WindowMut<'_, u8>, win : WindowMut<'_, u8>) {

    use crate::foreign::ipp::ippi::*;
    use crate::foreign::ipp::ippcv;
    use crate::image::ipputils;

    let (step, sz) = ipputils::step_and_size_for_window_mut(&win);
    let mut buffer = ipputils::allocate_buffer_with(|buf_sz|
        ippcv::ippiSegmentGradientGetBufferSize_8u_C1R(mem::transmute(sz), buf_sz)
    );

    // Alt: IPP_SEGMENT_DISTANCE + IPP_SEGMENT_BORDER_8
    let flags = IPP_SEGMENT_QUEUE + IPP_SEGMENT_BORDER_4;

    // alt norms: ippiNormInf, ippiNormL1, IppiNormL2, IppiNormFM
    let norm = _IppiNorm_ippiNormInf;
    let status = ippcv::ippiSegmentGradient_8u_C1IR(
        win.offset_ptr_mut(),
        step,
        labels.offset_ptr_mut(),
        step,
        mem::transmute(sz),
        norm,
        flags as i32,
        &mut buffer[0] as *mut _
    );
    assert!(status == 0);
}

/* The function finds small connected components and set them to the speckleVal value. This function marks
only components with size that is less than, or equal to maxSpeckleSize (irrespective of their color).
Pixels of the image belong to the
same connected component if the difference between adjacent pixels (considering 4-connected adjacency) is
less than, or equal to the maxSpeckleSize value. */
#[cfg(feature="ipp")]
unsafe fn ipp_mark_speckels(mut win : WindowMut<'_, u8>, max_diff : u8, max_sz : usize, new_speckle_val : u8) {

    use crate::foreign::ipp::ippi::*;
    use crate::foreign::ipp::ippcv;
    use crate::foreign::ipp::ippcore::IppDataType_ipp8u;
    use crate::image::ipputils;

    let (step, sz) = ipputils::step_and_size_for_window_mut(&win);
    let mut buffer = ipputils::allocate_buffer_with(|buf_sz|
        ippcv::ippiMarkSpecklesGetBufferSize(mem::transmute(sz), IppDataType_ipp8u, 1, buf_sz)
    );

    // alt norms: ippiNormInf, ippiNormL1, IppiNormL2, IppiNormFM
    let norm = _IppiNorm_ippiNormInf;
    let status = ippcv::ippiMarkSpeckles_8u_C1IR(
        win.offset_ptr_mut(),
        step,
        mem::transmute(sz),
        new_speckle_val,
        max_sz as i32,
        max_diff,
        norm,
        &mut buffer[0] as *mut _
    );
    assert!(status == 0);
}

// Each connected set of non-zero image pixels is treated as the separate marker
// The image with labeled markers can be used as the seed image for
// segmentation by functions ippiSegmentWatershed or ippiSegmentGradient functions.
// This works with a binary image.
#[cfg(feature="ipp")]
pub unsafe fn ipp_label_markers(mut win : WindowMut<'_, u8>) {

    use crate::foreign::ipp::ippi::*;
    use crate::foreign::ipp::ippcv;
    use crate::image::ipputils;

    let (step, sz) = crate::image::ipputils::step_and_size_for_window_mut(&win);
    let mut buffer = ipputils::allocate_buffer_with(|buf_sz|
        ippcv::ippiLabelMarkersGetBufferSize_8u_C1R(mem::transmute(sz), buf_sz)
    );

    // alt norms: ippiNormInf, ippiNormL1, IppiNormL2, IppiNormFM
    let norm = _IppiNorm_ippiNormInf;

    let min_label = 0;
    let max_label = 255;

    let mut n_labels : i32 = 0;
    let status = ippcv::ippiLabelMarkers_8u_C1IR(
        win.offset_ptr_mut(),
        step,
        mem::transmute(sz),
        min_label,
        max_label,
        norm,
        &mut n_labels as *mut _,
        &mut buffer[0] as *mut _
    );
    assert!(status == 0);
}

#[cfg(feature="ipp")]
pub unsafe fn ipp_flood_fill(mut img : WindowMut<'_, u8>, seed : (usize, usize), label : u8) {

    use crate::foreign::ipp::ippi::*;
    use crate::foreign::ipp::ippcv;
    let (step, sz) = crate::image::ipputils::step_and_size_for_window_mut(&img);

    let mut buf_sz : i32 = 0;
    let status =  ippcv::ippiFloodFillGetSize(mem::transmute(sz), &mut buf_sz as *mut _);
    assert!(status == 0 && buf_sz > 0);
    let mut buffer : Vec<u8> = Vec::from_iter((0..buf_sz).map(|_| 0u8 ) );

    // value is the grayscale value of the connected component, rect the bounding rectangle.
    let mut conn_comp = ippcv::IppiConnectedComp {
        area: 0.,
        value: [0., 0., 0.],
        rect: ippcv::IppiRect { x : 0, y : 0, width : 0, height : 0 }
    };

    // 4-neighborhood
    let status = ippcv::ippiFloodFill_4Con_8u_C1IR(
        img.offset_ptr_mut(),
        step,
        mem::transmute(sz),
        ippcv::IppiPoint { x : seed.1 as i32, y : seed.0 as i32},
        label,
        &mut conn_comp as *mut _,
        &mut buffer[0] as *mut _
    );

    // 8-Neighborhood
    /*IppStatus ippiFloodFill_8Con_<mod>(Ipp<DataType>* pImage, int imageStep, IppiSize
    roiSize, IppiPoint seed, Ipp<datatype> newVal, IppiConnectedComp* pRegion, Ipp8u*
    pBuffer);*/

    // IppStatus ippiFloodFillGetSize_Grad(IppiSize roiSize, int* pBufSize);

    // Grad mode (validate all pixels such that the pixel is within (min_delta, max_delta) of at least one of its already filled neighbors
    // IppStatus ippiFloodFill_Grad4Con_<mod>(Ipp<DataType>* pImage, int imageStep, IppiSize
    // roiSize, IppiPoint seed, Ipp<datatype>* pNewVal, Ipp<datatype>* pMinDelta,
    // Ipp<datatype>* pMaxDelta, IppiConnectedComp* pRegion, Ipp8u* pBuffer);

    // Range mode (validate all pixels within min_delta, max_delta of seed)
    // IppStatus ippiFloodFill_Range4Con_<mod>(Ipp<DataType>* pImage, int imageStep, IppiSize
    // roiSize, IppiPoint seed, Ipp<datatype>* pNewVal, Ipp<datatype>* pMinDelta,
    // Ipp<datatype>* pMaxDelta, IppiConnectedComp* pRegion, Ipp8u* pBuffer);
}

// After Burger & Burge (p. 203)
pub fn depth_flood_fill(mut win : WindowMut<'_, u8>, seed : (usize, usize), color : u8, label : u8) {
    let mut pxs = Vec::new();
    pxs.push(seed);
    let (w, h) = (win.width(), win.height());
    while let Some(px) = pxs.pop() {
        if win[px] == color {
            win[px] = label;
            if px.1 < w-1 {
                pxs.push((px.0, px.1 + 1));
            }
            if px.0 < h-1 {
                pxs.push((px.0 + 1, px.1));
            }
            if px.0 > 1 {
                pxs.push((px.0-1, px.1));
            }
            if px.1 > 1 {
                pxs.push((px.0, px.1 - 1));
            }
        }
    }
}

// After Burger & Burge (p. 203)
pub fn breadth_flood_fill(mut win : WindowMut<'_, u8>, seed : (usize, usize), color : u8, label : u8) {
    let mut pxs = VecDeque::new();
    pxs.push_back(seed);
    let (w, h) = (win.width(), win.height());
    while let Some(px) = pxs.pop_front() {
        if win[px] == color {
            win[px] = label;
            if px.1 < w-1 {
                pxs.push_back((px.0, px.1 + 1));
            }
            if px.0 < h-1 {
                pxs.push_back((px.0 + 1, px.1));
            }
            if px.0 > 1 {
                pxs.push_back((px.0-1, px.1));
            }
            if px.1 > 1 {
                pxs.push_back((px.0, px.1 - 1));
            }
        }
    }
}

use petgraph::{graph::DiGraph, graph::NodeIndex, data::Build, Direction};

pub struct QuadTree<'a, N>(DiGraph<Window<'a, N>, ()>) where N : Scalar + Clone + Copy + Debug;

impl<'a, N> AsRef<DiGraph<Window<'a, N>, ()>> for QuadTree<'a, N>
where
    N : Scalar + Clone + Copy + Debug
{

    fn as_ref(&self) -> &DiGraph<Window<'a, N>, ()> {
        &self.0
    }
}

impl<'a, N> QuadTree<'a, N>
where
    N : Scalar + Clone + Copy + Debug
{

    /// Return the four window roots (same as self.levels(0)). All windows will
    /// have size len/2
    pub fn root_widows(&'a self) -> impl Iterator<Item=Window<'a, N>> + 'a {
        self.0.externals(Direction::Incoming).map(move |ix| self.0[ix].clone() )
    }

    /// Returns an unknown number of window leafs (i.e. windows without further decomposition).
    /// Windows might have different sizes.
    pub fn leaf_windows(&'a self) -> impl Iterator<Item=Window<'a, N>> + 'a {
        self.0.externals(Direction::Outgoing).map(move |ix| self.0[ix].clone() )
    }

    /// Returns all windows of this quadtree, including the intermediate
    /// partitions.
    pub fn all_windows(&'a self) -> impl Iterator<Item=Window<'a, N>> + 'a {
        self.0.raw_nodes().iter().map(move |node| node.weight.clone() )
    }

}

/// Recursively partition the image into a quad-tree. Define regions by the equality
/// or small difference of all pixels within a quadtree region. A reasonabe segmentation
/// strategy is to start with a quadtree segmentation over a coarse version of the image,
/// and use the center of each rect in the graph as the seed for growth strategies at
/// the full image scale. In this way, small or large regions depending on the application
/// can be ruled out before processing at the more expensive detail scale. This does a recursive
/// call, and will fail if you don't set the minimum size to a value that will not exhaust the
/// call stack. The criteria function should be any transitive property (which guarantees pairwise
/// comparison between any two pixels means that the relation holds for all pixels). The quadtree
/// segmenter is costly, since each pixel might be evaluated as many times as the region must be
/// divided further; so restrict its use when your image is small and/or reasonably well delimited,
/// and you will benefit from its richer representation (i.e. you want to filter out windows with
/// a given size for further processing). It is trivial to build a parallel version
/// of this algorithm, since each region is evaluated independently of the others (just keep the
/// graph inside a Arc<Mutex<T>>, locking the mutex to append nodes only, leaving the pixel
/// evaluation to run in parallel). The decomposition stops when crit is satisfied for all pixel
/// pairs or the minimum window size is achieved. QuadTree might also be useful when processing videos,
/// where large homogeneous regions are expected to be stabe (just sample a few pixels randomly or uniformly
/// or at its edges to verify if the region is still stable at a next frame. If it is, ignore it for
/// processing at the current frame). This is basically the split-and-merge segmentation strategy.
pub fn quad_tree_segmenter<'a>(
    win : &'a Window<'a, u8>,
    crit : fn(u8,u8)->bool,
    min_sz : usize
) -> QuadTree<'a, u8> {

    assert!(win.width() % 4 == 0);
    assert!(win.height() % 4 == 0);

    let mut qt = DiGraph::new();

    // Iterate over top-level elements of quad-tree, depth-first
    for w in win.clone().equivalent_windows(2, 2) {
        let ix = qt.add_node(w.clone());
        serial_quad_tree_segmenter_step(&mut qt, w.clone(), ix, crit, min_sz);
    }

    QuadTree(qt)
}

// Iterate over a newly-inserted node, depth-first.
fn serial_quad_tree_segmenter_step<'a>(
    qt : &mut DiGraph<Window<'a, u8>, ()>,
    parent_win : Window<'a, u8>,
    parent_ix : NodeIndex,
    crit : fn(u8, u8)->bool,
    min_sz : usize
) {

    // By transtiveness of equality (or difference), the comparison of any two pixels
    // means the satisfaction of the condition for all pixels. The comparison of left with
    // right is purely by convention and convenience (any other exhaustive pairwise comparison
    // that contains all pixels would work). The "small difference" criteria is not really transitive,
    // but we approximate it nevertheless.
    let all_satisfy = parent_win.pixels(1).zip(parent_win.pixels(1).skip(1)).all(|(a, b)| crit(*a, *b) );

    if !all_satisfy && parent_win.width() / 2 >= min_sz && parent_win.height() / 2 >= min_sz {

        // Iterate over new region, depth-first
        for child_win in parent_win.clone().equivalent_windows(2, 2) {
            let child_ix = qt.add_node(child_win.clone());
            qt.add_edge(parent_ix, child_ix, ());
            serial_quad_tree_segmenter_step(qt, child_win, child_ix, crit, min_sz);
        }
    }

    // If all pixels satisfy the condition, do nothing, keeping the currently-inserted node as a left node.
    // If minimum window size was reached at previous iteration, also do nothing.
}

/*
TODO a better algorithm starts by comparing all pairwise pixels within windows of min_sz over the
whole window. Then, we merge those smaller windows by comparing the first pixel of each subwindow,
and so on and so forth, until windows get to the size of length/2.
*/

/// Builds a LUT between labels and pixels, such that the color for each label
/// is the color of the first pixel found with that label.
pub fn recolor_with_labels<'a>(mut img : WindowMut<'a, u8>, labels : &'a Window<'a, u8>) {

    assert!(img.width() == labels.width() && img.height() == labels.height());

    let mut lut = HashMap::new();

    /*for (px, lbl) in img.pixels_mut(1).zip(labels.pixels(1)) {
        match lut.get(lbl) {
            Some(color) => {
                *px = *color;
            },
            None => {
                lut.insert(lbl, *px);
                // Just keep the pixel color the first time this label is found.
            }
        }
    }*/
    for r in 0..img.height() {
        for c in 0..img.width() {
            let lbl = img[(r, c)];
            match lut.get(&lbl) {
                Some(color) => {
                    img[(r, c)] = *color;
                },
                None => {
                    lut.insert(lbl, img[(r, c)]);
                }
            }
        }
    }
}
