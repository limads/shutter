use crate::image::*;
use std::collections::{VecDeque, HashMap};
use std::iter::FromIterator;
use std::mem;
use std::convert::AsRef;
use std::fmt::Debug;
use nalgebra::Scalar;
use crate::gray::Foreground;
use std::cmp::Ordering;
use crate::draw::*;

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

/*pub struct Boundary {

}

impl Boundary {

    /// Extract a dense pixel coordinate vector from a labeled window.
    pub fn extract(labels : &Window<'_, u8>) -> Vec<Self> {
        unimplemented!()
    }
}*/

/*pub struct Contour {

}

impl Contour {

    /// Extract a dense pixel coordinate vector from a labeled window.
    pub fn extract(labels : &Window<'_, u8>) -> Vec<Self> {
        unimplemented!()
    }
}*/

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

/*/// Segmenters use fillers and/or tracers to separate distinct image regions.
/// Labels images using fillers or other strategies. (i.e. attribute unique integer labels of same numeric type as the image
/// for foreground regions that are disconnected; connected foreground images have the
/// same label). Segmenters can output a labeled image via segment_labels or output a set of Region
/// implementors (patches, boundaries and contours).
pub trait Segmenter {

    fn segment_labels();

    fn segment_boundaries();

    fn segment_contours();

}*/

// Labeling of a binary image specified by Myler & Weeks (p. 75). Joined regions marked as foreground
// with size > max_clust_dist will be marked as new regions, even if they are connected.
// The image will be labeled with the cluster index label. The algorithm will fail if more than
// 254 clusters are found, and will stop iterating in this case. This can be used for binary images
// over edges, using a distance that considers both orientation and magnitude.
fn rough_binary_region_labelling(mut win : WindowMut<'_, u8>, max_clust_dist : f32) 
where
    for<'a> &'a mut [u8] : StorageMut<u8> ,
{

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
                    let dist_k = crate::shape::point_euclidian((r, c), last_clust_px[k]);
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
pub unsafe fn ipp_watershed(mut labels : WindowMut<'_, u8>, win : &Window<'_, u8>) 
where
    for<'a> &'a mut [u8] : StorageMut<u8> ,
    for<'a> &'a [u8] : Storage<u8> 
{

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
        win.as_ptr(),
        src_step,
        labels.as_mut_ptr(),
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
pub unsafe fn ipp_segment_gradient(mut labels : WindowMut<'_, u8>, win : WindowMut<'_, u8>) 
where
    for<'a> &'a mut [u8] : StorageMut<u8> ,
    for<'a> &'a [u8] : Storage<u8> 
{

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
        win.as_ptr(),
        step,
        labels.as_mut_ptr(),
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
        win.as_mut_ptr(),
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
        win.as_mut_ptr(),
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
        img.as_mut_ptr(),
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

/*#[test]
fn homogeneous_regions() {

    use crate::prelude::*;
    use crate::feature::patch::*;
    use crate::feature::patch::raster::*;
    use crate::draw::*;
    use crate::local::*;
    use crate::gray::*;

    let img = crate::io::decode_from_file("/home/diego/Downloads/eyes.png").unwrap();
    img.show();

    let mut fimg = Image::<f32>::new_constant(img.height(), img.width(), 0.);
    fimg.full_window_mut().convert_from(img.full_window(), Conversion::Shrink);
    let out = fimg.full_window().convolve(&crate::local::edge::LAPLACE, Convolution::Linear);

    let mut out_c = Image::<u8>::new_constant(out.height(), out.width(), 0);
    out_c.full_window_mut().convert_from(out.full_window(), Conversion::Stretch);
    // out_c.show();
    //let out_thr = FixedThreshold::new(Foreground::Above(10)).threshold(&out_c.full_window());
    //let mut out = out_thr.clone();
    //crate::gray::supress_binary_speckles(out_thr.as_ref(), out.full_window_mut(), 3);

    // let mut seg = RasterSegmenter::new(out_thr.shape(), 1);
    // for patch in seg.segment_single_color(&out_thr.full_window(), |px| px == 0, ExpansionMode::Contour).iter() {
    //    out_thr.full_window_mut().draw_patch_contour(&patch, 255);
    // }

    // out.show();
}*/

pub struct PointExtractor {
    pts : Vec<(usize, usize)>
}

impl PointExtractor {

    pub fn new() -> Self {
        Self { pts : Vec::new() }
    }

    /// Extract all points of image with threshold above a given point.
    pub fn extract(&mut self, win : &Window<'_, u8>, fg : Foreground) -> &[(usize, usize)] {
        self.pts.clear();
        match fg {
            Foreground::Exactly(v) => {
                win.labeled_pixels::<usize, _>(1)
                    .for_each(|(r, c, px)| if px == v { self.pts.push((r, c)) } );
            },
            Foreground::Below(v) => {
                win.labeled_pixels::<usize, _>(1)
                    .for_each(|(r, c, px)| if px <= v { self.pts.push((r, c)) } );
            },
            Foreground::Above(v) => {
                win.labeled_pixels::<usize, _>(1)
                    .for_each(|(r, c, px)| if px >= v { self.pts.push((r, c)) } );
            },
            Foreground::Inside(a, b) => {
                win.labeled_pixels::<usize, _>(1)
                    .for_each(|(r, c, px)| if px >= a && px <= b { self.pts.push((r, c)) } );
            },
            Foreground::Outside(a, b) => {
                win.labeled_pixels::<usize, _>(1)
                    .for_each(|(r, c, px)| if px <= a && px >= b { self.pts.push((r, c)) } );
            }
        }
        &self.pts[..]
    }

}

#[cfg(feature="opencv")]
pub fn connected_components<S>(
    win : &Image<u8, S>, 
    out : &mut WindowMut<'_, u16>
) -> Vec<(usize, usize, usize, usize)> 
where
    u8 : Pixel,
    S : Storage<u8>
{

    use opencv::core::*;
    use opencv::imgproc::*;
    use opencv::prelude::*;

    let input : Mat = win.full_window().into();
    let mut output : Mat = out.into();

    let mut centroids = Mat::default();
    let mut stats = Mat::default();
    let n_labels = opencv::imgproc::connected_components_with_stats(
        &input,
        &mut output,
        &mut stats,
        &mut centroids,
        8,
        CV_16U
    ).unwrap();
    let mut rects = Vec::new();
    for lbl in 0..(n_labels as usize) {
        // let x = centroids.at_2d::<f64>(label, 0);
        // let y = centroids.at_2d::<f64>(label, 1);
        let left = stats.at_2d::<i32>(lbl as i32, CC_STAT_LEFT).unwrap();
        let top = stats.at_2d::<i32>(lbl as i32, CC_STAT_TOP).unwrap();
        let width = stats.at_2d::<i32>(lbl as i32, CC_STAT_WIDTH).unwrap();
        let height = stats.at_2d::<i32>(lbl as i32, CC_STAT_HEIGHT).unwrap();
        rects.push((*top as usize, *left as usize, *height as usize, *width as usize));
        // let area = stats.at(label, CC_STAT_AREA);
    }
    rects
}

pub struct GraphSegmenter {

}

impl GraphSegmenter {

    pub fn calculate() {
        // Declare: Graph with each pixel as an unconnected node.
        // Begin:
        // For all pixels:
        //      Examine 8-neighborhood. Is difference < threshold? then add_edge(px, neighborhood)
        // End;
        // The threshold is a function of the set size and global segmentation scale.
    }
}


/*pub struct RegionCache {

    area : RefCell<usize>,

    bbox : RefCell<(usize, usize, usize, usize)>,

}*/

pub trait Segmenter<'a> {

    type Output;

    fn segment(&'a mut self) -> &'a Self::Output;

}

/* impl Segmenter<&[RasterLine]> for RasterSegmenter */
/* impl Segmenter<&[Boundary]> for RasterSegmenter */
/* impl Segmenter<&[Contour]> for RasterSegmenter */
/* impl Segmenter<&[Patch]> for RasterSegmenter */

// Represent a contour that is not necessarily ordered
// (which cannot necessarily be drawn by tracing the positions). If you aim to just calcualte statistics
// such as central moments or view the segment limits, the boundary gives the same
// results as the contour without the computational cost of re-ordering the array.
// Every contour is also a boundary, so impl From<Contour> for Boundary is a simple vector move.
#[derive(Debug, Clone)]
pub struct Boundary(Vec<(usize, usize)>);

// Represents the limits of a segmented region that is ordered in such a way
// so that a line can be traced between the points.
#[derive(Debug, Clone)]
pub struct Contour {

    pxs : Vec<(usize, usize)>,

    bbox : (usize, usize, usize, usize)

}

impl Contour {

    pub fn bounding_box(&self) -> &(usize, usize, usize, usize) {
        &self.bbox
    }

    pub fn contacts(&self, other : &Self) -> bool {
        // crate::feature::shape::rect_contacts(self, other.bbox)
        unimplemented!()
    }

    pub fn center(&self) -> (usize, usize) {
        let bbox = &self.bbox;
        (bbox.0 + bbox.2 / 2, bbox.1 + bbox.3 / 2)
    }

    pub fn width(&self) -> usize {
        self.bbox.3
    }

    pub fn height(&self) -> usize {
        self.bbox.2
    }

    pub fn points(&self) -> &[(usize, usize)] {
        &self.pxs
    }

    /*pub fn bounding_box(&self) -> (usize, usize, usize, usize) {
        let n = self.0.len();
        let top = self.0[0].0;
        let bottom = self.0[n-1].0;
        let left = self.0.iter().min_by(|pt| pt.1 ).copied().unwrap().1;
        let right = self.0.iter().max_by(|pt| pt.1 ).copied().unwrap().1;
        (top, left, bottom - top, right - left)
    }

    pub fn area(&self) -> usize {

    }

    pub fn perimeter(&self) -> usize {
        let n = self.0.len();
        let top_perim = self.0[1].1 - self.0[0].1;
        let bottom_perim = self.0[n-2].1 - self.0[n-1].1;
        let mut perim = top_perim + bottom_perim;
        for ix in 0..n {
            perim += crate::shape::point_euclidian(pt1, pt2);
            perim += crate::shape::point_euclidian(pt3, pt4);
        }
        perim
    }*/

    // Since the RasterSegmenter organizes the image contour
    //
    //    l1 r1
    //   l2   r2
    //  l3      r3
    //
    // as l3 l2 l1 r1 r2 r3 (i.e. left pixels in decreasing order; right pixels in increasing order),
    // we can iterate over the inner pixels without furter
    // calculations by just iterating over this natural order from the middle
    // to the extremities.
    /*pub fn inner_pixels<'a>(&'a self, win : &'a Window<'a, u8>) -> impl Iterator<Item=u8> +'a {
        /*let n = self.pxs.len();
        assert!(n % 2 == 0);
        (0..(n/2+1)).rev().zip((n/2+1)..n)
            .map(move |(left, right)| { assert!(self.pxs[left].0 == self.pxs[right].0); win.row(self.pxs[left].0).unwrap()[self.pxs[left].1..self.pxs[right].1].iter().copied() })
            .flatten()*/
            unimplemented!()
    }*/

}

/*impl AsRef<[(usize, usize)]> for Contour {
}*/

/// Represents a short section of the image rasterization process.
pub struct RasterLines {
    line : usize,
    cols : (usize, usize),
    color : u8
}

/// Holds a run-length encoding (RLE) of an image. Each inner vector at index i
/// represents row i of the image. The elements of the inner vectors represent
/// the start column, end column and a dominant pixel color of the raster scan,
/// whose value depend on the rule adopted: the first rule used the color of the first
/// pixel; the central rule the color of the center-most pixel; the mean rule the
/// running average of the pixel color values (at a slightly higher computational cost).
#[derive(Debug, Clone)]
pub struct Rasterizer(Vec<Vec<(usize, usize, u8)>>);

pub fn overlap_ratio(a : (usize, usize), b : (usize, usize)) -> f32 {
    let overlap_ext = (
        (a.1 as f32 - b.0 as f32).max(0.) -
        (a.0 as f32 - b.0 as f32).max(0.) -
        (a.1 as f32 - b.1 as f32).max(0.)
    ).max(0.);

    let a_ext = (a.1 as f32 - a.0 as f32);
    let b_ext = (b.1 as f32 - b.0 as f32);

    // The largest possible overlap is the same size as the smallest interval.
    assert!(overlap_ext <= a_ext && overlap_ext <= b_ext);

    if a_ext > b_ext {
        overlap_ext / a_ext
    } else {
        overlap_ext / b_ext
    }
}

pub fn interval_overlap(a : (usize, usize), b : (usize, usize)) -> bool {
    /*(a.0 <= b.0 && a.1 >= b.0) ||
        (a.0 <= b.1 && a.1 >= b.1) ||
        (a.0 >= b.0 && a.1 <= b.1) ||
        (a.0 <= b.0 && a.1 >= b.1)*/
    !((a.0 < b.0 && a.1 < b.1) || (a.0 > b.0 && a.1 > b.1))
}

#[derive(Debug, Clone)]
pub struct OpenContour {

    // Holds left and (if applicable) right pixel for each row
    pxs : Vec<(usize, usize)>,

    pxs_back : Vec<(usize, usize)>,

    fst_col_last_row : usize,

    lst_col_last_row : usize,

    fst_row : usize,

    last_row : usize,

    min_col : usize,

    max_col : usize,

    // Updates the color at each raster line. The final color
    // will be the color of the last raster line (which might
    // be undesirable, since it might be an edge).
    col : u8
}

impl OpenContour {

    fn close(mut self) -> Contour {
        let bbox = (self.fst_row, self.min_col, self.last_row - self.fst_row + 1, self.max_col - self.min_col + 1);
        let mut pxs_back = std::mem::take(&mut self.pxs_back);
        let mut pxs = self.pxs;
        pxs.extend(pxs_back.drain(..).rev());
        Contour {
            bbox,
            pxs
        }
    }

}

fn innaugurate_contour(contours : &mut Vec<OpenContour>, row : usize, c1 : usize, c2 : usize, color : u8) {
    let mut pxs = Vec::new();
    let mut pxs_back = Vec::new();
    pxs.push((row, c1));
    assert!(c2 >= c1);
    if c1 != c2 {
        pxs_back.push((row, c2));
    }
    contours.push(OpenContour {
        fst_col_last_row : c1,
        lst_col_last_row : c2,
        col : color,
        fst_row : row,
        last_row : row,
        min_col : c1,
        max_col : c2,
        pxs,
        pxs_back
    });
}

fn overlap_open_contour(open : &OpenContour, this_c1 : usize, this_c2 : usize, this_color : u8, abs_diff : u8) -> Ordering {
    let overlaps = interval_overlap((open.fst_col_last_row, open.lst_col_last_row), (this_c1, this_c2));
    if overlaps {
        Ordering::Equal
    } else {

        // Compare by first. Just as same as comparing by last.
        open.fst_col_last_row.cmp(&this_c1)
    }
}

fn append_rows_to_open(open_contours : &mut [OpenContour], editions : &mut HashMap<usize, (usize, usize, u8)>, r : usize) {
    for (ix, (c1, c2, new_color)) in editions.drain() {

        // TODO just push pixels to the end of vector, and then
        // at the end of contour tracing
        // swap odd entries at 0..n/2 to n/2..n to avoid moving
        // so much data.

        // Insert new left col at even index to preserve contour order
        // open_contours[ix].pxs.insert(0, (r, c1));
        open_contours[ix].pxs.push((r, c1));

        // Insert new right col at odd index to preserve contour order
        // open_contours[ix].pxs.push((r, c2));
        open_contours[ix].pxs_back.push((r, c2));

        open_contours[ix].last_row = r;
        open_contours[ix].fst_col_last_row = c1;
        open_contours[ix].lst_col_last_row = c2;

        // Sets color to the open contour as the color of the last raster segment.
        // open_contours[ix].color = ((open_contours[ix].color as f32 + new_color as f32) / 2.) as u8;
        open_contours[ix].col = new_color;

        if c1 < open_contours[ix].min_col {
            open_contours[ix].min_col = c1;
        }
        if c2 > open_contours[ix].max_col {
            open_contours[ix].max_col = c2;
        }
    }
}

fn close_unmatched(
    open_contours : &mut Vec<OpenContour>,
    closed_contours : &mut Vec<Contour>,
    closed_colors : &mut Vec<u8>,
    matched_contours : &mut Vec<usize>
) {
    // Close all contours that weren't matched at this row.
    let unmatched_contours = (0..open_contours.len())
        .filter(|ix| matched_contours.iter().find(|c| *c == ix ).is_none() );

    // Reverse the iterator and remove from the end to the start so
    // as to not mess up with the indices.
    for ix in unmatched_contours.rev() {
        let open = open_contours.remove(ix);
        closed_colors.push(open.col);
        closed_contours.push(open.close());
    }

    matched_contours.clear();
}

fn innaugurate_segment(row : &mut Vec<(usize, usize, u8)>, sum : &mut f32, c : usize, px : u8) {
    row.push((c, c, px));
    *sum = px as f32;
}

fn extend_segment(row : &mut Vec<(usize, usize, u8)>, sum : &mut f32, px : u8, mean : bool) {
    row.last_mut().unwrap().1 += 1;

    // Update color with the average. If mean=false, will use color of first pixel.
    // An alternative would be to use color of the central pixel, to avoid edge
    // pixels determining the patch color.

    *sum += px as f32;
    let n = (row.last().unwrap().1 - row.last().unwrap().0 + 1) as f32;
    assert!(n >= 1.);
    if mean {
        row.last_mut().unwrap().2 = (*sum / n) as u8;
    }
}

fn trim_to_mean(r : usize, row : &mut Vec<(usize, usize, u8)>, win : &Window<'_, u8>, abs_diff : u8) {
    for seg in row {
        let mut start = seg.0;
        let mut end = seg.1;
        while (win[(r, start)] as i16 - seg.2 as i16).abs() > abs_diff as i16 {
            if start < end {
                start += 1;
            } else {
                break;
            }
        }

        while (win[(r, end)] as i16 - seg.2 as i16).abs() > abs_diff as i16 {
            if end > start {
                end -= 1;
            } else {
                break;
            }
        }

        seg.0 = start;
        seg.1 = end;
    }
}

fn exclude_small(row : &mut Vec<(usize, usize, u8)>, sz : usize) {
    for ix in (0..row.len()).rev() {
        if row[ix].1 - row[ix].0 <= sz {
            row.swap_remove(ix);
        }
    }
    row.sort_by(|a, b| a.0.cmp(&b.0) );
}

fn merge_similar(row : &mut Vec<(usize, usize, u8)>, abs_diff : u8) {
    let mut ix = 0;
    loop {
        if ix == row.len() || ix+1 == row.len() {
            return;
        }

        // Merge only when one of the segments is sufficiently small
        if row[ix].1 - row[ix].0 <= 2 || row[ix+1].1 - row[ix+1].0 <= 2 {
            if (row[ix].2 as i16 - row[ix+1].2 as i16).abs() <= abs_diff as i16 {
                row[ix].1 = row[ix+1].1;
                row[ix].2 = ((row[ix].2 as f32 + row[ix+1].2 as f32) / 2.) as u8;
                row.remove(ix+1);
            } else {
                ix += 1;
            }
        } else {
            ix += 1;
        }
    }
}

fn merge_small_to_left_or_right(r : usize, row : &mut Vec<(usize, usize, u8)>, win : &Window<'_, u8>, abs_diff : u8) {
    let mut ix = 0;
    loop {
        if ix == row.len() || ix+1 == row.len() {
            return;
        }

        // If this isn't a small raster, look for the next one.
        if row[ix].1 - row[ix].0 > 2 {
            ix += 1;
            continue;
        }

        let left = if ix == 0 {
            None
        } else {
            Some(row[ix-1].clone())
        };

        let right = if ix == row.len() - 1 {
            None
        } else {
            Some(row[ix+1].clone())
        };

        match (left, right) {
            (Some(l), None) => {
                if (row[ix].2 as i16 - l.2 as i16).abs() <= abs_diff as i16 {
                    row[ix-1].1 = row[ix].1;
                    row.remove(ix);
                } else {
                    ix += 1;
                }
            },
            (None, Some(r)) => {
                if (row[ix].2 as i16 - r.2 as i16).abs() <= abs_diff as i16 {
                    row[ix+1].0 = row[ix].0;
                    row.remove(ix);
                } else {
                    ix += 1;
                }
            },
            (Some(l), Some(r)) => {
                let left_diff = (row[ix].2 as i16 - l.2 as i16).abs();
                let rigth_diff = (row[ix].2 as i16 - r.2 as i16).abs();
                if left_diff.min(rigth_diff) < abs_diff as i16 {
                    if left_diff < rigth_diff {
                        row[ix-1].1 = row[ix].1;
                    } else {
                        row[ix+1].0 = row[ix].0;
                    }
                    row.remove(ix);
                } else {
                    ix += 1;
                }
            }
            (None, None) => {
                return;
            }
        }
    }
}

/// Since iteration is resolved from left to right,  it is natural the left
/// rasters will have preference fo gobble pixels even when they are best
/// matches for the right pixels. This functionn iterate over the borders,
/// re-assigning pixels to the right if necessary.
fn resolve_disputes(r : usize, row : &mut Vec<(usize, usize, u8)>, win : &Window<'_, u8>) {
    for ix in 1..row.len() {
        let mut n = 1;
        while let Some(left_pos) = row[ix-1].1.checked_sub(n) {
            if left_pos <= row[ix-1].0 {
                break;
            }

            let col_left = row[ix-1].2;
            let col_right = row[ix].2;

            if let Some(px) = win.get((r, left_pos)) {
                let diff_left = (*px as i16 - col_left as i16).abs();
                let diff_right = (*px as i16 - col_right as i16).abs();

                if diff_right < diff_left {
                    row[ix-1].1 -= 1;
                    row[ix].0 -= 1;
                } else {
                    return;
                }
                n += 1;
            } else {
                break;
            }
        }
    }
}

fn update_overlap(open_contours : &[OpenContour], found : usize, sign : isize, this_c1 : usize, this_c2 : usize) -> usize {
    let mut ext : isize = 1;
    let mut new_found = found;
    loop {
        let new_ix = found as isize + ext*sign;
        if new_ix < 0 {
            return new_found;
        }
        if let Some(open) = open_contours.get(new_ix as usize) {
            if interval_overlap((open.fst_col_last_row, open.lst_col_last_row), (this_c1, this_c2)) {
                new_found = new_ix as usize;
                ext += 1;
            } else {
                return new_found;
            }
        } else {
            return new_found;
        }
    }
}

impl Rasterizer {

    // As an object moves through the screen, assume the change in the segmented boundaries
    // can be reasonably modelled by just updating the left and right columns of each raster
    // scan. This means big computational savings, since there is no need to iterate
    // over pixels at the segment middle for segments that stay static across frames, such as background
    // or static objects. This is a reasonable assumption, since objects usually change by being translated in front or
    // outside the screen, and don't usually "pop-in". Objects that "pop-in" wouldn't be
    // captured in situations when they fall exactly in the middle of a segment and don't touch
    // any segmented boundaries.
    // For row r:
    // For segment i of row r:
    // (1) If color start[fst+1] = past color, do nothing
    //     Else while color start[fst+j] != color { start +=1; if !(past_segment) { create_past_segment(color of fst+j) } else { past_segment.end += 1 } j += 1
    // (2) If color[end-1] = past color, do nothing
    //     Else while color of end-j != color { end-=1; if !(next_segment) { create next segment(color of end-j) } else { next_segment.start -= 1} j += 1 }.
    // if len(segment i) == 0 remove it from raster vector.
    //
    // And of course now re-calculate all contours.
    /*pub fn update(&mut self, win : &Window<'_, u8>, abs_diff : u8) {

        for r in 0..self.0.len() {

            // Holds the position where edition will happen insertion and a list of items that will be
            // inserted there instead. If this is empty, just remove the segment at this position.
            let mut new_segs : Vec<(usize, Vec<(usize, usize, u8)>)> = Vec::new();

            for seg in 0..(self.0[r].len()) {
                let curr_start = self.0[r][seg].0;
                let curr_end = self.0[r][seg].1;
                let curr_color = self.0[r][seg].2;

                let color_mismatch_start = (win[(r,curr_start)] as i16 - curr_color as i16).abs() > abs_diff;
                let color_mismatch_end = (win[(r,curr_end)] as i16 - curr_color as i16).abs() > abs_diff;

                if color_mismatch_start && seg == 0 {
                    // If color mismatch at first pixel of first segment

                    if color_mismatch_end {
                        // If mismatch is also at end, re-classify whole segment.

                    } else {
                        // If mismatch is at start but not at end, shrink segment until a match is
                        // found (guaranteed to have at least the end pixel) and re-classify all of the start.

                    }

                } else if color_mismatch_start && seg == self.0[r].len()-1 {
                    // If color mismatch at first pixel of last segment

                    if color_mismatch_end {
                        // If mismatch also at end, re-classify whole segment.
                    } else {
                        // If mismatch at start but not at end, shrink segment until a match
                        // is found (guaranteed to have at least the end pixel) and re-classify
                        // until this match is found.
                    }

                } else if color_mismatch_start {
                    // If color mismatch for any segment in the middle

                    if color_mismatch_end {
                        // Re-classify whole segment.
                    } else {
                        //
                    }
                }

                // If the alteration was not called previously by being tiggered by a start
                // color change, also verify an end change.
                if !color_mismatch_start && color_mismatch_end && seg == self.0[r].len()-1 {
                    // If color mismatch at last pixel of last segment
                }

                /*let mut decrement = 1;
                let mut prev_matches = false;
                if seg > 0 {
                    let prev_color = self.0[r][seg-1].2;
                    loop {
                        if (curr_start as i32 - decrement as i32) < 0 {
                            break;
                        }

                        let this_matches = (win[(r, curr_start-decrement)] as i16 - curr_color as i16).abs() < abs_diff;
                        let prev_matches = (win[(r, curr_start-decrement)] as i16 - prev_color as i16).abs() < abs_diff;

                        match (this_matches, prev_matches) {
                            (false, true) => {
                                decrement += 1;
                            },
                            (false, false) => {
                                // Create new segment
                                let new_seg = ()
                                new_segs.push(new_seg);
                            },
                            (true, false) => {
                                self.0[r][seg].0 -= decrement;
                                self.0[r][seg-1].1 += decrement;
                                break;
                            },
                            (true, true) => {
                                // merge segments
                            }
                        }

                        // Did not match previvous segment; create new in this case.
                        if prev_matches {
                            new_segs.push(curr_start-1);
                        } else {

                        }
                            }
                        }

                if seg < self.0[r].len() - 1 {
                    let post_color = self.0[r][seg+1].2;
                    loop {

                    }
                }*/


            }
        }
    }*/

    pub fn new(height : usize) -> Self {
        Rasterizer((0..height).map(|_| Vec::<(usize, usize, u8)>::new() ).collect())
    }

    pub fn segments(&self) -> &[Vec<(usize, usize, u8)>] {
        &self.0
    }

    /*pub fn calculate_limits(&mut self, win : &Window<'_, u8>, low : u8, high : u8) {
        assert!(win.height() == self.0.len());
        self.0.iter_mut().for_each(|row| row.clear() );
        for (r, c, px) in win.labeled_pixels::<usize, _>(1) {
            if px > low && px < high {
                if self.0[r].len() == 0 {
                    self.0[r].push((c, c, px));
                }

                } else {
                    if (px as i16 - self.0[r].last().unwrap().2 as i16).abs() <= abs_diff as i16 {
                        self.0[r].last_mut().unwrap().1 += 1;

                        // TODO perhaps set color to the mean to improve future matches.
                        // Also keep a pixel sum for each raster to calculate the mean more easily.
                    } else {
                        self.0[r].push((c, c, px));
                    }
                }
            }
        }
    }*/

    // The mean rule updates the color of each raster segment to the current running average,
    // and matches any further pixels to this value. We could also calculate a running variance.
    // This makes for more robust matches, since it avoids blurred pixels at the edges of dominating
    // the color of the patch. A center-most pixel rule could also be adopted.
    pub fn calculate(&mut self, win : &Window<'_, u8>, mean : bool, abs_diff : u8) {
        assert!(win.height() == self.0.len());
        self.0.iter_mut().for_each(|row| row.clear() );
        let mut sum : f32 = 0.;

        for (r, c, px) in win.labeled_pixels::<usize, _>(1) {
            if c == 0 {
                assert!(self.0[r].len() == 0);
                innaugurate_segment(&mut self.0[r], &mut sum, c, px);
            } else {
                let color_match = (px as i16 - self.0[r].last().unwrap().2 as i16).abs() <= abs_diff as i16;
                if color_match {
                    extend_segment(&mut self.0[r], &mut sum, px, mean);
                } else {
                    innaugurate_segment(&mut self.0[r], &mut sum, c, px);
                }
            }
        }

        assert!(self.0.len() == win.height());
        for (r, row) in (self.0.iter_mut().enumerate()) {
            // trim_to_mean(r, row, win, abs_diff);
            // exclude_small(row, 2);
            // merge_similar(row, abs_diff);
            resolve_disputes(r, row, win);
            merge_small_to_left_or_right(r, row, win, abs_diff);
            assert!(row.len() >= 1);
        }
    }

    pub fn contours(&self, abs_diff : u8, req_overlap : f32) -> (Vec<Contour>, Vec<u8>) {
        let mut open_contours : Vec<OpenContour> = Vec::new();
        let mut closed_contours : Vec<Contour> = Vec::new();
        let mut closed_colors : Vec<u8> = Vec::new();

        // Innaugurate contours for first line.
        for (c1, c2, col) in &self.0[0] {
            innaugurate_contour(&mut open_contours, 0, *c1, *c2, *col);
        }

        open_contours.sort_by(|a, b| a.fst_col_last_row.cmp(&b.fst_col_last_row) );

        let mut matched_contours : Vec<usize> = Vec::new();
        let mut new_contours : Vec<OpenContour> = Vec::new();

        // Carries index of opened patch and the new (start, end) columns.
        let mut editions : HashMap<usize, (usize, usize, u8)> = HashMap::new();

        for r in 1..(self.0.len()) {

            // Iterate over the segments in this raster row.
            for (this_c1, this_c2, this_color) in &self.0[r] {

                assert!(open_contours.iter().all(|c| c.last_row == r-1 ));

                // Take any of the potential overlaps
                let res_prev_ix = open_contours.binary_search_by(|open| {
                    overlap_open_contour(open, *this_c1, *this_c2, *this_color, abs_diff)
                });

                if let Ok(cand_ix) = res_prev_ix {

                    let fst_overlap = update_overlap(&open_contours, cand_ix, -1, *this_c1, *this_c2);
                    let lst_overlap = update_overlap(&open_contours, cand_ix, 1, *this_c1, *this_c2);

                    assert!(interval_overlap((open_contours[fst_overlap].fst_col_last_row, open_contours[fst_overlap].lst_col_last_row), (*this_c1, *this_c2)));
                    assert!(interval_overlap((open_contours[lst_overlap].fst_col_last_row, open_contours[lst_overlap].lst_col_last_row), (*this_c1, *this_c2)));

                    let mut matched_color = false;
                    let mut best_ix = fst_overlap;
                    // let mut largest_top = 0;
                    let mut largest_overlap = 0.;
                    let mut closest_color = i16::MAX;
                    for ix in fst_overlap..(lst_overlap+1) {
                        let ovlp_ratio = overlap_ratio((open_contours[ix].fst_col_last_row, open_contours[ix].lst_col_last_row), (*this_c1, *this_c2));
                        let color_match = (open_contours[ix].col as i16 - *this_color as i16).abs() <= abs_diff as i16;
                        // let sz_diff = open_contours[ix].lst_col_last_row - open_contours[ix].fst_col_last_row;
                        let col_diff = (open_contours[ix].col as i16 - *this_color as i16).abs();
                        // if color_match && sz_diff >= largest_top  {
                        if color_match && ovlp_ratio > largest_overlap {
                            best_ix = ix;
                            matched_color = true;
                            // largest_top = sz_diff;
                            largest_overlap = ovlp_ratio;
                            closest_color = col_diff;
                        }
                    }

                    // Case this top segment has been matched to a previous bottom segment (top
                    // matches two or more at bottom, which will happen when we have a larger top raster segment and two smaller
                    // bottom segments), continue the top segment only with the largest match, and
                    // innaugurate a new segment for the current segment.
                    if matched_color {
                        if editions.contains_key(&best_ix) {

                            // Decide to substitute the previous edition for this one based
                            // on which one is the largest. This assumes smaller regions are more
                            // likely to be start and end of contours.

                            // TODO consider the sequence with the closest color instead.
                            // if editions[&best_ix].1 - editions[&best_ix].0 < *this_c2 - *this_c1 {

                            let prev_ratio = overlap_ratio(
                                (open_contours[best_ix].fst_col_last_row, open_contours[best_ix].lst_col_last_row),
                                (editions[&best_ix].0, editions[&best_ix].1)
                            );
                            let curr_ratio = overlap_ratio(
                                (open_contours[best_ix].fst_col_last_row, open_contours[best_ix].lst_col_last_row),
                                (*this_c1, *this_c2)
                            );
                            // if editions[&best_ix].1 - editions[&best_ix].0 < *this_c2 - *this_c1 {
                            if prev_ratio < curr_ratio {

                                // innaugurate_contour(&mut new_contours, r, editions[&best_ix].0, editions[&best_ix].1, editions[&best_ix].2);
                                editions.get_mut(&best_ix).unwrap().0 = *this_c1;
                                editions.get_mut(&best_ix).unwrap().1 = *this_c2;
                                editions.get_mut(&best_ix).unwrap().2 = *this_color;
                            } else {

                                // Previous is larger - Keep it there and innaugurate a new open contour in this case
                                innaugurate_contour(&mut new_contours, r, *this_c1, *this_c2, *this_color);
                            }

                        } else {
                            editions.insert(best_ix, (*this_c1, *this_c2, *this_color));
                            matched_contours.push(best_ix);
                        }
                    } else {
                        innaugurate_contour(&mut new_contours, r, *this_c1, *this_c2, *this_color);
                    }
                } else {

                    // Innaugurate new contour. Insert at another vector so they won't be matched now
                    // and mess with the indices of OpenContours.
                    innaugurate_contour(&mut new_contours, r, *this_c1, *this_c2, *this_color);

                }
            }

            append_rows_to_open(&mut open_contours[..], &mut editions, r);
            close_unmatched(&mut open_contours, &mut closed_contours, &mut closed_colors, &mut matched_contours);

            for contour in new_contours.drain(..) {
                open_contours.push(contour);
            }

            open_contours.sort_by(|a, b| a.fst_col_last_row.cmp(&b.fst_col_last_row) );

            /*// Verify if there are any "holes" generated by the bottom raster segments
            // at the bottom line when they match the top segments. We should create new segments
            // to fill the holes in those cases.
            if let Some(fst) = open_contours.first() {

            }
            for (open_left, open_right)
            if let Some(lst) = open_contours.last() {

            }

            open_contours.sort_by(|a, b| a.fst_col_last_row.cmp(&b.fst_col_last_row) );*/
        }

        // Push remaining contours after all rows are finished
        for open in open_contours {
            closed_colors.push(open.col);
            closed_contours.push(open.close());
        }

        /*// Rearrange pixel order here. Give the option to deliver
        // them unordered as a Boundary.
        for cont in closed_contours.iter_mut() {
            let n = cont.0.len();

            if n > 2 {

                // This will work for contours with even number of elements.
                // But contours might have odd number of elements, since the
                // raster might return a single pixel at the first row to be
                // combined with more pixels at the rows below.
                for i in (1..(n/2)).step_by(2) {
                    assert!(i % 2 != 0);
                    cont.0.swap(i, n - i - 1);
                }
            }
        }*/

        (closed_contours, closed_colors)
    }

}

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
    where N : Pixel + Scalar + Copy + Default,
    &'a [N] : Storage<N>,
    &'a mut [N] : StorageMut<N>,
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

// Draws each raster with its color, and two black points delimiting each raster.
pub fn draw_segments<'a>(raster : &Rasterizer, mut img : WindowMut<'a, u8>) {

    use crate::draw::Mark;

    for (r, segs) in raster.segments().iter().enumerate() {
        for s in segs {
            img.draw(crate::draw::Mark::Line((r, s.0), (r, s.1)), s.2);
            img[(r,s.0)] = 0;
            img[(r,s.1)] = 0;
        }
    }
}
