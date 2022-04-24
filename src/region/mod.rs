use crate::image::*;
use std::collections::VecDeque;
use std::iter::FromIterator;

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
// 254 clusters are found, and will stop iterating in this case.
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
unsafe fn ipp_watershed(mut labels : WindowMut<'_, u8>, win : &Window<'_, u8>) {

    use crate::foreign::ipp::ippi::*;
    use crate::image::ipputils;

    assert!(labels.width() == win.width() && labels.height() == windows.height());
    let (src_step, src_sz) = ipputils::step_and_size_for_window(win);
    let (label_step, label_sz)  = ipputils::step_and_size_for_window(win);

    let mut buffer = ipputils::allocate_buffer_with(|buf_sz|
        ippiSegmentWatershedGetBufferSize_8u_C1R(src_sz, buf_sz)
    );

    // Alt: IPP_SEGMENT_DISTANCE + IPP_SEGMENT_BORDER_8
    let flags = IPP_SEGMENT_QUEUE + IPP_SEGMENT_BORDER_4;

    // alt norms: ippiNormInf, ippiNormL1, IppiNormL2, IppiNormFM
    let norm = _IppiNorm_ippiNormInf;
    let status = ippiSegmentWatershed_8u_C1IR(
        win.offset_ptr(),
        src_step,
        labels.offset_ptr_mut(),
        label_step,
        src_sz,
        norm,
        flags,
        &mut buffer[0] as *mut _
    );
    assert!(status == 0);
}

// Grows to the least-gradient direction
#[cfg(feature="ipp")]
unsafe fn ipp_segment_gradient(mut labels : WindowMut<'_, u8>, win : WindowMut<'_, u8>) {

    use crate::foreign::ipp::ippi::*;
    use crate::image::ipputils;

    let (step, sz) = ipputils::step_and_size_for_window(win);
    let mut buffer = ipputils::allocate_buffer_with(|buf_sz|
        ippiSegmentGradientGetBufferSize_8u_C1R(sz, buf_sz)
    );

    // Alt: IPP_SEGMENT_DISTANCE + IPP_SEGMENT_BORDER_8
    let flags = IPP_SEGMENT_QUEUE + IPP_SEGMENT_BORDER_4;

    // alt norms: ippiNormInf, ippiNormL1, IppiNormL2, IppiNormFM
    let norm = _IppiNorm_ippiNormInf;
    let status = ippiSegmentGradient_8u_C1IR(
        win.offset_ptr_mut(),
        step,
        labels.offset_ptr_mut(),
        step,
        sz,
        norm,
        flags,
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
    use crate::foreign::ipp::ippcore::IppDataType_ipp8u;
    use crate::image::ipputils;

    let (step, sz) = ipputils::step_and_size_for_window(win);
    let mut buffer = ipputils::allocate_buffer_with(|buf_sz|
        ippiMarkSpecklesGetBufferSize(sz, IppDataType_ipp8u, 1, buf_sz)
    );

    // alt norms: ippiNormInf, ippiNormL1, IppiNormL2, IppiNormFM
    let norm = _IppiNorm_ippiNormInf;
    let status = ippiMarkSpeckles_8u_C1R(
        win.offset_ptr_mut(),
        step,
        sz,
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
unsafe fn ipp_label_markers(mut win : WindowMut<'_, u8>) {

    use crate::foreign::ipp::ippi::*;
    use crate::image::ipputils;

    let (step, sz) = crate::image::ipputils::step_and_size_for_window(win);
    let mut buffer = ipputils::allocate_buffer_with(|buf_sz| ippiLabelMarkersGetBufferSize_8u_C1IR(sz, buf_sz) );

    // alt norms: ippiNormInf, ippiNormL1, IppiNormL2, IppiNormFM
    let norm = _IppiNorm_ippiNormInf;

    let min_label = 0;
    let max_label = 255;

    let mut n_labels : i32 = 0;
    let status = ippiLabelMarkers_8u_C1IR(
        win.offset_ptr_mut(),
        step,
        sz,
        min_label,
        max_label,
        norm,
        &mut n_labels as *mut _,
        &mut buffer[0] as *mut _
    );
    assert!(status == 0);
}

#[cfg(feature="ipp")]
unsafe fn ipp_flood_fill(img : &Image<u8>, seed : (usize, usize), label : u8) {

    use crate::foreign::ipp::ippi::*;
    let (step, sz) = crate::image::ipputils::step_and_size_for_image(img);

    let mut buf_sz : i32 = 0;
    let status =  ippiFloodFillGetSize(sz, &mut buf_sz as *mut _);
    assert!(status == 0 && buf_sz > 0);
    let mut buffer : Vec<u8> = Vec::from_iter((0..buf_sz).map(|_| 0u8 ) );

    // value is the grayscale value of the connected component, rect the bounding rectangle.
    let mut conn_comp = IppiConnectedComp {
        area: 0.,
        value: [0., 0., 0.],
        rect: IppiRect { x : 0, y : 0, width : 0, height : 0 }
    };

    // 4-neighborhood
    let status = ippiFloodFill_4Con_8u_C1R(
        &img.buf[0] as *const _,
        step,
        sz,
        sz,
        IppiPoint { x : seed.1 as i32, y : seed.0 as i32},
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
fn depth_flood_fill(mut win : WindowMut<'_, u8>, seed : (usize, usize), color : u8, label : u8) {
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
fn breadth_flood_fill(mut win : WindowMut<'_, u8>, seed : (usize, usize), color : u8, label : u8) {
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


