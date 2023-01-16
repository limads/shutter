#![allow(warnings)]

#![doc(html_logo_url = "https://raw.githubusercontent.com/limads/shutter/master/assets/logo.png")]

// #![feature(int_log)]

/**

# Basic image processing

Operations defined for raw image buffers that also result in raw image buffers

| Operation                                             | u8 | i16 | i32 | f32 |
|:-----------------------------------------------------:|:--:|:---:|:---:|:---:|
| Arithmetic (add, sub, mul, div)                       | ✓  |  ✓  |  ✓  |  ✓  |
| Scalar (add, sub, mul, div, invert)                   | ✓  |  ✓  |  ✓  |  ✓  |
| Convolution                                           | ✓  |  ✓  |  ✓  |  ✓  |
| Non-linear filtering (median, rank-order)             | ✓  |  ✓  |  ✓  |  ✓  |
| Resampling                                            | ✓  |  ✓  |  ✓  |  ✓  |
| Warping                                               | ✓  |  ✓  |  ✓  |  ✓  |
| Polling (min, max, median)                            | ✓  |  ✓  |  ✓  |  ✓  |
| Logical (and, or, not, xor)                           | ✓  |     |     |     |
| Grayscale (threshold, truncate, dither, quantization) | ✓  |     |     |     |
| Morphology (erode, dilate, thin, thicken)             | ✓  |     |     |     |
| Signed (abs, truncate)                                |    |  ✓  |  ✓  |  ✓  |

# Transforms

IntegralTransform, DistanceTransform, DiffShiftTransform

# Spatial encoding

Operations define lossless binary image representations (operating on u8) in
terms of spatial encodings (containers of foreground pixel positions and/or displacements).

RunLengthEncoding
PointEncoding
ChainEncoding

# Shape detection, operating exclusively on u8

HoughLines

# Segmentation (i.e. disconnected region labeling), operating exclusively on u8

# Corner detection

# Edge detection

# Texture/pixel statistics

# Motion analysis

# Digital geometry utilities (geom module)

# Raster drawing utilities (text, circle, rect, lines)

Canny

**/

mod export;

pub mod image;

pub mod io;

pub mod ops;

pub mod stat;

pub mod scalar;

pub mod gray;

pub mod hist;

pub mod draw;

pub mod shape;

pub mod conv;

pub mod local;

pub mod morph;

pub mod patch;

pub mod code;

pub mod graph;

pub mod texture;

pub mod warp;

pub mod optics;

pub mod stereo;

// pub mod motion;

pub mod resample;

pub mod edge;

pub mod integral;

pub mod bitonal;

pub mod pyr;

/// Common optical manipulations (calculation of focal depth, magnification, camera calibration, lens distortion correction),
// pub mod optics;

// pub mod warp;

// pub mod edge;

// pub mod convert;

// pub mod resample;

// pub mod motion;

// pub mod stereo;

// pub mod object;

// pub mod graph;

/*
TODO traits:

SignedImage : i16 i32 (abs)

UnsignedImage : u8

FloatImage : f32

AnyImage : (abs_diff)

DepthConvert

increase_depth

decrease_depth

// depth_to_interval / interval_to_depth saturates all fp values outside the interval.
float_depth -> linearize_depth / depth_to_interval / interval_to_depth

*/

// Structures and algorithms for sparse binary image representation (in terms of graphs, ordered arrays, etc).
// Offers Run-length encoding of binary image representation, and other alternative representations.
// pub mod sparse;

// Contains algorithms to partition the images over homogeneous regions, with a criterion for
// homegeneity that is algorithm-specific. Contains dense and sparse data structures to represent
// image regions.
// pub mod region;

// Binary image operations
// pub mod binary;

// Pattern and texture analysis operations (segmentation, classification)
// pub mod texture;

/// Image-to-image grayscale operations (thresholding, segmentation)
// pub mod gray;

// Color image operations
// pub mod color;

// Point-wise operations (enhance, equalize, normalize, etc).
// pub mod point;

// Global operations (mean, max, avg)
// pub mod global;

// Local (non-filtering) operations (median, min, max)
// pub mod local;

// pub mod ffi;

// pub(crate) mod foreign;

pub mod profile;

pub mod util;

// Scalar image operations
// pub mod scalar;

// Low-level image features
// pub mod feature;

// pub mod path;

// pub mod raster;

// #[cfg(feature="opencvlib")]
// pub mod tracking;

// pub mod template;

// #[cfg(feature="opencvlib")]
// pub mod matching;

// #[cfg(feature="opencvlib")]
// pub mod flow;

// #[cfg(feature="opencv")]
// pub mod contour;

// pub mod corner;



// pub mod edge;

// pub mod threshold;

// pub mod feature;

// pub mod filter;

// Defines operations on binary images.
// #[cfg(feature="opencv")]
// pub mod morph;

// #[cfg(feature="opencv")]
// pub mod filter;

// pub mod cluster;

// pub mod segmentation;

// #[cfg(feature="opencv")]
// pub mod geom;

// #[cfg(feature="opencvlib")]
// pub mod detection;

pub mod foreign;

/*pub mod prelude {

    pub use super::image::*;

    pub use super::raster::*;

    pub use super::local::*;

    pub use super::convert::*;

    pub use super::resample::*;

    pub use super::draw::*;

}*/

/*IppStatus ippiApplyHaarClassifier_32f_C1R(const Ipp32f* pSrc, int srcStep, const
Ipp32f* pNorm, int normStep, Ipp8u* pMask, int maskStep, IppiSize roiSize, int*
pPositive, Ipp32f threshold, IppiHaarClassifier_32f* pState );

IppStatus ippiLBPImageMode3x3_<mod>(const Ipp<srcDatatype>* pSrc, int srcStep,
Ipp<dstDatatype>* pDst, int dstStep, IppiSize dstRoiSize, int mode, IppiBorderType
borderType, const Ipp<srcDatatype>* borderValue );
*/

/*#[cfg(feature="mlua")]
impl mlua::UserData for crate::image::Image<u8> {

    fn add_methods<'lua, M: mlua::UserDataMethods<'lua, Self>>(methods: &mut M) {

        use crate::image::*;

        methods.add_method("show", |_ : &mlua::Lua, this : &Image<u8>, _: ()| {
            this.show();
            Ok(())
        });

    }

}

#[cfg(feature="mlua")]
#[mlua::lua_module]
fn libshutter(lua : &mlua::Lua) -> mlua::Result<mlua::Table> {

    use mlua::{Table, Lua};
    use crate::image::*;

    let exports = lua.create_table()?;

    exports.set("open", lua.create_function(|_ : &Lua, path : String|->mlua::Result<Image<u8>> {
        let img = crate::io::decode_from_file(&path)
            .map_err(|e| mlua::Error::RuntimeError(format!("{}",e)) )?;
        Ok(img)
    })?)?;

    Ok(exports)
}*/

/*
/// This should contain Patch (what is today at segmentation) Keypoint and similar structures and algorithms.

/*IppStatus ippiFastN_8u_C1R(const Ipp8u* pSrc, int srcStep, Ipp8u* pDstCorner, int
dstCornerStep, Ipp8u* pDstScore, int dstScoreStep, int* pNumCorner, IppiPoint
srcRoiOffset, IppiSize dstRoiSize, IppiFastNSpec* pSpec, Ipp8u* pBuffer );

IppStatus ippiHarrisCorner_8u32f_C1R(const Ipp8u* pSrc, int srcStep, Ipp32f* pDst, int
dstStep, IppiSize roiSize, IppiDifferentialKernel filterType, IppiMaskSize filterMask,
Ipp32u avgWndSize, float k, float scale, IppiBorderType borderType, Ipp8u borderValue,
Ipp8u* pBuffer );*/

/*IppStatus ippiCanny_16s8u_C1R(Ipp16s* pSrcDx, int srcDxStep, Ipp16s* pSrcDy, int
srcDyStep, Ipp8u* pDstEdges, int dstEdgeStep, IppiSize roiSize, Ipp32f lowThreshold,
Ipp32f highThreshold, Ipp8u* pBuffer );*/

/*IppStatus ippiEigenValsVecs_8u32f_C1R(const Ipp8u* pSrc, int srcStep, Ipp32f* pEigenVV,
int eigStep, IppiSize roiSize, IppiKernelType kernType, int apertureSize, int
avgWindow, Ipp8u* pBuffer );

IppStatus ippiHOG_<mod>(const Ipp<srcDatatype>* pSrc, int srcStep, IppiSize roiSize,
const IppiPoint* pLocation, int nLocations, Ipp32f* pDst, const IppiHOGSpec* pHOGSpec,
IppiBorderType borderID, Ipp<srcDatatype> borderValue, Ipp8u* pBuffer );

IppStatus ippiHoughLine_8u32f_C1R(const Ipp8u* pSrc, int srcStep, IppiSize roiSize,
IppPointPolar delta, int threshold, IppPointPolar* pLine, int maxLineCount, int*
pLineCount, Ipp8u* pBuffer );

IppStatus ippiFloodFill_4Con_<mod>(Ipp<datatype>* pImage, int imageStep, IppiSize
roiSize, IppiPoint seed, Ipp<datatype> newVal, IppiConnectedComp* pRegion, Ipp8u*
pBuffer );
*/

/// Contains utilities to represent dense pixel regions (lists of pixel coordinates)
pub mod patch;

pub mod corner;

// Contains utilities for shape analysis (regions defined by their boundary or contour).
// pub mod shape;

pub mod point;

// pub mod edge;

// pub mod color;

*/

