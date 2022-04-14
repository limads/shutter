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

*/

/// Contains utilities to represent dense pixel regions (lists of pixel coordinates)
pub mod patch;

pub mod corner;

/// Contains utilities for shape analysis (regions defined by their boundary or contour).
pub mod shape;

pub mod point;

pub mod edge;

// pub mod color;


