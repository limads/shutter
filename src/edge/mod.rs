/* Distante & Distante (p. 291)
(1) Find pixel with magn > thresh
(2) Take neighbor at 4 or 8-connected region w/ highest magnitude and similar orientation (within pi/4).
Repeat this step while condition is met.
(3) Case (2) not met
*/

/*IppStatus ippiCannyBorderGetSize(IppiSize roiSize, IppiDifferentialKernel filterType,
IppiMaskSize mask, IppDataType dataType, int* pBufferSize);

IppStatus ippiCannyBorder_8u_C1R(const Ipp8u* pSrc, int srcStep, Ipp8u* pDst, int
dstStep, IppiSize roiSize, IppiDifferentialKernel filterType, IppiMaskSize mask,
IppiBorderType borderType, Ipp8u borderValue, Ipp32f lowThresh, Ipp32f highThresh,
IppNormType norm, Ipp8u* pBuffer);

IppStatus ippiCannyGetSize(IppiSize roiSize, int* pBufferSize);

IppStatus ippiCanny_16s8u_C1R(Ipp16s* pSrcDx, int srcDxStep, Ipp16s* pSrcDy, int
srcDyStep, Ipp8u* pDstEdges, int dstEdgeStep, IppiSize roiSize, Ipp32f lowThreshold,
Ipp32f highThreshold, Ipp8u* pBuffer);

IppStatus ippiCanny_32f8u_C1R(Ipp32f* pSrcDx, int srcDxStep, Ipp32f* pSrcDy, int
srcDyStep, Ipp8u* pDstEdges, int dstEdgeStep, IppiSize roiSize, Ipp32f lowThreshold,
Ipp32f highThreshold, Ipp8u* pBuffer);

*/
