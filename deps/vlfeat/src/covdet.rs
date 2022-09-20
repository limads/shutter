use ::libc;
extern "C" {
    pub type _VlScaleSpace;
    fn atan2(_: libc::c_double, _: libc::c_double) -> libc::c_double;
    fn cos(_: libc::c_double) -> libc::c_double;
    fn sin(_: libc::c_double) -> libc::c_double;
    fn exp(_: libc::c_double) -> libc::c_double;
    fn pow(_: libc::c_double, _: libc::c_double) -> libc::c_double;
    fn sqrt(_: libc::c_double) -> libc::c_double;
    fn ceil(_: libc::c_double) -> libc::c_double;
    fn fabs(_: libc::c_double) -> libc::c_double;
    fn floor(_: libc::c_double) -> libc::c_double;
    fn qsort(
        __base: *mut libc::c_void,
        __nmemb: size_t,
        __size: size_t,
        __compar: __compar_fn_t,
    );
    fn __assert_fail(
        __assertion: *const libc::c_char,
        __file: *const libc::c_char,
        __line: libc::c_uint,
        __function: *const libc::c_char,
    ) -> !;
    fn vl_set_last_error(
        error: libc::c_int,
        errorMessage: *const libc::c_char,
        _: ...
    ) -> libc::c_int;
    fn vl_malloc(n: size_t) -> *mut libc::c_void;
    fn vl_realloc(ptr: *mut libc::c_void, n: size_t) -> *mut libc::c_void;
    fn vl_calloc(n: size_t, size: size_t) -> *mut libc::c_void;
    fn vl_free(ptr: *mut libc::c_void);
    fn vl_imsmooth_f(
        smoothed: *mut libc::c_float,
        smoothedStride: vl_size,
        image: *const libc::c_float,
        width: vl_size,
        height: vl_size,
        stride: vl_size,
        sigmax: libc::c_double,
        sigmay: libc::c_double,
    );
    fn vl_imgradient_polar_f(
        amplitudeGradient: *mut libc::c_float,
        angleGradient: *mut libc::c_float,
        gradWidthStride: vl_size,
        gradHeightStride: vl_size,
        image: *const libc::c_float,
        imageWidth: vl_size,
        imageHeight: vl_size,
        imageStride: vl_size,
    );
    fn vl_imgradient_f(
        xGradient: *mut libc::c_float,
        yGradient: *mut libc::c_float,
        gradWidthStride: vl_size,
        gradHeightStride: vl_size,
        image: *const libc::c_float,
        imageWidth: vl_size,
        imageHeight: vl_size,
        imageStride: vl_size,
    );
    fn abort() -> !;
    fn vl_svd2(
        S: *mut libc::c_double,
        U: *mut libc::c_double,
        V: *mut libc::c_double,
        M: *const libc::c_double,
    );
    fn vl_solve_linear_system_3(
        x: *mut libc::c_double,
        A: *const libc::c_double,
        b: *const libc::c_double,
    ) -> libc::c_int;
    fn vl_solve_linear_system_2(
        x: *mut libc::c_double,
        A: *const libc::c_double,
        b: *const libc::c_double,
    ) -> libc::c_int;
    fn vl_scalespacegeometry_is_equal(
        a: VlScaleSpaceGeometry,
        b: VlScaleSpaceGeometry,
    ) -> vl_bool;
    fn vl_scalespace_get_default_geometry(
        width: vl_size,
        height: vl_size,
    ) -> VlScaleSpaceGeometry;
    fn vl_scalespace_new_with_geometry(geom: VlScaleSpaceGeometry) -> *mut VlScaleSpace;
    fn vl_scalespace_delete(self_0: *mut VlScaleSpace);
    fn vl_scalespace_put_image(self_0: *mut VlScaleSpace, image: *const libc::c_float);
    fn vl_scalespace_get_geometry(self_0: *const VlScaleSpace) -> VlScaleSpaceGeometry;
    fn vl_scalespace_get_octave_geometry(
        self_0: *const VlScaleSpace,
        o: vl_index,
    ) -> VlScaleSpaceOctaveGeometry;
    fn vl_scalespace_get_level(
        self_0: *mut VlScaleSpace,
        o: vl_index,
        s: vl_index,
    ) -> *mut libc::c_float;
    fn vl_scalespace_get_level_sigma(
        self_0: *const VlScaleSpace,
        o: vl_index,
        s: vl_index,
    ) -> libc::c_double;
    fn memcpy(
        _: *mut libc::c_void,
        _: *const libc::c_void,
        _: libc::c_ulong,
    ) -> *mut libc::c_void;
    fn memset(
        _: *mut libc::c_void,
        _: libc::c_int,
        _: libc::c_ulong,
    ) -> *mut libc::c_void;
}
pub type vl_int64 = libc::c_longlong;
pub type vl_uint64 = libc::c_ulonglong;
pub type vl_bool = libc::c_int;
pub type vl_size = vl_uint64;
pub type vl_index = vl_int64;
pub type size_t = libc::c_ulong;
pub type __compar_fn_t = Option::<
    unsafe extern "C" fn(*const libc::c_void, *const libc::c_void) -> libc::c_int,
>;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlEnumerator {
    pub name: *const libc::c_char,
    pub value: vl_index,
}
pub type VlEnumerator = _VlEnumerator;
#[derive(Copy, Clone)]
#[repr(C)]
pub union C2RustUnnamed {
    pub raw: vl_uint64,
    pub value: libc::c_double,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlScaleSpaceGeometry {
    pub width: vl_size,
    pub height: vl_size,
    pub firstOctave: vl_index,
    pub lastOctave: vl_index,
    pub octaveResolution: vl_size,
    pub octaveFirstSubdivision: vl_index,
    pub octaveLastSubdivision: vl_index,
    pub baseScale: libc::c_double,
    pub nominalScale: libc::c_double,
}
pub type VlScaleSpaceGeometry = _VlScaleSpaceGeometry;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlScaleSpaceOctaveGeometry {
    pub width: vl_size,
    pub height: vl_size,
    pub step: libc::c_double,
}
pub type VlScaleSpaceOctaveGeometry = _VlScaleSpaceOctaveGeometry;
pub type VlScaleSpace = _VlScaleSpace;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlFrameOrientedEllipse {
    pub x: libc::c_float,
    pub y: libc::c_float,
    pub a11: libc::c_float,
    pub a12: libc::c_float,
    pub a21: libc::c_float,
    pub a22: libc::c_float,
}
pub type VlFrameOrientedEllipse = _VlFrameOrientedEllipse;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlCovDetFeature {
    pub frame: VlFrameOrientedEllipse,
    pub peakScore: libc::c_float,
    pub edgeScore: libc::c_float,
    pub orientationScore: libc::c_float,
    pub laplacianScaleScore: libc::c_float,
}
pub type VlCovDetFeature = _VlCovDetFeature;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlCovDetFeatureOrientation {
    pub angle: libc::c_double,
    pub score: libc::c_double,
}
pub type VlCovDetFeatureOrientation = _VlCovDetFeatureOrientation;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlCovDetFeatureLaplacianScale {
    pub scale: libc::c_double,
    pub score: libc::c_double,
}
pub type VlCovDetFeatureLaplacianScale = _VlCovDetFeatureLaplacianScale;
pub type _VlCovDetMethod = libc::c_uint;
pub const VL_COVDET_METHOD_NUM: _VlCovDetMethod = 7;
pub const VL_COVDET_METHOD_MULTISCALE_HARRIS: _VlCovDetMethod = 6;
pub const VL_COVDET_METHOD_MULTISCALE_HESSIAN: _VlCovDetMethod = 5;
pub const VL_COVDET_METHOD_HARRIS_LAPLACE: _VlCovDetMethod = 4;
pub const VL_COVDET_METHOD_HESSIAN_LAPLACE: _VlCovDetMethod = 3;
pub const VL_COVDET_METHOD_HESSIAN: _VlCovDetMethod = 2;
pub const VL_COVDET_METHOD_DOG: _VlCovDetMethod = 1;
pub type VlCovDetMethod = _VlCovDetMethod;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlCovDet {
    pub gss: *mut VlScaleSpace,
    pub css: *mut VlScaleSpace,
    pub method: VlCovDetMethod,
    pub peakThreshold: libc::c_double,
    pub edgeThreshold: libc::c_double,
    pub lapPeakThreshold: libc::c_double,
    pub octaveResolution: vl_size,
    pub numOctaves: vl_index,
    pub firstOctave: vl_index,
    pub baseScale: libc::c_double,
    pub maxNumOrientations: vl_size,
    pub nonExtremaSuppression: libc::c_double,
    pub numNonExtremaSuppressed: vl_size,
    pub features: *mut VlCovDetFeature,
    pub numFeatures: vl_size,
    pub numFeatureBufferSize: vl_size,
    pub patch: *mut libc::c_float,
    pub patchBufferSize: vl_size,
    pub transposed: vl_bool,
    pub orientations: [VlCovDetFeatureOrientation; 4],
    pub scales: [VlCovDetFeatureLaplacianScale; 4],
    pub aaAccurateSmoothing: vl_bool,
    pub aaPatch: [libc::c_float; 1681],
    pub aaPatchX: [libc::c_float; 1681],
    pub aaPatchY: [libc::c_float; 1681],
    pub aaMask: [libc::c_float; 1681],
    pub lapPatch: [libc::c_float; 1089],
    pub laplacians: [libc::c_float; 10890],
    pub numFeaturesWithNumScales: [vl_size; 5],
    pub allowPaddedWarping: vl_bool,
}
pub type VlCovDet = _VlCovDet;
pub type VlCovDetExtremum2 = _VlCovDetExtremum2;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlCovDetExtremum2 {
    pub xi: vl_index,
    pub yi: vl_index,
    pub x: libc::c_float,
    pub y: libc::c_float,
    pub peakScore: libc::c_float,
    pub edgeScore: libc::c_float,
}
pub type VlCovDetExtremum3 = _VlCovDetExtremum3;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlCovDetExtremum3 {
    pub xi: vl_index,
    pub yi: vl_index,
    pub zi: vl_index,
    pub x: libc::c_float,
    pub y: libc::c_float,
    pub z: libc::c_float,
    pub peakScore: libc::c_float,
    pub edgeScore: libc::c_float,
}
static mut vl_infinity_d: C2RustUnnamed = C2RustUnnamed {
    raw: 0x7ff0000000000000 as libc::c_ulonglong,
};
#[inline]
unsafe extern "C" fn vl_floor_d(mut x: libc::c_double) -> libc::c_long {
    let mut xi: libc::c_long = x as libc::c_long;
    if x >= 0 as libc::c_int as libc::c_double || xi as libc::c_double == x {
        return xi
    } else {
        return xi - 1 as libc::c_int as libc::c_long
    };
}
#[inline]
unsafe extern "C" fn vl_abs_d(mut x: libc::c_double) -> libc::c_double {
    return x.abs();
}
unsafe extern "C" fn _vl_resize_buffer(
    mut buffer: *mut *mut libc::c_void,
    mut bufferSize: *mut vl_size,
    mut targetSize: vl_size,
) -> libc::c_int {
    let mut newBuffer: *mut libc::c_void = 0 as *mut libc::c_void;
    if (*buffer).is_null() {
        *buffer = vl_malloc(targetSize as size_t);
        if !(*buffer).is_null() {
            *bufferSize = targetSize;
            return 0 as libc::c_int;
        } else {
            *bufferSize = 0 as libc::c_int as vl_size;
            return 2 as libc::c_int;
        }
    }
    newBuffer = vl_realloc(*buffer, targetSize as size_t);
    if !newBuffer.is_null() {
        *buffer = newBuffer;
        *bufferSize = targetSize;
        return 0 as libc::c_int;
    } else {
        return 2 as libc::c_int
    };
}
#[no_mangle]
pub unsafe extern "C" fn vl_find_local_extrema_3(
    mut extrema: *mut *mut vl_index,
    mut bufferSize: *mut vl_size,
    mut map: *const libc::c_float,
    mut width: vl_size,
    mut height: vl_size,
    mut depth: vl_size,
    mut threshold: libc::c_double,
) -> vl_size {
    let mut x: vl_index = 0;
    let mut y: vl_index = 0;
    let mut z: vl_index = 0;
    let xo: vl_size = 1 as libc::c_int as vl_size;
    let yo: vl_size = width;
    let zo: vl_size = width.wrapping_mul(height);
    let mut pt: *const libc::c_float = map
        .offset(xo as isize)
        .offset(yo as isize)
        .offset(zo as isize);
    let mut numExtrema: vl_size = 0 as libc::c_int as vl_size;
    let mut requiredSize: vl_size = 0 as libc::c_int as vl_size;
    z = 1 as libc::c_int as vl_index;
    while z < (depth as libc::c_int - 1 as libc::c_int) as libc::c_longlong {
        y = 1 as libc::c_int as vl_index;
        while y < (height as libc::c_int - 1 as libc::c_int) as libc::c_longlong {
            x = 1 as libc::c_int as vl_index;
            while x < (width as libc::c_int - 1 as libc::c_int) as libc::c_longlong {
                let mut value: libc::c_float = *pt;
                if value as libc::c_double >= threshold
                    && value > *pt.offset(xo as isize)
                    && value > *pt.offset(-(xo as isize))
                    && value > *pt.offset(zo as isize)
                    && value > *pt.offset(-(zo as isize))
                    && value > *pt.offset(yo as isize)
                    && value > *pt.offset(-(yo as isize))
                    && value > *pt.offset(yo as isize).offset(xo as isize)
                    && value > *pt.offset(yo as isize).offset(-(xo as isize))
                    && value > *pt.offset(-(yo as isize)).offset(xo as isize)
                    && value > *pt.offset(-(yo as isize)).offset(-(xo as isize))
                    && value > *pt.offset(xo as isize).offset(zo as isize)
                    && value > *pt.offset(-(xo as isize)).offset(zo as isize)
                    && value > *pt.offset(yo as isize).offset(zo as isize)
                    && value > *pt.offset(-(yo as isize)).offset(zo as isize)
                    && value
                        > *pt.offset(yo as isize).offset(xo as isize).offset(zo as isize)
                    && value
                        > *pt
                            .offset(yo as isize)
                            .offset(-(xo as isize))
                            .offset(zo as isize)
                    && value
                        > *pt
                            .offset(-(yo as isize))
                            .offset(xo as isize)
                            .offset(zo as isize)
                    && value
                        > *pt
                            .offset(-(yo as isize))
                            .offset(-(xo as isize))
                            .offset(zo as isize)
                    && value > *pt.offset(xo as isize).offset(-(zo as isize))
                    && value > *pt.offset(-(xo as isize)).offset(-(zo as isize))
                    && value > *pt.offset(yo as isize).offset(-(zo as isize))
                    && value > *pt.offset(-(yo as isize)).offset(-(zo as isize))
                    && value
                        > *pt
                            .offset(yo as isize)
                            .offset(xo as isize)
                            .offset(-(zo as isize))
                    && value
                        > *pt
                            .offset(yo as isize)
                            .offset(-(xo as isize))
                            .offset(-(zo as isize))
                    && value
                        > *pt
                            .offset(-(yo as isize))
                            .offset(xo as isize)
                            .offset(-(zo as isize))
                    && value
                        > *pt
                            .offset(-(yo as isize))
                            .offset(-(xo as isize))
                            .offset(-(zo as isize))
                    || value as libc::c_double <= -threshold
                        && value < *pt.offset(xo as isize)
                        && value < *pt.offset(-(xo as isize))
                        && value < *pt.offset(zo as isize)
                        && value < *pt.offset(-(zo as isize))
                        && value < *pt.offset(yo as isize)
                        && value < *pt.offset(-(yo as isize))
                        && value < *pt.offset(yo as isize).offset(xo as isize)
                        && value < *pt.offset(yo as isize).offset(-(xo as isize))
                        && value < *pt.offset(-(yo as isize)).offset(xo as isize)
                        && value < *pt.offset(-(yo as isize)).offset(-(xo as isize))
                        && value < *pt.offset(xo as isize).offset(zo as isize)
                        && value < *pt.offset(-(xo as isize)).offset(zo as isize)
                        && value < *pt.offset(yo as isize).offset(zo as isize)
                        && value < *pt.offset(-(yo as isize)).offset(zo as isize)
                        && value
                            < *pt
                                .offset(yo as isize)
                                .offset(xo as isize)
                                .offset(zo as isize)
                        && value
                            < *pt
                                .offset(yo as isize)
                                .offset(-(xo as isize))
                                .offset(zo as isize)
                        && value
                            < *pt
                                .offset(-(yo as isize))
                                .offset(xo as isize)
                                .offset(zo as isize)
                        && value
                            < *pt
                                .offset(-(yo as isize))
                                .offset(-(xo as isize))
                                .offset(zo as isize)
                        && value < *pt.offset(xo as isize).offset(-(zo as isize))
                        && value < *pt.offset(-(xo as isize)).offset(-(zo as isize))
                        && value < *pt.offset(yo as isize).offset(-(zo as isize))
                        && value < *pt.offset(-(yo as isize)).offset(-(zo as isize))
                        && value
                            < *pt
                                .offset(yo as isize)
                                .offset(xo as isize)
                                .offset(-(zo as isize))
                        && value
                            < *pt
                                .offset(yo as isize)
                                .offset(-(xo as isize))
                                .offset(-(zo as isize))
                        && value
                            < *pt
                                .offset(-(yo as isize))
                                .offset(xo as isize)
                                .offset(-(zo as isize))
                        && value
                            < *pt
                                .offset(-(yo as isize))
                                .offset(-(xo as isize))
                                .offset(-(zo as isize))
                {
                    numExtrema = numExtrema.wrapping_add(1);
                    requiredSize = (requiredSize as libc::c_ulonglong)
                        .wrapping_add(
                            (::core::mem::size_of::<vl_index>() as libc::c_ulong)
                                .wrapping_mul(3 as libc::c_int as libc::c_ulong)
                                as libc::c_ulonglong,
                        ) as vl_size as vl_size;
                    if *bufferSize < requiredSize {
                        let mut err: libc::c_int = _vl_resize_buffer(
                            extrema as *mut *mut libc::c_void,
                            bufferSize,
                            requiredSize
                                .wrapping_add(
                                    ((2000 as libc::c_int * 3 as libc::c_int) as libc::c_ulong)
                                        .wrapping_mul(
                                            ::core::mem::size_of::<vl_index>() as libc::c_ulong,
                                        ) as libc::c_ulonglong,
                                ),
                        );
                        if err != 0 as libc::c_int {
                            abort();
                        }
                    }
                    *(*extrema)
                        .offset(
                            (3 as libc::c_int as libc::c_ulonglong)
                                .wrapping_mul(
                                    numExtrema
                                        .wrapping_sub(1 as libc::c_int as libc::c_ulonglong),
                                )
                                .wrapping_add(0 as libc::c_int as libc::c_ulonglong)
                                as isize,
                        ) = x;
                    *(*extrema)
                        .offset(
                            (3 as libc::c_int as libc::c_ulonglong)
                                .wrapping_mul(
                                    numExtrema
                                        .wrapping_sub(1 as libc::c_int as libc::c_ulonglong),
                                )
                                .wrapping_add(1 as libc::c_int as libc::c_ulonglong)
                                as isize,
                        ) = y;
                    *(*extrema)
                        .offset(
                            (3 as libc::c_int as libc::c_ulonglong)
                                .wrapping_mul(
                                    numExtrema
                                        .wrapping_sub(1 as libc::c_int as libc::c_ulonglong),
                                )
                                .wrapping_add(2 as libc::c_int as libc::c_ulonglong)
                                as isize,
                        ) = z;
                }
                pt = pt.offset(xo as isize);
                x += 1;
            }
            pt = pt
                .offset(
                    (2 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize,
                );
            y += 1;
        }
        pt = pt
            .offset((2 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize);
        z += 1;
    }
    return numExtrema;
}
#[no_mangle]
pub unsafe extern "C" fn vl_find_local_extrema_2(
    mut extrema: *mut *mut vl_index,
    mut bufferSize: *mut vl_size,
    mut map: *const libc::c_float,
    mut width: vl_size,
    mut height: vl_size,
    mut threshold: libc::c_double,
) -> vl_size {
    let mut x: vl_index = 0;
    let mut y: vl_index = 0;
    let xo: vl_size = 1 as libc::c_int as vl_size;
    let yo: vl_size = width;
    let mut pt: *const libc::c_float = map.offset(xo as isize).offset(yo as isize);
    let mut numExtrema: vl_size = 0 as libc::c_int as vl_size;
    let mut requiredSize: vl_size = 0 as libc::c_int as vl_size;
    y = 1 as libc::c_int as vl_index;
    while y < (height as libc::c_int - 1 as libc::c_int) as libc::c_longlong {
        x = 1 as libc::c_int as vl_index;
        while x < (width as libc::c_int - 1 as libc::c_int) as libc::c_longlong {
            let mut value: libc::c_float = *pt;
            if value as libc::c_double >= threshold && value > *pt.offset(xo as isize)
                && value > *pt.offset(-(xo as isize)) && value > *pt.offset(yo as isize)
                && value > *pt.offset(-(yo as isize))
                && value > *pt.offset(yo as isize).offset(xo as isize)
                && value > *pt.offset(yo as isize).offset(-(xo as isize))
                && value > *pt.offset(-(yo as isize)).offset(xo as isize)
                && value > *pt.offset(-(yo as isize)).offset(-(xo as isize))
                || value as libc::c_double <= -threshold
                    && value < *pt.offset(xo as isize)
                    && value < *pt.offset(-(xo as isize))
                    && value < *pt.offset(yo as isize)
                    && value < *pt.offset(-(yo as isize))
                    && value < *pt.offset(yo as isize).offset(xo as isize)
                    && value < *pt.offset(yo as isize).offset(-(xo as isize))
                    && value < *pt.offset(-(yo as isize)).offset(xo as isize)
                    && value < *pt.offset(-(yo as isize)).offset(-(xo as isize))
            {
                numExtrema = numExtrema.wrapping_add(1);
                requiredSize = (requiredSize as libc::c_ulonglong)
                    .wrapping_add(
                        (::core::mem::size_of::<vl_index>() as libc::c_ulong)
                            .wrapping_mul(2 as libc::c_int as libc::c_ulong)
                            as libc::c_ulonglong,
                    ) as vl_size as vl_size;
                if *bufferSize < requiredSize {
                    let mut err: libc::c_int = _vl_resize_buffer(
                        extrema as *mut *mut libc::c_void,
                        bufferSize,
                        requiredSize
                            .wrapping_add(
                                ((2000 as libc::c_int * 2 as libc::c_int) as libc::c_ulong)
                                    .wrapping_mul(
                                        ::core::mem::size_of::<vl_index>() as libc::c_ulong,
                                    ) as libc::c_ulonglong,
                            ),
                    );
                    if err != 0 as libc::c_int {
                        abort();
                    }
                }
                *(*extrema)
                    .offset(
                        (2 as libc::c_int as libc::c_ulonglong)
                            .wrapping_mul(
                                numExtrema
                                    .wrapping_sub(1 as libc::c_int as libc::c_ulonglong),
                            )
                            .wrapping_add(0 as libc::c_int as libc::c_ulonglong) as isize,
                    ) = x;
                *(*extrema)
                    .offset(
                        (2 as libc::c_int as libc::c_ulonglong)
                            .wrapping_mul(
                                numExtrema
                                    .wrapping_sub(1 as libc::c_int as libc::c_ulonglong),
                            )
                            .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as isize,
                    ) = y;
            }
            pt = pt.offset(xo as isize);
            x += 1;
        }
        pt = pt
            .offset((2 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize);
        y += 1;
    }
    return numExtrema;
}
#[no_mangle]
pub unsafe extern "C" fn vl_refine_local_extreum_3(
    mut refined: *mut VlCovDetExtremum3,
    mut map: *const libc::c_float,
    mut width: vl_size,
    mut height: vl_size,
    mut depth: vl_size,
    mut x: vl_index,
    mut y: vl_index,
    mut z: vl_index,
) -> vl_bool {
    let xo: vl_size = 1 as libc::c_int as vl_size;
    let yo: vl_size = width;
    let zo: vl_size = width.wrapping_mul(height);
    let mut Dx: libc::c_double = 0 as libc::c_int as libc::c_double;
    let mut Dy: libc::c_double = 0 as libc::c_int as libc::c_double;
    let mut Dz: libc::c_double = 0 as libc::c_int as libc::c_double;
    let mut Dxx: libc::c_double = 0 as libc::c_int as libc::c_double;
    let mut Dyy: libc::c_double = 0 as libc::c_int as libc::c_double;
    let mut Dzz: libc::c_double = 0 as libc::c_int as libc::c_double;
    let mut Dxy: libc::c_double = 0 as libc::c_int as libc::c_double;
    let mut Dxz: libc::c_double = 0 as libc::c_int as libc::c_double;
    let mut Dyz: libc::c_double = 0 as libc::c_int as libc::c_double;
    let mut A: [libc::c_double; 9] = [0.; 9];
    let mut b: [libc::c_double; 3] = [0.; 3];
    let mut pt: *const libc::c_float = 0 as *const libc::c_float;
    let mut dx: vl_index = 0 as libc::c_int as vl_index;
    let mut dy: vl_index = 0 as libc::c_int as vl_index;
    let mut iter: vl_index = 0;
    let mut err: libc::c_int = 0;
    if !map.is_null() {} else {
        __assert_fail(
            b"map\0" as *const u8 as *const libc::c_char,
            b"vl/covdet.c\0" as *const u8 as *const libc::c_char,
            1228 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 127],
                &[libc::c_char; 127],
            >(
                b"vl_bool vl_refine_local_extreum_3(VlCovDetExtremum3 *, const float *, vl_size, vl_size, vl_size, vl_index, vl_index, vl_index)\0",
            ))
                .as_ptr(),
        );
    }
    if 1 as libc::c_int as libc::c_longlong <= x
        && x <= (width as libc::c_int - 2 as libc::c_int) as libc::c_longlong
    {} else {
        __assert_fail(
            b"1 <= x && x <= (signed)width - 2\0" as *const u8 as *const libc::c_char,
            b"vl/covdet.c\0" as *const u8 as *const libc::c_char,
            1229 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 127],
                &[libc::c_char; 127],
            >(
                b"vl_bool vl_refine_local_extreum_3(VlCovDetExtremum3 *, const float *, vl_size, vl_size, vl_size, vl_index, vl_index, vl_index)\0",
            ))
                .as_ptr(),
        );
    }
    if 1 as libc::c_int as libc::c_longlong <= y
        && y <= (height as libc::c_int - 2 as libc::c_int) as libc::c_longlong
    {} else {
        __assert_fail(
            b"1 <= y && y <= (signed)height - 2\0" as *const u8 as *const libc::c_char,
            b"vl/covdet.c\0" as *const u8 as *const libc::c_char,
            1230 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 127],
                &[libc::c_char; 127],
            >(
                b"vl_bool vl_refine_local_extreum_3(VlCovDetExtremum3 *, const float *, vl_size, vl_size, vl_size, vl_index, vl_index, vl_index)\0",
            ))
                .as_ptr(),
        );
    }
    if 1 as libc::c_int as libc::c_longlong <= z
        && z <= (depth as libc::c_int - 2 as libc::c_int) as libc::c_longlong
    {} else {
        __assert_fail(
            b"1 <= z && z <= (signed)depth - 2\0" as *const u8 as *const libc::c_char,
            b"vl/covdet.c\0" as *const u8 as *const libc::c_char,
            1231 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 127],
                &[libc::c_char; 127],
            >(
                b"vl_bool vl_refine_local_extreum_3(VlCovDetExtremum3 *, const float *, vl_size, vl_size, vl_size, vl_index, vl_index, vl_index)\0",
            ))
                .as_ptr(),
        );
    }
    iter = 0 as libc::c_int as vl_index;
    while iter < 5 as libc::c_int as libc::c_longlong {
        x += dx;
        y += dy;
        pt = map
            .offset((x as libc::c_ulonglong).wrapping_mul(xo) as isize)
            .offset((y as libc::c_ulonglong).wrapping_mul(yo) as isize)
            .offset((z as libc::c_ulonglong).wrapping_mul(zo) as isize);
        Dx = 0.5f64
            * (*pt
                .offset(
                    (1 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize,
                )
                .offset(
                    (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize,
                )
                .offset(
                    (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(zo) as isize,
                )
                - *pt
                    .offset(
                        (-(1 as libc::c_int) as libc::c_ulonglong).wrapping_mul(xo)
                            as isize,
                    )
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize,
                    )
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(zo) as isize,
                    )) as libc::c_double;
        Dy = 0.5f64
            * (*pt
                .offset(
                    (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize,
                )
                .offset(
                    (1 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize,
                )
                .offset(
                    (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(zo) as isize,
                )
                - *pt
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize,
                    )
                    .offset(
                        (-(1 as libc::c_int) as libc::c_ulonglong).wrapping_mul(yo)
                            as isize,
                    )
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(zo) as isize,
                    )) as libc::c_double;
        Dz = 0.5f64
            * (*pt
                .offset(
                    (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize,
                )
                .offset(
                    (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize,
                )
                .offset(
                    (1 as libc::c_int as libc::c_ulonglong).wrapping_mul(zo) as isize,
                )
                - *pt
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize,
                    )
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize,
                    )
                    .offset(
                        (-(1 as libc::c_int) as libc::c_ulonglong).wrapping_mul(zo)
                            as isize,
                    )) as libc::c_double;
        Dxx = (*pt
            .offset((1 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize)
            .offset((0 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize)
            .offset((0 as libc::c_int as libc::c_ulonglong).wrapping_mul(zo) as isize)
            + *pt
                .offset(
                    (-(1 as libc::c_int) as libc::c_ulonglong).wrapping_mul(xo) as isize,
                )
                .offset(
                    (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize,
                )
                .offset(
                    (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(zo) as isize,
                )) as libc::c_double
            - 2.0f64
                * *pt
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize,
                    )
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize,
                    )
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(zo) as isize,
                    ) as libc::c_double;
        Dyy = (*pt
            .offset((0 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize)
            .offset((1 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize)
            .offset((0 as libc::c_int as libc::c_ulonglong).wrapping_mul(zo) as isize)
            + *pt
                .offset(
                    (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize,
                )
                .offset(
                    (-(1 as libc::c_int) as libc::c_ulonglong).wrapping_mul(yo) as isize,
                )
                .offset(
                    (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(zo) as isize,
                )) as libc::c_double
            - 2.0f64
                * *pt
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize,
                    )
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize,
                    )
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(zo) as isize,
                    ) as libc::c_double;
        Dzz = (*pt
            .offset((0 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize)
            .offset((0 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize)
            .offset((1 as libc::c_int as libc::c_ulonglong).wrapping_mul(zo) as isize)
            + *pt
                .offset(
                    (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize,
                )
                .offset(
                    (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize,
                )
                .offset(
                    (-(1 as libc::c_int) as libc::c_ulonglong).wrapping_mul(zo) as isize,
                )) as libc::c_double
            - 2.0f64
                * *pt
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize,
                    )
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize,
                    )
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(zo) as isize,
                    ) as libc::c_double;
        Dxy = 0.25f64
            * (*pt
                .offset(
                    (1 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize,
                )
                .offset(
                    (1 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize,
                )
                .offset(
                    (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(zo) as isize,
                )
                + *pt
                    .offset(
                        (-(1 as libc::c_int) as libc::c_ulonglong).wrapping_mul(xo)
                            as isize,
                    )
                    .offset(
                        (-(1 as libc::c_int) as libc::c_ulonglong).wrapping_mul(yo)
                            as isize,
                    )
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(zo) as isize,
                    )
                - *pt
                    .offset(
                        (-(1 as libc::c_int) as libc::c_ulonglong).wrapping_mul(xo)
                            as isize,
                    )
                    .offset(
                        (1 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize,
                    )
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(zo) as isize,
                    )
                - *pt
                    .offset(
                        (1 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize,
                    )
                    .offset(
                        (-(1 as libc::c_int) as libc::c_ulonglong).wrapping_mul(yo)
                            as isize,
                    )
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(zo) as isize,
                    )) as libc::c_double;
        Dxz = 0.25f64
            * (*pt
                .offset(
                    (1 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize,
                )
                .offset(
                    (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize,
                )
                .offset(
                    (1 as libc::c_int as libc::c_ulonglong).wrapping_mul(zo) as isize,
                )
                + *pt
                    .offset(
                        (-(1 as libc::c_int) as libc::c_ulonglong).wrapping_mul(xo)
                            as isize,
                    )
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize,
                    )
                    .offset(
                        (-(1 as libc::c_int) as libc::c_ulonglong).wrapping_mul(zo)
                            as isize,
                    )
                - *pt
                    .offset(
                        (-(1 as libc::c_int) as libc::c_ulonglong).wrapping_mul(xo)
                            as isize,
                    )
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize,
                    )
                    .offset(
                        (1 as libc::c_int as libc::c_ulonglong).wrapping_mul(zo) as isize,
                    )
                - *pt
                    .offset(
                        (1 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize,
                    )
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize,
                    )
                    .offset(
                        (-(1 as libc::c_int) as libc::c_ulonglong).wrapping_mul(zo)
                            as isize,
                    )) as libc::c_double;
        Dyz = 0.25f64
            * (*pt
                .offset(
                    (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize,
                )
                .offset(
                    (1 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize,
                )
                .offset(
                    (1 as libc::c_int as libc::c_ulonglong).wrapping_mul(zo) as isize,
                )
                + *pt
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize,
                    )
                    .offset(
                        (-(1 as libc::c_int) as libc::c_ulonglong).wrapping_mul(yo)
                            as isize,
                    )
                    .offset(
                        (-(1 as libc::c_int) as libc::c_ulonglong).wrapping_mul(zo)
                            as isize,
                    )
                - *pt
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize,
                    )
                    .offset(
                        (-(1 as libc::c_int) as libc::c_ulonglong).wrapping_mul(yo)
                            as isize,
                    )
                    .offset(
                        (1 as libc::c_int as libc::c_ulonglong).wrapping_mul(zo) as isize,
                    )
                - *pt
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize,
                    )
                    .offset(
                        (1 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize,
                    )
                    .offset(
                        (-(1 as libc::c_int) as libc::c_ulonglong).wrapping_mul(zo)
                            as isize,
                    )) as libc::c_double;
        A[(0 as libc::c_int + 0 as libc::c_int * 3 as libc::c_int) as usize] = Dxx;
        A[(1 as libc::c_int + 1 as libc::c_int * 3 as libc::c_int) as usize] = Dyy;
        A[(2 as libc::c_int + 2 as libc::c_int * 3 as libc::c_int) as usize] = Dzz;
        A[(1 as libc::c_int + 0 as libc::c_int * 3 as libc::c_int) as usize] = Dxy;
        A[(0 as libc::c_int + 1 as libc::c_int * 3 as libc::c_int)
            as usize] = A[(1 as libc::c_int + 0 as libc::c_int * 3 as libc::c_int)
            as usize];
        A[(2 as libc::c_int + 0 as libc::c_int * 3 as libc::c_int) as usize] = Dxz;
        A[(0 as libc::c_int + 2 as libc::c_int * 3 as libc::c_int)
            as usize] = A[(2 as libc::c_int + 0 as libc::c_int * 3 as libc::c_int)
            as usize];
        A[(2 as libc::c_int + 1 as libc::c_int * 3 as libc::c_int) as usize] = Dyz;
        A[(1 as libc::c_int + 2 as libc::c_int * 3 as libc::c_int)
            as usize] = A[(2 as libc::c_int + 1 as libc::c_int * 3 as libc::c_int)
            as usize];
        b[0 as libc::c_int as usize] = -Dx;
        b[1 as libc::c_int as usize] = -Dy;
        b[2 as libc::c_int as usize] = -Dz;
        err = vl_solve_linear_system_3(b.as_mut_ptr(), A.as_mut_ptr(), b.as_mut_ptr());
        if err != 0 as libc::c_int {
            b[0 as libc::c_int as usize] = 0 as libc::c_int as libc::c_double;
            b[1 as libc::c_int as usize] = 0 as libc::c_int as libc::c_double;
            b[2 as libc::c_int as usize] = 0 as libc::c_int as libc::c_double;
            break;
        } else {
            dx = ((if b[0 as libc::c_int as usize] > 0.6f64
                && x < (width as libc::c_int - 2 as libc::c_int) as libc::c_longlong
            {
                1 as libc::c_int
            } else {
                0 as libc::c_int
            })
                + (if b[0 as libc::c_int as usize] < -0.6f64
                    && x > 1 as libc::c_int as libc::c_longlong
                {
                    -(1 as libc::c_int)
                } else {
                    0 as libc::c_int
                })) as vl_index;
            dy = ((if b[1 as libc::c_int as usize] > 0.6f64
                && y < (height as libc::c_int - 2 as libc::c_int) as libc::c_longlong
            {
                1 as libc::c_int
            } else {
                0 as libc::c_int
            })
                + (if b[1 as libc::c_int as usize] < -0.6f64
                    && y > 1 as libc::c_int as libc::c_longlong
                {
                    -(1 as libc::c_int)
                } else {
                    0 as libc::c_int
                })) as vl_index;
            if dx == 0 as libc::c_int as libc::c_longlong
                && dy == 0 as libc::c_int as libc::c_longlong
            {
                break;
            }
            iter += 1;
        }
    }
    let mut peakScore: libc::c_double = *pt
        .offset((0 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize)
        .offset((0 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize)
        .offset((0 as libc::c_int as libc::c_ulonglong).wrapping_mul(zo) as isize)
        as libc::c_double
        + 0.5f64
            * (Dx * b[0 as libc::c_int as usize] + Dy * b[1 as libc::c_int as usize]
                + Dz * b[2 as libc::c_int as usize]);
    let mut alpha: libc::c_double = (Dxx + Dyy) * (Dxx + Dyy) / (Dxx * Dyy - Dxy * Dxy);
    let mut edgeScore: libc::c_double = 0.;
    if alpha < 0 as libc::c_int as libc::c_double {
        edgeScore = vl_infinity_d.value;
    } else {
        edgeScore = 0.5f64 * alpha - 1 as libc::c_int as libc::c_double
            + sqrt(
                (if 0.25f64 * alpha - 1 as libc::c_int as libc::c_double
                    > 0 as libc::c_int as libc::c_double
                {
                    0.25f64 * alpha - 1 as libc::c_int as libc::c_double
                } else {
                    0 as libc::c_int as libc::c_double
                }) * alpha,
            );
    }
    (*refined).xi = x;
    (*refined).yi = y;
    (*refined).zi = z;
    (*refined).x = (x as libc::c_double + b[0 as libc::c_int as usize]) as libc::c_float;
    (*refined).y = (y as libc::c_double + b[1 as libc::c_int as usize]) as libc::c_float;
    (*refined).z = (z as libc::c_double + b[2 as libc::c_int as usize]) as libc::c_float;
    (*refined).peakScore = peakScore as libc::c_float;
    (*refined).edgeScore = edgeScore as libc::c_float;
    return (err == 0 as libc::c_int && vl_abs_d(b[0 as libc::c_int as usize]) < 1.5f64
        && vl_abs_d(b[1 as libc::c_int as usize]) < 1.5f64
        && vl_abs_d(b[2 as libc::c_int as usize]) < 1.5f64
        && 0 as libc::c_int as libc::c_float <= (*refined).x
        && (*refined).x <= (width as libc::c_int - 1 as libc::c_int) as libc::c_float
        && 0 as libc::c_int as libc::c_float <= (*refined).y
        && (*refined).y <= (height as libc::c_int - 1 as libc::c_int) as libc::c_float
        && 0 as libc::c_int as libc::c_float <= (*refined).z
        && (*refined).z <= (depth as libc::c_int - 1 as libc::c_int) as libc::c_float)
        as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn vl_refine_local_extreum_2(
    mut refined: *mut VlCovDetExtremum2,
    mut map: *const libc::c_float,
    mut width: vl_size,
    mut height: vl_size,
    mut x: vl_index,
    mut y: vl_index,
) -> vl_bool {
    let xo: vl_size = 1 as libc::c_int as vl_size;
    let yo: vl_size = width;
    let mut Dx: libc::c_double = 0 as libc::c_int as libc::c_double;
    let mut Dy: libc::c_double = 0 as libc::c_int as libc::c_double;
    let mut Dxx: libc::c_double = 0 as libc::c_int as libc::c_double;
    let mut Dyy: libc::c_double = 0 as libc::c_int as libc::c_double;
    let mut Dxy: libc::c_double = 0 as libc::c_int as libc::c_double;
    let mut A: [libc::c_double; 4] = [0.; 4];
    let mut b: [libc::c_double; 2] = [0.; 2];
    let mut pt: *const libc::c_float = 0 as *const libc::c_float;
    let mut dx: vl_index = 0 as libc::c_int as vl_index;
    let mut dy: vl_index = 0 as libc::c_int as vl_index;
    let mut iter: vl_index = 0;
    let mut err: libc::c_int = 0;
    if !map.is_null() {} else {
        __assert_fail(
            b"map\0" as *const u8 as *const libc::c_char,
            b"vl/covdet.c\0" as *const u8 as *const libc::c_char,
            1352 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 108],
                &[libc::c_char; 108],
            >(
                b"vl_bool vl_refine_local_extreum_2(VlCovDetExtremum2 *, const float *, vl_size, vl_size, vl_index, vl_index)\0",
            ))
                .as_ptr(),
        );
    }
    if 1 as libc::c_int as libc::c_longlong <= x
        && x <= (width as libc::c_int - 2 as libc::c_int) as libc::c_longlong
    {} else {
        __assert_fail(
            b"1 <= x && x <= (signed)width - 2\0" as *const u8 as *const libc::c_char,
            b"vl/covdet.c\0" as *const u8 as *const libc::c_char,
            1353 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 108],
                &[libc::c_char; 108],
            >(
                b"vl_bool vl_refine_local_extreum_2(VlCovDetExtremum2 *, const float *, vl_size, vl_size, vl_index, vl_index)\0",
            ))
                .as_ptr(),
        );
    }
    if 1 as libc::c_int as libc::c_longlong <= y
        && y <= (height as libc::c_int - 2 as libc::c_int) as libc::c_longlong
    {} else {
        __assert_fail(
            b"1 <= y && y <= (signed)height - 2\0" as *const u8 as *const libc::c_char,
            b"vl/covdet.c\0" as *const u8 as *const libc::c_char,
            1354 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 108],
                &[libc::c_char; 108],
            >(
                b"vl_bool vl_refine_local_extreum_2(VlCovDetExtremum2 *, const float *, vl_size, vl_size, vl_index, vl_index)\0",
            ))
                .as_ptr(),
        );
    }
    iter = 0 as libc::c_int as vl_index;
    while iter < 5 as libc::c_int as libc::c_longlong {
        x += dx;
        y += dy;
        pt = map
            .offset((x as libc::c_ulonglong).wrapping_mul(xo) as isize)
            .offset((y as libc::c_ulonglong).wrapping_mul(yo) as isize);
        Dx = 0.5f64
            * (*pt
                .offset(
                    (1 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize,
                )
                .offset(
                    (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize,
                )
                - *pt
                    .offset(
                        (-(1 as libc::c_int) as libc::c_ulonglong).wrapping_mul(xo)
                            as isize,
                    )
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize,
                    )) as libc::c_double;
        Dy = 0.5f64
            * (*pt
                .offset(
                    (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize,
                )
                .offset(
                    (1 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize,
                )
                - *pt
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize,
                    )
                    .offset(
                        (-(1 as libc::c_int) as libc::c_ulonglong).wrapping_mul(yo)
                            as isize,
                    )) as libc::c_double;
        Dxx = (*pt
            .offset((1 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize)
            .offset((0 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize)
            + *pt
                .offset(
                    (-(1 as libc::c_int) as libc::c_ulonglong).wrapping_mul(xo) as isize,
                )
                .offset(
                    (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize,
                )) as libc::c_double
            - 2.0f64
                * *pt
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize,
                    )
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize,
                    ) as libc::c_double;
        Dyy = (*pt
            .offset((0 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize)
            .offset((1 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize)
            + *pt
                .offset(
                    (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize,
                )
                .offset(
                    (-(1 as libc::c_int) as libc::c_ulonglong).wrapping_mul(yo) as isize,
                )) as libc::c_double
            - 2.0f64
                * *pt
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize,
                    )
                    .offset(
                        (0 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize,
                    ) as libc::c_double;
        Dxy = 0.25f64
            * (*pt
                .offset(
                    (1 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize,
                )
                .offset(
                    (1 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize,
                )
                + *pt
                    .offset(
                        (-(1 as libc::c_int) as libc::c_ulonglong).wrapping_mul(xo)
                            as isize,
                    )
                    .offset(
                        (-(1 as libc::c_int) as libc::c_ulonglong).wrapping_mul(yo)
                            as isize,
                    )
                - *pt
                    .offset(
                        (-(1 as libc::c_int) as libc::c_ulonglong).wrapping_mul(xo)
                            as isize,
                    )
                    .offset(
                        (1 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize,
                    )
                - *pt
                    .offset(
                        (1 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize,
                    )
                    .offset(
                        (-(1 as libc::c_int) as libc::c_ulonglong).wrapping_mul(yo)
                            as isize,
                    )) as libc::c_double;
        A[(0 as libc::c_int + 0 as libc::c_int * 2 as libc::c_int) as usize] = Dxx;
        A[(1 as libc::c_int + 1 as libc::c_int * 2 as libc::c_int) as usize] = Dyy;
        A[(1 as libc::c_int + 0 as libc::c_int * 2 as libc::c_int) as usize] = Dxy;
        A[(0 as libc::c_int + 1 as libc::c_int * 2 as libc::c_int)
            as usize] = A[(1 as libc::c_int + 0 as libc::c_int * 2 as libc::c_int)
            as usize];
        b[0 as libc::c_int as usize] = -Dx;
        b[1 as libc::c_int as usize] = -Dy;
        err = vl_solve_linear_system_2(b.as_mut_ptr(), A.as_mut_ptr(), b.as_mut_ptr());
        if err != 0 as libc::c_int {
            b[0 as libc::c_int as usize] = 0 as libc::c_int as libc::c_double;
            b[1 as libc::c_int as usize] = 0 as libc::c_int as libc::c_double;
            break;
        } else {
            dx = ((if b[0 as libc::c_int as usize] > 0.6f64
                && x < (width as libc::c_int - 2 as libc::c_int) as libc::c_longlong
            {
                1 as libc::c_int
            } else {
                0 as libc::c_int
            })
                + (if b[0 as libc::c_int as usize] < -0.6f64
                    && x > 1 as libc::c_int as libc::c_longlong
                {
                    -(1 as libc::c_int)
                } else {
                    0 as libc::c_int
                })) as vl_index;
            dy = ((if b[1 as libc::c_int as usize] > 0.6f64
                && y < (height as libc::c_int - 2 as libc::c_int) as libc::c_longlong
            {
                1 as libc::c_int
            } else {
                0 as libc::c_int
            })
                + (if b[1 as libc::c_int as usize] < -0.6f64
                    && y > 1 as libc::c_int as libc::c_longlong
                {
                    -(1 as libc::c_int)
                } else {
                    0 as libc::c_int
                })) as vl_index;
            if dx == 0 as libc::c_int as libc::c_longlong
                && dy == 0 as libc::c_int as libc::c_longlong
            {
                break;
            }
            iter += 1;
        }
    }
    let mut peakScore: libc::c_double = *pt
        .offset((0 as libc::c_int as libc::c_ulonglong).wrapping_mul(xo) as isize)
        .offset((0 as libc::c_int as libc::c_ulonglong).wrapping_mul(yo) as isize)
        as libc::c_double
        + 0.5f64
            * (Dx * b[0 as libc::c_int as usize] + Dy * b[1 as libc::c_int as usize]);
    let mut alpha: libc::c_double = (Dxx + Dyy) * (Dxx + Dyy) / (Dxx * Dyy - Dxy * Dxy);
    let mut edgeScore: libc::c_double = 0.;
    if alpha < 0 as libc::c_int as libc::c_double {
        edgeScore = vl_infinity_d.value;
    } else {
        edgeScore = 0.5f64 * alpha - 1 as libc::c_int as libc::c_double
            + sqrt(
                (if 0.25f64 * alpha - 1 as libc::c_int as libc::c_double
                    > 0 as libc::c_int as libc::c_double
                {
                    0.25f64 * alpha - 1 as libc::c_int as libc::c_double
                } else {
                    0 as libc::c_int as libc::c_double
                }) * alpha,
            );
    }
    (*refined).xi = x;
    (*refined).yi = y;
    (*refined).x = (x as libc::c_double + b[0 as libc::c_int as usize]) as libc::c_float;
    (*refined).y = (y as libc::c_double + b[1 as libc::c_int as usize]) as libc::c_float;
    (*refined).peakScore = peakScore as libc::c_float;
    (*refined).edgeScore = edgeScore as libc::c_float;
    return (err == 0 as libc::c_int && vl_abs_d(b[0 as libc::c_int as usize]) < 1.5f64
        && vl_abs_d(b[1 as libc::c_int as usize]) < 1.5f64
        && 0 as libc::c_int as libc::c_float <= (*refined).x
        && (*refined).x <= (width as libc::c_int - 1 as libc::c_int) as libc::c_float
        && 0 as libc::c_int as libc::c_float <= (*refined).y
        && (*refined).y <= (height as libc::c_int - 1 as libc::c_int) as libc::c_float)
        as libc::c_int;
}
#[no_mangle]
pub static mut vlCovdetMethods: [VlEnumerator; 7] = [
    {
        let mut init = _VlEnumerator {
            name: b"DoG\0" as *const u8 as *const libc::c_char,
            value: VL_COVDET_METHOD_DOG as libc::c_int as vl_index,
        };
        init
    },
    {
        let mut init = _VlEnumerator {
            name: b"Hessian\0" as *const u8 as *const libc::c_char,
            value: VL_COVDET_METHOD_HESSIAN as libc::c_int as vl_index,
        };
        init
    },
    {
        let mut init = _VlEnumerator {
            name: b"HessianLaplace\0" as *const u8 as *const libc::c_char,
            value: VL_COVDET_METHOD_HESSIAN_LAPLACE as libc::c_int as vl_index,
        };
        init
    },
    {
        let mut init = _VlEnumerator {
            name: b"HarrisLaplace\0" as *const u8 as *const libc::c_char,
            value: VL_COVDET_METHOD_HARRIS_LAPLACE as libc::c_int as vl_index,
        };
        init
    },
    {
        let mut init = _VlEnumerator {
            name: b"MultiscaleHessian\0" as *const u8 as *const libc::c_char,
            value: VL_COVDET_METHOD_MULTISCALE_HESSIAN as libc::c_int as vl_index,
        };
        init
    },
    {
        let mut init = _VlEnumerator {
            name: b"MultiscaleHarris\0" as *const u8 as *const libc::c_char,
            value: VL_COVDET_METHOD_MULTISCALE_HARRIS as libc::c_int as vl_index,
        };
        init
    },
    {
        let mut init = _VlEnumerator {
            name: 0 as *const libc::c_char,
            value: 0 as libc::c_int as vl_index,
        };
        init
    },
];
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_new(mut method: VlCovDetMethod) -> *mut VlCovDet {
    let mut self_0: *mut VlCovDet = vl_calloc(
        ::core::mem::size_of::<VlCovDet>() as libc::c_ulong,
        1 as libc::c_int as size_t,
    ) as *mut VlCovDet;
    (*self_0).method = method;
    (*self_0).octaveResolution = 3 as libc::c_int as vl_size;
    (*self_0).numOctaves = -(1 as libc::c_int) as vl_index;
    (*self_0).firstOctave = -(1 as libc::c_int) as vl_index;
    (*self_0).baseScale = 1.6f64;
    (*self_0).maxNumOrientations = 4 as libc::c_int as vl_size;
    match (*self_0).method as libc::c_uint {
        1 => {
            (*self_0).peakThreshold = 0.01f64;
            (*self_0).edgeThreshold = 10.0f64;
            (*self_0).lapPeakThreshold = 0 as libc::c_int as libc::c_double;
        }
        4 | 6 => {
            (*self_0).peakThreshold = 0.000002f64;
            (*self_0).edgeThreshold = 10.0f64;
            (*self_0).lapPeakThreshold = 0.01f64;
        }
        2 | 3 | 5 => {
            (*self_0).peakThreshold = 0.003f64;
            (*self_0).edgeThreshold = 10.0f64;
            (*self_0).lapPeakThreshold = 0.01f64;
        }
        _ => {
            __assert_fail(
                b"0\0" as *const u8 as *const libc::c_char,
                b"vl/covdet.c\0" as *const u8 as *const libc::c_char,
                1542 as libc::c_int as libc::c_uint,
                (*::core::mem::transmute::<
                    &[u8; 40],
                    &[libc::c_char; 40],
                >(b"VlCovDet *vl_covdet_new(VlCovDetMethod)\0"))
                    .as_ptr(),
            );
        }
    }
    (*self_0).nonExtremaSuppression = 0.5f64;
    (*self_0).features = 0 as *mut VlCovDetFeature;
    (*self_0).numFeatures = 0 as libc::c_int as vl_size;
    (*self_0).numFeatureBufferSize = 0 as libc::c_int as vl_size;
    (*self_0).patch = 0 as *mut libc::c_float;
    (*self_0).patchBufferSize = 0 as libc::c_int as vl_size;
    (*self_0).transposed = 0 as libc::c_int;
    (*self_0).aaAccurateSmoothing = 0 as libc::c_int;
    (*self_0).allowPaddedWarping = 1 as libc::c_int;
    let w: vl_index = 20 as libc::c_int as vl_index;
    let mut i: vl_index = 0;
    let mut j: vl_index = 0;
    let mut step: libc::c_double = 2.0f64
        * (3 as libc::c_int * 3 as libc::c_int) as libc::c_double
        / (2 as libc::c_int as libc::c_longlong * w
            + 1 as libc::c_int as libc::c_longlong) as libc::c_double;
    let mut sigma: libc::c_double = 3 as libc::c_int as libc::c_double;
    j = -w;
    while j <= w {
        i = -w;
        while i <= w {
            let mut dx: libc::c_double = i as libc::c_double * step / sigma;
            let mut dy: libc::c_double = j as libc::c_double * step / sigma;
            (*self_0)
                .aaMask[(i + w
                + (2 as libc::c_int as libc::c_longlong * w
                    + 1 as libc::c_int as libc::c_longlong) * (j + w))
                as usize] = exp(-0.5f64 * (dx * dx + dy * dy)) as libc::c_float;
            i += 1;
        }
        j += 1;
    }
    let mut s: vl_index = 0;
    s = 0 as libc::c_int as vl_index;
    while s < 10 as libc::c_int as libc::c_longlong {
        let mut sigmaLap: libc::c_double = pow(
            2.0f64,
            -0.5f64
                + s as libc::c_double
                    / (10 as libc::c_int - 1 as libc::c_int) as libc::c_double,
        );
        let sigmaImage: libc::c_double = 1.0f64 / sqrt(2.0f64);
        let step_0: libc::c_double = 0.5f64 * sigmaImage;
        let sigmaDelta: libc::c_double = sqrt(
            sigmaLap * sigmaLap - sigmaImage * sigmaImage,
        );
        let w_0: vl_size = 16 as libc::c_int as vl_size;
        let num: vl_size = (2 as libc::c_int as libc::c_ulonglong)
            .wrapping_mul(w_0)
            .wrapping_add(1 as libc::c_int as libc::c_ulonglong);
        let mut pt: *mut libc::c_float = ((*self_0).laplacians)
            .as_mut_ptr()
            .offset(
                (s as libc::c_ulonglong).wrapping_mul(num.wrapping_mul(num)) as isize,
            );
        memset(
            pt as *mut libc::c_void,
            0 as libc::c_int,
            num
                .wrapping_mul(num)
                .wrapping_mul(
                    ::core::mem::size_of::<libc::c_float>() as libc::c_ulong
                        as libc::c_ulonglong,
                ) as libc::c_ulong,
        );
        *pt
            .offset(
                (0 as libc::c_int as libc::c_ulonglong)
                    .wrapping_add(w_0)
                    .wrapping_add(
                        (0 as libc::c_int as libc::c_ulonglong)
                            .wrapping_add(w_0)
                            .wrapping_mul(
                                (2 as libc::c_int as libc::c_ulonglong)
                                    .wrapping_mul(w_0)
                                    .wrapping_add(1 as libc::c_int as libc::c_ulonglong),
                            ),
                    ) as isize,
            ) = -4.0f64 as libc::c_float;
        *pt
            .offset(
                (-(1 as libc::c_int) as libc::c_ulonglong)
                    .wrapping_add(w_0)
                    .wrapping_add(
                        (0 as libc::c_int as libc::c_ulonglong)
                            .wrapping_add(w_0)
                            .wrapping_mul(
                                (2 as libc::c_int as libc::c_ulonglong)
                                    .wrapping_mul(w_0)
                                    .wrapping_add(1 as libc::c_int as libc::c_ulonglong),
                            ),
                    ) as isize,
            ) = 1.0f64 as libc::c_float;
        *pt
            .offset(
                (1 as libc::c_int as libc::c_ulonglong)
                    .wrapping_add(w_0)
                    .wrapping_add(
                        (0 as libc::c_int as libc::c_ulonglong)
                            .wrapping_add(w_0)
                            .wrapping_mul(
                                (2 as libc::c_int as libc::c_ulonglong)
                                    .wrapping_mul(w_0)
                                    .wrapping_add(1 as libc::c_int as libc::c_ulonglong),
                            ),
                    ) as isize,
            ) = 1.0f64 as libc::c_float;
        *pt
            .offset(
                (0 as libc::c_int as libc::c_ulonglong)
                    .wrapping_add(w_0)
                    .wrapping_add(
                        (1 as libc::c_int as libc::c_ulonglong)
                            .wrapping_add(w_0)
                            .wrapping_mul(
                                (2 as libc::c_int as libc::c_ulonglong)
                                    .wrapping_mul(w_0)
                                    .wrapping_add(1 as libc::c_int as libc::c_ulonglong),
                            ),
                    ) as isize,
            ) = 1.0f64 as libc::c_float;
        *pt
            .offset(
                (0 as libc::c_int as libc::c_ulonglong)
                    .wrapping_add(w_0)
                    .wrapping_add(
                        (-(1 as libc::c_int) as libc::c_ulonglong)
                            .wrapping_add(w_0)
                            .wrapping_mul(
                                (2 as libc::c_int as libc::c_ulonglong)
                                    .wrapping_mul(w_0)
                                    .wrapping_add(1 as libc::c_int as libc::c_ulonglong),
                            ),
                    ) as isize,
            ) = 1.0f64 as libc::c_float;
        vl_imsmooth_f(
            pt,
            num,
            pt,
            num,
            num,
            num,
            sigmaDelta / step_0,
            sigmaDelta / step_0,
        );
        s += 1;
    }
    return self_0;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_reset(mut self_0: *mut VlCovDet) {
    if !((*self_0).features).is_null() {
        vl_free((*self_0).features as *mut libc::c_void);
        (*self_0).features = 0 as *mut VlCovDetFeature;
    }
    if !((*self_0).css).is_null() {
        vl_scalespace_delete((*self_0).css);
        (*self_0).css = 0 as *mut VlScaleSpace;
    }
    if !((*self_0).gss).is_null() {
        vl_scalespace_delete((*self_0).gss);
        (*self_0).gss = 0 as *mut VlScaleSpace;
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_delete(mut self_0: *mut VlCovDet) {
    vl_covdet_reset(self_0);
    if !((*self_0).patch).is_null() {
        vl_free((*self_0).patch as *mut libc::c_void);
    }
    vl_free(self_0 as *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_append_feature(
    mut self_0: *mut VlCovDet,
    mut feature: *const VlCovDetFeature,
) -> libc::c_int {
    let mut requiredSize: vl_size = 0;
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/covdet.c\0" as *const u8 as *const libc::c_char,
            1661 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 66],
                &[libc::c_char; 66],
            >(b"int vl_covdet_append_feature(VlCovDet *, const VlCovDetFeature *)\0"))
                .as_ptr(),
        );
    }
    if !feature.is_null() {} else {
        __assert_fail(
            b"feature\0" as *const u8 as *const libc::c_char,
            b"vl/covdet.c\0" as *const u8 as *const libc::c_char,
            1662 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 66],
                &[libc::c_char; 66],
            >(b"int vl_covdet_append_feature(VlCovDet *, const VlCovDetFeature *)\0"))
                .as_ptr(),
        );
    }
    (*self_0).numFeatures = ((*self_0).numFeatures).wrapping_add(1);
    requiredSize = ((*self_0).numFeatures)
        .wrapping_mul(
            ::core::mem::size_of::<VlCovDetFeature>() as libc::c_ulong
                as libc::c_ulonglong,
        );
    if requiredSize > (*self_0).numFeatureBufferSize {
        let mut err: libc::c_int = _vl_resize_buffer(
            &mut (*self_0).features as *mut *mut VlCovDetFeature
                as *mut *mut libc::c_void,
            &mut (*self_0).numFeatureBufferSize,
            ((*self_0).numFeatures)
                .wrapping_add(1000 as libc::c_int as libc::c_ulonglong)
                .wrapping_mul(
                    ::core::mem::size_of::<VlCovDetFeature>() as libc::c_ulong
                        as libc::c_ulonglong,
                ),
        );
        if err != 0 {
            (*self_0).numFeatures = ((*self_0).numFeatures).wrapping_sub(1);
            return err;
        }
    }
    *((*self_0).features)
        .offset(
            ((*self_0).numFeatures).wrapping_sub(1 as libc::c_int as libc::c_ulonglong)
                as isize,
        ) = *feature;
    return 0 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_put_image(
    mut self_0: *mut VlCovDet,
    mut image: *const libc::c_float,
    mut width: vl_size,
    mut height: vl_size,
) -> libc::c_int {
    let minOctaveSize: vl_size = 16 as libc::c_int as vl_size;
    let mut lastOctave: vl_index = 0;
    let mut octaveFirstSubdivision: vl_index = 0;
    let mut octaveLastSubdivision: vl_index = 0;
    let mut geom: VlScaleSpaceGeometry = vl_scalespace_get_default_geometry(
        width,
        height,
    );
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/covdet.c\0" as *const u8 as *const libc::c_char,
            1703 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 69],
                &[libc::c_char; 69],
            >(b"int vl_covdet_put_image(VlCovDet *, const float *, vl_size, vl_size)\0"))
                .as_ptr(),
        );
    }
    if !image.is_null() {} else {
        __assert_fail(
            b"image\0" as *const u8 as *const libc::c_char,
            b"vl/covdet.c\0" as *const u8 as *const libc::c_char,
            1704 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 69],
                &[libc::c_char; 69],
            >(b"int vl_covdet_put_image(VlCovDet *, const float *, vl_size, vl_size)\0"))
                .as_ptr(),
        );
    }
    if width >= 1 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"width >= 1\0" as *const u8 as *const libc::c_char,
            b"vl/covdet.c\0" as *const u8 as *const libc::c_char,
            1705 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 69],
                &[libc::c_char; 69],
            >(b"int vl_covdet_put_image(VlCovDet *, const float *, vl_size, vl_size)\0"))
                .as_ptr(),
        );
    }
    if height >= 1 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"height >= 1\0" as *const u8 as *const libc::c_char,
            b"vl/covdet.c\0" as *const u8 as *const libc::c_char,
            1706 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 69],
                &[libc::c_char; 69],
            >(b"int vl_covdet_put_image(VlCovDet *, const float *, vl_size, vl_size)\0"))
                .as_ptr(),
        );
    }
    lastOctave = vl_floor_d(
        f64::log2(
            (if (width as libc::c_double - 1 as libc::c_int as libc::c_double)
                < height as libc::c_double - 1 as libc::c_int as libc::c_double
            {
                width as libc::c_double - 1 as libc::c_int as libc::c_double
            } else {
                height as libc::c_double - 1 as libc::c_int as libc::c_double
            })
                / minOctaveSize.wrapping_sub(1 as libc::c_int as libc::c_ulonglong)
                    as libc::c_double,
        ),
    ) as vl_index;
    if (*self_0).numOctaves > 0 as libc::c_int as libc::c_longlong {
        lastOctave = if ((*self_0).numOctaves - (*self_0).firstOctave
            - 1 as libc::c_int as libc::c_longlong) < lastOctave
        {
            (*self_0).numOctaves - (*self_0).firstOctave
                - 1 as libc::c_int as libc::c_longlong
        } else {
            lastOctave
        };
    }
    if (*self_0).method as libc::c_uint
        == VL_COVDET_METHOD_DOG as libc::c_int as libc::c_uint
    {
        octaveFirstSubdivision = -(1 as libc::c_int) as vl_index;
        octaveLastSubdivision = ((*self_0).octaveResolution)
            .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as vl_index;
    } else if (*self_0).method as libc::c_uint
        == VL_COVDET_METHOD_HESSIAN as libc::c_int as libc::c_uint
    {
        octaveFirstSubdivision = -(1 as libc::c_int) as vl_index;
        octaveLastSubdivision = (*self_0).octaveResolution as vl_index;
    } else {
        octaveFirstSubdivision = 0 as libc::c_int as vl_index;
        octaveLastSubdivision = ((*self_0).octaveResolution)
            .wrapping_sub(1 as libc::c_int as libc::c_ulonglong) as vl_index;
    }
    geom.width = width;
    geom.height = height;
    geom.firstOctave = (*self_0).firstOctave;
    geom.lastOctave = lastOctave;
    geom.octaveResolution = (*self_0).octaveResolution;
    geom
        .baseScale = (*self_0).baseScale
        * pow(2.0f64, 1.0f64 / (*self_0).octaveResolution as libc::c_double);
    geom.octaveFirstSubdivision = octaveFirstSubdivision;
    geom.octaveLastSubdivision = octaveLastSubdivision;
    if ((*self_0).gss).is_null()
        || vl_scalespacegeometry_is_equal(
            geom,
            vl_scalespace_get_geometry((*self_0).gss),
        ) == 0
    {
        if !((*self_0).gss).is_null() {
            vl_scalespace_delete((*self_0).gss);
        }
        (*self_0).gss = vl_scalespace_new_with_geometry(geom);
        if ((*self_0).gss).is_null() {
            return 2 as libc::c_int;
        }
    }
    vl_scalespace_put_image((*self_0).gss, image);
    return 0 as libc::c_int;
}
unsafe extern "C" fn _vl_det_hessian_response(
    mut hessian: *mut libc::c_float,
    mut image: *const libc::c_float,
    mut width: vl_size,
    mut height: vl_size,
    mut step: libc::c_double,
    mut sigma: libc::c_double,
) {
    let mut factor: libc::c_float = pow(sigma / step, 4.0f64) as libc::c_float;
    let xo: vl_index = 1 as libc::c_int as vl_index;
    let yo: vl_index = width as vl_index;
    let mut r: vl_size = 0;
    let mut c: vl_size = 0;
    let mut p11: libc::c_float = 0.;
    let mut p12: libc::c_float = 0.;
    let mut p13: libc::c_float = 0.;
    let mut p21: libc::c_float = 0.;
    let mut p22: libc::c_float = 0.;
    let mut p23: libc::c_float = 0.;
    let mut p31: libc::c_float = 0.;
    let mut p32: libc::c_float = 0.;
    let mut p33: libc::c_float = 0.;
    let mut in_0: *const libc::c_float = image.offset(yo as isize);
    let mut out: *mut libc::c_float = hessian.offset(xo as isize).offset(yo as isize);
    r = 1 as libc::c_int as vl_size;
    while r < height.wrapping_sub(1 as libc::c_int as libc::c_ulonglong) {
        p11 = *in_0.offset(-yo as isize);
        p12 = *in_0.offset((xo - yo) as isize);
        p21 = *in_0.offset(0 as libc::c_int as isize);
        p22 = *in_0.offset(xo as isize);
        p31 = *in_0.offset(yo as isize);
        p32 = *in_0.offset((xo + yo) as isize);
        in_0 = in_0.offset(2 as libc::c_int as isize);
        c = 1 as libc::c_int as vl_size;
        while c < width.wrapping_sub(1 as libc::c_int as libc::c_ulonglong) {
            let mut Lxx: libc::c_float = 0.;
            let mut Lyy: libc::c_float = 0.;
            let mut Lxy: libc::c_float = 0.;
            p13 = *in_0.offset(-yo as isize);
            p23 = *in_0;
            p33 = *in_0.offset(yo as isize);
            Lxx = -p21 + 2 as libc::c_int as libc::c_float * p22 - p23;
            Lyy = -p12 + 2 as libc::c_int as libc::c_float * p22 - p32;
            Lxy = (p11 - p31 - p13 + p33) / 4.0f32;
            *out = (Lxx * Lyy - Lxy * Lxy) * factor;
            p11 = p12;
            p12 = p13;
            p21 = p22;
            p22 = p23;
            p31 = p32;
            p32 = p33;
            in_0 = in_0.offset(1);
            out = out.offset(1);
            c = c.wrapping_add(1);
        }
        out = out.offset(2 as libc::c_int as isize);
        r = r.wrapping_add(1);
    }
    in_0 = hessian.offset(yo as isize).offset(xo as isize);
    out = hessian.offset(xo as isize);
    memcpy(
        out as *mut libc::c_void,
        in_0 as *const libc::c_void,
        width
            .wrapping_sub(2 as libc::c_int as libc::c_ulonglong)
            .wrapping_mul(
                ::core::mem::size_of::<libc::c_float>() as libc::c_ulong
                    as libc::c_ulonglong,
            ) as libc::c_ulong,
    );
    out = out.offset(-1);
    in_0 = in_0.offset(-(yo as isize));
    r = 0 as libc::c_int as vl_size;
    while r < height.wrapping_sub(1 as libc::c_int as libc::c_ulonglong) {
        *out = *in_0;
        *out
            .offset(yo as isize)
            .offset(
                -(1 as libc::c_int as isize),
            ) = *in_0.offset(yo as isize).offset(-(3 as libc::c_int as isize));
        in_0 = in_0.offset(yo as isize);
        out = out.offset(yo as isize);
        r = r.wrapping_add(1);
    }
    in_0 = in_0.offset(-(yo as isize));
    *out = *in_0;
    *out
        .offset(yo as isize)
        .offset(
            -(1 as libc::c_int as isize),
        ) = *in_0.offset(yo as isize).offset(-(3 as libc::c_int as isize));
    out = out.offset(1);
    memcpy(
        out as *mut libc::c_void,
        in_0 as *const libc::c_void,
        width
            .wrapping_sub(2 as libc::c_int as libc::c_ulonglong)
            .wrapping_mul(
                ::core::mem::size_of::<libc::c_float>() as libc::c_ulong
                    as libc::c_ulonglong,
            ) as libc::c_ulong,
    );
}
unsafe extern "C" fn _vl_harris_response(
    mut harris: *mut libc::c_float,
    mut image: *const libc::c_float,
    mut width: vl_size,
    mut height: vl_size,
    mut step: libc::c_double,
    mut sigma: libc::c_double,
    mut sigmaI: libc::c_double,
    mut alpha: libc::c_double,
) {
    let mut factor: libc::c_float = pow(sigma / step, 4.0f64) as libc::c_float;
    let mut k: vl_index = 0;
    let mut LxLx: *mut libc::c_float = 0 as *mut libc::c_float;
    let mut LyLy: *mut libc::c_float = 0 as *mut libc::c_float;
    let mut LxLy: *mut libc::c_float = 0 as *mut libc::c_float;
    LxLx = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(width)
            .wrapping_mul(height) as size_t,
    ) as *mut libc::c_float;
    LyLy = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(width)
            .wrapping_mul(height) as size_t,
    ) as *mut libc::c_float;
    LxLy = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(width)
            .wrapping_mul(height) as size_t,
    ) as *mut libc::c_float;
    vl_imgradient_f(
        LxLx,
        LyLy,
        1 as libc::c_int as vl_size,
        width,
        image,
        width,
        height,
        width,
    );
    k = 0 as libc::c_int as vl_index;
    while k < width.wrapping_mul(height) as libc::c_int as libc::c_longlong {
        let mut dx: libc::c_float = *LxLx.offset(k as isize);
        let mut dy: libc::c_float = *LyLy.offset(k as isize);
        *LxLx.offset(k as isize) = dx * dx;
        *LyLy.offset(k as isize) = dy * dy;
        *LxLy.offset(k as isize) = dx * dy;
        k += 1;
    }
    vl_imsmooth_f(LxLx, width, LxLx, width, height, width, sigmaI / step, sigmaI / step);
    vl_imsmooth_f(LyLy, width, LyLy, width, height, width, sigmaI / step, sigmaI / step);
    vl_imsmooth_f(LxLy, width, LxLy, width, height, width, sigmaI / step, sigmaI / step);
    k = 0 as libc::c_int as vl_index;
    while k < width.wrapping_mul(height) as libc::c_int as libc::c_longlong {
        let mut a: libc::c_float = *LxLx.offset(k as isize);
        let mut b: libc::c_float = *LyLy.offset(k as isize);
        let mut c: libc::c_float = *LxLy.offset(k as isize);
        let mut determinant: libc::c_float = a * b - c * c;
        let mut trace: libc::c_float = a + b;
        *harris
            .offset(
                k as isize,
            ) = (factor as libc::c_double
            * (determinant as libc::c_double
                - alpha * (trace * trace) as libc::c_double)) as libc::c_float;
        k += 1;
    }
    vl_free(LxLy as *mut libc::c_void);
    vl_free(LyLy as *mut libc::c_void);
    vl_free(LxLx as *mut libc::c_void);
}
unsafe extern "C" fn _vl_dog_response(
    mut dog: *mut libc::c_float,
    mut level1: *const libc::c_float,
    mut level2: *const libc::c_float,
    mut width: vl_size,
    mut height: vl_size,
) {
    let mut k: vl_index = 0;
    k = 0 as libc::c_int as vl_index;
    while k < width.wrapping_mul(height) as libc::c_int as libc::c_longlong {
        *dog
            .offset(
                k as isize,
            ) = *level2.offset(k as isize) - *level1.offset(k as isize);
        k += 1;
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_detect(mut self_0: *mut VlCovDet) {
    let mut geom: VlScaleSpaceGeometry = vl_scalespace_get_geometry((*self_0).gss);
    let mut cgeom: VlScaleSpaceGeometry = VlScaleSpaceGeometry {
        width: 0,
        height: 0,
        firstOctave: 0,
        lastOctave: 0,
        octaveResolution: 0,
        octaveFirstSubdivision: 0,
        octaveLastSubdivision: 0,
        baseScale: 0.,
        nominalScale: 0.,
    };
    let mut levelxx: *mut libc::c_float = 0 as *mut libc::c_float;
    let mut levelyy: *mut libc::c_float = 0 as *mut libc::c_float;
    let mut levelxy: *mut libc::c_float = 0 as *mut libc::c_float;
    let mut o: vl_index = 0;
    let mut s: vl_index = 0;
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/covdet.c\0" as *const u8 as *const libc::c_char,
            1944 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 34],
                &[libc::c_char; 34],
            >(b"void vl_covdet_detect(VlCovDet *)\0"))
                .as_ptr(),
        );
    }
    if !((*self_0).gss).is_null() {} else {
        __assert_fail(
            b"self->gss\0" as *const u8 as *const libc::c_char,
            b"vl/covdet.c\0" as *const u8 as *const libc::c_char,
            1945 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 34],
                &[libc::c_char; 34],
            >(b"void vl_covdet_detect(VlCovDet *)\0"))
                .as_ptr(),
        );
    }
    (*self_0).numFeatures = 0 as libc::c_int as vl_size;
    cgeom = geom;
    if (*self_0).method as libc::c_uint
        == VL_COVDET_METHOD_DOG as libc::c_int as libc::c_uint
    {
        cgeom.octaveLastSubdivision -= 1 as libc::c_int as libc::c_longlong;
    }
    if ((*self_0).css).is_null()
        || vl_scalespacegeometry_is_equal(
            cgeom,
            vl_scalespace_get_geometry((*self_0).css),
        ) == 0
    {
        if !((*self_0).css).is_null() {
            vl_scalespace_delete((*self_0).css);
        }
        (*self_0).css = vl_scalespace_new_with_geometry(cgeom);
    }
    if (*self_0).method as libc::c_uint
        == VL_COVDET_METHOD_HARRIS_LAPLACE as libc::c_int as libc::c_uint
        || (*self_0).method as libc::c_uint
            == VL_COVDET_METHOD_MULTISCALE_HARRIS as libc::c_int as libc::c_uint
    {
        let mut oct: VlScaleSpaceOctaveGeometry = vl_scalespace_get_octave_geometry(
            (*self_0).gss,
            geom.firstOctave,
        );
        levelxx = vl_malloc(
            (oct.width)
                .wrapping_mul(oct.height)
                .wrapping_mul(
                    ::core::mem::size_of::<libc::c_float>() as libc::c_ulong
                        as libc::c_ulonglong,
                ) as size_t,
        ) as *mut libc::c_float;
        levelyy = vl_malloc(
            (oct.width)
                .wrapping_mul(oct.height)
                .wrapping_mul(
                    ::core::mem::size_of::<libc::c_float>() as libc::c_ulong
                        as libc::c_ulonglong,
                ) as size_t,
        ) as *mut libc::c_float;
        levelxy = vl_malloc(
            (oct.width)
                .wrapping_mul(oct.height)
                .wrapping_mul(
                    ::core::mem::size_of::<libc::c_float>() as libc::c_ulong
                        as libc::c_ulonglong,
                ) as size_t,
        ) as *mut libc::c_float;
    }
    o = cgeom.firstOctave;
    while o <= cgeom.lastOctave {
        let mut oct_0: VlScaleSpaceOctaveGeometry = vl_scalespace_get_octave_geometry(
            (*self_0).css,
            o,
        );
        s = cgeom.octaveFirstSubdivision;
        while s <= cgeom.octaveLastSubdivision {
            let mut level: *mut libc::c_float = vl_scalespace_get_level(
                (*self_0).gss,
                o,
                s,
            );
            let mut clevel: *mut libc::c_float = vl_scalespace_get_level(
                (*self_0).css,
                o,
                s,
            );
            let mut sigma: libc::c_double = vl_scalespace_get_level_sigma(
                (*self_0).css,
                o,
                s,
            );
            match (*self_0).method as libc::c_uint {
                1 => {
                    _vl_dog_response(
                        clevel,
                        vl_scalespace_get_level(
                            (*self_0).gss,
                            o,
                            s + 1 as libc::c_int as libc::c_longlong,
                        ),
                        level,
                        oct_0.width,
                        oct_0.height,
                    );
                }
                4 | 6 => {
                    _vl_harris_response(
                        clevel,
                        level,
                        oct_0.width,
                        oct_0.height,
                        oct_0.step,
                        sigma,
                        1.4f64 * sigma,
                        0.05f64,
                    );
                }
                2 | 3 | 5 => {
                    _vl_det_hessian_response(
                        clevel,
                        level,
                        oct_0.width,
                        oct_0.height,
                        oct_0.step,
                        sigma,
                    );
                }
                _ => {
                    __assert_fail(
                        b"0\0" as *const u8 as *const libc::c_char,
                        b"vl/covdet.c\0" as *const u8 as *const libc::c_char,
                        2000 as libc::c_int as libc::c_uint,
                        (*::core::mem::transmute::<
                            &[u8; 34],
                            &[libc::c_char; 34],
                        >(b"void vl_covdet_detect(VlCovDet *)\0"))
                            .as_ptr(),
                    );
                }
            }
            s += 1;
        }
        o += 1;
    }
    let mut extrema: *mut vl_index = 0 as *mut vl_index;
    let mut extremaBufferSize: vl_size = 0 as libc::c_int as vl_size;
    let mut numExtrema: vl_size = 0;
    let mut index: vl_size = 0;
    o = cgeom.firstOctave;
    while o <= cgeom.lastOctave {
        let mut octgeom: VlScaleSpaceOctaveGeometry = vl_scalespace_get_octave_geometry(
            (*self_0).css,
            o,
        );
        let mut step: libc::c_double = octgeom.step;
        let mut width: vl_size = octgeom.width;
        let mut height: vl_size = octgeom.height;
        let mut depth: vl_size = (cgeom.octaveLastSubdivision
            - cgeom.octaveFirstSubdivision + 1 as libc::c_int as libc::c_longlong)
            as vl_size;
        match (*self_0).method as libc::c_uint {
            1 | 2 => {
                let mut octave: *const libc::c_float = vl_scalespace_get_level(
                    (*self_0).css,
                    o,
                    cgeom.octaveFirstSubdivision,
                );
                numExtrema = vl_find_local_extrema_3(
                    &mut extrema,
                    &mut extremaBufferSize,
                    octave,
                    width,
                    height,
                    depth,
                    0.8f64 * (*self_0).peakThreshold,
                );
                index = 0 as libc::c_int as vl_size;
                while index < numExtrema {
                    let mut refined: VlCovDetExtremum3 = VlCovDetExtremum3 {
                        xi: 0,
                        yi: 0,
                        zi: 0,
                        x: 0.,
                        y: 0.,
                        z: 0.,
                        peakScore: 0.,
                        edgeScore: 0.,
                    };
                    let mut feature: VlCovDetFeature = VlCovDetFeature {
                        frame: VlFrameOrientedEllipse {
                            x: 0.,
                            y: 0.,
                            a11: 0.,
                            a12: 0.,
                            a21: 0.,
                            a22: 0.,
                        },
                        peakScore: 0.,
                        edgeScore: 0.,
                        orientationScore: 0.,
                        laplacianScaleScore: 0.,
                    };
                    let mut ok: vl_bool = 0;
                    memset(
                        &mut feature as *mut VlCovDetFeature as *mut libc::c_void,
                        0 as libc::c_int,
                        ::core::mem::size_of::<VlCovDetFeature>() as libc::c_ulong,
                    );
                    ok = vl_refine_local_extreum_3(
                        &mut refined,
                        octave,
                        width,
                        height,
                        depth,
                        *extrema
                            .offset(
                                (3 as libc::c_int as libc::c_ulonglong)
                                    .wrapping_mul(index)
                                    .wrapping_add(0 as libc::c_int as libc::c_ulonglong)
                                    as isize,
                            ),
                        *extrema
                            .offset(
                                (3 as libc::c_int as libc::c_ulonglong)
                                    .wrapping_mul(index)
                                    .wrapping_add(1 as libc::c_int as libc::c_ulonglong)
                                    as isize,
                            ),
                        *extrema
                            .offset(
                                (3 as libc::c_int as libc::c_ulonglong)
                                    .wrapping_mul(index)
                                    .wrapping_add(2 as libc::c_int as libc::c_ulonglong)
                                    as isize,
                            ),
                    );
                    ok
                        &= (fabs(refined.peakScore as libc::c_double)
                            > (*self_0).peakThreshold) as libc::c_int;
                    ok
                        &= ((refined.edgeScore as libc::c_double)
                            < (*self_0).edgeThreshold) as libc::c_int;
                    if ok != 0 {
                        let mut sigma_0: libc::c_double = cgeom.baseScale
                            * pow(
                                2.0f64,
                                (o as libc::c_float
                                    + (refined.z
                                        + cgeom.octaveFirstSubdivision as libc::c_float)
                                        / cgeom.octaveResolution as libc::c_float) as libc::c_double,
                            );
                        feature
                            .frame
                            .x = (refined.x as libc::c_double * step) as libc::c_float;
                        feature
                            .frame
                            .y = (refined.y as libc::c_double * step) as libc::c_float;
                        feature.frame.a11 = sigma_0 as libc::c_float;
                        feature.frame.a12 = 0.0f64 as libc::c_float;
                        feature.frame.a21 = 0.0f64 as libc::c_float;
                        feature.frame.a22 = sigma_0 as libc::c_float;
                        feature.peakScore = refined.peakScore;
                        feature.edgeScore = refined.edgeScore;
                        vl_covdet_append_feature(self_0, &mut feature);
                    }
                    index = index.wrapping_add(1);
                }
            }
            _ => {
                s = cgeom.octaveFirstSubdivision;
                while s < cgeom.octaveLastSubdivision {
                    let mut level_0: *const libc::c_float = vl_scalespace_get_level(
                        (*self_0).css,
                        o,
                        s,
                    );
                    numExtrema = vl_find_local_extrema_2(
                        &mut extrema,
                        &mut extremaBufferSize,
                        level_0,
                        width,
                        height,
                        0.8f64 * (*self_0).peakThreshold,
                    );
                    index = 0 as libc::c_int as vl_size;
                    while index < numExtrema {
                        let mut refined_0: VlCovDetExtremum2 = VlCovDetExtremum2 {
                            xi: 0,
                            yi: 0,
                            x: 0.,
                            y: 0.,
                            peakScore: 0.,
                            edgeScore: 0.,
                        };
                        let mut feature_0: VlCovDetFeature = VlCovDetFeature {
                            frame: VlFrameOrientedEllipse {
                                x: 0.,
                                y: 0.,
                                a11: 0.,
                                a12: 0.,
                                a21: 0.,
                                a22: 0.,
                            },
                            peakScore: 0.,
                            edgeScore: 0.,
                            orientationScore: 0.,
                            laplacianScaleScore: 0.,
                        };
                        let mut ok_0: vl_bool = 0;
                        memset(
                            &mut feature_0 as *mut VlCovDetFeature as *mut libc::c_void,
                            0 as libc::c_int,
                            ::core::mem::size_of::<VlCovDetFeature>() as libc::c_ulong,
                        );
                        ok_0 = vl_refine_local_extreum_2(
                            &mut refined_0,
                            level_0,
                            width,
                            height,
                            *extrema
                                .offset(
                                    (2 as libc::c_int as libc::c_ulonglong)
                                        .wrapping_mul(index)
                                        .wrapping_add(0 as libc::c_int as libc::c_ulonglong)
                                        as isize,
                                ),
                            *extrema
                                .offset(
                                    (2 as libc::c_int as libc::c_ulonglong)
                                        .wrapping_mul(index)
                                        .wrapping_add(1 as libc::c_int as libc::c_ulonglong)
                                        as isize,
                                ),
                        );
                        ok_0
                            &= (fabs(refined_0.peakScore as libc::c_double)
                                > (*self_0).peakThreshold) as libc::c_int;
                        ok_0
                            &= ((refined_0.edgeScore as libc::c_double)
                                < (*self_0).edgeThreshold) as libc::c_int;
                        if ok_0 != 0 {
                            let mut sigma_1: libc::c_double = cgeom.baseScale
                                * pow(
                                    2.0f64,
                                    o as libc::c_double
                                        + s as libc::c_double
                                            / cgeom.octaveResolution as libc::c_double,
                                );
                            feature_0
                                .frame
                                .x = (refined_0.x as libc::c_double * step)
                                as libc::c_float;
                            feature_0
                                .frame
                                .y = (refined_0.y as libc::c_double * step)
                                as libc::c_float;
                            feature_0.frame.a11 = sigma_1 as libc::c_float;
                            feature_0.frame.a12 = 0.0f64 as libc::c_float;
                            feature_0.frame.a21 = 0.0f64 as libc::c_float;
                            feature_0.frame.a22 = sigma_1 as libc::c_float;
                            feature_0.peakScore = refined_0.peakScore;
                            feature_0.edgeScore = refined_0.edgeScore;
                            vl_covdet_append_feature(self_0, &mut feature_0);
                        }
                        index = index.wrapping_add(1);
                    }
                    s += 1;
                }
            }
        }
        o += 1;
    }
    if !extrema.is_null() {
        vl_free(extrema as *mut libc::c_void);
        extrema = 0 as *mut vl_index;
    }
    match (*self_0).method as libc::c_uint {
        4 | 3 => {
            vl_covdet_extract_laplacian_scales(self_0);
        }
        _ => {}
    }
    if (*self_0).nonExtremaSuppression != 0. {
        let mut i: vl_index = 0;
        let mut j: vl_index = 0;
        let mut tol: libc::c_double = (*self_0).nonExtremaSuppression;
        (*self_0).numNonExtremaSuppressed = 0 as libc::c_int as vl_size;
        i = 0 as libc::c_int as vl_index;
        while i < (*self_0).numFeatures as libc::c_int as libc::c_longlong {
            let mut x: libc::c_double = (*((*self_0).features).offset(i as isize))
                .frame
                .x as libc::c_double;
            let mut y: libc::c_double = (*((*self_0).features).offset(i as isize))
                .frame
                .y as libc::c_double;
            let mut sigma_2: libc::c_double = (*((*self_0).features).offset(i as isize))
                .frame
                .a11 as libc::c_double;
            let mut score: libc::c_double = (*((*self_0).features).offset(i as isize))
                .peakScore as libc::c_double;
            j = 0 as libc::c_int as vl_index;
            while j < (*self_0).numFeatures as libc::c_int as libc::c_longlong {
                let mut dx_: libc::c_double = (*((*self_0).features).offset(j as isize))
                    .frame
                    .x as libc::c_double - x;
                let mut dy_: libc::c_double = (*((*self_0).features).offset(j as isize))
                    .frame
                    .y as libc::c_double - y;
                let mut sigma_: libc::c_double = (*((*self_0).features)
                    .offset(j as isize))
                    .frame
                    .a11 as libc::c_double;
                let mut score_: libc::c_double = (*((*self_0).features)
                    .offset(j as isize))
                    .peakScore as libc::c_double;
                if !(score_ == 0 as libc::c_int as libc::c_double) {
                    if sigma_2 < (1 as libc::c_int as libc::c_double + tol) * sigma_
                        && sigma_ < (1 as libc::c_int as libc::c_double + tol) * sigma_2
                        && vl_abs_d(dx_) < tol * sigma_2 && vl_abs_d(dy_) < tol * sigma_2
                        && vl_abs_d(score) > vl_abs_d(score_)
                    {
                        (*((*self_0).features).offset(j as isize))
                            .peakScore = 0 as libc::c_int as libc::c_float;
                        (*self_0)
                            .numNonExtremaSuppressed = ((*self_0)
                            .numNonExtremaSuppressed)
                            .wrapping_add(1);
                    }
                }
                j += 1;
            }
            i += 1;
        }
        j = 0 as libc::c_int as vl_index;
        i = 0 as libc::c_int as vl_index;
        while i < (*self_0).numFeatures as libc::c_int as libc::c_longlong {
            let mut feature_1: VlCovDetFeature = *((*self_0).features)
                .offset(i as isize);
            if (*((*self_0).features).offset(i as isize)).peakScore
                != 0 as libc::c_int as libc::c_float
            {
                let fresh0 = j;
                j = j + 1;
                *((*self_0).features).offset(fresh0 as isize) = feature_1;
            }
            i += 1;
        }
        (*self_0).numFeatures = j as vl_size;
    }
    if !levelxx.is_null() {
        vl_free(levelxx as *mut libc::c_void);
    }
    if !levelyy.is_null() {
        vl_free(levelyy as *mut libc::c_void);
    }
    if !levelxy.is_null() {
        vl_free(levelxy as *mut libc::c_void);
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_extract_patch_helper(
    mut self_0: *mut VlCovDet,
    mut sigma1: *mut libc::c_double,
    mut sigma2: *mut libc::c_double,
    mut patch: *mut libc::c_float,
    mut resolution: vl_size,
    mut extent: libc::c_double,
    mut sigma: libc::c_double,
    mut A_: *mut libc::c_double,
    mut T_: *mut libc::c_double,
    mut d1: libc::c_double,
    mut d2: libc::c_double,
) -> vl_bool {
    let mut o: vl_index = 0;
    let mut s: vl_index = 0;
    let mut factor: libc::c_double = 0.;
    let mut sigma_: libc::c_double = 0.;
    let mut level: *const libc::c_float = 0 as *const libc::c_float;
    let mut width: vl_size = 0;
    let mut height: vl_size = 0;
    let mut step: libc::c_double = 0.;
    let mut A: [libc::c_double; 4] = [
        *A_.offset(0 as libc::c_int as isize),
        *A_.offset(1 as libc::c_int as isize),
        *A_.offset(2 as libc::c_int as isize),
        *A_.offset(3 as libc::c_int as isize),
    ];
    let mut T: [libc::c_double; 2] = [
        *T_.offset(0 as libc::c_int as isize),
        *T_.offset(1 as libc::c_int as isize),
    ];
    let mut geom: VlScaleSpaceGeometry = vl_scalespace_get_geometry((*self_0).gss);
    let mut oct: VlScaleSpaceOctaveGeometry = VlScaleSpaceOctaveGeometry {
        width: 0,
        height: 0,
        step: 0.,
    };
    factor = 1.0f64 / (if d1 < d2 { d1 } else { d2 });
    o = geom.firstOctave + 1 as libc::c_int as libc::c_longlong;
    while o <= geom.lastOctave {
        s = vl_floor_d(
            f64::log2(sigma / (factor * geom.baseScale)) - o as libc::c_double,
        ) as vl_index;
        s = if s > geom.octaveFirstSubdivision {
            s
        } else {
            geom.octaveFirstSubdivision
        };
        s = if s < geom.octaveLastSubdivision { s } else { geom.octaveLastSubdivision };
        sigma_ = geom.baseScale
            * pow(
                2.0f64,
                o as libc::c_double
                    + s as libc::c_double / geom.octaveResolution as libc::c_double,
            );
        if factor * sigma_ > sigma {
            o -= 1;
            break;
        } else {
            o += 1;
        }
    }
    o = if o < geom.lastOctave { o } else { geom.lastOctave };
    s = vl_floor_d(f64::log2(sigma / (factor * geom.baseScale)) - o as libc::c_double)
        as vl_index;
    s = if s > geom.octaveFirstSubdivision { s } else { geom.octaveFirstSubdivision };
    s = if s < geom.octaveLastSubdivision { s } else { geom.octaveLastSubdivision };
    sigma_ = geom.baseScale
        * pow(
            2.0f64,
            o as libc::c_double
                + s as libc::c_double / geom.octaveResolution as libc::c_double,
        );
    if !sigma1.is_null() {
        *sigma1 = sigma_ / d1;
    }
    if !sigma2.is_null() {
        *sigma2 = sigma_ / d2;
    }
    level = vl_scalespace_get_level((*self_0).gss, o, s);
    oct = vl_scalespace_get_octave_geometry((*self_0).gss, o);
    width = oct.width;
    height = oct.height;
    step = oct.step;
    A[0 as libc::c_int as usize] /= step;
    A[1 as libc::c_int as usize] /= step;
    A[2 as libc::c_int as usize] /= step;
    A[3 as libc::c_int as usize] /= step;
    T[0 as libc::c_int as usize] /= step;
    T[1 as libc::c_int as usize] /= step;
    let mut x0i: vl_index = 0;
    let mut y0i: vl_index = 0;
    let mut x1i: vl_index = 0;
    let mut y1i: vl_index = 0;
    let mut x0: libc::c_double = vl_infinity_d.value;
    let mut x1: libc::c_double = -vl_infinity_d.value;
    let mut y0: libc::c_double = vl_infinity_d.value;
    let mut y1: libc::c_double = -vl_infinity_d.value;
    let mut boxx: [libc::c_double; 4] = [extent, extent, -extent, -extent];
    let mut boxy: [libc::c_double; 4] = [-extent, extent, extent, -extent];
    let mut i: libc::c_int = 0;
    i = 0 as libc::c_int;
    while i < 4 as libc::c_int {
        let mut x: libc::c_double = A[0 as libc::c_int as usize] * boxx[i as usize]
            + A[2 as libc::c_int as usize] * boxy[i as usize]
            + T[0 as libc::c_int as usize];
        let mut y: libc::c_double = A[1 as libc::c_int as usize] * boxx[i as usize]
            + A[3 as libc::c_int as usize] * boxy[i as usize]
            + T[1 as libc::c_int as usize];
        x0 = if x0 < x { x0 } else { x };
        x1 = if x1 > x { x1 } else { x };
        y0 = if y0 < y { y0 } else { y };
        y1 = if y1 > y { y1 } else { y };
        i += 1;
    }
    if (x0 < 0 as libc::c_int as libc::c_double
        || x1 > (width as libc::c_int - 1 as libc::c_int) as libc::c_double
        || y0 < 0 as libc::c_int as libc::c_double
        || y1 > (height as libc::c_int - 1 as libc::c_int) as libc::c_double)
        && (*self_0).allowPaddedWarping == 0
    {
        return vl_set_last_error(
            5 as libc::c_int,
            b"Frame out of image.\0" as *const u8 as *const libc::c_char,
        );
    }
    x0i = (floor(x0) - 1 as libc::c_int as libc::c_double) as vl_index;
    y0i = (floor(y0) - 1 as libc::c_int as libc::c_double) as vl_index;
    x1i = (ceil(x1) + 1 as libc::c_int as libc::c_double) as vl_index;
    y1i = (ceil(y1) + 1 as libc::c_int as libc::c_double) as vl_index;
    if x0i < 0 as libc::c_int as libc::c_longlong
        || x1i > (width as libc::c_int - 1 as libc::c_int) as libc::c_longlong
        || y0i < 0 as libc::c_int as libc::c_longlong
        || y1i > (height as libc::c_int - 1 as libc::c_int) as libc::c_longlong
    {
        let mut xi: vl_index = 0;
        let mut yi: vl_index = 0;
        let mut padx0: vl_index = if 0 as libc::c_int as libc::c_longlong > -x0i {
            0 as libc::c_int as libc::c_longlong
        } else {
            -x0i
        };
        let mut pady0: vl_index = if 0 as libc::c_int as libc::c_longlong > -y0i {
            0 as libc::c_int as libc::c_longlong
        } else {
            -y0i
        };
        let mut padx1: vl_index = if 0 as libc::c_int as libc::c_longlong
            > x1i - (width as libc::c_int - 1 as libc::c_int) as libc::c_longlong
        {
            0 as libc::c_int as libc::c_longlong
        } else {
            x1i - (width as libc::c_int - 1 as libc::c_int) as libc::c_longlong
        };
        let mut pady1: vl_index = if 0 as libc::c_int as libc::c_longlong
            > y1i - (height as libc::c_int - 1 as libc::c_int) as libc::c_longlong
        {
            0 as libc::c_int as libc::c_longlong
        } else {
            y1i - (height as libc::c_int - 1 as libc::c_int) as libc::c_longlong
        };
        let mut patchWidth: vl_index = x1i - x0i + 1 as libc::c_int as libc::c_longlong;
        let mut patchHeight: vl_index = y1i - y0i + 1 as libc::c_int as libc::c_longlong;
        let mut patchBufferSize: vl_size = ((patchWidth * patchHeight)
            as libc::c_ulonglong)
            .wrapping_mul(
                ::core::mem::size_of::<libc::c_float>() as libc::c_ulong
                    as libc::c_ulonglong,
            );
        if patchBufferSize > (*self_0).patchBufferSize {
            let mut err: libc::c_int = _vl_resize_buffer(
                &mut (*self_0).patch as *mut *mut libc::c_float
                    as *mut *mut libc::c_void,
                &mut (*self_0).patchBufferSize,
                patchBufferSize,
            );
            if err != 0 {
                return vl_set_last_error(
                    2 as libc::c_int,
                    b"Unable to allocate data.\0" as *const u8 as *const libc::c_char,
                );
            }
        }
        if pady0 < patchHeight - pady1 {
            yi = y0i + pady0;
            while yi < y0i + patchHeight - pady1 {
                let mut dst: *mut libc::c_float = ((*self_0).patch)
                    .offset(((yi - y0i) * patchWidth) as isize);
                let mut src: *const libc::c_float = level
                    .offset((yi as libc::c_ulonglong).wrapping_mul(width) as isize)
                    .offset(
                        (if (if 0 as libc::c_int as libc::c_longlong > x0i {
                            0 as libc::c_int as libc::c_longlong
                        } else {
                            x0i
                        })
                            < (width as libc::c_int - 1 as libc::c_int)
                                as libc::c_longlong
                        {
                            (if 0 as libc::c_int as libc::c_longlong > x0i {
                                0 as libc::c_int as libc::c_longlong
                            } else {
                                x0i
                            })
                        } else {
                            (width as libc::c_int - 1 as libc::c_int) as libc::c_longlong
                        }) as isize,
                    );
                xi = x0i;
                while xi < x0i + padx0 {
                    let fresh1 = dst;
                    dst = dst.offset(1);
                    *fresh1 = *src;
                    xi += 1;
                }
                while xi
                    < x0i + patchWidth - padx1 - 2 as libc::c_int as libc::c_longlong
                {
                    let fresh2 = src;
                    src = src.offset(1);
                    let fresh3 = dst;
                    dst = dst.offset(1);
                    *fresh3 = *fresh2;
                    xi += 1;
                }
                while xi < x0i + patchWidth {
                    let fresh4 = dst;
                    dst = dst.offset(1);
                    *fresh4 = *src;
                    xi += 1;
                }
                yi += 1;
            }
            yi = 0 as libc::c_int as vl_index;
            while yi < pady0 {
                memcpy(
                    ((*self_0).patch).offset((yi * patchWidth) as isize)
                        as *mut libc::c_void,
                    ((*self_0).patch).offset((pady0 * patchWidth) as isize)
                        as *const libc::c_void,
                    (patchWidth as libc::c_ulonglong)
                        .wrapping_mul(
                            ::core::mem::size_of::<libc::c_float>() as libc::c_ulong
                                as libc::c_ulonglong,
                        ) as libc::c_ulong,
                );
                yi += 1;
            }
            yi = patchHeight - pady1;
            while yi < patchHeight {
                memcpy(
                    ((*self_0).patch).offset((yi * patchWidth) as isize)
                        as *mut libc::c_void,
                    ((*self_0).patch)
                        .offset(
                            ((patchHeight - pady1 - 1 as libc::c_int as libc::c_longlong)
                                * patchWidth) as isize,
                        ) as *const libc::c_void,
                    (patchWidth as libc::c_ulonglong)
                        .wrapping_mul(
                            ::core::mem::size_of::<libc::c_float>() as libc::c_ulong
                                as libc::c_ulonglong,
                        ) as libc::c_ulong,
                );
                yi += 1;
            }
        } else {
            memset(
                (*self_0).patch as *mut libc::c_void,
                0 as libc::c_int,
                (*self_0).patchBufferSize as libc::c_ulong,
            );
        }
        level = (*self_0).patch;
        width = patchWidth as vl_size;
        height = patchHeight as vl_size;
        T[0 as libc::c_int as usize] -= x0i as libc::c_double;
        T[1 as libc::c_int as usize] -= y0i as libc::c_double;
    }
    let mut pt: *mut libc::c_float = patch;
    let mut yhat: libc::c_double = -extent;
    let mut xxi: vl_index = 0;
    let mut yyi: vl_index = 0;
    let mut stephat: libc::c_double = extent / resolution as libc::c_double;
    yyi = 0 as libc::c_int as vl_index;
    while yyi
        < (2 as libc::c_int * resolution as libc::c_int + 1 as libc::c_int)
            as libc::c_longlong
    {
        let mut xhat: libc::c_double = -extent;
        let mut rx: libc::c_double = A[2 as libc::c_int as usize] * yhat
            + T[0 as libc::c_int as usize];
        let mut ry: libc::c_double = A[3 as libc::c_int as usize] * yhat
            + T[1 as libc::c_int as usize];
        xxi = 0 as libc::c_int as vl_index;
        while xxi
            < (2 as libc::c_int * resolution as libc::c_int + 1 as libc::c_int)
                as libc::c_longlong
        {
            let mut x_0: libc::c_double = A[0 as libc::c_int as usize] * xhat + rx;
            let mut y_0: libc::c_double = A[1 as libc::c_int as usize] * xhat + ry;
            let mut xi_0: vl_index = vl_floor_d(x_0) as vl_index;
            let mut yi_0: vl_index = vl_floor_d(y_0) as vl_index;
            let mut i00: libc::c_double = *level
                .offset(
                    (yi_0 as libc::c_ulonglong)
                        .wrapping_mul(width)
                        .wrapping_add(xi_0 as libc::c_ulonglong) as isize,
                ) as libc::c_double;
            let mut i10: libc::c_double = *level
                .offset(
                    (yi_0 as libc::c_ulonglong)
                        .wrapping_mul(width)
                        .wrapping_add(xi_0 as libc::c_ulonglong)
                        .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as isize,
                ) as libc::c_double;
            let mut i01: libc::c_double = *level
                .offset(
                    ((yi_0 + 1 as libc::c_int as libc::c_longlong) as libc::c_ulonglong)
                        .wrapping_mul(width)
                        .wrapping_add(xi_0 as libc::c_ulonglong) as isize,
                ) as libc::c_double;
            let mut i11: libc::c_double = *level
                .offset(
                    ((yi_0 + 1 as libc::c_int as libc::c_longlong) as libc::c_ulonglong)
                        .wrapping_mul(width)
                        .wrapping_add(xi_0 as libc::c_ulonglong)
                        .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as isize,
                ) as libc::c_double;
            let mut wx: libc::c_double = x_0 - xi_0 as libc::c_double;
            let mut wy: libc::c_double = y_0 - yi_0 as libc::c_double;
            if xi_0 >= 0 as libc::c_int as libc::c_longlong
                && xi_0 <= (width as libc::c_int - 1 as libc::c_int) as libc::c_longlong
            {} else {
                __assert_fail(
                    b"xi >= 0 && xi <= (signed)width - 1\0" as *const u8
                        as *const libc::c_char,
                    b"vl/covdet.c\0" as *const u8 as *const libc::c_char,
                    2400 as libc::c_int as libc::c_uint,
                    (*::core::mem::transmute::<
                        &[u8; 141],
                        &[libc::c_char; 141],
                    >(
                        b"vl_bool vl_covdet_extract_patch_helper(VlCovDet *, double *, double *, float *, vl_size, double, double, double *, double *, double, double)\0",
                    ))
                        .as_ptr(),
                );
            }
            if yi_0 >= 0 as libc::c_int as libc::c_longlong
                && yi_0 <= (height as libc::c_int - 1 as libc::c_int) as libc::c_longlong
            {} else {
                __assert_fail(
                    b"yi >= 0 && yi <= (signed)height - 1\0" as *const u8
                        as *const libc::c_char,
                    b"vl/covdet.c\0" as *const u8 as *const libc::c_char,
                    2401 as libc::c_int as libc::c_uint,
                    (*::core::mem::transmute::<
                        &[u8; 141],
                        &[libc::c_char; 141],
                    >(
                        b"vl_bool vl_covdet_extract_patch_helper(VlCovDet *, double *, double *, float *, vl_size, double, double, double *, double *, double, double)\0",
                    ))
                        .as_ptr(),
                );
            }
            let fresh5 = pt;
            pt = pt.offset(1);
            *fresh5 = ((1.0f64 - wy) * ((1.0f64 - wx) * i00 + wx * i10)
                + wy * ((1.0f64 - wx) * i01 + wx * i11)) as libc::c_float;
            xhat += stephat;
            xxi += 1;
        }
        yhat += stephat;
        yyi += 1;
    }
    return 0 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_extract_patch_for_frame(
    mut self_0: *mut VlCovDet,
    mut patch: *mut libc::c_float,
    mut resolution: vl_size,
    mut extent: libc::c_double,
    mut sigma: libc::c_double,
    mut frame: VlFrameOrientedEllipse,
) -> vl_bool {
    let mut A: [libc::c_double; 4] = [
        frame.a11 as libc::c_double,
        frame.a21 as libc::c_double,
        frame.a12 as libc::c_double,
        frame.a22 as libc::c_double,
    ];
    let mut T: [libc::c_double; 2] = [
        frame.x as libc::c_double,
        frame.y as libc::c_double,
    ];
    let mut D: [libc::c_double; 4] = [0.; 4];
    let mut U: [libc::c_double; 4] = [0.; 4];
    let mut V: [libc::c_double; 4] = [0.; 4];
    vl_svd2(D.as_mut_ptr(), U.as_mut_ptr(), V.as_mut_ptr(), A.as_mut_ptr());
    return vl_covdet_extract_patch_helper(
        self_0,
        0 as *mut libc::c_double,
        0 as *mut libc::c_double,
        patch,
        resolution,
        extent,
        sigma,
        A.as_mut_ptr(),
        T.as_mut_ptr(),
        D[0 as libc::c_int as usize],
        D[3 as libc::c_int as usize],
    );
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_extract_affine_shape_for_frame(
    mut self_0: *mut VlCovDet,
    mut adapted: *mut VlFrameOrientedEllipse,
    mut frame: VlFrameOrientedEllipse,
) -> libc::c_int {
    let mut iter: vl_index = 0 as libc::c_int as vl_index;
    let mut A: [libc::c_double; 4] = [
        frame.a11 as libc::c_double,
        frame.a21 as libc::c_double,
        frame.a12 as libc::c_double,
        frame.a22 as libc::c_double,
    ];
    let mut T: [libc::c_double; 2] = [
        frame.x as libc::c_double,
        frame.y as libc::c_double,
    ];
    let mut U: [libc::c_double; 4] = [0.; 4];
    let mut V: [libc::c_double; 4] = [0.; 4];
    let mut D: [libc::c_double; 4] = [0.; 4];
    let mut M: [libc::c_double; 4] = [0.; 4];
    let mut P: [libc::c_double; 4] = [0.; 4];
    let mut P_: [libc::c_double; 4] = [0.; 4];
    let mut Q: [libc::c_double; 4] = [0.; 4];
    let mut sigma1: libc::c_double = 0.;
    let mut sigma2: libc::c_double = 0.;
    let mut sigmaD: libc::c_double = 1 as libc::c_int as libc::c_double;
    let mut factor: libc::c_double = 0.;
    let mut anisotropy: libc::c_double = 0.;
    let mut referenceScale: libc::c_double = 0.;
    let resolution: vl_size = 20 as libc::c_int as vl_size;
    let side: vl_size = (2 as libc::c_int * 20 as libc::c_int + 1 as libc::c_int)
        as vl_size;
    let extent: libc::c_double = (3 as libc::c_int * 3 as libc::c_int) as libc::c_double;
    *adapted = frame;
    loop {
        let mut lxx: libc::c_double = 0 as libc::c_int as libc::c_double;
        let mut lxy: libc::c_double = 0 as libc::c_int as libc::c_double;
        let mut lyy: libc::c_double = 0 as libc::c_int as libc::c_double;
        let mut k: vl_index = 0;
        let mut err: libc::c_int = 0;
        vl_svd2(D.as_mut_ptr(), U.as_mut_ptr(), V.as_mut_ptr(), A.as_mut_ptr());
        anisotropy = if D[0 as libc::c_int as usize] / D[3 as libc::c_int as usize]
            > D[3 as libc::c_int as usize] / D[0 as libc::c_int as usize]
        {
            D[0 as libc::c_int as usize] / D[3 as libc::c_int as usize]
        } else {
            D[3 as libc::c_int as usize] / D[0 as libc::c_int as usize]
        };
        if anisotropy > 5 as libc::c_int as libc::c_double {
            break;
        }
        if iter == 0 as libc::c_int as libc::c_longlong {
            referenceScale = if D[0 as libc::c_int as usize]
                < D[3 as libc::c_int as usize]
            {
                D[0 as libc::c_int as usize]
            } else {
                D[3 as libc::c_int as usize]
            };
            factor = 1.0f64;
        } else {
            factor = referenceScale
                / (if D[0 as libc::c_int as usize] < D[3 as libc::c_int as usize] {
                    D[0 as libc::c_int as usize]
                } else {
                    D[3 as libc::c_int as usize]
                });
        }
        D[0 as libc::c_int as usize] *= factor;
        D[3 as libc::c_int as usize] *= factor;
        A[0 as libc::c_int
            as usize] = U[0 as libc::c_int as usize] * D[0 as libc::c_int as usize];
        A[1 as libc::c_int
            as usize] = U[1 as libc::c_int as usize] * D[0 as libc::c_int as usize];
        A[2 as libc::c_int
            as usize] = U[2 as libc::c_int as usize] * D[3 as libc::c_int as usize];
        A[3 as libc::c_int
            as usize] = U[3 as libc::c_int as usize] * D[3 as libc::c_int as usize];
        (*adapted).a11 = A[0 as libc::c_int as usize] as libc::c_float;
        (*adapted).a21 = A[1 as libc::c_int as usize] as libc::c_float;
        (*adapted).a12 = A[2 as libc::c_int as usize] as libc::c_float;
        (*adapted).a22 = A[3 as libc::c_int as usize] as libc::c_float;
        iter += 1;
        if iter >= 15 as libc::c_int as libc::c_longlong {
            break;
        }
        err = vl_covdet_extract_patch_helper(
            self_0,
            &mut sigma1,
            &mut sigma2,
            ((*self_0).aaPatch).as_mut_ptr(),
            resolution,
            extent,
            sigmaD,
            A.as_mut_ptr(),
            T.as_mut_ptr(),
            D[0 as libc::c_int as usize],
            D[3 as libc::c_int as usize],
        );
        if err != 0 {
            return err;
        }
        if (*self_0).aaAccurateSmoothing != 0 {
            let mut deltaSigma1: libc::c_double = sqrt(
                if sigmaD * sigmaD - sigma1 * sigma1 > 0 as libc::c_int as libc::c_double
                {
                    sigmaD * sigmaD - sigma1 * sigma1
                } else {
                    0 as libc::c_int as libc::c_double
                },
            );
            let mut deltaSigma2: libc::c_double = sqrt(
                if sigmaD * sigmaD - sigma2 * sigma2 > 0 as libc::c_int as libc::c_double
                {
                    sigmaD * sigmaD - sigma2 * sigma2
                } else {
                    0 as libc::c_int as libc::c_double
                },
            );
            let mut stephat: libc::c_double = extent / resolution as libc::c_double;
            vl_imsmooth_f(
                ((*self_0).aaPatch).as_mut_ptr(),
                side,
                ((*self_0).aaPatch).as_mut_ptr(),
                side,
                side,
                side,
                deltaSigma1 / stephat,
                deltaSigma2 / stephat,
            );
        }
        vl_imgradient_f(
            ((*self_0).aaPatchX).as_mut_ptr(),
            ((*self_0).aaPatchY).as_mut_ptr(),
            1 as libc::c_int as vl_size,
            side,
            ((*self_0).aaPatch).as_mut_ptr(),
            side,
            side,
            side,
        );
        k = 0 as libc::c_int as vl_index;
        while k < side.wrapping_mul(side) as libc::c_int as libc::c_longlong {
            let mut lx: libc::c_double = (*self_0).aaPatchX[k as usize]
                as libc::c_double;
            let mut ly: libc::c_double = (*self_0).aaPatchY[k as usize]
                as libc::c_double;
            lxx += lx * lx * (*self_0).aaMask[k as usize] as libc::c_double;
            lyy += ly * ly * (*self_0).aaMask[k as usize] as libc::c_double;
            lxy += lx * ly * (*self_0).aaMask[k as usize] as libc::c_double;
            k += 1;
        }
        M[0 as libc::c_int as usize] = lxx;
        M[1 as libc::c_int as usize] = lxy;
        M[2 as libc::c_int as usize] = lxy;
        M[3 as libc::c_int as usize] = lyy;
        if lxx == 0 as libc::c_int as libc::c_double
            || lyy == 0 as libc::c_int as libc::c_double
        {
            *adapted = frame;
            break;
        } else {
            vl_svd2(Q.as_mut_ptr(), P.as_mut_ptr(), P_.as_mut_ptr(), M.as_mut_ptr());
            if Q[3 as libc::c_int as usize] / Q[0 as libc::c_int as usize] < 1.001f64
                && Q[0 as libc::c_int as usize] / Q[3 as libc::c_int as usize] < 1.001f64
            {
                break;
            }
            let mut Ap: [libc::c_double; 4] = [0.; 4];
            let mut q0: libc::c_double = sqrt(Q[0 as libc::c_int as usize]);
            let mut q1: libc::c_double = sqrt(Q[3 as libc::c_int as usize]);
            Ap[0 as libc::c_int
                as usize] = (A[0 as libc::c_int as usize] * P[0 as libc::c_int as usize]
                + A[2 as libc::c_int as usize] * P[1 as libc::c_int as usize]) / q0;
            Ap[1 as libc::c_int
                as usize] = (A[1 as libc::c_int as usize] * P[0 as libc::c_int as usize]
                + A[3 as libc::c_int as usize] * P[1 as libc::c_int as usize]) / q0;
            Ap[2 as libc::c_int
                as usize] = (A[0 as libc::c_int as usize] * P[2 as libc::c_int as usize]
                + A[2 as libc::c_int as usize] * P[3 as libc::c_int as usize]) / q1;
            Ap[3 as libc::c_int
                as usize] = (A[1 as libc::c_int as usize] * P[2 as libc::c_int as usize]
                + A[3 as libc::c_int as usize] * P[3 as libc::c_int as usize]) / q1;
            memcpy(
                A.as_mut_ptr() as *mut libc::c_void,
                Ap.as_mut_ptr() as *const libc::c_void,
                (4 as libc::c_int as libc::c_ulong)
                    .wrapping_mul(
                        ::core::mem::size_of::<libc::c_double>() as libc::c_ulong,
                    ),
            );
        }
    }
    let mut A_0: [libc::c_double; 4] = [
        (*adapted).a11 as libc::c_double,
        (*adapted).a21 as libc::c_double,
        (*adapted).a12 as libc::c_double,
        (*adapted).a22 as libc::c_double,
    ];
    let mut ref_0: [libc::c_double; 2] = [0.; 2];
    let mut ref_: [libc::c_double; 2] = [0.; 2];
    let mut angle: libc::c_double = 0.;
    let mut angle_: libc::c_double = 0.;
    let mut dangle: libc::c_double = 0.;
    let mut r1: libc::c_double = 0.;
    let mut r2: libc::c_double = 0.;
    if (*self_0).transposed != 0 {
        ref_0[0 as libc::c_int as usize] = 1 as libc::c_int as libc::c_double;
        ref_0[1 as libc::c_int as usize] = 0 as libc::c_int as libc::c_double;
    } else {
        ref_0[0 as libc::c_int as usize] = 0 as libc::c_int as libc::c_double;
        ref_0[1 as libc::c_int as usize] = 1 as libc::c_int as libc::c_double;
    }
    vl_solve_linear_system_2(ref_.as_mut_ptr(), A_0.as_mut_ptr(), ref_0.as_mut_ptr());
    angle = atan2(ref_0[1 as libc::c_int as usize], ref_0[0 as libc::c_int as usize]);
    angle_ = atan2(ref_[1 as libc::c_int as usize], ref_[0 as libc::c_int as usize]);
    dangle = angle_ - angle;
    r1 = cos(dangle);
    r2 = sin(dangle);
    (*adapted)
        .a11 = (A_0[0 as libc::c_int as usize] * r1
        + A_0[2 as libc::c_int as usize] * r2) as libc::c_float;
    (*adapted)
        .a21 = (A_0[1 as libc::c_int as usize] * r1
        + A_0[3 as libc::c_int as usize] * r2) as libc::c_float;
    (*adapted)
        .a12 = (-A_0[0 as libc::c_int as usize] * r2
        + A_0[2 as libc::c_int as usize] * r1) as libc::c_float;
    (*adapted)
        .a22 = (-A_0[1 as libc::c_int as usize] * r2
        + A_0[3 as libc::c_int as usize] * r1) as libc::c_float;
    return 0 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_extract_affine_shape(mut self_0: *mut VlCovDet) {
    let mut i: vl_index = 0;
    let mut j: vl_index = 0 as libc::c_int as vl_index;
    let mut numFeatures: vl_size = vl_covdet_get_num_features(self_0);
    let mut feature: *mut VlCovDetFeature = vl_covdet_get_features(self_0)
        as *mut VlCovDetFeature;
    i = 0 as libc::c_int as vl_index;
    while i < numFeatures as libc::c_int as libc::c_longlong {
        let mut status: libc::c_int = 0;
        let mut adapted: VlFrameOrientedEllipse = VlFrameOrientedEllipse {
            x: 0.,
            y: 0.,
            a11: 0.,
            a12: 0.,
            a21: 0.,
            a22: 0.,
        };
        status = vl_covdet_extract_affine_shape_for_frame(
            self_0,
            &mut adapted,
            (*feature.offset(i as isize)).frame,
        );
        if status == 0 as libc::c_int {
            *feature.offset(j as isize) = *feature.offset(i as isize);
            (*feature.offset(j as isize)).frame = adapted;
            j += 1;
        }
        i += 1;
    }
    (*self_0).numFeatures = j as vl_size;
}
unsafe extern "C" fn _vl_covdet_compare_orientations_descending(
    mut a_: *const libc::c_void,
    mut b_: *const libc::c_void,
) -> libc::c_int {
    let mut a: *const VlCovDetFeatureOrientation = a_
        as *const VlCovDetFeatureOrientation;
    let mut b: *const VlCovDetFeatureOrientation = b_
        as *const VlCovDetFeatureOrientation;
    if (*a).score > (*b).score {
        return -(1 as libc::c_int);
    }
    if (*a).score < (*b).score {
        return 1 as libc::c_int;
    }
    return 0 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_extract_orientations_for_frame(
    mut self_0: *mut VlCovDet,
    mut numOrientations: *mut vl_size,
    mut frame: VlFrameOrientedEllipse,
) -> *mut VlCovDetFeatureOrientation {
    let mut err: libc::c_int = 0;
    let mut k: vl_index = 0;
    let mut i: vl_index = 0;
    let mut iter: vl_index = 0;
    let mut extent: libc::c_double = (3 as libc::c_int * 3 as libc::c_int)
        as libc::c_double;
    let mut resolution: vl_size = 20 as libc::c_int as vl_size;
    let mut side: vl_size = (2 as libc::c_int as libc::c_ulonglong)
        .wrapping_mul(resolution)
        .wrapping_add(1 as libc::c_int as libc::c_ulonglong);
    let numBins: vl_size = 36 as libc::c_int as vl_size;
    let mut hist: [libc::c_double; 36] = [0.; 36];
    let binExtent: libc::c_double = 2 as libc::c_int as libc::c_double
        * 3.141592653589793f64 / 36 as libc::c_int as libc::c_double;
    let peakRelativeSize: libc::c_double = 0.8f64;
    let mut maxPeakValue: libc::c_double = 0.;
    let mut A: [libc::c_double; 4] = [
        frame.a11 as libc::c_double,
        frame.a21 as libc::c_double,
        frame.a12 as libc::c_double,
        frame.a22 as libc::c_double,
    ];
    let mut T: [libc::c_double; 2] = [
        frame.x as libc::c_double,
        frame.y as libc::c_double,
    ];
    let mut U: [libc::c_double; 4] = [0.; 4];
    let mut V: [libc::c_double; 4] = [0.; 4];
    let mut D: [libc::c_double; 4] = [0.; 4];
    let mut sigma1: libc::c_double = 0.;
    let mut sigma2: libc::c_double = 0.;
    let mut sigmaD: libc::c_double = 1.0f64;
    let mut theta0: libc::c_double = 0.;
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/covdet.c\0" as *const u8 as *const libc::c_char,
            2737 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 116],
                &[libc::c_char; 116],
            >(
                b"VlCovDetFeatureOrientation *vl_covdet_extract_orientations_for_frame(VlCovDet *, vl_size *, VlFrameOrientedEllipse)\0",
            ))
                .as_ptr(),
        );
    }
    if !numOrientations.is_null() {} else {
        __assert_fail(
            b"numOrientations\0" as *const u8 as *const libc::c_char,
            b"vl/covdet.c\0" as *const u8 as *const libc::c_char,
            2738 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 116],
                &[libc::c_char; 116],
            >(
                b"VlCovDetFeatureOrientation *vl_covdet_extract_orientations_for_frame(VlCovDet *, vl_size *, VlFrameOrientedEllipse)\0",
            ))
                .as_ptr(),
        );
    }
    vl_svd2(D.as_mut_ptr(), U.as_mut_ptr(), V.as_mut_ptr(), A.as_mut_ptr());
    A[0 as libc::c_int
        as usize] = U[0 as libc::c_int as usize] * D[0 as libc::c_int as usize];
    A[1 as libc::c_int
        as usize] = U[1 as libc::c_int as usize] * D[0 as libc::c_int as usize];
    A[2 as libc::c_int
        as usize] = U[2 as libc::c_int as usize] * D[3 as libc::c_int as usize];
    A[3 as libc::c_int
        as usize] = U[3 as libc::c_int as usize] * D[3 as libc::c_int as usize];
    theta0 = atan2(V[1 as libc::c_int as usize], V[0 as libc::c_int as usize]);
    err = vl_covdet_extract_patch_helper(
        self_0,
        &mut sigma1,
        &mut sigma2,
        ((*self_0).aaPatch).as_mut_ptr(),
        resolution,
        extent,
        sigmaD,
        A.as_mut_ptr(),
        T.as_mut_ptr(),
        D[0 as libc::c_int as usize],
        D[3 as libc::c_int as usize],
    );
    if err != 0 {
        *numOrientations = 0 as libc::c_int as vl_size;
        return 0 as *mut VlCovDetFeatureOrientation;
    }
    let mut deltaSigma1: libc::c_double = sqrt(
        if sigmaD * sigmaD - sigma1 * sigma1 > 0 as libc::c_int as libc::c_double {
            sigmaD * sigmaD - sigma1 * sigma1
        } else {
            0 as libc::c_int as libc::c_double
        },
    );
    let mut deltaSigma2: libc::c_double = sqrt(
        if sigmaD * sigmaD - sigma2 * sigma2 > 0 as libc::c_int as libc::c_double {
            sigmaD * sigmaD - sigma2 * sigma2
        } else {
            0 as libc::c_int as libc::c_double
        },
    );
    let mut stephat: libc::c_double = extent / resolution as libc::c_double;
    vl_imsmooth_f(
        ((*self_0).aaPatch).as_mut_ptr(),
        side,
        ((*self_0).aaPatch).as_mut_ptr(),
        side,
        side,
        side,
        deltaSigma1 / stephat,
        deltaSigma2 / stephat,
    );
    vl_imgradient_polar_f(
        ((*self_0).aaPatchX).as_mut_ptr(),
        ((*self_0).aaPatchY).as_mut_ptr(),
        1 as libc::c_int as vl_size,
        side,
        ((*self_0).aaPatch).as_mut_ptr(),
        side,
        side,
        side,
    );
    memset(
        hist.as_mut_ptr() as *mut libc::c_void,
        0 as libc::c_int,
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numBins) as libc::c_ulong,
    );
    k = 0 as libc::c_int as vl_index;
    while k < side.wrapping_mul(side) as libc::c_int as libc::c_longlong {
        let mut modulus: libc::c_double = (*self_0).aaPatchX[k as usize]
            as libc::c_double;
        let mut angle: libc::c_double = (*self_0).aaPatchY[k as usize] as libc::c_double;
        let mut weight: libc::c_double = (*self_0).aaMask[k as usize] as libc::c_double;
        let mut x: libc::c_double = angle / binExtent;
        let mut bin: vl_index = vl_floor_d(x) as vl_index;
        let mut w2: libc::c_double = x - bin as libc::c_double;
        let mut w1: libc::c_double = 1.0f64 - w2;
        hist[(bin as libc::c_ulonglong).wrapping_add(numBins).wrapping_rem(numBins)
            as usize] += w1 * (modulus * weight);
        hist[(bin as libc::c_ulonglong)
            .wrapping_add(numBins)
            .wrapping_add(1 as libc::c_int as libc::c_ulonglong)
            .wrapping_rem(numBins) as usize] += w2 * (modulus * weight);
        k += 1;
    }
    iter = 0 as libc::c_int as vl_index;
    while iter < 6 as libc::c_int as libc::c_longlong {
        let mut prev: libc::c_double = hist[numBins
            .wrapping_sub(1 as libc::c_int as libc::c_ulonglong) as usize];
        let mut first: libc::c_double = hist[0 as libc::c_int as usize];
        let mut i_0: vl_index = 0;
        i_0 = 0 as libc::c_int as vl_index;
        while i_0 < (numBins as libc::c_int - 1 as libc::c_int) as libc::c_longlong {
            let mut curr: libc::c_double = (prev + hist[i_0 as usize]
                + hist[((i_0 + 1 as libc::c_int as libc::c_longlong)
                    as libc::c_ulonglong)
                    .wrapping_rem(numBins) as usize]) / 3.0f64;
            prev = hist[i_0 as usize];
            hist[i_0 as usize] = curr;
            i_0 += 1;
        }
        hist[i_0 as usize] = (prev + hist[i_0 as usize] + first) / 3.0f64;
        iter += 1;
    }
    maxPeakValue = 0 as libc::c_int as libc::c_double;
    i = 0 as libc::c_int as vl_index;
    while i < numBins as libc::c_int as libc::c_longlong {
        maxPeakValue = if maxPeakValue > hist[i as usize] {
            maxPeakValue
        } else {
            hist[i as usize]
        };
        i += 1;
    }
    *numOrientations = 0 as libc::c_int as vl_size;
    i = 0 as libc::c_int as vl_index;
    while i < numBins as libc::c_int as libc::c_longlong {
        let mut h0: libc::c_double = hist[i as usize];
        let mut hm: libc::c_double = hist[((i - 1 as libc::c_int as libc::c_longlong)
            as libc::c_ulonglong)
            .wrapping_add(numBins)
            .wrapping_rem(numBins) as usize];
        let mut hp: libc::c_double = hist[((i + 1 as libc::c_int as libc::c_longlong)
            as libc::c_ulonglong)
            .wrapping_add(numBins)
            .wrapping_rem(numBins) as usize];
        if h0 > peakRelativeSize * maxPeakValue && h0 > hm && h0 > hp {
            let mut di: libc::c_double = -0.5f64 * (hp - hm)
                / (hp + hm - 2 as libc::c_int as libc::c_double * h0);
            let mut th: libc::c_double = binExtent * (i as libc::c_double + di) + theta0;
            if (*self_0).transposed != 0 {
                th = th - 3.141592653589793f64 / 2 as libc::c_int as libc::c_double;
            }
            (*self_0).orientations[*numOrientations as usize].angle = th;
            (*self_0).orientations[*numOrientations as usize].score = h0;
            *numOrientations = (*numOrientations as libc::c_ulonglong)
                .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as vl_size
                as vl_size;
            if *numOrientations >= (*self_0).maxNumOrientations {
                break;
            }
        }
        i += 1;
    }
    qsort(
        ((*self_0).orientations).as_mut_ptr() as *mut libc::c_void,
        *numOrientations as size_t,
        ::core::mem::size_of::<VlCovDetFeatureOrientation>() as libc::c_ulong,
        Some(
            _vl_covdet_compare_orientations_descending
                as unsafe extern "C" fn(
                    *const libc::c_void,
                    *const libc::c_void,
                ) -> libc::c_int,
        ),
    );
    return ((*self_0).orientations).as_mut_ptr();
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_extract_orientations(mut self_0: *mut VlCovDet) {
    let mut i: vl_index = 0;
    let mut j: vl_index = 0;
    let mut numFeatures: vl_size = vl_covdet_get_num_features(self_0);
    i = 0 as libc::c_int as vl_index;
    while i < numFeatures as libc::c_int as libc::c_longlong {
        let mut numOrientations: vl_size = 0;
        let mut feature: VlCovDetFeature = *((*self_0).features).offset(i as isize);
        let mut orientations: *mut VlCovDetFeatureOrientation = vl_covdet_extract_orientations_for_frame(
            self_0,
            &mut numOrientations,
            feature.frame,
        );
        j = 0 as libc::c_int as vl_index;
        while j < numOrientations as libc::c_int as libc::c_longlong {
            let mut A: [libc::c_double; 4] = [
                feature.frame.a11 as libc::c_double,
                feature.frame.a21 as libc::c_double,
                feature.frame.a12 as libc::c_double,
                feature.frame.a22 as libc::c_double,
            ];
            let mut r1: libc::c_double = cos((*orientations.offset(j as isize)).angle);
            let mut r2: libc::c_double = sin((*orientations.offset(j as isize)).angle);
            let mut oriented: *mut VlCovDetFeature = 0 as *mut VlCovDetFeature;
            if j == 0 as libc::c_int as libc::c_longlong {
                oriented = &mut *((*self_0).features).offset(i as isize)
                    as *mut VlCovDetFeature;
            } else {
                vl_covdet_append_feature(self_0, &mut feature);
                oriented = &mut *((*self_0).features)
                    .offset(
                        ((*self_0).numFeatures)
                            .wrapping_sub(1 as libc::c_int as libc::c_ulonglong) as isize,
                    ) as *mut VlCovDetFeature;
            }
            (*oriented)
                .orientationScore = (*orientations.offset(j as isize)).score
                as libc::c_float;
            (*oriented)
                .frame
                .a11 = (A[0 as libc::c_int as usize] * r1
                + A[2 as libc::c_int as usize] * r2) as libc::c_float;
            (*oriented)
                .frame
                .a21 = (A[1 as libc::c_int as usize] * r1
                + A[3 as libc::c_int as usize] * r2) as libc::c_float;
            (*oriented)
                .frame
                .a12 = (-A[0 as libc::c_int as usize] * r2
                + A[2 as libc::c_int as usize] * r1) as libc::c_float;
            (*oriented)
                .frame
                .a22 = (-A[1 as libc::c_int as usize] * r2
                + A[3 as libc::c_int as usize] * r1) as libc::c_float;
            j += 1;
        }
        i += 1;
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_extract_laplacian_scales_for_frame(
    mut self_0: *mut VlCovDet,
    mut numScales: *mut vl_size,
    mut frame: VlFrameOrientedEllipse,
) -> *mut VlCovDetFeatureLaplacianScale {
    let mut err: libc::c_int = 0;
    let sigmaImage: libc::c_double = 1.0f64 / sqrt(2.0f64);
    let step: libc::c_double = 0.5f64 * sigmaImage;
    let mut actualSigmaImage: libc::c_double = 0.;
    let resolution: vl_size = 16 as libc::c_int as vl_size;
    let num: vl_size = (2 as libc::c_int as libc::c_ulonglong)
        .wrapping_mul(resolution)
        .wrapping_add(1 as libc::c_int as libc::c_ulonglong);
    let mut extent: libc::c_double = step * resolution as libc::c_double;
    let mut scores: [libc::c_double; 10] = [0.; 10];
    let mut factor: libc::c_double = 1.0f64;
    let mut pt: *const libc::c_float = 0 as *const libc::c_float;
    let mut k: vl_index = 0;
    let mut A: [libc::c_double; 4] = [
        frame.a11 as libc::c_double,
        frame.a21 as libc::c_double,
        frame.a12 as libc::c_double,
        frame.a22 as libc::c_double,
    ];
    let mut T: [libc::c_double; 2] = [
        frame.x as libc::c_double,
        frame.y as libc::c_double,
    ];
    let mut D: [libc::c_double; 4] = [0.; 4];
    let mut U: [libc::c_double; 4] = [0.; 4];
    let mut V: [libc::c_double; 4] = [0.; 4];
    let mut sigma1: libc::c_double = 0.;
    let mut sigma2: libc::c_double = 0.;
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/covdet.c\0" as *const u8 as *const libc::c_char,
            2954 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 123],
                &[libc::c_char; 123],
            >(
                b"VlCovDetFeatureLaplacianScale *vl_covdet_extract_laplacian_scales_for_frame(VlCovDet *, vl_size *, VlFrameOrientedEllipse)\0",
            ))
                .as_ptr(),
        );
    }
    if !numScales.is_null() {} else {
        __assert_fail(
            b"numScales\0" as *const u8 as *const libc::c_char,
            b"vl/covdet.c\0" as *const u8 as *const libc::c_char,
            2955 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 123],
                &[libc::c_char; 123],
            >(
                b"VlCovDetFeatureLaplacianScale *vl_covdet_extract_laplacian_scales_for_frame(VlCovDet *, vl_size *, VlFrameOrientedEllipse)\0",
            ))
                .as_ptr(),
        );
    }
    *numScales = 0 as libc::c_int as vl_size;
    vl_svd2(D.as_mut_ptr(), U.as_mut_ptr(), V.as_mut_ptr(), A.as_mut_ptr());
    err = vl_covdet_extract_patch_helper(
        self_0,
        &mut sigma1,
        &mut sigma2,
        ((*self_0).lapPatch).as_mut_ptr(),
        resolution,
        extent,
        sigmaImage,
        A.as_mut_ptr(),
        T.as_mut_ptr(),
        D[0 as libc::c_int as usize],
        D[3 as libc::c_int as usize],
    );
    if err != 0 {
        return 0 as *mut VlCovDetFeatureLaplacianScale;
    }
    if sigma1 == sigma2 {
        actualSigmaImage = sigma1;
    } else {
        actualSigmaImage = sqrt(sigma1 * sigma2);
    }
    pt = ((*self_0).laplacians).as_mut_ptr();
    k = 0 as libc::c_int as vl_index;
    while k < 10 as libc::c_int as libc::c_longlong {
        let mut q: vl_index = 0;
        let mut score: libc::c_double = 0 as libc::c_int as libc::c_double;
        let mut sigmaLap: libc::c_double = pow(
            2.0f64,
            -0.5f64
                + k as libc::c_double
                    / (10 as libc::c_int - 1 as libc::c_int) as libc::c_double,
        );
        sigmaLap = sqrt(
            sigmaLap * sigmaLap - sigmaImage * sigmaImage
                + actualSigmaImage * actualSigmaImage,
        );
        q = 0 as libc::c_int as vl_index;
        while q < num.wrapping_mul(num) as libc::c_int as libc::c_longlong {
            let fresh6 = pt;
            pt = pt.offset(1);
            score += (*fresh6 * (*self_0).lapPatch[q as usize]) as libc::c_double;
            q += 1;
        }
        scores[k as usize] = score * sigmaLap * sigmaLap;
        k += 1;
    }
    k = 1 as libc::c_int as vl_index;
    while k < (10 as libc::c_int - 1 as libc::c_int) as libc::c_longlong {
        let mut a: libc::c_double = scores[(k - 1 as libc::c_int as libc::c_longlong)
            as usize];
        let mut b: libc::c_double = scores[k as usize];
        let mut c: libc::c_double = scores[(k + 1 as libc::c_int as libc::c_longlong)
            as usize];
        let mut t: libc::c_double = (*self_0).lapPeakThreshold;
        if (b > a && b > c || b < a && b < c) && vl_abs_d(b) >= t {
            let mut dk: libc::c_double = -0.5f64 * (c - a)
                / (c + a - 2 as libc::c_int as libc::c_double * b);
            let mut s: libc::c_double = k as libc::c_double + dk;
            let mut sigmaLap_0: libc::c_double = pow(
                2.0f64,
                -0.5f64 + s / (10 as libc::c_int - 1 as libc::c_int) as libc::c_double,
            );
            let mut scale: libc::c_double = 0.;
            sigmaLap_0 = sqrt(
                sigmaLap_0 * sigmaLap_0 - sigmaImage * sigmaImage
                    + actualSigmaImage * actualSigmaImage,
            );
            scale = sigmaLap_0 / 1.0f64;
            if *numScales < 4 as libc::c_int as libc::c_ulonglong {
                (*self_0).scales[*numScales as usize].scale = scale * factor;
                (*self_0).scales[*numScales as usize].score = b + 0.5f64 * (c - a) * dk;
                *numScales = (*numScales as libc::c_ulonglong)
                    .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as vl_size
                    as vl_size;
            }
        }
        k += 1;
    }
    return ((*self_0).scales).as_mut_ptr();
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_extract_laplacian_scales(mut self_0: *mut VlCovDet) {
    let mut i: vl_index = 0;
    let mut j: vl_index = 0;
    let mut dropFeaturesWithoutScale: vl_bool = 1 as libc::c_int;
    let mut numFeatures: vl_size = vl_covdet_get_num_features(self_0);
    memset(
        ((*self_0).numFeaturesWithNumScales).as_mut_ptr() as *mut libc::c_void,
        0 as libc::c_int,
        ::core::mem::size_of::<[vl_size; 5]>() as libc::c_ulong,
    );
    i = 0 as libc::c_int as vl_index;
    while i < numFeatures as libc::c_int as libc::c_longlong {
        let mut numScales: vl_size = 0;
        let mut feature: VlCovDetFeature = *((*self_0).features).offset(i as isize);
        let mut scales: *const VlCovDetFeatureLaplacianScale = vl_covdet_extract_laplacian_scales_for_frame(
            self_0,
            &mut numScales,
            feature.frame,
        );
        (*self_0)
            .numFeaturesWithNumScales[numScales
            as usize] = ((*self_0).numFeaturesWithNumScales[numScales as usize])
            .wrapping_add(1);
        if numScales == 0 as libc::c_int as libc::c_ulonglong
            && dropFeaturesWithoutScale != 0
        {
            (*((*self_0).features).offset(i as isize))
                .peakScore = 0 as libc::c_int as libc::c_float;
        }
        j = 0 as libc::c_int as vl_index;
        while j < numScales as libc::c_int as libc::c_longlong {
            let mut scaled: *mut VlCovDetFeature = 0 as *mut VlCovDetFeature;
            if j == 0 as libc::c_int as libc::c_longlong {
                scaled = &mut *((*self_0).features).offset(i as isize)
                    as *mut VlCovDetFeature;
            } else {
                vl_covdet_append_feature(self_0, &mut feature);
                scaled = &mut *((*self_0).features)
                    .offset(
                        ((*self_0).numFeatures)
                            .wrapping_sub(1 as libc::c_int as libc::c_ulonglong) as isize,
                    ) as *mut VlCovDetFeature;
            }
            (*scaled)
                .laplacianScaleScore = (*scales.offset(j as isize)).score
                as libc::c_float;
            (*scaled)
                .frame
                .a11 = ((*scaled).frame.a11 as libc::c_double
                * (*scales.offset(j as isize)).scale) as libc::c_float;
            (*scaled)
                .frame
                .a21 = ((*scaled).frame.a21 as libc::c_double
                * (*scales.offset(j as isize)).scale) as libc::c_float;
            (*scaled)
                .frame
                .a12 = ((*scaled).frame.a12 as libc::c_double
                * (*scales.offset(j as isize)).scale) as libc::c_float;
            (*scaled)
                .frame
                .a22 = ((*scaled).frame.a22 as libc::c_double
                * (*scales.offset(j as isize)).scale) as libc::c_float;
            j += 1;
        }
        i += 1;
    }
    if dropFeaturesWithoutScale != 0 {
        j = 0 as libc::c_int as vl_index;
        i = 0 as libc::c_int as vl_index;
        while i < (*self_0).numFeatures as libc::c_int as libc::c_longlong {
            let mut feature_0: VlCovDetFeature = *((*self_0).features)
                .offset(i as isize);
            if feature_0.peakScore != 0. {
                let fresh7 = j;
                j = j + 1;
                *((*self_0).features).offset(fresh7 as isize) = feature_0;
            }
            i += 1;
        }
        (*self_0).numFeatures = j as vl_size;
    }
}
#[no_mangle]
pub unsafe extern "C" fn _vl_covdet_check_frame_inside(
    mut self_0: *mut VlCovDet,
    mut frame: VlFrameOrientedEllipse,
    mut margin: libc::c_double,
) -> vl_bool {
    let mut extent: libc::c_double = margin;
    let mut A: [libc::c_double; 4] = [
        frame.a11 as libc::c_double,
        frame.a21 as libc::c_double,
        frame.a12 as libc::c_double,
        frame.a22 as libc::c_double,
    ];
    let mut T: [libc::c_double; 2] = [
        frame.x as libc::c_double,
        frame.y as libc::c_double,
    ];
    let mut x0: libc::c_double = vl_infinity_d.value;
    let mut x1: libc::c_double = -vl_infinity_d.value;
    let mut y0: libc::c_double = vl_infinity_d.value;
    let mut y1: libc::c_double = -vl_infinity_d.value;
    let mut boxx: [libc::c_double; 4] = [extent, extent, -extent, -extent];
    let mut boxy: [libc::c_double; 4] = [-extent, extent, extent, -extent];
    let mut geom: VlScaleSpaceGeometry = vl_scalespace_get_geometry((*self_0).gss);
    let mut i: libc::c_int = 0;
    i = 0 as libc::c_int;
    while i < 4 as libc::c_int {
        let mut x: libc::c_double = A[0 as libc::c_int as usize] * boxx[i as usize]
            + A[2 as libc::c_int as usize] * boxy[i as usize]
            + T[0 as libc::c_int as usize];
        let mut y: libc::c_double = A[1 as libc::c_int as usize] * boxx[i as usize]
            + A[3 as libc::c_int as usize] * boxy[i as usize]
            + T[1 as libc::c_int as usize];
        x0 = if x0 < x { x0 } else { x };
        x1 = if x1 > x { x1 } else { x };
        y0 = if y0 < y { y0 } else { y };
        y1 = if y1 > y { y1 } else { y };
        i += 1;
    }
    return (0 as libc::c_int as libc::c_double <= x0
        && x1
            <= (geom.width).wrapping_sub(1 as libc::c_int as libc::c_ulonglong)
                as libc::c_double && 0 as libc::c_int as libc::c_double <= y0
        && y1
            <= (geom.height).wrapping_sub(1 as libc::c_int as libc::c_ulonglong)
                as libc::c_double) as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_drop_features_outside(
    mut self_0: *mut VlCovDet,
    mut margin: libc::c_double,
) {
    let mut i: vl_index = 0;
    let mut j: vl_index = 0 as libc::c_int as vl_index;
    let mut numFeatures: vl_size = vl_covdet_get_num_features(self_0);
    i = 0 as libc::c_int as vl_index;
    while i < numFeatures as libc::c_int as libc::c_longlong {
        let mut inside: vl_bool = _vl_covdet_check_frame_inside(
            self_0,
            (*((*self_0).features).offset(i as isize)).frame,
            margin,
        );
        if inside != 0 {
            *((*self_0).features)
                .offset(j as isize) = *((*self_0).features).offset(i as isize);
            j += 1;
        }
        i += 1;
    }
    (*self_0).numFeatures = j as vl_size;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_get_transposed(
    mut self_0: *const VlCovDet,
) -> vl_bool {
    return (*self_0).transposed;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_set_transposed(
    mut self_0: *mut VlCovDet,
    mut t: vl_bool,
) {
    (*self_0).transposed = t;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_get_edge_threshold(
    mut self_0: *const VlCovDet,
) -> libc::c_double {
    return (*self_0).edgeThreshold;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_set_edge_threshold(
    mut self_0: *mut VlCovDet,
    mut edgeThreshold: libc::c_double,
) {
    if edgeThreshold >= 0 as libc::c_int as libc::c_double {} else {
        __assert_fail(
            b"edgeThreshold >= 0\0" as *const u8 as *const libc::c_char,
            b"vl/covdet.c\0" as *const u8 as *const libc::c_char,
            3188 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 54],
                &[libc::c_char; 54],
            >(b"void vl_covdet_set_edge_threshold(VlCovDet *, double)\0"))
                .as_ptr(),
        );
    }
    (*self_0).edgeThreshold = edgeThreshold;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_get_peak_threshold(
    mut self_0: *const VlCovDet,
) -> libc::c_double {
    return (*self_0).peakThreshold;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_set_peak_threshold(
    mut self_0: *mut VlCovDet,
    mut peakThreshold: libc::c_double,
) {
    if peakThreshold >= 0 as libc::c_int as libc::c_double {} else {
        __assert_fail(
            b"peakThreshold >= 0\0" as *const u8 as *const libc::c_char,
            b"vl/covdet.c\0" as *const u8 as *const libc::c_char,
            3212 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 54],
                &[libc::c_char; 54],
            >(b"void vl_covdet_set_peak_threshold(VlCovDet *, double)\0"))
                .as_ptr(),
        );
    }
    (*self_0).peakThreshold = peakThreshold;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_get_laplacian_peak_threshold(
    mut self_0: *const VlCovDet,
) -> libc::c_double {
    return (*self_0).lapPeakThreshold;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_set_laplacian_peak_threshold(
    mut self_0: *mut VlCovDet,
    mut peakThreshold: libc::c_double,
) {
    if peakThreshold >= 0 as libc::c_int as libc::c_double {} else {
        __assert_fail(
            b"peakThreshold >= 0\0" as *const u8 as *const libc::c_char,
            b"vl/covdet.c\0" as *const u8 as *const libc::c_char,
            3239 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 64],
                &[libc::c_char; 64],
            >(b"void vl_covdet_set_laplacian_peak_threshold(VlCovDet *, double)\0"))
                .as_ptr(),
        );
    }
    (*self_0).lapPeakThreshold = peakThreshold;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_get_first_octave(
    mut self_0: *const VlCovDet,
) -> vl_index {
    return (*self_0).firstOctave;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_set_first_octave(
    mut self_0: *mut VlCovDet,
    mut o: vl_index,
) {
    (*self_0).firstOctave = o;
    vl_covdet_reset(self_0);
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_get_num_octaves(
    mut self_0: *const VlCovDet,
) -> vl_size {
    return (*self_0).numOctaves as vl_size;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_get_base_scale(
    mut self_0: *const VlCovDet,
) -> libc::c_double {
    return (*self_0).baseScale;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_set_num_octaves(
    mut self_0: *mut VlCovDet,
    mut o: vl_size,
) {
    (*self_0).numOctaves = o as vl_index;
    vl_covdet_reset(self_0);
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_set_base_scale(
    mut self_0: *mut VlCovDet,
    mut s: libc::c_double,
) {
    (*self_0).baseScale = s;
    vl_covdet_reset(self_0);
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_set_max_num_orientations(
    mut self_0: *mut VlCovDet,
    mut m: vl_size,
) {
    (*self_0).maxNumOrientations = m;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_get_octave_resolution(
    mut self_0: *const VlCovDet,
) -> vl_size {
    return (*self_0).octaveResolution;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_set_octave_resolution(
    mut self_0: *mut VlCovDet,
    mut r: vl_size,
) {
    (*self_0).octaveResolution = r;
    vl_covdet_reset(self_0);
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_get_aa_accurate_smoothing(
    mut self_0: *const VlCovDet,
) -> vl_bool {
    return (*self_0).aaAccurateSmoothing;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_set_aa_accurate_smoothing(
    mut self_0: *mut VlCovDet,
    mut x: vl_bool,
) {
    (*self_0).aaAccurateSmoothing = x;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_get_max_num_orientations(
    mut self_0: *const VlCovDet,
) -> vl_size {
    return (*self_0).maxNumOrientations;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_get_non_extrema_suppression_threshold(
    mut self_0: *const VlCovDet,
) -> libc::c_double {
    return (*self_0).nonExtremaSuppression;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_set_non_extrema_suppression_threshold(
    mut self_0: *mut VlCovDet,
    mut x: libc::c_double,
) {
    (*self_0).nonExtremaSuppression = x;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_get_num_non_extrema_suppressed(
    mut self_0: *const VlCovDet,
) -> vl_size {
    return (*self_0).numNonExtremaSuppressed;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_get_num_features(
    mut self_0: *const VlCovDet,
) -> vl_size {
    return (*self_0).numFeatures;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_get_features(
    mut self_0: *mut VlCovDet,
) -> *mut libc::c_void {
    return (*self_0).features as *mut libc::c_void;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_get_gss(
    mut self_0: *const VlCovDet,
) -> *mut VlScaleSpace {
    return (*self_0).gss;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_get_css(
    mut self_0: *const VlCovDet,
) -> *mut VlScaleSpace {
    return (*self_0).css;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_get_laplacian_scales_statistics(
    mut self_0: *const VlCovDet,
    mut numScales: *mut vl_size,
) -> *const vl_size {
    *numScales = 4 as libc::c_int as vl_size;
    return ((*self_0).numFeaturesWithNumScales).as_ptr();
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_get_allow_padded_warping(
    mut self_0: *const VlCovDet,
) -> vl_bool {
    return (*self_0).allowPaddedWarping;
}
#[no_mangle]
pub unsafe extern "C" fn vl_covdet_set_allow_padded_warping(
    mut self_0: *mut VlCovDet,
    mut t: vl_bool,
) {
    (*self_0).allowPaddedWarping = t;
}
