use ::libc;
extern "C" {
    fn vl_free(ptr: *mut libc::c_void);
    fn vl_malloc(n: size_t) -> *mut libc::c_void;
    fn expf(_: libc::c_float) -> libc::c_float;
    fn fabsf(_: libc::c_float) -> libc::c_float;
    fn vl_imconvcol_vf(
        dst: *mut libc::c_float,
        dst_stride: vl_size,
        src: *const libc::c_float,
        src_width: vl_size,
        src_height: vl_size,
        src_stride: vl_size,
        filt: *const libc::c_float,
        filt_begin: vl_index,
        filt_end: vl_index,
        step: libc::c_int,
        flags: libc::c_uint,
    );
    fn vl_imconvcoltri_f(
        dest: *mut libc::c_float,
        destStride: vl_size,
        image: *const libc::c_float,
        imageWidth: vl_size,
        imageHeight: vl_size,
        imageStride: vl_size,
        filterSize: vl_size,
        step: vl_size,
        flags: libc::c_uint,
    );
    fn memset(
        _: *mut libc::c_void,
        _: libc::c_int,
        _: libc::c_ulong,
    ) -> *mut libc::c_void;
}
pub type vl_int64 = libc::c_longlong;
pub type vl_int32 = libc::c_int;
pub type vl_uint64 = libc::c_ulonglong;
pub type vl_size = vl_uint64;
pub type vl_index = vl_int64;
pub type size_t = libc::c_ulong;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct VlDsiftKeypoint_ {
    pub x: libc::c_double,
    pub y: libc::c_double,
    pub s: libc::c_double,
    pub norm: libc::c_double,
}
pub type VlDsiftKeypoint = VlDsiftKeypoint_;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct VlDsiftDescriptorGeometry_ {
    pub numBinT: libc::c_int,
    pub numBinX: libc::c_int,
    pub numBinY: libc::c_int,
    pub binSizeX: libc::c_int,
    pub binSizeY: libc::c_int,
}
pub type VlDsiftDescriptorGeometry = VlDsiftDescriptorGeometry_;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct VlDsiftFilter_ {
    pub imWidth: libc::c_int,
    pub imHeight: libc::c_int,
    pub stepX: libc::c_int,
    pub stepY: libc::c_int,
    pub boundMinX: libc::c_int,
    pub boundMinY: libc::c_int,
    pub boundMaxX: libc::c_int,
    pub boundMaxY: libc::c_int,
    pub geom: VlDsiftDescriptorGeometry,
    pub useFlatWindow: libc::c_int,
    pub windowSize: libc::c_double,
    pub numFrames: libc::c_int,
    pub descrSize: libc::c_int,
    pub frames: *mut VlDsiftKeypoint,
    pub descrs: *mut libc::c_float,
    pub numBinAlloc: libc::c_int,
    pub numFrameAlloc: libc::c_int,
    pub numGradAlloc: libc::c_int,
    pub grads: *mut *mut libc::c_float,
    pub convTmp1: *mut libc::c_float,
    pub convTmp2: *mut libc::c_float,
}
pub type VlDsiftFilter = VlDsiftFilter_;
#[derive(Copy, Clone)]
#[repr(C)]
pub union C2RustUnnamed {
    pub x: libc::c_float,
    pub i: vl_int32,
}
unsafe extern "C" fn vl_dsift_get_keypoint_num(
    mut self_0: *const VlDsiftFilter,
) -> libc::c_int {
    return (*self_0).numFrames;
}
unsafe extern "C" fn vl_dsift_get_descriptor_size(
    mut self_0: *const VlDsiftFilter,
) -> libc::c_int {
    return (*self_0).descrSize;
}
unsafe extern "C" fn vl_dsift_set_geometry(
    mut self_0: *mut VlDsiftFilter,
    mut geom: *const VlDsiftDescriptorGeometry,
) {
    (*self_0).geom = *geom;
    _vl_dsift_update_buffers(self_0);
}
unsafe extern "C" fn vl_dsift_get_geometry(
    mut self_0: *const VlDsiftFilter,
) -> *const VlDsiftDescriptorGeometry {
    return &(*self_0).geom;
}
unsafe extern "C" fn vl_dsift_set_steps(
    mut self_0: *mut VlDsiftFilter,
    mut stepX: libc::c_int,
    mut stepY: libc::c_int,
) {
    (*self_0).stepX = stepX;
    (*self_0).stepY = stepY;
    _vl_dsift_update_buffers(self_0);
}
#[inline]
unsafe extern "C" fn vl_floor_f(mut x: libc::c_float) -> libc::c_long {
    let mut xi: libc::c_long = x as libc::c_long;
    if x >= 0 as libc::c_int as libc::c_float || xi as libc::c_float == x {
        return xi
    } else {
        return xi - 1 as libc::c_int as libc::c_long
    };
}
#[inline]
unsafe extern "C" fn vl_mod_2pi_f(mut x: libc::c_float) -> libc::c_float {
    while x
        > (2 as libc::c_int as libc::c_double * 3.141592653589793f64) as libc::c_float
    {
        x
            -= (2 as libc::c_int as libc::c_double * 3.141592653589793f64)
                as libc::c_float;
    }
    while x < 0.0f32 {
        x
            += (2 as libc::c_int as libc::c_double * 3.141592653589793f64)
                as libc::c_float;
    }
    return x;
}
#[inline]
unsafe extern "C" fn vl_fast_atan2_f(
    mut y: libc::c_float,
    mut x: libc::c_float,
) -> libc::c_float {
    let mut angle: libc::c_float = 0.;
    let mut r: libc::c_float = 0.;
    let c3: libc::c_float = 0.1821f32;
    let c1: libc::c_float = 0.9675f32;
    let mut abs_y: libc::c_float = vl_abs_f(y) + 1.19209290E-07f32;
    if x >= 0 as libc::c_int as libc::c_float {
        r = (x - abs_y) / (x + abs_y);
        angle = (3.141592653589793f64 / 4 as libc::c_int as libc::c_double)
            as libc::c_float;
    } else {
        r = (x + abs_y) / (abs_y - x);
        angle = (3 as libc::c_int as libc::c_double * 3.141592653589793f64
            / 4 as libc::c_int as libc::c_double) as libc::c_float;
    }
    angle += (c3 * r * r - c1) * r;
    return if y < 0 as libc::c_int as libc::c_float { -angle } else { angle };
}
#[inline]
unsafe extern "C" fn vl_abs_f(mut x: libc::c_float) -> libc::c_float {
    return x.abs();
}
#[inline]
unsafe extern "C" fn vl_fast_sqrt_f(mut x: libc::c_float) -> libc::c_float {
    return if (x as libc::c_double) < 1e-8f64 {
        0 as libc::c_int as libc::c_float
    } else {
        x * vl_fast_resqrt_f(x)
    };
}
#[inline]
unsafe extern "C" fn vl_fast_resqrt_f(mut x: libc::c_float) -> libc::c_float {
    let mut u: C2RustUnnamed = C2RustUnnamed { x: 0. };
    let mut xhalf: libc::c_float = 0.5f64 as libc::c_float * x;
    u.x = x;
    u.i = 0x5f3759df as libc::c_int - (u.i >> 1 as libc::c_int);
    u.x = u.x * (1.5f64 as libc::c_float - xhalf * u.x * u.x);
    u.x = u.x * (1.5f64 as libc::c_float - xhalf * u.x * u.x);
    return u.x;
}
#[no_mangle]
pub unsafe extern "C" fn _vl_dsift_new_kernel(
    mut binSize: libc::c_int,
    mut numBins: libc::c_int,
    mut binIndex: libc::c_int,
    mut windowSize: libc::c_double,
) -> *mut libc::c_float {
    let mut filtLen: libc::c_int = 2 as libc::c_int * binSize - 1 as libc::c_int;
    let mut ker: *mut libc::c_float = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong)
            .wrapping_mul(filtLen as libc::c_ulong),
    ) as *mut libc::c_float;
    let mut kerIter: *mut libc::c_float = ker;
    let mut delta: libc::c_float = binSize as libc::c_float
        * (binIndex as libc::c_float
            - 0.5f32 * (numBins - 1 as libc::c_int) as libc::c_float);
    let mut sigma: libc::c_float = binSize as libc::c_float
        * windowSize as libc::c_float;
    let mut x: libc::c_int = 0;
    x = -binSize + 1 as libc::c_int;
    while x <= binSize - 1 as libc::c_int {
        let mut z: libc::c_float = (x as libc::c_float - delta) / sigma;
        let fresh0 = kerIter;
        kerIter = kerIter.offset(1);
        *fresh0 = (1.0f32 - fabsf(x as libc::c_float) / binSize as libc::c_float)
            * (if binIndex >= 0 as libc::c_int {
                expf(-0.5f32 * z * z)
            } else {
                1.0f32
            });
        x += 1;
    }
    return ker;
}
unsafe extern "C" fn _vl_dsift_get_bin_window_mean(
    mut binSize: libc::c_int,
    mut numBins: libc::c_int,
    mut binIndex: libc::c_int,
    mut windowSize: libc::c_double,
) -> libc::c_float {
    let mut delta: libc::c_float = binSize as libc::c_float
        * (binIndex as libc::c_float
            - 0.5f32 * (numBins - 1 as libc::c_int) as libc::c_float);
    let mut sigma: libc::c_float = binSize as libc::c_float
        * windowSize as libc::c_float;
    let mut x: libc::c_int = 0;
    let mut acc: libc::c_float = 0.0f64 as libc::c_float;
    x = -binSize + 1 as libc::c_int;
    while x <= binSize - 1 as libc::c_int {
        let mut z: libc::c_float = (x as libc::c_float - delta) / sigma;
        acc += if binIndex >= 0 as libc::c_int { expf(-0.5f32 * z * z) } else { 1.0f32 };
        x += 1;
    }
    acc /= (2 as libc::c_int * binSize - 1 as libc::c_int) as libc::c_float;
    return acc;
}
#[inline]
unsafe extern "C" fn _vl_dsift_normalize_histogram(
    mut begin: *mut libc::c_float,
    mut end: *mut libc::c_float,
) -> libc::c_float {
    let mut iter: *mut libc::c_float = 0 as *mut libc::c_float;
    let mut norm: libc::c_float = 0.0f32;
    iter = begin;
    while iter < end {
        norm += *iter * *iter;
        iter = iter.offset(1);
    }
    norm = vl_fast_sqrt_f(norm) + 1.19209290E-07f32;
    iter = begin;
    while iter < end {
        *iter /= norm;
        iter = iter.offset(1);
    }
    return norm;
}
unsafe extern "C" fn _vl_dsift_free_buffers(mut self_0: *mut VlDsiftFilter) {
    if !((*self_0).frames).is_null() {
        vl_free((*self_0).frames as *mut libc::c_void);
        (*self_0).frames = 0 as *mut VlDsiftKeypoint;
    }
    if !((*self_0).descrs).is_null() {
        vl_free((*self_0).descrs as *mut libc::c_void);
        (*self_0).descrs = 0 as *mut libc::c_float;
    }
    if !((*self_0).grads).is_null() {
        let mut t: libc::c_int = 0;
        t = 0 as libc::c_int;
        while t < (*self_0).numGradAlloc {
            if !(*((*self_0).grads).offset(t as isize)).is_null() {
                vl_free(*((*self_0).grads).offset(t as isize) as *mut libc::c_void);
            }
            t += 1;
        }
        vl_free((*self_0).grads as *mut libc::c_void);
        (*self_0).grads = 0 as *mut *mut libc::c_float;
    }
    (*self_0).numFrameAlloc = 0 as libc::c_int;
    (*self_0).numBinAlloc = 0 as libc::c_int;
    (*self_0).numGradAlloc = 0 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn _vl_dsift_update_buffers(mut self_0: *mut VlDsiftFilter) {
    let mut x1: libc::c_int = (*self_0).boundMinX;
    let mut x2: libc::c_int = (*self_0).boundMaxX;
    let mut y1: libc::c_int = (*self_0).boundMinY;
    let mut y2: libc::c_int = (*self_0).boundMaxY;
    let mut rangeX: libc::c_int = x2 - x1
        - ((*self_0).geom.numBinX - 1 as libc::c_int) * (*self_0).geom.binSizeX;
    let mut rangeY: libc::c_int = y2 - y1
        - ((*self_0).geom.numBinY - 1 as libc::c_int) * (*self_0).geom.binSizeY;
    let mut numFramesX: libc::c_int = if rangeX >= 0 as libc::c_int {
        rangeX / (*self_0).stepX + 1 as libc::c_int
    } else {
        0 as libc::c_int
    };
    let mut numFramesY: libc::c_int = if rangeY >= 0 as libc::c_int {
        rangeY / (*self_0).stepY + 1 as libc::c_int
    } else {
        0 as libc::c_int
    };
    (*self_0).numFrames = numFramesX * numFramesY;
    (*self_0)
        .descrSize = (*self_0).geom.numBinT * (*self_0).geom.numBinX
        * (*self_0).geom.numBinY;
}
unsafe extern "C" fn _vl_dsift_alloc_buffers(mut self_0: *mut VlDsiftFilter) {
    _vl_dsift_update_buffers(self_0);
    let mut numFrameAlloc: libc::c_int = vl_dsift_get_keypoint_num(self_0);
    let mut numBinAlloc: libc::c_int = vl_dsift_get_descriptor_size(self_0);
    let mut numGradAlloc: libc::c_int = (*self_0).geom.numBinT;
    if numBinAlloc != (*self_0).numBinAlloc || numGradAlloc != (*self_0).numGradAlloc
        || numFrameAlloc != (*self_0).numFrameAlloc
    {
        let mut t: libc::c_int = 0;
        _vl_dsift_free_buffers(self_0);
        (*self_0)
            .frames = vl_malloc(
            (::core::mem::size_of::<VlDsiftKeypoint>() as libc::c_ulong)
                .wrapping_mul(numFrameAlloc as libc::c_ulong),
        ) as *mut VlDsiftKeypoint;
        (*self_0)
            .descrs = vl_malloc(
            (::core::mem::size_of::<libc::c_float>() as libc::c_ulong)
                .wrapping_mul(numBinAlloc as libc::c_ulong)
                .wrapping_mul(numFrameAlloc as libc::c_ulong),
        ) as *mut libc::c_float;
        (*self_0)
            .grads = vl_malloc(
            (::core::mem::size_of::<*mut libc::c_float>() as libc::c_ulong)
                .wrapping_mul(numGradAlloc as libc::c_ulong),
        ) as *mut *mut libc::c_float;
        t = 0 as libc::c_int;
        while t < numGradAlloc {
            let ref mut fresh1 = *((*self_0).grads).offset(t as isize);
            *fresh1 = vl_malloc(
                (::core::mem::size_of::<libc::c_float>() as libc::c_ulong)
                    .wrapping_mul((*self_0).imWidth as libc::c_ulong)
                    .wrapping_mul((*self_0).imHeight as libc::c_ulong),
            ) as *mut libc::c_float;
            t += 1;
        }
        (*self_0).numBinAlloc = numBinAlloc;
        (*self_0).numGradAlloc = numGradAlloc;
        (*self_0).numFrameAlloc = numFrameAlloc;
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_dsift_new(
    mut imWidth: libc::c_int,
    mut imHeight: libc::c_int,
) -> *mut VlDsiftFilter {
    let mut self_0: *mut VlDsiftFilter = vl_malloc(
        ::core::mem::size_of::<VlDsiftFilter>() as libc::c_ulong,
    ) as *mut VlDsiftFilter;
    (*self_0).imWidth = imWidth;
    (*self_0).imHeight = imHeight;
    (*self_0).stepX = 5 as libc::c_int;
    (*self_0).stepY = 5 as libc::c_int;
    (*self_0).boundMinX = 0 as libc::c_int;
    (*self_0).boundMinY = 0 as libc::c_int;
    (*self_0).boundMaxX = imWidth - 1 as libc::c_int;
    (*self_0).boundMaxY = imHeight - 1 as libc::c_int;
    (*self_0).geom.numBinX = 4 as libc::c_int;
    (*self_0).geom.numBinY = 4 as libc::c_int;
    (*self_0).geom.numBinT = 8 as libc::c_int;
    (*self_0).geom.binSizeX = 5 as libc::c_int;
    (*self_0).geom.binSizeY = 5 as libc::c_int;
    (*self_0).useFlatWindow = 0 as libc::c_int;
    (*self_0).windowSize = 2.0f64;
    (*self_0)
        .convTmp1 = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong)
            .wrapping_mul((*self_0).imWidth as libc::c_ulong)
            .wrapping_mul((*self_0).imHeight as libc::c_ulong),
    ) as *mut libc::c_float;
    (*self_0)
        .convTmp2 = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong)
            .wrapping_mul((*self_0).imWidth as libc::c_ulong)
            .wrapping_mul((*self_0).imHeight as libc::c_ulong),
    ) as *mut libc::c_float;
    (*self_0).numBinAlloc = 0 as libc::c_int;
    (*self_0).numFrameAlloc = 0 as libc::c_int;
    (*self_0).numGradAlloc = 0 as libc::c_int;
    (*self_0).descrSize = 0 as libc::c_int;
    (*self_0).numFrames = 0 as libc::c_int;
    (*self_0).grads = 0 as *mut *mut libc::c_float;
    (*self_0).frames = 0 as *mut VlDsiftKeypoint;
    (*self_0).descrs = 0 as *mut libc::c_float;
    _vl_dsift_update_buffers(self_0);
    return self_0;
}
#[no_mangle]
pub unsafe extern "C" fn vl_dsift_new_basic(
    mut imWidth: libc::c_int,
    mut imHeight: libc::c_int,
    mut step: libc::c_int,
    mut binSize: libc::c_int,
) -> *mut VlDsiftFilter {
    let mut self_0: *mut VlDsiftFilter = vl_dsift_new(imWidth, imHeight);
    let mut geom: VlDsiftDescriptorGeometry = *vl_dsift_get_geometry(self_0);
    geom.binSizeX = binSize;
    geom.binSizeY = binSize;
    vl_dsift_set_geometry(self_0, &mut geom);
    vl_dsift_set_steps(self_0, step, step);
    return self_0;
}
#[no_mangle]
pub unsafe extern "C" fn vl_dsift_delete(mut self_0: *mut VlDsiftFilter) {
    _vl_dsift_free_buffers(self_0);
    if !((*self_0).convTmp2).is_null() {
        vl_free((*self_0).convTmp2 as *mut libc::c_void);
    }
    if !((*self_0).convTmp1).is_null() {
        vl_free((*self_0).convTmp1 as *mut libc::c_void);
    }
    vl_free(self_0 as *mut libc::c_void);
}
#[inline]
unsafe extern "C" fn _vl_dsift_with_gaussian_window(mut self_0: *mut VlDsiftFilter) {
    let mut binx: libc::c_int = 0;
    let mut biny: libc::c_int = 0;
    let mut bint: libc::c_int = 0;
    let mut framex: libc::c_int = 0;
    let mut framey: libc::c_int = 0;
    let mut xker: *mut libc::c_float = 0 as *mut libc::c_float;
    let mut yker: *mut libc::c_float = 0 as *mut libc::c_float;
    let mut Wx: libc::c_int = (*self_0).geom.binSizeX - 1 as libc::c_int;
    let mut Wy: libc::c_int = (*self_0).geom.binSizeY - 1 as libc::c_int;
    biny = 0 as libc::c_int;
    while biny < (*self_0).geom.numBinY {
        yker = _vl_dsift_new_kernel(
            (*self_0).geom.binSizeY,
            (*self_0).geom.numBinY,
            biny,
            (*self_0).windowSize,
        );
        binx = 0 as libc::c_int;
        while binx < (*self_0).geom.numBinX {
            xker = _vl_dsift_new_kernel(
                (*self_0).geom.binSizeX,
                (*self_0).geom.numBinX,
                binx,
                (*self_0).windowSize,
            );
            bint = 0 as libc::c_int;
            while bint < (*self_0).geom.numBinT {
                vl_imconvcol_vf(
                    (*self_0).convTmp1,
                    (*self_0).imHeight as vl_size,
                    *((*self_0).grads).offset(bint as isize),
                    (*self_0).imWidth as vl_size,
                    (*self_0).imHeight as vl_size,
                    (*self_0).imWidth as vl_size,
                    yker,
                    -Wy as vl_index,
                    Wy as vl_index,
                    1 as libc::c_int,
                    ((0x1 as libc::c_int) << 0 as libc::c_int
                        | (0x1 as libc::c_int) << 2 as libc::c_int) as libc::c_uint,
                );
                vl_imconvcol_vf(
                    (*self_0).convTmp2,
                    (*self_0).imWidth as vl_size,
                    (*self_0).convTmp1,
                    (*self_0).imHeight as vl_size,
                    (*self_0).imWidth as vl_size,
                    (*self_0).imHeight as vl_size,
                    xker,
                    -Wx as vl_index,
                    Wx as vl_index,
                    1 as libc::c_int,
                    ((0x1 as libc::c_int) << 0 as libc::c_int
                        | (0x1 as libc::c_int) << 2 as libc::c_int) as libc::c_uint,
                );
                let mut dst: *mut libc::c_float = ((*self_0).descrs)
                    .offset(bint as isize)
                    .offset((binx * (*self_0).geom.numBinT) as isize)
                    .offset(
                        (biny * ((*self_0).geom.numBinX * (*self_0).geom.numBinT))
                            as isize,
                    );
                let mut src: *mut libc::c_float = (*self_0).convTmp2;
                let mut frameSizeX: libc::c_int = (*self_0).geom.binSizeX
                    * ((*self_0).geom.numBinX - 1 as libc::c_int) + 1 as libc::c_int;
                let mut frameSizeY: libc::c_int = (*self_0).geom.binSizeY
                    * ((*self_0).geom.numBinY - 1 as libc::c_int) + 1 as libc::c_int;
                let mut descrSize: libc::c_int = vl_dsift_get_descriptor_size(self_0);
                framey = (*self_0).boundMinY;
                while framey <= (*self_0).boundMaxY - frameSizeY + 1 as libc::c_int {
                    framex = (*self_0).boundMinX;
                    while framex <= (*self_0).boundMaxX - frameSizeX + 1 as libc::c_int {
                        *dst = *src
                            .offset(
                                ((framex + binx * (*self_0).geom.binSizeX)
                                    * 1 as libc::c_int
                                    + (framey + biny * (*self_0).geom.binSizeY)
                                        * (*self_0).imWidth) as isize,
                            );
                        dst = dst.offset(descrSize as isize);
                        framex += (*self_0).stepX;
                    }
                    framey += (*self_0).stepY;
                }
                bint += 1;
            }
            vl_free(xker as *mut libc::c_void);
            binx += 1;
        }
        vl_free(yker as *mut libc::c_void);
        biny += 1;
    }
}
#[inline]
unsafe extern "C" fn _vl_dsift_with_flat_window(mut self_0: *mut VlDsiftFilter) {
    let mut binx: libc::c_int = 0;
    let mut biny: libc::c_int = 0;
    let mut bint: libc::c_int = 0;
    let mut framex: libc::c_int = 0;
    let mut framey: libc::c_int = 0;
    bint = 0 as libc::c_int;
    while bint < (*self_0).geom.numBinT {
        vl_imconvcoltri_f(
            (*self_0).convTmp1,
            (*self_0).imHeight as vl_size,
            *((*self_0).grads).offset(bint as isize),
            (*self_0).imWidth as vl_size,
            (*self_0).imHeight as vl_size,
            (*self_0).imWidth as vl_size,
            (*self_0).geom.binSizeY as vl_size,
            1 as libc::c_int as vl_size,
            ((0x1 as libc::c_int) << 0 as libc::c_int
                | (0x1 as libc::c_int) << 2 as libc::c_int) as libc::c_uint,
        );
        vl_imconvcoltri_f(
            (*self_0).convTmp2,
            (*self_0).imWidth as vl_size,
            (*self_0).convTmp1,
            (*self_0).imHeight as vl_size,
            (*self_0).imWidth as vl_size,
            (*self_0).imHeight as vl_size,
            (*self_0).geom.binSizeX as vl_size,
            1 as libc::c_int as vl_size,
            ((0x1 as libc::c_int) << 0 as libc::c_int
                | (0x1 as libc::c_int) << 2 as libc::c_int) as libc::c_uint,
        );
        biny = 0 as libc::c_int;
        while biny < (*self_0).geom.numBinY {
            let mut wy: libc::c_float = _vl_dsift_get_bin_window_mean(
                (*self_0).geom.binSizeY,
                (*self_0).geom.numBinY,
                biny,
                (*self_0).windowSize,
            );
            wy *= (*self_0).geom.binSizeY as libc::c_float;
            binx = 0 as libc::c_int;
            while binx < (*self_0).geom.numBinX {
                let mut w: libc::c_float = 0.;
                let mut wx: libc::c_float = _vl_dsift_get_bin_window_mean(
                    (*self_0).geom.binSizeX,
                    (*self_0).geom.numBinX,
                    binx,
                    (*self_0).windowSize,
                );
                let mut dst: *mut libc::c_float = ((*self_0).descrs)
                    .offset(bint as isize)
                    .offset((binx * (*self_0).geom.numBinT) as isize)
                    .offset(
                        (biny * ((*self_0).geom.numBinX * (*self_0).geom.numBinT))
                            as isize,
                    );
                let mut src: *mut libc::c_float = (*self_0).convTmp2;
                let mut frameSizeX: libc::c_int = (*self_0).geom.binSizeX
                    * ((*self_0).geom.numBinX - 1 as libc::c_int) + 1 as libc::c_int;
                let mut frameSizeY: libc::c_int = (*self_0).geom.binSizeY
                    * ((*self_0).geom.numBinY - 1 as libc::c_int) + 1 as libc::c_int;
                let mut descrSize: libc::c_int = vl_dsift_get_descriptor_size(self_0);
                wx *= (*self_0).geom.binSizeX as libc::c_float;
                w = wx * wy;
                framey = (*self_0).boundMinY;
                while framey <= (*self_0).boundMaxY - frameSizeY + 1 as libc::c_int {
                    framex = (*self_0).boundMinX;
                    while framex <= (*self_0).boundMaxX - frameSizeX + 1 as libc::c_int {
                        *dst = w
                            * *src
                                .offset(
                                    ((framex + binx * (*self_0).geom.binSizeX)
                                        * 1 as libc::c_int
                                        + (framey + biny * (*self_0).geom.binSizeY)
                                            * (*self_0).imWidth) as isize,
                                );
                        dst = dst.offset(descrSize as isize);
                        framex += (*self_0).stepX;
                    }
                    framey += (*self_0).stepY;
                }
                binx += 1;
            }
            biny += 1;
        }
        bint += 1;
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_dsift_process(
    mut self_0: *mut VlDsiftFilter,
    mut im: *const libc::c_float,
) {
    let mut t: libc::c_int = 0;
    let mut x: libc::c_int = 0;
    let mut y: libc::c_int = 0;
    _vl_dsift_alloc_buffers(self_0);
    t = 0 as libc::c_int;
    while t < (*self_0).geom.numBinT {
        memset(
            *((*self_0).grads).offset(t as isize) as *mut libc::c_void,
            0 as libc::c_int,
            (::core::mem::size_of::<libc::c_float>() as libc::c_ulong)
                .wrapping_mul((*self_0).imWidth as libc::c_ulong)
                .wrapping_mul((*self_0).imHeight as libc::c_ulong),
        );
        t += 1;
    }
    y = 0 as libc::c_int;
    while y < (*self_0).imHeight {
        x = 0 as libc::c_int;
        while x < (*self_0).imWidth {
            let mut gx: libc::c_float = 0.;
            let mut gy: libc::c_float = 0.;
            let mut angle: libc::c_float = 0.;
            let mut mod_0: libc::c_float = 0.;
            let mut nt: libc::c_float = 0.;
            let mut rbint: libc::c_float = 0.;
            let mut bint: libc::c_int = 0;
            if y == 0 as libc::c_int {
                gy = *im
                    .offset(((y + 1 as libc::c_int) * (*self_0).imWidth + x) as isize)
                    - *im.offset((y * (*self_0).imWidth + x) as isize);
            } else if y == (*self_0).imHeight - 1 as libc::c_int {
                gy = *im.offset((y * (*self_0).imWidth + x) as isize)
                    - *im
                        .offset(
                            ((y - 1 as libc::c_int) * (*self_0).imWidth + x) as isize,
                        );
            } else {
                gy = 0.5f32
                    * (*im
                        .offset(
                            ((y + 1 as libc::c_int) * (*self_0).imWidth + x) as isize,
                        )
                        - *im
                            .offset(
                                ((y - 1 as libc::c_int) * (*self_0).imWidth + x) as isize,
                            ));
            }
            if x == 0 as libc::c_int {
                gx = *im
                    .offset((y * (*self_0).imWidth + (x + 1 as libc::c_int)) as isize)
                    - *im.offset((y * (*self_0).imWidth + x) as isize);
            } else if x == (*self_0).imWidth - 1 as libc::c_int {
                gx = *im.offset((y * (*self_0).imWidth + x) as isize)
                    - *im
                        .offset(
                            (y * (*self_0).imWidth + (x - 1 as libc::c_int)) as isize,
                        );
            } else {
                gx = 0.5f32
                    * (*im
                        .offset(
                            (y * (*self_0).imWidth + (x + 1 as libc::c_int)) as isize,
                        )
                        - *im
                            .offset(
                                (y * (*self_0).imWidth + (x - 1 as libc::c_int)) as isize,
                            ));
            }
            angle = vl_fast_atan2_f(gy, gx);
            mod_0 = vl_fast_sqrt_f(gx * gx + gy * gy);
            nt = (vl_mod_2pi_f(angle) as libc::c_double
                * ((*self_0).geom.numBinT as libc::c_double
                    / (2 as libc::c_int as libc::c_double * 3.141592653589793f64)))
                as libc::c_float;
            bint = vl_floor_f(nt) as libc::c_int;
            rbint = nt - bint as libc::c_float;
            *(*((*self_0).grads).offset((bint % (*self_0).geom.numBinT) as isize))
                .offset(
                    (x + y * (*self_0).imWidth) as isize,
                ) = (1 as libc::c_int as libc::c_float - rbint) * mod_0;
            *(*((*self_0).grads)
                .offset(((bint + 1 as libc::c_int) % (*self_0).geom.numBinT) as isize))
                .offset((x + y * (*self_0).imWidth) as isize) = rbint * mod_0;
            x += 1;
        }
        y += 1;
    }
    if (*self_0).useFlatWindow != 0 {
        _vl_dsift_with_flat_window(self_0);
    } else {
        _vl_dsift_with_gaussian_window(self_0);
    }
    let mut frameIter: *mut VlDsiftKeypoint = (*self_0).frames;
    let mut descrIter: *mut libc::c_float = (*self_0).descrs;
    let mut framex: libc::c_int = 0;
    let mut framey: libc::c_int = 0;
    let mut bint_0: libc::c_int = 0;
    let mut frameSizeX: libc::c_int = (*self_0).geom.binSizeX
        * ((*self_0).geom.numBinX - 1 as libc::c_int) + 1 as libc::c_int;
    let mut frameSizeY: libc::c_int = (*self_0).geom.binSizeY
        * ((*self_0).geom.numBinY - 1 as libc::c_int) + 1 as libc::c_int;
    let mut descrSize: libc::c_int = vl_dsift_get_descriptor_size(self_0);
    let mut deltaCenterX: libc::c_float = 0.5f32
        * (*self_0).geom.binSizeX as libc::c_float
        * ((*self_0).geom.numBinX - 1 as libc::c_int) as libc::c_float;
    let mut deltaCenterY: libc::c_float = 0.5f32
        * (*self_0).geom.binSizeY as libc::c_float
        * ((*self_0).geom.numBinY - 1 as libc::c_int) as libc::c_float;
    let mut normConstant: libc::c_float = (frameSizeX * frameSizeY) as libc::c_float;
    framey = (*self_0).boundMinY;
    while framey <= (*self_0).boundMaxY - frameSizeY + 1 as libc::c_int {
        framex = (*self_0).boundMinX;
        while framex <= (*self_0).boundMaxX - frameSizeX + 1 as libc::c_int {
            (*frameIter).x = (framex as libc::c_float + deltaCenterX) as libc::c_double;
            (*frameIter).y = (framey as libc::c_float + deltaCenterY) as libc::c_double;
            let mut mass: libc::c_float = 0 as libc::c_int as libc::c_float;
            bint_0 = 0 as libc::c_int;
            while bint_0 < descrSize {
                mass += *descrIter.offset(bint_0 as isize);
                bint_0 += 1;
            }
            mass /= normConstant;
            (*frameIter).norm = mass as libc::c_double;
            _vl_dsift_normalize_histogram(
                descrIter,
                descrIter.offset(descrSize as isize),
            );
            bint_0 = 0 as libc::c_int;
            while bint_0 < descrSize {
                if *descrIter.offset(bint_0 as isize) > 0.2f32 {
                    *descrIter.offset(bint_0 as isize) = 0.2f32;
                }
                bint_0 += 1;
            }
            _vl_dsift_normalize_histogram(
                descrIter,
                descrIter.offset(descrSize as isize),
            );
            frameIter = frameIter.offset(1);
            descrIter = descrIter.offset(descrSize as isize);
            framex += (*self_0).stepX;
        }
        framey += (*self_0).stepY;
    }
}
