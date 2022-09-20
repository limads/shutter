use ::libc;
extern "C" {
    fn vl_free(ptr: *mut libc::c_void);
    fn vl_calloc(n: size_t, size: size_t) -> *mut libc::c_void;
    fn vl_malloc(n: size_t) -> *mut libc::c_void;
    fn __assert_fail(
        __assertion: *const libc::c_char,
        __file: *const libc::c_char,
        __line: libc::c_uint,
        __function: *const libc::c_char,
    ) -> !;
    fn atan2(_: libc::c_double, _: libc::c_double) -> libc::c_double;
    fn cos(_: libc::c_double) -> libc::c_double;
    fn sin(_: libc::c_double) -> libc::c_double;
    fn sqrt(_: libc::c_double) -> libc::c_double;
    fn memset(
        _: *mut libc::c_void,
        _: libc::c_int,
        _: libc::c_ulong,
    ) -> *mut libc::c_void;
}
pub type vl_int64 = libc::c_longlong;
pub type vl_uint64 = libc::c_ulonglong;
pub type vl_int = libc::c_int;
pub type vl_size = vl_uint64;
pub type vl_index = vl_int64;
pub type vl_uindex = vl_uint64;
pub type size_t = libc::c_ulong;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlLiopDesc {
    pub numNeighbours: vl_int,
    pub numSpatialBins: vl_int,
    pub intensityThreshold: libc::c_float,
    pub dimension: vl_size,
    pub patchSideLength: vl_size,
    pub patchSize: vl_size,
    pub patchPixels: *mut vl_uindex,
    pub patchIntensities: *mut libc::c_float,
    pub patchPermutation: *mut vl_uindex,
    pub neighRadius: libc::c_float,
    pub neighIntensities: *mut libc::c_float,
    pub neighPermutation: *mut vl_uindex,
    pub neighSamplesX: *mut libc::c_double,
    pub neighSamplesY: *mut libc::c_double,
}
pub type VlLiopDesc = _VlLiopDesc;
#[inline]
unsafe extern "C" fn vl_floor_d(mut x: libc::c_double) -> libc::c_long {
    let mut xi: libc::c_long = x as libc::c_long;
    if x >= 0 as libc::c_int as libc::c_double || xi as libc::c_double == x {
        return xi
    } else {
        return xi - 1 as libc::c_int as libc::c_long
    };
}
unsafe extern "C" fn factorial(mut num: vl_int) -> vl_int {
    let mut result: vl_int = 1 as libc::c_int;
    while num > 1 as libc::c_int {
        result = num * result;
        num -= 1;
    }
    return result;
}
#[inline]
unsafe extern "C" fn get_permutation_index(
    mut permutation: *mut vl_uindex,
    mut size: vl_size,
) -> vl_index {
    let mut index: vl_index = 0 as libc::c_int as vl_index;
    let mut i: vl_index = 0;
    let mut j: vl_index = 0;
    i = 0 as libc::c_int as vl_index;
    while i < size as libc::c_int as libc::c_longlong {
        index = ((index * (size as libc::c_int as libc::c_longlong - i))
            as libc::c_ulonglong)
            .wrapping_add(*permutation.offset(i as isize)) as vl_index;
        j = i + 1 as libc::c_int as libc::c_longlong;
        while j < size as libc::c_int as libc::c_longlong {
            if *permutation.offset(j as isize) > *permutation.offset(i as isize) {
                let ref mut fresh0 = *permutation.offset(j as isize);
                *fresh0 = (*fresh0).wrapping_sub(1);
            }
            j += 1;
        }
        i += 1;
    }
    return index;
}
#[inline]
unsafe extern "C" fn patch_cmp(
    mut liop: *mut VlLiopDesc,
    mut i: vl_index,
    mut j: vl_index,
) -> libc::c_float {
    let mut ii: vl_index = *((*liop).patchPermutation).offset(i as isize) as vl_index;
    let mut jj: vl_index = *((*liop).patchPermutation).offset(j as isize) as vl_index;
    return *((*liop).patchIntensities).offset(ii as isize)
        - *((*liop).patchIntensities).offset(jj as isize);
}
#[inline]
unsafe extern "C" fn patch_swap(
    mut liop: *mut VlLiopDesc,
    mut i: vl_index,
    mut j: vl_index,
) {
    let mut tmp: vl_index = *((*liop).patchPermutation).offset(i as isize) as vl_index;
    *((*liop).patchPermutation)
        .offset(i as isize) = *((*liop).patchPermutation).offset(j as isize);
    *((*liop).patchPermutation).offset(j as isize) = tmp as vl_uindex;
}
#[inline]
unsafe extern "C" fn patch_sort_recursive(
    mut array: *mut VlLiopDesc,
    mut begin: vl_uindex,
    mut end: vl_uindex,
) {
    let mut pivot: vl_uindex = end
        .wrapping_add(begin)
        .wrapping_div(2 as libc::c_int as libc::c_ulonglong);
    let mut lowPart: vl_uindex = 0;
    let mut i: vl_uindex = 0;
    if begin <= end {} else {
        __assert_fail(
            b"begin <= end\0" as *const u8 as *const libc::c_char,
            b"vl/qsort-def.h\0" as *const u8 as *const libc::c_char,
            133 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 62],
                &[libc::c_char; 62],
            >(b"void patch_sort_recursive(VlLiopDesc *, vl_uindex, vl_uindex)\0"))
                .as_ptr(),
        );
    }
    patch_swap(array, pivot as vl_index, end as vl_index);
    pivot = end;
    lowPart = begin;
    i = begin;
    while i < end {
        if patch_cmp(array, i as vl_index, pivot as vl_index)
            <= 0 as libc::c_int as libc::c_float
        {
            patch_swap(array, lowPart as vl_index, i as vl_index);
            lowPart = lowPart.wrapping_add(1);
        }
        i = i.wrapping_add(1);
    }
    patch_swap(array, lowPart as vl_index, pivot as vl_index);
    pivot = lowPart;
    if pivot > begin {
        patch_sort_recursive(
            array,
            begin,
            pivot.wrapping_sub(1 as libc::c_int as libc::c_ulonglong),
        );
    }
    if pivot < end {
        patch_sort_recursive(
            array,
            pivot.wrapping_add(1 as libc::c_int as libc::c_ulonglong),
            end,
        );
    }
}
#[inline]
unsafe extern "C" fn neigh_sort_recursive(
    mut array: *mut VlLiopDesc,
    mut begin: vl_uindex,
    mut end: vl_uindex,
) {
    let mut pivot: vl_uindex = end
        .wrapping_add(begin)
        .wrapping_div(2 as libc::c_int as libc::c_ulonglong);
    let mut lowPart: vl_uindex = 0;
    let mut i: vl_uindex = 0;
    if begin <= end {} else {
        __assert_fail(
            b"begin <= end\0" as *const u8 as *const libc::c_char,
            b"vl/qsort-def.h\0" as *const u8 as *const libc::c_char,
            133 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 62],
                &[libc::c_char; 62],
            >(b"void neigh_sort_recursive(VlLiopDesc *, vl_uindex, vl_uindex)\0"))
                .as_ptr(),
        );
    }
    neigh_swap(array, pivot as vl_index, end as vl_index);
    pivot = end;
    lowPart = begin;
    i = begin;
    while i < end {
        if neigh_cmp(array, i as vl_index, pivot as vl_index)
            <= 0 as libc::c_int as libc::c_float
        {
            neigh_swap(array, lowPart as vl_index, i as vl_index);
            lowPart = lowPart.wrapping_add(1);
        }
        i = i.wrapping_add(1);
    }
    neigh_swap(array, lowPart as vl_index, pivot as vl_index);
    pivot = lowPart;
    if pivot > begin {
        neigh_sort_recursive(
            array,
            begin,
            pivot.wrapping_sub(1 as libc::c_int as libc::c_ulonglong),
        );
    }
    if pivot < end {
        neigh_sort_recursive(
            array,
            pivot.wrapping_add(1 as libc::c_int as libc::c_ulonglong),
            end,
        );
    }
}
#[inline]
unsafe extern "C" fn patch_sort(mut array: *mut VlLiopDesc, mut size: vl_size) {
    if size >= 1 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"size >= 1\0" as *const u8 as *const libc::c_char,
            b"vl/qsort-def.h\0" as *const u8 as *const libc::c_char,
            186 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 39],
                &[libc::c_char; 39],
            >(b"void patch_sort(VlLiopDesc *, vl_size)\0"))
                .as_ptr(),
        );
    }
    patch_sort_recursive(
        array,
        0 as libc::c_int as vl_uindex,
        size.wrapping_sub(1 as libc::c_int as libc::c_ulonglong),
    );
}
#[inline]
unsafe extern "C" fn neigh_sort(mut array: *mut VlLiopDesc, mut size: vl_size) {
    if size >= 1 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"size >= 1\0" as *const u8 as *const libc::c_char,
            b"vl/qsort-def.h\0" as *const u8 as *const libc::c_char,
            186 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 39],
                &[libc::c_char; 39],
            >(b"void neigh_sort(VlLiopDesc *, vl_size)\0"))
                .as_ptr(),
        );
    }
    neigh_sort_recursive(
        array,
        0 as libc::c_int as vl_uindex,
        size.wrapping_sub(1 as libc::c_int as libc::c_ulonglong),
    );
}
#[inline]
unsafe extern "C" fn neigh_cmp(
    mut liop: *mut VlLiopDesc,
    mut i: vl_index,
    mut j: vl_index,
) -> libc::c_float {
    let mut ii: vl_index = *((*liop).neighPermutation).offset(i as isize) as vl_index;
    let mut jj: vl_index = *((*liop).neighPermutation).offset(j as isize) as vl_index;
    return *((*liop).neighIntensities).offset(ii as isize)
        - *((*liop).neighIntensities).offset(jj as isize);
}
#[inline]
unsafe extern "C" fn neigh_swap(
    mut liop: *mut VlLiopDesc,
    mut i: vl_index,
    mut j: vl_index,
) {
    let mut tmp: vl_index = *((*liop).neighPermutation).offset(i as isize) as vl_index;
    *((*liop).neighPermutation)
        .offset(i as isize) = *((*liop).neighPermutation).offset(j as isize);
    *((*liop).neighPermutation).offset(j as isize) = tmp as vl_uindex;
}
#[no_mangle]
pub unsafe extern "C" fn vl_liopdesc_new(
    mut numNeighbours: vl_int,
    mut numSpatialBins: vl_int,
    mut radius: libc::c_float,
    mut sideLength: vl_size,
) -> *mut VlLiopDesc {
    let mut i: vl_index = 0;
    let mut t: vl_index = 0;
    let mut self_0: *mut VlLiopDesc = vl_calloc(
        ::core::mem::size_of::<VlLiopDesc>() as libc::c_ulong,
        1 as libc::c_int as size_t,
    ) as *mut VlLiopDesc;
    if radius
        <= sideLength.wrapping_div(2 as libc::c_int as libc::c_ulonglong)
            as libc::c_float
    {} else {
        __assert_fail(
            b"radius <= sideLength/2\0" as *const u8 as *const libc::c_char,
            b"vl/liop.c\0" as *const u8 as *const libc::c_char,
            327 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 60],
                &[libc::c_char; 60],
            >(b"VlLiopDesc *vl_liopdesc_new(vl_int, vl_int, float, vl_size)\0"))
                .as_ptr(),
        );
    }
    (*self_0).numNeighbours = numNeighbours;
    (*self_0).numSpatialBins = numSpatialBins;
    (*self_0).neighRadius = radius;
    (*self_0)
        .intensityThreshold = -(5.0f64 / 255 as libc::c_int as libc::c_double)
        as libc::c_float;
    (*self_0).dimension = (factorial(numNeighbours) * numSpatialBins) as vl_size;
    (*self_0).patchSize = 0 as libc::c_int as vl_size;
    (*self_0)
        .patchPixels = vl_malloc(
        (::core::mem::size_of::<vl_uindex>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(sideLength)
            .wrapping_mul(sideLength) as size_t,
    ) as *mut vl_uindex;
    (*self_0).patchSideLength = sideLength;
    let mut x: vl_index = 0;
    let mut y: vl_index = 0;
    let mut center: vl_index = sideLength
        .wrapping_sub(1 as libc::c_int as libc::c_ulonglong)
        .wrapping_div(2 as libc::c_int as libc::c_ulonglong) as vl_index;
    let mut t_0: libc::c_double = (center as libc::c_float - radius) as libc::c_double
        + 0.6f64;
    let mut t2: vl_index = (t_0 * t_0) as vl_index;
    y = 0 as libc::c_int as vl_index;
    while y < sideLength as libc::c_int as libc::c_longlong {
        x = 0 as libc::c_int as vl_index;
        while x < sideLength as libc::c_int as libc::c_longlong {
            let mut dx: vl_index = x - center;
            let mut dy: vl_index = y - center;
            if !(x == 0 as libc::c_int as libc::c_longlong
                && y == 0 as libc::c_int as libc::c_longlong)
            {
                if dx * dx + dy * dy <= t2 {
                    let fresh1 = (*self_0).patchSize;
                    (*self_0).patchSize = ((*self_0).patchSize).wrapping_add(1);
                    *((*self_0).patchPixels)
                        .offset(
                            fresh1 as isize,
                        ) = (x as libc::c_ulonglong)
                        .wrapping_add((y as libc::c_ulonglong).wrapping_mul(sideLength));
                }
            }
            x += 1;
        }
        y += 1;
    }
    (*self_0)
        .patchIntensities = vl_malloc(
        (::core::mem::size_of::<vl_uindex>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).patchSize) as size_t,
    ) as *mut libc::c_float;
    (*self_0)
        .patchPermutation = vl_malloc(
        (::core::mem::size_of::<vl_uindex>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).patchSize) as size_t,
    ) as *mut vl_uindex;
    (*self_0)
        .neighPermutation = vl_malloc(
        (::core::mem::size_of::<vl_uindex>() as libc::c_ulong)
            .wrapping_mul((*self_0).numNeighbours as libc::c_ulong),
    ) as *mut vl_uindex;
    (*self_0)
        .neighIntensities = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong)
            .wrapping_mul((*self_0).numNeighbours as libc::c_ulong),
    ) as *mut libc::c_float;
    (*self_0)
        .neighSamplesX = vl_calloc(
        ::core::mem::size_of::<libc::c_double>() as libc::c_ulong,
        ((*self_0).numNeighbours as libc::c_ulonglong).wrapping_mul((*self_0).patchSize)
            as size_t,
    ) as *mut libc::c_double;
    (*self_0)
        .neighSamplesY = vl_calloc(
        ::core::mem::size_of::<libc::c_double>() as libc::c_ulong,
        ((*self_0).numNeighbours as libc::c_ulonglong).wrapping_mul((*self_0).patchSize)
            as size_t,
    ) as *mut libc::c_double;
    i = 0 as libc::c_int as vl_index;
    while i < (*self_0).patchSize as libc::c_int as libc::c_longlong {
        let mut pixel: vl_index = 0;
        let mut x_0: libc::c_double = 0.;
        let mut y_0: libc::c_double = 0.;
        let mut dangle: libc::c_double = 2 as libc::c_int as libc::c_double
            * 3.141592653589793f64 / (*self_0).numNeighbours as libc::c_double;
        let mut angle0: libc::c_double = 0.;
        let mut center_0: vl_index = sideLength
            .wrapping_sub(1 as libc::c_int as libc::c_ulonglong)
            .wrapping_div(2 as libc::c_int as libc::c_ulonglong) as vl_index;
        pixel = *((*self_0).patchPixels).offset(i as isize) as vl_index;
        x_0 = (pixel % (*self_0).patchSideLength as libc::c_int as libc::c_longlong
            - center_0) as libc::c_double;
        y_0 = (pixel / (*self_0).patchSideLength as libc::c_int as libc::c_longlong
            - center_0) as libc::c_double;
        angle0 = atan2(y_0, x_0);
        t = 0 as libc::c_int as vl_index;
        while t < (*self_0).numNeighbours as libc::c_longlong {
            let mut x1: libc::c_double = x_0
                + radius as libc::c_double * cos(angle0 + dangle * t as libc::c_double)
                + center_0 as libc::c_double;
            let mut y1: libc::c_double = y_0
                + radius as libc::c_double * sin(angle0 + dangle * t as libc::c_double)
                + center_0 as libc::c_double;
            *((*self_0).neighSamplesX)
                .offset(
                    (t + (*self_0).numNeighbours as libc::c_longlong * i) as isize,
                ) = x1;
            *((*self_0).neighSamplesY)
                .offset(
                    (t + (*self_0).numNeighbours as libc::c_longlong * i) as isize,
                ) = y1;
            t += 1;
        }
        i += 1;
    }
    return self_0;
}
#[no_mangle]
pub unsafe extern "C" fn vl_liopdesc_new_basic(
    mut sideLength: vl_size,
) -> *mut VlLiopDesc {
    return vl_liopdesc_new(
        4 as libc::c_int,
        6 as libc::c_int,
        6.0f64 as libc::c_float,
        sideLength,
    );
}
#[no_mangle]
pub unsafe extern "C" fn vl_liopdesc_delete(mut self_0: *mut VlLiopDesc) {
    vl_free((*self_0).patchPixels as *mut libc::c_void);
    vl_free((*self_0).patchIntensities as *mut libc::c_void);
    vl_free((*self_0).patchPermutation as *mut libc::c_void);
    vl_free((*self_0).neighPermutation as *mut libc::c_void);
    vl_free((*self_0).neighIntensities as *mut libc::c_void);
    vl_free((*self_0).neighSamplesX as *mut libc::c_void);
    vl_free((*self_0).neighSamplesY as *mut libc::c_void);
    vl_free(self_0 as *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn vl_liopdesc_process(
    mut self_0: *mut VlLiopDesc,
    mut desc: *mut libc::c_float,
    mut patch: *const libc::c_float,
) {
    let mut i: vl_index = 0;
    let mut t: vl_index = 0;
    let mut offset: vl_index = 0;
    let mut numPermutations: vl_index = 0;
    let mut spatialBinArea: vl_index = 0;
    let mut spatialBinEnd: vl_index = 0;
    let mut spatialBinIndex: vl_index = 0;
    let mut threshold: libc::c_float = 0.;
    memset(
        desc as *mut libc::c_void,
        0 as libc::c_int,
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).dimension) as libc::c_ulong,
    );
    i = 0 as libc::c_int as vl_index;
    while i < (*self_0).patchSize as libc::c_int as libc::c_longlong {
        let mut pixel: vl_index = *((*self_0).patchPixels).offset(i as isize)
            as vl_index;
        *((*self_0).patchIntensities).offset(i as isize) = *patch.offset(pixel as isize);
        *((*self_0).patchPermutation).offset(i as isize) = i as vl_uindex;
        i += 1;
    }
    patch_sort(self_0, (*self_0).patchSize);
    if (*self_0).intensityThreshold < 0 as libc::c_int as libc::c_float {
        i = *((*self_0).patchPermutation).offset(0 as libc::c_int as isize) as vl_index;
        t = *((*self_0).patchPermutation)
            .offset(
                ((*self_0).patchSize).wrapping_sub(1 as libc::c_int as libc::c_ulonglong)
                    as isize,
            ) as vl_index;
        threshold = -(*self_0).intensityThreshold
            * (*((*self_0).patchIntensities).offset(t as isize)
                - *((*self_0).patchIntensities).offset(i as isize));
    } else {
        threshold = (*self_0).intensityThreshold;
    }
    numPermutations = factorial((*self_0).numNeighbours) as vl_index;
    spatialBinArea = ((*self_0).patchSize)
        .wrapping_div((*self_0).numSpatialBins as libc::c_ulonglong) as vl_index;
    spatialBinEnd = spatialBinArea;
    spatialBinIndex = 0 as libc::c_int as vl_index;
    offset = 0 as libc::c_int as vl_index;
    i = 0 as libc::c_int as vl_index;
    while i < (*self_0).patchSize as libc::c_int as libc::c_longlong {
        let mut permIndex: vl_index = 0;
        let mut sx: *mut libc::c_double = 0 as *mut libc::c_double;
        let mut sy: *mut libc::c_double = 0 as *mut libc::c_double;
        if i >= spatialBinEnd as libc::c_int as libc::c_longlong
            && spatialBinIndex
                < ((*self_0).numSpatialBins - 1 as libc::c_int) as libc::c_longlong
        {
            spatialBinEnd += spatialBinArea;
            spatialBinIndex += 1;
            offset += numPermutations;
        }
        sx = ((*self_0).neighSamplesX)
            .offset(
                ((*self_0).numNeighbours as libc::c_ulonglong)
                    .wrapping_mul(*((*self_0).patchPermutation).offset(i as isize))
                    as isize,
            );
        sy = ((*self_0).neighSamplesY)
            .offset(
                ((*self_0).numNeighbours as libc::c_ulonglong)
                    .wrapping_mul(*((*self_0).patchPermutation).offset(i as isize))
                    as isize,
            );
        t = 0 as libc::c_int as vl_index;
        while t < (*self_0).numNeighbours as libc::c_longlong {
            let fresh2 = sx;
            sx = sx.offset(1);
            let mut x: libc::c_double = *fresh2;
            let fresh3 = sy;
            sy = sy.offset(1);
            let mut y: libc::c_double = *fresh3;
            let mut ix: vl_index = vl_floor_d(x) as vl_index;
            let mut iy: vl_index = vl_floor_d(y) as vl_index;
            let mut wx: libc::c_double = x - ix as libc::c_double;
            let mut wy: libc::c_double = y - iy as libc::c_double;
            let mut a: libc::c_double = 0 as libc::c_int as libc::c_double;
            let mut b: libc::c_double = 0 as libc::c_int as libc::c_double;
            let mut c: libc::c_double = 0 as libc::c_int as libc::c_double;
            let mut d: libc::c_double = 0 as libc::c_int as libc::c_double;
            let mut L: libc::c_int = (*self_0).patchSideLength as libc::c_int;
            if ix >= 0 as libc::c_int as libc::c_longlong
                && iy >= 0 as libc::c_int as libc::c_longlong
            {
                a = *patch.offset((ix + iy * L as libc::c_longlong) as isize)
                    as libc::c_double;
            }
            if ix < (L - 1 as libc::c_int) as libc::c_longlong
                && iy >= 0 as libc::c_int as libc::c_longlong
            {
                b = *patch
                    .offset(
                        (ix + 1 as libc::c_int as libc::c_longlong
                            + iy * L as libc::c_longlong) as isize,
                    ) as libc::c_double;
            }
            if ix >= 0 as libc::c_int as libc::c_longlong
                && iy < (L - 1 as libc::c_int) as libc::c_longlong
            {
                c = *patch
                    .offset(
                        (ix
                            + (iy + 1 as libc::c_int as libc::c_longlong)
                                * L as libc::c_longlong) as isize,
                    ) as libc::c_double;
            }
            if ix < (L - 1 as libc::c_int) as libc::c_longlong
                && iy < (L - 1 as libc::c_int) as libc::c_longlong
            {
                d = *patch
                    .offset(
                        (ix + 1 as libc::c_int as libc::c_longlong
                            + (iy + 1 as libc::c_int as libc::c_longlong)
                                * L as libc::c_longlong) as isize,
                    ) as libc::c_double;
            }
            *((*self_0).neighPermutation).offset(t as isize) = t as vl_uindex;
            *((*self_0).neighIntensities)
                .offset(
                    t as isize,
                ) = ((1 as libc::c_int as libc::c_double - wy) * (a + (b - a) * wx)
                + wy * (c + (d - c) * wx)) as libc::c_float;
            t += 1;
        }
        neigh_sort(self_0, (*self_0).numNeighbours as vl_size);
        permIndex = get_permutation_index(
            (*self_0).neighPermutation,
            (*self_0).numNeighbours as vl_size,
        );
        let mut k: libc::c_int = 0;
        let mut t_0: libc::c_int = 0;
        let mut weight: libc::c_float = 0 as libc::c_int as libc::c_float;
        k = 0 as libc::c_int;
        while k < (*self_0).numNeighbours {
            t_0 = k + 1 as libc::c_int;
            while t_0 < (*self_0).numNeighbours {
                let mut a_0: libc::c_double = *((*self_0).neighIntensities)
                    .offset(k as isize) as libc::c_double;
                let mut b_0: libc::c_double = *((*self_0).neighIntensities)
                    .offset(t_0 as isize) as libc::c_double;
                weight
                    += (a_0 > b_0 + threshold as libc::c_double
                        || b_0 > a_0 + threshold as libc::c_double) as libc::c_int
                        as libc::c_float;
                t_0 += 1;
            }
            k += 1;
        }
        *desc.offset((permIndex + offset) as isize) += weight;
        i += 1;
    }
    let mut norm: libc::c_float = 0 as libc::c_int as libc::c_float;
    i = 0 as libc::c_int as vl_index;
    while i < (*self_0).dimension as libc::c_int as libc::c_longlong {
        norm += *desc.offset(i as isize) * *desc.offset(i as isize);
        i += 1;
    }
    norm = (if sqrt(norm as libc::c_double) > 1e-12f64 {
        sqrt(norm as libc::c_double)
    } else {
        1e-12f64
    }) as libc::c_float;
    i = 0 as libc::c_int as vl_index;
    while i < (*self_0).dimension as libc::c_int as libc::c_longlong {
        *desc.offset(i as isize) /= norm;
        i += 1;
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_liopdesc_get_dimension(
    mut self_0: *const VlLiopDesc,
) -> vl_size {
    return (*self_0).dimension;
}
#[no_mangle]
pub unsafe extern "C" fn vl_liopdesc_get_num_neighbours(
    mut self_0: *const VlLiopDesc,
) -> vl_size {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/liop.c\0" as *const u8 as *const libc::c_char,
            583 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 59],
                &[libc::c_char; 59],
            >(b"vl_size vl_liopdesc_get_num_neighbours(const VlLiopDesc *)\0"))
                .as_ptr(),
        );
    }
    return (*self_0).numNeighbours as vl_size;
}
#[no_mangle]
pub unsafe extern "C" fn vl_liopdesc_get_intensity_threshold(
    mut self_0: *const VlLiopDesc,
) -> libc::c_float {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/liop.c\0" as *const u8 as *const libc::c_char,
            596 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 62],
                &[libc::c_char; 62],
            >(b"float vl_liopdesc_get_intensity_threshold(const VlLiopDesc *)\0"))
                .as_ptr(),
        );
    }
    return (*self_0).intensityThreshold;
}
#[no_mangle]
pub unsafe extern "C" fn vl_liopdesc_set_intensity_threshold(
    mut self_0: *mut VlLiopDesc,
    mut x: libc::c_float,
) {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/liop.c\0" as *const u8 as *const libc::c_char,
            615 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 62],
                &[libc::c_char; 62],
            >(b"void vl_liopdesc_set_intensity_threshold(VlLiopDesc *, float)\0"))
                .as_ptr(),
        );
    }
    (*self_0).intensityThreshold = x;
}
#[no_mangle]
pub unsafe extern "C" fn vl_liopdesc_get_neighbourhood_radius(
    mut self_0: *const VlLiopDesc,
) -> libc::c_double {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/liop.c\0" as *const u8 as *const libc::c_char,
            627 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 64],
                &[libc::c_char; 64],
            >(b"double vl_liopdesc_get_neighbourhood_radius(const VlLiopDesc *)\0"))
                .as_ptr(),
        );
    }
    return (*self_0).neighRadius as libc::c_double;
}
#[no_mangle]
pub unsafe extern "C" fn vl_liopdesc_get_num_spatial_bins(
    mut self_0: *const VlLiopDesc,
) -> vl_size {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/liop.c\0" as *const u8 as *const libc::c_char,
            639 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 61],
                &[libc::c_char; 61],
            >(b"vl_size vl_liopdesc_get_num_spatial_bins(const VlLiopDesc *)\0"))
                .as_ptr(),
        );
    }
    return (*self_0).numSpatialBins as vl_size;
}
