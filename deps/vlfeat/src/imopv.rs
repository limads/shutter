use ::libc;
extern "C" {
    fn vl_get_simd_enabled() -> vl_bool;
    fn __assert_fail(
        __assertion: *const libc::c_char,
        __file: *const libc::c_char,
        __line: libc::c_uint,
        __function: *const libc::c_char,
    ) -> !;
    fn vl_cpu_has_sse2() -> vl_bool;
    fn vl_malloc(n: size_t) -> *mut libc::c_void;
    fn vl_free(ptr: *mut libc::c_void);
    fn _vl_imconvcol_vf_sse2(
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
    fn _vl_imconvcol_vd_sse2(
        dst: *mut libc::c_double,
        dst_stride: vl_size,
        src: *const libc::c_double,
        src_width: vl_size,
        src_height: vl_size,
        src_stride: vl_size,
        filt: *const libc::c_double,
        filt_begin: vl_index,
        filt_end: vl_index,
        step: libc::c_int,
        flags: libc::c_uint,
    );
    fn exp(_: libc::c_double) -> libc::c_double;
}
pub type vl_int64 = libc::c_longlong;
pub type vl_int32 = libc::c_int;
pub type vl_uint64 = libc::c_ulonglong;
pub type vl_uint32 = libc::c_uint;
pub type vl_bool = libc::c_int;
pub type vl_size = vl_uint64;
pub type vl_index = vl_int64;
pub type vl_uindex = vl_uint64;
pub type size_t = libc::c_ulong;
#[derive(Copy, Clone)]
#[repr(C)]
pub union C2RustUnnamed {
    pub raw: vl_uint64,
    pub value: libc::c_double,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub union C2RustUnnamed_0 {
    pub raw: vl_uint32,
    pub value: libc::c_float,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub union C2RustUnnamed_1 {
    pub x: libc::c_float,
    pub i: vl_int32,
}
static mut vl_infinity_d: C2RustUnnamed = C2RustUnnamed {
    raw: 0x7ff0000000000000 as libc::c_ulonglong,
};
static mut vl_infinity_f: C2RustUnnamed_0 = C2RustUnnamed_0 {
    raw: 0x7f800000 as libc::c_ulong as vl_uint32,
};
#[inline]
unsafe extern "C" fn vl_fast_resqrt_f(mut x: libc::c_float) -> libc::c_float {
    let mut u: C2RustUnnamed_1 = C2RustUnnamed_1 { x: 0. };
    let mut xhalf: libc::c_float = 0.5f64 as libc::c_float * x;
    u.x = x;
    u.i = 0x5f3759df as libc::c_int - (u.i >> 1 as libc::c_int);
    u.x = u.x * (1.5f64 as libc::c_float - xhalf * u.x * u.x);
    u.x = u.x * (1.5f64 as libc::c_float - xhalf * u.x * u.x);
    return u.x;
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
unsafe extern "C" fn vl_abs_f(mut x: libc::c_float) -> libc::c_float {
    return x.abs();
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
#[no_mangle]
pub unsafe extern "C" fn vl_imconvcol_vf(
    mut dst: *mut libc::c_float,
    mut dst_stride: vl_size,
    mut src: *const libc::c_float,
    mut src_width: vl_size,
    mut src_height: vl_size,
    mut src_stride: vl_size,
    mut filt: *const libc::c_float,
    mut filt_begin: vl_index,
    mut filt_end: vl_index,
    mut step: libc::c_int,
    mut flags: libc::c_uint,
) {
    let mut x: vl_index = 0 as libc::c_int as vl_index;
    let mut y: vl_index = 0;
    let mut dheight: vl_index = src_height
        .wrapping_sub(1 as libc::c_int as libc::c_ulonglong)
        .wrapping_div(step as libc::c_ulonglong)
        .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as vl_index;
    let mut transp: vl_bool = (flags
        & ((0x1 as libc::c_int) << 2 as libc::c_int) as libc::c_uint) as vl_bool;
    let mut zeropad: vl_bool = (flags & 0x3 as libc::c_int as libc::c_uint
        == ((0 as libc::c_int) << 0 as libc::c_int) as libc::c_uint) as libc::c_int;
    if vl_cpu_has_sse2() != 0 && vl_get_simd_enabled() != 0 {
        _vl_imconvcol_vf_sse2(
            dst,
            dst_stride,
            src,
            src_width,
            src_height,
            src_stride,
            filt,
            filt_begin,
            filt_end,
            step,
            flags,
        );
        return;
    }
    filt = filt.offset((filt_end - filt_begin) as isize);
    while x < src_width as libc::c_int as libc::c_longlong {
        let mut filti: *const libc::c_float = 0 as *const libc::c_float;
        let mut stop: vl_index = 0;
        y = 0 as libc::c_int as vl_index;
        while y < src_height as libc::c_int as libc::c_longlong {
            let mut acc: libc::c_float = 0 as libc::c_int as libc::c_float;
            let mut v: libc::c_float = 0 as libc::c_int as libc::c_float;
            let mut c: libc::c_float = 0.;
            let mut srci: *const libc::c_float = 0 as *const libc::c_float;
            filti = filt;
            stop = filt_end - y;
            srci = src
                .offset(x as isize)
                .offset(
                    -((stop as libc::c_ulonglong).wrapping_mul(src_stride) as isize),
                );
            if stop > 0 as libc::c_int as libc::c_longlong {
                if zeropad != 0 {
                    v = 0 as libc::c_int as libc::c_float;
                } else {
                    v = *src.offset(x as isize);
                }
                while filti > filt.offset(-(stop as isize)) {
                    let fresh0 = filti;
                    filti = filti.offset(-1);
                    c = *fresh0;
                    acc += v * c;
                    srci = srci.offset(src_stride as isize);
                }
            }
            stop = filt_end
                - (if filt_begin
                    > y - src_height as libc::c_int as libc::c_longlong
                        + 1 as libc::c_int as libc::c_longlong
                {
                    filt_begin
                } else {
                    y - src_height as libc::c_int as libc::c_longlong
                        + 1 as libc::c_int as libc::c_longlong
                }) + 1 as libc::c_int as libc::c_longlong;
            while filti > filt.offset(-(stop as isize)) {
                v = *srci;
                let fresh1 = filti;
                filti = filti.offset(-1);
                c = *fresh1;
                acc += v * c;
                srci = srci.offset(src_stride as isize);
            }
            if zeropad != 0 {
                v = 0 as libc::c_int as libc::c_float;
            }
            stop = filt_end - filt_begin + 1 as libc::c_int as libc::c_longlong;
            while filti > filt.offset(-(stop as isize)) {
                let fresh2 = filti;
                filti = filti.offset(-1);
                c = *fresh2;
                acc += v * c;
            }
            if transp != 0 {
                *dst = acc;
                dst = dst.offset(1 as libc::c_int as isize);
            } else {
                *dst = acc;
                dst = dst.offset(dst_stride as isize);
            }
            y += step as libc::c_longlong;
        }
        if transp != 0 {
            dst = dst
                .offset(
                    (1 as libc::c_int as libc::c_ulonglong)
                        .wrapping_mul(dst_stride)
                        .wrapping_sub(
                            (dheight * 1 as libc::c_int as libc::c_longlong)
                                as libc::c_ulonglong,
                        ) as isize,
                );
        } else {
            dst = dst
                .offset(
                    ((1 as libc::c_int * 1 as libc::c_int) as libc::c_ulonglong)
                        .wrapping_sub(
                            (dheight as libc::c_ulonglong).wrapping_mul(dst_stride),
                        ) as isize,
                );
        }
        x += 1 as libc::c_int as libc::c_longlong;
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_imconvcol_vd(
    mut dst: *mut libc::c_double,
    mut dst_stride: vl_size,
    mut src: *const libc::c_double,
    mut src_width: vl_size,
    mut src_height: vl_size,
    mut src_stride: vl_size,
    mut filt: *const libc::c_double,
    mut filt_begin: vl_index,
    mut filt_end: vl_index,
    mut step: libc::c_int,
    mut flags: libc::c_uint,
) {
    let mut x: vl_index = 0 as libc::c_int as vl_index;
    let mut y: vl_index = 0;
    let mut dheight: vl_index = src_height
        .wrapping_sub(1 as libc::c_int as libc::c_ulonglong)
        .wrapping_div(step as libc::c_ulonglong)
        .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as vl_index;
    let mut transp: vl_bool = (flags
        & ((0x1 as libc::c_int) << 2 as libc::c_int) as libc::c_uint) as vl_bool;
    let mut zeropad: vl_bool = (flags & 0x3 as libc::c_int as libc::c_uint
        == ((0 as libc::c_int) << 0 as libc::c_int) as libc::c_uint) as libc::c_int;
    if vl_cpu_has_sse2() != 0 && vl_get_simd_enabled() != 0 {
        _vl_imconvcol_vd_sse2(
            dst,
            dst_stride,
            src,
            src_width,
            src_height,
            src_stride,
            filt,
            filt_begin,
            filt_end,
            step,
            flags,
        );
        return;
    }
    filt = filt.offset((filt_end - filt_begin) as isize);
    while x < src_width as libc::c_int as libc::c_longlong {
        let mut filti: *const libc::c_double = 0 as *const libc::c_double;
        let mut stop: vl_index = 0;
        y = 0 as libc::c_int as vl_index;
        while y < src_height as libc::c_int as libc::c_longlong {
            let mut acc: libc::c_double = 0 as libc::c_int as libc::c_double;
            let mut v: libc::c_double = 0 as libc::c_int as libc::c_double;
            let mut c: libc::c_double = 0.;
            let mut srci: *const libc::c_double = 0 as *const libc::c_double;
            filti = filt;
            stop = filt_end - y;
            srci = src
                .offset(x as isize)
                .offset(
                    -((stop as libc::c_ulonglong).wrapping_mul(src_stride) as isize),
                );
            if stop > 0 as libc::c_int as libc::c_longlong {
                if zeropad != 0 {
                    v = 0 as libc::c_int as libc::c_double;
                } else {
                    v = *src.offset(x as isize);
                }
                while filti > filt.offset(-(stop as isize)) {
                    let fresh3 = filti;
                    filti = filti.offset(-1);
                    c = *fresh3;
                    acc += v * c;
                    srci = srci.offset(src_stride as isize);
                }
            }
            stop = filt_end
                - (if filt_begin
                    > y - src_height as libc::c_int as libc::c_longlong
                        + 1 as libc::c_int as libc::c_longlong
                {
                    filt_begin
                } else {
                    y - src_height as libc::c_int as libc::c_longlong
                        + 1 as libc::c_int as libc::c_longlong
                }) + 1 as libc::c_int as libc::c_longlong;
            while filti > filt.offset(-(stop as isize)) {
                v = *srci;
                let fresh4 = filti;
                filti = filti.offset(-1);
                c = *fresh4;
                acc += v * c;
                srci = srci.offset(src_stride as isize);
            }
            if zeropad != 0 {
                v = 0 as libc::c_int as libc::c_double;
            }
            stop = filt_end - filt_begin + 1 as libc::c_int as libc::c_longlong;
            while filti > filt.offset(-(stop as isize)) {
                let fresh5 = filti;
                filti = filti.offset(-1);
                c = *fresh5;
                acc += v * c;
            }
            if transp != 0 {
                *dst = acc;
                dst = dst.offset(1 as libc::c_int as isize);
            } else {
                *dst = acc;
                dst = dst.offset(dst_stride as isize);
            }
            y += step as libc::c_longlong;
        }
        if transp != 0 {
            dst = dst
                .offset(
                    (1 as libc::c_int as libc::c_ulonglong)
                        .wrapping_mul(dst_stride)
                        .wrapping_sub(
                            (dheight * 1 as libc::c_int as libc::c_longlong)
                                as libc::c_ulonglong,
                        ) as isize,
                );
        } else {
            dst = dst
                .offset(
                    ((1 as libc::c_int * 1 as libc::c_int) as libc::c_ulonglong)
                        .wrapping_sub(
                            (dheight as libc::c_ulonglong).wrapping_mul(dst_stride),
                        ) as isize,
                );
        }
        x += 1 as libc::c_int as libc::c_longlong;
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_image_distance_transform_f(
    mut image: *const libc::c_float,
    mut numColumns: vl_size,
    mut numRows: vl_size,
    mut columnStride: vl_size,
    mut rowStride: vl_size,
    mut distanceTransform: *mut libc::c_float,
    mut indexes: *mut vl_uindex,
    mut coeff: libc::c_float,
    mut offset: libc::c_float,
) {
    let mut x: vl_uindex = 0;
    let mut y: vl_uindex = 0;
    let mut from: *mut libc::c_float = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numColumns.wrapping_add(1 as libc::c_int as libc::c_ulonglong))
            as size_t,
    ) as *mut libc::c_float;
    let mut base: *mut libc::c_float = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numColumns) as size_t,
    ) as *mut libc::c_float;
    let mut baseIndexes: *mut vl_uindex = vl_malloc(
        (::core::mem::size_of::<vl_uindex>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numColumns) as size_t,
    ) as *mut vl_uindex;
    let mut which: *mut vl_uindex = vl_malloc(
        (::core::mem::size_of::<vl_uindex>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numColumns) as size_t,
    ) as *mut vl_uindex;
    let mut num: vl_uindex = 0 as libc::c_int as vl_uindex;
    y = 0 as libc::c_int as vl_uindex;
    while y < numRows {
        num = 0 as libc::c_int as vl_uindex;
        x = 0 as libc::c_int as vl_uindex;
        while x < numColumns {
            let mut r: libc::c_float = *image
                .offset(
                    x.wrapping_mul(columnStride).wrapping_add(y.wrapping_mul(rowStride))
                        as isize,
                );
            let mut x2: libc::c_float = x.wrapping_mul(x) as libc::c_float;
            let mut from_: libc::c_float = -vl_infinity_f.value;
            while num >= 1 as libc::c_int as libc::c_ulonglong {
                let mut x_: vl_uindex = *which
                    .offset(
                        num.wrapping_sub(1 as libc::c_int as libc::c_ulonglong) as isize,
                    );
                let mut x2_: libc::c_float = x_.wrapping_mul(x_) as libc::c_float;
                let mut r_: libc::c_float = *image
                    .offset(
                        x_
                            .wrapping_mul(columnStride)
                            .wrapping_add(y.wrapping_mul(rowStride)) as isize,
                    );
                let mut inters: libc::c_float = 0.;
                if r == r_ {
                    inters = (x.wrapping_add(x_) as libc::c_double / 2.0f64
                        + offset as libc::c_double) as libc::c_float;
                } else if coeff > 1.19209290E-07f32 {
                    inters = (r - r_ + coeff * (x2 - x2_))
                        / x.wrapping_sub(x_) as libc::c_float
                        / (2 as libc::c_int as libc::c_float * coeff) + offset;
                } else {
                    inters = if r < r_ {
                        -vl_infinity_f.value
                    } else {
                        vl_infinity_f.value
                    };
                }
                if inters
                    <= *from
                        .offset(
                            num.wrapping_sub(1 as libc::c_int as libc::c_ulonglong)
                                as isize,
                        )
                {
                    num = num.wrapping_sub(1);
                } else {
                    from_ = inters;
                    break;
                }
            }
            *which.offset(num as isize) = x;
            *from.offset(num as isize) = from_;
            *base.offset(num as isize) = r;
            if !indexes.is_null() {
                *baseIndexes
                    .offset(
                        num as isize,
                    ) = *indexes
                    .offset(
                        x
                            .wrapping_mul(columnStride)
                            .wrapping_add(y.wrapping_mul(rowStride)) as isize,
                    );
            }
            num = num.wrapping_add(1);
            x = x.wrapping_add(1);
        }
        *from.offset(num as isize) = vl_infinity_f.value;
        num = 0 as libc::c_int as vl_uindex;
        x = 0 as libc::c_int as vl_uindex;
        while x < numColumns {
            let mut delta: libc::c_double = 0.;
            while x as libc::c_float
                >= *from
                    .offset(
                        num.wrapping_add(1 as libc::c_int as libc::c_ulonglong) as isize,
                    )
            {
                num = num.wrapping_add(1);
            }
            delta = x as libc::c_double - *which.offset(num as isize) as libc::c_double
                - offset as libc::c_double;
            *distanceTransform
                .offset(
                    x.wrapping_mul(columnStride).wrapping_add(y.wrapping_mul(rowStride))
                        as isize,
                ) = (*base.offset(num as isize) as libc::c_double
                + coeff as libc::c_double * delta * delta) as libc::c_float;
            if !indexes.is_null() {
                *indexes
                    .offset(
                        x
                            .wrapping_mul(columnStride)
                            .wrapping_add(y.wrapping_mul(rowStride)) as isize,
                    ) = *baseIndexes.offset(num as isize);
            }
            x = x.wrapping_add(1);
        }
        y = y.wrapping_add(1);
    }
    vl_free(from as *mut libc::c_void);
    vl_free(which as *mut libc::c_void);
    vl_free(base as *mut libc::c_void);
    vl_free(baseIndexes as *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn vl_image_distance_transform_d(
    mut image: *const libc::c_double,
    mut numColumns: vl_size,
    mut numRows: vl_size,
    mut columnStride: vl_size,
    mut rowStride: vl_size,
    mut distanceTransform: *mut libc::c_double,
    mut indexes: *mut vl_uindex,
    mut coeff: libc::c_double,
    mut offset: libc::c_double,
) {
    let mut x: vl_uindex = 0;
    let mut y: vl_uindex = 0;
    let mut from: *mut libc::c_double = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numColumns.wrapping_add(1 as libc::c_int as libc::c_ulonglong))
            as size_t,
    ) as *mut libc::c_double;
    let mut base: *mut libc::c_double = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numColumns) as size_t,
    ) as *mut libc::c_double;
    let mut baseIndexes: *mut vl_uindex = vl_malloc(
        (::core::mem::size_of::<vl_uindex>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numColumns) as size_t,
    ) as *mut vl_uindex;
    let mut which: *mut vl_uindex = vl_malloc(
        (::core::mem::size_of::<vl_uindex>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numColumns) as size_t,
    ) as *mut vl_uindex;
    let mut num: vl_uindex = 0 as libc::c_int as vl_uindex;
    y = 0 as libc::c_int as vl_uindex;
    while y < numRows {
        num = 0 as libc::c_int as vl_uindex;
        x = 0 as libc::c_int as vl_uindex;
        while x < numColumns {
            let mut r: libc::c_double = *image
                .offset(
                    x.wrapping_mul(columnStride).wrapping_add(y.wrapping_mul(rowStride))
                        as isize,
                );
            let mut x2: libc::c_double = x.wrapping_mul(x) as libc::c_double;
            let mut from_: libc::c_double = -vl_infinity_d.value;
            while num >= 1 as libc::c_int as libc::c_ulonglong {
                let mut x_: vl_uindex = *which
                    .offset(
                        num.wrapping_sub(1 as libc::c_int as libc::c_ulonglong) as isize,
                    );
                let mut x2_: libc::c_double = x_.wrapping_mul(x_) as libc::c_double;
                let mut r_: libc::c_double = *image
                    .offset(
                        x_
                            .wrapping_mul(columnStride)
                            .wrapping_add(y.wrapping_mul(rowStride)) as isize,
                    );
                let mut inters: libc::c_double = 0.;
                if r == r_ {
                    inters = x.wrapping_add(x_) as libc::c_double / 2.0f64 + offset;
                } else if coeff > 2.220446049250313e-16f64 {
                    inters = (r - r_ + coeff * (x2 - x2_))
                        / x.wrapping_sub(x_) as libc::c_double
                        / (2 as libc::c_int as libc::c_double * coeff) + offset;
                } else {
                    inters = if r < r_ {
                        -vl_infinity_d.value
                    } else {
                        vl_infinity_d.value
                    };
                }
                if inters
                    <= *from
                        .offset(
                            num.wrapping_sub(1 as libc::c_int as libc::c_ulonglong)
                                as isize,
                        )
                {
                    num = num.wrapping_sub(1);
                } else {
                    from_ = inters;
                    break;
                }
            }
            *which.offset(num as isize) = x;
            *from.offset(num as isize) = from_;
            *base.offset(num as isize) = r;
            if !indexes.is_null() {
                *baseIndexes
                    .offset(
                        num as isize,
                    ) = *indexes
                    .offset(
                        x
                            .wrapping_mul(columnStride)
                            .wrapping_add(y.wrapping_mul(rowStride)) as isize,
                    );
            }
            num = num.wrapping_add(1);
            x = x.wrapping_add(1);
        }
        *from.offset(num as isize) = vl_infinity_d.value;
        num = 0 as libc::c_int as vl_uindex;
        x = 0 as libc::c_int as vl_uindex;
        while x < numColumns {
            let mut delta: libc::c_double = 0.;
            while x as libc::c_double
                >= *from
                    .offset(
                        num.wrapping_add(1 as libc::c_int as libc::c_ulonglong) as isize,
                    )
            {
                num = num.wrapping_add(1);
            }
            delta = x as libc::c_double - *which.offset(num as isize) as libc::c_double
                - offset;
            *distanceTransform
                .offset(
                    x.wrapping_mul(columnStride).wrapping_add(y.wrapping_mul(rowStride))
                        as isize,
                ) = *base.offset(num as isize) + coeff * delta * delta;
            if !indexes.is_null() {
                *indexes
                    .offset(
                        x
                            .wrapping_mul(columnStride)
                            .wrapping_add(y.wrapping_mul(rowStride)) as isize,
                    ) = *baseIndexes.offset(num as isize);
            }
            x = x.wrapping_add(1);
        }
        y = y.wrapping_add(1);
    }
    vl_free(from as *mut libc::c_void);
    vl_free(which as *mut libc::c_void);
    vl_free(base as *mut libc::c_void);
    vl_free(baseIndexes as *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn vl_imconvcoltri_f(
    mut dest: *mut libc::c_float,
    mut destStride: vl_size,
    mut image: *const libc::c_float,
    mut imageWidth: vl_size,
    mut imageHeight: vl_size,
    mut imageStride: vl_size,
    mut filterSize: vl_size,
    mut step: vl_size,
    mut flags: libc::c_uint,
) {
    let mut x: vl_index = 0;
    let mut y: vl_index = 0;
    let mut dheight: vl_index = 0;
    let mut transp: vl_bool = (flags
        & ((0x1 as libc::c_int) << 2 as libc::c_int) as libc::c_uint) as vl_bool;
    let mut zeropad: vl_bool = (flags & 0x3 as libc::c_int as libc::c_uint
        == ((0 as libc::c_int) << 0 as libc::c_int) as libc::c_uint) as libc::c_int;
    let mut scale: libc::c_float = (1.0f64
        / (filterSize as libc::c_double * filterSize as libc::c_double))
        as libc::c_float;
    let mut buffer: *mut libc::c_float = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(imageHeight.wrapping_add(filterSize)) as size_t,
    ) as *mut libc::c_float;
    buffer = buffer.offset(filterSize as isize);
    if imageHeight == 0 as libc::c_int as libc::c_ulonglong {
        return;
    }
    x = 0 as libc::c_int as vl_index;
    dheight = imageHeight
        .wrapping_sub(1 as libc::c_int as libc::c_ulonglong)
        .wrapping_div(step)
        .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as vl_index;
    while x < imageWidth as libc::c_int as libc::c_longlong {
        let mut imagei: *const libc::c_float = 0 as *const libc::c_float;
        imagei = image
            .offset(x as isize)
            .offset(
                imageStride
                    .wrapping_mul(
                        imageHeight.wrapping_sub(1 as libc::c_int as libc::c_ulonglong),
                    ) as isize,
            );
        *buffer
            .offset(
                imageHeight.wrapping_sub(1 as libc::c_int as libc::c_ulonglong) as isize,
            ) = *imagei;
        y = (imageHeight as libc::c_int - 2 as libc::c_int) as vl_index;
        while y >= 0 as libc::c_int as libc::c_longlong {
            imagei = imagei.offset(-(imageStride as isize));
            *buffer
                .offset(
                    y as isize,
                ) = *buffer.offset((y + 1 as libc::c_int as libc::c_longlong) as isize)
                + *imagei;
            y -= 1;
        }
        if zeropad != 0 {
            while y >= -(filterSize as libc::c_int) as libc::c_longlong {
                *buffer
                    .offset(
                        y as isize,
                    ) = *buffer
                    .offset((y + 1 as libc::c_int as libc::c_longlong) as isize);
                y -= 1;
            }
        } else {
            while y >= -(filterSize as libc::c_int) as libc::c_longlong {
                *buffer
                    .offset(
                        y as isize,
                    ) = *buffer
                    .offset((y + 1 as libc::c_int as libc::c_longlong) as isize)
                    + *imagei;
                y -= 1;
            }
        }
        y = -(filterSize as libc::c_int) as vl_index;
        while y
            < (imageHeight as libc::c_int - filterSize as libc::c_int)
                as libc::c_longlong
        {
            *buffer
                .offset(
                    y as isize,
                ) = *buffer.offset(y as isize)
                - *buffer
                    .offset((y as libc::c_ulonglong).wrapping_add(filterSize) as isize);
            y += 1;
        }
        if zeropad == 0 {
            y = (imageHeight as libc::c_int - filterSize as libc::c_int) as vl_index;
            while y < imageHeight as libc::c_int as libc::c_longlong {
                *buffer
                    .offset(
                        y as isize,
                    ) = *buffer.offset(y as isize)
                    - *buffer
                        .offset(
                            imageHeight
                                .wrapping_sub(1 as libc::c_int as libc::c_ulonglong)
                                as isize,
                        )
                        * ((imageHeight as libc::c_int - filterSize as libc::c_int)
                            as libc::c_longlong - y) as libc::c_float;
                y += 1;
            }
        }
        y = (-(filterSize as libc::c_int) + 1 as libc::c_int) as vl_index;
        while y < imageHeight as libc::c_int as libc::c_longlong {
            *buffer.offset(y as isize)
                += *buffer.offset((y - 1 as libc::c_int as libc::c_longlong) as isize);
            y += 1;
        }
        let mut stride: vl_size = if transp != 0 {
            1 as libc::c_int as libc::c_ulonglong
        } else {
            destStride
        };
        dest = dest.offset((dheight as libc::c_ulonglong).wrapping_mul(stride) as isize);
        y = step
            .wrapping_mul(
                (dheight - 1 as libc::c_int as libc::c_longlong) as libc::c_ulonglong,
            ) as vl_index;
        while y >= 0 as libc::c_int as libc::c_longlong {
            dest = dest.offset(-(stride as isize));
            *dest = scale
                * (*buffer.offset(y as isize)
                    - *buffer
                        .offset(
                            (y - filterSize as libc::c_int as libc::c_longlong) as isize,
                        ));
            y = (y as libc::c_ulonglong).wrapping_sub(step) as vl_index as vl_index;
        }
        dest = dest
            .offset(
                (if transp != 0 {
                    destStride
                } else {
                    1 as libc::c_int as libc::c_ulonglong
                }) as isize,
            );
        x += 1 as libc::c_int as libc::c_longlong;
    }
    vl_free(buffer.offset(-(filterSize as isize)) as *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn vl_imconvcoltri_d(
    mut dest: *mut libc::c_double,
    mut destStride: vl_size,
    mut image: *const libc::c_double,
    mut imageWidth: vl_size,
    mut imageHeight: vl_size,
    mut imageStride: vl_size,
    mut filterSize: vl_size,
    mut step: vl_size,
    mut flags: libc::c_uint,
) {
    let mut x: vl_index = 0;
    let mut y: vl_index = 0;
    let mut dheight: vl_index = 0;
    let mut transp: vl_bool = (flags
        & ((0x1 as libc::c_int) << 2 as libc::c_int) as libc::c_uint) as vl_bool;
    let mut zeropad: vl_bool = (flags & 0x3 as libc::c_int as libc::c_uint
        == ((0 as libc::c_int) << 0 as libc::c_int) as libc::c_uint) as libc::c_int;
    let mut scale: libc::c_double = 1.0f64
        / (filterSize as libc::c_double * filterSize as libc::c_double);
    let mut buffer: *mut libc::c_double = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(imageHeight.wrapping_add(filterSize)) as size_t,
    ) as *mut libc::c_double;
    buffer = buffer.offset(filterSize as isize);
    if imageHeight == 0 as libc::c_int as libc::c_ulonglong {
        return;
    }
    x = 0 as libc::c_int as vl_index;
    dheight = imageHeight
        .wrapping_sub(1 as libc::c_int as libc::c_ulonglong)
        .wrapping_div(step)
        .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as vl_index;
    while x < imageWidth as libc::c_int as libc::c_longlong {
        let mut imagei: *const libc::c_double = 0 as *const libc::c_double;
        imagei = image
            .offset(x as isize)
            .offset(
                imageStride
                    .wrapping_mul(
                        imageHeight.wrapping_sub(1 as libc::c_int as libc::c_ulonglong),
                    ) as isize,
            );
        *buffer
            .offset(
                imageHeight.wrapping_sub(1 as libc::c_int as libc::c_ulonglong) as isize,
            ) = *imagei;
        y = (imageHeight as libc::c_int - 2 as libc::c_int) as vl_index;
        while y >= 0 as libc::c_int as libc::c_longlong {
            imagei = imagei.offset(-(imageStride as isize));
            *buffer
                .offset(
                    y as isize,
                ) = *buffer.offset((y + 1 as libc::c_int as libc::c_longlong) as isize)
                + *imagei;
            y -= 1;
        }
        if zeropad != 0 {
            while y >= -(filterSize as libc::c_int) as libc::c_longlong {
                *buffer
                    .offset(
                        y as isize,
                    ) = *buffer
                    .offset((y + 1 as libc::c_int as libc::c_longlong) as isize);
                y -= 1;
            }
        } else {
            while y >= -(filterSize as libc::c_int) as libc::c_longlong {
                *buffer
                    .offset(
                        y as isize,
                    ) = *buffer
                    .offset((y + 1 as libc::c_int as libc::c_longlong) as isize)
                    + *imagei;
                y -= 1;
            }
        }
        y = -(filterSize as libc::c_int) as vl_index;
        while y
            < (imageHeight as libc::c_int - filterSize as libc::c_int)
                as libc::c_longlong
        {
            *buffer
                .offset(
                    y as isize,
                ) = *buffer.offset(y as isize)
                - *buffer
                    .offset((y as libc::c_ulonglong).wrapping_add(filterSize) as isize);
            y += 1;
        }
        if zeropad == 0 {
            y = (imageHeight as libc::c_int - filterSize as libc::c_int) as vl_index;
            while y < imageHeight as libc::c_int as libc::c_longlong {
                *buffer
                    .offset(
                        y as isize,
                    ) = *buffer.offset(y as isize)
                    - *buffer
                        .offset(
                            imageHeight
                                .wrapping_sub(1 as libc::c_int as libc::c_ulonglong)
                                as isize,
                        )
                        * ((imageHeight as libc::c_int - filterSize as libc::c_int)
                            as libc::c_longlong - y) as libc::c_double;
                y += 1;
            }
        }
        y = (-(filterSize as libc::c_int) + 1 as libc::c_int) as vl_index;
        while y < imageHeight as libc::c_int as libc::c_longlong {
            *buffer.offset(y as isize)
                += *buffer.offset((y - 1 as libc::c_int as libc::c_longlong) as isize);
            y += 1;
        }
        let mut stride: vl_size = if transp != 0 {
            1 as libc::c_int as libc::c_ulonglong
        } else {
            destStride
        };
        dest = dest.offset((dheight as libc::c_ulonglong).wrapping_mul(stride) as isize);
        y = step
            .wrapping_mul(
                (dheight - 1 as libc::c_int as libc::c_longlong) as libc::c_ulonglong,
            ) as vl_index;
        while y >= 0 as libc::c_int as libc::c_longlong {
            dest = dest.offset(-(stride as isize));
            *dest = scale
                * (*buffer.offset(y as isize)
                    - *buffer
                        .offset(
                            (y - filterSize as libc::c_int as libc::c_longlong) as isize,
                        ));
            y = (y as libc::c_ulonglong).wrapping_sub(step) as vl_index as vl_index;
        }
        dest = dest
            .offset(
                (if transp != 0 {
                    destStride
                } else {
                    1 as libc::c_int as libc::c_ulonglong
                }) as isize,
            );
        x += 1 as libc::c_int as libc::c_longlong;
    }
    vl_free(buffer.offset(-(filterSize as isize)) as *mut libc::c_void);
}
unsafe extern "C" fn _vl_new_gaussian_fitler_d(
    mut size: *mut vl_size,
    mut sigma: libc::c_double,
) -> *mut libc::c_double {
    let mut filter: *mut libc::c_double = 0 as *mut libc::c_double;
    let mut mass: libc::c_double = 1.0f64;
    let mut i: vl_index = 0;
    let mut width: vl_size = f64::ceil(sigma * 3.0f64) as vl_size;
    *size = (2 as libc::c_int as libc::c_ulonglong)
        .wrapping_mul(width)
        .wrapping_add(1 as libc::c_int as libc::c_ulonglong);
    if !size.is_null() {} else {
        __assert_fail(
            b"size\0" as *const u8 as *const libc::c_char,
            b"vl/imopv.c\0" as *const u8 as *const libc::c_char,
            629 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 53],
                &[libc::c_char; 53],
            >(b"double *_vl_new_gaussian_fitler_d(vl_size *, double)\0"))
                .as_ptr(),
        );
    }
    filter = vl_malloc(
        (*size)
            .wrapping_mul(
                ::core::mem::size_of::<libc::c_double>() as libc::c_ulong
                    as libc::c_ulonglong,
            ) as size_t,
    ) as *mut libc::c_double;
    *filter.offset(width as isize) = 1.0f64;
    i = 1 as libc::c_int as vl_index;
    while i <= width as libc::c_int as libc::c_longlong {
        let mut x: libc::c_double = i as libc::c_double / sigma;
        let mut g: libc::c_double = exp(-0.5f64 * x * x);
        mass += g + g;
        *filter.offset(width.wrapping_sub(i as libc::c_ulonglong) as isize) = g;
        *filter.offset(width.wrapping_add(i as libc::c_ulonglong) as isize) = g;
        i += 1;
    }
    i = 0 as libc::c_int as vl_index;
    while i < *size as libc::c_int as libc::c_longlong {
        *filter.offset(i as isize) /= mass;
        i += 1;
    }
    return filter;
}
unsafe extern "C" fn _vl_new_gaussian_fitler_f(
    mut size: *mut vl_size,
    mut sigma: libc::c_double,
) -> *mut libc::c_float {
    let mut filter: *mut libc::c_float = 0 as *mut libc::c_float;
    let mut mass: libc::c_float = 1.0f64 as libc::c_float;
    let mut i: vl_index = 0;
    let mut width: vl_size = f64::ceil(sigma * 3.0f64) as vl_size;
    *size = (2 as libc::c_int as libc::c_ulonglong)
        .wrapping_mul(width)
        .wrapping_add(1 as libc::c_int as libc::c_ulonglong);
    if !size.is_null() {} else {
        __assert_fail(
            b"size\0" as *const u8 as *const libc::c_char,
            b"vl/imopv.c\0" as *const u8 as *const libc::c_char,
            629 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 52],
                &[libc::c_char; 52],
            >(b"float *_vl_new_gaussian_fitler_f(vl_size *, double)\0"))
                .as_ptr(),
        );
    }
    filter = vl_malloc(
        (*size)
            .wrapping_mul(
                ::core::mem::size_of::<libc::c_float>() as libc::c_ulong
                    as libc::c_ulonglong,
            ) as size_t,
    ) as *mut libc::c_float;
    *filter.offset(width as isize) = 1.0f64 as libc::c_float;
    i = 1 as libc::c_int as vl_index;
    while i <= width as libc::c_int as libc::c_longlong {
        let mut x: libc::c_double = i as libc::c_double / sigma;
        let mut g: libc::c_double = exp(-0.5f64 * x * x);
        mass = (mass as libc::c_double + (g + g)) as libc::c_float;
        *filter
            .offset(
                width.wrapping_sub(i as libc::c_ulonglong) as isize,
            ) = g as libc::c_float;
        *filter
            .offset(
                width.wrapping_add(i as libc::c_ulonglong) as isize,
            ) = g as libc::c_float;
        i += 1;
    }
    i = 0 as libc::c_int as vl_index;
    while i < *size as libc::c_int as libc::c_longlong {
        *filter.offset(i as isize) /= mass;
        i += 1;
    }
    return filter;
}
#[no_mangle]
pub unsafe extern "C" fn vl_imsmooth_f(
    mut smoothed: *mut libc::c_float,
    mut smoothedStride: vl_size,
    mut image: *const libc::c_float,
    mut width: vl_size,
    mut height: vl_size,
    mut stride: vl_size,
    mut sigmax: libc::c_double,
    mut sigmay: libc::c_double,
) {
    let mut filterx: *mut libc::c_float = 0 as *mut libc::c_float;
    let mut filtery: *mut libc::c_float = 0 as *mut libc::c_float;
    let mut buffer: *mut libc::c_float = 0 as *mut libc::c_float;
    let mut sizex: vl_size = 0;
    let mut sizey: vl_size = 0;
    filterx = _vl_new_gaussian_fitler_f(&mut sizex, sigmax);
    if sigmax == sigmay {
        filtery = filterx;
        sizey = sizex;
    } else {
        filtery = _vl_new_gaussian_fitler_f(&mut sizey, sigmay);
    }
    buffer = vl_malloc(
        width
            .wrapping_mul(height)
            .wrapping_mul(
                ::core::mem::size_of::<libc::c_float>() as libc::c_ulong
                    as libc::c_ulonglong,
            ) as size_t,
    ) as *mut libc::c_float;
    vl_imconvcol_vf(
        buffer,
        height,
        image,
        width,
        height,
        stride,
        filtery,
        (-(sizey as libc::c_int - 1 as libc::c_int) / 2 as libc::c_int) as vl_index,
        ((sizey as libc::c_int - 1 as libc::c_int) / 2 as libc::c_int) as vl_index,
        1 as libc::c_int,
        ((0x1 as libc::c_int) << 0 as libc::c_int
            | (0x1 as libc::c_int) << 2 as libc::c_int) as libc::c_uint,
    );
    vl_imconvcol_vf(
        smoothed,
        smoothedStride,
        buffer,
        height,
        width,
        height,
        filterx,
        (-(sizex as libc::c_int - 1 as libc::c_int) / 2 as libc::c_int) as vl_index,
        ((sizex as libc::c_int - 1 as libc::c_int) / 2 as libc::c_int) as vl_index,
        1 as libc::c_int,
        ((0x1 as libc::c_int) << 0 as libc::c_int
            | (0x1 as libc::c_int) << 2 as libc::c_int) as libc::c_uint,
    );
    vl_free(buffer as *mut libc::c_void);
    vl_free(filterx as *mut libc::c_void);
    if sigmax != sigmay {
        vl_free(filtery as *mut libc::c_void);
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_imsmooth_d(
    mut smoothed: *mut libc::c_double,
    mut smoothedStride: vl_size,
    mut image: *const libc::c_double,
    mut width: vl_size,
    mut height: vl_size,
    mut stride: vl_size,
    mut sigmax: libc::c_double,
    mut sigmay: libc::c_double,
) {
    let mut filterx: *mut libc::c_double = 0 as *mut libc::c_double;
    let mut filtery: *mut libc::c_double = 0 as *mut libc::c_double;
    let mut buffer: *mut libc::c_double = 0 as *mut libc::c_double;
    let mut sizex: vl_size = 0;
    let mut sizey: vl_size = 0;
    filterx = _vl_new_gaussian_fitler_d(&mut sizex, sigmax);
    if sigmax == sigmay {
        filtery = filterx;
        sizey = sizex;
    } else {
        filtery = _vl_new_gaussian_fitler_d(&mut sizey, sigmay);
    }
    buffer = vl_malloc(
        width
            .wrapping_mul(height)
            .wrapping_mul(
                ::core::mem::size_of::<libc::c_double>() as libc::c_ulong
                    as libc::c_ulonglong,
            ) as size_t,
    ) as *mut libc::c_double;
    vl_imconvcol_vd(
        buffer,
        height,
        image,
        width,
        height,
        stride,
        filtery,
        (-(sizey as libc::c_int - 1 as libc::c_int) / 2 as libc::c_int) as vl_index,
        ((sizey as libc::c_int - 1 as libc::c_int) / 2 as libc::c_int) as vl_index,
        1 as libc::c_int,
        ((0x1 as libc::c_int) << 0 as libc::c_int
            | (0x1 as libc::c_int) << 2 as libc::c_int) as libc::c_uint,
    );
    vl_imconvcol_vd(
        smoothed,
        smoothedStride,
        buffer,
        height,
        width,
        height,
        filterx,
        (-(sizex as libc::c_int - 1 as libc::c_int) / 2 as libc::c_int) as vl_index,
        ((sizex as libc::c_int - 1 as libc::c_int) / 2 as libc::c_int) as vl_index,
        1 as libc::c_int,
        ((0x1 as libc::c_int) << 0 as libc::c_int
            | (0x1 as libc::c_int) << 2 as libc::c_int) as libc::c_uint,
    );
    vl_free(buffer as *mut libc::c_void);
    vl_free(filterx as *mut libc::c_void);
    if sigmax != sigmay {
        vl_free(filtery as *mut libc::c_void);
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_imgradient_d(
    mut xGradient: *mut libc::c_double,
    mut yGradient: *mut libc::c_double,
    mut gradWidthStride: vl_size,
    mut gradHeightStride: vl_size,
    mut image: *const libc::c_double,
    mut imageWidth: vl_size,
    mut imageHeight: vl_size,
    mut imageStride: vl_size,
) {
    let xo: vl_index = 1 as libc::c_int as vl_index;
    let yo: vl_index = imageStride as vl_index;
    let w: vl_size = imageWidth;
    let h: vl_size = imageHeight;
    let mut src: *const libc::c_double = 0 as *const libc::c_double;
    let mut end: *const libc::c_double = 0 as *const libc::c_double;
    let mut pgrad_x: *mut libc::c_double = 0 as *mut libc::c_double;
    let mut pgrad_y: *mut libc::c_double = 0 as *mut libc::c_double;
    let mut y: vl_size = 0;
    src = image;
    pgrad_x = xGradient;
    pgrad_y = yGradient;
    *pgrad_x = *src.offset(xo as isize) - *src.offset(0 as libc::c_int as isize);
    pgrad_x = pgrad_x.offset(gradWidthStride as isize);
    *pgrad_y = *src.offset(yo as isize) - *src.offset(0 as libc::c_int as isize);
    pgrad_y = pgrad_y.offset(gradWidthStride as isize);
    src = src.offset(1);
    end = src
        .offset(-(1 as libc::c_int as isize))
        .offset(w as isize)
        .offset(-(1 as libc::c_int as isize));
    while src < end {
        *pgrad_x = 0.5f64 * (*src.offset(xo as isize) - *src.offset(-xo as isize));
        pgrad_x = pgrad_x.offset(gradWidthStride as isize);
        *pgrad_y = *src.offset(yo as isize) - *src.offset(0 as libc::c_int as isize);
        pgrad_y = pgrad_y.offset(gradWidthStride as isize);
        src = src.offset(1);
    }
    *pgrad_x = *src.offset(0 as libc::c_int as isize) - *src.offset(-xo as isize);
    pgrad_x = pgrad_x.offset(gradWidthStride as isize);
    *pgrad_y = *src.offset(yo as isize) - *src.offset(0 as libc::c_int as isize);
    pgrad_y = pgrad_y.offset(gradWidthStride as isize);
    src = src.offset(1);
    xGradient = xGradient.offset(gradHeightStride as isize);
    pgrad_x = xGradient;
    yGradient = yGradient.offset(gradHeightStride as isize);
    pgrad_y = yGradient;
    image = image.offset(yo as isize);
    src = image;
    y = 1 as libc::c_int as vl_size;
    while y < h.wrapping_sub(1 as libc::c_int as libc::c_ulonglong) {
        *pgrad_x = *src.offset(xo as isize) - *src.offset(0 as libc::c_int as isize);
        pgrad_x = pgrad_x.offset(gradWidthStride as isize);
        *pgrad_y = 0.5f64 * (*src.offset(yo as isize) - *src.offset(-yo as isize));
        pgrad_y = pgrad_y.offset(gradWidthStride as isize);
        src = src.offset(1);
        end = src
            .offset(-(1 as libc::c_int as isize))
            .offset(w as isize)
            .offset(-(1 as libc::c_int as isize));
        while src < end {
            *pgrad_x = 0.5f64 * (*src.offset(xo as isize) - *src.offset(-xo as isize));
            pgrad_x = pgrad_x.offset(gradWidthStride as isize);
            *pgrad_y = 0.5f64 * (*src.offset(yo as isize) - *src.offset(-yo as isize));
            pgrad_y = pgrad_y.offset(gradWidthStride as isize);
            src = src.offset(1);
        }
        *pgrad_x = *src.offset(0 as libc::c_int as isize) - *src.offset(-xo as isize);
        pgrad_x = pgrad_x.offset(gradWidthStride as isize);
        *pgrad_y = 0.5f64 * (*src.offset(yo as isize) - *src.offset(-yo as isize));
        pgrad_y = pgrad_y.offset(gradWidthStride as isize);
        src = src.offset(1);
        xGradient = xGradient.offset(gradHeightStride as isize);
        pgrad_x = xGradient;
        yGradient = yGradient.offset(gradHeightStride as isize);
        pgrad_y = yGradient;
        image = image.offset(yo as isize);
        src = image;
        y = y.wrapping_add(1);
    }
    *pgrad_x = *src.offset(xo as isize) - *src.offset(0 as libc::c_int as isize);
    pgrad_x = pgrad_x.offset(gradWidthStride as isize);
    *pgrad_y = *src.offset(0 as libc::c_int as isize) - *src.offset(-yo as isize);
    pgrad_y = pgrad_y.offset(gradWidthStride as isize);
    src = src.offset(1);
    end = src
        .offset(-(1 as libc::c_int as isize))
        .offset(w as isize)
        .offset(-(1 as libc::c_int as isize));
    while src < end {
        *pgrad_x = 0.5f64 * (*src.offset(xo as isize) - *src.offset(-xo as isize));
        pgrad_x = pgrad_x.offset(gradWidthStride as isize);
        *pgrad_y = *src.offset(0 as libc::c_int as isize) - *src.offset(-yo as isize);
        pgrad_y = pgrad_y.offset(gradWidthStride as isize);
        src = src.offset(1);
    }
    *pgrad_x = *src.offset(0 as libc::c_int as isize) - *src.offset(-xo as isize);
    *pgrad_y = *src.offset(0 as libc::c_int as isize) - *src.offset(-yo as isize);
}
#[no_mangle]
pub unsafe extern "C" fn vl_imgradient_f(
    mut xGradient: *mut libc::c_float,
    mut yGradient: *mut libc::c_float,
    mut gradWidthStride: vl_size,
    mut gradHeightStride: vl_size,
    mut image: *const libc::c_float,
    mut imageWidth: vl_size,
    mut imageHeight: vl_size,
    mut imageStride: vl_size,
) {
    let xo: vl_index = 1 as libc::c_int as vl_index;
    let yo: vl_index = imageStride as vl_index;
    let w: vl_size = imageWidth;
    let h: vl_size = imageHeight;
    let mut src: *const libc::c_float = 0 as *const libc::c_float;
    let mut end: *const libc::c_float = 0 as *const libc::c_float;
    let mut pgrad_x: *mut libc::c_float = 0 as *mut libc::c_float;
    let mut pgrad_y: *mut libc::c_float = 0 as *mut libc::c_float;
    let mut y: vl_size = 0;
    src = image;
    pgrad_x = xGradient;
    pgrad_y = yGradient;
    *pgrad_x = *src.offset(xo as isize) - *src.offset(0 as libc::c_int as isize);
    pgrad_x = pgrad_x.offset(gradWidthStride as isize);
    *pgrad_y = *src.offset(yo as isize) - *src.offset(0 as libc::c_int as isize);
    pgrad_y = pgrad_y.offset(gradWidthStride as isize);
    src = src.offset(1);
    end = src
        .offset(-(1 as libc::c_int as isize))
        .offset(w as isize)
        .offset(-(1 as libc::c_int as isize));
    while src < end {
        *pgrad_x = (0.5f64
            * (*src.offset(xo as isize) - *src.offset(-xo as isize)) as libc::c_double)
            as libc::c_float;
        pgrad_x = pgrad_x.offset(gradWidthStride as isize);
        *pgrad_y = *src.offset(yo as isize) - *src.offset(0 as libc::c_int as isize);
        pgrad_y = pgrad_y.offset(gradWidthStride as isize);
        src = src.offset(1);
    }
    *pgrad_x = *src.offset(0 as libc::c_int as isize) - *src.offset(-xo as isize);
    pgrad_x = pgrad_x.offset(gradWidthStride as isize);
    *pgrad_y = *src.offset(yo as isize) - *src.offset(0 as libc::c_int as isize);
    pgrad_y = pgrad_y.offset(gradWidthStride as isize);
    src = src.offset(1);
    xGradient = xGradient.offset(gradHeightStride as isize);
    pgrad_x = xGradient;
    yGradient = yGradient.offset(gradHeightStride as isize);
    pgrad_y = yGradient;
    image = image.offset(yo as isize);
    src = image;
    y = 1 as libc::c_int as vl_size;
    while y < h.wrapping_sub(1 as libc::c_int as libc::c_ulonglong) {
        *pgrad_x = *src.offset(xo as isize) - *src.offset(0 as libc::c_int as isize);
        pgrad_x = pgrad_x.offset(gradWidthStride as isize);
        *pgrad_y = (0.5f64
            * (*src.offset(yo as isize) - *src.offset(-yo as isize)) as libc::c_double)
            as libc::c_float;
        pgrad_y = pgrad_y.offset(gradWidthStride as isize);
        src = src.offset(1);
        end = src
            .offset(-(1 as libc::c_int as isize))
            .offset(w as isize)
            .offset(-(1 as libc::c_int as isize));
        while src < end {
            *pgrad_x = (0.5f64
                * (*src.offset(xo as isize) - *src.offset(-xo as isize))
                    as libc::c_double) as libc::c_float;
            pgrad_x = pgrad_x.offset(gradWidthStride as isize);
            *pgrad_y = (0.5f64
                * (*src.offset(yo as isize) - *src.offset(-yo as isize))
                    as libc::c_double) as libc::c_float;
            pgrad_y = pgrad_y.offset(gradWidthStride as isize);
            src = src.offset(1);
        }
        *pgrad_x = *src.offset(0 as libc::c_int as isize) - *src.offset(-xo as isize);
        pgrad_x = pgrad_x.offset(gradWidthStride as isize);
        *pgrad_y = (0.5f64
            * (*src.offset(yo as isize) - *src.offset(-yo as isize)) as libc::c_double)
            as libc::c_float;
        pgrad_y = pgrad_y.offset(gradWidthStride as isize);
        src = src.offset(1);
        xGradient = xGradient.offset(gradHeightStride as isize);
        pgrad_x = xGradient;
        yGradient = yGradient.offset(gradHeightStride as isize);
        pgrad_y = yGradient;
        image = image.offset(yo as isize);
        src = image;
        y = y.wrapping_add(1);
    }
    *pgrad_x = *src.offset(xo as isize) - *src.offset(0 as libc::c_int as isize);
    pgrad_x = pgrad_x.offset(gradWidthStride as isize);
    *pgrad_y = *src.offset(0 as libc::c_int as isize) - *src.offset(-yo as isize);
    pgrad_y = pgrad_y.offset(gradWidthStride as isize);
    src = src.offset(1);
    end = src
        .offset(-(1 as libc::c_int as isize))
        .offset(w as isize)
        .offset(-(1 as libc::c_int as isize));
    while src < end {
        *pgrad_x = (0.5f64
            * (*src.offset(xo as isize) - *src.offset(-xo as isize)) as libc::c_double)
            as libc::c_float;
        pgrad_x = pgrad_x.offset(gradWidthStride as isize);
        *pgrad_y = *src.offset(0 as libc::c_int as isize) - *src.offset(-yo as isize);
        pgrad_y = pgrad_y.offset(gradWidthStride as isize);
        src = src.offset(1);
    }
    *pgrad_x = *src.offset(0 as libc::c_int as isize) - *src.offset(-xo as isize);
    *pgrad_y = *src.offset(0 as libc::c_int as isize) - *src.offset(-yo as isize);
}
#[no_mangle]
pub unsafe extern "C" fn vl_imgradient_polar_d(
    mut gradientModulus: *mut libc::c_double,
    mut gradientAngle: *mut libc::c_double,
    mut gradientHorizontalStride: vl_size,
    mut gradHeightStride: vl_size,
    mut image: *const libc::c_double,
    mut imageWidth: vl_size,
    mut imageHeight: vl_size,
    mut imageStride: vl_size,
) {
    let xo: vl_index = 1 as libc::c_int as vl_index;
    let yo: vl_index = imageStride as vl_index;
    let w: vl_size = imageWidth;
    let h: vl_size = imageHeight;
    let mut src: *const libc::c_double = 0 as *const libc::c_double;
    let mut end: *const libc::c_double = 0 as *const libc::c_double;
    let mut pgrad_angl: *mut libc::c_double = 0 as *mut libc::c_double;
    let mut pgrad_ampl: *mut libc::c_double = 0 as *mut libc::c_double;
    let mut gx: libc::c_double = 0.;
    let mut gy: libc::c_double = 0.;
    let mut y: vl_size = 0;
    src = image;
    pgrad_angl = gradientAngle;
    pgrad_ampl = gradientModulus;
    gx = *src.offset(xo as isize) - *src.offset(0 as libc::c_int as isize);
    gy = *src.offset(yo as isize) - *src.offset(0 as libc::c_int as isize);
    *pgrad_ampl = vl_fast_sqrt_f((gx * gx + gy * gy) as libc::c_float) as libc::c_double;
    pgrad_ampl = pgrad_ampl.offset(gradientHorizontalStride as isize);
    *pgrad_angl = vl_mod_2pi_f(
        (vl_fast_atan2_f(gy as libc::c_float, gx as libc::c_float) as libc::c_double
            + 2 as libc::c_int as libc::c_double * 3.141592653589793f64) as libc::c_float,
    ) as libc::c_double;
    pgrad_angl = pgrad_angl.offset(gradientHorizontalStride as isize);
    src = src.offset(1);
    end = src
        .offset(-(1 as libc::c_int as isize))
        .offset(w as isize)
        .offset(-(1 as libc::c_int as isize));
    while src < end {
        gx = 0.5f64 * (*src.offset(xo as isize) - *src.offset(-xo as isize));
        gy = *src.offset(yo as isize) - *src.offset(0 as libc::c_int as isize);
        *pgrad_ampl = vl_fast_sqrt_f((gx * gx + gy * gy) as libc::c_float)
            as libc::c_double;
        pgrad_ampl = pgrad_ampl.offset(gradientHorizontalStride as isize);
        *pgrad_angl = vl_mod_2pi_f(
            (vl_fast_atan2_f(gy as libc::c_float, gx as libc::c_float) as libc::c_double
                + 2 as libc::c_int as libc::c_double * 3.141592653589793f64)
                as libc::c_float,
        ) as libc::c_double;
        pgrad_angl = pgrad_angl.offset(gradientHorizontalStride as isize);
        src = src.offset(1);
    }
    gx = *src.offset(0 as libc::c_int as isize) - *src.offset(-xo as isize);
    gy = *src.offset(yo as isize) - *src.offset(0 as libc::c_int as isize);
    *pgrad_ampl = vl_fast_sqrt_f((gx * gx + gy * gy) as libc::c_float) as libc::c_double;
    pgrad_ampl = pgrad_ampl.offset(gradientHorizontalStride as isize);
    *pgrad_angl = vl_mod_2pi_f(
        (vl_fast_atan2_f(gy as libc::c_float, gx as libc::c_float) as libc::c_double
            + 2 as libc::c_int as libc::c_double * 3.141592653589793f64) as libc::c_float,
    ) as libc::c_double;
    pgrad_angl = pgrad_angl.offset(gradientHorizontalStride as isize);
    src = src.offset(1);
    gradientModulus = gradientModulus.offset(gradHeightStride as isize);
    pgrad_ampl = gradientModulus;
    gradientAngle = gradientAngle.offset(gradHeightStride as isize);
    pgrad_angl = gradientAngle;
    image = image.offset(imageStride as isize);
    src = image;
    y = 1 as libc::c_int as vl_size;
    while y < h.wrapping_sub(1 as libc::c_int as libc::c_ulonglong) {
        gx = *src.offset(xo as isize) - *src.offset(0 as libc::c_int as isize);
        gy = 0.5f64 * (*src.offset(yo as isize) - *src.offset(-yo as isize));
        *pgrad_ampl = vl_fast_sqrt_f((gx * gx + gy * gy) as libc::c_float)
            as libc::c_double;
        pgrad_ampl = pgrad_ampl.offset(gradientHorizontalStride as isize);
        *pgrad_angl = vl_mod_2pi_f(
            (vl_fast_atan2_f(gy as libc::c_float, gx as libc::c_float) as libc::c_double
                + 2 as libc::c_int as libc::c_double * 3.141592653589793f64)
                as libc::c_float,
        ) as libc::c_double;
        pgrad_angl = pgrad_angl.offset(gradientHorizontalStride as isize);
        src = src.offset(1);
        end = src
            .offset(-(1 as libc::c_int as isize))
            .offset(w as isize)
            .offset(-(1 as libc::c_int as isize));
        while src < end {
            gx = 0.5f64 * (*src.offset(xo as isize) - *src.offset(-xo as isize));
            gy = 0.5f64 * (*src.offset(yo as isize) - *src.offset(-yo as isize));
            *pgrad_ampl = vl_fast_sqrt_f((gx * gx + gy * gy) as libc::c_float)
                as libc::c_double;
            pgrad_ampl = pgrad_ampl.offset(gradientHorizontalStride as isize);
            *pgrad_angl = vl_mod_2pi_f(
                (vl_fast_atan2_f(gy as libc::c_float, gx as libc::c_float)
                    as libc::c_double
                    + 2 as libc::c_int as libc::c_double * 3.141592653589793f64)
                    as libc::c_float,
            ) as libc::c_double;
            pgrad_angl = pgrad_angl.offset(gradientHorizontalStride as isize);
            src = src.offset(1);
        }
        gx = *src.offset(0 as libc::c_int as isize) - *src.offset(-xo as isize);
        gy = 0.5f64 * (*src.offset(yo as isize) - *src.offset(-yo as isize));
        *pgrad_ampl = vl_fast_sqrt_f((gx * gx + gy * gy) as libc::c_float)
            as libc::c_double;
        pgrad_ampl = pgrad_ampl.offset(gradientHorizontalStride as isize);
        *pgrad_angl = vl_mod_2pi_f(
            (vl_fast_atan2_f(gy as libc::c_float, gx as libc::c_float) as libc::c_double
                + 2 as libc::c_int as libc::c_double * 3.141592653589793f64)
                as libc::c_float,
        ) as libc::c_double;
        pgrad_angl = pgrad_angl.offset(gradientHorizontalStride as isize);
        src = src.offset(1);
        gradientModulus = gradientModulus.offset(gradHeightStride as isize);
        pgrad_ampl = gradientModulus;
        gradientAngle = gradientAngle.offset(gradHeightStride as isize);
        pgrad_angl = gradientAngle;
        image = image.offset(imageStride as isize);
        src = image;
        y = y.wrapping_add(1);
    }
    gx = *src.offset(xo as isize) - *src.offset(0 as libc::c_int as isize);
    gy = *src.offset(0 as libc::c_int as isize) - *src.offset(-yo as isize);
    *pgrad_ampl = vl_fast_sqrt_f((gx * gx + gy * gy) as libc::c_float) as libc::c_double;
    pgrad_ampl = pgrad_ampl.offset(gradientHorizontalStride as isize);
    *pgrad_angl = vl_mod_2pi_f(
        (vl_fast_atan2_f(gy as libc::c_float, gx as libc::c_float) as libc::c_double
            + 2 as libc::c_int as libc::c_double * 3.141592653589793f64) as libc::c_float,
    ) as libc::c_double;
    pgrad_angl = pgrad_angl.offset(gradientHorizontalStride as isize);
    src = src.offset(1);
    end = src
        .offset(-(1 as libc::c_int as isize))
        .offset(w as isize)
        .offset(-(1 as libc::c_int as isize));
    while src < end {
        gx = 0.5f64 * (*src.offset(xo as isize) - *src.offset(-xo as isize));
        gy = *src.offset(0 as libc::c_int as isize) - *src.offset(-yo as isize);
        *pgrad_ampl = vl_fast_sqrt_f((gx * gx + gy * gy) as libc::c_float)
            as libc::c_double;
        pgrad_ampl = pgrad_ampl.offset(gradientHorizontalStride as isize);
        *pgrad_angl = vl_mod_2pi_f(
            (vl_fast_atan2_f(gy as libc::c_float, gx as libc::c_float) as libc::c_double
                + 2 as libc::c_int as libc::c_double * 3.141592653589793f64)
                as libc::c_float,
        ) as libc::c_double;
        pgrad_angl = pgrad_angl.offset(gradientHorizontalStride as isize);
        src = src.offset(1);
    }
    gx = *src.offset(0 as libc::c_int as isize) - *src.offset(-xo as isize);
    gy = *src.offset(0 as libc::c_int as isize) - *src.offset(-yo as isize);
    *pgrad_ampl = vl_fast_sqrt_f((gx * gx + gy * gy) as libc::c_float) as libc::c_double;
    pgrad_ampl = pgrad_ampl.offset(gradientHorizontalStride as isize);
    *pgrad_angl = vl_mod_2pi_f(
        (vl_fast_atan2_f(gy as libc::c_float, gx as libc::c_float) as libc::c_double
            + 2 as libc::c_int as libc::c_double * 3.141592653589793f64) as libc::c_float,
    ) as libc::c_double;
    pgrad_angl = pgrad_angl.offset(gradientHorizontalStride as isize);
    src = src.offset(1);
}
#[no_mangle]
pub unsafe extern "C" fn vl_imgradient_polar_f(
    mut gradientModulus: *mut libc::c_float,
    mut gradientAngle: *mut libc::c_float,
    mut gradientHorizontalStride: vl_size,
    mut gradHeightStride: vl_size,
    mut image: *const libc::c_float,
    mut imageWidth: vl_size,
    mut imageHeight: vl_size,
    mut imageStride: vl_size,
) {
    let xo: vl_index = 1 as libc::c_int as vl_index;
    let yo: vl_index = imageStride as vl_index;
    let w: vl_size = imageWidth;
    let h: vl_size = imageHeight;
    let mut src: *const libc::c_float = 0 as *const libc::c_float;
    let mut end: *const libc::c_float = 0 as *const libc::c_float;
    let mut pgrad_angl: *mut libc::c_float = 0 as *mut libc::c_float;
    let mut pgrad_ampl: *mut libc::c_float = 0 as *mut libc::c_float;
    let mut gx: libc::c_float = 0.;
    let mut gy: libc::c_float = 0.;
    let mut y: vl_size = 0;
    src = image;
    pgrad_angl = gradientAngle;
    pgrad_ampl = gradientModulus;
    gx = *src.offset(xo as isize) - *src.offset(0 as libc::c_int as isize);
    gy = *src.offset(yo as isize) - *src.offset(0 as libc::c_int as isize);
    *pgrad_ampl = vl_fast_sqrt_f(gx * gx + gy * gy);
    pgrad_ampl = pgrad_ampl.offset(gradientHorizontalStride as isize);
    *pgrad_angl = vl_mod_2pi_f(
        (vl_fast_atan2_f(gy, gx) as libc::c_double
            + 2 as libc::c_int as libc::c_double * 3.141592653589793f64) as libc::c_float,
    );
    pgrad_angl = pgrad_angl.offset(gradientHorizontalStride as isize);
    src = src.offset(1);
    end = src
        .offset(-(1 as libc::c_int as isize))
        .offset(w as isize)
        .offset(-(1 as libc::c_int as isize));
    while src < end {
        gx = (0.5f64
            * (*src.offset(xo as isize) - *src.offset(-xo as isize)) as libc::c_double)
            as libc::c_float;
        gy = *src.offset(yo as isize) - *src.offset(0 as libc::c_int as isize);
        *pgrad_ampl = vl_fast_sqrt_f(gx * gx + gy * gy);
        pgrad_ampl = pgrad_ampl.offset(gradientHorizontalStride as isize);
        *pgrad_angl = vl_mod_2pi_f(
            (vl_fast_atan2_f(gy, gx) as libc::c_double
                + 2 as libc::c_int as libc::c_double * 3.141592653589793f64)
                as libc::c_float,
        );
        pgrad_angl = pgrad_angl.offset(gradientHorizontalStride as isize);
        src = src.offset(1);
    }
    gx = *src.offset(0 as libc::c_int as isize) - *src.offset(-xo as isize);
    gy = *src.offset(yo as isize) - *src.offset(0 as libc::c_int as isize);
    *pgrad_ampl = vl_fast_sqrt_f(gx * gx + gy * gy);
    pgrad_ampl = pgrad_ampl.offset(gradientHorizontalStride as isize);
    *pgrad_angl = vl_mod_2pi_f(
        (vl_fast_atan2_f(gy, gx) as libc::c_double
            + 2 as libc::c_int as libc::c_double * 3.141592653589793f64) as libc::c_float,
    );
    pgrad_angl = pgrad_angl.offset(gradientHorizontalStride as isize);
    src = src.offset(1);
    gradientModulus = gradientModulus.offset(gradHeightStride as isize);
    pgrad_ampl = gradientModulus;
    gradientAngle = gradientAngle.offset(gradHeightStride as isize);
    pgrad_angl = gradientAngle;
    image = image.offset(imageStride as isize);
    src = image;
    y = 1 as libc::c_int as vl_size;
    while y < h.wrapping_sub(1 as libc::c_int as libc::c_ulonglong) {
        gx = *src.offset(xo as isize) - *src.offset(0 as libc::c_int as isize);
        gy = (0.5f64
            * (*src.offset(yo as isize) - *src.offset(-yo as isize)) as libc::c_double)
            as libc::c_float;
        *pgrad_ampl = vl_fast_sqrt_f(gx * gx + gy * gy);
        pgrad_ampl = pgrad_ampl.offset(gradientHorizontalStride as isize);
        *pgrad_angl = vl_mod_2pi_f(
            (vl_fast_atan2_f(gy, gx) as libc::c_double
                + 2 as libc::c_int as libc::c_double * 3.141592653589793f64)
                as libc::c_float,
        );
        pgrad_angl = pgrad_angl.offset(gradientHorizontalStride as isize);
        src = src.offset(1);
        end = src
            .offset(-(1 as libc::c_int as isize))
            .offset(w as isize)
            .offset(-(1 as libc::c_int as isize));
        while src < end {
            gx = (0.5f64
                * (*src.offset(xo as isize) - *src.offset(-xo as isize))
                    as libc::c_double) as libc::c_float;
            gy = (0.5f64
                * (*src.offset(yo as isize) - *src.offset(-yo as isize))
                    as libc::c_double) as libc::c_float;
            *pgrad_ampl = vl_fast_sqrt_f(gx * gx + gy * gy);
            pgrad_ampl = pgrad_ampl.offset(gradientHorizontalStride as isize);
            *pgrad_angl = vl_mod_2pi_f(
                (vl_fast_atan2_f(gy, gx) as libc::c_double
                    + 2 as libc::c_int as libc::c_double * 3.141592653589793f64)
                    as libc::c_float,
            );
            pgrad_angl = pgrad_angl.offset(gradientHorizontalStride as isize);
            src = src.offset(1);
        }
        gx = *src.offset(0 as libc::c_int as isize) - *src.offset(-xo as isize);
        gy = (0.5f64
            * (*src.offset(yo as isize) - *src.offset(-yo as isize)) as libc::c_double)
            as libc::c_float;
        *pgrad_ampl = vl_fast_sqrt_f(gx * gx + gy * gy);
        pgrad_ampl = pgrad_ampl.offset(gradientHorizontalStride as isize);
        *pgrad_angl = vl_mod_2pi_f(
            (vl_fast_atan2_f(gy, gx) as libc::c_double
                + 2 as libc::c_int as libc::c_double * 3.141592653589793f64)
                as libc::c_float,
        );
        pgrad_angl = pgrad_angl.offset(gradientHorizontalStride as isize);
        src = src.offset(1);
        gradientModulus = gradientModulus.offset(gradHeightStride as isize);
        pgrad_ampl = gradientModulus;
        gradientAngle = gradientAngle.offset(gradHeightStride as isize);
        pgrad_angl = gradientAngle;
        image = image.offset(imageStride as isize);
        src = image;
        y = y.wrapping_add(1);
    }
    gx = *src.offset(xo as isize) - *src.offset(0 as libc::c_int as isize);
    gy = *src.offset(0 as libc::c_int as isize) - *src.offset(-yo as isize);
    *pgrad_ampl = vl_fast_sqrt_f(gx * gx + gy * gy);
    pgrad_ampl = pgrad_ampl.offset(gradientHorizontalStride as isize);
    *pgrad_angl = vl_mod_2pi_f(
        (vl_fast_atan2_f(gy, gx) as libc::c_double
            + 2 as libc::c_int as libc::c_double * 3.141592653589793f64) as libc::c_float,
    );
    pgrad_angl = pgrad_angl.offset(gradientHorizontalStride as isize);
    src = src.offset(1);
    end = src
        .offset(-(1 as libc::c_int as isize))
        .offset(w as isize)
        .offset(-(1 as libc::c_int as isize));
    while src < end {
        gx = (0.5f64
            * (*src.offset(xo as isize) - *src.offset(-xo as isize)) as libc::c_double)
            as libc::c_float;
        gy = *src.offset(0 as libc::c_int as isize) - *src.offset(-yo as isize);
        *pgrad_ampl = vl_fast_sqrt_f(gx * gx + gy * gy);
        pgrad_ampl = pgrad_ampl.offset(gradientHorizontalStride as isize);
        *pgrad_angl = vl_mod_2pi_f(
            (vl_fast_atan2_f(gy, gx) as libc::c_double
                + 2 as libc::c_int as libc::c_double * 3.141592653589793f64)
                as libc::c_float,
        );
        pgrad_angl = pgrad_angl.offset(gradientHorizontalStride as isize);
        src = src.offset(1);
    }
    gx = *src.offset(0 as libc::c_int as isize) - *src.offset(-xo as isize);
    gy = *src.offset(0 as libc::c_int as isize) - *src.offset(-yo as isize);
    *pgrad_ampl = vl_fast_sqrt_f(gx * gx + gy * gy);
    pgrad_ampl = pgrad_ampl.offset(gradientHorizontalStride as isize);
    *pgrad_angl = vl_mod_2pi_f(
        (vl_fast_atan2_f(gy, gx) as libc::c_double
            + 2 as libc::c_int as libc::c_double * 3.141592653589793f64) as libc::c_float,
    );
    pgrad_angl = pgrad_angl.offset(gradientHorizontalStride as isize);
    src = src.offset(1);
}
#[no_mangle]
pub unsafe extern "C" fn vl_imintegral_d(
    mut integral: *mut libc::c_double,
    mut integralStride: vl_size,
    mut image: *const libc::c_double,
    mut imageWidth: vl_size,
    mut imageHeight: vl_size,
    mut imageStride: vl_size,
) {
    let mut x: vl_uindex = 0;
    let mut y: vl_uindex = 0;
    let mut temp: libc::c_double = 0 as libc::c_int as libc::c_double;
    if imageHeight > 0 as libc::c_int as libc::c_ulonglong {
        x = 0 as libc::c_int as vl_uindex;
        while x < imageWidth {
            let fresh6 = image;
            image = image.offset(1);
            temp += *fresh6;
            let fresh7 = integral;
            integral = integral.offset(1);
            *fresh7 = temp;
            x = x.wrapping_add(1);
        }
    }
    y = 1 as libc::c_int as vl_uindex;
    while y < imageHeight {
        let mut integralPrev: *mut libc::c_double = 0 as *mut libc::c_double;
        integral = integral.offset(integralStride.wrapping_sub(imageWidth) as isize);
        image = image.offset(imageStride.wrapping_sub(imageWidth) as isize);
        integralPrev = integral.offset(-(integralStride as isize));
        temp = 0 as libc::c_int as libc::c_double;
        x = 0 as libc::c_int as vl_uindex;
        while x < imageWidth {
            let fresh8 = image;
            image = image.offset(1);
            temp += *fresh8;
            let fresh9 = integralPrev;
            integralPrev = integralPrev.offset(1);
            let fresh10 = integral;
            integral = integral.offset(1);
            *fresh10 = *fresh9 + temp;
            x = x.wrapping_add(1);
        }
        y = y.wrapping_add(1);
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_imintegral_f(
    mut integral: *mut libc::c_float,
    mut integralStride: vl_size,
    mut image: *const libc::c_float,
    mut imageWidth: vl_size,
    mut imageHeight: vl_size,
    mut imageStride: vl_size,
) {
    let mut x: vl_uindex = 0;
    let mut y: vl_uindex = 0;
    let mut temp: libc::c_float = 0 as libc::c_int as libc::c_float;
    if imageHeight > 0 as libc::c_int as libc::c_ulonglong {
        x = 0 as libc::c_int as vl_uindex;
        while x < imageWidth {
            let fresh11 = image;
            image = image.offset(1);
            temp += *fresh11;
            let fresh12 = integral;
            integral = integral.offset(1);
            *fresh12 = temp;
            x = x.wrapping_add(1);
        }
    }
    y = 1 as libc::c_int as vl_uindex;
    while y < imageHeight {
        let mut integralPrev: *mut libc::c_float = 0 as *mut libc::c_float;
        integral = integral.offset(integralStride.wrapping_sub(imageWidth) as isize);
        image = image.offset(imageStride.wrapping_sub(imageWidth) as isize);
        integralPrev = integral.offset(-(integralStride as isize));
        temp = 0 as libc::c_int as libc::c_float;
        x = 0 as libc::c_int as vl_uindex;
        while x < imageWidth {
            let fresh13 = image;
            image = image.offset(1);
            temp += *fresh13;
            let fresh14 = integralPrev;
            integralPrev = integralPrev.offset(1);
            let fresh15 = integral;
            integral = integral.offset(1);
            *fresh15 = *fresh14 + temp;
            x = x.wrapping_add(1);
        }
        y = y.wrapping_add(1);
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_imintegral_ui32(
    mut integral: *mut vl_uint32,
    mut integralStride: vl_size,
    mut image: *const vl_uint32,
    mut imageWidth: vl_size,
    mut imageHeight: vl_size,
    mut imageStride: vl_size,
) {
    let mut x: vl_uindex = 0;
    let mut y: vl_uindex = 0;
    let mut temp: vl_uint32 = 0 as libc::c_int as vl_uint32;
    if imageHeight > 0 as libc::c_int as libc::c_ulonglong {
        x = 0 as libc::c_int as vl_uindex;
        while x < imageWidth {
            let fresh16 = image;
            image = image.offset(1);
            temp = (temp as libc::c_uint).wrapping_add(*fresh16) as vl_uint32
                as vl_uint32;
            let fresh17 = integral;
            integral = integral.offset(1);
            *fresh17 = temp;
            x = x.wrapping_add(1);
        }
    }
    y = 1 as libc::c_int as vl_uindex;
    while y < imageHeight {
        let mut integralPrev: *mut vl_uint32 = 0 as *mut vl_uint32;
        integral = integral.offset(integralStride.wrapping_sub(imageWidth) as isize);
        image = image.offset(imageStride.wrapping_sub(imageWidth) as isize);
        integralPrev = integral.offset(-(integralStride as isize));
        temp = 0 as libc::c_int as vl_uint32;
        x = 0 as libc::c_int as vl_uindex;
        while x < imageWidth {
            let fresh18 = image;
            image = image.offset(1);
            temp = (temp as libc::c_uint).wrapping_add(*fresh18) as vl_uint32
                as vl_uint32;
            let fresh19 = integralPrev;
            integralPrev = integralPrev.offset(1);
            let fresh20 = integral;
            integral = integral.offset(1);
            *fresh20 = (*fresh19).wrapping_add(temp);
            x = x.wrapping_add(1);
        }
        y = y.wrapping_add(1);
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_imintegral_i32(
    mut integral: *mut vl_int32,
    mut integralStride: vl_size,
    mut image: *const vl_int32,
    mut imageWidth: vl_size,
    mut imageHeight: vl_size,
    mut imageStride: vl_size,
) {
    let mut x: vl_uindex = 0;
    let mut y: vl_uindex = 0;
    let mut temp: vl_int32 = 0 as libc::c_int;
    if imageHeight > 0 as libc::c_int as libc::c_ulonglong {
        x = 0 as libc::c_int as vl_uindex;
        while x < imageWidth {
            let fresh21 = image;
            image = image.offset(1);
            temp += *fresh21;
            let fresh22 = integral;
            integral = integral.offset(1);
            *fresh22 = temp;
            x = x.wrapping_add(1);
        }
    }
    y = 1 as libc::c_int as vl_uindex;
    while y < imageHeight {
        let mut integralPrev: *mut vl_int32 = 0 as *mut vl_int32;
        integral = integral.offset(integralStride.wrapping_sub(imageWidth) as isize);
        image = image.offset(imageStride.wrapping_sub(imageWidth) as isize);
        integralPrev = integral.offset(-(integralStride as isize));
        temp = 0 as libc::c_int;
        x = 0 as libc::c_int as vl_uindex;
        while x < imageWidth {
            let fresh23 = image;
            image = image.offset(1);
            temp += *fresh23;
            let fresh24 = integralPrev;
            integralPrev = integralPrev.offset(1);
            let fresh25 = integral;
            integral = integral.offset(1);
            *fresh25 = *fresh24 + temp;
            x = x.wrapping_add(1);
        }
        y = y.wrapping_add(1);
    }
}
