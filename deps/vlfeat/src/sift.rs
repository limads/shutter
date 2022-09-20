use ::libc;
extern "C" {
    fn vl_malloc(n: size_t) -> *mut libc::c_void;
    fn vl_realloc(ptr: *mut libc::c_void, n: size_t) -> *mut libc::c_void;
    fn vl_free(ptr: *mut libc::c_void);
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
    fn cos(_: libc::c_double) -> libc::c_double;
    fn sin(_: libc::c_double) -> libc::c_double;
    fn exp(_: libc::c_double) -> libc::c_double;
    fn log(_: libc::c_double) -> libc::c_double;
    fn powf(_: libc::c_float, _: libc::c_float) -> libc::c_float;
    fn pow(_: libc::c_double, _: libc::c_double) -> libc::c_double;
    fn sqrt(_: libc::c_double) -> libc::c_double;
    fn ceil(_: libc::c_double) -> libc::c_double;
    fn floor(_: libc::c_double) -> libc::c_double;
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
pub type size_t = libc::c_ulong;
pub type vl_int64 = libc::c_longlong;
pub type vl_int32 = libc::c_int;
pub type vl_uint64 = libc::c_ulonglong;
pub type vl_bool = libc::c_int;
pub type vl_size = vl_uint64;
pub type vl_index = vl_int64;
pub type vl_uindex = vl_uint64;
pub type vl_sift_pix = libc::c_float;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlSiftKeypoint {
    pub o: libc::c_int,
    pub ix: libc::c_int,
    pub iy: libc::c_int,
    pub is: libc::c_int,
    pub x: libc::c_float,
    pub y: libc::c_float,
    pub s: libc::c_float,
    pub sigma: libc::c_float,
}
pub type VlSiftKeypoint = _VlSiftKeypoint;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlSiftFilt {
    pub sigman: libc::c_double,
    pub sigma0: libc::c_double,
    pub sigmak: libc::c_double,
    pub dsigma0: libc::c_double,
    pub width: libc::c_int,
    pub height: libc::c_int,
    pub O: libc::c_int,
    pub S: libc::c_int,
    pub o_min: libc::c_int,
    pub s_min: libc::c_int,
    pub s_max: libc::c_int,
    pub o_cur: libc::c_int,
    pub temp: *mut vl_sift_pix,
    pub octave: *mut vl_sift_pix,
    pub dog: *mut vl_sift_pix,
    pub octave_width: libc::c_int,
    pub octave_height: libc::c_int,
    pub gaussFilter: *mut vl_sift_pix,
    pub gaussFilterSigma: libc::c_double,
    pub gaussFilterWidth: vl_size,
    pub keys: *mut VlSiftKeypoint,
    pub nkeys: libc::c_int,
    pub keys_res: libc::c_int,
    pub peak_thresh: libc::c_double,
    pub edge_thresh: libc::c_double,
    pub norm_thresh: libc::c_double,
    pub magnif: libc::c_double,
    pub windowSize: libc::c_double,
    pub grad: *mut vl_sift_pix,
    pub grad_o: libc::c_int,
}
pub type VlSiftFilt = _VlSiftFilt;
pub const nbins: C2RustUnnamed_0 = 36;
#[derive(Copy, Clone)]
#[repr(C)]
pub union C2RustUnnamed {
    pub x: libc::c_float,
    pub i: vl_int32,
}
pub type C2RustUnnamed_0 = libc::c_uint;
#[inline]
unsafe extern "C" fn vl_sift_get_octave(
    mut f: *const VlSiftFilt,
    mut s: libc::c_int,
) -> *mut vl_sift_pix {
    let mut w: libc::c_int = vl_sift_get_octave_width(f);
    let mut h: libc::c_int = vl_sift_get_octave_height(f);
    return ((*f).octave).offset((w * h * (s - (*f).s_min)) as isize);
}
#[inline]
unsafe extern "C" fn vl_sift_get_octave_height(mut f: *const VlSiftFilt) -> libc::c_int {
    return (*f).octave_height;
}
#[inline]
unsafe extern "C" fn vl_sift_get_octave_width(mut f: *const VlSiftFilt) -> libc::c_int {
    return (*f).octave_width;
}
#[inline]
unsafe extern "C" fn vl_abs_d(mut x: libc::c_double) -> libc::c_double {
    return x.abs();
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
unsafe extern "C" fn vl_floor_f(mut x: libc::c_float) -> libc::c_long {
    let mut xi: libc::c_long = x as libc::c_long;
    if x >= 0 as libc::c_int as libc::c_float || xi as libc::c_float == x {
        return xi
    } else {
        return xi - 1 as libc::c_int as libc::c_long
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
#[inline]
unsafe extern "C" fn vl_fast_sqrt_f(mut x: libc::c_float) -> libc::c_float {
    return if (x as libc::c_double) < 1e-8f64 {
        0 as libc::c_int as libc::c_float
    } else {
        x * vl_fast_resqrt_f(x)
    };
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
#[inline]
unsafe extern "C" fn vl_floor_d(mut x: libc::c_double) -> libc::c_long {
    let mut xi: libc::c_long = x as libc::c_long;
    if x >= 0 as libc::c_int as libc::c_double || xi as libc::c_double == x {
        return xi
    } else {
        return xi - 1 as libc::c_int as libc::c_long
    };
}
#[no_mangle]
pub static mut expn_tab: [libc::c_double; 257] = [0.; 257];
#[inline]
unsafe extern "C" fn fast_expn(mut x: libc::c_double) -> libc::c_double {
    let mut a: libc::c_double = 0.;
    let mut b: libc::c_double = 0.;
    let mut r: libc::c_double = 0.;
    let mut i: libc::c_int = 0;
    if x > 25.0f64 {
        return 0.0f64;
    }
    x *= 256 as libc::c_int as libc::c_double / 25.0f64;
    i = vl_floor_d(x) as libc::c_int;
    r = x - i as libc::c_double;
    a = expn_tab[i as usize];
    b = expn_tab[(i + 1 as libc::c_int) as usize];
    return a + r * (b - a);
}
#[inline]
unsafe extern "C" fn fast_expn_init() {
    let mut k: libc::c_int = 0;
    k = 0 as libc::c_int;
    while k < 256 as libc::c_int + 1 as libc::c_int {
        expn_tab[k
            as usize] = exp(
            -(k as libc::c_double) * (25.0f64 / 256 as libc::c_int as libc::c_double),
        );
        k += 1;
    }
}
unsafe extern "C" fn copy_and_upsample_rows(
    mut dst: *mut vl_sift_pix,
    mut src: *const vl_sift_pix,
    mut width: libc::c_int,
    mut height: libc::c_int,
) {
    let mut x: libc::c_int = 0;
    let mut y: libc::c_int = 0;
    let mut a: vl_sift_pix = 0.;
    let mut b: vl_sift_pix = 0.;
    y = 0 as libc::c_int;
    while y < height {
        let fresh0 = src;
        src = src.offset(1);
        a = *fresh0;
        b = a;
        x = 0 as libc::c_int;
        while x < width - 1 as libc::c_int {
            let fresh1 = src;
            src = src.offset(1);
            b = *fresh1;
            *dst = a;
            dst = dst.offset(height as isize);
            *dst = (0.5f64 * (a + b) as libc::c_double) as vl_sift_pix;
            dst = dst.offset(height as isize);
            a = b;
            x += 1;
        }
        *dst = b;
        dst = dst.offset(height as isize);
        *dst = b;
        dst = dst.offset(height as isize);
        dst = dst
            .offset((1 as libc::c_int - width * 2 as libc::c_int * height) as isize);
        y += 1;
    }
}
unsafe extern "C" fn _vl_sift_smooth(
    mut self_0: *mut VlSiftFilt,
    mut outputImage: *mut vl_sift_pix,
    mut tempImage: *mut vl_sift_pix,
    mut inputImage: *const vl_sift_pix,
    mut width: vl_size,
    mut height: vl_size,
    mut sigma: libc::c_double,
) {
    if (*self_0).gaussFilterSigma != sigma {
        let mut j: vl_uindex = 0;
        let mut acc: vl_sift_pix = 0 as libc::c_int as vl_sift_pix;
        if !((*self_0).gaussFilter).is_null() {
            vl_free((*self_0).gaussFilter as *mut libc::c_void);
        }
        (*self_0)
            .gaussFilterWidth = (if ceil(4.0f64 * sigma)
            > 1 as libc::c_int as libc::c_double
        {
            ceil(4.0f64 * sigma)
        } else {
            1 as libc::c_int as libc::c_double
        }) as vl_size;
        (*self_0).gaussFilterSigma = sigma;
        (*self_0)
            .gaussFilter = vl_malloc(
            (::core::mem::size_of::<vl_sift_pix>() as libc::c_ulong as libc::c_ulonglong)
                .wrapping_mul(
                    (2 as libc::c_int as libc::c_ulonglong)
                        .wrapping_mul((*self_0).gaussFilterWidth)
                        .wrapping_add(1 as libc::c_int as libc::c_ulonglong),
                ) as size_t,
        ) as *mut vl_sift_pix;
        j = 0 as libc::c_int as vl_uindex;
        while j
            < (2 as libc::c_int as libc::c_ulonglong)
                .wrapping_mul((*self_0).gaussFilterWidth)
                .wrapping_add(1 as libc::c_int as libc::c_ulonglong)
        {
            let mut d: vl_sift_pix = (j as libc::c_int
                - (*self_0).gaussFilterWidth as libc::c_int) as vl_sift_pix
                / sigma as vl_sift_pix;
            *((*self_0).gaussFilter)
                .offset(
                    j as isize,
                ) = exp(-0.5f64 * (d * d) as libc::c_double) as vl_sift_pix;
            acc += *((*self_0).gaussFilter).offset(j as isize);
            j = j.wrapping_add(1);
        }
        j = 0 as libc::c_int as vl_uindex;
        while j
            < (2 as libc::c_int as libc::c_ulonglong)
                .wrapping_mul((*self_0).gaussFilterWidth)
                .wrapping_add(1 as libc::c_int as libc::c_ulonglong)
        {
            let ref mut fresh2 = *((*self_0).gaussFilter).offset(j as isize);
            *fresh2 /= acc;
            j = j.wrapping_add(1);
        }
    }
    if (*self_0).gaussFilterWidth == 0 as libc::c_int as libc::c_ulonglong {
        memcpy(
            outputImage as *mut libc::c_void,
            inputImage as *const libc::c_void,
            (::core::mem::size_of::<vl_sift_pix>() as libc::c_ulong as libc::c_ulonglong)
                .wrapping_mul(width)
                .wrapping_mul(height) as libc::c_ulong,
        );
        return;
    }
    vl_imconvcol_vf(
        tempImage,
        height,
        inputImage,
        width,
        height,
        width,
        (*self_0).gaussFilter,
        ((*self_0).gaussFilterWidth).wrapping_neg() as vl_index,
        (*self_0).gaussFilterWidth as vl_index,
        1 as libc::c_int,
        ((0x1 as libc::c_int) << 0 as libc::c_int
            | (0x1 as libc::c_int) << 2 as libc::c_int) as libc::c_uint,
    );
    vl_imconvcol_vf(
        outputImage,
        width,
        tempImage,
        height,
        width,
        height,
        (*self_0).gaussFilter,
        ((*self_0).gaussFilterWidth).wrapping_neg() as vl_index,
        (*self_0).gaussFilterWidth as vl_index,
        1 as libc::c_int,
        ((0x1 as libc::c_int) << 0 as libc::c_int
            | (0x1 as libc::c_int) << 2 as libc::c_int) as libc::c_uint,
    );
}
unsafe extern "C" fn copy_and_downsample(
    mut dst: *mut vl_sift_pix,
    mut src: *const vl_sift_pix,
    mut width: libc::c_int,
    mut height: libc::c_int,
    mut d: libc::c_int,
) {
    let mut x: libc::c_int = 0;
    let mut y: libc::c_int = 0;
    d = (1 as libc::c_int) << d;
    y = 0 as libc::c_int;
    while y < height {
        let mut srcrowp: *const vl_sift_pix = src.offset((y * width) as isize);
        x = 0 as libc::c_int;
        while x < width - (d - 1 as libc::c_int) {
            let fresh3 = dst;
            dst = dst.offset(1);
            *fresh3 = *srcrowp;
            srcrowp = srcrowp.offset(d as isize);
            x += d;
        }
        y += d;
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_sift_new(
    mut width: libc::c_int,
    mut height: libc::c_int,
    mut noctaves: libc::c_int,
    mut nlevels: libc::c_int,
    mut o_min: libc::c_int,
) -> *mut VlSiftFilt {
    let mut f: *mut VlSiftFilt = vl_malloc(
        ::core::mem::size_of::<VlSiftFilt>() as libc::c_ulong,
    ) as *mut VlSiftFilt;
    let mut w: libc::c_int = if -o_min >= 0 as libc::c_int {
        width << -o_min
    } else {
        width >> --o_min
    };
    let mut h: libc::c_int = if -o_min >= 0 as libc::c_int {
        height << -o_min
    } else {
        height >> --o_min
    };
    let mut nel: libc::c_int = w * h;
    if noctaves < 0 as libc::c_int {
        noctaves = (if floor(
            log((if width < height { width } else { height }) as libc::c_double)
                / 0.693147180559945f64,
        ) - o_min as libc::c_double - 3 as libc::c_int as libc::c_double
            > 1 as libc::c_int as libc::c_double
        {
            floor(
                log((if width < height { width } else { height }) as libc::c_double)
                    / 0.693147180559945f64,
            ) - o_min as libc::c_double - 3 as libc::c_int as libc::c_double
        } else {
            1 as libc::c_int as libc::c_double
        }) as libc::c_int;
    }
    (*f).width = width;
    (*f).height = height;
    (*f).O = noctaves;
    (*f).S = nlevels;
    (*f).o_min = o_min;
    (*f).s_min = -(1 as libc::c_int);
    (*f).s_max = nlevels + 1 as libc::c_int;
    (*f).o_cur = o_min;
    (*f)
        .temp = vl_malloc(
        (::core::mem::size_of::<vl_sift_pix>() as libc::c_ulong)
            .wrapping_mul(nel as libc::c_ulong),
    ) as *mut vl_sift_pix;
    (*f)
        .octave = vl_malloc(
        (::core::mem::size_of::<vl_sift_pix>() as libc::c_ulong)
            .wrapping_mul(nel as libc::c_ulong)
            .wrapping_mul(((*f).s_max - (*f).s_min + 1 as libc::c_int) as libc::c_ulong),
    ) as *mut vl_sift_pix;
    (*f)
        .dog = vl_malloc(
        (::core::mem::size_of::<vl_sift_pix>() as libc::c_ulong)
            .wrapping_mul(nel as libc::c_ulong)
            .wrapping_mul(((*f).s_max - (*f).s_min) as libc::c_ulong),
    ) as *mut vl_sift_pix;
    (*f)
        .grad = vl_malloc(
        (::core::mem::size_of::<vl_sift_pix>() as libc::c_ulong)
            .wrapping_mul(nel as libc::c_ulong)
            .wrapping_mul(2 as libc::c_int as libc::c_ulong)
            .wrapping_mul(((*f).s_max - (*f).s_min) as libc::c_ulong),
    ) as *mut vl_sift_pix;
    (*f).sigman = 0.5f64;
    (*f).sigmak = pow(2.0f64, 1.0f64 / nlevels as libc::c_double);
    (*f).sigma0 = 1.6f64 * (*f).sigmak;
    (*f).dsigma0 = (*f).sigma0 * sqrt(1.0f64 - 1.0f64 / ((*f).sigmak * (*f).sigmak));
    (*f).gaussFilter = 0 as *mut vl_sift_pix;
    (*f).gaussFilterSigma = 0 as libc::c_int as libc::c_double;
    (*f).gaussFilterWidth = 0 as libc::c_int as vl_size;
    (*f).octave_width = 0 as libc::c_int;
    (*f).octave_height = 0 as libc::c_int;
    (*f).keys = 0 as *mut VlSiftKeypoint;
    (*f).nkeys = 0 as libc::c_int;
    (*f).keys_res = 0 as libc::c_int;
    (*f).peak_thresh = 0.0f64;
    (*f).edge_thresh = 10.0f64;
    (*f).norm_thresh = 0.0f64;
    (*f).magnif = 3.0f64;
    (*f).windowSize = (4 as libc::c_int / 2 as libc::c_int) as libc::c_double;
    (*f).grad_o = o_min - 1 as libc::c_int;
    fast_expn_init();
    return f;
}
#[no_mangle]
pub unsafe extern "C" fn vl_sift_delete(mut f: *mut VlSiftFilt) {
    if !f.is_null() {
        if !((*f).keys).is_null() {
            vl_free((*f).keys as *mut libc::c_void);
        }
        if !((*f).grad).is_null() {
            vl_free((*f).grad as *mut libc::c_void);
        }
        if !((*f).dog).is_null() {
            vl_free((*f).dog as *mut libc::c_void);
        }
        if !((*f).octave).is_null() {
            vl_free((*f).octave as *mut libc::c_void);
        }
        if !((*f).temp).is_null() {
            vl_free((*f).temp as *mut libc::c_void);
        }
        if !((*f).gaussFilter).is_null() {
            vl_free((*f).gaussFilter as *mut libc::c_void);
        }
        vl_free(f as *mut libc::c_void);
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_sift_process_first_octave(
    mut f: *mut VlSiftFilt,
    mut im: *const vl_sift_pix,
) -> libc::c_int {
    let mut o: libc::c_int = 0;
    let mut s: libc::c_int = 0;
    let mut h: libc::c_int = 0;
    let mut w: libc::c_int = 0;
    let mut sa: libc::c_double = 0.;
    let mut sb: libc::c_double = 0.;
    let mut octave: *mut vl_sift_pix = 0 as *mut vl_sift_pix;
    let mut temp: *mut vl_sift_pix = (*f).temp;
    let mut width: libc::c_int = (*f).width;
    let mut height: libc::c_int = (*f).height;
    let mut o_min: libc::c_int = (*f).o_min;
    let mut s_min: libc::c_int = (*f).s_min;
    let mut s_max: libc::c_int = (*f).s_max;
    let mut sigma0: libc::c_double = (*f).sigma0;
    let mut sigmak: libc::c_double = (*f).sigmak;
    let mut sigman: libc::c_double = (*f).sigman;
    let mut dsigma0: libc::c_double = (*f).dsigma0;
    (*f).o_cur = o_min;
    (*f).nkeys = 0 as libc::c_int;
    (*f)
        .octave_width = if -(*f).o_cur >= 0 as libc::c_int {
        (*f).width << -(*f).o_cur
    } else {
        (*f).width >> --(*f).o_cur
    };
    w = (*f).octave_width;
    (*f)
        .octave_height = if -(*f).o_cur >= 0 as libc::c_int {
        (*f).height << -(*f).o_cur
    } else {
        (*f).height >> --(*f).o_cur
    };
    h = (*f).octave_height;
    if (*f).O == 0 as libc::c_int {
        return 5 as libc::c_int;
    }
    octave = vl_sift_get_octave(f, s_min);
    if o_min < 0 as libc::c_int {
        copy_and_upsample_rows(temp, im, width, height);
        copy_and_upsample_rows(octave, temp, height, 2 as libc::c_int * width);
        o = -(1 as libc::c_int);
        while o > o_min {
            copy_and_upsample_rows(temp, octave, width << -o, height << -o);
            copy_and_upsample_rows(
                octave,
                temp,
                width << -o,
                2 as libc::c_int * (height << -o),
            );
            o -= 1;
        }
    } else if o_min > 0 as libc::c_int {
        copy_and_downsample(octave, im, width, height, o_min);
    } else {
        memcpy(
            octave as *mut libc::c_void,
            im as *const libc::c_void,
            (::core::mem::size_of::<vl_sift_pix>() as libc::c_ulong)
                .wrapping_mul(width as libc::c_ulong)
                .wrapping_mul(height as libc::c_ulong),
        );
    }
    sa = sigma0 * pow(sigmak, s_min as libc::c_double);
    sb = sigman * pow(2.0f64, -o_min as libc::c_double);
    if sa > sb {
        let mut sd: libc::c_double = sqrt(sa * sa - sb * sb);
        _vl_sift_smooth(f, octave, temp, octave, w as vl_size, h as vl_size, sd);
    }
    s = s_min + 1 as libc::c_int;
    while s <= s_max {
        let mut sd_0: libc::c_double = dsigma0 * pow(sigmak, s as libc::c_double);
        _vl_sift_smooth(
            f,
            vl_sift_get_octave(f, s),
            temp,
            vl_sift_get_octave(f, s - 1 as libc::c_int),
            w as vl_size,
            h as vl_size,
            sd_0,
        );
        s += 1;
    }
    return 0 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn vl_sift_process_next_octave(
    mut f: *mut VlSiftFilt,
) -> libc::c_int {
    let mut s: libc::c_int = 0;
    let mut h: libc::c_int = 0;
    let mut w: libc::c_int = 0;
    let mut s_best: libc::c_int = 0;
    let mut sa: libc::c_double = 0.;
    let mut sb: libc::c_double = 0.;
    let mut octave: *mut vl_sift_pix = 0 as *mut vl_sift_pix;
    let mut pt: *mut vl_sift_pix = 0 as *mut vl_sift_pix;
    let mut temp: *mut vl_sift_pix = (*f).temp;
    let mut O: libc::c_int = (*f).O;
    let mut S: libc::c_int = (*f).S;
    let mut o_min: libc::c_int = (*f).o_min;
    let mut s_min: libc::c_int = (*f).s_min;
    let mut s_max: libc::c_int = (*f).s_max;
    let mut sigma0: libc::c_double = (*f).sigma0;
    let mut sigmak: libc::c_double = (*f).sigmak;
    let mut dsigma0: libc::c_double = (*f).dsigma0;
    if (*f).o_cur == o_min + O - 1 as libc::c_int {
        return 5 as libc::c_int;
    }
    s_best = if s_min + S < s_max { s_min + S } else { s_max };
    w = vl_sift_get_octave_width(f);
    h = vl_sift_get_octave_height(f);
    pt = vl_sift_get_octave(f, s_best);
    octave = vl_sift_get_octave(f, s_min);
    copy_and_downsample(octave, pt, w, h, 1 as libc::c_int);
    (*f).o_cur += 1 as libc::c_int;
    (*f).nkeys = 0 as libc::c_int;
    (*f)
        .octave_width = if -(*f).o_cur >= 0 as libc::c_int {
        (*f).width << -(*f).o_cur
    } else {
        (*f).width >> --(*f).o_cur
    };
    w = (*f).octave_width;
    (*f)
        .octave_height = if -(*f).o_cur >= 0 as libc::c_int {
        (*f).height << -(*f).o_cur
    } else {
        (*f).height >> --(*f).o_cur
    };
    h = (*f).octave_height;
    sa = sigma0
        * powf(sigmak as libc::c_float, s_min as libc::c_float) as libc::c_double;
    sb = sigma0
        * powf(sigmak as libc::c_float, (s_best - S) as libc::c_float) as libc::c_double;
    if sa > sb {
        let mut sd: libc::c_double = sqrt(sa * sa - sb * sb);
        _vl_sift_smooth(f, octave, temp, octave, w as vl_size, h as vl_size, sd);
    }
    s = s_min + 1 as libc::c_int;
    while s <= s_max {
        let mut sd_0: libc::c_double = dsigma0 * pow(sigmak, s as libc::c_double);
        _vl_sift_smooth(
            f,
            vl_sift_get_octave(f, s),
            temp,
            vl_sift_get_octave(f, s - 1 as libc::c_int),
            w as vl_size,
            h as vl_size,
            sd_0,
        );
        s += 1;
    }
    return 0 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn vl_sift_detect(mut f: *mut VlSiftFilt) {
    let mut dog: *mut vl_sift_pix = (*f).dog;
    let mut s_min: libc::c_int = (*f).s_min;
    let mut s_max: libc::c_int = (*f).s_max;
    let mut w: libc::c_int = (*f).octave_width;
    let mut h: libc::c_int = (*f).octave_height;
    let mut te: libc::c_double = (*f).edge_thresh;
    let mut tp: libc::c_double = (*f).peak_thresh;
    let xo: libc::c_int = 1 as libc::c_int;
    let yo: libc::c_int = w;
    let so: libc::c_int = w * h;
    let mut xper: libc::c_double = pow(2.0f64, (*f).o_cur as libc::c_double);
    let mut x: libc::c_int = 0;
    let mut y: libc::c_int = 0;
    let mut s: libc::c_int = 0;
    let mut i: libc::c_int = 0;
    let mut ii: libc::c_int = 0;
    let mut jj: libc::c_int = 0;
    let mut pt: *mut vl_sift_pix = 0 as *mut vl_sift_pix;
    let mut v: vl_sift_pix = 0.;
    let mut k: *mut VlSiftKeypoint = 0 as *mut VlSiftKeypoint;
    (*f).nkeys = 0 as libc::c_int;
    pt = (*f).dog;
    s = s_min;
    while s <= s_max - 1 as libc::c_int {
        let mut src_a: *mut vl_sift_pix = vl_sift_get_octave(f, s);
        let mut src_b: *mut vl_sift_pix = vl_sift_get_octave(f, s + 1 as libc::c_int);
        let mut end_a: *mut vl_sift_pix = src_a.offset((w * h) as isize);
        while src_a != end_a {
            let fresh4 = src_b;
            src_b = src_b.offset(1);
            let fresh5 = src_a;
            src_a = src_a.offset(1);
            let fresh6 = pt;
            pt = pt.offset(1);
            *fresh6 = *fresh4 - *fresh5;
        }
        s += 1;
    }
    pt = dog.offset(xo as isize).offset(yo as isize).offset(so as isize);
    s = s_min + 1 as libc::c_int;
    while s <= s_max - 2 as libc::c_int {
        y = 1 as libc::c_int;
        while y < h - 1 as libc::c_int {
            x = 1 as libc::c_int;
            while x < w - 1 as libc::c_int {
                v = *pt;
                if v as libc::c_double >= 0.8f64 * tp && v > *pt.offset(xo as isize)
                    && v > *pt.offset(-(xo as isize)) && v > *pt.offset(so as isize)
                    && v > *pt.offset(-(so as isize)) && v > *pt.offset(yo as isize)
                    && v > *pt.offset(-(yo as isize))
                    && v > *pt.offset(yo as isize).offset(xo as isize)
                    && v > *pt.offset(yo as isize).offset(-(xo as isize))
                    && v > *pt.offset(-(yo as isize)).offset(xo as isize)
                    && v > *pt.offset(-(yo as isize)).offset(-(xo as isize))
                    && v > *pt.offset(xo as isize).offset(so as isize)
                    && v > *pt.offset(-(xo as isize)).offset(so as isize)
                    && v > *pt.offset(yo as isize).offset(so as isize)
                    && v > *pt.offset(-(yo as isize)).offset(so as isize)
                    && v
                        > *pt.offset(yo as isize).offset(xo as isize).offset(so as isize)
                    && v
                        > *pt
                            .offset(yo as isize)
                            .offset(-(xo as isize))
                            .offset(so as isize)
                    && v
                        > *pt
                            .offset(-(yo as isize))
                            .offset(xo as isize)
                            .offset(so as isize)
                    && v
                        > *pt
                            .offset(-(yo as isize))
                            .offset(-(xo as isize))
                            .offset(so as isize)
                    && v > *pt.offset(xo as isize).offset(-(so as isize))
                    && v > *pt.offset(-(xo as isize)).offset(-(so as isize))
                    && v > *pt.offset(yo as isize).offset(-(so as isize))
                    && v > *pt.offset(-(yo as isize)).offset(-(so as isize))
                    && v
                        > *pt
                            .offset(yo as isize)
                            .offset(xo as isize)
                            .offset(-(so as isize))
                    && v
                        > *pt
                            .offset(yo as isize)
                            .offset(-(xo as isize))
                            .offset(-(so as isize))
                    && v
                        > *pt
                            .offset(-(yo as isize))
                            .offset(xo as isize)
                            .offset(-(so as isize))
                    && v
                        > *pt
                            .offset(-(yo as isize))
                            .offset(-(xo as isize))
                            .offset(-(so as isize))
                    || v as libc::c_double <= -0.8f64 * tp && v < *pt.offset(xo as isize)
                        && v < *pt.offset(-(xo as isize)) && v < *pt.offset(so as isize)
                        && v < *pt.offset(-(so as isize)) && v < *pt.offset(yo as isize)
                        && v < *pt.offset(-(yo as isize))
                        && v < *pt.offset(yo as isize).offset(xo as isize)
                        && v < *pt.offset(yo as isize).offset(-(xo as isize))
                        && v < *pt.offset(-(yo as isize)).offset(xo as isize)
                        && v < *pt.offset(-(yo as isize)).offset(-(xo as isize))
                        && v < *pt.offset(xo as isize).offset(so as isize)
                        && v < *pt.offset(-(xo as isize)).offset(so as isize)
                        && v < *pt.offset(yo as isize).offset(so as isize)
                        && v < *pt.offset(-(yo as isize)).offset(so as isize)
                        && v
                            < *pt
                                .offset(yo as isize)
                                .offset(xo as isize)
                                .offset(so as isize)
                        && v
                            < *pt
                                .offset(yo as isize)
                                .offset(-(xo as isize))
                                .offset(so as isize)
                        && v
                            < *pt
                                .offset(-(yo as isize))
                                .offset(xo as isize)
                                .offset(so as isize)
                        && v
                            < *pt
                                .offset(-(yo as isize))
                                .offset(-(xo as isize))
                                .offset(so as isize)
                        && v < *pt.offset(xo as isize).offset(-(so as isize))
                        && v < *pt.offset(-(xo as isize)).offset(-(so as isize))
                        && v < *pt.offset(yo as isize).offset(-(so as isize))
                        && v < *pt.offset(-(yo as isize)).offset(-(so as isize))
                        && v
                            < *pt
                                .offset(yo as isize)
                                .offset(xo as isize)
                                .offset(-(so as isize))
                        && v
                            < *pt
                                .offset(yo as isize)
                                .offset(-(xo as isize))
                                .offset(-(so as isize))
                        && v
                            < *pt
                                .offset(-(yo as isize))
                                .offset(xo as isize)
                                .offset(-(so as isize))
                        && v
                            < *pt
                                .offset(-(yo as isize))
                                .offset(-(xo as isize))
                                .offset(-(so as isize))
                {
                    if (*f).nkeys >= (*f).keys_res {
                        (*f).keys_res += 500 as libc::c_int;
                        if !((*f).keys).is_null() {
                            (*f)
                                .keys = vl_realloc(
                                (*f).keys as *mut libc::c_void,
                                ((*f).keys_res as libc::c_ulong)
                                    .wrapping_mul(
                                        ::core::mem::size_of::<VlSiftKeypoint>() as libc::c_ulong,
                                    ),
                            ) as *mut VlSiftKeypoint;
                        } else {
                            (*f)
                                .keys = vl_malloc(
                                ((*f).keys_res as libc::c_ulong)
                                    .wrapping_mul(
                                        ::core::mem::size_of::<VlSiftKeypoint>() as libc::c_ulong,
                                    ),
                            ) as *mut VlSiftKeypoint;
                        }
                    }
                    let fresh7 = (*f).nkeys;
                    (*f).nkeys = (*f).nkeys + 1;
                    k = ((*f).keys).offset(fresh7 as isize);
                    (*k).ix = x;
                    (*k).iy = y;
                    (*k).is = s;
                }
                pt = pt.offset(1 as libc::c_int as isize);
                x += 1;
            }
            pt = pt.offset(2 as libc::c_int as isize);
            y += 1;
        }
        pt = pt.offset((2 as libc::c_int * yo) as isize);
        s += 1;
    }
    k = (*f).keys;
    i = 0 as libc::c_int;
    while i < (*f).nkeys {
        let mut x_0: libc::c_int = (*((*f).keys).offset(i as isize)).ix;
        let mut y_0: libc::c_int = (*((*f).keys).offset(i as isize)).iy;
        let mut s_0: libc::c_int = (*((*f).keys).offset(i as isize)).is;
        let mut Dx: libc::c_double = 0 as libc::c_int as libc::c_double;
        let mut Dy: libc::c_double = 0 as libc::c_int as libc::c_double;
        let mut Ds: libc::c_double = 0 as libc::c_int as libc::c_double;
        let mut Dxx: libc::c_double = 0 as libc::c_int as libc::c_double;
        let mut Dyy: libc::c_double = 0 as libc::c_int as libc::c_double;
        let mut Dss: libc::c_double = 0 as libc::c_int as libc::c_double;
        let mut Dxy: libc::c_double = 0 as libc::c_int as libc::c_double;
        let mut Dxs: libc::c_double = 0 as libc::c_int as libc::c_double;
        let mut Dys: libc::c_double = 0 as libc::c_int as libc::c_double;
        let mut A: [libc::c_double; 9] = [0.; 9];
        let mut b: [libc::c_double; 3] = [0.; 3];
        let mut dx: libc::c_int = 0 as libc::c_int;
        let mut dy: libc::c_int = 0 as libc::c_int;
        let mut iter: libc::c_int = 0;
        let mut i_0: libc::c_int = 0;
        let mut j: libc::c_int = 0;
        iter = 0 as libc::c_int;
        while iter < 5 as libc::c_int {
            x_0 += dx;
            y_0 += dy;
            pt = dog
                .offset((xo * x_0) as isize)
                .offset((yo * y_0) as isize)
                .offset((so * (s_0 - s_min)) as isize);
            Dx = 0.5f64
                * (*pt
                    .offset((1 as libc::c_int * xo) as isize)
                    .offset((0 as libc::c_int * yo) as isize)
                    .offset((0 as libc::c_int * so) as isize)
                    - *pt
                        .offset((-(1 as libc::c_int) * xo) as isize)
                        .offset((0 as libc::c_int * yo) as isize)
                        .offset((0 as libc::c_int * so) as isize)) as libc::c_double;
            Dy = 0.5f64
                * (*pt
                    .offset((0 as libc::c_int * xo) as isize)
                    .offset((1 as libc::c_int * yo) as isize)
                    .offset((0 as libc::c_int * so) as isize)
                    - *pt
                        .offset((0 as libc::c_int * xo) as isize)
                        .offset((-(1 as libc::c_int) * yo) as isize)
                        .offset((0 as libc::c_int * so) as isize)) as libc::c_double;
            Ds = 0.5f64
                * (*pt
                    .offset((0 as libc::c_int * xo) as isize)
                    .offset((0 as libc::c_int * yo) as isize)
                    .offset((1 as libc::c_int * so) as isize)
                    - *pt
                        .offset((0 as libc::c_int * xo) as isize)
                        .offset((0 as libc::c_int * yo) as isize)
                        .offset((-(1 as libc::c_int) * so) as isize)) as libc::c_double;
            Dxx = (*pt
                .offset((1 as libc::c_int * xo) as isize)
                .offset((0 as libc::c_int * yo) as isize)
                .offset((0 as libc::c_int * so) as isize)
                + *pt
                    .offset((-(1 as libc::c_int) * xo) as isize)
                    .offset((0 as libc::c_int * yo) as isize)
                    .offset((0 as libc::c_int * so) as isize)) as libc::c_double
                - 2.0f64
                    * *pt
                        .offset((0 as libc::c_int * xo) as isize)
                        .offset((0 as libc::c_int * yo) as isize)
                        .offset((0 as libc::c_int * so) as isize) as libc::c_double;
            Dyy = (*pt
                .offset((0 as libc::c_int * xo) as isize)
                .offset((1 as libc::c_int * yo) as isize)
                .offset((0 as libc::c_int * so) as isize)
                + *pt
                    .offset((0 as libc::c_int * xo) as isize)
                    .offset((-(1 as libc::c_int) * yo) as isize)
                    .offset((0 as libc::c_int * so) as isize)) as libc::c_double
                - 2.0f64
                    * *pt
                        .offset((0 as libc::c_int * xo) as isize)
                        .offset((0 as libc::c_int * yo) as isize)
                        .offset((0 as libc::c_int * so) as isize) as libc::c_double;
            Dss = (*pt
                .offset((0 as libc::c_int * xo) as isize)
                .offset((0 as libc::c_int * yo) as isize)
                .offset((1 as libc::c_int * so) as isize)
                + *pt
                    .offset((0 as libc::c_int * xo) as isize)
                    .offset((0 as libc::c_int * yo) as isize)
                    .offset((-(1 as libc::c_int) * so) as isize)) as libc::c_double
                - 2.0f64
                    * *pt
                        .offset((0 as libc::c_int * xo) as isize)
                        .offset((0 as libc::c_int * yo) as isize)
                        .offset((0 as libc::c_int * so) as isize) as libc::c_double;
            Dxy = 0.25f64
                * (*pt
                    .offset((1 as libc::c_int * xo) as isize)
                    .offset((1 as libc::c_int * yo) as isize)
                    .offset((0 as libc::c_int * so) as isize)
                    + *pt
                        .offset((-(1 as libc::c_int) * xo) as isize)
                        .offset((-(1 as libc::c_int) * yo) as isize)
                        .offset((0 as libc::c_int * so) as isize)
                    - *pt
                        .offset((-(1 as libc::c_int) * xo) as isize)
                        .offset((1 as libc::c_int * yo) as isize)
                        .offset((0 as libc::c_int * so) as isize)
                    - *pt
                        .offset((1 as libc::c_int * xo) as isize)
                        .offset((-(1 as libc::c_int) * yo) as isize)
                        .offset((0 as libc::c_int * so) as isize)) as libc::c_double;
            Dxs = 0.25f64
                * (*pt
                    .offset((1 as libc::c_int * xo) as isize)
                    .offset((0 as libc::c_int * yo) as isize)
                    .offset((1 as libc::c_int * so) as isize)
                    + *pt
                        .offset((-(1 as libc::c_int) * xo) as isize)
                        .offset((0 as libc::c_int * yo) as isize)
                        .offset((-(1 as libc::c_int) * so) as isize)
                    - *pt
                        .offset((-(1 as libc::c_int) * xo) as isize)
                        .offset((0 as libc::c_int * yo) as isize)
                        .offset((1 as libc::c_int * so) as isize)
                    - *pt
                        .offset((1 as libc::c_int * xo) as isize)
                        .offset((0 as libc::c_int * yo) as isize)
                        .offset((-(1 as libc::c_int) * so) as isize)) as libc::c_double;
            Dys = 0.25f64
                * (*pt
                    .offset((0 as libc::c_int * xo) as isize)
                    .offset((1 as libc::c_int * yo) as isize)
                    .offset((1 as libc::c_int * so) as isize)
                    + *pt
                        .offset((0 as libc::c_int * xo) as isize)
                        .offset((-(1 as libc::c_int) * yo) as isize)
                        .offset((-(1 as libc::c_int) * so) as isize)
                    - *pt
                        .offset((0 as libc::c_int * xo) as isize)
                        .offset((-(1 as libc::c_int) * yo) as isize)
                        .offset((1 as libc::c_int * so) as isize)
                    - *pt
                        .offset((0 as libc::c_int * xo) as isize)
                        .offset((1 as libc::c_int * yo) as isize)
                        .offset((-(1 as libc::c_int) * so) as isize)) as libc::c_double;
            A[(0 as libc::c_int + 0 as libc::c_int * 3 as libc::c_int) as usize] = Dxx;
            A[(1 as libc::c_int + 1 as libc::c_int * 3 as libc::c_int) as usize] = Dyy;
            A[(2 as libc::c_int + 2 as libc::c_int * 3 as libc::c_int) as usize] = Dss;
            A[(1 as libc::c_int + 0 as libc::c_int * 3 as libc::c_int) as usize] = Dxy;
            A[(0 as libc::c_int + 1 as libc::c_int * 3 as libc::c_int)
                as usize] = A[(1 as libc::c_int + 0 as libc::c_int * 3 as libc::c_int)
                as usize];
            A[(2 as libc::c_int + 0 as libc::c_int * 3 as libc::c_int) as usize] = Dxs;
            A[(0 as libc::c_int + 2 as libc::c_int * 3 as libc::c_int)
                as usize] = A[(2 as libc::c_int + 0 as libc::c_int * 3 as libc::c_int)
                as usize];
            A[(2 as libc::c_int + 1 as libc::c_int * 3 as libc::c_int) as usize] = Dys;
            A[(1 as libc::c_int + 2 as libc::c_int * 3 as libc::c_int)
                as usize] = A[(2 as libc::c_int + 1 as libc::c_int * 3 as libc::c_int)
                as usize];
            b[0 as libc::c_int as usize] = -Dx;
            b[1 as libc::c_int as usize] = -Dy;
            b[2 as libc::c_int as usize] = -Ds;
            j = 0 as libc::c_int;
            while j < 3 as libc::c_int {
                let mut maxa: libc::c_double = 0 as libc::c_int as libc::c_double;
                let mut maxabsa: libc::c_double = 0 as libc::c_int as libc::c_double;
                let mut maxi: libc::c_int = -(1 as libc::c_int);
                let mut tmp: libc::c_double = 0.;
                i_0 = j;
                while i_0 < 3 as libc::c_int {
                    let mut a: libc::c_double = A[(i_0 + j * 3 as libc::c_int) as usize];
                    let mut absa: libc::c_double = vl_abs_d(a);
                    if absa > maxabsa {
                        maxa = a;
                        maxabsa = absa;
                        maxi = i_0;
                    }
                    i_0 += 1;
                }
                if maxabsa < 1e-10f32 as libc::c_double {
                    b[0 as libc::c_int as usize] = 0 as libc::c_int as libc::c_double;
                    b[1 as libc::c_int as usize] = 0 as libc::c_int as libc::c_double;
                    b[2 as libc::c_int as usize] = 0 as libc::c_int as libc::c_double;
                    break;
                } else {
                    i_0 = maxi;
                    jj = j;
                    while jj < 3 as libc::c_int {
                        tmp = A[(i_0 + jj * 3 as libc::c_int) as usize];
                        A[(i_0 + jj * 3 as libc::c_int)
                            as usize] = A[(j + jj * 3 as libc::c_int) as usize];
                        A[(j + jj * 3 as libc::c_int) as usize] = tmp;
                        A[(j + jj * 3 as libc::c_int) as usize] /= maxa;
                        jj += 1;
                    }
                    tmp = b[j as usize];
                    b[j as usize] = b[i_0 as usize];
                    b[i_0 as usize] = tmp;
                    b[j as usize] /= maxa;
                    ii = j + 1 as libc::c_int;
                    while ii < 3 as libc::c_int {
                        let mut x_1: libc::c_double = A[(ii + j * 3 as libc::c_int)
                            as usize];
                        jj = j;
                        while jj < 3 as libc::c_int {
                            A[(ii + jj * 3 as libc::c_int) as usize]
                                -= x_1 * A[(j + jj * 3 as libc::c_int) as usize];
                            jj += 1;
                        }
                        b[ii as usize] -= x_1 * b[j as usize];
                        ii += 1;
                    }
                    j += 1;
                }
            }
            i_0 = 2 as libc::c_int;
            while i_0 > 0 as libc::c_int {
                let mut x_2: libc::c_double = b[i_0 as usize];
                ii = i_0 - 1 as libc::c_int;
                while ii >= 0 as libc::c_int {
                    b[ii as usize] -= x_2 * A[(ii + i_0 * 3 as libc::c_int) as usize];
                    ii -= 1;
                }
                i_0 -= 1;
            }
            dx = (if b[0 as libc::c_int as usize] > 0.6f64 && x_0 < w - 2 as libc::c_int
            {
                1 as libc::c_int
            } else {
                0 as libc::c_int
            })
                + (if b[0 as libc::c_int as usize] < -0.6f64 && x_0 > 1 as libc::c_int {
                    -(1 as libc::c_int)
                } else {
                    0 as libc::c_int
                });
            dy = (if b[1 as libc::c_int as usize] > 0.6f64 && y_0 < h - 2 as libc::c_int
            {
                1 as libc::c_int
            } else {
                0 as libc::c_int
            })
                + (if b[1 as libc::c_int as usize] < -0.6f64 && y_0 > 1 as libc::c_int {
                    -(1 as libc::c_int)
                } else {
                    0 as libc::c_int
                });
            if dx == 0 as libc::c_int && dy == 0 as libc::c_int {
                break;
            }
            iter += 1;
        }
        let mut val: libc::c_double = *pt
            .offset((0 as libc::c_int * xo) as isize)
            .offset((0 as libc::c_int * yo) as isize)
            .offset((0 as libc::c_int * so) as isize) as libc::c_double
            + 0.5f64
                * (Dx * b[0 as libc::c_int as usize] + Dy * b[1 as libc::c_int as usize]
                    + Ds * b[2 as libc::c_int as usize]);
        let mut score: libc::c_double = (Dxx + Dyy) * (Dxx + Dyy)
            / (Dxx * Dyy - Dxy * Dxy);
        let mut xn: libc::c_double = x_0 as libc::c_double
            + b[0 as libc::c_int as usize];
        let mut yn: libc::c_double = y_0 as libc::c_double
            + b[1 as libc::c_int as usize];
        let mut sn: libc::c_double = s_0 as libc::c_double
            + b[2 as libc::c_int as usize];
        let mut good: vl_bool = (vl_abs_d(val) > tp
            && score
                < (te + 1 as libc::c_int as libc::c_double)
                    * (te + 1 as libc::c_int as libc::c_double) / te
            && score >= 0 as libc::c_int as libc::c_double
            && vl_abs_d(b[0 as libc::c_int as usize]) < 1.5f64
            && vl_abs_d(b[1 as libc::c_int as usize]) < 1.5f64
            && vl_abs_d(b[2 as libc::c_int as usize]) < 1.5f64
            && xn >= 0 as libc::c_int as libc::c_double
            && xn <= (w - 1 as libc::c_int) as libc::c_double
            && yn >= 0 as libc::c_int as libc::c_double
            && yn <= (h - 1 as libc::c_int) as libc::c_double
            && sn >= s_min as libc::c_double && sn <= s_max as libc::c_double)
            as libc::c_int;
        if good != 0 {
            (*k).o = (*f).o_cur;
            (*k).ix = x_0;
            (*k).iy = y_0;
            (*k).is = s_0;
            (*k).s = sn as libc::c_float;
            (*k).x = (xn * xper) as libc::c_float;
            (*k).y = (yn * xper) as libc::c_float;
            (*k)
                .sigma = ((*f).sigma0 * pow(2.0f64, sn / (*f).S as libc::c_double)
                * xper) as libc::c_float;
            k = k.offset(1);
        }
        i += 1;
    }
    (*f).nkeys = k.offset_from((*f).keys) as libc::c_long as libc::c_int;
}
unsafe extern "C" fn update_gradient(mut f: *mut VlSiftFilt) {
    let mut s_min: libc::c_int = (*f).s_min;
    let mut s_max: libc::c_int = (*f).s_max;
    let mut w: libc::c_int = vl_sift_get_octave_width(f);
    let mut h: libc::c_int = vl_sift_get_octave_height(f);
    let xo: libc::c_int = 1 as libc::c_int;
    let yo: libc::c_int = w;
    let so: libc::c_int = h * w;
    let mut y: libc::c_int = 0;
    let mut s: libc::c_int = 0;
    if (*f).grad_o == (*f).o_cur {
        return;
    }
    s = s_min + 1 as libc::c_int;
    while s <= s_max - 2 as libc::c_int {
        let mut src: *mut vl_sift_pix = 0 as *mut vl_sift_pix;
        let mut end: *mut vl_sift_pix = 0 as *mut vl_sift_pix;
        let mut grad: *mut vl_sift_pix = 0 as *mut vl_sift_pix;
        let mut gx: vl_sift_pix = 0.;
        let mut gy: vl_sift_pix = 0.;
        src = vl_sift_get_octave(f, s);
        grad = ((*f).grad)
            .offset((2 as libc::c_int * so * (s - s_min - 1 as libc::c_int)) as isize);
        gx = *src.offset(xo as isize) - *src.offset(0 as libc::c_int as isize);
        gy = *src.offset(yo as isize) - *src.offset(0 as libc::c_int as isize);
        let fresh8 = grad;
        grad = grad.offset(1);
        *fresh8 = vl_fast_sqrt_f(gx * gx + gy * gy);
        let fresh9 = grad;
        grad = grad.offset(1);
        *fresh9 = vl_mod_2pi_f(
            (vl_fast_atan2_f(gy, gx) as libc::c_double
                + 2 as libc::c_int as libc::c_double * 3.141592653589793f64)
                as libc::c_float,
        );
        src = src.offset(1);
        end = src
            .offset(-(1 as libc::c_int as isize))
            .offset(w as isize)
            .offset(-(1 as libc::c_int as isize));
        while src < end {
            gx = (0.5f64
                * (*src.offset(xo as isize) - *src.offset(-xo as isize))
                    as libc::c_double) as vl_sift_pix;
            gy = *src.offset(yo as isize) - *src.offset(0 as libc::c_int as isize);
            let fresh10 = grad;
            grad = grad.offset(1);
            *fresh10 = vl_fast_sqrt_f(gx * gx + gy * gy);
            let fresh11 = grad;
            grad = grad.offset(1);
            *fresh11 = vl_mod_2pi_f(
                (vl_fast_atan2_f(gy, gx) as libc::c_double
                    + 2 as libc::c_int as libc::c_double * 3.141592653589793f64)
                    as libc::c_float,
            );
            src = src.offset(1);
        }
        gx = *src.offset(0 as libc::c_int as isize) - *src.offset(-xo as isize);
        gy = *src.offset(yo as isize) - *src.offset(0 as libc::c_int as isize);
        let fresh12 = grad;
        grad = grad.offset(1);
        *fresh12 = vl_fast_sqrt_f(gx * gx + gy * gy);
        let fresh13 = grad;
        grad = grad.offset(1);
        *fresh13 = vl_mod_2pi_f(
            (vl_fast_atan2_f(gy, gx) as libc::c_double
                + 2 as libc::c_int as libc::c_double * 3.141592653589793f64)
                as libc::c_float,
        );
        src = src.offset(1);
        y = 1 as libc::c_int;
        while y < h - 1 as libc::c_int {
            gx = *src.offset(xo as isize) - *src.offset(0 as libc::c_int as isize);
            gy = (0.5f64
                * (*src.offset(yo as isize) - *src.offset(-yo as isize))
                    as libc::c_double) as vl_sift_pix;
            let fresh14 = grad;
            grad = grad.offset(1);
            *fresh14 = vl_fast_sqrt_f(gx * gx + gy * gy);
            let fresh15 = grad;
            grad = grad.offset(1);
            *fresh15 = vl_mod_2pi_f(
                (vl_fast_atan2_f(gy, gx) as libc::c_double
                    + 2 as libc::c_int as libc::c_double * 3.141592653589793f64)
                    as libc::c_float,
            );
            src = src.offset(1);
            end = src
                .offset(-(1 as libc::c_int as isize))
                .offset(w as isize)
                .offset(-(1 as libc::c_int as isize));
            while src < end {
                gx = (0.5f64
                    * (*src.offset(xo as isize) - *src.offset(-xo as isize))
                        as libc::c_double) as vl_sift_pix;
                gy = (0.5f64
                    * (*src.offset(yo as isize) - *src.offset(-yo as isize))
                        as libc::c_double) as vl_sift_pix;
                let fresh16 = grad;
                grad = grad.offset(1);
                *fresh16 = vl_fast_sqrt_f(gx * gx + gy * gy);
                let fresh17 = grad;
                grad = grad.offset(1);
                *fresh17 = vl_mod_2pi_f(
                    (vl_fast_atan2_f(gy, gx) as libc::c_double
                        + 2 as libc::c_int as libc::c_double * 3.141592653589793f64)
                        as libc::c_float,
                );
                src = src.offset(1);
            }
            gx = *src.offset(0 as libc::c_int as isize) - *src.offset(-xo as isize);
            gy = (0.5f64
                * (*src.offset(yo as isize) - *src.offset(-yo as isize))
                    as libc::c_double) as vl_sift_pix;
            let fresh18 = grad;
            grad = grad.offset(1);
            *fresh18 = vl_fast_sqrt_f(gx * gx + gy * gy);
            let fresh19 = grad;
            grad = grad.offset(1);
            *fresh19 = vl_mod_2pi_f(
                (vl_fast_atan2_f(gy, gx) as libc::c_double
                    + 2 as libc::c_int as libc::c_double * 3.141592653589793f64)
                    as libc::c_float,
            );
            src = src.offset(1);
            y += 1;
        }
        gx = *src.offset(xo as isize) - *src.offset(0 as libc::c_int as isize);
        gy = *src.offset(0 as libc::c_int as isize) - *src.offset(-yo as isize);
        let fresh20 = grad;
        grad = grad.offset(1);
        *fresh20 = vl_fast_sqrt_f(gx * gx + gy * gy);
        let fresh21 = grad;
        grad = grad.offset(1);
        *fresh21 = vl_mod_2pi_f(
            (vl_fast_atan2_f(gy, gx) as libc::c_double
                + 2 as libc::c_int as libc::c_double * 3.141592653589793f64)
                as libc::c_float,
        );
        src = src.offset(1);
        end = src
            .offset(-(1 as libc::c_int as isize))
            .offset(w as isize)
            .offset(-(1 as libc::c_int as isize));
        while src < end {
            gx = (0.5f64
                * (*src.offset(xo as isize) - *src.offset(-xo as isize))
                    as libc::c_double) as vl_sift_pix;
            gy = *src.offset(0 as libc::c_int as isize) - *src.offset(-yo as isize);
            let fresh22 = grad;
            grad = grad.offset(1);
            *fresh22 = vl_fast_sqrt_f(gx * gx + gy * gy);
            let fresh23 = grad;
            grad = grad.offset(1);
            *fresh23 = vl_mod_2pi_f(
                (vl_fast_atan2_f(gy, gx) as libc::c_double
                    + 2 as libc::c_int as libc::c_double * 3.141592653589793f64)
                    as libc::c_float,
            );
            src = src.offset(1);
        }
        gx = *src.offset(0 as libc::c_int as isize) - *src.offset(-xo as isize);
        gy = *src.offset(0 as libc::c_int as isize) - *src.offset(-yo as isize);
        let fresh24 = grad;
        grad = grad.offset(1);
        *fresh24 = vl_fast_sqrt_f(gx * gx + gy * gy);
        let fresh25 = grad;
        grad = grad.offset(1);
        *fresh25 = vl_mod_2pi_f(
            (vl_fast_atan2_f(gy, gx) as libc::c_double
                + 2 as libc::c_int as libc::c_double * 3.141592653589793f64)
                as libc::c_float,
        );
        src = src.offset(1);
        s += 1;
    }
    (*f).grad_o = (*f).o_cur;
}
#[no_mangle]
pub unsafe extern "C" fn vl_sift_calc_keypoint_orientations(
    mut f: *mut VlSiftFilt,
    mut angles: *mut libc::c_double,
    mut k: *const VlSiftKeypoint,
) -> libc::c_int {
    let winf: libc::c_double = 1.5f64;
    let mut xper: libc::c_double = pow(2.0f64, (*f).o_cur as libc::c_double);
    let mut w: libc::c_int = (*f).octave_width;
    let mut h: libc::c_int = (*f).octave_height;
    let xo: libc::c_int = 2 as libc::c_int;
    let yo: libc::c_int = 2 as libc::c_int * w;
    let so: libc::c_int = 2 as libc::c_int * w * h;
    let mut x: libc::c_double = (*k).x as libc::c_double / xper;
    let mut y: libc::c_double = (*k).y as libc::c_double / xper;
    let mut sigma: libc::c_double = (*k).sigma as libc::c_double / xper;
    let mut xi: libc::c_int = (x + 0.5f64) as libc::c_int;
    let mut yi: libc::c_int = (y + 0.5f64) as libc::c_int;
    let mut si: libc::c_int = (*k).is;
    let sigmaw: libc::c_double = winf * sigma;
    let mut W: libc::c_int = (if floor(3.0f64 * sigmaw)
        > 1 as libc::c_int as libc::c_double
    {
        floor(3.0f64 * sigmaw)
    } else {
        1 as libc::c_int as libc::c_double
    }) as libc::c_int;
    let mut nangles: libc::c_int = 0 as libc::c_int;
    let mut hist: [libc::c_double; 36] = [0.; 36];
    let mut maxh: libc::c_double = 0.;
    let mut pt: *const vl_sift_pix = 0 as *const vl_sift_pix;
    let mut xs: libc::c_int = 0;
    let mut ys: libc::c_int = 0;
    let mut iter: libc::c_int = 0;
    let mut i: libc::c_int = 0;
    if (*k).o != (*f).o_cur {
        return 0 as libc::c_int;
    }
    if xi < 0 as libc::c_int || xi > w - 1 as libc::c_int || yi < 0 as libc::c_int
        || yi > h - 1 as libc::c_int || si < (*f).s_min + 1 as libc::c_int
        || si > (*f).s_max - 2 as libc::c_int
    {
        return 0 as libc::c_int;
    }
    update_gradient(f);
    memset(
        hist.as_mut_ptr() as *mut libc::c_void,
        0 as libc::c_int,
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong)
            .wrapping_mul(nbins as libc::c_int as libc::c_ulong),
    );
    pt = ((*f).grad)
        .offset((xo * xi) as isize)
        .offset((yo * yi) as isize)
        .offset((so * (si - (*f).s_min - 1 as libc::c_int)) as isize);
    ys = if -W > -yi { -W } else { -yi };
    while ys
        <= (if W < h - 1 as libc::c_int - yi { W } else { h - 1 as libc::c_int - yi })
    {
        xs = if -W > -xi { -W } else { -xi };
        while xs
            <= (if W < w - 1 as libc::c_int - xi {
                W
            } else {
                w - 1 as libc::c_int - xi
            })
        {
            let mut dx: libc::c_double = (xi + xs) as libc::c_double - x;
            let mut dy: libc::c_double = (yi + ys) as libc::c_double - y;
            let mut r2: libc::c_double = dx * dx + dy * dy;
            let mut wgt: libc::c_double = 0.;
            let mut mod_0: libc::c_double = 0.;
            let mut ang: libc::c_double = 0.;
            let mut fbin: libc::c_double = 0.;
            if !(r2 >= (W * W) as libc::c_double + 0.6f64) {
                wgt = fast_expn(
                    r2 / (2 as libc::c_int as libc::c_double * sigmaw * sigmaw),
                );
                mod_0 = *pt.offset((xs * xo) as isize).offset((ys * yo) as isize)
                    as libc::c_double;
                ang = *pt
                    .offset((xs * xo) as isize)
                    .offset((ys * yo) as isize)
                    .offset(1 as libc::c_int as isize) as libc::c_double;
                fbin = nbins as libc::c_int as libc::c_double * ang
                    / (2 as libc::c_int as libc::c_double * 3.141592653589793f64);
                let mut bin: libc::c_int = vl_floor_d(fbin - 0.5f64) as libc::c_int;
                let mut rbin: libc::c_double = fbin - bin as libc::c_double - 0.5f64;
                hist[((bin + nbins as libc::c_int) % nbins as libc::c_int) as usize]
                    += (1 as libc::c_int as libc::c_double - rbin) * mod_0 * wgt;
                hist[((bin + 1 as libc::c_int) % nbins as libc::c_int) as usize]
                    += rbin * mod_0 * wgt;
            }
            xs += 1;
        }
        ys += 1;
    }
    iter = 0 as libc::c_int;
    while iter < 6 as libc::c_int {
        let mut prev: libc::c_double = hist[(nbins as libc::c_int - 1 as libc::c_int)
            as usize];
        let mut first: libc::c_double = hist[0 as libc::c_int as usize];
        let mut i_0: libc::c_int = 0;
        i_0 = 0 as libc::c_int;
        while i_0 < nbins as libc::c_int - 1 as libc::c_int {
            let mut newh: libc::c_double = (prev + hist[i_0 as usize]
                + hist[((i_0 + 1 as libc::c_int) % nbins as libc::c_int) as usize])
                / 3.0f64;
            prev = hist[i_0 as usize];
            hist[i_0 as usize] = newh;
            i_0 += 1;
        }
        hist[i_0 as usize] = (prev + hist[i_0 as usize] + first) / 3.0f64;
        iter += 1;
    }
    maxh = 0 as libc::c_int as libc::c_double;
    i = 0 as libc::c_int;
    while i < nbins as libc::c_int {
        maxh = if maxh > hist[i as usize] { maxh } else { hist[i as usize] };
        i += 1;
    }
    nangles = 0 as libc::c_int;
    i = 0 as libc::c_int;
    while i < nbins as libc::c_int {
        let mut h0: libc::c_double = hist[i as usize];
        let mut hm: libc::c_double = hist[((i - 1 as libc::c_int + nbins as libc::c_int)
            % nbins as libc::c_int) as usize];
        let mut hp: libc::c_double = hist[((i + 1 as libc::c_int + nbins as libc::c_int)
            % nbins as libc::c_int) as usize];
        if h0 > 0.8f64 * maxh && h0 > hm && h0 > hp {
            let mut di: libc::c_double = -0.5f64 * (hp - hm)
                / (hp + hm - 2 as libc::c_int as libc::c_double * h0);
            let mut th: libc::c_double = 2 as libc::c_int as libc::c_double
                * 3.141592653589793f64 * (i as libc::c_double + di + 0.5f64)
                / nbins as libc::c_int as libc::c_double;
            let fresh26 = nangles;
            nangles = nangles + 1;
            *angles.offset(fresh26 as isize) = th;
            if nangles == 4 as libc::c_int {
                break;
            }
        }
        i += 1;
    }
    return nangles;
}
#[inline]
unsafe extern "C" fn normalize_histogram(
    mut begin: *mut vl_sift_pix,
    mut end: *mut vl_sift_pix,
) -> vl_sift_pix {
    let mut iter: *mut vl_sift_pix = 0 as *mut vl_sift_pix;
    let mut norm: vl_sift_pix = 0.0f64 as vl_sift_pix;
    iter = begin;
    while iter != end {
        norm += *iter * *iter;
        iter = iter.offset(1);
    }
    norm = vl_fast_sqrt_f(norm) + 1.19209290E-07f32;
    iter = begin;
    while iter != end {
        *iter /= norm;
        iter = iter.offset(1);
    }
    return norm;
}
#[no_mangle]
pub unsafe extern "C" fn vl_sift_calc_raw_descriptor(
    mut f: *const VlSiftFilt,
    mut grad: *const vl_sift_pix,
    mut descr: *mut vl_sift_pix,
    mut width: libc::c_int,
    mut height: libc::c_int,
    mut x: libc::c_double,
    mut y: libc::c_double,
    mut sigma: libc::c_double,
    mut angle0: libc::c_double,
) {
    let magnif: libc::c_double = (*f).magnif;
    let mut w: libc::c_int = width;
    let mut h: libc::c_int = height;
    let xo: libc::c_int = 2 as libc::c_int;
    let yo: libc::c_int = 2 as libc::c_int * w;
    let mut xi: libc::c_int = (x + 0.5f64) as libc::c_int;
    let mut yi: libc::c_int = (y + 0.5f64) as libc::c_int;
    let st0: libc::c_double = sin(angle0);
    let ct0: libc::c_double = cos(angle0);
    let SBP: libc::c_double = magnif * sigma + 2.220446049250313e-16f64;
    let W: libc::c_int = floor(
        sqrt(2.0f64) * SBP * (4 as libc::c_int + 1 as libc::c_int) as libc::c_double
            / 2.0f64 + 0.5f64,
    ) as libc::c_int;
    let binto: libc::c_int = 1 as libc::c_int;
    let binyo: libc::c_int = 8 as libc::c_int * 4 as libc::c_int;
    let binxo: libc::c_int = 8 as libc::c_int;
    let mut bin: libc::c_int = 0;
    let mut dxi: libc::c_int = 0;
    let mut dyi: libc::c_int = 0;
    let mut pt: *const vl_sift_pix = 0 as *const vl_sift_pix;
    let mut dpt: *mut vl_sift_pix = 0 as *mut vl_sift_pix;
    if xi < 0 as libc::c_int || xi >= w || yi < 0 as libc::c_int
        || yi >= h - 1 as libc::c_int
    {
        return;
    }
    memset(
        descr as *mut libc::c_void,
        0 as libc::c_int,
        (::core::mem::size_of::<vl_sift_pix>() as libc::c_ulong)
            .wrapping_mul(8 as libc::c_int as libc::c_ulong)
            .wrapping_mul(4 as libc::c_int as libc::c_ulong)
            .wrapping_mul(4 as libc::c_int as libc::c_ulong),
    );
    pt = grad.offset((xi * xo) as isize).offset((yi * yo) as isize);
    dpt = descr
        .offset((4 as libc::c_int / 2 as libc::c_int * binyo) as isize)
        .offset((4 as libc::c_int / 2 as libc::c_int * binxo) as isize);
    dyi = if -W > -yi { -W } else { -yi };
    while dyi
        <= (if W < h - yi - 1 as libc::c_int { W } else { h - yi - 1 as libc::c_int })
    {
        dxi = if -W > -xi { -W } else { -xi };
        while dxi
            <= (if W < w - xi - 1 as libc::c_int {
                W
            } else {
                w - xi - 1 as libc::c_int
            })
        {
            let mut mod_0: vl_sift_pix = *pt
                .offset((dxi * xo) as isize)
                .offset((dyi * yo) as isize)
                .offset(0 as libc::c_int as isize);
            let mut angle: vl_sift_pix = *pt
                .offset((dxi * xo) as isize)
                .offset((dyi * yo) as isize)
                .offset(1 as libc::c_int as isize);
            let mut theta: vl_sift_pix = vl_mod_2pi_f(
                (angle as libc::c_double - angle0) as libc::c_float,
            );
            let mut dx: vl_sift_pix = ((xi + dxi) as libc::c_double - x) as vl_sift_pix;
            let mut dy: vl_sift_pix = ((yi + dyi) as libc::c_double - y) as vl_sift_pix;
            let mut nx: vl_sift_pix = ((ct0 * dx as libc::c_double
                + st0 * dy as libc::c_double) / SBP) as vl_sift_pix;
            let mut ny: vl_sift_pix = ((-st0 * dx as libc::c_double
                + ct0 * dy as libc::c_double) / SBP) as vl_sift_pix;
            let mut nt: vl_sift_pix = ((8 as libc::c_int as libc::c_float * theta)
                as libc::c_double
                / (2 as libc::c_int as libc::c_double * 3.141592653589793f64))
                as vl_sift_pix;
            let wsigma: vl_sift_pix = (*f).windowSize as vl_sift_pix;
            let mut win: vl_sift_pix = fast_expn(
                (nx * nx + ny * ny) as libc::c_double
                    / (2.0f64 * wsigma as libc::c_double * wsigma as libc::c_double),
            ) as vl_sift_pix;
            let mut binx: libc::c_int = vl_floor_f(
                (nx as libc::c_double - 0.5f64) as libc::c_float,
            ) as libc::c_int;
            let mut biny: libc::c_int = vl_floor_f(
                (ny as libc::c_double - 0.5f64) as libc::c_float,
            ) as libc::c_int;
            let mut bint: libc::c_int = vl_floor_f(nt) as libc::c_int;
            let mut rbinx: vl_sift_pix = (nx as libc::c_double
                - (binx as libc::c_double + 0.5f64)) as vl_sift_pix;
            let mut rbiny: vl_sift_pix = (ny as libc::c_double
                - (biny as libc::c_double + 0.5f64)) as vl_sift_pix;
            let mut rbint: vl_sift_pix = nt - bint as libc::c_float;
            let mut dbinx: libc::c_int = 0;
            let mut dbiny: libc::c_int = 0;
            let mut dbint: libc::c_int = 0;
            dbinx = 0 as libc::c_int;
            while dbinx < 2 as libc::c_int {
                dbiny = 0 as libc::c_int;
                while dbiny < 2 as libc::c_int {
                    dbint = 0 as libc::c_int;
                    while dbint < 2 as libc::c_int {
                        if binx + dbinx >= -(4 as libc::c_int / 2 as libc::c_int)
                            && binx + dbinx < 4 as libc::c_int / 2 as libc::c_int
                            && biny + dbiny >= -(4 as libc::c_int / 2 as libc::c_int)
                            && biny + dbiny < 4 as libc::c_int / 2 as libc::c_int
                        {
                            let mut weight: vl_sift_pix = win * mod_0
                                * vl_abs_f(
                                    (1 as libc::c_int - dbinx) as libc::c_float - rbinx,
                                )
                                * vl_abs_f(
                                    (1 as libc::c_int - dbiny) as libc::c_float - rbiny,
                                )
                                * vl_abs_f(
                                    (1 as libc::c_int - dbint) as libc::c_float - rbint,
                                );
                            let ref mut fresh27 = *dpt
                                .offset(
                                    ((bint + dbint) % 8 as libc::c_int * binto) as isize,
                                )
                                .offset(((biny + dbiny) * binyo) as isize)
                                .offset(((binx + dbinx) * binxo) as isize);
                            *fresh27 += weight;
                        }
                        dbint += 1;
                    }
                    dbiny += 1;
                }
                dbinx += 1;
            }
            dxi += 1;
        }
        dyi += 1;
    }
    let mut norm: vl_sift_pix = normalize_histogram(
        descr,
        descr.offset((8 as libc::c_int * 4 as libc::c_int * 4 as libc::c_int) as isize),
    );
    let mut numSamples: libc::c_int = ((if W < w - xi - 1 as libc::c_int {
        W
    } else {
        w - xi - 1 as libc::c_int
    }) - (if -W > -xi { -W } else { -xi }) + 1 as libc::c_int)
        * ((if W < h - yi - 1 as libc::c_int { W } else { h - yi - 1 as libc::c_int })
            - (if -W > -yi { -W } else { -yi }) + 1 as libc::c_int);
    if (*f).norm_thresh != 0.
        && (norm as libc::c_double) < (*f).norm_thresh * numSamples as libc::c_double
    {
        bin = 0 as libc::c_int;
        while bin < 8 as libc::c_int * 4 as libc::c_int * 4 as libc::c_int {
            *descr.offset(bin as isize) = 0 as libc::c_int as vl_sift_pix;
            bin += 1;
        }
    } else {
        bin = 0 as libc::c_int;
        while bin < 8 as libc::c_int * 4 as libc::c_int * 4 as libc::c_int {
            if *descr.offset(bin as isize) as libc::c_double > 0.2f64 {
                *descr.offset(bin as isize) = 0.2f64 as vl_sift_pix;
            }
            bin += 1;
        }
        normalize_histogram(
            descr,
            descr
                .offset(
                    (8 as libc::c_int * 4 as libc::c_int * 4 as libc::c_int) as isize,
                ),
        );
    };
}
#[no_mangle]
pub unsafe extern "C" fn vl_sift_calc_keypoint_descriptor(
    mut f: *mut VlSiftFilt,
    mut descr: *mut vl_sift_pix,
    mut k: *const VlSiftKeypoint,
    mut angle0: libc::c_double,
) {
    let magnif: libc::c_double = (*f).magnif;
    let mut xper: libc::c_double = pow(2.0f64, (*f).o_cur as libc::c_double);
    let mut w: libc::c_int = (*f).octave_width;
    let mut h: libc::c_int = (*f).octave_height;
    let xo: libc::c_int = 2 as libc::c_int;
    let yo: libc::c_int = 2 as libc::c_int * w;
    let so: libc::c_int = 2 as libc::c_int * w * h;
    let mut x: libc::c_double = (*k).x as libc::c_double / xper;
    let mut y: libc::c_double = (*k).y as libc::c_double / xper;
    let mut sigma: libc::c_double = (*k).sigma as libc::c_double / xper;
    let mut xi: libc::c_int = (x + 0.5f64) as libc::c_int;
    let mut yi: libc::c_int = (y + 0.5f64) as libc::c_int;
    let mut si: libc::c_int = (*k).is;
    let st0: libc::c_double = sin(angle0);
    let ct0: libc::c_double = cos(angle0);
    let SBP: libc::c_double = magnif * sigma + 2.220446049250313e-16f64;
    let W: libc::c_int = floor(
        sqrt(2.0f64) * SBP * (4 as libc::c_int + 1 as libc::c_int) as libc::c_double
            / 2.0f64 + 0.5f64,
    ) as libc::c_int;
    let binto: libc::c_int = 1 as libc::c_int;
    let binyo: libc::c_int = 8 as libc::c_int * 4 as libc::c_int;
    let binxo: libc::c_int = 8 as libc::c_int;
    let mut bin: libc::c_int = 0;
    let mut dxi: libc::c_int = 0;
    let mut dyi: libc::c_int = 0;
    let mut pt: *const vl_sift_pix = 0 as *const vl_sift_pix;
    let mut dpt: *mut vl_sift_pix = 0 as *mut vl_sift_pix;
    if (*k).o != (*f).o_cur || xi < 0 as libc::c_int || xi >= w || yi < 0 as libc::c_int
        || yi >= h - 1 as libc::c_int || si < (*f).s_min + 1 as libc::c_int
        || si > (*f).s_max - 2 as libc::c_int
    {
        return;
    }
    update_gradient(f);
    memset(
        descr as *mut libc::c_void,
        0 as libc::c_int,
        (::core::mem::size_of::<vl_sift_pix>() as libc::c_ulong)
            .wrapping_mul(8 as libc::c_int as libc::c_ulong)
            .wrapping_mul(4 as libc::c_int as libc::c_ulong)
            .wrapping_mul(4 as libc::c_int as libc::c_ulong),
    );
    pt = ((*f).grad)
        .offset((xi * xo) as isize)
        .offset((yi * yo) as isize)
        .offset(((si - (*f).s_min - 1 as libc::c_int) * so) as isize);
    dpt = descr
        .offset((4 as libc::c_int / 2 as libc::c_int * binyo) as isize)
        .offset((4 as libc::c_int / 2 as libc::c_int * binxo) as isize);
    dyi = if -W > 1 as libc::c_int - yi { -W } else { 1 as libc::c_int - yi };
    while dyi
        <= (if W < h - yi - 2 as libc::c_int { W } else { h - yi - 2 as libc::c_int })
    {
        dxi = if -W > 1 as libc::c_int - xi { -W } else { 1 as libc::c_int - xi };
        while dxi
            <= (if W < w - xi - 2 as libc::c_int {
                W
            } else {
                w - xi - 2 as libc::c_int
            })
        {
            let mut mod_0: vl_sift_pix = *pt
                .offset((dxi * xo) as isize)
                .offset((dyi * yo) as isize)
                .offset(0 as libc::c_int as isize);
            let mut angle: vl_sift_pix = *pt
                .offset((dxi * xo) as isize)
                .offset((dyi * yo) as isize)
                .offset(1 as libc::c_int as isize);
            let mut theta: vl_sift_pix = vl_mod_2pi_f(
                (angle as libc::c_double - angle0) as libc::c_float,
            );
            let mut dx: vl_sift_pix = ((xi + dxi) as libc::c_double - x) as vl_sift_pix;
            let mut dy: vl_sift_pix = ((yi + dyi) as libc::c_double - y) as vl_sift_pix;
            let mut nx: vl_sift_pix = ((ct0 * dx as libc::c_double
                + st0 * dy as libc::c_double) / SBP) as vl_sift_pix;
            let mut ny: vl_sift_pix = ((-st0 * dx as libc::c_double
                + ct0 * dy as libc::c_double) / SBP) as vl_sift_pix;
            let mut nt: vl_sift_pix = ((8 as libc::c_int as libc::c_float * theta)
                as libc::c_double
                / (2 as libc::c_int as libc::c_double * 3.141592653589793f64))
                as vl_sift_pix;
            let wsigma: vl_sift_pix = (*f).windowSize as vl_sift_pix;
            let mut win: vl_sift_pix = fast_expn(
                (nx * nx + ny * ny) as libc::c_double
                    / (2.0f64 * wsigma as libc::c_double * wsigma as libc::c_double),
            ) as vl_sift_pix;
            let mut binx: libc::c_int = vl_floor_f(
                (nx as libc::c_double - 0.5f64) as libc::c_float,
            ) as libc::c_int;
            let mut biny: libc::c_int = vl_floor_f(
                (ny as libc::c_double - 0.5f64) as libc::c_float,
            ) as libc::c_int;
            let mut bint: libc::c_int = vl_floor_f(nt) as libc::c_int;
            let mut rbinx: vl_sift_pix = (nx as libc::c_double
                - (binx as libc::c_double + 0.5f64)) as vl_sift_pix;
            let mut rbiny: vl_sift_pix = (ny as libc::c_double
                - (biny as libc::c_double + 0.5f64)) as vl_sift_pix;
            let mut rbint: vl_sift_pix = nt - bint as libc::c_float;
            let mut dbinx: libc::c_int = 0;
            let mut dbiny: libc::c_int = 0;
            let mut dbint: libc::c_int = 0;
            dbinx = 0 as libc::c_int;
            while dbinx < 2 as libc::c_int {
                dbiny = 0 as libc::c_int;
                while dbiny < 2 as libc::c_int {
                    dbint = 0 as libc::c_int;
                    while dbint < 2 as libc::c_int {
                        if binx + dbinx >= -(4 as libc::c_int / 2 as libc::c_int)
                            && binx + dbinx < 4 as libc::c_int / 2 as libc::c_int
                            && biny + dbiny >= -(4 as libc::c_int / 2 as libc::c_int)
                            && biny + dbiny < 4 as libc::c_int / 2 as libc::c_int
                        {
                            let mut weight: vl_sift_pix = win * mod_0
                                * vl_abs_f(
                                    (1 as libc::c_int - dbinx) as libc::c_float - rbinx,
                                )
                                * vl_abs_f(
                                    (1 as libc::c_int - dbiny) as libc::c_float - rbiny,
                                )
                                * vl_abs_f(
                                    (1 as libc::c_int - dbint) as libc::c_float - rbint,
                                );
                            let ref mut fresh28 = *dpt
                                .offset(
                                    ((bint + dbint) % 8 as libc::c_int * binto) as isize,
                                )
                                .offset(((biny + dbiny) * binyo) as isize)
                                .offset(((binx + dbinx) * binxo) as isize);
                            *fresh28 += weight;
                        }
                        dbint += 1;
                    }
                    dbiny += 1;
                }
                dbinx += 1;
            }
            dxi += 1;
        }
        dyi += 1;
    }
    let mut norm: vl_sift_pix = normalize_histogram(
        descr,
        descr.offset((8 as libc::c_int * 4 as libc::c_int * 4 as libc::c_int) as isize),
    );
    if (*f).norm_thresh != 0. && (norm as libc::c_double) < (*f).norm_thresh {
        bin = 0 as libc::c_int;
        while bin < 8 as libc::c_int * 4 as libc::c_int * 4 as libc::c_int {
            *descr.offset(bin as isize) = 0 as libc::c_int as vl_sift_pix;
            bin += 1;
        }
    } else {
        bin = 0 as libc::c_int;
        while bin < 8 as libc::c_int * 4 as libc::c_int * 4 as libc::c_int {
            if *descr.offset(bin as isize) as libc::c_double > 0.2f64 {
                *descr.offset(bin as isize) = 0.2f64 as vl_sift_pix;
            }
            bin += 1;
        }
        normalize_histogram(
            descr,
            descr
                .offset(
                    (8 as libc::c_int * 4 as libc::c_int * 4 as libc::c_int) as isize,
                ),
        );
    };
}
#[no_mangle]
pub unsafe extern "C" fn vl_sift_keypoint_init(
    mut f: *const VlSiftFilt,
    mut k: *mut VlSiftKeypoint,
    mut x: libc::c_double,
    mut y: libc::c_double,
    mut sigma: libc::c_double,
) {
    let mut o: libc::c_int = 0;
    let mut ix: libc::c_int = 0;
    let mut iy: libc::c_int = 0;
    let mut is: libc::c_int = 0;
    let mut s: libc::c_double = 0.;
    let mut phi: libc::c_double = 0.;
    let mut xper: libc::c_double = 0.;
    phi = log((sigma + 2.220446049250313e-16f64) / (*f).sigma0) / 0.693147180559945f64;
    o = vl_floor_d(
        phi - ((*f).s_min as libc::c_double + 0.5f64) / (*f).S as libc::c_double,
    ) as libc::c_int;
    o = if o < (*f).o_min + (*f).O - 1 as libc::c_int {
        o
    } else {
        (*f).o_min + (*f).O - 1 as libc::c_int
    };
    o = if o > (*f).o_min { o } else { (*f).o_min };
    s = (*f).S as libc::c_double * (phi - o as libc::c_double);
    is = (s + 0.5f64) as libc::c_int;
    is = if is < (*f).s_max - 2 as libc::c_int {
        is
    } else {
        (*f).s_max - 2 as libc::c_int
    };
    is = if is > (*f).s_min + 1 as libc::c_int {
        is
    } else {
        (*f).s_min + 1 as libc::c_int
    };
    xper = pow(2.0f64, o as libc::c_double);
    ix = (x / xper + 0.5f64) as libc::c_int;
    iy = (y / xper + 0.5f64) as libc::c_int;
    (*k).o = o;
    (*k).ix = ix;
    (*k).iy = iy;
    (*k).is = is;
    (*k).x = x as libc::c_float;
    (*k).y = y as libc::c_float;
    (*k).s = s as libc::c_float;
    (*k).sigma = sigma as libc::c_float;
}
