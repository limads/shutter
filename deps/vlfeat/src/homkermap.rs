use ::libc;
extern "C" {
    fn abort() -> !;
    fn __assert_fail(
        __assertion: *const libc::c_char,
        __file: *const libc::c_char,
        __line: libc::c_uint,
        __function: *const libc::c_char,
    ) -> !;
    fn vl_malloc(n: size_t) -> *mut libc::c_void;
    fn vl_free(ptr: *mut libc::c_void);
    fn cos(_: libc::c_double) -> libc::c_double;
    fn sin(_: libc::c_double) -> libc::c_double;
    fn exp(_: libc::c_double) -> libc::c_double;
    fn frexp(_: libc::c_double, _: *mut libc::c_int) -> libc::c_double;
    fn ldexp(_: libc::c_double, _: libc::c_int) -> libc::c_double;
    fn log(_: libc::c_double) -> libc::c_double;
    fn pow(_: libc::c_double, _: libc::c_double) -> libc::c_double;
    fn sqrt(_: libc::c_double) -> libc::c_double;
}
pub type vl_int64 = libc::c_longlong;
pub type vl_uint64 = libc::c_ulonglong;
pub type vl_size = vl_uint64;
pub type vl_index = vl_int64;
pub type vl_uindex = vl_uint64;
pub type size_t = libc::c_ulong;
pub type VlHomogeneousKernelType = libc::c_uint;
pub const VlHomogeneousKernelJS: VlHomogeneousKernelType = 2;
pub const VlHomogeneousKernelChi2: VlHomogeneousKernelType = 1;
pub const VlHomogeneousKernelIntersection: VlHomogeneousKernelType = 0;
pub type VlHomogeneousKernelMapWindowType = libc::c_uint;
pub const VlHomogeneousKernelMapWindowRectangular: VlHomogeneousKernelMapWindowType = 1;
pub const VlHomogeneousKernelMapWindowUniform: VlHomogeneousKernelMapWindowType = 0;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlHomogeneousKernelMap {
    pub kernelType: VlHomogeneousKernelType,
    pub gamma: libc::c_double,
    pub windowType: VlHomogeneousKernelMapWindowType,
    pub order: vl_size,
    pub period: libc::c_double,
    pub numSubdivisions: vl_size,
    pub subdivision: libc::c_double,
    pub minExponent: vl_index,
    pub maxExponent: vl_index,
    pub table: *mut libc::c_double,
}
pub type VlHomogeneousKernelMap = _VlHomogeneousKernelMap;
#[inline]
unsafe extern "C" fn vl_homogeneouskernelmap_get_spectrum(
    mut self_0: *const VlHomogeneousKernelMap,
    mut omega: libc::c_double,
) -> libc::c_double {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/homkermap.c\0" as *const u8 as *const libc::c_char,
            242 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 84],
                &[libc::c_char; 84],
            >(
                b"double vl_homogeneouskernelmap_get_spectrum(const VlHomogeneousKernelMap *, double)\0",
            ))
                .as_ptr(),
        );
    }
    match (*self_0).kernelType as libc::c_uint {
        0 => {
            return 2.0f64 / 3.141592653589793f64
                / (1 as libc::c_int as libc::c_double
                    + 4 as libc::c_int as libc::c_double * omega * omega);
        }
        1 => {
            return 2.0f64
                / (exp(3.141592653589793f64 * omega)
                    + exp(-3.141592653589793f64 * omega));
        }
        2 => {
            return 2.0f64 / log(4.0f64) * 2.0f64
                / (exp(3.141592653589793f64 * omega)
                    + exp(-3.141592653589793f64 * omega))
                / (1 as libc::c_int as libc::c_double
                    + 4 as libc::c_int as libc::c_double * omega * omega);
        }
        _ => {
            abort();
        }
    };
}
#[inline]
unsafe extern "C" fn sinc(mut x: libc::c_double) -> libc::c_double {
    if x == 0.0f64 {
        return 1.0f64;
    }
    return sin(x) / x;
}
#[inline]
unsafe extern "C" fn vl_homogeneouskernelmap_get_smooth_spectrum(
    mut self_0: *const VlHomogeneousKernelMap,
    mut omega: libc::c_double,
) -> libc::c_double {
    let mut kappa_hat: libc::c_double = 0 as libc::c_int as libc::c_double;
    let mut omegap: libc::c_double = 0.;
    let mut epsilon: libc::c_double = 1e-2f64;
    let omegaRange: libc::c_double = 2.0f64 / ((*self_0).period * epsilon);
    let domega: libc::c_double = 2 as libc::c_int as libc::c_double * omegaRange
        / (2 as libc::c_int as libc::c_double * 1024.0f64
            + 1 as libc::c_int as libc::c_double);
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/homkermap.c\0" as *const u8 as *const libc::c_char,
            278 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 91],
                &[libc::c_char; 91],
            >(
                b"double vl_homogeneouskernelmap_get_smooth_spectrum(const VlHomogeneousKernelMap *, double)\0",
            ))
                .as_ptr(),
        );
    }
    match (*self_0).windowType as libc::c_uint {
        0 => {
            kappa_hat = vl_homogeneouskernelmap_get_spectrum(self_0, omega);
        }
        1 => {
            omegap = -omegaRange;
            while omegap <= omegaRange {
                let mut win: libc::c_double = sinc((*self_0).period / 2.0f64 * omegap);
                win *= (*self_0).period / (2.0f64 * 3.141592653589793f64);
                kappa_hat
                    += win
                        * vl_homogeneouskernelmap_get_spectrum(self_0, omegap + omega);
                omegap += domega;
            }
            kappa_hat *= domega;
            kappa_hat = if kappa_hat > 0.0f64 { kappa_hat } else { 0.0f64 };
        }
        _ => {
            abort();
        }
    }
    return kappa_hat;
}
#[no_mangle]
pub unsafe extern "C" fn vl_homogeneouskernelmap_new(
    mut kernelType: VlHomogeneousKernelType,
    mut gamma: libc::c_double,
    mut order: vl_size,
    mut period: libc::c_double,
    mut windowType: VlHomogeneousKernelMapWindowType,
) -> *mut VlHomogeneousKernelMap {
    let mut tableWidth: libc::c_int = 0;
    let mut tableHeight: libc::c_int = 0;
    let mut self_0: *mut VlHomogeneousKernelMap = vl_malloc(
        ::core::mem::size_of::<VlHomogeneousKernelMap>() as libc::c_ulong,
    ) as *mut VlHomogeneousKernelMap;
    if self_0.is_null() {
        return 0 as *mut VlHomogeneousKernelMap;
    }
    if gamma > 0 as libc::c_int as libc::c_double {} else {
        __assert_fail(
            b"gamma > 0\0" as *const u8 as *const libc::c_char,
            b"vl/homkermap.c\0" as *const u8 as *const libc::c_char,
            336 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 136],
                &[libc::c_char; 136],
            >(
                b"VlHomogeneousKernelMap *vl_homogeneouskernelmap_new(VlHomogeneousKernelType, double, vl_size, double, VlHomogeneousKernelMapWindowType)\0",
            ))
                .as_ptr(),
        );
    }
    if kernelType as libc::c_uint
        == VlHomogeneousKernelIntersection as libc::c_int as libc::c_uint
        || kernelType as libc::c_uint
            == VlHomogeneousKernelChi2 as libc::c_int as libc::c_uint
        || kernelType as libc::c_uint
            == VlHomogeneousKernelJS as libc::c_int as libc::c_uint
    {} else {
        __assert_fail(
            b"kernelType == VlHomogeneousKernelIntersection || kernelType == VlHomogeneousKernelChi2 || kernelType == VlHomogeneousKernelJS\0"
                as *const u8 as *const libc::c_char,
            b"vl/homkermap.c\0" as *const u8 as *const libc::c_char,
            340 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 136],
                &[libc::c_char; 136],
            >(
                b"VlHomogeneousKernelMap *vl_homogeneouskernelmap_new(VlHomogeneousKernelType, double, vl_size, double, VlHomogeneousKernelMapWindowType)\0",
            ))
                .as_ptr(),
        );
    }
    if windowType as libc::c_uint
        == VlHomogeneousKernelMapWindowUniform as libc::c_int as libc::c_uint
        || windowType as libc::c_uint
            == VlHomogeneousKernelMapWindowRectangular as libc::c_int as libc::c_uint
    {} else {
        __assert_fail(
            b"windowType == VlHomogeneousKernelMapWindowUniform || windowType == VlHomogeneousKernelMapWindowRectangular\0"
                as *const u8 as *const libc::c_char,
            b"vl/homkermap.c\0" as *const u8 as *const libc::c_char,
            343 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 136],
                &[libc::c_char; 136],
            >(
                b"VlHomogeneousKernelMap *vl_homogeneouskernelmap_new(VlHomogeneousKernelType, double, vl_size, double, VlHomogeneousKernelMapWindowType)\0",
            ))
                .as_ptr(),
        );
    }
    if period < 0 as libc::c_int as libc::c_double {
        match windowType as libc::c_uint {
            0 => {
                match kernelType as libc::c_uint {
                    1 => {
                        period = 5.86f64
                            * sqrt(
                                order.wrapping_add(0 as libc::c_int as libc::c_ulonglong)
                                    as libc::c_double,
                            ) + 3.65f64;
                    }
                    2 => {
                        period = 6.64f64
                            * sqrt(
                                order.wrapping_add(0 as libc::c_int as libc::c_ulonglong)
                                    as libc::c_double,
                            ) + 7.24f64;
                    }
                    0 => {
                        period = 2.38f64 * log(order as libc::c_double + 0.8f64)
                            + 5.6f64;
                    }
                    _ => {}
                }
            }
            1 => {
                match kernelType as libc::c_uint {
                    1 => {
                        period = 8.80f64 * sqrt(order as libc::c_double + 4.44f64)
                            - 12.6f64;
                    }
                    2 => {
                        period = 9.63f64 * sqrt(order as libc::c_double + 1.00f64)
                            - 2.93f64;
                    }
                    0 => {
                        period = 2.00f64 * log(order as libc::c_double + 0.99f64)
                            + 3.52f64;
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        period = if period > 1.0f64 { period } else { 1.0f64 };
    }
    (*self_0).kernelType = kernelType;
    (*self_0).windowType = windowType;
    (*self_0).gamma = gamma;
    (*self_0).order = order;
    (*self_0).period = period;
    (*self_0)
        .numSubdivisions = (8 as libc::c_int as libc::c_ulonglong)
        .wrapping_add((8 as libc::c_int as libc::c_ulonglong).wrapping_mul(order));
    (*self_0).subdivision = 1.0f64 / (*self_0).numSubdivisions as libc::c_double;
    (*self_0).minExponent = -(20 as libc::c_int) as vl_index;
    (*self_0).maxExponent = 8 as libc::c_int as vl_index;
    tableHeight = (2 as libc::c_int as libc::c_ulonglong)
        .wrapping_mul((*self_0).order)
        .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as libc::c_int;
    tableWidth = ((*self_0).numSubdivisions)
        .wrapping_mul(
            ((*self_0).maxExponent - (*self_0).minExponent
                + 1 as libc::c_int as libc::c_longlong) as libc::c_ulonglong,
        ) as libc::c_int;
    (*self_0)
        .table = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(
                ((tableHeight * tableWidth) as libc::c_ulonglong)
                    .wrapping_add(
                        (2 as libc::c_int as libc::c_ulonglong)
                            .wrapping_mul(
                                (1 as libc::c_int as libc::c_ulonglong)
                                    .wrapping_add((*self_0).order),
                            ),
                    ),
            ) as size_t,
    ) as *mut libc::c_double;
    if ((*self_0).table).is_null() {
        vl_free(self_0 as *mut libc::c_void);
        return 0 as *mut VlHomogeneousKernelMap;
    }
    let mut exponent: vl_index = 0;
    let mut i: vl_uindex = 0;
    let mut j: vl_uindex = 0;
    let mut tablep: *mut libc::c_double = (*self_0).table;
    let mut kappa: *mut libc::c_double = ((*self_0).table)
        .offset((tableHeight * tableWidth) as isize);
    let mut freq: *mut libc::c_double = kappa
        .offset(
            (1 as libc::c_int as libc::c_ulonglong).wrapping_add((*self_0).order)
                as isize,
        );
    let mut L: libc::c_double = 2.0f64 * 3.141592653589793f64 / (*self_0).period;
    j = 0 as libc::c_int as vl_uindex;
    i = 0 as libc::c_int as vl_uindex;
    while i <= (*self_0).order {
        *freq.offset(i as isize) = j as libc::c_double;
        *kappa
            .offset(
                i as isize,
            ) = vl_homogeneouskernelmap_get_smooth_spectrum(
            self_0,
            j as libc::c_double * L,
        );
        j = j.wrapping_add(1);
        if *kappa.offset(i as isize) > 0 as libc::c_int as libc::c_double
            || j >= (3 as libc::c_int as libc::c_ulonglong).wrapping_mul(i)
        {
            i = i.wrapping_add(1);
        }
    }
    exponent = (*self_0).minExponent;
    while exponent <= (*self_0).maxExponent {
        let mut x: libc::c_double = 0.;
        let mut Lxgamma: libc::c_double = 0.;
        let mut Llogx: libc::c_double = 0.;
        let mut xgamma: libc::c_double = 0.;
        let mut sqrt2kappaLxgamma: libc::c_double = 0.;
        let mut mantissa: libc::c_double = 1.0f64;
        i = 0 as libc::c_int as vl_uindex;
        while i < (*self_0).numSubdivisions {
            x = ldexp(mantissa, exponent as libc::c_int);
            xgamma = pow(x, (*self_0).gamma);
            Lxgamma = L * xgamma;
            Llogx = L * log(x);
            let fresh0 = tablep;
            tablep = tablep.offset(1);
            *fresh0 = sqrt(Lxgamma * *kappa.offset(0 as libc::c_int as isize));
            j = 1 as libc::c_int as vl_uindex;
            while j <= (*self_0).order {
                sqrt2kappaLxgamma = sqrt(2.0f64 * Lxgamma * *kappa.offset(j as isize));
                let fresh1 = tablep;
                tablep = tablep.offset(1);
                *fresh1 = sqrt2kappaLxgamma * cos(*freq.offset(j as isize) * Llogx);
                let fresh2 = tablep;
                tablep = tablep.offset(1);
                *fresh2 = sqrt2kappaLxgamma * sin(*freq.offset(j as isize) * Llogx);
                j = j.wrapping_add(1);
            }
            i = i.wrapping_add(1);
            mantissa += (*self_0).subdivision;
        }
        exponent += 1;
    }
    return self_0;
}
#[no_mangle]
pub unsafe extern "C" fn vl_homogeneouskernelmap_delete(
    mut self_0: *mut VlHomogeneousKernelMap,
) {
    vl_free((*self_0).table as *mut libc::c_void);
    (*self_0).table = 0 as *mut libc::c_double;
    vl_free(self_0 as *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn vl_homogeneouskernelmap_get_order(
    mut self_0: *const VlHomogeneousKernelMap,
) -> vl_size {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/homkermap.c\0" as *const u8 as *const libc::c_char,
            454 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 74],
                &[libc::c_char; 74],
            >(
                b"vl_size vl_homogeneouskernelmap_get_order(const VlHomogeneousKernelMap *)\0",
            ))
                .as_ptr(),
        );
    }
    return (*self_0).order;
}
#[no_mangle]
pub unsafe extern "C" fn vl_homogeneouskernelmap_get_dimension(
    mut self_0: *const VlHomogeneousKernelMap,
) -> vl_size {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/homkermap.c\0" as *const u8 as *const libc::c_char,
            466 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 78],
                &[libc::c_char; 78],
            >(
                b"vl_size vl_homogeneouskernelmap_get_dimension(const VlHomogeneousKernelMap *)\0",
            ))
                .as_ptr(),
        );
    }
    return (2 as libc::c_int as libc::c_ulonglong)
        .wrapping_mul((*self_0).order)
        .wrapping_add(1 as libc::c_int as libc::c_ulonglong);
}
#[no_mangle]
pub unsafe extern "C" fn vl_homogeneouskernelmap_get_kernel_type(
    mut self_0: *const VlHomogeneousKernelMap,
) -> VlHomogeneousKernelType {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/homkermap.c\0" as *const u8 as *const libc::c_char,
            478 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 96],
                &[libc::c_char; 96],
            >(
                b"VlHomogeneousKernelType vl_homogeneouskernelmap_get_kernel_type(const VlHomogeneousKernelMap *)\0",
            ))
                .as_ptr(),
        );
    }
    return (*self_0).kernelType;
}
#[no_mangle]
pub unsafe extern "C" fn vl_homogeneouskernelmap_get_window_type(
    mut self_0: *const VlHomogeneousKernelMap,
) -> VlHomogeneousKernelMapWindowType {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/homkermap.c\0" as *const u8 as *const libc::c_char,
            490 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 105],
                &[libc::c_char; 105],
            >(
                b"VlHomogeneousKernelMapWindowType vl_homogeneouskernelmap_get_window_type(const VlHomogeneousKernelMap *)\0",
            ))
                .as_ptr(),
        );
    }
    return (*self_0).windowType;
}
#[no_mangle]
pub unsafe extern "C" fn vl_homogeneouskernelmap_evaluate_d(
    mut self_0: *const VlHomogeneousKernelMap,
    mut destination: *mut libc::c_double,
    mut stride: vl_size,
    mut x: libc::c_double,
) {
    let mut exponent: libc::c_int = 0;
    let mut j: libc::c_uint = 0;
    let mut mantissa: libc::c_double = frexp(x, &mut exponent);
    let mut sign: libc::c_double = if mantissa >= 0.0f64 { 1.0f64 } else { -1.0f64 };
    mantissa *= 2 as libc::c_int as libc::c_double * sign;
    exponent -= 1;
    if mantissa == 0 as libc::c_int as libc::c_double
        || exponent as libc::c_longlong <= (*self_0).minExponent
        || exponent as libc::c_longlong >= (*self_0).maxExponent
    {
        j = 0 as libc::c_int as libc::c_uint;
        while (j as libc::c_ulonglong)
            < (2 as libc::c_int as libc::c_ulonglong)
                .wrapping_mul((*self_0).order)
                .wrapping_add(1 as libc::c_int as libc::c_ulonglong)
        {
            *destination = 0.0f64;
            destination = destination.offset(stride as isize);
            j = j.wrapping_add(1);
        }
        return;
    }
    let mut featureDimension: vl_size = (2 as libc::c_int as libc::c_ulonglong)
        .wrapping_mul((*self_0).order)
        .wrapping_add(1 as libc::c_int as libc::c_ulonglong);
    let mut v1: *const libc::c_double = ((*self_0).table)
        .offset(
            ((exponent as libc::c_longlong - (*self_0).minExponent) as libc::c_ulonglong)
                .wrapping_mul((*self_0).numSubdivisions)
                .wrapping_mul(featureDimension) as isize,
        );
    let mut v2: *const libc::c_double = 0 as *const libc::c_double;
    let mut f1: libc::c_double = 0.;
    let mut f2: libc::c_double = 0.;
    mantissa -= 1.0f64;
    while mantissa >= (*self_0).subdivision {
        mantissa -= (*self_0).subdivision;
        v1 = v1.offset(featureDimension as isize);
    }
    v2 = v1.offset(featureDimension as isize);
    j = 0 as libc::c_int as libc::c_uint;
    while (j as libc::c_ulonglong) < featureDimension {
        let fresh3 = v1;
        v1 = v1.offset(1);
        f1 = *fresh3;
        let fresh4 = v2;
        v2 = v2.offset(1);
        f2 = *fresh4;
        *destination = sign
            * ((f2 - f1) * ((*self_0).numSubdivisions as libc::c_double * mantissa)
                + f1);
        destination = destination.offset(stride as isize);
        j = j.wrapping_add(1);
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_homogeneouskernelmap_evaluate_f(
    mut self_0: *const VlHomogeneousKernelMap,
    mut destination: *mut libc::c_float,
    mut stride: vl_size,
    mut x: libc::c_double,
) {
    let mut exponent: libc::c_int = 0;
    let mut j: libc::c_uint = 0;
    let mut mantissa: libc::c_double = frexp(x, &mut exponent);
    let mut sign: libc::c_double = if mantissa >= 0.0f64 { 1.0f64 } else { -1.0f64 };
    mantissa *= 2 as libc::c_int as libc::c_double * sign;
    exponent -= 1;
    if mantissa == 0 as libc::c_int as libc::c_double
        || exponent as libc::c_longlong <= (*self_0).minExponent
        || exponent as libc::c_longlong >= (*self_0).maxExponent
    {
        j = 0 as libc::c_int as libc::c_uint;
        while (j as libc::c_ulonglong)
            < (2 as libc::c_int as libc::c_ulonglong)
                .wrapping_mul((*self_0).order)
                .wrapping_add(1 as libc::c_int as libc::c_ulonglong)
        {
            *destination = 0.0f64 as libc::c_float;
            destination = destination.offset(stride as isize);
            j = j.wrapping_add(1);
        }
        return;
    }
    let mut featureDimension: vl_size = (2 as libc::c_int as libc::c_ulonglong)
        .wrapping_mul((*self_0).order)
        .wrapping_add(1 as libc::c_int as libc::c_ulonglong);
    let mut v1: *const libc::c_double = ((*self_0).table)
        .offset(
            ((exponent as libc::c_longlong - (*self_0).minExponent) as libc::c_ulonglong)
                .wrapping_mul((*self_0).numSubdivisions)
                .wrapping_mul(featureDimension) as isize,
        );
    let mut v2: *const libc::c_double = 0 as *const libc::c_double;
    let mut f1: libc::c_double = 0.;
    let mut f2: libc::c_double = 0.;
    mantissa -= 1.0f64;
    while mantissa >= (*self_0).subdivision {
        mantissa -= (*self_0).subdivision;
        v1 = v1.offset(featureDimension as isize);
    }
    v2 = v1.offset(featureDimension as isize);
    j = 0 as libc::c_int as libc::c_uint;
    while (j as libc::c_ulonglong) < featureDimension {
        let fresh5 = v1;
        v1 = v1.offset(1);
        f1 = *fresh5;
        let fresh6 = v2;
        v2 = v2.offset(1);
        f2 = *fresh6;
        *destination = (sign as libc::c_float as libc::c_double
            * ((f2 - f1) * ((*self_0).numSubdivisions as libc::c_double * mantissa)
                + f1)) as libc::c_float;
        destination = destination.offset(stride as isize);
        j = j.wrapping_add(1);
    }
}
