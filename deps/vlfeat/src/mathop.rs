use ::libc;
extern "C" {
    fn abort() -> !;
    fn __assert_fail(
        __assertion: *const libc::c_char,
        __file: *const libc::c_char,
        __line: libc::c_uint,
        __function: *const libc::c_char,
    ) -> !;
    fn vl_get_simd_enabled() -> vl_bool;
    fn vl_cpu_has_avx() -> vl_bool;
    fn vl_cpu_has_sse2() -> vl_bool;
    fn sqrt(_: libc::c_double) -> libc::c_double;
    fn fabs(_: libc::c_double) -> libc::c_double;
    fn sqrtf(_: libc::c_float) -> libc::c_float;
    fn _vl_distance_l2_sse2_f(
        dimension: vl_size,
        X: *const libc::c_float,
        Y: *const libc::c_float,
    ) -> libc::c_float;
    fn _vl_distance_l2_sse2_d(
        dimension: vl_size,
        X: *const libc::c_double,
        Y: *const libc::c_double,
    ) -> libc::c_double;
    fn _vl_distance_l1_sse2_f(
        dimension: vl_size,
        X: *const libc::c_float,
        Y: *const libc::c_float,
    ) -> libc::c_float;
    fn _vl_distance_l1_sse2_d(
        dimension: vl_size,
        X: *const libc::c_double,
        Y: *const libc::c_double,
    ) -> libc::c_double;
    fn _vl_distance_chi2_sse2_d(
        dimension: vl_size,
        X: *const libc::c_double,
        Y: *const libc::c_double,
    ) -> libc::c_double;
    fn _vl_distance_chi2_sse2_f(
        dimension: vl_size,
        X: *const libc::c_float,
        Y: *const libc::c_float,
    ) -> libc::c_float;
    fn _vl_kernel_l2_sse2_d(
        dimension: vl_size,
        X: *const libc::c_double,
        Y: *const libc::c_double,
    ) -> libc::c_double;
    fn _vl_kernel_l2_sse2_f(
        dimension: vl_size,
        X: *const libc::c_float,
        Y: *const libc::c_float,
    ) -> libc::c_float;
    fn _vl_kernel_l1_sse2_d(
        dimension: vl_size,
        X: *const libc::c_double,
        Y: *const libc::c_double,
    ) -> libc::c_double;
    fn _vl_kernel_l1_sse2_f(
        dimension: vl_size,
        X: *const libc::c_float,
        Y: *const libc::c_float,
    ) -> libc::c_float;
    fn _vl_kernel_chi2_sse2_f(
        dimension: vl_size,
        X: *const libc::c_float,
        Y: *const libc::c_float,
    ) -> libc::c_float;
    fn _vl_kernel_chi2_sse2_d(
        dimension: vl_size,
        X: *const libc::c_double,
        Y: *const libc::c_double,
    ) -> libc::c_double;
    fn _vl_distance_mahalanobis_sq_sse2_f(
        dimension: vl_size,
        X: *const libc::c_float,
        MU: *const libc::c_float,
        S: *const libc::c_float,
    ) -> libc::c_float;
    fn _vl_distance_mahalanobis_sq_sse2_d(
        dimension: vl_size,
        X: *const libc::c_double,
        MU: *const libc::c_double,
        S: *const libc::c_double,
    ) -> libc::c_double;
    fn _vl_distance_mahalanobis_sq_avx_d(
        dimension: vl_size,
        X: *const libc::c_double,
        MU: *const libc::c_double,
        S: *const libc::c_double,
    ) -> libc::c_double;
    fn _vl_distance_mahalanobis_sq_avx_f(
        dimension: vl_size,
        X: *const libc::c_float,
        MU: *const libc::c_float,
        S: *const libc::c_float,
    ) -> libc::c_float;
    fn _vl_distance_l2_avx_d(
        dimension: vl_size,
        X: *const libc::c_double,
        Y: *const libc::c_double,
    ) -> libc::c_double;
    fn _vl_distance_l2_avx_f(
        dimension: vl_size,
        X: *const libc::c_float,
        Y: *const libc::c_float,
    ) -> libc::c_float;
}
pub type vl_int64 = libc::c_longlong;
pub type vl_uint64 = libc::c_ulonglong;
pub type vl_bool = libc::c_int;
pub type vl_size = vl_uint64;
pub type vl_index = vl_int64;
pub type vl_uindex = vl_uint64;
pub type VlFloatVectorComparisonFunction = Option::<
    unsafe extern "C" fn(
        vl_size,
        *const libc::c_float,
        *const libc::c_float,
    ) -> libc::c_float,
>;
pub type VlDoubleVectorComparisonFunction = Option::<
    unsafe extern "C" fn(
        vl_size,
        *const libc::c_double,
        *const libc::c_double,
    ) -> libc::c_double,
>;
pub type VlFloatVector3ComparisonFunction = Option::<
    unsafe extern "C" fn(
        vl_size,
        *const libc::c_float,
        *const libc::c_float,
        *const libc::c_float,
    ) -> libc::c_float,
>;
pub type VlDoubleVector3ComparisonFunction = Option::<
    unsafe extern "C" fn(
        vl_size,
        *const libc::c_double,
        *const libc::c_double,
        *const libc::c_double,
    ) -> libc::c_double,
>;
pub type _VlVectorComparisonType = libc::c_uint;
pub const VlKernelJS: _VlVectorComparisonType = 10;
pub const VlKernelHellinger: _VlVectorComparisonType = 9;
pub const VlKernelChi2: _VlVectorComparisonType = 8;
pub const VlKernelL2: _VlVectorComparisonType = 7;
pub const VlKernelL1: _VlVectorComparisonType = 6;
pub const VlDistanceMahalanobis: _VlVectorComparisonType = 5;
pub const VlDistanceJS: _VlVectorComparisonType = 4;
pub const VlDistanceHellinger: _VlVectorComparisonType = 3;
pub const VlDistanceChi2: _VlVectorComparisonType = 2;
pub const VlDistanceL2: _VlVectorComparisonType = 1;
pub const VlDistanceL1: _VlVectorComparisonType = 0;
pub type VlVectorComparisonType = _VlVectorComparisonType;
#[inline]
unsafe extern "C" fn vl_abs_f(mut x: libc::c_float) -> libc::c_float {
    return x.abs();
}
#[inline]
unsafe extern "C" fn vl_abs_d(mut x: libc::c_double) -> libc::c_double {
    return x.abs();
}
#[no_mangle]
pub unsafe extern "C" fn _vl_distance_l2_d(
    mut dimension: vl_size,
    mut X: *const libc::c_double,
    mut Y: *const libc::c_double,
) -> libc::c_double {
    let mut X_end: *const libc::c_double = X.offset(dimension as isize);
    let mut acc: libc::c_double = 0.0f64;
    while X < X_end {
        let fresh0 = X;
        X = X.offset(1);
        let fresh1 = Y;
        Y = Y.offset(1);
        let mut d: libc::c_double = *fresh0 - *fresh1;
        acc += d * d;
    }
    return acc;
}
#[no_mangle]
pub unsafe extern "C" fn _vl_distance_l2_f(
    mut dimension: vl_size,
    mut X: *const libc::c_float,
    mut Y: *const libc::c_float,
) -> libc::c_float {
    let mut X_end: *const libc::c_float = X.offset(dimension as isize);
    let mut acc: libc::c_float = 0.0f64 as libc::c_float;
    while X < X_end {
        let fresh2 = X;
        X = X.offset(1);
        let fresh3 = Y;
        Y = Y.offset(1);
        let mut d: libc::c_float = *fresh2 - *fresh3;
        acc += d * d;
    }
    return acc;
}
#[no_mangle]
pub unsafe extern "C" fn _vl_distance_l1_f(
    mut dimension: vl_size,
    mut X: *const libc::c_float,
    mut Y: *const libc::c_float,
) -> libc::c_float {
    let mut X_end: *const libc::c_float = X.offset(dimension as isize);
    let mut acc: libc::c_float = 0.0f64 as libc::c_float;
    while X < X_end {
        let fresh4 = X;
        X = X.offset(1);
        let fresh5 = Y;
        Y = Y.offset(1);
        let mut d: libc::c_float = *fresh4 - *fresh5;
        acc += if d > -d { d } else { -d };
    }
    return acc;
}
#[no_mangle]
pub unsafe extern "C" fn _vl_distance_l1_d(
    mut dimension: vl_size,
    mut X: *const libc::c_double,
    mut Y: *const libc::c_double,
) -> libc::c_double {
    let mut X_end: *const libc::c_double = X.offset(dimension as isize);
    let mut acc: libc::c_double = 0.0f64;
    while X < X_end {
        let fresh6 = X;
        X = X.offset(1);
        let fresh7 = Y;
        Y = Y.offset(1);
        let mut d: libc::c_double = *fresh6 - *fresh7;
        acc += if d > -d { d } else { -d };
    }
    return acc;
}
#[no_mangle]
pub unsafe extern "C" fn _vl_distance_chi2_f(
    mut dimension: vl_size,
    mut X: *const libc::c_float,
    mut Y: *const libc::c_float,
) -> libc::c_float {
    let mut X_end: *const libc::c_float = X.offset(dimension as isize);
    let mut acc: libc::c_float = 0.0f64 as libc::c_float;
    while X < X_end {
        let fresh8 = X;
        X = X.offset(1);
        let mut a: libc::c_float = *fresh8;
        let fresh9 = Y;
        Y = Y.offset(1);
        let mut b: libc::c_float = *fresh9;
        let mut delta: libc::c_float = a - b;
        let mut denom: libc::c_float = a + b;
        let mut numer: libc::c_float = delta * delta;
        if denom != 0. {
            let mut ratio: libc::c_float = numer / denom;
            acc += ratio;
        }
    }
    return acc;
}
#[no_mangle]
pub unsafe extern "C" fn _vl_distance_chi2_d(
    mut dimension: vl_size,
    mut X: *const libc::c_double,
    mut Y: *const libc::c_double,
) -> libc::c_double {
    let mut X_end: *const libc::c_double = X.offset(dimension as isize);
    let mut acc: libc::c_double = 0.0f64;
    while X < X_end {
        let fresh10 = X;
        X = X.offset(1);
        let mut a: libc::c_double = *fresh10;
        let fresh11 = Y;
        Y = Y.offset(1);
        let mut b: libc::c_double = *fresh11;
        let mut delta: libc::c_double = a - b;
        let mut denom: libc::c_double = a + b;
        let mut numer: libc::c_double = delta * delta;
        if denom != 0. {
            let mut ratio: libc::c_double = numer / denom;
            acc += ratio;
        }
    }
    return acc;
}
#[no_mangle]
pub unsafe extern "C" fn _vl_distance_hellinger_f(
    mut dimension: vl_size,
    mut X: *const libc::c_float,
    mut Y: *const libc::c_float,
) -> libc::c_float {
    let mut X_end: *const libc::c_float = X.offset(dimension as isize);
    let mut acc: libc::c_float = 0.0f64 as libc::c_float;
    while X < X_end {
        let fresh12 = X;
        X = X.offset(1);
        let mut a: libc::c_float = *fresh12;
        let fresh13 = Y;
        Y = Y.offset(1);
        let mut b: libc::c_float = *fresh13;
        acc = (acc as libc::c_double
            + ((a + b) as libc::c_double - 2.0f64 * sqrtf(a * b) as libc::c_double))
            as libc::c_float;
    }
    return acc;
}
#[no_mangle]
pub unsafe extern "C" fn _vl_distance_hellinger_d(
    mut dimension: vl_size,
    mut X: *const libc::c_double,
    mut Y: *const libc::c_double,
) -> libc::c_double {
    let mut X_end: *const libc::c_double = X.offset(dimension as isize);
    let mut acc: libc::c_double = 0.0f64;
    while X < X_end {
        let fresh14 = X;
        X = X.offset(1);
        let mut a: libc::c_double = *fresh14;
        let fresh15 = Y;
        Y = Y.offset(1);
        let mut b: libc::c_double = *fresh15;
        acc += a + b - 2.0f64 * sqrt(a * b);
    }
    return acc;
}
#[no_mangle]
pub unsafe extern "C" fn _vl_distance_js_f(
    mut dimension: vl_size,
    mut X: *const libc::c_float,
    mut Y: *const libc::c_float,
) -> libc::c_float {
    let mut X_end: *const libc::c_float = X.offset(dimension as isize);
    let mut acc: libc::c_float = 0.0f64 as libc::c_float;
    while X < X_end {
        let fresh16 = X;
        X = X.offset(1);
        let mut x: libc::c_float = *fresh16;
        let fresh17 = Y;
        Y = Y.offset(1);
        let mut y: libc::c_float = *fresh17;
        if x != 0. {
            acc += x - x * f32::log2(1 as libc::c_int as libc::c_float + y / x);
        }
        if y != 0. {
            acc += y - y * f32::log2(1 as libc::c_int as libc::c_float + x / y);
        }
    }
    return acc;
}
#[no_mangle]
pub unsafe extern "C" fn _vl_distance_js_d(
    mut dimension: vl_size,
    mut X: *const libc::c_double,
    mut Y: *const libc::c_double,
) -> libc::c_double {
    let mut X_end: *const libc::c_double = X.offset(dimension as isize);
    let mut acc: libc::c_double = 0.0f64;
    while X < X_end {
        let fresh18 = X;
        X = X.offset(1);
        let mut x: libc::c_double = *fresh18;
        let fresh19 = Y;
        Y = Y.offset(1);
        let mut y: libc::c_double = *fresh19;
        if x != 0. {
            acc += x - x * f64::log2(1 as libc::c_int as libc::c_double + y / x);
        }
        if y != 0. {
            acc += y - y * f64::log2(1 as libc::c_int as libc::c_double + x / y);
        }
    }
    return acc;
}
#[no_mangle]
pub unsafe extern "C" fn _vl_kernel_l2_d(
    mut dimension: vl_size,
    mut X: *const libc::c_double,
    mut Y: *const libc::c_double,
) -> libc::c_double {
    let mut X_end: *const libc::c_double = X.offset(dimension as isize);
    let mut acc: libc::c_double = 0.0f64;
    while X < X_end {
        let fresh20 = X;
        X = X.offset(1);
        let mut a: libc::c_double = *fresh20;
        let fresh21 = Y;
        Y = Y.offset(1);
        let mut b: libc::c_double = *fresh21;
        acc += a * b;
    }
    return acc;
}
#[no_mangle]
pub unsafe extern "C" fn _vl_kernel_l2_f(
    mut dimension: vl_size,
    mut X: *const libc::c_float,
    mut Y: *const libc::c_float,
) -> libc::c_float {
    let mut X_end: *const libc::c_float = X.offset(dimension as isize);
    let mut acc: libc::c_float = 0.0f64 as libc::c_float;
    while X < X_end {
        let fresh22 = X;
        X = X.offset(1);
        let mut a: libc::c_float = *fresh22;
        let fresh23 = Y;
        Y = Y.offset(1);
        let mut b: libc::c_float = *fresh23;
        acc += a * b;
    }
    return acc;
}
#[no_mangle]
pub unsafe extern "C" fn _vl_kernel_l1_f(
    mut dimension: vl_size,
    mut X: *const libc::c_float,
    mut Y: *const libc::c_float,
) -> libc::c_float {
    let mut X_end: *const libc::c_float = X.offset(dimension as isize);
    let mut acc: libc::c_float = 0.0f64 as libc::c_float;
    while X < X_end {
        let fresh24 = X;
        X = X.offset(1);
        let mut a: libc::c_float = *fresh24;
        let fresh25 = Y;
        Y = Y.offset(1);
        let mut b: libc::c_float = *fresh25;
        let mut a_: libc::c_float = vl_abs_f(a);
        let mut b_: libc::c_float = vl_abs_f(b);
        acc += a_ + b_ - vl_abs_f(a - b);
    }
    return acc / 2 as libc::c_int as libc::c_float;
}
#[no_mangle]
pub unsafe extern "C" fn _vl_kernel_l1_d(
    mut dimension: vl_size,
    mut X: *const libc::c_double,
    mut Y: *const libc::c_double,
) -> libc::c_double {
    let mut X_end: *const libc::c_double = X.offset(dimension as isize);
    let mut acc: libc::c_double = 0.0f64;
    while X < X_end {
        let fresh26 = X;
        X = X.offset(1);
        let mut a: libc::c_double = *fresh26;
        let fresh27 = Y;
        Y = Y.offset(1);
        let mut b: libc::c_double = *fresh27;
        let mut a_: libc::c_double = vl_abs_d(a);
        let mut b_: libc::c_double = vl_abs_d(b);
        acc += a_ + b_ - vl_abs_d(a - b);
    }
    return acc / 2 as libc::c_int as libc::c_double;
}
#[no_mangle]
pub unsafe extern "C" fn _vl_kernel_chi2_d(
    mut dimension: vl_size,
    mut X: *const libc::c_double,
    mut Y: *const libc::c_double,
) -> libc::c_double {
    let mut X_end: *const libc::c_double = X.offset(dimension as isize);
    let mut acc: libc::c_double = 0.0f64;
    while X < X_end {
        let fresh28 = X;
        X = X.offset(1);
        let mut a: libc::c_double = *fresh28;
        let fresh29 = Y;
        Y = Y.offset(1);
        let mut b: libc::c_double = *fresh29;
        let mut denom: libc::c_double = a + b;
        if denom != 0. {
            let mut numer: libc::c_double = 2 as libc::c_int as libc::c_double * a * b;
            let mut ratio: libc::c_double = numer / denom;
            acc += ratio;
        }
    }
    return acc;
}
#[no_mangle]
pub unsafe extern "C" fn _vl_kernel_chi2_f(
    mut dimension: vl_size,
    mut X: *const libc::c_float,
    mut Y: *const libc::c_float,
) -> libc::c_float {
    let mut X_end: *const libc::c_float = X.offset(dimension as isize);
    let mut acc: libc::c_float = 0.0f64 as libc::c_float;
    while X < X_end {
        let fresh30 = X;
        X = X.offset(1);
        let mut a: libc::c_float = *fresh30;
        let fresh31 = Y;
        Y = Y.offset(1);
        let mut b: libc::c_float = *fresh31;
        let mut denom: libc::c_float = a + b;
        if denom != 0. {
            let mut numer: libc::c_float = 2 as libc::c_int as libc::c_float * a * b;
            let mut ratio: libc::c_float = numer / denom;
            acc += ratio;
        }
    }
    return acc;
}
#[no_mangle]
pub unsafe extern "C" fn _vl_kernel_hellinger_d(
    mut dimension: vl_size,
    mut X: *const libc::c_double,
    mut Y: *const libc::c_double,
) -> libc::c_double {
    let mut X_end: *const libc::c_double = X.offset(dimension as isize);
    let mut acc: libc::c_double = 0.0f64;
    while X < X_end {
        let fresh32 = X;
        X = X.offset(1);
        let mut a: libc::c_double = *fresh32;
        let fresh33 = Y;
        Y = Y.offset(1);
        let mut b: libc::c_double = *fresh33;
        acc += sqrt(a * b);
    }
    return acc;
}
#[no_mangle]
pub unsafe extern "C" fn _vl_kernel_hellinger_f(
    mut dimension: vl_size,
    mut X: *const libc::c_float,
    mut Y: *const libc::c_float,
) -> libc::c_float {
    let mut X_end: *const libc::c_float = X.offset(dimension as isize);
    let mut acc: libc::c_float = 0.0f64 as libc::c_float;
    while X < X_end {
        let fresh34 = X;
        X = X.offset(1);
        let mut a: libc::c_float = *fresh34;
        let fresh35 = Y;
        Y = Y.offset(1);
        let mut b: libc::c_float = *fresh35;
        acc += sqrtf(a * b);
    }
    return acc;
}
#[no_mangle]
pub unsafe extern "C" fn _vl_kernel_js_d(
    mut dimension: vl_size,
    mut X: *const libc::c_double,
    mut Y: *const libc::c_double,
) -> libc::c_double {
    let mut X_end: *const libc::c_double = X.offset(dimension as isize);
    let mut acc: libc::c_double = 0.0f64;
    while X < X_end {
        let fresh36 = X;
        X = X.offset(1);
        let mut x: libc::c_double = *fresh36;
        let fresh37 = Y;
        Y = Y.offset(1);
        let mut y: libc::c_double = *fresh37;
        if x != 0. {
            acc += x * f64::log2(1 as libc::c_int as libc::c_double + y / x);
        }
        if y != 0. {
            acc += y * f64::log2(1 as libc::c_int as libc::c_double + x / y);
        }
    }
    return 0.5f64 * acc;
}
#[no_mangle]
pub unsafe extern "C" fn _vl_kernel_js_f(
    mut dimension: vl_size,
    mut X: *const libc::c_float,
    mut Y: *const libc::c_float,
) -> libc::c_float {
    let mut X_end: *const libc::c_float = X.offset(dimension as isize);
    let mut acc: libc::c_float = 0.0f64 as libc::c_float;
    while X < X_end {
        let fresh38 = X;
        X = X.offset(1);
        let mut x: libc::c_float = *fresh38;
        let fresh39 = Y;
        Y = Y.offset(1);
        let mut y: libc::c_float = *fresh39;
        if x != 0. {
            acc += x * f32::log2(1 as libc::c_int as libc::c_float + y / x);
        }
        if y != 0. {
            acc += y * f32::log2(1 as libc::c_int as libc::c_float + x / y);
        }
    }
    return 0.5f64 as libc::c_float * acc;
}
#[no_mangle]
pub unsafe extern "C" fn _vl_distance_mahalanobis_sq_f(
    mut dimension: vl_size,
    mut X: *const libc::c_float,
    mut MU: *const libc::c_float,
    mut S: *const libc::c_float,
) -> libc::c_float {
    let mut X_end: *const libc::c_float = X.offset(dimension as isize);
    let mut acc: libc::c_float = 0.0f64 as libc::c_float;
    while X < X_end {
        let fresh40 = X;
        X = X.offset(1);
        let fresh41 = MU;
        MU = MU.offset(1);
        let mut d: libc::c_float = *fresh40 - *fresh41;
        let fresh42 = S;
        S = S.offset(1);
        acc += d * d * *fresh42;
    }
    return acc;
}
#[no_mangle]
pub unsafe extern "C" fn _vl_distance_mahalanobis_sq_d(
    mut dimension: vl_size,
    mut X: *const libc::c_double,
    mut MU: *const libc::c_double,
    mut S: *const libc::c_double,
) -> libc::c_double {
    let mut X_end: *const libc::c_double = X.offset(dimension as isize);
    let mut acc: libc::c_double = 0.0f64;
    while X < X_end {
        let fresh43 = X;
        X = X.offset(1);
        let fresh44 = MU;
        MU = MU.offset(1);
        let mut d: libc::c_double = *fresh43 - *fresh44;
        let fresh45 = S;
        S = S.offset(1);
        acc += d * d * *fresh45;
    }
    return acc;
}
#[no_mangle]
pub unsafe extern "C" fn vl_get_vector_comparison_function_d(
    mut type_0: VlVectorComparisonType,
) -> VlDoubleVectorComparisonFunction {
    let mut function: VlDoubleVectorComparisonFunction = None;
    match type_0 as libc::c_uint {
        1 => {
            function = Some(
                _vl_distance_l2_d
                    as unsafe extern "C" fn(
                        vl_size,
                        *const libc::c_double,
                        *const libc::c_double,
                    ) -> libc::c_double,
            );
        }
        0 => {
            function = Some(
                _vl_distance_l1_d
                    as unsafe extern "C" fn(
                        vl_size,
                        *const libc::c_double,
                        *const libc::c_double,
                    ) -> libc::c_double,
            );
        }
        2 => {
            function = Some(
                _vl_distance_chi2_d
                    as unsafe extern "C" fn(
                        vl_size,
                        *const libc::c_double,
                        *const libc::c_double,
                    ) -> libc::c_double,
            );
        }
        3 => {
            function = Some(
                _vl_distance_hellinger_d
                    as unsafe extern "C" fn(
                        vl_size,
                        *const libc::c_double,
                        *const libc::c_double,
                    ) -> libc::c_double,
            );
        }
        4 => {
            function = Some(
                _vl_distance_js_d
                    as unsafe extern "C" fn(
                        vl_size,
                        *const libc::c_double,
                        *const libc::c_double,
                    ) -> libc::c_double,
            );
        }
        7 => {
            function = Some(
                _vl_kernel_l2_d
                    as unsafe extern "C" fn(
                        vl_size,
                        *const libc::c_double,
                        *const libc::c_double,
                    ) -> libc::c_double,
            );
        }
        6 => {
            function = Some(
                _vl_kernel_l1_d
                    as unsafe extern "C" fn(
                        vl_size,
                        *const libc::c_double,
                        *const libc::c_double,
                    ) -> libc::c_double,
            );
        }
        8 => {
            function = Some(
                _vl_kernel_chi2_d
                    as unsafe extern "C" fn(
                        vl_size,
                        *const libc::c_double,
                        *const libc::c_double,
                    ) -> libc::c_double,
            );
        }
        9 => {
            function = Some(
                _vl_kernel_hellinger_d
                    as unsafe extern "C" fn(
                        vl_size,
                        *const libc::c_double,
                        *const libc::c_double,
                    ) -> libc::c_double,
            );
        }
        10 => {
            function = Some(
                _vl_kernel_js_d
                    as unsafe extern "C" fn(
                        vl_size,
                        *const libc::c_double,
                        *const libc::c_double,
                    ) -> libc::c_double,
            );
        }
        _ => {
            abort();
        }
    }
    if vl_cpu_has_sse2() != 0 && vl_get_simd_enabled() != 0 {
        match type_0 as libc::c_uint {
            1 => {
                function = Some(
                    _vl_distance_l2_sse2_d
                        as unsafe extern "C" fn(
                            vl_size,
                            *const libc::c_double,
                            *const libc::c_double,
                        ) -> libc::c_double,
                );
            }
            0 => {
                function = Some(
                    _vl_distance_l1_sse2_d
                        as unsafe extern "C" fn(
                            vl_size,
                            *const libc::c_double,
                            *const libc::c_double,
                        ) -> libc::c_double,
                );
            }
            2 => {
                function = Some(
                    _vl_distance_chi2_sse2_d
                        as unsafe extern "C" fn(
                            vl_size,
                            *const libc::c_double,
                            *const libc::c_double,
                        ) -> libc::c_double,
                );
            }
            7 => {
                function = Some(
                    _vl_kernel_l2_sse2_d
                        as unsafe extern "C" fn(
                            vl_size,
                            *const libc::c_double,
                            *const libc::c_double,
                        ) -> libc::c_double,
                );
            }
            6 => {
                function = Some(
                    _vl_kernel_l1_sse2_d
                        as unsafe extern "C" fn(
                            vl_size,
                            *const libc::c_double,
                            *const libc::c_double,
                        ) -> libc::c_double,
                );
            }
            8 => {
                function = Some(
                    _vl_kernel_chi2_sse2_d
                        as unsafe extern "C" fn(
                            vl_size,
                            *const libc::c_double,
                            *const libc::c_double,
                        ) -> libc::c_double,
                );
            }
            _ => {}
        }
    }
    if vl_cpu_has_avx() != 0 && vl_get_simd_enabled() != 0 {
        match type_0 as libc::c_uint {
            1 => {
                function = Some(
                    _vl_distance_l2_avx_d
                        as unsafe extern "C" fn(
                            vl_size,
                            *const libc::c_double,
                            *const libc::c_double,
                        ) -> libc::c_double,
                );
            }
            _ => {}
        }
    }
    return function;
}
#[no_mangle]
pub unsafe extern "C" fn vl_get_vector_comparison_function_f(
    mut type_0: VlVectorComparisonType,
) -> VlFloatVectorComparisonFunction {
    let mut function: VlFloatVectorComparisonFunction = None;
    match type_0 as libc::c_uint {
        1 => {
            function = Some(
                _vl_distance_l2_f
                    as unsafe extern "C" fn(
                        vl_size,
                        *const libc::c_float,
                        *const libc::c_float,
                    ) -> libc::c_float,
            );
        }
        0 => {
            function = Some(
                _vl_distance_l1_f
                    as unsafe extern "C" fn(
                        vl_size,
                        *const libc::c_float,
                        *const libc::c_float,
                    ) -> libc::c_float,
            );
        }
        2 => {
            function = Some(
                _vl_distance_chi2_f
                    as unsafe extern "C" fn(
                        vl_size,
                        *const libc::c_float,
                        *const libc::c_float,
                    ) -> libc::c_float,
            );
        }
        3 => {
            function = Some(
                _vl_distance_hellinger_f
                    as unsafe extern "C" fn(
                        vl_size,
                        *const libc::c_float,
                        *const libc::c_float,
                    ) -> libc::c_float,
            );
        }
        4 => {
            function = Some(
                _vl_distance_js_f
                    as unsafe extern "C" fn(
                        vl_size,
                        *const libc::c_float,
                        *const libc::c_float,
                    ) -> libc::c_float,
            );
        }
        7 => {
            function = Some(
                _vl_kernel_l2_f
                    as unsafe extern "C" fn(
                        vl_size,
                        *const libc::c_float,
                        *const libc::c_float,
                    ) -> libc::c_float,
            );
        }
        6 => {
            function = Some(
                _vl_kernel_l1_f
                    as unsafe extern "C" fn(
                        vl_size,
                        *const libc::c_float,
                        *const libc::c_float,
                    ) -> libc::c_float,
            );
        }
        8 => {
            function = Some(
                _vl_kernel_chi2_f
                    as unsafe extern "C" fn(
                        vl_size,
                        *const libc::c_float,
                        *const libc::c_float,
                    ) -> libc::c_float,
            );
        }
        9 => {
            function = Some(
                _vl_kernel_hellinger_f
                    as unsafe extern "C" fn(
                        vl_size,
                        *const libc::c_float,
                        *const libc::c_float,
                    ) -> libc::c_float,
            );
        }
        10 => {
            function = Some(
                _vl_kernel_js_f
                    as unsafe extern "C" fn(
                        vl_size,
                        *const libc::c_float,
                        *const libc::c_float,
                    ) -> libc::c_float,
            );
        }
        _ => {
            abort();
        }
    }
    if vl_cpu_has_sse2() != 0 && vl_get_simd_enabled() != 0 {
        match type_0 as libc::c_uint {
            1 => {
                function = Some(
                    _vl_distance_l2_sse2_f
                        as unsafe extern "C" fn(
                            vl_size,
                            *const libc::c_float,
                            *const libc::c_float,
                        ) -> libc::c_float,
                );
            }
            0 => {
                function = Some(
                    _vl_distance_l1_sse2_f
                        as unsafe extern "C" fn(
                            vl_size,
                            *const libc::c_float,
                            *const libc::c_float,
                        ) -> libc::c_float,
                );
            }
            2 => {
                function = Some(
                    _vl_distance_chi2_sse2_f
                        as unsafe extern "C" fn(
                            vl_size,
                            *const libc::c_float,
                            *const libc::c_float,
                        ) -> libc::c_float,
                );
            }
            7 => {
                function = Some(
                    _vl_kernel_l2_sse2_f
                        as unsafe extern "C" fn(
                            vl_size,
                            *const libc::c_float,
                            *const libc::c_float,
                        ) -> libc::c_float,
                );
            }
            6 => {
                function = Some(
                    _vl_kernel_l1_sse2_f
                        as unsafe extern "C" fn(
                            vl_size,
                            *const libc::c_float,
                            *const libc::c_float,
                        ) -> libc::c_float,
                );
            }
            8 => {
                function = Some(
                    _vl_kernel_chi2_sse2_f
                        as unsafe extern "C" fn(
                            vl_size,
                            *const libc::c_float,
                            *const libc::c_float,
                        ) -> libc::c_float,
                );
            }
            _ => {}
        }
    }
    if vl_cpu_has_avx() != 0 && vl_get_simd_enabled() != 0 {
        match type_0 as libc::c_uint {
            1 => {
                function = Some(
                    _vl_distance_l2_avx_f
                        as unsafe extern "C" fn(
                            vl_size,
                            *const libc::c_float,
                            *const libc::c_float,
                        ) -> libc::c_float,
                );
            }
            _ => {}
        }
    }
    return function;
}
#[no_mangle]
pub unsafe extern "C" fn vl_get_vector_3_comparison_function_f(
    mut type_0: VlVectorComparisonType,
) -> VlFloatVector3ComparisonFunction {
    let mut function: VlFloatVector3ComparisonFunction = None;
    match type_0 as libc::c_uint {
        5 => {
            function = Some(
                _vl_distance_mahalanobis_sq_f
                    as unsafe extern "C" fn(
                        vl_size,
                        *const libc::c_float,
                        *const libc::c_float,
                        *const libc::c_float,
                    ) -> libc::c_float,
            );
        }
        _ => {
            abort();
        }
    }
    if vl_cpu_has_sse2() != 0 && vl_get_simd_enabled() != 0 {
        match type_0 as libc::c_uint {
            5 => {
                function = Some(
                    _vl_distance_mahalanobis_sq_sse2_f
                        as unsafe extern "C" fn(
                            vl_size,
                            *const libc::c_float,
                            *const libc::c_float,
                            *const libc::c_float,
                        ) -> libc::c_float,
                );
            }
            _ => {}
        }
    }
    if vl_cpu_has_avx() != 0 && vl_get_simd_enabled() != 0 {
        match type_0 as libc::c_uint {
            5 => {
                function = Some(
                    _vl_distance_mahalanobis_sq_avx_f
                        as unsafe extern "C" fn(
                            vl_size,
                            *const libc::c_float,
                            *const libc::c_float,
                            *const libc::c_float,
                        ) -> libc::c_float,
                );
            }
            _ => {}
        }
    }
    return function;
}
#[no_mangle]
pub unsafe extern "C" fn vl_get_vector_3_comparison_function_d(
    mut type_0: VlVectorComparisonType,
) -> VlDoubleVector3ComparisonFunction {
    let mut function: VlDoubleVector3ComparisonFunction = None;
    match type_0 as libc::c_uint {
        5 => {
            function = Some(
                _vl_distance_mahalanobis_sq_d
                    as unsafe extern "C" fn(
                        vl_size,
                        *const libc::c_double,
                        *const libc::c_double,
                        *const libc::c_double,
                    ) -> libc::c_double,
            );
        }
        _ => {
            abort();
        }
    }
    if vl_cpu_has_sse2() != 0 && vl_get_simd_enabled() != 0 {
        match type_0 as libc::c_uint {
            5 => {
                function = Some(
                    _vl_distance_mahalanobis_sq_sse2_d
                        as unsafe extern "C" fn(
                            vl_size,
                            *const libc::c_double,
                            *const libc::c_double,
                            *const libc::c_double,
                        ) -> libc::c_double,
                );
            }
            _ => {}
        }
    }
    if vl_cpu_has_avx() != 0 && vl_get_simd_enabled() != 0 {
        match type_0 as libc::c_uint {
            5 => {
                function = Some(
                    _vl_distance_mahalanobis_sq_avx_d
                        as unsafe extern "C" fn(
                            vl_size,
                            *const libc::c_double,
                            *const libc::c_double,
                            *const libc::c_double,
                        ) -> libc::c_double,
                );
            }
            _ => {}
        }
    }
    return function;
}
#[no_mangle]
pub unsafe extern "C" fn vl_eval_vector_comparison_on_all_pairs_d(
    mut result: *mut libc::c_double,
    mut dimension: vl_size,
    mut X: *const libc::c_double,
    mut numDataX: vl_size,
    mut Y: *const libc::c_double,
    mut numDataY: vl_size,
    mut function: VlDoubleVectorComparisonFunction,
) {
    let mut xi: vl_uindex = 0;
    let mut yi: vl_uindex = 0;
    if dimension == 0 as libc::c_int as libc::c_ulonglong {
        return;
    }
    if numDataX == 0 as libc::c_int as libc::c_ulonglong {
        return;
    }
    if !X.is_null() {} else {
        __assert_fail(
            b"X\0" as *const u8 as *const libc::c_char,
            b"vl/mathop.c\0" as *const u8 as *const libc::c_char,
            566 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 149],
                &[libc::c_char; 149],
            >(
                b"void vl_eval_vector_comparison_on_all_pairs_d(double *, vl_size, const double *, vl_size, const double *, vl_size, VlDoubleVectorComparisonFunction)\0",
            ))
                .as_ptr(),
        );
    }
    if !Y.is_null() {
        if numDataY == 0 as libc::c_int as libc::c_ulonglong {
            return;
        }
        yi = 0 as libc::c_int as vl_uindex;
        while yi < numDataY {
            xi = 0 as libc::c_int as vl_uindex;
            while xi < numDataX {
                let fresh46 = result;
                result = result.offset(1);
                *fresh46 = (Some(function.expect("non-null function pointer")))
                    .expect("non-null function pointer")(dimension, X, Y);
                X = X.offset(dimension as isize);
                xi = xi.wrapping_add(1);
            }
            X = X.offset(-(dimension.wrapping_mul(numDataX) as isize));
            Y = Y.offset(dimension as isize);
            yi = yi.wrapping_add(1);
        }
    } else {
        let mut resultTransp: *mut libc::c_double = result;
        Y = X;
        yi = 0 as libc::c_int as vl_uindex;
        while yi < numDataX {
            xi = 0 as libc::c_int as vl_uindex;
            while xi <= yi {
                let mut z: libc::c_double = (Some(
                    function.expect("non-null function pointer"),
                ))
                    .expect("non-null function pointer")(dimension, X, Y);
                X = X.offset(dimension as isize);
                *result = z;
                *resultTransp = z;
                result = result.offset(1 as libc::c_int as isize);
                resultTransp = resultTransp.offset(numDataX as isize);
                xi = xi.wrapping_add(1);
            }
            X = X
                .offset(
                    -(dimension
                        .wrapping_mul(
                            yi.wrapping_add(1 as libc::c_int as libc::c_ulonglong),
                        ) as isize),
                );
            Y = Y.offset(dimension as isize);
            result = result
                .offset(
                    numDataX
                        .wrapping_sub(
                            yi.wrapping_add(1 as libc::c_int as libc::c_ulonglong),
                        ) as isize,
                );
            resultTransp = resultTransp
                .offset(
                    (1 as libc::c_int as libc::c_ulonglong)
                        .wrapping_sub(
                            yi
                                .wrapping_add(1 as libc::c_int as libc::c_ulonglong)
                                .wrapping_mul(numDataX),
                        ) as isize,
                );
            yi = yi.wrapping_add(1);
        }
    };
}
#[no_mangle]
pub unsafe extern "C" fn vl_eval_vector_comparison_on_all_pairs_f(
    mut result: *mut libc::c_float,
    mut dimension: vl_size,
    mut X: *const libc::c_float,
    mut numDataX: vl_size,
    mut Y: *const libc::c_float,
    mut numDataY: vl_size,
    mut function: VlFloatVectorComparisonFunction,
) {
    let mut xi: vl_uindex = 0;
    let mut yi: vl_uindex = 0;
    if dimension == 0 as libc::c_int as libc::c_ulonglong {
        return;
    }
    if numDataX == 0 as libc::c_int as libc::c_ulonglong {
        return;
    }
    if !X.is_null() {} else {
        __assert_fail(
            b"X\0" as *const u8 as *const libc::c_char,
            b"vl/mathop.c\0" as *const u8 as *const libc::c_char,
            566 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 145],
                &[libc::c_char; 145],
            >(
                b"void vl_eval_vector_comparison_on_all_pairs_f(float *, vl_size, const float *, vl_size, const float *, vl_size, VlFloatVectorComparisonFunction)\0",
            ))
                .as_ptr(),
        );
    }
    if !Y.is_null() {
        if numDataY == 0 as libc::c_int as libc::c_ulonglong {
            return;
        }
        yi = 0 as libc::c_int as vl_uindex;
        while yi < numDataY {
            xi = 0 as libc::c_int as vl_uindex;
            while xi < numDataX {
                let fresh47 = result;
                result = result.offset(1);
                *fresh47 = (Some(function.expect("non-null function pointer")))
                    .expect("non-null function pointer")(dimension, X, Y);
                X = X.offset(dimension as isize);
                xi = xi.wrapping_add(1);
            }
            X = X.offset(-(dimension.wrapping_mul(numDataX) as isize));
            Y = Y.offset(dimension as isize);
            yi = yi.wrapping_add(1);
        }
    } else {
        let mut resultTransp: *mut libc::c_float = result;
        Y = X;
        yi = 0 as libc::c_int as vl_uindex;
        while yi < numDataX {
            xi = 0 as libc::c_int as vl_uindex;
            while xi <= yi {
                let mut z: libc::c_float = (Some(
                    function.expect("non-null function pointer"),
                ))
                    .expect("non-null function pointer")(dimension, X, Y);
                X = X.offset(dimension as isize);
                *result = z;
                *resultTransp = z;
                result = result.offset(1 as libc::c_int as isize);
                resultTransp = resultTransp.offset(numDataX as isize);
                xi = xi.wrapping_add(1);
            }
            X = X
                .offset(
                    -(dimension
                        .wrapping_mul(
                            yi.wrapping_add(1 as libc::c_int as libc::c_ulonglong),
                        ) as isize),
                );
            Y = Y.offset(dimension as isize);
            result = result
                .offset(
                    numDataX
                        .wrapping_sub(
                            yi.wrapping_add(1 as libc::c_int as libc::c_ulonglong),
                        ) as isize,
                );
            resultTransp = resultTransp
                .offset(
                    (1 as libc::c_int as libc::c_ulonglong)
                        .wrapping_sub(
                            yi
                                .wrapping_add(1 as libc::c_int as libc::c_ulonglong)
                                .wrapping_mul(numDataX),
                        ) as isize,
                );
            yi = yi.wrapping_add(1);
        }
    };
}
#[no_mangle]
pub unsafe extern "C" fn vl_svd2(
    mut S: *mut libc::c_double,
    mut U: *mut libc::c_double,
    mut V: *mut libc::c_double,
    mut M: *const libc::c_double,
) {
    let mut m11: libc::c_double = *M.offset(0 as libc::c_int as isize);
    let mut m21: libc::c_double = *M.offset(1 as libc::c_int as isize);
    let mut m12: libc::c_double = *M.offset(2 as libc::c_int as isize);
    let mut m22: libc::c_double = *M.offset(3 as libc::c_int as isize);
    let mut cu1: libc::c_double = m11;
    let mut su1: libc::c_double = m21;
    let mut norm: libc::c_double = sqrt(cu1 * cu1 + su1 * su1);
    let mut cu2: libc::c_double = 0.;
    let mut su2: libc::c_double = 0.;
    let mut cv2: libc::c_double = 0.;
    let mut sv2: libc::c_double = 0.;
    let mut f: libc::c_double = 0.;
    let mut g: libc::c_double = 0.;
    let mut h: libc::c_double = 0.;
    let mut smin: libc::c_double = 0.;
    let mut smax: libc::c_double = 0.;
    cu1 /= norm;
    su1 /= norm;
    f = cu1 * m11 + su1 * m21;
    g = cu1 * m12 + su1 * m22;
    h = -su1 * m12 + cu1 * m22;
    vl_lapack_dlasv2(
        &mut smin,
        &mut smax,
        &mut sv2,
        &mut cv2,
        &mut su2,
        &mut cu2,
        f,
        g,
        h,
    );
    if !S.is_null() {} else {
        __assert_fail(
            b"S\0" as *const u8 as *const libc::c_char,
            b"vl/mathop.c\0" as *const u8 as *const libc::c_char,
            665 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 59],
                &[libc::c_char; 59],
            >(b"void vl_svd2(double *, double *, double *, const double *)\0"))
                .as_ptr(),
        );
    }
    *S.offset(0 as libc::c_int as isize) = smax;
    *S.offset(1 as libc::c_int as isize) = 0 as libc::c_int as libc::c_double;
    *S.offset(2 as libc::c_int as isize) = 0 as libc::c_int as libc::c_double;
    *S.offset(3 as libc::c_int as isize) = smin;
    if !U.is_null() {
        *U.offset(0 as libc::c_int as isize) = cu2 * cu1 - su2 * su1;
        *U.offset(1 as libc::c_int as isize) = su2 * cu1 + cu2 * su1;
        *U.offset(2 as libc::c_int as isize) = -cu2 * su1 - su2 * cu1;
        *U.offset(3 as libc::c_int as isize) = -su2 * su1 + cu2 * cu1;
    }
    if !V.is_null() {
        *V.offset(0 as libc::c_int as isize) = cv2;
        *V.offset(1 as libc::c_int as isize) = sv2;
        *V.offset(2 as libc::c_int as isize) = -sv2;
        *V.offset(3 as libc::c_int as isize) = cv2;
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_lapack_dlasv2(
    mut smin: *mut libc::c_double,
    mut smax: *mut libc::c_double,
    mut sv: *mut libc::c_double,
    mut cv: *mut libc::c_double,
    mut su: *mut libc::c_double,
    mut cu: *mut libc::c_double,
    mut f: libc::c_double,
    mut g: libc::c_double,
    mut h: libc::c_double,
) {
    let mut svt: libc::c_double = 0.;
    let mut cvt: libc::c_double = 0.;
    let mut sut: libc::c_double = 0.;
    let mut cut: libc::c_double = 0.;
    let mut ft: libc::c_double = f;
    let mut gt: libc::c_double = g;
    let mut ht: libc::c_double = h;
    let mut fa: libc::c_double = fabs(f);
    let mut ga: libc::c_double = fabs(g);
    let mut ha: libc::c_double = fabs(h);
    let mut pmax: libc::c_int = 1 as libc::c_int;
    let mut swap: libc::c_int = 0 as libc::c_int;
    let mut glarge: libc::c_int = 0 as libc::c_int;
    let mut tsign: libc::c_int = 0;
    let mut fmh: libc::c_double = 0.;
    let mut d: libc::c_double = 0.;
    let mut dd: libc::c_double = 0.;
    let mut q: libc::c_double = 0.;
    let mut qq: libc::c_double = 0.;
    let mut s: libc::c_double = 0.;
    let mut ss: libc::c_double = 0.;
    let mut spq: libc::c_double = 0.;
    let mut dpq: libc::c_double = 0.;
    let mut a: libc::c_double = 0.;
    let mut tmp: libc::c_double = 0.;
    let mut tt: libc::c_double = 0.;
    if fa < ha {
        pmax = 3 as libc::c_int;
        tmp = ft;
        ft = ht;
        ht = tmp;
        tmp = fa;
        fa = ha;
        ha = tmp;
        swap = 1 as libc::c_int;
    }
    if ga == 0.0f64 {
        *smin = ha;
        *smax = fa;
        cut = 1.0f64;
        sut = 0.0f64;
        cvt = 1.0f64;
        svt = 0.0f64;
    } else {
        if ga > fa {
            pmax = 2 as libc::c_int;
            if fa / ga < 2.220446049250313e-16f64 {
                glarge = 1 as libc::c_int;
                *smax = ga;
                if ha > 1.0f64 {
                    *smin = fa / (ga / ha);
                } else {
                    *smin = fa / ga * ha;
                }
                cut = 1.0f64;
                sut = ht / gt;
                cvt = 1.0f64;
                svt = ft / gt;
            }
        }
        if glarge == 0 as libc::c_int {
            fmh = fa - ha;
            if fmh == fa {
                d = 1.0f64;
            } else {
                d = fmh / fa;
            }
            q = gt / ft;
            s = 2.0f64 - d;
            dd = d * d;
            qq = q * q;
            ss = s * s;
            spq = sqrt(ss + qq);
            if d == 0.0f64 {
                dpq = fabs(q);
            } else {
                dpq = sqrt(dd + qq);
            }
            a = 0.5f64 * (spq + dpq);
            *smin = ha / a;
            *smax = fa * a;
            if qq == 0.0f64 {
                if d == 0.0f64 {
                    tmp = ((if ft < 0.0f64 {
                        -(1 as libc::c_int)
                    } else {
                        1 as libc::c_int
                    }) * 2 as libc::c_int
                        * (if gt < 0.0f64 {
                            -(1 as libc::c_int)
                        } else {
                            1 as libc::c_int
                        })) as libc::c_double;
                } else {
                    tmp = gt
                        / ((if ft < 0.0f64 {
                            -(1 as libc::c_int)
                        } else {
                            1 as libc::c_int
                        }) as libc::c_double * fmh) + q / s;
                }
            } else {
                tmp = (q / (spq + s) + q / (dpq + d)) * (1.0f64 + a);
            }
            tt = sqrt(tmp * tmp + 4.0f64);
            cvt = 2.0f64 / tt;
            svt = tmp / tt;
            cut = (cvt + svt * q) / a;
            sut = ht / ft * svt / a;
        }
    }
    if swap == 1 as libc::c_int {
        *cu = svt;
        *su = cvt;
        *cv = sut;
        *sv = cut;
    } else {
        *cu = cut;
        *su = sut;
        *cv = cvt;
        *sv = svt;
    }
    if pmax == 1 as libc::c_int {
        tsign = (if *cv < 0.0f64 { -(1 as libc::c_int) } else { 1 as libc::c_int })
            * (if *cu < 0.0f64 { -(1 as libc::c_int) } else { 1 as libc::c_int })
            * (if f < 0.0f64 { -(1 as libc::c_int) } else { 1 as libc::c_int });
    }
    if pmax == 2 as libc::c_int {
        tsign = (if *sv < 0.0f64 { -(1 as libc::c_int) } else { 1 as libc::c_int })
            * (if *cu < 0.0f64 { -(1 as libc::c_int) } else { 1 as libc::c_int })
            * (if g < 0.0f64 { -(1 as libc::c_int) } else { 1 as libc::c_int });
    }
    if pmax == 3 as libc::c_int {
        tsign = (if *sv < 0.0f64 { -(1 as libc::c_int) } else { 1 as libc::c_int })
            * (if *su < 0.0f64 { -(1 as libc::c_int) } else { 1 as libc::c_int })
            * (if h < 0.0f64 { -(1 as libc::c_int) } else { 1 as libc::c_int });
    }
    *smax = (if tsign < 0 as libc::c_int {
        -(1 as libc::c_int)
    } else {
        1 as libc::c_int
    }) as libc::c_double * *smax;
    *smin = (if tsign * (if f < 0.0f64 { -(1 as libc::c_int) } else { 1 as libc::c_int })
        * (if h < 0.0f64 { -(1 as libc::c_int) } else { 1 as libc::c_int })
        < 0 as libc::c_int
    {
        -(1 as libc::c_int)
    } else {
        1 as libc::c_int
    }) as libc::c_double * *smin;
}
#[no_mangle]
pub unsafe extern "C" fn vl_solve_linear_system_3(
    mut x: *mut libc::c_double,
    mut A: *const libc::c_double,
    mut b: *const libc::c_double,
) -> libc::c_int {
    let mut err: libc::c_int = 0;
    let mut M: [libc::c_double; 12] = [0.; 12];
    M[0 as libc::c_int as usize] = *A.offset(0 as libc::c_int as isize);
    M[1 as libc::c_int as usize] = *A.offset(1 as libc::c_int as isize);
    M[2 as libc::c_int as usize] = *A.offset(2 as libc::c_int as isize);
    M[3 as libc::c_int as usize] = *A.offset(3 as libc::c_int as isize);
    M[4 as libc::c_int as usize] = *A.offset(4 as libc::c_int as isize);
    M[5 as libc::c_int as usize] = *A.offset(5 as libc::c_int as isize);
    M[6 as libc::c_int as usize] = *A.offset(6 as libc::c_int as isize);
    M[7 as libc::c_int as usize] = *A.offset(7 as libc::c_int as isize);
    M[8 as libc::c_int as usize] = *A.offset(8 as libc::c_int as isize);
    M[9 as libc::c_int as usize] = *b.offset(0 as libc::c_int as isize);
    M[10 as libc::c_int as usize] = *b.offset(1 as libc::c_int as isize);
    M[11 as libc::c_int as usize] = *b.offset(2 as libc::c_int as isize);
    err = vl_gaussian_elimination(
        M.as_mut_ptr(),
        3 as libc::c_int as vl_size,
        4 as libc::c_int as vl_size,
    );
    *x.offset(0 as libc::c_int as isize) = M[9 as libc::c_int as usize];
    *x.offset(1 as libc::c_int as isize) = M[10 as libc::c_int as usize];
    *x.offset(2 as libc::c_int as isize) = M[11 as libc::c_int as usize];
    return err;
}
#[no_mangle]
pub unsafe extern "C" fn vl_solve_linear_system_2(
    mut x: *mut libc::c_double,
    mut A: *const libc::c_double,
    mut b: *const libc::c_double,
) -> libc::c_int {
    let mut err: libc::c_int = 0;
    let mut M: [libc::c_double; 6] = [0.; 6];
    M[0 as libc::c_int as usize] = *A.offset(0 as libc::c_int as isize);
    M[1 as libc::c_int as usize] = *A.offset(1 as libc::c_int as isize);
    M[2 as libc::c_int as usize] = *A.offset(2 as libc::c_int as isize);
    M[3 as libc::c_int as usize] = *A.offset(3 as libc::c_int as isize);
    M[4 as libc::c_int as usize] = *b.offset(0 as libc::c_int as isize);
    M[5 as libc::c_int as usize] = *b.offset(1 as libc::c_int as isize);
    err = vl_gaussian_elimination(
        M.as_mut_ptr(),
        2 as libc::c_int as vl_size,
        3 as libc::c_int as vl_size,
    );
    *x.offset(0 as libc::c_int as isize) = M[4 as libc::c_int as usize];
    *x.offset(1 as libc::c_int as isize) = M[5 as libc::c_int as usize];
    return err;
}
#[no_mangle]
pub unsafe extern "C" fn vl_gaussian_elimination(
    mut A: *mut libc::c_double,
    mut numRows: vl_size,
    mut numColumns: vl_size,
) -> libc::c_int {
    let mut i: vl_index = 0;
    let mut j: vl_index = 0;
    let mut ii: vl_index = 0;
    let mut jj: vl_index = 0;
    if !A.is_null() {} else {
        __assert_fail(
            b"A\0" as *const u8 as *const libc::c_char,
            b"vl/mathop.c\0" as *const u8 as *const libc::c_char,
            909 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 56],
                &[libc::c_char; 56],
            >(b"int vl_gaussian_elimination(double *, vl_size, vl_size)\0"))
                .as_ptr(),
        );
    }
    if numRows <= numColumns {} else {
        __assert_fail(
            b"numRows <= numColumns\0" as *const u8 as *const libc::c_char,
            b"vl/mathop.c\0" as *const u8 as *const libc::c_char,
            910 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 56],
                &[libc::c_char; 56],
            >(b"int vl_gaussian_elimination(double *, vl_size, vl_size)\0"))
                .as_ptr(),
        );
    }
    j = 0 as libc::c_int as vl_index;
    while j < numRows as libc::c_int as libc::c_longlong {
        let mut maxa: libc::c_double = 0 as libc::c_int as libc::c_double;
        let mut maxabsa: libc::c_double = 0 as libc::c_int as libc::c_double;
        let mut maxi: vl_index = -(1 as libc::c_int) as vl_index;
        let mut tmp: libc::c_double = 0.;
        i = j;
        while i < numRows as libc::c_int as libc::c_longlong {
            let mut a: libc::c_double = *A
                .offset(
                    (i as libc::c_ulonglong)
                        .wrapping_add((j as libc::c_ulonglong).wrapping_mul(numRows))
                        as isize,
                );
            let mut absa: libc::c_double = vl_abs_d(a);
            if absa > maxabsa {
                maxa = a;
                maxabsa = absa;
                maxi = i;
            }
            i += 1;
        }
        i = maxi;
        if maxabsa < 1e-10f64 {
            return 1 as libc::c_int;
        }
        jj = j;
        while jj < numColumns as libc::c_int as libc::c_longlong {
            tmp = *A
                .offset(
                    (i as libc::c_ulonglong)
                        .wrapping_add((jj as libc::c_ulonglong).wrapping_mul(numRows))
                        as isize,
                );
            *A
                .offset(
                    (i as libc::c_ulonglong)
                        .wrapping_add((jj as libc::c_ulonglong).wrapping_mul(numRows))
                        as isize,
                ) = *A
                .offset(
                    (j as libc::c_ulonglong)
                        .wrapping_add((jj as libc::c_ulonglong).wrapping_mul(numRows))
                        as isize,
                );
            *A
                .offset(
                    (j as libc::c_ulonglong)
                        .wrapping_add((jj as libc::c_ulonglong).wrapping_mul(numRows))
                        as isize,
                ) = tmp;
            *A
                .offset(
                    (j as libc::c_ulonglong)
                        .wrapping_add((jj as libc::c_ulonglong).wrapping_mul(numRows))
                        as isize,
                ) /= maxa;
            jj += 1;
        }
        ii = j + 1 as libc::c_int as libc::c_longlong;
        while ii < numRows as libc::c_int as libc::c_longlong {
            let mut x: libc::c_double = *A
                .offset(
                    (ii as libc::c_ulonglong)
                        .wrapping_add((j as libc::c_ulonglong).wrapping_mul(numRows))
                        as isize,
                );
            jj = j;
            while jj < numColumns as libc::c_int as libc::c_longlong {
                *A
                    .offset(
                        (ii as libc::c_ulonglong)
                            .wrapping_add(
                                (jj as libc::c_ulonglong).wrapping_mul(numRows),
                            ) as isize,
                    )
                    -= x
                        * *A
                            .offset(
                                (j as libc::c_ulonglong)
                                    .wrapping_add(
                                        (jj as libc::c_ulonglong).wrapping_mul(numRows),
                                    ) as isize,
                            );
                jj += 1;
            }
            ii += 1;
        }
        j += 1;
    }
    i = numRows.wrapping_sub(1 as libc::c_int as libc::c_ulonglong) as vl_index;
    while i > 0 as libc::c_int as libc::c_longlong {
        ii = i - 1 as libc::c_int as libc::c_longlong;
        while ii >= 0 as libc::c_int as libc::c_longlong {
            let mut x_0: libc::c_double = *A
                .offset(
                    (ii as libc::c_ulonglong)
                        .wrapping_add((i as libc::c_ulonglong).wrapping_mul(numRows))
                        as isize,
                );
            j = numRows as vl_index;
            while j < numColumns as libc::c_int as libc::c_longlong {
                *A
                    .offset(
                        (ii as libc::c_ulonglong)
                            .wrapping_add((j as libc::c_ulonglong).wrapping_mul(numRows))
                            as isize,
                    )
                    -= x_0
                        * *A
                            .offset(
                                (i as libc::c_ulonglong)
                                    .wrapping_add(
                                        (j as libc::c_ulonglong).wrapping_mul(numRows),
                                    ) as isize,
                            );
                j += 1;
            }
            ii -= 1;
        }
        i -= 1;
    }
    return 0 as libc::c_int;
}
