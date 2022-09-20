use ::libc;
extern "C" {
    fn abort() -> !;
    fn __assert_fail(
        __assertion: *const libc::c_char,
        __file: *const libc::c_char,
        __line: libc::c_uint,
        __function: *const libc::c_char,
    ) -> !;
    fn vl_free(ptr: *mut libc::c_void);
    fn vl_malloc(n: size_t) -> *mut libc::c_void;
    fn sqrt(_: libc::c_double) -> libc::c_double;
    fn vl_get_gmm_data_posteriors_f(
        posteriors: *mut libc::c_float,
        numClusters: vl_size,
        numData: vl_size,
        priors: *const libc::c_float,
        means: *const libc::c_float,
        dimension: vl_size,
        covariances: *const libc::c_float,
        data: *const libc::c_float,
    ) -> libc::c_double;
    fn vl_get_gmm_data_posteriors_d(
        posteriors: *mut libc::c_double,
        numClusters: vl_size,
        numData: vl_size,
        priors: *const libc::c_double,
        means: *const libc::c_double,
        dimension: vl_size,
        covariances: *const libc::c_double,
        data: *const libc::c_double,
    ) -> libc::c_double;
    fn memset(
        _: *mut libc::c_void,
        _: libc::c_int,
        _: libc::c_ulong,
    ) -> *mut libc::c_void;
}
pub type vl_int64 = libc::c_longlong;
pub type vl_uint64 = libc::c_ulonglong;
pub type vl_uint32 = libc::c_uint;
pub type vl_size = vl_uint64;
pub type vl_index = vl_int64;
pub type size_t = libc::c_ulong;
pub type vl_type = vl_uint32;
unsafe extern "C" fn _vl_fisher_encode_f(
    mut enc: *mut libc::c_float,
    mut means: *const libc::c_float,
    mut dimension: vl_size,
    mut numClusters: vl_size,
    mut covariances: *const libc::c_float,
    mut priors: *const libc::c_float,
    mut data: *const libc::c_float,
    mut numData: vl_size,
    mut flags: libc::c_int,
) -> vl_size {
    let mut dim: vl_size = 0;
    let mut i_cl: vl_index = 0;
    let mut i_d: vl_index = 0;
    let mut numTerms: vl_size = 0 as libc::c_int as vl_size;
    let mut posteriors: *mut libc::c_float = 0 as *mut libc::c_float;
    let mut sqrtInvSigma: *mut libc::c_float = 0 as *mut libc::c_float;
    if numClusters >= 1 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"numClusters >= 1\0" as *const u8 as *const libc::c_char,
            b"vl/fisher.c\0" as *const u8 as *const libc::c_char,
            378 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 129],
                &[libc::c_char; 129],
            >(
                b"vl_size _vl_fisher_encode_f(float *, const float *, vl_size, vl_size, const float *, const float *, const float *, vl_size, int)\0",
            ))
                .as_ptr(),
        );
    }
    if dimension >= 1 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"dimension >= 1\0" as *const u8 as *const libc::c_char,
            b"vl/fisher.c\0" as *const u8 as *const libc::c_char,
            379 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 129],
                &[libc::c_char; 129],
            >(
                b"vl_size _vl_fisher_encode_f(float *, const float *, vl_size, vl_size, const float *, const float *, const float *, vl_size, int)\0",
            ))
                .as_ptr(),
        );
    }
    posteriors = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numClusters)
            .wrapping_mul(numData) as size_t,
    ) as *mut libc::c_float;
    sqrtInvSigma = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(dimension)
            .wrapping_mul(numClusters) as size_t,
    ) as *mut libc::c_float;
    memset(
        enc as *mut libc::c_void,
        0 as libc::c_int,
        ((::core::mem::size_of::<libc::c_float>() as libc::c_ulong)
            .wrapping_mul(2 as libc::c_int as libc::c_ulong) as libc::c_ulonglong)
            .wrapping_mul(dimension)
            .wrapping_mul(numClusters) as libc::c_ulong,
    );
    i_cl = 0 as libc::c_int as vl_index;
    while i_cl < numClusters as libc::c_int as libc::c_longlong {
        dim = 0 as libc::c_int as vl_size;
        while dim < dimension {
            *sqrtInvSigma
                .offset(
                    (i_cl as libc::c_ulonglong).wrapping_mul(dimension).wrapping_add(dim)
                        as isize,
                ) = sqrt(
                1.0f64
                    / *covariances
                        .offset(
                            (i_cl as libc::c_ulonglong)
                                .wrapping_mul(dimension)
                                .wrapping_add(dim) as isize,
                        ) as libc::c_double,
            ) as libc::c_float;
            dim = dim.wrapping_add(1);
        }
        i_cl += 1;
    }
    vl_get_gmm_data_posteriors_f(
        posteriors,
        numClusters,
        numData,
        priors,
        means,
        dimension,
        covariances,
        data,
    );
    if flags & (0x1 as libc::c_int) << 2 as libc::c_int != 0 {
        i_d = 0 as libc::c_int as vl_index;
        while i_d < numData as libc::c_int as libc::c_longlong {
            let mut best: vl_index = 0 as libc::c_int as vl_index;
            let mut bestValue: libc::c_float = *posteriors
                .offset((i_d as libc::c_ulonglong).wrapping_mul(numClusters) as isize);
            i_cl = 1 as libc::c_int as vl_index;
            while i_cl < numClusters as libc::c_int as libc::c_longlong {
                let mut p: libc::c_float = *posteriors
                    .offset(
                        (i_cl as libc::c_ulonglong)
                            .wrapping_add(
                                (i_d as libc::c_ulonglong).wrapping_mul(numClusters),
                            ) as isize,
                    );
                if p > bestValue {
                    bestValue = p;
                    best = i_cl;
                }
                i_cl += 1;
            }
            i_cl = 0 as libc::c_int as vl_index;
            while i_cl < numClusters as libc::c_int as libc::c_longlong {
                *posteriors
                    .offset(
                        (i_cl as libc::c_ulonglong)
                            .wrapping_add(
                                (i_d as libc::c_ulonglong).wrapping_mul(numClusters),
                            ) as isize,
                    ) = (i_cl == best) as libc::c_int as libc::c_float;
                i_cl += 1;
            }
            i_d += 1;
        }
    }
    i_cl = 0 as libc::c_int as vl_index;
    while i_cl < numClusters as libc::c_int as libc::c_longlong {
        let mut uprefix: libc::c_float = 0.;
        let mut vprefix: libc::c_float = 0.;
        let mut uk: *mut libc::c_float = enc
            .offset((i_cl as libc::c_ulonglong).wrapping_mul(dimension) as isize);
        let mut vk: *mut libc::c_float = enc
            .offset((i_cl as libc::c_ulonglong).wrapping_mul(dimension) as isize)
            .offset(numClusters.wrapping_mul(dimension) as isize);
        if !((*priors.offset(i_cl as isize) as libc::c_double) < 1e-6f64) {
            i_d = 0 as libc::c_int as vl_index;
            while i_d < numData as libc::c_int as libc::c_longlong {
                let mut p_0: libc::c_float = *posteriors
                    .offset(
                        (i_cl as libc::c_ulonglong)
                            .wrapping_add(
                                (i_d as libc::c_ulonglong).wrapping_mul(numClusters),
                            ) as isize,
                    );
                if !((p_0 as libc::c_double) < 1e-6f64) {
                    numTerms = (numTerms as libc::c_ulonglong)
                        .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as vl_size
                        as vl_size;
                    dim = 0 as libc::c_int as vl_size;
                    while dim < dimension {
                        let mut diff: libc::c_float = *data
                            .offset(
                                (i_d as libc::c_ulonglong)
                                    .wrapping_mul(dimension)
                                    .wrapping_add(dim) as isize,
                            )
                            - *means
                                .offset(
                                    (i_cl as libc::c_ulonglong)
                                        .wrapping_mul(dimension)
                                        .wrapping_add(dim) as isize,
                                );
                        diff
                            *= *sqrtInvSigma
                                .offset(
                                    (i_cl as libc::c_ulonglong)
                                        .wrapping_mul(dimension)
                                        .wrapping_add(dim) as isize,
                                );
                        *uk.offset(dim as isize) += p_0 * diff;
                        *vk.offset(dim as isize)
                            += p_0 * (diff * diff - 1 as libc::c_int as libc::c_float);
                        dim = dim.wrapping_add(1);
                    }
                }
                i_d += 1;
            }
            if numData > 0 as libc::c_int as libc::c_ulonglong {
                uprefix = (1 as libc::c_int as libc::c_double
                    / (numData as libc::c_double
                        * sqrt(*priors.offset(i_cl as isize) as libc::c_double)))
                    as libc::c_float;
                vprefix = (1 as libc::c_int as libc::c_double
                    / (numData as libc::c_double
                        * sqrt(
                            (2 as libc::c_int as libc::c_float
                                * *priors.offset(i_cl as isize)) as libc::c_double,
                        ))) as libc::c_float;
                dim = 0 as libc::c_int as vl_size;
                while dim < dimension {
                    *uk.offset(dim as isize) = *uk.offset(dim as isize) * uprefix;
                    *vk.offset(dim as isize) = *vk.offset(dim as isize) * vprefix;
                    dim = dim.wrapping_add(1);
                }
            }
        }
        i_cl += 1;
    }
    vl_free(posteriors as *mut libc::c_void);
    vl_free(sqrtInvSigma as *mut libc::c_void);
    if flags & (0x1 as libc::c_int) << 0 as libc::c_int != 0 {
        dim = 0 as libc::c_int as vl_size;
        while dim
            < (2 as libc::c_int as libc::c_ulonglong)
                .wrapping_mul(dimension)
                .wrapping_mul(numClusters)
        {
            let mut z: libc::c_float = *enc.offset(dim as isize);
            if z >= 0 as libc::c_int as libc::c_float {
                *enc.offset(dim as isize) = f32::sqrt(z);
            } else {
                *enc.offset(dim as isize) = -f32::sqrt(-z);
            }
            dim = dim.wrapping_add(1);
        }
    }
    if flags & (0x1 as libc::c_int) << 1 as libc::c_int != 0 {
        let mut n: libc::c_float = 0 as libc::c_int as libc::c_float;
        dim = 0 as libc::c_int as vl_size;
        while dim
            < (2 as libc::c_int as libc::c_ulonglong)
                .wrapping_mul(dimension)
                .wrapping_mul(numClusters)
        {
            let mut z_0: libc::c_float = *enc.offset(dim as isize);
            n += z_0 * z_0;
            dim = dim.wrapping_add(1);
        }
        n = f32::sqrt(n);
        n = (if n as libc::c_double > 1e-12f64 { n as libc::c_double } else { 1e-12f64 })
            as libc::c_float;
        dim = 0 as libc::c_int as vl_size;
        while dim
            < (2 as libc::c_int as libc::c_ulonglong)
                .wrapping_mul(dimension)
                .wrapping_mul(numClusters)
        {
            *enc.offset(dim as isize) /= n;
            dim = dim.wrapping_add(1);
        }
    }
    return numTerms;
}
unsafe extern "C" fn _vl_fisher_encode_d(
    mut enc: *mut libc::c_double,
    mut means: *const libc::c_double,
    mut dimension: vl_size,
    mut numClusters: vl_size,
    mut covariances: *const libc::c_double,
    mut priors: *const libc::c_double,
    mut data: *const libc::c_double,
    mut numData: vl_size,
    mut flags: libc::c_int,
) -> vl_size {
    let mut dim: vl_size = 0;
    let mut i_cl: vl_index = 0;
    let mut i_d: vl_index = 0;
    let mut numTerms: vl_size = 0 as libc::c_int as vl_size;
    let mut posteriors: *mut libc::c_double = 0 as *mut libc::c_double;
    let mut sqrtInvSigma: *mut libc::c_double = 0 as *mut libc::c_double;
    if numClusters >= 1 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"numClusters >= 1\0" as *const u8 as *const libc::c_char,
            b"vl/fisher.c\0" as *const u8 as *const libc::c_char,
            378 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 134],
                &[libc::c_char; 134],
            >(
                b"vl_size _vl_fisher_encode_d(double *, const double *, vl_size, vl_size, const double *, const double *, const double *, vl_size, int)\0",
            ))
                .as_ptr(),
        );
    }
    if dimension >= 1 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"dimension >= 1\0" as *const u8 as *const libc::c_char,
            b"vl/fisher.c\0" as *const u8 as *const libc::c_char,
            379 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 134],
                &[libc::c_char; 134],
            >(
                b"vl_size _vl_fisher_encode_d(double *, const double *, vl_size, vl_size, const double *, const double *, const double *, vl_size, int)\0",
            ))
                .as_ptr(),
        );
    }
    posteriors = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numClusters)
            .wrapping_mul(numData) as size_t,
    ) as *mut libc::c_double;
    sqrtInvSigma = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(dimension)
            .wrapping_mul(numClusters) as size_t,
    ) as *mut libc::c_double;
    memset(
        enc as *mut libc::c_void,
        0 as libc::c_int,
        ((::core::mem::size_of::<libc::c_double>() as libc::c_ulong)
            .wrapping_mul(2 as libc::c_int as libc::c_ulong) as libc::c_ulonglong)
            .wrapping_mul(dimension)
            .wrapping_mul(numClusters) as libc::c_ulong,
    );
    i_cl = 0 as libc::c_int as vl_index;
    while i_cl < numClusters as libc::c_int as libc::c_longlong {
        dim = 0 as libc::c_int as vl_size;
        while dim < dimension {
            *sqrtInvSigma
                .offset(
                    (i_cl as libc::c_ulonglong).wrapping_mul(dimension).wrapping_add(dim)
                        as isize,
                ) = sqrt(
                1.0f64
                    / *covariances
                        .offset(
                            (i_cl as libc::c_ulonglong)
                                .wrapping_mul(dimension)
                                .wrapping_add(dim) as isize,
                        ),
            );
            dim = dim.wrapping_add(1);
        }
        i_cl += 1;
    }
    vl_get_gmm_data_posteriors_d(
        posteriors,
        numClusters,
        numData,
        priors,
        means,
        dimension,
        covariances,
        data,
    );
    if flags & (0x1 as libc::c_int) << 2 as libc::c_int != 0 {
        i_d = 0 as libc::c_int as vl_index;
        while i_d < numData as libc::c_int as libc::c_longlong {
            let mut best: vl_index = 0 as libc::c_int as vl_index;
            let mut bestValue: libc::c_double = *posteriors
                .offset((i_d as libc::c_ulonglong).wrapping_mul(numClusters) as isize);
            i_cl = 1 as libc::c_int as vl_index;
            while i_cl < numClusters as libc::c_int as libc::c_longlong {
                let mut p: libc::c_double = *posteriors
                    .offset(
                        (i_cl as libc::c_ulonglong)
                            .wrapping_add(
                                (i_d as libc::c_ulonglong).wrapping_mul(numClusters),
                            ) as isize,
                    );
                if p > bestValue {
                    bestValue = p;
                    best = i_cl;
                }
                i_cl += 1;
            }
            i_cl = 0 as libc::c_int as vl_index;
            while i_cl < numClusters as libc::c_int as libc::c_longlong {
                *posteriors
                    .offset(
                        (i_cl as libc::c_ulonglong)
                            .wrapping_add(
                                (i_d as libc::c_ulonglong).wrapping_mul(numClusters),
                            ) as isize,
                    ) = (i_cl == best) as libc::c_int as libc::c_double;
                i_cl += 1;
            }
            i_d += 1;
        }
    }
    i_cl = 0 as libc::c_int as vl_index;
    while i_cl < numClusters as libc::c_int as libc::c_longlong {
        let mut uprefix: libc::c_double = 0.;
        let mut vprefix: libc::c_double = 0.;
        let mut uk: *mut libc::c_double = enc
            .offset((i_cl as libc::c_ulonglong).wrapping_mul(dimension) as isize);
        let mut vk: *mut libc::c_double = enc
            .offset((i_cl as libc::c_ulonglong).wrapping_mul(dimension) as isize)
            .offset(numClusters.wrapping_mul(dimension) as isize);
        if !(*priors.offset(i_cl as isize) < 1e-6f64) {
            i_d = 0 as libc::c_int as vl_index;
            while i_d < numData as libc::c_int as libc::c_longlong {
                let mut p_0: libc::c_double = *posteriors
                    .offset(
                        (i_cl as libc::c_ulonglong)
                            .wrapping_add(
                                (i_d as libc::c_ulonglong).wrapping_mul(numClusters),
                            ) as isize,
                    );
                if !(p_0 < 1e-6f64) {
                    numTerms = (numTerms as libc::c_ulonglong)
                        .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as vl_size
                        as vl_size;
                    dim = 0 as libc::c_int as vl_size;
                    while dim < dimension {
                        let mut diff: libc::c_double = *data
                            .offset(
                                (i_d as libc::c_ulonglong)
                                    .wrapping_mul(dimension)
                                    .wrapping_add(dim) as isize,
                            )
                            - *means
                                .offset(
                                    (i_cl as libc::c_ulonglong)
                                        .wrapping_mul(dimension)
                                        .wrapping_add(dim) as isize,
                                );
                        diff
                            *= *sqrtInvSigma
                                .offset(
                                    (i_cl as libc::c_ulonglong)
                                        .wrapping_mul(dimension)
                                        .wrapping_add(dim) as isize,
                                );
                        *uk.offset(dim as isize) += p_0 * diff;
                        *vk.offset(dim as isize)
                            += p_0 * (diff * diff - 1 as libc::c_int as libc::c_double);
                        dim = dim.wrapping_add(1);
                    }
                }
                i_d += 1;
            }
            if numData > 0 as libc::c_int as libc::c_ulonglong {
                uprefix = 1 as libc::c_int as libc::c_double
                    / (numData as libc::c_double * sqrt(*priors.offset(i_cl as isize)));
                vprefix = 1 as libc::c_int as libc::c_double
                    / (numData as libc::c_double
                        * sqrt(
                            2 as libc::c_int as libc::c_double
                                * *priors.offset(i_cl as isize),
                        ));
                dim = 0 as libc::c_int as vl_size;
                while dim < dimension {
                    *uk.offset(dim as isize) = *uk.offset(dim as isize) * uprefix;
                    *vk.offset(dim as isize) = *vk.offset(dim as isize) * vprefix;
                    dim = dim.wrapping_add(1);
                }
            }
        }
        i_cl += 1;
    }
    vl_free(posteriors as *mut libc::c_void);
    vl_free(sqrtInvSigma as *mut libc::c_void);
    if flags & (0x1 as libc::c_int) << 0 as libc::c_int != 0 {
        dim = 0 as libc::c_int as vl_size;
        while dim
            < (2 as libc::c_int as libc::c_ulonglong)
                .wrapping_mul(dimension)
                .wrapping_mul(numClusters)
        {
            let mut z: libc::c_double = *enc.offset(dim as isize);
            if z >= 0 as libc::c_int as libc::c_double {
                *enc.offset(dim as isize) = f64::sqrt(z);
            } else {
                *enc.offset(dim as isize) = -f64::sqrt(-z);
            }
            dim = dim.wrapping_add(1);
        }
    }
    if flags & (0x1 as libc::c_int) << 1 as libc::c_int != 0 {
        let mut n: libc::c_double = 0 as libc::c_int as libc::c_double;
        dim = 0 as libc::c_int as vl_size;
        while dim
            < (2 as libc::c_int as libc::c_ulonglong)
                .wrapping_mul(dimension)
                .wrapping_mul(numClusters)
        {
            let mut z_0: libc::c_double = *enc.offset(dim as isize);
            n += z_0 * z_0;
            dim = dim.wrapping_add(1);
        }
        n = f64::sqrt(n);
        n = if n > 1e-12f64 { n } else { 1e-12f64 };
        dim = 0 as libc::c_int as vl_size;
        while dim
            < (2 as libc::c_int as libc::c_ulonglong)
                .wrapping_mul(dimension)
                .wrapping_mul(numClusters)
        {
            *enc.offset(dim as isize) /= n;
            dim = dim.wrapping_add(1);
        }
    }
    return numTerms;
}
#[no_mangle]
pub unsafe extern "C" fn vl_fisher_encode(
    mut enc: *mut libc::c_void,
    mut dataType: vl_type,
    mut means: *const libc::c_void,
    mut dimension: vl_size,
    mut numClusters: vl_size,
    mut covariances: *const libc::c_void,
    mut priors: *const libc::c_void,
    mut data: *const libc::c_void,
    mut numData: vl_size,
    mut flags: libc::c_int,
) -> vl_size {
    match dataType {
        1 => {
            return _vl_fisher_encode_f(
                enc as *mut libc::c_float,
                means as *const libc::c_float,
                dimension,
                numClusters,
                covariances as *const libc::c_float,
                priors as *const libc::c_float,
                data as *const libc::c_float,
                numData,
                flags,
            );
        }
        2 => {
            return _vl_fisher_encode_d(
                enc as *mut libc::c_double,
                means as *const libc::c_double,
                dimension,
                numClusters,
                covariances as *const libc::c_double,
                priors as *const libc::c_double,
                data as *const libc::c_double,
                numData,
                flags,
            );
        }
        _ => {
            abort();
        }
    };
}
