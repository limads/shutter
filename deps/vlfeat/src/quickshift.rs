use ::libc;
extern "C" {
    fn vl_malloc(n: size_t) -> *mut libc::c_void;
    fn vl_calloc(n: size_t, size: size_t) -> *mut libc::c_void;
    fn vl_free(ptr: *mut libc::c_void);
    fn exp(_: libc::c_double) -> libc::c_double;
    fn sqrt(_: libc::c_double) -> libc::c_double;
    fn ceil(_: libc::c_double) -> libc::c_double;
}
pub type vl_uint64 = libc::c_ulonglong;
pub type vl_bool = libc::c_int;
pub type size_t = libc::c_ulong;
#[derive(Copy, Clone)]
#[repr(C)]
pub union C2RustUnnamed {
    pub raw: vl_uint64,
    pub value: libc::c_double,
}
pub type vl_qs_type = libc::c_double;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlQS {
    pub image: *mut vl_qs_type,
    pub height: libc::c_int,
    pub width: libc::c_int,
    pub channels: libc::c_int,
    pub medoid: vl_bool,
    pub sigma: vl_qs_type,
    pub tau: vl_qs_type,
    pub parents: *mut libc::c_int,
    pub dists: *mut vl_qs_type,
    pub density: *mut vl_qs_type,
}
pub type VlQS = _VlQS;
static mut vl_infinity_d: C2RustUnnamed = C2RustUnnamed {
    raw: 0x7ff0000000000000 as libc::c_ulonglong,
};
#[inline]
unsafe extern "C" fn vl_quickshift_distance(
    mut I: *const vl_qs_type,
    mut N1: libc::c_int,
    mut N2: libc::c_int,
    mut K: libc::c_int,
    mut i1: libc::c_int,
    mut i2: libc::c_int,
    mut j1: libc::c_int,
    mut j2: libc::c_int,
) -> vl_qs_type {
    let mut dist: vl_qs_type = 0 as libc::c_int as vl_qs_type;
    let mut d1: libc::c_int = j1 - i1;
    let mut d2: libc::c_int = j2 - i2;
    let mut k: libc::c_int = 0;
    dist += (d1 * d1 + d2 * d2) as libc::c_double;
    k = 0 as libc::c_int;
    while k < K {
        let mut d: vl_qs_type = *I.offset((i1 + N1 * i2 + N1 * N2 * k) as isize)
            - *I.offset((j1 + N1 * j2 + N1 * N2 * k) as isize);
        dist += d * d;
        k += 1;
    }
    return dist;
}
#[inline]
unsafe extern "C" fn vl_quickshift_inner(
    mut I: *const vl_qs_type,
    mut N1: libc::c_int,
    mut N2: libc::c_int,
    mut K: libc::c_int,
    mut i1: libc::c_int,
    mut i2: libc::c_int,
    mut j1: libc::c_int,
    mut j2: libc::c_int,
) -> vl_qs_type {
    let mut ker: vl_qs_type = 0 as libc::c_int as vl_qs_type;
    let mut k: libc::c_int = 0;
    ker += (i1 * j1 + i2 * j2) as libc::c_double;
    k = 0 as libc::c_int;
    while k < K {
        ker
            += *I.offset((i1 + N1 * i2 + N1 * N2 * k) as isize)
                * *I.offset((j1 + N1 * j2 + N1 * N2 * k) as isize);
        k += 1;
    }
    return ker;
}
#[no_mangle]
pub unsafe extern "C" fn vl_quickshift_new(
    mut image: *const vl_qs_type,
    mut height: libc::c_int,
    mut width: libc::c_int,
    mut channels: libc::c_int,
) -> *mut VlQS {
    let mut q: *mut VlQS = vl_malloc(::core::mem::size_of::<VlQS>() as libc::c_ulong)
        as *mut VlQS;
    (*q).image = image as *mut vl_qs_type;
    (*q).height = height;
    (*q).width = width;
    (*q).channels = channels;
    (*q).medoid = 0 as libc::c_int;
    (*q)
        .tau = ((if height > width { height } else { width }) / 50 as libc::c_int)
        as vl_qs_type;
    (*q)
        .sigma = if 2 as libc::c_int as libc::c_double
        > (*q).tau / 3 as libc::c_int as libc::c_double
    {
        2 as libc::c_int as libc::c_double
    } else {
        (*q).tau / 3 as libc::c_int as libc::c_double
    };
    (*q)
        .dists = vl_calloc(
        (height * width) as size_t,
        ::core::mem::size_of::<vl_qs_type>() as libc::c_ulong,
    ) as *mut vl_qs_type;
    (*q)
        .parents = vl_calloc(
        (height * width) as size_t,
        ::core::mem::size_of::<libc::c_int>() as libc::c_ulong,
    ) as *mut libc::c_int;
    (*q)
        .density = vl_calloc(
        (height * width) as size_t,
        ::core::mem::size_of::<vl_qs_type>() as libc::c_ulong,
    ) as *mut vl_qs_type;
    return q;
}
#[no_mangle]
pub unsafe extern "C" fn vl_quickshift_process(mut q: *mut VlQS) {
    let mut I: *const vl_qs_type = (*q).image;
    let mut parents: *mut libc::c_int = (*q).parents;
    let mut E: *mut vl_qs_type = (*q).density;
    let mut dists: *mut vl_qs_type = (*q).dists;
    let mut M: *mut vl_qs_type = 0 as *mut vl_qs_type;
    let mut n: *mut vl_qs_type = 0 as *mut vl_qs_type;
    let mut sigma: vl_qs_type = (*q).sigma;
    let mut tau: vl_qs_type = (*q).tau;
    let mut tau2: vl_qs_type = tau * tau;
    let mut K: libc::c_int = (*q).channels;
    let mut d: libc::c_int = 0;
    let mut N1: libc::c_int = (*q).height;
    let mut N2: libc::c_int = (*q).width;
    let mut i1: libc::c_int = 0;
    let mut i2: libc::c_int = 0;
    let mut j1: libc::c_int = 0;
    let mut j2: libc::c_int = 0;
    let mut R: libc::c_int = 0;
    let mut tR: libc::c_int = 0;
    d = 2 as libc::c_int + K;
    if (*q).medoid != 0 {
        M = vl_calloc(
            (N1 * N2 * d) as size_t,
            ::core::mem::size_of::<vl_qs_type>() as libc::c_ulong,
        ) as *mut vl_qs_type;
        n = vl_calloc(
            (N1 * N2) as size_t,
            ::core::mem::size_of::<vl_qs_type>() as libc::c_ulong,
        ) as *mut vl_qs_type;
    }
    R = ceil(3 as libc::c_int as libc::c_double * sigma) as libc::c_int;
    tR = ceil(tau) as libc::c_int;
    if !n.is_null() {
        i2 = 0 as libc::c_int;
        while i2 < N2 {
            i1 = 0 as libc::c_int;
            while i1 < N1 {
                *n
                    .offset(
                        (i1 + N1 * i2) as isize,
                    ) = vl_quickshift_inner(I, N1, N2, K, i1, i2, i1, i2);
                i1 += 1;
            }
            i2 += 1;
        }
    }
    i2 = 0 as libc::c_int;
    while i2 < N2 {
        i1 = 0 as libc::c_int;
        while i1 < N1 {
            let mut j1min: libc::c_int = if i1 - R > 0 as libc::c_int {
                i1 - R
            } else {
                0 as libc::c_int
            };
            let mut j1max: libc::c_int = if i1 + R < N1 - 1 as libc::c_int {
                i1 + R
            } else {
                N1 - 1 as libc::c_int
            };
            let mut j2min: libc::c_int = if i2 - R > 0 as libc::c_int {
                i2 - R
            } else {
                0 as libc::c_int
            };
            let mut j2max: libc::c_int = if i2 + R < N2 - 1 as libc::c_int {
                i2 + R
            } else {
                N2 - 1 as libc::c_int
            };
            j2 = j2min;
            while j2 <= j2max {
                j1 = j1min;
                while j1 <= j1max {
                    let mut Dij: vl_qs_type = vl_quickshift_distance(
                        I,
                        N1,
                        N2,
                        K,
                        i1,
                        i2,
                        j1,
                        j2,
                    );
                    let mut Fij: vl_qs_type = -exp(
                        -Dij / (2 as libc::c_int as libc::c_double * sigma * sigma),
                    );
                    let ref mut fresh0 = *E.offset((i1 + N1 * i2) as isize);
                    *fresh0 -= Fij;
                    if !M.is_null() {
                        let mut k: libc::c_int = 0;
                        let ref mut fresh1 = *M
                            .offset(
                                (i1 + N1 * i2 + N1 * N2 * 0 as libc::c_int) as isize,
                            );
                        *fresh1 += j1 as libc::c_double * Fij;
                        let ref mut fresh2 = *M
                            .offset(
                                (i1 + N1 * i2 + N1 * N2 * 1 as libc::c_int) as isize,
                            );
                        *fresh2 += j2 as libc::c_double * Fij;
                        k = 0 as libc::c_int;
                        while k < K {
                            let ref mut fresh3 = *M
                                .offset(
                                    (i1 + N1 * i2 + N1 * N2 * (k + 2 as libc::c_int)) as isize,
                                );
                            *fresh3
                                += *I.offset((j1 + N1 * j2 + N1 * N2 * k) as isize) * Fij;
                            k += 1;
                        }
                    }
                    j1 += 1;
                }
                j2 += 1;
            }
            i1 += 1;
        }
        i2 += 1;
    }
    if (*q).medoid != 0 {
        i2 = 0 as libc::c_int;
        while i2 < N2 {
            i1 = 0 as libc::c_int;
            while i1 < N1 {
                let mut sc_best: vl_qs_type = 0 as libc::c_int as vl_qs_type;
                let mut j1_best: vl_qs_type = i1 as vl_qs_type;
                let mut j2_best: vl_qs_type = i2 as vl_qs_type;
                let mut j1min_0: libc::c_int = if i1 - R > 0 as libc::c_int {
                    i1 - R
                } else {
                    0 as libc::c_int
                };
                let mut j1max_0: libc::c_int = if i1 + R < N1 - 1 as libc::c_int {
                    i1 + R
                } else {
                    N1 - 1 as libc::c_int
                };
                let mut j2min_0: libc::c_int = if i2 - R > 0 as libc::c_int {
                    i2 - R
                } else {
                    0 as libc::c_int
                };
                let mut j2max_0: libc::c_int = if i2 + R < N2 - 1 as libc::c_int {
                    i2 + R
                } else {
                    N2 - 1 as libc::c_int
                };
                j2 = j2min_0;
                while j2 <= j2max_0 {
                    j1 = j1min_0;
                    while j1 <= j1max_0 {
                        let mut Qij: vl_qs_type = -*n.offset((j1 + j2 * N1) as isize)
                            * *E.offset((i1 + i2 * N1) as isize);
                        let mut k_0: libc::c_int = 0;
                        Qij
                            -= (2 as libc::c_int * j1) as libc::c_double
                                * *M
                                    .offset(
                                        (i1 + i2 * N1 + N1 * N2 * 0 as libc::c_int) as isize,
                                    );
                        Qij
                            -= (2 as libc::c_int * j2) as libc::c_double
                                * *M
                                    .offset(
                                        (i1 + i2 * N1 + N1 * N2 * 1 as libc::c_int) as isize,
                                    );
                        k_0 = 0 as libc::c_int;
                        while k_0 < K {
                            Qij
                                -= 2 as libc::c_int as libc::c_double
                                    * *I.offset((j1 + j2 * N1 + N1 * N2 * k_0) as isize)
                                    * *M
                                        .offset(
                                            (i1 + i2 * N1 + N1 * N2 * (k_0 + 2 as libc::c_int)) as isize,
                                        );
                            k_0 += 1;
                        }
                        if Qij > sc_best {
                            sc_best = Qij;
                            j1_best = j1 as vl_qs_type;
                            j2_best = j2 as vl_qs_type;
                        }
                        j1 += 1;
                    }
                    j2 += 1;
                }
                *parents
                    .offset(
                        (i1 + N1 * i2) as isize,
                    ) = (j1_best + N1 as libc::c_double * j2_best) as libc::c_int;
                *dists.offset((i1 + N1 * i2) as isize) = sc_best;
                i1 += 1;
            }
            i2 += 1;
        }
    } else {
        i2 = 0 as libc::c_int;
        while i2 < N2 {
            i1 = 0 as libc::c_int;
            while i1 < N1 {
                let mut E0: vl_qs_type = *E.offset((i1 + N1 * i2) as isize);
                let mut d_best: vl_qs_type = vl_infinity_d.value;
                let mut j1_best_0: vl_qs_type = i1 as vl_qs_type;
                let mut j2_best_0: vl_qs_type = i2 as vl_qs_type;
                let mut j1min_1: libc::c_int = if i1 - tR > 0 as libc::c_int {
                    i1 - tR
                } else {
                    0 as libc::c_int
                };
                let mut j1max_1: libc::c_int = if i1 + tR < N1 - 1 as libc::c_int {
                    i1 + tR
                } else {
                    N1 - 1 as libc::c_int
                };
                let mut j2min_1: libc::c_int = if i2 - tR > 0 as libc::c_int {
                    i2 - tR
                } else {
                    0 as libc::c_int
                };
                let mut j2max_1: libc::c_int = if i2 + tR < N2 - 1 as libc::c_int {
                    i2 + tR
                } else {
                    N2 - 1 as libc::c_int
                };
                j2 = j2min_1;
                while j2 <= j2max_1 {
                    j1 = j1min_1;
                    while j1 <= j1max_1 {
                        if *E.offset((j1 + N1 * j2) as isize) > E0 {
                            let mut Dij_0: vl_qs_type = vl_quickshift_distance(
                                I,
                                N1,
                                N2,
                                K,
                                i1,
                                i2,
                                j1,
                                j2,
                            );
                            if Dij_0 <= tau2 && Dij_0 < d_best {
                                d_best = Dij_0;
                                j1_best_0 = j1 as vl_qs_type;
                                j2_best_0 = j2 as vl_qs_type;
                            }
                        }
                        j1 += 1;
                    }
                    j2 += 1;
                }
                *parents
                    .offset(
                        (i1 + N1 * i2) as isize,
                    ) = (j1_best_0 + N1 as libc::c_double * j2_best_0) as libc::c_int;
                *dists.offset((i1 + N1 * i2) as isize) = sqrt(d_best);
                i1 += 1;
            }
            i2 += 1;
        }
    }
    if !M.is_null() {
        vl_free(M as *mut libc::c_void);
    }
    if !n.is_null() {
        vl_free(n as *mut libc::c_void);
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_quickshift_delete(mut q: *mut VlQS) {
    if !q.is_null() {
        if !((*q).parents).is_null() {
            vl_free((*q).parents as *mut libc::c_void);
        }
        if !((*q).dists).is_null() {
            vl_free((*q).dists as *mut libc::c_void);
        }
        if !((*q).density).is_null() {
            vl_free((*q).density as *mut libc::c_void);
        }
        vl_free(q as *mut libc::c_void);
    }
}
