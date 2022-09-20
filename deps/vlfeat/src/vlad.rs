use ::libc;
extern "C" {
    fn abort() -> !;
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
pub type vl_uindex = vl_uint64;
pub type vl_type = vl_uint32;
unsafe extern "C" fn _vl_vlad_encode_f(
    mut enc: *mut libc::c_float,
    mut means: *const libc::c_float,
    mut dimension: vl_size,
    mut numClusters: vl_size,
    mut data: *const libc::c_float,
    mut numData: vl_size,
    mut assignments: *const libc::c_float,
    mut flags: libc::c_int,
) {
    let mut dim: vl_uindex = 0;
    let mut i_cl: vl_index = 0;
    let mut i_d: vl_index = 0;
    memset(
        enc as *mut libc::c_void,
        0 as libc::c_int,
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(dimension)
            .wrapping_mul(numClusters) as libc::c_ulong,
    );
    i_cl = 0 as libc::c_int as vl_index;
    while i_cl < numClusters as libc::c_int as libc::c_longlong {
        let mut clusterMass: libc::c_double = 0 as libc::c_int as libc::c_double;
        i_d = 0 as libc::c_int as vl_index;
        while i_d < numData as libc::c_int as libc::c_longlong {
            if *assignments
                .offset(
                    (i_d as libc::c_ulonglong)
                        .wrapping_mul(numClusters)
                        .wrapping_add(i_cl as libc::c_ulonglong) as isize,
                ) > 0 as libc::c_int as libc::c_float
            {
                let mut q: libc::c_double = *assignments
                    .offset(
                        (i_d as libc::c_ulonglong)
                            .wrapping_mul(numClusters)
                            .wrapping_add(i_cl as libc::c_ulonglong) as isize,
                    ) as libc::c_double;
                clusterMass += q;
                dim = 0 as libc::c_int as vl_uindex;
                while dim < dimension {
                    let ref mut fresh0 = *enc
                        .offset(
                            (i_cl as libc::c_ulonglong)
                                .wrapping_mul(dimension)
                                .wrapping_add(dim) as isize,
                        );
                    *fresh0 = (*fresh0 as libc::c_double
                        + q
                            * *data
                                .offset(
                                    (i_d as libc::c_ulonglong)
                                        .wrapping_mul(dimension)
                                        .wrapping_add(dim) as isize,
                                ) as libc::c_double) as libc::c_float;
                    dim = dim.wrapping_add(1);
                }
            }
            i_d += 1;
        }
        if clusterMass > 0 as libc::c_int as libc::c_double {
            if flags & (0x1 as libc::c_int) << 3 as libc::c_int != 0 {
                dim = 0 as libc::c_int as vl_uindex;
                while dim < dimension {
                    let ref mut fresh1 = *enc
                        .offset(
                            (i_cl as libc::c_ulonglong)
                                .wrapping_mul(dimension)
                                .wrapping_add(dim) as isize,
                        );
                    *fresh1 = (*fresh1 as libc::c_double / clusterMass) as libc::c_float;
                    *enc
                        .offset(
                            (i_cl as libc::c_ulonglong)
                                .wrapping_mul(dimension)
                                .wrapping_add(dim) as isize,
                        )
                        -= *means
                            .offset(
                                (i_cl as libc::c_ulonglong)
                                    .wrapping_mul(dimension)
                                    .wrapping_add(dim) as isize,
                            );
                    dim = dim.wrapping_add(1);
                }
            } else {
                dim = 0 as libc::c_int as vl_uindex;
                while dim < dimension {
                    let ref mut fresh2 = *enc
                        .offset(
                            (i_cl as libc::c_ulonglong)
                                .wrapping_mul(dimension)
                                .wrapping_add(dim) as isize,
                        );
                    *fresh2 = (*fresh2 as libc::c_double
                        - clusterMass
                            * *means
                                .offset(
                                    (i_cl as libc::c_ulonglong)
                                        .wrapping_mul(dimension)
                                        .wrapping_add(dim) as isize,
                                ) as libc::c_double) as libc::c_float;
                    dim = dim.wrapping_add(1);
                }
            }
        }
        if flags & (0x1 as libc::c_int) << 1 as libc::c_int != 0 {
            dim = 0 as libc::c_int as vl_uindex;
            while dim < dimension {
                let mut z: libc::c_float = *enc
                    .offset(
                        (i_cl as libc::c_ulonglong)
                            .wrapping_mul(dimension)
                            .wrapping_add(dim) as isize,
                    );
                if z >= 0 as libc::c_int as libc::c_float {
                    *enc
                        .offset(
                            (i_cl as libc::c_ulonglong)
                                .wrapping_mul(dimension)
                                .wrapping_add(dim) as isize,
                        ) = f32::sqrt(z);
                } else {
                    *enc
                        .offset(
                            (i_cl as libc::c_ulonglong)
                                .wrapping_mul(dimension)
                                .wrapping_add(dim) as isize,
                        ) = -f32::sqrt(-z);
                }
                dim = dim.wrapping_add(1);
            }
        }
        if flags & (0x1 as libc::c_int) << 0 as libc::c_int != 0 {
            let mut n: libc::c_float = 0 as libc::c_int as libc::c_float;
            dim = 0 as libc::c_int as vl_uindex;
            dim = 0 as libc::c_int as vl_uindex;
            while dim < dimension {
                let mut z_0: libc::c_float = *enc
                    .offset(
                        (i_cl as libc::c_ulonglong)
                            .wrapping_mul(dimension)
                            .wrapping_add(dim) as isize,
                    );
                n += z_0 * z_0;
                dim = dim.wrapping_add(1);
            }
            n = f32::sqrt(n);
            n = (if n as libc::c_double > 1e-12f64 {
                n as libc::c_double
            } else {
                1e-12f64
            }) as libc::c_float;
            dim = 0 as libc::c_int as vl_uindex;
            while dim < dimension {
                *enc
                    .offset(
                        (i_cl as libc::c_ulonglong)
                            .wrapping_mul(dimension)
                            .wrapping_add(dim) as isize,
                    ) /= n;
                dim = dim.wrapping_add(1);
            }
        }
        i_cl += 1;
    }
    if flags & (0x1 as libc::c_int) << 2 as libc::c_int == 0 {
        let mut n_0: libc::c_float = 0 as libc::c_int as libc::c_float;
        dim = 0 as libc::c_int as vl_uindex;
        while dim < dimension.wrapping_mul(numClusters) {
            let mut z_1: libc::c_float = *enc.offset(dim as isize);
            n_0 += z_1 * z_1;
            dim = dim.wrapping_add(1);
        }
        n_0 = f32::sqrt(n_0);
        n_0 = (if n_0 as libc::c_double > 1e-12f64 {
            n_0 as libc::c_double
        } else {
            1e-12f64
        }) as libc::c_float;
        dim = 0 as libc::c_int as vl_uindex;
        while dim < dimension.wrapping_mul(numClusters) {
            *enc.offset(dim as isize) /= n_0;
            dim = dim.wrapping_add(1);
        }
    }
}
unsafe extern "C" fn _vl_vlad_encode_d(
    mut enc: *mut libc::c_double,
    mut means: *const libc::c_double,
    mut dimension: vl_size,
    mut numClusters: vl_size,
    mut data: *const libc::c_double,
    mut numData: vl_size,
    mut assignments: *const libc::c_double,
    mut flags: libc::c_int,
) {
    let mut dim: vl_uindex = 0;
    let mut i_cl: vl_index = 0;
    let mut i_d: vl_index = 0;
    memset(
        enc as *mut libc::c_void,
        0 as libc::c_int,
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(dimension)
            .wrapping_mul(numClusters) as libc::c_ulong,
    );
    i_cl = 0 as libc::c_int as vl_index;
    while i_cl < numClusters as libc::c_int as libc::c_longlong {
        let mut clusterMass: libc::c_double = 0 as libc::c_int as libc::c_double;
        i_d = 0 as libc::c_int as vl_index;
        while i_d < numData as libc::c_int as libc::c_longlong {
            if *assignments
                .offset(
                    (i_d as libc::c_ulonglong)
                        .wrapping_mul(numClusters)
                        .wrapping_add(i_cl as libc::c_ulonglong) as isize,
                ) > 0 as libc::c_int as libc::c_double
            {
                let mut q: libc::c_double = *assignments
                    .offset(
                        (i_d as libc::c_ulonglong)
                            .wrapping_mul(numClusters)
                            .wrapping_add(i_cl as libc::c_ulonglong) as isize,
                    );
                clusterMass += q;
                dim = 0 as libc::c_int as vl_uindex;
                while dim < dimension {
                    *enc
                        .offset(
                            (i_cl as libc::c_ulonglong)
                                .wrapping_mul(dimension)
                                .wrapping_add(dim) as isize,
                        )
                        += q
                            * *data
                                .offset(
                                    (i_d as libc::c_ulonglong)
                                        .wrapping_mul(dimension)
                                        .wrapping_add(dim) as isize,
                                );
                    dim = dim.wrapping_add(1);
                }
            }
            i_d += 1;
        }
        if clusterMass > 0 as libc::c_int as libc::c_double {
            if flags & (0x1 as libc::c_int) << 3 as libc::c_int != 0 {
                dim = 0 as libc::c_int as vl_uindex;
                while dim < dimension {
                    *enc
                        .offset(
                            (i_cl as libc::c_ulonglong)
                                .wrapping_mul(dimension)
                                .wrapping_add(dim) as isize,
                        ) /= clusterMass;
                    *enc
                        .offset(
                            (i_cl as libc::c_ulonglong)
                                .wrapping_mul(dimension)
                                .wrapping_add(dim) as isize,
                        )
                        -= *means
                            .offset(
                                (i_cl as libc::c_ulonglong)
                                    .wrapping_mul(dimension)
                                    .wrapping_add(dim) as isize,
                            );
                    dim = dim.wrapping_add(1);
                }
            } else {
                dim = 0 as libc::c_int as vl_uindex;
                while dim < dimension {
                    *enc
                        .offset(
                            (i_cl as libc::c_ulonglong)
                                .wrapping_mul(dimension)
                                .wrapping_add(dim) as isize,
                        )
                        -= clusterMass
                            * *means
                                .offset(
                                    (i_cl as libc::c_ulonglong)
                                        .wrapping_mul(dimension)
                                        .wrapping_add(dim) as isize,
                                );
                    dim = dim.wrapping_add(1);
                }
            }
        }
        if flags & (0x1 as libc::c_int) << 1 as libc::c_int != 0 {
            dim = 0 as libc::c_int as vl_uindex;
            while dim < dimension {
                let mut z: libc::c_double = *enc
                    .offset(
                        (i_cl as libc::c_ulonglong)
                            .wrapping_mul(dimension)
                            .wrapping_add(dim) as isize,
                    );
                if z >= 0 as libc::c_int as libc::c_double {
                    *enc
                        .offset(
                            (i_cl as libc::c_ulonglong)
                                .wrapping_mul(dimension)
                                .wrapping_add(dim) as isize,
                        ) = f64::sqrt(z);
                } else {
                    *enc
                        .offset(
                            (i_cl as libc::c_ulonglong)
                                .wrapping_mul(dimension)
                                .wrapping_add(dim) as isize,
                        ) = -f64::sqrt(-z);
                }
                dim = dim.wrapping_add(1);
            }
        }
        if flags & (0x1 as libc::c_int) << 0 as libc::c_int != 0 {
            let mut n: libc::c_double = 0 as libc::c_int as libc::c_double;
            dim = 0 as libc::c_int as vl_uindex;
            dim = 0 as libc::c_int as vl_uindex;
            while dim < dimension {
                let mut z_0: libc::c_double = *enc
                    .offset(
                        (i_cl as libc::c_ulonglong)
                            .wrapping_mul(dimension)
                            .wrapping_add(dim) as isize,
                    );
                n += z_0 * z_0;
                dim = dim.wrapping_add(1);
            }
            n = f64::sqrt(n);
            n = if n > 1e-12f64 { n } else { 1e-12f64 };
            dim = 0 as libc::c_int as vl_uindex;
            while dim < dimension {
                *enc
                    .offset(
                        (i_cl as libc::c_ulonglong)
                            .wrapping_mul(dimension)
                            .wrapping_add(dim) as isize,
                    ) /= n;
                dim = dim.wrapping_add(1);
            }
        }
        i_cl += 1;
    }
    if flags & (0x1 as libc::c_int) << 2 as libc::c_int == 0 {
        let mut n_0: libc::c_double = 0 as libc::c_int as libc::c_double;
        dim = 0 as libc::c_int as vl_uindex;
        while dim < dimension.wrapping_mul(numClusters) {
            let mut z_1: libc::c_double = *enc.offset(dim as isize);
            n_0 += z_1 * z_1;
            dim = dim.wrapping_add(1);
        }
        n_0 = f64::sqrt(n_0);
        n_0 = if n_0 > 1e-12f64 { n_0 } else { 1e-12f64 };
        dim = 0 as libc::c_int as vl_uindex;
        while dim < dimension.wrapping_mul(numClusters) {
            *enc.offset(dim as isize) /= n_0;
            dim = dim.wrapping_add(1);
        }
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_vlad_encode(
    mut enc: *mut libc::c_void,
    mut dataType: vl_type,
    mut means: *const libc::c_void,
    mut dimension: vl_size,
    mut numClusters: vl_size,
    mut data: *const libc::c_void,
    mut numData: vl_size,
    mut assignments: *const libc::c_void,
    mut flags: libc::c_int,
) {
    match dataType {
        1 => {
            _vl_vlad_encode_f(
                enc as *mut libc::c_float,
                means as *const libc::c_float,
                dimension,
                numClusters,
                data as *const libc::c_float,
                numData,
                assignments as *const libc::c_float,
                flags,
            );
        }
        2 => {
            _vl_vlad_encode_d(
                enc as *mut libc::c_double,
                means as *const libc::c_double,
                dimension,
                numClusters,
                data as *const libc::c_double,
                numData,
                assignments as *const libc::c_double,
                flags,
            );
        }
        _ => {
            abort();
        }
    };
}
