use ::libc;
extern "C" {
    fn acos(_: libc::c_double) -> libc::c_double;
    fn cos(_: libc::c_double) -> libc::c_double;
    fn sin(_: libc::c_double) -> libc::c_double;
    fn sqrt(_: libc::c_double) -> libc::c_double;
    fn fabs(_: libc::c_double) -> libc::c_double;
}
pub type vl_uint64 = libc::c_ulonglong;
#[derive(Copy, Clone)]
#[repr(C)]
pub union C2RustUnnamed {
    pub raw: vl_uint64,
    pub value: libc::c_double,
}
static mut vl_nan_d: C2RustUnnamed = C2RustUnnamed {
    raw: 0x7ff8000000000000 as libc::c_ulonglong,
};
#[no_mangle]
pub unsafe extern "C" fn vl_rodrigues(
    mut R_pt: *mut libc::c_double,
    mut dR_pt: *mut libc::c_double,
    mut om_pt: *const libc::c_double,
) {
    let small: libc::c_double = 1e-6f64;
    let mut th: libc::c_double = sqrt(
        *om_pt.offset(0 as libc::c_int as isize)
            * *om_pt.offset(0 as libc::c_int as isize)
            + *om_pt.offset(1 as libc::c_int as isize)
                * *om_pt.offset(1 as libc::c_int as isize)
            + *om_pt.offset(2 as libc::c_int as isize)
                * *om_pt.offset(2 as libc::c_int as isize),
    );
    if th < small {
        *R_pt
            .offset(
                (0 as libc::c_int + 3 as libc::c_int * 0 as libc::c_int) as isize,
            ) = 1.0f64;
        *R_pt
            .offset(
                (0 as libc::c_int + 3 as libc::c_int * 1 as libc::c_int) as isize,
            ) = 0.0f64;
        *R_pt
            .offset(
                (0 as libc::c_int + 3 as libc::c_int * 2 as libc::c_int) as isize,
            ) = 0.0f64;
        *R_pt
            .offset(
                (1 as libc::c_int + 3 as libc::c_int * 0 as libc::c_int) as isize,
            ) = 0.0f64;
        *R_pt
            .offset(
                (1 as libc::c_int + 3 as libc::c_int * 1 as libc::c_int) as isize,
            ) = 1.0f64;
        *R_pt
            .offset(
                (1 as libc::c_int + 3 as libc::c_int * 2 as libc::c_int) as isize,
            ) = 0.0f64;
        *R_pt
            .offset(
                (2 as libc::c_int + 3 as libc::c_int * 0 as libc::c_int) as isize,
            ) = 0.0f64;
        *R_pt
            .offset(
                (2 as libc::c_int + 3 as libc::c_int * 1 as libc::c_int) as isize,
            ) = 0.0f64;
        *R_pt
            .offset(
                (2 as libc::c_int + 3 as libc::c_int * 2 as libc::c_int) as isize,
            ) = 1.0f64;
        if !dR_pt.is_null() {
            *dR_pt
                .offset(
                    (0 as libc::c_int + 9 as libc::c_int * 0 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dR_pt
                .offset(
                    (0 as libc::c_int + 9 as libc::c_int * 1 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dR_pt
                .offset(
                    (0 as libc::c_int + 9 as libc::c_int * 2 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dR_pt
                .offset(
                    (1 as libc::c_int + 9 as libc::c_int * 0 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dR_pt
                .offset(
                    (1 as libc::c_int + 9 as libc::c_int * 1 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dR_pt
                .offset(
                    (1 as libc::c_int + 9 as libc::c_int * 2 as libc::c_int) as isize,
                ) = 1 as libc::c_int as libc::c_double;
            *dR_pt
                .offset(
                    (2 as libc::c_int + 9 as libc::c_int * 0 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dR_pt
                .offset(
                    (2 as libc::c_int + 9 as libc::c_int * 1 as libc::c_int) as isize,
                ) = -(1 as libc::c_int) as libc::c_double;
            *dR_pt
                .offset(
                    (2 as libc::c_int + 9 as libc::c_int * 2 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dR_pt
                .offset(
                    (3 as libc::c_int + 9 as libc::c_int * 0 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dR_pt
                .offset(
                    (3 as libc::c_int + 9 as libc::c_int * 1 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dR_pt
                .offset(
                    (3 as libc::c_int + 9 as libc::c_int * 2 as libc::c_int) as isize,
                ) = -(1 as libc::c_int) as libc::c_double;
            *dR_pt
                .offset(
                    (4 as libc::c_int + 9 as libc::c_int * 0 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dR_pt
                .offset(
                    (4 as libc::c_int + 9 as libc::c_int * 1 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dR_pt
                .offset(
                    (4 as libc::c_int + 9 as libc::c_int * 2 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dR_pt
                .offset(
                    (5 as libc::c_int + 9 as libc::c_int * 0 as libc::c_int) as isize,
                ) = 1 as libc::c_int as libc::c_double;
            *dR_pt
                .offset(
                    (5 as libc::c_int + 9 as libc::c_int * 1 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dR_pt
                .offset(
                    (5 as libc::c_int + 9 as libc::c_int * 2 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dR_pt
                .offset(
                    (6 as libc::c_int + 9 as libc::c_int * 0 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dR_pt
                .offset(
                    (6 as libc::c_int + 9 as libc::c_int * 1 as libc::c_int) as isize,
                ) = 1 as libc::c_int as libc::c_double;
            *dR_pt
                .offset(
                    (6 as libc::c_int + 9 as libc::c_int * 2 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dR_pt
                .offset(
                    (7 as libc::c_int + 9 as libc::c_int * 0 as libc::c_int) as isize,
                ) = -(1 as libc::c_int) as libc::c_double;
            *dR_pt
                .offset(
                    (7 as libc::c_int + 9 as libc::c_int * 1 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dR_pt
                .offset(
                    (7 as libc::c_int + 9 as libc::c_int * 2 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dR_pt
                .offset(
                    (8 as libc::c_int + 9 as libc::c_int * 0 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dR_pt
                .offset(
                    (8 as libc::c_int + 9 as libc::c_int * 1 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dR_pt
                .offset(
                    (8 as libc::c_int + 9 as libc::c_int * 2 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
        }
        return;
    }
    let mut x: libc::c_double = *om_pt.offset(0 as libc::c_int as isize) / th;
    let mut y: libc::c_double = *om_pt.offset(1 as libc::c_int as isize) / th;
    let mut z: libc::c_double = *om_pt.offset(2 as libc::c_int as isize) / th;
    let mut xx: libc::c_double = x * x;
    let mut xy: libc::c_double = x * y;
    let mut xz: libc::c_double = x * z;
    let mut yy: libc::c_double = y * y;
    let mut yz: libc::c_double = y * z;
    let mut zz: libc::c_double = z * z;
    let yx: libc::c_double = xy;
    let zx: libc::c_double = xz;
    let zy: libc::c_double = yz;
    let mut sth: libc::c_double = sin(th);
    let mut cth: libc::c_double = cos(th);
    let mut mcth: libc::c_double = 1.0f64 - cth;
    *R_pt
        .offset(
            (0 as libc::c_int + 3 as libc::c_int * 0 as libc::c_int) as isize,
        ) = 1 as libc::c_int as libc::c_double - mcth * (yy + zz);
    *R_pt
        .offset(
            (1 as libc::c_int + 3 as libc::c_int * 0 as libc::c_int) as isize,
        ) = sth * z + mcth * xy;
    *R_pt
        .offset(
            (2 as libc::c_int + 3 as libc::c_int * 0 as libc::c_int) as isize,
        ) = -sth * y + mcth * xz;
    *R_pt
        .offset(
            (0 as libc::c_int + 3 as libc::c_int * 1 as libc::c_int) as isize,
        ) = -sth * z + mcth * yx;
    *R_pt
        .offset(
            (1 as libc::c_int + 3 as libc::c_int * 1 as libc::c_int) as isize,
        ) = 1 as libc::c_int as libc::c_double - mcth * (zz + xx);
    *R_pt
        .offset(
            (2 as libc::c_int + 3 as libc::c_int * 1 as libc::c_int) as isize,
        ) = sth * x + mcth * yz;
    *R_pt
        .offset(
            (0 as libc::c_int + 3 as libc::c_int * 2 as libc::c_int) as isize,
        ) = sth * y + mcth * xz;
    *R_pt
        .offset(
            (1 as libc::c_int + 3 as libc::c_int * 2 as libc::c_int) as isize,
        ) = -sth * x + mcth * yz;
    *R_pt
        .offset(
            (2 as libc::c_int + 3 as libc::c_int * 2 as libc::c_int) as isize,
        ) = 1 as libc::c_int as libc::c_double - mcth * (xx + yy);
    if !dR_pt.is_null() {
        let mut a: libc::c_double = sth / th;
        let mut b: libc::c_double = mcth / th;
        let mut c: libc::c_double = cth - a;
        let mut d: libc::c_double = sth - 2 as libc::c_int as libc::c_double * b;
        *dR_pt
            .offset(
                (0 as libc::c_int + 9 as libc::c_int * 0 as libc::c_int) as isize,
            ) = -d * (yy + zz) * x;
        *dR_pt
            .offset(
                (1 as libc::c_int + 9 as libc::c_int * 0 as libc::c_int) as isize,
            ) = b * y + c * zx + d * xy * x;
        *dR_pt
            .offset(
                (2 as libc::c_int + 9 as libc::c_int * 0 as libc::c_int) as isize,
            ) = b * z - c * yx + d * xz * x;
        *dR_pt
            .offset(
                (3 as libc::c_int + 9 as libc::c_int * 0 as libc::c_int) as isize,
            ) = b * y - c * zx + d * xy * x;
        *dR_pt
            .offset(
                (4 as libc::c_int + 9 as libc::c_int * 0 as libc::c_int) as isize,
            ) = -(2 as libc::c_int) as libc::c_double * b * x - d * (zz + xx) * x;
        *dR_pt
            .offset(
                (5 as libc::c_int + 9 as libc::c_int * 0 as libc::c_int) as isize,
            ) = a + c * xx + d * yz * x;
        *dR_pt
            .offset(
                (6 as libc::c_int + 9 as libc::c_int * 0 as libc::c_int) as isize,
            ) = b * z + c * yx + d * zx * x;
        *dR_pt
            .offset(
                (7 as libc::c_int + 9 as libc::c_int * 0 as libc::c_int) as isize,
            ) = -a - c * xx + d * zy * x;
        *dR_pt
            .offset(
                (8 as libc::c_int + 9 as libc::c_int * 0 as libc::c_int) as isize,
            ) = -(2 as libc::c_int) as libc::c_double * b * x - d * (yy + xx) * x;
        *dR_pt
            .offset(
                (0 as libc::c_int + 9 as libc::c_int * 1 as libc::c_int) as isize,
            ) = -(2 as libc::c_int) as libc::c_double * b * y - d * (yy + zz) * y;
        *dR_pt
            .offset(
                (1 as libc::c_int + 9 as libc::c_int * 1 as libc::c_int) as isize,
            ) = b * x + c * zy + d * xy * y;
        *dR_pt
            .offset(
                (2 as libc::c_int + 9 as libc::c_int * 1 as libc::c_int) as isize,
            ) = -a - c * yy + d * xz * y;
        *dR_pt
            .offset(
                (3 as libc::c_int + 9 as libc::c_int * 1 as libc::c_int) as isize,
            ) = b * x - c * zy + d * xy * y;
        *dR_pt
            .offset(
                (4 as libc::c_int + 9 as libc::c_int * 1 as libc::c_int) as isize,
            ) = -d * (zz + xx) * y;
        *dR_pt
            .offset(
                (5 as libc::c_int + 9 as libc::c_int * 1 as libc::c_int) as isize,
            ) = b * z + c * xy + d * yz * y;
        *dR_pt
            .offset(
                (6 as libc::c_int + 9 as libc::c_int * 1 as libc::c_int) as isize,
            ) = a + c * yy + d * zx * y;
        *dR_pt
            .offset(
                (7 as libc::c_int + 9 as libc::c_int * 1 as libc::c_int) as isize,
            ) = b * z - c * xy + d * zy * y;
        *dR_pt
            .offset(
                (8 as libc::c_int + 9 as libc::c_int * 1 as libc::c_int) as isize,
            ) = -(2 as libc::c_int) as libc::c_double * b * y - d * (yy + xx) * y;
        *dR_pt
            .offset(
                (0 as libc::c_int + 9 as libc::c_int * 2 as libc::c_int) as isize,
            ) = -(2 as libc::c_int) as libc::c_double * b * z - d * (yy + zz) * z;
        *dR_pt
            .offset(
                (1 as libc::c_int + 9 as libc::c_int * 2 as libc::c_int) as isize,
            ) = a + c * zz + d * xy * z;
        *dR_pt
            .offset(
                (2 as libc::c_int + 9 as libc::c_int * 2 as libc::c_int) as isize,
            ) = b * x - c * yz + d * xz * z;
        *dR_pt
            .offset(
                (3 as libc::c_int + 9 as libc::c_int * 2 as libc::c_int) as isize,
            ) = -a - c * zz + d * xy * z;
        *dR_pt
            .offset(
                (4 as libc::c_int + 9 as libc::c_int * 2 as libc::c_int) as isize,
            ) = -(2 as libc::c_int) as libc::c_double * b * z - d * (zz + xx) * z;
        *dR_pt
            .offset(
                (5 as libc::c_int + 9 as libc::c_int * 2 as libc::c_int) as isize,
            ) = b * y + c * xz + d * yz * z;
        *dR_pt
            .offset(
                (6 as libc::c_int + 9 as libc::c_int * 2 as libc::c_int) as isize,
            ) = b * x + c * yz + d * zx * z;
        *dR_pt
            .offset(
                (7 as libc::c_int + 9 as libc::c_int * 2 as libc::c_int) as isize,
            ) = b * y - c * xz + d * zy * z;
        *dR_pt
            .offset(
                (8 as libc::c_int + 9 as libc::c_int * 2 as libc::c_int) as isize,
            ) = -d * (yy + xx) * z;
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_irodrigues(
    mut om_pt: *mut libc::c_double,
    mut dom_pt: *mut libc::c_double,
    mut R_pt: *const libc::c_double,
) {
    let small: libc::c_double = 1e-6f64;
    let mut th: libc::c_double = acos(
        0.5f64
            * ((if *R_pt
                .offset(
                    (0 as libc::c_int + 3 as libc::c_int * 0 as libc::c_int) as isize,
                )
                + *R_pt
                    .offset(
                        (1 as libc::c_int + 3 as libc::c_int * 1 as libc::c_int) as isize,
                    )
                + *R_pt
                    .offset(
                        (2 as libc::c_int + 3 as libc::c_int * 2 as libc::c_int) as isize,
                    ) > -1.0f64
            {
                *R_pt
                    .offset(
                        (0 as libc::c_int + 3 as libc::c_int * 0 as libc::c_int) as isize,
                    )
                    + *R_pt
                        .offset(
                            (1 as libc::c_int + 3 as libc::c_int * 1 as libc::c_int)
                                as isize,
                        )
                    + *R_pt
                        .offset(
                            (2 as libc::c_int + 3 as libc::c_int * 2 as libc::c_int)
                                as isize,
                        )
            } else {
                -1.0f64
            }) - 1.0f64),
    );
    let mut sth: libc::c_double = sin(th);
    let mut cth: libc::c_double = cos(th);
    if fabs(sth) < small && cth < 0 as libc::c_int as libc::c_double {
        let mut W_pt: [libc::c_double; 9] = [0.; 9];
        let mut x: libc::c_double = 0.;
        let mut y: libc::c_double = 0.;
        let mut z: libc::c_double = 0.;
        W_pt[0 as libc::c_int
            as usize] = 0.5f64
            * (*R_pt
                .offset(
                    (0 as libc::c_int + 3 as libc::c_int * 0 as libc::c_int) as isize,
                )
                + *R_pt
                    .offset(
                        (0 as libc::c_int + 3 as libc::c_int * 0 as libc::c_int) as isize,
                    )) - 1.0f64;
        W_pt[1 as libc::c_int
            as usize] = 0.5f64
            * (*R_pt
                .offset(
                    (1 as libc::c_int + 3 as libc::c_int * 0 as libc::c_int) as isize,
                )
                + *R_pt
                    .offset(
                        (0 as libc::c_int + 3 as libc::c_int * 1 as libc::c_int) as isize,
                    ));
        W_pt[2 as libc::c_int
            as usize] = 0.5f64
            * (*R_pt
                .offset(
                    (2 as libc::c_int + 3 as libc::c_int * 0 as libc::c_int) as isize,
                )
                + *R_pt
                    .offset(
                        (0 as libc::c_int + 3 as libc::c_int * 2 as libc::c_int) as isize,
                    ));
        W_pt[3 as libc::c_int
            as usize] = 0.5f64
            * (*R_pt
                .offset(
                    (0 as libc::c_int + 3 as libc::c_int * 1 as libc::c_int) as isize,
                )
                + *R_pt
                    .offset(
                        (1 as libc::c_int + 3 as libc::c_int * 0 as libc::c_int) as isize,
                    ));
        W_pt[4 as libc::c_int
            as usize] = 0.5f64
            * (*R_pt
                .offset(
                    (1 as libc::c_int + 3 as libc::c_int * 1 as libc::c_int) as isize,
                )
                + *R_pt
                    .offset(
                        (1 as libc::c_int + 3 as libc::c_int * 1 as libc::c_int) as isize,
                    )) - 1.0f64;
        W_pt[5 as libc::c_int
            as usize] = 0.5f64
            * (*R_pt
                .offset(
                    (2 as libc::c_int + 3 as libc::c_int * 1 as libc::c_int) as isize,
                )
                + *R_pt
                    .offset(
                        (1 as libc::c_int + 3 as libc::c_int * 2 as libc::c_int) as isize,
                    ));
        W_pt[6 as libc::c_int
            as usize] = 0.5f64
            * (*R_pt
                .offset(
                    (0 as libc::c_int + 3 as libc::c_int * 2 as libc::c_int) as isize,
                )
                + *R_pt
                    .offset(
                        (2 as libc::c_int + 3 as libc::c_int * 0 as libc::c_int) as isize,
                    ));
        W_pt[7 as libc::c_int
            as usize] = 0.5f64
            * (*R_pt
                .offset(
                    (1 as libc::c_int + 3 as libc::c_int * 2 as libc::c_int) as isize,
                )
                + *R_pt
                    .offset(
                        (2 as libc::c_int + 3 as libc::c_int * 1 as libc::c_int) as isize,
                    ));
        W_pt[8 as libc::c_int
            as usize] = 0.5f64
            * (*R_pt
                .offset(
                    (2 as libc::c_int + 3 as libc::c_int * 2 as libc::c_int) as isize,
                )
                + *R_pt
                    .offset(
                        (2 as libc::c_int + 3 as libc::c_int * 2 as libc::c_int) as isize,
                    )) - 1.0f64;
        x = sqrt(
            0.5f64
                * (W_pt[(0 as libc::c_int + 3 as libc::c_int * 0 as libc::c_int)
                    as usize]
                    - W_pt[(1 as libc::c_int + 3 as libc::c_int * 1 as libc::c_int)
                        as usize]
                    - W_pt[(2 as libc::c_int + 3 as libc::c_int * 2 as libc::c_int)
                        as usize]),
        );
        y = sqrt(
            0.5f64
                * (W_pt[(1 as libc::c_int + 3 as libc::c_int * 1 as libc::c_int)
                    as usize]
                    - W_pt[(2 as libc::c_int + 3 as libc::c_int * 2 as libc::c_int)
                        as usize]
                    - W_pt[(0 as libc::c_int + 3 as libc::c_int * 0 as libc::c_int)
                        as usize]),
        );
        z = sqrt(
            0.5f64
                * (W_pt[(2 as libc::c_int + 3 as libc::c_int * 2 as libc::c_int)
                    as usize]
                    - W_pt[(0 as libc::c_int + 3 as libc::c_int * 0 as libc::c_int)
                        as usize]
                    - W_pt[(1 as libc::c_int + 3 as libc::c_int * 1 as libc::c_int)
                        as usize]),
        );
        if x >= y && x >= z {
            y = if W_pt[(1 as libc::c_int + 3 as libc::c_int * 0 as libc::c_int)
                as usize] >= 0 as libc::c_int as libc::c_double
            {
                y
            } else {
                -y
            };
            z = if W_pt[(2 as libc::c_int + 3 as libc::c_int * 0 as libc::c_int)
                as usize] >= 0 as libc::c_int as libc::c_double
            {
                z
            } else {
                -z
            };
        } else if y >= x && y >= z {
            z = if W_pt[(2 as libc::c_int + 3 as libc::c_int * 1 as libc::c_int)
                as usize] >= 0 as libc::c_int as libc::c_double
            {
                z
            } else {
                -z
            };
            x = if W_pt[(1 as libc::c_int + 3 as libc::c_int * 0 as libc::c_int)
                as usize] >= 0 as libc::c_int as libc::c_double
            {
                x
            } else {
                -x
            };
        } else {
            x = if W_pt[(2 as libc::c_int + 3 as libc::c_int * 0 as libc::c_int)
                as usize] >= 0 as libc::c_int as libc::c_double
            {
                x
            } else {
                -x
            };
            y = if W_pt[(2 as libc::c_int + 3 as libc::c_int * 1 as libc::c_int)
                as usize] >= 0 as libc::c_int as libc::c_double
            {
                y
            } else {
                -y
            };
        }
        let mut scale: libc::c_double = th
            / sqrt(1 as libc::c_int as libc::c_double - cth);
        *om_pt.offset(0 as libc::c_int as isize) = scale * x;
        *om_pt.offset(1 as libc::c_int as isize) = scale * y;
        *om_pt.offset(2 as libc::c_int as isize) = scale * z;
        if !dom_pt.is_null() {
            let mut k: libc::c_int = 0;
            k = 0 as libc::c_int;
            while k < 3 as libc::c_int * 9 as libc::c_int {
                *dom_pt.offset(k as isize) = vl_nan_d.value;
                k += 1;
            }
        }
        return;
    } else {
        let mut a: libc::c_double = if fabs(sth) < small {
            1 as libc::c_int as libc::c_double
        } else {
            th / sin(th)
        };
        let mut b: libc::c_double = 0.;
        *om_pt
            .offset(
                0 as libc::c_int as isize,
            ) = 0.5f64 * a
            * (*R_pt
                .offset(
                    (2 as libc::c_int + 3 as libc::c_int * 1 as libc::c_int) as isize,
                )
                - *R_pt
                    .offset(
                        (1 as libc::c_int + 3 as libc::c_int * 2 as libc::c_int) as isize,
                    ));
        *om_pt
            .offset(
                1 as libc::c_int as isize,
            ) = 0.5f64 * a
            * (*R_pt
                .offset(
                    (0 as libc::c_int + 3 as libc::c_int * 2 as libc::c_int) as isize,
                )
                - *R_pt
                    .offset(
                        (2 as libc::c_int + 3 as libc::c_int * 0 as libc::c_int) as isize,
                    ));
        *om_pt
            .offset(
                2 as libc::c_int as isize,
            ) = 0.5f64 * a
            * (*R_pt
                .offset(
                    (1 as libc::c_int + 3 as libc::c_int * 0 as libc::c_int) as isize,
                )
                - *R_pt
                    .offset(
                        (0 as libc::c_int + 3 as libc::c_int * 1 as libc::c_int) as isize,
                    ));
        if !dom_pt.is_null() {
            if fabs(sth) < small {
                a = 0.5f64;
                b = 0 as libc::c_int as libc::c_double;
            } else {
                a = th / (2 as libc::c_int as libc::c_double * sth);
                b = (th * cth - sth) / (2 as libc::c_int as libc::c_double * sth * sth)
                    / th;
            }
            *dom_pt
                .offset(
                    (0 as libc::c_int + 3 as libc::c_int * 0 as libc::c_int) as isize,
                ) = b * *om_pt.offset(0 as libc::c_int as isize);
            *dom_pt
                .offset(
                    (1 as libc::c_int + 3 as libc::c_int * 0 as libc::c_int) as isize,
                ) = b * *om_pt.offset(1 as libc::c_int as isize);
            *dom_pt
                .offset(
                    (2 as libc::c_int + 3 as libc::c_int * 0 as libc::c_int) as isize,
                ) = b * *om_pt.offset(2 as libc::c_int as isize);
            *dom_pt
                .offset(
                    (0 as libc::c_int + 3 as libc::c_int * 1 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dom_pt
                .offset(
                    (1 as libc::c_int + 3 as libc::c_int * 1 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dom_pt
                .offset(
                    (2 as libc::c_int + 3 as libc::c_int * 1 as libc::c_int) as isize,
                ) = a;
            *dom_pt
                .offset(
                    (0 as libc::c_int + 3 as libc::c_int * 2 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dom_pt
                .offset(
                    (1 as libc::c_int + 3 as libc::c_int * 2 as libc::c_int) as isize,
                ) = -a;
            *dom_pt
                .offset(
                    (2 as libc::c_int + 3 as libc::c_int * 2 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dom_pt
                .offset(
                    (0 as libc::c_int + 3 as libc::c_int * 3 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dom_pt
                .offset(
                    (1 as libc::c_int + 3 as libc::c_int * 3 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dom_pt
                .offset(
                    (2 as libc::c_int + 3 as libc::c_int * 3 as libc::c_int) as isize,
                ) = -a;
            *dom_pt
                .offset(
                    (0 as libc::c_int + 3 as libc::c_int * 4 as libc::c_int) as isize,
                ) = b * *om_pt.offset(0 as libc::c_int as isize);
            *dom_pt
                .offset(
                    (1 as libc::c_int + 3 as libc::c_int * 4 as libc::c_int) as isize,
                ) = b * *om_pt.offset(1 as libc::c_int as isize);
            *dom_pt
                .offset(
                    (2 as libc::c_int + 3 as libc::c_int * 4 as libc::c_int) as isize,
                ) = b * *om_pt.offset(2 as libc::c_int as isize);
            *dom_pt
                .offset(
                    (0 as libc::c_int + 3 as libc::c_int * 5 as libc::c_int) as isize,
                ) = a;
            *dom_pt
                .offset(
                    (1 as libc::c_int + 3 as libc::c_int * 5 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dom_pt
                .offset(
                    (2 as libc::c_int + 3 as libc::c_int * 5 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dom_pt
                .offset(
                    (0 as libc::c_int + 3 as libc::c_int * 6 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dom_pt
                .offset(
                    (1 as libc::c_int + 3 as libc::c_int * 6 as libc::c_int) as isize,
                ) = a;
            *dom_pt
                .offset(
                    (2 as libc::c_int + 3 as libc::c_int * 6 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dom_pt
                .offset(
                    (0 as libc::c_int + 3 as libc::c_int * 7 as libc::c_int) as isize,
                ) = -a;
            *dom_pt
                .offset(
                    (1 as libc::c_int + 3 as libc::c_int * 7 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dom_pt
                .offset(
                    (2 as libc::c_int + 3 as libc::c_int * 7 as libc::c_int) as isize,
                ) = 0 as libc::c_int as libc::c_double;
            *dom_pt
                .offset(
                    (0 as libc::c_int + 3 as libc::c_int * 8 as libc::c_int) as isize,
                ) = b * *om_pt.offset(0 as libc::c_int as isize);
            *dom_pt
                .offset(
                    (1 as libc::c_int + 3 as libc::c_int * 8 as libc::c_int) as isize,
                ) = b * *om_pt.offset(1 as libc::c_int as isize);
            *dom_pt
                .offset(
                    (2 as libc::c_int + 3 as libc::c_int * 8 as libc::c_int) as isize,
                ) = b * *om_pt.offset(2 as libc::c_int as isize);
        }
    };
}
