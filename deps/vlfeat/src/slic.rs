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
    fn floor(_: libc::c_double) -> libc::c_double;
    fn ceil(_: libc::c_double) -> libc::c_double;
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
pub type vl_int64 = libc::c_longlong;
pub type vl_uint64 = libc::c_ulonglong;
pub type vl_uint32 = libc::c_uint;
pub type vl_size = vl_uint64;
pub type vl_index = vl_int64;
pub type vl_uindex = vl_uint64;
pub type size_t = libc::c_ulong;
#[derive(Copy, Clone)]
#[repr(C)]
pub union C2RustUnnamed {
    pub raw: vl_uint32,
    pub value: libc::c_float,
}
static mut vl_infinity_f: C2RustUnnamed = C2RustUnnamed {
    raw: 0x7f800000 as libc::c_ulong as vl_uint32,
};
#[no_mangle]
pub unsafe extern "C" fn vl_slic_segment(
    mut segmentation: *mut vl_uint32,
    mut image: *const libc::c_float,
    mut width: vl_size,
    mut height: vl_size,
    mut numChannels: vl_size,
    mut regionSize: vl_size,
    mut regularization: libc::c_float,
    mut minRegionSize: vl_size,
) {
    let mut i: vl_index = 0;
    let mut x: vl_index = 0;
    let mut y: vl_index = 0;
    let mut u: vl_index = 0;
    let mut v: vl_index = 0;
    let mut k: vl_index = 0;
    let mut region: vl_index = 0;
    let mut iter: vl_uindex = 0;
    let numRegionsX: vl_size = ceil(
        width as libc::c_double / regionSize as libc::c_double,
    ) as vl_size;
    let numRegionsY: vl_size = ceil(
        height as libc::c_double / regionSize as libc::c_double,
    ) as vl_size;
    let numRegions: vl_size = numRegionsX.wrapping_mul(numRegionsY);
    let numPixels: vl_size = width.wrapping_mul(height);
    let mut centers: *mut libc::c_float = 0 as *mut libc::c_float;
    let mut edgeMap: *mut libc::c_float = 0 as *mut libc::c_float;
    let mut previousEnergy: libc::c_float = vl_infinity_f.value;
    let mut startingEnergy: libc::c_float = 0.;
    let mut masses: *mut vl_uint32 = 0 as *mut vl_uint32;
    let maxNumIterations: vl_size = 100 as libc::c_int as vl_size;
    if !segmentation.is_null() {} else {
        __assert_fail(
            b"segmentation\0" as *const u8 as *const libc::c_char,
            b"vl/slic.c\0" as *const u8 as *const libc::c_char,
            192 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 101],
                &[libc::c_char; 101],
            >(
                b"void vl_slic_segment(vl_uint32 *, const float *, vl_size, vl_size, vl_size, vl_size, float, vl_size)\0",
            ))
                .as_ptr(),
        );
    }
    if !image.is_null() {} else {
        __assert_fail(
            b"image\0" as *const u8 as *const libc::c_char,
            b"vl/slic.c\0" as *const u8 as *const libc::c_char,
            193 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 101],
                &[libc::c_char; 101],
            >(
                b"void vl_slic_segment(vl_uint32 *, const float *, vl_size, vl_size, vl_size, vl_size, float, vl_size)\0",
            ))
                .as_ptr(),
        );
    }
    if width >= 1 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"width >= 1\0" as *const u8 as *const libc::c_char,
            b"vl/slic.c\0" as *const u8 as *const libc::c_char,
            194 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 101],
                &[libc::c_char; 101],
            >(
                b"void vl_slic_segment(vl_uint32 *, const float *, vl_size, vl_size, vl_size, vl_size, float, vl_size)\0",
            ))
                .as_ptr(),
        );
    }
    if height >= 1 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"height >= 1\0" as *const u8 as *const libc::c_char,
            b"vl/slic.c\0" as *const u8 as *const libc::c_char,
            195 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 101],
                &[libc::c_char; 101],
            >(
                b"void vl_slic_segment(vl_uint32 *, const float *, vl_size, vl_size, vl_size, vl_size, float, vl_size)\0",
            ))
                .as_ptr(),
        );
    }
    if numChannels >= 1 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"numChannels >= 1\0" as *const u8 as *const libc::c_char,
            b"vl/slic.c\0" as *const u8 as *const libc::c_char,
            196 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 101],
                &[libc::c_char; 101],
            >(
                b"void vl_slic_segment(vl_uint32 *, const float *, vl_size, vl_size, vl_size, vl_size, float, vl_size)\0",
            ))
                .as_ptr(),
        );
    }
    if regionSize >= 1 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"regionSize >= 1\0" as *const u8 as *const libc::c_char,
            b"vl/slic.c\0" as *const u8 as *const libc::c_char,
            197 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 101],
                &[libc::c_char; 101],
            >(
                b"void vl_slic_segment(vl_uint32 *, const float *, vl_size, vl_size, vl_size, vl_size, float, vl_size)\0",
            ))
                .as_ptr(),
        );
    }
    if regularization >= 0 as libc::c_int as libc::c_float {} else {
        __assert_fail(
            b"regularization >= 0\0" as *const u8 as *const libc::c_char,
            b"vl/slic.c\0" as *const u8 as *const libc::c_char,
            198 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 101],
                &[libc::c_char; 101],
            >(
                b"void vl_slic_segment(vl_uint32 *, const float *, vl_size, vl_size, vl_size, vl_size, float, vl_size)\0",
            ))
                .as_ptr(),
        );
    }
    edgeMap = vl_calloc(
        numPixels as size_t,
        ::core::mem::size_of::<libc::c_float>() as libc::c_ulong,
    ) as *mut libc::c_float;
    masses = vl_malloc(
        (::core::mem::size_of::<vl_uint32>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numPixels) as size_t,
    ) as *mut vl_uint32;
    centers = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(
                (2 as libc::c_int as libc::c_ulonglong).wrapping_add(numChannels),
            )
            .wrapping_mul(numRegions) as size_t,
    ) as *mut libc::c_float;
    k = 0 as libc::c_int as vl_index;
    while k < numChannels as libc::c_int as libc::c_longlong {
        y = 1 as libc::c_int as vl_index;
        while y < (height as libc::c_int - 1 as libc::c_int) as libc::c_longlong {
            x = 1 as libc::c_int as vl_index;
            while x < (width as libc::c_int - 1 as libc::c_int) as libc::c_longlong {
                let mut a: libc::c_float = *image
                    .offset(
                        ((x - 1 as libc::c_int as libc::c_longlong) as libc::c_ulonglong)
                            .wrapping_add((y as libc::c_ulonglong).wrapping_mul(width))
                            .wrapping_add(
                                (k as libc::c_ulonglong)
                                    .wrapping_mul(width)
                                    .wrapping_mul(height),
                            ) as isize,
                    );
                let mut b: libc::c_float = *image
                    .offset(
                        ((x + 1 as libc::c_int as libc::c_longlong) as libc::c_ulonglong)
                            .wrapping_add((y as libc::c_ulonglong).wrapping_mul(width))
                            .wrapping_add(
                                (k as libc::c_ulonglong)
                                    .wrapping_mul(width)
                                    .wrapping_mul(height),
                            ) as isize,
                    );
                let mut c: libc::c_float = *image
                    .offset(
                        (x as libc::c_ulonglong)
                            .wrapping_add(
                                ((y + 1 as libc::c_int as libc::c_longlong)
                                    as libc::c_ulonglong)
                                    .wrapping_mul(width),
                            )
                            .wrapping_add(
                                (k as libc::c_ulonglong)
                                    .wrapping_mul(width)
                                    .wrapping_mul(height),
                            ) as isize,
                    );
                let mut d: libc::c_float = *image
                    .offset(
                        (x as libc::c_ulonglong)
                            .wrapping_add(
                                ((y - 1 as libc::c_int as libc::c_longlong)
                                    as libc::c_ulonglong)
                                    .wrapping_mul(width),
                            )
                            .wrapping_add(
                                (k as libc::c_ulonglong)
                                    .wrapping_mul(width)
                                    .wrapping_mul(height),
                            ) as isize,
                    );
                *edgeMap
                    .offset(
                        (x as libc::c_ulonglong)
                            .wrapping_add((y as libc::c_ulonglong).wrapping_mul(width))
                            as isize,
                    ) += (a - b) * (a - b) + (c - d) * (c - d);
                x += 1;
            }
            y += 1;
        }
        k += 1;
    }
    i = 0 as libc::c_int as vl_index;
    v = 0 as libc::c_int as vl_index;
    while v < numRegionsY as libc::c_int as libc::c_longlong {
        u = 0 as libc::c_int as vl_index;
        while u < numRegionsX as libc::c_int as libc::c_longlong {
            let mut xp: vl_index = 0;
            let mut yp: vl_index = 0;
            let mut centerx: vl_index = 0 as libc::c_int as vl_index;
            let mut centery: vl_index = 0 as libc::c_int as vl_index;
            let mut minEdgeValue: libc::c_float = vl_infinity_f.value;
            x = f64::round(regionSize as libc::c_double * (u as libc::c_double + 0.5f64))
                as vl_index;
            y = f64::round(regionSize as libc::c_double * (v as libc::c_double + 0.5f64))
                as vl_index;
            x = if (if x < (width as libc::c_int - 1 as libc::c_int) as libc::c_longlong
            {
                x
            } else {
                (width as libc::c_int - 1 as libc::c_int) as libc::c_longlong
            }) > 0 as libc::c_int as libc::c_longlong
            {
                if x < (width as libc::c_int - 1 as libc::c_int) as libc::c_longlong {
                    x
                } else {
                    (width as libc::c_int - 1 as libc::c_int) as libc::c_longlong
                }
            } else {
                0 as libc::c_int as libc::c_longlong
            };
            y = if (if y < (height as libc::c_int - 1 as libc::c_int) as libc::c_longlong
            {
                y
            } else {
                (height as libc::c_int - 1 as libc::c_int) as libc::c_longlong
            }) > 0 as libc::c_int as libc::c_longlong
            {
                if y < (height as libc::c_int - 1 as libc::c_int) as libc::c_longlong {
                    y
                } else {
                    (height as libc::c_int - 1 as libc::c_int) as libc::c_longlong
                }
            } else {
                0 as libc::c_int as libc::c_longlong
            };
            yp = if 0 as libc::c_int as libc::c_longlong
                > y - 1 as libc::c_int as libc::c_longlong
            {
                0 as libc::c_int as libc::c_longlong
            } else {
                y - 1 as libc::c_int as libc::c_longlong
            };
            while yp
                <= (if ((height as libc::c_int - 1 as libc::c_int) as libc::c_longlong)
                    < y + 1 as libc::c_int as libc::c_longlong
                {
                    (height as libc::c_int - 1 as libc::c_int) as libc::c_longlong
                } else {
                    y + 1 as libc::c_int as libc::c_longlong
                })
            {
                xp = if 0 as libc::c_int as libc::c_longlong
                    > x - 1 as libc::c_int as libc::c_longlong
                {
                    0 as libc::c_int as libc::c_longlong
                } else {
                    x - 1 as libc::c_int as libc::c_longlong
                };
                while xp
                    <= (if ((width as libc::c_int - 1 as libc::c_int)
                        as libc::c_longlong) < x + 1 as libc::c_int as libc::c_longlong
                    {
                        (width as libc::c_int - 1 as libc::c_int) as libc::c_longlong
                    } else {
                        x + 1 as libc::c_int as libc::c_longlong
                    })
                {
                    let mut thisEdgeValue: libc::c_float = *edgeMap
                        .offset(
                            (xp as libc::c_ulonglong)
                                .wrapping_add((yp as libc::c_ulonglong).wrapping_mul(width))
                                as isize,
                        );
                    if thisEdgeValue < minEdgeValue {
                        minEdgeValue = thisEdgeValue;
                        centerx = xp;
                        centery = yp;
                    }
                    xp += 1;
                }
                yp += 1;
            }
            let fresh0 = i;
            i = i + 1;
            *centers.offset(fresh0 as isize) = centerx as libc::c_float;
            let fresh1 = i;
            i = i + 1;
            *centers.offset(fresh1 as isize) = centery as libc::c_float;
            k = 0 as libc::c_int as vl_index;
            while k < numChannels as libc::c_int as libc::c_longlong {
                let fresh2 = i;
                i = i + 1;
                *centers
                    .offset(
                        fresh2 as isize,
                    ) = *image
                    .offset(
                        (centerx as libc::c_ulonglong)
                            .wrapping_add(
                                (centery as libc::c_ulonglong).wrapping_mul(width),
                            )
                            .wrapping_add(
                                (k as libc::c_ulonglong)
                                    .wrapping_mul(width)
                                    .wrapping_mul(height),
                            ) as isize,
                    );
                k += 1;
            }
            u += 1;
        }
        v += 1;
    }
    iter = 0 as libc::c_int as vl_uindex;
    while iter < maxNumIterations {
        let mut factor: libc::c_float = regularization
            / regionSize.wrapping_mul(regionSize) as libc::c_float;
        let mut energy: libc::c_float = 0 as libc::c_int as libc::c_float;
        y = 0 as libc::c_int as vl_index;
        while y < height as libc::c_int as libc::c_longlong {
            x = 0 as libc::c_int as vl_index;
            while x < width as libc::c_int as libc::c_longlong {
                let mut u_0: vl_index = floor(
                    x as libc::c_double / regionSize as libc::c_double - 0.5f64,
                ) as vl_index;
                let mut v_0: vl_index = floor(
                    y as libc::c_double / regionSize as libc::c_double - 0.5f64,
                ) as vl_index;
                let mut up: vl_index = 0;
                let mut vp: vl_index = 0;
                let mut minDistance: libc::c_float = vl_infinity_f.value;
                vp = if 0 as libc::c_int as libc::c_longlong > v_0 {
                    0 as libc::c_int as libc::c_longlong
                } else {
                    v_0
                };
                while vp
                    <= (if ((numRegionsY as libc::c_int - 1 as libc::c_int)
                        as libc::c_longlong) < v_0 + 1 as libc::c_int as libc::c_longlong
                    {
                        (numRegionsY as libc::c_int - 1 as libc::c_int)
                            as libc::c_longlong
                    } else {
                        v_0 + 1 as libc::c_int as libc::c_longlong
                    })
                {
                    up = if 0 as libc::c_int as libc::c_longlong > u_0 {
                        0 as libc::c_int as libc::c_longlong
                    } else {
                        u_0
                    };
                    while up
                        <= (if ((numRegionsX as libc::c_int - 1 as libc::c_int)
                            as libc::c_longlong)
                            < u_0 + 1 as libc::c_int as libc::c_longlong
                        {
                            (numRegionsX as libc::c_int - 1 as libc::c_int)
                                as libc::c_longlong
                        } else {
                            u_0 + 1 as libc::c_int as libc::c_longlong
                        })
                    {
                        let mut region_0: vl_index = (up as libc::c_ulonglong)
                            .wrapping_add(
                                (vp as libc::c_ulonglong).wrapping_mul(numRegionsX),
                            ) as vl_index;
                        let mut centerx_0: libc::c_float = *centers
                            .offset(
                                (2 as libc::c_int as libc::c_ulonglong)
                                    .wrapping_add(numChannels)
                                    .wrapping_mul(region_0 as libc::c_ulonglong)
                                    .wrapping_add(0 as libc::c_int as libc::c_ulonglong)
                                    as isize,
                            );
                        let mut centery_0: libc::c_float = *centers
                            .offset(
                                (2 as libc::c_int as libc::c_ulonglong)
                                    .wrapping_add(numChannels)
                                    .wrapping_mul(region_0 as libc::c_ulonglong)
                                    .wrapping_add(1 as libc::c_int as libc::c_ulonglong)
                                    as isize,
                            );
                        let mut spatial: libc::c_float = (x as libc::c_float - centerx_0)
                            * (x as libc::c_float - centerx_0)
                            + (y as libc::c_float - centery_0)
                                * (y as libc::c_float - centery_0);
                        let mut appearance: libc::c_float = 0 as libc::c_int
                            as libc::c_float;
                        let mut distance: libc::c_float = 0.;
                        k = 0 as libc::c_int as vl_index;
                        while k < numChannels as libc::c_int as libc::c_longlong {
                            let mut centerz: libc::c_float = *centers
                                .offset(
                                    (2 as libc::c_int as libc::c_ulonglong)
                                        .wrapping_add(numChannels)
                                        .wrapping_mul(region_0 as libc::c_ulonglong)
                                        .wrapping_add(k as libc::c_ulonglong)
                                        .wrapping_add(2 as libc::c_int as libc::c_ulonglong)
                                        as isize,
                                );
                            let mut z: libc::c_float = *image
                                .offset(
                                    (x as libc::c_ulonglong)
                                        .wrapping_add((y as libc::c_ulonglong).wrapping_mul(width))
                                        .wrapping_add(
                                            (k as libc::c_ulonglong)
                                                .wrapping_mul(width)
                                                .wrapping_mul(height),
                                        ) as isize,
                                );
                            appearance += (z - centerz) * (z - centerz);
                            k += 1;
                        }
                        distance = appearance + factor * spatial;
                        if minDistance > distance {
                            minDistance = distance;
                            *segmentation
                                .offset(
                                    (x as libc::c_ulonglong)
                                        .wrapping_add((y as libc::c_ulonglong).wrapping_mul(width))
                                        as isize,
                                ) = region_0 as vl_uint32;
                        }
                        up += 1;
                    }
                    vp += 1;
                }
                energy += minDistance;
                x += 1;
            }
            y += 1;
        }
        if iter == 0 as libc::c_int as libc::c_ulonglong {
            startingEnergy = energy;
        } else if ((previousEnergy - energy) as libc::c_double)
            < 1e-5f64 * (startingEnergy - energy) as libc::c_double
        {
            break;
        }
        previousEnergy = energy;
        memset(
            masses as *mut libc::c_void,
            0 as libc::c_int,
            (::core::mem::size_of::<vl_uint32>() as libc::c_ulong as libc::c_ulonglong)
                .wrapping_mul(width)
                .wrapping_mul(height) as libc::c_ulong,
        );
        memset(
            centers as *mut libc::c_void,
            0 as libc::c_int,
            (::core::mem::size_of::<libc::c_float>() as libc::c_ulong
                as libc::c_ulonglong)
                .wrapping_mul(
                    (2 as libc::c_int as libc::c_ulonglong).wrapping_add(numChannels),
                )
                .wrapping_mul(numRegions) as libc::c_ulong,
        );
        y = 0 as libc::c_int as vl_index;
        while y < height as libc::c_int as libc::c_longlong {
            x = 0 as libc::c_int as vl_index;
            while x < width as libc::c_int as libc::c_longlong {
                let mut pixel: vl_index = (x as libc::c_ulonglong)
                    .wrapping_add((y as libc::c_ulonglong).wrapping_mul(width))
                    as vl_index;
                let mut region_1: vl_index = *segmentation.offset(pixel as isize)
                    as vl_index;
                let ref mut fresh3 = *masses.offset(region_1 as isize);
                *fresh3 = (*fresh3).wrapping_add(1);
                *centers
                    .offset(
                        (region_1 as libc::c_ulonglong)
                            .wrapping_mul(
                                (2 as libc::c_int as libc::c_ulonglong)
                                    .wrapping_add(numChannels),
                            )
                            .wrapping_add(0 as libc::c_int as libc::c_ulonglong) as isize,
                    ) += x as libc::c_float;
                *centers
                    .offset(
                        (region_1 as libc::c_ulonglong)
                            .wrapping_mul(
                                (2 as libc::c_int as libc::c_ulonglong)
                                    .wrapping_add(numChannels),
                            )
                            .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as isize,
                    ) += y as libc::c_float;
                k = 0 as libc::c_int as vl_index;
                while k < numChannels as libc::c_int as libc::c_longlong {
                    *centers
                        .offset(
                            (region_1 as libc::c_ulonglong)
                                .wrapping_mul(
                                    (2 as libc::c_int as libc::c_ulonglong)
                                        .wrapping_add(numChannels),
                                )
                                .wrapping_add(k as libc::c_ulonglong)
                                .wrapping_add(2 as libc::c_int as libc::c_ulonglong)
                                as isize,
                        )
                        += *image
                            .offset(
                                (x as libc::c_ulonglong)
                                    .wrapping_add((y as libc::c_ulonglong).wrapping_mul(width))
                                    .wrapping_add(
                                        (k as libc::c_ulonglong)
                                            .wrapping_mul(width)
                                            .wrapping_mul(height),
                                    ) as isize,
                            );
                    k += 1;
                }
                x += 1;
            }
            y += 1;
        }
        region = 0 as libc::c_int as vl_index;
        while region < numRegions as libc::c_int as libc::c_longlong {
            let mut mass: libc::c_float = (if *masses.offset(region as isize)
                as libc::c_double > 1e-8f64
            {
                *masses.offset(region as isize) as libc::c_double
            } else {
                1e-8f64
            }) as libc::c_float;
            i = (2 as libc::c_int as libc::c_ulonglong)
                .wrapping_add(numChannels)
                .wrapping_mul(region as libc::c_ulonglong) as vl_index;
            while i
                < (2 as libc::c_int as libc::c_ulonglong).wrapping_add(numChannels)
                    as libc::c_int as libc::c_longlong
                    * (region + 1 as libc::c_int as libc::c_longlong)
            {
                *centers.offset(i as isize) /= mass;
                i += 1;
            }
            region += 1;
        }
        iter = iter.wrapping_add(1);
    }
    vl_free(masses as *mut libc::c_void);
    vl_free(centers as *mut libc::c_void);
    vl_free(edgeMap as *mut libc::c_void);
    let mut cleaned: *mut vl_uint32 = vl_calloc(
        numPixels as size_t,
        ::core::mem::size_of::<vl_uint32>() as libc::c_ulong,
    ) as *mut vl_uint32;
    let mut segment: *mut vl_uindex = vl_malloc(
        (::core::mem::size_of::<vl_uindex>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numPixels) as size_t,
    ) as *mut vl_uindex;
    let mut segmentSize: vl_size = 0;
    let mut label: vl_uint32 = 0;
    let mut cleanedLabel: vl_uint32 = 0;
    let mut numExpanded: vl_size = 0;
    let dx: [vl_index; 4] = [
        1 as libc::c_int as vl_index,
        -(1 as libc::c_int) as vl_index,
        0 as libc::c_int as vl_index,
        0 as libc::c_int as vl_index,
    ];
    let dy: [vl_index; 4] = [
        0 as libc::c_int as vl_index,
        0 as libc::c_int as vl_index,
        1 as libc::c_int as vl_index,
        -(1 as libc::c_int) as vl_index,
    ];
    let mut direction: vl_index = 0;
    let mut pixel_0: vl_index = 0;
    pixel_0 = 0 as libc::c_int as vl_index;
    while pixel_0 < numPixels as libc::c_int as libc::c_longlong {
        if !(*cleaned.offset(pixel_0 as isize) != 0) {
            label = *segmentation.offset(pixel_0 as isize);
            numExpanded = 0 as libc::c_int as vl_size;
            segmentSize = 0 as libc::c_int as vl_size;
            let fresh4 = segmentSize;
            segmentSize = segmentSize.wrapping_add(1);
            *segment.offset(fresh4 as isize) = pixel_0 as vl_uindex;
            cleanedLabel = label.wrapping_add(1 as libc::c_int as libc::c_uint);
            *cleaned
                .offset(
                    pixel_0 as isize,
                ) = label.wrapping_add(1 as libc::c_int as libc::c_uint);
            x = (pixel_0 as libc::c_ulonglong).wrapping_rem(width) as vl_index;
            y = (pixel_0 as libc::c_ulonglong).wrapping_div(width) as vl_index;
            direction = 0 as libc::c_int as vl_index;
            while direction < 4 as libc::c_int as libc::c_longlong {
                let mut xp_0: vl_index = x + dx[direction as usize];
                let mut yp_0: vl_index = y + dy[direction as usize];
                let mut neighbor: vl_index = (xp_0 as libc::c_ulonglong)
                    .wrapping_add((yp_0 as libc::c_ulonglong).wrapping_mul(width))
                    as vl_index;
                if 0 as libc::c_int as libc::c_longlong <= xp_0
                    && xp_0 < width as libc::c_int as libc::c_longlong
                    && 0 as libc::c_int as libc::c_longlong <= yp_0
                    && yp_0 < height as libc::c_int as libc::c_longlong
                    && *cleaned.offset(neighbor as isize) != 0
                {
                    cleanedLabel = *cleaned.offset(neighbor as isize);
                }
                direction += 1;
            }
            while numExpanded < segmentSize {
                let fresh5 = numExpanded;
                numExpanded = numExpanded.wrapping_add(1);
                let mut open: vl_index = *segment.offset(fresh5 as isize) as vl_index;
                x = (open as libc::c_ulonglong).wrapping_rem(width) as vl_index;
                y = (open as libc::c_ulonglong).wrapping_div(width) as vl_index;
                direction = 0 as libc::c_int as vl_index;
                while direction < 4 as libc::c_int as libc::c_longlong {
                    let mut xp_1: vl_index = x + dx[direction as usize];
                    let mut yp_1: vl_index = y + dy[direction as usize];
                    let mut neighbor_0: vl_index = (xp_1 as libc::c_ulonglong)
                        .wrapping_add((yp_1 as libc::c_ulonglong).wrapping_mul(width))
                        as vl_index;
                    if 0 as libc::c_int as libc::c_longlong <= xp_1
                        && xp_1 < width as libc::c_int as libc::c_longlong
                        && 0 as libc::c_int as libc::c_longlong <= yp_1
                        && yp_1 < height as libc::c_int as libc::c_longlong
                        && *cleaned.offset(neighbor_0 as isize)
                            == 0 as libc::c_int as libc::c_uint
                        && *segmentation.offset(neighbor_0 as isize) == label
                    {
                        *cleaned
                            .offset(
                                neighbor_0 as isize,
                            ) = label.wrapping_add(1 as libc::c_int as libc::c_uint);
                        let fresh6 = segmentSize;
                        segmentSize = segmentSize.wrapping_add(1);
                        *segment.offset(fresh6 as isize) = neighbor_0 as vl_uindex;
                    }
                    direction += 1;
                }
            }
            if segmentSize < minRegionSize {
                while segmentSize > 0 as libc::c_int as libc::c_ulonglong {
                    segmentSize = segmentSize.wrapping_sub(1);
                    *cleaned
                        .offset(
                            *segment.offset(segmentSize as isize) as isize,
                        ) = cleanedLabel;
                }
            }
        }
        pixel_0 += 1;
    }
    pixel_0 = 0 as libc::c_int as vl_index;
    while pixel_0 < numPixels as libc::c_int as libc::c_longlong {
        let ref mut fresh7 = *cleaned.offset(pixel_0 as isize);
        *fresh7 = (*fresh7).wrapping_sub(1);
        pixel_0 += 1;
    }
    memcpy(
        segmentation as *mut libc::c_void,
        cleaned as *const libc::c_void,
        numPixels
            .wrapping_mul(
                ::core::mem::size_of::<vl_uint32>() as libc::c_ulong as libc::c_ulonglong,
            ) as libc::c_ulong,
    );
    vl_free(cleaned as *mut libc::c_void);
    vl_free(segment as *mut libc::c_void);
}
