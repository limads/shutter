use ::libc;
extern "C" {
    fn __assert_fail(
        __assertion: *const libc::c_char,
        __file: *const libc::c_char,
        __line: libc::c_uint,
        __function: *const libc::c_char,
    ) -> !;
    fn vl_malloc(n: size_t) -> *mut libc::c_void;
    fn vl_calloc(n: size_t, size: size_t) -> *mut libc::c_void;
    fn vl_free(ptr: *mut libc::c_void);
    fn vl_imsmooth_f(
        smoothed: *mut libc::c_float,
        smoothedStride: vl_size,
        image: *const libc::c_float,
        width: vl_size,
        height: vl_size,
        stride: vl_size,
        sigmax: libc::c_double,
        sigmay: libc::c_double,
    );
    fn pow(_: libc::c_double, _: libc::c_double) -> libc::c_double;
    fn sqrt(_: libc::c_double) -> libc::c_double;
    fn floor(_: libc::c_double) -> libc::c_double;
    fn sqrtf(_: libc::c_float) -> libc::c_float;
    fn memcpy(
        _: *mut libc::c_void,
        _: *const libc::c_void,
        _: libc::c_ulong,
    ) -> *mut libc::c_void;
}
pub type vl_int64 = libc::c_longlong;
pub type vl_uint64 = libc::c_ulonglong;
pub type vl_bool = libc::c_int;
pub type vl_size = vl_uint64;
pub type vl_index = vl_int64;
pub type size_t = libc::c_ulong;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlScaleSpaceGeometry {
    pub width: vl_size,
    pub height: vl_size,
    pub firstOctave: vl_index,
    pub lastOctave: vl_index,
    pub octaveResolution: vl_size,
    pub octaveFirstSubdivision: vl_index,
    pub octaveLastSubdivision: vl_index,
    pub baseScale: libc::c_double,
    pub nominalScale: libc::c_double,
}
pub type VlScaleSpaceGeometry = _VlScaleSpaceGeometry;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlScaleSpaceOctaveGeometry {
    pub width: vl_size,
    pub height: vl_size,
    pub step: libc::c_double,
}
pub type VlScaleSpaceOctaveGeometry = _VlScaleSpaceOctaveGeometry;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlScaleSpace {
    pub geom: VlScaleSpaceGeometry,
    pub octaves: *mut *mut libc::c_float,
}
pub type VlScaleSpace = _VlScaleSpace;
#[no_mangle]
pub unsafe extern "C" fn vl_scalespace_get_default_geometry(
    mut width: vl_size,
    mut height: vl_size,
) -> VlScaleSpaceGeometry {
    let mut geom: VlScaleSpaceGeometry = VlScaleSpaceGeometry {
        width: 0,
        height: 0,
        firstOctave: 0,
        lastOctave: 0,
        octaveResolution: 0,
        octaveFirstSubdivision: 0,
        octaveLastSubdivision: 0,
        baseScale: 0.,
        nominalScale: 0.,
    };
    if width >= 1 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"width >= 1\0" as *const u8 as *const libc::c_char,
            b"vl/scalespace.c\0" as *const u8 as *const libc::c_char,
            309 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 74],
                &[libc::c_char; 74],
            >(
                b"VlScaleSpaceGeometry vl_scalespace_get_default_geometry(vl_size, vl_size)\0",
            ))
                .as_ptr(),
        );
    }
    if height >= 1 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"height >= 1\0" as *const u8 as *const libc::c_char,
            b"vl/scalespace.c\0" as *const u8 as *const libc::c_char,
            310 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 74],
                &[libc::c_char; 74],
            >(
                b"VlScaleSpaceGeometry vl_scalespace_get_default_geometry(vl_size, vl_size)\0",
            ))
                .as_ptr(),
        );
    }
    geom.width = width;
    geom.height = height;
    geom.firstOctave = 0 as libc::c_int as vl_index;
    geom
        .lastOctave = (if floor(
        f64::log2((if width < height { width } else { height }) as libc::c_double),
    ) - 3 as libc::c_int as libc::c_double > 0 as libc::c_int as libc::c_double
    {
        floor(f64::log2((if width < height { width } else { height }) as libc::c_double))
            - 3 as libc::c_int as libc::c_double
    } else {
        0 as libc::c_int as libc::c_double
    }) as vl_index;
    geom.octaveResolution = 3 as libc::c_int as vl_size;
    geom.octaveFirstSubdivision = 0 as libc::c_int as vl_index;
    geom
        .octaveLastSubdivision = (geom.octaveResolution)
        .wrapping_sub(1 as libc::c_int as libc::c_ulonglong) as vl_index;
    geom
        .baseScale = 1.6f64
        * pow(2.0f64, 1.0f64 / geom.octaveResolution as libc::c_double);
    geom.nominalScale = 0.5f64;
    return geom;
}
#[no_mangle]
pub unsafe extern "C" fn vl_scalespacegeometry_is_equal(
    mut a: VlScaleSpaceGeometry,
    mut b: VlScaleSpaceGeometry,
) -> vl_bool {
    return (a.width == b.width && a.height == b.height && a.firstOctave == b.firstOctave
        && a.lastOctave == b.lastOctave && a.octaveResolution == b.octaveResolution
        && a.octaveFirstSubdivision == b.octaveLastSubdivision
        && a.baseScale == b.baseScale && a.nominalScale == b.nominalScale)
        as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn vl_scalespace_get_geometry(
    mut self_0: *const VlScaleSpace,
) -> VlScaleSpaceGeometry {
    return (*self_0).geom;
}
#[no_mangle]
pub unsafe extern "C" fn vl_scalespace_get_octave_geometry(
    mut self_0: *const VlScaleSpace,
    mut o: vl_index,
) -> VlScaleSpaceOctaveGeometry {
    let mut ogeom: VlScaleSpaceOctaveGeometry = VlScaleSpaceOctaveGeometry {
        width: 0,
        height: 0,
        step: 0.,
    };
    ogeom
        .width = if -o >= 0 as libc::c_int as libc::c_longlong {
        (*self_0).geom.width << -o
    } else {
        (*self_0).geom.width >> --o
    };
    ogeom
        .height = if -o >= 0 as libc::c_int as libc::c_longlong {
        (*self_0).geom.height << -o
    } else {
        (*self_0).geom.height >> --o
    };
    ogeom.step = pow(2.0f64, o as libc::c_double);
    return ogeom;
}
#[no_mangle]
pub unsafe extern "C" fn vl_scalespace_get_level(
    mut self_0: *mut VlScaleSpace,
    mut o: vl_index,
    mut s: vl_index,
) -> *mut libc::c_float {
    let mut ogeom: VlScaleSpaceOctaveGeometry = vl_scalespace_get_octave_geometry(
        self_0,
        o,
    );
    let mut octave: *mut libc::c_float = 0 as *mut libc::c_float;
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/scalespace.c\0" as *const u8 as *const libc::c_char,
            394 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 67],
                &[libc::c_char; 67],
            >(b"float *vl_scalespace_get_level(VlScaleSpace *, vl_index, vl_index)\0"))
                .as_ptr(),
        );
    }
    if o >= (*self_0).geom.firstOctave {} else {
        __assert_fail(
            b"o >= self->geom.firstOctave\0" as *const u8 as *const libc::c_char,
            b"vl/scalespace.c\0" as *const u8 as *const libc::c_char,
            395 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 67],
                &[libc::c_char; 67],
            >(b"float *vl_scalespace_get_level(VlScaleSpace *, vl_index, vl_index)\0"))
                .as_ptr(),
        );
    }
    if o <= (*self_0).geom.lastOctave {} else {
        __assert_fail(
            b"o <= self->geom.lastOctave\0" as *const u8 as *const libc::c_char,
            b"vl/scalespace.c\0" as *const u8 as *const libc::c_char,
            396 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 67],
                &[libc::c_char; 67],
            >(b"float *vl_scalespace_get_level(VlScaleSpace *, vl_index, vl_index)\0"))
                .as_ptr(),
        );
    }
    if s >= (*self_0).geom.octaveFirstSubdivision {} else {
        __assert_fail(
            b"s >= self->geom.octaveFirstSubdivision\0" as *const u8
                as *const libc::c_char,
            b"vl/scalespace.c\0" as *const u8 as *const libc::c_char,
            397 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 67],
                &[libc::c_char; 67],
            >(b"float *vl_scalespace_get_level(VlScaleSpace *, vl_index, vl_index)\0"))
                .as_ptr(),
        );
    }
    if s <= (*self_0).geom.octaveLastSubdivision {} else {
        __assert_fail(
            b"s <= self->geom.octaveLastSubdivision\0" as *const u8
                as *const libc::c_char,
            b"vl/scalespace.c\0" as *const u8 as *const libc::c_char,
            398 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 67],
                &[libc::c_char; 67],
            >(b"float *vl_scalespace_get_level(VlScaleSpace *, vl_index, vl_index)\0"))
                .as_ptr(),
        );
    }
    octave = *((*self_0).octaves).offset((o - (*self_0).geom.firstOctave) as isize);
    return octave
        .offset(
            (ogeom.width)
                .wrapping_mul(ogeom.height)
                .wrapping_mul(
                    (s - (*self_0).geom.octaveFirstSubdivision) as libc::c_ulonglong,
                ) as isize,
        );
}
#[no_mangle]
pub unsafe extern "C" fn vl_scalespace_get_level_const(
    mut self_0: *const VlScaleSpace,
    mut o: vl_index,
    mut s: vl_index,
) -> *const libc::c_float {
    return vl_scalespace_get_level(self_0 as *mut VlScaleSpace, o, s);
}
#[no_mangle]
pub unsafe extern "C" fn vl_scalespace_get_level_sigma(
    mut self_0: *const VlScaleSpace,
    mut o: vl_index,
    mut s: vl_index,
) -> libc::c_double {
    return (*self_0).geom.baseScale
        * pow(
            2.0f64,
            o as libc::c_double
                + s as libc::c_double / (*self_0).geom.octaveResolution as libc::c_double,
        );
}
unsafe extern "C" fn copy_and_upsample(
    mut destination: *mut libc::c_float,
    mut source: *const libc::c_float,
    mut width: vl_size,
    mut height: vl_size,
) {
    let mut x: vl_index = 0;
    let mut y: vl_index = 0;
    let mut ox: vl_index = 0;
    let mut oy: vl_index = 0;
    let mut v00: libc::c_float = 0.;
    let mut v10: libc::c_float = 0.;
    let mut v01: libc::c_float = 0.;
    let mut v11: libc::c_float = 0.;
    if !destination.is_null() {} else {
        __assert_fail(
            b"destination\0" as *const u8 as *const libc::c_char,
            b"vl/scalespace.c\0" as *const u8 as *const libc::c_char,
            458 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 65],
                &[libc::c_char; 65],
            >(b"void copy_and_upsample(float *, const float *, vl_size, vl_size)\0"))
                .as_ptr(),
        );
    }
    if !source.is_null() {} else {
        __assert_fail(
            b"source\0" as *const u8 as *const libc::c_char,
            b"vl/scalespace.c\0" as *const u8 as *const libc::c_char,
            459 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 65],
                &[libc::c_char; 65],
            >(b"void copy_and_upsample(float *, const float *, vl_size, vl_size)\0"))
                .as_ptr(),
        );
    }
    y = 0 as libc::c_int as vl_index;
    while y < height as libc::c_int as libc::c_longlong {
        oy = ((y < (height as libc::c_int - 1 as libc::c_int) as libc::c_longlong)
            as libc::c_int as libc::c_ulonglong)
            .wrapping_mul(width) as vl_index;
        v10 = *source.offset(0 as libc::c_int as isize);
        v11 = *source.offset(oy as isize);
        x = 0 as libc::c_int as vl_index;
        while x < width as libc::c_int as libc::c_longlong {
            ox = (x < (width as libc::c_int - 1 as libc::c_int) as libc::c_longlong)
                as libc::c_int as vl_index;
            v00 = v10;
            v01 = v11;
            v10 = *source.offset(ox as isize);
            v11 = *source.offset((ox + oy) as isize);
            *destination.offset(0 as libc::c_int as isize) = v00;
            *destination.offset(1 as libc::c_int as isize) = 0.5f32 * (v00 + v10);
            *destination
                .offset(
                    (2 as libc::c_int as libc::c_ulonglong).wrapping_mul(width) as isize,
                ) = 0.5f32 * (v00 + v01);
            *destination
                .offset(
                    (2 as libc::c_int as libc::c_ulonglong)
                        .wrapping_mul(width)
                        .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as isize,
                ) = 0.25f32 * (v00 + v01 + v10 + v11);
            destination = destination.offset(2 as libc::c_int as isize);
            source = source.offset(1);
            x += 1;
        }
        destination = destination
            .offset(
                (2 as libc::c_int as libc::c_ulonglong).wrapping_mul(width) as isize,
            );
        y += 1;
    }
}
unsafe extern "C" fn copy_and_downsample(
    mut destination: *mut libc::c_float,
    mut source: *const libc::c_float,
    mut width: vl_size,
    mut height: vl_size,
    mut numOctaves: vl_size,
) {
    let mut x: vl_index = 0;
    let mut y: vl_index = 0;
    let mut step: vl_size = ((1 as libc::c_int) << numOctaves) as vl_size;
    if !destination.is_null() {} else {
        __assert_fail(
            b"destination\0" as *const u8 as *const libc::c_char,
            b"vl/scalespace.c\0" as *const u8 as *const libc::c_char,
            506 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 76],
                &[libc::c_char; 76],
            >(
                b"void copy_and_downsample(float *, const float *, vl_size, vl_size, vl_size)\0",
            ))
                .as_ptr(),
        );
    }
    if !source.is_null() {} else {
        __assert_fail(
            b"source\0" as *const u8 as *const libc::c_char,
            b"vl/scalespace.c\0" as *const u8 as *const libc::c_char,
            507 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 76],
                &[libc::c_char; 76],
            >(
                b"void copy_and_downsample(float *, const float *, vl_size, vl_size, vl_size)\0",
            ))
                .as_ptr(),
        );
    }
    if numOctaves == 0 as libc::c_int as libc::c_ulonglong {
        memcpy(
            destination as *mut libc::c_void,
            source as *const libc::c_void,
            (::core::mem::size_of::<libc::c_float>() as libc::c_ulong
                as libc::c_ulonglong)
                .wrapping_mul(width)
                .wrapping_mul(height) as libc::c_ulong,
        );
    } else {
        y = 0 as libc::c_int as vl_index;
        while y < height as libc::c_int as libc::c_longlong {
            let mut p: *const libc::c_float = source
                .offset((y as libc::c_ulonglong).wrapping_mul(width) as isize);
            x = 0 as libc::c_int as vl_index;
            while x
                < (width as libc::c_int - (step as libc::c_int - 1 as libc::c_int))
                    as libc::c_longlong
            {
                let fresh0 = destination;
                destination = destination.offset(1);
                *fresh0 = *p;
                p = p.offset(step as isize);
                x = (x as libc::c_ulonglong).wrapping_add(step) as vl_index as vl_index;
            }
            y = (y as libc::c_ulonglong).wrapping_add(step) as vl_index as vl_index;
        }
    };
}
#[no_mangle]
pub unsafe extern "C" fn vl_scalespace_new(
    mut width: vl_size,
    mut height: vl_size,
) -> *mut VlScaleSpace {
    let mut geom: VlScaleSpaceGeometry = VlScaleSpaceGeometry {
        width: 0,
        height: 0,
        firstOctave: 0,
        lastOctave: 0,
        octaveResolution: 0,
        octaveFirstSubdivision: 0,
        octaveLastSubdivision: 0,
        baseScale: 0.,
        nominalScale: 0.,
    };
    geom = vl_scalespace_get_default_geometry(width, height);
    return vl_scalespace_new_with_geometry(geom);
}
#[no_mangle]
pub unsafe extern "C" fn vl_scalespace_new_with_geometry(
    mut geom: VlScaleSpaceGeometry,
) -> *mut VlScaleSpace {
    let mut current_block: u64;
    let mut o: vl_index = 0;
    let mut numSublevels: vl_size = (geom.octaveLastSubdivision
        - geom.octaveFirstSubdivision + 1 as libc::c_int as libc::c_longlong) as vl_size;
    let mut numOctaves: vl_size = (geom.lastOctave - geom.firstOctave
        + 1 as libc::c_int as libc::c_longlong) as vl_size;
    let mut self_0: *mut VlScaleSpace = 0 as *mut VlScaleSpace;
    if geom.firstOctave <= geom.lastOctave
        && geom.octaveResolution >= 1 as libc::c_int as libc::c_ulonglong
        && geom.octaveFirstSubdivision <= geom.octaveLastSubdivision
        && geom.baseScale >= 0.0f64 && geom.nominalScale >= 0.0f64
    {} else {
        __assert_fail(
            b"is_valid_geometry(geom)\0" as *const u8 as *const libc::c_char,
            b"vl/scalespace.c\0" as *const u8 as *const libc::c_char,
            566 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 68],
                &[libc::c_char; 68],
            >(b"VlScaleSpace *vl_scalespace_new_with_geometry(VlScaleSpaceGeometry)\0"))
                .as_ptr(),
        );
    }
    numOctaves = (geom.lastOctave - geom.firstOctave
        + 1 as libc::c_int as libc::c_longlong) as vl_size;
    numSublevels = (geom.octaveLastSubdivision - geom.octaveFirstSubdivision
        + 1 as libc::c_int as libc::c_longlong) as vl_size;
    self_0 = vl_calloc(
        1 as libc::c_int as size_t,
        ::core::mem::size_of::<VlScaleSpace>() as libc::c_ulong,
    ) as *mut VlScaleSpace;
    if !self_0.is_null() {
        (*self_0).geom = geom;
        (*self_0)
            .octaves = vl_calloc(
            numOctaves as size_t,
            ::core::mem::size_of::<*mut libc::c_float>() as libc::c_ulong,
        ) as *mut *mut libc::c_float;
        if !((*self_0).octaves).is_null() {
            o = (*self_0).geom.firstOctave;
            loop {
                if !(o <= (*self_0).geom.lastOctave) {
                    current_block = 7651349459974463963;
                    break;
                }
                let mut ogeom: VlScaleSpaceOctaveGeometry = vl_scalespace_get_octave_geometry(
                    self_0,
                    o,
                );
                let mut octaveSize: vl_size = (ogeom.width)
                    .wrapping_mul(ogeom.height)
                    .wrapping_mul(numSublevels);
                let ref mut fresh1 = *((*self_0).octaves)
                    .offset((o - (*self_0).geom.firstOctave) as isize);
                *fresh1 = vl_malloc(
                    octaveSize
                        .wrapping_mul(
                            ::core::mem::size_of::<libc::c_float>() as libc::c_ulong
                                as libc::c_ulonglong,
                        ) as size_t,
                ) as *mut libc::c_float;
                if (*((*self_0).octaves)
                    .offset((o - (*self_0).geom.firstOctave) as isize))
                    .is_null()
                {
                    current_block = 16414700378068096622;
                    break;
                }
                o += 1;
            }
            match current_block {
                7651349459974463963 => return self_0,
                _ => {
                    o = (*self_0).geom.firstOctave;
                    while o <= (*self_0).geom.lastOctave {
                        if !(*((*self_0).octaves)
                            .offset((o - (*self_0).geom.firstOctave) as isize))
                            .is_null()
                        {
                            vl_free(
                                *((*self_0).octaves)
                                    .offset((o - (*self_0).geom.firstOctave) as isize)
                                    as *mut libc::c_void,
                            );
                        }
                        o += 1;
                    }
                }
            }
        }
        vl_free(self_0 as *mut libc::c_void);
    }
    return 0 as *mut VlScaleSpace;
}
#[no_mangle]
pub unsafe extern "C" fn vl_scalespace_new_copy(
    mut self_0: *mut VlScaleSpace,
) -> *mut VlScaleSpace {
    let mut o: vl_index = 0;
    let mut copy: *mut VlScaleSpace = vl_scalespace_new_shallow_copy(self_0);
    if copy.is_null() {
        return 0 as *mut VlScaleSpace;
    }
    o = (*self_0).geom.firstOctave;
    while o <= (*self_0).geom.lastOctave {
        let mut ogeom: VlScaleSpaceOctaveGeometry = vl_scalespace_get_octave_geometry(
            self_0,
            o,
        );
        let mut numSubevels: vl_size = ((*self_0).geom.octaveLastSubdivision
            - (*self_0).geom.octaveFirstSubdivision
            + 1 as libc::c_int as libc::c_longlong) as vl_size;
        memcpy(
            *((*copy).octaves).offset((o - (*self_0).geom.firstOctave) as isize)
                as *mut libc::c_void,
            *((*self_0).octaves).offset((o - (*self_0).geom.firstOctave) as isize)
                as *const libc::c_void,
            (ogeom.width)
                .wrapping_mul(ogeom.height)
                .wrapping_mul(numSubevels)
                .wrapping_mul(
                    ::core::mem::size_of::<libc::c_float>() as libc::c_ulong
                        as libc::c_ulonglong,
                ) as libc::c_ulong,
        );
        o += 1;
    }
    return copy;
}
#[no_mangle]
pub unsafe extern "C" fn vl_scalespace_new_shallow_copy(
    mut self_0: *mut VlScaleSpace,
) -> *mut VlScaleSpace {
    return vl_scalespace_new_with_geometry((*self_0).geom);
}
#[no_mangle]
pub unsafe extern "C" fn vl_scalespace_delete(mut self_0: *mut VlScaleSpace) {
    if !self_0.is_null() {
        if !((*self_0).octaves).is_null() {
            let mut o: vl_index = 0;
            o = (*self_0).geom.firstOctave;
            while o <= (*self_0).geom.lastOctave {
                if !(*((*self_0).octaves)
                    .offset((o - (*self_0).geom.firstOctave) as isize))
                    .is_null()
                {
                    vl_free(
                        *((*self_0).octaves)
                            .offset((o - (*self_0).geom.firstOctave) as isize)
                            as *mut libc::c_void,
                    );
                }
                o += 1;
            }
            vl_free((*self_0).octaves as *mut libc::c_void);
        }
        vl_free(self_0 as *mut libc::c_void);
    }
}
#[no_mangle]
pub unsafe extern "C" fn _vl_scalespace_fill_octave(
    mut self_0: *mut VlScaleSpace,
    mut o: vl_index,
) {
    let mut s: vl_index = 0;
    let mut ogeom: VlScaleSpaceOctaveGeometry = vl_scalespace_get_octave_geometry(
        self_0,
        o,
    );
    s = (*self_0).geom.octaveFirstSubdivision + 1 as libc::c_int as libc::c_longlong;
    while s <= (*self_0).geom.octaveLastSubdivision {
        let mut sigma: libc::c_double = vl_scalespace_get_level_sigma(self_0, o, s);
        let mut previousSigma: libc::c_double = vl_scalespace_get_level_sigma(
            self_0,
            o,
            s - 1 as libc::c_int as libc::c_longlong,
        );
        let mut deltaSigma: libc::c_double = sqrtf(
            (sigma * sigma - previousSigma * previousSigma) as libc::c_float,
        ) as libc::c_double;
        let mut level: *mut libc::c_float = vl_scalespace_get_level(self_0, o, s);
        let mut previous: *mut libc::c_float = vl_scalespace_get_level(
            self_0,
            o,
            s - 1 as libc::c_int as libc::c_longlong,
        );
        vl_imsmooth_f(
            level,
            ogeom.width,
            previous,
            ogeom.width,
            ogeom.height,
            ogeom.width,
            deltaSigma / ogeom.step,
            deltaSigma / ogeom.step,
        );
        s += 1;
    }
}
unsafe extern "C" fn _vl_scalespace_start_octave_from_image(
    mut self_0: *mut VlScaleSpace,
    mut image: *const libc::c_float,
    mut o: vl_index,
) {
    let mut level: *mut libc::c_float = 0 as *mut libc::c_float;
    let mut sigma: libc::c_double = 0.;
    let mut imageSigma: libc::c_double = 0.;
    let mut op: vl_index = 0;
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/scalespace.c\0" as *const u8 as *const libc::c_char,
            708 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 85],
                &[libc::c_char; 85],
            >(
                b"void _vl_scalespace_start_octave_from_image(VlScaleSpace *, const float *, vl_index)\0",
            ))
                .as_ptr(),
        );
    }
    if !image.is_null() {} else {
        __assert_fail(
            b"image\0" as *const u8 as *const libc::c_char,
            b"vl/scalespace.c\0" as *const u8 as *const libc::c_char,
            709 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 85],
                &[libc::c_char; 85],
            >(
                b"void _vl_scalespace_start_octave_from_image(VlScaleSpace *, const float *, vl_index)\0",
            ))
                .as_ptr(),
        );
    }
    if o >= (*self_0).geom.firstOctave {} else {
        __assert_fail(
            b"o >= self->geom.firstOctave\0" as *const u8 as *const libc::c_char,
            b"vl/scalespace.c\0" as *const u8 as *const libc::c_char,
            710 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 85],
                &[libc::c_char; 85],
            >(
                b"void _vl_scalespace_start_octave_from_image(VlScaleSpace *, const float *, vl_index)\0",
            ))
                .as_ptr(),
        );
    }
    if o <= (*self_0).geom.lastOctave {} else {
        __assert_fail(
            b"o <= self->geom.lastOctave\0" as *const u8 as *const libc::c_char,
            b"vl/scalespace.c\0" as *const u8 as *const libc::c_char,
            711 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 85],
                &[libc::c_char; 85],
            >(
                b"void _vl_scalespace_start_octave_from_image(VlScaleSpace *, const float *, vl_index)\0",
            ))
                .as_ptr(),
        );
    }
    level = vl_scalespace_get_level(
        self_0,
        if 0 as libc::c_int as libc::c_longlong > o {
            0 as libc::c_int as libc::c_longlong
        } else {
            o
        },
        (*self_0).geom.octaveFirstSubdivision,
    );
    copy_and_downsample(
        level,
        image,
        (*self_0).geom.width,
        (*self_0).geom.height,
        (if 0 as libc::c_int as libc::c_longlong > o {
            0 as libc::c_int as libc::c_longlong
        } else {
            o
        }) as vl_size,
    );
    op = -(1 as libc::c_int) as vl_index;
    while op >= o {
        let mut ogeom: VlScaleSpaceOctaveGeometry = vl_scalespace_get_octave_geometry(
            self_0,
            op + 1 as libc::c_int as libc::c_longlong,
        );
        let mut succLevel: *mut libc::c_float = vl_scalespace_get_level(
            self_0,
            op + 1 as libc::c_int as libc::c_longlong,
            (*self_0).geom.octaveFirstSubdivision,
        );
        level = vl_scalespace_get_level(
            self_0,
            op,
            (*self_0).geom.octaveFirstSubdivision,
        );
        copy_and_upsample(level, succLevel, ogeom.width, ogeom.height);
        op -= 1;
    }
    sigma = vl_scalespace_get_level_sigma(
        self_0,
        o,
        (*self_0).geom.octaveFirstSubdivision,
    );
    imageSigma = (*self_0).geom.nominalScale;
    if sigma > imageSigma {
        let mut ogeom_0: VlScaleSpaceOctaveGeometry = vl_scalespace_get_octave_geometry(
            self_0,
            o,
        );
        let mut deltaSigma: libc::c_double = sqrt(
            sigma * sigma - imageSigma * imageSigma,
        );
        level = vl_scalespace_get_level(
            self_0,
            o,
            (*self_0).geom.octaveFirstSubdivision,
        );
        vl_imsmooth_f(
            level,
            ogeom_0.width,
            level,
            ogeom_0.width,
            ogeom_0.height,
            ogeom_0.width,
            deltaSigma / ogeom_0.step,
            deltaSigma / ogeom_0.step,
        );
    }
}
unsafe extern "C" fn _vl_scalespace_start_octave_from_previous_octave(
    mut self_0: *mut VlScaleSpace,
    mut o: vl_index,
) {
    let mut sigma: libc::c_double = 0.;
    let mut prevSigma: libc::c_double = 0.;
    let mut level: *mut libc::c_float = 0 as *mut libc::c_float;
    let mut prevLevel: *mut libc::c_float = 0 as *mut libc::c_float;
    let mut prevLevelIndex: vl_index = 0;
    let mut ogeom: VlScaleSpaceOctaveGeometry = VlScaleSpaceOctaveGeometry {
        width: 0,
        height: 0,
        step: 0.,
    };
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/scalespace.c\0" as *const u8 as *const libc::c_char,
            763 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 80],
                &[libc::c_char; 80],
            >(
                b"void _vl_scalespace_start_octave_from_previous_octave(VlScaleSpace *, vl_index)\0",
            ))
                .as_ptr(),
        );
    }
    if o > (*self_0).geom.firstOctave {} else {
        __assert_fail(
            b"o > self->geom.firstOctave\0" as *const u8 as *const libc::c_char,
            b"vl/scalespace.c\0" as *const u8 as *const libc::c_char,
            764 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 80],
                &[libc::c_char; 80],
            >(
                b"void _vl_scalespace_start_octave_from_previous_octave(VlScaleSpace *, vl_index)\0",
            ))
                .as_ptr(),
        );
    }
    if o <= (*self_0).geom.lastOctave {} else {
        __assert_fail(
            b"o <= self->geom.lastOctave\0" as *const u8 as *const libc::c_char,
            b"vl/scalespace.c\0" as *const u8 as *const libc::c_char,
            765 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 80],
                &[libc::c_char; 80],
            >(
                b"void _vl_scalespace_start_octave_from_previous_octave(VlScaleSpace *, vl_index)\0",
            ))
                .as_ptr(),
        );
    }
    prevLevelIndex = if ((*self_0).geom.octaveFirstSubdivision
        + (*self_0).geom.octaveResolution as libc::c_int as libc::c_longlong)
        < (*self_0).geom.octaveLastSubdivision
    {
        (*self_0).geom.octaveFirstSubdivision
            + (*self_0).geom.octaveResolution as libc::c_int as libc::c_longlong
    } else {
        (*self_0).geom.octaveLastSubdivision
    };
    prevLevel = vl_scalespace_get_level(
        self_0,
        o - 1 as libc::c_int as libc::c_longlong,
        prevLevelIndex,
    );
    level = vl_scalespace_get_level(self_0, o, (*self_0).geom.octaveFirstSubdivision);
    ogeom = vl_scalespace_get_octave_geometry(
        self_0,
        o - 1 as libc::c_int as libc::c_longlong,
    );
    copy_and_downsample(
        level,
        prevLevel,
        ogeom.width,
        ogeom.height,
        1 as libc::c_int as vl_size,
    );
    sigma = vl_scalespace_get_level_sigma(
        self_0,
        o,
        (*self_0).geom.octaveFirstSubdivision,
    );
    prevSigma = vl_scalespace_get_level_sigma(
        self_0,
        o - 1 as libc::c_int as libc::c_longlong,
        prevLevelIndex,
    );
    if sigma > prevSigma {
        let mut ogeom_0: VlScaleSpaceOctaveGeometry = vl_scalespace_get_octave_geometry(
            self_0,
            o,
        );
        let mut deltaSigma: libc::c_double = sqrt(sigma * sigma - prevSigma * prevSigma);
        level = vl_scalespace_get_level(
            self_0,
            o,
            (*self_0).geom.octaveFirstSubdivision,
        );
        vl_imsmooth_f(
            level,
            ogeom_0.width,
            level,
            ogeom_0.width,
            ogeom_0.height,
            ogeom_0.width,
            deltaSigma / ogeom_0.step,
            deltaSigma / ogeom_0.step,
        );
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_scalespace_put_image(
    mut self_0: *mut VlScaleSpace,
    mut image: *const libc::c_float,
) {
    let mut o: vl_index = 0;
    _vl_scalespace_start_octave_from_image(self_0, image, (*self_0).geom.firstOctave);
    _vl_scalespace_fill_octave(self_0, (*self_0).geom.firstOctave);
    o = (*self_0).geom.firstOctave + 1 as libc::c_int as libc::c_longlong;
    while o <= (*self_0).geom.lastOctave {
        _vl_scalespace_start_octave_from_previous_octave(self_0, o);
        _vl_scalespace_fill_octave(self_0, o);
        o += 1;
    }
}
