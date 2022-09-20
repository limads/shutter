use ::libc;
extern "C" {
    fn exit(_: libc::c_int) -> !;
    fn vl_set_last_error(
        error: libc::c_int,
        errorMessage: *const libc::c_char,
        _: ...
    ) -> libc::c_int;
    fn vl_malloc(n: size_t) -> *mut libc::c_void;
    fn vl_free(ptr: *mut libc::c_void);
    fn sqrtf(_: libc::c_float) -> libc::c_float;
    fn memset(
        _: *mut libc::c_void,
        _: libc::c_int,
        _: libc::c_ulong,
    ) -> *mut libc::c_void;
}
pub type vl_int64 = libc::c_longlong;
pub type vl_uint64 = libc::c_ulonglong;
pub type vl_uint8 = libc::c_uchar;
pub type vl_bool = libc::c_int;
pub type vl_size = vl_uint64;
pub type vl_index = vl_int64;
pub type size_t = libc::c_ulong;
pub type _VlLbpMappingType = libc::c_uint;
pub const VlLbpUniform: _VlLbpMappingType = 0;
pub type VlLbpMappingType = _VlLbpMappingType;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct VlLbp_ {
    pub dimension: vl_size,
    pub mapping: [vl_uint8; 256],
    pub transposed: vl_bool,
}
pub type VlLbp = VlLbp_;
#[inline]
unsafe extern "C" fn vl_floor_f(mut x: libc::c_float) -> libc::c_long {
    let mut xi: libc::c_long = x as libc::c_long;
    if x >= 0 as libc::c_int as libc::c_float || xi as libc::c_float == x {
        return xi
    } else {
        return xi - 1 as libc::c_int as libc::c_long
    };
}
unsafe extern "C" fn _vl_lbp_init_uniform(mut self_0: *mut VlLbp) {
    let mut i: libc::c_int = 0;
    let mut j: libc::c_int = 0;
    (*self_0).dimension = 58 as libc::c_int as vl_size;
    i = 0 as libc::c_int;
    while i < 256 as libc::c_int {
        (*self_0).mapping[i as usize] = 57 as libc::c_int as vl_uint8;
        i += 1;
    }
    (*self_0).mapping[0 as libc::c_int as usize] = 56 as libc::c_int as vl_uint8;
    (*self_0).mapping[0xff as libc::c_int as usize] = 56 as libc::c_int as vl_uint8;
    i = 0 as libc::c_int;
    while i < 8 as libc::c_int {
        j = 1 as libc::c_int;
        while j <= 7 as libc::c_int {
            let mut ip: libc::c_int = 0;
            let mut string: libc::c_uint = 0;
            if (*self_0).transposed != 0 {
                ip = (-i + 2 as libc::c_int - (j - 1 as libc::c_int) + 16 as libc::c_int)
                    % 8 as libc::c_int;
            } else {
                ip = i;
            }
            string = (((1 as libc::c_int) << j) - 1 as libc::c_int) as libc::c_uint;
            string <<= ip;
            string = (string | string >> 8 as libc::c_int)
                & 0xff as libc::c_int as libc::c_uint;
            (*self_0)
                .mapping[string
                as usize] = (i * 7 as libc::c_int + (j - 1 as libc::c_int)) as vl_uint8;
            j += 1;
        }
        i += 1;
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_lbp_new(
    mut type_0: VlLbpMappingType,
    mut transposed: vl_bool,
) -> *mut VlLbp {
    let mut self_0: *mut VlLbp = vl_malloc(
        ::core::mem::size_of::<VlLbp>() as libc::c_ulong,
    ) as *mut VlLbp;
    if self_0.is_null() {
        vl_set_last_error(2 as libc::c_int, 0 as *const libc::c_char);
        return 0 as *mut VlLbp;
    }
    (*self_0).transposed = transposed;
    match type_0 as libc::c_uint {
        0 => {
            _vl_lbp_init_uniform(self_0);
        }
        _ => {
            exit(1 as libc::c_int);
        }
    }
    return self_0;
}
#[no_mangle]
pub unsafe extern "C" fn vl_lbp_delete(mut self_0: *mut VlLbp) {
    vl_free(self_0 as *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn vl_lbp_get_dimension(mut self_0: *mut VlLbp) -> vl_size {
    return (*self_0).dimension;
}
#[no_mangle]
pub unsafe extern "C" fn vl_lbp_process(
    mut self_0: *mut VlLbp,
    mut features: *mut libc::c_float,
    mut image: *mut libc::c_float,
    mut width: vl_size,
    mut height: vl_size,
    mut cellSize: vl_size,
) {
    let mut cwidth: vl_size = width.wrapping_div(cellSize);
    let mut cheight: vl_size = height.wrapping_div(cellSize);
    let mut cstride: vl_size = cwidth.wrapping_mul(cheight);
    let mut cdimension: vl_size = vl_lbp_get_dimension(self_0);
    let mut x: vl_index = 0;
    let mut y: vl_index = 0;
    let mut cx: vl_index = 0;
    let mut cy: vl_index = 0;
    let mut k: vl_index = 0;
    let mut bin: vl_index = 0;
    memset(
        features as *mut libc::c_void,
        0 as libc::c_int,
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(cdimension)
            .wrapping_mul(cstride) as libc::c_ulong,
    );
    y = 1 as libc::c_int as vl_index;
    while y < (height as libc::c_int - 1 as libc::c_int) as libc::c_longlong {
        let mut wy1: libc::c_float = (y as libc::c_float + 0.5f32)
            / cellSize as libc::c_float - 0.5f32;
        let mut cy1: libc::c_int = vl_floor_f(wy1) as libc::c_int;
        let mut cy2: libc::c_int = cy1 + 1 as libc::c_int;
        let mut wy2: libc::c_float = wy1 - cy1 as libc::c_float;
        wy1 = 1.0f32 - wy2;
        if !(cy1 >= cheight as libc::c_int) {
            x = 1 as libc::c_int as vl_index;
            while x < (width as libc::c_int - 1 as libc::c_int) as libc::c_longlong {
                let mut wx1: libc::c_float = (x as libc::c_float + 0.5f32)
                    / cellSize as libc::c_float - 0.5f32;
                let mut cx1: libc::c_int = vl_floor_f(wx1) as libc::c_int;
                let mut cx2: libc::c_int = cx1 + 1 as libc::c_int;
                let mut wx2: libc::c_float = wx1 - cx1 as libc::c_float;
                wx1 = 1.0f32 - wx2;
                if !(cx1 >= cwidth as libc::c_int) {
                    let mut bitString: libc::c_uint = 0 as libc::c_int as libc::c_uint;
                    let mut center: libc::c_float = *image
                        .offset(width.wrapping_mul(y as libc::c_ulonglong) as isize)
                        .offset(x as isize);
                    if *image
                        .offset(
                            width
                                .wrapping_mul(
                                    (y + 0 as libc::c_int as libc::c_longlong)
                                        as libc::c_ulonglong,
                                ) as isize,
                        )
                        .offset((x + 1 as libc::c_int as libc::c_longlong) as isize)
                        > center
                    {
                        bitString
                            |= ((0x1 as libc::c_int) << 0 as libc::c_int)
                                as libc::c_uint;
                    }
                    if *image
                        .offset(
                            width
                                .wrapping_mul(
                                    (y + 1 as libc::c_int as libc::c_longlong)
                                        as libc::c_ulonglong,
                                ) as isize,
                        )
                        .offset((x + 1 as libc::c_int as libc::c_longlong) as isize)
                        > center
                    {
                        bitString
                            |= ((0x1 as libc::c_int) << 1 as libc::c_int)
                                as libc::c_uint;
                    }
                    if *image
                        .offset(
                            width
                                .wrapping_mul(
                                    (y + 1 as libc::c_int as libc::c_longlong)
                                        as libc::c_ulonglong,
                                ) as isize,
                        )
                        .offset((x + 0 as libc::c_int as libc::c_longlong) as isize)
                        > center
                    {
                        bitString
                            |= ((0x1 as libc::c_int) << 2 as libc::c_int)
                                as libc::c_uint;
                    }
                    if *image
                        .offset(
                            width
                                .wrapping_mul(
                                    (y + 1 as libc::c_int as libc::c_longlong)
                                        as libc::c_ulonglong,
                                ) as isize,
                        )
                        .offset((x - 1 as libc::c_int as libc::c_longlong) as isize)
                        > center
                    {
                        bitString
                            |= ((0x1 as libc::c_int) << 3 as libc::c_int)
                                as libc::c_uint;
                    }
                    if *image
                        .offset(
                            width
                                .wrapping_mul(
                                    (y + 0 as libc::c_int as libc::c_longlong)
                                        as libc::c_ulonglong,
                                ) as isize,
                        )
                        .offset((x - 1 as libc::c_int as libc::c_longlong) as isize)
                        > center
                    {
                        bitString
                            |= ((0x1 as libc::c_int) << 4 as libc::c_int)
                                as libc::c_uint;
                    }
                    if *image
                        .offset(
                            width
                                .wrapping_mul(
                                    (y - 1 as libc::c_int as libc::c_longlong)
                                        as libc::c_ulonglong,
                                ) as isize,
                        )
                        .offset((x - 1 as libc::c_int as libc::c_longlong) as isize)
                        > center
                    {
                        bitString
                            |= ((0x1 as libc::c_int) << 5 as libc::c_int)
                                as libc::c_uint;
                    }
                    if *image
                        .offset(
                            width
                                .wrapping_mul(
                                    (y - 1 as libc::c_int as libc::c_longlong)
                                        as libc::c_ulonglong,
                                ) as isize,
                        )
                        .offset((x + 0 as libc::c_int as libc::c_longlong) as isize)
                        > center
                    {
                        bitString
                            |= ((0x1 as libc::c_int) << 6 as libc::c_int)
                                as libc::c_uint;
                    }
                    if *image
                        .offset(
                            width
                                .wrapping_mul(
                                    (y - 1 as libc::c_int as libc::c_longlong)
                                        as libc::c_ulonglong,
                                ) as isize,
                        )
                        .offset((x + 1 as libc::c_int as libc::c_longlong) as isize)
                        > center
                    {
                        bitString
                            |= ((0x1 as libc::c_int) << 7 as libc::c_int)
                                as libc::c_uint;
                    }
                    bin = (*self_0).mapping[bitString as usize] as vl_index;
                    if (cx1 >= 0 as libc::c_int) as libc::c_int
                        & (cy1 >= 0 as libc::c_int) as libc::c_int != 0
                    {
                        *features
                            .offset(
                                cstride.wrapping_mul(bin as libc::c_ulonglong) as isize,
                            )
                            .offset(
                                cwidth.wrapping_mul(cy1 as libc::c_ulonglong) as isize,
                            )
                            .offset(cx1 as isize) += wx1 * wy1;
                    }
                    if (cx2 < cwidth as libc::c_int) as libc::c_int
                        & (cy1 >= 0 as libc::c_int) as libc::c_int != 0
                    {
                        *features
                            .offset(
                                cstride.wrapping_mul(bin as libc::c_ulonglong) as isize,
                            )
                            .offset(
                                cwidth.wrapping_mul(cy1 as libc::c_ulonglong) as isize,
                            )
                            .offset(cx2 as isize) += wx2 * wy1;
                    }
                    if (cx1 >= 0 as libc::c_int) as libc::c_int
                        & (cy2 < cheight as libc::c_int) as libc::c_int != 0
                    {
                        *features
                            .offset(
                                cstride.wrapping_mul(bin as libc::c_ulonglong) as isize,
                            )
                            .offset(
                                cwidth.wrapping_mul(cy2 as libc::c_ulonglong) as isize,
                            )
                            .offset(cx1 as isize) += wx1 * wy2;
                    }
                    if (cx2 < cwidth as libc::c_int) as libc::c_int
                        & (cy2 < cheight as libc::c_int) as libc::c_int != 0
                    {
                        *features
                            .offset(
                                cstride.wrapping_mul(bin as libc::c_ulonglong) as isize,
                            )
                            .offset(
                                cwidth.wrapping_mul(cy2 as libc::c_ulonglong) as isize,
                            )
                            .offset(cx2 as isize) += wx2 * wy2;
                    }
                }
                x += 1;
            }
        }
        y += 1;
    }
    cy = 0 as libc::c_int as vl_index;
    while cy < cheight as libc::c_int as libc::c_longlong {
        cx = 0 as libc::c_int as vl_index;
        while cx < cwidth as libc::c_int as libc::c_longlong {
            let mut norm: libc::c_float = 0 as libc::c_int as libc::c_float;
            k = 0 as libc::c_int as vl_index;
            while k < cdimension as libc::c_int as libc::c_longlong {
                norm
                    += *features
                        .offset((k as libc::c_ulonglong).wrapping_mul(cstride) as isize);
                k += 1;
            }
            norm = sqrtf(norm) + 1e-10f32;
            k = 0 as libc::c_int as vl_index;
            while k < cdimension as libc::c_int as libc::c_longlong {
                *features
                    .offset(
                        (k as libc::c_ulonglong).wrapping_mul(cstride) as isize,
                    ) = sqrtf(
                    *features
                        .offset((k as libc::c_ulonglong).wrapping_mul(cstride) as isize),
                ) / norm;
                k += 1;
            }
            features = features.offset(1 as libc::c_int as isize);
            cx += 1;
        }
        cy += 1;
    }
}
