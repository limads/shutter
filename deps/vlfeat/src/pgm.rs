use ::libc;
extern "C" {
    pub type _IO_wide_data;
    pub type _IO_codecvt;
    pub type _IO_marker;
    fn vl_set_last_error(
        error: libc::c_int,
        errorMessage: *const libc::c_char,
        _: ...
    ) -> libc::c_int;
    fn vl_malloc(n: size_t) -> *mut libc::c_void;
    fn vl_free(ptr: *mut libc::c_void);
    fn fclose(__stream: *mut FILE) -> libc::c_int;
    fn fopen(_: *const libc::c_char, _: *const libc::c_char) -> *mut FILE;
    fn fprintf(_: *mut FILE, _: *const libc::c_char, _: ...) -> libc::c_int;
    fn fscanf(_: *mut FILE, _: *const libc::c_char, _: ...) -> libc::c_int;
    fn fgetc(__stream: *mut FILE) -> libc::c_int;
    fn ungetc(__c: libc::c_int, __stream: *mut FILE) -> libc::c_int;
    fn fread(
        _: *mut libc::c_void,
        _: libc::c_ulong,
        _: libc::c_ulong,
        _: *mut FILE,
    ) -> libc::c_ulong;
    fn fwrite(
        _: *const libc::c_void,
        _: libc::c_ulong,
        _: libc::c_ulong,
        _: *mut FILE,
    ) -> libc::c_ulong;
    fn memcpy(
        _: *mut libc::c_void,
        _: *const libc::c_void,
        _: libc::c_ulong,
    ) -> *mut libc::c_void;
}
pub type vl_uint64 = libc::c_ulonglong;
pub type vl_uint32 = libc::c_uint;
pub type vl_uint16 = libc::c_ushort;
pub type vl_uint8 = libc::c_uchar;
pub type vl_bool = libc::c_int;
pub type vl_size = vl_uint64;
pub type vl_uindex = vl_uint64;
pub type size_t = libc::c_ulong;
pub type __off_t = libc::c_long;
pub type __off64_t = libc::c_long;
#[derive(Copy, Clone)]
#[repr(C)]
pub union C2RustUnnamed {
    pub raw: vl_uint32,
    pub value: libc::c_float,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _IO_FILE {
    pub _flags: libc::c_int,
    pub _IO_read_ptr: *mut libc::c_char,
    pub _IO_read_end: *mut libc::c_char,
    pub _IO_read_base: *mut libc::c_char,
    pub _IO_write_base: *mut libc::c_char,
    pub _IO_write_ptr: *mut libc::c_char,
    pub _IO_write_end: *mut libc::c_char,
    pub _IO_buf_base: *mut libc::c_char,
    pub _IO_buf_end: *mut libc::c_char,
    pub _IO_save_base: *mut libc::c_char,
    pub _IO_backup_base: *mut libc::c_char,
    pub _IO_save_end: *mut libc::c_char,
    pub _markers: *mut _IO_marker,
    pub _chain: *mut _IO_FILE,
    pub _fileno: libc::c_int,
    pub _flags2: libc::c_int,
    pub _old_offset: __off_t,
    pub _cur_column: libc::c_ushort,
    pub _vtable_offset: libc::c_schar,
    pub _shortbuf: [libc::c_char; 1],
    pub _lock: *mut libc::c_void,
    pub _offset: __off64_t,
    pub _codecvt: *mut _IO_codecvt,
    pub _wide_data: *mut _IO_wide_data,
    pub _freeres_list: *mut _IO_FILE,
    pub _freeres_buf: *mut libc::c_void,
    pub __pad5: size_t,
    pub _mode: libc::c_int,
    pub _unused2: [libc::c_char; 20],
}
pub type _IO_lock_t = ();
pub type FILE = _IO_FILE;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlPgmImage {
    pub width: vl_size,
    pub height: vl_size,
    pub max_value: vl_size,
    pub is_raw: vl_bool,
}
pub type VlPgmImage = _VlPgmImage;
static mut vl_infinity_f: C2RustUnnamed = C2RustUnnamed {
    raw: 0x7f800000 as libc::c_ulong as vl_uint32,
};
unsafe extern "C" fn remove_line(mut f: *mut FILE) -> libc::c_int {
    let mut count: libc::c_int = 0 as libc::c_int;
    let mut c: libc::c_int = 0;
    loop {
        c = fgetc(f);
        count += 1;
        match c {
            10 => {
                break;
            }
            -1 => {}
            _ => {
                continue;
            }
        }
        count -= 1;
        break;
    }
    return count;
}
unsafe extern "C" fn remove_blanks(mut f: *mut FILE) -> libc::c_int {
    let mut count: libc::c_int = 0 as libc::c_int;
    let mut c: libc::c_int = 0;
    loop {
        c = fgetc(f);
        match c {
            9 | 10 | 13 | 32 => {
                count += 1;
            }
            35 => {
                count += 1 as libc::c_int + remove_line(f);
            }
            -1 => {
                break;
            }
            _ => {
                ungetc(c, f);
                break;
            }
        }
    }
    return count;
}
#[no_mangle]
pub unsafe extern "C" fn vl_pgm_get_npixels(mut im: *const VlPgmImage) -> vl_size {
    return ((*im).width).wrapping_mul((*im).height);
}
#[no_mangle]
pub unsafe extern "C" fn vl_pgm_get_bpp(mut im: *const VlPgmImage) -> vl_size {
    return (((*im).max_value >= 256 as libc::c_int as libc::c_ulonglong) as libc::c_int
        + 1 as libc::c_int) as vl_size;
}
#[no_mangle]
pub unsafe extern "C" fn vl_pgm_extract_head(
    mut f: *mut FILE,
    mut im: *mut VlPgmImage,
) -> libc::c_int {
    let mut magic: [libc::c_char; 2] = [0; 2];
    let mut c: libc::c_int = 0;
    let mut is_raw: libc::c_int = 0;
    let mut width: libc::c_int = 0;
    let mut height: libc::c_int = 0;
    let mut max_value: libc::c_int = 0;
    let mut sz: size_t = 0;
    let mut good: vl_bool = 0;
    sz = fread(
        magic.as_mut_ptr() as *mut libc::c_void,
        1 as libc::c_int as libc::c_ulong,
        2 as libc::c_int as libc::c_ulong,
        f,
    );
    if sz < 2 as libc::c_int as libc::c_ulong {
        return vl_set_last_error(
            101 as libc::c_int,
            b"Invalid PGM header\0" as *const u8 as *const libc::c_char,
        );
    }
    good = (magic[0 as libc::c_int as usize] as libc::c_int == 'P' as i32)
        as libc::c_int;
    match magic[1 as libc::c_int as usize] as libc::c_int {
        50 => {
            is_raw = 0 as libc::c_int;
        }
        53 => {
            is_raw = 1 as libc::c_int;
        }
        _ => {
            good = 0 as libc::c_int;
        }
    }
    if good == 0 {
        return vl_set_last_error(
            101 as libc::c_int,
            b"Invalid PGM header\0" as *const u8 as *const libc::c_char,
        );
    }
    good = 1 as libc::c_int;
    c = remove_blanks(f);
    good &= (c > 0 as libc::c_int) as libc::c_int;
    c = fscanf(
        f,
        b"%d\0" as *const u8 as *const libc::c_char,
        &mut width as *mut libc::c_int,
    );
    good &= (c == 1 as libc::c_int) as libc::c_int;
    c = remove_blanks(f);
    good &= (c > 0 as libc::c_int) as libc::c_int;
    c = fscanf(
        f,
        b"%d\0" as *const u8 as *const libc::c_char,
        &mut height as *mut libc::c_int,
    );
    good &= (c == 1 as libc::c_int) as libc::c_int;
    c = remove_blanks(f);
    good &= (c > 0 as libc::c_int) as libc::c_int;
    c = fscanf(
        f,
        b"%d\0" as *const u8 as *const libc::c_char,
        &mut max_value as *mut libc::c_int,
    );
    good &= (c == 1 as libc::c_int) as libc::c_int;
    c = fgetc(f);
    good
        &= (c == '\n' as i32 || c == '\t' as i32 || c == ' ' as i32 || c == '\r' as i32)
            as libc::c_int;
    if good == 0 {
        return vl_set_last_error(
            102 as libc::c_int,
            b"Invalid PGM meta information\0" as *const u8 as *const libc::c_char,
        );
    }
    if !(max_value >= 65536 as libc::c_int) {
        return vl_set_last_error(
            102 as libc::c_int,
            b"Invalid PGM meta information\0" as *const u8 as *const libc::c_char,
        );
    }
    (*im).width = width as vl_size;
    (*im).height = height as vl_size;
    (*im).max_value = max_value as vl_size;
    (*im).is_raw = is_raw;
    return 0 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn vl_pgm_extract_data(
    mut f: *mut FILE,
    mut im: *const VlPgmImage,
    mut data: *mut libc::c_void,
) -> libc::c_int {
    let mut bpp: vl_size = vl_pgm_get_bpp(im);
    let mut data_size: vl_size = vl_pgm_get_npixels(im);
    let mut good: vl_bool = 1 as libc::c_int;
    let mut c: size_t = 0;
    if (*im).is_raw != 0 {
        c = fread(data, bpp as libc::c_ulong, data_size as libc::c_ulong, f);
        good = (c as libc::c_ulonglong == data_size) as libc::c_int;
        if bpp == 2 as libc::c_int as libc::c_ulonglong {
            let mut i: vl_uindex = 0;
            let mut pt: *mut vl_uint8 = data as *mut vl_uint8;
            i = 0 as libc::c_int as vl_uindex;
            while i < (2 as libc::c_int as libc::c_ulonglong).wrapping_mul(data_size) {
                let mut tmp: vl_uint8 = *pt.offset(i as isize);
                *pt
                    .offset(
                        i as isize,
                    ) = *pt
                    .offset(
                        i.wrapping_add(1 as libc::c_int as libc::c_ulonglong) as isize,
                    );
                *pt
                    .offset(
                        i.wrapping_add(1 as libc::c_int as libc::c_ulonglong) as isize,
                    ) = tmp;
                i = (i as libc::c_ulonglong)
                    .wrapping_add(2 as libc::c_int as libc::c_ulonglong) as vl_uindex
                    as vl_uindex;
            }
        }
    } else {
        let mut i_0: vl_uindex = 0;
        let mut v: libc::c_uint = 0;
        good = 1 as libc::c_int;
        i_0 = 0 as libc::c_int as vl_uindex;
        while i_0 < data_size && good != 0 {
            c = fscanf(
                f,
                b" %ud\0" as *const u8 as *const libc::c_char,
                &mut v as *mut libc::c_uint,
            ) as size_t;
            if bpp == 1 as libc::c_int as libc::c_ulonglong {
                *(data as *mut vl_uint8).offset(i_0 as isize) = v as vl_uint8;
            } else {
                *(data as *mut vl_uint16).offset(i_0 as isize) = v as vl_uint16;
            }
            good &= (c == 1 as libc::c_int as libc::c_ulong) as libc::c_int;
            i_0 = i_0.wrapping_add(1);
        }
    }
    if good == 0 {
        return vl_set_last_error(
            103 as libc::c_int,
            b"Invalid PGM data\0" as *const u8 as *const libc::c_char,
        );
    }
    return 0 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn vl_pgm_insert(
    mut f: *mut FILE,
    mut im: *const VlPgmImage,
    mut data: *const libc::c_void,
) -> libc::c_int {
    let mut bpp: vl_size = vl_pgm_get_bpp(im);
    let mut data_size: vl_size = vl_pgm_get_npixels(im);
    let mut c: size_t = 0;
    fprintf(
        f,
        b"P5\n%d\n%d\n%d\n\0" as *const u8 as *const libc::c_char,
        (*im).width as libc::c_int,
        (*im).height as libc::c_int,
        (*im).max_value as libc::c_int,
    );
    if bpp == 2 as libc::c_int as libc::c_ulonglong {
        let mut i: vl_uindex = 0;
        let mut temp: *mut vl_uint8 = vl_malloc(
            (2 as libc::c_int as libc::c_ulonglong).wrapping_mul(data_size) as size_t,
        ) as *mut vl_uint8;
        memcpy(
            temp as *mut libc::c_void,
            data,
            (2 as libc::c_int as libc::c_ulonglong).wrapping_mul(data_size)
                as libc::c_ulong,
        );
        i = 0 as libc::c_int as vl_uindex;
        while i < (2 as libc::c_int as libc::c_ulonglong).wrapping_mul(data_size) {
            let mut tmp: vl_uint8 = *temp.offset(i as isize);
            *temp
                .offset(
                    i as isize,
                ) = *temp
                .offset(i.wrapping_add(1 as libc::c_int as libc::c_ulonglong) as isize);
            *temp
                .offset(
                    i.wrapping_add(1 as libc::c_int as libc::c_ulonglong) as isize,
                ) = tmp;
            i = (i as libc::c_ulonglong)
                .wrapping_add(2 as libc::c_int as libc::c_ulonglong) as vl_uindex
                as vl_uindex;
        }
        c = fwrite(
            temp as *const libc::c_void,
            2 as libc::c_int as libc::c_ulong,
            data_size as libc::c_ulong,
            f,
        );
        vl_free(temp as *mut libc::c_void);
    } else {
        c = fwrite(data, bpp as libc::c_ulong, data_size as libc::c_ulong, f);
    }
    if c as libc::c_ulonglong != data_size {
        return vl_set_last_error(
            104 as libc::c_int,
            b"Error writing PGM data\0" as *const u8 as *const libc::c_char,
        );
    }
    return 0 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn vl_pgm_read_new(
    mut name: *const libc::c_char,
    mut im: *mut VlPgmImage,
    mut data: *mut *mut vl_uint8,
) -> libc::c_int {
    let mut err: libc::c_int = 0 as libc::c_int;
    let mut f: *mut FILE = fopen(name, b"rb\0" as *const u8 as *const libc::c_char);
    if f.is_null() {
        return vl_set_last_error(
            104 as libc::c_int,
            b"Error opening PGM file `%s' for reading\0" as *const u8
                as *const libc::c_char,
            name,
        );
    }
    err = vl_pgm_extract_head(f, im);
    if err != 0 {
        fclose(f);
        return err;
    }
    if vl_pgm_get_bpp(im) > 1 as libc::c_int as libc::c_ulonglong {
        return vl_set_last_error(
            3 as libc::c_int,
            b"PGM with BPP > 1 not supported\0" as *const u8 as *const libc::c_char,
        );
    }
    *data = vl_malloc(
        (vl_pgm_get_npixels(im))
            .wrapping_mul(
                ::core::mem::size_of::<vl_uint8>() as libc::c_ulong as libc::c_ulonglong,
            ) as size_t,
    ) as *mut vl_uint8;
    err = vl_pgm_extract_data(f, im, *data as *mut libc::c_void);
    if err != 0 {
        vl_free(data as *mut libc::c_void);
        fclose(f);
    }
    fclose(f);
    return err;
}
#[no_mangle]
pub unsafe extern "C" fn vl_pgm_read_new_f(
    mut name: *const libc::c_char,
    mut im: *mut VlPgmImage,
    mut data: *mut *mut libc::c_float,
) -> libc::c_int {
    let mut err: libc::c_int = 0 as libc::c_int;
    let mut npixels: size_t = 0;
    let mut idata: *mut vl_uint8 = 0 as *mut vl_uint8;
    err = vl_pgm_read_new(name, im, &mut idata);
    if err != 0 {
        return err;
    }
    npixels = vl_pgm_get_npixels(im) as size_t;
    *data = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong).wrapping_mul(npixels),
    ) as *mut libc::c_float;
    let mut k: size_t = 0;
    let mut scale: libc::c_float = 1.0f32 / (*im).max_value as libc::c_float;
    k = 0 as libc::c_int as size_t;
    while k < npixels {
        *(*data)
            .offset(
                k as isize,
            ) = scale * *idata.offset(k as isize) as libc::c_int as libc::c_float;
        k = k.wrapping_add(1);
    }
    vl_free(idata as *mut libc::c_void);
    return 0 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn vl_pgm_write(
    mut name: *const libc::c_char,
    mut data: *const vl_uint8,
    mut width: libc::c_int,
    mut height: libc::c_int,
) -> libc::c_int {
    let mut err: libc::c_int = 0 as libc::c_int;
    let mut pgm: VlPgmImage = VlPgmImage {
        width: 0,
        height: 0,
        max_value: 0,
        is_raw: 0,
    };
    let mut f: *mut FILE = fopen(name, b"wb\0" as *const u8 as *const libc::c_char);
    if f.is_null() {
        return vl_set_last_error(
            104 as libc::c_int,
            b"Error opening PGM file '%s' for writing\0" as *const u8
                as *const libc::c_char,
            name,
        );
    }
    pgm.width = width as vl_size;
    pgm.height = height as vl_size;
    pgm.is_raw = 1 as libc::c_int;
    pgm.max_value = 255 as libc::c_int as vl_size;
    err = vl_pgm_insert(f, &mut pgm, data as *const libc::c_void);
    fclose(f);
    return err;
}
#[no_mangle]
pub unsafe extern "C" fn vl_pgm_write_f(
    mut name: *const libc::c_char,
    mut data: *const libc::c_float,
    mut width: libc::c_int,
    mut height: libc::c_int,
) -> libc::c_int {
    let mut err: libc::c_int = 0 as libc::c_int;
    let mut k: libc::c_int = 0;
    let mut min: libc::c_float = vl_infinity_f.value;
    let mut max: libc::c_float = -vl_infinity_f.value;
    let mut scale: libc::c_float = 0.;
    let mut buffer: *mut vl_uint8 = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong)
            .wrapping_mul(width as libc::c_ulong)
            .wrapping_mul(height as libc::c_ulong),
    ) as *mut vl_uint8;
    k = 0 as libc::c_int;
    while k < width * height {
        min = if min < *data.offset(k as isize) {
            min
        } else {
            *data.offset(k as isize)
        };
        max = if max > *data.offset(k as isize) {
            max
        } else {
            *data.offset(k as isize)
        };
        k += 1;
    }
    scale = 255 as libc::c_int as libc::c_float / (max - min + 1.19209290E-07f32);
    k = 0 as libc::c_int;
    while k < width * height {
        *buffer
            .offset(k as isize) = ((*data.offset(k as isize) - min) * scale) as vl_uint8;
        k += 1;
    }
    err = vl_pgm_write(name, buffer, width, height);
    vl_free(buffer as *mut libc::c_void);
    return err;
}
