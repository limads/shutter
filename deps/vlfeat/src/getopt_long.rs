use ::libc;
extern "C" {
    pub type _IO_wide_data;
    pub type _IO_codecvt;
    pub type _IO_marker;
    fn strncmp(
        _: *const libc::c_char,
        _: *const libc::c_char,
        _: libc::c_ulong,
    ) -> libc::c_int;
    fn strchr(_: *const libc::c_char, _: libc::c_int) -> *mut libc::c_char;
    fn strcspn(_: *const libc::c_char, _: *const libc::c_char) -> libc::c_ulong;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
    static mut stderr: *mut FILE;
    fn fprintf(_: *mut FILE, _: *const libc::c_char, _: ...) -> libc::c_int;
}
pub type size_t = libc::c_ulong;
pub type __off_t = libc::c_long;
pub type __off64_t = libc::c_long;
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
pub struct option {
    pub name: *const libc::c_char,
    pub has_arg: libc::c_int,
    pub flag: *mut libc::c_int,
    pub val: libc::c_int,
}
#[no_mangle]
pub static mut opterr: libc::c_int = 1 as libc::c_int;
#[no_mangle]
pub static mut optind: libc::c_int = 1 as libc::c_int;
#[no_mangle]
pub static mut optopt: libc::c_int = 0;
#[no_mangle]
pub static mut optarg: *mut libc::c_char = 0 as *const libc::c_char as *mut libc::c_char;
#[no_mangle]
pub static mut optreset: libc::c_int = 0;
#[no_mangle]
pub unsafe extern "C" fn getopt_long(
    mut argc: libc::c_int,
    mut argv: *const *mut libc::c_char,
    mut optstring: *const libc::c_char,
    mut longopts: *const option,
    mut longindex: *mut libc::c_int,
) -> libc::c_int {
    let mut current_block: u64;
    static mut place: *mut libc::c_char = b"\0" as *const u8 as *const libc::c_char
        as *mut libc::c_char;
    static mut optbegin: libc::c_int = 0 as libc::c_int;
    static mut optend: libc::c_int = 0 as libc::c_int;
    let mut oli: *mut libc::c_char = 0 as *mut libc::c_char;
    let mut has_colon: libc::c_int = 0 as libc::c_int;
    let mut ret_val: libc::c_int = 0 as libc::c_int;
    has_colon = (!optstring.is_null()
        && *optstring.offset(0 as libc::c_int as isize) as libc::c_int == ':' as i32)
        as libc::c_int;
    if has_colon != 0 {
        optstring = optstring.offset(1);
    }
    if optreset != 0 || *place as libc::c_int == '\0' as i32 {
        optreset = 0 as libc::c_int;
        if optind >= argc {
            place = b"\0" as *const u8 as *const libc::c_char as *mut libc::c_char;
            return -(1 as libc::c_int);
        }
        optbegin = optind;
        loop {
            place = *argv.offset(optbegin as isize);
            if !(*place.offset(0 as libc::c_int as isize) as libc::c_int != '-' as i32) {
                break;
            }
            optbegin += 1;
            if optbegin >= argc {
                place = b"\0" as *const u8 as *const libc::c_char as *mut libc::c_char;
                return -(1 as libc::c_int);
            }
        }
        place = place.offset(1);
        optend = optbegin + 1 as libc::c_int;
        optarg = 0 as *mut libc::c_char;
        if *place.offset(0 as libc::c_int as isize) as libc::c_int != 0
            && *place.offset(0 as libc::c_int as isize) as libc::c_int == '-' as i32
            && *place.offset(1 as libc::c_int as isize) as libc::c_int == '\0' as i32
        {
            optind = optend;
            place = b"\0" as *const u8 as *const libc::c_char as *mut libc::c_char;
            ret_val = -(1 as libc::c_int);
            current_block = 11159877328049367773;
        } else if *place.offset(0 as libc::c_int as isize) as libc::c_int != 0
            && *place.offset(0 as libc::c_int as isize) as libc::c_int == '-' as i32
            && *place.offset(1 as libc::c_int as isize) as libc::c_int != 0
        {
            let mut namelen: size_t = 0;
            let mut i: libc::c_int = 0;
            place = place.offset(1);
            namelen = strcspn(place, b"=\0" as *const u8 as *const libc::c_char);
            i = 0 as libc::c_int;
            loop {
                if ((*longopts.offset(i as isize)).name).is_null() {
                    current_block = 12997042908615822766;
                    break;
                }
                if strlen((*longopts.offset(i as isize)).name) == namelen
                    && strncmp(place, (*longopts.offset(i as isize)).name, namelen)
                        == 0 as libc::c_int
                {
                    if !longindex.is_null() {
                        *longindex = i;
                    }
                    if (*longopts.offset(i as isize)).has_arg == 1 as libc::c_int
                        || (*longopts.offset(i as isize)).has_arg == 2 as libc::c_int
                    {
                        if *place.offset(namelen as isize) as libc::c_int == '=' as i32 {
                            optarg = place
                                .offset(namelen as isize)
                                .offset(1 as libc::c_int as isize);
                        } else if (*longopts.offset(i as isize)).has_arg
                            == 1 as libc::c_int
                        {
                            if optbegin >= argc - 1 as libc::c_int {
                                if has_colon == 0 && opterr != 0 {
                                    fprintf(
                                        stderr,
                                        b"%s: option requires an argument -- %s\n\0" as *const u8
                                            as *const libc::c_char,
                                        *argv.offset(0 as libc::c_int as isize),
                                        place,
                                    );
                                }
                                place = b"\0" as *const u8 as *const libc::c_char
                                    as *mut libc::c_char;
                                ret_val = if has_colon != 0 {
                                    ':' as i32
                                } else {
                                    '?' as i32
                                };
                                current_block = 11159877328049367773;
                                break;
                            } else {
                                optarg = *argv.offset(optend as isize);
                                optend += 1;
                            }
                        }
                    }
                    if ((*longopts.offset(i as isize)).flag).is_null() {
                        ret_val = (*longopts.offset(i as isize)).val;
                    } else {
                        *(*longopts.offset(i as isize))
                            .flag = (*longopts.offset(i as isize)).val;
                        ret_val = 0 as libc::c_int;
                    }
                    place = b"\0" as *const u8 as *const libc::c_char
                        as *mut libc::c_char;
                    current_block = 11159877328049367773;
                    break;
                } else {
                    i += 1;
                }
            }
            match current_block {
                11159877328049367773 => {}
                _ => {
                    if has_colon == 0 && opterr != 0 {
                        fprintf(
                            stderr,
                            b"%s: illegal option -- %s\n\0" as *const u8
                                as *const libc::c_char,
                            *argv.offset(0 as libc::c_int as isize),
                            place,
                        );
                    }
                    place = b"\0" as *const u8 as *const libc::c_char
                        as *mut libc::c_char;
                    ret_val = '?' as i32;
                    current_block = 11159877328049367773;
                }
            }
        } else {
            current_block = 313581471991351815;
        }
    } else {
        current_block = 313581471991351815;
    }
    match current_block {
        313581471991351815 => {
            let fresh0 = place;
            place = place.offset(1);
            optopt = *fresh0 as libc::c_int;
            oli = strchr(optstring, optopt);
            if oli.is_null() {
                if has_colon == 0 && opterr != 0 {
                    fprintf(
                        stderr,
                        b"%s: illegal option -- %c\n\0" as *const u8
                            as *const libc::c_char,
                        *argv.offset(0 as libc::c_int as isize),
                        optopt,
                    );
                }
                if *place != 0 {
                    return '?' as i32
                } else {
                    place = b"\0" as *const u8 as *const libc::c_char
                        as *mut libc::c_char;
                    ret_val = '?' as i32;
                }
            } else if *oli.offset(1 as libc::c_int as isize) as libc::c_int != ':' as i32
            {
                if *place != 0 {
                    return optopt
                } else {
                    place = b"\0" as *const u8 as *const libc::c_char
                        as *mut libc::c_char;
                    ret_val = optopt;
                }
            } else if *place != 0 {
                optarg = place;
                place = b"\0" as *const u8 as *const libc::c_char as *mut libc::c_char;
                ret_val = optopt;
            } else if optbegin >= argc - 1 as libc::c_int {
                if has_colon == 0 && opterr != 0 {
                    fprintf(
                        stderr,
                        b"%s: option requires an argument -- %c\n\0" as *const u8
                            as *const libc::c_char,
                        *argv.offset(0 as libc::c_int as isize),
                        optopt,
                    );
                }
                place = b"\0" as *const u8 as *const libc::c_char as *mut libc::c_char;
                ret_val = if has_colon != 0 { ':' as i32 } else { '?' as i32 };
            } else {
                optarg = *argv.offset(optend as isize);
                optend += 1;
                place = b"\0" as *const u8 as *const libc::c_char as *mut libc::c_char;
                ret_val = optopt;
            }
        }
        _ => {}
    }
    let mut pos: libc::c_int = optend - optbegin;
    let mut c: libc::c_int = pos;
    loop {
        let fresh1 = c;
        c = c - 1;
        if !(fresh1 != 0) {
            break;
        }
        let mut i_0: libc::c_int = 0;
        let mut tmp: *mut libc::c_char = *argv
            .offset((optend - 1 as libc::c_int) as isize);
        i_0 = optend - 1 as libc::c_int;
        while i_0 > optind {
            let ref mut fresh2 = *(argv as *mut *mut libc::c_char).offset(i_0 as isize);
            *fresh2 = *argv.offset((i_0 - 1 as libc::c_int) as isize);
            i_0 -= 1;
        }
        let ref mut fresh3 = *(argv as *mut *mut libc::c_char).offset(optind as isize);
        *fresh3 = tmp;
    }
    optind += pos;
    return ret_val;
}
