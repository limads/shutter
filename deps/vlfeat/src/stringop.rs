use ::libc;
extern "C" {
    fn __assert_fail(
        __assertion: *const libc::c_char,
        __file: *const libc::c_char,
        __line: libc::c_uint,
        __function: *const libc::c_char,
    ) -> !;
    fn strcmp(_: *const libc::c_char, _: *const libc::c_char) -> libc::c_int;
    fn strncmp(
        _: *const libc::c_char,
        _: *const libc::c_char,
        _: libc::c_ulong,
    ) -> libc::c_int;
    fn strstr(_: *const libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
    fn tolower(_: libc::c_int) -> libc::c_int;
}
pub type vl_int64 = libc::c_longlong;
pub type vl_uint64 = libc::c_ulonglong;
pub type vl_bool = libc::c_int;
pub type vl_size = vl_uint64;
pub type vl_index = vl_int64;
pub type vl_uindex = vl_uint64;
pub type C2RustUnnamed = libc::c_int;
pub const VL_PROT_BINARY: C2RustUnnamed = 2;
pub const VL_PROT_ASCII: C2RustUnnamed = 1;
pub const VL_PROT_NONE: C2RustUnnamed = 0;
pub const VL_PROT_UNKNOWN: C2RustUnnamed = -1;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlEnumerator {
    pub name: *const libc::c_char,
    pub value: vl_index,
}
pub type VlEnumerator = _VlEnumerator;
#[no_mangle]
pub unsafe extern "C" fn vl_string_parse_protocol(
    mut string: *const libc::c_char,
    mut protocol: *mut libc::c_int,
) -> *mut libc::c_char {
    let mut cpt: *const libc::c_char = 0 as *const libc::c_char;
    let mut dummy: libc::c_int = 0;
    if protocol.is_null() {
        protocol = &mut dummy;
    }
    cpt = strstr(string, b"://\0" as *const u8 as *const libc::c_char);
    if cpt.is_null() {
        *protocol = VL_PROT_NONE as libc::c_int;
        cpt = string;
    } else {
        if strncmp(
            string,
            b"ascii\0" as *const u8 as *const libc::c_char,
            cpt.offset_from(string) as libc::c_long as libc::c_ulong,
        ) == 0 as libc::c_int
        {
            *protocol = VL_PROT_ASCII as libc::c_int;
        } else if strncmp(
            string,
            b"bin\0" as *const u8 as *const libc::c_char,
            cpt.offset_from(string) as libc::c_long as libc::c_ulong,
        ) == 0 as libc::c_int
        {
            *protocol = VL_PROT_BINARY as libc::c_int;
        } else {
            *protocol = VL_PROT_UNKNOWN as libc::c_int;
        }
        cpt = cpt.offset(3 as libc::c_int as isize);
    }
    return cpt as *mut libc::c_char;
}
#[no_mangle]
pub unsafe extern "C" fn vl_string_protocol_name(
    mut protocol: libc::c_int,
) -> *const libc::c_char {
    match protocol {
        1 => return b"ascii\0" as *const u8 as *const libc::c_char,
        2 => return b"bin\0" as *const u8 as *const libc::c_char,
        0 => return b"\0" as *const u8 as *const libc::c_char,
        _ => return 0 as *const libc::c_char,
    };
}
#[no_mangle]
pub unsafe extern "C" fn vl_string_basename(
    mut destination: *mut libc::c_char,
    mut destinationSize: vl_size,
    mut source: *const libc::c_char,
    mut maxNumStrippedExtensions: vl_size,
) -> vl_size {
    let mut c: libc::c_char = 0;
    let mut k: vl_uindex = 0 as libc::c_int as vl_uindex;
    let mut beg: vl_uindex = 0;
    let mut end: vl_uindex = 0;
    beg = 0 as libc::c_int as vl_uindex;
    k = 0 as libc::c_int as vl_uindex;
    loop {
        c = *source.offset(k as isize);
        if !(c != 0) {
            break;
        }
        if c as libc::c_int == '\\' as i32 || c as libc::c_int == '/' as i32 {
            beg = k.wrapping_add(1 as libc::c_int as libc::c_ulonglong);
        }
        k = k.wrapping_add(1);
    }
    end = strlen(source) as vl_uindex;
    k = end;
    while k > beg {
        if *source.offset(k.wrapping_sub(1 as libc::c_int as libc::c_ulonglong) as isize)
            as libc::c_int == '.' as i32
            && maxNumStrippedExtensions > 0 as libc::c_int as libc::c_ulonglong
        {
            maxNumStrippedExtensions = maxNumStrippedExtensions.wrapping_sub(1);
            end = k.wrapping_sub(1 as libc::c_int as libc::c_ulonglong);
        }
        k = k.wrapping_sub(1);
    }
    return vl_string_copy_sub(
        destination,
        destinationSize,
        source.offset(beg as isize),
        source.offset(end as isize),
    );
}
#[no_mangle]
pub unsafe extern "C" fn vl_string_replace_wildcard(
    mut destination: *mut libc::c_char,
    mut destinationSize: vl_size,
    mut source: *const libc::c_char,
    mut wildcardChar: libc::c_char,
    mut escapeChar: libc::c_char,
    mut replacement: *const libc::c_char,
) -> vl_size {
    let mut c: libc::c_char = 0;
    let mut k: vl_uindex = 0 as libc::c_int as vl_uindex;
    let mut escape: vl_bool = 0 as libc::c_int;
    loop {
        let fresh0 = source;
        source = source.offset(1);
        c = *fresh0;
        if !(c != 0) {
            break;
        }
        if escape == 0 && c as libc::c_int == escapeChar as libc::c_int {
            escape = 1 as libc::c_int;
        } else {
            if escape == 0 && c as libc::c_int == wildcardChar as libc::c_int {
                let mut repl: *const libc::c_char = replacement;
                loop {
                    let fresh1 = repl;
                    repl = repl.offset(1);
                    c = *fresh1;
                    if !(c != 0) {
                        break;
                    }
                    if !destination.is_null()
                        && k.wrapping_add(1 as libc::c_int as libc::c_ulonglong)
                            < destinationSize
                    {
                        *destination.offset(k as isize) = c;
                    }
                    k = k.wrapping_add(1);
                }
            } else {
                if !destination.is_null()
                    && k.wrapping_add(1 as libc::c_int as libc::c_ulonglong)
                        < destinationSize
                {
                    *destination.offset(k as isize) = c;
                }
                k = k.wrapping_add(1);
            }
            escape = 0 as libc::c_int;
        }
    }
    if destinationSize > 0 as libc::c_int as libc::c_ulonglong {
        *destination
            .offset(
                (if k
                    < destinationSize.wrapping_sub(1 as libc::c_int as libc::c_ulonglong)
                {
                    k
                } else {
                    destinationSize.wrapping_sub(1 as libc::c_int as libc::c_ulonglong)
                }) as isize,
            ) = 0 as libc::c_int as libc::c_char;
    }
    return k;
}
#[no_mangle]
pub unsafe extern "C" fn vl_string_copy(
    mut destination: *mut libc::c_char,
    mut destinationSize: vl_size,
    mut source: *const libc::c_char,
) -> vl_size {
    let mut c: libc::c_char = 0;
    let mut k: vl_uindex = 0 as libc::c_int as vl_uindex;
    loop {
        let fresh2 = source;
        source = source.offset(1);
        c = *fresh2;
        if !(c != 0) {
            break;
        }
        if !destination.is_null()
            && k.wrapping_add(1 as libc::c_int as libc::c_ulonglong) < destinationSize
        {
            *destination.offset(k as isize) = c;
        }
        k = k.wrapping_add(1);
    }
    if destinationSize > 0 as libc::c_int as libc::c_ulonglong {
        *destination
            .offset(
                (if k
                    < destinationSize.wrapping_sub(1 as libc::c_int as libc::c_ulonglong)
                {
                    k
                } else {
                    destinationSize.wrapping_sub(1 as libc::c_int as libc::c_ulonglong)
                }) as isize,
            ) = 0 as libc::c_int as libc::c_char;
    }
    return k;
}
#[no_mangle]
pub unsafe extern "C" fn vl_string_copy_sub(
    mut destination: *mut libc::c_char,
    mut destinationSize: vl_size,
    mut beginning: *const libc::c_char,
    mut end: *const libc::c_char,
) -> vl_size {
    let mut c: libc::c_char = 0;
    let mut k: vl_uindex = 0 as libc::c_int as vl_uindex;
    while beginning < end
        && {
            let fresh3 = beginning;
            beginning = beginning.offset(1);
            c = *fresh3;
            c as libc::c_int != 0
        }
    {
        if !destination.is_null()
            && k.wrapping_add(1 as libc::c_int as libc::c_ulonglong) < destinationSize
        {
            *destination.offset(k as isize) = c;
        }
        k = k.wrapping_add(1);
    }
    if destinationSize > 0 as libc::c_int as libc::c_ulonglong {
        *destination
            .offset(
                (if k
                    < destinationSize.wrapping_sub(1 as libc::c_int as libc::c_ulonglong)
                {
                    k
                } else {
                    destinationSize.wrapping_sub(1 as libc::c_int as libc::c_ulonglong)
                }) as isize,
            ) = 0 as libc::c_int as libc::c_char;
    }
    return k;
}
#[no_mangle]
pub unsafe extern "C" fn vl_string_find_char_rev(
    mut beginning: *const libc::c_char,
    mut end: *const libc::c_char,
    mut c: libc::c_char,
) -> *mut libc::c_char {
    loop {
        let fresh4 = end;
        end = end.offset(-1);
        if !(fresh4 != beginning) {
            break;
        }
        if *end as libc::c_int == c as libc::c_int {
            return end as *mut libc::c_char;
        }
    }
    return 0 as *mut libc::c_char;
}
#[no_mangle]
pub unsafe extern "C" fn vl_string_length(mut string: *const libc::c_char) -> vl_size {
    let mut i: vl_uindex = 0;
    i = 0 as libc::c_int as vl_uindex;
    while *string.offset(i as isize) != 0 {
        i = i.wrapping_add(1);
    }
    return i;
}
#[no_mangle]
pub unsafe extern "C" fn vl_string_casei_cmp(
    mut string1: *const libc::c_char,
    mut string2: *const libc::c_char,
) -> libc::c_int {
    while tolower(*string1 as libc::c_uchar as libc::c_int)
        == tolower(*string2 as libc::c_uchar as libc::c_int)
    {
        if *string1 as libc::c_int == 0 as libc::c_int {
            return 0 as libc::c_int;
        }
        string1 = string1.offset(1);
        string2 = string2.offset(1);
    }
    return tolower(*string1 as libc::c_uchar as libc::c_int)
        - tolower(*string2 as libc::c_uchar as libc::c_int);
}
#[no_mangle]
pub unsafe extern "C" fn vl_enumeration_get(
    mut enumeration: *const VlEnumerator,
    mut name: *const libc::c_char,
) -> *mut VlEnumerator {
    if !enumeration.is_null() {} else {
        __assert_fail(
            b"enumeration\0" as *const u8 as *const libc::c_char,
            b"vl/stringop.c\0" as *const u8 as *const libc::c_char,
            411 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 69],
                &[libc::c_char; 69],
            >(b"VlEnumerator *vl_enumeration_get(const VlEnumerator *, const char *)\0"))
                .as_ptr(),
        );
    }
    while !((*enumeration).name).is_null() {
        if strcmp(name, (*enumeration).name) == 0 as libc::c_int {
            return enumeration as *mut VlEnumerator;
        }
        enumeration = enumeration.offset(1);
    }
    return 0 as *mut VlEnumerator;
}
#[no_mangle]
pub unsafe extern "C" fn vl_enumeration_get_casei(
    mut enumeration: *const VlEnumerator,
    mut name: *const libc::c_char,
) -> *mut VlEnumerator {
    if !enumeration.is_null() {} else {
        __assert_fail(
            b"enumeration\0" as *const u8 as *const libc::c_char,
            b"vl/stringop.c\0" as *const u8 as *const libc::c_char,
            433 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 75],
                &[libc::c_char; 75],
            >(
                b"VlEnumerator *vl_enumeration_get_casei(const VlEnumerator *, const char *)\0",
            ))
                .as_ptr(),
        );
    }
    while !((*enumeration).name).is_null() {
        if vl_string_casei_cmp(name, (*enumeration).name) == 0 as libc::c_int {
            return enumeration as *mut VlEnumerator;
        }
        enumeration = enumeration.offset(1);
    }
    return 0 as *mut VlEnumerator;
}
#[no_mangle]
pub unsafe extern "C" fn vl_enumeration_get_by_value(
    mut enumeration: *const VlEnumerator,
    mut value: vl_index,
) -> *mut VlEnumerator {
    if !enumeration.is_null() {} else {
        __assert_fail(
            b"enumeration\0" as *const u8 as *const libc::c_char,
            b"vl/stringop.c\0" as *const u8 as *const libc::c_char,
            455 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 74],
                &[libc::c_char; 74],
            >(
                b"VlEnumerator *vl_enumeration_get_by_value(const VlEnumerator *, vl_index)\0",
            ))
                .as_ptr(),
        );
    }
    while !((*enumeration).name).is_null() {
        if (*enumeration).value == value {
            return enumeration as *mut VlEnumerator;
        }
        enumeration = enumeration.offset(1);
    }
    return 0 as *mut VlEnumerator;
}
