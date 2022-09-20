use ::libc;
use core::arch::asm;
extern "C" {
    fn vl_malloc(n: size_t) -> *mut libc::c_void;
    fn snprintf(
        _: *mut libc::c_char,
        _: libc::c_ulong,
        _: *const libc::c_char,
        _: ...
    ) -> libc::c_int;
}
pub type size_t = libc::c_ulong;
pub type vl_int32 = libc::c_int;
pub type vl_uint32 = libc::c_uint;
pub type vl_bool = libc::c_int;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlX86CpuInfo {
    pub vendor: C2RustUnnamed,
    pub hasAVX: vl_bool,
    pub hasSSE42: vl_bool,
    pub hasSSE41: vl_bool,
    pub hasSSE3: vl_bool,
    pub hasSSE2: vl_bool,
    pub hasSSE: vl_bool,
    pub hasMMX: vl_bool,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub union C2RustUnnamed {
    pub string: [libc::c_char; 32],
    pub words: [vl_uint32; 8],
}
pub type VlX86CpuInfo = _VlX86CpuInfo;
#[inline]
unsafe extern "C" fn _vl_cpuid(mut info: *mut vl_int32, mut function: libc::c_int) {
    asm!(
        "cpuid\nmov {restmp0:x}, %bx", restmp0 = lateout(reg) * info.offset(1 as
        libc::c_int as isize), inlateout("ax") function => * info.offset(0 as libc::c_int
        as isize), lateout("cx") * info.offset(2 as libc::c_int as isize), lateout("dx")
        * info.offset(3 as libc::c_int as isize), options(att_syntax)
    );
}
#[no_mangle]
pub unsafe extern "C" fn _vl_x86cpu_info_init(mut self_0: *mut VlX86CpuInfo) {
    let mut info: [vl_int32; 4] = [0; 4];
    let mut max_func: libc::c_int = 0 as libc::c_int;
    _vl_cpuid(info.as_mut_ptr(), 0 as libc::c_int);
    max_func = info[0 as libc::c_int as usize];
    (*self_0)
        .vendor
        .words[0 as libc::c_int as usize] = info[1 as libc::c_int as usize] as vl_uint32;
    (*self_0)
        .vendor
        .words[1 as libc::c_int as usize] = info[3 as libc::c_int as usize] as vl_uint32;
    (*self_0)
        .vendor
        .words[2 as libc::c_int as usize] = info[2 as libc::c_int as usize] as vl_uint32;
    if max_func >= 1 as libc::c_int {
        _vl_cpuid(info.as_mut_ptr(), 1 as libc::c_int);
        (*self_0)
            .hasMMX = info[3 as libc::c_int as usize]
            & (1 as libc::c_int) << 23 as libc::c_int;
        (*self_0)
            .hasSSE = info[3 as libc::c_int as usize]
            & (1 as libc::c_int) << 25 as libc::c_int;
        (*self_0)
            .hasSSE2 = info[3 as libc::c_int as usize]
            & (1 as libc::c_int) << 26 as libc::c_int;
        (*self_0)
            .hasSSE3 = info[2 as libc::c_int as usize]
            & (1 as libc::c_int) << 0 as libc::c_int;
        (*self_0)
            .hasSSE41 = info[2 as libc::c_int as usize]
            & (1 as libc::c_int) << 19 as libc::c_int;
        (*self_0)
            .hasSSE42 = info[2 as libc::c_int as usize]
            & (1 as libc::c_int) << 20 as libc::c_int;
        (*self_0)
            .hasAVX = info[2 as libc::c_int as usize]
            & (1 as libc::c_int) << 28 as libc::c_int;
    }
}
#[no_mangle]
pub unsafe extern "C" fn _vl_x86cpu_info_to_string_copy(
    mut self_0: *const VlX86CpuInfo,
) -> *mut libc::c_char {
    let mut string: *mut libc::c_char = 0 as *mut libc::c_char;
    let mut length: libc::c_int = 0 as libc::c_int;
    while string.is_null() {
        if length > 0 as libc::c_int {
            string = vl_malloc(
                (::core::mem::size_of::<libc::c_char>() as libc::c_ulong)
                    .wrapping_mul(length as libc::c_ulong),
            ) as *mut libc::c_char;
            if string.is_null() {
                break;
            }
        }
        length = snprintf(
            string,
            length as libc::c_ulong,
            b"%s%s%s%s%s%s%s%s\0" as *const u8 as *const libc::c_char,
            ((*self_0).vendor.string).as_ptr(),
            if (*self_0).hasMMX != 0 {
                b" MMX\0" as *const u8 as *const libc::c_char
            } else {
                b"\0" as *const u8 as *const libc::c_char
            },
            if (*self_0).hasSSE != 0 {
                b" SSE\0" as *const u8 as *const libc::c_char
            } else {
                b"\0" as *const u8 as *const libc::c_char
            },
            if (*self_0).hasSSE2 != 0 {
                b" SSE2\0" as *const u8 as *const libc::c_char
            } else {
                b"\0" as *const u8 as *const libc::c_char
            },
            if (*self_0).hasSSE3 != 0 {
                b" SSE3\0" as *const u8 as *const libc::c_char
            } else {
                b"\0" as *const u8 as *const libc::c_char
            },
            if (*self_0).hasSSE41 != 0 {
                b" SSE41\0" as *const u8 as *const libc::c_char
            } else {
                b"\0" as *const u8 as *const libc::c_char
            },
            if (*self_0).hasSSE42 != 0 {
                b" SSE42\0" as *const u8 as *const libc::c_char
            } else {
                b"\0" as *const u8 as *const libc::c_char
            },
            if (*self_0).hasAVX != 0 {
                b" AVX\0" as *const u8 as *const libc::c_char
            } else {
                b"\0" as *const u8 as *const libc::c_char
            },
        );
        length += 1 as libc::c_int;
    }
    return string;
}
#[no_mangle]
pub unsafe extern "C" fn vl_static_configuration_to_string_copy() -> *mut libc::c_char {
    let mut hostString: *const libc::c_char = b"X64, little_endian\0" as *const u8
        as *const libc::c_char;
    let mut compilerString: [libc::c_char; 1024] = [0; 1024];
    let mut libraryString: *const libc::c_char = b"POSIX_threads, SSE2\0" as *const u8
        as *const libc::c_char;
    snprintf(
        compilerString.as_mut_ptr(),
        1024 as libc::c_int as libc::c_ulong,
        b"GNU C %d LP64\0" as *const u8 as *const libc::c_char,
        4 as libc::c_int * 10000 as libc::c_int + 2 as libc::c_int * 100 as libc::c_int
            + 1 as libc::c_int,
    );
    let mut string: *mut libc::c_char = 0 as *mut libc::c_char;
    let mut length: libc::c_int = 0 as libc::c_int;
    while string.is_null() {
        if length > 0 as libc::c_int {
            string = vl_malloc(
                (::core::mem::size_of::<libc::c_char>() as libc::c_ulong)
                    .wrapping_mul(length as libc::c_ulong),
            ) as *mut libc::c_char;
            if string.is_null() {
                break;
            }
        }
        length = snprintf(
            string,
            length as libc::c_ulong,
            b"%s, %s, %s\0" as *const u8 as *const libc::c_char,
            hostString,
            compilerString.as_mut_ptr(),
            libraryString,
        );
        length += 1 as libc::c_int;
    }
    return string;
}
