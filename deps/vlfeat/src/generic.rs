use ::libc;
extern "C" {
    fn _vl_x86cpu_info_to_string_copy(self_0: *const VlX86CpuInfo) -> *mut libc::c_char;
    fn vl_rand_init(self_0: *mut VlRand);
    fn vl_static_configuration_to_string_copy() -> *mut libc::c_char;
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn clock() -> clock_t;
    fn snprintf(
        _: *mut libc::c_char,
        _: libc::c_ulong,
        _: *const libc::c_char,
        _: ...
    ) -> libc::c_int;
    fn vsnprintf(
        _: *mut libc::c_char,
        _: libc::c_ulong,
        _: *const libc::c_char,
        _: ::core::ffi::VaList,
    ) -> libc::c_int;
    fn pthread_self() -> pthread_t;
    fn pthread_equal(__thread1: pthread_t, __thread2: pthread_t) -> libc::c_int;
    fn pthread_mutex_lock(__mutex: *mut pthread_mutex_t) -> libc::c_int;
    fn pthread_mutex_unlock(__mutex: *mut pthread_mutex_t) -> libc::c_int;
    fn pthread_cond_signal(__cond: *mut pthread_cond_t) -> libc::c_int;
    fn pthread_cond_wait(
        __cond: *mut pthread_cond_t,
        __mutex: *mut pthread_mutex_t,
    ) -> libc::c_int;
    fn pthread_setspecific(
        __key: pthread_key_t,
        __pointer: *const libc::c_void,
    ) -> libc::c_int;
    fn pthread_getspecific(__key: pthread_key_t) -> *mut libc::c_void;
}
pub type __builtin_va_list = [__va_list_tag; 1];
#[derive(Copy, Clone)]
#[repr(C)]
pub struct __va_list_tag {
    pub gp_offset: libc::c_uint,
    pub fp_offset: libc::c_uint,
    pub overflow_arg_area: *mut libc::c_void,
    pub reg_save_area: *mut libc::c_void,
}
pub type vl_uint64 = libc::c_ulonglong;
pub type vl_uint32 = libc::c_uint;
pub type vl_bool = libc::c_int;
pub type vl_size = vl_uint64;
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
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlRand {
    pub mt: [vl_uint32; 624],
    pub mti: vl_uint32,
}
pub type VlRand = _VlRand;
pub type size_t = libc::c_ulong;
pub type __clock_t = libc::c_long;
pub type clock_t = __clock_t;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct __pthread_internal_list {
    pub __prev: *mut __pthread_internal_list,
    pub __next: *mut __pthread_internal_list,
}
pub type __pthread_list_t = __pthread_internal_list;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct __pthread_mutex_s {
    pub __lock: libc::c_int,
    pub __count: libc::c_uint,
    pub __owner: libc::c_int,
    pub __nusers: libc::c_uint,
    pub __kind: libc::c_int,
    pub __spins: libc::c_short,
    pub __elision: libc::c_short,
    pub __list: __pthread_list_t,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct __pthread_cond_s {
    pub c2rust_unnamed: C2RustUnnamed_2,
    pub c2rust_unnamed_0: C2RustUnnamed_0,
    pub __g_refs: [libc::c_uint; 2],
    pub __g_size: [libc::c_uint; 2],
    pub __g1_orig_size: libc::c_uint,
    pub __wrefs: libc::c_uint,
    pub __g_signals: [libc::c_uint; 2],
}
#[derive(Copy, Clone)]
#[repr(C)]
pub union C2RustUnnamed_0 {
    pub __g1_start: libc::c_ulonglong,
    pub __g1_start32: C2RustUnnamed_1,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct C2RustUnnamed_1 {
    pub __low: libc::c_uint,
    pub __high: libc::c_uint,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub union C2RustUnnamed_2 {
    pub __wseq: libc::c_ulonglong,
    pub __wseq32: C2RustUnnamed_3,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct C2RustUnnamed_3 {
    pub __low: libc::c_uint,
    pub __high: libc::c_uint,
}
pub type pthread_t = libc::c_ulong;
pub type pthread_key_t = libc::c_uint;
#[derive(Copy, Clone)]
#[repr(C)]
pub union pthread_mutex_t {
    pub __data: __pthread_mutex_s,
    pub __size: [libc::c_char; 40],
    pub __align: libc::c_long,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub union pthread_cond_t {
    pub __data: __pthread_cond_s,
    pub __size: [libc::c_char; 48],
    pub __align: libc::c_longlong,
}
pub type VlState = _VlState;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlState {
    pub threadKey: pthread_key_t,
    pub mutex: pthread_mutex_t,
    pub mutexOwner: pthread_t,
    pub mutexCondition: pthread_cond_t,
    pub mutexCount: size_t,
    pub printf_func: Option::<
        unsafe extern "C" fn(*const libc::c_char, ...) -> libc::c_int,
    >,
    pub malloc_func: Option::<unsafe extern "C" fn(size_t) -> *mut libc::c_void>,
    pub realloc_func: Option::<
        unsafe extern "C" fn(*mut libc::c_void, size_t) -> *mut libc::c_void,
    >,
    pub calloc_func: Option::<unsafe extern "C" fn(size_t, size_t) -> *mut libc::c_void>,
    pub free_func: Option::<unsafe extern "C" fn(*mut libc::c_void) -> ()>,
    pub cpuInfo: VlX86CpuInfo,
    pub numCPUs: vl_size,
    pub simdEnabled: vl_bool,
    pub numThreads: vl_size,
}
pub type VlThreadState = _VlThreadState;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlThreadState {
    pub lastError: libc::c_int,
    pub lastErrorMessage: [libc::c_char; 1024],
    pub rand: VlRand,
    pub ticMark: clock_t,
}
pub type va_list = __builtin_va_list;
pub type printf_func_t = Option::<
    unsafe extern "C" fn(*const libc::c_char, ...) -> libc::c_int,
>;
#[no_mangle]
pub static mut _vl_state: VlState = VlState {
    threadKey: 0,
    mutex: pthread_mutex_t {
        __data: __pthread_mutex_s {
            __lock: 0,
            __count: 0,
            __owner: 0,
            __nusers: 0,
            __kind: 0,
            __spins: 0,
            __elision: 0,
            __list: __pthread_list_t {
                __prev: 0 as *const __pthread_internal_list
                    as *mut __pthread_internal_list,
                __next: 0 as *const __pthread_internal_list
                    as *mut __pthread_internal_list,
            },
        },
    },
    mutexOwner: 0,
    mutexCondition: pthread_cond_t {
        __data: __pthread_cond_s {
            c2rust_unnamed: C2RustUnnamed_2 { __wseq: 0 },
            c2rust_unnamed_0: C2RustUnnamed_0 { __g1_start: 0 },
            __g_refs: [0; 2],
            __g_size: [0; 2],
            __g1_orig_size: 0,
            __wrefs: 0,
            __g_signals: [0; 2],
        },
    },
    mutexCount: 0,
    printf_func: None,
    malloc_func: None,
    realloc_func: None,
    calloc_func: None,
    free_func: None,
    cpuInfo: VlX86CpuInfo {
        vendor: C2RustUnnamed { string: [0; 32] },
        hasAVX: 0,
        hasSSE42: 0,
        hasSSE41: 0,
        hasSSE3: 0,
        hasSSE2: 0,
        hasSSE: 0,
        hasMMX: 0,
    },
    numCPUs: 0,
    simdEnabled: 0,
    numThreads: 0,
};
#[no_mangle]
pub unsafe extern "C" fn vl_get_version_string() -> *const libc::c_char {
    return b"0.9.21\0" as *const u8 as *const libc::c_char;
}
#[no_mangle]
pub unsafe extern "C" fn vl_configuration_to_string_copy() -> *mut libc::c_char {
    let mut string: *mut libc::c_char = 0 as *mut libc::c_char;
    let mut length: libc::c_int = 0 as libc::c_int;
    let mut staticString: *mut libc::c_char = vl_static_configuration_to_string_copy();
    let mut cpuString: *mut libc::c_char = _vl_x86cpu_info_to_string_copy(
        &mut (*::core::mem::transmute::<
            unsafe extern "C" fn() -> *mut VlState,
            unsafe extern "C" fn() -> *mut VlState,
        >(vl_get_state)())
            .cpuInfo,
    );
    let debug: libc::c_int = 0 as libc::c_int;
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
            b"VLFeat version %s\n    Static config: %s\n    %llu CPU(s): %s\n    Debug: %s\n\0"
                as *const u8 as *const libc::c_char,
            vl_get_version_string(),
            staticString,
            vl_get_num_cpus(),
            cpuString,
            if debug != 0 {
                b"yes\0" as *const u8 as *const libc::c_char
            } else {
                b"no\0" as *const u8 as *const libc::c_char
            },
        );
        length += 1 as libc::c_int;
    }
    if !staticString.is_null() {
        vl_free(staticString as *mut libc::c_void);
    }
    if !cpuString.is_null() {
        vl_free(cpuString as *mut libc::c_void);
    }
    return string;
}
unsafe extern "C" fn do_nothing_printf(
    mut format: *const libc::c_char,
    mut args: ...
) -> libc::c_int {
    return 0 as libc::c_int;
}
unsafe extern "C" fn vl_lock_state() {
    let mut state: *mut VlState = vl_get_state();
    let mut thisThread: pthread_t = pthread_self();
    pthread_mutex_lock(&mut (*state).mutex);
    if (*state).mutexCount >= 1 as libc::c_int as libc::c_ulong
        && pthread_equal((*state).mutexOwner, thisThread) != 0
    {
        (*state).mutexCount = ((*state).mutexCount).wrapping_add(1);
    } else {
        while (*state).mutexCount >= 1 as libc::c_int as libc::c_ulong {
            pthread_cond_wait(&mut (*state).mutexCondition, &mut (*state).mutex);
        }
        (*state).mutexOwner = thisThread;
        (*state).mutexCount = 1 as libc::c_int as size_t;
    }
    pthread_mutex_unlock(&mut (*state).mutex);
}
unsafe extern "C" fn vl_unlock_state() {
    let mut state: *mut VlState = vl_get_state();
    pthread_mutex_lock(&mut (*state).mutex);
    (*state).mutexCount = ((*state).mutexCount).wrapping_sub(1);
    if (*state).mutexCount == 0 as libc::c_int as libc::c_ulong {
        pthread_cond_signal(&mut (*state).mutexCondition);
    }
    pthread_mutex_unlock(&mut (*state).mutex);
}
#[inline]
unsafe extern "C" fn vl_get_state() -> *mut VlState {
    return &mut _vl_state;
}
#[inline]
unsafe extern "C" fn vl_get_thread_specific_state() -> *mut VlThreadState {
    let mut state: *mut VlState = 0 as *mut VlState;
    let mut threadState: *mut VlThreadState = 0 as *mut VlThreadState;
    vl_lock_state();
    state = vl_get_state();
    threadState = pthread_getspecific((*state).threadKey) as *mut VlThreadState;
    if threadState.is_null() {
        threadState = vl_thread_specific_state_new();
    }
    pthread_setspecific((*state).threadKey, threadState as *const libc::c_void);
    vl_unlock_state();
    return threadState;
}
#[no_mangle]
pub unsafe extern "C" fn vl_get_num_cpus() -> vl_size {
    return (*vl_get_state()).numCPUs;
}
#[no_mangle]
pub unsafe extern "C" fn vl_set_simd_enabled(mut x: vl_bool) {
    (*vl_get_state()).simdEnabled = x;
}
#[no_mangle]
pub unsafe extern "C" fn vl_get_simd_enabled() -> vl_bool {
    return (*vl_get_state()).simdEnabled;
}
#[no_mangle]
pub unsafe extern "C" fn vl_cpu_has_avx() -> vl_bool {
    return (*vl_get_state()).cpuInfo.hasAVX;
}
#[no_mangle]
pub unsafe extern "C" fn vl_cpu_has_sse3() -> vl_bool {
    return (*vl_get_state()).cpuInfo.hasSSE3;
}
#[no_mangle]
pub unsafe extern "C" fn vl_cpu_has_sse2() -> vl_bool {
    return (*vl_get_state()).cpuInfo.hasSSE2;
}
#[no_mangle]
pub unsafe extern "C" fn vl_get_thread_limit() -> vl_size {
    return 1 as libc::c_int as vl_size;
}
#[no_mangle]
pub unsafe extern "C" fn vl_get_max_threads() -> vl_size {
    return 1 as libc::c_int as vl_size;
}
#[no_mangle]
pub unsafe extern "C" fn vl_set_num_threads(mut numThreads: vl_size) {}
#[no_mangle]
pub unsafe extern "C" fn vl_set_last_error(
    mut error: libc::c_int,
    mut errorMessage: *const libc::c_char,
    mut args: ...
) -> libc::c_int {
    let mut state: *mut VlThreadState = vl_get_thread_specific_state();
    let mut args_0: ::core::ffi::VaListImpl;
    args_0 = args.clone();
    if !errorMessage.is_null() {
        vsnprintf(
            ((*state).lastErrorMessage).as_mut_ptr(),
            (::core::mem::size_of::<[libc::c_char; 1024]>() as libc::c_ulong)
                .wrapping_div(::core::mem::size_of::<libc::c_char>() as libc::c_ulong),
            errorMessage,
            args_0.as_va_list(),
        );
    } else {
        (*state)
            .lastErrorMessage[0 as libc::c_int
            as usize] = 0 as libc::c_int as libc::c_char;
    }
    (*state).lastError = error;
    return error;
}
#[no_mangle]
pub unsafe extern "C" fn vl_get_last_error() -> libc::c_int {
    return (*vl_get_thread_specific_state()).lastError;
}
#[no_mangle]
pub unsafe extern "C" fn vl_get_last_error_message() -> *const libc::c_char {
    return ((*vl_get_thread_specific_state()).lastErrorMessage).as_mut_ptr();
}
#[no_mangle]
pub unsafe extern "C" fn vl_set_alloc_func(
    mut malloc_func: Option::<unsafe extern "C" fn(size_t) -> *mut libc::c_void>,
    mut realloc_func: Option::<
        unsafe extern "C" fn(*mut libc::c_void, size_t) -> *mut libc::c_void,
    >,
    mut calloc_func: Option::<unsafe extern "C" fn(size_t, size_t) -> *mut libc::c_void>,
    mut free_func: Option::<unsafe extern "C" fn(*mut libc::c_void) -> ()>,
) {
    let mut state: *mut VlState = 0 as *mut VlState;
    vl_lock_state();
    state = vl_get_state();
    (*state).malloc_func = malloc_func;
    (*state).realloc_func = realloc_func;
    (*state).calloc_func = calloc_func;
    (*state).free_func = free_func;
    vl_unlock_state();
}
#[no_mangle]
pub unsafe extern "C" fn vl_malloc(mut n: size_t) -> *mut libc::c_void {
    return ((*vl_get_state()).malloc_func).expect("non-null function pointer")(n);
}
#[no_mangle]
pub unsafe extern "C" fn vl_realloc(
    mut ptr: *mut libc::c_void,
    mut n: size_t,
) -> *mut libc::c_void {
    return ((*vl_get_state()).realloc_func).expect("non-null function pointer")(ptr, n);
}
#[no_mangle]
pub unsafe extern "C" fn vl_calloc(
    mut n: size_t,
    mut size: size_t,
) -> *mut libc::c_void {
    return ((*vl_get_state()).calloc_func).expect("non-null function pointer")(n, size);
}
#[no_mangle]
pub unsafe extern "C" fn vl_free(mut ptr: *mut libc::c_void) {
    ((*vl_get_state()).free_func).expect("non-null function pointer")(ptr);
}
#[no_mangle]
pub unsafe extern "C" fn vl_set_printf_func(mut printf_func: printf_func_t) {
    let ref mut fresh0 = (*vl_get_state()).printf_func;
    *fresh0 = if printf_func.is_some() {
        printf_func
    } else {
        Some(
            do_nothing_printf
                as unsafe extern "C" fn(*const libc::c_char, ...) -> libc::c_int,
        )
    };
}
#[no_mangle]
pub unsafe extern "C" fn vl_get_printf_func() -> printf_func_t {
    return (*vl_get_state()).printf_func;
}
#[no_mangle]
pub unsafe extern "C" fn vl_get_cpu_time() -> libc::c_double {
    return clock() as libc::c_double
        / 1000000 as libc::c_int as __clock_t as libc::c_double;
}
#[no_mangle]
pub unsafe extern "C" fn vl_tic() {
    let mut threadState: *mut VlThreadState = vl_get_thread_specific_state();
    (*threadState).ticMark = clock();
}
#[no_mangle]
pub unsafe extern "C" fn vl_toc() -> libc::c_double {
    let mut threadState: *mut VlThreadState = vl_get_thread_specific_state();
    return (clock() - (*threadState).ticMark) as libc::c_double
        / 1000000 as libc::c_int as __clock_t as libc::c_double;
}
#[no_mangle]
pub unsafe extern "C" fn vl_get_rand() -> *mut VlRand {
    return &mut (*(vl_get_thread_specific_state
        as unsafe extern "C" fn() -> *mut VlThreadState)())
        .rand;
}
unsafe extern "C" fn vl_thread_specific_state_new() -> *mut VlThreadState {
    let mut self_0: *mut VlThreadState = 0 as *mut VlThreadState;
    self_0 = malloc(::core::mem::size_of::<VlThreadState>() as libc::c_ulong)
        as *mut VlThreadState;
    (*self_0).lastError = 0 as libc::c_int;
    (*self_0)
        .lastErrorMessage[0 as libc::c_int as usize] = 0 as libc::c_int as libc::c_char;
    (*self_0).ticMark = 0 as libc::c_int as clock_t;
    vl_rand_init(&mut (*self_0).rand);
    return self_0;
}
