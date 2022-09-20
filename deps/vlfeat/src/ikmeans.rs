use ::libc;
extern "C" {
    fn vl_rand_uint32(self_0: *mut VlRand) -> vl_uint32;
    fn abort() -> !;
    fn qsort(
        __base: *mut libc::c_void,
        __nmemb: size_t,
        __size: size_t,
        __compar: __compar_fn_t,
    );
    fn vl_get_rand() -> *mut VlRand;
    fn vl_malloc(n: size_t) -> *mut libc::c_void;
    fn vl_calloc(n: size_t, size: size_t) -> *mut libc::c_void;
    fn vl_free(ptr: *mut libc::c_void);
    fn vl_get_printf_func() -> printf_func_t;
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
pub type vl_int32 = libc::c_int;
pub type vl_uint64 = libc::c_ulonglong;
pub type vl_uint32 = libc::c_uint;
pub type vl_uint8 = libc::c_uchar;
pub type vl_uint = libc::c_uint;
pub type vl_bool = libc::c_int;
pub type vl_size = vl_uint64;
pub type vl_index = vl_int64;
pub type vl_uindex = vl_uint64;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlRand {
    pub mt: [vl_uint32; 624],
    pub mti: vl_uint32,
}
pub type VlRand = _VlRand;
pub type size_t = libc::c_ulong;
pub type __compar_fn_t = Option::<
    unsafe extern "C" fn(*const libc::c_void, *const libc::c_void) -> libc::c_int,
>;
pub type printf_func_t = Option::<
    unsafe extern "C" fn(*const libc::c_char, ...) -> libc::c_int,
>;
pub type vl_ikmacc_t = vl_int32;
pub type VlIKMAlgorithms = libc::c_uint;
pub const VL_IKM_ELKAN: VlIKMAlgorithms = 1;
pub const VL_IKM_LLOYD: VlIKMAlgorithms = 0;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlIKMFilt {
    pub M: vl_size,
    pub K: vl_size,
    pub max_niters: vl_size,
    pub method: libc::c_int,
    pub verb: libc::c_int,
    pub centers: *mut vl_ikmacc_t,
    pub inter_dist: *mut vl_ikmacc_t,
}
pub type VlIKMFilt = _VlIKMFilt;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct pair_t {
    pub w: vl_index,
    pub j: vl_index,
}
#[no_mangle]
pub unsafe extern "C" fn vl_ikm_new(mut method: libc::c_int) -> *mut VlIKMFilt {
    let mut f: *mut VlIKMFilt = vl_calloc(
        ::core::mem::size_of::<VlIKMFilt>() as libc::c_ulong,
        1 as libc::c_int as size_t,
    ) as *mut VlIKMFilt;
    (*f).method = method;
    (*f).max_niters = 200 as libc::c_int as vl_size;
    return f;
}
#[no_mangle]
pub unsafe extern "C" fn vl_ikm_delete(mut f: *mut VlIKMFilt) {
    if !f.is_null() {
        if !((*f).centers).is_null() {
            vl_free((*f).centers as *mut libc::c_void);
        }
        if !((*f).inter_dist).is_null() {
            vl_free((*f).inter_dist as *mut libc::c_void);
        }
        vl_free(f as *mut libc::c_void);
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_ikm_train(
    mut f: *mut VlIKMFilt,
    mut data: *const vl_uint8,
    mut N: vl_size,
) -> libc::c_int {
    let mut err: libc::c_int = 0;
    if (*f).verb != 0 {
        (Some(
            ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                .expect("non-null function pointer"),
        ))
            .expect(
                "non-null function pointer",
            )(b"ikm: training with %d data\n\0" as *const u8 as *const libc::c_char, N);
        (Some(
            ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                .expect("non-null function pointer"),
        ))
            .expect(
                "non-null function pointer",
            )(b"ikm: %d clusters\n\0" as *const u8 as *const libc::c_char, (*f).K);
    }
    match (*f).method {
        0 => {
            err = vl_ikm_train_lloyd(f, data, N);
        }
        1 => {
            err = vl_ikm_train_elkan(f, data, N);
        }
        _ => {
            abort();
        }
    }
    return err;
}
#[no_mangle]
pub unsafe extern "C" fn vl_ikm_push(
    mut f: *mut VlIKMFilt,
    mut asgn: *mut vl_uint32,
    mut data: *const vl_uint8,
    mut N: vl_size,
) {
    match (*f).method {
        0 => {
            vl_ikm_push_lloyd(f, asgn, data, N);
        }
        1 => {
            vl_ikm_push_elkan(f, asgn, data, N);
        }
        _ => {
            abort();
        }
    };
}
#[no_mangle]
pub unsafe extern "C" fn vl_ikm_push_one(
    mut centers: *const vl_ikmacc_t,
    mut data: *const vl_uint8,
    mut M: vl_size,
    mut K: vl_size,
) -> vl_uint {
    let mut i: vl_uindex = 0;
    let mut k: vl_uindex = 0;
    let mut best: vl_uindex = -(1 as libc::c_int) as vl_uindex;
    let mut best_dist: vl_ikmacc_t = 0 as libc::c_int;
    k = 0 as libc::c_int as vl_uindex;
    while k < K {
        let mut dist: vl_ikmacc_t = 0 as libc::c_int;
        i = 0 as libc::c_int as vl_uindex;
        while i < M {
            let mut delta: vl_ikmacc_t = *data.offset(i as isize) as vl_ikmacc_t
                - *centers.offset(k.wrapping_mul(M).wrapping_add(i) as isize);
            dist += delta * delta;
            i = i.wrapping_add(1);
        }
        if best == -(1 as libc::c_int) as vl_uindex || dist < best_dist {
            best = k;
            best_dist = dist;
        }
        k = k.wrapping_add(1);
    }
    return best as vl_uint32;
}
#[no_mangle]
pub unsafe extern "C" fn vl_ikm_get_ndims(mut f: *const VlIKMFilt) -> vl_size {
    return (*f).M;
}
#[no_mangle]
pub unsafe extern "C" fn vl_ikm_get_K(mut f: *const VlIKMFilt) -> vl_size {
    return (*f).K;
}
#[no_mangle]
pub unsafe extern "C" fn vl_ikm_get_verbosity(mut f: *const VlIKMFilt) -> libc::c_int {
    return (*f).verb;
}
#[no_mangle]
pub unsafe extern "C" fn vl_ikm_get_max_niters(mut f: *const VlIKMFilt) -> vl_size {
    return (*f).max_niters;
}
#[no_mangle]
pub unsafe extern "C" fn vl_ikm_get_centers(
    mut f: *const VlIKMFilt,
) -> *const vl_ikmacc_t {
    return (*f).centers;
}
#[no_mangle]
pub unsafe extern "C" fn vl_ikm_set_verbosity(
    mut f: *mut VlIKMFilt,
    mut verb: libc::c_int,
) {
    (*f).verb = if 0 as libc::c_int > verb { 0 as libc::c_int } else { verb };
}
#[no_mangle]
pub unsafe extern "C" fn vl_ikm_set_max_niters(
    mut f: *mut VlIKMFilt,
    mut max_niters: vl_size,
) {
    (*f).max_niters = max_niters;
}
unsafe extern "C" fn cmp_pair(
    mut a: *const libc::c_void,
    mut b: *const libc::c_void,
) -> libc::c_int {
    let mut pa: *mut pair_t = a as *mut pair_t;
    let mut pb: *mut pair_t = b as *mut pair_t;
    let mut d: libc::c_int = ((*pa).w - (*pb).w) as libc::c_int;
    if d != 0 {
        return d;
    }
    return ((*pa).j - (*pb).j) as libc::c_int;
}
#[inline]
unsafe extern "C" fn calc_dist2(
    mut A: *const vl_ikmacc_t,
    mut B: *const vl_uint8,
    mut M: vl_size,
) -> vl_ikmacc_t {
    let mut acc: vl_ikmacc_t = 0 as libc::c_int;
    let mut i: vl_uindex = 0 as libc::c_int as vl_uindex;
    i = 0 as libc::c_int as vl_uindex;
    while i < M {
        let mut dist: vl_ikmacc_t = *A.offset(i as isize)
            - *B.offset(i as isize) as vl_ikmacc_t;
        acc = (acc as libc::c_ulonglong).wrapping_add((dist * dist) as vl_uindex)
            as vl_ikmacc_t as vl_ikmacc_t;
        i = i.wrapping_add(1);
    }
    return acc;
}
unsafe extern "C" fn alloc(mut f: *mut VlIKMFilt, mut M: vl_size, mut K: vl_size) {
    if !((*f).centers).is_null() {
        vl_free((*f).centers as *mut libc::c_void);
    }
    (*f).K = K;
    (*f).M = M;
    (*f)
        .centers = vl_malloc(
        (::core::mem::size_of::<vl_ikmacc_t>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(M)
            .wrapping_mul(K) as size_t,
    ) as *mut vl_ikmacc_t;
}
unsafe extern "C" fn vl_ikm_init_helper(mut f: *mut VlIKMFilt) {
    match (*f).method {
        0 => {
            vl_ikm_init_lloyd(f);
        }
        1 => {
            vl_ikm_init_elkan(f);
        }
        _ => {}
    };
}
#[no_mangle]
pub unsafe extern "C" fn vl_ikm_init(
    mut f: *mut VlIKMFilt,
    mut centers: *const vl_ikmacc_t,
    mut M: vl_size,
    mut K: vl_size,
) {
    alloc(f, M, K);
    memcpy(
        (*f).centers as *mut libc::c_void,
        centers as *const libc::c_void,
        (::core::mem::size_of::<vl_ikmacc_t>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(M)
            .wrapping_mul(K) as libc::c_ulong,
    );
    vl_ikm_init_helper(f);
}
#[no_mangle]
pub unsafe extern "C" fn vl_ikm_init_rand(
    mut f: *mut VlIKMFilt,
    mut M: vl_size,
    mut K: vl_size,
) {
    let mut k: vl_uindex = 0;
    let mut i: vl_uindex = 0;
    let mut rand: *mut VlRand = vl_get_rand();
    alloc(f, M, K);
    k = 0 as libc::c_int as vl_uindex;
    while k < K {
        i = 0 as libc::c_int as vl_uindex;
        while i < M {
            *((*f).centers)
                .offset(
                    k.wrapping_mul(M).wrapping_add(i) as isize,
                ) = vl_rand_uint32(rand) as vl_ikmacc_t;
            i = i.wrapping_add(1);
        }
        k = k.wrapping_add(1);
    }
    vl_ikm_init_helper(f);
}
#[no_mangle]
pub unsafe extern "C" fn vl_ikm_init_rand_data(
    mut f: *mut VlIKMFilt,
    mut data: *const vl_uint8,
    mut M: vl_size,
    mut N: vl_size,
    mut K: vl_size,
) {
    let mut i: vl_uindex = 0;
    let mut j: vl_uindex = 0;
    let mut k: vl_uindex = 0;
    let mut rand: *mut VlRand = vl_get_rand();
    let mut pairs: *mut pair_t = vl_malloc(
        (::core::mem::size_of::<pair_t>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(N) as size_t,
    ) as *mut pair_t;
    alloc(f, M, K);
    j = 0 as libc::c_int as vl_uindex;
    while j < N {
        (*pairs.offset(j as isize)).j = j as vl_index;
        (*pairs.offset(j as isize))
            .w = (vl_rand_uint32(rand) as vl_int32 >> 2 as libc::c_int) as vl_index;
        j = j.wrapping_add(1);
    }
    qsort(
        pairs as *mut libc::c_void,
        N as size_t,
        ::core::mem::size_of::<pair_t>() as libc::c_ulong,
        Some(
            cmp_pair
                as unsafe extern "C" fn(
                    *const libc::c_void,
                    *const libc::c_void,
                ) -> libc::c_int,
        ),
    );
    j = 0 as libc::c_int as vl_uindex;
    k = 0 as libc::c_int as vl_uindex;
    while k < K {
        while j < N.wrapping_sub(1 as libc::c_int as libc::c_ulonglong) {
            let mut prevk: vl_uindex = 0 as libc::c_int as vl_uindex;
            prevk = 0 as libc::c_int as vl_uindex;
            while prevk < k {
                let mut dist: vl_ikmacc_t = calc_dist2(
                    ((*f).centers).offset(prevk.wrapping_mul(M) as isize),
                    data
                        .offset(
                            ((*pairs.offset(j as isize)).j as libc::c_ulonglong)
                                .wrapping_mul(M) as isize,
                        ),
                    M,
                );
                if dist == 0 as libc::c_int {
                    break;
                }
                prevk = prevk.wrapping_add(1);
            }
            if prevk == k {
                break;
            }
            j = j.wrapping_add(1);
        }
        i = 0 as libc::c_int as vl_uindex;
        while i < M {
            *((*f).centers)
                .offset(
                    k.wrapping_mul(M).wrapping_add(i) as isize,
                ) = *data
                .offset(
                    ((*pairs.offset(j as isize)).j as vl_uint64)
                        .wrapping_mul(M)
                        .wrapping_add(i) as isize,
                ) as vl_ikmacc_t;
            i = i.wrapping_add(1);
        }
        if j < N.wrapping_sub(1 as libc::c_int as libc::c_ulonglong) {
            j = j.wrapping_add(1);
        }
        k = k.wrapping_add(1);
    }
    vl_free(pairs as *mut libc::c_void);
    vl_ikm_init_helper(f);
}
unsafe extern "C" fn vl_ikm_init_lloyd(mut f: *mut VlIKMFilt) {}
unsafe extern "C" fn vl_ikm_train_lloyd(
    mut f: *mut VlIKMFilt,
    mut data: *const vl_uint8,
    mut N: vl_size,
) -> libc::c_int {
    let mut err: libc::c_int = 0 as libc::c_int;
    let mut iter: vl_uindex = 0;
    let mut i: vl_uindex = 0;
    let mut j: vl_uindex = 0;
    let mut k: vl_uindex = 0;
    let mut asgn: *mut vl_uint32 = vl_malloc(
        (::core::mem::size_of::<vl_uint32>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(N) as size_t,
    ) as *mut vl_uint32;
    let mut counts: *mut vl_uint32 = vl_malloc(
        (::core::mem::size_of::<vl_uint32>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(N) as size_t,
    ) as *mut vl_uint32;
    iter = 0 as libc::c_int as vl_uindex;
    loop {
        let mut done: vl_bool = 1 as libc::c_int;
        j = 0 as libc::c_int as vl_uindex;
        while j < N {
            let mut best_dist: vl_ikmacc_t = 0 as libc::c_int;
            let mut best: vl_index = -(1 as libc::c_int) as vl_index;
            k = 0 as libc::c_int as vl_uindex;
            while k < (*f).K {
                let mut dist: vl_ikmacc_t = 0 as libc::c_int;
                i = 0 as libc::c_int as vl_uindex;
                while i < (*f).M {
                    let mut delta: vl_ikmacc_t = *data
                        .offset(j.wrapping_mul((*f).M).wrapping_add(i) as isize)
                        as libc::c_int
                        - *((*f).centers)
                            .offset(k.wrapping_mul((*f).M).wrapping_add(i) as isize);
                    dist += delta * delta;
                    i = i.wrapping_add(1);
                }
                if best == -(1 as libc::c_int) as libc::c_longlong || dist < best_dist {
                    best = k as vl_index;
                    best_dist = dist;
                }
                k = k.wrapping_add(1);
            }
            if *asgn.offset(j as isize) as libc::c_longlong != best {
                *asgn.offset(j as isize) = best as vl_uint32;
                done = 0 as libc::c_int;
            }
            j = j.wrapping_add(1);
        }
        if done != 0 || iter == (*f).max_niters {
            break;
        }
        memset(
            (*f).centers as *mut libc::c_void,
            0 as libc::c_int,
            (::core::mem::size_of::<vl_ikmacc_t>() as libc::c_ulong as libc::c_ulonglong)
                .wrapping_mul((*f).M)
                .wrapping_mul((*f).K) as libc::c_ulong,
        );
        memset(
            counts as *mut libc::c_void,
            0 as libc::c_int,
            (::core::mem::size_of::<vl_uint32>() as libc::c_ulong as libc::c_ulonglong)
                .wrapping_mul((*f).K) as libc::c_ulong,
        );
        j = 0 as libc::c_int as vl_uindex;
        while j < N {
            let mut this_center: vl_uindex = *asgn.offset(j as isize) as vl_uindex;
            let ref mut fresh0 = *counts.offset(this_center as isize);
            *fresh0 = (*fresh0).wrapping_add(1);
            i = 0 as libc::c_int as vl_uindex;
            while i < (*f).M {
                let ref mut fresh1 = *((*f).centers)
                    .offset(this_center.wrapping_mul((*f).M).wrapping_add(i) as isize);
                *fresh1
                    += *data.offset(j.wrapping_mul((*f).M).wrapping_add(i) as isize)
                        as libc::c_int;
                i = i.wrapping_add(1);
            }
            j = j.wrapping_add(1);
        }
        k = 0 as libc::c_int as vl_uindex;
        while k < (*f).K {
            let mut n: vl_index = *counts.offset(k as isize) as vl_index;
            if n > 0xffffff as libc::c_int as libc::c_longlong {
                err = 1 as libc::c_int;
            }
            if n > 0 as libc::c_int as libc::c_longlong {
                i = 0 as libc::c_int as vl_uindex;
                while i < (*f).M {
                    let ref mut fresh2 = *((*f).centers)
                        .offset(k.wrapping_mul((*f).M).wrapping_add(i) as isize);
                    *fresh2 = (*fresh2 as libc::c_longlong / n) as vl_ikmacc_t;
                    i = i.wrapping_add(1);
                }
            }
            k = k.wrapping_add(1);
        }
        iter = iter.wrapping_add(1);
    }
    vl_free(counts as *mut libc::c_void);
    vl_free(asgn as *mut libc::c_void);
    return err;
}
unsafe extern "C" fn vl_ikm_push_lloyd(
    mut f: *mut VlIKMFilt,
    mut asgn: *mut vl_uint32,
    mut data: *const vl_uint8,
    mut N: vl_size,
) {
    let mut j: vl_uindex = 0;
    j = 0 as libc::c_int as vl_uindex;
    while j < N {
        *asgn
            .offset(
                j as isize,
            ) = vl_ikm_push_one(
            (*f).centers,
            data.offset(j.wrapping_mul((*f).M) as isize),
            (*f).M,
            (*f).K,
        );
        j = j.wrapping_add(1);
    }
}
unsafe extern "C" fn vl_ikm_train_elkan(
    mut f: *mut VlIKMFilt,
    mut data: *const vl_uint8,
    mut N: vl_size,
) -> libc::c_int {
    let mut i: vl_uindex = 0;
    let mut pass: vl_uindex = 0;
    let mut c: vl_uindex = 0;
    let mut cp: vl_uindex = 0;
    let mut x: vl_uindex = 0;
    let mut cx: vl_uindex = 0;
    let mut dist_calc: vl_size = 0 as libc::c_int as vl_size;
    let mut dist: vl_ikmacc_t = 0;
    let mut m_pt: *mut vl_ikmacc_t = vl_malloc(
        (::core::mem::size_of::<vl_ikmacc_t>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*f).M)
            .wrapping_mul((*f).K) as size_t,
    ) as *mut vl_ikmacc_t;
    let mut u_pt: *mut vl_ikmacc_t = vl_malloc(
        (::core::mem::size_of::<vl_ikmacc_t>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(N) as size_t,
    ) as *mut vl_ikmacc_t;
    let mut r_pt: *mut libc::c_char = vl_malloc(
        ((::core::mem::size_of::<libc::c_char>() as libc::c_ulong)
            .wrapping_mul(1 as libc::c_int as libc::c_ulong) as libc::c_ulonglong)
            .wrapping_mul(N) as size_t,
    ) as *mut libc::c_char;
    let mut s_pt: *mut vl_ikmacc_t = vl_malloc(
        (::core::mem::size_of::<vl_ikmacc_t>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*f).K) as size_t,
    ) as *mut vl_ikmacc_t;
    let mut l_pt: *mut vl_ikmacc_t = vl_malloc(
        (::core::mem::size_of::<vl_ikmacc_t>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(N)
            .wrapping_mul((*f).K) as size_t,
    ) as *mut vl_ikmacc_t;
    let mut d_pt: *mut vl_ikmacc_t = (*f).inter_dist;
    let mut asgn: *mut vl_uint32 = vl_malloc(
        (::core::mem::size_of::<vl_uint32>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(N) as size_t,
    ) as *mut vl_uint32;
    let mut counts: *mut vl_uint32 = vl_malloc(
        (::core::mem::size_of::<vl_uint32>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(N) as size_t,
    ) as *mut vl_uint32;
    let mut done: libc::c_int = 0 as libc::c_int;
    vl_ikm_elkan_update_inter_dist(f);
    memset(
        l_pt as *mut libc::c_void,
        0 as libc::c_int,
        (::core::mem::size_of::<vl_ikmacc_t>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(N)
            .wrapping_mul((*f).K) as libc::c_ulong,
    );
    memset(
        u_pt as *mut libc::c_void,
        0 as libc::c_int,
        (::core::mem::size_of::<vl_ikmacc_t>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(N) as libc::c_ulong,
    );
    memset(
        r_pt as *mut libc::c_void,
        0 as libc::c_int,
        (::core::mem::size_of::<libc::c_char>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(N) as libc::c_ulong,
    );
    x = 0 as libc::c_int as vl_uindex;
    while x < N {
        let mut best_dist: vl_ikmacc_t = 0;
        dist_calc = dist_calc.wrapping_add(1);
        dist = 0 as libc::c_int;
        i = 0 as libc::c_int as vl_uindex;
        while i < (*f).M {
            let mut delta: vl_ikmacc_t = *data
                .offset(x.wrapping_mul((*f).M).wrapping_add(i) as isize) as vl_ikmacc_t
                - *((*f).centers).offset(i as isize);
            dist += delta * delta;
            i = i.wrapping_add(1);
        }
        cx = 0 as libc::c_int as vl_uindex;
        best_dist = dist;
        *l_pt.offset(x as isize) = dist;
        c = 1 as libc::c_int as vl_uindex;
        while c < (*f).K {
            if *d_pt.offset(((*f).K).wrapping_mul(cx).wrapping_add(c) as isize)
                < best_dist
            {
                dist_calc = dist_calc.wrapping_add(1);
                dist = 0 as libc::c_int;
                i = 0 as libc::c_int as vl_uindex;
                while i < (*f).M {
                    let mut delta_0: vl_ikmacc_t = *data
                        .offset(x.wrapping_mul((*f).M).wrapping_add(i) as isize)
                        as vl_ikmacc_t
                        - *((*f).centers)
                            .offset(c.wrapping_mul((*f).M).wrapping_add(i) as isize);
                    dist += delta_0 * delta_0;
                    i = i.wrapping_add(1);
                }
                *l_pt.offset(N.wrapping_mul(c).wrapping_add(x) as isize) = dist;
                if dist < best_dist {
                    best_dist = dist;
                    cx = c;
                }
            }
            c = c.wrapping_add(1);
        }
        *asgn.offset(x as isize) = cx as vl_uint32;
        *u_pt.offset(x as isize) = best_dist;
        x = x.wrapping_add(1);
    }
    pass = 0 as libc::c_int as vl_uindex;
    loop {
        memset(
            m_pt as *mut libc::c_void,
            0 as libc::c_int,
            (::core::mem::size_of::<vl_ikmacc_t>() as libc::c_ulong as libc::c_ulonglong)
                .wrapping_mul((*f).M)
                .wrapping_mul((*f).K) as libc::c_ulong,
        );
        memset(
            counts as *mut libc::c_void,
            0 as libc::c_int,
            (::core::mem::size_of::<vl_uint32>() as libc::c_ulong as libc::c_ulonglong)
                .wrapping_mul((*f).K) as libc::c_ulong,
        );
        x = 0 as libc::c_int as vl_uindex;
        while x < N {
            let mut cx_0: libc::c_int = *asgn.offset(x as isize) as libc::c_int;
            let ref mut fresh3 = *counts.offset(cx_0 as isize);
            *fresh3 = (*fresh3).wrapping_add(1);
            i = 0 as libc::c_int as vl_uindex;
            while i < (*f).M {
                let ref mut fresh4 = *m_pt
                    .offset(
                        (cx_0 as libc::c_ulonglong).wrapping_mul((*f).M).wrapping_add(i)
                            as isize,
                    );
                *fresh4
                    += *data.offset(x.wrapping_mul((*f).M).wrapping_add(i) as isize)
                        as libc::c_int;
                i = i.wrapping_add(1);
            }
            x = x.wrapping_add(1);
        }
        c = 0 as libc::c_int as vl_uindex;
        while c < (*f).K {
            let mut n: vl_ikmacc_t = *counts.offset(c as isize) as vl_ikmacc_t;
            if n > 0 as libc::c_int {
                i = 0 as libc::c_int as vl_uindex;
                while i < (*f).M {
                    let ref mut fresh5 = *m_pt
                        .offset(c.wrapping_mul((*f).M).wrapping_add(i) as isize);
                    *fresh5 /= n;
                    i = i.wrapping_add(1);
                }
            } else {
                i = 0 as libc::c_int as vl_uindex;
                while i < (*f).M {
                    i = i.wrapping_add(1);
                }
            }
            c = c.wrapping_add(1);
        }
        c = 0 as libc::c_int as vl_uindex;
        while c < (*f).K {
            dist_calc = dist_calc.wrapping_add(1);
            dist = 0 as libc::c_int;
            i = 0 as libc::c_int as vl_uindex;
            while i < (*f).M {
                let mut delta_1: vl_ikmacc_t = *m_pt
                    .offset(c.wrapping_mul((*f).M).wrapping_add(i) as isize)
                    - *((*f).centers)
                        .offset(c.wrapping_mul((*f).M).wrapping_add(i) as isize);
                *((*f).centers)
                    .offset(
                        c.wrapping_mul((*f).M).wrapping_add(i) as isize,
                    ) = *m_pt.offset(c.wrapping_mul((*f).M).wrapping_add(i) as isize);
                dist += delta_1 * delta_1;
                i = i.wrapping_add(1);
            }
            x = 0 as libc::c_int as vl_uindex;
            while x < N {
                let mut lxc: vl_ikmacc_t = *l_pt
                    .offset(c.wrapping_mul(N).wrapping_add(x) as isize);
                let mut cx_1: vl_uindex = *asgn.offset(x as isize) as libc::c_int
                    as vl_uindex;
                if dist < lxc {
                    lxc = ((lxc + dist) as libc::c_ulonglong)
                        .wrapping_sub(
                            (2 as libc::c_int as libc::c_ulonglong)
                                .wrapping_mul(
                                    (vl_fast_sqrt_ui64(lxc as vl_uint64))
                                        .wrapping_add(1 as libc::c_int as libc::c_ulonglong),
                                )
                                .wrapping_mul(
                                    (vl_fast_sqrt_ui64(dist as vl_uint64))
                                        .wrapping_add(1 as libc::c_int as libc::c_ulonglong),
                                ),
                        ) as vl_ikmacc_t;
                } else {
                    lxc = 0 as libc::c_int;
                }
                *l_pt.offset(c.wrapping_mul(N).wrapping_add(x) as isize) = lxc;
                if c == cx_1 {
                    let mut ux: vl_ikmacc_t = *u_pt.offset(x as isize);
                    *u_pt
                        .offset(
                            x as isize,
                        ) = ((ux + dist) as libc::c_ulonglong)
                        .wrapping_add(
                            (2 as libc::c_int as libc::c_ulonglong)
                                .wrapping_mul(
                                    (vl_fast_sqrt_ui64(ux as vl_uint64))
                                        .wrapping_add(1 as libc::c_int as libc::c_ulonglong),
                                )
                                .wrapping_mul(
                                    (vl_fast_sqrt_ui64(dist as vl_uint64))
                                        .wrapping_add(1 as libc::c_int as libc::c_ulonglong),
                                ),
                        ) as vl_ikmacc_t;
                    *r_pt.offset(x as isize) = 1 as libc::c_int as libc::c_char;
                }
                x = x.wrapping_add(1);
            }
            c = c.wrapping_add(1);
        }
        c = 0 as libc::c_int as vl_uindex;
        while c < (*f).K {
            cp = 0 as libc::c_int as vl_uindex;
            while cp < (*f).K {
                dist = 0 as libc::c_int;
                if c != cp {
                    dist_calc = dist_calc.wrapping_add(1);
                    i = 0 as libc::c_int as vl_uindex;
                    while i < (*f).M {
                        let mut delta_2: vl_ikmacc_t = *((*f).centers)
                            .offset(cp.wrapping_mul((*f).M).wrapping_add(i) as isize)
                            - *((*f).centers)
                                .offset(c.wrapping_mul((*f).M).wrapping_add(i) as isize);
                        dist += delta_2 * delta_2;
                        i = i.wrapping_add(1);
                    }
                }
                let ref mut fresh6 = *d_pt
                    .offset(cp.wrapping_mul((*f).K).wrapping_add(c) as isize);
                *fresh6 = dist >> 2 as libc::c_int;
                *d_pt.offset(c.wrapping_mul((*f).K).wrapping_add(cp) as isize) = *fresh6;
                cp = cp.wrapping_add(1);
            }
            c = c.wrapping_add(1);
        }
        c = 0 as libc::c_int as vl_uindex;
        while c < (*f).K {
            let mut best_dist_0: vl_ikmacc_t = 0x7fffffff as libc::c_ulong
                as vl_ikmacc_t;
            cp = 0 as libc::c_int as vl_uindex;
            while cp < (*f).K {
                dist = *d_pt.offset(c.wrapping_mul((*f).K).wrapping_add(cp) as isize);
                if c != cp && dist < best_dist_0 {
                    best_dist_0 = dist;
                }
                cp = cp.wrapping_add(1);
            }
            *s_pt.offset(c as isize) = best_dist_0 >> 2 as libc::c_int;
            c = c.wrapping_add(1);
        }
        done = 1 as libc::c_int;
        x = 0 as libc::c_int as vl_uindex;
        while x < N {
            let mut cx_2: vl_uindex = *asgn.offset(x as isize) as vl_uindex;
            let mut ux_0: vl_ikmacc_t = *u_pt.offset(x as isize);
            if !(ux_0 <= *s_pt.offset(cx_2 as isize)) {
                let mut current_block_123: u64;
                c = 0 as libc::c_int as vl_uindex;
                while c < (*f).K {
                    let mut dist_0: vl_ikmacc_t = 0 as libc::c_int;
                    if !(c == cx_2
                        || ux_0
                            <= *l_pt.offset(N.wrapping_mul(c).wrapping_add(x) as isize)
                        || ux_0
                            <= *d_pt
                                .offset(
                                    ((*f).K).wrapping_mul(c).wrapping_add(cx_2) as isize,
                                ))
                    {
                        if *r_pt.offset(x as isize) != 0 {
                            dist_calc = dist_calc.wrapping_add(1);
                            dist_0 = 0 as libc::c_int;
                            i = 0 as libc::c_int as vl_uindex;
                            while i < (*f).M {
                                let mut delta_3: vl_ikmacc_t = *data
                                    .offset(x.wrapping_mul((*f).M).wrapping_add(i) as isize)
                                    as vl_ikmacc_t
                                    - *((*f).centers)
                                        .offset(cx_2.wrapping_mul((*f).M).wrapping_add(i) as isize);
                                dist_0 += delta_3 * delta_3;
                                i = i.wrapping_add(1);
                            }
                            let ref mut fresh7 = *u_pt.offset(x as isize);
                            *fresh7 = dist_0;
                            ux_0 = *fresh7;
                            *r_pt.offset(x as isize) = 0 as libc::c_int as libc::c_char;
                            if ux_0
                                <= *l_pt.offset(N.wrapping_mul(c).wrapping_add(x) as isize)
                                || ux_0
                                    <= *d_pt
                                        .offset(
                                            ((*f).K).wrapping_mul(c).wrapping_add(cx_2) as isize,
                                        )
                            {
                                current_block_123 = 722119776535234387;
                            } else {
                                current_block_123 = 1677945370889843322;
                            }
                        } else {
                            current_block_123 = 1677945370889843322;
                        }
                        match current_block_123 {
                            722119776535234387 => {}
                            _ => {
                                dist_calc = dist_calc.wrapping_add(1);
                                dist_0 = 0 as libc::c_int;
                                i = 0 as libc::c_int as vl_uindex;
                                while i < (*f).M {
                                    let mut delta_4: vl_ikmacc_t = *data
                                        .offset(x.wrapping_mul((*f).M).wrapping_add(i) as isize)
                                        as vl_ikmacc_t
                                        - *((*f).centers)
                                            .offset(c.wrapping_mul((*f).M).wrapping_add(i) as isize);
                                    dist_0 += delta_4 * delta_4;
                                    i = i.wrapping_add(1);
                                }
                                *l_pt
                                    .offset(
                                        N.wrapping_mul(c).wrapping_add(x) as isize,
                                    ) = dist_0;
                                if dist_0 < ux_0 {
                                    let ref mut fresh8 = *u_pt.offset(x as isize);
                                    *fresh8 = dist_0;
                                    ux_0 = *fresh8;
                                    *asgn.offset(x as isize) = c as vl_uint32;
                                    done = 0 as libc::c_int;
                                }
                            }
                        }
                    }
                    c = c.wrapping_add(1);
                }
            }
            x = x.wrapping_add(1);
        }
        if done != 0 || pass == (*f).max_niters {
            break;
        }
        pass = pass.wrapping_add(1);
    }
    vl_free(counts as *mut libc::c_void);
    vl_free(asgn as *mut libc::c_void);
    vl_free(l_pt as *mut libc::c_void);
    vl_free(s_pt as *mut libc::c_void);
    vl_free(r_pt as *mut libc::c_void);
    vl_free(u_pt as *mut libc::c_void);
    vl_free(m_pt as *mut libc::c_void);
    if (*f).verb != 0 {
        (Some(
            ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                .expect("non-null function pointer"),
        ))
            .expect(
                "non-null function pointer",
            )(
            b"ikm: Elkan algorithm: total iterations: %d\n\0" as *const u8
                as *const libc::c_char,
            pass,
        );
        (Some(
            ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                .expect("non-null function pointer"),
        ))
            .expect(
                "non-null function pointer",
            )(
            b"ikm: Elkan algorithm: distance calculations: %d (speedup: %.2f)\n\0"
                as *const u8 as *const libc::c_char,
            dist_calc,
            (N as libc::c_float * (*f).K as libc::c_float
                * pass.wrapping_add(2 as libc::c_int as libc::c_ulonglong)
                    as libc::c_float / dist_calc as libc::c_float
                - 1 as libc::c_int as libc::c_float) as libc::c_double,
        );
    }
    return 0 as libc::c_int;
}
#[inline]
unsafe extern "C" fn vl_fast_sqrt_ui64(mut x: vl_uint64) -> vl_uint64 {
    let mut y: vl_uint64 = 0 as libc::c_int as vl_uint64;
    let mut tmp: vl_uint64 = 0 as libc::c_int as vl_uint64;
    let mut twice_k: libc::c_int = 0;
    twice_k = (8 as libc::c_int as libc::c_ulong)
        .wrapping_mul(::core::mem::size_of::<vl_uint64>() as libc::c_ulong)
        .wrapping_sub(2 as libc::c_int as libc::c_ulong) as libc::c_int;
    while twice_k >= 0 as libc::c_int {
        y <<= 1 as libc::c_int;
        tmp = (2 as libc::c_int as libc::c_ulonglong)
            .wrapping_mul(y)
            .wrapping_add(1 as libc::c_int as libc::c_ulonglong) << twice_k;
        if x >= tmp {
            x = (x as libc::c_ulonglong).wrapping_sub(tmp) as vl_uint64 as vl_uint64;
            y = (y as libc::c_ulonglong)
                .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as vl_uint64
                as vl_uint64;
        }
        twice_k -= 2 as libc::c_int;
    }
    return y;
}
unsafe extern "C" fn vl_ikm_init_elkan(mut f: *mut VlIKMFilt) {
    if !((*f).inter_dist).is_null() {
        vl_free((*f).inter_dist as *mut libc::c_void);
    }
    (*f)
        .inter_dist = vl_malloc(
        (::core::mem::size_of::<vl_ikmacc_t>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*f).K)
            .wrapping_mul((*f).K) as size_t,
    ) as *mut vl_ikmacc_t;
    vl_ikm_elkan_update_inter_dist(f);
}
unsafe extern "C" fn vl_ikm_elkan_update_inter_dist(mut f: *mut VlIKMFilt) {
    let mut i: vl_uindex = 0;
    let mut k: vl_uindex = 0;
    let mut kp: vl_uindex = 0;
    k = 0 as libc::c_int as vl_uindex;
    while k < (*f).K {
        kp = 0 as libc::c_int as vl_uindex;
        while kp < (*f).K {
            let mut dist: vl_ikmacc_t = 0 as libc::c_int;
            if k != kp {
                i = 0 as libc::c_int as vl_uindex;
                while i < (*f).M {
                    let mut delta: vl_ikmacc_t = *((*f).centers)
                        .offset(kp.wrapping_mul((*f).M).wrapping_add(i) as isize)
                        - *((*f).centers)
                            .offset(k.wrapping_mul((*f).M).wrapping_add(i) as isize);
                    dist += delta * delta;
                    i = i.wrapping_add(1);
                }
            }
            let ref mut fresh9 = *((*f).inter_dist)
                .offset(kp.wrapping_mul((*f).K).wrapping_add(k) as isize);
            *fresh9 = dist >> 2 as libc::c_int;
            *((*f).inter_dist)
                .offset(k.wrapping_mul((*f).K).wrapping_add(kp) as isize) = *fresh9;
            kp = kp.wrapping_add(1);
        }
        k = k.wrapping_add(1);
    }
}
unsafe extern "C" fn vl_ikm_push_elkan(
    mut f: *mut VlIKMFilt,
    mut asgn: *mut vl_uint32,
    mut data: *const vl_uint8,
    mut N: vl_size,
) {
    let mut i: vl_uindex = 0;
    let mut c: vl_uindex = 0;
    let mut cx: vl_uindex = 0;
    let mut x: vl_uindex = 0;
    let mut dist_calc: vl_size = 0 as libc::c_int as vl_size;
    let mut dist: vl_ikmacc_t = 0;
    let mut best_dist: vl_ikmacc_t = 0;
    let mut d_pt: *mut vl_ikmacc_t = (*f).inter_dist;
    x = 0 as libc::c_int as vl_uindex;
    while x < N {
        best_dist = 0x7fffffff as libc::c_ulong as vl_ikmacc_t;
        cx = 0 as libc::c_int as vl_uindex;
        c = 0 as libc::c_int as vl_uindex;
        while c < (*f).K {
            if *d_pt.offset(((*f).K).wrapping_mul(cx).wrapping_add(c) as isize)
                < best_dist
            {
                dist_calc = dist_calc.wrapping_add(1);
                dist = 0 as libc::c_int;
                i = 0 as libc::c_int as vl_uindex;
                while i < (*f).M {
                    let mut delta: vl_ikmacc_t = *data
                        .offset(x.wrapping_mul((*f).M).wrapping_add(i) as isize)
                        as libc::c_int
                        - *((*f).centers)
                            .offset(c.wrapping_mul((*f).M).wrapping_add(i) as isize);
                    dist += delta * delta;
                    i = i.wrapping_add(1);
                }
                if dist < best_dist {
                    best_dist = dist;
                    cx = c;
                }
            }
            c = c.wrapping_add(1);
        }
        *asgn.offset(x as isize) = cx as vl_uint32;
        x = x.wrapping_add(1);
    }
}
