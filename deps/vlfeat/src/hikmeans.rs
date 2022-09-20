use ::libc;
extern "C" {
    fn memcpy(
        _: *mut libc::c_void,
        _: *const libc::c_void,
        _: libc::c_ulong,
    ) -> *mut libc::c_void;
    fn __assert_fail(
        __assertion: *const libc::c_char,
        __file: *const libc::c_char,
        __line: libc::c_uint,
        __function: *const libc::c_char,
    ) -> !;
    fn vl_ikm_set_max_niters(f: *mut VlIKMFilt, max_niters: vl_size);
    fn vl_ikm_set_verbosity(f: *mut VlIKMFilt, verb: libc::c_int);
    fn vl_ikm_get_K(f: *const VlIKMFilt) -> vl_size;
    fn vl_ikm_push(
        f: *mut VlIKMFilt,
        asgn: *mut vl_uint32,
        data: *const vl_uint8,
        N: vl_size,
    );
    fn vl_ikm_train(f: *mut VlIKMFilt, data: *const vl_uint8, N: vl_size) -> libc::c_int;
    fn vl_ikm_init_rand_data(
        f: *mut VlIKMFilt,
        data: *const vl_uint8,
        M: vl_size,
        N: vl_size,
        K: vl_size,
    );
    fn vl_ikm_new(method: libc::c_int) -> *mut VlIKMFilt;
    fn vl_ikm_delete(f: *mut VlIKMFilt);
    fn vl_malloc(n: size_t) -> *mut libc::c_void;
    fn vl_calloc(n: size_t, size: size_t) -> *mut libc::c_void;
    fn vl_free(ptr: *mut libc::c_void);
    fn vl_get_printf_func() -> printf_func_t;
}
pub type size_t = libc::c_ulong;
pub type vl_int32 = libc::c_int;
pub type vl_uint64 = libc::c_ulonglong;
pub type vl_uint32 = libc::c_uint;
pub type vl_uint8 = libc::c_uchar;
pub type vl_size = vl_uint64;
pub type vl_uindex = vl_uint64;
pub type printf_func_t = Option::<
    unsafe extern "C" fn(*const libc::c_char, ...) -> libc::c_int,
>;
pub type vl_ikmacc_t = vl_int32;
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
pub struct _VlHIKMNode {
    pub filter: *mut VlIKMFilt,
    pub children: *mut *mut _VlHIKMNode,
}
pub type VlHIKMNode = _VlHIKMNode;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlHIKMTree {
    pub M: vl_size,
    pub K: vl_size,
    pub depth: vl_size,
    pub max_niters: vl_size,
    pub method: libc::c_int,
    pub verb: libc::c_int,
    pub root: *mut VlHIKMNode,
}
pub type VlHIKMTree = _VlHIKMTree;
#[no_mangle]
pub unsafe extern "C" fn vl_hikm_copy_subset(
    mut data: *const vl_uint8,
    mut ids: *mut vl_uint32,
    mut N: vl_size,
    mut M: vl_size,
    mut id: vl_uint32,
    mut N2: *mut vl_size,
) -> *mut vl_uint8 {
    let mut i: vl_uindex = 0;
    let mut count: vl_size = 0 as libc::c_int as vl_size;
    i = 0 as libc::c_int as vl_uindex;
    while i < N {
        if *ids.offset(i as isize) == id {
            count = count.wrapping_add(1);
        }
        i = i.wrapping_add(1);
    }
    *N2 = count;
    let mut new_data: *mut vl_uint8 = vl_malloc(
        (::core::mem::size_of::<vl_uint8>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(M)
            .wrapping_mul(count) as size_t,
    ) as *mut vl_uint8;
    count = 0 as libc::c_int as vl_size;
    i = 0 as libc::c_int as vl_uindex;
    while i < N {
        if *ids.offset(i as isize) == id {
            memcpy(
                new_data.offset(count.wrapping_mul(M) as isize) as *mut libc::c_void,
                data.offset(i.wrapping_mul(M) as isize) as *const libc::c_void,
                (::core::mem::size_of::<vl_uint8>() as libc::c_ulong
                    as libc::c_ulonglong)
                    .wrapping_mul(M) as libc::c_ulong,
            );
            count = count.wrapping_add(1);
        }
        i = i.wrapping_add(1);
    }
    *N2 = count;
    return new_data;
}
unsafe extern "C" fn xmeans(
    mut tree: *mut VlHIKMTree,
    mut data: *const vl_uint8,
    mut N: vl_size,
    mut K: vl_size,
    mut height: vl_size,
) -> *mut VlHIKMNode {
    let mut node: *mut VlHIKMNode = vl_malloc(
        ::core::mem::size_of::<VlHIKMNode>() as libc::c_ulong,
    ) as *mut VlHIKMNode;
    let mut ids: *mut vl_uint32 = vl_malloc(
        (::core::mem::size_of::<vl_uint32>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(N) as size_t,
    ) as *mut vl_uint32;
    (*node).filter = vl_ikm_new((*tree).method);
    (*node)
        .children = (if height == 1 as libc::c_int as libc::c_ulonglong {
        0 as *mut libc::c_void
    } else {
        vl_malloc(
            (::core::mem::size_of::<*mut _VlHIKMNode>() as libc::c_ulong
                as libc::c_ulonglong)
                .wrapping_mul(K) as size_t,
        )
    }) as *mut *mut _VlHIKMNode;
    vl_ikm_set_max_niters((*node).filter, (*tree).max_niters);
    vl_ikm_set_verbosity((*node).filter, (*tree).verb - 1 as libc::c_int);
    vl_ikm_init_rand_data((*node).filter, data, (*tree).M, N, K);
    vl_ikm_train((*node).filter, data, N);
    vl_ikm_push((*node).filter, ids, data, N);
    if height > 1 as libc::c_int as libc::c_ulonglong {
        let mut k: vl_uindex = 0;
        k = 0 as libc::c_int as vl_uindex;
        while k < K {
            let mut partition_N: vl_size = 0;
            let mut partition_K: vl_size = 0;
            let mut partition: *mut vl_uint8 = 0 as *mut vl_uint8;
            partition = vl_hikm_copy_subset(
                data,
                ids,
                N,
                (*tree).M,
                k as vl_uint32,
                &mut partition_N,
            );
            partition_K = if K < partition_N { K } else { partition_N };
            let ref mut fresh0 = *((*node).children).offset(k as isize);
            *fresh0 = xmeans(
                tree,
                partition,
                partition_N,
                partition_K,
                height.wrapping_sub(1 as libc::c_int as libc::c_ulonglong),
            );
            vl_free(partition as *mut libc::c_void);
            if (*tree).verb > (*tree).depth as libc::c_int - height as libc::c_int {
                (Some(
                    ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                        .expect("non-null function pointer"),
                ))
                    .expect(
                        "non-null function pointer",
                    )(
                    b"hikmeans: branch at depth %d: %6.1f %% completed\n\0" as *const u8
                        as *const libc::c_char,
                    ((*tree).depth).wrapping_sub(height),
                    k.wrapping_add(1 as libc::c_int as libc::c_ulonglong)
                        as libc::c_double / K as libc::c_double
                        * 100 as libc::c_int as libc::c_double,
                );
            }
            k = k.wrapping_add(1);
        }
    }
    vl_free(ids as *mut libc::c_void);
    return node;
}
unsafe extern "C" fn xdelete(mut node: *mut VlHIKMNode) {
    if !node.is_null() {
        let mut k: vl_uindex = 0;
        if !((*node).children).is_null() {
            k = 0 as libc::c_int as vl_uindex;
            while k < vl_ikm_get_K((*node).filter) {
                xdelete(*((*node).children).offset(k as isize));
                k = k.wrapping_add(1);
            }
            vl_free((*node).children as *mut libc::c_void);
        }
        if !((*node).filter).is_null() {
            vl_ikm_delete((*node).filter);
        }
        vl_free(node as *mut libc::c_void);
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_hikm_new(mut method: libc::c_int) -> *mut VlHIKMTree {
    let mut f: *mut VlHIKMTree = vl_calloc(
        ::core::mem::size_of::<VlHIKMTree>() as libc::c_ulong,
        1 as libc::c_int as size_t,
    ) as *mut VlHIKMTree;
    (*f).max_niters = 200 as libc::c_int as vl_size;
    (*f).method = method;
    return f;
}
#[no_mangle]
pub unsafe extern "C" fn vl_hikm_delete(mut f: *mut VlHIKMTree) {
    if !f.is_null() {
        xdelete((*f).root);
        vl_free(f as *mut libc::c_void);
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_hikm_init(
    mut f: *mut VlHIKMTree,
    mut M: vl_size,
    mut K: vl_size,
    mut depth: vl_size,
) {
    if depth > 0 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"depth > 0\0" as *const u8 as *const libc::c_char,
            b"vl/hikmeans.c\0" as *const u8 as *const libc::c_char,
            219 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 59],
                &[libc::c_char; 59],
            >(b"void vl_hikm_init(VlHIKMTree *, vl_size, vl_size, vl_size)\0"))
                .as_ptr(),
        );
    }
    if M > 0 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"M > 0\0" as *const u8 as *const libc::c_char,
            b"vl/hikmeans.c\0" as *const u8 as *const libc::c_char,
            220 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 59],
                &[libc::c_char; 59],
            >(b"void vl_hikm_init(VlHIKMTree *, vl_size, vl_size, vl_size)\0"))
                .as_ptr(),
        );
    }
    if K > 0 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"K > 0\0" as *const u8 as *const libc::c_char,
            b"vl/hikmeans.c\0" as *const u8 as *const libc::c_char,
            221 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 59],
                &[libc::c_char; 59],
            >(b"void vl_hikm_init(VlHIKMTree *, vl_size, vl_size, vl_size)\0"))
                .as_ptr(),
        );
    }
    xdelete((*f).root);
    (*f).root = 0 as *mut VlHIKMNode;
    (*f).M = M;
    (*f).K = K;
    (*f).depth = depth;
}
#[no_mangle]
pub unsafe extern "C" fn vl_hikm_train(
    mut f: *mut VlHIKMTree,
    mut data: *const vl_uint8,
    mut N: vl_size,
) {
    (*f).root = xmeans(f, data, N, if (*f).K < N { (*f).K } else { N }, (*f).depth);
}
#[no_mangle]
pub unsafe extern "C" fn vl_hikm_push(
    mut f: *mut VlHIKMTree,
    mut asgn: *mut vl_uint32,
    mut data: *const vl_uint8,
    mut N: vl_size,
) {
    let mut i: vl_uindex = 0;
    let mut d: vl_uindex = 0;
    let mut M: vl_size = vl_hikm_get_ndims(f);
    let mut depth: vl_size = vl_hikm_get_depth(f);
    i = 0 as libc::c_int as vl_uindex;
    while i < N {
        let mut node: *mut VlHIKMNode = (*f).root;
        d = 0 as libc::c_int as vl_uindex;
        while !node.is_null() {
            let mut best: vl_uint32 = 0;
            vl_ikm_push(
                (*node).filter,
                &mut best,
                data.offset(i.wrapping_mul(M) as isize),
                1 as libc::c_int as vl_size,
            );
            *asgn.offset(i.wrapping_mul(depth).wrapping_add(d) as isize) = best;
            d = d.wrapping_add(1);
            if ((*node).children).is_null() {
                break;
            }
            node = *((*node).children).offset(best as isize);
        }
        i = i.wrapping_add(1);
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_hikm_get_ndims(mut f: *const VlHIKMTree) -> vl_size {
    return (*f).M;
}
#[no_mangle]
pub unsafe extern "C" fn vl_hikm_get_K(mut f: *const VlHIKMTree) -> vl_size {
    return (*f).K;
}
#[no_mangle]
pub unsafe extern "C" fn vl_hikm_get_depth(mut f: *const VlHIKMTree) -> vl_size {
    return (*f).depth;
}
#[no_mangle]
pub unsafe extern "C" fn vl_hikm_get_verbosity(mut f: *const VlHIKMTree) -> libc::c_int {
    return (*f).verb;
}
#[no_mangle]
pub unsafe extern "C" fn vl_hikm_get_max_niters(mut f: *const VlHIKMTree) -> vl_size {
    return (*f).max_niters;
}
#[no_mangle]
pub unsafe extern "C" fn vl_hikm_get_root(
    mut f: *const VlHIKMTree,
) -> *const VlHIKMNode {
    return (*f).root;
}
#[no_mangle]
pub unsafe extern "C" fn vl_hikm_set_verbosity(
    mut f: *mut VlHIKMTree,
    mut verb: libc::c_int,
) {
    (*f).verb = verb;
}
#[no_mangle]
pub unsafe extern "C" fn vl_hikm_set_max_niters(
    mut f: *mut VlHIKMTree,
    mut max_niters: libc::c_int,
) {
    (*f).max_niters = max_niters as vl_size;
}
