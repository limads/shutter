use ::libc;
extern "C" {
    fn log(_: libc::c_double) -> libc::c_double;
    fn vl_get_printf_func() -> printf_func_t;
    fn vl_free(ptr: *mut libc::c_void);
    fn vl_malloc(n: size_t) -> *mut libc::c_void;
}
pub type vl_uint64 = libc::c_ulonglong;
pub type vl_uint = libc::c_uint;
pub type size_t = libc::c_ulong;
pub type printf_func_t = Option::<
    unsafe extern "C" fn(*const libc::c_char, ...) -> libc::c_int,
>;
#[derive(Copy, Clone)]
#[repr(C)]
pub union C2RustUnnamed {
    pub raw: vl_uint64,
    pub value: libc::c_double,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlAIB {
    pub nodes: *mut vl_uint,
    pub nentries: vl_uint,
    pub beta: *mut libc::c_double,
    pub bidx: *mut vl_uint,
    pub which: *mut vl_uint,
    pub nwhich: vl_uint,
    pub Pcx: *mut libc::c_double,
    pub Px: *mut libc::c_double,
    pub Pc: *mut libc::c_double,
    pub nvalues: vl_uint,
    pub nlabels: vl_uint,
    pub parents: *mut vl_uint,
    pub costs: *mut libc::c_double,
    pub verbosity: vl_uint,
}
pub type VlAIB = _VlAIB;
static mut vl_nan_d: C2RustUnnamed = C2RustUnnamed {
    raw: 0x7ff8000000000000 as libc::c_ulonglong,
};
#[no_mangle]
pub unsafe extern "C" fn vl_aib_normalize_P(
    mut P: *mut libc::c_double,
    mut nelem: vl_uint,
) {
    let mut i: vl_uint = 0;
    let mut sum: libc::c_double = 0 as libc::c_int as libc::c_double;
    i = 0 as libc::c_int as vl_uint;
    while i < nelem {
        sum += *P.offset(i as isize);
        i = i.wrapping_add(1);
    }
    i = 0 as libc::c_int as vl_uint;
    while i < nelem {
        *P.offset(i as isize) /= sum;
        i = i.wrapping_add(1);
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_aib_new_nodelist(mut nentries: vl_uint) -> *mut vl_uint {
    let mut nodelist: *mut vl_uint = vl_malloc(
        (::core::mem::size_of::<vl_uint>() as libc::c_ulong)
            .wrapping_mul(nentries as libc::c_ulong),
    ) as *mut vl_uint;
    let mut n: vl_uint = 0;
    n = 0 as libc::c_int as vl_uint;
    while n < nentries {
        *nodelist.offset(n as isize) = n;
        n = n.wrapping_add(1);
    }
    return nodelist;
}
#[no_mangle]
pub unsafe extern "C" fn vl_aib_new_Px(
    mut Pcx: *mut libc::c_double,
    mut nvalues: vl_uint,
    mut nlabels: vl_uint,
) -> *mut libc::c_double {
    let mut Px: *mut libc::c_double = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong)
            .wrapping_mul(nvalues as libc::c_ulong),
    ) as *mut libc::c_double;
    let mut r: vl_uint = 0;
    let mut c: vl_uint = 0;
    r = 0 as libc::c_int as vl_uint;
    while r < nvalues {
        let mut sum: libc::c_double = 0 as libc::c_int as libc::c_double;
        c = 0 as libc::c_int as vl_uint;
        while c < nlabels {
            sum += *Pcx.offset(r.wrapping_mul(nlabels).wrapping_add(c) as isize);
            c = c.wrapping_add(1);
        }
        *Px.offset(r as isize) = sum;
        r = r.wrapping_add(1);
    }
    return Px;
}
#[no_mangle]
pub unsafe extern "C" fn vl_aib_new_Pc(
    mut Pcx: *mut libc::c_double,
    mut nvalues: vl_uint,
    mut nlabels: vl_uint,
) -> *mut libc::c_double {
    let mut Pc: *mut libc::c_double = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong)
            .wrapping_mul(nlabels as libc::c_ulong),
    ) as *mut libc::c_double;
    let mut r: vl_uint = 0;
    let mut c: vl_uint = 0;
    c = 0 as libc::c_int as vl_uint;
    while c < nlabels {
        let mut sum: libc::c_double = 0 as libc::c_int as libc::c_double;
        r = 0 as libc::c_int as vl_uint;
        while r < nvalues {
            sum += *Pcx.offset(r.wrapping_mul(nlabels).wrapping_add(c) as isize);
            r = r.wrapping_add(1);
        }
        *Pc.offset(c as isize) = sum;
        c = c.wrapping_add(1);
    }
    return Pc;
}
#[no_mangle]
pub unsafe extern "C" fn vl_aib_min_beta(
    mut aib: *mut VlAIB,
    mut besti: *mut vl_uint,
    mut bestj: *mut vl_uint,
    mut minbeta: *mut libc::c_double,
) {
    let mut i: vl_uint = 0;
    *minbeta = *((*aib).beta).offset(0 as libc::c_int as isize);
    *besti = 0 as libc::c_int as vl_uint;
    *bestj = *((*aib).bidx).offset(0 as libc::c_int as isize);
    i = 0 as libc::c_int as vl_uint;
    while i < (*aib).nentries {
        if *((*aib).beta).offset(i as isize) < *minbeta {
            *minbeta = *((*aib).beta).offset(i as isize);
            *besti = i;
            *bestj = *((*aib).bidx).offset(i as isize);
        }
        i = i.wrapping_add(1);
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_aib_merge_nodes(
    mut aib: *mut VlAIB,
    mut i: vl_uint,
    mut j: vl_uint,
    mut new: vl_uint,
) {
    let mut last_entry: vl_uint = ((*aib).nentries)
        .wrapping_sub(1 as libc::c_int as libc::c_uint);
    let mut c: vl_uint = 0;
    let mut n: vl_uint = 0;
    (*aib).nwhich = 0 as libc::c_int as vl_uint;
    if i > j {
        let mut tmp: vl_uint = j;
        j = i;
        i = tmp;
    }
    *((*aib).Px).offset(i as isize) += *((*aib).Px).offset(j as isize);
    *((*aib).beta).offset(i as isize) = 1.7976931348623157e+308f64;
    *((*aib).nodes).offset(i as isize) = new;
    c = 0 as libc::c_int as vl_uint;
    while c < (*aib).nlabels {
        *((*aib).Pcx).offset(i.wrapping_mul((*aib).nlabels).wrapping_add(c) as isize)
            += *((*aib).Pcx)
                .offset(j.wrapping_mul((*aib).nlabels).wrapping_add(c) as isize);
        c = c.wrapping_add(1);
    }
    *((*aib).Px).offset(j as isize) = *((*aib).Px).offset(last_entry as isize);
    *((*aib).beta).offset(j as isize) = *((*aib).beta).offset(last_entry as isize);
    *((*aib).bidx).offset(j as isize) = *((*aib).bidx).offset(last_entry as isize);
    *((*aib).nodes).offset(j as isize) = *((*aib).nodes).offset(last_entry as isize);
    c = 0 as libc::c_int as vl_uint;
    while c < (*aib).nlabels {
        *((*aib).Pcx)
            .offset(
                j.wrapping_mul((*aib).nlabels).wrapping_add(c) as isize,
            ) = *((*aib).Pcx)
            .offset(last_entry.wrapping_mul((*aib).nlabels).wrapping_add(c) as isize);
        c = c.wrapping_add(1);
    }
    (*aib).nentries = ((*aib).nentries).wrapping_sub(1);
    n = 0 as libc::c_int as vl_uint;
    while n < (*aib).nentries {
        if *((*aib).bidx).offset(n as isize) == i
            || *((*aib).bidx).offset(n as isize) == j
        {
            *((*aib).bidx).offset(n as isize) = 0 as libc::c_int as vl_uint;
            *((*aib).beta).offset(n as isize) = 1.7976931348623157e+308f64;
            let fresh0 = (*aib).nwhich;
            (*aib).nwhich = ((*aib).nwhich).wrapping_add(1);
            *((*aib).which).offset(fresh0 as isize) = n;
        } else if *((*aib).bidx).offset(n as isize) == last_entry {
            *((*aib).bidx).offset(n as isize) = j;
        }
        n = n.wrapping_add(1);
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_aib_update_beta(mut aib: *mut VlAIB) {
    let mut i: vl_uint = 0;
    let mut Px: *mut libc::c_double = (*aib).Px;
    let mut Pcx: *mut libc::c_double = (*aib).Pcx;
    let mut tmp: *mut libc::c_double = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong)
            .wrapping_mul((*aib).nentries as libc::c_ulong),
    ) as *mut libc::c_double;
    let mut a: vl_uint = 0;
    let mut b: vl_uint = 0;
    let mut c: vl_uint = 0;
    a = 0 as libc::c_int as vl_uint;
    while a < (*aib).nentries {
        *tmp.offset(a as isize) = 0 as libc::c_int as libc::c_double;
        c = 0 as libc::c_int as vl_uint;
        while c < (*aib).nlabels {
            let mut Pac: libc::c_double = *Pcx
                .offset(a.wrapping_mul((*aib).nlabels).wrapping_add(c) as isize);
            if Pac != 0 as libc::c_int as libc::c_double {
                *tmp.offset(a as isize) += Pac * log(Pac / *Px.offset(a as isize));
            }
            c = c.wrapping_add(1);
        }
        a = a.wrapping_add(1);
    }
    i = 0 as libc::c_int as vl_uint;
    while i < (*aib).nwhich {
        a = *((*aib).which).offset(i as isize);
        b = 0 as libc::c_int as vl_uint;
        while b < (*aib).nentries {
            let mut T1: libc::c_double = 0 as libc::c_int as libc::c_double;
            if !(a == b || *Px.offset(a as isize) == 0 as libc::c_int as libc::c_double
                || *Px.offset(b as isize) == 0 as libc::c_int as libc::c_double)
            {
                T1 = (*Px.offset(a as isize) + *Px.offset(b as isize))
                    * log(*Px.offset(a as isize) + *Px.offset(b as isize));
                T1 += *tmp.offset(a as isize) + *tmp.offset(b as isize);
                c = 0 as libc::c_int as vl_uint;
                while c < (*aib).nlabels {
                    let mut Pac_0: libc::c_double = *Pcx
                        .offset(a.wrapping_mul((*aib).nlabels).wrapping_add(c) as isize);
                    let mut Pbc: libc::c_double = *Pcx
                        .offset(b.wrapping_mul((*aib).nlabels).wrapping_add(c) as isize);
                    if !(Pac_0 == 0 as libc::c_int as libc::c_double
                        && Pbc == 0 as libc::c_int as libc::c_double)
                    {
                        T1 += -((Pac_0 + Pbc) * log(Pac_0 + Pbc));
                    }
                    c = c.wrapping_add(1);
                }
                let mut beta: libc::c_double = T1;
                if beta < *((*aib).beta).offset(a as isize) {
                    *((*aib).beta).offset(a as isize) = beta;
                    *((*aib).bidx).offset(a as isize) = b;
                }
                if beta < *((*aib).beta).offset(b as isize) {
                    *((*aib).beta).offset(b as isize) = beta;
                    *((*aib).bidx).offset(b as isize) = a;
                }
            }
            b = b.wrapping_add(1);
        }
        i = i.wrapping_add(1);
    }
    vl_free(tmp as *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn vl_aib_calculate_information(
    mut aib: *mut VlAIB,
    mut I: *mut libc::c_double,
    mut H: *mut libc::c_double,
) {
    let mut r: vl_uint = 0;
    let mut c: vl_uint = 0;
    *H = 0 as libc::c_int as libc::c_double;
    *I = 0 as libc::c_int as libc::c_double;
    r = 0 as libc::c_int as vl_uint;
    while r < (*aib).nentries {
        if !(*((*aib).Px).offset(r as isize) == 0 as libc::c_int as libc::c_double) {
            *H
                += -log(*((*aib).Px).offset(r as isize))
                    * *((*aib).Px).offset(r as isize);
            c = 0 as libc::c_int as vl_uint;
            while c < (*aib).nlabels {
                if !(*((*aib).Pcx)
                    .offset(r.wrapping_mul((*aib).nlabels).wrapping_add(c) as isize)
                    == 0 as libc::c_int as libc::c_double)
                {
                    *I
                        += *((*aib).Pcx)
                            .offset(
                                r.wrapping_mul((*aib).nlabels).wrapping_add(c) as isize,
                            )
                            * log(
                                *((*aib).Pcx)
                                    .offset(
                                        r.wrapping_mul((*aib).nlabels).wrapping_add(c) as isize,
                                    )
                                    / (*((*aib).Px).offset(r as isize)
                                        * *((*aib).Pc).offset(c as isize)),
                            );
                }
                c = c.wrapping_add(1);
            }
        }
        r = r.wrapping_add(1);
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_aib_new(
    mut Pcx: *mut libc::c_double,
    mut nvalues: vl_uint,
    mut nlabels: vl_uint,
) -> *mut VlAIB {
    let mut aib: *mut VlAIB = vl_malloc(::core::mem::size_of::<VlAIB>() as libc::c_ulong)
        as *mut VlAIB;
    let mut i: vl_uint = 0;
    (*aib).verbosity = 0 as libc::c_int as vl_uint;
    (*aib).Pcx = Pcx;
    (*aib).nvalues = nvalues;
    (*aib).nlabels = nlabels;
    vl_aib_normalize_P((*aib).Pcx, ((*aib).nvalues).wrapping_mul((*aib).nlabels));
    (*aib).Px = vl_aib_new_Px((*aib).Pcx, (*aib).nvalues, (*aib).nlabels);
    (*aib).Pc = vl_aib_new_Pc((*aib).Pcx, (*aib).nvalues, (*aib).nlabels);
    (*aib).nentries = (*aib).nvalues;
    (*aib).nodes = vl_aib_new_nodelist((*aib).nentries);
    (*aib)
        .beta = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong)
            .wrapping_mul((*aib).nentries as libc::c_ulong),
    ) as *mut libc::c_double;
    (*aib)
        .bidx = vl_malloc(
        (::core::mem::size_of::<vl_uint>() as libc::c_ulong)
            .wrapping_mul((*aib).nentries as libc::c_ulong),
    ) as *mut vl_uint;
    i = 0 as libc::c_int as vl_uint;
    while i < (*aib).nentries {
        *((*aib).beta).offset(i as isize) = 1.7976931348623157e+308f64;
        i = i.wrapping_add(1);
    }
    (*aib).nwhich = (*aib).nvalues;
    (*aib).which = vl_aib_new_nodelist((*aib).nwhich);
    (*aib)
        .parents = vl_malloc(
        (::core::mem::size_of::<vl_uint>() as libc::c_ulong)
            .wrapping_mul(
                ((*aib).nvalues)
                    .wrapping_mul(2 as libc::c_int as libc::c_uint)
                    .wrapping_sub(1 as libc::c_int as libc::c_uint) as libc::c_ulong,
            ),
    ) as *mut vl_uint;
    i = 0 as libc::c_int as vl_uint;
    while i
        < (2 as libc::c_int as libc::c_uint)
            .wrapping_mul((*aib).nvalues)
            .wrapping_sub(1 as libc::c_int as libc::c_uint)
    {
        *((*aib).parents)
            .offset(
                i as isize,
            ) = (2 as libc::c_int as libc::c_uint).wrapping_mul((*aib).nvalues);
        i = i.wrapping_add(1);
    }
    (*aib)
        .costs = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong)
            .wrapping_mul(
                ((*aib).nvalues)
                    .wrapping_sub(1 as libc::c_int as libc::c_uint)
                    .wrapping_add(1 as libc::c_int as libc::c_uint) as libc::c_ulong,
            ),
    ) as *mut libc::c_double;
    return aib;
}
#[no_mangle]
pub unsafe extern "C" fn vl_aib_delete(mut aib: *mut VlAIB) {
    if !aib.is_null() {
        if !((*aib).nodes).is_null() {
            vl_free((*aib).nodes as *mut libc::c_void);
        }
        if !((*aib).beta).is_null() {
            vl_free((*aib).beta as *mut libc::c_void);
        }
        if !((*aib).bidx).is_null() {
            vl_free((*aib).bidx as *mut libc::c_void);
        }
        if !((*aib).which).is_null() {
            vl_free((*aib).which as *mut libc::c_void);
        }
        if !((*aib).Px).is_null() {
            vl_free((*aib).Px as *mut libc::c_void);
        }
        if !((*aib).Pc).is_null() {
            vl_free((*aib).Pc as *mut libc::c_void);
        }
        if !((*aib).parents).is_null() {
            vl_free((*aib).parents as *mut libc::c_void);
        }
        if !((*aib).costs).is_null() {
            vl_free((*aib).costs as *mut libc::c_void);
        }
        vl_free(aib as *mut libc::c_void);
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_aib_process(mut aib: *mut VlAIB) {
    let mut i: vl_uint = 0;
    let mut besti: vl_uint = 0;
    let mut bestj: vl_uint = 0;
    let mut newnode: vl_uint = 0;
    let mut nodei: vl_uint = 0;
    let mut nodej: vl_uint = 0;
    let mut I: libc::c_double = 0.;
    let mut H: libc::c_double = 0.;
    let mut minbeta: libc::c_double = 0.;
    vl_aib_calculate_information(aib, &mut I, &mut H);
    *((*aib).costs).offset(0 as libc::c_int as isize) = I;
    i = 0 as libc::c_int as vl_uint;
    while i < ((*aib).nvalues).wrapping_sub(1 as libc::c_int as libc::c_uint) {
        vl_aib_update_beta(aib);
        vl_aib_min_beta(aib, &mut besti, &mut bestj, &mut minbeta);
        if minbeta == 1.7976931348623157e+308f64 {
            break;
        }
        newnode = ((*aib).nvalues).wrapping_add(i);
        nodei = *((*aib).nodes).offset(besti as isize);
        nodej = *((*aib).nodes).offset(bestj as isize);
        *((*aib).parents).offset(nodei as isize) = newnode;
        *((*aib).parents).offset(nodej as isize) = newnode;
        *((*aib).parents).offset(newnode as isize) = 0 as libc::c_int as vl_uint;
        vl_aib_merge_nodes(aib, besti, bestj, newnode);
        vl_aib_calculate_information(aib, &mut I, &mut H);
        *((*aib).costs)
            .offset(i.wrapping_add(1 as libc::c_int as libc::c_uint) as isize) = I;
        if (*aib).verbosity > 0 as libc::c_int as libc::c_uint {
            (Some(
                ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                    .expect("non-null function pointer"),
            ))
                .expect(
                    "non-null function pointer",
                )(
                b"aib: (%5d,%5d)=%5d dE: %10.3g I: %6.4g H: %6.4g updt: %5d\n\0"
                    as *const u8 as *const libc::c_char,
                nodei,
                nodej,
                newnode,
                minbeta,
                I,
                H,
                (*aib).nwhich,
            );
        }
        i = i.wrapping_add(1);
    }
    while i < ((*aib).nvalues).wrapping_sub(1 as libc::c_int as libc::c_uint) {
        *((*aib).costs)
            .offset(
                i.wrapping_add(1 as libc::c_int as libc::c_uint) as isize,
            ) = vl_nan_d.value;
        i = i.wrapping_add(1);
    }
}
