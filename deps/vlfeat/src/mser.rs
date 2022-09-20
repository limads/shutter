use ::libc;
extern "C" {
    fn vl_malloc(n: size_t) -> *mut libc::c_void;
    fn vl_calloc(n: size_t, size: size_t) -> *mut libc::c_void;
    fn vl_free(ptr: *mut libc::c_void);
    fn memset(
        _: *mut libc::c_void,
        _: libc::c_int,
        _: libc::c_ulong,
    ) -> *mut libc::c_void;
}
pub type vl_uint8 = libc::c_uchar;
pub type vl_uint = libc::c_uint;
pub type vl_bool = libc::c_int;
pub type size_t = libc::c_ulong;
pub type vl_mser_pix = vl_uint8;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlMserFilt {
    pub ndims: libc::c_int,
    pub dims: *mut libc::c_int,
    pub nel: libc::c_int,
    pub subs: *mut libc::c_int,
    pub dsubs: *mut libc::c_int,
    pub strides: *mut libc::c_int,
    pub perm: *mut vl_uint,
    pub joins: *mut vl_uint,
    pub njoins: libc::c_int,
    pub r: *mut VlMserReg,
    pub er: *mut VlMserExtrReg,
    pub mer: *mut vl_uint,
    pub ner: libc::c_int,
    pub nmer: libc::c_int,
    pub rer: libc::c_int,
    pub rmer: libc::c_int,
    pub acc: *mut libc::c_float,
    pub ell: *mut libc::c_float,
    pub rell: libc::c_int,
    pub nell: libc::c_int,
    pub dof: libc::c_int,
    pub verbose: vl_bool,
    pub delta: libc::c_int,
    pub max_area: libc::c_double,
    pub min_area: libc::c_double,
    pub max_variation: libc::c_double,
    pub min_diversity: libc::c_double,
    pub stats: VlMserStats,
}
pub type VlMserStats = _VlMserStats;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlMserStats {
    pub num_extremal: libc::c_int,
    pub num_unstable: libc::c_int,
    pub num_abs_unstable: libc::c_int,
    pub num_too_big: libc::c_int,
    pub num_too_small: libc::c_int,
    pub num_duplicates: libc::c_int,
}
pub type VlMserExtrReg = _VlMserExtrReg;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlMserExtrReg {
    pub parent: libc::c_int,
    pub index: libc::c_int,
    pub value: vl_mser_pix,
    pub shortcut: vl_uint,
    pub area: vl_uint,
    pub variation: libc::c_float,
    pub max_stable: vl_uint,
}
pub type VlMserReg = _VlMserReg;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlMserReg {
    pub parent: vl_uint,
    pub shortcut: vl_uint,
    pub height: vl_uint,
    pub area: vl_uint,
}
pub type VlMserFilt = _VlMserFilt;
pub type vl_mser_acc = libc::c_float;
#[inline]
unsafe extern "C" fn adv(
    mut ndims: libc::c_int,
    mut dims: *const libc::c_int,
    mut subs: *mut libc::c_int,
) {
    let mut d: libc::c_int = 0 as libc::c_int;
    while d < ndims {
        let ref mut fresh0 = *subs.offset(d as isize);
        *fresh0 += 1;
        if *fresh0 < *dims.offset(d as isize) {
            return;
        }
        let fresh1 = d;
        d = d + 1;
        *subs.offset(fresh1 as isize) = 0 as libc::c_int;
    }
}
#[inline]
unsafe extern "C" fn climb(mut r: *mut VlMserReg, mut idx: vl_uint) -> vl_uint {
    let mut prev_idx: vl_uint = idx;
    let mut next_idx: vl_uint = 0;
    let mut root_idx: vl_uint = 0;
    loop {
        next_idx = (*r.offset(idx as isize)).shortcut;
        (*r.offset(idx as isize)).shortcut = prev_idx;
        if next_idx == idx {
            break;
        }
        prev_idx = idx;
        idx = next_idx;
    }
    root_idx = idx;
    loop {
        prev_idx = (*r.offset(idx as isize)).shortcut;
        (*r.offset(idx as isize)).shortcut = root_idx;
        if prev_idx == idx {
            break;
        }
        idx = prev_idx;
    }
    return root_idx;
}
#[no_mangle]
pub unsafe extern "C" fn vl_mser_new(
    mut ndims: libc::c_int,
    mut dims: *const libc::c_int,
) -> *mut VlMserFilt {
    let mut f: *mut VlMserFilt = 0 as *mut VlMserFilt;
    let mut strides: *mut libc::c_int = 0 as *mut libc::c_int;
    let mut k: libc::c_int = 0;
    f = vl_calloc(
        ::core::mem::size_of::<VlMserFilt>() as libc::c_ulong,
        1 as libc::c_int as size_t,
    ) as *mut VlMserFilt;
    (*f).ndims = ndims;
    (*f)
        .dims = vl_malloc(
        (::core::mem::size_of::<libc::c_int>() as libc::c_ulong)
            .wrapping_mul(ndims as libc::c_ulong),
    ) as *mut libc::c_int;
    (*f)
        .subs = vl_malloc(
        (::core::mem::size_of::<libc::c_int>() as libc::c_ulong)
            .wrapping_mul(ndims as libc::c_ulong),
    ) as *mut libc::c_int;
    (*f)
        .dsubs = vl_malloc(
        (::core::mem::size_of::<libc::c_int>() as libc::c_ulong)
            .wrapping_mul(ndims as libc::c_ulong),
    ) as *mut libc::c_int;
    (*f)
        .strides = vl_malloc(
        (::core::mem::size_of::<libc::c_int>() as libc::c_ulong)
            .wrapping_mul(ndims as libc::c_ulong),
    ) as *mut libc::c_int;
    strides = (*f).strides;
    k = 0 as libc::c_int;
    while k < ndims {
        *((*f).dims).offset(k as isize) = *dims.offset(k as isize);
        k += 1;
    }
    *strides.offset(0 as libc::c_int as isize) = 1 as libc::c_int;
    k = 1 as libc::c_int;
    while k < ndims {
        *strides
            .offset(
                k as isize,
            ) = *strides.offset((k - 1 as libc::c_int) as isize)
            * *dims.offset((k - 1 as libc::c_int) as isize);
        k += 1;
    }
    (*f)
        .nel = *strides.offset((ndims - 1 as libc::c_int) as isize)
        * *dims.offset((ndims - 1 as libc::c_int) as isize);
    (*f).dof = ndims * (ndims + 1 as libc::c_int) / 2 as libc::c_int + ndims;
    (*f)
        .perm = vl_malloc(
        (::core::mem::size_of::<vl_uint>() as libc::c_ulong)
            .wrapping_mul((*f).nel as libc::c_ulong),
    ) as *mut vl_uint;
    (*f)
        .joins = vl_malloc(
        (::core::mem::size_of::<vl_uint>() as libc::c_ulong)
            .wrapping_mul((*f).nel as libc::c_ulong),
    ) as *mut vl_uint;
    (*f)
        .r = vl_malloc(
        (::core::mem::size_of::<VlMserReg>() as libc::c_ulong)
            .wrapping_mul((*f).nel as libc::c_ulong),
    ) as *mut VlMserReg;
    (*f).er = 0 as *mut VlMserExtrReg;
    (*f).rer = 0 as libc::c_int;
    (*f).mer = 0 as *mut vl_uint;
    (*f).rmer = 0 as libc::c_int;
    (*f).ell = 0 as *mut libc::c_float;
    (*f).rell = 0 as libc::c_int;
    (*f).delta = 5 as libc::c_int;
    (*f).max_area = 0.75f64;
    (*f).min_area = 3.0f64 / (*f).nel as libc::c_double;
    (*f).max_variation = 0.25f64;
    (*f).min_diversity = 0.2f64;
    return f;
}
#[no_mangle]
pub unsafe extern "C" fn vl_mser_delete(mut f: *mut VlMserFilt) {
    if !f.is_null() {
        if !((*f).acc).is_null() {
            vl_free((*f).acc as *mut libc::c_void);
        }
        if !((*f).ell).is_null() {
            vl_free((*f).ell as *mut libc::c_void);
        }
        if !((*f).er).is_null() {
            vl_free((*f).er as *mut libc::c_void);
        }
        if !((*f).r).is_null() {
            vl_free((*f).r as *mut libc::c_void);
        }
        if !((*f).joins).is_null() {
            vl_free((*f).joins as *mut libc::c_void);
        }
        if !((*f).perm).is_null() {
            vl_free((*f).perm as *mut libc::c_void);
        }
        if !((*f).strides).is_null() {
            vl_free((*f).strides as *mut libc::c_void);
        }
        if !((*f).dsubs).is_null() {
            vl_free((*f).dsubs as *mut libc::c_void);
        }
        if !((*f).subs).is_null() {
            vl_free((*f).subs as *mut libc::c_void);
        }
        if !((*f).dims).is_null() {
            vl_free((*f).dims as *mut libc::c_void);
        }
        if !((*f).mer).is_null() {
            vl_free((*f).mer as *mut libc::c_void);
        }
        vl_free(f as *mut libc::c_void);
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_mser_process(
    mut f: *mut VlMserFilt,
    mut im: *const vl_mser_pix,
) {
    let mut current_block: u64;
    let mut nel: vl_uint = (*f).nel as vl_uint;
    let mut perm: *mut vl_uint = (*f).perm;
    let mut joins: *mut vl_uint = (*f).joins;
    let mut ndims: libc::c_int = (*f).ndims;
    let mut dims: *mut libc::c_int = (*f).dims;
    let mut subs: *mut libc::c_int = (*f).subs;
    let mut dsubs: *mut libc::c_int = (*f).dsubs;
    let mut strides: *mut libc::c_int = (*f).strides;
    let mut r: *mut VlMserReg = (*f).r;
    let mut er: *mut VlMserExtrReg = (*f).er;
    let mut mer: *mut vl_uint = (*f).mer;
    let mut delta: libc::c_int = (*f).delta;
    let mut njoins: libc::c_int = 0 as libc::c_int;
    let mut ner: libc::c_int = 0 as libc::c_int;
    let mut nmer: libc::c_int = 0 as libc::c_int;
    let mut nbig: libc::c_int = 0 as libc::c_int;
    let mut nsmall: libc::c_int = 0 as libc::c_int;
    let mut nbad: libc::c_int = 0 as libc::c_int;
    let mut ndup: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0;
    let mut j: libc::c_int = 0;
    let mut k: libc::c_int = 0;
    (*f).nell = 0 as libc::c_int;
    let mut buckets: [vl_uint; 256] = [0; 256];
    memset(
        buckets.as_mut_ptr() as *mut libc::c_void,
        0 as libc::c_int,
        (::core::mem::size_of::<vl_uint>() as libc::c_ulong)
            .wrapping_mul(256 as libc::c_int as libc::c_ulong),
    );
    i = 0 as libc::c_int;
    while i < nel as libc::c_int {
        let mut v: vl_mser_pix = *im.offset(i as isize);
        buckets[v as usize] = (buckets[v as usize]).wrapping_add(1);
        i += 1;
    }
    i = 1 as libc::c_int;
    while i < 256 as libc::c_int {
        buckets[i
            as usize] = (buckets[i as usize] as libc::c_uint)
            .wrapping_add(buckets[(i - 1 as libc::c_int) as usize]) as vl_uint
            as vl_uint;
        i += 1;
    }
    i = nel as libc::c_int;
    while i >= 1 as libc::c_int {
        i -= 1;
        let mut v_0: vl_mser_pix = *im.offset(i as isize);
        buckets[v_0 as usize] = (buckets[v_0 as usize]).wrapping_sub(1);
        let mut j_0: vl_uint = buckets[v_0 as usize];
        *perm.offset(j_0 as isize) = i as vl_uint;
    }
    i = 0 as libc::c_int;
    while i < nel as libc::c_int {
        (*r.offset(i as isize))
            .parent = ((1 as libc::c_ulonglong) << 32 as libc::c_int)
            .wrapping_sub(1 as libc::c_int as libc::c_ulonglong) as vl_uint;
        i += 1;
    }
    i = 0 as libc::c_int;
    while i < nel as libc::c_int {
        let mut idx: vl_uint = *perm.offset(i as isize);
        let mut val: vl_mser_pix = *im.offset(idx as isize);
        let mut r_idx: vl_uint = 0;
        (*r.offset(idx as isize)).parent = idx;
        (*r.offset(idx as isize)).shortcut = idx;
        (*r.offset(idx as isize)).area = 1 as libc::c_int as vl_uint;
        (*r.offset(idx as isize)).height = 1 as libc::c_int as vl_uint;
        r_idx = idx;
        let mut temp: vl_uint = idx;
        k = ndims - 1 as libc::c_int;
        while k >= 0 as libc::c_int {
            *dsubs.offset(k as isize) = -(1 as libc::c_int);
            *subs
                .offset(
                    k as isize,
                ) = temp.wrapping_div(*strides.offset(k as isize) as libc::c_uint)
                as libc::c_int;
            temp = temp.wrapping_rem(*strides.offset(k as isize) as libc::c_uint);
            k -= 1;
        }
        's_194: loop {
            let mut n_idx: vl_uint = 0 as libc::c_int as vl_uint;
            let mut good: vl_bool = 1 as libc::c_int;
            k = 0 as libc::c_int;
            while k < ndims && good != 0 {
                let mut temp_0: libc::c_int = *dsubs.offset(k as isize)
                    + *subs.offset(k as isize);
                good
                    &= (0 as libc::c_int <= temp_0 && temp_0 < *dims.offset(k as isize))
                        as libc::c_int;
                n_idx = (n_idx as libc::c_uint)
                    .wrapping_add((temp_0 * *strides.offset(k as isize)) as libc::c_uint)
                    as vl_uint as vl_uint;
                k += 1;
            }
            if good != 0 && n_idx != idx
                && (*r.offset(n_idx as isize)).parent as libc::c_ulonglong
                    != ((1 as libc::c_ulonglong) << 32 as libc::c_int)
                        .wrapping_sub(1 as libc::c_int as libc::c_ulonglong)
            {
                let mut nr_val: vl_mser_pix = 0 as libc::c_int as vl_mser_pix;
                let mut nr_idx: vl_uint = 0 as libc::c_int as vl_uint;
                let mut hgt: libc::c_int = (*r.offset(r_idx as isize)).height
                    as libc::c_int;
                let mut n_hgt: libc::c_int = (*r.offset(nr_idx as isize)).height
                    as libc::c_int;
                r_idx = climb(r, idx);
                nr_idx = climb(r, n_idx);
                if r_idx != nr_idx {
                    nr_val = *im.offset(nr_idx as isize);
                    if nr_val as libc::c_int == val as libc::c_int && hgt < n_hgt {
                        (*r.offset(r_idx as isize)).parent = nr_idx;
                        (*r.offset(r_idx as isize)).shortcut = nr_idx;
                        let ref mut fresh2 = (*r.offset(nr_idx as isize)).area;
                        *fresh2 = (*fresh2 as libc::c_uint)
                            .wrapping_add((*r.offset(r_idx as isize)).area) as vl_uint
                            as vl_uint;
                        (*r.offset(nr_idx as isize))
                            .height = (if n_hgt > hgt + 1 as libc::c_int {
                            n_hgt
                        } else {
                            hgt + 1 as libc::c_int
                        }) as vl_uint;
                        let fresh3 = njoins;
                        njoins = njoins + 1;
                        *joins.offset(fresh3 as isize) = r_idx;
                    } else {
                        (*r.offset(nr_idx as isize)).parent = r_idx;
                        (*r.offset(nr_idx as isize)).shortcut = r_idx;
                        let ref mut fresh4 = (*r.offset(r_idx as isize)).area;
                        *fresh4 = (*fresh4 as libc::c_uint)
                            .wrapping_add((*r.offset(nr_idx as isize)).area) as vl_uint
                            as vl_uint;
                        (*r.offset(r_idx as isize))
                            .height = (if hgt > n_hgt + 1 as libc::c_int {
                            hgt
                        } else {
                            n_hgt + 1 as libc::c_int
                        }) as vl_uint;
                        let fresh5 = njoins;
                        njoins = njoins + 1;
                        *joins.offset(fresh5 as isize) = nr_idx;
                        if nr_val as libc::c_int != val as libc::c_int {
                            ner += 1;
                        }
                    }
                }
            }
            k = 0 as libc::c_int;
            loop {
                let ref mut fresh6 = *dsubs.offset(k as isize);
                *fresh6 += 1;
                if !(*fresh6 > 1 as libc::c_int) {
                    break;
                }
                let fresh7 = k;
                k = k + 1;
                *dsubs.offset(fresh7 as isize) = -(1 as libc::c_int);
                if k == ndims {
                    break 's_194;
                }
            }
        }
        i += 1;
    }
    ner += 1;
    (*f).njoins = njoins;
    (*f).stats.num_extremal = ner;
    if (*f).rer < ner {
        if !er.is_null() {
            vl_free(er as *mut libc::c_void);
        }
        er = vl_malloc(
            (::core::mem::size_of::<VlMserExtrReg>() as libc::c_ulong)
                .wrapping_mul(ner as libc::c_ulong),
        ) as *mut VlMserExtrReg;
        (*f).er = er;
        (*f).rer = ner;
    }
    (*f).nmer = ner;
    ner = 0 as libc::c_int;
    i = 0 as libc::c_int;
    while i < nel as libc::c_int {
        let mut idx_0: vl_uint = *perm.offset(i as isize);
        let mut val_0: vl_mser_pix = *im.offset(idx_0 as isize);
        let mut p_idx: vl_uint = (*r.offset(idx_0 as isize)).parent;
        let mut p_val: vl_mser_pix = *im.offset(p_idx as isize);
        let mut is_extr: vl_bool = (p_val as libc::c_int > val_0 as libc::c_int
            || idx_0 == p_idx) as libc::c_int;
        if is_extr != 0 {
            (*er.offset(ner as isize)).index = idx_0 as libc::c_int;
            (*er.offset(ner as isize)).parent = ner;
            (*er.offset(ner as isize)).value = *im.offset(idx_0 as isize);
            (*er.offset(ner as isize)).area = (*r.offset(idx_0 as isize)).area;
            (*r.offset(idx_0 as isize)).shortcut = ner as vl_uint;
            ner += 1;
        } else {
            (*r.offset(idx_0 as isize))
                .shortcut = ((1 as libc::c_ulonglong) << 32 as libc::c_int)
                .wrapping_sub(1 as libc::c_int as libc::c_ulonglong) as vl_uint;
        }
        i += 1;
    }
    i = 0 as libc::c_int;
    while i < ner {
        let mut idx_1: vl_uint = (*er.offset(i as isize)).index as vl_uint;
        loop {
            idx_1 = (*r.offset(idx_1 as isize)).parent;
            if !((*r.offset(idx_1 as isize)).shortcut as libc::c_ulonglong
                == ((1 as libc::c_ulonglong) << 32 as libc::c_int)
                    .wrapping_sub(1 as libc::c_int as libc::c_ulonglong))
            {
                break;
            }
        }
        (*er.offset(i as isize))
            .parent = (*r.offset(idx_1 as isize)).shortcut as libc::c_int;
        (*er.offset(i as isize)).shortcut = i as vl_uint;
        i += 1;
    }
    i = 0 as libc::c_int;
    while i < ner {
        let mut top_val: libc::c_int = (*er.offset(i as isize)).value as libc::c_int
            + delta;
        let mut top: libc::c_int = (*er.offset(i as isize)).shortcut as libc::c_int;
        loop {
            let mut next: libc::c_int = (*er.offset(top as isize)).parent;
            let mut next_val: libc::c_int = (*er.offset(next as isize)).value
                as libc::c_int;
            if next == top || next_val > top_val {
                break;
            }
            top = next;
        }
        let mut area: libc::c_int = (*er.offset(i as isize)).area as libc::c_int;
        let mut area_top: libc::c_int = (*er.offset(top as isize)).area as libc::c_int;
        (*er.offset(i as isize))
            .variation = (area_top - area) as libc::c_float / area as libc::c_float;
        (*er.offset(i as isize)).max_stable = 1 as libc::c_int as vl_uint;
        let mut parent: libc::c_int = (*er.offset(i as isize)).parent;
        let mut curr: libc::c_int = (*er.offset(parent as isize)).shortcut
            as libc::c_int;
        (*er.offset(parent as isize))
            .shortcut = (if top > curr { top } else { curr }) as vl_uint;
        i += 1;
    }
    nmer = ner;
    i = 0 as libc::c_int;
    while i < ner {
        let mut parent_0: vl_uint = (*er.offset(i as isize)).parent as vl_uint;
        let mut val_1: vl_mser_pix = (*er.offset(i as isize)).value;
        let mut var: libc::c_float = (*er.offset(i as isize)).variation;
        let mut p_val_0: vl_mser_pix = (*er.offset(parent_0 as isize)).value;
        let mut p_var: libc::c_float = (*er.offset(parent_0 as isize)).variation;
        let mut loser: vl_uint = 0;
        if !(p_val_0 as libc::c_int > val_1 as libc::c_int + 1 as libc::c_int) {
            if var < p_var {
                loser = parent_0;
            } else {
                loser = i as vl_uint;
            }
            if (*er.offset(loser as isize)).max_stable != 0 {
                nmer -= 1;
                (*er.offset(loser as isize)).max_stable = 0 as libc::c_int as vl_uint;
            }
        }
        i += 1;
    }
    (*f).stats.num_unstable = ner - nmer;
    let mut max_area: libc::c_float = (*f).max_area as libc::c_float
        * nel as libc::c_float;
    let mut min_area: libc::c_float = (*f).min_area as libc::c_float
        * nel as libc::c_float;
    let mut max_var: libc::c_float = (*f).max_variation as libc::c_float;
    let mut min_div: libc::c_float = (*f).min_diversity as libc::c_float;
    i = ner - 1 as libc::c_int;
    while i as libc::c_long >= 0 as libc::c_long {
        if !((*er.offset(i as isize)).max_stable == 0) {
            if (*er.offset(i as isize)).variation >= max_var {
                nbad += 1;
                current_block = 6944873078863795104;
            } else if (*er.offset(i as isize)).area as libc::c_float > max_area {
                nbig += 1;
                current_block = 6944873078863795104;
            } else if ((*er.offset(i as isize)).area as libc::c_float) < min_area {
                nsmall += 1;
                current_block = 6944873078863795104;
            } else if (min_div as libc::c_double) < 1.0f64 {
                let mut parent_1: vl_uint = (*er.offset(i as isize)).parent as vl_uint;
                let mut area_0: libc::c_int = 0;
                let mut p_area: libc::c_int = 0;
                let mut div: libc::c_float = 0.;
                if parent_1 as libc::c_int != i {
                    while (*er.offset(parent_1 as isize)).max_stable == 0 {
                        let mut next_0: vl_uint = (*er.offset(parent_1 as isize)).parent
                            as vl_uint;
                        if next_0 == parent_1 {
                            break;
                        }
                        parent_1 = next_0;
                    }
                    area_0 = (*er.offset(i as isize)).area as libc::c_int;
                    p_area = (*er.offset(parent_1 as isize)).area as libc::c_int;
                    div = (p_area - area_0) as libc::c_float / p_area as libc::c_float;
                    if div < min_div {
                        ndup += 1;
                        current_block = 6944873078863795104;
                    } else {
                        current_block = 1425453989644512380;
                    }
                } else {
                    current_block = 1425453989644512380;
                }
            } else {
                current_block = 1425453989644512380;
            }
            match current_block {
                1425453989644512380 => {}
                _ => {
                    (*er.offset(i as isize)).max_stable = 0 as libc::c_int as vl_uint;
                    nmer -= 1;
                }
            }
        }
        i -= 1;
    }
    (*f).stats.num_abs_unstable = nbad;
    (*f).stats.num_too_big = nbig;
    (*f).stats.num_too_small = nsmall;
    (*f).stats.num_duplicates = ndup;
    if (*f).rmer < nmer {
        if !mer.is_null() {
            vl_free(mer as *mut libc::c_void);
        }
        mer = vl_malloc(
            (::core::mem::size_of::<vl_uint>() as libc::c_ulong)
                .wrapping_mul(nmer as libc::c_ulong),
        ) as *mut vl_uint;
        (*f).mer = mer;
        (*f).rmer = nmer;
    }
    (*f).nmer = nmer;
    j = 0 as libc::c_int;
    i = 0 as libc::c_int;
    while i < ner {
        if (*er.offset(i as isize)).max_stable != 0 {
            let fresh8 = j;
            j = j + 1;
            *mer.offset(fresh8 as isize) = (*er.offset(i as isize)).index as vl_uint;
        }
        i += 1;
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_mser_ell_fit(mut f: *mut VlMserFilt) {
    let mut nel: libc::c_int = (*f).nel;
    let mut dof: libc::c_int = (*f).dof;
    let mut dims: *mut libc::c_int = (*f).dims;
    let mut ndims: libc::c_int = (*f).ndims;
    let mut subs: *mut libc::c_int = (*f).subs;
    let mut njoins: libc::c_int = (*f).njoins;
    let mut joins: *mut vl_uint = (*f).joins;
    let mut r: *mut VlMserReg = (*f).r;
    let mut mer: *mut vl_uint = (*f).mer;
    let mut nmer: libc::c_int = (*f).nmer;
    let mut acc: *mut vl_mser_acc = (*f).acc;
    let mut ell: *mut vl_mser_acc = (*f).ell;
    let mut d: libc::c_int = 0;
    let mut index: libc::c_int = 0;
    let mut i: libc::c_int = 0;
    let mut j: libc::c_int = 0;
    if (*f).nell == (*f).nmer {
        return;
    }
    if (*f).rell < (*f).nmer {
        if !((*f).ell).is_null() {
            vl_free((*f).ell as *mut libc::c_void);
        }
        (*f)
            .ell = vl_malloc(
            (::core::mem::size_of::<libc::c_float>() as libc::c_ulong)
                .wrapping_mul((*f).nmer as libc::c_ulong)
                .wrapping_mul((*f).dof as libc::c_ulong),
        ) as *mut libc::c_float;
        (*f).rell = (*f).nmer;
    }
    if ((*f).acc).is_null() {
        (*f)
            .acc = vl_malloc(
            (::core::mem::size_of::<libc::c_float>() as libc::c_ulong)
                .wrapping_mul((*f).nel as libc::c_ulong),
        ) as *mut libc::c_float;
    }
    acc = (*f).acc;
    ell = (*f).ell;
    d = 0 as libc::c_int;
    while d < (*f).dof {
        memset(
            subs as *mut libc::c_void,
            0 as libc::c_int,
            (::core::mem::size_of::<libc::c_int>() as libc::c_ulong)
                .wrapping_mul(ndims as libc::c_ulong),
        );
        if d < ndims {
            index = 0 as libc::c_int;
            while index < nel {
                *acc.offset(index as isize) = *subs.offset(d as isize) as vl_mser_acc;
                adv(ndims, dims, subs);
                index += 1;
            }
        } else {
            i = d - ndims;
            j = 0 as libc::c_int;
            while i > j {
                i -= j + 1 as libc::c_int;
                j += 1;
            }
            index = 0 as libc::c_int;
            while index < nel {
                *acc
                    .offset(
                        index as isize,
                    ) = (*subs.offset(i as isize) * *subs.offset(j as isize))
                    as vl_mser_acc;
                adv(ndims, dims, subs);
                index += 1;
            }
        }
        i = 0 as libc::c_int;
        while i < njoins {
            let mut index_0: vl_uint = *joins.offset(i as isize);
            let mut parent: vl_uint = (*r.offset(index_0 as isize)).parent;
            let ref mut fresh9 = *acc.offset(parent as isize);
            *fresh9 += *acc.offset(index_0 as isize);
            i += 1;
        }
        i = 0 as libc::c_int;
        while i < nmer {
            let mut idx: vl_uint = *mer.offset(i as isize);
            *ell.offset((d + dof * i) as isize) = *acc.offset(idx as isize);
            i += 1;
        }
        d += 1;
    }
    index = 0 as libc::c_int;
    while index < nmer {
        let mut pt: *mut libc::c_float = ell.offset((index * dof) as isize);
        let mut idx_0: vl_uint = *mer.offset(index as isize);
        let mut area: libc::c_float = (*r.offset(idx_0 as isize)).area as libc::c_float;
        d = 0 as libc::c_int;
        while d < dof {
            *pt.offset(d as isize) /= area;
            if d >= ndims {
                i = d - ndims;
                j = 0 as libc::c_int;
                while i > j {
                    i -= j + 1 as libc::c_int;
                    j += 1;
                }
                *pt.offset(d as isize)
                    -= *pt.offset(i as isize) * *pt.offset(j as isize);
            }
            d += 1;
        }
        index += 1;
    }
    (*f).nell = nmer;
}
