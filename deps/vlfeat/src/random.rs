use ::libc;
extern "C" {
    fn memset(
        _: *mut libc::c_void,
        _: libc::c_int,
        _: libc::c_ulong,
    ) -> *mut libc::c_void;
}
pub type vl_int64 = libc::c_longlong;
pub type vl_uint64 = libc::c_ulonglong;
pub type vl_uint32 = libc::c_uint;
pub type vl_int = libc::c_int;
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
#[inline]
unsafe extern "C" fn vl_rand_uindex(
    mut self_0: *mut VlRand,
    mut range: vl_uindex,
) -> vl_uindex {
    if range <= 0xffffffff as libc::c_uint as libc::c_ulonglong {
        return (vl_rand_uint32(self_0)).wrapping_rem(range as vl_uint32) as vl_uindex
    } else {
        return (vl_rand_uint64(self_0)).wrapping_rem(range)
    };
}
#[inline]
unsafe extern "C" fn vl_rand_uint64(mut self_0: *mut VlRand) -> vl_uint64 {
    let mut a: vl_uint64 = vl_rand_uint32(self_0) as vl_uint64;
    let mut b: vl_uint64 = vl_rand_uint32(self_0) as vl_uint64;
    return a << 32 as libc::c_int | b;
}
#[no_mangle]
pub unsafe extern "C" fn vl_rand_init(mut self_0: *mut VlRand) {
    memset(
        ((*self_0).mt).as_mut_ptr() as *mut libc::c_void,
        0 as libc::c_int,
        (::core::mem::size_of::<vl_uint32>() as libc::c_ulong)
            .wrapping_mul(624 as libc::c_int as libc::c_ulong),
    );
    (*self_0).mti = (624 as libc::c_int + 1 as libc::c_int) as vl_uint32;
}
#[no_mangle]
pub unsafe extern "C" fn vl_rand_seed(mut self_0: *mut VlRand, mut s: vl_uint32) {
    (*self_0).mt[0 as libc::c_int as usize] = s & 0xffffffff as libc::c_uint;
    (*self_0).mti = 1 as libc::c_int as vl_uint32;
    while (*self_0).mti < 624 as libc::c_int as libc::c_uint {
        (*self_0)
            .mt[(*self_0).mti
            as usize] = (1812433253 as libc::c_uint)
            .wrapping_mul(
                (*self_0)
                    .mt[((*self_0).mti).wrapping_sub(1 as libc::c_int as libc::c_uint)
                    as usize]
                    ^ (*self_0)
                        .mt[((*self_0).mti)
                        .wrapping_sub(1 as libc::c_int as libc::c_uint) as usize]
                        >> 30 as libc::c_int,
            )
            .wrapping_add((*self_0).mti);
        (*self_0).mt[(*self_0).mti as usize] &= 0xffffffff as libc::c_uint;
        (*self_0).mti = ((*self_0).mti).wrapping_add(1);
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_rand_seed_by_array(
    mut self_0: *mut VlRand,
    mut key: *const vl_uint32,
    mut keySize: vl_size,
) {
    let mut i: libc::c_int = 0;
    let mut j: libc::c_int = 0;
    let mut k: libc::c_int = 0;
    vl_rand_seed(self_0, 19650218 as libc::c_uint);
    i = 1 as libc::c_int;
    j = 0 as libc::c_int;
    k = if 624 as libc::c_int as libc::c_ulonglong > keySize {
        624 as libc::c_int
    } else {
        keySize as libc::c_int
    };
    while k != 0 {
        (*self_0)
            .mt[i
            as usize] = ((*self_0).mt[i as usize]
            ^ ((*self_0).mt[(i - 1 as libc::c_int) as usize]
                ^ (*self_0).mt[(i - 1 as libc::c_int) as usize] >> 30 as libc::c_int)
                .wrapping_mul(1664525 as libc::c_uint))
            .wrapping_add(*key.offset(j as isize))
            .wrapping_add(j as libc::c_uint);
        (*self_0).mt[i as usize] &= 0xffffffff as libc::c_uint;
        i += 1;
        j += 1;
        if i >= 624 as libc::c_int {
            (*self_0)
                .mt[0 as libc::c_int
                as usize] = (*self_0)
                .mt[(624 as libc::c_int - 1 as libc::c_int) as usize];
            i = 1 as libc::c_int;
        }
        if j >= keySize as libc::c_int {
            j = 0 as libc::c_int;
        }
        k -= 1;
    }
    k = 624 as libc::c_int - 1 as libc::c_int;
    while k != 0 {
        (*self_0)
            .mt[i
            as usize] = ((*self_0).mt[i as usize]
            ^ ((*self_0).mt[(i - 1 as libc::c_int) as usize]
                ^ (*self_0).mt[(i - 1 as libc::c_int) as usize] >> 30 as libc::c_int)
                .wrapping_mul(1566083941 as libc::c_uint))
            .wrapping_sub(i as libc::c_uint);
        (*self_0).mt[i as usize] &= 0xffffffff as libc::c_uint;
        i += 1;
        if i >= 624 as libc::c_int {
            (*self_0)
                .mt[0 as libc::c_int
                as usize] = (*self_0)
                .mt[(624 as libc::c_int - 1 as libc::c_int) as usize];
            i = 1 as libc::c_int;
        }
        k -= 1;
    }
    (*self_0).mt[0 as libc::c_int as usize] = 0x80000000 as libc::c_uint;
}
#[no_mangle]
pub unsafe extern "C" fn vl_rand_permute_indexes(
    mut self_0: *mut VlRand,
    mut array: *mut vl_index,
    mut size: vl_size,
) {
    let mut i: vl_index = 0;
    let mut j: vl_index = 0;
    let mut tmp: vl_index = 0;
    i = size.wrapping_sub(1 as libc::c_int as libc::c_ulonglong) as vl_index;
    while i > 0 as libc::c_int as libc::c_longlong {
        j = vl_rand_uindex(
            self_0,
            (i + 1 as libc::c_int as libc::c_longlong) as vl_uindex,
        ) as vl_int as vl_index;
        tmp = *array.offset(i as isize);
        *array.offset(i as isize) = *array.offset(j as isize);
        *array.offset(j as isize) = tmp;
        i -= 1;
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_rand_uint32(mut self_0: *mut VlRand) -> vl_uint32 {
    let mut y: vl_uint32 = 0;
    static mut mag01: [vl_uint32; 2] = [0 as libc::c_uint, 0x9908b0df as libc::c_uint];
    if (*self_0).mti >= 624 as libc::c_int as libc::c_uint {
        let mut kk: libc::c_int = 0;
        if (*self_0).mti == (624 as libc::c_int + 1 as libc::c_int) as libc::c_uint {
            vl_rand_seed(self_0, 5489 as libc::c_uint);
        }
        kk = 0 as libc::c_int;
        while kk < 624 as libc::c_int - 397 as libc::c_int {
            y = (*self_0).mt[kk as usize] & 0x80000000 as libc::c_uint
                | (*self_0).mt[(kk + 1 as libc::c_int) as usize]
                    & 0x7fffffff as libc::c_uint;
            (*self_0)
                .mt[kk
                as usize] = (*self_0).mt[(kk + 397 as libc::c_int) as usize]
                ^ y >> 1 as libc::c_int ^ mag01[(y & 0x1 as libc::c_uint) as usize];
            kk += 1;
        }
        while kk < 624 as libc::c_int - 1 as libc::c_int {
            y = (*self_0).mt[kk as usize] & 0x80000000 as libc::c_uint
                | (*self_0).mt[(kk + 1 as libc::c_int) as usize]
                    & 0x7fffffff as libc::c_uint;
            (*self_0)
                .mt[kk
                as usize] = (*self_0)
                .mt[(kk + (397 as libc::c_int - 624 as libc::c_int)) as usize]
                ^ y >> 1 as libc::c_int ^ mag01[(y & 0x1 as libc::c_uint) as usize];
            kk += 1;
        }
        y = (*self_0).mt[(624 as libc::c_int - 1 as libc::c_int) as usize]
            & 0x80000000 as libc::c_uint
            | (*self_0).mt[0 as libc::c_int as usize] & 0x7fffffff as libc::c_uint;
        (*self_0)
            .mt[(624 as libc::c_int - 1 as libc::c_int)
            as usize] = (*self_0).mt[(397 as libc::c_int - 1 as libc::c_int) as usize]
            ^ y >> 1 as libc::c_int ^ mag01[(y & 0x1 as libc::c_uint) as usize];
        (*self_0).mti = 0 as libc::c_int as vl_uint32;
    }
    let fresh0 = (*self_0).mti;
    (*self_0).mti = ((*self_0).mti).wrapping_add(1);
    y = (*self_0).mt[fresh0 as usize];
    y ^= y >> 11 as libc::c_int;
    y ^= y << 7 as libc::c_int & 0x9d2c5680 as libc::c_uint;
    y ^= y << 15 as libc::c_int & 0xefc60000 as libc::c_uint;
    y ^= y >> 18 as libc::c_int;
    return y;
}
