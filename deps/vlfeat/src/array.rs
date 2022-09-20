use ::libc;
extern "C" {
    fn abort() -> !;
    fn __assert_fail(
        __assertion: *const libc::c_char,
        __file: *const libc::c_char,
        __line: libc::c_uint,
        __function: *const libc::c_char,
    ) -> !;
    fn vl_free(ptr: *mut libc::c_void);
    fn vl_malloc(n: size_t) -> *mut libc::c_void;
    fn memcpy(
        _: *mut libc::c_void,
        _: *const libc::c_void,
        _: libc::c_ulong,
    ) -> *mut libc::c_void;
}
pub type vl_int64 = libc::c_longlong;
pub type vl_int32 = libc::c_int;
pub type vl_int16 = libc::c_short;
pub type vl_int8 = libc::c_char;
pub type vl_uint64 = libc::c_ulonglong;
pub type vl_uint32 = libc::c_uint;
pub type vl_bool = libc::c_int;
pub type vl_size = vl_uint64;
pub type vl_uindex = vl_uint64;
pub type size_t = libc::c_ulong;
pub type vl_type = vl_uint32;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlArray {
    pub type_0: vl_type,
    pub isEnvelope: vl_bool,
    pub isSparse: vl_bool,
    pub numDimensions: vl_size,
    pub dimensions: [vl_size; 16],
    pub data: *mut libc::c_void,
    pub rowPointers: *mut libc::c_void,
    pub columnPointers: *mut libc::c_void,
}
pub type VlArray = _VlArray;
#[inline]
unsafe extern "C" fn vl_get_type_size(mut type_0: vl_type) -> vl_size {
    let mut dataSize: vl_size = 0 as libc::c_int as vl_size;
    match type_0 {
        2 => {
            dataSize = ::core::mem::size_of::<libc::c_double>() as libc::c_ulong
                as vl_size;
        }
        1 => {
            dataSize = ::core::mem::size_of::<libc::c_float>() as libc::c_ulong
                as vl_size;
        }
        9 | 10 => {
            dataSize = ::core::mem::size_of::<vl_int64>() as libc::c_ulong as vl_size;
        }
        7 | 8 => {
            dataSize = ::core::mem::size_of::<vl_int32>() as libc::c_ulong as vl_size;
        }
        5 | 6 => {
            dataSize = ::core::mem::size_of::<vl_int16>() as libc::c_ulong as vl_size;
        }
        3 | 4 => {
            dataSize = ::core::mem::size_of::<vl_int8>() as libc::c_ulong as vl_size;
        }
        _ => {
            abort();
        }
    }
    return dataSize;
}
#[no_mangle]
pub unsafe extern "C" fn vl_array_get_num_elements(
    mut self_0: *const VlArray,
) -> vl_size {
    let mut numElements: vl_size = 1 as libc::c_int as vl_size;
    let mut k: vl_uindex = 0;
    if (*self_0).numDimensions == 0 as libc::c_int as libc::c_ulonglong {
        return 0 as libc::c_int as vl_size;
    }
    k = 0 as libc::c_int as vl_uindex;
    while k < (*self_0).numDimensions {
        numElements = (numElements as libc::c_ulonglong)
            .wrapping_mul((*self_0).dimensions[k as usize]) as vl_size as vl_size;
        k = k.wrapping_add(1);
    }
    return numElements;
}
#[no_mangle]
pub unsafe extern "C" fn vl_array_init(
    mut self_0: *mut VlArray,
    mut type_0: vl_type,
    mut numDimensions: vl_size,
    mut dimensions: *const vl_size,
) -> *mut VlArray {
    if numDimensions <= 16 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"numDimensions <= VL_ARRAY_MAX_NUM_DIMENSIONS\0" as *const u8
                as *const libc::c_char,
            b"vl/array.c\0" as *const u8 as *const libc::c_char,
            54 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 69],
                &[libc::c_char; 69],
            >(b"VlArray *vl_array_init(VlArray *, vl_type, vl_size, const vl_size *)\0"))
                .as_ptr(),
        );
    }
    (*self_0).type_0 = type_0;
    (*self_0).numDimensions = numDimensions;
    memcpy(
        ((*self_0).dimensions).as_mut_ptr() as *mut libc::c_void,
        dimensions as *const libc::c_void,
        (::core::mem::size_of::<vl_size>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numDimensions) as libc::c_ulong,
    );
    (*self_0)
        .data = vl_malloc(
        (vl_get_type_size(type_0)).wrapping_mul(vl_array_get_num_elements(self_0))
            as size_t,
    );
    (*self_0).isEnvelope = 0 as libc::c_int;
    (*self_0).isSparse = 0 as libc::c_int;
    return self_0;
}
#[no_mangle]
pub unsafe extern "C" fn vl_array_init_envelope(
    mut self_0: *mut VlArray,
    mut data: *mut libc::c_void,
    mut type_0: vl_type,
    mut numDimensions: vl_size,
    mut dimensions: *const vl_size,
) -> *mut VlArray {
    if numDimensions <= 16 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"numDimensions <= VL_ARRAY_MAX_NUM_DIMENSIONS\0" as *const u8
                as *const libc::c_char,
            b"vl/array.c\0" as *const u8 as *const libc::c_char,
            79 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 86],
                &[libc::c_char; 86],
            >(
                b"VlArray *vl_array_init_envelope(VlArray *, void *, vl_type, vl_size, const vl_size *)\0",
            ))
                .as_ptr(),
        );
    }
    (*self_0).type_0 = type_0;
    (*self_0).numDimensions = numDimensions;
    memcpy(
        ((*self_0).dimensions).as_mut_ptr() as *mut libc::c_void,
        dimensions as *const libc::c_void,
        (::core::mem::size_of::<vl_size>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numDimensions) as libc::c_ulong,
    );
    (*self_0).data = data;
    (*self_0).isEnvelope = 1 as libc::c_int;
    (*self_0).isSparse = 0 as libc::c_int;
    return self_0;
}
#[no_mangle]
pub unsafe extern "C" fn vl_array_init_matrix(
    mut self_0: *mut VlArray,
    mut type_0: vl_type,
    mut numRows: vl_size,
    mut numColumns: vl_size,
) -> *mut VlArray {
    let mut dimensions: [vl_size; 2] = [numRows, numColumns];
    return vl_array_init(
        self_0,
        type_0,
        2 as libc::c_int as vl_size,
        dimensions.as_mut_ptr(),
    );
}
#[no_mangle]
pub unsafe extern "C" fn vl_array_init_matrix_envelope(
    mut self_0: *mut VlArray,
    mut data: *mut libc::c_void,
    mut type_0: vl_type,
    mut numRows: vl_size,
    mut numColumns: vl_size,
) -> *mut VlArray {
    let mut dimensions: [vl_size; 2] = [numRows, numColumns];
    return vl_array_init_envelope(
        self_0,
        data,
        type_0,
        2 as libc::c_int as vl_size,
        dimensions.as_mut_ptr(),
    );
}
#[no_mangle]
pub unsafe extern "C" fn vl_array_dealloc(mut self_0: *mut VlArray) {
    if (*self_0).isEnvelope == 0 {
        if !((*self_0).data).is_null() {
            vl_free((*self_0).data);
            (*self_0).data = 0 as *mut libc::c_void;
        }
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_array_new(
    mut type_0: vl_type,
    mut numDimensions: vl_size,
    mut dimensions: *const vl_size,
) -> *mut VlArray {
    let mut self_0: *mut VlArray = vl_malloc(
        ::core::mem::size_of::<VlArray>() as libc::c_ulong,
    ) as *mut VlArray;
    return vl_array_init(self_0, type_0, numDimensions, dimensions);
}
#[no_mangle]
pub unsafe extern "C" fn vl_array_new_matrix(
    mut type_0: vl_type,
    mut numRows: vl_size,
    mut numColumns: vl_size,
) -> *mut VlArray {
    let mut dimensions: [vl_size; 2] = [numRows, numColumns];
    return vl_array_new(type_0, 2 as libc::c_int as vl_size, dimensions.as_mut_ptr());
}
#[no_mangle]
pub unsafe extern "C" fn vl_array_new_envelope(
    mut data: *mut libc::c_void,
    mut type_0: vl_type,
    mut numDimensions: vl_size,
    mut dimensions: *const vl_size,
) -> *mut VlArray {
    let mut self_0: *mut VlArray = vl_malloc(
        ::core::mem::size_of::<VlArray>() as libc::c_ulong,
    ) as *mut VlArray;
    return vl_array_init_envelope(self_0, data, type_0, numDimensions, dimensions);
}
#[no_mangle]
pub unsafe extern "C" fn vl_array_new_matrix_envelope(
    mut data: *mut libc::c_void,
    mut type_0: vl_type,
    mut numRows: vl_size,
    mut numColumns: vl_size,
) -> *mut VlArray {
    let mut dimensions: [vl_size; 2] = [numRows, numColumns];
    return vl_array_new_envelope(
        data,
        type_0,
        2 as libc::c_int as vl_size,
        dimensions.as_mut_ptr(),
    );
}
#[no_mangle]
pub unsafe extern "C" fn vl_array_delete(mut self_0: *mut VlArray) {
    vl_array_dealloc(self_0);
    vl_free(self_0 as *mut libc::c_void);
}
