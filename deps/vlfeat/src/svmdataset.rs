use ::libc;
extern "C" {
    pub type _VlHomogeneousKernelMap;
    fn abort() -> !;
    fn __assert_fail(
        __assertion: *const libc::c_char,
        __file: *const libc::c_char,
        __line: libc::c_uint,
        __function: *const libc::c_char,
    ) -> !;
    fn vl_calloc(n: size_t, size: size_t) -> *mut libc::c_void;
    fn vl_free(ptr: *mut libc::c_void);
    fn vl_homogeneouskernelmap_evaluate_d(
        self_0: *const VlHomogeneousKernelMap,
        destination: *mut libc::c_double,
        stride: vl_size,
        x: libc::c_double,
    );
    fn vl_homogeneouskernelmap_evaluate_f(
        self_0: *const VlHomogeneousKernelMap,
        destination: *mut libc::c_float,
        stride: vl_size,
        x: libc::c_double,
    );
    fn vl_homogeneouskernelmap_get_dimension(
        self_0: *const VlHomogeneousKernelMap,
    ) -> vl_size;
}
pub type vl_int64 = libc::c_longlong;
pub type vl_int32 = libc::c_int;
pub type vl_int16 = libc::c_short;
pub type vl_int8 = libc::c_char;
pub type vl_uint64 = libc::c_ulonglong;
pub type vl_uint32 = libc::c_uint;
pub type vl_size = vl_uint64;
pub type vl_uindex = vl_uint64;
pub type size_t = libc::c_ulong;
pub type vl_type = vl_uint32;
pub type VlHomogeneousKernelMap = _VlHomogeneousKernelMap;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct VlSvmDataset_ {
    pub dataType: vl_type,
    pub data: *mut libc::c_void,
    pub numData: vl_size,
    pub dimension: vl_size,
    pub hom: *mut VlHomogeneousKernelMap,
    pub homBuffer: *mut libc::c_void,
    pub homDimension: vl_size,
}
pub type VlSvmDataset = VlSvmDataset_;
pub type VlSvmInnerProductFunction = Option::<
    unsafe extern "C" fn(
        *const libc::c_void,
        vl_uindex,
        *mut libc::c_double,
    ) -> libc::c_double,
>;
pub type VlSvmAccumulateFunction = Option::<
    unsafe extern "C" fn(
        *const libc::c_void,
        vl_uindex,
        *mut libc::c_double,
        libc::c_double,
    ) -> (),
>;
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
pub unsafe extern "C" fn vl_svmdataset_new(
    mut dataType: vl_type,
    mut data: *mut libc::c_void,
    mut dimension: vl_size,
    mut numData: vl_size,
) -> *mut VlSvmDataset {
    let mut self_0: *mut VlSvmDataset = 0 as *mut VlSvmDataset;
    if dataType == 2 as libc::c_int as libc::c_uint
        || dataType == 1 as libc::c_int as libc::c_uint
    {} else {
        __assert_fail(
            b"dataType == VL_TYPE_DOUBLE || dataType == VL_TYPE_FLOAT\0" as *const u8
                as *const libc::c_char,
            b"vl/svmdataset.c\0" as *const u8 as *const libc::c_char,
            141 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 67],
                &[libc::c_char; 67],
            >(b"VlSvmDataset *vl_svmdataset_new(vl_type, void *, vl_size, vl_size)\0"))
                .as_ptr(),
        );
    }
    if !data.is_null() {} else {
        __assert_fail(
            b"data\0" as *const u8 as *const libc::c_char,
            b"vl/svmdataset.c\0" as *const u8 as *const libc::c_char,
            142 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 67],
                &[libc::c_char; 67],
            >(b"VlSvmDataset *vl_svmdataset_new(vl_type, void *, vl_size, vl_size)\0"))
                .as_ptr(),
        );
    }
    self_0 = vl_calloc(
        1 as libc::c_int as size_t,
        ::core::mem::size_of::<VlSvmDataset>() as libc::c_ulong,
    ) as *mut VlSvmDataset;
    if self_0.is_null() {
        return 0 as *mut VlSvmDataset;
    }
    (*self_0).dataType = dataType;
    (*self_0).data = data;
    (*self_0).dimension = dimension;
    (*self_0).numData = numData;
    (*self_0).hom = 0 as *mut VlHomogeneousKernelMap;
    (*self_0).homBuffer = 0 as *mut libc::c_void;
    return self_0;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svmdataset_delete(mut self_0: *mut VlSvmDataset) {
    if !((*self_0).homBuffer).is_null() {
        vl_free((*self_0).homBuffer);
        (*self_0).homBuffer = 0 as *mut libc::c_void;
    }
    vl_free(self_0 as *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn vl_svmdataset_get_data(
    mut self_0: *const VlSvmDataset,
) -> *mut libc::c_void {
    return (*self_0).data;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svmdataset_get_num_data(
    mut self_0: *const VlSvmDataset,
) -> vl_size {
    return (*self_0).numData;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svmdataset_get_dimension(
    mut self_0: *const VlSvmDataset,
) -> vl_size {
    if !((*self_0).hom).is_null() {
        return ((*self_0).dimension)
            .wrapping_mul(vl_homogeneouskernelmap_get_dimension((*self_0).hom));
    }
    return (*self_0).dimension;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svmdataset_get_homogeneous_kernel_map(
    mut self_0: *const VlSvmDataset,
) -> *mut VlHomogeneousKernelMap {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svmdataset.c\0" as *const u8 as *const libc::c_char,
            217 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 87],
                &[libc::c_char; 87],
            >(
                b"VlHomogeneousKernelMap *vl_svmdataset_get_homogeneous_kernel_map(const VlSvmDataset *)\0",
            ))
                .as_ptr(),
        );
    }
    return (*self_0).hom;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svmdataset_set_homogeneous_kernel_map(
    mut self_0: *mut VlSvmDataset,
    mut hom: *mut VlHomogeneousKernelMap,
) {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svmdataset.c\0" as *const u8 as *const libc::c_char,
            241 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 88],
                &[libc::c_char; 88],
            >(
                b"void vl_svmdataset_set_homogeneous_kernel_map(VlSvmDataset *, VlHomogeneousKernelMap *)\0",
            ))
                .as_ptr(),
        );
    }
    (*self_0).hom = hom;
    (*self_0).homDimension = 0 as libc::c_int as vl_size;
    if !((*self_0).homBuffer).is_null() {
        vl_free((*self_0).homBuffer);
        (*self_0).homBuffer = 0 as *mut libc::c_void;
    }
    if !((*self_0).hom).is_null() {
        (*self_0).homDimension = vl_homogeneouskernelmap_get_dimension((*self_0).hom);
        (*self_0)
            .homBuffer = vl_calloc(
            (*self_0).homDimension as size_t,
            vl_get_type_size((*self_0).dataType) as size_t,
        );
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_svmdataset_get_accumulate_function(
    mut self_0: *const VlSvmDataset,
) -> VlSvmAccumulateFunction {
    if ((*self_0).hom).is_null() {
        match (*self_0).dataType {
            1 => {
                return ::core::mem::transmute::<
                    Option::<
                        unsafe extern "C" fn(
                            *const VlSvmDataset,
                            vl_uindex,
                            *mut libc::c_double,
                            libc::c_double,
                        ) -> (),
                    >,
                    VlSvmAccumulateFunction,
                >(
                    Some(
                        vl_svmdataset_accumulate_f
                            as unsafe extern "C" fn(
                                *const VlSvmDataset,
                                vl_uindex,
                                *mut libc::c_double,
                                libc::c_double,
                            ) -> (),
                    ),
                );
            }
            2 => {
                return ::core::mem::transmute::<
                    Option::<
                        unsafe extern "C" fn(
                            *const VlSvmDataset,
                            vl_uindex,
                            *mut libc::c_double,
                            libc::c_double,
                        ) -> (),
                    >,
                    VlSvmAccumulateFunction,
                >(
                    Some(
                        vl_svmdataset_accumulate_d
                            as unsafe extern "C" fn(
                                *const VlSvmDataset,
                                vl_uindex,
                                *mut libc::c_double,
                                libc::c_double,
                            ) -> (),
                    ),
                );
            }
            _ => {}
        }
    } else {
        match (*self_0).dataType {
            1 => {
                return ::core::mem::transmute::<
                    Option::<
                        unsafe extern "C" fn(
                            *const VlSvmDataset,
                            vl_uindex,
                            *mut libc::c_double,
                            libc::c_double,
                        ) -> (),
                    >,
                    VlSvmAccumulateFunction,
                >(
                    Some(
                        vl_svmdataset_accumulate_hom_f
                            as unsafe extern "C" fn(
                                *const VlSvmDataset,
                                vl_uindex,
                                *mut libc::c_double,
                                libc::c_double,
                            ) -> (),
                    ),
                );
            }
            2 => {
                return ::core::mem::transmute::<
                    Option::<
                        unsafe extern "C" fn(
                            *const VlSvmDataset,
                            vl_uindex,
                            *mut libc::c_double,
                            libc::c_double,
                        ) -> (),
                    >,
                    VlSvmAccumulateFunction,
                >(
                    Some(
                        vl_svmdataset_accumulate_hom_d
                            as unsafe extern "C" fn(
                                *const VlSvmDataset,
                                vl_uindex,
                                *mut libc::c_double,
                                libc::c_double,
                            ) -> (),
                    ),
                );
            }
            _ => {}
        }
    }
    __assert_fail(
        b"0\0" as *const u8 as *const libc::c_char,
        b"vl/svmdataset.c\0" as *const u8 as *const libc::c_char,
        281 as libc::c_int as libc::c_uint,
        (*::core::mem::transmute::<
            &[u8; 84],
            &[libc::c_char; 84],
        >(
            b"VlSvmAccumulateFunction vl_svmdataset_get_accumulate_function(const VlSvmDataset *)\0",
        ))
            .as_ptr(),
    );
    return None;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svmdataset_get_inner_product_function(
    mut self_0: *const VlSvmDataset,
) -> VlSvmInnerProductFunction {
    if ((*self_0).hom).is_null() {
        match (*self_0).dataType {
            1 => {
                return ::core::mem::transmute::<
                    Option::<
                        unsafe extern "C" fn(
                            *const VlSvmDataset,
                            vl_uindex,
                            *const libc::c_double,
                        ) -> libc::c_double,
                    >,
                    VlSvmInnerProductFunction,
                >(
                    Some(
                        _vl_svmdataset_inner_product_f
                            as unsafe extern "C" fn(
                                *const VlSvmDataset,
                                vl_uindex,
                                *const libc::c_double,
                            ) -> libc::c_double,
                    ),
                );
            }
            2 => {
                return ::core::mem::transmute::<
                    Option::<
                        unsafe extern "C" fn(
                            *const VlSvmDataset,
                            vl_uindex,
                            *const libc::c_double,
                        ) -> libc::c_double,
                    >,
                    VlSvmInnerProductFunction,
                >(
                    Some(
                        _vl_svmdataset_inner_product_d
                            as unsafe extern "C" fn(
                                *const VlSvmDataset,
                                vl_uindex,
                                *const libc::c_double,
                            ) -> libc::c_double,
                    ),
                );
            }
            _ => {
                __assert_fail(
                    b"0\0" as *const u8 as *const libc::c_char,
                    b"vl/svmdataset.c\0" as *const u8 as *const libc::c_char,
                    302 as libc::c_int as libc::c_uint,
                    (*::core::mem::transmute::<
                        &[u8; 89],
                        &[libc::c_char; 89],
                    >(
                        b"VlSvmInnerProductFunction vl_svmdataset_get_inner_product_function(const VlSvmDataset *)\0",
                    ))
                        .as_ptr(),
                );
            }
        }
    } else {
        match (*self_0).dataType {
            1 => {
                return ::core::mem::transmute::<
                    Option::<
                        unsafe extern "C" fn(
                            *const VlSvmDataset,
                            vl_uindex,
                            *const libc::c_double,
                        ) -> libc::c_double,
                    >,
                    VlSvmInnerProductFunction,
                >(
                    Some(
                        _vl_svmdataset_inner_product_hom_f
                            as unsafe extern "C" fn(
                                *const VlSvmDataset,
                                vl_uindex,
                                *const libc::c_double,
                            ) -> libc::c_double,
                    ),
                );
            }
            2 => {
                return ::core::mem::transmute::<
                    Option::<
                        unsafe extern "C" fn(
                            *const VlSvmDataset,
                            vl_uindex,
                            *const libc::c_double,
                        ) -> libc::c_double,
                    >,
                    VlSvmInnerProductFunction,
                >(
                    Some(
                        _vl_svmdataset_inner_product_hom_d
                            as unsafe extern "C" fn(
                                *const VlSvmDataset,
                                vl_uindex,
                                *const libc::c_double,
                            ) -> libc::c_double,
                    ),
                );
            }
            _ => {
                __assert_fail(
                    b"0\0" as *const u8 as *const libc::c_char,
                    b"vl/svmdataset.c\0" as *const u8 as *const libc::c_char,
                    313 as libc::c_int as libc::c_uint,
                    (*::core::mem::transmute::<
                        &[u8; 89],
                        &[libc::c_char; 89],
                    >(
                        b"VlSvmInnerProductFunction vl_svmdataset_get_inner_product_function(const VlSvmDataset *)\0",
                    ))
                        .as_ptr(),
                );
            }
        }
    }
    return None;
}
#[no_mangle]
pub unsafe extern "C" fn _vl_svmdataset_inner_product_f(
    mut self_0: *const VlSvmDataset,
    mut element: vl_uindex,
    mut model: *const libc::c_double,
) -> libc::c_double {
    let mut product: libc::c_double = 0 as libc::c_int as libc::c_double;
    let mut data: *mut libc::c_float = ((*self_0).data as *mut libc::c_float)
        .offset(((*self_0).dimension).wrapping_mul(element) as isize);
    let mut end: *mut libc::c_float = data.offset((*self_0).dimension as isize);
    while data != end {
        let fresh0 = data;
        data = data.offset(1);
        let fresh1 = model;
        model = model.offset(1);
        product += *fresh0 as libc::c_double * *fresh1;
    }
    return product;
}
#[no_mangle]
pub unsafe extern "C" fn _vl_svmdataset_inner_product_d(
    mut self_0: *const VlSvmDataset,
    mut element: vl_uindex,
    mut model: *const libc::c_double,
) -> libc::c_double {
    let mut product: libc::c_double = 0 as libc::c_int as libc::c_double;
    let mut data: *mut libc::c_double = ((*self_0).data as *mut libc::c_double)
        .offset(((*self_0).dimension).wrapping_mul(element) as isize);
    let mut end: *mut libc::c_double = data.offset((*self_0).dimension as isize);
    while data != end {
        let fresh2 = data;
        data = data.offset(1);
        let fresh3 = model;
        model = model.offset(1);
        product += *fresh2 * *fresh3;
    }
    return product;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svmdataset_accumulate_f(
    mut self_0: *const VlSvmDataset,
    mut element: vl_uindex,
    mut model: *mut libc::c_double,
    multiplier: libc::c_double,
) {
    let mut data: *mut libc::c_float = ((*self_0).data as *mut libc::c_float)
        .offset(((*self_0).dimension).wrapping_mul(element) as isize);
    let mut end: *mut libc::c_float = data.offset((*self_0).dimension as isize);
    while data != end {
        let fresh4 = data;
        data = data.offset(1);
        *model += *fresh4 as libc::c_double * multiplier;
        model = model.offset(1);
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_svmdataset_accumulate_d(
    mut self_0: *const VlSvmDataset,
    mut element: vl_uindex,
    mut model: *mut libc::c_double,
    multiplier: libc::c_double,
) {
    let mut data: *mut libc::c_double = ((*self_0).data as *mut libc::c_double)
        .offset(((*self_0).dimension).wrapping_mul(element) as isize);
    let mut end: *mut libc::c_double = data.offset((*self_0).dimension as isize);
    while data != end {
        let fresh5 = data;
        data = data.offset(1);
        *model += *fresh5 * multiplier;
        model = model.offset(1);
    }
}
#[no_mangle]
pub unsafe extern "C" fn _vl_svmdataset_inner_product_hom_d(
    mut self_0: *const VlSvmDataset,
    mut element: vl_uindex,
    mut model: *const libc::c_double,
) -> libc::c_double {
    let mut product: libc::c_double = 0 as libc::c_int as libc::c_double;
    let mut data: *mut libc::c_double = ((*self_0).data as *mut libc::c_double)
        .offset(((*self_0).dimension).wrapping_mul(element) as isize);
    let mut end: *mut libc::c_double = data.offset((*self_0).dimension as isize);
    let mut bufEnd: *mut libc::c_double = ((*self_0).homBuffer as *mut libc::c_double)
        .offset((*self_0).homDimension as isize);
    while data != end {
        let mut buf: *mut libc::c_double = (*self_0).homBuffer as *mut libc::c_double;
        let fresh6 = data;
        data = data.offset(1);
        vl_homogeneouskernelmap_evaluate_d(
            (*self_0).hom,
            (*self_0).homBuffer as *mut libc::c_double,
            1 as libc::c_int as vl_size,
            *fresh6,
        );
        while buf != bufEnd {
            let fresh7 = buf;
            buf = buf.offset(1);
            let fresh8 = model;
            model = model.offset(1);
            product += *fresh7 * *fresh8;
        }
    }
    return product;
}
#[no_mangle]
pub unsafe extern "C" fn _vl_svmdataset_inner_product_hom_f(
    mut self_0: *const VlSvmDataset,
    mut element: vl_uindex,
    mut model: *const libc::c_double,
) -> libc::c_double {
    let mut product: libc::c_double = 0 as libc::c_int as libc::c_double;
    let mut data: *mut libc::c_float = ((*self_0).data as *mut libc::c_float)
        .offset(((*self_0).dimension).wrapping_mul(element) as isize);
    let mut end: *mut libc::c_float = data.offset((*self_0).dimension as isize);
    let mut bufEnd: *mut libc::c_float = ((*self_0).homBuffer as *mut libc::c_float)
        .offset((*self_0).homDimension as isize);
    while data != end {
        let mut buf: *mut libc::c_float = (*self_0).homBuffer as *mut libc::c_float;
        let fresh9 = data;
        data = data.offset(1);
        vl_homogeneouskernelmap_evaluate_f(
            (*self_0).hom,
            (*self_0).homBuffer as *mut libc::c_float,
            1 as libc::c_int as vl_size,
            *fresh9 as libc::c_double,
        );
        while buf != bufEnd {
            let fresh10 = buf;
            buf = buf.offset(1);
            let fresh11 = model;
            model = model.offset(1);
            product += *fresh10 as libc::c_double * *fresh11;
        }
    }
    return product;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svmdataset_accumulate_hom_f(
    mut self_0: *const VlSvmDataset,
    mut element: vl_uindex,
    mut model: *mut libc::c_double,
    multiplier: libc::c_double,
) {
    let mut data: *mut libc::c_float = ((*self_0).data as *mut libc::c_float)
        .offset(((*self_0).dimension).wrapping_mul(element) as isize);
    let mut end: *mut libc::c_float = data.offset((*self_0).dimension as isize);
    let mut bufEnd: *mut libc::c_float = ((*self_0).homBuffer as *mut libc::c_float)
        .offset((*self_0).homDimension as isize);
    while data != end {
        let mut buf: *mut libc::c_float = (*self_0).homBuffer as *mut libc::c_float;
        let fresh12 = data;
        data = data.offset(1);
        vl_homogeneouskernelmap_evaluate_f(
            (*self_0).hom,
            (*self_0).homBuffer as *mut libc::c_float,
            1 as libc::c_int as vl_size,
            *fresh12 as libc::c_double,
        );
        while buf != bufEnd {
            let fresh13 = buf;
            buf = buf.offset(1);
            *model += *fresh13 as libc::c_double * multiplier;
            model = model.offset(1);
        }
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_svmdataset_accumulate_hom_d(
    mut self_0: *const VlSvmDataset,
    mut element: vl_uindex,
    mut model: *mut libc::c_double,
    multiplier: libc::c_double,
) {
    let mut data: *mut libc::c_double = ((*self_0).data as *mut libc::c_double)
        .offset(((*self_0).dimension).wrapping_mul(element) as isize);
    let mut end: *mut libc::c_double = data.offset((*self_0).dimension as isize);
    let mut bufEnd: *mut libc::c_double = ((*self_0).homBuffer as *mut libc::c_double)
        .offset((*self_0).homDimension as isize);
    while data != end {
        let mut buf: *mut libc::c_double = (*self_0).homBuffer as *mut libc::c_double;
        let fresh14 = data;
        data = data.offset(1);
        vl_homogeneouskernelmap_evaluate_d(
            (*self_0).hom,
            (*self_0).homBuffer as *mut libc::c_double,
            1 as libc::c_int as vl_size,
            *fresh14,
        );
        while buf != bufEnd {
            let fresh15 = buf;
            buf = buf.offset(1);
            *model += *fresh15 * multiplier;
            model = model.offset(1);
        }
    }
}
