use ::libc;
extern "C" {
    pub type VlSvmDataset_;
    fn __assert_fail(
        __assertion: *const libc::c_char,
        __file: *const libc::c_char,
        __line: libc::c_uint,
        __function: *const libc::c_char,
    ) -> !;
    fn vl_get_rand() -> *mut VlRand;
    fn vl_calloc(n: size_t, size: size_t) -> *mut libc::c_void;
    fn vl_free(ptr: *mut libc::c_void);
    fn vl_get_cpu_time() -> libc::c_double;
    fn exp(_: libc::c_double) -> libc::c_double;
    fn log(_: libc::c_double) -> libc::c_double;
    fn sqrt(_: libc::c_double) -> libc::c_double;
    fn vl_rand_permute_indexes(self_0: *mut VlRand, array: *mut vl_index, size: vl_size);
    fn vl_svmdataset_new(
        dataType: vl_type,
        data: *mut libc::c_void,
        dimension: vl_size,
        numData: vl_size,
    ) -> *mut VlSvmDataset;
    fn vl_svmdataset_delete(dataset: *mut VlSvmDataset);
    fn vl_svmdataset_get_num_data(self_0: *const VlSvmDataset) -> vl_size;
    fn vl_svmdataset_get_dimension(self_0: *const VlSvmDataset) -> vl_size;
    fn vl_svmdataset_get_accumulate_function(
        self_0: *const VlSvmDataset,
    ) -> VlSvmAccumulateFunction;
    fn vl_svmdataset_get_inner_product_function(
        self_0: *const VlSvmDataset,
    ) -> VlSvmInnerProductFunction;
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
pub type vl_uint64 = libc::c_ulonglong;
pub type vl_uint32 = libc::c_uint;
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
pub type vl_type = vl_uint32;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct VlSvm_ {
    pub solver: VlSvmSolverType,
    pub dimension: vl_size,
    pub model: *mut libc::c_double,
    pub bias: libc::c_double,
    pub biasMultiplier: libc::c_double,
    pub lambda: libc::c_double,
    pub data: *const libc::c_void,
    pub numData: vl_size,
    pub labels: *const libc::c_double,
    pub weights: *const libc::c_double,
    pub ownDataset: *mut VlSvmDataset,
    pub diagnosticFn: VlSvmDiagnosticFunction,
    pub diagnosticFnData: *mut libc::c_void,
    pub diagnosticFrequency: vl_size,
    pub lossFn: VlSvmLossFunction,
    pub conjugateLossFn: VlSvmLossFunction,
    pub lossDerivativeFn: VlSvmLossFunction,
    pub dcaUpdateFn: VlSvmDcaUpdateFunction,
    pub innerProductFn: VlSvmInnerProductFunction,
    pub accumulateFn: VlSvmAccumulateFunction,
    pub iteration: vl_size,
    pub maxNumIterations: vl_size,
    pub epsilon: libc::c_double,
    pub statistics: VlSvmStatistics,
    pub scores: *mut libc::c_double,
    pub biasLearningRate: libc::c_double,
    pub alpha: *mut libc::c_double,
}
pub type VlSvmStatistics = VlSvmStatistics_;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct VlSvmStatistics_ {
    pub status: VlSvmSolverStatus,
    pub iteration: vl_size,
    pub epoch: vl_size,
    pub objective: libc::c_double,
    pub regularizer: libc::c_double,
    pub loss: libc::c_double,
    pub dualObjective: libc::c_double,
    pub dualLoss: libc::c_double,
    pub dualityGap: libc::c_double,
    pub scoresVariation: libc::c_double,
    pub elapsedTime: libc::c_double,
}
pub type VlSvmSolverStatus = libc::c_uint;
pub const VlSvmStatusMaxNumIterationsReached: VlSvmSolverStatus = 3;
pub const VlSvmStatusConverged: VlSvmSolverStatus = 2;
pub const VlSvmStatusTraining: VlSvmSolverStatus = 1;
pub type VlSvmAccumulateFunction = Option::<
    unsafe extern "C" fn(
        *const libc::c_void,
        vl_uindex,
        *mut libc::c_double,
        libc::c_double,
    ) -> (),
>;
pub type VlSvmInnerProductFunction = Option::<
    unsafe extern "C" fn(
        *const libc::c_void,
        vl_uindex,
        *mut libc::c_double,
    ) -> libc::c_double,
>;
pub type VlSvmDcaUpdateFunction = Option::<
    unsafe extern "C" fn(
        libc::c_double,
        libc::c_double,
        libc::c_double,
        libc::c_double,
    ) -> libc::c_double,
>;
pub type VlSvmLossFunction = Option::<
    unsafe extern "C" fn(libc::c_double, libc::c_double) -> libc::c_double,
>;
pub type VlSvmDiagnosticFunction = Option::<
    unsafe extern "C" fn(*mut VlSvm_, *mut libc::c_void) -> (),
>;
pub type VlSvmDataset = VlSvmDataset_;
pub type VlSvmSolverType = libc::c_uint;
pub const VlSvmSolverSdca: VlSvmSolverType = 2;
pub const VlSvmSolverSgd: VlSvmSolverType = 1;
pub const VlSvmSolverNone: VlSvmSolverType = 0;
pub type VlSvm = VlSvm_;
pub type VlSvmLossType = libc::c_uint;
pub const VlSvmLossLogistic: VlSvmLossType = 4;
pub const VlSvmLossL2: VlSvmLossType = 3;
pub const VlSvmLossL1: VlSvmLossType = 2;
pub const VlSvmLossHinge2: VlSvmLossType = 1;
pub const VlSvmLossHinge: VlSvmLossType = 0;
#[derive(Copy, Clone)]
#[repr(C)]
pub union C2RustUnnamed {
    pub raw: vl_uint64,
    pub value: libc::c_double,
}
static mut vl_infinity_d: C2RustUnnamed = C2RustUnnamed {
    raw: 0x7ff0000000000000 as libc::c_ulonglong,
};
#[inline]
unsafe extern "C" fn vl_abs_d(mut x: libc::c_double) -> libc::c_double {
    return x.abs();
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_new(
    mut type_0: VlSvmSolverType,
    mut data: *const libc::c_double,
    mut dimension: vl_size,
    mut numData: vl_size,
    mut labels: *const libc::c_double,
    mut lambda: libc::c_double,
) -> *mut VlSvm {
    let mut dataset: *mut VlSvmDataset = vl_svmdataset_new(
        2 as libc::c_int as vl_type,
        data as *mut libc::c_void,
        dimension,
        numData,
    );
    let mut self_0: *mut VlSvm = vl_svm_new_with_dataset(
        type_0,
        dataset,
        labels,
        lambda,
    );
    (*self_0).ownDataset = dataset;
    return self_0;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_new_with_dataset(
    mut solver: VlSvmSolverType,
    mut dataset: *mut VlSvmDataset,
    mut labels: *const libc::c_double,
    mut lambda: libc::c_double,
) -> *mut VlSvm {
    let mut self_0: *mut VlSvm = vl_svm_new_with_abstract_data(
        solver,
        dataset as *mut libc::c_void,
        vl_svmdataset_get_dimension(dataset),
        vl_svmdataset_get_num_data(dataset),
        labels,
        lambda,
    );
    vl_svm_set_data_functions(
        self_0,
        vl_svmdataset_get_inner_product_function(dataset),
        vl_svmdataset_get_accumulate_function(dataset),
    );
    return self_0;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_new_with_abstract_data(
    mut solver: VlSvmSolverType,
    mut data: *mut libc::c_void,
    mut dimension: vl_size,
    mut numData: vl_size,
    mut labels: *const libc::c_double,
    mut lambda: libc::c_double,
) -> *mut VlSvm {
    let mut current_block: u64;
    let mut self_0: *mut VlSvm = vl_calloc(
        1 as libc::c_int as size_t,
        ::core::mem::size_of::<VlSvm>() as libc::c_ulong,
    ) as *mut VlSvm;
    if dimension >= 1 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"dimension >= 1\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1023 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 104],
                &[libc::c_char; 104],
            >(
                b"VlSvm *vl_svm_new_with_abstract_data(VlSvmSolverType, void *, vl_size, vl_size, const double *, double)\0",
            ))
                .as_ptr(),
        );
    }
    if numData >= 1 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"numData >= 1\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1024 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 104],
                &[libc::c_char; 104],
            >(
                b"VlSvm *vl_svm_new_with_abstract_data(VlSvmSolverType, void *, vl_size, vl_size, const double *, double)\0",
            ))
                .as_ptr(),
        );
    }
    if !labels.is_null() {} else {
        __assert_fail(
            b"labels\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1025 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 104],
                &[libc::c_char; 104],
            >(
                b"VlSvm *vl_svm_new_with_abstract_data(VlSvmSolverType, void *, vl_size, vl_size, const double *, double)\0",
            ))
                .as_ptr(),
        );
    }
    (*self_0).solver = solver;
    (*self_0).dimension = dimension;
    (*self_0).model = 0 as *mut libc::c_double;
    (*self_0).bias = 0 as libc::c_int as libc::c_double;
    (*self_0).biasMultiplier = 1.0f64;
    (*self_0).lambda = lambda;
    (*self_0).data = data;
    (*self_0).numData = numData;
    (*self_0).labels = labels;
    (*self_0).diagnosticFrequency = numData;
    (*self_0).diagnosticFn = None;
    (*self_0).diagnosticFnData = 0 as *mut libc::c_void;
    (*self_0)
        .lossFn = Some(
        vl_svm_hinge_loss
            as unsafe extern "C" fn(libc::c_double, libc::c_double) -> libc::c_double,
    );
    (*self_0)
        .conjugateLossFn = Some(
        vl_svm_hinge_conjugate_loss
            as unsafe extern "C" fn(libc::c_double, libc::c_double) -> libc::c_double,
    );
    (*self_0)
        .lossDerivativeFn = Some(
        vl_svm_hinge_loss_derivative
            as unsafe extern "C" fn(libc::c_double, libc::c_double) -> libc::c_double,
    );
    (*self_0)
        .dcaUpdateFn = Some(
        vl_svm_hinge_dca_update
            as unsafe extern "C" fn(
                libc::c_double,
                libc::c_double,
                libc::c_double,
                libc::c_double,
            ) -> libc::c_double,
    );
    (*self_0).innerProductFn = None;
    (*self_0).accumulateFn = None;
    (*self_0).iteration = 0 as libc::c_int as vl_size;
    (*self_0)
        .maxNumIterations = (if numData as libc::c_double
        > f32::ceil((10.0f64 / lambda) as libc::c_float) as libc::c_double
    {
        numData as libc::c_double
    } else {
        f32::ceil((10.0f64 / lambda) as libc::c_float) as libc::c_double
    }) as vl_size;
    (*self_0).epsilon = 1e-2f64;
    (*self_0).biasLearningRate = 0.01f64;
    (*self_0).alpha = 0 as *mut libc::c_double;
    (*self_0)
        .model = vl_calloc(
        dimension as size_t,
        ::core::mem::size_of::<libc::c_double>() as libc::c_ulong,
    ) as *mut libc::c_double;
    if !((*self_0).model).is_null() {
        if (*self_0).solver as libc::c_uint
            == VlSvmSolverSdca as libc::c_int as libc::c_uint
        {
            (*self_0)
                .alpha = vl_calloc(
                (*self_0).numData as size_t,
                ::core::mem::size_of::<libc::c_double>() as libc::c_ulong,
            ) as *mut libc::c_double;
            if ((*self_0).alpha).is_null() {
                current_block = 6991336440260680384;
            } else {
                current_block = 26972500619410423;
            }
        } else {
            current_block = 26972500619410423;
        }
        match current_block {
            6991336440260680384 => {}
            _ => {
                (*self_0)
                    .scores = vl_calloc(
                    numData as size_t,
                    ::core::mem::size_of::<libc::c_double>() as libc::c_ulong,
                ) as *mut libc::c_double;
                if !((*self_0).scores).is_null() {
                    return self_0;
                }
            }
        }
    }
    if !((*self_0).scores).is_null() {
        vl_free((*self_0).scores as *mut libc::c_void);
        (*self_0).scores = 0 as *mut libc::c_double;
    }
    if !((*self_0).model).is_null() {
        vl_free((*self_0).model as *mut libc::c_void);
        (*self_0).model = 0 as *mut libc::c_double;
    }
    if !((*self_0).alpha).is_null() {
        vl_free((*self_0).alpha as *mut libc::c_void);
        (*self_0).alpha = 0 as *mut libc::c_double;
    }
    return 0 as *mut VlSvm;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_delete(mut self_0: *mut VlSvm) {
    if !((*self_0).model).is_null() {
        vl_free((*self_0).model as *mut libc::c_void);
        (*self_0).model = 0 as *mut libc::c_double;
    }
    if !((*self_0).alpha).is_null() {
        vl_free((*self_0).alpha as *mut libc::c_void);
        (*self_0).alpha = 0 as *mut libc::c_double;
    }
    if !((*self_0).ownDataset).is_null() {
        vl_svmdataset_delete((*self_0).ownDataset);
        (*self_0).ownDataset = 0 as *mut VlSvmDataset;
    }
    vl_free(self_0 as *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_set_epsilon(
    mut self_0: *mut VlSvm,
    mut epsilon: libc::c_double,
) {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1125 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 41],
                &[libc::c_char; 41],
            >(b"void vl_svm_set_epsilon(VlSvm *, double)\0"))
                .as_ptr(),
        );
    }
    if epsilon >= 0 as libc::c_int as libc::c_double {} else {
        __assert_fail(
            b"epsilon >= 0\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1126 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 41],
                &[libc::c_char; 41],
            >(b"void vl_svm_set_epsilon(VlSvm *, double)\0"))
                .as_ptr(),
        );
    }
    (*self_0).epsilon = epsilon;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_get_epsilon(mut self_0: *const VlSvm) -> libc::c_double {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1137 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 41],
                &[libc::c_char; 41],
            >(b"double vl_svm_get_epsilon(const VlSvm *)\0"))
                .as_ptr(),
        );
    }
    return (*self_0).epsilon;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_set_bias_learning_rate(
    mut self_0: *mut VlSvm,
    mut rate: libc::c_double,
) {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1150 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 52],
                &[libc::c_char; 52],
            >(b"void vl_svm_set_bias_learning_rate(VlSvm *, double)\0"))
                .as_ptr(),
        );
    }
    if rate > 0 as libc::c_int as libc::c_double {} else {
        __assert_fail(
            b"rate > 0\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1151 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 52],
                &[libc::c_char; 52],
            >(b"void vl_svm_set_bias_learning_rate(VlSvm *, double)\0"))
                .as_ptr(),
        );
    }
    (*self_0).biasLearningRate = rate;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_get_bias_learning_rate(
    mut self_0: *const VlSvm,
) -> libc::c_double {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1162 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 52],
                &[libc::c_char; 52],
            >(b"double vl_svm_get_bias_learning_rate(const VlSvm *)\0"))
                .as_ptr(),
        );
    }
    return (*self_0).biasLearningRate;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_set_bias_multiplier(
    mut self_0: *mut VlSvm,
    mut b: libc::c_double,
) {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1176 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 49],
                &[libc::c_char; 49],
            >(b"void vl_svm_set_bias_multiplier(VlSvm *, double)\0"))
                .as_ptr(),
        );
    }
    if b >= 0 as libc::c_int as libc::c_double {} else {
        __assert_fail(
            b"b >= 0\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1177 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 49],
                &[libc::c_char; 49],
            >(b"void vl_svm_set_bias_multiplier(VlSvm *, double)\0"))
                .as_ptr(),
        );
    }
    (*self_0).biasMultiplier = b;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_get_bias_multiplier(
    mut self_0: *const VlSvm,
) -> libc::c_double {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1188 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 49],
                &[libc::c_char; 49],
            >(b"double vl_svm_get_bias_multiplier(const VlSvm *)\0"))
                .as_ptr(),
        );
    }
    return (*self_0).biasMultiplier;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_set_iteration_number(
    mut self_0: *mut VlSvm,
    mut n: vl_uindex,
) {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1203 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 53],
                &[libc::c_char; 53],
            >(b"void vl_svm_set_iteration_number(VlSvm *, vl_uindex)\0"))
                .as_ptr(),
        );
    }
    (*self_0).iteration = n;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_get_iteration_number(
    mut self_0: *const VlSvm,
) -> vl_size {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1214 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 51],
                &[libc::c_char; 51],
            >(b"vl_size vl_svm_get_iteration_number(const VlSvm *)\0"))
                .as_ptr(),
        );
    }
    return (*self_0).iteration;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_set_max_num_iterations(
    mut self_0: *mut VlSvm,
    mut n: vl_size,
) {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1225 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 53],
                &[libc::c_char; 53],
            >(b"void vl_svm_set_max_num_iterations(VlSvm *, vl_size)\0"))
                .as_ptr(),
        );
    }
    (*self_0).maxNumIterations = n;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_get_max_num_iterations(
    mut self_0: *const VlSvm,
) -> vl_size {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1236 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 53],
                &[libc::c_char; 53],
            >(b"vl_size vl_svm_get_max_num_iterations(const VlSvm *)\0"))
                .as_ptr(),
        );
    }
    return (*self_0).maxNumIterations;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_set_diagnostic_frequency(
    mut self_0: *mut VlSvm,
    mut f: vl_size,
) {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1250 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 55],
                &[libc::c_char; 55],
            >(b"void vl_svm_set_diagnostic_frequency(VlSvm *, vl_size)\0"))
                .as_ptr(),
        );
    }
    if f > 0 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"f > 0\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1251 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 55],
                &[libc::c_char; 55],
            >(b"void vl_svm_set_diagnostic_frequency(VlSvm *, vl_size)\0"))
                .as_ptr(),
        );
    }
    (*self_0).diagnosticFrequency = f;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_get_diagnostic_frequency(
    mut self_0: *const VlSvm,
) -> vl_size {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1262 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 55],
                &[libc::c_char; 55],
            >(b"vl_size vl_svm_get_diagnostic_frequency(const VlSvm *)\0"))
                .as_ptr(),
        );
    }
    return (*self_0).diagnosticFrequency;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_get_solver(mut self_0: *const VlSvm) -> VlSvmSolverType {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1273 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 49],
                &[libc::c_char; 49],
            >(b"VlSvmSolverType vl_svm_get_solver(const VlSvm *)\0"))
                .as_ptr(),
        );
    }
    return (*self_0).solver;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_set_lambda(
    mut self_0: *mut VlSvm,
    mut lambda: libc::c_double,
) {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1290 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 40],
                &[libc::c_char; 40],
            >(b"void vl_svm_set_lambda(VlSvm *, double)\0"))
                .as_ptr(),
        );
    }
    if lambda >= 0 as libc::c_int as libc::c_double {} else {
        __assert_fail(
            b"lambda >= 0\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1291 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 40],
                &[libc::c_char; 40],
            >(b"void vl_svm_set_lambda(VlSvm *, double)\0"))
                .as_ptr(),
        );
    }
    (*self_0).lambda = lambda;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_get_lambda(mut self_0: *const VlSvm) -> libc::c_double {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1302 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 40],
                &[libc::c_char; 40],
            >(b"double vl_svm_get_lambda(const VlSvm *)\0"))
                .as_ptr(),
        );
    }
    return (*self_0).lambda;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_set_weights(
    mut self_0: *mut VlSvm,
    mut weights: *const libc::c_double,
) {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1322 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 49],
                &[libc::c_char; 49],
            >(b"void vl_svm_set_weights(VlSvm *, const double *)\0"))
                .as_ptr(),
        );
    }
    (*self_0).weights = weights;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_get_weights(
    mut self_0: *const VlSvm,
) -> *const libc::c_double {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1333 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 48],
                &[libc::c_char; 48],
            >(b"const double *vl_svm_get_weights(const VlSvm *)\0"))
                .as_ptr(),
        );
    }
    return (*self_0).weights;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_get_dimension(mut self_0: *mut VlSvm) -> vl_size {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1350 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 38],
                &[libc::c_char; 38],
            >(b"vl_size vl_svm_get_dimension(VlSvm *)\0"))
                .as_ptr(),
        );
    }
    return (*self_0).dimension;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_get_num_data(mut self_0: *mut VlSvm) -> vl_size {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1363 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 37],
                &[libc::c_char; 37],
            >(b"vl_size vl_svm_get_num_data(VlSvm *)\0"))
                .as_ptr(),
        );
    }
    return (*self_0).numData;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_get_model(
    mut self_0: *const VlSvm,
) -> *const libc::c_double {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1376 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 46],
                &[libc::c_char; 46],
            >(b"const double *vl_svm_get_model(const VlSvm *)\0"))
                .as_ptr(),
        );
    }
    return (*self_0).model;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_set_model(
    mut self_0: *mut VlSvm,
    mut model: *const libc::c_double,
) {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1391 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 47],
                &[libc::c_char; 47],
            >(b"void vl_svm_set_model(VlSvm *, const double *)\0"))
                .as_ptr(),
        );
    }
    if !model.is_null() {} else {
        __assert_fail(
            b"model\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1392 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 47],
                &[libc::c_char; 47],
            >(b"void vl_svm_set_model(VlSvm *, const double *)\0"))
                .as_ptr(),
        );
    }
    memcpy(
        (*self_0).model as *mut libc::c_void,
        model as *const libc::c_void,
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(vl_svm_get_dimension(self_0)) as libc::c_ulong,
    );
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_set_bias(mut self_0: *mut VlSvm, mut b: libc::c_double) {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1408 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 38],
                &[libc::c_char; 38],
            >(b"void vl_svm_set_bias(VlSvm *, double)\0"))
                .as_ptr(),
        );
    }
    if (*self_0).biasMultiplier != 0. {
        (*self_0).bias = b / (*self_0).biasMultiplier;
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_get_bias(mut self_0: *const VlSvm) -> libc::c_double {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1424 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 38],
                &[libc::c_char; 38],
            >(b"double vl_svm_get_bias(const VlSvm *)\0"))
                .as_ptr(),
        );
    }
    return (*self_0).bias * (*self_0).biasMultiplier;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_get_statistics(
    mut self_0: *const VlSvm,
) -> *const VlSvmStatistics {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1435 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 60],
                &[libc::c_char; 60],
            >(b"const VlSvmStatistics *vl_svm_get_statistics(const VlSvm *)\0"))
                .as_ptr(),
        );
    }
    return &(*self_0).statistics;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_get_scores(
    mut self_0: *const VlSvm,
) -> *const libc::c_double {
    return (*self_0).scores;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_set_diagnostic_function(
    mut self_0: *mut VlSvm,
    mut f: VlSvmDiagnosticFunction,
    mut data: *mut libc::c_void,
) {
    (*self_0).diagnosticFn = f;
    (*self_0).diagnosticFnData = data;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_set_data_functions(
    mut self_0: *mut VlSvm,
    mut inner: VlSvmInnerProductFunction,
    mut acc: VlSvmAccumulateFunction,
) {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1494 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 92],
                &[libc::c_char; 92],
            >(
                b"void vl_svm_set_data_functions(VlSvm *, VlSvmInnerProductFunction, VlSvmAccumulateFunction)\0",
            ))
                .as_ptr(),
        );
    }
    if inner.is_some() {} else {
        __assert_fail(
            b"inner\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1495 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 92],
                &[libc::c_char; 92],
            >(
                b"void vl_svm_set_data_functions(VlSvm *, VlSvmInnerProductFunction, VlSvmAccumulateFunction)\0",
            ))
                .as_ptr(),
        );
    }
    if acc.is_some() {} else {
        __assert_fail(
            b"acc\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1496 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 92],
                &[libc::c_char; 92],
            >(
                b"void vl_svm_set_data_functions(VlSvm *, VlSvmInnerProductFunction, VlSvmAccumulateFunction)\0",
            ))
                .as_ptr(),
        );
    }
    (*self_0).innerProductFn = inner;
    (*self_0).accumulateFn = acc;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_set_loss_function(
    mut self_0: *mut VlSvm,
    mut f: VlSvmLossFunction,
) {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1511 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 58],
                &[libc::c_char; 58],
            >(b"void vl_svm_set_loss_function(VlSvm *, VlSvmLossFunction)\0"))
                .as_ptr(),
        );
    }
    (*self_0).lossFn = f;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_set_loss_derivative_function(
    mut self_0: *mut VlSvm,
    mut f: VlSvmLossFunction,
) {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1521 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 69],
                &[libc::c_char; 69],
            >(b"void vl_svm_set_loss_derivative_function(VlSvm *, VlSvmLossFunction)\0"))
                .as_ptr(),
        );
    }
    (*self_0).lossDerivativeFn = f;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_set_conjugate_loss_function(
    mut self_0: *mut VlSvm,
    mut f: VlSvmLossFunction,
) {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1531 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 68],
                &[libc::c_char; 68],
            >(b"void vl_svm_set_conjugate_loss_function(VlSvm *, VlSvmLossFunction)\0"))
                .as_ptr(),
        );
    }
    (*self_0).conjugateLossFn = f;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_set_dca_update_function(
    mut self_0: *mut VlSvm,
    mut f: VlSvmDcaUpdateFunction,
) {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            1541 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 69],
                &[libc::c_char; 69],
            >(b"void vl_svm_set_dca_update_function(VlSvm *, VlSvmDcaUpdateFunction)\0"))
                .as_ptr(),
        );
    }
    (*self_0).dcaUpdateFn = f;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_set_loss(
    mut self_0: *mut VlSvm,
    mut loss: VlSvmLossType,
) {
    match loss as libc::c_uint {
        0 => {
            vl_svm_set_loss_function(
                self_0,
                Some(
                    vl_svm_hinge_loss
                        as unsafe extern "C" fn(
                            libc::c_double,
                            libc::c_double,
                        ) -> libc::c_double,
                ),
            );
            vl_svm_set_loss_derivative_function(
                self_0,
                Some(
                    vl_svm_hinge_loss_derivative
                        as unsafe extern "C" fn(
                            libc::c_double,
                            libc::c_double,
                        ) -> libc::c_double,
                ),
            );
            vl_svm_set_conjugate_loss_function(
                self_0,
                Some(
                    vl_svm_hinge_conjugate_loss
                        as unsafe extern "C" fn(
                            libc::c_double,
                            libc::c_double,
                        ) -> libc::c_double,
                ),
            );
            vl_svm_set_dca_update_function(
                self_0,
                Some(
                    vl_svm_hinge_dca_update
                        as unsafe extern "C" fn(
                            libc::c_double,
                            libc::c_double,
                            libc::c_double,
                            libc::c_double,
                        ) -> libc::c_double,
                ),
            );
        }
        1 => {
            vl_svm_set_loss_function(
                self_0,
                Some(
                    vl_svm_hinge2_loss
                        as unsafe extern "C" fn(
                            libc::c_double,
                            libc::c_double,
                        ) -> libc::c_double,
                ),
            );
            vl_svm_set_loss_derivative_function(
                self_0,
                Some(
                    vl_svm_hinge2_loss_derivative
                        as unsafe extern "C" fn(
                            libc::c_double,
                            libc::c_double,
                        ) -> libc::c_double,
                ),
            );
            vl_svm_set_conjugate_loss_function(
                self_0,
                Some(
                    vl_svm_hinge2_conjugate_loss
                        as unsafe extern "C" fn(
                            libc::c_double,
                            libc::c_double,
                        ) -> libc::c_double,
                ),
            );
            vl_svm_set_dca_update_function(
                self_0,
                Some(
                    vl_svm_hinge2_dca_update
                        as unsafe extern "C" fn(
                            libc::c_double,
                            libc::c_double,
                            libc::c_double,
                            libc::c_double,
                        ) -> libc::c_double,
                ),
            );
        }
        2 => {
            vl_svm_set_loss_function(
                self_0,
                Some(
                    vl_svm_l1_loss
                        as unsafe extern "C" fn(
                            libc::c_double,
                            libc::c_double,
                        ) -> libc::c_double,
                ),
            );
            vl_svm_set_loss_derivative_function(
                self_0,
                Some(
                    vl_svm_l1_loss_derivative
                        as unsafe extern "C" fn(
                            libc::c_double,
                            libc::c_double,
                        ) -> libc::c_double,
                ),
            );
            vl_svm_set_conjugate_loss_function(
                self_0,
                Some(
                    vl_svm_l1_conjugate_loss
                        as unsafe extern "C" fn(
                            libc::c_double,
                            libc::c_double,
                        ) -> libc::c_double,
                ),
            );
            vl_svm_set_dca_update_function(
                self_0,
                Some(
                    vl_svm_l1_dca_update
                        as unsafe extern "C" fn(
                            libc::c_double,
                            libc::c_double,
                            libc::c_double,
                            libc::c_double,
                        ) -> libc::c_double,
                ),
            );
        }
        3 => {
            vl_svm_set_loss_function(
                self_0,
                Some(
                    vl_svm_l2_loss
                        as unsafe extern "C" fn(
                            libc::c_double,
                            libc::c_double,
                        ) -> libc::c_double,
                ),
            );
            vl_svm_set_loss_derivative_function(
                self_0,
                Some(
                    vl_svm_l2_loss_derivative
                        as unsafe extern "C" fn(
                            libc::c_double,
                            libc::c_double,
                        ) -> libc::c_double,
                ),
            );
            vl_svm_set_conjugate_loss_function(
                self_0,
                Some(
                    vl_svm_l2_conjugate_loss
                        as unsafe extern "C" fn(
                            libc::c_double,
                            libc::c_double,
                        ) -> libc::c_double,
                ),
            );
            vl_svm_set_dca_update_function(
                self_0,
                Some(
                    vl_svm_l2_dca_update
                        as unsafe extern "C" fn(
                            libc::c_double,
                            libc::c_double,
                            libc::c_double,
                            libc::c_double,
                        ) -> libc::c_double,
                ),
            );
        }
        4 => {
            vl_svm_set_loss_function(
                self_0,
                Some(
                    vl_svm_logistic_loss
                        as unsafe extern "C" fn(
                            libc::c_double,
                            libc::c_double,
                        ) -> libc::c_double,
                ),
            );
            vl_svm_set_loss_derivative_function(
                self_0,
                Some(
                    vl_svm_logistic_loss_derivative
                        as unsafe extern "C" fn(
                            libc::c_double,
                            libc::c_double,
                        ) -> libc::c_double,
                ),
            );
            vl_svm_set_conjugate_loss_function(
                self_0,
                Some(
                    vl_svm_logistic_conjugate_loss
                        as unsafe extern "C" fn(
                            libc::c_double,
                            libc::c_double,
                        ) -> libc::c_double,
                ),
            );
            vl_svm_set_dca_update_function(
                self_0,
                Some(
                    vl_svm_logistic_dca_update
                        as unsafe extern "C" fn(
                            libc::c_double,
                            libc::c_double,
                            libc::c_double,
                            libc::c_double,
                        ) -> libc::c_double,
                ),
            );
        }
        _ => {
            __assert_fail(
                b"0\0" as *const u8 as *const libc::c_char,
                b"vl/svm.c\0" as *const u8 as *const libc::c_char,
                1569 as libc::c_int as libc::c_uint,
                (*::core::mem::transmute::<
                    &[u8; 45],
                    &[libc::c_char; 45],
                >(b"void vl_svm_set_loss(VlSvm *, VlSvmLossType)\0"))
                    .as_ptr(),
            );
        }
    };
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_hinge_loss(
    mut inner: libc::c_double,
    mut label: libc::c_double,
) -> libc::c_double {
    return if 1 as libc::c_int as libc::c_double - label * inner > 0.0f64 {
        1 as libc::c_int as libc::c_double - label * inner
    } else {
        0.0f64
    };
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_hinge_loss_derivative(
    mut inner: libc::c_double,
    mut label: libc::c_double,
) -> libc::c_double {
    if label * inner < 1.0f64 { return -label } else { return 0.0f64 };
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_hinge_conjugate_loss(
    mut u: libc::c_double,
    mut label: libc::c_double,
) -> libc::c_double {
    let mut z: libc::c_double = label * u;
    if -(1 as libc::c_int) as libc::c_double <= z
        && z <= 0 as libc::c_int as libc::c_double
    {
        return label * u
    } else {
        return vl_infinity_d.value
    };
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_hinge_dca_update(
    mut alpha: libc::c_double,
    mut inner: libc::c_double,
    mut norm2: libc::c_double,
    mut label: libc::c_double,
) -> libc::c_double {
    let mut palpha: libc::c_double = (label - inner) / norm2 + alpha;
    return label
        * (if 0 as libc::c_int as libc::c_double
            > (if (1 as libc::c_int as libc::c_double) < label * palpha {
                1 as libc::c_int as libc::c_double
            } else {
                label * palpha
            })
        {
            0 as libc::c_int as libc::c_double
        } else {
            (if (1 as libc::c_int as libc::c_double) < label * palpha {
                1 as libc::c_int as libc::c_double
            } else {
                label * palpha
            })
        }) - alpha;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_hinge2_loss(
    mut inner: libc::c_double,
    mut label: libc::c_double,
) -> libc::c_double {
    let mut z: libc::c_double = if 1 as libc::c_int as libc::c_double - label * inner
        > 0.0f64
    {
        1 as libc::c_int as libc::c_double - label * inner
    } else {
        0.0f64
    };
    return z * z;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_hinge2_loss_derivative(
    mut inner: libc::c_double,
    mut label: libc::c_double,
) -> libc::c_double {
    if label * inner < 1.0f64 {
        return 2 as libc::c_int as libc::c_double * (inner - label)
    } else {
        return 0 as libc::c_int as libc::c_double
    };
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_hinge2_conjugate_loss(
    mut u: libc::c_double,
    mut label: libc::c_double,
) -> libc::c_double {
    if label * u <= 0 as libc::c_int as libc::c_double {
        return (label + u / 4 as libc::c_int as libc::c_double) * u
    } else {
        return vl_infinity_d.value
    };
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_hinge2_dca_update(
    mut alpha: libc::c_double,
    mut inner: libc::c_double,
    mut norm2: libc::c_double,
    mut label: libc::c_double,
) -> libc::c_double {
    let mut palpha: libc::c_double = (label - inner - 0.5f64 * alpha) / (norm2 + 0.5f64)
        + alpha;
    return label
        * (if 0 as libc::c_int as libc::c_double > label * palpha {
            0 as libc::c_int as libc::c_double
        } else {
            label * palpha
        }) - alpha;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_l1_loss(
    mut inner: libc::c_double,
    mut label: libc::c_double,
) -> libc::c_double {
    return vl_abs_d(label - inner);
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_l1_loss_derivative(
    mut inner: libc::c_double,
    mut label: libc::c_double,
) -> libc::c_double {
    if label > inner { return -1.0f64 } else { return 1.0f64 };
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_l1_conjugate_loss(
    mut u: libc::c_double,
    mut label: libc::c_double,
) -> libc::c_double {
    if vl_abs_d(u) <= 1 as libc::c_int as libc::c_double {
        return label * u
    } else {
        return vl_infinity_d.value
    };
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_l1_dca_update(
    mut alpha: libc::c_double,
    mut inner: libc::c_double,
    mut norm2: libc::c_double,
    mut label: libc::c_double,
) -> libc::c_double {
    if vl_abs_d(alpha) <= 1 as libc::c_int as libc::c_double {
        let mut palpha: libc::c_double = (label - inner) / norm2 + alpha;
        return (if -1.0f64 > (if 1.0f64 < palpha { 1.0f64 } else { palpha }) {
            -1.0f64
        } else {
            (if 1.0f64 < palpha { 1.0f64 } else { palpha })
        }) - alpha;
    } else {
        return vl_infinity_d.value
    };
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_l2_loss(
    mut inner: libc::c_double,
    mut label: libc::c_double,
) -> libc::c_double {
    let mut z: libc::c_double = label - inner;
    return z * z;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_l2_loss_derivative(
    mut inner: libc::c_double,
    mut label: libc::c_double,
) -> libc::c_double {
    return -(2 as libc::c_int) as libc::c_double * (label - inner);
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_l2_conjugate_loss(
    mut u: libc::c_double,
    mut label: libc::c_double,
) -> libc::c_double {
    return (label + u / 4 as libc::c_int as libc::c_double) * u;
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_l2_dca_update(
    mut alpha: libc::c_double,
    mut inner: libc::c_double,
    mut norm2: libc::c_double,
    mut label: libc::c_double,
) -> libc::c_double {
    return (label - inner - 0.5f64 * alpha) / (norm2 + 0.5f64);
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_logistic_loss(
    mut inner: libc::c_double,
    mut label: libc::c_double,
) -> libc::c_double {
    let mut z: libc::c_double = label * inner;
    if z >= 0 as libc::c_int as libc::c_double {
        return log(1.0f64 + exp(-z))
    } else {
        return -z + log(exp(z) + 1.0f64)
    };
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_logistic_loss_derivative(
    mut inner: libc::c_double,
    mut label: libc::c_double,
) -> libc::c_double {
    let mut z: libc::c_double = label * inner;
    let mut t: libc::c_double = 1 as libc::c_int as libc::c_double
        / (1 as libc::c_int as libc::c_double + exp(-z));
    return label * (t - 1 as libc::c_int as libc::c_double);
}
#[inline]
unsafe extern "C" fn xlogx(mut x: libc::c_double) -> libc::c_double {
    if x <= 1e-10f64 {
        return 0 as libc::c_int as libc::c_double;
    }
    return x * log(x);
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_logistic_conjugate_loss(
    mut u: libc::c_double,
    mut label: libc::c_double,
) -> libc::c_double {
    let mut z: libc::c_double = label * u;
    if -(1 as libc::c_int) as libc::c_double <= z
        && z <= 0 as libc::c_int as libc::c_double
    {
        return xlogx(-z) + xlogx(1 as libc::c_int as libc::c_double + z)
    } else {
        return vl_infinity_d.value
    };
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_logistic_dca_update(
    mut alpha: libc::c_double,
    mut inner: libc::c_double,
    mut norm2: libc::c_double,
    mut label: libc::c_double,
) -> libc::c_double {
    let mut df: libc::c_double = 0.;
    let mut ddf: libc::c_double = 0.;
    let mut der: libc::c_double = 0.;
    let mut dder: libc::c_double = 0.;
    let mut t: vl_index = 0;
    let mut beta1: libc::c_double = 0 as libc::c_int as libc::c_double;
    let mut beta2: libc::c_double = 1 as libc::c_int as libc::c_double;
    let mut beta: libc::c_double = 0.5f64;
    t = 0 as libc::c_int as vl_index;
    while t < 5 as libc::c_int as libc::c_longlong {
        df = log(beta) - log(1 as libc::c_int as libc::c_double - beta);
        der = norm2 * beta + label * (inner - norm2 * alpha) + df;
        if der >= 0 as libc::c_int as libc::c_double {
            beta2 = beta;
        } else {
            beta1 = beta;
        }
        beta = 0.5f64 * (beta1 + beta2);
        t += 1;
    }
    t = 0 as libc::c_int as vl_index;
    while (t < 2 as libc::c_int as libc::c_longlong) as libc::c_int
        & (beta > 2.220446049250313e-16f64) as libc::c_int
        & (beta < 1 as libc::c_int as libc::c_double - 2.220446049250313e-16f64)
            as libc::c_int != 0
    {
        df = log(beta) - log(1 as libc::c_int as libc::c_double - beta);
        ddf = 1 as libc::c_int as libc::c_double
            / (beta * (1 as libc::c_int as libc::c_double - beta));
        der = norm2 * beta + label * (inner - norm2 * alpha) + df;
        dder = norm2 + ddf;
        beta -= der / dder;
        beta = if 0 as libc::c_int as libc::c_double
            > (if (1 as libc::c_int as libc::c_double) < beta {
                1 as libc::c_int as libc::c_double
            } else {
                beta
            })
        {
            0 as libc::c_int as libc::c_double
        } else if (1 as libc::c_int as libc::c_double) < beta {
            1 as libc::c_int as libc::c_double
        } else {
            beta
        };
        t += 1;
    }
    return label * beta - alpha;
}
#[no_mangle]
pub unsafe extern "C" fn _vl_svm_update_statistics(mut self_0: *mut VlSvm) {
    let mut i: vl_size = 0;
    let mut k: vl_size = 0;
    let mut inner: libc::c_double = 0.;
    let mut p: libc::c_double = 0.;
    memset(
        &mut (*self_0).statistics as *mut VlSvmStatistics as *mut libc::c_void,
        0 as libc::c_int,
        ::core::mem::size_of::<VlSvmStatistics>() as libc::c_ulong,
    );
    (*self_0).statistics.regularizer = (*self_0).bias * (*self_0).bias;
    i = 0 as libc::c_int as vl_size;
    while i < (*self_0).dimension {
        (*self_0).statistics.regularizer
            += *((*self_0).model).offset(i as isize)
                * *((*self_0).model).offset(i as isize);
        i = i.wrapping_add(1);
    }
    (*self_0).statistics.regularizer *= (*self_0).lambda * 0.5f64;
    k = 0 as libc::c_int as vl_size;
    while k < (*self_0).numData {
        p = if !((*self_0).weights).is_null() {
            *((*self_0).weights).offset(k as isize)
        } else {
            1.0f64
        };
        if !(p <= 0 as libc::c_int as libc::c_double) {
            inner = ((*self_0).innerProductFn)
                .expect("non-null function pointer")((*self_0).data, k, (*self_0).model);
            inner += (*self_0).bias * (*self_0).biasMultiplier;
            *((*self_0).scores).offset(k as isize) = inner;
            (*self_0).statistics.loss
                += p
                    * ((*self_0).lossFn)
                        .expect(
                            "non-null function pointer",
                        )(inner, *((*self_0).labels).offset(k as isize));
            if (*self_0).solver as libc::c_uint
                == VlSvmSolverSdca as libc::c_int as libc::c_uint
            {
                (*self_0).statistics.dualLoss
                    -= p
                        * ((*self_0).conjugateLossFn)
                            .expect(
                                "non-null function pointer",
                            )(
                            -*((*self_0).alpha).offset(k as isize) / p,
                            *((*self_0).labels).offset(k as isize),
                        );
            }
        }
        k = k.wrapping_add(1);
    }
    (*self_0).statistics.loss /= (*self_0).numData as libc::c_double;
    (*self_0)
        .statistics
        .objective = (*self_0).statistics.regularizer + (*self_0).statistics.loss;
    if (*self_0).solver as libc::c_uint == VlSvmSolverSdca as libc::c_int as libc::c_uint
    {
        (*self_0).statistics.dualLoss /= (*self_0).numData as libc::c_double;
        (*self_0)
            .statistics
            .dualObjective = -(*self_0).statistics.regularizer
            + (*self_0).statistics.dualLoss;
        (*self_0)
            .statistics
            .dualityGap = (*self_0).statistics.objective
            - (*self_0).statistics.dualObjective;
    }
}
#[no_mangle]
pub unsafe extern "C" fn _vl_svm_evaluate(mut self_0: *mut VlSvm) {
    let mut startTime: libc::c_double = vl_get_cpu_time();
    _vl_svm_update_statistics(self_0);
    (*self_0).statistics.elapsedTime = vl_get_cpu_time() - startTime;
    (*self_0).statistics.iteration = 0 as libc::c_int as vl_size;
    (*self_0).statistics.epoch = 0 as libc::c_int as vl_size;
    (*self_0).statistics.status = VlSvmStatusConverged;
    if ((*self_0).diagnosticFn).is_some() {
        ((*self_0).diagnosticFn)
            .expect("non-null function pointer")(self_0, (*self_0).diagnosticFnData);
    }
}
#[no_mangle]
pub unsafe extern "C" fn _vl_svm_sdca_train(mut self_0: *mut VlSvm) {
    let mut norm2: *mut libc::c_double = 0 as *mut libc::c_double;
    let mut permutation: *mut vl_index = 0 as *mut vl_index;
    let mut i: vl_uindex = 0;
    let mut t: vl_uindex = 0;
    let mut inner: libc::c_double = 0.;
    let mut delta: libc::c_double = 0.;
    let mut multiplier: libc::c_double = 0.;
    let mut p: libc::c_double = 0.;
    let mut startTime: libc::c_double = vl_get_cpu_time();
    let mut rand: *mut VlRand = vl_get_rand();
    norm2 = vl_calloc(
        (*self_0).numData as size_t,
        ::core::mem::size_of::<libc::c_double>() as libc::c_ulong,
    ) as *mut libc::c_double;
    permutation = vl_calloc(
        (*self_0).numData as size_t,
        ::core::mem::size_of::<vl_index>() as libc::c_ulong,
    ) as *mut vl_index;
    let mut buffer: *mut libc::c_double = vl_calloc(
        (*self_0).dimension as size_t,
        ::core::mem::size_of::<libc::c_double>() as libc::c_ulong,
    ) as *mut libc::c_double;
    i = 0 as libc::c_int as vl_uindex;
    while i < (*self_0).numData as libc::c_uint as libc::c_ulonglong {
        let mut n2: libc::c_double = 0.;
        *permutation.offset(i as isize) = i as vl_index;
        memset(
            buffer as *mut libc::c_void,
            0 as libc::c_int,
            ((*self_0).dimension)
                .wrapping_mul(
                    ::core::mem::size_of::<libc::c_double>() as libc::c_ulong
                        as libc::c_ulonglong,
                ) as libc::c_ulong,
        );
        ((*self_0).accumulateFn)
            .expect(
                "non-null function pointer",
            )((*self_0).data, i, buffer, 1 as libc::c_int as libc::c_double);
        n2 = ((*self_0).innerProductFn)
            .expect("non-null function pointer")((*self_0).data, i, buffer);
        n2 += (*self_0).biasMultiplier * (*self_0).biasMultiplier;
        *norm2
            .offset(
                i as isize,
            ) = n2 / ((*self_0).lambda * (*self_0).numData as libc::c_double);
        i = i.wrapping_add(1);
    }
    vl_free(buffer as *mut libc::c_void);
    t = 0 as libc::c_int as vl_uindex;
    loop {
        if t.wrapping_rem((*self_0).numData) == 0 as libc::c_int as libc::c_ulonglong {
            vl_rand_permute_indexes(rand, permutation, (*self_0).numData);
        }
        i = *permutation.offset(t.wrapping_rem((*self_0).numData) as isize) as vl_uindex;
        p = if !((*self_0).weights).is_null() {
            *((*self_0).weights).offset(i as isize)
        } else {
            1.0f64
        };
        if p > 0 as libc::c_int as libc::c_double {
            inner = ((*self_0).innerProductFn)
                .expect("non-null function pointer")((*self_0).data, i, (*self_0).model);
            inner += (*self_0).bias * (*self_0).biasMultiplier;
            delta = p
                * ((*self_0).dcaUpdateFn)
                    .expect(
                        "non-null function pointer",
                    )(
                    *((*self_0).alpha).offset(i as isize) / p,
                    inner,
                    p * *norm2.offset(i as isize),
                    *((*self_0).labels).offset(i as isize),
                );
        } else {
            delta = 0 as libc::c_int as libc::c_double;
        }
        if delta != 0 as libc::c_int as libc::c_double {
            *((*self_0).alpha).offset(i as isize) += delta;
            multiplier = delta
                / ((*self_0).numData as libc::c_double * (*self_0).lambda);
            ((*self_0).accumulateFn)
                .expect(
                    "non-null function pointer",
                )((*self_0).data, i, (*self_0).model, multiplier);
            (*self_0).bias += (*self_0).biasMultiplier * multiplier;
        }
        if t
            .wrapping_add(1 as libc::c_int as libc::c_ulonglong)
            .wrapping_rem((*self_0).diagnosticFrequency)
            == 0 as libc::c_int as libc::c_ulonglong
            || t.wrapping_add(1 as libc::c_int as libc::c_ulonglong)
                == (*self_0).maxNumIterations
        {
            _vl_svm_update_statistics(self_0);
            (*self_0).statistics.elapsedTime = vl_get_cpu_time() - startTime;
            (*self_0).statistics.iteration = t;
            (*self_0).statistics.epoch = t.wrapping_div((*self_0).numData);
            (*self_0).statistics.status = VlSvmStatusTraining;
            if (*self_0).statistics.dualityGap < (*self_0).epsilon {
                (*self_0).statistics.status = VlSvmStatusConverged;
            } else if t.wrapping_add(1 as libc::c_int as libc::c_ulonglong)
                == (*self_0).maxNumIterations
            {
                (*self_0).statistics.status = VlSvmStatusMaxNumIterationsReached;
            }
            if ((*self_0).diagnosticFn).is_some() {
                ((*self_0).diagnosticFn)
                    .expect(
                        "non-null function pointer",
                    )(self_0, (*self_0).diagnosticFnData);
            }
            if (*self_0).statistics.status as libc::c_uint
                != VlSvmStatusTraining as libc::c_int as libc::c_uint
            {
                break;
            }
        }
        t = t.wrapping_add(1);
    }
    vl_free(norm2 as *mut libc::c_void);
    vl_free(permutation as *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn _vl_svm_sgd_train(mut self_0: *mut VlSvm) {
    let mut permutation: *mut vl_index = 0 as *mut vl_index;
    let mut scores: *mut libc::c_double = 0 as *mut libc::c_double;
    let mut previousScores: *mut libc::c_double = 0 as *mut libc::c_double;
    let mut i: vl_uindex = 0;
    let mut t: vl_uindex = 0;
    let mut k: vl_uindex = 0;
    let mut inner: libc::c_double = 0.;
    let mut gradient: libc::c_double = 0.;
    let mut rate: libc::c_double = 0.;
    let mut biasRate: libc::c_double = 0.;
    let mut p: libc::c_double = 0.;
    let mut factor: libc::c_double = 1.0f64;
    let mut biasFactor: libc::c_double = 1.0f64;
    let mut t0: vl_index = (if 2 as libc::c_int as libc::c_long
        > f64::ceil(1.0f64 / (*self_0).lambda) as libc::c_long
    {
        2 as libc::c_int as libc::c_long
    } else {
        f64::ceil(1.0f64 / (*self_0).lambda) as libc::c_long
    }) as vl_index;
    let mut startTime: libc::c_double = vl_get_cpu_time();
    let mut rand: *mut VlRand = vl_get_rand();
    permutation = vl_calloc(
        (*self_0).numData as size_t,
        ::core::mem::size_of::<vl_index>() as libc::c_ulong,
    ) as *mut vl_index;
    scores = vl_calloc(
        ((*self_0).numData).wrapping_mul(2 as libc::c_int as libc::c_ulonglong)
            as size_t,
        ::core::mem::size_of::<libc::c_double>() as libc::c_ulong,
    ) as *mut libc::c_double;
    previousScores = scores.offset((*self_0).numData as isize);
    i = 0 as libc::c_int as vl_uindex;
    while i < (*self_0).numData as libc::c_uint as libc::c_ulonglong {
        *permutation.offset(i as isize) = i as vl_index;
        *previousScores.offset(i as isize) = -vl_infinity_d.value;
        i = i.wrapping_add(1);
    }
    t = 0 as libc::c_int as vl_uindex;
    loop {
        if t.wrapping_rem((*self_0).numData) == 0 as libc::c_int as libc::c_ulonglong {
            vl_rand_permute_indexes(rand, permutation, (*self_0).numData);
        }
        i = *permutation.offset(t.wrapping_rem((*self_0).numData) as isize) as vl_uindex;
        p = if !((*self_0).weights).is_null() {
            *((*self_0).weights).offset(i as isize)
        } else {
            1.0f64
        };
        p = if 0.0f64 > p { 0.0f64 } else { p };
        inner = factor
            * ((*self_0).innerProductFn)
                .expect("non-null function pointer")((*self_0).data, i, (*self_0).model);
        inner += biasFactor * ((*self_0).biasMultiplier * (*self_0).bias);
        gradient = p
            * ((*self_0).lossDerivativeFn)
                .expect(
                    "non-null function pointer",
                )(inner, *((*self_0).labels).offset(i as isize));
        *previousScores.offset(i as isize) = *scores.offset(i as isize);
        *scores.offset(i as isize) = inner;
        rate = 1.0f64
            / ((*self_0).lambda
                * t.wrapping_add(t0 as libc::c_ulonglong) as libc::c_double);
        biasRate = rate * (*self_0).biasLearningRate;
        factor *= 1.0f64 - (*self_0).lambda * rate;
        biasFactor *= 1.0f64 - (*self_0).lambda * biasRate;
        if gradient != 0 as libc::c_int as libc::c_double {
            ((*self_0).accumulateFn)
                .expect(
                    "non-null function pointer",
                )((*self_0).data, i, (*self_0).model, -gradient * rate / factor);
            (*self_0).bias
                += (*self_0).biasMultiplier * (-gradient * biasRate / biasFactor);
        }
        if t
            .wrapping_add(1 as libc::c_int as libc::c_ulonglong)
            .wrapping_rem((*self_0).diagnosticFrequency)
            == 0 as libc::c_int as libc::c_ulonglong
            || t.wrapping_add(1 as libc::c_int as libc::c_ulonglong)
                == (*self_0).maxNumIterations
        {
            k = 0 as libc::c_int as vl_uindex;
            while k < (*self_0).dimension {
                *((*self_0).model).offset(k as isize) *= factor;
                k = k.wrapping_add(1);
            }
            (*self_0).bias *= biasFactor;
            factor = 1.0f64;
            biasFactor = 1.0f64;
            _vl_svm_update_statistics(self_0);
            k = 0 as libc::c_int as vl_uindex;
            while k < (*self_0).numData {
                let mut delta: libc::c_double = *scores.offset(k as isize)
                    - *previousScores.offset(k as isize);
                (*self_0).statistics.scoresVariation += delta * delta;
                k = k.wrapping_add(1);
            }
            (*self_0)
                .statistics
                .scoresVariation = sqrt((*self_0).statistics.scoresVariation)
                / (*self_0).numData as libc::c_double;
            (*self_0).statistics.elapsedTime = vl_get_cpu_time() - startTime;
            (*self_0).statistics.iteration = t;
            (*self_0).statistics.epoch = t.wrapping_div((*self_0).numData);
            (*self_0).statistics.status = VlSvmStatusTraining;
            if (*self_0).statistics.scoresVariation < (*self_0).epsilon {
                (*self_0).statistics.status = VlSvmStatusConverged;
            } else if t.wrapping_add(1 as libc::c_int as libc::c_ulonglong)
                == (*self_0).maxNumIterations
            {
                (*self_0).statistics.status = VlSvmStatusMaxNumIterationsReached;
            }
            if ((*self_0).diagnosticFn).is_some() {
                ((*self_0).diagnosticFn)
                    .expect(
                        "non-null function pointer",
                    )(self_0, (*self_0).diagnosticFnData);
            }
            if (*self_0).statistics.status as libc::c_uint
                != VlSvmStatusTraining as libc::c_int as libc::c_uint
            {
                break;
            }
        }
        t = t.wrapping_add(1);
    }
    vl_free(scores as *mut libc::c_void);
    vl_free(permutation as *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn vl_svm_train(mut self_0: *mut VlSvm) {
    if !self_0.is_null() {} else {
        __assert_fail(
            b"self\0" as *const u8 as *const libc::c_char,
            b"vl/svm.c\0" as *const u8 as *const libc::c_char,
            2158 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 27],
                &[libc::c_char; 27],
            >(b"void vl_svm_train(VlSvm *)\0"))
                .as_ptr(),
        );
    }
    match (*self_0).solver as libc::c_uint {
        2 => {
            _vl_svm_sdca_train(self_0);
        }
        1 => {
            _vl_svm_sgd_train(self_0);
        }
        0 => {
            _vl_svm_evaluate(self_0);
        }
        _ => {
            __assert_fail(
                b"0\0" as *const u8 as *const libc::c_char,
                b"vl/svm.c\0" as *const u8 as *const libc::c_char,
                2170 as libc::c_int as libc::c_uint,
                (*::core::mem::transmute::<
                    &[u8; 27],
                    &[libc::c_char; 27],
                >(b"void vl_svm_train(VlSvm *)\0"))
                    .as_ptr(),
            );
        }
    };
}
