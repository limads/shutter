use ::libc;
extern "C" {
    fn exp(_: libc::c_double) -> libc::c_double;
    fn log(_: libc::c_double) -> libc::c_double;
    fn abort() -> !;
    fn __assert_fail(
        __assertion: *const libc::c_char,
        __file: *const libc::c_char,
        __line: libc::c_uint,
        __function: *const libc::c_char,
    ) -> !;
    fn vl_get_cpu_time() -> libc::c_double;
    fn vl_get_printf_func() -> printf_func_t;
    fn vl_free(ptr: *mut libc::c_void);
    fn vl_calloc(n: size_t, size: size_t) -> *mut libc::c_void;
    fn vl_malloc(n: size_t) -> *mut libc::c_void;
    fn vl_cpu_has_sse2() -> vl_bool;
    fn vl_cpu_has_avx() -> vl_bool;
    fn vl_get_simd_enabled() -> vl_bool;
    fn vl_get_vector_3_comparison_function_f(
        type_0: VlVectorComparisonType,
    ) -> VlFloatVector3ComparisonFunction;
    fn vl_get_vector_3_comparison_function_d(
        type_0: VlVectorComparisonType,
    ) -> VlDoubleVector3ComparisonFunction;
    fn vl_kmeans_new(
        dataType: vl_type,
        distance: VlVectorComparisonType,
    ) -> *mut VlKMeans;
    fn vl_kmeans_delete(self_0: *mut VlKMeans);
    fn vl_kmeans_cluster(
        self_0: *mut VlKMeans,
        data: *const libc::c_void,
        dimension: vl_size,
        numData: vl_size,
        numCenters: vl_size,
    ) -> libc::c_double;
    fn vl_kmeans_quantize(
        self_0: *mut VlKMeans,
        assignments: *mut vl_uint32,
        distances: *mut libc::c_void,
        data: *const libc::c_void,
        numData: vl_size,
    );
    fn vl_kmeans_init_centers_plus_plus(
        self_0: *mut VlKMeans,
        data: *const libc::c_void,
        dimensions: vl_size,
        numData: vl_size,
        numCenters: vl_size,
    );
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
    fn _vl_weighted_sigma_sse2_f(
        dimension: vl_size,
        S: *mut libc::c_float,
        X: *const libc::c_float,
        Y: *const libc::c_float,
        W: libc::c_float,
    );
    fn _vl_weighted_sigma_sse2_d(
        dimension: vl_size,
        S: *mut libc::c_double,
        X: *const libc::c_double,
        Y: *const libc::c_double,
        W: libc::c_double,
    );
    fn _vl_weighted_mean_sse2_d(
        dimension: vl_size,
        MU: *mut libc::c_double,
        X: *const libc::c_double,
        W: libc::c_double,
    );
    fn _vl_weighted_mean_sse2_f(
        dimension: vl_size,
        MU: *mut libc::c_float,
        X: *const libc::c_float,
        W: libc::c_float,
    );
    fn _vl_weighted_sigma_avx_d(
        dimension: vl_size,
        S: *mut libc::c_double,
        X: *const libc::c_double,
        Y: *const libc::c_double,
        W: libc::c_double,
    );
    fn _vl_weighted_sigma_avx_f(
        dimension: vl_size,
        S: *mut libc::c_float,
        X: *const libc::c_float,
        Y: *const libc::c_float,
        W: libc::c_float,
    );
    fn _vl_weighted_mean_avx_d(
        dimension: vl_size,
        MU: *mut libc::c_double,
        X: *const libc::c_double,
        W: libc::c_double,
    );
    fn _vl_weighted_mean_avx_f(
        dimension: vl_size,
        MU: *mut libc::c_float,
        X: *const libc::c_float,
        W: libc::c_float,
    );
}
pub type vl_int64 = libc::c_longlong;
pub type vl_int32 = libc::c_int;
pub type vl_int16 = libc::c_short;
pub type vl_int8 = libc::c_char;
pub type vl_uint64 = libc::c_ulonglong;
pub type vl_uint32 = libc::c_uint;
pub type vl_bool = libc::c_int;
pub type vl_size = vl_uint64;
pub type vl_index = vl_int64;
pub type vl_uindex = vl_uint64;
pub type size_t = libc::c_ulong;
pub type vl_type = vl_uint32;
pub type printf_func_t = Option::<
    unsafe extern "C" fn(*const libc::c_char, ...) -> libc::c_int,
>;
#[derive(Copy, Clone)]
#[repr(C)]
pub union C2RustUnnamed {
    pub raw: vl_uint64,
    pub value: libc::c_double,
}
pub type VlFloatVectorComparisonFunction = Option::<
    unsafe extern "C" fn(
        vl_size,
        *const libc::c_float,
        *const libc::c_float,
    ) -> libc::c_float,
>;
pub type VlDoubleVectorComparisonFunction = Option::<
    unsafe extern "C" fn(
        vl_size,
        *const libc::c_double,
        *const libc::c_double,
    ) -> libc::c_double,
>;
pub type VlFloatVector3ComparisonFunction = Option::<
    unsafe extern "C" fn(
        vl_size,
        *const libc::c_float,
        *const libc::c_float,
        *const libc::c_float,
    ) -> libc::c_float,
>;
pub type VlDoubleVector3ComparisonFunction = Option::<
    unsafe extern "C" fn(
        vl_size,
        *const libc::c_double,
        *const libc::c_double,
        *const libc::c_double,
    ) -> libc::c_double,
>;
pub type _VlVectorComparisonType = libc::c_uint;
pub const VlKernelJS: _VlVectorComparisonType = 10;
pub const VlKernelHellinger: _VlVectorComparisonType = 9;
pub const VlKernelChi2: _VlVectorComparisonType = 8;
pub const VlKernelL2: _VlVectorComparisonType = 7;
pub const VlKernelL1: _VlVectorComparisonType = 6;
pub const VlDistanceMahalanobis: _VlVectorComparisonType = 5;
pub const VlDistanceJS: _VlVectorComparisonType = 4;
pub const VlDistanceHellinger: _VlVectorComparisonType = 3;
pub const VlDistanceChi2: _VlVectorComparisonType = 2;
pub const VlDistanceL2: _VlVectorComparisonType = 1;
pub const VlDistanceL1: _VlVectorComparisonType = 0;
pub type VlVectorComparisonType = _VlVectorComparisonType;
pub type _VlKMeansAlgorithm = libc::c_uint;
pub const VlKMeansANN: _VlKMeansAlgorithm = 2;
pub const VlKMeansElkan: _VlKMeansAlgorithm = 1;
pub const VlKMeansLloyd: _VlKMeansAlgorithm = 0;
pub type VlKMeansAlgorithm = _VlKMeansAlgorithm;
pub type _VlKMeansInitialization = libc::c_uint;
pub const VlKMeansPlusPlus: _VlKMeansInitialization = 1;
pub const VlKMeansRandomSelection: _VlKMeansInitialization = 0;
pub type VlKMeansInitialization = _VlKMeansInitialization;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlKMeans {
    pub dataType: vl_type,
    pub dimension: vl_size,
    pub numCenters: vl_size,
    pub numTrees: vl_size,
    pub maxNumComparisons: vl_size,
    pub initialization: VlKMeansInitialization,
    pub algorithm: VlKMeansAlgorithm,
    pub distance: VlVectorComparisonType,
    pub maxNumIterations: vl_size,
    pub minEnergyVariation: libc::c_double,
    pub numRepetitions: vl_size,
    pub verbosity: libc::c_int,
    pub centers: *mut libc::c_void,
    pub centerDistances: *mut libc::c_void,
    pub energy: libc::c_double,
    pub floatVectorComparisonFn: VlFloatVectorComparisonFunction,
    pub doubleVectorComparisonFn: VlDoubleVectorComparisonFunction,
}
pub type VlKMeans = _VlKMeans;
pub type _VlGMMInitialization = libc::c_uint;
pub const VlGMMCustom: _VlGMMInitialization = 2;
pub const VlGMMRand: _VlGMMInitialization = 1;
pub const VlGMMKMeans: _VlGMMInitialization = 0;
pub type VlGMMInitialization = _VlGMMInitialization;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlGMM {
    pub dataType: vl_type,
    pub dimension: vl_size,
    pub numClusters: vl_size,
    pub numData: vl_size,
    pub maxNumIterations: vl_size,
    pub numRepetitions: vl_size,
    pub verbosity: libc::c_int,
    pub means: *mut libc::c_void,
    pub covariances: *mut libc::c_void,
    pub priors: *mut libc::c_void,
    pub posteriors: *mut libc::c_void,
    pub sigmaLowBound: *mut libc::c_double,
    pub initialization: VlGMMInitialization,
    pub kmeansInit: *mut VlKMeans,
    pub LL: libc::c_double,
    pub kmeansInitIsOwner: vl_bool,
}
pub type VlGMM = _VlGMM;
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
static mut vl_infinity_d: C2RustUnnamed = C2RustUnnamed {
    raw: 0x7ff0000000000000 as libc::c_ulonglong,
};
#[inline]
unsafe extern "C" fn vl_abs_d(mut x: libc::c_double) -> libc::c_double {
    return x.abs();
}
#[inline]
unsafe extern "C" fn vl_kmeans_get_centers(
    mut self_0: *const VlKMeans,
) -> *const libc::c_void {
    return (*self_0).centers;
}
#[inline]
unsafe extern "C" fn vl_kmeans_set_algorithm(
    mut self_0: *mut VlKMeans,
    mut algorithm: VlKMeansAlgorithm,
) {
    (*self_0).algorithm = algorithm;
}
#[inline]
unsafe extern "C" fn vl_kmeans_set_initialization(
    mut self_0: *mut VlKMeans,
    mut initialization: VlKMeansInitialization,
) {
    (*self_0).initialization = initialization;
}
#[inline]
unsafe extern "C" fn vl_kmeans_set_num_repetitions(
    mut self_0: *mut VlKMeans,
    mut numRepetitions: vl_size,
) {
    if numRepetitions >= 1 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"numRepetitions >= 1\0" as *const u8 as *const libc::c_char,
            b"vl/kmeans.h\0" as *const u8 as *const libc::c_char,
            302 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 56],
                &[libc::c_char; 56],
            >(b"void vl_kmeans_set_num_repetitions(VlKMeans *, vl_size)\0"))
                .as_ptr(),
        );
    }
    (*self_0).numRepetitions = numRepetitions;
}
#[inline]
unsafe extern "C" fn vl_kmeans_set_max_num_iterations(
    mut self_0: *mut VlKMeans,
    mut maxNumIterations: vl_size,
) {
    (*self_0).maxNumIterations = maxNumIterations;
}
#[inline]
unsafe extern "C" fn vl_kmeans_set_verbosity(
    mut self_0: *mut VlKMeans,
    mut verbosity: libc::c_int,
) {
    (*self_0).verbosity = verbosity;
}
#[inline]
unsafe extern "C" fn vl_kmeans_set_max_num_comparisons(
    mut self_0: *mut VlKMeans,
    mut maxNumComparisons: vl_size,
) {
    (*self_0).maxNumComparisons = maxNumComparisons;
}
#[inline]
unsafe extern "C" fn vl_kmeans_set_num_trees(
    mut self_0: *mut VlKMeans,
    mut numTrees: vl_size,
) {
    (*self_0).numTrees = numTrees;
}
unsafe extern "C" fn _vl_gmm_prepare_for_data(
    mut self_0: *mut VlGMM,
    mut numData: vl_size,
) {
    if (*self_0).numData < numData {
        vl_free((*self_0).posteriors);
        (*self_0)
            .posteriors = vl_malloc(
            (vl_get_type_size((*self_0).dataType))
                .wrapping_mul(numData)
                .wrapping_mul((*self_0).numClusters) as size_t,
        );
    }
    (*self_0).numData = numData;
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_new(
    mut dataType: vl_type,
    mut dimension: vl_size,
    mut numComponents: vl_size,
) -> *mut VlGMM {
    let mut i: vl_index = 0;
    let mut size: vl_size = vl_get_type_size(dataType);
    let mut self_0: *mut VlGMM = vl_calloc(
        1 as libc::c_int as size_t,
        ::core::mem::size_of::<VlGMM>() as libc::c_ulong,
    ) as *mut VlGMM;
    (*self_0).dataType = dataType;
    (*self_0).numClusters = numComponents;
    (*self_0).numData = 0 as libc::c_int as vl_size;
    (*self_0).dimension = dimension;
    (*self_0).initialization = VlGMMRand;
    (*self_0).verbosity = 0 as libc::c_int;
    (*self_0).maxNumIterations = 50 as libc::c_int as vl_size;
    (*self_0).numRepetitions = 1 as libc::c_int as vl_size;
    (*self_0).sigmaLowBound = 0 as *mut libc::c_double;
    (*self_0).priors = 0 as *mut libc::c_void;
    (*self_0).covariances = 0 as *mut libc::c_void;
    (*self_0).means = 0 as *mut libc::c_void;
    (*self_0).posteriors = 0 as *mut libc::c_void;
    (*self_0).kmeansInit = 0 as *mut VlKMeans;
    (*self_0).kmeansInitIsOwner = 0 as libc::c_int;
    (*self_0).priors = vl_calloc(numComponents as size_t, size as size_t);
    (*self_0)
        .means = vl_calloc(
        numComponents.wrapping_mul(dimension) as size_t,
        size as size_t,
    );
    (*self_0)
        .covariances = vl_calloc(
        numComponents.wrapping_mul(dimension) as size_t,
        size as size_t,
    );
    (*self_0)
        .sigmaLowBound = vl_calloc(
        dimension as size_t,
        ::core::mem::size_of::<libc::c_double>() as libc::c_ulong,
    ) as *mut libc::c_double;
    i = 0 as libc::c_int as vl_index;
    while i < (*self_0).dimension as libc::c_uint as libc::c_longlong {
        *((*self_0).sigmaLowBound).offset(i as isize) = 1e-4f64;
        i += 1;
    }
    return self_0;
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_reset(mut self_0: *mut VlGMM) {
    if !((*self_0).posteriors).is_null() {
        vl_free((*self_0).posteriors);
        (*self_0).posteriors = 0 as *mut libc::c_void;
        (*self_0).numData = 0 as libc::c_int as vl_size;
    }
    if !((*self_0).kmeansInit).is_null() && (*self_0).kmeansInitIsOwner != 0 {
        vl_kmeans_delete((*self_0).kmeansInit);
        (*self_0).kmeansInit = 0 as *mut VlKMeans;
        (*self_0).kmeansInitIsOwner = 0 as libc::c_int;
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_delete(mut self_0: *mut VlGMM) {
    if !((*self_0).means).is_null() {
        vl_free((*self_0).means);
    }
    if !((*self_0).covariances).is_null() {
        vl_free((*self_0).covariances);
    }
    if !((*self_0).priors).is_null() {
        vl_free((*self_0).priors);
    }
    if !((*self_0).posteriors).is_null() {
        vl_free((*self_0).posteriors);
    }
    if !((*self_0).kmeansInit).is_null() && (*self_0).kmeansInitIsOwner != 0 {
        vl_kmeans_delete((*self_0).kmeansInit);
    }
    vl_free(self_0 as *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_get_data_type(mut self_0: *const VlGMM) -> vl_type {
    return (*self_0).dataType;
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_get_num_clusters(mut self_0: *const VlGMM) -> vl_size {
    return (*self_0).numClusters;
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_get_num_data(mut self_0: *const VlGMM) -> vl_size {
    return (*self_0).numData;
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_get_loglikelihood(
    mut self_0: *const VlGMM,
) -> libc::c_double {
    return (*self_0).LL;
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_get_verbosity(mut self_0: *const VlGMM) -> libc::c_int {
    return (*self_0).verbosity;
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_set_verbosity(
    mut self_0: *mut VlGMM,
    mut verbosity: libc::c_int,
) {
    (*self_0).verbosity = verbosity;
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_get_means(
    mut self_0: *const VlGMM,
) -> *const libc::c_void {
    return (*self_0).means;
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_get_covariances(
    mut self_0: *const VlGMM,
) -> *const libc::c_void {
    return (*self_0).covariances;
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_get_priors(
    mut self_0: *const VlGMM,
) -> *const libc::c_void {
    return (*self_0).priors;
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_get_posteriors(
    mut self_0: *const VlGMM,
) -> *const libc::c_void {
    return (*self_0).posteriors;
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_get_max_num_iterations(
    mut self_0: *const VlGMM,
) -> vl_size {
    return (*self_0).maxNumIterations;
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_set_max_num_iterations(
    mut self_0: *mut VlGMM,
    mut maxNumIterations: vl_size,
) {
    (*self_0).maxNumIterations = maxNumIterations;
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_get_num_repetitions(
    mut self_0: *const VlGMM,
) -> vl_size {
    return (*self_0).numRepetitions;
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_set_num_repetitions(
    mut self_0: *mut VlGMM,
    mut numRepetitions: vl_size,
) {
    if numRepetitions >= 1 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"numRepetitions >= 1\0" as *const u8 as *const libc::c_char,
            b"vl/gmm.c\0" as *const u8 as *const libc::c_char,
            582 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 50],
                &[libc::c_char; 50],
            >(b"void vl_gmm_set_num_repetitions(VlGMM *, vl_size)\0"))
                .as_ptr(),
        );
    }
    (*self_0).numRepetitions = numRepetitions;
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_get_dimension(mut self_0: *const VlGMM) -> vl_size {
    return (*self_0).dimension;
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_get_initialization(
    mut self_0: *const VlGMM,
) -> VlGMMInitialization {
    return (*self_0).initialization;
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_set_initialization(
    mut self_0: *mut VlGMM,
    mut init: VlGMMInitialization,
) {
    (*self_0).initialization = init;
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_get_kmeans_init_object(
    mut self_0: *const VlGMM,
) -> *mut VlKMeans {
    return (*self_0).kmeansInit;
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_set_kmeans_init_object(
    mut self_0: *mut VlGMM,
    mut kmeans: *mut VlKMeans,
) {
    if !((*self_0).kmeansInit).is_null() && (*self_0).kmeansInitIsOwner != 0 {
        vl_kmeans_delete((*self_0).kmeansInit);
    }
    (*self_0).kmeansInit = kmeans;
    (*self_0).kmeansInitIsOwner = 0 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_get_covariance_lower_bounds(
    mut self_0: *const VlGMM,
) -> *const libc::c_double {
    return (*self_0).sigmaLowBound;
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_set_covariance_lower_bounds(
    mut self_0: *mut VlGMM,
    mut bounds: *const libc::c_double,
) {
    memcpy(
        (*self_0).sigmaLowBound as *mut libc::c_void,
        bounds as *const libc::c_void,
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).dimension) as libc::c_ulong,
    );
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_set_covariance_lower_bound(
    mut self_0: *mut VlGMM,
    mut bound: libc::c_double,
) {
    let mut i: libc::c_int = 0;
    i = 0 as libc::c_int;
    while i < (*self_0).dimension as libc::c_int {
        *((*self_0).sigmaLowBound).offset(i as isize) = bound;
        i += 1;
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_get_gmm_data_posteriors_d(
    mut posteriors: *mut libc::c_double,
    mut numClusters: vl_size,
    mut numData: vl_size,
    mut priors: *const libc::c_double,
    mut means: *const libc::c_double,
    mut dimension: vl_size,
    mut covariances: *const libc::c_double,
    mut data: *const libc::c_double,
) -> libc::c_double {
    let mut i_d: vl_index = 0;
    let mut i_cl: vl_index = 0;
    let mut dim: vl_size = 0;
    let mut LL: libc::c_double = 0 as libc::c_int as libc::c_double;
    let mut halfDimLog2Pi: libc::c_double = dimension as libc::c_double / 2.0f64
        * log(2.0f64 * 3.141592653589793f64);
    let mut logCovariances: *mut libc::c_double = 0 as *mut libc::c_double;
    let mut logWeights: *mut libc::c_double = 0 as *mut libc::c_double;
    let mut invCovariances: *mut libc::c_double = 0 as *mut libc::c_double;
    let mut distFn: VlDoubleVector3ComparisonFunction = vl_get_vector_3_comparison_function_d(
        VlDistanceMahalanobis,
    );
    logCovariances = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numClusters) as size_t,
    ) as *mut libc::c_double;
    invCovariances = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numClusters)
            .wrapping_mul(dimension) as size_t,
    ) as *mut libc::c_double;
    logWeights = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numClusters) as size_t,
    ) as *mut libc::c_double;
    i_cl = 0 as libc::c_int as vl_index;
    while i_cl < numClusters as libc::c_int as libc::c_longlong {
        let mut logSigma: libc::c_double = 0 as libc::c_int as libc::c_double;
        if *priors.offset(i_cl as isize) < 1e-6f64 {
            *logWeights.offset(i_cl as isize) = -vl_infinity_d.value;
        } else {
            *logWeights.offset(i_cl as isize) = log(*priors.offset(i_cl as isize));
        }
        dim = 0 as libc::c_int as vl_size;
        while dim < dimension {
            logSigma
                += log(
                    *covariances
                        .offset(
                            (i_cl as libc::c_ulonglong)
                                .wrapping_mul(dimension)
                                .wrapping_add(dim) as isize,
                        ),
                );
            *invCovariances
                .offset(
                    (i_cl as libc::c_ulonglong).wrapping_mul(dimension).wrapping_add(dim)
                        as isize,
                ) = 1.0f64
                / *covariances
                    .offset(
                        (i_cl as libc::c_ulonglong)
                            .wrapping_mul(dimension)
                            .wrapping_add(dim) as isize,
                    );
            dim = dim.wrapping_add(1);
        }
        *logCovariances.offset(i_cl as isize) = logSigma;
        i_cl += 1;
    }
    i_d = 0 as libc::c_int as vl_index;
    while i_d < numData as libc::c_int as libc::c_longlong {
        let mut clusterPosteriorsSum: libc::c_double = 0 as libc::c_int
            as libc::c_double;
        let mut maxPosterior: libc::c_double = -vl_infinity_d.value;
        i_cl = 0 as libc::c_int as vl_index;
        while i_cl < numClusters as libc::c_int as libc::c_longlong {
            let mut p: libc::c_double = *logWeights.offset(i_cl as isize) - halfDimLog2Pi
                - 0.5f64 * *logCovariances.offset(i_cl as isize)
                - 0.5f64
                    * distFn
                        .expect(
                            "non-null function pointer",
                        )(
                        dimension,
                        data
                            .offset(
                                (i_d as libc::c_ulonglong).wrapping_mul(dimension) as isize,
                            ),
                        means
                            .offset(
                                (i_cl as libc::c_ulonglong).wrapping_mul(dimension) as isize,
                            ),
                        invCovariances
                            .offset(
                                (i_cl as libc::c_ulonglong).wrapping_mul(dimension) as isize,
                            ),
                    );
            *posteriors
                .offset(
                    (i_cl as libc::c_ulonglong)
                        .wrapping_add(
                            (i_d as libc::c_ulonglong).wrapping_mul(numClusters),
                        ) as isize,
                ) = p;
            if p > maxPosterior {
                maxPosterior = p;
            }
            i_cl += 1;
        }
        i_cl = 0 as libc::c_int as vl_index;
        while i_cl < numClusters as libc::c_int as libc::c_longlong {
            let mut p_0: libc::c_double = *posteriors
                .offset(
                    (i_cl as libc::c_ulonglong)
                        .wrapping_add(
                            (i_d as libc::c_ulonglong).wrapping_mul(numClusters),
                        ) as isize,
                );
            p_0 = exp(p_0 - maxPosterior);
            *posteriors
                .offset(
                    (i_cl as libc::c_ulonglong)
                        .wrapping_add(
                            (i_d as libc::c_ulonglong).wrapping_mul(numClusters),
                        ) as isize,
                ) = p_0;
            clusterPosteriorsSum += p_0;
            i_cl += 1;
        }
        LL += log(clusterPosteriorsSum) + maxPosterior;
        i_cl = 0 as libc::c_int as vl_index;
        while i_cl < numClusters as libc::c_int as libc::c_longlong {
            *posteriors
                .offset(
                    (i_cl as libc::c_ulonglong)
                        .wrapping_add(
                            (i_d as libc::c_ulonglong).wrapping_mul(numClusters),
                        ) as isize,
                ) /= clusterPosteriorsSum;
            i_cl += 1;
        }
        i_d += 1;
    }
    vl_free(logCovariances as *mut libc::c_void);
    vl_free(logWeights as *mut libc::c_void);
    vl_free(invCovariances as *mut libc::c_void);
    return LL;
}
#[no_mangle]
pub unsafe extern "C" fn vl_get_gmm_data_posteriors_f(
    mut posteriors: *mut libc::c_float,
    mut numClusters: vl_size,
    mut numData: vl_size,
    mut priors: *const libc::c_float,
    mut means: *const libc::c_float,
    mut dimension: vl_size,
    mut covariances: *const libc::c_float,
    mut data: *const libc::c_float,
) -> libc::c_double {
    let mut i_d: vl_index = 0;
    let mut i_cl: vl_index = 0;
    let mut dim: vl_size = 0;
    let mut LL: libc::c_double = 0 as libc::c_int as libc::c_double;
    let mut halfDimLog2Pi: libc::c_float = (dimension as libc::c_double / 2.0f64
        * log(2.0f64 * 3.141592653589793f64)) as libc::c_float;
    let mut logCovariances: *mut libc::c_float = 0 as *mut libc::c_float;
    let mut logWeights: *mut libc::c_float = 0 as *mut libc::c_float;
    let mut invCovariances: *mut libc::c_float = 0 as *mut libc::c_float;
    let mut distFn: VlFloatVector3ComparisonFunction = vl_get_vector_3_comparison_function_f(
        VlDistanceMahalanobis,
    );
    logCovariances = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numClusters) as size_t,
    ) as *mut libc::c_float;
    invCovariances = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numClusters)
            .wrapping_mul(dimension) as size_t,
    ) as *mut libc::c_float;
    logWeights = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numClusters) as size_t,
    ) as *mut libc::c_float;
    i_cl = 0 as libc::c_int as vl_index;
    while i_cl < numClusters as libc::c_int as libc::c_longlong {
        let mut logSigma: libc::c_float = 0 as libc::c_int as libc::c_float;
        if (*priors.offset(i_cl as isize) as libc::c_double) < 1e-6f64 {
            *logWeights.offset(i_cl as isize) = -(vl_infinity_d.value as libc::c_float);
        } else {
            *logWeights
                .offset(
                    i_cl as isize,
                ) = log(*priors.offset(i_cl as isize) as libc::c_double)
                as libc::c_float;
        }
        dim = 0 as libc::c_int as vl_size;
        while dim < dimension {
            logSigma = (logSigma as libc::c_double
                + log(
                    *covariances
                        .offset(
                            (i_cl as libc::c_ulonglong)
                                .wrapping_mul(dimension)
                                .wrapping_add(dim) as isize,
                        ) as libc::c_double,
                )) as libc::c_float;
            *invCovariances
                .offset(
                    (i_cl as libc::c_ulonglong).wrapping_mul(dimension).wrapping_add(dim)
                        as isize,
                ) = 1.0f64 as libc::c_float
                / *covariances
                    .offset(
                        (i_cl as libc::c_ulonglong)
                            .wrapping_mul(dimension)
                            .wrapping_add(dim) as isize,
                    );
            dim = dim.wrapping_add(1);
        }
        *logCovariances.offset(i_cl as isize) = logSigma;
        i_cl += 1;
    }
    i_d = 0 as libc::c_int as vl_index;
    while i_d < numData as libc::c_int as libc::c_longlong {
        let mut clusterPosteriorsSum: libc::c_float = 0 as libc::c_int as libc::c_float;
        let mut maxPosterior: libc::c_float = -vl_infinity_d.value as libc::c_float;
        i_cl = 0 as libc::c_int as vl_index;
        while i_cl < numClusters as libc::c_int as libc::c_longlong {
            let mut p: libc::c_float = ((*logWeights.offset(i_cl as isize)
                - halfDimLog2Pi) as libc::c_double
                - 0.5f64 * *logCovariances.offset(i_cl as isize) as libc::c_double
                - 0.5f64
                    * distFn
                        .expect(
                            "non-null function pointer",
                        )(
                        dimension,
                        data
                            .offset(
                                (i_d as libc::c_ulonglong).wrapping_mul(dimension) as isize,
                            ),
                        means
                            .offset(
                                (i_cl as libc::c_ulonglong).wrapping_mul(dimension) as isize,
                            ),
                        invCovariances
                            .offset(
                                (i_cl as libc::c_ulonglong).wrapping_mul(dimension) as isize,
                            ),
                    ) as libc::c_double) as libc::c_float;
            *posteriors
                .offset(
                    (i_cl as libc::c_ulonglong)
                        .wrapping_add(
                            (i_d as libc::c_ulonglong).wrapping_mul(numClusters),
                        ) as isize,
                ) = p;
            if p > maxPosterior {
                maxPosterior = p;
            }
            i_cl += 1;
        }
        i_cl = 0 as libc::c_int as vl_index;
        while i_cl < numClusters as libc::c_int as libc::c_longlong {
            let mut p_0: libc::c_float = *posteriors
                .offset(
                    (i_cl as libc::c_ulonglong)
                        .wrapping_add(
                            (i_d as libc::c_ulonglong).wrapping_mul(numClusters),
                        ) as isize,
                );
            p_0 = exp((p_0 - maxPosterior) as libc::c_double) as libc::c_float;
            *posteriors
                .offset(
                    (i_cl as libc::c_ulonglong)
                        .wrapping_add(
                            (i_d as libc::c_ulonglong).wrapping_mul(numClusters),
                        ) as isize,
                ) = p_0;
            clusterPosteriorsSum += p_0;
            i_cl += 1;
        }
        LL
            += log(clusterPosteriorsSum as libc::c_double)
                + maxPosterior as libc::c_double;
        i_cl = 0 as libc::c_int as vl_index;
        while i_cl < numClusters as libc::c_int as libc::c_longlong {
            *posteriors
                .offset(
                    (i_cl as libc::c_ulonglong)
                        .wrapping_add(
                            (i_d as libc::c_ulonglong).wrapping_mul(numClusters),
                        ) as isize,
                ) /= clusterPosteriorsSum;
            i_cl += 1;
        }
        i_d += 1;
    }
    vl_free(logCovariances as *mut libc::c_void);
    vl_free(logWeights as *mut libc::c_void);
    vl_free(invCovariances as *mut libc::c_void);
    return LL;
}
unsafe extern "C" fn _vl_gmm_restart_empty_modes_f(
    mut self_0: *mut VlGMM,
    mut data: *const libc::c_float,
) -> vl_size {
    let mut dimension: vl_size = (*self_0).dimension;
    let mut numClusters: vl_size = (*self_0).numClusters;
    let mut i_cl: vl_index = 0;
    let mut j_cl: vl_index = 0;
    let mut i_d: vl_index = 0;
    let mut d: vl_index = 0;
    let mut zeroWNum: vl_size = 0 as libc::c_int as vl_size;
    let mut priors: *mut libc::c_float = (*self_0).priors as *mut libc::c_float;
    let mut means: *mut libc::c_float = (*self_0).means as *mut libc::c_float;
    let mut covariances: *mut libc::c_float = (*self_0).covariances
        as *mut libc::c_float;
    let mut posteriors: *mut libc::c_float = (*self_0).posteriors as *mut libc::c_float;
    let mut mass: *mut libc::c_float = vl_calloc(
        ::core::mem::size_of::<libc::c_float>() as libc::c_ulong,
        (*self_0).numClusters as size_t,
    ) as *mut libc::c_float;
    if numClusters <= 1 as libc::c_int as libc::c_ulonglong {
        return 0 as libc::c_int as vl_size;
    }
    let mut i: vl_uindex = 0;
    let mut k: vl_uindex = 0;
    let mut numNullAssignments: vl_size = 0 as libc::c_int as vl_size;
    i = 0 as libc::c_int as vl_uindex;
    while i < (*self_0).numData {
        k = 0 as libc::c_int as vl_uindex;
        while k < (*self_0).numClusters {
            let mut p: libc::c_float = *((*self_0).posteriors as *mut libc::c_float)
                .offset(k.wrapping_add(i.wrapping_mul((*self_0).numClusters)) as isize);
            *mass.offset(k as isize) += p;
            if (p as libc::c_double) < 1e-2f64 {
                numNullAssignments = numNullAssignments.wrapping_add(1);
            }
            k = k.wrapping_add(1);
        }
        i = i.wrapping_add(1);
    }
    if (*self_0).verbosity != 0 {
        (Some(
            ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                .expect("non-null function pointer"),
        ))
            .expect(
                "non-null function pointer",
            )(
            b"gmm: sparsity of data posterior: %.1f%%\n\0" as *const u8
                as *const libc::c_char,
            numNullAssignments as libc::c_double
                / ((*self_0).numData).wrapping_mul((*self_0).numClusters)
                    as libc::c_double * 100 as libc::c_int as libc::c_double,
        );
    }
    i_cl = 0 as libc::c_int as vl_index;
    while i_cl < numClusters as libc::c_int as libc::c_longlong {
        let mut size: libc::c_double = -vl_infinity_d.value;
        let mut best: vl_index = -(1 as libc::c_int) as vl_index;
        if !(*mass.offset(i_cl as isize) as libc::c_double
            >= 1e-2f64
                * (if 1.0f64
                    > (*self_0).numData as libc::c_double
                        / (*self_0).numClusters as libc::c_double
                {
                    1.0f64
                } else {
                    (*self_0).numData as libc::c_double
                        / (*self_0).numClusters as libc::c_double
                }))
        {
            if (*self_0).verbosity != 0 {
                (Some(
                    ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                        .expect("non-null function pointer"),
                ))
                    .expect(
                        "non-null function pointer",
                    )(
                    b"gmm: mode %d is nearly empty (mass %f)\n\0" as *const u8
                        as *const libc::c_char,
                    i_cl,
                    *mass.offset(i_cl as isize) as libc::c_double,
                );
            }
            j_cl = 0 as libc::c_int as vl_index;
            while j_cl < numClusters as libc::c_int as libc::c_longlong {
                let mut size_: libc::c_double = 0.;
                if !((*priors.offset(j_cl as isize) as libc::c_double) < 1e-6f64) {
                    size_ = 0.5f64 * dimension as libc::c_double
                        * (1.0f64
                            + log(
                                2 as libc::c_int as libc::c_double * 3.141592653589793f64,
                            ));
                    d = 0 as libc::c_int as vl_index;
                    while d < dimension as libc::c_int as libc::c_longlong {
                        let mut sigma2: libc::c_double = *covariances
                            .offset(
                                (j_cl as libc::c_ulonglong)
                                    .wrapping_mul(dimension)
                                    .wrapping_add(d as libc::c_ulonglong) as isize,
                            ) as libc::c_double;
                        size_ += 0.5f64 * log(sigma2);
                        d += 1;
                    }
                    size_ = *priors.offset(j_cl as isize) as libc::c_double
                        * (size_ - log(*priors.offset(j_cl as isize) as libc::c_double));
                    if (*self_0).verbosity > 1 as libc::c_int {
                        (Some(
                            ((vl_get_printf_func
                                as unsafe extern "C" fn() -> printf_func_t)())
                                .expect("non-null function pointer"),
                        ))
                            .expect(
                                "non-null function pointer",
                            )(
                            b"gmm: mode %d: prior %f, mass %f, entropy contribution %f\n\0"
                                as *const u8 as *const libc::c_char,
                            j_cl,
                            *priors.offset(j_cl as isize) as libc::c_double,
                            *mass.offset(j_cl as isize) as libc::c_double,
                            size_,
                        );
                    }
                    if size_ > size {
                        size = size_;
                        best = j_cl;
                    }
                }
                j_cl += 1;
            }
            j_cl = best;
            if j_cl == i_cl || j_cl < 0 as libc::c_int as libc::c_longlong {
                if (*self_0).verbosity != 0 {
                    (Some(
                        ((vl_get_printf_func
                            as unsafe extern "C" fn() -> printf_func_t)())
                            .expect("non-null function pointer"),
                    ))
                        .expect(
                            "non-null function pointer",
                        )(
                        b"gmm: mode %d is empty, but no other mode to split could be found\n\0"
                            as *const u8 as *const libc::c_char,
                        i_cl,
                    );
                }
            } else {
                if (*self_0).verbosity != 0 {
                    (Some(
                        ((vl_get_printf_func
                            as unsafe extern "C" fn() -> printf_func_t)())
                            .expect("non-null function pointer"),
                    ))
                        .expect(
                            "non-null function pointer",
                        )(
                        b"gmm: reinitializing empty mode %d with mode %d (prior %f, mass %f, score %f)\n\0"
                            as *const u8 as *const libc::c_char,
                        i_cl,
                        j_cl,
                        *priors.offset(j_cl as isize) as libc::c_double,
                        *mass.offset(j_cl as isize) as libc::c_double,
                        size,
                    );
                }
                size = -vl_infinity_d.value;
                best = -(1 as libc::c_int) as vl_index;
                d = 0 as libc::c_int as vl_index;
                while d < dimension as libc::c_int as libc::c_longlong {
                    let mut sigma2_0: libc::c_double = *covariances
                        .offset(
                            (j_cl as libc::c_ulonglong)
                                .wrapping_mul(dimension)
                                .wrapping_add(d as libc::c_ulonglong) as isize,
                        ) as libc::c_double;
                    if sigma2_0 > size {
                        size = sigma2_0;
                        best = d;
                    }
                    d += 1;
                }
                let mut mu: libc::c_float = *means
                    .offset(
                        (best as libc::c_ulonglong)
                            .wrapping_add(
                                (j_cl as libc::c_ulonglong)
                                    .wrapping_mul((*self_0).dimension),
                            ) as isize,
                    );
                i_d = 0 as libc::c_int as vl_index;
                while i_d < (*self_0).numData as libc::c_int as libc::c_longlong {
                    let mut p_0: libc::c_float = *posteriors
                        .offset(
                            (j_cl as libc::c_ulonglong)
                                .wrapping_add(
                                    ((*self_0).numClusters)
                                        .wrapping_mul(i_d as libc::c_ulonglong),
                                ) as isize,
                        );
                    let mut q: libc::c_float = *posteriors
                        .offset(
                            (i_cl as libc::c_ulonglong)
                                .wrapping_add(
                                    ((*self_0).numClusters)
                                        .wrapping_mul(i_d as libc::c_ulonglong),
                                ) as isize,
                        );
                    if *data
                        .offset(
                            (best as libc::c_ulonglong)
                                .wrapping_add(
                                    (i_d as libc::c_ulonglong).wrapping_mul((*self_0).dimension),
                                ) as isize,
                        ) < mu
                    {
                        *posteriors
                            .offset(
                                (i_cl as libc::c_ulonglong)
                                    .wrapping_add(
                                        ((*self_0).numClusters)
                                            .wrapping_mul(i_d as libc::c_ulonglong),
                                    ) as isize,
                            ) = p_0 + q;
                        *posteriors
                            .offset(
                                (j_cl as libc::c_ulonglong)
                                    .wrapping_add(
                                        ((*self_0).numClusters)
                                            .wrapping_mul(i_d as libc::c_ulonglong),
                                    ) as isize,
                            ) = 0 as libc::c_int as libc::c_float;
                    } else {
                        *posteriors
                            .offset(
                                (i_cl as libc::c_ulonglong)
                                    .wrapping_add(
                                        ((*self_0).numClusters)
                                            .wrapping_mul(i_d as libc::c_ulonglong),
                                    ) as isize,
                            ) = 0 as libc::c_int as libc::c_float;
                        *posteriors
                            .offset(
                                (j_cl as libc::c_ulonglong)
                                    .wrapping_add(
                                        ((*self_0).numClusters)
                                            .wrapping_mul(i_d as libc::c_ulonglong),
                                    ) as isize,
                            ) = p_0 + q;
                    }
                    i_d += 1;
                }
                _vl_gmm_maximization_f(
                    self_0,
                    posteriors,
                    priors,
                    covariances,
                    means,
                    data,
                    (*self_0).numData,
                );
            }
        }
        i_cl += 1;
    }
    return zeroWNum;
}
unsafe extern "C" fn _vl_gmm_restart_empty_modes_d(
    mut self_0: *mut VlGMM,
    mut data: *const libc::c_double,
) -> vl_size {
    let mut dimension: vl_size = (*self_0).dimension;
    let mut numClusters: vl_size = (*self_0).numClusters;
    let mut i_cl: vl_index = 0;
    let mut j_cl: vl_index = 0;
    let mut i_d: vl_index = 0;
    let mut d: vl_index = 0;
    let mut zeroWNum: vl_size = 0 as libc::c_int as vl_size;
    let mut priors: *mut libc::c_double = (*self_0).priors as *mut libc::c_double;
    let mut means: *mut libc::c_double = (*self_0).means as *mut libc::c_double;
    let mut covariances: *mut libc::c_double = (*self_0).covariances
        as *mut libc::c_double;
    let mut posteriors: *mut libc::c_double = (*self_0).posteriors
        as *mut libc::c_double;
    let mut mass: *mut libc::c_double = vl_calloc(
        ::core::mem::size_of::<libc::c_double>() as libc::c_ulong,
        (*self_0).numClusters as size_t,
    ) as *mut libc::c_double;
    if numClusters <= 1 as libc::c_int as libc::c_ulonglong {
        return 0 as libc::c_int as vl_size;
    }
    let mut i: vl_uindex = 0;
    let mut k: vl_uindex = 0;
    let mut numNullAssignments: vl_size = 0 as libc::c_int as vl_size;
    i = 0 as libc::c_int as vl_uindex;
    while i < (*self_0).numData {
        k = 0 as libc::c_int as vl_uindex;
        while k < (*self_0).numClusters {
            let mut p: libc::c_double = *((*self_0).posteriors as *mut libc::c_double)
                .offset(k.wrapping_add(i.wrapping_mul((*self_0).numClusters)) as isize);
            *mass.offset(k as isize) += p;
            if p < 1e-2f64 {
                numNullAssignments = numNullAssignments.wrapping_add(1);
            }
            k = k.wrapping_add(1);
        }
        i = i.wrapping_add(1);
    }
    if (*self_0).verbosity != 0 {
        (Some(
            ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                .expect("non-null function pointer"),
        ))
            .expect(
                "non-null function pointer",
            )(
            b"gmm: sparsity of data posterior: %.1f%%\n\0" as *const u8
                as *const libc::c_char,
            numNullAssignments as libc::c_double
                / ((*self_0).numData).wrapping_mul((*self_0).numClusters)
                    as libc::c_double * 100 as libc::c_int as libc::c_double,
        );
    }
    i_cl = 0 as libc::c_int as vl_index;
    while i_cl < numClusters as libc::c_int as libc::c_longlong {
        let mut size: libc::c_double = -vl_infinity_d.value;
        let mut best: vl_index = -(1 as libc::c_int) as vl_index;
        if !(*mass.offset(i_cl as isize)
            >= 1e-2f64
                * (if 1.0f64
                    > (*self_0).numData as libc::c_double
                        / (*self_0).numClusters as libc::c_double
                {
                    1.0f64
                } else {
                    (*self_0).numData as libc::c_double
                        / (*self_0).numClusters as libc::c_double
                }))
        {
            if (*self_0).verbosity != 0 {
                (Some(
                    ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                        .expect("non-null function pointer"),
                ))
                    .expect(
                        "non-null function pointer",
                    )(
                    b"gmm: mode %d is nearly empty (mass %f)\n\0" as *const u8
                        as *const libc::c_char,
                    i_cl,
                    *mass.offset(i_cl as isize),
                );
            }
            j_cl = 0 as libc::c_int as vl_index;
            while j_cl < numClusters as libc::c_int as libc::c_longlong {
                let mut size_: libc::c_double = 0.;
                if !(*priors.offset(j_cl as isize) < 1e-6f64) {
                    size_ = 0.5f64 * dimension as libc::c_double
                        * (1.0f64
                            + log(
                                2 as libc::c_int as libc::c_double * 3.141592653589793f64,
                            ));
                    d = 0 as libc::c_int as vl_index;
                    while d < dimension as libc::c_int as libc::c_longlong {
                        let mut sigma2: libc::c_double = *covariances
                            .offset(
                                (j_cl as libc::c_ulonglong)
                                    .wrapping_mul(dimension)
                                    .wrapping_add(d as libc::c_ulonglong) as isize,
                            );
                        size_ += 0.5f64 * log(sigma2);
                        d += 1;
                    }
                    size_ = *priors.offset(j_cl as isize)
                        * (size_ - log(*priors.offset(j_cl as isize)));
                    if (*self_0).verbosity > 1 as libc::c_int {
                        (Some(
                            ((vl_get_printf_func
                                as unsafe extern "C" fn() -> printf_func_t)())
                                .expect("non-null function pointer"),
                        ))
                            .expect(
                                "non-null function pointer",
                            )(
                            b"gmm: mode %d: prior %f, mass %f, entropy contribution %f\n\0"
                                as *const u8 as *const libc::c_char,
                            j_cl,
                            *priors.offset(j_cl as isize),
                            *mass.offset(j_cl as isize),
                            size_,
                        );
                    }
                    if size_ > size {
                        size = size_;
                        best = j_cl;
                    }
                }
                j_cl += 1;
            }
            j_cl = best;
            if j_cl == i_cl || j_cl < 0 as libc::c_int as libc::c_longlong {
                if (*self_0).verbosity != 0 {
                    (Some(
                        ((vl_get_printf_func
                            as unsafe extern "C" fn() -> printf_func_t)())
                            .expect("non-null function pointer"),
                    ))
                        .expect(
                            "non-null function pointer",
                        )(
                        b"gmm: mode %d is empty, but no other mode to split could be found\n\0"
                            as *const u8 as *const libc::c_char,
                        i_cl,
                    );
                }
            } else {
                if (*self_0).verbosity != 0 {
                    (Some(
                        ((vl_get_printf_func
                            as unsafe extern "C" fn() -> printf_func_t)())
                            .expect("non-null function pointer"),
                    ))
                        .expect(
                            "non-null function pointer",
                        )(
                        b"gmm: reinitializing empty mode %d with mode %d (prior %f, mass %f, score %f)\n\0"
                            as *const u8 as *const libc::c_char,
                        i_cl,
                        j_cl,
                        *priors.offset(j_cl as isize),
                        *mass.offset(j_cl as isize),
                        size,
                    );
                }
                size = -vl_infinity_d.value;
                best = -(1 as libc::c_int) as vl_index;
                d = 0 as libc::c_int as vl_index;
                while d < dimension as libc::c_int as libc::c_longlong {
                    let mut sigma2_0: libc::c_double = *covariances
                        .offset(
                            (j_cl as libc::c_ulonglong)
                                .wrapping_mul(dimension)
                                .wrapping_add(d as libc::c_ulonglong) as isize,
                        );
                    if sigma2_0 > size {
                        size = sigma2_0;
                        best = d;
                    }
                    d += 1;
                }
                let mut mu: libc::c_double = *means
                    .offset(
                        (best as libc::c_ulonglong)
                            .wrapping_add(
                                (j_cl as libc::c_ulonglong)
                                    .wrapping_mul((*self_0).dimension),
                            ) as isize,
                    );
                i_d = 0 as libc::c_int as vl_index;
                while i_d < (*self_0).numData as libc::c_int as libc::c_longlong {
                    let mut p_0: libc::c_double = *posteriors
                        .offset(
                            (j_cl as libc::c_ulonglong)
                                .wrapping_add(
                                    ((*self_0).numClusters)
                                        .wrapping_mul(i_d as libc::c_ulonglong),
                                ) as isize,
                        );
                    let mut q: libc::c_double = *posteriors
                        .offset(
                            (i_cl as libc::c_ulonglong)
                                .wrapping_add(
                                    ((*self_0).numClusters)
                                        .wrapping_mul(i_d as libc::c_ulonglong),
                                ) as isize,
                        );
                    if *data
                        .offset(
                            (best as libc::c_ulonglong)
                                .wrapping_add(
                                    (i_d as libc::c_ulonglong).wrapping_mul((*self_0).dimension),
                                ) as isize,
                        ) < mu
                    {
                        *posteriors
                            .offset(
                                (i_cl as libc::c_ulonglong)
                                    .wrapping_add(
                                        ((*self_0).numClusters)
                                            .wrapping_mul(i_d as libc::c_ulonglong),
                                    ) as isize,
                            ) = p_0 + q;
                        *posteriors
                            .offset(
                                (j_cl as libc::c_ulonglong)
                                    .wrapping_add(
                                        ((*self_0).numClusters)
                                            .wrapping_mul(i_d as libc::c_ulonglong),
                                    ) as isize,
                            ) = 0 as libc::c_int as libc::c_double;
                    } else {
                        *posteriors
                            .offset(
                                (i_cl as libc::c_ulonglong)
                                    .wrapping_add(
                                        ((*self_0).numClusters)
                                            .wrapping_mul(i_d as libc::c_ulonglong),
                                    ) as isize,
                            ) = 0 as libc::c_int as libc::c_double;
                        *posteriors
                            .offset(
                                (j_cl as libc::c_ulonglong)
                                    .wrapping_add(
                                        ((*self_0).numClusters)
                                            .wrapping_mul(i_d as libc::c_ulonglong),
                                    ) as isize,
                            ) = p_0 + q;
                    }
                    i_d += 1;
                }
                _vl_gmm_maximization_d(
                    self_0,
                    posteriors,
                    priors,
                    covariances,
                    means,
                    data,
                    (*self_0).numData,
                );
            }
        }
        i_cl += 1;
    }
    return zeroWNum;
}
unsafe extern "C" fn _vl_gmm_apply_bounds_f(mut self_0: *mut VlGMM) {
    let mut dim: vl_uindex = 0;
    let mut k: vl_uindex = 0;
    let mut numAdjusted: vl_size = 0 as libc::c_int as vl_size;
    let mut cov: *mut libc::c_float = (*self_0).covariances as *mut libc::c_float;
    let mut lbs: *const libc::c_double = (*self_0).sigmaLowBound;
    k = 0 as libc::c_int as vl_uindex;
    while k < (*self_0).numClusters {
        let mut adjusted: vl_bool = 0 as libc::c_int;
        dim = 0 as libc::c_int as vl_uindex;
        while dim < (*self_0).dimension {
            if (*cov
                .offset(k.wrapping_mul((*self_0).dimension).wrapping_add(dim) as isize)
                as libc::c_double) < *lbs.offset(dim as isize)
            {
                *cov
                    .offset(
                        k.wrapping_mul((*self_0).dimension).wrapping_add(dim) as isize,
                    ) = *lbs.offset(dim as isize) as libc::c_float;
                adjusted = 1 as libc::c_int;
            }
            dim = dim.wrapping_add(1);
        }
        if adjusted != 0 {
            numAdjusted = numAdjusted.wrapping_add(1);
        }
        k = k.wrapping_add(1);
    }
    if numAdjusted > 0 as libc::c_int as libc::c_ulonglong
        && (*self_0).verbosity > 0 as libc::c_int
    {
        (Some(
            ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                .expect("non-null function pointer"),
        ))
            .expect(
                "non-null function pointer",
            )(
            b"gmm: detected %d of %d modes with at least one dimension with covariance too small (set to lower bound)\n\0"
                as *const u8 as *const libc::c_char,
            numAdjusted,
            (*self_0).numClusters,
        );
    }
}
unsafe extern "C" fn _vl_gmm_apply_bounds_d(mut self_0: *mut VlGMM) {
    let mut dim: vl_uindex = 0;
    let mut k: vl_uindex = 0;
    let mut numAdjusted: vl_size = 0 as libc::c_int as vl_size;
    let mut cov: *mut libc::c_double = (*self_0).covariances as *mut libc::c_double;
    let mut lbs: *const libc::c_double = (*self_0).sigmaLowBound;
    k = 0 as libc::c_int as vl_uindex;
    while k < (*self_0).numClusters {
        let mut adjusted: vl_bool = 0 as libc::c_int;
        dim = 0 as libc::c_int as vl_uindex;
        while dim < (*self_0).dimension {
            if *cov
                .offset(k.wrapping_mul((*self_0).dimension).wrapping_add(dim) as isize)
                < *lbs.offset(dim as isize)
            {
                *cov
                    .offset(
                        k.wrapping_mul((*self_0).dimension).wrapping_add(dim) as isize,
                    ) = *lbs.offset(dim as isize);
                adjusted = 1 as libc::c_int;
            }
            dim = dim.wrapping_add(1);
        }
        if adjusted != 0 {
            numAdjusted = numAdjusted.wrapping_add(1);
        }
        k = k.wrapping_add(1);
    }
    if numAdjusted > 0 as libc::c_int as libc::c_ulonglong
        && (*self_0).verbosity > 0 as libc::c_int
    {
        (Some(
            ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                .expect("non-null function pointer"),
        ))
            .expect(
                "non-null function pointer",
            )(
            b"gmm: detected %d of %d modes with at least one dimension with covariance too small (set to lower bound)\n\0"
                as *const u8 as *const libc::c_char,
            numAdjusted,
            (*self_0).numClusters,
        );
    }
}
unsafe extern "C" fn _vl_gmm_maximization_d(
    mut self_0: *mut VlGMM,
    mut posteriors: *mut libc::c_double,
    mut priors: *mut libc::c_double,
    mut covariances: *mut libc::c_double,
    mut means: *mut libc::c_double,
    mut data: *const libc::c_double,
    mut numData: vl_size,
) {
    let mut numClusters: vl_size = (*self_0).numClusters;
    let mut i_d: vl_index = 0;
    let mut i_cl: vl_index = 0;
    let mut dim: vl_size = 0;
    let mut oldMeans: *mut libc::c_double = 0 as *mut libc::c_double;
    let mut time: libc::c_double = 0 as libc::c_int as libc::c_double;
    if (*self_0).verbosity > 1 as libc::c_int {
        (Some(
            ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                .expect("non-null function pointer"),
        ))
            .expect(
                "non-null function pointer",
            )(
            b"gmm: em: entering maximization step\n\0" as *const u8
                as *const libc::c_char,
        );
        time = vl_get_cpu_time();
    }
    oldMeans = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).dimension)
            .wrapping_mul(numClusters) as size_t,
    ) as *mut libc::c_double;
    memcpy(
        oldMeans as *mut libc::c_void,
        means as *const libc::c_void,
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).dimension)
            .wrapping_mul(numClusters) as libc::c_ulong,
    );
    memset(
        priors as *mut libc::c_void,
        0 as libc::c_int,
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numClusters) as libc::c_ulong,
    );
    memset(
        means as *mut libc::c_void,
        0 as libc::c_int,
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).dimension)
            .wrapping_mul(numClusters) as libc::c_ulong,
    );
    memset(
        covariances as *mut libc::c_void,
        0 as libc::c_int,
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).dimension)
            .wrapping_mul(numClusters) as libc::c_ulong,
    );
    let mut clusterPosteriorSum_: *mut libc::c_double = 0 as *mut libc::c_double;
    let mut means_: *mut libc::c_double = 0 as *mut libc::c_double;
    let mut covariances_: *mut libc::c_double = 0 as *mut libc::c_double;
    clusterPosteriorSum_ = vl_calloc(
        ::core::mem::size_of::<libc::c_double>() as libc::c_ulong,
        numClusters as size_t,
    ) as *mut libc::c_double;
    means_ = vl_calloc(
        ::core::mem::size_of::<libc::c_double>() as libc::c_ulong,
        ((*self_0).dimension).wrapping_mul(numClusters) as size_t,
    ) as *mut libc::c_double;
    covariances_ = vl_calloc(
        ::core::mem::size_of::<libc::c_double>() as libc::c_ulong,
        ((*self_0).dimension).wrapping_mul(numClusters) as size_t,
    ) as *mut libc::c_double;
    i_d = 0 as libc::c_int as vl_index;
    while i_d < numData as libc::c_int as libc::c_longlong {
        i_cl = 0 as libc::c_int as vl_index;
        while i_cl < numClusters as libc::c_int as libc::c_longlong {
            let mut p: libc::c_double = *posteriors
                .offset(
                    (i_cl as libc::c_ulonglong)
                        .wrapping_add(
                            (i_d as libc::c_ulonglong)
                                .wrapping_mul((*self_0).numClusters),
                        ) as isize,
                );
            let mut calculated: vl_bool = 0 as libc::c_int;
            if !(p < 1e-2f64 / numClusters as libc::c_double) {
                *clusterPosteriorSum_.offset(i_cl as isize) += p;
                if vl_get_simd_enabled() != 0 && vl_cpu_has_avx() != 0 {
                    _vl_weighted_mean_avx_d(
                        (*self_0).dimension,
                        means_
                            .offset(
                                (i_cl as libc::c_ulonglong)
                                    .wrapping_mul((*self_0).dimension) as isize,
                            ),
                        data
                            .offset(
                                (i_d as libc::c_ulonglong).wrapping_mul((*self_0).dimension)
                                    as isize,
                            ),
                        p,
                    );
                    _vl_weighted_sigma_avx_d(
                        (*self_0).dimension,
                        covariances_
                            .offset(
                                (i_cl as libc::c_ulonglong)
                                    .wrapping_mul((*self_0).dimension) as isize,
                            ),
                        data
                            .offset(
                                (i_d as libc::c_ulonglong).wrapping_mul((*self_0).dimension)
                                    as isize,
                            ),
                        oldMeans
                            .offset(
                                (i_cl as libc::c_ulonglong)
                                    .wrapping_mul((*self_0).dimension) as isize,
                            ),
                        p,
                    );
                    calculated = 1 as libc::c_int;
                }
                if vl_get_simd_enabled() != 0 && vl_cpu_has_sse2() != 0
                    && calculated == 0
                {
                    _vl_weighted_mean_sse2_d(
                        (*self_0).dimension,
                        means_
                            .offset(
                                (i_cl as libc::c_ulonglong)
                                    .wrapping_mul((*self_0).dimension) as isize,
                            ),
                        data
                            .offset(
                                (i_d as libc::c_ulonglong).wrapping_mul((*self_0).dimension)
                                    as isize,
                            ),
                        p,
                    );
                    _vl_weighted_sigma_sse2_d(
                        (*self_0).dimension,
                        covariances_
                            .offset(
                                (i_cl as libc::c_ulonglong)
                                    .wrapping_mul((*self_0).dimension) as isize,
                            ),
                        data
                            .offset(
                                (i_d as libc::c_ulonglong).wrapping_mul((*self_0).dimension)
                                    as isize,
                            ),
                        oldMeans
                            .offset(
                                (i_cl as libc::c_ulonglong)
                                    .wrapping_mul((*self_0).dimension) as isize,
                            ),
                        p,
                    );
                    calculated = 1 as libc::c_int;
                }
                if calculated == 0 {
                    dim = 0 as libc::c_int as vl_size;
                    while dim < (*self_0).dimension {
                        let mut x: libc::c_double = *data
                            .offset(
                                (i_d as libc::c_ulonglong)
                                    .wrapping_mul((*self_0).dimension)
                                    .wrapping_add(dim) as isize,
                            );
                        let mut mu: libc::c_double = *oldMeans
                            .offset(
                                (i_cl as libc::c_ulonglong)
                                    .wrapping_mul((*self_0).dimension)
                                    .wrapping_add(dim) as isize,
                            );
                        let mut diff: libc::c_double = x - mu;
                        *means_
                            .offset(
                                (i_cl as libc::c_ulonglong)
                                    .wrapping_mul((*self_0).dimension)
                                    .wrapping_add(dim) as isize,
                            ) += p * x;
                        *covariances_
                            .offset(
                                (i_cl as libc::c_ulonglong)
                                    .wrapping_mul((*self_0).dimension)
                                    .wrapping_add(dim) as isize,
                            ) += p * (diff * diff);
                        dim = dim.wrapping_add(1);
                    }
                }
            }
            i_cl += 1;
        }
        i_d += 1;
    }
    i_cl = 0 as libc::c_int as vl_index;
    while i_cl < numClusters as libc::c_int as libc::c_longlong {
        *priors.offset(i_cl as isize) += *clusterPosteriorSum_.offset(i_cl as isize);
        dim = 0 as libc::c_int as vl_size;
        while dim < (*self_0).dimension {
            *means
                .offset(
                    (i_cl as libc::c_ulonglong)
                        .wrapping_mul((*self_0).dimension)
                        .wrapping_add(dim) as isize,
                )
                += *means_
                    .offset(
                        (i_cl as libc::c_ulonglong)
                            .wrapping_mul((*self_0).dimension)
                            .wrapping_add(dim) as isize,
                    );
            *covariances
                .offset(
                    (i_cl as libc::c_ulonglong)
                        .wrapping_mul((*self_0).dimension)
                        .wrapping_add(dim) as isize,
                )
                += *covariances_
                    .offset(
                        (i_cl as libc::c_ulonglong)
                            .wrapping_mul((*self_0).dimension)
                            .wrapping_add(dim) as isize,
                    );
            dim = dim.wrapping_add(1);
        }
        i_cl += 1;
    }
    vl_free(means_ as *mut libc::c_void);
    vl_free(covariances_ as *mut libc::c_void);
    vl_free(clusterPosteriorSum_ as *mut libc::c_void);
    i_cl = 0 as libc::c_int as vl_index;
    while i_cl < numClusters as libc::c_int as libc::c_longlong {
        let mut mass: libc::c_double = *priors.offset(i_cl as isize);
        if mass >= 1e-6f64 / numClusters as libc::c_double {
            dim = 0 as libc::c_int as vl_size;
            while dim < (*self_0).dimension {
                *means
                    .offset(
                        (i_cl as libc::c_ulonglong)
                            .wrapping_mul((*self_0).dimension)
                            .wrapping_add(dim) as isize,
                    ) /= mass;
                *covariances
                    .offset(
                        (i_cl as libc::c_ulonglong)
                            .wrapping_mul((*self_0).dimension)
                            .wrapping_add(dim) as isize,
                    ) /= mass;
                dim = dim.wrapping_add(1);
            }
        }
        i_cl += 1;
    }
    i_cl = 0 as libc::c_int as vl_index;
    while i_cl < numClusters as libc::c_int as libc::c_longlong {
        let mut mass_0: libc::c_double = *priors.offset(i_cl as isize);
        if mass_0 >= 1e-6f64 / numClusters as libc::c_double {
            dim = 0 as libc::c_int as vl_size;
            while dim < (*self_0).dimension {
                let mut mu_0: libc::c_double = *means
                    .offset(
                        (i_cl as libc::c_ulonglong)
                            .wrapping_mul((*self_0).dimension)
                            .wrapping_add(dim) as isize,
                    );
                let mut oldMu: libc::c_double = *oldMeans
                    .offset(
                        (i_cl as libc::c_ulonglong)
                            .wrapping_mul((*self_0).dimension)
                            .wrapping_add(dim) as isize,
                    );
                let mut diff_0: libc::c_double = mu_0 - oldMu;
                *covariances
                    .offset(
                        (i_cl as libc::c_ulonglong)
                            .wrapping_mul((*self_0).dimension)
                            .wrapping_add(dim) as isize,
                    ) -= diff_0 * diff_0;
                dim = dim.wrapping_add(1);
            }
        }
        i_cl += 1;
    }
    _vl_gmm_apply_bounds_d(self_0);
    let mut sum: libc::c_double = 0 as libc::c_int as libc::c_double;
    i_cl = 0 as libc::c_int as vl_index;
    while i_cl < numClusters as libc::c_int as libc::c_longlong {
        sum += *priors.offset(i_cl as isize);
        i_cl += 1;
    }
    sum = if sum > 1e-12f64 { sum } else { 1e-12f64 };
    i_cl = 0 as libc::c_int as vl_index;
    while i_cl < numClusters as libc::c_int as libc::c_longlong {
        *priors.offset(i_cl as isize) /= sum;
        i_cl += 1;
    }
    if (*self_0).verbosity > 1 as libc::c_int {
        (Some(
            ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                .expect("non-null function pointer"),
        ))
            .expect(
                "non-null function pointer",
            )(
            b"gmm: em: maximization step completed in %.2f s\n\0" as *const u8
                as *const libc::c_char,
            vl_get_cpu_time() - time,
        );
    }
    vl_free(oldMeans as *mut libc::c_void);
}
unsafe extern "C" fn _vl_gmm_maximization_f(
    mut self_0: *mut VlGMM,
    mut posteriors: *mut libc::c_float,
    mut priors: *mut libc::c_float,
    mut covariances: *mut libc::c_float,
    mut means: *mut libc::c_float,
    mut data: *const libc::c_float,
    mut numData: vl_size,
) {
    let mut numClusters: vl_size = (*self_0).numClusters;
    let mut i_d: vl_index = 0;
    let mut i_cl: vl_index = 0;
    let mut dim: vl_size = 0;
    let mut oldMeans: *mut libc::c_float = 0 as *mut libc::c_float;
    let mut time: libc::c_double = 0 as libc::c_int as libc::c_double;
    if (*self_0).verbosity > 1 as libc::c_int {
        (Some(
            ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                .expect("non-null function pointer"),
        ))
            .expect(
                "non-null function pointer",
            )(
            b"gmm: em: entering maximization step\n\0" as *const u8
                as *const libc::c_char,
        );
        time = vl_get_cpu_time();
    }
    oldMeans = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).dimension)
            .wrapping_mul(numClusters) as size_t,
    ) as *mut libc::c_float;
    memcpy(
        oldMeans as *mut libc::c_void,
        means as *const libc::c_void,
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).dimension)
            .wrapping_mul(numClusters) as libc::c_ulong,
    );
    memset(
        priors as *mut libc::c_void,
        0 as libc::c_int,
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numClusters) as libc::c_ulong,
    );
    memset(
        means as *mut libc::c_void,
        0 as libc::c_int,
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).dimension)
            .wrapping_mul(numClusters) as libc::c_ulong,
    );
    memset(
        covariances as *mut libc::c_void,
        0 as libc::c_int,
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).dimension)
            .wrapping_mul(numClusters) as libc::c_ulong,
    );
    let mut clusterPosteriorSum_: *mut libc::c_float = 0 as *mut libc::c_float;
    let mut means_: *mut libc::c_float = 0 as *mut libc::c_float;
    let mut covariances_: *mut libc::c_float = 0 as *mut libc::c_float;
    clusterPosteriorSum_ = vl_calloc(
        ::core::mem::size_of::<libc::c_float>() as libc::c_ulong,
        numClusters as size_t,
    ) as *mut libc::c_float;
    means_ = vl_calloc(
        ::core::mem::size_of::<libc::c_float>() as libc::c_ulong,
        ((*self_0).dimension).wrapping_mul(numClusters) as size_t,
    ) as *mut libc::c_float;
    covariances_ = vl_calloc(
        ::core::mem::size_of::<libc::c_float>() as libc::c_ulong,
        ((*self_0).dimension).wrapping_mul(numClusters) as size_t,
    ) as *mut libc::c_float;
    i_d = 0 as libc::c_int as vl_index;
    while i_d < numData as libc::c_int as libc::c_longlong {
        i_cl = 0 as libc::c_int as vl_index;
        while i_cl < numClusters as libc::c_int as libc::c_longlong {
            let mut p: libc::c_float = *posteriors
                .offset(
                    (i_cl as libc::c_ulonglong)
                        .wrapping_add(
                            (i_d as libc::c_ulonglong)
                                .wrapping_mul((*self_0).numClusters),
                        ) as isize,
                );
            let mut calculated: vl_bool = 0 as libc::c_int;
            if !((p as libc::c_double) < 1e-2f64 / numClusters as libc::c_double) {
                *clusterPosteriorSum_.offset(i_cl as isize) += p;
                if vl_get_simd_enabled() != 0 && vl_cpu_has_avx() != 0 {
                    _vl_weighted_mean_avx_f(
                        (*self_0).dimension,
                        means_
                            .offset(
                                (i_cl as libc::c_ulonglong)
                                    .wrapping_mul((*self_0).dimension) as isize,
                            ),
                        data
                            .offset(
                                (i_d as libc::c_ulonglong).wrapping_mul((*self_0).dimension)
                                    as isize,
                            ),
                        p,
                    );
                    _vl_weighted_sigma_avx_f(
                        (*self_0).dimension,
                        covariances_
                            .offset(
                                (i_cl as libc::c_ulonglong)
                                    .wrapping_mul((*self_0).dimension) as isize,
                            ),
                        data
                            .offset(
                                (i_d as libc::c_ulonglong).wrapping_mul((*self_0).dimension)
                                    as isize,
                            ),
                        oldMeans
                            .offset(
                                (i_cl as libc::c_ulonglong)
                                    .wrapping_mul((*self_0).dimension) as isize,
                            ),
                        p,
                    );
                    calculated = 1 as libc::c_int;
                }
                if vl_get_simd_enabled() != 0 && vl_cpu_has_sse2() != 0
                    && calculated == 0
                {
                    _vl_weighted_mean_sse2_f(
                        (*self_0).dimension,
                        means_
                            .offset(
                                (i_cl as libc::c_ulonglong)
                                    .wrapping_mul((*self_0).dimension) as isize,
                            ),
                        data
                            .offset(
                                (i_d as libc::c_ulonglong).wrapping_mul((*self_0).dimension)
                                    as isize,
                            ),
                        p,
                    );
                    _vl_weighted_sigma_sse2_f(
                        (*self_0).dimension,
                        covariances_
                            .offset(
                                (i_cl as libc::c_ulonglong)
                                    .wrapping_mul((*self_0).dimension) as isize,
                            ),
                        data
                            .offset(
                                (i_d as libc::c_ulonglong).wrapping_mul((*self_0).dimension)
                                    as isize,
                            ),
                        oldMeans
                            .offset(
                                (i_cl as libc::c_ulonglong)
                                    .wrapping_mul((*self_0).dimension) as isize,
                            ),
                        p,
                    );
                    calculated = 1 as libc::c_int;
                }
                if calculated == 0 {
                    dim = 0 as libc::c_int as vl_size;
                    while dim < (*self_0).dimension {
                        let mut x: libc::c_float = *data
                            .offset(
                                (i_d as libc::c_ulonglong)
                                    .wrapping_mul((*self_0).dimension)
                                    .wrapping_add(dim) as isize,
                            );
                        let mut mu: libc::c_float = *oldMeans
                            .offset(
                                (i_cl as libc::c_ulonglong)
                                    .wrapping_mul((*self_0).dimension)
                                    .wrapping_add(dim) as isize,
                            );
                        let mut diff: libc::c_float = x - mu;
                        *means_
                            .offset(
                                (i_cl as libc::c_ulonglong)
                                    .wrapping_mul((*self_0).dimension)
                                    .wrapping_add(dim) as isize,
                            ) += p * x;
                        *covariances_
                            .offset(
                                (i_cl as libc::c_ulonglong)
                                    .wrapping_mul((*self_0).dimension)
                                    .wrapping_add(dim) as isize,
                            ) += p * (diff * diff);
                        dim = dim.wrapping_add(1);
                    }
                }
            }
            i_cl += 1;
        }
        i_d += 1;
    }
    i_cl = 0 as libc::c_int as vl_index;
    while i_cl < numClusters as libc::c_int as libc::c_longlong {
        *priors.offset(i_cl as isize) += *clusterPosteriorSum_.offset(i_cl as isize);
        dim = 0 as libc::c_int as vl_size;
        while dim < (*self_0).dimension {
            *means
                .offset(
                    (i_cl as libc::c_ulonglong)
                        .wrapping_mul((*self_0).dimension)
                        .wrapping_add(dim) as isize,
                )
                += *means_
                    .offset(
                        (i_cl as libc::c_ulonglong)
                            .wrapping_mul((*self_0).dimension)
                            .wrapping_add(dim) as isize,
                    );
            *covariances
                .offset(
                    (i_cl as libc::c_ulonglong)
                        .wrapping_mul((*self_0).dimension)
                        .wrapping_add(dim) as isize,
                )
                += *covariances_
                    .offset(
                        (i_cl as libc::c_ulonglong)
                            .wrapping_mul((*self_0).dimension)
                            .wrapping_add(dim) as isize,
                    );
            dim = dim.wrapping_add(1);
        }
        i_cl += 1;
    }
    vl_free(means_ as *mut libc::c_void);
    vl_free(covariances_ as *mut libc::c_void);
    vl_free(clusterPosteriorSum_ as *mut libc::c_void);
    i_cl = 0 as libc::c_int as vl_index;
    while i_cl < numClusters as libc::c_int as libc::c_longlong {
        let mut mass: libc::c_float = *priors.offset(i_cl as isize);
        if mass as libc::c_double >= 1e-6f64 / numClusters as libc::c_double {
            dim = 0 as libc::c_int as vl_size;
            while dim < (*self_0).dimension {
                *means
                    .offset(
                        (i_cl as libc::c_ulonglong)
                            .wrapping_mul((*self_0).dimension)
                            .wrapping_add(dim) as isize,
                    ) /= mass;
                *covariances
                    .offset(
                        (i_cl as libc::c_ulonglong)
                            .wrapping_mul((*self_0).dimension)
                            .wrapping_add(dim) as isize,
                    ) /= mass;
                dim = dim.wrapping_add(1);
            }
        }
        i_cl += 1;
    }
    i_cl = 0 as libc::c_int as vl_index;
    while i_cl < numClusters as libc::c_int as libc::c_longlong {
        let mut mass_0: libc::c_float = *priors.offset(i_cl as isize);
        if mass_0 as libc::c_double >= 1e-6f64 / numClusters as libc::c_double {
            dim = 0 as libc::c_int as vl_size;
            while dim < (*self_0).dimension {
                let mut mu_0: libc::c_float = *means
                    .offset(
                        (i_cl as libc::c_ulonglong)
                            .wrapping_mul((*self_0).dimension)
                            .wrapping_add(dim) as isize,
                    );
                let mut oldMu: libc::c_float = *oldMeans
                    .offset(
                        (i_cl as libc::c_ulonglong)
                            .wrapping_mul((*self_0).dimension)
                            .wrapping_add(dim) as isize,
                    );
                let mut diff_0: libc::c_float = mu_0 - oldMu;
                *covariances
                    .offset(
                        (i_cl as libc::c_ulonglong)
                            .wrapping_mul((*self_0).dimension)
                            .wrapping_add(dim) as isize,
                    ) -= diff_0 * diff_0;
                dim = dim.wrapping_add(1);
            }
        }
        i_cl += 1;
    }
    _vl_gmm_apply_bounds_f(self_0);
    let mut sum: libc::c_float = 0 as libc::c_int as libc::c_float;
    i_cl = 0 as libc::c_int as vl_index;
    while i_cl < numClusters as libc::c_int as libc::c_longlong {
        sum += *priors.offset(i_cl as isize);
        i_cl += 1;
    }
    sum = (if sum as libc::c_double > 1e-12f64 {
        sum as libc::c_double
    } else {
        1e-12f64
    }) as libc::c_float;
    i_cl = 0 as libc::c_int as vl_index;
    while i_cl < numClusters as libc::c_int as libc::c_longlong {
        *priors.offset(i_cl as isize) /= sum;
        i_cl += 1;
    }
    if (*self_0).verbosity > 1 as libc::c_int {
        (Some(
            ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                .expect("non-null function pointer"),
        ))
            .expect(
                "non-null function pointer",
            )(
            b"gmm: em: maximization step completed in %.2f s\n\0" as *const u8
                as *const libc::c_char,
            vl_get_cpu_time() - time,
        );
    }
    vl_free(oldMeans as *mut libc::c_void);
}
unsafe extern "C" fn _vl_gmm_em_d(
    mut self_0: *mut VlGMM,
    mut data: *const libc::c_double,
    mut numData: vl_size,
) -> libc::c_double {
    let mut iteration: vl_size = 0;
    let mut restarted: vl_size = 0;
    let mut previousLL: libc::c_double = -vl_infinity_d.value;
    let mut LL: libc::c_double = -vl_infinity_d.value;
    let mut time: libc::c_double = 0 as libc::c_int as libc::c_double;
    _vl_gmm_prepare_for_data(self_0, numData);
    _vl_gmm_apply_bounds_d(self_0);
    iteration = 0 as libc::c_int as vl_size;
    loop {
        let mut eps: libc::c_double = 0.;
        if (*self_0).verbosity > 1 as libc::c_int {
            (Some(
                ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                    .expect("non-null function pointer"),
            ))
                .expect(
                    "non-null function pointer",
                )(
                b"gmm: em: entering expectation step\n\0" as *const u8
                    as *const libc::c_char,
            );
            time = vl_get_cpu_time();
        }
        LL = vl_get_gmm_data_posteriors_d(
            (*self_0).posteriors as *mut libc::c_double,
            (*self_0).numClusters,
            numData,
            (*self_0).priors as *const libc::c_double,
            (*self_0).means as *const libc::c_double,
            (*self_0).dimension,
            (*self_0).covariances as *const libc::c_double,
            data,
        );
        if (*self_0).verbosity > 1 as libc::c_int {
            (Some(
                ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                    .expect("non-null function pointer"),
            ))
                .expect(
                    "non-null function pointer",
                )(
                b"gmm: em: expectation step completed in %.2f s\n\0" as *const u8
                    as *const libc::c_char,
                vl_get_cpu_time() - time,
            );
        }
        if (*self_0).verbosity != 0 {
            (Some(
                ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                    .expect("non-null function pointer"),
            ))
                .expect(
                    "non-null function pointer",
                )(
                b"gmm: em: iteration %d: loglikelihood = %f (variation = %f)\n\0"
                    as *const u8 as *const libc::c_char,
                iteration,
                LL,
                LL - previousLL,
            );
        }
        if iteration >= (*self_0).maxNumIterations {
            if (*self_0).verbosity != 0 {
                (Some(
                    ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                        .expect("non-null function pointer"),
                ))
                    .expect(
                        "non-null function pointer",
                    )(
                    b"gmm: em: terminating because the maximum number of iterations (%d) has been reached.\n\0"
                        as *const u8 as *const libc::c_char,
                    (*self_0).maxNumIterations,
                );
            }
            break;
        } else {
            eps = vl_abs_d((LL - previousLL) / LL);
            if iteration > 0 as libc::c_int as libc::c_ulonglong && eps < 0.00001f64 {
                if (*self_0).verbosity != 0 {
                    (Some(
                        ((vl_get_printf_func
                            as unsafe extern "C" fn() -> printf_func_t)())
                            .expect("non-null function pointer"),
                    ))
                        .expect(
                            "non-null function pointer",
                        )(
                        b"gmm: em: terminating because the algorithm fully converged (log-likelihood variation = %f).\n\0"
                            as *const u8 as *const libc::c_char,
                        eps,
                    );
                }
                break;
            } else {
                previousLL = LL;
                if iteration > 1 as libc::c_int as libc::c_ulonglong {
                    restarted = _vl_gmm_restart_empty_modes_d(self_0, data);
                    if (restarted > 0 as libc::c_int as libc::c_ulonglong) as libc::c_int
                        & ((*self_0).verbosity > 0 as libc::c_int) as libc::c_int != 0
                    {
                        (Some(
                            ((vl_get_printf_func
                                as unsafe extern "C" fn() -> printf_func_t)())
                                .expect("non-null function pointer"),
                        ))
                            .expect(
                                "non-null function pointer",
                            )(
                            b"gmm: em: %d Gaussian modes restarted because they had become empty.\n\0"
                                as *const u8 as *const libc::c_char,
                            restarted,
                        );
                    }
                }
                _vl_gmm_maximization_d(
                    self_0,
                    (*self_0).posteriors as *mut libc::c_double,
                    (*self_0).priors as *mut libc::c_double,
                    (*self_0).covariances as *mut libc::c_double,
                    (*self_0).means as *mut libc::c_double,
                    data,
                    numData,
                );
                iteration = iteration.wrapping_add(1);
            }
        }
    }
    return LL;
}
unsafe extern "C" fn _vl_gmm_em_f(
    mut self_0: *mut VlGMM,
    mut data: *const libc::c_float,
    mut numData: vl_size,
) -> libc::c_double {
    let mut iteration: vl_size = 0;
    let mut restarted: vl_size = 0;
    let mut previousLL: libc::c_double = -vl_infinity_d.value as libc::c_float
        as libc::c_double;
    let mut LL: libc::c_double = -vl_infinity_d.value as libc::c_float as libc::c_double;
    let mut time: libc::c_double = 0 as libc::c_int as libc::c_double;
    _vl_gmm_prepare_for_data(self_0, numData);
    _vl_gmm_apply_bounds_f(self_0);
    iteration = 0 as libc::c_int as vl_size;
    loop {
        let mut eps: libc::c_double = 0.;
        if (*self_0).verbosity > 1 as libc::c_int {
            (Some(
                ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                    .expect("non-null function pointer"),
            ))
                .expect(
                    "non-null function pointer",
                )(
                b"gmm: em: entering expectation step\n\0" as *const u8
                    as *const libc::c_char,
            );
            time = vl_get_cpu_time();
        }
        LL = vl_get_gmm_data_posteriors_f(
            (*self_0).posteriors as *mut libc::c_float,
            (*self_0).numClusters,
            numData,
            (*self_0).priors as *const libc::c_float,
            (*self_0).means as *const libc::c_float,
            (*self_0).dimension,
            (*self_0).covariances as *const libc::c_float,
            data,
        );
        if (*self_0).verbosity > 1 as libc::c_int {
            (Some(
                ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                    .expect("non-null function pointer"),
            ))
                .expect(
                    "non-null function pointer",
                )(
                b"gmm: em: expectation step completed in %.2f s\n\0" as *const u8
                    as *const libc::c_char,
                vl_get_cpu_time() - time,
            );
        }
        if (*self_0).verbosity != 0 {
            (Some(
                ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                    .expect("non-null function pointer"),
            ))
                .expect(
                    "non-null function pointer",
                )(
                b"gmm: em: iteration %d: loglikelihood = %f (variation = %f)\n\0"
                    as *const u8 as *const libc::c_char,
                iteration,
                LL,
                LL - previousLL,
            );
        }
        if iteration >= (*self_0).maxNumIterations {
            if (*self_0).verbosity != 0 {
                (Some(
                    ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                        .expect("non-null function pointer"),
                ))
                    .expect(
                        "non-null function pointer",
                    )(
                    b"gmm: em: terminating because the maximum number of iterations (%d) has been reached.\n\0"
                        as *const u8 as *const libc::c_char,
                    (*self_0).maxNumIterations,
                );
            }
            break;
        } else {
            eps = vl_abs_d((LL - previousLL) / LL);
            if iteration > 0 as libc::c_int as libc::c_ulonglong && eps < 0.00001f64 {
                if (*self_0).verbosity != 0 {
                    (Some(
                        ((vl_get_printf_func
                            as unsafe extern "C" fn() -> printf_func_t)())
                            .expect("non-null function pointer"),
                    ))
                        .expect(
                            "non-null function pointer",
                        )(
                        b"gmm: em: terminating because the algorithm fully converged (log-likelihood variation = %f).\n\0"
                            as *const u8 as *const libc::c_char,
                        eps,
                    );
                }
                break;
            } else {
                previousLL = LL;
                if iteration > 1 as libc::c_int as libc::c_ulonglong {
                    restarted = _vl_gmm_restart_empty_modes_f(self_0, data);
                    if (restarted > 0 as libc::c_int as libc::c_ulonglong) as libc::c_int
                        & ((*self_0).verbosity > 0 as libc::c_int) as libc::c_int != 0
                    {
                        (Some(
                            ((vl_get_printf_func
                                as unsafe extern "C" fn() -> printf_func_t)())
                                .expect("non-null function pointer"),
                        ))
                            .expect(
                                "non-null function pointer",
                            )(
                            b"gmm: em: %d Gaussian modes restarted because they had become empty.\n\0"
                                as *const u8 as *const libc::c_char,
                            restarted,
                        );
                    }
                }
                _vl_gmm_maximization_f(
                    self_0,
                    (*self_0).posteriors as *mut libc::c_float,
                    (*self_0).priors as *mut libc::c_float,
                    (*self_0).covariances as *mut libc::c_float,
                    (*self_0).means as *mut libc::c_float,
                    data,
                    numData,
                );
                iteration = iteration.wrapping_add(1);
            }
        }
    }
    return LL;
}
unsafe extern "C" fn _vl_gmm_init_with_kmeans_d(
    mut self_0: *mut VlGMM,
    mut data: *const libc::c_double,
    mut numData: vl_size,
    mut kmeansInit: *mut VlKMeans,
) {
    let mut i_d: vl_size = 0;
    let mut assignments: *mut vl_uint32 = vl_malloc(
        (::core::mem::size_of::<vl_uint32>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut vl_uint32;
    _vl_gmm_prepare_for_data(self_0, numData);
    memset(
        (*self_0).means,
        0 as libc::c_int,
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).numClusters)
            .wrapping_mul((*self_0).dimension) as libc::c_ulong,
    );
    memset(
        (*self_0).priors,
        0 as libc::c_int,
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).numClusters) as libc::c_ulong,
    );
    memset(
        (*self_0).covariances,
        0 as libc::c_int,
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).numClusters)
            .wrapping_mul((*self_0).dimension) as libc::c_ulong,
    );
    memset(
        (*self_0).posteriors,
        0 as libc::c_int,
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).numClusters)
            .wrapping_mul(numData) as libc::c_ulong,
    );
    if !kmeansInit.is_null() {
        vl_gmm_set_kmeans_init_object(self_0, kmeansInit);
    }
    if ((*self_0).kmeansInit).is_null() {
        let mut ncomparisons: vl_size = if numData
            .wrapping_div(4 as libc::c_int as libc::c_ulonglong)
            > 10 as libc::c_int as libc::c_ulonglong
        {
            numData.wrapping_div(4 as libc::c_int as libc::c_ulonglong)
        } else {
            10 as libc::c_int as libc::c_ulonglong
        };
        let mut niter: vl_size = 5 as libc::c_int as vl_size;
        let mut ntrees: vl_size = 1 as libc::c_int as vl_size;
        let mut nrepetitions: vl_size = 1 as libc::c_int as vl_size;
        let mut algorithm: VlKMeansAlgorithm = VlKMeansANN;
        let mut initialization: VlKMeansInitialization = VlKMeansRandomSelection;
        let mut kmeansInitDefault: *mut VlKMeans = vl_kmeans_new(
            (*self_0).dataType,
            VlDistanceL2,
        );
        vl_kmeans_set_initialization(kmeansInitDefault, initialization);
        vl_kmeans_set_max_num_iterations(kmeansInitDefault, niter);
        vl_kmeans_set_max_num_comparisons(kmeansInitDefault, ncomparisons);
        vl_kmeans_set_num_trees(kmeansInitDefault, ntrees);
        vl_kmeans_set_algorithm(kmeansInitDefault, algorithm);
        vl_kmeans_set_num_repetitions(kmeansInitDefault, nrepetitions);
        vl_kmeans_set_verbosity(kmeansInitDefault, (*self_0).verbosity);
        (*self_0).kmeansInit = kmeansInitDefault;
        (*self_0).kmeansInitIsOwner = 1 as libc::c_int;
    }
    vl_kmeans_cluster(
        (*self_0).kmeansInit,
        data as *const libc::c_void,
        (*self_0).dimension,
        numData,
        (*self_0).numClusters,
    );
    vl_kmeans_quantize(
        (*self_0).kmeansInit,
        assignments,
        0 as *mut libc::c_void,
        data as *const libc::c_void,
        numData,
    );
    i_d = 0 as libc::c_int as vl_size;
    while i_d < numData {
        *((*self_0).posteriors as *mut libc::c_double)
            .offset(
                (*assignments.offset(i_d as isize) as libc::c_ulonglong)
                    .wrapping_add(i_d.wrapping_mul((*self_0).numClusters)) as isize,
            ) = 1.0f64;
        i_d = i_d.wrapping_add(1);
    }
    _vl_gmm_maximization_d(
        self_0,
        (*self_0).posteriors as *mut libc::c_double,
        (*self_0).priors as *mut libc::c_double,
        (*self_0).covariances as *mut libc::c_double,
        (*self_0).means as *mut libc::c_double,
        data,
        numData,
    );
    vl_free(assignments as *mut libc::c_void);
}
unsafe extern "C" fn _vl_gmm_init_with_kmeans_f(
    mut self_0: *mut VlGMM,
    mut data: *const libc::c_float,
    mut numData: vl_size,
    mut kmeansInit: *mut VlKMeans,
) {
    let mut i_d: vl_size = 0;
    let mut assignments: *mut vl_uint32 = vl_malloc(
        (::core::mem::size_of::<vl_uint32>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut vl_uint32;
    _vl_gmm_prepare_for_data(self_0, numData);
    memset(
        (*self_0).means,
        0 as libc::c_int,
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).numClusters)
            .wrapping_mul((*self_0).dimension) as libc::c_ulong,
    );
    memset(
        (*self_0).priors,
        0 as libc::c_int,
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).numClusters) as libc::c_ulong,
    );
    memset(
        (*self_0).covariances,
        0 as libc::c_int,
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).numClusters)
            .wrapping_mul((*self_0).dimension) as libc::c_ulong,
    );
    memset(
        (*self_0).posteriors,
        0 as libc::c_int,
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).numClusters)
            .wrapping_mul(numData) as libc::c_ulong,
    );
    if !kmeansInit.is_null() {
        vl_gmm_set_kmeans_init_object(self_0, kmeansInit);
    }
    if ((*self_0).kmeansInit).is_null() {
        let mut ncomparisons: vl_size = if numData
            .wrapping_div(4 as libc::c_int as libc::c_ulonglong)
            > 10 as libc::c_int as libc::c_ulonglong
        {
            numData.wrapping_div(4 as libc::c_int as libc::c_ulonglong)
        } else {
            10 as libc::c_int as libc::c_ulonglong
        };
        let mut niter: vl_size = 5 as libc::c_int as vl_size;
        let mut ntrees: vl_size = 1 as libc::c_int as vl_size;
        let mut nrepetitions: vl_size = 1 as libc::c_int as vl_size;
        let mut algorithm: VlKMeansAlgorithm = VlKMeansANN;
        let mut initialization: VlKMeansInitialization = VlKMeansRandomSelection;
        let mut kmeansInitDefault: *mut VlKMeans = vl_kmeans_new(
            (*self_0).dataType,
            VlDistanceL2,
        );
        vl_kmeans_set_initialization(kmeansInitDefault, initialization);
        vl_kmeans_set_max_num_iterations(kmeansInitDefault, niter);
        vl_kmeans_set_max_num_comparisons(kmeansInitDefault, ncomparisons);
        vl_kmeans_set_num_trees(kmeansInitDefault, ntrees);
        vl_kmeans_set_algorithm(kmeansInitDefault, algorithm);
        vl_kmeans_set_num_repetitions(kmeansInitDefault, nrepetitions);
        vl_kmeans_set_verbosity(kmeansInitDefault, (*self_0).verbosity);
        (*self_0).kmeansInit = kmeansInitDefault;
        (*self_0).kmeansInitIsOwner = 1 as libc::c_int;
    }
    vl_kmeans_cluster(
        (*self_0).kmeansInit,
        data as *const libc::c_void,
        (*self_0).dimension,
        numData,
        (*self_0).numClusters,
    );
    vl_kmeans_quantize(
        (*self_0).kmeansInit,
        assignments,
        0 as *mut libc::c_void,
        data as *const libc::c_void,
        numData,
    );
    i_d = 0 as libc::c_int as vl_size;
    while i_d < numData {
        *((*self_0).posteriors as *mut libc::c_float)
            .offset(
                (*assignments.offset(i_d as isize) as libc::c_ulonglong)
                    .wrapping_add(i_d.wrapping_mul((*self_0).numClusters)) as isize,
            ) = 1.0f64 as libc::c_float;
        i_d = i_d.wrapping_add(1);
    }
    _vl_gmm_maximization_f(
        self_0,
        (*self_0).posteriors as *mut libc::c_float,
        (*self_0).priors as *mut libc::c_float,
        (*self_0).covariances as *mut libc::c_float,
        (*self_0).means as *mut libc::c_float,
        data,
        numData,
    );
    vl_free(assignments as *mut libc::c_void);
}
unsafe extern "C" fn _vl_gmm_compute_init_sigma_d(
    mut self_0: *mut VlGMM,
    mut data: *const libc::c_double,
    mut initSigma: *mut libc::c_double,
    mut dimension: vl_size,
    mut numData: vl_size,
) {
    let mut dim: vl_size = 0;
    let mut i: vl_uindex = 0;
    let mut dataMean: *mut libc::c_double = 0 as *mut libc::c_double;
    memset(
        initSigma as *mut libc::c_void,
        0 as libc::c_int,
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(dimension) as libc::c_ulong,
    );
    if numData <= 1 as libc::c_int as libc::c_ulonglong {
        return;
    }
    dataMean = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(dimension) as size_t,
    ) as *mut libc::c_double;
    memset(
        dataMean as *mut libc::c_void,
        0 as libc::c_int,
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(dimension) as libc::c_ulong,
    );
    dim = 0 as libc::c_int as vl_size;
    while dim < dimension {
        i = 0 as libc::c_int as vl_uindex;
        while i < numData {
            *dataMean.offset(dim as isize)
                += *data.offset(i.wrapping_mul(dimension).wrapping_add(dim) as isize);
            i = i.wrapping_add(1);
        }
        *dataMean.offset(dim as isize) /= numData as libc::c_double;
        dim = dim.wrapping_add(1);
    }
    dim = 0 as libc::c_int as vl_size;
    while dim < dimension {
        i = 0 as libc::c_int as vl_uindex;
        while i < numData {
            let mut diff: libc::c_double = *data
                .offset(i.wrapping_mul((*self_0).dimension).wrapping_add(dim) as isize)
                - *dataMean.offset(dim as isize);
            *initSigma.offset(dim as isize) += diff * diff;
            i = i.wrapping_add(1);
        }
        *initSigma.offset(dim as isize)
            /= numData.wrapping_sub(1 as libc::c_int as libc::c_ulonglong)
                as libc::c_double;
        dim = dim.wrapping_add(1);
    }
    vl_free(dataMean as *mut libc::c_void);
}
unsafe extern "C" fn _vl_gmm_compute_init_sigma_f(
    mut self_0: *mut VlGMM,
    mut data: *const libc::c_float,
    mut initSigma: *mut libc::c_float,
    mut dimension: vl_size,
    mut numData: vl_size,
) {
    let mut dim: vl_size = 0;
    let mut i: vl_uindex = 0;
    let mut dataMean: *mut libc::c_float = 0 as *mut libc::c_float;
    memset(
        initSigma as *mut libc::c_void,
        0 as libc::c_int,
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(dimension) as libc::c_ulong,
    );
    if numData <= 1 as libc::c_int as libc::c_ulonglong {
        return;
    }
    dataMean = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(dimension) as size_t,
    ) as *mut libc::c_float;
    memset(
        dataMean as *mut libc::c_void,
        0 as libc::c_int,
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(dimension) as libc::c_ulong,
    );
    dim = 0 as libc::c_int as vl_size;
    while dim < dimension {
        i = 0 as libc::c_int as vl_uindex;
        while i < numData {
            *dataMean.offset(dim as isize)
                += *data.offset(i.wrapping_mul(dimension).wrapping_add(dim) as isize);
            i = i.wrapping_add(1);
        }
        *dataMean.offset(dim as isize) /= numData as libc::c_float;
        dim = dim.wrapping_add(1);
    }
    dim = 0 as libc::c_int as vl_size;
    while dim < dimension {
        i = 0 as libc::c_int as vl_uindex;
        while i < numData {
            let mut diff: libc::c_float = *data
                .offset(i.wrapping_mul((*self_0).dimension).wrapping_add(dim) as isize)
                - *dataMean.offset(dim as isize);
            *initSigma.offset(dim as isize) += diff * diff;
            i = i.wrapping_add(1);
        }
        *initSigma.offset(dim as isize)
            /= numData.wrapping_sub(1 as libc::c_int as libc::c_ulonglong)
                as libc::c_float;
        dim = dim.wrapping_add(1);
    }
    vl_free(dataMean as *mut libc::c_void);
}
unsafe extern "C" fn _vl_gmm_init_with_rand_data_d(
    mut self_0: *mut VlGMM,
    mut data: *const libc::c_double,
    mut numData: vl_size,
) {
    let mut i: vl_uindex = 0;
    let mut k: vl_uindex = 0;
    let mut dim: vl_uindex = 0;
    let mut kmeans: *mut VlKMeans = 0 as *mut VlKMeans;
    _vl_gmm_prepare_for_data(self_0, numData);
    i = 0 as libc::c_int as vl_uindex;
    while i < (*self_0).numClusters {
        *((*self_0).priors as *mut libc::c_double)
            .offset(i as isize) = 1.0f64 / (*self_0).numClusters as libc::c_double;
        i = i.wrapping_add(1);
    }
    _vl_gmm_compute_init_sigma_d(
        self_0,
        data,
        (*self_0).covariances as *mut libc::c_double,
        (*self_0).dimension,
        numData,
    );
    k = 1 as libc::c_int as vl_uindex;
    while k < (*self_0).numClusters {
        dim = 0 as libc::c_int as vl_uindex;
        while dim < (*self_0).dimension {
            *((*self_0).covariances as *mut libc::c_double)
                .offset(k.wrapping_mul((*self_0).dimension) as isize)
                .offset(
                    dim as isize,
                ) = *((*self_0).covariances as *mut libc::c_double).offset(dim as isize);
            dim = dim.wrapping_add(1);
        }
        k = k.wrapping_add(1);
    }
    kmeans = vl_kmeans_new((*self_0).dataType, VlDistanceL2);
    vl_kmeans_init_centers_plus_plus(
        kmeans,
        data as *const libc::c_void,
        (*self_0).dimension,
        numData,
        (*self_0).numClusters,
    );
    memcpy(
        (*self_0).means,
        vl_kmeans_get_centers(kmeans),
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).dimension)
            .wrapping_mul((*self_0).numClusters) as libc::c_ulong,
    );
    vl_kmeans_delete(kmeans);
}
unsafe extern "C" fn _vl_gmm_init_with_rand_data_f(
    mut self_0: *mut VlGMM,
    mut data: *const libc::c_float,
    mut numData: vl_size,
) {
    let mut i: vl_uindex = 0;
    let mut k: vl_uindex = 0;
    let mut dim: vl_uindex = 0;
    let mut kmeans: *mut VlKMeans = 0 as *mut VlKMeans;
    _vl_gmm_prepare_for_data(self_0, numData);
    i = 0 as libc::c_int as vl_uindex;
    while i < (*self_0).numClusters {
        *((*self_0).priors as *mut libc::c_float)
            .offset(
                i as isize,
            ) = (1.0f64 / (*self_0).numClusters as libc::c_double) as libc::c_float;
        i = i.wrapping_add(1);
    }
    _vl_gmm_compute_init_sigma_f(
        self_0,
        data,
        (*self_0).covariances as *mut libc::c_float,
        (*self_0).dimension,
        numData,
    );
    k = 1 as libc::c_int as vl_uindex;
    while k < (*self_0).numClusters {
        dim = 0 as libc::c_int as vl_uindex;
        while dim < (*self_0).dimension {
            *((*self_0).covariances as *mut libc::c_float)
                .offset(k.wrapping_mul((*self_0).dimension) as isize)
                .offset(
                    dim as isize,
                ) = *((*self_0).covariances as *mut libc::c_float).offset(dim as isize);
            dim = dim.wrapping_add(1);
        }
        k = k.wrapping_add(1);
    }
    kmeans = vl_kmeans_new((*self_0).dataType, VlDistanceL2);
    vl_kmeans_init_centers_plus_plus(
        kmeans,
        data as *const libc::c_void,
        (*self_0).dimension,
        numData,
        (*self_0).numClusters,
    );
    memcpy(
        (*self_0).means,
        vl_kmeans_get_centers(kmeans),
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).dimension)
            .wrapping_mul((*self_0).numClusters) as libc::c_ulong,
    );
    vl_kmeans_delete(kmeans);
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_new_copy(mut self_0: *const VlGMM) -> *mut VlGMM {
    let mut size: vl_size = vl_get_type_size((*self_0).dataType);
    let mut gmm: *mut VlGMM = vl_gmm_new(
        (*self_0).dataType,
        (*self_0).dimension,
        (*self_0).numClusters,
    );
    (*gmm).initialization = (*self_0).initialization;
    (*gmm).maxNumIterations = (*self_0).maxNumIterations;
    (*gmm).numRepetitions = (*self_0).numRepetitions;
    (*gmm).verbosity = (*self_0).verbosity;
    (*gmm).LL = (*self_0).LL;
    memcpy(
        (*gmm).means,
        (*self_0).means,
        size.wrapping_mul((*self_0).numClusters).wrapping_mul((*self_0).dimension)
            as libc::c_ulong,
    );
    memcpy(
        (*gmm).covariances,
        (*self_0).covariances,
        size.wrapping_mul((*self_0).numClusters).wrapping_mul((*self_0).dimension)
            as libc::c_ulong,
    );
    memcpy(
        (*gmm).priors,
        (*self_0).priors,
        size.wrapping_mul((*self_0).numClusters) as libc::c_ulong,
    );
    return gmm;
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_init_with_rand_data(
    mut self_0: *mut VlGMM,
    mut data: *const libc::c_void,
    mut numData: vl_size,
) {
    vl_gmm_reset(self_0);
    match (*self_0).dataType {
        1 => {
            _vl_gmm_init_with_rand_data_f(self_0, data as *const libc::c_float, numData);
        }
        2 => {
            _vl_gmm_init_with_rand_data_d(
                self_0,
                data as *const libc::c_double,
                numData,
            );
        }
        _ => {
            abort();
        }
    };
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_init_with_kmeans(
    mut self_0: *mut VlGMM,
    mut data: *const libc::c_void,
    mut numData: vl_size,
    mut kmeansInit: *mut VlKMeans,
) {
    vl_gmm_reset(self_0);
    match (*self_0).dataType {
        1 => {
            _vl_gmm_init_with_kmeans_f(
                self_0,
                data as *const libc::c_float,
                numData,
                kmeansInit,
            );
        }
        2 => {
            _vl_gmm_init_with_kmeans_d(
                self_0,
                data as *const libc::c_double,
                numData,
                kmeansInit,
            );
        }
        _ => {
            abort();
        }
    };
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_cluster(
    mut self_0: *mut VlGMM,
    mut data: *const libc::c_void,
    mut numData: vl_size,
) -> libc::c_double {
    let mut bestPriors: *mut libc::c_void = 0 as *mut libc::c_void;
    let mut bestMeans: *mut libc::c_void = 0 as *mut libc::c_void;
    let mut bestCovariances: *mut libc::c_void = 0 as *mut libc::c_void;
    let mut bestPosteriors: *mut libc::c_void = 0 as *mut libc::c_void;
    let mut size: vl_size = vl_get_type_size((*self_0).dataType);
    let mut bestLL: libc::c_double = -vl_infinity_d.value;
    let mut repetition: vl_uindex = 0;
    if (*self_0).numRepetitions >= 1 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"self->numRepetitions >=1\0" as *const u8 as *const libc::c_char,
            b"vl/gmm.c\0" as *const u8 as *const libc::c_char,
            1569 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 54],
                &[libc::c_char; 54],
            >(b"double vl_gmm_cluster(VlGMM *, const void *, vl_size)\0"))
                .as_ptr(),
        );
    }
    bestPriors = vl_malloc(size.wrapping_mul((*self_0).numClusters) as size_t);
    bestMeans = vl_malloc(
        size.wrapping_mul((*self_0).dimension).wrapping_mul((*self_0).numClusters)
            as size_t,
    );
    bestCovariances = vl_malloc(
        size.wrapping_mul((*self_0).dimension).wrapping_mul((*self_0).numClusters)
            as size_t,
    );
    bestPosteriors = vl_malloc(
        size.wrapping_mul((*self_0).numClusters).wrapping_mul(numData) as size_t,
    );
    repetition = 0 as libc::c_int as vl_uindex;
    while repetition < (*self_0).numRepetitions {
        let mut LL: libc::c_double = 0.;
        let mut timeRef: libc::c_double = 0.;
        if (*self_0).verbosity != 0 {
            (Some(
                ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                    .expect("non-null function pointer"),
            ))
                .expect(
                    "non-null function pointer",
                )(
                b"gmm: clustering: starting repetition %d of %d\n\0" as *const u8
                    as *const libc::c_char,
                repetition.wrapping_add(1 as libc::c_int as libc::c_ulonglong),
                (*self_0).numRepetitions,
            );
        }
        timeRef = vl_get_cpu_time();
        match (*self_0).initialization as libc::c_uint {
            0 => {
                vl_gmm_init_with_kmeans(self_0, data, numData, 0 as *mut VlKMeans);
            }
            1 => {
                vl_gmm_init_with_rand_data(self_0, data, numData);
            }
            2 => {}
            _ => {
                abort();
            }
        }
        if (*self_0).verbosity != 0 {
            (Some(
                ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                    .expect("non-null function pointer"),
            ))
                .expect(
                    "non-null function pointer",
                )(
                b"gmm: model initialized in %.2f s\n\0" as *const u8
                    as *const libc::c_char,
                vl_get_cpu_time() - timeRef,
            );
        }
        timeRef = vl_get_cpu_time();
        LL = vl_gmm_em(self_0, data, numData);
        if (*self_0).verbosity != 0 {
            (Some(
                ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                    .expect("non-null function pointer"),
            ))
                .expect(
                    "non-null function pointer",
                )(
                b"gmm: optimization terminated in %.2f s with loglikelihood %f\n\0"
                    as *const u8 as *const libc::c_char,
                vl_get_cpu_time() - timeRef,
                LL,
            );
        }
        if LL > bestLL || repetition == 0 as libc::c_int as libc::c_ulonglong {
            let mut temp: *mut libc::c_void = 0 as *mut libc::c_void;
            temp = bestPriors;
            bestPriors = (*self_0).priors;
            (*self_0).priors = temp;
            temp = bestMeans;
            bestMeans = (*self_0).means;
            (*self_0).means = temp;
            temp = bestCovariances;
            bestCovariances = (*self_0).covariances;
            (*self_0).covariances = temp;
            temp = bestPosteriors;
            bestPosteriors = (*self_0).posteriors;
            (*self_0).posteriors = temp;
            bestLL = LL;
        }
        repetition = repetition.wrapping_add(1);
    }
    vl_free((*self_0).priors);
    vl_free((*self_0).means);
    vl_free((*self_0).covariances);
    vl_free((*self_0).posteriors);
    (*self_0).priors = bestPriors;
    (*self_0).means = bestMeans;
    (*self_0).covariances = bestCovariances;
    (*self_0).posteriors = bestPosteriors;
    (*self_0).LL = bestLL;
    if (*self_0).verbosity != 0 {
        (Some(
            ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                .expect("non-null function pointer"),
        ))
            .expect(
                "non-null function pointer",
            )(
            b"gmm: all repetitions terminated with final loglikelihood %f\n\0"
                as *const u8 as *const libc::c_char,
            (*self_0).LL,
        );
    }
    return bestLL;
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_em(
    mut self_0: *mut VlGMM,
    mut data: *const libc::c_void,
    mut numData: vl_size,
) -> libc::c_double {
    match (*self_0).dataType {
        1 => return _vl_gmm_em_f(self_0, data as *const libc::c_float, numData),
        2 => return _vl_gmm_em_d(self_0, data as *const libc::c_double, numData),
        _ => {
            abort();
        }
    };
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_set_means(
    mut self_0: *mut VlGMM,
    mut means: *const libc::c_void,
) {
    memcpy(
        (*self_0).means,
        means,
        ((*self_0).dimension)
            .wrapping_mul((*self_0).numClusters)
            .wrapping_mul(vl_get_type_size((*self_0).dataType)) as libc::c_ulong,
    );
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_set_covariances(
    mut self_0: *mut VlGMM,
    mut covariances: *const libc::c_void,
) {
    memcpy(
        (*self_0).covariances,
        covariances,
        ((*self_0).dimension)
            .wrapping_mul((*self_0).numClusters)
            .wrapping_mul(vl_get_type_size((*self_0).dataType)) as libc::c_ulong,
    );
}
#[no_mangle]
pub unsafe extern "C" fn vl_gmm_set_priors(
    mut self_0: *mut VlGMM,
    mut priors: *const libc::c_void,
) {
    memcpy(
        (*self_0).priors,
        priors,
        ((*self_0).numClusters).wrapping_mul(vl_get_type_size((*self_0).dataType))
            as libc::c_ulong,
    );
}
