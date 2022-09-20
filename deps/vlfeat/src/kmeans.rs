use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn free(_: *mut libc::c_void);
    fn abort() -> !;
    fn __assert_fail(
        __assertion: *const libc::c_char,
        __file: *const libc::c_char,
        __line: libc::c_uint,
        __function: *const libc::c_char,
    ) -> !;
    fn vl_get_rand() -> *mut VlRand;
    fn vl_malloc(n: size_t) -> *mut libc::c_void;
    fn vl_calloc(n: size_t, size: size_t) -> *mut libc::c_void;
    fn vl_free(ptr: *mut libc::c_void);
    fn vl_get_printf_func() -> printf_func_t;
    fn vl_get_cpu_time() -> libc::c_double;
    fn sqrt(_: libc::c_double) -> libc::c_double;
    fn vl_rand_uint32(self_0: *mut VlRand) -> vl_uint32;
    fn sqrtf(_: libc::c_float) -> libc::c_float;
    fn vl_get_vector_comparison_function_f(
        type_0: VlVectorComparisonType,
    ) -> VlFloatVectorComparisonFunction;
    fn vl_get_vector_comparison_function_d(
        type_0: VlVectorComparisonType,
    ) -> VlDoubleVectorComparisonFunction;
    fn vl_eval_vector_comparison_on_all_pairs_f(
        result: *mut libc::c_float,
        dimension: vl_size,
        X: *const libc::c_float,
        numDataX: vl_size,
        Y: *const libc::c_float,
        numDataY: vl_size,
        function: VlFloatVectorComparisonFunction,
    );
    fn vl_eval_vector_comparison_on_all_pairs_d(
        result: *mut libc::c_double,
        dimension: vl_size,
        X: *const libc::c_double,
        numDataX: vl_size,
        Y: *const libc::c_double,
        numDataY: vl_size,
        function: VlDoubleVectorComparisonFunction,
    );
    fn vl_kdforest_new(
        dataType: vl_type,
        dimension: vl_size,
        numTrees: vl_size,
        normType: VlVectorComparisonType,
    ) -> *mut VlKDForest;
    fn vl_kdforest_new_searcher(kdforest: *mut VlKDForest) -> *mut VlKDForestSearcher;
    fn vl_kdforest_delete(self_0: *mut VlKDForest);
    fn vl_kdforest_build(
        self_0: *mut VlKDForest,
        numData: vl_size,
        data: *const libc::c_void,
    );
    fn vl_kdforestsearcher_query(
        self_0: *mut VlKDForestSearcher,
        neighbors: *mut VlKDForestNeighbor,
        numNeighbors: vl_size,
        query: *const libc::c_void,
    ) -> vl_size;
    fn vl_kdforest_set_max_num_comparisons(self_0: *mut VlKDForest, n: vl_size);
    fn vl_kdforest_set_thresholding_method(
        self_0: *mut VlKDForest,
        method: VlKDTreeThresholdingMethod,
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
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlRand {
    pub mt: [vl_uint32; 624],
    pub mti: vl_uint32,
}
pub type VlRand = _VlRand;
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
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlKDTreeNode {
    pub parent: vl_uindex,
    pub lowerChild: vl_index,
    pub upperChild: vl_index,
    pub splitDimension: libc::c_uint,
    pub splitThreshold: libc::c_double,
    pub lowerBound: libc::c_double,
    pub upperBound: libc::c_double,
}
pub type VlKDTreeNode = _VlKDTreeNode;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlKDTreeSplitDimension {
    pub dimension: libc::c_uint,
    pub mean: libc::c_double,
    pub variance: libc::c_double,
}
pub type VlKDTreeSplitDimension = _VlKDTreeSplitDimension;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlKDTreeDataIndexEntry {
    pub index: vl_index,
    pub value: libc::c_double,
}
pub type VlKDTreeDataIndexEntry = _VlKDTreeDataIndexEntry;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlKDForestSearchState {
    pub tree: *mut VlKDTree,
    pub nodeIndex: vl_uindex,
    pub distanceLowerBound: libc::c_double,
}
pub type VlKDTree = _VlKDTree;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlKDTree {
    pub nodes: *mut VlKDTreeNode,
    pub numUsedNodes: vl_size,
    pub numAllocatedNodes: vl_size,
    pub dataIndex: *mut VlKDTreeDataIndexEntry,
    pub depth: libc::c_uint,
}
pub type VlKDForestSearchState = _VlKDForestSearchState;
pub type _VlKDTreeThresholdingMethod = libc::c_uint;
pub const VL_KDTREE_MEAN: _VlKDTreeThresholdingMethod = 1;
pub const VL_KDTREE_MEDIAN: _VlKDTreeThresholdingMethod = 0;
pub type VlKDTreeThresholdingMethod = _VlKDTreeThresholdingMethod;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlKDForestNeighbor {
    pub distance: libc::c_double,
    pub index: vl_uindex,
}
pub type VlKDForestNeighbor = _VlKDForestNeighbor;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlKDForestSearcher {
    pub next: *mut _VlKDForestSearcher,
    pub previous: *mut _VlKDForestSearcher,
    pub searchIdBook: *mut vl_uindex,
    pub searchHeapArray: *mut VlKDForestSearchState,
    pub forest: *mut VlKDForest,
    pub searchNumComparisons: vl_size,
    pub searchNumRecursions: vl_size,
    pub searchNumSimplifications: vl_size,
    pub searchHeapNumNodes: vl_size,
    pub searchId: vl_uindex,
}
pub type VlKDForest = _VlKDForest;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlKDForest {
    pub dimension: vl_size,
    pub rand: *mut VlRand,
    pub dataType: vl_type,
    pub data: *const libc::c_void,
    pub numData: vl_size,
    pub distance: VlVectorComparisonType,
    pub distanceFunction: Option::<unsafe extern "C" fn() -> ()>,
    pub trees: *mut *mut VlKDTree,
    pub numTrees: vl_size,
    pub thresholdingMethod: VlKDTreeThresholdingMethod,
    pub splitHeapArray: [VlKDTreeSplitDimension; 5],
    pub splitHeapNumNodes: vl_size,
    pub splitHeapSize: vl_size,
    pub maxNumNodes: vl_size,
    pub searchMaxNumComparisons: vl_size,
    pub numSearchers: vl_size,
    pub headSearcher: *mut _VlKDForestSearcher,
}
pub type VlKDForestSearcher = _VlKDForestSearcher;
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
pub type VlKMeansSortWrapper = _VlKMeansSortWrapper;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _VlKMeansSortWrapper {
    pub permutation: *mut vl_uint32,
    pub data: *const libc::c_void,
    pub stride: vl_size,
}
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
unsafe extern "C" fn vl_rand_real1(mut self_0: *mut VlRand) -> libc::c_double {
    return vl_rand_uint32(self_0) as libc::c_double * (1.0f64 / 4294967295.0f64);
}
#[inline]
unsafe extern "C" fn vl_rand_uint64(mut self_0: *mut VlRand) -> vl_uint64 {
    let mut a: vl_uint64 = vl_rand_uint32(self_0) as vl_uint64;
    let mut b: vl_uint64 = vl_rand_uint32(self_0) as vl_uint64;
    return a << 32 as libc::c_int | b;
}
static mut vl_infinity_d: C2RustUnnamed = C2RustUnnamed {
    raw: 0x7ff0000000000000 as libc::c_ulonglong,
};
#[no_mangle]
pub unsafe extern "C" fn vl_kmeans_reset(mut self_0: *mut VlKMeans) {
    (*self_0).numCenters = 0 as libc::c_int as vl_size;
    (*self_0).dimension = 0 as libc::c_int as vl_size;
    if !((*self_0).centers).is_null() {
        vl_free((*self_0).centers);
    }
    if !((*self_0).centerDistances).is_null() {
        vl_free((*self_0).centerDistances);
    }
    (*self_0).centers = 0 as *mut libc::c_void;
    (*self_0).centerDistances = 0 as *mut libc::c_void;
}
#[no_mangle]
pub unsafe extern "C" fn vl_kmeans_new(
    mut dataType: vl_type,
    mut distance: VlVectorComparisonType,
) -> *mut VlKMeans {
    let mut self_0: *mut VlKMeans = vl_calloc(
        1 as libc::c_int as size_t,
        ::core::mem::size_of::<VlKMeans>() as libc::c_ulong,
    ) as *mut VlKMeans;
    (*self_0).algorithm = VlKMeansLloyd;
    (*self_0).distance = distance;
    (*self_0).dataType = dataType;
    (*self_0).verbosity = 0 as libc::c_int;
    (*self_0).maxNumIterations = 100 as libc::c_int as vl_size;
    (*self_0).minEnergyVariation = 1e-4f64;
    (*self_0).numRepetitions = 1 as libc::c_int as vl_size;
    (*self_0).centers = 0 as *mut libc::c_void;
    (*self_0).centerDistances = 0 as *mut libc::c_void;
    (*self_0).numTrees = 3 as libc::c_int as vl_size;
    (*self_0).maxNumComparisons = 100 as libc::c_int as vl_size;
    vl_kmeans_reset(self_0);
    return self_0;
}
#[no_mangle]
pub unsafe extern "C" fn vl_kmeans_new_copy(
    mut kmeans: *const VlKMeans,
) -> *mut VlKMeans {
    let mut self_0: *mut VlKMeans = vl_malloc(
        ::core::mem::size_of::<VlKMeans>() as libc::c_ulong,
    ) as *mut VlKMeans;
    (*self_0).algorithm = (*kmeans).algorithm;
    (*self_0).distance = (*kmeans).distance;
    (*self_0).dataType = (*kmeans).dataType;
    (*self_0).verbosity = (*kmeans).verbosity;
    (*self_0).maxNumIterations = (*kmeans).maxNumIterations;
    (*self_0).numRepetitions = (*kmeans).numRepetitions;
    (*self_0).dimension = (*kmeans).dimension;
    (*self_0).numCenters = (*kmeans).numCenters;
    (*self_0).centers = 0 as *mut libc::c_void;
    (*self_0).centerDistances = 0 as *mut libc::c_void;
    (*self_0).numTrees = (*kmeans).numTrees;
    (*self_0).maxNumComparisons = (*kmeans).maxNumComparisons;
    if !((*kmeans).centers).is_null() {
        let mut dataSize: vl_size = (vl_get_type_size((*self_0).dataType))
            .wrapping_mul((*self_0).dimension)
            .wrapping_mul((*self_0).numCenters);
        (*self_0).centers = vl_malloc(dataSize as size_t);
        memcpy((*self_0).centers, (*kmeans).centers, dataSize as libc::c_ulong);
    }
    if !((*kmeans).centerDistances).is_null() {
        let mut dataSize_0: vl_size = (vl_get_type_size((*self_0).dataType))
            .wrapping_mul((*self_0).numCenters)
            .wrapping_mul((*self_0).numCenters);
        (*self_0).centerDistances = vl_malloc(dataSize_0 as size_t);
        memcpy(
            (*self_0).centerDistances,
            (*kmeans).centerDistances,
            dataSize_0 as libc::c_ulong,
        );
    }
    return self_0;
}
#[no_mangle]
pub unsafe extern "C" fn vl_kmeans_delete(mut self_0: *mut VlKMeans) {
    vl_kmeans_reset(self_0);
    vl_free(self_0 as *mut libc::c_void);
}
#[inline]
unsafe extern "C" fn _vl_kmeans_swap(
    mut array: *mut vl_uindex,
    mut indexA: vl_uindex,
    mut indexB: vl_uindex,
) {
    let mut t: vl_uindex = *array.offset(indexA as isize);
    *array.offset(indexA as isize) = *array.offset(indexB as isize);
    *array.offset(indexB as isize) = t;
}
#[inline]
unsafe extern "C" fn _vl_kmeans_shuffle(
    mut array: *mut vl_uindex,
    mut size: vl_size,
    mut rand: *mut VlRand,
) {
    let mut n: vl_uindex = size;
    while n > 1 as libc::c_int as libc::c_ulonglong {
        let mut k: vl_uindex = vl_rand_uindex(rand, n);
        n = n.wrapping_sub(1);
        _vl_kmeans_swap(array, n, k);
    }
}
unsafe extern "C" fn _vl_kmeans_set_centers_f(
    mut self_0: *mut VlKMeans,
    mut centers: *const libc::c_float,
    mut dimension: vl_size,
    mut numCenters: vl_size,
) {
    (*self_0).dimension = dimension;
    (*self_0).numCenters = numCenters;
    (*self_0)
        .centers = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(dimension)
            .wrapping_mul(numCenters) as size_t,
    );
    memcpy(
        (*self_0).centers as *mut libc::c_float as *mut libc::c_void,
        centers as *const libc::c_void,
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(dimension)
            .wrapping_mul(numCenters) as libc::c_ulong,
    );
}
unsafe extern "C" fn _vl_kmeans_set_centers_d(
    mut self_0: *mut VlKMeans,
    mut centers: *const libc::c_double,
    mut dimension: vl_size,
    mut numCenters: vl_size,
) {
    (*self_0).dimension = dimension;
    (*self_0).numCenters = numCenters;
    (*self_0)
        .centers = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(dimension)
            .wrapping_mul(numCenters) as size_t,
    );
    memcpy(
        (*self_0).centers as *mut libc::c_double as *mut libc::c_void,
        centers as *const libc::c_void,
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(dimension)
            .wrapping_mul(numCenters) as libc::c_ulong,
    );
}
unsafe extern "C" fn _vl_kmeans_init_centers_with_rand_data_f(
    mut self_0: *mut VlKMeans,
    mut data: *const libc::c_float,
    mut dimension: vl_size,
    mut numData: vl_size,
    mut numCenters: vl_size,
) {
    let mut i: vl_uindex = 0;
    let mut j: vl_uindex = 0;
    let mut k: vl_uindex = 0;
    let mut rand: *mut VlRand = vl_get_rand();
    (*self_0).dimension = dimension;
    (*self_0).numCenters = numCenters;
    (*self_0)
        .centers = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(dimension)
            .wrapping_mul(numCenters) as size_t,
    );
    let mut perm: *mut vl_uindex = vl_malloc(
        (::core::mem::size_of::<vl_uindex>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut vl_uindex;
    let mut distFn: VlFloatVectorComparisonFunction = vl_get_vector_comparison_function_f(
        (*self_0).distance,
    );
    let mut distances: *mut libc::c_float = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numCenters) as size_t,
    ) as *mut libc::c_float;
    i = 0 as libc::c_int as vl_uindex;
    while i < numData {
        *perm.offset(i as isize) = i;
        i = i.wrapping_add(1);
    }
    _vl_kmeans_shuffle(perm, numData, rand);
    let mut current_block_14: u64;
    k = 0 as libc::c_int as vl_uindex;
    i = 0 as libc::c_int as vl_uindex;
    while k < numCenters {
        if numCenters.wrapping_sub(k) < numData.wrapping_sub(i) {
            let mut duplicateDetected: vl_bool = 0 as libc::c_int;
            vl_eval_vector_comparison_on_all_pairs_f(
                distances,
                dimension,
                data.offset(dimension.wrapping_mul(*perm.offset(i as isize)) as isize),
                1 as libc::c_int as vl_size,
                (*self_0).centers as *mut libc::c_float,
                k,
                distFn,
            );
            j = 0 as libc::c_int as vl_uindex;
            while j < k {
                duplicateDetected
                    |= (*distances.offset(j as isize)
                        == 0 as libc::c_int as libc::c_float) as libc::c_int;
                j = j.wrapping_add(1);
            }
            if duplicateDetected != 0 {
                current_block_14 = 10886091980245723256;
            } else {
                current_block_14 = 1054647088692577877;
            }
        } else {
            current_block_14 = 1054647088692577877;
        }
        match current_block_14 {
            1054647088692577877 => {
                memcpy(
                    ((*self_0).centers as *mut libc::c_float)
                        .offset(dimension.wrapping_mul(k) as isize) as *mut libc::c_void,
                    data
                        .offset(
                            dimension.wrapping_mul(*perm.offset(i as isize)) as isize,
                        ) as *const libc::c_void,
                    (::core::mem::size_of::<libc::c_float>() as libc::c_ulong
                        as libc::c_ulonglong)
                        .wrapping_mul(dimension) as libc::c_ulong,
                );
                k = k.wrapping_add(1);
            }
            _ => {}
        }
        i = i.wrapping_add(1);
    }
    vl_free(distances as *mut libc::c_void);
    vl_free(perm as *mut libc::c_void);
}
unsafe extern "C" fn _vl_kmeans_init_centers_with_rand_data_d(
    mut self_0: *mut VlKMeans,
    mut data: *const libc::c_double,
    mut dimension: vl_size,
    mut numData: vl_size,
    mut numCenters: vl_size,
) {
    let mut i: vl_uindex = 0;
    let mut j: vl_uindex = 0;
    let mut k: vl_uindex = 0;
    let mut rand: *mut VlRand = vl_get_rand();
    (*self_0).dimension = dimension;
    (*self_0).numCenters = numCenters;
    (*self_0)
        .centers = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(dimension)
            .wrapping_mul(numCenters) as size_t,
    );
    let mut perm: *mut vl_uindex = vl_malloc(
        (::core::mem::size_of::<vl_uindex>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut vl_uindex;
    let mut distFn: VlDoubleVectorComparisonFunction = vl_get_vector_comparison_function_d(
        (*self_0).distance,
    );
    let mut distances: *mut libc::c_double = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numCenters) as size_t,
    ) as *mut libc::c_double;
    i = 0 as libc::c_int as vl_uindex;
    while i < numData {
        *perm.offset(i as isize) = i;
        i = i.wrapping_add(1);
    }
    _vl_kmeans_shuffle(perm, numData, rand);
    let mut current_block_14: u64;
    k = 0 as libc::c_int as vl_uindex;
    i = 0 as libc::c_int as vl_uindex;
    while k < numCenters {
        if numCenters.wrapping_sub(k) < numData.wrapping_sub(i) {
            let mut duplicateDetected: vl_bool = 0 as libc::c_int;
            vl_eval_vector_comparison_on_all_pairs_d(
                distances,
                dimension,
                data.offset(dimension.wrapping_mul(*perm.offset(i as isize)) as isize),
                1 as libc::c_int as vl_size,
                (*self_0).centers as *mut libc::c_double,
                k,
                distFn,
            );
            j = 0 as libc::c_int as vl_uindex;
            while j < k {
                duplicateDetected
                    |= (*distances.offset(j as isize)
                        == 0 as libc::c_int as libc::c_double) as libc::c_int;
                j = j.wrapping_add(1);
            }
            if duplicateDetected != 0 {
                current_block_14 = 10886091980245723256;
            } else {
                current_block_14 = 1054647088692577877;
            }
        } else {
            current_block_14 = 1054647088692577877;
        }
        match current_block_14 {
            1054647088692577877 => {
                memcpy(
                    ((*self_0).centers as *mut libc::c_double)
                        .offset(dimension.wrapping_mul(k) as isize) as *mut libc::c_void,
                    data
                        .offset(
                            dimension.wrapping_mul(*perm.offset(i as isize)) as isize,
                        ) as *const libc::c_void,
                    (::core::mem::size_of::<libc::c_double>() as libc::c_ulong
                        as libc::c_ulonglong)
                        .wrapping_mul(dimension) as libc::c_ulong,
                );
                k = k.wrapping_add(1);
            }
            _ => {}
        }
        i = i.wrapping_add(1);
    }
    vl_free(distances as *mut libc::c_void);
    vl_free(perm as *mut libc::c_void);
}
unsafe extern "C" fn _vl_kmeans_init_centers_plus_plus_f(
    mut self_0: *mut VlKMeans,
    mut data: *const libc::c_float,
    mut dimension: vl_size,
    mut numData: vl_size,
    mut numCenters: vl_size,
) {
    let mut x: vl_uindex = 0;
    let mut c: vl_uindex = 0;
    let mut rand: *mut VlRand = vl_get_rand();
    let mut distances: *mut libc::c_float = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut libc::c_float;
    let mut minDistances: *mut libc::c_float = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut libc::c_float;
    let mut distFn: VlFloatVectorComparisonFunction = vl_get_vector_comparison_function_f(
        (*self_0).distance,
    );
    (*self_0).dimension = dimension;
    (*self_0).numCenters = numCenters;
    (*self_0)
        .centers = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(dimension)
            .wrapping_mul(numCenters) as size_t,
    );
    x = 0 as libc::c_int as vl_uindex;
    while x < numData {
        *minDistances.offset(x as isize) = vl_infinity_d.value as libc::c_float;
        x = x.wrapping_add(1);
    }
    x = vl_rand_uindex(rand, numData);
    c = 0 as libc::c_int as vl_uindex;
    loop {
        let mut energy: libc::c_float = 0 as libc::c_int as libc::c_float;
        let mut acc: libc::c_float = 0 as libc::c_int as libc::c_float;
        let mut thresh: libc::c_float = vl_rand_real1(rand) as libc::c_float;
        memcpy(
            ((*self_0).centers as *mut libc::c_float)
                .offset(c.wrapping_mul(dimension) as isize) as *mut libc::c_void,
            data.offset(x.wrapping_mul(dimension) as isize) as *const libc::c_void,
            (::core::mem::size_of::<libc::c_float>() as libc::c_ulong
                as libc::c_ulonglong)
                .wrapping_mul(dimension) as libc::c_ulong,
        );
        c = c.wrapping_add(1);
        if c == numCenters {
            break;
        }
        vl_eval_vector_comparison_on_all_pairs_f(
            distances,
            dimension,
            ((*self_0).centers as *mut libc::c_float)
                .offset(
                    c
                        .wrapping_sub(1 as libc::c_int as libc::c_ulonglong)
                        .wrapping_mul(dimension) as isize,
                ),
            1 as libc::c_int as vl_size,
            data,
            numData,
            distFn,
        );
        x = 0 as libc::c_int as vl_uindex;
        while x < numData {
            *minDistances
                .offset(
                    x as isize,
                ) = if *minDistances.offset(x as isize) < *distances.offset(x as isize) {
                *minDistances.offset(x as isize)
            } else {
                *distances.offset(x as isize)
            };
            energy += *minDistances.offset(x as isize);
            x = x.wrapping_add(1);
        }
        x = 0 as libc::c_int as vl_uindex;
        while x < numData.wrapping_sub(1 as libc::c_int as libc::c_ulonglong) {
            acc += *minDistances.offset(x as isize);
            if acc >= thresh * energy {
                break;
            }
            x = x.wrapping_add(1);
        }
    }
    vl_free(distances as *mut libc::c_void);
    vl_free(minDistances as *mut libc::c_void);
}
unsafe extern "C" fn _vl_kmeans_init_centers_plus_plus_d(
    mut self_0: *mut VlKMeans,
    mut data: *const libc::c_double,
    mut dimension: vl_size,
    mut numData: vl_size,
    mut numCenters: vl_size,
) {
    let mut x: vl_uindex = 0;
    let mut c: vl_uindex = 0;
    let mut rand: *mut VlRand = vl_get_rand();
    let mut distances: *mut libc::c_double = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut libc::c_double;
    let mut minDistances: *mut libc::c_double = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut libc::c_double;
    let mut distFn: VlDoubleVectorComparisonFunction = vl_get_vector_comparison_function_d(
        (*self_0).distance,
    );
    (*self_0).dimension = dimension;
    (*self_0).numCenters = numCenters;
    (*self_0)
        .centers = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(dimension)
            .wrapping_mul(numCenters) as size_t,
    );
    x = 0 as libc::c_int as vl_uindex;
    while x < numData {
        *minDistances.offset(x as isize) = vl_infinity_d.value;
        x = x.wrapping_add(1);
    }
    x = vl_rand_uindex(rand, numData);
    c = 0 as libc::c_int as vl_uindex;
    loop {
        let mut energy: libc::c_double = 0 as libc::c_int as libc::c_double;
        let mut acc: libc::c_double = 0 as libc::c_int as libc::c_double;
        let mut thresh: libc::c_double = vl_rand_real1(rand);
        memcpy(
            ((*self_0).centers as *mut libc::c_double)
                .offset(c.wrapping_mul(dimension) as isize) as *mut libc::c_void,
            data.offset(x.wrapping_mul(dimension) as isize) as *const libc::c_void,
            (::core::mem::size_of::<libc::c_double>() as libc::c_ulong
                as libc::c_ulonglong)
                .wrapping_mul(dimension) as libc::c_ulong,
        );
        c = c.wrapping_add(1);
        if c == numCenters {
            break;
        }
        vl_eval_vector_comparison_on_all_pairs_d(
            distances,
            dimension,
            ((*self_0).centers as *mut libc::c_double)
                .offset(
                    c
                        .wrapping_sub(1 as libc::c_int as libc::c_ulonglong)
                        .wrapping_mul(dimension) as isize,
                ),
            1 as libc::c_int as vl_size,
            data,
            numData,
            distFn,
        );
        x = 0 as libc::c_int as vl_uindex;
        while x < numData {
            *minDistances
                .offset(
                    x as isize,
                ) = if *minDistances.offset(x as isize) < *distances.offset(x as isize) {
                *minDistances.offset(x as isize)
            } else {
                *distances.offset(x as isize)
            };
            energy += *minDistances.offset(x as isize);
            x = x.wrapping_add(1);
        }
        x = 0 as libc::c_int as vl_uindex;
        while x < numData.wrapping_sub(1 as libc::c_int as libc::c_ulonglong) {
            acc += *minDistances.offset(x as isize);
            if acc >= thresh * energy {
                break;
            }
            x = x.wrapping_add(1);
        }
    }
    vl_free(distances as *mut libc::c_void);
    vl_free(minDistances as *mut libc::c_void);
}
unsafe extern "C" fn _vl_kmeans_quantize_f(
    mut self_0: *mut VlKMeans,
    mut assignments: *mut vl_uint32,
    mut distances: *mut libc::c_float,
    mut data: *const libc::c_float,
    mut numData: vl_size,
) {
    let mut i: vl_index = 0;
    let mut distFn: VlFloatVectorComparisonFunction = vl_get_vector_comparison_function_f(
        (*self_0).distance,
    );
    let mut distanceToCenters: *mut libc::c_float = malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).numCenters) as libc::c_ulong,
    ) as *mut libc::c_float;
    i = 0 as libc::c_int as vl_index;
    while i < numData as libc::c_int as libc::c_longlong {
        let mut k: vl_uindex = 0;
        let mut bestDistance: libc::c_float = vl_infinity_d.value as libc::c_float;
        vl_eval_vector_comparison_on_all_pairs_f(
            distanceToCenters,
            (*self_0).dimension,
            data
                .offset(
                    ((*self_0).dimension).wrapping_mul(i as libc::c_ulonglong) as isize,
                ),
            1 as libc::c_int as vl_size,
            (*self_0).centers as *mut libc::c_float,
            (*self_0).numCenters,
            distFn,
        );
        k = 0 as libc::c_int as vl_uindex;
        while k < (*self_0).numCenters {
            if *distanceToCenters.offset(k as isize) < bestDistance {
                bestDistance = *distanceToCenters.offset(k as isize);
                *assignments.offset(i as isize) = k as vl_uint32;
            }
            k = k.wrapping_add(1);
        }
        if !distances.is_null() {
            *distances.offset(i as isize) = bestDistance;
        }
        i += 1;
    }
    free(distanceToCenters as *mut libc::c_void);
}
unsafe extern "C" fn _vl_kmeans_quantize_d(
    mut self_0: *mut VlKMeans,
    mut assignments: *mut vl_uint32,
    mut distances: *mut libc::c_double,
    mut data: *const libc::c_double,
    mut numData: vl_size,
) {
    let mut i: vl_index = 0;
    let mut distFn: VlDoubleVectorComparisonFunction = vl_get_vector_comparison_function_d(
        (*self_0).distance,
    );
    let mut distanceToCenters: *mut libc::c_double = malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).numCenters) as libc::c_ulong,
    ) as *mut libc::c_double;
    i = 0 as libc::c_int as vl_index;
    while i < numData as libc::c_int as libc::c_longlong {
        let mut k: vl_uindex = 0;
        let mut bestDistance: libc::c_double = vl_infinity_d.value;
        vl_eval_vector_comparison_on_all_pairs_d(
            distanceToCenters,
            (*self_0).dimension,
            data
                .offset(
                    ((*self_0).dimension).wrapping_mul(i as libc::c_ulonglong) as isize,
                ),
            1 as libc::c_int as vl_size,
            (*self_0).centers as *mut libc::c_double,
            (*self_0).numCenters,
            distFn,
        );
        k = 0 as libc::c_int as vl_uindex;
        while k < (*self_0).numCenters {
            if *distanceToCenters.offset(k as isize) < bestDistance {
                bestDistance = *distanceToCenters.offset(k as isize);
                *assignments.offset(i as isize) = k as vl_uint32;
            }
            k = k.wrapping_add(1);
        }
        if !distances.is_null() {
            *distances.offset(i as isize) = bestDistance;
        }
        i += 1;
    }
    free(distanceToCenters as *mut libc::c_void);
}
unsafe extern "C" fn _vl_kmeans_quantize_ann_d(
    mut self_0: *mut VlKMeans,
    mut assignments: *mut vl_uint32,
    mut distances: *mut libc::c_double,
    mut data: *const libc::c_double,
    mut numData: vl_size,
    mut update: vl_bool,
) {
    let mut distFn: VlDoubleVectorComparisonFunction = vl_get_vector_comparison_function_d(
        (*self_0).distance,
    );
    let mut forest: *mut VlKDForest = vl_kdforest_new(
        (*self_0).dataType,
        (*self_0).dimension,
        (*self_0).numTrees,
        (*self_0).distance,
    );
    vl_kdforest_set_max_num_comparisons(forest, (*self_0).maxNumComparisons);
    vl_kdforest_set_thresholding_method(forest, VL_KDTREE_MEDIAN);
    vl_kdforest_build(forest, (*self_0).numCenters, (*self_0).centers);
    let mut neighbor: VlKDForestNeighbor = VlKDForestNeighbor {
        distance: 0.,
        index: 0,
    };
    let mut searcher: *mut VlKDForestSearcher = 0 as *mut VlKDForestSearcher;
    let mut x: vl_index = 0;
    searcher = vl_kdforest_new_searcher(forest);
    x = 0 as libc::c_int as vl_index;
    while x < numData as libc::c_int as libc::c_longlong {
        vl_kdforestsearcher_query(
            searcher,
            &mut neighbor,
            1 as libc::c_int as vl_size,
            data
                .offset(
                    (x as libc::c_ulonglong).wrapping_mul((*self_0).dimension) as isize,
                ) as *const libc::c_void,
        );
        if !distances.is_null() {
            if update == 0 {
                *distances.offset(x as isize) = neighbor.distance;
                *assignments.offset(x as isize) = neighbor.index as vl_uint32;
            } else {
                let mut prevDist: libc::c_double = distFn
                    .expect(
                        "non-null function pointer",
                    )(
                    (*self_0).dimension,
                    data
                        .offset(
                            ((*self_0).dimension).wrapping_mul(x as libc::c_ulonglong)
                                as isize,
                        ),
                    ((*self_0).centers as *mut libc::c_double)
                        .offset(
                            ((*self_0).dimension)
                                .wrapping_mul(
                                    *assignments.offset(x as isize) as libc::c_ulonglong,
                                ) as isize,
                        ),
                );
                if prevDist > neighbor.distance {
                    *distances.offset(x as isize) = neighbor.distance;
                    *assignments.offset(x as isize) = neighbor.index as vl_uint32;
                } else {
                    *distances.offset(x as isize) = prevDist;
                }
            }
        } else {
            *assignments.offset(x as isize) = neighbor.index as vl_uint32;
        }
        x += 1;
    }
    vl_kdforest_delete(forest);
}
unsafe extern "C" fn _vl_kmeans_quantize_ann_f(
    mut self_0: *mut VlKMeans,
    mut assignments: *mut vl_uint32,
    mut distances: *mut libc::c_float,
    mut data: *const libc::c_float,
    mut numData: vl_size,
    mut update: vl_bool,
) {
    let mut distFn: VlFloatVectorComparisonFunction = vl_get_vector_comparison_function_f(
        (*self_0).distance,
    );
    let mut forest: *mut VlKDForest = vl_kdforest_new(
        (*self_0).dataType,
        (*self_0).dimension,
        (*self_0).numTrees,
        (*self_0).distance,
    );
    vl_kdforest_set_max_num_comparisons(forest, (*self_0).maxNumComparisons);
    vl_kdforest_set_thresholding_method(forest, VL_KDTREE_MEDIAN);
    vl_kdforest_build(forest, (*self_0).numCenters, (*self_0).centers);
    let mut neighbor: VlKDForestNeighbor = VlKDForestNeighbor {
        distance: 0.,
        index: 0,
    };
    let mut searcher: *mut VlKDForestSearcher = 0 as *mut VlKDForestSearcher;
    let mut x: vl_index = 0;
    searcher = vl_kdforest_new_searcher(forest);
    x = 0 as libc::c_int as vl_index;
    while x < numData as libc::c_int as libc::c_longlong {
        vl_kdforestsearcher_query(
            searcher,
            &mut neighbor,
            1 as libc::c_int as vl_size,
            data
                .offset(
                    (x as libc::c_ulonglong).wrapping_mul((*self_0).dimension) as isize,
                ) as *const libc::c_void,
        );
        if !distances.is_null() {
            if update == 0 {
                *distances.offset(x as isize) = neighbor.distance as libc::c_float;
                *assignments.offset(x as isize) = neighbor.index as vl_uint32;
            } else {
                let mut prevDist: libc::c_float = distFn
                    .expect(
                        "non-null function pointer",
                    )(
                    (*self_0).dimension,
                    data
                        .offset(
                            ((*self_0).dimension).wrapping_mul(x as libc::c_ulonglong)
                                as isize,
                        ),
                    ((*self_0).centers as *mut libc::c_float)
                        .offset(
                            ((*self_0).dimension)
                                .wrapping_mul(
                                    *assignments.offset(x as isize) as libc::c_ulonglong,
                                ) as isize,
                        ),
                );
                if prevDist > neighbor.distance as libc::c_float {
                    *distances.offset(x as isize) = neighbor.distance as libc::c_float;
                    *assignments.offset(x as isize) = neighbor.index as vl_uint32;
                } else {
                    *distances.offset(x as isize) = prevDist;
                }
            }
        } else {
            *assignments.offset(x as isize) = neighbor.index as vl_uint32;
        }
        x += 1;
    }
    vl_kdforest_delete(forest);
}
#[inline]
unsafe extern "C" fn _vl_kmeans_d_qsort_cmp(
    mut array: *mut VlKMeansSortWrapper,
    mut indexA: vl_uindex,
    mut indexB: vl_uindex,
) -> libc::c_double {
    return *((*array).data as *mut libc::c_double)
        .offset(
            (*((*array).permutation).offset(indexA as isize) as libc::c_ulonglong)
                .wrapping_mul((*array).stride) as isize,
        )
        - *((*array).data as *mut libc::c_double)
            .offset(
                (*((*array).permutation).offset(indexB as isize) as libc::c_ulonglong)
                    .wrapping_mul((*array).stride) as isize,
            );
}
#[inline]
unsafe extern "C" fn _vl_kmeans_f_qsort_cmp(
    mut array: *mut VlKMeansSortWrapper,
    mut indexA: vl_uindex,
    mut indexB: vl_uindex,
) -> libc::c_float {
    return *((*array).data as *mut libc::c_float)
        .offset(
            (*((*array).permutation).offset(indexA as isize) as libc::c_ulonglong)
                .wrapping_mul((*array).stride) as isize,
        )
        - *((*array).data as *mut libc::c_float)
            .offset(
                (*((*array).permutation).offset(indexB as isize) as libc::c_ulonglong)
                    .wrapping_mul((*array).stride) as isize,
            );
}
#[inline]
unsafe extern "C" fn _vl_kmeans_f_qsort_swap(
    mut array: *mut VlKMeansSortWrapper,
    mut indexA: vl_uindex,
    mut indexB: vl_uindex,
) {
    let mut tmp: vl_uint32 = *((*array).permutation).offset(indexA as isize);
    *((*array).permutation)
        .offset(indexA as isize) = *((*array).permutation).offset(indexB as isize);
    *((*array).permutation).offset(indexB as isize) = tmp;
}
#[inline]
unsafe extern "C" fn _vl_kmeans_d_qsort_swap(
    mut array: *mut VlKMeansSortWrapper,
    mut indexA: vl_uindex,
    mut indexB: vl_uindex,
) {
    let mut tmp: vl_uint32 = *((*array).permutation).offset(indexA as isize);
    *((*array).permutation)
        .offset(indexA as isize) = *((*array).permutation).offset(indexB as isize);
    *((*array).permutation).offset(indexB as isize) = tmp;
}
unsafe extern "C" fn _vl_kmeans_sort_data_helper_d(
    mut self_0: *mut VlKMeans,
    mut permutations: *mut vl_uint32,
    mut data: *const libc::c_double,
    mut numData: vl_size,
) {
    let mut d: vl_uindex = 0;
    let mut x: vl_uindex = 0;
    d = 0 as libc::c_int as vl_uindex;
    while d < (*self_0).dimension {
        let mut array: VlKMeansSortWrapper = VlKMeansSortWrapper {
            permutation: 0 as *mut vl_uint32,
            data: 0 as *const libc::c_void,
            stride: 0,
        };
        array.permutation = permutations.offset(d.wrapping_mul(numData) as isize);
        array.data = data.offset(d as isize) as *const libc::c_void;
        array.stride = (*self_0).dimension;
        x = 0 as libc::c_int as vl_uindex;
        while x < numData {
            *(array.permutation).offset(x as isize) = x as vl_uint32;
            x = x.wrapping_add(1);
        }
        _vl_kmeans_d_qsort_sort(&mut array, numData);
        d = d.wrapping_add(1);
    }
}
unsafe extern "C" fn _vl_kmeans_sort_data_helper_f(
    mut self_0: *mut VlKMeans,
    mut permutations: *mut vl_uint32,
    mut data: *const libc::c_float,
    mut numData: vl_size,
) {
    let mut d: vl_uindex = 0;
    let mut x: vl_uindex = 0;
    d = 0 as libc::c_int as vl_uindex;
    while d < (*self_0).dimension {
        let mut array: VlKMeansSortWrapper = VlKMeansSortWrapper {
            permutation: 0 as *mut vl_uint32,
            data: 0 as *const libc::c_void,
            stride: 0,
        };
        array.permutation = permutations.offset(d.wrapping_mul(numData) as isize);
        array.data = data.offset(d as isize) as *const libc::c_void;
        array.stride = (*self_0).dimension;
        x = 0 as libc::c_int as vl_uindex;
        while x < numData {
            *(array.permutation).offset(x as isize) = x as vl_uint32;
            x = x.wrapping_add(1);
        }
        _vl_kmeans_f_qsort_sort(&mut array, numData);
        d = d.wrapping_add(1);
    }
}
unsafe extern "C" fn _vl_kmeans_refine_centers_lloyd_d(
    mut self_0: *mut VlKMeans,
    mut data: *const libc::c_double,
    mut numData: vl_size,
) -> libc::c_double {
    let mut c: vl_size = 0;
    let mut d: vl_size = 0;
    let mut x: vl_size = 0;
    let mut iteration: vl_size = 0;
    let mut previousEnergy: libc::c_double = vl_infinity_d.value;
    let mut initialEnergy: libc::c_double = vl_infinity_d.value;
    let mut energy: libc::c_double = 0.;
    let mut distances: *mut libc::c_double = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut libc::c_double;
    let mut assignments: *mut vl_uint32 = vl_malloc(
        (::core::mem::size_of::<vl_uint32>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut vl_uint32;
    let mut clusterMasses: *mut vl_size = vl_malloc(
        (::core::mem::size_of::<vl_size>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut vl_size;
    let mut permutations: *mut vl_uint32 = 0 as *mut vl_uint32;
    let mut numSeenSoFar: *mut vl_size = 0 as *mut vl_size;
    let mut rand: *mut VlRand = vl_get_rand();
    let mut totNumRestartedCenters: vl_size = 0 as libc::c_int as vl_size;
    let mut numRestartedCenters: vl_size = 0 as libc::c_int as vl_size;
    if (*self_0).distance as libc::c_uint == VlDistanceL1 as libc::c_int as libc::c_uint
    {
        permutations = vl_malloc(
            (::core::mem::size_of::<vl_uint32>() as libc::c_ulong as libc::c_ulonglong)
                .wrapping_mul(numData)
                .wrapping_mul((*self_0).dimension) as size_t,
        ) as *mut vl_uint32;
        numSeenSoFar = vl_malloc(
            (::core::mem::size_of::<vl_size>() as libc::c_ulong as libc::c_ulonglong)
                .wrapping_mul((*self_0).numCenters) as size_t,
        ) as *mut vl_size;
        _vl_kmeans_sort_data_helper_d(self_0, permutations, data, numData);
    }
    energy = vl_infinity_d.value;
    iteration = 0 as libc::c_int as vl_size;
    loop {
        _vl_kmeans_quantize_d(self_0, assignments, distances, data, numData);
        energy = 0 as libc::c_int as libc::c_double;
        x = 0 as libc::c_int as vl_size;
        while x < numData {
            energy += *distances.offset(x as isize);
            x = x.wrapping_add(1);
        }
        if (*self_0).verbosity != 0 {
            (Some(
                ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                    .expect("non-null function pointer"),
            ))
                .expect(
                    "non-null function pointer",
                )(
                b"kmeans: Lloyd iter %d: energy = %g\n\0" as *const u8
                    as *const libc::c_char,
                iteration,
                energy,
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
                    b"kmeans: Lloyd terminating because maximum number of iterations reached\n\0"
                        as *const u8 as *const libc::c_char,
                );
            }
            break;
        } else if energy == previousEnergy {
            if (*self_0).verbosity != 0 {
                (Some(
                    ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                        .expect("non-null function pointer"),
                ))
                    .expect(
                        "non-null function pointer",
                    )(
                    b"kmeans: Lloyd terminating because the algorithm fully converged\n\0"
                        as *const u8 as *const libc::c_char,
                );
            }
            break;
        } else {
            if iteration == 0 as libc::c_int as libc::c_ulonglong {
                initialEnergy = energy;
            } else {
                let mut eps: libc::c_double = (previousEnergy - energy)
                    / (initialEnergy - energy);
                if eps < (*self_0).minEnergyVariation {
                    if (*self_0).verbosity != 0 {
                        (Some(
                            ((vl_get_printf_func
                                as unsafe extern "C" fn() -> printf_func_t)())
                                .expect("non-null function pointer"),
                        ))
                            .expect(
                                "non-null function pointer",
                            )(
                            b"kmeans: ANN terminating because the energy relative variation was less than %f\n\0"
                                as *const u8 as *const libc::c_char,
                            (*self_0).minEnergyVariation,
                        );
                    }
                    break;
                }
            }
            previousEnergy = energy;
            memset(
                clusterMasses as *mut libc::c_void,
                0 as libc::c_int,
                (::core::mem::size_of::<vl_size>() as libc::c_ulong as libc::c_ulonglong)
                    .wrapping_mul(numData) as libc::c_ulong,
            );
            x = 0 as libc::c_int as vl_size;
            while x < numData {
                let ref mut fresh0 = *clusterMasses
                    .offset(*assignments.offset(x as isize) as isize);
                *fresh0 = (*fresh0).wrapping_add(1);
                x = x.wrapping_add(1);
            }
            numRestartedCenters = 0 as libc::c_int as vl_size;
            match (*self_0).distance as libc::c_uint {
                1 => {
                    memset(
                        (*self_0).centers,
                        0 as libc::c_int,
                        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong
                            as libc::c_ulonglong)
                            .wrapping_mul((*self_0).dimension)
                            .wrapping_mul((*self_0).numCenters) as libc::c_ulong,
                    );
                    x = 0 as libc::c_int as vl_size;
                    while x < numData {
                        let mut cpt: *mut libc::c_double = ((*self_0).centers
                            as *mut libc::c_double)
                            .offset(
                                (*assignments.offset(x as isize) as libc::c_ulonglong)
                                    .wrapping_mul((*self_0).dimension) as isize,
                            );
                        let mut xpt: *const libc::c_double = data
                            .offset(x.wrapping_mul((*self_0).dimension) as isize);
                        d = 0 as libc::c_int as vl_size;
                        while d < (*self_0).dimension {
                            *cpt.offset(d as isize) += *xpt.offset(d as isize);
                            d = d.wrapping_add(1);
                        }
                        x = x.wrapping_add(1);
                    }
                    c = 0 as libc::c_int as vl_size;
                    while c < (*self_0).numCenters {
                        let mut cpt_0: *mut libc::c_double = ((*self_0).centers
                            as *mut libc::c_double)
                            .offset(c.wrapping_mul((*self_0).dimension) as isize);
                        if *clusterMasses.offset(c as isize)
                            > 0 as libc::c_int as libc::c_ulonglong
                        {
                            let mut mass: libc::c_double = *clusterMasses
                                .offset(c as isize) as libc::c_double;
                            d = 0 as libc::c_int as vl_size;
                            while d < (*self_0).dimension {
                                *cpt_0.offset(d as isize) /= mass;
                                d = d.wrapping_add(1);
                            }
                        } else {
                            let mut x_0: vl_uindex = vl_rand_uindex(rand, numData);
                            numRestartedCenters = numRestartedCenters.wrapping_add(1);
                            d = 0 as libc::c_int as vl_size;
                            while d < (*self_0).dimension {
                                *cpt_0
                                    .offset(
                                        d as isize,
                                    ) = *data
                                    .offset(
                                        x_0.wrapping_mul((*self_0).dimension).wrapping_add(d)
                                            as isize,
                                    );
                                d = d.wrapping_add(1);
                            }
                        }
                        c = c.wrapping_add(1);
                    }
                }
                0 => {
                    d = 0 as libc::c_int as vl_size;
                    while d < (*self_0).dimension {
                        let mut perm: *mut vl_uint32 = permutations
                            .offset(d.wrapping_mul(numData) as isize);
                        memset(
                            numSeenSoFar as *mut libc::c_void,
                            0 as libc::c_int,
                            (::core::mem::size_of::<vl_size>() as libc::c_ulong
                                as libc::c_ulonglong)
                                .wrapping_mul((*self_0).numCenters) as libc::c_ulong,
                        );
                        x = 0 as libc::c_int as vl_size;
                        while x < numData {
                            c = *assignments.offset(*perm.offset(x as isize) as isize)
                                as vl_size;
                            if (2 as libc::c_int as libc::c_ulonglong)
                                .wrapping_mul(*numSeenSoFar.offset(c as isize))
                                < *clusterMasses.offset(c as isize)
                            {
                                *((*self_0).centers as *mut libc::c_double)
                                    .offset(
                                        d.wrapping_add(c.wrapping_mul((*self_0).dimension)) as isize,
                                    ) = *data
                                    .offset(
                                        d
                                            .wrapping_add(
                                                (*perm.offset(x as isize) as libc::c_ulonglong)
                                                    .wrapping_mul((*self_0).dimension),
                                            ) as isize,
                                    );
                            }
                            let ref mut fresh1 = *numSeenSoFar.offset(c as isize);
                            *fresh1 = (*fresh1).wrapping_add(1);
                            x = x.wrapping_add(1);
                        }
                        c = 0 as libc::c_int as vl_size;
                        while c < (*self_0).numCenters {
                            if *clusterMasses.offset(c as isize)
                                == 0 as libc::c_int as libc::c_ulonglong
                            {
                                let mut cpt_1: *mut libc::c_double = ((*self_0).centers
                                    as *mut libc::c_double)
                                    .offset(c.wrapping_mul((*self_0).dimension) as isize);
                                let mut x_1: vl_uindex = vl_rand_uindex(rand, numData);
                                numRestartedCenters = numRestartedCenters.wrapping_add(1);
                                d = 0 as libc::c_int as vl_size;
                                while d < (*self_0).dimension {
                                    *cpt_1
                                        .offset(
                                            d as isize,
                                        ) = *data
                                        .offset(
                                            x_1.wrapping_mul((*self_0).dimension).wrapping_add(d)
                                                as isize,
                                        );
                                    d = d.wrapping_add(1);
                                }
                            }
                            c = c.wrapping_add(1);
                        }
                        d = d.wrapping_add(1);
                    }
                }
                _ => {
                    abort();
                }
            }
            totNumRestartedCenters = (totNumRestartedCenters as libc::c_ulonglong)
                .wrapping_add(numRestartedCenters) as vl_size as vl_size;
            if (*self_0).verbosity != 0 && numRestartedCenters != 0 {
                (Some(
                    ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                        .expect("non-null function pointer"),
                ))
                    .expect(
                        "non-null function pointer",
                    )(
                    b"kmeans: Lloyd iter %d: restarted %d centers\n\0" as *const u8
                        as *const libc::c_char,
                    iteration,
                    numRestartedCenters,
                );
            }
            iteration = iteration.wrapping_add(1);
        }
    }
    if !permutations.is_null() {
        vl_free(permutations as *mut libc::c_void);
    }
    if !numSeenSoFar.is_null() {
        vl_free(numSeenSoFar as *mut libc::c_void);
    }
    vl_free(distances as *mut libc::c_void);
    vl_free(assignments as *mut libc::c_void);
    vl_free(clusterMasses as *mut libc::c_void);
    return energy;
}
unsafe extern "C" fn _vl_kmeans_refine_centers_lloyd_f(
    mut self_0: *mut VlKMeans,
    mut data: *const libc::c_float,
    mut numData: vl_size,
) -> libc::c_double {
    let mut c: vl_size = 0;
    let mut d: vl_size = 0;
    let mut x: vl_size = 0;
    let mut iteration: vl_size = 0;
    let mut previousEnergy: libc::c_double = vl_infinity_d.value;
    let mut initialEnergy: libc::c_double = vl_infinity_d.value;
    let mut energy: libc::c_double = 0.;
    let mut distances: *mut libc::c_float = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut libc::c_float;
    let mut assignments: *mut vl_uint32 = vl_malloc(
        (::core::mem::size_of::<vl_uint32>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut vl_uint32;
    let mut clusterMasses: *mut vl_size = vl_malloc(
        (::core::mem::size_of::<vl_size>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut vl_size;
    let mut permutations: *mut vl_uint32 = 0 as *mut vl_uint32;
    let mut numSeenSoFar: *mut vl_size = 0 as *mut vl_size;
    let mut rand: *mut VlRand = vl_get_rand();
    let mut totNumRestartedCenters: vl_size = 0 as libc::c_int as vl_size;
    let mut numRestartedCenters: vl_size = 0 as libc::c_int as vl_size;
    if (*self_0).distance as libc::c_uint == VlDistanceL1 as libc::c_int as libc::c_uint
    {
        permutations = vl_malloc(
            (::core::mem::size_of::<vl_uint32>() as libc::c_ulong as libc::c_ulonglong)
                .wrapping_mul(numData)
                .wrapping_mul((*self_0).dimension) as size_t,
        ) as *mut vl_uint32;
        numSeenSoFar = vl_malloc(
            (::core::mem::size_of::<vl_size>() as libc::c_ulong as libc::c_ulonglong)
                .wrapping_mul((*self_0).numCenters) as size_t,
        ) as *mut vl_size;
        _vl_kmeans_sort_data_helper_f(self_0, permutations, data, numData);
    }
    energy = vl_infinity_d.value;
    iteration = 0 as libc::c_int as vl_size;
    loop {
        _vl_kmeans_quantize_f(self_0, assignments, distances, data, numData);
        energy = 0 as libc::c_int as libc::c_double;
        x = 0 as libc::c_int as vl_size;
        while x < numData {
            energy += *distances.offset(x as isize) as libc::c_double;
            x = x.wrapping_add(1);
        }
        if (*self_0).verbosity != 0 {
            (Some(
                ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                    .expect("non-null function pointer"),
            ))
                .expect(
                    "non-null function pointer",
                )(
                b"kmeans: Lloyd iter %d: energy = %g\n\0" as *const u8
                    as *const libc::c_char,
                iteration,
                energy,
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
                    b"kmeans: Lloyd terminating because maximum number of iterations reached\n\0"
                        as *const u8 as *const libc::c_char,
                );
            }
            break;
        } else if energy == previousEnergy {
            if (*self_0).verbosity != 0 {
                (Some(
                    ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                        .expect("non-null function pointer"),
                ))
                    .expect(
                        "non-null function pointer",
                    )(
                    b"kmeans: Lloyd terminating because the algorithm fully converged\n\0"
                        as *const u8 as *const libc::c_char,
                );
            }
            break;
        } else {
            if iteration == 0 as libc::c_int as libc::c_ulonglong {
                initialEnergy = energy;
            } else {
                let mut eps: libc::c_double = (previousEnergy - energy)
                    / (initialEnergy - energy);
                if eps < (*self_0).minEnergyVariation {
                    if (*self_0).verbosity != 0 {
                        (Some(
                            ((vl_get_printf_func
                                as unsafe extern "C" fn() -> printf_func_t)())
                                .expect("non-null function pointer"),
                        ))
                            .expect(
                                "non-null function pointer",
                            )(
                            b"kmeans: ANN terminating because the energy relative variation was less than %f\n\0"
                                as *const u8 as *const libc::c_char,
                            (*self_0).minEnergyVariation,
                        );
                    }
                    break;
                }
            }
            previousEnergy = energy;
            memset(
                clusterMasses as *mut libc::c_void,
                0 as libc::c_int,
                (::core::mem::size_of::<vl_size>() as libc::c_ulong as libc::c_ulonglong)
                    .wrapping_mul(numData) as libc::c_ulong,
            );
            x = 0 as libc::c_int as vl_size;
            while x < numData {
                let ref mut fresh2 = *clusterMasses
                    .offset(*assignments.offset(x as isize) as isize);
                *fresh2 = (*fresh2).wrapping_add(1);
                x = x.wrapping_add(1);
            }
            numRestartedCenters = 0 as libc::c_int as vl_size;
            match (*self_0).distance as libc::c_uint {
                1 => {
                    memset(
                        (*self_0).centers,
                        0 as libc::c_int,
                        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong
                            as libc::c_ulonglong)
                            .wrapping_mul((*self_0).dimension)
                            .wrapping_mul((*self_0).numCenters) as libc::c_ulong,
                    );
                    x = 0 as libc::c_int as vl_size;
                    while x < numData {
                        let mut cpt: *mut libc::c_float = ((*self_0).centers
                            as *mut libc::c_float)
                            .offset(
                                (*assignments.offset(x as isize) as libc::c_ulonglong)
                                    .wrapping_mul((*self_0).dimension) as isize,
                            );
                        let mut xpt: *const libc::c_float = data
                            .offset(x.wrapping_mul((*self_0).dimension) as isize);
                        d = 0 as libc::c_int as vl_size;
                        while d < (*self_0).dimension {
                            *cpt.offset(d as isize) += *xpt.offset(d as isize);
                            d = d.wrapping_add(1);
                        }
                        x = x.wrapping_add(1);
                    }
                    c = 0 as libc::c_int as vl_size;
                    while c < (*self_0).numCenters {
                        let mut cpt_0: *mut libc::c_float = ((*self_0).centers
                            as *mut libc::c_float)
                            .offset(c.wrapping_mul((*self_0).dimension) as isize);
                        if *clusterMasses.offset(c as isize)
                            > 0 as libc::c_int as libc::c_ulonglong
                        {
                            let mut mass: libc::c_float = *clusterMasses
                                .offset(c as isize) as libc::c_float;
                            d = 0 as libc::c_int as vl_size;
                            while d < (*self_0).dimension {
                                *cpt_0.offset(d as isize) /= mass;
                                d = d.wrapping_add(1);
                            }
                        } else {
                            let mut x_0: vl_uindex = vl_rand_uindex(rand, numData);
                            numRestartedCenters = numRestartedCenters.wrapping_add(1);
                            d = 0 as libc::c_int as vl_size;
                            while d < (*self_0).dimension {
                                *cpt_0
                                    .offset(
                                        d as isize,
                                    ) = *data
                                    .offset(
                                        x_0.wrapping_mul((*self_0).dimension).wrapping_add(d)
                                            as isize,
                                    );
                                d = d.wrapping_add(1);
                            }
                        }
                        c = c.wrapping_add(1);
                    }
                }
                0 => {
                    d = 0 as libc::c_int as vl_size;
                    while d < (*self_0).dimension {
                        let mut perm: *mut vl_uint32 = permutations
                            .offset(d.wrapping_mul(numData) as isize);
                        memset(
                            numSeenSoFar as *mut libc::c_void,
                            0 as libc::c_int,
                            (::core::mem::size_of::<vl_size>() as libc::c_ulong
                                as libc::c_ulonglong)
                                .wrapping_mul((*self_0).numCenters) as libc::c_ulong,
                        );
                        x = 0 as libc::c_int as vl_size;
                        while x < numData {
                            c = *assignments.offset(*perm.offset(x as isize) as isize)
                                as vl_size;
                            if (2 as libc::c_int as libc::c_ulonglong)
                                .wrapping_mul(*numSeenSoFar.offset(c as isize))
                                < *clusterMasses.offset(c as isize)
                            {
                                *((*self_0).centers as *mut libc::c_float)
                                    .offset(
                                        d.wrapping_add(c.wrapping_mul((*self_0).dimension)) as isize,
                                    ) = *data
                                    .offset(
                                        d
                                            .wrapping_add(
                                                (*perm.offset(x as isize) as libc::c_ulonglong)
                                                    .wrapping_mul((*self_0).dimension),
                                            ) as isize,
                                    );
                            }
                            let ref mut fresh3 = *numSeenSoFar.offset(c as isize);
                            *fresh3 = (*fresh3).wrapping_add(1);
                            x = x.wrapping_add(1);
                        }
                        c = 0 as libc::c_int as vl_size;
                        while c < (*self_0).numCenters {
                            if *clusterMasses.offset(c as isize)
                                == 0 as libc::c_int as libc::c_ulonglong
                            {
                                let mut cpt_1: *mut libc::c_float = ((*self_0).centers
                                    as *mut libc::c_float)
                                    .offset(c.wrapping_mul((*self_0).dimension) as isize);
                                let mut x_1: vl_uindex = vl_rand_uindex(rand, numData);
                                numRestartedCenters = numRestartedCenters.wrapping_add(1);
                                d = 0 as libc::c_int as vl_size;
                                while d < (*self_0).dimension {
                                    *cpt_1
                                        .offset(
                                            d as isize,
                                        ) = *data
                                        .offset(
                                            x_1.wrapping_mul((*self_0).dimension).wrapping_add(d)
                                                as isize,
                                        );
                                    d = d.wrapping_add(1);
                                }
                            }
                            c = c.wrapping_add(1);
                        }
                        d = d.wrapping_add(1);
                    }
                }
                _ => {
                    abort();
                }
            }
            totNumRestartedCenters = (totNumRestartedCenters as libc::c_ulonglong)
                .wrapping_add(numRestartedCenters) as vl_size as vl_size;
            if (*self_0).verbosity != 0 && numRestartedCenters != 0 {
                (Some(
                    ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                        .expect("non-null function pointer"),
                ))
                    .expect(
                        "non-null function pointer",
                    )(
                    b"kmeans: Lloyd iter %d: restarted %d centers\n\0" as *const u8
                        as *const libc::c_char,
                    iteration,
                    numRestartedCenters,
                );
            }
            iteration = iteration.wrapping_add(1);
        }
    }
    if !permutations.is_null() {
        vl_free(permutations as *mut libc::c_void);
    }
    if !numSeenSoFar.is_null() {
        vl_free(numSeenSoFar as *mut libc::c_void);
    }
    vl_free(distances as *mut libc::c_void);
    vl_free(assignments as *mut libc::c_void);
    vl_free(clusterMasses as *mut libc::c_void);
    return energy;
}
unsafe extern "C" fn _vl_kmeans_update_center_distances_d(
    mut self_0: *mut VlKMeans,
) -> libc::c_double {
    let mut distFn: VlDoubleVectorComparisonFunction = vl_get_vector_comparison_function_d(
        (*self_0).distance,
    );
    if ((*self_0).centerDistances).is_null() {
        (*self_0)
            .centerDistances = vl_malloc(
            (::core::mem::size_of::<libc::c_double>() as libc::c_ulong
                as libc::c_ulonglong)
                .wrapping_mul((*self_0).numCenters)
                .wrapping_mul((*self_0).numCenters) as size_t,
        );
    }
    vl_eval_vector_comparison_on_all_pairs_d(
        (*self_0).centerDistances as *mut libc::c_double,
        (*self_0).dimension,
        (*self_0).centers as *const libc::c_double,
        (*self_0).numCenters,
        0 as *const libc::c_double,
        0 as libc::c_int as vl_size,
        distFn,
    );
    return ((*self_0).numCenters)
        .wrapping_mul(
            ((*self_0).numCenters).wrapping_sub(1 as libc::c_int as libc::c_ulonglong),
        )
        .wrapping_div(2 as libc::c_int as libc::c_ulonglong) as libc::c_double;
}
unsafe extern "C" fn _vl_kmeans_update_center_distances_f(
    mut self_0: *mut VlKMeans,
) -> libc::c_double {
    let mut distFn: VlFloatVectorComparisonFunction = vl_get_vector_comparison_function_f(
        (*self_0).distance,
    );
    if ((*self_0).centerDistances).is_null() {
        (*self_0)
            .centerDistances = vl_malloc(
            (::core::mem::size_of::<libc::c_float>() as libc::c_ulong
                as libc::c_ulonglong)
                .wrapping_mul((*self_0).numCenters)
                .wrapping_mul((*self_0).numCenters) as size_t,
        );
    }
    vl_eval_vector_comparison_on_all_pairs_f(
        (*self_0).centerDistances as *mut libc::c_float,
        (*self_0).dimension,
        (*self_0).centers as *const libc::c_float,
        (*self_0).numCenters,
        0 as *const libc::c_float,
        0 as libc::c_int as vl_size,
        distFn,
    );
    return ((*self_0).numCenters)
        .wrapping_mul(
            ((*self_0).numCenters).wrapping_sub(1 as libc::c_int as libc::c_ulonglong),
        )
        .wrapping_div(2 as libc::c_int as libc::c_ulonglong) as libc::c_double;
}
unsafe extern "C" fn _vl_kmeans_refine_centers_ann_f(
    mut self_0: *mut VlKMeans,
    mut data: *const libc::c_float,
    mut numData: vl_size,
) -> libc::c_double {
    let mut c: vl_size = 0;
    let mut d: vl_size = 0;
    let mut x: vl_size = 0;
    let mut iteration: vl_size = 0;
    let mut initialEnergy: libc::c_double = vl_infinity_d.value;
    let mut previousEnergy: libc::c_double = vl_infinity_d.value;
    let mut energy: libc::c_double = 0.;
    let mut permutations: *mut vl_uint32 = 0 as *mut vl_uint32;
    let mut numSeenSoFar: *mut vl_size = 0 as *mut vl_size;
    let mut rand: *mut VlRand = vl_get_rand();
    let mut totNumRestartedCenters: vl_size = 0 as libc::c_int as vl_size;
    let mut numRestartedCenters: vl_size = 0 as libc::c_int as vl_size;
    let mut assignments: *mut vl_uint32 = vl_malloc(
        (::core::mem::size_of::<vl_uint32>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut vl_uint32;
    let mut clusterMasses: *mut vl_size = vl_malloc(
        (::core::mem::size_of::<vl_size>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut vl_size;
    let mut distances: *mut libc::c_float = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut libc::c_float;
    if (*self_0).distance as libc::c_uint == VlDistanceL1 as libc::c_int as libc::c_uint
    {
        permutations = vl_malloc(
            (::core::mem::size_of::<vl_uint32>() as libc::c_ulong as libc::c_ulonglong)
                .wrapping_mul(numData)
                .wrapping_mul((*self_0).dimension) as size_t,
        ) as *mut vl_uint32;
        numSeenSoFar = vl_malloc(
            (::core::mem::size_of::<vl_size>() as libc::c_ulong as libc::c_ulonglong)
                .wrapping_mul((*self_0).numCenters) as size_t,
        ) as *mut vl_size;
        _vl_kmeans_sort_data_helper_f(self_0, permutations, data, numData);
    }
    energy = vl_infinity_d.value;
    iteration = 0 as libc::c_int as vl_size;
    loop {
        _vl_kmeans_quantize_ann_f(
            self_0,
            assignments,
            distances,
            data,
            numData,
            (iteration > 0 as libc::c_int as libc::c_ulonglong) as libc::c_int,
        );
        energy = 0 as libc::c_int as libc::c_double;
        x = 0 as libc::c_int as vl_size;
        while x < numData {
            energy += *distances.offset(x as isize) as libc::c_double;
            x = x.wrapping_add(1);
        }
        if (*self_0).verbosity != 0 {
            (Some(
                ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                    .expect("non-null function pointer"),
            ))
                .expect(
                    "non-null function pointer",
                )(
                b"kmeans: ANN iter %d: energy = %g\n\0" as *const u8
                    as *const libc::c_char,
                iteration,
                energy,
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
                    b"kmeans: ANN terminating because the maximum number of iterations has been reached\n\0"
                        as *const u8 as *const libc::c_char,
                );
            }
            break;
        } else if energy == previousEnergy {
            if (*self_0).verbosity != 0 {
                (Some(
                    ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                        .expect("non-null function pointer"),
                ))
                    .expect(
                        "non-null function pointer",
                    )(
                    b"kmeans: ANN terminating because the algorithm fully converged\n\0"
                        as *const u8 as *const libc::c_char,
                );
            }
            break;
        } else {
            if iteration == 0 as libc::c_int as libc::c_ulonglong {
                initialEnergy = energy;
            } else {
                let mut eps: libc::c_double = (previousEnergy - energy)
                    / (initialEnergy - energy);
                if eps < (*self_0).minEnergyVariation {
                    if (*self_0).verbosity != 0 {
                        (Some(
                            ((vl_get_printf_func
                                as unsafe extern "C" fn() -> printf_func_t)())
                                .expect("non-null function pointer"),
                        ))
                            .expect(
                                "non-null function pointer",
                            )(
                            b"kmeans: ANN terminating because the energy relative variation was less than %f\n\0"
                                as *const u8 as *const libc::c_char,
                            (*self_0).minEnergyVariation,
                        );
                    }
                    break;
                }
            }
            previousEnergy = energy;
            memset(
                clusterMasses as *mut libc::c_void,
                0 as libc::c_int,
                (::core::mem::size_of::<vl_size>() as libc::c_ulong as libc::c_ulonglong)
                    .wrapping_mul(numData) as libc::c_ulong,
            );
            x = 0 as libc::c_int as vl_size;
            while x < numData {
                let ref mut fresh4 = *clusterMasses
                    .offset(*assignments.offset(x as isize) as isize);
                *fresh4 = (*fresh4).wrapping_add(1);
                x = x.wrapping_add(1);
            }
            numRestartedCenters = 0 as libc::c_int as vl_size;
            match (*self_0).distance as libc::c_uint {
                1 => {
                    memset(
                        (*self_0).centers,
                        0 as libc::c_int,
                        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong
                            as libc::c_ulonglong)
                            .wrapping_mul((*self_0).dimension)
                            .wrapping_mul((*self_0).numCenters) as libc::c_ulong,
                    );
                    x = 0 as libc::c_int as vl_size;
                    while x < numData {
                        let mut cpt: *mut libc::c_float = ((*self_0).centers
                            as *mut libc::c_float)
                            .offset(
                                (*assignments.offset(x as isize) as libc::c_ulonglong)
                                    .wrapping_mul((*self_0).dimension) as isize,
                            );
                        let mut xpt: *const libc::c_float = data
                            .offset(x.wrapping_mul((*self_0).dimension) as isize);
                        d = 0 as libc::c_int as vl_size;
                        while d < (*self_0).dimension {
                            *cpt.offset(d as isize) += *xpt.offset(d as isize);
                            d = d.wrapping_add(1);
                        }
                        x = x.wrapping_add(1);
                    }
                    c = 0 as libc::c_int as vl_size;
                    while c < (*self_0).numCenters {
                        let mut cpt_0: *mut libc::c_float = ((*self_0).centers
                            as *mut libc::c_float)
                            .offset(c.wrapping_mul((*self_0).dimension) as isize);
                        if *clusterMasses.offset(c as isize)
                            > 0 as libc::c_int as libc::c_ulonglong
                        {
                            let mut mass: libc::c_float = *clusterMasses
                                .offset(c as isize) as libc::c_float;
                            d = 0 as libc::c_int as vl_size;
                            while d < (*self_0).dimension {
                                *cpt_0.offset(d as isize) /= mass;
                                d = d.wrapping_add(1);
                            }
                        } else {
                            let mut x_0: vl_uindex = vl_rand_uindex(rand, numData);
                            numRestartedCenters = numRestartedCenters.wrapping_add(1);
                            d = 0 as libc::c_int as vl_size;
                            while d < (*self_0).dimension {
                                *cpt_0
                                    .offset(
                                        d as isize,
                                    ) = *data
                                    .offset(
                                        x_0.wrapping_mul((*self_0).dimension).wrapping_add(d)
                                            as isize,
                                    );
                                d = d.wrapping_add(1);
                            }
                        }
                        c = c.wrapping_add(1);
                    }
                }
                0 => {
                    d = 0 as libc::c_int as vl_size;
                    while d < (*self_0).dimension {
                        let mut perm: *mut vl_uint32 = permutations
                            .offset(d.wrapping_mul(numData) as isize);
                        memset(
                            numSeenSoFar as *mut libc::c_void,
                            0 as libc::c_int,
                            (::core::mem::size_of::<vl_size>() as libc::c_ulong
                                as libc::c_ulonglong)
                                .wrapping_mul((*self_0).numCenters) as libc::c_ulong,
                        );
                        x = 0 as libc::c_int as vl_size;
                        while x < numData {
                            c = *assignments.offset(*perm.offset(x as isize) as isize)
                                as vl_size;
                            if (2 as libc::c_int as libc::c_ulonglong)
                                .wrapping_mul(*numSeenSoFar.offset(c as isize))
                                < *clusterMasses.offset(c as isize)
                            {
                                *((*self_0).centers as *mut libc::c_float)
                                    .offset(
                                        d.wrapping_add(c.wrapping_mul((*self_0).dimension)) as isize,
                                    ) = *data
                                    .offset(
                                        d
                                            .wrapping_add(
                                                (*perm.offset(x as isize) as libc::c_ulonglong)
                                                    .wrapping_mul((*self_0).dimension),
                                            ) as isize,
                                    );
                            }
                            let ref mut fresh5 = *numSeenSoFar.offset(c as isize);
                            *fresh5 = (*fresh5).wrapping_add(1);
                            x = x.wrapping_add(1);
                        }
                        c = 0 as libc::c_int as vl_size;
                        while c < (*self_0).numCenters {
                            if *clusterMasses.offset(c as isize)
                                == 0 as libc::c_int as libc::c_ulonglong
                            {
                                let mut cpt_1: *mut libc::c_float = ((*self_0).centers
                                    as *mut libc::c_float)
                                    .offset(c.wrapping_mul((*self_0).dimension) as isize);
                                let mut x_1: vl_uindex = vl_rand_uindex(rand, numData);
                                numRestartedCenters = numRestartedCenters.wrapping_add(1);
                                d = 0 as libc::c_int as vl_size;
                                while d < (*self_0).dimension {
                                    *cpt_1
                                        .offset(
                                            d as isize,
                                        ) = *data
                                        .offset(
                                            x_1.wrapping_mul((*self_0).dimension).wrapping_add(d)
                                                as isize,
                                        );
                                    d = d.wrapping_add(1);
                                }
                            }
                            c = c.wrapping_add(1);
                        }
                        d = d.wrapping_add(1);
                    }
                }
                _ => {
                    (Some(
                        ((vl_get_printf_func
                            as unsafe extern "C" fn() -> printf_func_t)())
                            .expect("non-null function pointer"),
                    ))
                        .expect(
                            "non-null function pointer",
                        )(
                        b"bad distance set: %d\n\0" as *const u8 as *const libc::c_char,
                        (*self_0).distance as libc::c_uint,
                    );
                    abort();
                }
            }
            totNumRestartedCenters = (totNumRestartedCenters as libc::c_ulonglong)
                .wrapping_add(numRestartedCenters) as vl_size as vl_size;
            if (*self_0).verbosity != 0 && numRestartedCenters != 0 {
                (Some(
                    ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                        .expect("non-null function pointer"),
                ))
                    .expect(
                        "non-null function pointer",
                    )(
                    b"kmeans: ANN iter %d: restarted %d centers\n\0" as *const u8
                        as *const libc::c_char,
                    iteration,
                    numRestartedCenters,
                );
            }
            iteration = iteration.wrapping_add(1);
        }
    }
    if !permutations.is_null() {
        vl_free(permutations as *mut libc::c_void);
    }
    if !numSeenSoFar.is_null() {
        vl_free(numSeenSoFar as *mut libc::c_void);
    }
    vl_free(distances as *mut libc::c_void);
    vl_free(assignments as *mut libc::c_void);
    vl_free(clusterMasses as *mut libc::c_void);
    return energy;
}
unsafe extern "C" fn _vl_kmeans_refine_centers_ann_d(
    mut self_0: *mut VlKMeans,
    mut data: *const libc::c_double,
    mut numData: vl_size,
) -> libc::c_double {
    let mut c: vl_size = 0;
    let mut d: vl_size = 0;
    let mut x: vl_size = 0;
    let mut iteration: vl_size = 0;
    let mut initialEnergy: libc::c_double = vl_infinity_d.value;
    let mut previousEnergy: libc::c_double = vl_infinity_d.value;
    let mut energy: libc::c_double = 0.;
    let mut permutations: *mut vl_uint32 = 0 as *mut vl_uint32;
    let mut numSeenSoFar: *mut vl_size = 0 as *mut vl_size;
    let mut rand: *mut VlRand = vl_get_rand();
    let mut totNumRestartedCenters: vl_size = 0 as libc::c_int as vl_size;
    let mut numRestartedCenters: vl_size = 0 as libc::c_int as vl_size;
    let mut assignments: *mut vl_uint32 = vl_malloc(
        (::core::mem::size_of::<vl_uint32>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut vl_uint32;
    let mut clusterMasses: *mut vl_size = vl_malloc(
        (::core::mem::size_of::<vl_size>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut vl_size;
    let mut distances: *mut libc::c_double = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut libc::c_double;
    if (*self_0).distance as libc::c_uint == VlDistanceL1 as libc::c_int as libc::c_uint
    {
        permutations = vl_malloc(
            (::core::mem::size_of::<vl_uint32>() as libc::c_ulong as libc::c_ulonglong)
                .wrapping_mul(numData)
                .wrapping_mul((*self_0).dimension) as size_t,
        ) as *mut vl_uint32;
        numSeenSoFar = vl_malloc(
            (::core::mem::size_of::<vl_size>() as libc::c_ulong as libc::c_ulonglong)
                .wrapping_mul((*self_0).numCenters) as size_t,
        ) as *mut vl_size;
        _vl_kmeans_sort_data_helper_d(self_0, permutations, data, numData);
    }
    energy = vl_infinity_d.value;
    iteration = 0 as libc::c_int as vl_size;
    loop {
        _vl_kmeans_quantize_ann_d(
            self_0,
            assignments,
            distances,
            data,
            numData,
            (iteration > 0 as libc::c_int as libc::c_ulonglong) as libc::c_int,
        );
        energy = 0 as libc::c_int as libc::c_double;
        x = 0 as libc::c_int as vl_size;
        while x < numData {
            energy += *distances.offset(x as isize);
            x = x.wrapping_add(1);
        }
        if (*self_0).verbosity != 0 {
            (Some(
                ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                    .expect("non-null function pointer"),
            ))
                .expect(
                    "non-null function pointer",
                )(
                b"kmeans: ANN iter %d: energy = %g\n\0" as *const u8
                    as *const libc::c_char,
                iteration,
                energy,
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
                    b"kmeans: ANN terminating because the maximum number of iterations has been reached\n\0"
                        as *const u8 as *const libc::c_char,
                );
            }
            break;
        } else if energy == previousEnergy {
            if (*self_0).verbosity != 0 {
                (Some(
                    ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                        .expect("non-null function pointer"),
                ))
                    .expect(
                        "non-null function pointer",
                    )(
                    b"kmeans: ANN terminating because the algorithm fully converged\n\0"
                        as *const u8 as *const libc::c_char,
                );
            }
            break;
        } else {
            if iteration == 0 as libc::c_int as libc::c_ulonglong {
                initialEnergy = energy;
            } else {
                let mut eps: libc::c_double = (previousEnergy - energy)
                    / (initialEnergy - energy);
                if eps < (*self_0).minEnergyVariation {
                    if (*self_0).verbosity != 0 {
                        (Some(
                            ((vl_get_printf_func
                                as unsafe extern "C" fn() -> printf_func_t)())
                                .expect("non-null function pointer"),
                        ))
                            .expect(
                                "non-null function pointer",
                            )(
                            b"kmeans: ANN terminating because the energy relative variation was less than %f\n\0"
                                as *const u8 as *const libc::c_char,
                            (*self_0).minEnergyVariation,
                        );
                    }
                    break;
                }
            }
            previousEnergy = energy;
            memset(
                clusterMasses as *mut libc::c_void,
                0 as libc::c_int,
                (::core::mem::size_of::<vl_size>() as libc::c_ulong as libc::c_ulonglong)
                    .wrapping_mul(numData) as libc::c_ulong,
            );
            x = 0 as libc::c_int as vl_size;
            while x < numData {
                let ref mut fresh6 = *clusterMasses
                    .offset(*assignments.offset(x as isize) as isize);
                *fresh6 = (*fresh6).wrapping_add(1);
                x = x.wrapping_add(1);
            }
            numRestartedCenters = 0 as libc::c_int as vl_size;
            match (*self_0).distance as libc::c_uint {
                1 => {
                    memset(
                        (*self_0).centers,
                        0 as libc::c_int,
                        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong
                            as libc::c_ulonglong)
                            .wrapping_mul((*self_0).dimension)
                            .wrapping_mul((*self_0).numCenters) as libc::c_ulong,
                    );
                    x = 0 as libc::c_int as vl_size;
                    while x < numData {
                        let mut cpt: *mut libc::c_double = ((*self_0).centers
                            as *mut libc::c_double)
                            .offset(
                                (*assignments.offset(x as isize) as libc::c_ulonglong)
                                    .wrapping_mul((*self_0).dimension) as isize,
                            );
                        let mut xpt: *const libc::c_double = data
                            .offset(x.wrapping_mul((*self_0).dimension) as isize);
                        d = 0 as libc::c_int as vl_size;
                        while d < (*self_0).dimension {
                            *cpt.offset(d as isize) += *xpt.offset(d as isize);
                            d = d.wrapping_add(1);
                        }
                        x = x.wrapping_add(1);
                    }
                    c = 0 as libc::c_int as vl_size;
                    while c < (*self_0).numCenters {
                        let mut cpt_0: *mut libc::c_double = ((*self_0).centers
                            as *mut libc::c_double)
                            .offset(c.wrapping_mul((*self_0).dimension) as isize);
                        if *clusterMasses.offset(c as isize)
                            > 0 as libc::c_int as libc::c_ulonglong
                        {
                            let mut mass: libc::c_double = *clusterMasses
                                .offset(c as isize) as libc::c_double;
                            d = 0 as libc::c_int as vl_size;
                            while d < (*self_0).dimension {
                                *cpt_0.offset(d as isize) /= mass;
                                d = d.wrapping_add(1);
                            }
                        } else {
                            let mut x_0: vl_uindex = vl_rand_uindex(rand, numData);
                            numRestartedCenters = numRestartedCenters.wrapping_add(1);
                            d = 0 as libc::c_int as vl_size;
                            while d < (*self_0).dimension {
                                *cpt_0
                                    .offset(
                                        d as isize,
                                    ) = *data
                                    .offset(
                                        x_0.wrapping_mul((*self_0).dimension).wrapping_add(d)
                                            as isize,
                                    );
                                d = d.wrapping_add(1);
                            }
                        }
                        c = c.wrapping_add(1);
                    }
                }
                0 => {
                    d = 0 as libc::c_int as vl_size;
                    while d < (*self_0).dimension {
                        let mut perm: *mut vl_uint32 = permutations
                            .offset(d.wrapping_mul(numData) as isize);
                        memset(
                            numSeenSoFar as *mut libc::c_void,
                            0 as libc::c_int,
                            (::core::mem::size_of::<vl_size>() as libc::c_ulong
                                as libc::c_ulonglong)
                                .wrapping_mul((*self_0).numCenters) as libc::c_ulong,
                        );
                        x = 0 as libc::c_int as vl_size;
                        while x < numData {
                            c = *assignments.offset(*perm.offset(x as isize) as isize)
                                as vl_size;
                            if (2 as libc::c_int as libc::c_ulonglong)
                                .wrapping_mul(*numSeenSoFar.offset(c as isize))
                                < *clusterMasses.offset(c as isize)
                            {
                                *((*self_0).centers as *mut libc::c_double)
                                    .offset(
                                        d.wrapping_add(c.wrapping_mul((*self_0).dimension)) as isize,
                                    ) = *data
                                    .offset(
                                        d
                                            .wrapping_add(
                                                (*perm.offset(x as isize) as libc::c_ulonglong)
                                                    .wrapping_mul((*self_0).dimension),
                                            ) as isize,
                                    );
                            }
                            let ref mut fresh7 = *numSeenSoFar.offset(c as isize);
                            *fresh7 = (*fresh7).wrapping_add(1);
                            x = x.wrapping_add(1);
                        }
                        c = 0 as libc::c_int as vl_size;
                        while c < (*self_0).numCenters {
                            if *clusterMasses.offset(c as isize)
                                == 0 as libc::c_int as libc::c_ulonglong
                            {
                                let mut cpt_1: *mut libc::c_double = ((*self_0).centers
                                    as *mut libc::c_double)
                                    .offset(c.wrapping_mul((*self_0).dimension) as isize);
                                let mut x_1: vl_uindex = vl_rand_uindex(rand, numData);
                                numRestartedCenters = numRestartedCenters.wrapping_add(1);
                                d = 0 as libc::c_int as vl_size;
                                while d < (*self_0).dimension {
                                    *cpt_1
                                        .offset(
                                            d as isize,
                                        ) = *data
                                        .offset(
                                            x_1.wrapping_mul((*self_0).dimension).wrapping_add(d)
                                                as isize,
                                        );
                                    d = d.wrapping_add(1);
                                }
                            }
                            c = c.wrapping_add(1);
                        }
                        d = d.wrapping_add(1);
                    }
                }
                _ => {
                    (Some(
                        ((vl_get_printf_func
                            as unsafe extern "C" fn() -> printf_func_t)())
                            .expect("non-null function pointer"),
                    ))
                        .expect(
                            "non-null function pointer",
                        )(
                        b"bad distance set: %d\n\0" as *const u8 as *const libc::c_char,
                        (*self_0).distance as libc::c_uint,
                    );
                    abort();
                }
            }
            totNumRestartedCenters = (totNumRestartedCenters as libc::c_ulonglong)
                .wrapping_add(numRestartedCenters) as vl_size as vl_size;
            if (*self_0).verbosity != 0 && numRestartedCenters != 0 {
                (Some(
                    ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                        .expect("non-null function pointer"),
                ))
                    .expect(
                        "non-null function pointer",
                    )(
                    b"kmeans: ANN iter %d: restarted %d centers\n\0" as *const u8
                        as *const libc::c_char,
                    iteration,
                    numRestartedCenters,
                );
            }
            iteration = iteration.wrapping_add(1);
        }
    }
    if !permutations.is_null() {
        vl_free(permutations as *mut libc::c_void);
    }
    if !numSeenSoFar.is_null() {
        vl_free(numSeenSoFar as *mut libc::c_void);
    }
    vl_free(distances as *mut libc::c_void);
    vl_free(assignments as *mut libc::c_void);
    vl_free(clusterMasses as *mut libc::c_void);
    return energy;
}
unsafe extern "C" fn _vl_kmeans_refine_centers_elkan_f(
    mut self_0: *mut VlKMeans,
    mut data: *const libc::c_float,
    mut numData: vl_size,
) -> libc::c_double {
    let mut d: vl_size = 0;
    let mut iteration: vl_size = 0;
    let mut x: vl_index = 0;
    let mut c: vl_uint32 = 0;
    let mut j: vl_uint32 = 0;
    let mut allDone: vl_bool = 0;
    let mut distances: *mut libc::c_float = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut libc::c_float;
    let mut assignments: *mut vl_uint32 = vl_malloc(
        (::core::mem::size_of::<vl_uint32>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut vl_uint32;
    let mut clusterMasses: *mut vl_size = vl_malloc(
        (::core::mem::size_of::<vl_size>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut vl_size;
    let mut rand: *mut VlRand = vl_get_rand();
    let mut distFn: VlFloatVectorComparisonFunction = vl_get_vector_comparison_function_f(
        (*self_0).distance,
    );
    let mut nextCenterDistances: *mut libc::c_float = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).numCenters) as size_t,
    ) as *mut libc::c_float;
    let mut pointToClosestCenterUB: *mut libc::c_float = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut libc::c_float;
    let mut pointToClosestCenterUBIsStrict: *mut vl_bool = vl_malloc(
        (::core::mem::size_of::<vl_bool>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut vl_bool;
    let mut pointToCenterLB: *mut libc::c_float = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData)
            .wrapping_mul((*self_0).numCenters) as size_t,
    ) as *mut libc::c_float;
    let mut newCenters: *mut libc::c_float = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).dimension)
            .wrapping_mul((*self_0).numCenters) as size_t,
    ) as *mut libc::c_float;
    let mut centerToNewCenterDistances: *mut libc::c_float = vl_malloc(
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).numCenters) as size_t,
    ) as *mut libc::c_float;
    let mut permutations: *mut vl_uint32 = 0 as *mut vl_uint32;
    let mut numSeenSoFar: *mut vl_size = 0 as *mut vl_size;
    let mut energy: libc::c_double = 0.;
    let mut totDistanceComputationsToInit: vl_size = 0 as libc::c_int as vl_size;
    let mut totDistanceComputationsToRefreshUB: vl_size = 0 as libc::c_int as vl_size;
    let mut totDistanceComputationsToRefreshLB: vl_size = 0 as libc::c_int as vl_size;
    let mut totDistanceComputationsToRefreshCenterDistances: vl_size = 0 as libc::c_int
        as vl_size;
    let mut totDistanceComputationsToNewCenters: vl_size = 0 as libc::c_int as vl_size;
    let mut totDistanceComputationsToFinalize: vl_size = 0 as libc::c_int as vl_size;
    let mut totNumRestartedCenters: vl_size = 0 as libc::c_int as vl_size;
    if (*self_0).distance as libc::c_uint == VlDistanceL1 as libc::c_int as libc::c_uint
    {
        permutations = vl_malloc(
            (::core::mem::size_of::<vl_uint32>() as libc::c_ulong as libc::c_ulonglong)
                .wrapping_mul(numData)
                .wrapping_mul((*self_0).dimension) as size_t,
        ) as *mut vl_uint32;
        numSeenSoFar = vl_malloc(
            (::core::mem::size_of::<vl_size>() as libc::c_ulong as libc::c_ulonglong)
                .wrapping_mul((*self_0).numCenters) as size_t,
        ) as *mut vl_size;
        _vl_kmeans_sort_data_helper_f(self_0, permutations, data, numData);
    }
    totDistanceComputationsToInit = (totDistanceComputationsToInit as libc::c_double
        + _vl_kmeans_update_center_distances_f(self_0)) as vl_size;
    memset(
        pointToCenterLB as *mut libc::c_void,
        0 as libc::c_int,
        (::core::mem::size_of::<libc::c_float>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).numCenters)
            .wrapping_mul(numData) as libc::c_ulong,
    );
    x = 0 as libc::c_int as vl_index;
    while x < numData as libc::c_int as libc::c_longlong {
        let mut distance: libc::c_float = 0.;
        *assignments.offset(x as isize) = 0 as libc::c_int as vl_uint32;
        distance = distFn
            .expect(
                "non-null function pointer",
            )(
            (*self_0).dimension,
            data
                .offset(
                    (x as libc::c_ulonglong).wrapping_mul((*self_0).dimension) as isize,
                ),
            ((*self_0).centers as *mut libc::c_float).offset(0 as libc::c_int as isize),
        );
        *pointToClosestCenterUB.offset(x as isize) = distance;
        *pointToClosestCenterUBIsStrict.offset(x as isize) = 1 as libc::c_int;
        *pointToCenterLB
            .offset(
                (0 as libc::c_int as libc::c_ulonglong)
                    .wrapping_add(
                        (x as libc::c_ulonglong).wrapping_mul((*self_0).numCenters),
                    ) as isize,
            ) = distance;
        totDistanceComputationsToInit = (totDistanceComputationsToInit
            as libc::c_ulonglong)
            .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as vl_size as vl_size;
        c = 1 as libc::c_int as vl_uint32;
        while (c as libc::c_ulonglong) < (*self_0).numCenters {
            if !((if (*self_0).distance as libc::c_uint
                == VlDistanceL1 as libc::c_int as libc::c_uint
            {
                2.0f64
            } else {
                4.0f64
            }) * *pointToClosestCenterUB.offset(x as isize) as libc::c_double
                <= *((*self_0).centerDistances as *mut libc::c_float)
                    .offset(
                        (c as libc::c_ulonglong)
                            .wrapping_add(
                                (*assignments.offset(x as isize) as libc::c_ulonglong)
                                    .wrapping_mul((*self_0).numCenters),
                            ) as isize,
                    ) as libc::c_double)
            {
                distance = distFn
                    .expect(
                        "non-null function pointer",
                    )(
                    (*self_0).dimension,
                    data
                        .offset(
                            (x as libc::c_ulonglong).wrapping_mul((*self_0).dimension)
                                as isize,
                        ),
                    ((*self_0).centers as *mut libc::c_float)
                        .offset(
                            (c as libc::c_ulonglong).wrapping_mul((*self_0).dimension)
                                as isize,
                        ),
                );
                *pointToCenterLB
                    .offset(
                        (c as libc::c_ulonglong)
                            .wrapping_add(
                                (x as libc::c_ulonglong).wrapping_mul((*self_0).numCenters),
                            ) as isize,
                    ) = distance;
                totDistanceComputationsToInit = (totDistanceComputationsToInit
                    as libc::c_ulonglong)
                    .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as vl_size
                    as vl_size;
                if distance < *pointToClosestCenterUB.offset(x as isize) {
                    *pointToClosestCenterUB.offset(x as isize) = distance;
                    *assignments.offset(x as isize) = c;
                }
            }
            c = c.wrapping_add(1);
        }
        x += 1;
    }
    energy = 0 as libc::c_int as libc::c_double;
    x = 0 as libc::c_int as vl_index;
    while x < numData as libc::c_int as libc::c_longlong {
        energy += *pointToClosestCenterUB.offset(x as isize) as libc::c_double;
        x += 1;
    }
    if (*self_0).verbosity != 0 {
        (Some(
            ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                .expect("non-null function pointer"),
        ))
            .expect(
                "non-null function pointer",
            )(
            b"kmeans: Elkan iter 0: energy = %g, dist. calc. = %d\n\0" as *const u8
                as *const libc::c_char,
            energy,
            totDistanceComputationsToInit,
        );
    }
    iteration = 1 as libc::c_int as vl_size;
    loop {
        let mut numDistanceComputationsToRefreshUB: vl_size = 0 as libc::c_int
            as vl_size;
        let mut numDistanceComputationsToRefreshLB: vl_size = 0 as libc::c_int
            as vl_size;
        let mut numDistanceComputationsToRefreshCenterDistances: vl_size = 0
            as libc::c_int as vl_size;
        let mut numDistanceComputationsToNewCenters: vl_size = 0 as libc::c_int
            as vl_size;
        let mut numRestartedCenters: vl_size = 0 as libc::c_int as vl_size;
        memset(
            clusterMasses as *mut libc::c_void,
            0 as libc::c_int,
            (::core::mem::size_of::<vl_size>() as libc::c_ulong as libc::c_ulonglong)
                .wrapping_mul(numData) as libc::c_ulong,
        );
        x = 0 as libc::c_int as vl_index;
        while x < numData as libc::c_int as libc::c_longlong {
            let ref mut fresh8 = *clusterMasses
                .offset(*assignments.offset(x as isize) as isize);
            *fresh8 = (*fresh8).wrapping_add(1);
            x += 1;
        }
        match (*self_0).distance as libc::c_uint {
            1 => {
                memset(
                    newCenters as *mut libc::c_void,
                    0 as libc::c_int,
                    (::core::mem::size_of::<libc::c_float>() as libc::c_ulong
                        as libc::c_ulonglong)
                        .wrapping_mul((*self_0).dimension)
                        .wrapping_mul((*self_0).numCenters) as libc::c_ulong,
                );
                x = 0 as libc::c_int as vl_index;
                while x < numData as libc::c_int as libc::c_longlong {
                    let mut cpt: *mut libc::c_float = newCenters
                        .offset(
                            (*assignments.offset(x as isize) as libc::c_ulonglong)
                                .wrapping_mul((*self_0).dimension) as isize,
                        );
                    let mut xpt: *const libc::c_float = data
                        .offset(
                            (x as libc::c_ulonglong).wrapping_mul((*self_0).dimension)
                                as isize,
                        );
                    d = 0 as libc::c_int as vl_size;
                    while d < (*self_0).dimension {
                        *cpt.offset(d as isize) += *xpt.offset(d as isize);
                        d = d.wrapping_add(1);
                    }
                    x += 1;
                }
                c = 0 as libc::c_int as vl_uint32;
                while (c as libc::c_ulonglong) < (*self_0).numCenters {
                    let mut cpt_0: *mut libc::c_float = newCenters
                        .offset(
                            (c as libc::c_ulonglong).wrapping_mul((*self_0).dimension)
                                as isize,
                        );
                    if *clusterMasses.offset(c as isize)
                        > 0 as libc::c_int as libc::c_ulonglong
                    {
                        let mut mass: libc::c_float = *clusterMasses.offset(c as isize)
                            as libc::c_float;
                        d = 0 as libc::c_int as vl_size;
                        while d < (*self_0).dimension {
                            *cpt_0.offset(d as isize) /= mass;
                            d = d.wrapping_add(1);
                        }
                    } else {
                        let mut x_0: vl_uindex = vl_rand_uindex(rand, numData);
                        numRestartedCenters = numRestartedCenters.wrapping_add(1);
                        d = 0 as libc::c_int as vl_size;
                        while d < (*self_0).dimension {
                            *cpt_0
                                .offset(
                                    d as isize,
                                ) = *data
                                .offset(
                                    x_0.wrapping_mul((*self_0).dimension).wrapping_add(d)
                                        as isize,
                                );
                            d = d.wrapping_add(1);
                        }
                    }
                    c = c.wrapping_add(1);
                }
            }
            0 => {
                d = 0 as libc::c_int as vl_size;
                while d < (*self_0).dimension {
                    let mut perm: *mut vl_uint32 = permutations
                        .offset(d.wrapping_mul(numData) as isize);
                    memset(
                        numSeenSoFar as *mut libc::c_void,
                        0 as libc::c_int,
                        (::core::mem::size_of::<vl_size>() as libc::c_ulong
                            as libc::c_ulonglong)
                            .wrapping_mul((*self_0).numCenters) as libc::c_ulong,
                    );
                    x = 0 as libc::c_int as vl_index;
                    while x < numData as libc::c_int as libc::c_longlong {
                        c = *assignments.offset(*perm.offset(x as isize) as isize);
                        if (2 as libc::c_int as libc::c_ulonglong)
                            .wrapping_mul(*numSeenSoFar.offset(c as isize))
                            < *clusterMasses.offset(c as isize)
                        {
                            *newCenters
                                .offset(
                                    d
                                        .wrapping_add(
                                            (c as libc::c_ulonglong).wrapping_mul((*self_0).dimension),
                                        ) as isize,
                                ) = *data
                                .offset(
                                    d
                                        .wrapping_add(
                                            (*perm.offset(x as isize) as libc::c_ulonglong)
                                                .wrapping_mul((*self_0).dimension),
                                        ) as isize,
                                );
                        }
                        let ref mut fresh9 = *numSeenSoFar.offset(c as isize);
                        *fresh9 = (*fresh9).wrapping_add(1);
                        x += 1;
                    }
                    d = d.wrapping_add(1);
                }
                c = 0 as libc::c_int as vl_uint32;
                while (c as libc::c_ulonglong) < (*self_0).numCenters {
                    if *clusterMasses.offset(c as isize)
                        == 0 as libc::c_int as libc::c_ulonglong
                    {
                        let mut cpt_1: *mut libc::c_float = newCenters
                            .offset(
                                (c as libc::c_ulonglong).wrapping_mul((*self_0).dimension)
                                    as isize,
                            );
                        let mut x_1: vl_uindex = vl_rand_uindex(rand, numData);
                        numRestartedCenters = numRestartedCenters.wrapping_add(1);
                        d = 0 as libc::c_int as vl_size;
                        while d < (*self_0).dimension {
                            *cpt_1
                                .offset(
                                    d as isize,
                                ) = *data
                                .offset(
                                    x_1.wrapping_mul((*self_0).dimension).wrapping_add(d)
                                        as isize,
                                );
                            d = d.wrapping_add(1);
                        }
                    }
                    c = c.wrapping_add(1);
                }
            }
            _ => {
                abort();
            }
        }
        c = 0 as libc::c_int as vl_uint32;
        while (c as libc::c_ulonglong) < (*self_0).numCenters {
            let mut distance_0: libc::c_float = distFn
                .expect(
                    "non-null function pointer",
                )(
                (*self_0).dimension,
                newCenters
                    .offset(
                        (c as libc::c_ulonglong).wrapping_mul((*self_0).dimension)
                            as isize,
                    ),
                ((*self_0).centers as *mut libc::c_float)
                    .offset(
                        (c as libc::c_ulonglong).wrapping_mul((*self_0).dimension)
                            as isize,
                    ),
            );
            *centerToNewCenterDistances.offset(c as isize) = distance_0;
            numDistanceComputationsToNewCenters = (numDistanceComputationsToNewCenters
                as libc::c_ulonglong)
                .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as vl_size
                as vl_size;
            c = c.wrapping_add(1);
        }
        let mut tmp: *mut libc::c_float = (*self_0).centers as *mut libc::c_float;
        (*self_0).centers = newCenters as *mut libc::c_void;
        newCenters = tmp;
        numDistanceComputationsToRefreshCenterDistances = (numDistanceComputationsToRefreshCenterDistances
            as libc::c_double + _vl_kmeans_update_center_distances_f(self_0)) as vl_size;
        c = 0 as libc::c_int as vl_uint32;
        while (c as libc::c_ulonglong) < (*self_0).numCenters {
            *nextCenterDistances
                .offset(c as isize) = vl_infinity_d.value as libc::c_float;
            j = 0 as libc::c_int as vl_uint32;
            while (j as libc::c_ulonglong) < (*self_0).numCenters {
                if !(j == c) {
                    *nextCenterDistances
                        .offset(
                            c as isize,
                        ) = if *nextCenterDistances.offset(c as isize)
                        < *((*self_0).centerDistances as *mut libc::c_float)
                            .offset(
                                (j as libc::c_ulonglong)
                                    .wrapping_add(
                                        (c as libc::c_ulonglong).wrapping_mul((*self_0).numCenters),
                                    ) as isize,
                            )
                    {
                        *nextCenterDistances.offset(c as isize)
                    } else {
                        *((*self_0).centerDistances as *mut libc::c_float)
                            .offset(
                                (j as libc::c_ulonglong)
                                    .wrapping_add(
                                        (c as libc::c_ulonglong).wrapping_mul((*self_0).numCenters),
                                    ) as isize,
                            )
                    };
                }
                j = j.wrapping_add(1);
            }
            c = c.wrapping_add(1);
        }
        x = 0 as libc::c_int as vl_index;
        while x < numData as libc::c_int as libc::c_longlong {
            let mut a: libc::c_float = *pointToClosestCenterUB.offset(x as isize);
            let mut b: libc::c_float = *centerToNewCenterDistances
                .offset(*assignments.offset(x as isize) as isize);
            if (*self_0).distance as libc::c_uint
                == VlDistanceL1 as libc::c_int as libc::c_uint
            {
                *pointToClosestCenterUB.offset(x as isize) = a + b;
            } else {
                let mut sqrtab: libc::c_float = sqrtf(a * b);
                *pointToClosestCenterUB
                    .offset(
                        x as isize,
                    ) = ((a + b) as libc::c_double + 2.0f64 * sqrtab as libc::c_double)
                    as libc::c_float;
            }
            *pointToClosestCenterUBIsStrict.offset(x as isize) = 0 as libc::c_int;
            x += 1;
        }
        x = 0 as libc::c_int as vl_index;
        while x < numData as libc::c_int as libc::c_longlong {
            c = 0 as libc::c_int as vl_uint32;
            while (c as libc::c_ulonglong) < (*self_0).numCenters {
                let mut a_0: libc::c_float = *pointToCenterLB
                    .offset(
                        (c as libc::c_ulonglong)
                            .wrapping_add(
                                (x as libc::c_ulonglong).wrapping_mul((*self_0).numCenters),
                            ) as isize,
                    );
                let mut b_0: libc::c_float = *centerToNewCenterDistances
                    .offset(c as isize);
                if a_0 < b_0 {
                    *pointToCenterLB
                        .offset(
                            (c as libc::c_ulonglong)
                                .wrapping_add(
                                    (x as libc::c_ulonglong).wrapping_mul((*self_0).numCenters),
                                ) as isize,
                        ) = 0 as libc::c_int as libc::c_float;
                } else if (*self_0).distance as libc::c_uint
                    == VlDistanceL1 as libc::c_int as libc::c_uint
                {
                    *pointToCenterLB
                        .offset(
                            (c as libc::c_ulonglong)
                                .wrapping_add(
                                    (x as libc::c_ulonglong).wrapping_mul((*self_0).numCenters),
                                ) as isize,
                        ) = a_0 - b_0;
                } else {
                    let mut sqrtab_0: libc::c_float = sqrtf(a_0 * b_0);
                    *pointToCenterLB
                        .offset(
                            (c as libc::c_ulonglong)
                                .wrapping_add(
                                    (x as libc::c_ulonglong).wrapping_mul((*self_0).numCenters),
                                ) as isize,
                        ) = ((a_0 + b_0) as libc::c_double
                        - 2.0f64 * sqrtab_0 as libc::c_double) as libc::c_float;
                }
                c = c.wrapping_add(1);
            }
            x += 1;
        }
        allDone = 1 as libc::c_int;
        x = 0 as libc::c_int as vl_index;
        while x < numData as libc::c_int as libc::c_longlong {
            if !((if (*self_0).distance as libc::c_uint
                == VlDistanceL1 as libc::c_int as libc::c_uint
            {
                2.0f64
            } else {
                4.0f64
            }) * *pointToClosestCenterUB.offset(x as isize) as libc::c_double
                <= *nextCenterDistances.offset(*assignments.offset(x as isize) as isize)
                    as libc::c_double)
            {
                let mut current_block_141: u64;
                c = 0 as libc::c_int as vl_uint32;
                while (c as libc::c_ulonglong) < (*self_0).numCenters {
                    let mut cx: vl_uint32 = *assignments.offset(x as isize);
                    let mut distance_1: libc::c_float = 0.;
                    if !(cx == c) {
                        if !((if (*self_0).distance as libc::c_uint
                            == VlDistanceL1 as libc::c_int as libc::c_uint
                        {
                            2.0f64
                        } else {
                            4.0f64
                        }) * *pointToClosestCenterUB.offset(x as isize) as libc::c_double
                            <= *((*self_0).centerDistances as *mut libc::c_float)
                                .offset(
                                    (c as libc::c_ulonglong)
                                        .wrapping_add(
                                            (cx as libc::c_ulonglong).wrapping_mul((*self_0).numCenters),
                                        ) as isize,
                                ) as libc::c_double)
                        {
                            if !(*pointToClosestCenterUB.offset(x as isize)
                                <= *pointToCenterLB
                                    .offset(
                                        (c as libc::c_ulonglong)
                                            .wrapping_add(
                                                (x as libc::c_ulonglong).wrapping_mul((*self_0).numCenters),
                                            ) as isize,
                                    ))
                            {
                                if *pointToClosestCenterUBIsStrict.offset(x as isize) == 0 {
                                    distance_1 = distFn
                                        .expect(
                                            "non-null function pointer",
                                        )(
                                        (*self_0).dimension,
                                        data
                                            .offset(
                                                ((*self_0).dimension).wrapping_mul(x as libc::c_ulonglong)
                                                    as isize,
                                            ),
                                        ((*self_0).centers as *mut libc::c_float)
                                            .offset(
                                                ((*self_0).dimension).wrapping_mul(cx as libc::c_ulonglong)
                                                    as isize,
                                            ),
                                    );
                                    *pointToClosestCenterUB.offset(x as isize) = distance_1;
                                    *pointToClosestCenterUBIsStrict
                                        .offset(x as isize) = 1 as libc::c_int;
                                    *pointToCenterLB
                                        .offset(
                                            (cx as libc::c_ulonglong)
                                                .wrapping_add(
                                                    (x as libc::c_ulonglong).wrapping_mul((*self_0).numCenters),
                                                ) as isize,
                                        ) = distance_1;
                                    numDistanceComputationsToRefreshUB = (numDistanceComputationsToRefreshUB
                                        as libc::c_ulonglong)
                                        .wrapping_add(1 as libc::c_int as libc::c_ulonglong)
                                        as vl_size as vl_size;
                                    if (if (*self_0).distance as libc::c_uint
                                        == VlDistanceL1 as libc::c_int as libc::c_uint
                                    {
                                        2.0f64
                                    } else {
                                        4.0f64
                                    })
                                        * *pointToClosestCenterUB.offset(x as isize)
                                            as libc::c_double
                                        <= *((*self_0).centerDistances as *mut libc::c_float)
                                            .offset(
                                                (c as libc::c_ulonglong)
                                                    .wrapping_add(
                                                        (cx as libc::c_ulonglong).wrapping_mul((*self_0).numCenters),
                                                    ) as isize,
                                            ) as libc::c_double
                                    {
                                        current_block_141 = 12608488225262500095;
                                    } else if *pointToClosestCenterUB.offset(x as isize)
                                        <= *pointToCenterLB
                                            .offset(
                                                (c as libc::c_ulonglong)
                                                    .wrapping_add(
                                                        (x as libc::c_ulonglong).wrapping_mul((*self_0).numCenters),
                                                    ) as isize,
                                            )
                                    {
                                        current_block_141 = 12608488225262500095;
                                    } else {
                                        current_block_141 = 12223373342341601825;
                                    }
                                } else {
                                    current_block_141 = 12223373342341601825;
                                }
                                match current_block_141 {
                                    12608488225262500095 => {}
                                    _ => {
                                        distance_1 = distFn
                                            .expect(
                                                "non-null function pointer",
                                            )(
                                            (*self_0).dimension,
                                            data
                                                .offset(
                                                    (x as libc::c_ulonglong).wrapping_mul((*self_0).dimension)
                                                        as isize,
                                                ),
                                            ((*self_0).centers as *mut libc::c_float)
                                                .offset(
                                                    (c as libc::c_ulonglong).wrapping_mul((*self_0).dimension)
                                                        as isize,
                                                ),
                                        );
                                        numDistanceComputationsToRefreshLB = (numDistanceComputationsToRefreshLB
                                            as libc::c_ulonglong)
                                            .wrapping_add(1 as libc::c_int as libc::c_ulonglong)
                                            as vl_size as vl_size;
                                        *pointToCenterLB
                                            .offset(
                                                (c as libc::c_ulonglong)
                                                    .wrapping_add(
                                                        (x as libc::c_ulonglong).wrapping_mul((*self_0).numCenters),
                                                    ) as isize,
                                            ) = distance_1;
                                        if distance_1 < *pointToClosestCenterUB.offset(x as isize) {
                                            *assignments.offset(x as isize) = c;
                                            *pointToClosestCenterUB.offset(x as isize) = distance_1;
                                            allDone = 0 as libc::c_int;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    c = c.wrapping_add(1);
                }
            }
            x += 1;
        }
        totDistanceComputationsToRefreshUB = (totDistanceComputationsToRefreshUB
            as libc::c_ulonglong)
            .wrapping_add(numDistanceComputationsToRefreshUB) as vl_size as vl_size;
        totDistanceComputationsToRefreshLB = (totDistanceComputationsToRefreshLB
            as libc::c_ulonglong)
            .wrapping_add(numDistanceComputationsToRefreshLB) as vl_size as vl_size;
        totDistanceComputationsToRefreshCenterDistances = (totDistanceComputationsToRefreshCenterDistances
            as libc::c_ulonglong)
            .wrapping_add(numDistanceComputationsToRefreshCenterDistances) as vl_size
            as vl_size;
        totDistanceComputationsToNewCenters = (totDistanceComputationsToNewCenters
            as libc::c_ulonglong)
            .wrapping_add(numDistanceComputationsToNewCenters) as vl_size as vl_size;
        totNumRestartedCenters = (totNumRestartedCenters as libc::c_ulonglong)
            .wrapping_add(numRestartedCenters) as vl_size as vl_size;
        energy = 0 as libc::c_int as libc::c_double;
        x = 0 as libc::c_int as vl_index;
        while x < numData as libc::c_int as libc::c_longlong {
            energy += *pointToClosestCenterUB.offset(x as isize) as libc::c_double;
            x += 1;
        }
        if (*self_0).verbosity != 0 {
            let mut numDistanceComputations: vl_size = numDistanceComputationsToRefreshUB
                .wrapping_add(numDistanceComputationsToRefreshLB)
                .wrapping_add(numDistanceComputationsToRefreshCenterDistances)
                .wrapping_add(numDistanceComputationsToNewCenters);
            (Some(
                ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                    .expect("non-null function pointer"),
            ))
                .expect(
                    "non-null function pointer",
                )(
                b"kmeans: Elkan iter %d: energy <= %g, dist. calc. = %d\n\0" as *const u8
                    as *const libc::c_char,
                iteration,
                energy,
                numDistanceComputations,
            );
            if numRestartedCenters != 0 {
                (Some(
                    ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                        .expect("non-null function pointer"),
                ))
                    .expect(
                        "non-null function pointer",
                    )(
                    b"kmeans: Elkan iter %d: restarted %d centers\n\0" as *const u8
                        as *const libc::c_char,
                    iteration,
                    energy,
                    numRestartedCenters,
                );
            }
            if (*self_0).verbosity > 1 as libc::c_int {
                (Some(
                    ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                        .expect("non-null function pointer"),
                ))
                    .expect(
                        "non-null function pointer",
                    )(
                    b"kmeans: Elkan iter %d: total dist. calc. per type: UB: %.1f%% (%d), LB: %.1f%% (%d), intra_center: %.1f%% (%d), new_center: %.1f%% (%d)\n\0"
                        as *const u8 as *const libc::c_char,
                    iteration,
                    100.0f64 * numDistanceComputationsToRefreshUB as libc::c_double
                        / numDistanceComputations as libc::c_double,
                    numDistanceComputationsToRefreshUB,
                    100.0f64 * numDistanceComputationsToRefreshLB as libc::c_double
                        / numDistanceComputations as libc::c_double,
                    numDistanceComputationsToRefreshLB,
                    100.0f64
                        * numDistanceComputationsToRefreshCenterDistances
                            as libc::c_double
                        / numDistanceComputations as libc::c_double,
                    numDistanceComputationsToRefreshCenterDistances,
                    100.0f64 * numDistanceComputationsToNewCenters as libc::c_double
                        / numDistanceComputations as libc::c_double,
                    numDistanceComputationsToNewCenters,
                );
            }
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
                    b"kmeans: Elkan terminating because maximum number of iterations reached\n\0"
                        as *const u8 as *const libc::c_char,
                );
            }
            break;
        } else if allDone != 0 {
            if (*self_0).verbosity != 0 {
                (Some(
                    ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                        .expect("non-null function pointer"),
                ))
                    .expect(
                        "non-null function pointer",
                    )(
                    b"kmeans: Elkan terminating because the algorithm fully converged\n\0"
                        as *const u8 as *const libc::c_char,
                );
            }
            break;
        } else {
            iteration = iteration.wrapping_add(1);
        }
    }
    energy = 0 as libc::c_int as libc::c_double;
    x = 0 as libc::c_int as vl_index;
    while x < numData as libc::c_int as libc::c_longlong {
        let mut cx_0: vl_uindex = *assignments.offset(x as isize) as vl_uindex;
        energy
            += distFn
                .expect(
                    "non-null function pointer",
                )(
                (*self_0).dimension,
                data
                    .offset(
                        ((*self_0).dimension).wrapping_mul(x as libc::c_ulonglong)
                            as isize,
                    ),
                ((*self_0).centers as *mut libc::c_float)
                    .offset(((*self_0).dimension).wrapping_mul(cx_0) as isize),
            ) as libc::c_double;
        totDistanceComputationsToFinalize = (totDistanceComputationsToFinalize
            as libc::c_ulonglong)
            .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as vl_size as vl_size;
        x += 1;
    }
    let mut totDistanceComputations: vl_size = totDistanceComputationsToInit
        .wrapping_add(totDistanceComputationsToRefreshUB)
        .wrapping_add(totDistanceComputationsToRefreshLB)
        .wrapping_add(totDistanceComputationsToRefreshCenterDistances)
        .wrapping_add(totDistanceComputationsToNewCenters)
        .wrapping_add(totDistanceComputationsToFinalize);
    let mut saving: libc::c_double = totDistanceComputations as libc::c_double
        / iteration.wrapping_mul((*self_0).numCenters).wrapping_mul(numData)
            as libc::c_double;
    if (*self_0).verbosity != 0 {
        (Some(
            ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                .expect("non-null function pointer"),
        ))
            .expect(
                "non-null function pointer",
            )(
            b"kmeans: Elkan: total dist. calc.: %d (%.2f %% of Lloyd)\n\0" as *const u8
                as *const libc::c_char,
            totDistanceComputations,
            saving * 100.0f64,
        );
        if totNumRestartedCenters != 0 {
            (Some(
                ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                    .expect("non-null function pointer"),
            ))
                .expect(
                    "non-null function pointer",
                )(
                b"kmeans: Elkan: there have been %d restarts\n\0" as *const u8
                    as *const libc::c_char,
                totNumRestartedCenters,
            );
        }
    }
    if (*self_0).verbosity > 1 as libc::c_int {
        (Some(
            ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                .expect("non-null function pointer"),
        ))
            .expect(
                "non-null function pointer",
            )(
            b"kmeans: Elkan: total dist. calc. per type: init: %.1f%% (%d), UB: %.1f%% (%d), LB: %.1f%% (%d), intra_center: %.1f%% (%d), new_center: %.1f%% (%d), finalize: %.1f%% (%d)\n\0"
                as *const u8 as *const libc::c_char,
            100.0f64 * totDistanceComputationsToInit as libc::c_double
                / totDistanceComputations as libc::c_double,
            totDistanceComputationsToInit,
            100.0f64 * totDistanceComputationsToRefreshUB as libc::c_double
                / totDistanceComputations as libc::c_double,
            totDistanceComputationsToRefreshUB,
            100.0f64 * totDistanceComputationsToRefreshLB as libc::c_double
                / totDistanceComputations as libc::c_double,
            totDistanceComputationsToRefreshLB,
            100.0f64 * totDistanceComputationsToRefreshCenterDistances as libc::c_double
                / totDistanceComputations as libc::c_double,
            totDistanceComputationsToRefreshCenterDistances,
            100.0f64 * totDistanceComputationsToNewCenters as libc::c_double
                / totDistanceComputations as libc::c_double,
            totDistanceComputationsToNewCenters,
            100.0f64 * totDistanceComputationsToFinalize as libc::c_double
                / totDistanceComputations as libc::c_double,
            totDistanceComputationsToFinalize,
        );
    }
    if !permutations.is_null() {
        vl_free(permutations as *mut libc::c_void);
    }
    if !numSeenSoFar.is_null() {
        vl_free(numSeenSoFar as *mut libc::c_void);
    }
    vl_free(distances as *mut libc::c_void);
    vl_free(assignments as *mut libc::c_void);
    vl_free(clusterMasses as *mut libc::c_void);
    vl_free(nextCenterDistances as *mut libc::c_void);
    vl_free(pointToClosestCenterUB as *mut libc::c_void);
    vl_free(pointToClosestCenterUBIsStrict as *mut libc::c_void);
    vl_free(pointToCenterLB as *mut libc::c_void);
    vl_free(newCenters as *mut libc::c_void);
    vl_free(centerToNewCenterDistances as *mut libc::c_void);
    return energy;
}
unsafe extern "C" fn _vl_kmeans_refine_centers_elkan_d(
    mut self_0: *mut VlKMeans,
    mut data: *const libc::c_double,
    mut numData: vl_size,
) -> libc::c_double {
    let mut d: vl_size = 0;
    let mut iteration: vl_size = 0;
    let mut x: vl_index = 0;
    let mut c: vl_uint32 = 0;
    let mut j: vl_uint32 = 0;
    let mut allDone: vl_bool = 0;
    let mut distances: *mut libc::c_double = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut libc::c_double;
    let mut assignments: *mut vl_uint32 = vl_malloc(
        (::core::mem::size_of::<vl_uint32>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut vl_uint32;
    let mut clusterMasses: *mut vl_size = vl_malloc(
        (::core::mem::size_of::<vl_size>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut vl_size;
    let mut rand: *mut VlRand = vl_get_rand();
    let mut distFn: VlDoubleVectorComparisonFunction = vl_get_vector_comparison_function_d(
        (*self_0).distance,
    );
    let mut nextCenterDistances: *mut libc::c_double = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).numCenters) as size_t,
    ) as *mut libc::c_double;
    let mut pointToClosestCenterUB: *mut libc::c_double = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut libc::c_double;
    let mut pointToClosestCenterUBIsStrict: *mut vl_bool = vl_malloc(
        (::core::mem::size_of::<vl_bool>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData) as size_t,
    ) as *mut vl_bool;
    let mut pointToCenterLB: *mut libc::c_double = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul(numData)
            .wrapping_mul((*self_0).numCenters) as size_t,
    ) as *mut libc::c_double;
    let mut newCenters: *mut libc::c_double = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).dimension)
            .wrapping_mul((*self_0).numCenters) as size_t,
    ) as *mut libc::c_double;
    let mut centerToNewCenterDistances: *mut libc::c_double = vl_malloc(
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).numCenters) as size_t,
    ) as *mut libc::c_double;
    let mut permutations: *mut vl_uint32 = 0 as *mut vl_uint32;
    let mut numSeenSoFar: *mut vl_size = 0 as *mut vl_size;
    let mut energy: libc::c_double = 0.;
    let mut totDistanceComputationsToInit: vl_size = 0 as libc::c_int as vl_size;
    let mut totDistanceComputationsToRefreshUB: vl_size = 0 as libc::c_int as vl_size;
    let mut totDistanceComputationsToRefreshLB: vl_size = 0 as libc::c_int as vl_size;
    let mut totDistanceComputationsToRefreshCenterDistances: vl_size = 0 as libc::c_int
        as vl_size;
    let mut totDistanceComputationsToNewCenters: vl_size = 0 as libc::c_int as vl_size;
    let mut totDistanceComputationsToFinalize: vl_size = 0 as libc::c_int as vl_size;
    let mut totNumRestartedCenters: vl_size = 0 as libc::c_int as vl_size;
    if (*self_0).distance as libc::c_uint == VlDistanceL1 as libc::c_int as libc::c_uint
    {
        permutations = vl_malloc(
            (::core::mem::size_of::<vl_uint32>() as libc::c_ulong as libc::c_ulonglong)
                .wrapping_mul(numData)
                .wrapping_mul((*self_0).dimension) as size_t,
        ) as *mut vl_uint32;
        numSeenSoFar = vl_malloc(
            (::core::mem::size_of::<vl_size>() as libc::c_ulong as libc::c_ulonglong)
                .wrapping_mul((*self_0).numCenters) as size_t,
        ) as *mut vl_size;
        _vl_kmeans_sort_data_helper_d(self_0, permutations, data, numData);
    }
    totDistanceComputationsToInit = (totDistanceComputationsToInit as libc::c_double
        + _vl_kmeans_update_center_distances_d(self_0)) as vl_size;
    memset(
        pointToCenterLB as *mut libc::c_void,
        0 as libc::c_int,
        (::core::mem::size_of::<libc::c_double>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).numCenters)
            .wrapping_mul(numData) as libc::c_ulong,
    );
    x = 0 as libc::c_int as vl_index;
    while x < numData as libc::c_int as libc::c_longlong {
        let mut distance: libc::c_double = 0.;
        *assignments.offset(x as isize) = 0 as libc::c_int as vl_uint32;
        distance = distFn
            .expect(
                "non-null function pointer",
            )(
            (*self_0).dimension,
            data
                .offset(
                    (x as libc::c_ulonglong).wrapping_mul((*self_0).dimension) as isize,
                ),
            ((*self_0).centers as *mut libc::c_double).offset(0 as libc::c_int as isize),
        );
        *pointToClosestCenterUB.offset(x as isize) = distance;
        *pointToClosestCenterUBIsStrict.offset(x as isize) = 1 as libc::c_int;
        *pointToCenterLB
            .offset(
                (0 as libc::c_int as libc::c_ulonglong)
                    .wrapping_add(
                        (x as libc::c_ulonglong).wrapping_mul((*self_0).numCenters),
                    ) as isize,
            ) = distance;
        totDistanceComputationsToInit = (totDistanceComputationsToInit
            as libc::c_ulonglong)
            .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as vl_size as vl_size;
        c = 1 as libc::c_int as vl_uint32;
        while (c as libc::c_ulonglong) < (*self_0).numCenters {
            if !((if (*self_0).distance as libc::c_uint
                == VlDistanceL1 as libc::c_int as libc::c_uint
            {
                2.0f64
            } else {
                4.0f64
            }) * *pointToClosestCenterUB.offset(x as isize)
                <= *((*self_0).centerDistances as *mut libc::c_double)
                    .offset(
                        (c as libc::c_ulonglong)
                            .wrapping_add(
                                (*assignments.offset(x as isize) as libc::c_ulonglong)
                                    .wrapping_mul((*self_0).numCenters),
                            ) as isize,
                    ))
            {
                distance = distFn
                    .expect(
                        "non-null function pointer",
                    )(
                    (*self_0).dimension,
                    data
                        .offset(
                            (x as libc::c_ulonglong).wrapping_mul((*self_0).dimension)
                                as isize,
                        ),
                    ((*self_0).centers as *mut libc::c_double)
                        .offset(
                            (c as libc::c_ulonglong).wrapping_mul((*self_0).dimension)
                                as isize,
                        ),
                );
                *pointToCenterLB
                    .offset(
                        (c as libc::c_ulonglong)
                            .wrapping_add(
                                (x as libc::c_ulonglong).wrapping_mul((*self_0).numCenters),
                            ) as isize,
                    ) = distance;
                totDistanceComputationsToInit = (totDistanceComputationsToInit
                    as libc::c_ulonglong)
                    .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as vl_size
                    as vl_size;
                if distance < *pointToClosestCenterUB.offset(x as isize) {
                    *pointToClosestCenterUB.offset(x as isize) = distance;
                    *assignments.offset(x as isize) = c;
                }
            }
            c = c.wrapping_add(1);
        }
        x += 1;
    }
    energy = 0 as libc::c_int as libc::c_double;
    x = 0 as libc::c_int as vl_index;
    while x < numData as libc::c_int as libc::c_longlong {
        energy += *pointToClosestCenterUB.offset(x as isize);
        x += 1;
    }
    if (*self_0).verbosity != 0 {
        (Some(
            ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                .expect("non-null function pointer"),
        ))
            .expect(
                "non-null function pointer",
            )(
            b"kmeans: Elkan iter 0: energy = %g, dist. calc. = %d\n\0" as *const u8
                as *const libc::c_char,
            energy,
            totDistanceComputationsToInit,
        );
    }
    iteration = 1 as libc::c_int as vl_size;
    loop {
        let mut numDistanceComputationsToRefreshUB: vl_size = 0 as libc::c_int
            as vl_size;
        let mut numDistanceComputationsToRefreshLB: vl_size = 0 as libc::c_int
            as vl_size;
        let mut numDistanceComputationsToRefreshCenterDistances: vl_size = 0
            as libc::c_int as vl_size;
        let mut numDistanceComputationsToNewCenters: vl_size = 0 as libc::c_int
            as vl_size;
        let mut numRestartedCenters: vl_size = 0 as libc::c_int as vl_size;
        memset(
            clusterMasses as *mut libc::c_void,
            0 as libc::c_int,
            (::core::mem::size_of::<vl_size>() as libc::c_ulong as libc::c_ulonglong)
                .wrapping_mul(numData) as libc::c_ulong,
        );
        x = 0 as libc::c_int as vl_index;
        while x < numData as libc::c_int as libc::c_longlong {
            let ref mut fresh10 = *clusterMasses
                .offset(*assignments.offset(x as isize) as isize);
            *fresh10 = (*fresh10).wrapping_add(1);
            x += 1;
        }
        match (*self_0).distance as libc::c_uint {
            1 => {
                memset(
                    newCenters as *mut libc::c_void,
                    0 as libc::c_int,
                    (::core::mem::size_of::<libc::c_double>() as libc::c_ulong
                        as libc::c_ulonglong)
                        .wrapping_mul((*self_0).dimension)
                        .wrapping_mul((*self_0).numCenters) as libc::c_ulong,
                );
                x = 0 as libc::c_int as vl_index;
                while x < numData as libc::c_int as libc::c_longlong {
                    let mut cpt: *mut libc::c_double = newCenters
                        .offset(
                            (*assignments.offset(x as isize) as libc::c_ulonglong)
                                .wrapping_mul((*self_0).dimension) as isize,
                        );
                    let mut xpt: *const libc::c_double = data
                        .offset(
                            (x as libc::c_ulonglong).wrapping_mul((*self_0).dimension)
                                as isize,
                        );
                    d = 0 as libc::c_int as vl_size;
                    while d < (*self_0).dimension {
                        *cpt.offset(d as isize) += *xpt.offset(d as isize);
                        d = d.wrapping_add(1);
                    }
                    x += 1;
                }
                c = 0 as libc::c_int as vl_uint32;
                while (c as libc::c_ulonglong) < (*self_0).numCenters {
                    let mut cpt_0: *mut libc::c_double = newCenters
                        .offset(
                            (c as libc::c_ulonglong).wrapping_mul((*self_0).dimension)
                                as isize,
                        );
                    if *clusterMasses.offset(c as isize)
                        > 0 as libc::c_int as libc::c_ulonglong
                    {
                        let mut mass: libc::c_double = *clusterMasses.offset(c as isize)
                            as libc::c_double;
                        d = 0 as libc::c_int as vl_size;
                        while d < (*self_0).dimension {
                            *cpt_0.offset(d as isize) /= mass;
                            d = d.wrapping_add(1);
                        }
                    } else {
                        let mut x_0: vl_uindex = vl_rand_uindex(rand, numData);
                        numRestartedCenters = numRestartedCenters.wrapping_add(1);
                        d = 0 as libc::c_int as vl_size;
                        while d < (*self_0).dimension {
                            *cpt_0
                                .offset(
                                    d as isize,
                                ) = *data
                                .offset(
                                    x_0.wrapping_mul((*self_0).dimension).wrapping_add(d)
                                        as isize,
                                );
                            d = d.wrapping_add(1);
                        }
                    }
                    c = c.wrapping_add(1);
                }
            }
            0 => {
                d = 0 as libc::c_int as vl_size;
                while d < (*self_0).dimension {
                    let mut perm: *mut vl_uint32 = permutations
                        .offset(d.wrapping_mul(numData) as isize);
                    memset(
                        numSeenSoFar as *mut libc::c_void,
                        0 as libc::c_int,
                        (::core::mem::size_of::<vl_size>() as libc::c_ulong
                            as libc::c_ulonglong)
                            .wrapping_mul((*self_0).numCenters) as libc::c_ulong,
                    );
                    x = 0 as libc::c_int as vl_index;
                    while x < numData as libc::c_int as libc::c_longlong {
                        c = *assignments.offset(*perm.offset(x as isize) as isize);
                        if (2 as libc::c_int as libc::c_ulonglong)
                            .wrapping_mul(*numSeenSoFar.offset(c as isize))
                            < *clusterMasses.offset(c as isize)
                        {
                            *newCenters
                                .offset(
                                    d
                                        .wrapping_add(
                                            (c as libc::c_ulonglong).wrapping_mul((*self_0).dimension),
                                        ) as isize,
                                ) = *data
                                .offset(
                                    d
                                        .wrapping_add(
                                            (*perm.offset(x as isize) as libc::c_ulonglong)
                                                .wrapping_mul((*self_0).dimension),
                                        ) as isize,
                                );
                        }
                        let ref mut fresh11 = *numSeenSoFar.offset(c as isize);
                        *fresh11 = (*fresh11).wrapping_add(1);
                        x += 1;
                    }
                    d = d.wrapping_add(1);
                }
                c = 0 as libc::c_int as vl_uint32;
                while (c as libc::c_ulonglong) < (*self_0).numCenters {
                    if *clusterMasses.offset(c as isize)
                        == 0 as libc::c_int as libc::c_ulonglong
                    {
                        let mut cpt_1: *mut libc::c_double = newCenters
                            .offset(
                                (c as libc::c_ulonglong).wrapping_mul((*self_0).dimension)
                                    as isize,
                            );
                        let mut x_1: vl_uindex = vl_rand_uindex(rand, numData);
                        numRestartedCenters = numRestartedCenters.wrapping_add(1);
                        d = 0 as libc::c_int as vl_size;
                        while d < (*self_0).dimension {
                            *cpt_1
                                .offset(
                                    d as isize,
                                ) = *data
                                .offset(
                                    x_1.wrapping_mul((*self_0).dimension).wrapping_add(d)
                                        as isize,
                                );
                            d = d.wrapping_add(1);
                        }
                    }
                    c = c.wrapping_add(1);
                }
            }
            _ => {
                abort();
            }
        }
        c = 0 as libc::c_int as vl_uint32;
        while (c as libc::c_ulonglong) < (*self_0).numCenters {
            let mut distance_0: libc::c_double = distFn
                .expect(
                    "non-null function pointer",
                )(
                (*self_0).dimension,
                newCenters
                    .offset(
                        (c as libc::c_ulonglong).wrapping_mul((*self_0).dimension)
                            as isize,
                    ),
                ((*self_0).centers as *mut libc::c_double)
                    .offset(
                        (c as libc::c_ulonglong).wrapping_mul((*self_0).dimension)
                            as isize,
                    ),
            );
            *centerToNewCenterDistances.offset(c as isize) = distance_0;
            numDistanceComputationsToNewCenters = (numDistanceComputationsToNewCenters
                as libc::c_ulonglong)
                .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as vl_size
                as vl_size;
            c = c.wrapping_add(1);
        }
        let mut tmp: *mut libc::c_double = (*self_0).centers as *mut libc::c_double;
        (*self_0).centers = newCenters as *mut libc::c_void;
        newCenters = tmp;
        numDistanceComputationsToRefreshCenterDistances = (numDistanceComputationsToRefreshCenterDistances
            as libc::c_double + _vl_kmeans_update_center_distances_d(self_0)) as vl_size;
        c = 0 as libc::c_int as vl_uint32;
        while (c as libc::c_ulonglong) < (*self_0).numCenters {
            *nextCenterDistances.offset(c as isize) = vl_infinity_d.value;
            j = 0 as libc::c_int as vl_uint32;
            while (j as libc::c_ulonglong) < (*self_0).numCenters {
                if !(j == c) {
                    *nextCenterDistances
                        .offset(
                            c as isize,
                        ) = if *nextCenterDistances.offset(c as isize)
                        < *((*self_0).centerDistances as *mut libc::c_double)
                            .offset(
                                (j as libc::c_ulonglong)
                                    .wrapping_add(
                                        (c as libc::c_ulonglong).wrapping_mul((*self_0).numCenters),
                                    ) as isize,
                            )
                    {
                        *nextCenterDistances.offset(c as isize)
                    } else {
                        *((*self_0).centerDistances as *mut libc::c_double)
                            .offset(
                                (j as libc::c_ulonglong)
                                    .wrapping_add(
                                        (c as libc::c_ulonglong).wrapping_mul((*self_0).numCenters),
                                    ) as isize,
                            )
                    };
                }
                j = j.wrapping_add(1);
            }
            c = c.wrapping_add(1);
        }
        x = 0 as libc::c_int as vl_index;
        while x < numData as libc::c_int as libc::c_longlong {
            let mut a: libc::c_double = *pointToClosestCenterUB.offset(x as isize);
            let mut b: libc::c_double = *centerToNewCenterDistances
                .offset(*assignments.offset(x as isize) as isize);
            if (*self_0).distance as libc::c_uint
                == VlDistanceL1 as libc::c_int as libc::c_uint
            {
                *pointToClosestCenterUB.offset(x as isize) = a + b;
            } else {
                let mut sqrtab: libc::c_double = sqrt(a * b);
                *pointToClosestCenterUB.offset(x as isize) = a + b + 2.0f64 * sqrtab;
            }
            *pointToClosestCenterUBIsStrict.offset(x as isize) = 0 as libc::c_int;
            x += 1;
        }
        x = 0 as libc::c_int as vl_index;
        while x < numData as libc::c_int as libc::c_longlong {
            c = 0 as libc::c_int as vl_uint32;
            while (c as libc::c_ulonglong) < (*self_0).numCenters {
                let mut a_0: libc::c_double = *pointToCenterLB
                    .offset(
                        (c as libc::c_ulonglong)
                            .wrapping_add(
                                (x as libc::c_ulonglong).wrapping_mul((*self_0).numCenters),
                            ) as isize,
                    );
                let mut b_0: libc::c_double = *centerToNewCenterDistances
                    .offset(c as isize);
                if a_0 < b_0 {
                    *pointToCenterLB
                        .offset(
                            (c as libc::c_ulonglong)
                                .wrapping_add(
                                    (x as libc::c_ulonglong).wrapping_mul((*self_0).numCenters),
                                ) as isize,
                        ) = 0 as libc::c_int as libc::c_double;
                } else if (*self_0).distance as libc::c_uint
                    == VlDistanceL1 as libc::c_int as libc::c_uint
                {
                    *pointToCenterLB
                        .offset(
                            (c as libc::c_ulonglong)
                                .wrapping_add(
                                    (x as libc::c_ulonglong).wrapping_mul((*self_0).numCenters),
                                ) as isize,
                        ) = a_0 - b_0;
                } else {
                    let mut sqrtab_0: libc::c_double = sqrt(a_0 * b_0);
                    *pointToCenterLB
                        .offset(
                            (c as libc::c_ulonglong)
                                .wrapping_add(
                                    (x as libc::c_ulonglong).wrapping_mul((*self_0).numCenters),
                                ) as isize,
                        ) = a_0 + b_0 - 2.0f64 * sqrtab_0;
                }
                c = c.wrapping_add(1);
            }
            x += 1;
        }
        allDone = 1 as libc::c_int;
        x = 0 as libc::c_int as vl_index;
        while x < numData as libc::c_int as libc::c_longlong {
            if !((if (*self_0).distance as libc::c_uint
                == VlDistanceL1 as libc::c_int as libc::c_uint
            {
                2.0f64
            } else {
                4.0f64
            }) * *pointToClosestCenterUB.offset(x as isize)
                <= *nextCenterDistances.offset(*assignments.offset(x as isize) as isize))
            {
                let mut current_block_141: u64;
                c = 0 as libc::c_int as vl_uint32;
                while (c as libc::c_ulonglong) < (*self_0).numCenters {
                    let mut cx: vl_uint32 = *assignments.offset(x as isize);
                    let mut distance_1: libc::c_double = 0.;
                    if !(cx == c) {
                        if !((if (*self_0).distance as libc::c_uint
                            == VlDistanceL1 as libc::c_int as libc::c_uint
                        {
                            2.0f64
                        } else {
                            4.0f64
                        }) * *pointToClosestCenterUB.offset(x as isize)
                            <= *((*self_0).centerDistances as *mut libc::c_double)
                                .offset(
                                    (c as libc::c_ulonglong)
                                        .wrapping_add(
                                            (cx as libc::c_ulonglong).wrapping_mul((*self_0).numCenters),
                                        ) as isize,
                                ))
                        {
                            if !(*pointToClosestCenterUB.offset(x as isize)
                                <= *pointToCenterLB
                                    .offset(
                                        (c as libc::c_ulonglong)
                                            .wrapping_add(
                                                (x as libc::c_ulonglong).wrapping_mul((*self_0).numCenters),
                                            ) as isize,
                                    ))
                            {
                                if *pointToClosestCenterUBIsStrict.offset(x as isize) == 0 {
                                    distance_1 = distFn
                                        .expect(
                                            "non-null function pointer",
                                        )(
                                        (*self_0).dimension,
                                        data
                                            .offset(
                                                ((*self_0).dimension).wrapping_mul(x as libc::c_ulonglong)
                                                    as isize,
                                            ),
                                        ((*self_0).centers as *mut libc::c_double)
                                            .offset(
                                                ((*self_0).dimension).wrapping_mul(cx as libc::c_ulonglong)
                                                    as isize,
                                            ),
                                    );
                                    *pointToClosestCenterUB.offset(x as isize) = distance_1;
                                    *pointToClosestCenterUBIsStrict
                                        .offset(x as isize) = 1 as libc::c_int;
                                    *pointToCenterLB
                                        .offset(
                                            (cx as libc::c_ulonglong)
                                                .wrapping_add(
                                                    (x as libc::c_ulonglong).wrapping_mul((*self_0).numCenters),
                                                ) as isize,
                                        ) = distance_1;
                                    numDistanceComputationsToRefreshUB = (numDistanceComputationsToRefreshUB
                                        as libc::c_ulonglong)
                                        .wrapping_add(1 as libc::c_int as libc::c_ulonglong)
                                        as vl_size as vl_size;
                                    if (if (*self_0).distance as libc::c_uint
                                        == VlDistanceL1 as libc::c_int as libc::c_uint
                                    {
                                        2.0f64
                                    } else {
                                        4.0f64
                                    }) * *pointToClosestCenterUB.offset(x as isize)
                                        <= *((*self_0).centerDistances as *mut libc::c_double)
                                            .offset(
                                                (c as libc::c_ulonglong)
                                                    .wrapping_add(
                                                        (cx as libc::c_ulonglong).wrapping_mul((*self_0).numCenters),
                                                    ) as isize,
                                            )
                                    {
                                        current_block_141 = 12608488225262500095;
                                    } else if *pointToClosestCenterUB.offset(x as isize)
                                        <= *pointToCenterLB
                                            .offset(
                                                (c as libc::c_ulonglong)
                                                    .wrapping_add(
                                                        (x as libc::c_ulonglong).wrapping_mul((*self_0).numCenters),
                                                    ) as isize,
                                            )
                                    {
                                        current_block_141 = 12608488225262500095;
                                    } else {
                                        current_block_141 = 12223373342341601825;
                                    }
                                } else {
                                    current_block_141 = 12223373342341601825;
                                }
                                match current_block_141 {
                                    12608488225262500095 => {}
                                    _ => {
                                        distance_1 = distFn
                                            .expect(
                                                "non-null function pointer",
                                            )(
                                            (*self_0).dimension,
                                            data
                                                .offset(
                                                    (x as libc::c_ulonglong).wrapping_mul((*self_0).dimension)
                                                        as isize,
                                                ),
                                            ((*self_0).centers as *mut libc::c_double)
                                                .offset(
                                                    (c as libc::c_ulonglong).wrapping_mul((*self_0).dimension)
                                                        as isize,
                                                ),
                                        );
                                        numDistanceComputationsToRefreshLB = (numDistanceComputationsToRefreshLB
                                            as libc::c_ulonglong)
                                            .wrapping_add(1 as libc::c_int as libc::c_ulonglong)
                                            as vl_size as vl_size;
                                        *pointToCenterLB
                                            .offset(
                                                (c as libc::c_ulonglong)
                                                    .wrapping_add(
                                                        (x as libc::c_ulonglong).wrapping_mul((*self_0).numCenters),
                                                    ) as isize,
                                            ) = distance_1;
                                        if distance_1 < *pointToClosestCenterUB.offset(x as isize) {
                                            *assignments.offset(x as isize) = c;
                                            *pointToClosestCenterUB.offset(x as isize) = distance_1;
                                            allDone = 0 as libc::c_int;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    c = c.wrapping_add(1);
                }
            }
            x += 1;
        }
        totDistanceComputationsToRefreshUB = (totDistanceComputationsToRefreshUB
            as libc::c_ulonglong)
            .wrapping_add(numDistanceComputationsToRefreshUB) as vl_size as vl_size;
        totDistanceComputationsToRefreshLB = (totDistanceComputationsToRefreshLB
            as libc::c_ulonglong)
            .wrapping_add(numDistanceComputationsToRefreshLB) as vl_size as vl_size;
        totDistanceComputationsToRefreshCenterDistances = (totDistanceComputationsToRefreshCenterDistances
            as libc::c_ulonglong)
            .wrapping_add(numDistanceComputationsToRefreshCenterDistances) as vl_size
            as vl_size;
        totDistanceComputationsToNewCenters = (totDistanceComputationsToNewCenters
            as libc::c_ulonglong)
            .wrapping_add(numDistanceComputationsToNewCenters) as vl_size as vl_size;
        totNumRestartedCenters = (totNumRestartedCenters as libc::c_ulonglong)
            .wrapping_add(numRestartedCenters) as vl_size as vl_size;
        energy = 0 as libc::c_int as libc::c_double;
        x = 0 as libc::c_int as vl_index;
        while x < numData as libc::c_int as libc::c_longlong {
            energy += *pointToClosestCenterUB.offset(x as isize);
            x += 1;
        }
        if (*self_0).verbosity != 0 {
            let mut numDistanceComputations: vl_size = numDistanceComputationsToRefreshUB
                .wrapping_add(numDistanceComputationsToRefreshLB)
                .wrapping_add(numDistanceComputationsToRefreshCenterDistances)
                .wrapping_add(numDistanceComputationsToNewCenters);
            (Some(
                ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                    .expect("non-null function pointer"),
            ))
                .expect(
                    "non-null function pointer",
                )(
                b"kmeans: Elkan iter %d: energy <= %g, dist. calc. = %d\n\0" as *const u8
                    as *const libc::c_char,
                iteration,
                energy,
                numDistanceComputations,
            );
            if numRestartedCenters != 0 {
                (Some(
                    ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                        .expect("non-null function pointer"),
                ))
                    .expect(
                        "non-null function pointer",
                    )(
                    b"kmeans: Elkan iter %d: restarted %d centers\n\0" as *const u8
                        as *const libc::c_char,
                    iteration,
                    energy,
                    numRestartedCenters,
                );
            }
            if (*self_0).verbosity > 1 as libc::c_int {
                (Some(
                    ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                        .expect("non-null function pointer"),
                ))
                    .expect(
                        "non-null function pointer",
                    )(
                    b"kmeans: Elkan iter %d: total dist. calc. per type: UB: %.1f%% (%d), LB: %.1f%% (%d), intra_center: %.1f%% (%d), new_center: %.1f%% (%d)\n\0"
                        as *const u8 as *const libc::c_char,
                    iteration,
                    100.0f64 * numDistanceComputationsToRefreshUB as libc::c_double
                        / numDistanceComputations as libc::c_double,
                    numDistanceComputationsToRefreshUB,
                    100.0f64 * numDistanceComputationsToRefreshLB as libc::c_double
                        / numDistanceComputations as libc::c_double,
                    numDistanceComputationsToRefreshLB,
                    100.0f64
                        * numDistanceComputationsToRefreshCenterDistances
                            as libc::c_double
                        / numDistanceComputations as libc::c_double,
                    numDistanceComputationsToRefreshCenterDistances,
                    100.0f64 * numDistanceComputationsToNewCenters as libc::c_double
                        / numDistanceComputations as libc::c_double,
                    numDistanceComputationsToNewCenters,
                );
            }
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
                    b"kmeans: Elkan terminating because maximum number of iterations reached\n\0"
                        as *const u8 as *const libc::c_char,
                );
            }
            break;
        } else if allDone != 0 {
            if (*self_0).verbosity != 0 {
                (Some(
                    ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                        .expect("non-null function pointer"),
                ))
                    .expect(
                        "non-null function pointer",
                    )(
                    b"kmeans: Elkan terminating because the algorithm fully converged\n\0"
                        as *const u8 as *const libc::c_char,
                );
            }
            break;
        } else {
            iteration = iteration.wrapping_add(1);
        }
    }
    energy = 0 as libc::c_int as libc::c_double;
    x = 0 as libc::c_int as vl_index;
    while x < numData as libc::c_int as libc::c_longlong {
        let mut cx_0: vl_uindex = *assignments.offset(x as isize) as vl_uindex;
        energy
            += distFn
                .expect(
                    "non-null function pointer",
                )(
                (*self_0).dimension,
                data
                    .offset(
                        ((*self_0).dimension).wrapping_mul(x as libc::c_ulonglong)
                            as isize,
                    ),
                ((*self_0).centers as *mut libc::c_double)
                    .offset(((*self_0).dimension).wrapping_mul(cx_0) as isize),
            );
        totDistanceComputationsToFinalize = (totDistanceComputationsToFinalize
            as libc::c_ulonglong)
            .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as vl_size as vl_size;
        x += 1;
    }
    let mut totDistanceComputations: vl_size = totDistanceComputationsToInit
        .wrapping_add(totDistanceComputationsToRefreshUB)
        .wrapping_add(totDistanceComputationsToRefreshLB)
        .wrapping_add(totDistanceComputationsToRefreshCenterDistances)
        .wrapping_add(totDistanceComputationsToNewCenters)
        .wrapping_add(totDistanceComputationsToFinalize);
    let mut saving: libc::c_double = totDistanceComputations as libc::c_double
        / iteration.wrapping_mul((*self_0).numCenters).wrapping_mul(numData)
            as libc::c_double;
    if (*self_0).verbosity != 0 {
        (Some(
            ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                .expect("non-null function pointer"),
        ))
            .expect(
                "non-null function pointer",
            )(
            b"kmeans: Elkan: total dist. calc.: %d (%.2f %% of Lloyd)\n\0" as *const u8
                as *const libc::c_char,
            totDistanceComputations,
            saving * 100.0f64,
        );
        if totNumRestartedCenters != 0 {
            (Some(
                ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                    .expect("non-null function pointer"),
            ))
                .expect(
                    "non-null function pointer",
                )(
                b"kmeans: Elkan: there have been %d restarts\n\0" as *const u8
                    as *const libc::c_char,
                totNumRestartedCenters,
            );
        }
    }
    if (*self_0).verbosity > 1 as libc::c_int {
        (Some(
            ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                .expect("non-null function pointer"),
        ))
            .expect(
                "non-null function pointer",
            )(
            b"kmeans: Elkan: total dist. calc. per type: init: %.1f%% (%d), UB: %.1f%% (%d), LB: %.1f%% (%d), intra_center: %.1f%% (%d), new_center: %.1f%% (%d), finalize: %.1f%% (%d)\n\0"
                as *const u8 as *const libc::c_char,
            100.0f64 * totDistanceComputationsToInit as libc::c_double
                / totDistanceComputations as libc::c_double,
            totDistanceComputationsToInit,
            100.0f64 * totDistanceComputationsToRefreshUB as libc::c_double
                / totDistanceComputations as libc::c_double,
            totDistanceComputationsToRefreshUB,
            100.0f64 * totDistanceComputationsToRefreshLB as libc::c_double
                / totDistanceComputations as libc::c_double,
            totDistanceComputationsToRefreshLB,
            100.0f64 * totDistanceComputationsToRefreshCenterDistances as libc::c_double
                / totDistanceComputations as libc::c_double,
            totDistanceComputationsToRefreshCenterDistances,
            100.0f64 * totDistanceComputationsToNewCenters as libc::c_double
                / totDistanceComputations as libc::c_double,
            totDistanceComputationsToNewCenters,
            100.0f64 * totDistanceComputationsToFinalize as libc::c_double
                / totDistanceComputations as libc::c_double,
            totDistanceComputationsToFinalize,
        );
    }
    if !permutations.is_null() {
        vl_free(permutations as *mut libc::c_void);
    }
    if !numSeenSoFar.is_null() {
        vl_free(numSeenSoFar as *mut libc::c_void);
    }
    vl_free(distances as *mut libc::c_void);
    vl_free(assignments as *mut libc::c_void);
    vl_free(clusterMasses as *mut libc::c_void);
    vl_free(nextCenterDistances as *mut libc::c_void);
    vl_free(pointToClosestCenterUB as *mut libc::c_void);
    vl_free(pointToClosestCenterUBIsStrict as *mut libc::c_void);
    vl_free(pointToCenterLB as *mut libc::c_void);
    vl_free(newCenters as *mut libc::c_void);
    vl_free(centerToNewCenterDistances as *mut libc::c_void);
    return energy;
}
unsafe extern "C" fn _vl_kmeans_refine_centers_f(
    mut self_0: *mut VlKMeans,
    mut data: *const libc::c_float,
    mut numData: vl_size,
) -> libc::c_double {
    match (*self_0).algorithm as libc::c_uint {
        0 => return _vl_kmeans_refine_centers_lloyd_f(self_0, data, numData),
        1 => return _vl_kmeans_refine_centers_elkan_f(self_0, data, numData),
        2 => return _vl_kmeans_refine_centers_ann_f(self_0, data, numData),
        _ => {
            abort();
        }
    };
}
unsafe extern "C" fn _vl_kmeans_refine_centers_d(
    mut self_0: *mut VlKMeans,
    mut data: *const libc::c_double,
    mut numData: vl_size,
) -> libc::c_double {
    match (*self_0).algorithm as libc::c_uint {
        0 => return _vl_kmeans_refine_centers_lloyd_d(self_0, data, numData),
        1 => return _vl_kmeans_refine_centers_elkan_d(self_0, data, numData),
        2 => return _vl_kmeans_refine_centers_ann_d(self_0, data, numData),
        _ => {
            abort();
        }
    };
}
#[inline]
unsafe extern "C" fn _vl_kmeans_f_qsort_sort_recursive(
    mut array: *mut VlKMeansSortWrapper,
    mut begin: vl_uindex,
    mut end: vl_uindex,
) {
    let mut pivot: vl_uindex = end
        .wrapping_add(begin)
        .wrapping_div(2 as libc::c_int as libc::c_ulonglong);
    let mut lowPart: vl_uindex = 0;
    let mut i: vl_uindex = 0;
    if begin <= end {} else {
        __assert_fail(
            b"begin <= end\0" as *const u8 as *const libc::c_char,
            b"vl/qsort-def.h\0" as *const u8 as *const libc::c_char,
            133 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 84],
                &[libc::c_char; 84],
            >(
                b"void _vl_kmeans_f_qsort_sort_recursive(VlKMeansSortWrapper *, vl_uindex, vl_uindex)\0",
            ))
                .as_ptr(),
        );
    }
    _vl_kmeans_f_qsort_swap(array, pivot, end);
    pivot = end;
    lowPart = begin;
    i = begin;
    while i < end {
        if _vl_kmeans_f_qsort_cmp(array, i, pivot) <= 0 as libc::c_int as libc::c_float {
            _vl_kmeans_f_qsort_swap(array, lowPart, i);
            lowPart = lowPart.wrapping_add(1);
        }
        i = i.wrapping_add(1);
    }
    _vl_kmeans_f_qsort_swap(array, lowPart, pivot);
    pivot = lowPart;
    if pivot > begin {
        _vl_kmeans_f_qsort_sort_recursive(
            array,
            begin,
            pivot.wrapping_sub(1 as libc::c_int as libc::c_ulonglong),
        );
    }
    if pivot < end {
        _vl_kmeans_f_qsort_sort_recursive(
            array,
            pivot.wrapping_add(1 as libc::c_int as libc::c_ulonglong),
            end,
        );
    }
}
#[inline]
unsafe extern "C" fn _vl_kmeans_d_qsort_sort_recursive(
    mut array: *mut VlKMeansSortWrapper,
    mut begin: vl_uindex,
    mut end: vl_uindex,
) {
    let mut pivot: vl_uindex = end
        .wrapping_add(begin)
        .wrapping_div(2 as libc::c_int as libc::c_ulonglong);
    let mut lowPart: vl_uindex = 0;
    let mut i: vl_uindex = 0;
    if begin <= end {} else {
        __assert_fail(
            b"begin <= end\0" as *const u8 as *const libc::c_char,
            b"vl/qsort-def.h\0" as *const u8 as *const libc::c_char,
            133 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 84],
                &[libc::c_char; 84],
            >(
                b"void _vl_kmeans_d_qsort_sort_recursive(VlKMeansSortWrapper *, vl_uindex, vl_uindex)\0",
            ))
                .as_ptr(),
        );
    }
    _vl_kmeans_d_qsort_swap(array, pivot, end);
    pivot = end;
    lowPart = begin;
    i = begin;
    while i < end {
        if _vl_kmeans_d_qsort_cmp(array, i, pivot) <= 0 as libc::c_int as libc::c_double
        {
            _vl_kmeans_d_qsort_swap(array, lowPart, i);
            lowPart = lowPart.wrapping_add(1);
        }
        i = i.wrapping_add(1);
    }
    _vl_kmeans_d_qsort_swap(array, lowPart, pivot);
    pivot = lowPart;
    if pivot > begin {
        _vl_kmeans_d_qsort_sort_recursive(
            array,
            begin,
            pivot.wrapping_sub(1 as libc::c_int as libc::c_ulonglong),
        );
    }
    if pivot < end {
        _vl_kmeans_d_qsort_sort_recursive(
            array,
            pivot.wrapping_add(1 as libc::c_int as libc::c_ulonglong),
            end,
        );
    }
}
#[inline]
unsafe extern "C" fn _vl_kmeans_f_qsort_sort(
    mut array: *mut VlKMeansSortWrapper,
    mut size: vl_size,
) {
    if size >= 1 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"size >= 1\0" as *const u8 as *const libc::c_char,
            b"vl/qsort-def.h\0" as *const u8 as *const libc::c_char,
            186 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 61],
                &[libc::c_char; 61],
            >(b"void _vl_kmeans_f_qsort_sort(VlKMeansSortWrapper *, vl_size)\0"))
                .as_ptr(),
        );
    }
    _vl_kmeans_f_qsort_sort_recursive(
        array,
        0 as libc::c_int as vl_uindex,
        size.wrapping_sub(1 as libc::c_int as libc::c_ulonglong),
    );
}
#[inline]
unsafe extern "C" fn _vl_kmeans_d_qsort_sort(
    mut array: *mut VlKMeansSortWrapper,
    mut size: vl_size,
) {
    if size >= 1 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"size >= 1\0" as *const u8 as *const libc::c_char,
            b"vl/qsort-def.h\0" as *const u8 as *const libc::c_char,
            186 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 61],
                &[libc::c_char; 61],
            >(b"void _vl_kmeans_d_qsort_sort(VlKMeansSortWrapper *, vl_size)\0"))
                .as_ptr(),
        );
    }
    _vl_kmeans_d_qsort_sort_recursive(
        array,
        0 as libc::c_int as vl_uindex,
        size.wrapping_sub(1 as libc::c_int as libc::c_ulonglong),
    );
}
#[no_mangle]
pub unsafe extern "C" fn vl_kmeans_set_centers(
    mut self_0: *mut VlKMeans,
    mut centers: *const libc::c_void,
    mut dimension: vl_size,
    mut numCenters: vl_size,
) {
    vl_kmeans_reset(self_0);
    match (*self_0).dataType {
        1 => {
            _vl_kmeans_set_centers_f(
                self_0,
                centers as *const libc::c_float,
                dimension,
                numCenters,
            );
        }
        2 => {
            _vl_kmeans_set_centers_d(
                self_0,
                centers as *const libc::c_double,
                dimension,
                numCenters,
            );
        }
        _ => {
            abort();
        }
    };
}
#[no_mangle]
pub unsafe extern "C" fn vl_kmeans_init_centers_with_rand_data(
    mut self_0: *mut VlKMeans,
    mut data: *const libc::c_void,
    mut dimension: vl_size,
    mut numData: vl_size,
    mut numCenters: vl_size,
) {
    vl_kmeans_reset(self_0);
    match (*self_0).dataType {
        1 => {
            _vl_kmeans_init_centers_with_rand_data_f(
                self_0,
                data as *const libc::c_float,
                dimension,
                numData,
                numCenters,
            );
        }
        2 => {
            _vl_kmeans_init_centers_with_rand_data_d(
                self_0,
                data as *const libc::c_double,
                dimension,
                numData,
                numCenters,
            );
        }
        _ => {
            abort();
        }
    };
}
#[no_mangle]
pub unsafe extern "C" fn vl_kmeans_init_centers_plus_plus(
    mut self_0: *mut VlKMeans,
    mut data: *const libc::c_void,
    mut dimension: vl_size,
    mut numData: vl_size,
    mut numCenters: vl_size,
) {
    vl_kmeans_reset(self_0);
    match (*self_0).dataType {
        1 => {
            _vl_kmeans_init_centers_plus_plus_f(
                self_0,
                data as *const libc::c_float,
                dimension,
                numData,
                numCenters,
            );
        }
        2 => {
            _vl_kmeans_init_centers_plus_plus_d(
                self_0,
                data as *const libc::c_double,
                dimension,
                numData,
                numCenters,
            );
        }
        _ => {
            abort();
        }
    };
}
#[no_mangle]
pub unsafe extern "C" fn vl_kmeans_quantize(
    mut self_0: *mut VlKMeans,
    mut assignments: *mut vl_uint32,
    mut distances: *mut libc::c_void,
    mut data: *const libc::c_void,
    mut numData: vl_size,
) {
    match (*self_0).dataType {
        1 => {
            _vl_kmeans_quantize_f(
                self_0,
                assignments,
                distances as *mut libc::c_float,
                data as *const libc::c_float,
                numData,
            );
        }
        2 => {
            _vl_kmeans_quantize_d(
                self_0,
                assignments,
                distances as *mut libc::c_double,
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
pub unsafe extern "C" fn vl_kmeans_quantize_ann(
    mut self_0: *mut VlKMeans,
    mut assignments: *mut vl_uint32,
    mut distances: *mut libc::c_void,
    mut data: *const libc::c_void,
    mut numData: vl_size,
    mut update: vl_bool,
) {
    match (*self_0).dataType {
        1 => {
            _vl_kmeans_quantize_ann_f(
                self_0,
                assignments,
                distances as *mut libc::c_float,
                data as *const libc::c_float,
                numData,
                update,
            );
        }
        2 => {
            _vl_kmeans_quantize_ann_d(
                self_0,
                assignments,
                distances as *mut libc::c_double,
                data as *const libc::c_double,
                numData,
                update,
            );
        }
        _ => {
            abort();
        }
    };
}
#[no_mangle]
pub unsafe extern "C" fn vl_kmeans_refine_centers(
    mut self_0: *mut VlKMeans,
    mut data: *const libc::c_void,
    mut numData: vl_size,
) -> libc::c_double {
    if !((*self_0).centers).is_null() {} else {
        __assert_fail(
            b"self->centers\0" as *const u8 as *const libc::c_char,
            b"vl/kmeans.c\0" as *const u8 as *const libc::c_char,
            1989 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 67],
                &[libc::c_char; 67],
            >(b"double vl_kmeans_refine_centers(VlKMeans *, const void *, vl_size)\0"))
                .as_ptr(),
        );
    }
    match (*self_0).dataType {
        1 => {
            return _vl_kmeans_refine_centers_f(
                self_0,
                data as *const libc::c_float,
                numData,
            );
        }
        2 => {
            return _vl_kmeans_refine_centers_d(
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
pub unsafe extern "C" fn vl_kmeans_cluster(
    mut self_0: *mut VlKMeans,
    mut data: *const libc::c_void,
    mut dimension: vl_size,
    mut numData: vl_size,
    mut numCenters: vl_size,
) -> libc::c_double {
    let mut repetition: vl_uindex = 0;
    let mut bestEnergy: libc::c_double = vl_infinity_d.value;
    let mut bestCenters: *mut libc::c_void = 0 as *mut libc::c_void;
    repetition = 0 as libc::c_int as vl_uindex;
    while repetition < (*self_0).numRepetitions {
        let mut energy: libc::c_double = 0.;
        let mut timeRef: libc::c_double = 0.;
        if (*self_0).verbosity != 0 {
            (Some(
                ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                    .expect("non-null function pointer"),
            ))
                .expect(
                    "non-null function pointer",
                )(
                b"kmeans: repetition %d of %d\n\0" as *const u8 as *const libc::c_char,
                repetition.wrapping_add(1 as libc::c_int as libc::c_ulonglong),
                (*self_0).numRepetitions,
            );
        }
        timeRef = vl_get_cpu_time();
        match (*self_0).initialization as libc::c_uint {
            0 => {
                vl_kmeans_init_centers_with_rand_data(
                    self_0,
                    data,
                    dimension,
                    numData,
                    numCenters,
                );
            }
            1 => {
                vl_kmeans_init_centers_plus_plus(
                    self_0,
                    data,
                    dimension,
                    numData,
                    numCenters,
                );
            }
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
                b"kmeans: K-means initialized in %.2f s\n\0" as *const u8
                    as *const libc::c_char,
                vl_get_cpu_time() - timeRef,
            );
        }
        timeRef = vl_get_cpu_time();
        energy = vl_kmeans_refine_centers(self_0, data, numData);
        if (*self_0).verbosity != 0 {
            (Some(
                ((vl_get_printf_func as unsafe extern "C" fn() -> printf_func_t)())
                    .expect("non-null function pointer"),
            ))
                .expect(
                    "non-null function pointer",
                )(
                b"kmeans: K-means terminated in %.2f s with energy %g\n\0" as *const u8
                    as *const libc::c_char,
                vl_get_cpu_time() - timeRef,
                energy,
            );
        }
        if energy < bestEnergy || repetition == 0 as libc::c_int as libc::c_ulonglong {
            let mut temp: *mut libc::c_void = 0 as *mut libc::c_void;
            bestEnergy = energy;
            if bestCenters.is_null() {
                bestCenters = vl_malloc(
                    (vl_get_type_size((*self_0).dataType))
                        .wrapping_mul((*self_0).dimension)
                        .wrapping_mul((*self_0).numCenters) as size_t,
                );
            }
            temp = bestCenters;
            bestCenters = (*self_0).centers;
            (*self_0).centers = temp;
        }
        repetition = repetition.wrapping_add(1);
    }
    vl_free((*self_0).centers);
    (*self_0).centers = bestCenters;
    return bestEnergy;
}
