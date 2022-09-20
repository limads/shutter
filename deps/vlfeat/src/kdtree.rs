use ::libc;
extern "C" {
    fn vl_get_vector_comparison_function_d(
        type_0: VlVectorComparisonType,
    ) -> VlDoubleVectorComparisonFunction;
    fn vl_get_vector_comparison_function_f(
        type_0: VlVectorComparisonType,
    ) -> VlFloatVectorComparisonFunction;
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
    fn vl_rand_uint32(self_0: *mut VlRand) -> vl_uint32;
    fn qsort(
        __base: *mut libc::c_void,
        __nmemb: size_t,
        __size: size_t,
        __compar: __compar_fn_t,
    );
    fn abort() -> !;
}
pub type vl_int64 = libc::c_longlong;
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
pub type __compar_fn_t = Option::<
    unsafe extern "C" fn(*const libc::c_void, *const libc::c_void) -> libc::c_int,
>;
pub type vl_type = vl_uint32;
#[derive(Copy, Clone)]
#[repr(C)]
pub union C2RustUnnamed {
    pub raw: vl_uint32,
    pub value: libc::c_float,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub union C2RustUnnamed_0 {
    pub raw: vl_uint32,
    pub value: libc::c_float,
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
static mut vl_nan_f: C2RustUnnamed = C2RustUnnamed {
    raw: 0x7fc00000 as libc::c_ulong as vl_uint32,
};
static mut vl_infinity_f: C2RustUnnamed_0 = C2RustUnnamed_0 {
    raw: 0x7f800000 as libc::c_ulong as vl_uint32,
};
#[inline]
unsafe extern "C" fn vl_heap_parent(mut index: vl_uindex) -> vl_uindex {
    if index == 0 as libc::c_int as libc::c_ulonglong {
        return 0 as libc::c_int as vl_uindex;
    }
    return index
        .wrapping_sub(1 as libc::c_int as libc::c_ulonglong)
        .wrapping_div(2 as libc::c_int as libc::c_ulonglong);
}
#[inline]
unsafe extern "C" fn vl_heap_left_child(mut index: vl_uindex) -> vl_uindex {
    return (2 as libc::c_int as libc::c_ulonglong)
        .wrapping_mul(index)
        .wrapping_add(1 as libc::c_int as libc::c_ulonglong);
}
#[inline]
unsafe extern "C" fn vl_heap_right_child(mut index: vl_uindex) -> vl_uindex {
    return (vl_heap_left_child(index))
        .wrapping_add(1 as libc::c_int as libc::c_ulonglong);
}
#[inline]
unsafe extern "C" fn vl_kdforest_neighbor_heap_swap(
    mut array: *mut VlKDForestNeighbor,
    mut indexA: vl_uindex,
    mut indexB: vl_uindex,
) {
    let mut t: VlKDForestNeighbor = *array.offset(indexA as isize);
    *array.offset(indexA as isize) = *array.offset(indexB as isize);
    *array.offset(indexB as isize) = t;
}
#[inline]
unsafe extern "C" fn vl_kdforest_search_heap_swap(
    mut array: *mut VlKDForestSearchState,
    mut indexA: vl_uindex,
    mut indexB: vl_uindex,
) {
    let mut t: VlKDForestSearchState = *array.offset(indexA as isize);
    *array.offset(indexA as isize) = *array.offset(indexB as isize);
    *array.offset(indexB as isize) = t;
}
#[inline]
unsafe extern "C" fn vl_kdtree_split_heap_swap(
    mut array: *mut VlKDTreeSplitDimension,
    mut indexA: vl_uindex,
    mut indexB: vl_uindex,
) {
    let mut t: VlKDTreeSplitDimension = *array.offset(indexA as isize);
    *array.offset(indexA as isize) = *array.offset(indexB as isize);
    *array.offset(indexB as isize) = t;
}
#[inline]
unsafe extern "C" fn vl_kdforest_search_heap_up(
    mut array: *mut VlKDForestSearchState,
    mut heapSize: vl_size,
    mut index: vl_uindex,
) {
    let mut leftIndex: vl_uindex = vl_heap_left_child(index);
    let mut rightIndex: vl_uindex = vl_heap_right_child(index);
    if leftIndex >= heapSize {
        return;
    }
    if rightIndex >= heapSize {
        if (*array.offset(index as isize)).distanceLowerBound
            - (*array.offset(leftIndex as isize)).distanceLowerBound
            > 0 as libc::c_int as libc::c_double
        {
            vl_kdforest_search_heap_swap(array, index, leftIndex);
        }
        return;
    }
    if (*array.offset(leftIndex as isize)).distanceLowerBound
        - (*array.offset(rightIndex as isize)).distanceLowerBound
        < 0 as libc::c_int as libc::c_double
    {
        if (*array.offset(index as isize)).distanceLowerBound
            - (*array.offset(leftIndex as isize)).distanceLowerBound
            > 0 as libc::c_int as libc::c_double
        {
            vl_kdforest_search_heap_swap(array, index, leftIndex);
            vl_kdforest_search_heap_up(array, heapSize, leftIndex);
        }
    } else if (*array.offset(index as isize)).distanceLowerBound
        - (*array.offset(rightIndex as isize)).distanceLowerBound
        > 0 as libc::c_int as libc::c_double
    {
        vl_kdforest_search_heap_swap(array, index, rightIndex);
        vl_kdforest_search_heap_up(array, heapSize, rightIndex);
    }
}
#[inline]
unsafe extern "C" fn vl_kdforest_neighbor_heap_up(
    mut array: *mut VlKDForestNeighbor,
    mut heapSize: vl_size,
    mut index: vl_uindex,
) {
    let mut leftIndex: vl_uindex = vl_heap_left_child(index);
    let mut rightIndex: vl_uindex = vl_heap_right_child(index);
    if leftIndex >= heapSize {
        return;
    }
    if rightIndex >= heapSize {
        if (*array.offset(leftIndex as isize)).distance
            - (*array.offset(index as isize)).distance
            > 0 as libc::c_int as libc::c_double
        {
            vl_kdforest_neighbor_heap_swap(array, index, leftIndex);
        }
        return;
    }
    if (*array.offset(rightIndex as isize)).distance
        - (*array.offset(leftIndex as isize)).distance
        < 0 as libc::c_int as libc::c_double
    {
        if (*array.offset(leftIndex as isize)).distance
            - (*array.offset(index as isize)).distance
            > 0 as libc::c_int as libc::c_double
        {
            vl_kdforest_neighbor_heap_swap(array, index, leftIndex);
            vl_kdforest_neighbor_heap_up(array, heapSize, leftIndex);
        }
    } else if (*array.offset(rightIndex as isize)).distance
        - (*array.offset(index as isize)).distance > 0 as libc::c_int as libc::c_double
    {
        vl_kdforest_neighbor_heap_swap(array, index, rightIndex);
        vl_kdforest_neighbor_heap_up(array, heapSize, rightIndex);
    }
}
#[inline]
unsafe extern "C" fn vl_kdtree_split_heap_up(
    mut array: *mut VlKDTreeSplitDimension,
    mut heapSize: vl_size,
    mut index: vl_uindex,
) {
    let mut leftIndex: vl_uindex = vl_heap_left_child(index);
    let mut rightIndex: vl_uindex = vl_heap_right_child(index);
    if leftIndex >= heapSize {
        return;
    }
    if rightIndex >= heapSize {
        if (*array.offset(index as isize)).variance
            - (*array.offset(leftIndex as isize)).variance
            > 0 as libc::c_int as libc::c_double
        {
            vl_kdtree_split_heap_swap(array, index, leftIndex);
        }
        return;
    }
    if (*array.offset(leftIndex as isize)).variance
        - (*array.offset(rightIndex as isize)).variance
        < 0 as libc::c_int as libc::c_double
    {
        if (*array.offset(index as isize)).variance
            - (*array.offset(leftIndex as isize)).variance
            > 0 as libc::c_int as libc::c_double
        {
            vl_kdtree_split_heap_swap(array, index, leftIndex);
            vl_kdtree_split_heap_up(array, heapSize, leftIndex);
        }
    } else if (*array.offset(index as isize)).variance
        - (*array.offset(rightIndex as isize)).variance
        > 0 as libc::c_int as libc::c_double
    {
        vl_kdtree_split_heap_swap(array, index, rightIndex);
        vl_kdtree_split_heap_up(array, heapSize, rightIndex);
    }
}
#[inline]
unsafe extern "C" fn vl_kdforest_search_heap_down(
    mut array: *mut VlKDForestSearchState,
    mut index: vl_uindex,
) {
    let mut parentIndex: vl_uindex = 0;
    if index == 0 as libc::c_int as libc::c_ulonglong {
        return;
    }
    parentIndex = vl_heap_parent(index);
    if (*array.offset(index as isize)).distanceLowerBound
        - (*array.offset(parentIndex as isize)).distanceLowerBound
        < 0 as libc::c_int as libc::c_double
    {
        vl_kdforest_search_heap_swap(array, index, parentIndex);
        vl_kdforest_search_heap_down(array, parentIndex);
    }
}
#[inline]
unsafe extern "C" fn vl_kdtree_split_heap_down(
    mut array: *mut VlKDTreeSplitDimension,
    mut index: vl_uindex,
) {
    let mut parentIndex: vl_uindex = 0;
    if index == 0 as libc::c_int as libc::c_ulonglong {
        return;
    }
    parentIndex = vl_heap_parent(index);
    if (*array.offset(index as isize)).variance
        - (*array.offset(parentIndex as isize)).variance
        < 0 as libc::c_int as libc::c_double
    {
        vl_kdtree_split_heap_swap(array, index, parentIndex);
        vl_kdtree_split_heap_down(array, parentIndex);
    }
}
#[inline]
unsafe extern "C" fn vl_kdforest_neighbor_heap_down(
    mut array: *mut VlKDForestNeighbor,
    mut index: vl_uindex,
) {
    let mut parentIndex: vl_uindex = 0;
    if index == 0 as libc::c_int as libc::c_ulonglong {
        return;
    }
    parentIndex = vl_heap_parent(index);
    if (*array.offset(parentIndex as isize)).distance
        - (*array.offset(index as isize)).distance < 0 as libc::c_int as libc::c_double
    {
        vl_kdforest_neighbor_heap_swap(array, index, parentIndex);
        vl_kdforest_neighbor_heap_down(array, parentIndex);
    }
}
#[inline]
unsafe extern "C" fn vl_kdforest_neighbor_heap_push(
    mut array: *mut VlKDForestNeighbor,
    mut heapSize: *mut vl_size,
) {
    vl_kdforest_neighbor_heap_down(array, *heapSize);
    *heapSize = (*heapSize as libc::c_ulonglong)
        .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as vl_size as vl_size;
}
#[inline]
unsafe extern "C" fn vl_kdtree_split_heap_push(
    mut array: *mut VlKDTreeSplitDimension,
    mut heapSize: *mut vl_size,
) {
    vl_kdtree_split_heap_down(array, *heapSize);
    *heapSize = (*heapSize as libc::c_ulonglong)
        .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as vl_size as vl_size;
}
#[inline]
unsafe extern "C" fn vl_kdforest_search_heap_push(
    mut array: *mut VlKDForestSearchState,
    mut heapSize: *mut vl_size,
) {
    vl_kdforest_search_heap_down(array, *heapSize);
    *heapSize = (*heapSize as libc::c_ulonglong)
        .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as vl_size as vl_size;
}
#[inline]
unsafe extern "C" fn vl_kdforest_neighbor_heap_pop(
    mut array: *mut VlKDForestNeighbor,
    mut heapSize: *mut vl_size,
) -> vl_uindex {
    if *heapSize != 0 {} else {
        __assert_fail(
            b"*heapSize\0" as *const u8 as *const libc::c_char,
            b"vl/heap-def.h\0" as *const u8 as *const libc::c_char,
            403 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 73],
                &[libc::c_char; 73],
            >(
                b"vl_uindex vl_kdforest_neighbor_heap_pop(VlKDForestNeighbor *, vl_size *)\0",
            ))
                .as_ptr(),
        );
    }
    *heapSize = (*heapSize as libc::c_ulonglong)
        .wrapping_sub(1 as libc::c_int as libc::c_ulonglong) as vl_size as vl_size;
    vl_kdforest_neighbor_heap_swap(array, 0 as libc::c_int as vl_uindex, *heapSize);
    if *heapSize > 1 as libc::c_int as libc::c_ulonglong {
        vl_kdforest_neighbor_heap_up(array, *heapSize, 0 as libc::c_int as vl_uindex);
    }
    return *heapSize;
}
#[inline]
unsafe extern "C" fn vl_kdforest_search_heap_pop(
    mut array: *mut VlKDForestSearchState,
    mut heapSize: *mut vl_size,
) -> vl_uindex {
    if *heapSize != 0 {} else {
        __assert_fail(
            b"*heapSize\0" as *const u8 as *const libc::c_char,
            b"vl/heap-def.h\0" as *const u8 as *const libc::c_char,
            403 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 74],
                &[libc::c_char; 74],
            >(
                b"vl_uindex vl_kdforest_search_heap_pop(VlKDForestSearchState *, vl_size *)\0",
            ))
                .as_ptr(),
        );
    }
    *heapSize = (*heapSize as libc::c_ulonglong)
        .wrapping_sub(1 as libc::c_int as libc::c_ulonglong) as vl_size as vl_size;
    vl_kdforest_search_heap_swap(array, 0 as libc::c_int as vl_uindex, *heapSize);
    if *heapSize > 1 as libc::c_int as libc::c_ulonglong {
        vl_kdforest_search_heap_up(array, *heapSize, 0 as libc::c_int as vl_uindex);
    }
    return *heapSize;
}
#[inline]
unsafe extern "C" fn vl_kdtree_split_heap_update(
    mut array: *mut VlKDTreeSplitDimension,
    mut heapSize: vl_size,
    mut index: vl_uindex,
) {
    vl_kdtree_split_heap_up(array, heapSize, index);
    vl_kdtree_split_heap_down(array, index);
}
#[inline]
unsafe extern "C" fn vl_kdforest_neighbor_heap_update(
    mut array: *mut VlKDForestNeighbor,
    mut heapSize: vl_size,
    mut index: vl_uindex,
) {
    vl_kdforest_neighbor_heap_up(array, heapSize, index);
    vl_kdforest_neighbor_heap_down(array, index);
}
unsafe extern "C" fn vl_kdtree_node_new(
    mut tree: *mut VlKDTree,
    mut parentIndex: vl_uindex,
) -> vl_uindex {
    let mut node: *mut VlKDTreeNode = 0 as *mut VlKDTreeNode;
    let mut nodeIndex: vl_uindex = (*tree).numUsedNodes;
    (*tree)
        .numUsedNodes = ((*tree).numUsedNodes as libc::c_ulonglong)
        .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as vl_size as vl_size;
    if (*tree).numUsedNodes <= (*tree).numAllocatedNodes {} else {
        __assert_fail(
            b"tree->numUsedNodes <= tree->numAllocatedNodes\0" as *const u8
                as *const libc::c_char,
            b"vl/kdtree.c\0" as *const u8 as *const libc::c_char,
            126 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 52],
                &[libc::c_char; 52],
            >(b"vl_uindex vl_kdtree_node_new(VlKDTree *, vl_uindex)\0"))
                .as_ptr(),
        );
    }
    node = ((*tree).nodes).offset(nodeIndex as isize);
    (*node).parent = parentIndex;
    (*node).lowerChild = 0 as libc::c_int as vl_index;
    (*node).upperChild = 0 as libc::c_int as vl_index;
    (*node).splitDimension = 0 as libc::c_int as libc::c_uint;
    (*node).splitThreshold = 0 as libc::c_int as libc::c_double;
    return nodeIndex;
}
#[inline]
unsafe extern "C" fn vl_kdtree_compare_index_entries(
    mut a: *const libc::c_void,
    mut b: *const libc::c_void,
) -> libc::c_int {
    let mut delta: libc::c_double = (*(a as *const VlKDTreeDataIndexEntry)).value
        - (*(b as *const VlKDTreeDataIndexEntry)).value;
    if delta < 0 as libc::c_int as libc::c_double {
        return -(1 as libc::c_int);
    }
    if delta > 0 as libc::c_int as libc::c_double {
        return 1 as libc::c_int;
    }
    return 0 as libc::c_int;
}
unsafe extern "C" fn vl_kdtree_build_recursively(
    mut forest: *mut VlKDForest,
    mut tree: *mut VlKDTree,
    mut nodeIndex: vl_uindex,
    mut dataBegin: vl_uindex,
    mut dataEnd: vl_uindex,
    mut depth: libc::c_uint,
) {
    let mut d: vl_uindex = 0;
    let mut i: vl_uindex = 0;
    let mut medianIndex: vl_uindex = 0;
    let mut splitIndex: vl_uindex = 0;
    let mut node: *mut VlKDTreeNode = ((*tree).nodes).offset(nodeIndex as isize);
    let mut splitDimension: *mut VlKDTreeSplitDimension = 0
        as *mut VlKDTreeSplitDimension;
    if dataEnd.wrapping_sub(dataBegin) <= 1 as libc::c_int as libc::c_ulonglong {
        if (*tree).depth < depth {
            (*tree).depth = depth;
        }
        (*node)
            .lowerChild = dataBegin
            .wrapping_neg()
            .wrapping_sub(1 as libc::c_int as libc::c_ulonglong) as vl_index;
        (*node)
            .upperChild = dataEnd
            .wrapping_neg()
            .wrapping_sub(1 as libc::c_int as libc::c_ulonglong) as vl_index;
        return;
    }
    (*forest).splitHeapNumNodes = 0 as libc::c_int as vl_size;
    d = 0 as libc::c_int as vl_uindex;
    while d < (*forest).dimension {
        let mut mean: libc::c_double = 0 as libc::c_int as libc::c_double;
        let mut secondMoment: libc::c_double = 0 as libc::c_int as libc::c_double;
        let mut variance: libc::c_double = 0 as libc::c_int as libc::c_double;
        let mut numSamples: vl_size = 1024 as libc::c_int as vl_size;
        let mut useAllData: vl_bool = 0 as libc::c_int;
        if dataEnd.wrapping_sub(dataBegin) <= 1024 as libc::c_int as libc::c_ulonglong {
            useAllData = 1 as libc::c_int;
            numSamples = dataEnd.wrapping_sub(dataBegin);
        }
        i = 0 as libc::c_int as vl_uindex;
        while i < numSamples {
            let mut sampleIndex: vl_uint32 = 0;
            let mut di: vl_index = 0;
            let mut datum: libc::c_double = 0.;
            if useAllData == 1 as libc::c_int {
                sampleIndex = i as vl_uint32;
            } else {
                sampleIndex = (vl_rand_uint32((*forest).rand))
                    .wrapping_rem(1024 as libc::c_int as libc::c_uint);
            }
            sampleIndex = (sampleIndex as libc::c_ulonglong).wrapping_add(dataBegin)
                as vl_uint32 as vl_uint32;
            di = (*((*tree).dataIndex).offset(sampleIndex as isize)).index;
            match (*forest).dataType {
                1 => {
                    datum = *((*forest).data as *const libc::c_float)
                        .offset(
                            (di as libc::c_ulonglong)
                                .wrapping_mul((*forest).dimension)
                                .wrapping_add(d) as isize,
                        ) as libc::c_double;
                }
                2 => {
                    datum = *((*forest).data as *const libc::c_double)
                        .offset(
                            (di as libc::c_ulonglong)
                                .wrapping_mul((*forest).dimension)
                                .wrapping_add(d) as isize,
                        );
                }
                _ => {
                    abort();
                }
            }
            mean += datum;
            secondMoment += datum * datum;
            i = i.wrapping_add(1);
        }
        mean /= numSamples as libc::c_double;
        secondMoment /= numSamples as libc::c_double;
        variance = secondMoment - mean * mean;
        if !(variance <= 0 as libc::c_int as libc::c_double) {
            if (*forest).splitHeapNumNodes < (*forest).splitHeapSize {
                let mut splitDimension_0: *mut VlKDTreeSplitDimension = ((*forest)
                    .splitHeapArray)
                    .as_mut_ptr()
                    .offset((*forest).splitHeapNumNodes as isize);
                (*splitDimension_0).dimension = d as libc::c_uint;
                (*splitDimension_0).mean = mean;
                (*splitDimension_0).variance = variance;
                vl_kdtree_split_heap_push(
                    ((*forest).splitHeapArray).as_mut_ptr(),
                    &mut (*forest).splitHeapNumNodes,
                );
            } else {
                let mut splitDimension_1: *mut VlKDTreeSplitDimension = ((*forest)
                    .splitHeapArray)
                    .as_mut_ptr()
                    .offset(0 as libc::c_int as isize);
                if (*splitDimension_1).variance < variance {
                    (*splitDimension_1).dimension = d as libc::c_uint;
                    (*splitDimension_1).mean = mean;
                    (*splitDimension_1).variance = variance;
                    vl_kdtree_split_heap_update(
                        ((*forest).splitHeapArray).as_mut_ptr(),
                        (*forest).splitHeapNumNodes,
                        0 as libc::c_int as vl_uindex,
                    );
                }
            }
        }
        d = d.wrapping_add(1);
    }
    if (*forest).splitHeapNumNodes == 0 as libc::c_int as libc::c_ulonglong {
        (*node)
            .lowerChild = dataBegin
            .wrapping_neg()
            .wrapping_sub(1 as libc::c_int as libc::c_ulonglong) as vl_index;
        (*node)
            .upperChild = dataEnd
            .wrapping_neg()
            .wrapping_sub(1 as libc::c_int as libc::c_ulonglong) as vl_index;
        return;
    }
    splitDimension = ((*forest).splitHeapArray)
        .as_mut_ptr()
        .offset(
            (vl_rand_uint32((*forest).rand) as libc::c_ulonglong)
                .wrapping_rem(
                    (if (*forest).splitHeapSize < (*forest).splitHeapNumNodes {
                        (*forest).splitHeapSize
                    } else {
                        (*forest).splitHeapNumNodes
                    }),
                ) as isize,
        );
    (*node).splitDimension = (*splitDimension).dimension;
    i = dataBegin;
    while i < dataEnd {
        let mut di_0: vl_index = (*((*tree).dataIndex).offset(i as isize)).index;
        let mut datum_0: libc::c_double = 0.;
        match (*forest).dataType {
            1 => {
                datum_0 = *((*forest).data as *const libc::c_float)
                    .offset(
                        (di_0 as libc::c_ulonglong)
                            .wrapping_mul((*forest).dimension)
                            .wrapping_add(
                                (*splitDimension).dimension as libc::c_ulonglong,
                            ) as isize,
                    ) as libc::c_double;
            }
            2 => {
                datum_0 = *((*forest).data as *const libc::c_double)
                    .offset(
                        (di_0 as libc::c_ulonglong)
                            .wrapping_mul((*forest).dimension)
                            .wrapping_add(
                                (*splitDimension).dimension as libc::c_ulonglong,
                            ) as isize,
                    );
            }
            _ => {
                abort();
            }
        }
        (*((*tree).dataIndex).offset(i as isize)).value = datum_0;
        i = i.wrapping_add(1);
    }
    qsort(
        ((*tree).dataIndex).offset(dataBegin as isize) as *mut libc::c_void,
        dataEnd.wrapping_sub(dataBegin) as size_t,
        ::core::mem::size_of::<VlKDTreeDataIndexEntry>() as libc::c_ulong,
        Some(
            vl_kdtree_compare_index_entries
                as unsafe extern "C" fn(
                    *const libc::c_void,
                    *const libc::c_void,
                ) -> libc::c_int,
        ),
    );
    let mut current_block_70: u64;
    match (*forest).thresholdingMethod as libc::c_uint {
        1 => {
            (*node).splitThreshold = (*splitDimension).mean;
            splitIndex = dataBegin;
            while splitIndex < dataEnd
                && (*((*tree).dataIndex).offset(splitIndex as isize)).value
                    <= (*node).splitThreshold
            {
                splitIndex = splitIndex.wrapping_add(1);
            }
            splitIndex = (splitIndex as libc::c_ulonglong)
                .wrapping_sub(1 as libc::c_int as libc::c_ulonglong) as vl_uindex
                as vl_uindex;
            if dataBegin <= splitIndex
                && splitIndex.wrapping_add(1 as libc::c_int as libc::c_ulonglong)
                    < dataEnd
            {
                current_block_70 = 8869332144787829186;
            } else {
                current_block_70 = 18148317310534040296;
            }
        }
        0 => {
            current_block_70 = 18148317310534040296;
        }
        _ => {
            abort();
        }
    }
    match current_block_70 {
        18148317310534040296 => {
            medianIndex = dataBegin
                .wrapping_add(dataEnd)
                .wrapping_sub(1 as libc::c_int as libc::c_ulonglong)
                .wrapping_div(2 as libc::c_int as libc::c_ulonglong);
            splitIndex = medianIndex;
            (*node)
                .splitThreshold = (*((*tree).dataIndex).offset(medianIndex as isize))
                .value;
        }
        _ => {}
    }
    (*node).lowerChild = vl_kdtree_node_new(tree, nodeIndex) as vl_index;
    vl_kdtree_build_recursively(
        forest,
        tree,
        (*node).lowerChild as vl_uindex,
        dataBegin,
        splitIndex.wrapping_add(1 as libc::c_int as libc::c_ulonglong),
        depth.wrapping_add(1 as libc::c_int as libc::c_uint),
    );
    (*node).upperChild = vl_kdtree_node_new(tree, nodeIndex) as vl_index;
    vl_kdtree_build_recursively(
        forest,
        tree,
        (*node).upperChild as vl_uindex,
        splitIndex.wrapping_add(1 as libc::c_int as libc::c_ulonglong),
        dataEnd,
        depth.wrapping_add(1 as libc::c_int as libc::c_uint),
    );
}
#[no_mangle]
pub unsafe extern "C" fn vl_kdforest_new(
    mut dataType: vl_type,
    mut dimension: vl_size,
    mut numTrees: vl_size,
    mut distance: VlVectorComparisonType,
) -> *mut VlKDForest {
    let mut self_0: *mut VlKDForest = vl_calloc(
        ::core::mem::size_of::<VlKDForest>() as libc::c_ulong,
        1 as libc::c_int as size_t,
    ) as *mut VlKDForest;
    if dataType == 1 as libc::c_int as libc::c_uint
        || dataType == 2 as libc::c_int as libc::c_uint
    {} else {
        __assert_fail(
            b"dataType == VL_TYPE_FLOAT || dataType == VL_TYPE_DOUBLE\0" as *const u8
                as *const libc::c_char,
            b"vl/kdtree.c\0" as *const u8 as *const libc::c_char,
            337 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 79],
                &[libc::c_char; 79],
            >(
                b"VlKDForest *vl_kdforest_new(vl_type, vl_size, vl_size, VlVectorComparisonType)\0",
            ))
                .as_ptr(),
        );
    }
    if dimension >= 1 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"dimension >= 1\0" as *const u8 as *const libc::c_char,
            b"vl/kdtree.c\0" as *const u8 as *const libc::c_char,
            338 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 79],
                &[libc::c_char; 79],
            >(
                b"VlKDForest *vl_kdforest_new(vl_type, vl_size, vl_size, VlVectorComparisonType)\0",
            ))
                .as_ptr(),
        );
    }
    if numTrees >= 1 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"numTrees >= 1\0" as *const u8 as *const libc::c_char,
            b"vl/kdtree.c\0" as *const u8 as *const libc::c_char,
            339 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 79],
                &[libc::c_char; 79],
            >(
                b"VlKDForest *vl_kdforest_new(vl_type, vl_size, vl_size, VlVectorComparisonType)\0",
            ))
                .as_ptr(),
        );
    }
    (*self_0).rand = vl_get_rand();
    (*self_0).dataType = dataType;
    (*self_0).numData = 0 as libc::c_int as vl_size;
    (*self_0).data = 0 as *const libc::c_void;
    (*self_0).dimension = dimension;
    (*self_0).numTrees = numTrees;
    (*self_0).trees = 0 as *mut *mut VlKDTree;
    (*self_0).thresholdingMethod = VL_KDTREE_MEDIAN;
    (*self_0)
        .splitHeapSize = if numTrees < 5 as libc::c_int as libc::c_ulonglong {
        numTrees
    } else {
        5 as libc::c_int as libc::c_ulonglong
    };
    (*self_0).splitHeapNumNodes = 0 as libc::c_int as vl_size;
    (*self_0).distance = distance;
    (*self_0).maxNumNodes = 0 as libc::c_int as vl_size;
    (*self_0).numSearchers = 0 as libc::c_int as vl_size;
    (*self_0).headSearcher = 0 as *mut _VlKDForestSearcher;
    match (*self_0).dataType {
        1 => {
            (*self_0)
                .distanceFunction = ::core::mem::transmute::<
                VlFloatVectorComparisonFunction,
                Option::<unsafe extern "C" fn() -> ()>,
            >(vl_get_vector_comparison_function_f(distance));
        }
        2 => {
            (*self_0)
                .distanceFunction = ::core::mem::transmute::<
                VlDoubleVectorComparisonFunction,
                Option::<unsafe extern "C" fn() -> ()>,
            >(vl_get_vector_comparison_function_d(distance));
        }
        _ => {
            abort();
        }
    }
    return self_0;
}
#[no_mangle]
pub unsafe extern "C" fn vl_kdforest_new_searcher(
    mut kdforest: *mut VlKDForest,
) -> *mut VlKDForestSearcher {
    let mut self_0: *mut VlKDForestSearcher = vl_calloc(
        ::core::mem::size_of::<VlKDForestSearcher>() as libc::c_ulong,
        1 as libc::c_int as size_t,
    ) as *mut VlKDForestSearcher;
    if (*kdforest).numSearchers == 0 as libc::c_int as libc::c_ulonglong {
        (*kdforest).headSearcher = self_0;
        (*self_0).previous = 0 as *mut _VlKDForestSearcher;
        (*self_0).next = 0 as *mut _VlKDForestSearcher;
    } else {
        let mut lastSearcher: *mut VlKDForestSearcher = (*kdforest).headSearcher;
        loop {
            if !((*lastSearcher).next).is_null() {
                lastSearcher = (*lastSearcher).next;
            } else {
                (*lastSearcher).next = self_0;
                (*self_0).previous = lastSearcher;
                (*self_0).next = 0 as *mut _VlKDForestSearcher;
                break;
            }
        }
    }
    (*kdforest).numSearchers = ((*kdforest).numSearchers).wrapping_add(1);
    (*self_0).forest = kdforest;
    (*self_0)
        .searchHeapArray = vl_malloc(
        (::core::mem::size_of::<VlKDForestSearchState>() as libc::c_ulong
            as libc::c_ulonglong)
            .wrapping_mul((*kdforest).maxNumNodes) as size_t,
    ) as *mut VlKDForestSearchState;
    (*self_0)
        .searchIdBook = vl_calloc(
        ::core::mem::size_of::<vl_uindex>() as libc::c_ulong,
        (*kdforest).numData as size_t,
    ) as *mut vl_uindex;
    return self_0;
}
#[no_mangle]
pub unsafe extern "C" fn vl_kdforestsearcher_delete(
    mut self_0: *mut VlKDForestSearcher,
) {
    if !((*self_0).previous).is_null() && !((*self_0).next).is_null() {
        (*(*self_0).previous).next = (*self_0).next;
        (*(*self_0).next).previous = (*self_0).previous;
    } else if !((*self_0).previous).is_null() && ((*self_0).next).is_null() {
        (*(*self_0).previous).next = 0 as *mut _VlKDForestSearcher;
    } else if ((*self_0).previous).is_null() && !((*self_0).next).is_null() {
        (*(*self_0).next).previous = 0 as *mut _VlKDForestSearcher;
        (*(*self_0).forest).headSearcher = (*self_0).next;
    } else {
        (*(*self_0).forest).headSearcher = 0 as *mut _VlKDForestSearcher;
    }
    (*(*self_0).forest)
        .numSearchers = ((*(*self_0).forest).numSearchers).wrapping_sub(1);
    vl_free((*self_0).searchHeapArray as *mut libc::c_void);
    vl_free((*self_0).searchIdBook as *mut libc::c_void);
    vl_free(self_0 as *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn vl_kdforest_get_searcher(
    mut self_0: *const VlKDForest,
    mut pos: vl_uindex,
) -> *mut VlKDForestSearcher {
    let mut lastSearcher: *mut VlKDForestSearcher = (*self_0).headSearcher;
    let mut i: vl_uindex = 0;
    i = 0 as libc::c_int as vl_uindex;
    while (i < pos) as libc::c_int
        & (lastSearcher != 0 as *mut libc::c_void as *mut VlKDForestSearcher)
            as libc::c_int != 0
    {
        lastSearcher = (*lastSearcher).next;
        i = i.wrapping_add(1);
    }
    return lastSearcher;
}
#[no_mangle]
pub unsafe extern "C" fn vl_kdforest_delete(mut self_0: *mut VlKDForest) {
    let mut ti: vl_uindex = 0;
    let mut searcher: *mut VlKDForestSearcher = 0 as *mut VlKDForestSearcher;
    loop {
        searcher = vl_kdforest_get_searcher(self_0, 0 as libc::c_int as vl_uindex);
        if searcher.is_null() {
            break;
        }
        vl_kdforestsearcher_delete(searcher);
    }
    if !((*self_0).trees).is_null() {
        ti = 0 as libc::c_int as vl_uindex;
        while ti < (*self_0).numTrees {
            if !(*((*self_0).trees).offset(ti as isize)).is_null() {
                if !((**((*self_0).trees).offset(ti as isize)).nodes).is_null() {
                    vl_free(
                        (**((*self_0).trees).offset(ti as isize)).nodes
                            as *mut libc::c_void,
                    );
                }
                if !((**((*self_0).trees).offset(ti as isize)).dataIndex).is_null() {
                    vl_free(
                        (**((*self_0).trees).offset(ti as isize)).dataIndex
                            as *mut libc::c_void,
                    );
                }
                vl_free(*((*self_0).trees).offset(ti as isize) as *mut libc::c_void);
            }
            ti = ti.wrapping_add(1);
        }
        vl_free((*self_0).trees as *mut libc::c_void);
    }
    vl_free(self_0 as *mut libc::c_void);
}
unsafe extern "C" fn vl_kdtree_calc_bounds_recursively(
    mut tree: *mut VlKDTree,
    mut nodeIndex: vl_uindex,
    mut searchBounds: *mut libc::c_double,
) {
    let mut node: *mut VlKDTreeNode = ((*tree).nodes).offset(nodeIndex as isize);
    let mut i: vl_uindex = (*node).splitDimension as vl_uindex;
    let mut t: libc::c_double = (*node).splitThreshold;
    (*node)
        .lowerBound = *searchBounds
        .offset(
            (2 as libc::c_int as libc::c_ulonglong)
                .wrapping_mul(i)
                .wrapping_add(0 as libc::c_int as libc::c_ulonglong) as isize,
        );
    (*node)
        .upperBound = *searchBounds
        .offset(
            (2 as libc::c_int as libc::c_ulonglong)
                .wrapping_mul(i)
                .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as isize,
        );
    if (*node).lowerChild > 0 as libc::c_int as libc::c_longlong {
        *searchBounds
            .offset(
                (2 as libc::c_int as libc::c_ulonglong)
                    .wrapping_mul(i)
                    .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as isize,
            ) = t;
        vl_kdtree_calc_bounds_recursively(
            tree,
            (*node).lowerChild as vl_uindex,
            searchBounds,
        );
        *searchBounds
            .offset(
                (2 as libc::c_int as libc::c_ulonglong)
                    .wrapping_mul(i)
                    .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as isize,
            ) = (*node).upperBound;
    }
    if (*node).upperChild > 0 as libc::c_int as libc::c_longlong {
        *searchBounds
            .offset(
                (2 as libc::c_int as libc::c_ulonglong)
                    .wrapping_mul(i)
                    .wrapping_add(0 as libc::c_int as libc::c_ulonglong) as isize,
            ) = t;
        vl_kdtree_calc_bounds_recursively(
            tree,
            (*node).upperChild as vl_uindex,
            searchBounds,
        );
        *searchBounds
            .offset(
                (2 as libc::c_int as libc::c_ulonglong)
                    .wrapping_mul(i)
                    .wrapping_add(0 as libc::c_int as libc::c_ulonglong) as isize,
            ) = (*node).lowerBound;
    }
}
#[no_mangle]
pub unsafe extern "C" fn vl_kdforest_build(
    mut self_0: *mut VlKDForest,
    mut numData: vl_size,
    mut data: *const libc::c_void,
) {
    let mut di: vl_uindex = 0;
    let mut ti: vl_uindex = 0;
    let mut maxNumNodes: vl_size = 0;
    let mut searchBounds: *mut libc::c_double = 0 as *mut libc::c_double;
    if !data.is_null() {} else {
        __assert_fail(
            b"data\0" as *const u8 as *const libc::c_char,
            b"vl/kdtree.c\0" as *const u8 as *const libc::c_char,
            536 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 60],
                &[libc::c_char; 60],
            >(b"void vl_kdforest_build(VlKDForest *, vl_size, const void *)\0"))
                .as_ptr(),
        );
    }
    if numData >= 1 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"numData >= 1\0" as *const u8 as *const libc::c_char,
            b"vl/kdtree.c\0" as *const u8 as *const libc::c_char,
            537 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 60],
                &[libc::c_char; 60],
            >(b"void vl_kdforest_build(VlKDForest *, vl_size, const void *)\0"))
                .as_ptr(),
        );
    }
    (*self_0).data = data;
    (*self_0).numData = numData;
    (*self_0)
        .trees = vl_malloc(
        (::core::mem::size_of::<*mut VlKDTree>() as libc::c_ulong as libc::c_ulonglong)
            .wrapping_mul((*self_0).numTrees) as size_t,
    ) as *mut *mut VlKDTree;
    maxNumNodes = 0 as libc::c_int as vl_size;
    ti = 0 as libc::c_int as vl_uindex;
    while ti < (*self_0).numTrees {
        let ref mut fresh0 = *((*self_0).trees).offset(ti as isize);
        *fresh0 = vl_malloc(::core::mem::size_of::<VlKDTree>() as libc::c_ulong)
            as *mut VlKDTree;
        let ref mut fresh1 = (**((*self_0).trees).offset(ti as isize)).dataIndex;
        *fresh1 = vl_malloc(
            (::core::mem::size_of::<VlKDTreeDataIndexEntry>() as libc::c_ulong
                as libc::c_ulonglong)
                .wrapping_mul((*self_0).numData) as size_t,
        ) as *mut VlKDTreeDataIndexEntry;
        di = 0 as libc::c_int as vl_uindex;
        while di < (*self_0).numData {
            (*((**((*self_0).trees).offset(ti as isize)).dataIndex).offset(di as isize))
                .index = di as vl_index;
            di = di.wrapping_add(1);
        }
        (**((*self_0).trees).offset(ti as isize))
            .numUsedNodes = 0 as libc::c_int as vl_size;
        (**((*self_0).trees).offset(ti as isize))
            .numAllocatedNodes = (2 as libc::c_int as libc::c_ulonglong)
            .wrapping_mul((*self_0).numData)
            .wrapping_sub(1 as libc::c_int as libc::c_ulonglong);
        let ref mut fresh2 = (**((*self_0).trees).offset(ti as isize)).nodes;
        *fresh2 = vl_malloc(
            (::core::mem::size_of::<VlKDTreeNode>() as libc::c_ulong
                as libc::c_ulonglong)
                .wrapping_mul(
                    (**((*self_0).trees).offset(ti as isize)).numAllocatedNodes,
                ) as size_t,
        ) as *mut VlKDTreeNode;
        (**((*self_0).trees).offset(ti as isize))
            .depth = 0 as libc::c_int as libc::c_uint;
        vl_kdtree_build_recursively(
            self_0,
            *((*self_0).trees).offset(ti as isize),
            vl_kdtree_node_new(
                *((*self_0).trees).offset(ti as isize),
                0 as libc::c_int as vl_uindex,
            ),
            0 as libc::c_int as vl_uindex,
            (*self_0).numData,
            0 as libc::c_int as libc::c_uint,
        );
        maxNumNodes = (maxNumNodes as libc::c_ulonglong)
            .wrapping_add((**((*self_0).trees).offset(ti as isize)).numUsedNodes)
            as vl_size as vl_size;
        ti = ti.wrapping_add(1);
    }
    searchBounds = vl_malloc(
        ((::core::mem::size_of::<libc::c_double>() as libc::c_ulong)
            .wrapping_mul(2 as libc::c_int as libc::c_ulong) as libc::c_ulonglong)
            .wrapping_mul((*self_0).dimension) as size_t,
    ) as *mut libc::c_double;
    ti = 0 as libc::c_int as vl_uindex;
    while ti < (*self_0).numTrees {
        let mut iter: *mut libc::c_double = searchBounds;
        let mut end: *mut libc::c_double = iter
            .offset(
                (2 as libc::c_int as libc::c_ulonglong).wrapping_mul((*self_0).dimension)
                    as isize,
            );
        while iter < end {
            let fresh3 = iter;
            iter = iter.offset(1);
            *fresh3 = -vl_infinity_f.value as libc::c_double;
            let fresh4 = iter;
            iter = iter.offset(1);
            *fresh4 = vl_infinity_f.value as libc::c_double;
        }
        vl_kdtree_calc_bounds_recursively(
            *((*self_0).trees).offset(ti as isize),
            0 as libc::c_int as vl_uindex,
            searchBounds,
        );
        ti = ti.wrapping_add(1);
    }
    vl_free(searchBounds as *mut libc::c_void);
    (*self_0).maxNumNodes = maxNumNodes;
}
#[no_mangle]
pub unsafe extern "C" fn vl_kdforest_query_recursively(
    mut searcher: *mut VlKDForestSearcher,
    mut tree: *mut VlKDTree,
    mut nodeIndex: vl_uindex,
    mut neighbors: *mut VlKDForestNeighbor,
    mut numNeighbors: vl_size,
    mut numAddedNeighbors: *mut vl_size,
    mut dist: libc::c_double,
    mut query: *const libc::c_void,
) -> vl_uindex {
    let mut node: *const VlKDTreeNode = ((*tree).nodes).offset(nodeIndex as isize);
    let mut i: vl_uindex = (*node).splitDimension as vl_uindex;
    let mut nextChild: vl_index = 0;
    let mut saveChild: vl_index = 0;
    let mut delta: libc::c_double = 0.;
    let mut saveDist: libc::c_double = 0.;
    let mut x: libc::c_double = 0.;
    let mut x1: libc::c_double = (*node).lowerBound;
    let mut x2: libc::c_double = (*node).splitThreshold;
    let mut x3: libc::c_double = (*node).upperBound;
    let mut searchState: *mut VlKDForestSearchState = 0 as *mut VlKDForestSearchState;
    (*searcher).searchNumRecursions = ((*searcher).searchNumRecursions).wrapping_add(1);
    match (*(*searcher).forest).dataType {
        1 => {
            x = *(query as *const libc::c_float).offset(i as isize) as libc::c_double;
        }
        2 => {
            x = *(query as *const libc::c_double).offset(i as isize);
        }
        _ => {
            abort();
        }
    }
    if (*node).lowerChild < 0 as libc::c_int as libc::c_longlong {
        let mut begin: vl_index = -(*node).lowerChild
            - 1 as libc::c_int as libc::c_longlong;
        let mut end: vl_index = -(*node).upperChild
            - 1 as libc::c_int as libc::c_longlong;
        let mut iter: vl_index = 0;
        iter = begin;
        while iter < end
            && ((*(*searcher).forest).searchMaxNumComparisons
                == 0 as libc::c_int as libc::c_ulonglong
                || (*searcher).searchNumComparisons
                    < (*(*searcher).forest).searchMaxNumComparisons)
        {
            let mut di: vl_index = (*((*tree).dataIndex).offset(iter as isize)).index;
            if !(*((*searcher).searchIdBook).offset(di as isize) == (*searcher).searchId)
            {
                *((*searcher).searchIdBook).offset(di as isize) = (*searcher).searchId;
                match (*(*searcher).forest).dataType {
                    1 => {
                        dist = (::core::mem::transmute::<
                            Option::<unsafe extern "C" fn() -> ()>,
                            VlFloatVectorComparisonFunction,
                        >((*(*searcher).forest).distanceFunction))
                            .expect(
                                "non-null function pointer",
                            )(
                            (*(*searcher).forest).dimension,
                            query as *const libc::c_float,
                            ((*(*searcher).forest).data as *const libc::c_float)
                                .offset(
                                    (di as libc::c_ulonglong)
                                        .wrapping_mul((*(*searcher).forest).dimension) as isize,
                                ),
                        ) as libc::c_double;
                    }
                    2 => {
                        dist = (::core::mem::transmute::<
                            Option::<unsafe extern "C" fn() -> ()>,
                            VlDoubleVectorComparisonFunction,
                        >((*(*searcher).forest).distanceFunction))
                            .expect(
                                "non-null function pointer",
                            )(
                            (*(*searcher).forest).dimension,
                            query as *const libc::c_double,
                            ((*(*searcher).forest).data as *const libc::c_double)
                                .offset(
                                    (di as libc::c_ulonglong)
                                        .wrapping_mul((*(*searcher).forest).dimension) as isize,
                                ),
                        );
                    }
                    _ => {
                        abort();
                    }
                }
                (*searcher)
                    .searchNumComparisons = ((*searcher).searchNumComparisons
                    as libc::c_ulonglong)
                    .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as vl_size
                    as vl_size;
                if *numAddedNeighbors < numNeighbors {
                    let mut newNeighbor: *mut VlKDForestNeighbor = neighbors
                        .offset(*numAddedNeighbors as isize);
                    (*newNeighbor).index = di as vl_uindex;
                    (*newNeighbor).distance = dist;
                    vl_kdforest_neighbor_heap_push(neighbors, numAddedNeighbors);
                } else {
                    let mut largestNeighbor: *mut VlKDForestNeighbor = neighbors
                        .offset(0 as libc::c_int as isize);
                    if (*largestNeighbor).distance > dist {
                        (*largestNeighbor).index = di as vl_uindex;
                        (*largestNeighbor).distance = dist;
                        vl_kdforest_neighbor_heap_update(
                            neighbors,
                            *numAddedNeighbors,
                            0 as libc::c_int as vl_uindex,
                        );
                    }
                }
            }
            iter += 1;
        }
        return nodeIndex;
    }
    delta = x - x2;
    saveDist = dist + delta * delta;
    if x <= x2 {
        nextChild = (*node).lowerChild;
        saveChild = (*node).upperChild;
        if x <= x1 {
            delta = x - x1;
            saveDist -= delta * delta;
        }
    } else {
        nextChild = (*node).upperChild;
        saveChild = (*node).lowerChild;
        if x > x3 {
            delta = x - x3;
            saveDist -= delta * delta;
        }
    }
    if *numAddedNeighbors < numNeighbors
        || (*neighbors.offset(0 as libc::c_int as isize)).distance > saveDist
    {
        searchState = ((*searcher).searchHeapArray)
            .offset((*searcher).searchHeapNumNodes as isize);
        (*searchState).tree = tree;
        (*searchState).nodeIndex = saveChild as vl_uindex;
        (*searchState).distanceLowerBound = saveDist;
        vl_kdforest_search_heap_push(
            (*searcher).searchHeapArray,
            &mut (*searcher).searchHeapNumNodes,
        );
    }
    return vl_kdforest_query_recursively(
        searcher,
        tree,
        nextChild as vl_uindex,
        neighbors,
        numNeighbors,
        numAddedNeighbors,
        dist,
        query,
    );
}
#[no_mangle]
pub unsafe extern "C" fn vl_kdforest_query(
    mut self_0: *mut VlKDForest,
    mut neighbors: *mut VlKDForestNeighbor,
    mut numNeighbors: vl_size,
    mut query: *const libc::c_void,
) -> vl_size {
    let mut searcher: *mut VlKDForestSearcher = vl_kdforest_get_searcher(
        self_0,
        0 as libc::c_int as vl_uindex,
    );
    if searcher.is_null() {
        searcher = vl_kdforest_new_searcher(self_0);
    }
    return vl_kdforestsearcher_query(searcher, neighbors, numNeighbors, query);
}
#[no_mangle]
pub unsafe extern "C" fn vl_kdforestsearcher_query(
    mut self_0: *mut VlKDForestSearcher,
    mut neighbors: *mut VlKDForestNeighbor,
    mut numNeighbors: vl_size,
    mut query: *const libc::c_void,
) -> vl_size {
    let mut i: vl_uindex = 0;
    let mut ti: vl_uindex = 0;
    let mut exactSearch: vl_bool = ((*(*self_0).forest).searchMaxNumComparisons
        == 0 as libc::c_int as libc::c_ulonglong) as libc::c_int;
    let mut searchState: *mut VlKDForestSearchState = 0 as *mut VlKDForestSearchState;
    let mut numAddedNeighbors: vl_size = 0 as libc::c_int as vl_size;
    if !neighbors.is_null() {} else {
        __assert_fail(
            b"neighbors\0" as *const u8 as *const libc::c_char,
            b"vl/kdtree.c\0" as *const u8 as *const libc::c_char,
            786 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 101],
                &[libc::c_char; 101],
            >(
                b"vl_size vl_kdforestsearcher_query(VlKDForestSearcher *, VlKDForestNeighbor *, vl_size, const void *)\0",
            ))
                .as_ptr(),
        );
    }
    if numNeighbors > 0 as libc::c_int as libc::c_ulonglong {} else {
        __assert_fail(
            b"numNeighbors > 0\0" as *const u8 as *const libc::c_char,
            b"vl/kdtree.c\0" as *const u8 as *const libc::c_char,
            787 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 101],
                &[libc::c_char; 101],
            >(
                b"vl_size vl_kdforestsearcher_query(VlKDForestSearcher *, VlKDForestNeighbor *, vl_size, const void *)\0",
            ))
                .as_ptr(),
        );
    }
    if !query.is_null() {} else {
        __assert_fail(
            b"query\0" as *const u8 as *const libc::c_char,
            b"vl/kdtree.c\0" as *const u8 as *const libc::c_char,
            788 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 101],
                &[libc::c_char; 101],
            >(
                b"vl_size vl_kdforestsearcher_query(VlKDForestSearcher *, VlKDForestNeighbor *, vl_size, const void *)\0",
            ))
                .as_ptr(),
        );
    }
    (*self_0)
        .searchId = ((*self_0).searchId as libc::c_ulonglong)
        .wrapping_add(1 as libc::c_int as libc::c_ulonglong) as vl_uindex as vl_uindex;
    (*self_0).searchNumRecursions = 0 as libc::c_int as vl_size;
    (*self_0).searchNumComparisons = 0 as libc::c_int as vl_size;
    (*self_0).searchNumSimplifications = 0 as libc::c_int as vl_size;
    (*self_0).searchHeapNumNodes = 0 as libc::c_int as vl_size;
    ti = 0 as libc::c_int as vl_uindex;
    while ti < (*(*self_0).forest).numTrees {
        searchState = ((*self_0).searchHeapArray)
            .offset((*self_0).searchHeapNumNodes as isize);
        (*searchState).tree = *((*(*self_0).forest).trees).offset(ti as isize);
        (*searchState).nodeIndex = 0 as libc::c_int as vl_uindex;
        (*searchState).distanceLowerBound = 0 as libc::c_int as libc::c_double;
        vl_kdforest_search_heap_push(
            (*self_0).searchHeapArray,
            &mut (*self_0).searchHeapNumNodes,
        );
        ti = ti.wrapping_add(1);
    }
    while exactSearch != 0
        || (*self_0).searchNumComparisons < (*(*self_0).forest).searchMaxNumComparisons
    {
        let mut searchState_0: *mut VlKDForestSearchState = 0
            as *mut VlKDForestSearchState;
        if (*self_0).searchHeapNumNodes == 0 as libc::c_int as libc::c_ulonglong {
            break;
        }
        searchState_0 = ((*self_0).searchHeapArray)
            .offset(
                vl_kdforest_search_heap_pop(
                    (*self_0).searchHeapArray,
                    &mut (*self_0).searchHeapNumNodes,
                ) as isize,
            );
        if numAddedNeighbors == numNeighbors
            && (*neighbors.offset(0 as libc::c_int as isize)).distance
                < (*searchState_0).distanceLowerBound
        {
            (*self_0)
                .searchNumSimplifications = ((*self_0).searchNumSimplifications)
                .wrapping_add(1);
            break;
        } else {
            vl_kdforest_query_recursively(
                self_0,
                (*searchState_0).tree,
                (*searchState_0).nodeIndex,
                neighbors,
                numNeighbors,
                &mut numAddedNeighbors,
                (*searchState_0).distanceLowerBound,
                query,
            );
        }
    }
    i = numAddedNeighbors;
    while i < numNeighbors {
        (*neighbors.offset(i as isize)).index = -(1 as libc::c_int) as vl_uindex;
        (*neighbors.offset(i as isize)).distance = vl_nan_f.value as libc::c_double;
        i = i.wrapping_add(1);
    }
    while numAddedNeighbors != 0 {
        vl_kdforest_neighbor_heap_pop(neighbors, &mut numAddedNeighbors);
    }
    return (*self_0).searchNumComparisons;
}
#[no_mangle]
pub unsafe extern "C" fn vl_kdforest_query_with_array(
    mut self_0: *mut VlKDForest,
    mut indexes: *mut vl_uint32,
    mut numNeighbors: vl_size,
    mut numQueries: vl_size,
    mut distances: *mut libc::c_void,
    mut queries: *const libc::c_void,
) -> vl_size {
    let mut numComparisons: vl_size = 0 as libc::c_int as vl_size;
    let mut dataType: vl_type = vl_kdforest_get_data_type(self_0);
    let mut dimension: vl_size = vl_kdforest_get_data_dimension(self_0);
    let mut qi: vl_index = 0;
    let mut thisNumComparisons: vl_size = 0 as libc::c_int as vl_size;
    let mut searcher: *mut VlKDForestSearcher = 0 as *mut VlKDForestSearcher;
    let mut neighbors: *mut VlKDForestNeighbor = 0 as *mut VlKDForestNeighbor;
    searcher = vl_kdforest_new_searcher(self_0);
    neighbors = vl_calloc(
        ::core::mem::size_of::<VlKDForestNeighbor>() as libc::c_ulong,
        numNeighbors as size_t,
    ) as *mut VlKDForestNeighbor;
    qi = 0 as libc::c_int as vl_index;
    while qi < numQueries as libc::c_int as libc::c_longlong {
        match dataType {
            1 => {
                let mut ni: vl_size = 0;
                thisNumComparisons = (thisNumComparisons as libc::c_ulonglong)
                    .wrapping_add(
                        vl_kdforestsearcher_query(
                            searcher,
                            neighbors,
                            numNeighbors,
                            (queries as *const libc::c_float)
                                .offset(
                                    (qi as libc::c_ulonglong).wrapping_mul(dimension) as isize,
                                ) as *const libc::c_void,
                        ),
                    ) as vl_size as vl_size;
                ni = 0 as libc::c_int as vl_size;
                while ni < numNeighbors {
                    *indexes
                        .offset(
                            (qi as libc::c_ulonglong)
                                .wrapping_mul(numNeighbors)
                                .wrapping_add(ni) as isize,
                        ) = (*neighbors.offset(ni as isize)).index as vl_uint32;
                    if !distances.is_null() {
                        *(distances as *mut libc::c_float)
                            .offset(
                                (qi as libc::c_ulonglong).wrapping_mul(numNeighbors)
                                    as isize,
                            )
                            .offset(
                                ni as isize,
                            ) = (*neighbors.offset(ni as isize)).distance
                            as libc::c_float;
                    }
                    ni = ni.wrapping_add(1);
                }
            }
            2 => {
                let mut ni_0: vl_size = 0;
                thisNumComparisons = (thisNumComparisons as libc::c_ulonglong)
                    .wrapping_add(
                        vl_kdforestsearcher_query(
                            searcher,
                            neighbors,
                            numNeighbors,
                            (queries as *const libc::c_double)
                                .offset(
                                    (qi as libc::c_ulonglong).wrapping_mul(dimension) as isize,
                                ) as *const libc::c_void,
                        ),
                    ) as vl_size as vl_size;
                ni_0 = 0 as libc::c_int as vl_size;
                while ni_0 < numNeighbors {
                    *indexes
                        .offset(
                            (qi as libc::c_ulonglong)
                                .wrapping_mul(numNeighbors)
                                .wrapping_add(ni_0) as isize,
                        ) = (*neighbors.offset(ni_0 as isize)).index as vl_uint32;
                    if !distances.is_null() {
                        *(distances as *mut libc::c_double)
                            .offset(
                                (qi as libc::c_ulonglong).wrapping_mul(numNeighbors)
                                    as isize,
                            )
                            .offset(
                                ni_0 as isize,
                            ) = (*neighbors.offset(ni_0 as isize)).distance;
                    }
                    ni_0 = ni_0.wrapping_add(1);
                }
            }
            _ => {
                abort();
            }
        }
        qi += 1;
    }
    numComparisons = (numComparisons as libc::c_ulonglong)
        .wrapping_add(thisNumComparisons) as vl_size as vl_size;
    vl_kdforestsearcher_delete(searcher);
    vl_free(neighbors as *mut libc::c_void);
    return numComparisons;
}
#[no_mangle]
pub unsafe extern "C" fn vl_kdforest_get_num_nodes_of_tree(
    mut self_0: *const VlKDForest,
    mut treeIndex: vl_uindex,
) -> vl_size {
    if treeIndex < (*self_0).numTrees {} else {
        __assert_fail(
            b"treeIndex < self->numTrees\0" as *const u8 as *const libc::c_char,
            b"vl/kdtree.c\0" as *const u8 as *const libc::c_char,
            954 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 73],
                &[libc::c_char; 73],
            >(
                b"vl_size vl_kdforest_get_num_nodes_of_tree(const VlKDForest *, vl_uindex)\0",
            ))
                .as_ptr(),
        );
    }
    return (**((*self_0).trees).offset(treeIndex as isize)).numUsedNodes;
}
#[no_mangle]
pub unsafe extern "C" fn vl_kdforest_get_depth_of_tree(
    mut self_0: *const VlKDForest,
    mut treeIndex: vl_uindex,
) -> vl_size {
    if treeIndex < (*self_0).numTrees {} else {
        __assert_fail(
            b"treeIndex < self->numTrees\0" as *const u8 as *const libc::c_char,
            b"vl/kdtree.c\0" as *const u8 as *const libc::c_char,
            968 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 69],
                &[libc::c_char; 69],
            >(b"vl_size vl_kdforest_get_depth_of_tree(const VlKDForest *, vl_uindex)\0"))
                .as_ptr(),
        );
    }
    return (**((*self_0).trees).offset(treeIndex as isize)).depth as vl_size;
}
#[no_mangle]
pub unsafe extern "C" fn vl_kdforest_get_num_trees(
    mut self_0: *const VlKDForest,
) -> vl_size {
    return (*self_0).numTrees;
}
#[no_mangle]
pub unsafe extern "C" fn vl_kdforest_set_max_num_comparisons(
    mut self_0: *mut VlKDForest,
    mut n: vl_size,
) {
    (*self_0).searchMaxNumComparisons = n;
}
#[no_mangle]
pub unsafe extern "C" fn vl_kdforest_get_max_num_comparisons(
    mut self_0: *mut VlKDForest,
) -> vl_size {
    return (*self_0).searchMaxNumComparisons;
}
#[no_mangle]
pub unsafe extern "C" fn vl_kdforest_set_thresholding_method(
    mut self_0: *mut VlKDForest,
    mut method: VlKDTreeThresholdingMethod,
) {
    if method as libc::c_uint == VL_KDTREE_MEDIAN as libc::c_int as libc::c_uint
        || method as libc::c_uint == VL_KDTREE_MEAN as libc::c_int as libc::c_uint
    {} else {
        __assert_fail(
            b"method == VL_KDTREE_MEDIAN || method == VL_KDTREE_MEAN\0" as *const u8
                as *const libc::c_char,
            b"vl/kdtree.c\0" as *const u8 as *const libc::c_char,
            1029 as libc::c_int as libc::c_uint,
            (*::core::mem::transmute::<
                &[u8; 83],
                &[libc::c_char; 83],
            >(
                b"void vl_kdforest_set_thresholding_method(VlKDForest *, VlKDTreeThresholdingMethod)\0",
            ))
                .as_ptr(),
        );
    }
    (*self_0).thresholdingMethod = method;
}
#[no_mangle]
pub unsafe extern "C" fn vl_kdforest_get_thresholding_method(
    mut self_0: *const VlKDForest,
) -> VlKDTreeThresholdingMethod {
    return (*self_0).thresholdingMethod;
}
#[no_mangle]
pub unsafe extern "C" fn vl_kdforest_get_data_dimension(
    mut self_0: *const VlKDForest,
) -> vl_size {
    return (*self_0).dimension;
}
#[no_mangle]
pub unsafe extern "C" fn vl_kdforest_get_data_type(
    mut self_0: *const VlKDForest,
) -> vl_type {
    return (*self_0).dataType;
}
#[no_mangle]
pub unsafe extern "C" fn vl_kdforestsearcher_get_forest(
    mut self_0: *const VlKDForestSearcher,
) -> *mut VlKDForest {
    return (*self_0).forest;
}
