use crate::foreign::vlfeat::generic::*;
use crate::foreign::vlfeat::hikmeans::*;
use nalgebra::DMatrix;

/// Hierarchical Integer K-means 
pub struct HIKM {
    tree : *mut VlHIKMTree,
    k : usize,      // Number of clusters
    dim : usize,    // Data dimensionality
    depth : usize   // Tree depth
}

impl HIKM {

    /// k : Number of clusters per node; dim : dimension of each integer data vector; depth : Tree depth.
    pub fn new(k : usize, dim : usize, depth : usize, n_iter : usize) -> Self {
        // VlIKMAlgorithms_VL_IKM_LLOYD; VlIKMAlgorithms_VL_IKM_ELKAN
        unsafe {
            let tree : *mut VlHIKMTree = vl_hikm_new(VlIKMAlgorithms_VL_IKM_ELKAN as i32);
            vl_hikm_set_verbosity(tree, 0); // OR 1
            vl_hikm_set_max_niters(tree, n_iter as i32);
            vl_hikm_init(tree, dim as u64, k as u64, depth as u64);
            Self{ tree, k, dim, depth }
        }
    }

    pub fn train(&mut self, data : &[u8]) {
        assert!(data.len() % self.dim == 0);
        let n_obs = data.len() / self.dim;
        unsafe {
            vl_hikm_train (self.tree, data.as_ptr(), n_obs as u64);
        }
    }
    
    pub fn predict(&mut self, data : &[u8]) -> DMatrix<u32> {
        // Takes the data rows (K x N_DATA POINTS) and write their path along the tree into the asgn variable. asgn must point
        // to a pre-allocated array of (TREE_DEPTH x N_DATA POINTS)
        assert!(data.len() % self.dim == 0);
        let n_obs = data.len() / self.dim;
        let mut asgn = DMatrix::<u32>::zeros(self.depth, n_obs);
        unsafe {
            vl_hikm_push (self.tree, asgn.as_mut_slice().as_mut_ptr(), data.as_ptr(), n_obs as u64);
        }  
        asgn  
    }
}

impl Drop for HIKM {

    fn drop(&mut self) {
        unsafe{ vl_hikm_delete(self.tree); }    
    }
    
}


