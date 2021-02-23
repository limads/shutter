use opencv::core;
use nalgebra::{DMatrix, DVector};
use crate::image::cvutils;

pub struct KMeans {
    // Number of clusters
    k : usize,
    
    // n x 1 cluster indices
    clust_ixs : DVector<i32>,
    
    // k x n_features
    clust_centers : DMatrix<f32>,
    
    max_iter : usize
}

impl KMeans {

    pub fn new(k : usize, feature_sz : usize, max_iter : usize) -> Self {
        let clust_centers = DMatrix::<f32>::zeros(k, feature_sz);
        let clust_ixs = DVector::<i32>::zeros(1);
        Self { k, clust_ixs, clust_centers, max_iter }
    } 
    
    /// Returns cluster indices from 0..(k-1) of the observation rows at obs.
    pub fn calculate(&mut self, obs : &DMatrix<f32>) -> &[i32] {
        let n = obs.len();
        if n > self.clust_ixs.len() {
            self.clust_ixs = DVector::zeros(n);
        }
        let ty = 1; //count=1; max_iter=1; eps=2;
        let term = core::TermCriteria { 
            typ : ty, 
            max_count : self.max_iter as i32, 
            epsilon : 1.0 
        };
        let n_attempts = 1;
        
        unsafe {
            let obs = cvutils::dmatrix_to_mat(&obs);
            let mut out_labels = cvutils::slice_to_mat(&self.clust_ixs.as_mut_slice()[0..n], 1, None);
            let mut centers = cvutils::dmatrix_to_mat(&self.clust_centers);
            
            core::kmeans(
                &obs, 
                self.k as i32, 
                &mut out_labels, 
                term, 
                n_attempts, 
                core::KMEANS_USE_INITIAL_LABELS, 
                &mut centers
            ).unwrap();
        }
        
        &self.clust_ixs.as_slice()[0..n]
    }
    
}

/*/// K-nearest/Hierarchical cluster bindings are not yet ready at opencv crate.
KDTree implementation of K-nearest neighbors.
pub fn knn(obs : &DMatrix<f64>, k : usize, preds : &mut DMatrix<f64>) {
    let obs = cvutils::dmatrix_to_mat(obs);
    let mut preds = cvutils::dmatrix_to_mat(preds);
    let kn = KNearest::create().unwrap();
    
    // Make predictions (write values of the k-nearest neighbors to each test input (which might be unseen).
    kn.find_nearest(
        &obs
        k as i32
        &mut preds,
        core::no_array(),
        core::no_array()
    ).unwrap();
    
) {
}
let hc = HierarchicalClusteringIndexParams::new(
    branching: i32,
    centers_init: flann_centers_init_t,
    trees: i32,
    leaf_size: i32
).unwrap()*/

/*
VLFeat solution to KMeans:
numData = 5000 ;
dimension = 2 ;
data = rand(dimension,numData) ;
numClusters = 30 ;
[centers, assignments] = vl_kmeans(data, numClusters);
x = rand(dimension, 1) ;
[~, k] = min(vl_alldist(x, centers)) ;

[centers, assignments] = vl_kmeans(data, numClusters, 'Initialization', 'plusplus') ;

'Algorithm' parameter to 'Lloyd', 'Elkan' or 'ANN'

When using the 'ANN' algorithm, the user can also specify the parameters 'MaxNumComparisons' and 'NumTrees'
to configure the KDTree.
*/


