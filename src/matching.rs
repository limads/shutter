use opencv::features2d;

/// A matching algorithm starts with two sets of size M and N. It assumes all observations
/// at the smallest set have a matching observation at the test set. The match explores the space of 
/// MN possible matches, extracting the ones that are most probably true. To resolve the match,
/// we iterate over the metric matrix, taking for example the 10% best matches based on some criteira
/// (irrespective of the keypoint positions). To make matches unique, each row/column must have at
/// most one matching observation. We can then build a histogram over match distances and relative
/// angles. If the histograms has low variance, it means the matches have a common source of variability
/// such as a common rotation or translation. We can estimate this translation/rotation from this histogram
/// to extract geometrical image transformations. If we expect severe scale changes, we do the match
/// across multiple DWT scales, trying all possible combinations of different scale matches (although
/// we should correct the translation values across different images). If we expect severe rotation changes,
/// we aument the test set (largest set) with sets of rotated versions of the keypoints, doing the match
/// across the multiple rotations and picking the one with the least global dissimilarity.
pub struct Match {

}


// features2d::FlannBasedMatcher::new(index_params, search_params);
