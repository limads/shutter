/// The object matching problem maps local features from an object or abstract object representation
/// to another object with the objective of alignment and recognition. For point matching, represent
/// a local feature with a numeric vector that can be compared with others in a metric space. As
/// long as this is possible, all vectors can be ordered w.r.t. a "vector origin", allowing binary search,
/// or can be positioned into a KDTree (RectTree) for spatial search.
pub trait Matching {

}

/// Local feature extractor
pub trait Extractor {

}

/// (1) Calculate dx and dy of image;
/// (2) Calculate three image products dxdx dxdy dydy
/// (3) Get local maxima of det(H) = dxdx * dydy - dxy^2 (non-maximum supression).
pub struct HessianExtractor {

}

pub struct HarrisExtractor {

}
