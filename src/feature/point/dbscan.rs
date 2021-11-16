use std::collections::HashMap;
use spade::rtree::RTree;

pub enum Assignment {
    Unvisited,
	Noise,
	Core(usize),
	Border(usize),
}

pub struct PointRef<'a> {
    pt : &'a (usize, usize),
    lin_ix : usize
}

impl<'a> spade::HasPosition for PointRef<'a> {

    type Point = [i32; 2];

    fn position(&self) -> Self::Point {
        [self.pt.0 as i32, self.pt.1 as i32]
    }

}

/// Returns index at parent vector and point.
fn local_neighborhood_linear(
    neigh : &mut Vec<usize>,
    ref_pt : (usize, usize),
    pts : &[(usize, usize)],
    min_dist : f64
) {
    neigh.clear();
	neigh.extend(pts.iter().cloned()
	    .enumerate()
	    .filter(|(_, pt)| *pt != ref_pt && super::point_dist(&ref_pt, pt) < min_dist )
	    .map(|(ix, _)| ix )
	);
}

fn local_neighborhood_indexed(
    neigh : &mut Vec<usize>,
    ref_pt : (usize, usize),
    pts : &[(usize, usize)],
    tree : &RTree<PointRef<'_>>,
    min_dist : f64
) {
    neigh.clear();
    let ref_pt_int = [ref_pt.0 as i32, ref_pt.1 as i32];
    neigh.extend(tree.nearest_neighbor_iterator(&ref_pt_int)
        .take_while(|pt| super::point_dist(&ref_pt, &pt.pt) < min_dist )
        .map(|pt| pt.lin_ix )
    );
}

fn expand_local_neighborhood(
    labels : &mut [Assignment],
    pts : &[(usize, usize)],
    tree : Option<&RTree<PointRef<'_>>>,
    min_dist : f64,
    min_cluster_sz : usize,
    curr_clust : usize,
    n_ix : usize
) {
    let mut inner_neigh = Vec::<usize>::with_capacity(pts.len() / 10);
    match labels[n_ix] {
	    Assignment::Unvisited => {
	        /*let inner_neigh =*/ if let Some(tree) = tree {
                local_neighborhood_indexed(&mut inner_neigh, pts[n_ix], pts, tree, min_dist)
	        } else {
	            local_neighborhood_linear(&mut inner_neigh, pts[n_ix], pts, min_dist)
	        };
	        if inner_neigh.len() > min_cluster_sz {
	            labels[n_ix] = Assignment::Core(curr_clust);
		        for inner_n_ix in inner_neigh {
			        expand_local_neighborhood(labels, pts, tree, min_dist, min_cluster_sz, curr_clust, inner_n_ix);
		        }
	        } else {
	            labels[n_ix] = Assignment::Border(curr_clust);
	        }
	    },
	    _ => {
            // labels[n_ix] = Assignment::Border(curr_clust);
	    }
	}
}

// dbscan implementation (https://en.wikipedia.org/wiki/DBSCAN)
// Receives minimum distance to define core points and minimum number
// of points to determine a cluster allocation. Finding predominantly
// vertical or predominanlty horizontal edges can be done by giving
// asymetric weghts w1, w2 \in [0,1] w1+w2=1 to the vertical and horizontal
// coordinates, effectively "compressing" the points closer either in the
// vertical or horizontal dimension. Returns custers and noise vector.
pub fn dbscan(
    pts : &[(usize, usize)],
    min_dist : f64,
    min_cluster_sz : usize,
    use_indexing : bool
) -> (HashMap<usize, Vec<(usize, usize)>>, Vec<(usize, usize)>) {

	// Point q is **directly reachable** from point p if it is within distance epsilon
	// of core point p.

	// A point q is **indirectly reachable** from p is there is a path of core points where
	// p_i+1 is directly reachable from p_i. q does not need to be a core points,
	// but all other p_i points must.

	// All points not reachable are outliers or noise.

    let tree = if use_indexing {
        let mut rtree : RTree<PointRef<'_>> = RTree::new();
        pts.iter().enumerate().for_each(|(lin_ix, pt)| { rtree.insert(PointRef { lin_ix, pt }); });
        Some(rtree)
    } else {
        None
    };

	let mut labels : Vec<Assignment> = (0..pts.len()).map(|_| Assignment::Unvisited ).collect();
	let mut curr_clust = 0;
    let mut neigh = Vec::new();

	// (1) For each point pt:
	for (ref_ix, ref_pt) in pts.iter().enumerate() {

        // Ignore points that might have been classified at previous iterations
	    match labels[ref_ix] {
	        Assignment::Border(_) | Assignment::Core(_) | Assignment::Noise => continue,
	        _ => { }
	    }

		/*let mut neigh =*/ if let Some(tree) = tree.as_ref() {
            local_neighborhood_indexed(&mut neigh, *ref_pt, pts, tree, min_dist)
		} else {
		    local_neighborhood_linear(&mut neigh, *ref_pt, pts, min_dist)
		};

		// If at least min_pts points are within distance epsilon of a point p,
		// it is a core point.
		if neigh.len() >= min_cluster_sz {
			labels[ref_ix] = Assignment::Core(curr_clust);

			// Search for directly-reachable points to this core point still classified as noise
			for n_ix in &neigh {

				// match labels[n_ix] {
					//Assignment::Noise => {
				expand_local_neighborhood(&mut labels, pts, tree.as_ref(), min_dist, min_cluster_sz, curr_clust, *n_ix);
					// },
					//_ => {
					// Ignore points already classified as core or border
					//}
				// }

			}

			curr_clust += 1;
		} else {
		    labels[ref_ix] = Assignment::Noise;
		}
	}

	let mut clusters = HashMap::new();
	let mut noise = Vec::new();
	for (lbl_ix, lbl) in labels.iter().enumerate() {
		match lbl {
			Assignment::Core(ix) | Assignment::Border(ix) => {
				clusters.entry(*ix).or_insert(Vec::new()).push(pts[lbl_ix]);
			},
			Assignment::Noise => {
                noise.push(pts[lbl_ix]);
			},
			Assignment::Unvisited => { panic!("Unvisited point"); }
		}
	}

	(clusters, noise)
}

