use petgraph::graph::UnGraph;
use petgraph::graph::NodeIndex;
use petgraph::unionfind::UnionFind;
use petgraph::visit::EdgeRef;
use petgraph::visit::NodeIndexable;
use std::ops::Range;
use std::collections::BTreeMap;
use std::iter::FromIterator;
use itertools::Itertools;

/// A split graph represents indices belonging to the same group contiguously,
/// and represents each group by a range of indices in the second vector.
/// Within each group, the node indices are ordered, which is useful if the
/// graph is also ordered.
pub struct SplitGraph {
    pub indices : Vec<NodeIndex>,
    pub ranges : Vec<Range<usize>>
}

impl SplitGraph {

    pub fn new<N, E>(
        graph : &UnGraph<N, E>,
        uf : &UnionFind<NodeIndex>
    ) -> SplitGraph {
        let mut labels = uf.clone().into_labeling();
        labels.truncate(graph.raw_nodes().len());
        let counts = labels.iter().counts();
        let total_len = counts.iter().fold(0, |total, (_, c)| total + c );
        let mut indices = Vec::from_iter((0..total_len).map(|_| NodeIndex::new(0) ));
        let mut ranges_map : BTreeMap<NodeIndex, Range<usize>> = BTreeMap::new();
        let mut curr_start = 0;
        for (k, v) in counts.iter() {
            ranges_map.insert(**k, Range { start : curr_start, end : curr_start });
            curr_start += v;
        }
        for node_ix in graph.node_indices() {
            // let parent_ix = uf.find(node_ix);
            let parent_ix = labels[node_ix.index()];
            indices[(ranges_map[&parent_ix].end)] = node_ix;
            ranges_map.get_mut(&parent_ix).unwrap().end += 1;
        }
        let ranges = ranges_map.iter().map(|(_, r)| r.clone() ).collect();
        let split = SplitGraph {
            indices,
            ranges
        };

        verify_integrity(&split);

        split
    }

    // Return a handle to a unique element of each group. Can be used
    // to start a search restricted to each group.
    pub fn first_nodes<'a>(&'a self) -> impl Iterator<Item=NodeIndex> + 'a {
        self.ranges.iter().map(move |r| self.indices[r.start] )
    }

}

fn verify_integrity(s : &SplitGraph) {

    // Verify group at each range is sorted.
    for r in &s.ranges {
        if r.end - r.start >= 2 {
            let iter = s.indices[r.clone()].iter();
            for (a, b) in iter.clone().zip(iter.skip(1)) {
                assert!(b >= a);
            }
        }
    }
}


