use petgraph::graph::UnGraph;
use petgraph::graph::NodeIndex;
use petgraph::unionfind::UnionFind;
use petgraph::visit::EdgeRef;
use petgraph::visit::NodeIndexable;
use std::ops::Range;
use std::collections::HashMap;
use std::iter::FromIterator;
use itertools::Itertools;

pub struct SplitGraph {
    pub indices : Vec<NodeIndex>,
    pub ranges : Vec<Range<usize>>
}

pub fn group_weights<N, E>(
    graph : &UnGraph<N, E>,
    uf : &UnionFind<NodeIndex>
) -> SplitGraph {
    let mut labels = uf.clone().into_labeling();
    labels.truncate(graph.raw_nodes().len());
    let counts = labels.iter().counts();
    let total_len = counts.iter().fold(0, |total, (_, c)| total + c );
    let mut indices = Vec::from_iter((0..total_len).map(|_| NodeIndex::new(0) ));
    let mut ranges_hash : HashMap<NodeIndex, Range<usize>> = HashMap::new();
    let mut curr_start = 0;
    for (k, v) in counts.iter() {
        ranges_hash.insert(**k, Range { start : curr_start, end : curr_start });
        curr_start += v;
    }
    for node_ix in graph.node_indices() {
        // let parent_ix = uf.find(node_ix);
        let parent_ix = labels[node_ix.index()];
        indices[(ranges_hash[&parent_ix].end)] = node_ix;
        ranges_hash.get_mut(&parent_ix).unwrap().end += 1;
    }
    SplitGraph {
        indices,
        ranges : ranges_hash.iter().map(|(_, r)| r.clone() ).collect()
    }
}


