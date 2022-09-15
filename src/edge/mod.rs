use std::mem;
use std::iter::FromIterator;
use std::cmp::PartialOrd;
use petgraph::visit;
use std::fmt::Debug;
use nalgebra::Scalar;
use num_traits::AsPrimitive;
use petgraph::graph::UnGraph;
use petgraph::graph::NodeIndex;
use petgraph::unionfind::UnionFind;
use petgraph::visit::EdgeRef;
use petgraph::visit::NodeIndexable;
use crate::code::Direction;
use itertools::Itertools;
use std::collections::HashMap;
use std::ops::Range;
use crate::image::*;

pub struct Edge(Vec<(usize, usize)>);

impl From<Vec<(usize, usize)>> for Edge {

    fn from(v : Vec<(usize, usize)>) -> Self {
        Self(v)
    }

}

impl Edge {

    pub fn points(&self) -> &[(usize, usize)] {
        &self.0[..]
    }

}

pub trait Tracer {

    fn trace<N>(&mut self, win : &Window<N>)
    where
        N : Pixel + PartialOrd + AsPrimitive<f32>,
        f32 : AsPrimitive<N>,
        for<'a> &'a [N] : Storage<N>;

}

// Carries coordinate and label. TODO also wrap the UnionFind here.
pub type EdgeGraph = UnGraph<(usize, usize), Direction>;

#[derive(Debug, Clone)]
pub struct EdgeGraphTracer {

    thr : f32,

    sub_sz : usize,

    pub graph : EdgeGraph,

    // Each node in the graph has a separate label held here, corresponding
    // to its assigned group.
    pub edge_sets : UnionFind<NodeIndex>
}

/*pub struct GroupedEdges<'a> {
    slices : Vec<&'a [NodeIndex]>,
    memberships : Vec<usize>
}*/

impl EdgeGraphTracer {

    pub fn grouped_weights(&self) -> crate::graph::SplitGraph {
        crate::graph::SplitGraph::new(&self.graph, &self.edge_sets)
    }

    // Makes two passes over the UnionFind (one at into_labeling, other at
    // the implementation here) BUT makes a single allocation and does not require moving content
    // while the pixel positions are being written.
    pub fn split(&self) -> crate::graph::SplitGraph {
        crate::graph::SplitGraph::new(&self.graph, &self.edge_sets)
    }

    // Make a single pass over the UnionFind, BUT must call one .insert(.) for each pixel.
    /*pub fn grouped_weights_with_insert(&self) -> (Vec<(usize, usize)>, Vec<Range<usize>>) {
        let mut grouped_ranges = HashMap::new();
        let mut pts = Vec::new();
        let mut ix_inserted : Option<usize> = None;
        for node_ix in self.graph.node_indices() {
            let parent_ix = self.edge_sets.find(node_ix);
            match grouped_ranges.get_mut(parent_ix) {
                Some(mut range) => {
                    pts.insert(range.end-1, self.graph[node_ix]);
                    range.end += 1;
                    ix_inserted = Some(range.end);
                },
                None => {
                    pts.push(self.graph[node_ix]);
                    grouped_ranges.insert(parent_ix, Range { start : pts.len()-1, end : pts.len() });
                    ix_inserted = None;
                }
            }
            if let Some(ix) = ix_inserted {
                for (_, mut v) in grouped_ranges.iter_mut() {
                    if v.start >= ix {
                        v.start += 1;
                        v.end += 1;
                    }
                }
            }
        }
        grouped
    }*/

    pub fn grouped_indices(&self) -> HashMap<NodeIndex, Vec<NodeIndex>> {
        let mut grouped = HashMap::new();
        for node_ix in self.graph.node_indices() {
            let parent_ix = self.edge_sets.find(node_ix);
            let mut v = grouped.entry(parent_ix).or_insert(Vec::default());
            v.push(node_ix);
        }
        grouped
    }

    /*fn row_grouped_edges(&self) -> Vec<&[NodeIndex]> {
        let mut last_parent = NodeIndex::new(0);
        let mut last_parent_offset = 0;
        let mut set_sz = 0;
        let mut curr_slice = &[];
        for (i, node_ix) in self.graph.node_indices().enumerate() {
            let parent_ix = edgeg.edge_sets.find(node_ix);
            if parent_ix == last_parent {
                set_sz += 1;
            } else {

            }
        }
    }*/

    /*fn group_edges(&self) -> Vec<Vec<NodeIndex>> {
        for node_ix in self.graph.node_indices() {
            let parent_ix = edgeg.edge_sets.find(node_ix);
            let mut e = colors.entry(parent_ix);
            match &e {
                Entry::Vacant(_) => {
                    if let Some(c) = color.checked_add(8) {
                        color = c;
                    } else {
                        color = 8;
                    }
                },
                _ => { }
            }
            let c = e.or_insert(color);
            canny_out[edgeg.graph[node_ix]] = *c;
        }
    }*/

    // This is based on petgraph::algo::connected_components.
    fn edge_groups(&self) -> Vec<usize> {
        let max_sz = self.graph.node_bound();
        let mut vertex_sets = UnionFind::new(max_sz);
        for edge in self.graph.edge_references() {
            let (a, b) = (edge.source(), edge.target());
            vertex_sets.union(self.graph.to_index(a), self.graph.to_index(b));
        }
        let mut labels = vertex_sets.into_labeling();
        println!("{:?}", labels);

        // labels.sort_unstable();
        // labels.dedup();
        // labels
        // unimplemented!()
        labels
    }

    // Returns disjoint edges (i.e. sets of pixels that are not connected between them,
    // but are connected within them).
    fn edge_groups_alt(&self) -> Vec<Vec<NodeIndex>> {
        let mut labeled : Vec<bool> = (0..self.graph.node_count()).map(|x| false ).collect();
        let mut groups = Vec::new();
        while let Some(ix) = labeled.iter().position(|is_labeled| !is_labeled ) {
            let mut new_group = Vec::new();
            let mut bfs = visit::Bfs::new(&self.graph, NodeIndex::new(ix));
            while let Some(reachable_ix) = bfs.next(&self.graph) {
                new_group.push(reachable_ix);
                labeled[self.graph.to_index(reachable_ix)] = true;
            }
            groups.push(new_group);
        }
        groups
    }

    // Keep only grouped pixels with two connections.
    fn clean_edge_groups(&self) -> Vec<Vec<NodeIndex>> {
        /*let mut groups = self.edge_groups();
        for mut group in groups.iter_mut() {
            for ix in (0..group.len()).rev() {
                let n_edges = self.graph.edges(group[ix]).count();
                if n_edges != 1 && n_edges != 2 {
                    group.remove(ix);
                }
            }
        }
        groups.retain(|g| g.len() >= 2 );*/

        // At this point there must be exactly two nodes with single links to the rest of
        // the graph. Those are the extreme points. Use them to verify via has_path_connecting.

        // By removing certain elements here, the groups do not necessarily
        // have a path between all their elements anymore. Keep only those that still have.
        // algo::has_path_connectinig(&graph, node_a, node_b, None);

        // groups
        unimplemented!()
    }

    pub fn new(thr : f32, sub_sz : usize) -> Self {
        Self { thr, sub_sz, graph : EdgeGraph::default(), edge_sets : UnionFind::new(1) }
    }

    pub fn state(&self) -> &EdgeGraph {
        &self.graph
    }

}

/*
Patches can be more economically represented via a run-length
encoding over the image, then as a graph of connected rows.
Rows still preserve the Direction connections. Connectivity is
then defined by row_vert_dist(row1, row2) < 1 and cols_overlap(row1, row2)

To retrieve homogenoeus regions, apply edge filter, then invert the image,
then calculate the run-length over the homoegeneous regions, which will be
foreground.

This is based on the fact that we ususally don't care about individual pixel
connections, as long as we know that a bunch of pixels side-by-side are
always connected (run-length encoding).
*/

/*
Representing image edges in an undirected graph + and unionfind has the nice implications that:

(1) if it has exactly two elements with a single graph edge each, it is an clean open edge.
(2) if all its elements have two graph edges each, it is a clean closed shape.
(3) If a single-cycle subgraph can be found, with all elements having at least two graph edges each,
it contains a noisy closed shape that can be extracte from it. If more cycles can be found, the graphs
can be partitioned accordingly.

Plus, it is cheaper to calculate than a full-pixel connected patches graph.
*/

fn graph_growth_step<'a>(
    graph : &mut EdgeGraph,
    edge_sets : &mut UnionFind<NodeIndex>,
    curr_row : &'a mut Vec<Option<NodeIndex>>,
    last_row : &'a mut Vec<Option<NodeIndex>>,
    vf : f32,
    thr : f32,
    row_ix : usize,
    col_ix : usize,
    max_cols : usize
) {
    if col_ix == 0 {
        mem::swap(last_row, curr_row);
    }

    if vf >= thr {
        // let new_graph_ix = graph.add_node((w.offset().0 + y, w.offset().1 + x));
        let new_graph_ix = graph.add_node((row_ix, col_ix));
        curr_row[col_ix] = Some(new_graph_ix);

        let tl = if col_ix == 0 { None } else { last_row[col_ix-1] };
        let top = last_row[col_ix];
        let tr = if col_ix == max_cols-1 { None } else { last_row[col_ix+1] };

        if let Some(tl_ix) = tl {
            graph.add_edge(tl_ix, new_graph_ix, Direction::NorthWest);
            edge_sets.union(tl_ix, new_graph_ix);
        }

        if let Some(top_ix) = top {
            graph.add_edge(top_ix, new_graph_ix, Direction::North);
            edge_sets.union(top_ix, new_graph_ix);
        }

        if let Some(tr_ix) = tr {
            graph.add_edge(tr_ix, new_graph_ix, Direction::NorthEast);
            edge_sets.union(tr_ix, new_graph_ix);
        }

        // Link to last element to left of current row.
        if col_ix > 0 {
            if let Some(left_ix) = curr_row[col_ix-1] {
                graph.add_edge(left_ix, new_graph_ix, Direction::West);
                edge_sets.union(left_ix, new_graph_ix);
            }
        }
    } else {
        curr_row[col_ix] = None;
    }
}

impl Tracer for EdgeGraphTracer {

    fn trace<N>(&mut self, win : &Window<N>)
    where
        N : Pixel + PartialOrd + AsPrimitive<f32>,
        f32 : AsPrimitive<N>,
        for<'a> &'a [N] : Storage<N>
    {
        self.graph.clear();

        // Ideally, this is the number of nodes in the graph. But we do not know it
        // yet, so there will be many unassigned sets of size 1, each with a different
        // index. But this does not matter, since all sets of size 1 are filtered out
        // in the end, not being important if they are assigned, 1-pixel sets or just
        // sets that do not represent any pixel.
        self.edge_sets = UnionFind::new((win.height() / 2) * (win.width() / 2));

        /*let wins_per_row = win.width() / self.sub_sz;
        let mut last_row = Vec::with_capacity(wins_per_row);
        let mut curr_row = Vec::with_capacity(wins_per_row);
        last_row.extend((0..wins_per_row).map(|_| None ));
        curr_row.extend((0..wins_per_row).map(|_| None ));

        let mut curr_label = 0i32;

        for (w_ix, w) in win.windows((self.sub_sz, self.sub_sz)).enumerate() {
            let col_ix = w_ix % wins_per_row;
            let row_ix = w_ix / wins_per_row;
            let (_y, _x, v) = crate::local::min_max_idx(&w, false, true).1.unwrap();

            // Note: We ignore (y, x) and use the window index instead (but we could
            // use the (y,x) to keep the original resolution, BUT it would require
            // chaging the "clean" step (which is when we switch rows).
            graph_growth_step(
                &mut self.graph,
                &mut self.edge_sets,
                &mut curr_row,
                &mut past_row,
                v.as_(),
                self.thr,
                row_ix,
                col_ix
            );
        }*/

        let mut last_row = Vec::with_capacity(win.width());
        last_row.extend((0..win.width()).map(|_| None ));
        let mut curr_row = last_row.clone();
        for (i, j, px) in win.labeled_pixels::<usize, _>(1) {
            graph_growth_step(
                &mut self.graph,
                &mut self.edge_sets,
                &mut curr_row,
                &mut last_row,
                px.as_(),
                self.thr,
                i,
                j,
                win.width()
            );
        }

        // Perhaps call graph.retain_nodes(|node| node.edges().count() >= 1 ) to remove isolated points.
    }
}

/* Distante & Distante (p. 291)
(1) Find pixel with magn > thresh
(2) Take neighbor at 4 or 8-connected region w/ highest magnitude and similar orientation (within pi/4).
Repeat this step while condition is met.
(3) Case (2) not met
*/

#[cfg(feature="ipp")]
pub unsafe fn ippi_canny(
    src_dx : &Window<f32>, 
    src_dy : &Window<f32>, 
    dst : &mut WindowMut<u8>, 
    thr : (f32, f32)
) {

    let (step_dx, sz_dx) = crate::image::ipputils::step_and_size_for_window(src_dx);
    let (step_dy, sz_dy) = crate::image::ipputils::step_and_size_for_window(src_dy);
    let (dst_step, dst_sz) = crate::image::ipputils::step_and_size_for_window_mut(&dst);

    let mut sz : i32 = 0;
    let ans = crate::foreign::ipp::ippcv::ippiCannyGetSize(mem::transmute(sz_dx), &mut sz as *mut _);
    assert!(ans == 0);
    let mut buf = Vec::from_iter((0..(sz as usize)).map(|_| 0u8 ));

    // TODO ipp requires sources to be mutable pointers. Verify they aren't changing the inputs.
    let ans = crate::foreign::ipp::ippcv::ippiCanny_32f8u_C1R(
        mem::transmute(src_dx.as_ptr()),
        step_dx,
        mem::transmute(src_dy.as_ptr()),
        step_dy,
        dst.as_mut_ptr(),
        mem::transmute(dst_step),
        mem::transmute(dst_sz),
        thr.0,
        thr.1,
        buf.as_mut_ptr()
    );
    assert!(ans==0);

}

/*
Canny:
(1) Convolve with vertical and horizontal sobel/prewit filters
(2) Calculate |g|=sqrt(dx^2+dy^2) (hypot) and theta=atan2(dy,dx).
(3) Round the real angle image to discrete labels mapping to 0,45,90 or 135 (horizontal, right_diagonal, vertical, left_diagonal).
(4) Apply local non-maximal supression, comparing the two pixels in the 3-neighborhood that have
direction matching the gradient label
(4) Apply the lower and upper gradient magnitudue threshold to remove remaining noise pixels not already removed
by non-maximal supression.
*/


