use crate::image::*;
use petgraph::graph::UnGraph;
use itertools::Itertools;
use std::ops::Range;
use parry2d::utils::Interval;
use petgraph::unionfind::UnionFind;
use std::mem;
use petgraph::graph::NodeIndex;
use crate::graph::SplitGraph;

#[repr(u8)]
#[derive(Debug, Clone)]
pub enum Neigborhood {

    /// Four-neighborhood (top, below, left, right pixels) (Aka. Von Neumann neighborhood),
    /// characterized by CardinalDirection
    Immediate,

    /// Eight-neighborhood (all elements at immediate neighborhood plus
    /// top-left, top-right, bottom-left and bottom-right pixels) (Aka. Moore neighborhood),
    /// characterized by Direction
    Extended

}

/// Represents one of the four cardinal directions in a 4-neighborhood pixel connectivity graph
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum CardinalDirection {
    North,
    East,
    South,
    West
}

/// Represents one of the eight directions in a 8-neighborhood pixel connectivity graph
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum Direction {
    North,
    NorthEast,
    East,
    SouthEast,
    South,
    SouthWest,
    West,
    NorthWest
}

/// Represents a horizontal sequence of homogeneous pixels by the coordinate of
/// the first pixel and the length of the sequence. The RunLength is the basis
/// for a sparse representation of a binary image.
#[derive(Debug, Clone, Copy)]
pub struct RunLength {
    pub start : (usize, usize),
    pub length : usize
}

impl RunLength {

    /// Represents a closed interval covered by this run-length
    pub fn interval(&self) -> Interval<usize> {
        Interval(self.start.1, self.start.1+self.length-1)
    }

    pub fn intersect(&self, other : &Self) -> Option<RunLength> {
        self.intersect_interval(other.interval())
    }

    /// Returns the subset of Self that intersects vertically with the given interval
    pub fn intersect_interval(&self, intv : Interval<usize>) -> Option<RunLength> {
        self.interval().intersect(intv)
            .map(|intv| RunLength { start : (self.start.0, intv.0), length : intv.1-intv.0+1 } )
    }

    // It never makes sense to check for contact of RLEs at the same row,
    // because two neighboring RLEs should have been merged into one. So
    // contacts only test elements in different rows.
    pub fn contacts(&self, other : &Self) -> bool {
        (self.start.0 as i32 - other.start.0 as i32).abs() == 1 &&
            self.interval_overlap(other)
    }

    // Panics if Other > self.
    pub fn interval_overlap(&self, other : &Self) -> bool {
        // crate::region::raster::interval_overlap(self.interval(), other.interval())
        self.interval().intersect(other.interval()).is_some()
    }

}

pub struct RunLengthEncoder {
    rles : Vec<RunLength>,
    rows : Vec<Range<usize>>
}

impl RunLengthEncoder {

    pub fn state(&self) -> (&[RunLength], &[Range<usize>]) {
        (&self.rles[..], &self.rows[..])
    }

    pub fn calculate(w : &Window<u8>) -> Self {
        let mut this = Self { rles : Vec::new(), rows : Vec::new() };
        this.update(w);
        this
    }

    // pub fn split(&self) -> crate::graph::SplitGraph {
    //    crate::graph::group_weights(&self.graph, &self.uf)
    // }

    pub fn build_graph(&self) -> RunLengthGraph {
        let nrows = self.rows.len();

        let mut graph = UnGraph::<RunLength, Interval<usize>>::new_undirected();
        let mut uf = UnionFind::new(self.rles.len());
        if self.rows.len() == 0 {
            return RunLengthGraph { graph, uf };
        }

        let mut last_matched_above : Option<&RunLength> = None;
        let mut curr_ixs = Vec::new();
        let mut past_ixs = Vec::new();

        // Populate graph initially with first row
        for rl in &self.rles[self.rows[0].clone()] {
            past_ixs.push(graph.add_node(*rl));
        }

        // println!("Intersection = {:?}", Interval(1usize,1usize).intersect(Interval(1usize,1usize)));

        let row_pair_iter = self.rows[0..(nrows-1)].iter()
            .zip(self.rows[1..nrows].iter());
        for (row_above, row_below) in row_pair_iter {
            for rl_below in &self.rles[row_below.clone()] {

                // Add current bottom RLE
                let below_ix = graph.add_node(*rl_below);
                curr_ixs.push(below_ix);

                /* Decide on the RunLenght connectivity strategy: If the strict overlap
                is desired, we are working with 4-neighborhood; If diagonal linking (8-neighborhood)
                is desired, the diagonally-connecting intervals will not share an overlap */

                // Iterate overr overlapping top RLEs (since they are ordered, there is no
                // need to check the overlap of intermediate elements:
                // only the start and end matching RLEs).
                let iter_above = self.rles[row_above.clone()].iter()
                    .enumerate()

                    // use this for diagonally-linking RLEs (CANNOT have intersections as weights)
                    // .skip_while(|(_, above)| (above.start.1+above.length+1 ) < rl_below.start.1 )
                    // .take_while(|(_, above)| above.start.1 < (rl_below.start.1+rl_below.length+1 ) )

                    // Use this for strictly overlapping RLEs (intersections can be weights).
                    .skip_while(|(_, above)| (above.start.1+above.length-1) < rl_below.start.1 )
                    .take_while(|(_, above)| above.start.1 <= (rl_below.start.1+rl_below.length-1) )

                    .map(|(ix, above)| (ix, rl_below.intersect(&above).unwrap().interval() ) );

                // Add edges to top RLEs
                for (above_ix, intv) in iter_above {
                    graph.add_edge(below_ix, past_ixs[above_ix], intv);
                    uf.union(below_ix, past_ixs[above_ix]);
                }
            }
            mem::swap(&mut past_ixs, &mut curr_ixs);
            curr_ixs.clear();
        }

        RunLengthGraph { graph, uf }
    }

    /*
    Run-length encoding is trivially parallelizable by splitting an image with n
    rows into 4 images with dimensions n/4 x m, running it separately on the images,
    then polling the results back together.
    */
    pub fn update(&mut self, w : &Window<u8>) -> (&[RunLength], &[Range<usize>]) {
        self.rles.clear();
        self.rows.clear();
        let mut last_rle : Option<RunLength> = None;
        for r in 0..w.height() {
            self.rows.push(Range { start : self.rles.len(), end : self.rles.len() });
            for c in 0..w.width() {
                if w[(r, c)] == 0 {
                    if let Some(rle) = last_rle.take() {
                        self.rles.push(rle);
                        self.rows.last_mut().unwrap().end += 1;
                    }
                } else {
                    if let Some(mut rle) = last_rle.as_mut() {
                        rle.length += 1;
                    } else {
                        last_rle = Some(RunLength { start : (r, c), length : 1 });
                    }
                }
            }
            if let Some(rle) = last_rle.take() {
                self.rles.push(rle);
                self.rows.last_mut().unwrap().end += 1;
            }
        }
        (&self.rles[..], &self.rows[..])
    }

}

pub fn draw_distinct(img : &mut WindowMut<u8>, graph : &UnGraph<RunLength, (usize, usize)>, split : &SplitGraph) {
    for range in &split.ranges  {
        let r : f64 = rand::random();
        let color : u8 = 64 + ((256. - 64.)*r) as u8;
        for ix in split.indices[range.clone()].iter() {
            img[graph[*ix]].iter_mut().for_each(|px| *px = color );
        }
    }
}

/// A graph of connected RunLength elements is much cheaper to represent than a graph of connected pixels,
/// and encode the same set of spatial relationships by making horizontal pixel
/// connections implicit (all pixels represented by a RunLength are connected; horizontal
/// pixels in the same row but different RunLengths are disconnected; vertical RunLengths
/// are connected by graph edges). The weights can represent start column and end column of the
/// relative overlap. RLEs stay ordered in the graph: A graph index B > graph Index A means
/// row(B) >= row(A), and col(B) > col(A). graph.node_indices() therefore also represent a
/// raster order.
pub struct RunLengthGraph {

    // Edges carry the intersection of the RunLenghts.
    pub graph : UnGraph<RunLength, Interval<usize>>,

    pub uf : UnionFind<NodeIndex>

}

impl RunLengthGraph {

    // Since the graph is in raster order, the index of all labels in the union find distinct
    // from the previous in a serial iterator over the union find candidate top-left
    // element in a new group. Sorting and taking the distinct of those candidates gives
    // the first top-left element. The first index of each group can be used as a handle to start
    // a depth or breadth search over the graph restricted to the current group. Consider using
    // splitgraph::first_nodes instead to avoid calling into_labeling more than once.
    pub fn first_nodes(&self) -> Vec<NodeIndex> {
        let labels = self.uf.clone().into_labeling();
        let mut fst_nodes = Vec::new();
        if labels.len() == 0 {
            return fst_nodes;
        }
        fst_nodes.push(NodeIndex::new(0));
        for (pair_ix, (prev, curr)) in labels.iter().zip(labels.iter().skip(1)).enumerate() {
            if *curr != *prev {
                fst_nodes.push(NodeIndex::new(pair_ix+1));
            }
        }
        fst_nodes.sort();
        fst_nodes.dedup();
        fst_nodes
    }

    /// Returns several rects contained in grouped  RunLengths. This returns
    /// a separate rect for all possible depth-first searches.
    pub fn inner_rects(&self) -> Vec<Vec<(usize, usize, usize, usize)>> {
        /*let mut rects = Vec::new();
        // The first nodeindex at each range is always the RLE to the top-left in raster order.
        for fst_ix in self.first_nodes() {

            let mut dfs = Dfs::new(&self.graph, &self.graph[fst_ix]);

            let mut grouped_rects = Vec::new();
            let mut overlap = self.graph[fst_ix];
            while let Some(nx) = dfs.next() {

            }
        }*/
        unimplemented!()
    }

    pub fn split(&self) -> crate::graph::SplitGraph {
        crate::graph::group_weights(&self.graph, &self.uf)
    }

}


