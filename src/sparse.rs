use crate::image::*;
use petgraph::graph::UnGraph;
use itertools::Itertools;
use std::ops::Range;
use parry2d::utils::Interval;
use petgraph::unionfind::UnionFind;
use std::mem;
use petgraph::graph::NodeIndex;

#[repr(u8)]
#[derive(Debug, Clone)]
pub enum Neigborhood {

    /// Four-neighborhood (top, below, left, right pixels) (Aka. Von Neumann neighborhood)
    Immediate,

    /// Eight-neighborhood (all elements at immediate neighborhood plus
    /// top-left, top-right, bottom-left and bottom-right pixels) (Aka. Moore neighborhood)
    Extended

}

#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum Direction {
    NorthWest,
    North,
    NorthEast,
    East,
    SouthEast,
    South,
    SouthWest,
    West
}

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

    pub fn build_graph(&self) -> RunLengthGraph {
        let nrows = self.rows.len();

        let mut graph = UnGraph::<RunLength, Direction>::new_undirected();
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

        let row_pair_iter = self.rows[0..(nrows-1)].iter()
            .zip(self.rows[1..nrows].iter());
        for (row_above, row_below) in row_pair_iter {
            for rl_below in &self.rles[row_below.clone()] {

                // Add current bottom RLE
                let below_ix = graph.add_node(*rl_below);
                curr_ixs.push(below_ix);

                // Iterate overr overlapping top RLEs (since they are ordered, there is no
                // need to check the overlap of intermediate elements:
                // only the start and end matching RLEs).
                let iter_above = self.rles[row_above.clone()].iter()
                    .enumerate()
                    .skip_while(|(_, r)| (r.start.1+r.length) < rl_below.start.1-1 )
                    .take_while(|(_, r)| r.start.1 < (rl_below.start.1+rl_below.length+1) );

                // Add edges to top RLEs
                for (above_ix, _) in iter_above {
                    graph.add_edge(below_ix, past_ixs[above_ix], Direction::North);
                    uf.union(below_ix, past_ixs[above_ix]);
                }
            }
            mem::swap(&mut past_ixs, &mut curr_ixs);
            curr_ixs.clear();
        }

        RunLengthGraph { graph, uf }
    }

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

pub struct RunLengthGraph {

    pub graph : UnGraph<RunLength, Direction>,

    pub uf : UnionFind<NodeIndex>

}

impl RunLengthGraph {

    pub fn split(&self) -> crate::graph::SplitGraph {
        crate::graph::group_weights(&self.graph, &self.uf)
    }

}


