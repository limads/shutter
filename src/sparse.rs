use crate::image::*;
use petgraph::graph::UnGraph;
use itertools::Itertools;
use std::ops::Range;
use parry2d::utils::Interval;
use petgraph::unionfind::UnionFind;
use std::mem;
use petgraph::graph::NodeIndex;
use crate::graph::SplitGraph;
use std::collections::BTreeMap;
use petgraph::prelude::*;
use petgraph::visit::*;
use std::convert::{AsRef, AsMut};
use std::cmp::Ordering;
use num_traits::bounds::Bounded;

/* Represents a binary image by a set of foreground pixel coordinates. Useful when
objects are expected to be speckle-like. For dense objects, consider using RunLengthEncoding
instead, since PointEncoding is an encoding that is potentially memory-heavy. But if objects are
speckle-like, then the RunLength encoding would mostly carry a start and length=1 that wouldn't
be very informative, so PointEncoding would be a best option. */
#[derive(Default)]
pub struct PointEncoding {
    pub pts :  Vec<(usize, usize)>,
    pub rows : Vec<Range<usize>>
}

impl Encoding for PointEncoding {

    fn points<'a>(&'a self) -> PointIter<'a> {
        PointIter(Box::new(self.pts.clone().into_iter()))
    }

    fn encode_distinct(img : &dyn AsRef<Window<u8>>) -> BTreeMap<u8, Self> {
        unimplemented!()
    }

    fn encode_from(&mut self, img : &dyn AsRef<Window<u8>>) {
        unimplemented!()
    }

    // Write the labels back to the image.
    fn decode_distinct_to(labels : &BTreeMap<u8, Self>, img : &dyn AsMut<WindowMut<u8>>) {
        unimplemented!()
    }

}
// Assume pts is sorted by rows and by cols within rows.
fn rows_for_sorted_points(pts : &[(usize, usize)], rows : &mut Vec<Range<usize>>) {
    rows.clear();
    let mut n = 0;
    for (_, mut pts) in &pts.iter().group_by(|pt| pt.0 ) {
        let row_len = pts.count();
        rows.push(Range { start : n, end : n+row_len } );
        n += row_len;
    }
}

impl PointEncoding {

    // Builds a PointEncoding from a set of points (assume neither order or uniqueness
    // of the underlying array).
    pub fn from_points(mut pts : Vec<(usize, usize)>) -> Self {
        pts.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)) );
        pts.dedup();
        let mut rows = Vec::new();
        rows_for_sorted_points(&pts[..], &mut rows);
        Self { pts, rows }
    }

    /// Returns coordinates of dense set of points that are nonzero.
    pub fn encode(win : &Window<'_, u8>) -> Self {
        let mut pts = Vec::new();
        let mut rows : Vec<Range<usize>> = Vec::new();
        pts.clear();
        for i in 0..win.height() {
            let mut found_at_row = 0;
            for j in 0..win.width() {
                if win[(i, j)] != 0 {
                    found_at_row += 1;
                    pts.push((i, j));
                }
            }
            if found_at_row > 0 {
                let last = rows.last().map(|l| l.end ).unwrap_or(0);
                rows.push(Range { start : last, end : last + found_at_row });
            }
        }
        Self { pts, rows }
    }

    pub fn connectivity_graph(&self, neigh : Neighborhood) -> PointGraph {
        let mut uf = UnionFind::new(self.pts.len());
        let mut graph = UnGraph::new_undirected();
        let mut found_pts = Vec::new();
        for (row_ix, row) in self.rows.iter().enumerate() {
            for pt_ix_at_row in 0..row.len() {
                let pt_ix = row.start + pt_ix_at_row;
                let graph_ix = graph.add_node(self.pts[pt_ix]);

                let this_pt_col = self.pts[pt_ix].1;
                let this_pt_row = self.pts[pt_ix].0;

                // Link to element(s) to left
                let mut left_offset = 1;
                if let Some(left_ix) = pt_ix_at_row.checked_sub(1) {
                    if this_pt_col - self.pts[row.clone()][left_ix].1 == 1 {
                        let left_ix = NodeIndex::new(graph_ix.index()-1);
                        graph.add_edge(graph_ix, left_ix, ());
                        uf.union(graph_ix, left_ix);
                    }
                }

                // Link to any element to top as long as it is below a given distance. Worst
                // case scenario, we have points in all rows above the point (but leave
                // early if not).
                let mut top_row_offset = 1;
                if let Some(top_row_ix) = row_ix.checked_sub(1) {

                    let top_row = &self.rows[top_row_ix];

                    // (Leave early if not)
                    if this_pt_row - self.pts[top_row.clone()][0].0 == 1 {

                        // TODO move this match to return a function pointer.
                        let res_pt = match neigh {
                            Neighborhood::Immediate => {
                                self.pts[top_row.clone()].binary_search_by(|pt| {
                                    pt.1.cmp(&this_pt_col)
                                })
                            },
                            Neighborhood::Extended => {
                                self.pts[top_row.clone()].binary_search_by(|pt| {
                                    if this_pt_col.abs_diff(pt.1) <= 1 {
                                        Ordering::Equal
                                    } else {
                                        pt.1.cmp(&this_pt_col)
                                    }
                                })
                            }
                        };
                        if let Ok(found) = res_pt {
                            match neigh {
                                Neighborhood::Immediate => {
                                    let top_ix = NodeIndex::new(top_row.start + found);
                                    graph.add_edge(graph_ix, top_ix, ());
                                    uf.union(graph_ix, top_ix);
                                },
                                Neighborhood::Extended => {
                                    found_pts.clear();
                                    found_pts.push(found);

                                    // Search for neighboring element when the binary search returns middle OR extreme element.
                                    if found > 0 && self.pts[top_row.start + found - 1].1.abs_diff(this_pt_col) <= 1 {
                                        found_pts.push(found - 1);
                                    }
                                    if found < top_row.len()-1 && self.pts[top_row.start + found + 1].1.abs_diff(this_pt_col) <= 1 {
                                        found_pts.push(found + 1);
                                    }

                                    // Search for two next elements when the binary search returns an extreme element
                                    if found > 1 && self.pts[top_row.start + found - 2].1.abs_diff(this_pt_col) <= 1 {
                                        found_pts.push(found - 2);
                                    }
                                    if top_row.len() >= 2 && found < top_row.len()-2 && self.pts[top_row.start + found + 2].1.abs_diff(this_pt_col) <= 1 {
                                        found_pts.push(found + 2);
                                    }

                                    for pt_ix in &found_pts {
                                        let top_ix = NodeIndex::new(top_row.start + pt_ix);
                                        graph.add_edge(graph_ix, top_ix, ());
                                        uf.union(graph_ix, top_ix);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        PointGraph { uf, graph }
    }

    pub fn proximity_graph(&self, max_dist : usize) -> PointGraph {
        let mut uf = UnionFind::new(self.pts.len());
        let mut graph = UnGraph::new_undirected();
        for (row_ix, row) in self.rows.iter().enumerate() {
            for pt_ix_at_row in 0..row.len() {
                let pt_ix = row.start + pt_ix_at_row;
                let graph_ix = graph.add_node(self.pts[pt_ix]);

                let this_pt_col = self.pts[pt_ix].1;
                let this_pt_row = self.pts[pt_ix].0;

                // Link to element(s) to left
                let mut left_offset = 1;
                while let Some(left_ix) = pt_ix_at_row.checked_sub(left_offset) {
                    if this_pt_col - self.pts[row.clone()][left_ix].1 <= max_dist {
                        // let left_ix = NodeIndex::new(graph_ix.index()-1);
                        let left_ix = NodeIndex::new(graph_ix.index()-left_offset);
                        graph.add_edge(graph_ix, left_ix, ());
                        uf.union(graph_ix, left_ix);
                    } else {
                        break;
                    }
                    left_offset += 1;
                }

                // Link to any element to top as long as it is below a given distance. Worst
                // case scenario, we have points in all rows above the point (but leave
                // early if not).
                let mut top_row_offset = 1;
                while top_row_offset <= max_dist && row_ix.checked_sub(top_row_offset).is_some() {

                    let top_row = &self.rows[row_ix - top_row_offset];

                    // (Leave early if not)
                    if this_pt_row - self.pts[top_row.clone()][0].0 > max_dist {
                        break;
                    }

                    let res_pt = self.pts[top_row.clone()].binary_search_by(|pt| {
                        if this_pt_col.abs_diff(pt.1) <= max_dist {
                            Ordering::Equal
                        } else {
                            pt.1.cmp(&this_pt_col)
                        }
                    });

                    if let Ok(mut start) = res_pt {
                        // Iterate over all points near this one.
                        let mut end = start+1;
                        loop {
                            if start > 0 && self.pts[top_row.start + start - 1].1.abs_diff(this_pt_col) <= max_dist {
                                start -= 1;
                            } else {
                                break;
                            }
                        }
                        loop {
                            if end < top_row.len()-1 && self.pts[top_row.start + end + 1].1.abs_diff(this_pt_col) <= max_dist {
                                end += 1;
                            } else {
                                break;
                            }
                        }
                        for pt_ix in start..end {
                            let top_ix = NodeIndex::new(top_row.start + pt_ix);
                            graph.add_edge(graph_ix, top_ix, ());
                            uf.union(graph_ix, top_ix);
                        }
                    }
                    top_row_offset += 1;
                }
            }
        }
        PointGraph { uf, graph }
    }

}

// Establish edges between points that are close together.
pub struct PointGraph {

    pub graph : UnGraph<(usize, usize), ()>,

    pub uf : UnionFind<NodeIndex<u32>>

}

impl PointGraph {

    pub fn enclosing_rects(&self) -> BTreeMap<NodeIndex<u32>, (usize, usize, usize, usize)> {
        let mut rects = BTreeMap::new();
        for (ix, pts) in self.groups() {
            let min_y = pts.iter().min_by(|a, b| a.0.cmp(&b.0) ).copied().unwrap().0;
            let min_x = pts.iter().min_by(|a, b| a.1.cmp(&b.1) ).copied().unwrap().1;
            let max_y = pts.iter().max_by(|a, b| a.0.cmp(&b.0) ).copied().unwrap().0;
            let max_x = pts.iter().max_by(|a, b| a.1.cmp(&b.1) ).copied().unwrap().1;
            rects.insert(ix, (min_y, min_x, max_y - min_y, max_x - min_x));
        }
        rects
    }

    pub fn groups(&self) -> BTreeMap<NodeIndex<u32>, Vec<(usize, usize)>> {
        let mut grs = BTreeMap::new();
        for ix in self.graph.node_indices() {
            let parent = self.uf.find(ix);
            grs.entry(parent).or_insert(Vec::new()).push(self.graph[ix]);
        }
        grs
    }

}

type BumpVecOpen<'a> = bumpalo::collections::Vec<'a, OpenChain<'a>>;

type BumpVecDir<'a> = bumpalo::collections::Vec<'a, Direction>;

#[derive(Clone)]
struct OpenChain<'a> {

    start : (usize, usize),

    end : (usize, usize),

    directions : BumpVecDir<'a>

}

fn is_close(a : (usize, usize), b : (usize, usize)) -> bool {
    a.1.abs_diff(b.1) <= 1 && a.0.abs_diff(b.0) <= 1
}

impl ChainEncoding {

    fn encode(bin_img : &Window<u8>) -> ChainEncoding {

        use bumpalo::Bump;

        let bump = Bump::with_capacity(std::mem::size_of::<Direction>() * bin_img.height() * bin_img.width());
        let mut starts = Vec::new();
        let mut ends = Vec::new();
        let mut ranges : Vec<Range<usize>> = Vec::new();
        let mut open_chains = BumpVecOpen::new_in(&bump);
        let mut row_chains = BumpVecOpen::new_in(&bump);
        let mut directions = Vec::new();

        // Holds indices of open_chains that matched the current row.
        let mut open_matched = bumpalo::collections::Vec::new_in(&bump);

        for (row_ix, row) in bin_img.rows().enumerate() {

            let mut past = row[0];
            if row[0] != 0 {
                // Innaugurante a chain when first pixel is nonzero.
                row_chains.push(OpenChain {
                    start : (row_ix, 0),
                    end : (row_ix, 0),
                    directions : BumpVecDir::new_in(&bump)
                });
            }

            for col_ix in 1..row.len() {
                match (past != 0, row[col_ix] != 0) {
                    (true, true) => {
                        // If left and this matched, just grow the last chain.
                        let last = row_chains.last_mut().unwrap();
                        last.end.1 += 1;
                        last.directions.push(Direction::East);
                    },
                    (false, true) => {
                        // If transition to new matched pixel, innaugurate a new chain.
                        row_chains.push(OpenChain { start : (row_ix, col_ix), end : (row_ix, col_ix), directions : BumpVecDir::new_in(&bump) });
                    },
                    (true, false) => {
                        // Positive->negative transitions can be ignored, since
                        // chains need to be kept open.
                    },
                    (false, false) => {
                        // Negative->negative transitions also can be ignored.
                    }
                }
                past = row[col_ix];
            }

            // row_chains now contains a set of horizontal chains accumulated over
            // the current row; open_chains all the previous chains. Combine the two.
            open_matched.clear();
            let n_before = open_chains.len();
            for mut row_chain in row_chains.drain(..) {
                let mut matched = None;
                for i in 0..n_before {

                    // Guarantees matched elements are not re-used.
                    if open_matched.binary_search(&i).is_ok() {
                        continue;
                    }

                    if is_close(open_chains[i].end, row_chain.start) {

                        // Extend by appending element (and going left-to-right if required).
                        let end = open_chains[i].end;
                        open_chains[i].directions.push(Direction::compare_below(end, row_chain.start));
                        open_chains[i].directions.extend(row_chain.directions.drain(..));
                        open_chains[i].end = row_chain.end;

                        matched = Some(i);
                        break;

                    // This second condition will only match for row chains that have at least
                    // two elements, since single-element ones would have already matched at the start
                    // in the first condition.
                    } else if is_close(open_chains[i].end, row_chain.end) {

                        // Extend now by going right-to-left through this row chain (Set all row_chain directions to west).
                        row_chain.directions.iter_mut().for_each(|d| d.reverse() );
                        let end = open_chains[i].end;
                        open_chains[i].directions.push(Direction::compare_below(end, row_chain.end));
                        open_chains[i].directions.extend(row_chain.directions.drain(..));
                        open_chains[i].end = row_chain.start;

                        matched = Some(i);
                        break;

                    // The two conditions below will trigger only for chains that have
                    // two or more elements (since connecting, single-element chains would
                    // have already matched the two previous conditions). The conditions will
                    // also match only when the previous open_chain[i] started and ended at the same
                    // row above (since any chains started at a row before will return false for is_close).
                    // The conditions below produce "snake-like" patterns for dense objects.
                    } else if is_close(open_chains[i].start, row_chain.start) {

                        // Revert chain at previous row (as if it started at the end) so it
                        // can be a part of the next chain.

                        open_chains[i].directions.iter_mut().for_each(|d| d.reverse() );

                        // The starts will only be equal to next start when the chain above is present on a single row.

                        // Pushing to the end of row_chains and reversing has the same effect as pushing
                        // to the start of open_chains[i], but there is no risk of re-allocating the vector.
                        let start = open_chains[i].start;
                        open_chains[i].directions.push(Direction::compare_below(start, row_chain.start));

                        // "snake turns right"
                        open_chains[i].directions.extend(row_chain.directions.drain(..));
                        open_chains[i].start = open_chains[i].end;
                        open_chains[i].end = row_chain.end;

                        matched = Some(i);
                        break;

                    } else if is_close(open_chains[i].start, row_chain.end) {

                        // The start will only be equal to next end when the chain above is present on a single row.
                        open_chains[i].directions.iter_mut().for_each(|d| d.reverse() );
                        row_chain.directions.iter_mut().for_each(|d| d.reverse() );
                        let start = open_chains[i].start;
                        open_chains[i].directions.push(Direction::compare_below(start, row_chain.end));

                        // "snake turns left"
                        open_chains[i].directions.extend(row_chain.directions.drain(..));
                        open_chains[i].start = open_chains[i].end;
                        open_chains[i].end = row_chain.start;

                        matched = Some(i);
                        break;
                    }
                }
                if let Some(matched_ix) = matched {
                    open_matched.push(matched_ix);
                } else {
                    // To be considered for next row iteration. This item should
                    // not be considered further at this iteration.
                    open_chains.push(row_chain);
                }
            }

            // Close unmatched chains, by iterating up to the size of open_chains before row
            // was evaluated.
            for i in (0..n_before).rev() {
                if open_matched.binary_search(&i).is_err() {
                    let closed = open_chains.swap_remove(i);
                    starts.push(closed.start);
                    ends.push(closed.end);

                    // Elements with a single element will yield a open range (start = end)
                    let start = ranges.last().map(|r| r.end ).unwrap_or(0);
                    ranges.push(Range { start, end : start + closed.directions.len() });
                    directions.extend(closed.directions);
                }
            }
        }

        ChainEncoding {
            starts,
            directions,
            ranges,
            ends
        }
    }

}

/// Wraps any iterator over (usize, usize)
pub struct PointIter<'a>(Box<dyn Iterator<Item=(usize, usize)> + 'a>);

impl<'a> Iterator for PointIter<'a> {

    type Item=(usize, usize);
    
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
    
}

pub trait Encoding
where
    Self : Default
{

    fn points<'a>(&'a self) -> PointIter<'a>;

    // If the argument is a labeled image, where distinct matched labels are values greater than or
    // equal to 1, and unmatched pixels are represented by zero, then encode_distinct
    // should return a set of distinct encodings that map to each label, as if each labeled
    // pixel were a binary image encoded as "this label against all others".
    fn encode_distinct(img : &dyn AsRef<Window<u8>>) -> BTreeMap<u8, Self>;

    fn encode_from(&mut self, img : &dyn AsRef<Window<u8>>);

    fn encode(img : &dyn AsRef<Window<u8>>) -> Self {
        let mut encoding : Self = Default::default();
        encoding.encode_from(img);
        encoding
    }

    // Tries to decode points into the image. If size is insufficient, return false
    // and stops.
    fn decode_to(&self, img : &mut dyn AsMut<WindowMut<u8>>) -> bool {
        let mut img = img.as_mut();
        img.fill(0);
        for pt in self.points().0 {
            if pt.0 < img.height() && pt.1 < img.width() {
                img[pt] = 255;
            } else {
                return false;
            }
        }
        true
    }

    fn decode(&self, shape : (usize, usize)) -> Option<Image<u8>> {
        // Assumes image must always be filled with zeros at the start of the call of decode_to
        let mut img = unsafe { Image::new_empty(shape.0, shape.1) };
        if self.decode_to(&mut img) {
            Some(img)
        } else {
            None
        }
    }

    // Write the labels back to the image.
    fn decode_distinct_to(labels : &BTreeMap<u8, Self>, img : &dyn AsMut<WindowMut<u8>>);

}

/* Chain encodings are mostly useful to encode binary images resulting from edge operators.
dense objects will be represented by snake-like chain patterns (pattern repeats forward and
backward across even and odd rows, so for dense images it is best to stick with RunLengthEncoding).
But for edge images, representing the (start, end) pair and a set of directions is much more
efficient, since each pixel is represented by a single byte tagging the direction of change,
instead of a full usize pair. The points can be recovered from any resolution desired by
calling trajectory(.) and sparse_trajectory(.). */
pub struct ChainEncoding {
    starts : Vec<(usize, usize)>,
    ends : Vec<(usize, usize)>,
    directions : Vec<Direction>,
    ranges : Vec<Range<usize>>
}

pub struct ChainIter<'a> {
    enc : &'a ChainEncoding,
    curr : usize
}

impl ChainEncoding {

    pub fn iter(&self) -> ChainIter {
        ChainIter {
            enc : self,
            curr : 0
        }
    }

}

/*impl<'a> Iterator for ChainIter<'a> {

    type Item = Chain<'a>;

    fn next(&mut self) -> Option<Chain> {
        let start = self.starts.get(self.curr)?;
        let traj = &self.directions[self.ranges.get(self.curr)?];
        self.curr += 1;
        Some(Chain { start, traj })
    }

}*/

/** Represents an edge using an 8-direction chain code.
Useful to represent binary images resulting from edge and contour operators. **/
#[derive(Debug, Clone, Copy)]
pub struct Chain<'a> {
    pub start : (usize, usize),
    pub end : (usize, usize),
    pub traj : &'a [Direction]
}

fn update_point(pt : &mut (usize, usize), d : Direction) {
    let off = d.offset();
    pt.0 = (pt.0 as i32 + off.0) as usize;
    pt.1 = (pt.1 as i32 + off.1) as usize;
}

impl<'a> Chain<'a> {

    /* Returns the full set of points this chain represents. The method is lightweight and
    does not allocate. */
    pub fn trajectory(&'a self) -> impl Iterator<Item=(usize, usize)> + 'a {
        std::iter::once(self.start)
            .chain(self.traj.iter().scan(self.start, move |pt, d| {
                update_point(pt, *d);
                Some(*pt)
            }))
    }

    /* Returns every nth point this chain represents. The method is lightweight and
    does not allocate. */
    pub fn sparse_trajectory(&'a self, step : usize) -> impl Iterator<Item=(usize, usize)> + 'a {
        std::iter::once(self.start)
            .chain((0..self.traj.len()).step_by(step).skip(1).scan(self.start, move |pt, i| {

                // Accumulate previous points (excluding the previous step)
                for j in (i-step-1)..(i+1) {
                    update_point(pt, self.traj[j]);
                }

                Some(*pt)
            }))
    }

}

#[repr(u8)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Neighborhood {

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

impl Direction {

    pub fn compare_below(above : (usize, usize), below : (usize, usize)) -> Direction {
        assert!(above.0 + 1 == below.0 && above.1.abs_diff(below.1) <= 1);
        if below.1 == above.1 {
            Direction::South
        } else if below.1 + 1 == above.1 {
            Direction::SouthWest
        } else if above.1 + 1 == below.1 {
            Direction::SouthEast
        } else {
            panic!("Invalid direction")
        }
    }

    pub fn reverse(&mut self) {
        match *self {
            Direction::North => { *self = Direction::South },
            Direction::NorthEast => { *self = Direction::SouthWest },
            Direction::East => { *self = Direction::West },
            Direction::SouthEast => { *self = Direction::NorthWest },
            Direction::South => { *self = Direction::North },
            Direction::SouthWest => { *self = Direction::NorthEast },
            Direction::West => { *self = Direction::East },
            Direction::NorthWest => { *self = Direction::SouthEast },
        }
    }

    pub fn offset(&self) -> (i32, i32) {
        match self {
            Direction::North => (-1, 0),
            Direction::NorthEast => (-1, 1),
            Direction::East => (0, 1),
            Direction::SouthEast => (1, 1),
            Direction::South => (1, 0),
            Direction::SouthWest => (1, -1),
            Direction::West => (0, -1),
            Direction::NorthWest => (-1, -1)
        }
    }

}

/// Represents a set of homogeneous pixels. Useful to represent binary images resulting
/// in dense regions (patches), such as the ones generated by thresholding operations.
/// A lightweight (copy) struct representing a horizontal sequence of homogeneous
/// pixels by the coordinate of the first pixel and the length of the sequence.
/// The RunLength is the basis for a sparse representation of a binary image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RunLength {
    pub start : (usize, usize),
    pub length : usize
}

impl RunLength {

    pub fn points<'a>(&'a self) -> impl Iterator<Item=(usize, usize)> + 'a {
        ((self.start.1)..(self.start.1 + self.length)).map(move |c| (self.start.0, c) )
    }

    pub fn middle_point(&self) -> (usize, usize) {
        (self.start.0, self.start.1 + self.length / 2 )
    }
    
    // TODO rename to inner_points
    pub fn middle_points<'a>(&'a self) -> impl Iterator<Item=(usize, usize)> + 'a {
        ((self.start.1+1)..(self.end().1.saturating_sub(1))).map(move |c| (self.start.0, c) )
    }

    pub fn end_col(&self) -> usize {
        self.start.1 + self.length-1
    }

    pub fn end(&self) -> (usize, usize) {
        (self.start.0, self.end_col())
    }

    pub fn as_rect(&self) -> (usize, usize, usize, usize) {
        (self.start.0, self.start.1, 1, self.length)
    }

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

pub fn strict_nonzero_bound_window<'a>(w : &'a Window<'a, u8>) -> Window<'a, u8> {
    let vert_win = vertical_nonzero_bound_window(w);
    let (min_r, mut min_c) = vert_win.offset();
    let shape = vert_win.shape();
    let (max_r, mut max_c) = (min_r + shape.0, min_c + shape.1);
    let sub_height = max_r - min_r;
    let left_win = w.sub_window((min_r, 0), (sub_height, min_c)).unwrap();
    let right_win = w.sub_window((min_r, max_c), (sub_height, w.width() - max_c)).unwrap();
    let mut r = min_r;
    while r < max_r {
        for c in 0..min_c {
            if w[(r, c)] == 255 && c < min_c {
                min_c = c;
                break;
            }
        }
        for c in (max_c..w.width()).rev() {
            if w[(r, c)] == 255 && c > max_c {
                max_c = c;
                break;
            }
        }
        r += 1;
    }
    let shape = (max_r - min_r, max_c - min_c);
    w.sub_window((min_r, min_c), shape).unwrap()
}

pub fn vertical_nonzero_bound_at_integral<'a>(w : &'a Window<'a, i32>) -> Window<'a, i32> {
    let mut tl = (0, 0);
    let mut br = (w.height()-1, w.width()-1);
    for r in 0..w.height() {
        for c in 0..w.width() {
            if w[(r, c)] != 0 {
                tl = (r, c);
                break;
            }
        }
    }
    let last = w[(w.height()-1, w.width()-1)];
    for r in (0..w.height()).rev() {
        for c in (0..w.width()).rev() {
            if w[(r, c)] != last {
                br = (r, c);
                break;
            }
        }
    }
    let shape = (br.0 - tl.0, br.1 - tl.1);
    w.sub_window(tl, shape).unwrap()
}

// Returns the subwindow of w that contains the smallest region containing
// the totality of the nonzero pixels of w (assumed binary). This can speed up
// encoding for sparse binary images, since the algorithm does not iterate over
// all pixels. Rather, it uses the row of the minimum column and the column of
// the minimum row to ignore full image regions at further iterations. If window
// contains no nozero pixels, return the full window. This verifies only the vertical dimension,
// if the strict minimal bound is desired, use the strict variant.
pub fn vertical_nonzero_bound_window<'a>(w : &'a Window<'a, u8>) -> Window<'a, u8> {

    assert!(w.height() > 1);

    let mut min_r = 0;
    let mut max_r = w.height() - 1;
    let mut min_c = 0;
    let mut max_c = w.width() - 1;

    // Iterate from bottom to top
    let mut r = 1;
    let mut c = 0;
    while r < w.height() - 1 {

        // Iterate from left-to-right
        while c < w.width() {
            if w[(r, c)] == 255 {
                min_c = c;
                max_c = c;
                min_r = r;
                break;
            }
            c += 1;
        }

        // If first nonzero pixel found here, iterate from right-to-left
        if min_r == r {
            c = w.width();
            while c > min_c {
                if w[(r, c)] == 255 {
                    max_c = c;
                    break;
                }
                c -= 1;
            }
            break;
        }

        r += 1;
    }

    // Iterate from top to bottom
    r = w.height() - 2;
    c = 0;
    while r > min_r {
        // Iterate from left-to-right
        while c < w.width() {
            if w[(r, c)] == 255 {
                if c < min_c {
                    min_c = c;
                }
                if c > max_c {
                    max_c = c;
                }
                max_r = r;
                break;
            }
            c += 1;
        }

        // If first nonzero pixel found here, iterate from right-to-left
        if max_r == r {
            c = w.width();
            while c > max_c {
                if w[(r, c)] == 255 {
                    if c > max_c {
                        max_c = c;
                    }
                    break;
                }
                c -= 1;
            }
            break;
        }

        r -= 1;
    }

    let shape = (max_r - min_r, max_c - min_c);
    w.sub_window((min_r, min_c), shape).unwrap()
}

// RunLength encodings are sorted by starting points within rows
// and also by rows, so to determine if a pixel is positive can be
// done by bisection.
#[derive(Debug, Clone, Default)]
pub struct RunLengthEncoding {
    pub rles : Vec<RunLength>,
    pub rows : Vec<Range<usize>>
}

pub struct RunLengthIter<'a> {
    enc : &'a RunLengthEncoding,
    curr : usize
}

impl Encoding for RunLengthEncoding {

    fn points<'a>(&'a self) -> PointIter<'a> {
        PointIter(Box::new(self.rles.iter().map(move |rle| rle.points() ).flatten()))
    }

    fn encode_distinct(img : &dyn AsRef<Window<u8>>) -> BTreeMap<u8, Self> {
        unimplemented!()
    }

    fn encode_from(&mut self, img : &dyn AsRef<Window<u8>>) {
        unimplemented!()
    }

    // Write the labels back to the image.
    fn decode_distinct_to(labels : &BTreeMap<u8, Self>, img : &dyn AsMut<WindowMut<u8>>) {
        unimplemented!()
    }

}

/* Given a (sorted) RLE vector, extract its row vector. */
fn rows_vector(rles : &[RunLength]) -> Vec<Range<usize>> {
    let mut rows = Vec::new();
    let mut start = 0;
    for (r, mut s) in &rles.iter().group_by(|v| v.start.0 ) {
        let row_len = s.count();
        rows.push(Range { start, end : start + row_len });
        start += row_len;
    }
    rows
}

// Recursively search for regions in a row of an accumulated binary image
// that might contain RunLengths, by just comparing the differences of the
// boundary pixels over a row segment.
fn calculate_valid_col_ranges(ranges : &mut Vec<Range<usize>>, w : &Window<i32>, r : usize, a : usize, b : usize, min_sz : usize) {

    // min_sz cannot be close to 2, because two pixels can be equal and be a white->dark transition.
    assert!(min_sz >= 4);

    assert!(b > a);

    // assert!((b - a) % 2 == 0, "a = {}; b = {}", a, b);

    // If left and right pixels are equal in the accumulated image,
    // this is a dark region at the binary image.
    if w[(r, b)] > w[(r, a)] {
        if b - a > min_sz {

            // Since a transition between foreground pixel to background pixel
            // always happen for i == j, it is critical that the valid range
            // contains the first pixel repetition.

            let half = ((a as f32 + b as f32) / 2.0).ceil() as usize;
            assert!((half - a) + (b - half) == b - a);

            calculate_valid_col_ranges(ranges, w, r, a, half, min_sz);
            calculate_valid_col_ranges(ranges, w, r, half+1, b, min_sz);
        } else {
            ranges.push(Range { start : a, end : b });
        }
    }
    // If w[a] and w[b] are equal, ignore the region for further processing
}

fn calculate_valid_row_ranges(ranges : &mut Vec<Range<usize>>, w : &Window<i32>, a : usize, b : usize, min_sz : usize) {
    assert!(b > a);
    // If top left and bottom right pixels in the row rergion are equal in the accumulated image,
    // this is a dark region at the binary image.
    if w[(a, 0)] < w[(b, w.width()-1)] {
        if b - a > min_sz {
            let half = ((a as f32 + b as f32) / 2.0).ceil() as usize;
            assert!((half - a) + (b - half) == b - a);
            calculate_valid_row_ranges(ranges, w, a, half, min_sz);
            calculate_valid_row_ranges(ranges, w, half+1, b, min_sz);
        } else {
            ranges.push(Range { start : a, end : b });
        }
    }
    // If w[a] and w[b] are equal, ignore the region for further processing
}

fn merge_col_ranges(ranges : &mut Vec<Range<usize>>) {
    if ranges.len() < 2 {
        return;
    }

    // Holds a "lead" range index (one that might be extended with zero, one or more "part" ranges
    let mut lead = 0;

    // Holds a "part" range index, that might either merge to the lead range before it if they are
    // contiguous, or become its own lead range if it is not contiguous to the past lead range.
    let mut part = 1;

    while part < ranges.len() {
        if ranges[lead].end + 1 == ranges[part].start {

            // Extend lead range
            ranges[lead].end = ranges[part].end;

            // Invalidate this part end
            ranges[part].end = ranges[part].start;

        } else {
            lead = part;
        }
        part += 1;
    }

    // Not strictly necessary (since start==end range iterators yield None, so
    // there is no harm in keeping them in the valid range iterator), but added
    // here for clarity.
    ranges.retain(|r| r.end > r.start );
}

// Perform a search for one or two RunLenghts recursively starting from the extremities. From the
// structure of the IntegralImage, we can know if we have one or more RunLengths in a pair of pixel
// boundaries by verifying their difference in pixel value against their horizontal distance (if they
// are equal, we have a single RunLenght and can stop iteration. If not, we iterate until we find the
// end of the first left runlength and the start of the last right runlenght, then call this recursively
// starting after the end of the left runlenght and before the start of the right runlength). This represents
// huge computational savings for sparse binary images, since whole rows or large segments of rows can be
// ignored during the search.
fn find_at_row_segment(rles : &mut Vec<RunLength>, w : &Window<i32>, r : usize, mut a : usize, mut b : usize) {
    assert!(b >= a, "a = {}; b = {}", a, b);
    let mut left_px = w[(r, a)];
    let mut right_px = w[(r, b)];
    let d = right_px - left_px;
    // println!("d = {d}");
    if d == 0 {
        return;
    } else {

        // TODO differentiate an "even" start from an "odd" start, since
        // we do not know if we will start at the beginning or end of the RLE.

        // Skip flat region to right (dark binary segment after any RLEs)
        /*while w[(r, b-1)] == right_px {
            b -= 1;
        }
        b -= 1;*/

        // Note the first white->dark transition repeats the last
        // white pixel value, which is why we evaluate one-past b.
        while w[(r, b)] - w[(r, b-1)] == 0 {
            b -= 1;
        }

        // Skip flat region to left (dark binary segment before any RLEs)
        /*while w[(r, a+1)] == left_px {
            a += 1;
        }*/
        while w[(r, a+1)] - w[(r, a)] == 0 {
            a += 1;
        }
        assert!(b >= a + 1);

        right_px = w[(r, b)];
        left_px = w[(r, a)];
        assert!(b >= a, "{}, {}", a, b);
        println!("Diff a = {}, b = {}", a, b);
        if (right_px - left_px) / 255 == (b - a) as i32 {
            if r == 0 {
                println!("Unique RLE found: {:?} between {:?}", RunLength { start : (r, a), length : b - a + 1 }, (a, b));
            }
            rles.push(RunLength { start : (r, a), length : b - a });
        } else {
            let mut left_length = 1;
            let mut right_length = 1;
            while w[(r, a+left_length+1)] - w[(r, a+left_length)] == 255 {
                left_length += 1;
            }

            // This iterator goes one-past the right length because the first dark
            // pixel repeats the last while pixel accumulated value.
            while w[(r, b-right_length-1)] - w[(r, b-right_length-2)] == 255 {
                right_length += 1;
            }

            // We add an offset of 1 to all RunLenghts because the accumulated image
            // is shifted by one relative to the original image (since the first source
            // pixel is equal to the first accumulated pixel).
            let left = RunLength { start : (r, a + 1), length : left_length };
            let right = RunLength { start : (r, b - right_length + 1), length : right_length };
            if r == 0 {
                println!("RLE pair found between {}-{} : {:?} {:?}",
                    a,
                    b,
                    left,
                    right
                );
            }
            rles.push(left);
            rles.push(right);

            // There should be at least one dark pixel between the two recently-inserted RLEs
            // (since it would have been considered a sigle RLE in the if clause otherwise).
            let new_a = a+left_length+1;
            let new_b = b-right_length-1;
            assert!(new_a < new_b, "a : {}, b : {}", new_a, new_b);
            find_at_row_segment(rles, w, r, new_a, new_b);
        }
    }
}

fn find_row_rles(
    valid_col_ranges : &mut Vec<Range<usize>>,
    rles : &mut Vec<RunLength>,
    rows : &mut Vec<Range<usize>>,
    w : &Window<i32>,
    r : usize,
    min_col_range : Option<usize>
) {
    calculate_valid_col_ranges(valid_col_ranges, w, r, 0, w.width() - 1, min_col_range.unwrap_or(MIN_VALID_INTEGRAL_COL_RANGE));
    valid_col_ranges.sort_by(|a, b| a.start.cmp(&b.start) );
    merge_col_ranges(valid_col_ranges);
    for range in valid_col_ranges.drain(..) {
        let n_before = rles.len();
        find_at_row_segment(rles, w, r, range.start, range.end);
        let n_after = rles.len();
        rles[n_before..n_after].sort_by(|a, b| a.start.1.cmp(&b.start.1) );
        rows.push(Range { start : n_before, end : n_after });
    }
}

// cargo test --all-features -- accumulated_rle --nocapture
#[test]
fn accumulated_rle() {
    let mut img = Image::new_checkerboard(100, 10);
    img[(0, 2)] = 255;
    img[(0, 3)] = 255;
    img[(0, 4)] = 255;

    img[(0, 7)] = 255;
    img[(0, 8)] = 255;

    img.show();
    println!("{:?}", img.row(0));

    let acc = crate::integral::Accumulated::calculate(img.as_ref());
    let iw : &Window<i32> = acc.as_ref();
    println!("{:?}", iw.row(0));
    // for i in 0..5 {
    //    println!("row {}: {:?}", i, iw.row(i).unwrap());
    // }
    // println!("Integral = {:?}", iw);
    // println!("{:?}", iw.row(1));
    let rle = RunLengthEncoding::calculate_from_accumulated(&acc, None, None);
    // println!("{:?}", rle);
}

pub const MIN_VALID_INTEGRAL_COL_RANGE : usize = 8;

pub const MIN_VALID_INTEGRAL_ROW_RANGE : usize = 8;

// Assume pts is sorted by rows and by cols within rows.
fn rows_for_sorted_rles(rles : &[RunLength], rows : &mut Vec<Range<usize>>) {
    rows.clear();
    let mut n = 0;
    for (_, mut rles) in &rles.iter().group_by(|r| r.start.0 ) {
        let row_len = rles.count();
        rows.push(Range { start : n, end : n+row_len } );
        n += row_len;
    }
}

impl RunLengthEncoding {

    /* Transpose this RunLenght, so that its decoding is equivalent to the decoding of the transposed image */
    pub fn transpose(mut self) -> Self {
        if self.rows.len() == 0 {
            return Default::default();
        }
        self.rles.iter_mut().for_each(|r| r.start = (r.start.1, r.start.0) );
        self.rles.sort_by(|a, b| 
            a.start.0.cmp(&b.start.0).then(a.start.1.cmp(&b.start.1)) 
        );
        rows_for_sorted_rles(&self.rles[..], &mut self.rows);
        self
    }
    
    pub fn preserve_larger(&self, min_len : usize) -> RunLengthEncoding {
        let mut new_rles = self.rles.clone();
        new_rles.retain(|rle| rle.length >= min_len );
        let mut new_rows : Vec<Range<usize>> = Vec::new();
        rows_for_sorted_rles(&new_rles[..], &mut new_rows);
        let new_rle = Self { rles : new_rles, rows : new_rows };
        verify_rle_state(&self);
        new_rle
    }

    pub fn iter(&self) -> RunLengthIter {
        RunLengthIter {
            enc : self,
            curr : 0
        }
    }

    // pub fn split(&self) -> crate::graph::SplitGraph {
    //    crate::graph::group_weights(&self.graph, &self.uf)
    // }

    /*/// Returns the complement of this RunLength.
    pub fn gaps(&self) -> RunLengthEncoding {
        let mut neg_rles = Vec::new();
        let mut neg_rows = Vec::new();
        let mut ix = 0;
        for r in 0..self.rows[self.rows.len()-1] {
            while r < rles[curr_ix].0 {
                neg_rles.push(full_row);

            }
            // Push RLEs at this row
        }
    }*/

    // Builds a RunLength encoding from offsets (useful when a filter
    // was applied to img.windows(.) and only a few windows were maintained.
    // offsets is assumed to be a non-overlapping set of matching windows of the
    // given shape.
    pub fn from_offsets(offsets : &[(usize, usize)], shape : (usize, usize)) -> Self {

        let mut offsets : Vec<_> = offsets.to_owned();
        offsets.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)) );

        // For a start, do only the first row of each window.
        let mut rles = Vec::new();
        let mut curr : Option<RunLength> = None;
        for off in offsets {
            if let Some(mut this_curr) = curr.take() {
                if off.0 > this_curr.start.0 {
                    rles.push(this_curr);
                    curr = Some(RunLength { start : off, length : shape.1 });
                } else if off.0 == this_curr.start.0 {
                    if let Some(dist) = off.1.checked_sub(this_curr.start.1+this_curr.length) {
                        if dist == shape.1 {
                            this_curr.length += shape.1;
                            curr = Some(this_curr);
                        } else {
                            rles.push(this_curr);
                            curr = Some(RunLength { start : off, length : shape.1 });
                        }
                    } else {
                        // println!("{:?}", (this_curr, off));
                        panic!("Offset element in non-increasing col order found");
                    }
                } else {
                    // off.0 < curr.0
                    panic!("Offset element in non-increasing row order found");
                }
            } else {
                curr = Some(RunLength { start : off, length : shape.1 });
            }
        }

        if let Some(last_curr) = curr.take() {
            rles.push(last_curr);
        }

        // Now, reproduce all RLEs for as many rows as the window has.
        if shape.0 > 1 {
            let cloned_rles = rles.clone();
            for extra_row in 1..shape.1 {
                let row_below = cloned_rles.iter()
                    .map(|r| RunLength { start : (r.start.0 + extra_row, r.start.1), length : r.length });
                rles.extend(row_below);
            }
        }
        rles.sort_by(|a, b| a.start.0.cmp(&b.start.0).then(a.start.1.cmp(&b.start.1)) );

        // Count elements per row
        let rows = rows_vector(&rles);
        let rle = Self { rles, rows };

        verify_rle_state(&rle);

        rle
    }

    pub fn calculate_from_accumulated(
        w : &dyn AsRef<Window<i32>>,
        min_valid_row_range : Option<usize>,
        min_valid_col_range : Option<usize>
    ) -> Self {
        let mut r = Self::default();
        r.update_from_accumulated(w, min_valid_row_range, min_valid_col_range);
        r
    }

    // Similar to prune, but leaves the extremities untouched and examine
    // only the middle of the RunLenghth, possibly splitting them into
    // two or more RunLengths depending on whether there are new unmatched
    // pixels separating them.
    pub fn split(&mut self, w : &dyn AsRef<Window<u8>>) {

    }

    // Similar to expand, but shortens unmatched RunLength regions (or remove
    // them entirely), by just examining their extremities.
    pub fn prune(&mut self, w : &dyn AsRef<Window<u8>>) {

    }

    // Prunes, then expands the current RunLenghs.
    pub fn prune_and_expand(&mut self, w : &dyn AsRef<Window<u8>>) {
        self.prune(w);
        self.expand(w);
    }

    /* Updates the current RLE by iterating only over pixels *outside* the RLE, but
    connected to it.
    This is useful for an iterative thresholding strategy: If the desired foreground
    is more extreme than the current one, we know the RLE should contain at least the
    current pixels, therefore it is not necessary to iterate over them anymore. For
    this strategy to work, always start from a more conservative estimate, and increment
    towards a more extreme value. This only examines pixels that are either at the start,
    end, top or bottom of existing RLEs, or pixels neihboring the new matched regions recursively.
    Pixel rows outside these neighborhoods are ignored. */
    pub fn expand(&mut self, w : &dyn AsRef<Window<u8>>) {
        let mut r = 0;
    }

    // Find RLEs by bisecting over running sum over rows only (each new row
    // start at zero, and goess up to 256 at the final column if we are at a u8 image).
    // Although slower than bisecting over a full accumulated image, since there is still
    // a linear search over rows, this can operate over u8 images directly if the
    // binary image uses 1 to represent foreground values. This fails if any rows
    // saturate their sum (last pixel is 256), since it is not possible to determine
    // if there are any RLEs after the saturated column. The performance gain here
    // explores the fact that running sums of u8 images can execute vectorized
    // instructions containing more pixels, and there isn't the cost of conversion
    // if you start with a u8 image already.
    pub fn update_from_accumulated_rows(
        &mut self,
        w : &dyn AsRef<Window<i32>>,
        min_col_range : Option<usize>
    ) -> bool {
        let w = w.as_ref();
        self.rles.clear();
        self.rows.clear();
        let mut valid_col_ranges = Vec::new();
        for r in 0..w.height() {
            if w[(r, w.width()-1)] == i32::max_value() {
                return false;
            }
            find_row_rles(&mut valid_col_ranges, &mut self.rles, &mut self.rows, w, r, min_col_range);
        }
        verify_rle_state(&self);
        true
    }

    /* A RunLength encoding can usually be calculated much faster by bisection
    over the rows and columns of an accumulator image (containing the running sum of
    pixels in raster order). At each iteration, check if the pixels of a region boundary
    are equal. If they are, ignore the row region. This can be repeated
    recursively over non-examined regions. When a region does not have equal
    boundary pixels, it must be examined to determine the n RunLengths.
    If the difference between pixels happen to be equal to the number of columns
    times the non-zero constant value, we know there is a single RunLength there.
    This might mean a performance gain because while calculating the accumulated image
    is a vectorized operation, pointwise evaluating the RunLenghts for flat images is not.
    min_valid_range is the row length at which bisection stops, and can be chosen based on
    the expected object width (smaller values mean more bisection iterations, which might
    increase performance if objects are smaller than the desired range, since it reduces
    the number of linear pixel iterations). A defaut value of 8 is used if no value is informed.
    The image can also be bisected over whole row ranges, by passing the min_row_range argument. */
    pub fn update_from_accumulated(
        &mut self,
        w : &dyn AsRef<Window<i32>>,
        min_row_range : Option<usize>,
        min_col_range : Option<usize>
    ) {
        let w = w.as_ref();
        self.rles.clear();
        self.rows.clear();
        let mut valid_rows = Vec::new();
        calculate_valid_row_ranges(&mut valid_rows, w, 0, w.height() - 1, min_row_range.unwrap_or(MIN_VALID_INTEGRAL_ROW_RANGE));
        valid_rows.sort_by(|a, b| a.start.cmp(&b.start) );
        let mut valid_col_ranges = Vec::new();
        for r in valid_rows.drain(..).flatten() {
            find_row_rles(&mut valid_col_ranges, &mut self.rles, &mut self.rows, w, r, min_col_range);
        }
        verify_rle_state(&self);
    }

    pub fn calculate(w : &Window<u8>) -> Self {
        let mut rle = Self::default();
        rle.update(w);
        rle
    }

    /*
    Builds a runlength from a dense binary image. Pixels that are set to zero are
    not encoded in the RunLenght; pixels that are non-zero are considered part
    of the RunLength.

    Run-length encoding is trivially parallelizable by splitting an image with n
    rows into 4 images with dimensions n/4 x m, running it separately on the images,
    then polling the results back together.
    */
    pub fn update(&mut self, w : &Window<u8>) {
        self.rles.clear();
        self.rows.clear();
        let mut last_rle : Option<RunLength> = None;
        for r in 0..w.height() {

            let mut curr_range = Range { start : self.rles.len(), end : self.rles.len() };

            update_range(w, r, 0..w.width(), &mut last_rle, &mut curr_range, &mut self.rles);

            // Push any remaining RunLenghts that haven't ended before the last column.
            if let Some(rle) = last_rle.take() {
                self.rles.push(rle);
                curr_range.end += 1;
            }

            // If at least one RunLength has been pushed, add this range to the
            // row count. If no RunLenghts have been pushed, ignore this range.
            if curr_range.end > curr_range.start {
                self.rows.push(curr_range);
            }
        }

        verify_rle_state(&self);
    }

    pub fn new() -> Self {
        Default::default()
    }

    pub fn encode(w : &Window<u8>) -> Self {
        let mut encoding = Self::default();
        encoding.update(w);
        encoding
    }

    pub fn graph(&self) -> RunLengthGraph {
        let nrows = self.rows.len();

        let mut graph = UnGraph::<RunLength, /*Interval<usize>*/ () >::new_undirected();

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

                    // Return this to use RLE intervals as edes
                    // .map(|(ix, above)| (ix, rl_below.intersect(&above).unwrap().interval() ) );

                    // Return this for edges without information.
                    .map(|(ix, _above)| ix );

                // Add edges to top RLEs
                // for (above_ix, intv) in iter_above {
                for above_ix in iter_above {
                    graph.add_edge(below_ix, past_ixs[above_ix], /*intv*/ () );
                    uf.union(below_ix, past_ixs[above_ix]);
                }
            }
            mem::swap(&mut past_ixs, &mut curr_ixs);
            curr_ixs.clear();
        }

        RunLengthGraph { graph, uf }
    }
}

fn update_range(
    w : &Window<u8>,
    r : usize,
    cols : Range<usize>,
    last_rle : &mut Option<RunLength>,
    curr_range : &mut Range<usize>,
    rles : &mut Vec<RunLength>
) {
    for c in cols {
        if w[(r, c)] == 0 {
            // End current run-length (if any) when match is zero.
            if let Some(rle) = last_rle.take() {
                rles.push(rle);
                curr_range.end += 1;
            }

        } else {
            if let Some(mut rle) = last_rle.as_mut() {
                // Extend current run-length (if any) for a nonzero match.
                rle.length += 1;
            } else {
                // Innaugurate a new run-length if there isn't one already.
                *last_rle = Some(RunLength { start : (r, c), length : 1 });
            }
        }
    }
}

fn verify_rle_state(rle : &RunLengthEncoding) {

    // Verify each row index at least one valid RLE
    assert!(rle.rows.iter().all(|r| r.end - r.start >= 1 ));

    // Verify the summed length of sub-indices equals the total length.
    assert!(rle.rles.len() == rle.rows.iter().fold(0, |total, r| total + rle.rles[r.clone()].len() ) );

    // Verify end of previous range (open) equals start of next range (closed).
    assert!(rle.rows.iter().zip(rle.rows.iter().skip(1)).all(|(a, b)| a.end == b.start ));

}

/*pub fn draw_distinct(graph : &RunLengthGraph, bin : &mut WindowMut<u8>) {
    let split = graph.graph.split();
    for r in split.ranges {
        let v : f64 = rand::random();
        for node in &split.indices[r] {
            for pt in graph.graph[*node].middle_points() {
                bin[pt] = (v*255.) as u8;
            }
        }
    }
}*/

pub fn draw_distinct(img : &mut WindowMut<u8>, graph : &UnGraph<RunLength, /*Interval<usize>*/ ()>, split : &SplitGraph) {
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
#[derive(Debug, Clone)]
pub struct RunLengthGraph {

    // Edges carry the intersection of the RunLenghts.
    pub graph : UnGraph<RunLength, /*Interval<usize>*/ ()>,

    pub uf : UnionFind<NodeIndex>

}

fn extend_extremities(ps : &mut Vec<(usize, usize)>, i : usize) {
    let diff = ps[i].1 as i32 - ps[i-2].1 as i32;

    // Left entries
    if i % 2 == 0 {
        if diff >= 2 {
            let a = ps[i-2].1+1;
            let b = ps[i].1.saturating_sub(1);
            let r = ps[i].0;
            ps.extend((a..b).map(|c| (r, c) ));
        }
        if diff <= -2 {
            let a = ps[i].1+1;
            let b = ps[i-2].1.saturating_sub(1);
            let r = ps[i-2].0;
            ps.extend((a..b).map(|c| (r, c) ));
        }

    // Right entries
    } else {
        if diff >= 2 {
            let a = ps[i].1+1;
            let b = ps[i-2].1.saturating_sub(1);
            let r = ps[i-2].0;
            ps.extend((a..b).map(|c| (r, c) ));
        }
        if diff <= -2 {
            let a = ps[i-2].1+1;
            let b = ps[i].1.saturating_sub(1);
            let r = ps[i].0;
            ps.extend((a..b).map(|c| (r, c) ));
        }
    }
}

/*fn push_rect(
    rect : &mut (usize, usize, usize, usize),
    out : &mut BTreeMap<NodeIndex<u32>, Vec<(usize, usize, usize, usize)>>,
    graph : &UnGraph<RunLength, Interval<usize>>,
    b : NodeIndex<u32>,
    parent_ix : NodeIndex<u32>
) {
    println!("pushed");
    out.entry(parent_ix).or_insert(Vec::new()).push(*rect);
    *rect = graph[b].as_rect();
}*/

/*fn add_finished(
    finished : &mut Vec<NodeIndex<u32>>,
    rects : &mut Vec<(usize, usize, usize, usize)>,
    graph : &UnGraph<RunLength, Interval<usize>>,
) {

    finished.clear();
}*/

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

    // Returns external points in unspecified order.
    pub fn outer_points(
        &self,
        pts : Option<BTreeMap<NodeIndex, Vec<(usize, usize)>>>
    ) -> BTreeMap<NodeIndex, Vec<(usize, usize)>> {
        let mut pts = pts.unwrap_or(BTreeMap::new());
        pts.clear();

        let n = self.graph.raw_nodes().len();
        if n == 0 {
            return pts;
        }

        for ix in self.graph.node_indices() {
            let parent = self.uf.find(ix);
            let rle = &self.graph[ix];
            if let Some(mut this_pts) = pts.get_mut(&parent) {
                let mut last = this_pts.last_mut().unwrap();
                if last.0 == rle.start.0 {
                    // Last is from current row. Just extend right point.
                    last.1 = rle.end().1;

                } else {
                    // Last is from previous row. Add two new points.
                    this_pts.push(rle.start);
                    this_pts.push(rle.end());
                }

            } else {
                pts.insert(parent, vec![rle.start, rle.end()]);
            }
        }

        // Extends top and bottom row with middle points
        for (_, ps) in &mut pts {

            // println!("{:?}", ps);

            // Fill bottom and top rows
            let n_ps = ps.len();
            let fst_row = ps[0].0;
            let fst_col_fst_row = (ps[0].1+1);
            let lst_col_fst_row = (ps[1].1.saturating_sub(1));
            let fst_col_lst_row = (ps[n_ps-2].1+1);
            let lst_col_lst_row = (ps[n_ps-1].1.saturating_sub(1));
            // assert!(ps[0].0 == ps[1].0);
            // assert!(ps[n-2].0 == ps[n-1].0);
            ps.extend((fst_col_fst_row..lst_col_fst_row).map(|c| (fst_row, c) ));
            if n_ps >= 4 {
                let lst_row = ps[n_ps-2].0;
                ps.extend((fst_col_lst_row..lst_col_lst_row).map(|c| (lst_row, c) ));
            }
        }

        // Extends all contiguous intervals that have a small overlap (large gaps)
        for (_, ps) in &mut pts {
            // Iterate over even (left) entries
            for i in (2..ps.len()).step_by(2) {
                extend_extremities(ps, i);
            }

            // Iterate over odd (right) entries
            for i in (3..ps.len()).step_by(2) {
                extend_extremities(ps, i);
            }
        }

        pts
    }

    /// Return a single outer rect per object.
    pub fn outer_rects(
        &self,
        rects : Option<BTreeMap<NodeIndex, (usize, usize, usize, usize)>>
    ) -> BTreeMap<NodeIndex, (usize, usize, usize, usize)> {
        let mut rects = rects.unwrap_or(BTreeMap::new());
        rects.clear();

        for ix in self.graph.node_indices() {
            let parent = self.uf.find(ix);
            let rle = &self.graph[ix];
            if let Some(mut r) = rects.get_mut(&parent) {

                // Since raster order is preserved as we iterate over graph indices,
                // only update the sides and the bottom regions of rect.
                let intv = rle.interval();

                // Update left
                if intv.0 < r.1 {
                    r.1 = intv.0;
                }

                // Update width
                if intv.1 > r.1+r.3 {
                    r.3 = intv.1 - r.1;
                }

                r.2 = (rle.start.0 - r.0);
            } else {
                rects.insert(parent, rle.as_rect());
            }
        }

        // let mut labels = self.uf.clone().into_labeling().iter().unique().count();
        // assert!(rects.len() == labels, "Has {} rects but {} parent labels", rects.len(), labels);

        rects
    }

    /// Returns several rects that are contained in grouped  RunLengths. The rects are found
    /// by a single depth-first search at the run-length encoding graph. The rects are minimally-overlapping
    /// and will have maximum dimension given by the desired arguments, but some of them might be smaller.
    /// It is important to give a maximum height that is much smaller than the expected object size, or
    /// else if the object geometry is too far away from a rect (think a blob or circle) the rects will be
    /// too thin to cover a good proportion of the object area. The maximum width, however, can be whatever
    /// is best for your application. Passing usize::MAX to max_width will make output rects cover the whole object
    /// width when possible. Returned rects are in an unspecified order. The covered proportion of the object
    /// can be found by rects.fold(0., |s, r| s + r.area() ) / rle.area(), which is useful when choosing appropriate
    //// max_height and max_width parameters.
    pub fn inner_rects(
        &self,
        max_height : usize,
        max_width : usize
    ) -> BTreeMap<NodeIndex<u32>, Vec<(usize, usize, usize, usize)>> {
        let split = SplitGraph::new(&self.graph, &self.uf);
        let mut out = BTreeMap::new();
        let mut finished = Vec::new();
        for group in split.ranges.iter() {

            // Used to start the DFS
            let fst_el = split.indices[group.start];

            // Do a depth-first search at the RLE graph. Vertically-neighboring RLES
            // will be grouped together, so the inner rects can be found by simply
            // iterating over neighboring entries if we order the finish event indices into
            // a vector and look for contiguous RLEs. All rects found at this step have the
            // maximum possible width.
            finished.clear();
            petgraph::visit::depth_first_search(&self.graph, [fst_el], |event| {
                match event {
                    DfsEvent::Finish(ix, _) => {
                        finished.push(ix);
                    },
                    _ => { }
                }
            });
            if finished.len() < 2 {
                continue;
            }

            // Rect currently being grown.
            let mut rect : Option<(usize, usize, usize, usize)> = Some(self.graph[finished[0]].as_rect());
            let n = finished.len();
            let mut prev_is_below : Option<bool> = None;
            let parent = self.uf.find(split.indices[group.start]);

            // Holds all rects for a given parent.
            let mut rects = out.entry(parent).or_insert(Vec::new());

            for i in 0..(n-1) {
                let a = finished[i];
                let b = finished[i+1];
                let row_abs_diff = self.graph[b].start.0.abs_diff(self.graph[a].start.0);
                let b_is_contiguous_a = row_abs_diff == 1;
                if b_is_contiguous_a {
                    let mut r = rect.take().unwrap_or(self.graph[a].as_rect());
                    let rect_intv = Interval(r.1, r.1 + r.3);
                    let b_intv = self.graph[b].interval();
                    if let Some(intc) = b_intv.intersect(rect_intv) {
                        let next_is_below = self.graph[b].start.0 > self.graph[a].start.0;
                        if next_is_below && (prev_is_below.is_none() || prev_is_below == Some(true)) {
                            grow_bottom(&mut r, &intc);
                        } else if !next_is_below && (prev_is_below.is_none() || prev_is_below == Some(false)) {
                            grow_top(&mut r, &intc);
                        } else {
                            rects.push(r);
                        };
                        prev_is_below = Some(next_is_below);
                        if r.2 >= max_height {
                            rects.push(r);
                        } else {
                            rect = Some(r);
                        }
                    }
                } else {
                    if let Some(r) = rect.take() {
                        if r.2 > 1 {
                            rects.push(r);
                        }
                    }
                }
            }
            if let Some(r) = rect {
                if r.2 > 1 {
                    rects.push(r);
                }
            }
        }

        // Now, split all found rects respecting the desired maximum width.
        if max_width < usize::MAX {
            for (_, rects) in &mut out {
                let n = rects.len();
                for i in 0..n {
                    if rects[i].3 > max_width {
                        let mut divisor = 2;
                        let mut w = rects[i].3 / divisor;
                        while rects[i].3 / divisor > max_width {
                            divisor += 1;
                            w = rects[i].3 / divisor;
                        }
                        for ix in 1..divisor {
                            let r = (rects[i].0, rects[i].1 + ix*w, rects[i].2, w);
                            rects.push(r);
                        }
                        rects[i].3 = w;
                    }
                }
            }
        }

        out
    }

    pub fn split(&self) -> crate::graph::SplitGraph {
        crate::graph::SplitGraph::new(&self.graph, &self.uf)
    }

}

fn shrink_sides(
    r : &mut (usize, usize, usize, usize),
    intc : &Interval<usize>
) {
    if intc.0 > r.1 {
        r.1 = intc.0;
        r.3 = (r.1 + r.3) - intc.0;
    }

    // Shrink right end of rect. Here, we can split the remainder into another rect.
    if intc.1 < (r.1+r.3) {
        r.3 = intc.1 - r.1;
    }
}

fn grow_bottom(
    r : &mut (usize, usize, usize, usize), intc : &Interval<usize>
) {
    shrink_sides(r, intc);
    r.2 += 1;
    // rems
}

fn grow_top(
    r : &mut (usize, usize, usize, usize), intc : &Interval<usize>
) {
    shrink_sides(r, intc);
    r.0 -= 1;
    r.2 += 1;
}

// cargo test --all-features --lib -- inner_rects --nocapture
#[test]
fn inner_rects() {
    use crate::draw::*;
    for path in ["/home/diego/Downloads/pattern-hole.png", "/home/diego/Downloads/pattern.png"] {
        let mut img = crate::io::decode_from_file(path).unwrap();
        let rle = RunLengthEncoding::calculate(img.as_ref());
        let graph = rle.graph();
        let rects = graph.inner_rects(16, 16);
        for (_, rs) in rects {
            for r in rs {
                img.draw(Mark::Rect((r.0, r.1), (r.2, r.3), 255));
            }
        }
        img.show();
    }
}

pub enum Knot {

    // Elements are certain to be connected.
    Tight,

    // Elements might be connected, pending an evaluation of
    // the 8-neighborhood.
    Loose

}

// Represents a binary image in terms of rects. The smallest rects are pixels;
// the largest rect is the full image. Always use the largest possible rect to
// represent an object.
pub struct RectEncoder {

    sum_dst : Image<u8>,

    // Indices of the local sum image.
    graph : UnGraph<(usize, usize), Knot>

}

pub enum Region {

    // A dense region (rect).
    Complete((usize, usize, usize, usize)),

    // A 8-neighborhood anchored at the top-left with some missing pixels.
    Incomplete((usize, usize), Pattern)

}

pub struct RegionEncoding {

    graph : UnGraph<Region, ()>

}

fn update_graph(
    graph : &mut UnGraph<(usize, usize), Knot>,
    last_left : &mut NodeIndex,
    last_tops : &mut [NodeIndex],
    s : &Image<u8>,
    i : usize,
    j : usize,
    can_be_tight : bool
) {
    let curr_ix = graph.add_node((i, j));
    if i >= 1 {
        if can_be_tight && s[(i-1, j)] >= 7 {
            graph.add_edge(curr_ix, last_tops[j], Knot::Tight);
        } else if s[(i-1, j)] > 0 {
            graph.add_edge(curr_ix, last_tops[j], Knot::Loose);
        }
    }

    if j >= 1 {
        if can_be_tight && s[(i, j-1)] >= 7 {
            graph.add_edge(curr_ix, *last_left, Knot::Tight);
        } else if s[(i, j-1)] > 0 {
            graph.add_edge(curr_ix, *last_left, Knot::Loose);
        }
    }
    *last_left = curr_ix;
    last_tops[j] = curr_ix;
}

impl RectEncoder {

    pub fn new(sz : (usize, usize)) -> Self {
        assert!(sz.0 % 3 == 0 && sz.1 % 3 == 0);
        Self {
            sum_dst : Image::new_constant(sz.0 / 3, sz.1 / 3, 0),
            graph : UnGraph::new_undirected()
        }
    }

    pub fn encode(&mut self, img : &Image<u8>) {
        crate::local::baseline_local_sum(
            &img.full_window(),
            &mut self.sum_dst.full_window_mut()
        );
        self.graph.clear();
        let s = &self.sum_dst;
        let (sum_height, sum_width) = self.sum_dst.shape();
        let mut last_tops : Vec<_> = (0..sum_width).map(|_| NodeIndex::from(0) ).collect();
        let mut last_left = NodeIndex::from(0);
        for i in 0..sum_height {
            for j in 0..sum_width {
                match s[(i,j)] {
                    0 => {  },
                    1..=6 => {
                        update_graph(&mut self.graph, &mut last_left, &mut last_tops, s, i, j, false);
                    },
                    7.. => {
                        update_graph(&mut self.graph, &mut last_left, &mut last_tops, s, i, j, true);
                    }
                }
            }
        }
    }

}

const FULL_WHITE_PATTERN : Pattern = Pattern {
    center : true,
    neigh : 0b11111111
};

/*const FULL_DARK_PATTERN : Pattern = {
    center : false,
    neigh : 0b00000000
};*/

/// Represents the binary pattern of the 8-neighborhood and the
/// center pixel on/off status. The connectedness of two such neighorhoods
/// can be determined via a single match expression.
#[derive(Debug, Clone, Copy)]
pub struct Pattern {
    pub center : bool,
    pub neigh : u8
}

pub struct PatternEncoding {

    // 8-bit neighborhood and center patterns
    patterns : Vec<Pattern>,

    // Scale ranges, starting at the original image scale and multiplying
    // by 3 at each iteration.
    scales : Vec<Range<usize>>,

    // Top left position that anchors each pattern (in the original image scale)
    positions : Vec<(usize, usize)>,
}

pub struct PatternEncoder {
    sums : Vec<Image<u8>>,
    binaries : Vec<Image<u8>>
}

const PX_EVALUATED : u8 = 10;

fn eval_to_next_binary_or_pattern(
    patterns : &mut Vec<Pattern>,
    scale_ranges : &mut Vec<Range<usize>>,
    positions : &mut Vec<(usize, usize)>,
    next_bin : &mut WindowMut<u8>,
    curr_bin : &Window<u8>,
    curr_sum : &Window<u8>,
) {
    assert!(curr_sum.width() * 3 == curr_bin.width());
    assert!(next_bin.width() == curr_sum.width());
    for i in 0..curr_sum.height() {
        for j in 0..curr_sum.width() {
            match curr_sum[(i, j)] {
                0 => {
                    // Write to be evaluated at the binary image of next level.
                    next_bin[(i, j)] = 0;
                },
                9 => {
                    // Write to be evaluated at the binary image of next level.
                    next_bin[(i, j)] = 1;
                },
                s => {
                    let offset = (i*3, j*3);
                    if s <= 8 {
                        // This means all iterations in the neighborhood were
                        // hit by 0 or 1 (sum=0 or sum=9). Therefore, evaluate
                        // the full pattern at the current level.
                        next_bin[(i, j)] = PX_EVALUATED;
                        patterns.push(eval_pattern(curr_bin, offset, s));
                        positions.push(offset);
                    } else {
                        // This means that at least one pixel received the PX_EVALUATED
                        // constant (10), therefore the remaining pixels that matched 1
                        // must be inserted at the  previous level as a full white pattern.
                        for k in 0..3 {
                            for l in 0..3 {
                                if curr_bin[(offset.0+k, offset.1+l)] == 1 {
                                    if let Some(mut r) = scale_ranges.last_mut() {
                                        patterns.push(FULL_WHITE_PATTERN);
                                        let last = patterns.len()-1;
                                        patterns.swap(r.end, last);
                                        positions.push(offset);
                                        positions.swap(r.end, last);
                                        r.end += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn px_at_pos(w : &Window<u8>, tl : (usize, usize), pos : u8) -> u8 {
    w[offset_at_pos(tl, pos)]
}

fn white_at_pos(pos : u8) -> u8 {
    /*match pos {
        0 => 0b10000000,
        1 => 0b01000000,
        2 => 0b00100000,
        3 => 0b00010000,
        4 => 0b00001000,
        5 => 0b00000100,
        6 => 0b00000010,
        7 => 0b00000001,
        _ => panic!()
    }*/
    0b10000000u8 >> pos
}

fn black_at_pos(pos : u8) -> u8 {
    /*match pos {
        0 => 0b01111111,
        1 => 0b10111111,
        2 => 0b11011111,
        3 => 0b11101111,
        4 => 0b11110111,
        5 => 0b11111011,
        6 => 0b11111101,
        7 => 0b11111110,
        _ => panic!()
    }*/
    (0b01111111u8).rotate_right(pos as u32)
}

fn linear_offset_at_pos(linear_tl : usize, width : usize, pos : u8) -> usize {
    match pos {
        0 => linear_tl,
        1 => linear_tl+1,
        2 => linear_tl+2,
        3 => linear_tl+width+2,
        4 => linear_tl+2*width+2,
        5 => linear_tl+2*width+1,
        6 => linear_tl+2*width,
        7 => linear_tl+width,
        _ => panic!()
    }
}

fn offset_at_pos(tl : (usize, usize), pos : u8) -> (usize, usize) {
    match pos {
        0 => tl,
        1 => (tl.0, tl.1+1),
        2 => (tl.0, tl.1+2),
        3 => (tl.0+1, tl.1+2),
        4 => (tl.0+2, tl.1+2),
        5 => (tl.0+2, tl.1+1),
        6 => (tl.0+2, tl.1),
        7 => (tl.0+1, tl.1),
        _ => panic!()
    }
}

fn eval_next_white(w : &Window<u8>, tl : (usize, usize), pos : u8, neigh : u8, sum : u8) -> u8 {
    if sum == 0 {
        neigh
    } else {
        if px_at_pos(w, tl, pos) == 1 {
            eval_next_white(w, tl, pos+1, neigh | white_at_pos(pos), sum-1)
        } else {
            eval_next_white(w, tl, pos+1, neigh, sum)
        }
    }
}

fn eval_next_black(w : &Window<u8>, tl : (usize, usize), pos : u8, neigh : u8, sum : u8) -> u8 {
    if sum == 0 {
        neigh
    } else {
        if px_at_pos(w, tl, pos) == 0 {
            eval_next_black(w, tl, pos+1, neigh | black_at_pos(pos), sum-1)
        } else {
            eval_next_black(w, tl, pos+1, neigh, sum)
        }
    }
}

fn eval_pattern(w : &Window<u8>, tl : (usize, usize), sum : u8) -> Pattern {
    let center = w[(tl.0+1, tl.1+1)] == 1;
    if center {
        if sum <= 4 {
            Pattern { center, neigh : eval_next_white(w, tl, 0, 0b00000000, sum - 1) }
        } else {
            Pattern { center, neigh : eval_next_black(w, tl, 0, 0b11111111, 8 - sum - 1) }
        }
    } else {
        if sum <= 4 {
            Pattern { center, neigh : eval_next_white(w, tl, 0, 0b00000000, sum) }
        } else {
            Pattern { center, neigh : eval_next_black(w, tl, 0, 0b11111111, 8 - sum) }
        }
    }
}

impl PatternEncoder {

    pub fn new(mut shape : (usize, usize)) -> Self {
        assert!(shape.0 % 3 == 0 && shape.1 % 3 == 0);
        let mut sums = Vec::new();
        while shape.0 >= 3 {
            shape.0 /= 3;
            shape.1 /= 3;
            sums.push(Image::new_constant(shape.0, shape.1, 0));
        }
        // The first binary is actually the received image, so we
        // need one less binary than sum images.
        let binaries = sums.clone();
        Self { sums, binaries }
    }

    pub fn encode(&mut self, img : &Window<u8>) {
        let mut patterns = Vec::new();
        let mut scales = Vec::new();
        let mut positions = Vec::new();
        for lvl in 0..self.sums.len() {

            // Calculate sum over previous binary image (the 0th sum
            // is the original image).
            if lvl == 0 {
                crate::local::baseline_local_sum(
                    &img,
                    &mut self.sums[0].as_mut()
                );
            } else {
                crate::local::baseline_local_sum(
                    &self.binaries[lvl-1].full_window(),
                    &mut self.sums[lvl].full_window_mut()
                );
            }

            let mut next_binary = mem::take(&mut self.binaries[lvl]);
            let curr_binary = if lvl == 0 {
                img.clone()
            } else {
                self.binaries[lvl-1].full_window()
            };

            let size_before = patterns.len();
            eval_to_next_binary_or_pattern(
                &mut patterns,
                &mut scales,
                &mut positions,
                &mut next_binary.full_window_mut(),

                &curr_binary,
                &self.sums[lvl].full_window()
            );
            self.binaries[lvl] = next_binary;

            let scale_range = Range { start : size_before, end : patterns.len() };
            scales.push(scale_range);

            // Empty scale range migt mean ALL patterns might be upgraded to
            // the next level, and does not mean level is empty.
            /*if scale_range.is_empty() {
                // This means all pixels were 0 at this level, so there
                // is no point in iterating on any remaining coarser scales.
                return;
            } else {
                scales.push(scale_range);
            }*/

        }

    }

}

#[test]
fn shl() {
    // println!("{:#08b} {:#08b}", ONE_TL, ONE_T);
}

/*const fn shr_by<const N : u8>(a : u8) -> u8 {
    a >> N
}

const fn shl_by<const N : u8>(a : u8) -> u8 {
    a << N
}

const fn bitor(a : u8, b : u8) -> u8 {
    a | b
}

pub const ZERO : u8 = 0b00000000;

pub const ONE_TL : u8 = 0b10000000;
pub const ONE_T : u8 = ONE_TL.rotate_right(1);
pub const ONE_TR : u8 = ONE_TL.rotate_right(2);
pub const ONE_R : u8 = ONE_TL.rotate_right(3);
pub const ONE_BR : u8 = ONE_TL.rotate_right(4);
pub const ONE_B : u8 = ONE_TL.rotate_right(5);
pub const ONE_BL : u8 = ONE_TL.rotate_right(6);
pub const ONE_L : u8 = ONE_TL.rotate_right(7);

// Two values close together.
pub const TWO_TL_T : u8 = 0b11000000;
pub const TWO_T_TR : u8 = TWO_TL_T.rotate_right(1);
pub const TWO_TR_R : u8 = TWO_TL_T.rotate_right(2);
pub const TWO_R_BR : u8 = TWO_TL_T.rotate_right(3);
pub const TWO_BR_B : u8 = TWO_TL_T.rotate_right(4);
pub const TWO_B_BL : u8 = TWO_TL_T.rotate_right(5);
pub const TWO_BL_L : u8 = TWO_TL_T.rotate_right(6);
pub const TWO_L_TL : u8 = TWO_TL_T.rotate_right(7);

// Two values separated by one 0.
pub const TWO_S1_TL_TR : u8 = 0b10100000;
pub const TWO_S1_T_R : u8 = TWO_S1_TL_TR.rotate_right(1);
pub const TWO_S1_TR_BR : u8 = TWO_S1_TL_TR.rotate_right(2);
pub const TWO_S1_R_B : u8 = TWO_S1_TL_TR.rotate_right(3);
pub const TWO_S1_BR_B : u8 = TWO_S1_TL_TR.rotate_right(4);
pub const TWO_S1_B_BL : u8 = TWO_S1_TL_TR.rotate_right(5);
pub const TWO_S1_BL_L : u8 = TWO_S1_TL_TR.rotate_right(6);
pub const TWO_S1_L_TL : u8 = TWO_S1_TL_TR.rotate_right(7);*/

// The values w/ 5 are the bit neg of 3; values w/7 are bitneg of 2
// values with/ 8 is unique. 3 and 4 are the hard ones.

/*// Two values separated by two 0s.
pub const TWO_S2_TL_TR : u8 = 0b100100000;
pub const TWO_S2_T_R : u8 = TWO_S1_TL_TR.rotate_right(1);
pub const TWO_S2_TR_BR : u8 = TWO_S1_TL_TR.rotate_right(2);
pub const TWO_S2_R_B : u8 = TWO_S1_TL_TR.rotate_right(3);
pub const TWO_S2_BR_B : u8 = TWO_S1_TL_TR.rotate_right(4);
pub const TWO_S2_B_BL : u8 = TWO_S1_TL_TR.rotate_right(5);
pub const TWO_S2_BL_L : u8 = TWO_S1_TL_TR.rotate_right(6);
pub const TWO_S2_L_TL : u8 = TWO_S1_TL_TR.rotate_right(7);*/









