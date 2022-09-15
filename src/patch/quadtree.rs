use crate::image::*;
use petgraph::{graph::DiGraph, graph::NodeIndex, data::Build, Direction};
use nalgebra::Scalar;
use std::fmt::Debug;
use std::collections::HashMap;

pub struct QuadTree<'a, N>(DiGraph<Window<'a, N>, ()>) where N :Pixel;

impl<'a, N> AsRef<DiGraph<Window<'a, N>, ()>> for QuadTree<'a, N>
where
    N : Pixel
{

    fn as_ref(&self) -> &DiGraph<Window<'a, N>, ()> {
        &self.0
    }
}

impl<'a, N> QuadTree<'a, N>
where
    N : Pixel + Scalar + Clone + Copy + Debug,
    &'a [N] : Storage<N>
{

    /// Return the four window roots (same as self.levels(0)). All windows will
    /// have size len/2
    pub fn root_widows(&'a self) -> impl Iterator<Item=Window<'a, N>> + 'a {
        self.0.externals(Direction::Incoming).map(move |ix| self.0[ix].clone() )
    }

    /// Returns an unknown number of window leafs (i.e. windows without further decomposition).
    /// Windows might have different sizes.
    pub fn leaf_windows(&'a self) -> impl Iterator<Item=Window<'a, N>> + 'a {
        self.0.externals(Direction::Outgoing).map(move |ix| self.0[ix].clone() )
    }

    /// Returns all windows of this quadtree, including the intermediate
    /// partitions.
    pub fn all_windows(&'a self) -> impl Iterator<Item=Window<'a, N>> + 'a {
        self.0.raw_nodes().iter().map(move |node| node.weight.clone() )
    }

}

/// Recursively partition the image into a quad-tree. Define regions by the equality
/// or small difference of all pixels within a quadtree region. A reasonabe segmentation
/// strategy is to start with a quadtree segmentation over a coarse version of the image,
/// and use the center of each rect in the graph as the seed for growth strategies at
/// the full image scale. In this way, small or large regions depending on the application
/// can be ruled out before processing at the more expensive detail scale. This does a recursive
/// call, and will fail if you don't set the minimum size to a value that will not exhaust the
/// call stack. The criteria function should be any transitive property (which guarantees pairwise
/// comparison between any two pixels means that the relation holds for all pixels). The quadtree
/// segmenter is costly, since each pixel might be evaluated as many times as the region must be
/// divided further; so restrict its use when your image is small and/or reasonably well delimited,
/// and you will benefit from its richer representation (i.e. you want to filter out windows with
/// a given size for further processing). It is trivial to build a parallel version
/// of this algorithm, since each region is evaluated independently of the others (just keep the
/// graph inside a Arc<Mutex<T>>, locking the mutex to append nodes only, leaving the pixel
/// evaluation to run in parallel). The decomposition stops when crit is satisfied for all pixel
/// pairs or the minimum window size is achieved. QuadTree might also be useful when processing videos,
/// where large homogeneous regions are expected to be stabe (just sample a few pixels randomly or uniformly
/// or at its edges to verify if the region is still stable at a next frame. If it is, ignore it for
/// processing at the current frame). This is basically the split-and-merge segmentation strategy.
pub fn quad_tree_segmenter(
    win : Window<u8>,
    crit : fn(u8,u8)->bool,
    min_sz : usize
) -> QuadTree<u8> {

    assert!(win.width() % 4 == 0);
    assert!(win.height() % 4 == 0);

    let mut qt = DiGraph::new();

    // Iterate over top-level elements of quad-tree, depth-first
    for w in win.split_equivalent_windows(2, 2) {
        let ix = qt.add_node(w.clone());
        serial_quad_tree_segmenter_step(&mut qt, w.clone(), ix, crit, min_sz);
    }

    QuadTree(qt)
}

// Iterate over a newly-inserted node, depth-first.
fn serial_quad_tree_segmenter_step<'a>(
    qt : &mut DiGraph<Window<'a, u8>, ()>,
    parent_win : Window<'a, u8>,
    parent_ix : NodeIndex,
    crit : fn(u8, u8)->bool,
    min_sz : usize
) {

    // By transtiveness of equality (or difference), the comparison of any two pixels
    // means the satisfaction of the condition for all pixels. The comparison of left with
    // right is purely by convention and convenience (any other exhaustive pairwise comparison
    // that contains all pixels would work). The "small difference" criteria is not really transitive,
    // but we approximate it nevertheless.
    let all_satisfy = parent_win.pixels(1)
        .zip(parent_win.pixels(1).skip(1)).all(|(a, b)| crit(*a, *b) );

    if !all_satisfy && parent_win.width() / 2 >= min_sz && parent_win.height() / 2 >= min_sz {

        // Iterate over new region, depth-first
        for child_win in parent_win.split_equivalent_windows(2, 2) {
            let child_ix = qt.add_node(child_win.clone());
            qt.add_edge(parent_ix, child_ix, ());
            serial_quad_tree_segmenter_step(qt, child_win.clone(), child_ix, crit, min_sz);
        }
    }

    // If all pixels satisfy the condition, do nothing, keeping the currently-inserted node as a left node.
    // If minimum window size was reached at previous iteration, also do nothing.
}

/*
TODO a better algorithm starts by comparing all pairwise pixels within windows of min_sz over the
whole window. Then, we merge those smaller windows by comparing the first pixel of each subwindow,
and so on and so forth, until windows get to the size of length/2.
*/

/// Builds a LUT between labels and pixels, such that the color for each label
/// is the color of the first pixel found with that label.
pub fn recolor_with_labels<'a>(mut img : WindowMut<'a, u8>, labels : &'a Window<'a, u8>) {

    assert!(img.width() == labels.width() && img.height() == labels.height());

    let mut lut = HashMap::new();

    /*for (px, lbl) in img.pixels_mut(1).zip(labels.pixels(1)) {
        match lut.get(lbl) {
            Some(color) => {
                *px = *color;
            },
            None => {
                lut.insert(lbl, *px);
                // Just keep the pixel color the first time this label is found.
            }
        }
    }*/
    for r in 0..img.height() {
        for c in 0..img.width() {
            let lbl = img[(r, c)];
            match lut.get(&lbl) {
                Some(color) => {
                    img[(r, c)] = *color;
                },
                None => {
                    lut.insert(lbl, img[(r, c)]);
                }
            }
        }
    }
}

