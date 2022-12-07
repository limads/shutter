use std::any::Any;
use crate::image::*;
use num_traits::Zero;
use nalgebra::*;
use std::mem;
use std::ops::{Add, Shl};
use std::ops::Range;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::Direction;
use petgraph::visit::DfsEvent;
use std::collections::{BTreeMap, BTreeSet};
use petgraph::unionfind::UnionFind;
use std::cmp::{PartialEq, Eq};
use smallvec::SmallVec;

#[cfg(feature="ipp")]
fn ipp_integral(win : &Window<u8>, dst : &mut WindowMut<i32>) {
    assert!(win.width() == dst.width() - 1);
    assert!(win.height() == dst.height() - 1);
    unsafe {
        let (src_step, src_sz) = crate::image::ipputils::step_and_size_for_window(win);
        let (dst_step, dst_sz) = crate::image::ipputils::step_and_size_for_window_mut(&dst);

        // Usually a negative value, which allows for larger images to be processed without
        // an integer overflow.
        let offset_val : i32 = 0;
        let ans = crate::foreign::ipp::ippcv::ippiIntegral_8u32s_C1R(
            win.as_ptr(),
            src_step,
            mem::transmute(dst.as_mut_ptr()),
            dst_step,
            std::mem::transmute(src_sz),
            offset_val
        );
        assert!(ans == 0, "Error processing integral image: {}", ans);
        return;
    }
}

/// The integral should have nrows+1, ncols+1 relative to the source image.
pub struct Integral<T>(ImageBuf<T>)
where
    T : Pixel + Scalar + Clone + Copy + Any;

impl<T> Integral<T>
where
    T : Pixel + Scalar + Clone + Copy + Any + Zero + From<u8> + Default + std::ops::AddAssign
{

    pub fn new_constant(height : usize, width : usize, val : T) -> Self {
        Integral(ImageBuf::<T>::new_constant(height, width, val))
    }

    /* Integral sums the rectangle (0, 0) (i, j) above the pixel (i, j).*/
    pub fn update(&mut self, win : &Window<'_, u8>) {

        #[cfg(feature="ipp")]
        unsafe {
            if self.0.pixel_is::<i32>() {
                return ipp_integral(win, mem::transmute(&mut self.0.full_window_mut()));
            }
        }
        panic!();

        // TODO  'attempt to add with overflow'
        let mut dst = &mut self.0;
        dst[(0usize, 0usize)] = T::from(win[(0 as usize, 0 as usize)]);
        unsafe {
            for ix in 1..dst.as_slice().len() {
                let prev = *dst.unchecked_linear_index(ix-1);
                *dst.unchecked_linear_index_mut(ix) += prev;
            }
        }
    }

    pub fn calculate(win : &Window<'_, u8>) -> Self {
        // TODO Make sure IppiIntegral overwrites all pixels.
        // The first pixel of the integral image equals the first pixel of the original
        // image. All other pixels are the sum of the previous pixels up to the current pixel.
        let mut dst = unsafe { Self(ImageBuf::<T>::new_empty(win.height() + 1, win.width() + 1)) };
        dst.update(win);
        dst
    }

}

// pub enum Region {
//    Empty
// }

/*// Carries a pair of cursor indices that change at each recursion (r1, r2)
// and auxiliary data (to index the non-recursing dimension). Recursion happens
// when a match does not happen.
pub fn search_recursively_not_matched<F>(
    ranges : &mut Vec<Range<usize>>,
    w : &Window<i32>,
    r1 : usize,
    r2 : usize,
    a : usize,
    comp : F,
    min_sz : usize
) where
    F : Fn(&Window<i32>, usize, usize, usize)->bool + Copy
{
    if r2 - r1 <= min_sz {
        return;
    }
    assert!(r2 > r1);
    if comp(w, r1, r2, a) {
        ranges.push(Range { start : r1, end : r2 });
    } else {
        let middle = r1 + (r2 - r1) / 2;
        search_recursively_not_matched(ranges, w, r1, middle-1, a, comp, min_sz);
        search_recursively_not_matched(ranges, w, middle, r2, a, comp, min_sz);
    }
}

// Recursion happens when a match happens.
pub fn search_recursively_matched<F>(
    ranges : &mut Vec<Range<usize>>,
    w : &Window<i32>,
    r1 : usize,
    r2 : usize,
    a : usize,
    comp : F,
    min_sz : usize
) where
    F : Fn(&Window<i32>, usize, usize, usize)->bool + Copy
{
    if r2 - r1 <= min_sz {
        return;
    }
    assert!(r2 > r1);
    if comp(w, r1, r2, a) {
        let middle = r1 + (r2 - r1) / 2;
        search_recursively_matched(ranges, w, r1, middle-1, a, comp, min_sz);
        search_recursively_matched(ranges, w, middle, r2, a, comp, min_sz);
    } else {
        ranges.push(Range { start : r1, end : r2 });
    }
}*/

/*fn empty_row_comp(w : &Window<i32>, r1 : usize, r2 : usize, c : usize) -> bool {
    w[(r2, c)] - w[(r1, c)] == 0
}

fn empty_col_comp(w : &Window<i32>, c1 : usize, c2 : usize, r : usize) -> bool {
    w[(r, c2)] - w[(r, c1)] == 0
}*/

#[derive(Debug, Clone, Copy)]
pub struct Quad {
    pub rect : (usize, usize, usize, usize),
    pub content : Content
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Content {
    Empty,
    Mixed,
    Filled
}

impl Content {

    pub fn contains_any(&self) -> bool {
        match self {
            Content::Mixed | Content::Filled => true,
            Content::Empty => false
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Content::Empty => true,
            _ => false
        }
    }

}

fn rect_area(w : &Window<i32>) -> i32 {
    let br = w[(w.height()-1,w.width()-1)];
    let tr = w[(0, w.width()-1)];
    let tl = w[(0usize, 0usize)];
    let bl = w[(w.height()-1, 0)];
    (br - (tr - tl) - bl)
}

fn rect_content(w : &Window<i32>) -> Content {
    let total = rect_area(w);
    let max = ((w.height()-1) * (w.width()-1)) as i32 * 255;
    if total == 0 {
        Content::Empty
    } else if total == max {
        Content::Filled
    } else {
        Content::Mixed
    }
}

pub fn iter_col(
    graph : &mut DiGraph<Quad, ()>,
    parent_ix : NodeIndex<u32>,
    w : &Window<i32>,
    min_sz : (usize, usize)
) {
    // if w.height() < min_sz.0 || w.width() < min_sz.1 {
    //    return;
    // }
    let rect = (w.offset().0, w.offset().1, w.height(), w.width());
    match rect_content(w) {
        Content::Empty => {
            let ix = graph.add_node(Quad { rect, content : Content::Empty });
            graph.add_edge(parent_ix, ix, ());
        },
        Content::Mixed => {
            let ix = graph.add_node(Quad { rect, content : Content::Mixed });
            graph.add_edge(parent_ix, ix, ());
            let half_w = w.width() / 2;
            if half_w >= min_sz.1 {
                let left = w.window((0, 0), (w.height(), half_w)).unwrap();
                let right = w.window((0, half_w), (w.height(), half_w)).unwrap();
                iter_row(graph, ix, &left, min_sz);
                iter_row(graph, ix, &right, min_sz);
            }
        },
        Content::Filled => {
            let ix = graph.add_node(Quad { rect, content : Content::Filled });
            graph.add_edge(parent_ix, ix, ());
        }
    }
}

pub fn iter_row(
    graph : &mut DiGraph<Quad, ()>,
    parent_ix : NodeIndex<u32>,
    w : &Window<i32>,
    min_sz : (usize, usize)
) {
    // if w.height() < min_sz.0 || w.width() < min_sz.1 {
    //    return;
    // }
    let rect = (w.offset().0, w.offset().1, w.height(), w.width());
    match rect_content(w) {
        Content::Empty => {
            let ix = graph.add_node(Quad { rect, content : Content::Empty });
            graph.add_edge(parent_ix, ix, ());
        },
        Content::Mixed => {
            let ix = graph.add_node(Quad { rect, content : Content::Mixed });
            graph.add_edge(parent_ix, ix, ());
            let half_h = w.height() / 2;
            if half_h >= min_sz.0 {
                let top = w.window((0, 0), (half_h, w.width())).unwrap();
                let bottom = w.window((half_h, 0), (half_h, w.width())).unwrap();
                iter_col(graph, ix, &top, min_sz);
                iter_col(graph, ix, &bottom, min_sz);
            }
        },
        Content::Filled => {
            let ix = graph.add_node(Quad { rect, content : Content::Filled });
            graph.add_edge(parent_ix, ix, ());
        }
    }
}

pub fn iter_quad(
    graph : &mut DiGraph<Quad, ()>,
    parent_ix : NodeIndex<u32>,
    w : &Window<i32>,
    min_sz : (usize, usize)
) {
    if w.height() < min_sz.0 || w.width() < min_sz.1 {
        return;
    }
    let rect = (w.offset().0, w.offset().1, w.height(), w.width());
    match rect_content(w) {
        Content::Empty => {
            let ix = graph.add_node(Quad { rect, content : Content::Empty });
            graph.add_edge(parent_ix, ix, ());
        },
        Content::Mixed => {
            let ix = graph.add_node(Quad { rect, content : Content::Mixed });
            graph.add_edge(parent_ix, ix, ());
            let half_h = w.height() / 2;
            let half_w = w.width() / 2;
            let top_left = w.window((0, 0), (half_h, half_w)).unwrap();
            let top_right = w.window((0, half_w), (half_h, half_w)).unwrap();
            let bottom_left = w.window((half_h, 0), (half_h, half_w)).unwrap();
            let bottom_right = w.window((half_h, half_w), (half_h, half_w)).unwrap();
            if half_h >= min_sz.0 && half_w >= min_sz.1 {
                iter_quad(graph, ix, &top_left, min_sz);
                iter_quad(graph, ix, &top_right, min_sz);
                iter_quad(graph, ix, &bottom_left, min_sz);
                iter_quad(graph, ix, &bottom_right, min_sz);
            }
        },
        Content::Filled => {
            let ix = graph.add_node(Quad { rect, content : Content::Filled });
            graph.add_edge(parent_ix, ix, ());
        }
    }
}

/*fn preserve_matching_regions(graph : &mut DiGraph<Rect, ()>) {
    let start = graph.externals(Direction::Incoming).next().unwrap();
    let mut is_solid : BTreeSet<NodeIndex<u32>> = BTreeSet::new();
    petgraph::visit::depth_first_search(&*graph, Some(start), |event| {
        match event {
            DfsEvent::Finish(u, _) => {

                if !graph[u].is_match {
                    return;
                }

                let children = graph.neighbors_directed(u, Direction::Outgoing);
                let is_leaf = children.clone().next().is_none();
                if is_leaf {
                    is_solid.insert(u);
                } else {
                    let solid = children.clone().all(|c| is_solid.contains(&c) );
                    if solid {
                        assert!(!is_solid.contains(&u));
                        for ch in children.clone() {
                            is_solid.remove(&ch);
                        }
                        is_solid.insert(u);
                        assert!(!children.clone().any(|ch| is_solid.contains(&ch) ));
                    } else {
                        assert!(!is_solid.contains(&u));
                    }
                }
            },
            _ => { }
        }
    });
    graph.retain_nodes(|_, ix| is_solid.contains(&ix) );
}*/

/// A QuadTree calculated from an integral image.
pub struct IntegralQuad(DiGraph<Quad, ()>);

impl std::convert::AsRef<DiGraph<Quad, ()>> for IntegralQuad {

    fn as_ref(&self) -> &DiGraph<Quad, ()> {
        &self.0
    }

}

#[derive(Debug, Clone, Copy)]
pub enum Split {

    // Split four quadrants at each iteration. Leads
    // to more symmetric quads, since iteration stops
    // when either one of (row, col) is smaller than
    // minimum area.
    Quadrants,

    // Split rows, then columns. Leads to rects that
    // reproduce the aspect ratio of the image, since
    // iteration stops when the min(row, row) is
    // smaller than the minimum area
    Halves

}

pub fn bounding_row_points(window : &Window<i32>, pts : &mut Vec<(usize, usize)>) {
    let width = window.width()-1;
    let mut pts = Vec::new();
    for i in 1..window.height() {
        let fst_px = window[(i, 0)] - window[(i-1, 0)];
        let lst_px = window[(i, width-1)] - window[(i-1, width-1)];
        if lst_px == fst_px {
            continue;
        }

        // TODO use sub_ptr instead of offset_from (that returns usize) when stabilized.
        let fst_addr = &window[(i, 0)] as *const i32;
        let start = window.row(i).unwrap().partition_point(|px| {
            let col = unsafe { (px as *const i32).offset_from(fst_addr) as usize };
            (px - window[(i-1, col)]) == fst_px
        });
        let mut end = window.row(i).unwrap()[start..].partition_point(|px| {
            let col = unsafe { (px as *const i32).offset_from(fst_addr) as usize };
            (px - window[(i-1, col)]) != lst_px
        });
        end += start;

        // TODO verify (win[end] - win[start]) / 255 == end - start.
        // Do not push in case this doesn't apply.
        pts.push((i - 1, start - 1));
        pts.push((i - 1, end - 1));
    }
}

impl IntegralQuad {

    pub fn new(window : &Window<i32>, min_sz : (usize, usize), method : Split) -> Self {
        let h = if window.height() % 2 == 0 { window.height() } else { window.height() - 1 };
        let w = if window.width() % 2 == 0 { window.width() } else { window.width() - 1 };
        let win = window.window((0, 0), (h, w)).unwrap();
        assert!(win.height() % 2 == 0 && win.width() % 2 == 0);
        let mut graph : DiGraph<Quad, ()> = DiGraph::new();
        let h = win.height();
        let w = win.width();
        let half_h = win.height()/2;
        let half_w = win.width()/2;
        let content = rect_content(&win);
        let ix = graph.add_node(Quad { rect : (0, 0, h, w), content : content.clone() });
        if content == Content::Mixed {
            match method {
                Split::Halves => {
                    let top = win.window((0, 0), (half_h, w)).unwrap();
                    let bottom = win.window((half_h, 0), (half_h, w)).unwrap();
                    iter_row(&mut graph, ix, &top, min_sz);
                    iter_row(&mut graph, ix, &bottom, min_sz);
                },
                Split::Quadrants => {
                    let top_left = win.window((0, 0), (half_h, half_w)).unwrap();
                    let top_right = win.window((0, half_w), (half_h, half_w)).unwrap();
                    let bottom_left = win.window((half_h, 0), (half_h, half_w)).unwrap();
                    let bottom_right = win.window((half_h, half_w), (half_h, half_w)).unwrap();
                    iter_quad(&mut graph, ix, &top_left, min_sz);
                    iter_quad(&mut graph, ix, &top_right, min_sz);
                    iter_quad(&mut graph, ix, &bottom_left, min_sz);
                    iter_quad(&mut graph, ix, &bottom_right, min_sz);
                }
            }
        }
        Self(graph)
    }

    /*pub fn empty_leaves(&self) -> Vec<Quad> {
        for i in 0..self.0.raw_nodes().len() {
            for j in (i+1)..self.0.raw_nodes().len() {

            }
        }
    }*/

    /// Returns all leaves of the graph that either contain mixed or filled content.
    pub fn leaves_with_content(&self) -> BTreeMap<usize, SmallVec<[Quad; 4]>> {
        use crate::shape::*;

        let mut externals : Vec<NodeIndex<u32>> = self.0.externals(Direction::Outgoing).collect();
        externals.retain(|n| self.0[*n].content.contains_any() );

        let mut uf = UnionFind::<usize>::new(externals.len());
        for i in 0..(externals.len().saturating_sub(1)) {
            for j in (i+1)..externals.len() {
                let ni = externals[i];
                let nj = externals[j];
                let r1 = Region::from_rect_tuple(&self.0[ni].rect);
                let r2 = Region::from_rect_tuple(&self.0[nj].rect);
                match r1.proximity(&r2) {
                    Proximity::Contact | Proximity::Overlap => {
                        uf.union(i, j);
                    },
                    _ => { }
                }
            }
        }

        let mut btm = BTreeMap::new();
        for (i, ni) in externals.drain(..).enumerate() {
            if self.0[ni].content.contains_any() {
                let key = uf.find(i);
                btm.entry(key).or_insert(SmallVec::new()).push(self.0[ni].clone());
            }
        }
        btm
    }

    pub fn matching_rects(&self) -> Vec<(usize, usize, usize, usize)> {
        let btm = self.leaves_with_content();
        let mut rects = Vec::new();
        for (_, mut qs) in btm {
            rects.extend(qs.drain(..).filter_map(|q| if q.content.contains_any() {
                Some(q.rect)
            } else {
                None
            }));
        }
        rects
    }

    pub fn enclosing_rects(&self) -> Vec<(usize, usize, usize, usize)> {
        let btm = self.leaves_with_content();
        let mut rects = Vec::new();
        for (_, qs) in &btm {
            let outer = crate::shape::enclosing_rect_for_rects(qs.iter().map(|q| q.rect.clone() )).unwrap();
            rects.push(outer);
        }
        rects
    }

}

// cargo test --features ipp --message-format short -- image_integral --nocapture
#[test]
fn image_integral() {

    use crate::draw::*;

    {
        let mut img = ImageBuf::<u8>::new_constant(16, 16, 1);
        let int = Integral::<i32>::calculate(&img.full_window());
        println!("{}", rect_area(&int.as_ref().window((8,8), (4,4)).unwrap()));
        return;
    }

    let mut target = ImageBuf::<u8>::new_constant(128, 128, 0);
    target.draw(Mark::Dot((32, 32), 8), 255);

    target.draw(Mark::Dot((92, 92), 8), 255);

    let int = Integral::<i32>::calculate(&target.full_window());

    // TODO repeat same function for x and y dimension. Then calculate
    // the intersections of empty x and empty y.

    let win = int.0.full_window();
    let iq = IntegralQuad::new(&win, (4, 4), Split::Halves);

    /*for node in petgraph::algo::toposort(&graph, None).unwrap() {
        let tl = (graph[node].rect.0 + 1, graph[node].rect.1 + 1);
        let sz = (graph[node].rect.2 - 1, graph[node].rect.3 - 1);
        println!("{:?}", (tl, sz, graph[node].is_match));
        let color = if graph[node].is_match { 180 } else { 60 };
        target.draw(Mark::Rect(tl, (sz.0-2, sz.1-2)), color);
        // target.show();
    }*/

    // preserve_matching_regions(&mut graph);

    /*for node in graph.node_indices() {
        let tl = (graph[node].rect.0 + 1, graph[node].rect.1 + 1);
        let sz = (graph[node].rect.2 - 1, graph[node].rect.3 - 1);
        println!("{:?}", (tl, sz, graph[node].is_match));
        let color = if graph[node].is_match { 180 } else { 60 };
        target.draw(Mark::Rect(tl, (sz.0-2, sz.1-2)), color);
    }*/

    target.show();

    // Now, to get the largest possible regions, just take
    // either leaves or non-leaf nodes that are true such that all children
    // are also true.

    /*let last_col = target.width()-1;
    search_recursively_not_matched(&mut ranges, &w, 0, w.height()-1, last_col, empty_row_comp, 2);
    for r in &ranges {
        rects.push((w.offset().0 + r.start, w.offset().1, (r.end - r.start)-1, w.size().1));
    }
    ranges.clear();

    // Do the col search only for nonmatched row regions.
    let last_row = target.height()-1;
    search_recursively_not_matched(&mut ranges, &w, 0, w.width()-1, last_row, empty_col_comp, 2);
    for c in &ranges {
        rects.push((w.offset().0, w.offset().1 + c.start, w.size().0, (c.end - c.start)-1));
    }
    for r in rects {
        println!("{:?}", r);
        // target.draw(Mark::Rect((r.0, r.1), (r.2-2, r.3-2)), 255);
        for i in 0..r.2 {
            for j in 0..r.3 {
                target[(r.0+i, r.1+j)] = 127;
            }
        }
    }*/

    // target.show();
}

impl<T> AsRef<ImageBuf<T>> for Integral<T>
where
    T : Pixel + Scalar + Clone + Copy + Any + Zero + From<u8>
{

    fn as_ref(&self) -> &ImageBuf<T> {
        &self.0
    }

}

/*impl<'a, T> AsRef<Window<'a, T>> for Integral<T>
where
    T : Scalar + Clone + Copy + Any + Zero + From<u8>
{

    fn as_ref(&self) -> &Window<'a, T> {
        self.0.as_ref()
    }

}*/

/*impl<'a, T> AsMut<WindowMut<'a, T>> for Integral<T>
where
    T : Scalar + Clone + Copy + Any + Zero + From<u8>
{

    fn as_mut(&mut self) -> &mut WindowMut<'a , T> {
        self.0.as_mut()
    }

}*/

pub struct Accumulated(ImageBuf<i32>);

impl Accumulated {

    pub fn calculate(win : &Window<u8>) -> Self {
        let mut dst = unsafe { Self(ImageBuf::<i32>::new_empty(win.height(), win.width())) };
        dst.update(win);
        dst
    }

    // This sets pixel (i,j) to the sum of all pixels before it. It is
    // calculated using IPP by re-using the IppIntegral, but interpreting
    // the image as a 1D buffer (which is why it requires a reference to an
    // owned buffer - this trick cannot be applied to image views).
    pub fn update(&mut self, w : &Window<u8>) {
        baseline_accumulate(&mut self.0.full_window_mut(), w);
    }

    pub fn update_vectorized(&mut self, w : &Window<i32>) {
        if w.width() % 8 == 0 && self.0.width() == w.width() {
            for (mut d, r) in self.0.full_window_mut().rows_mut().zip(w.rows()) {
                unsafe { vectorized_cumulative_sum(r, d) };
            }
        } else {
            panic!()
        }
    }

}

fn baseline_accumulate(dst : &mut WindowMut<i32>, src : &Window<u8>) {
    assert!(dst.shape() == src.shape());
    let mut s = 0;
    for (mut d, px) in dst.pixels_mut(1).zip(src.pixels(1)) {
        s += *px as i32;
        *d = s;
    }
}

/*impl<'a> AsRef<Window<'a, i32>> for Accumulated
{

    fn as_ref(&self) -> &Window<'a, i32> {
        self.0.as_ref()
    }

}

impl<'a> AsMut<WindowMut<'a, i32>> for Accumulated
{

    fn as_mut(&mut self) -> &mut WindowMut<'a , i32> {
        self.0.as_mut()
    }

}*/

// cargo test -- foo --nocapture
#[test]
fn foo() {

    use std::arch::x86_64::*;

    unsafe {
        let mut s : __m128i = _mm_setzero_si128();
        println!("{:?}", s);
    }
}

// cargo test -- prefix_sum --nocapture
#[test]
fn prefix_sum() {

    let a : [i32; 16] = [1,2,3,4,5,6,7,8,9,10,11,12,13,15,15,16];
    let mut s : [i32; 16] = [0; 16];

    // Never transmute the array pointer like *mut i32 to __mm128 pointer, transmute the dereferenced array to __mm128
    // let mut ptr = s.as_mut_ptr() as *mut [i32; 4];
    // unsafe { println!("{:?}", mem::transmute::<_, [i32; 4]>(int::local_prefix4(mem::transmute(*ptr)))); };
    // Using types from wide, however, pointer cast will work.

    let mut s2 = s.clone();
    unsafe {
        vectorized_cumulative_sum(&a[..], &mut s[..]);
        baseline_cumulative_sum(&a[..], &mut s2[..]);
        // println!("{:?}", s);
        // println!("{:?}", s2);
        assert!(s == s2);
    }
}

// rustc -C target-feature=+sse2
pub mod int {

    use std::arch::x86_64::*;
    use std::mem;

    pub unsafe fn local_prefix4(mut x : __m128i) -> __m128i {
        // x = 1, 2, 3, 4
        x = _mm_add_epi32(x, _mm_slli_si128(x, 4));
        // x = 1, 2, 3, 4
        //   + 0, 1, 2, 3
        //   = 1, 3, 5, 7
        x = _mm_add_epi32(x, _mm_slli_si128(x, 8));
        // x = 1, 3, 5, 7
        //   + 0, 0, 1, 3
        //   = 1, 3, 6, 10
        return x;
    }

    // This does the first step (local sums in 8-lane i32) */
    // #[target_feature(enable="sse2")]
    // #[cfg(target_feature = "avx")]
    pub unsafe fn local_prefix8(p : *mut i32) {
        println!("{:p}", p);
        println!("{:p}", p.offset(7));
        println!("a");
        if !is_x86_feature_detected!( "sse2" ) {
            panic!("SSE2 not detected");
        }

        let mut x : __m256i = _mm256_load_si256(p as *const _);
        println!("done");
        x = _mm256_add_epi32(x, _mm256_slli_si256(x, 4));
        x = _mm256_add_epi32(x, _mm256_slli_si256(x, 8));
        _mm256_store_si256(p as *mut _, x);
        // }
        // unimplemented!()
    }

    // This does the second step.
    unsafe fn accumulate(p : *mut i32, s : __m128i) -> __m128i {
        let d : __m128i = mem::transmute(_mm_broadcast_ss(mem::transmute::<_, &f32>(p.offset(3))));
        let mut x : __m128i = _mm_load_si128(p as *const _);
        x = _mm_add_epi32(s, x);
        _mm_store_si128(p as *mut _, x);
        return _mm_add_epi32(s, d);
    }

    // This calls both steps
    pub unsafe fn prefix(a : &mut [i32]) {
        for i in (0..a.len()).step_by(8) {
            local_prefix8(&mut a[i] as *mut i32);
        }

        let mut s : __m128i = _mm_setzero_si128();
        for i in (4..a.len()).step_by(4) {
            s = accumulate(&mut a[i] as *mut i32, s);
        }
    }
}
/*/*
_mm_shuffle_ps can be used for integer vectors as *pA = _mm_shuffle_epi32(_mm_unpacklo_epi32(*pA, _mm_shuffle_epi32(*pB, 0xe)), 0xd8);
OR
*pA = _mm_blend_epi16(*pA, *pB, 0xf0);
https://stackoverflow.com/questions/26983569/implications-of-using-mm-shuffle-ps-on-integer-vector
*/

use wide::*;

// _mm_castps_si128 cast [f32; 4] into [i32; 4]
// _mm_castsi128_ps cast [i32; 4] into [f32; 4]
// _mm_slli_si128 Shifts first [i32; 4] argment to left by N bytes while shifting in zeros.
fn scan_SSE(mut x : wide::i32x4) -> wide::i32x4 {
    x = x + x.rotate_left(i32x4::splat(4));
    x = x + x.rotate_left(i32x4::splat(8));
    x
}

// local_prefix_wide implements that using simd.
fn cumulative_sum(a : &mut [i32]) {
    let n = a.len();
    for l in 0..(n.log(10)) {
        let m = 1 << l;
        for i in m..n {
            a[i] += a[i - m];
        }
    }
}

unsafe fn local_prefix_wide4(p : &mut [i32; 4]) {
    let mut x = wide::i32x4::from(*p);
    // rotate_lanes_left(4)
    println!("x = {:?}", x);
    let xs4 = x.rotate_left(i32x4::splat(4));
    println!("xs4 = {:?}", xs4);
    x += xs4;
    let xs8 = x.rotate_left(i32x4::splat(8));
    println!("xs8 = {:?}", xs8);
    x += xs8;
    *p = x.into();
}*/

fn baseline_cumulative_sum<T>(a : &[T], dst : &mut [T])
where
    T : Copy + Add<Output=T>
{
    assert!(a.len() == dst.len());
    dst[0] = a[0];
    for i in 1..a.len() {
        dst[i] = dst[i-1] + a[i];
    }
}

/* Based on https://en.algorithmica.org/hpc/algorithms/prefix/
(1) Do local prefix sums within a vector in parallel (each sub-vector has the size of the SIMD lane)
(2) Use SIMD to add the previous vector to the current vector */
unsafe fn vectorized_cumulative_sum(a : &[i32], dst : &mut [i32]) {
    assert!(a.len() == dst.len());
    assert!(a.len() % 8 == 0);
    for i in (0..a.len()).step_by(8) {
        local_prefix_wide8(a[i..].as_ptr() as *const _, dst[i..].as_mut_ptr() as *mut _);
    }

    let mut s = wide::i32x4::ZERO;
    for i in (0..a.len()).step_by(4) {
        s = accumulate_wide(mem::transmute(&mut dst[i]), s);
    }
}

// This does the same thing as scan_SSE before.
unsafe fn local_prefix_wide8(p : *const [i32; 8], s : *mut [i32; 8]) {
    use std::arch::x86_64::*;
    let mut x = wide::i32x8::from(*p);
    x += mem::transmute::<_, wide::i32x8>(_mm256_slli_si256(mem::transmute(x), 4));
    x += mem::transmute::<_, wide::i32x8>(_mm256_slli_si256(mem::transmute(x), 8));
    *s = x.into();
}

unsafe fn accumulate_wide(dst : &mut [i32; 4], s : wide::i32x4) -> wide::i32x4 {
    let d : wide::i32x4 = mem::transmute(_mm_broadcast_ss(mem::transmute::<_, &f32>(&dst[3])));
    let mut x : wide::i32x4 = (*dst).into();
    x += s;
    *dst = x.into();
    s + d
}

/*unsafe fn vectorized_u8_cumulative_sum(a : &[u8], dst : &mut [u8]) {
    assert!(a.len() == dst.len());
    assert!(a.len() % 16 == 0);
    for i in (0..a.len()).step_by(16) {
        local_u8_prefix_wide8(a[i..].as_ptr() as *const _, dst[i..].as_mut_ptr() as *mut _);
    }
    let mut s = wide::u8x16::ZERO;
    for i in (0..a.len()).step_by(16) {
        s = accumulate_u8_wide(mem::transmute(&mut dst[i]), s);
    }
}
// This does the same thing as scan_SSE before.
unsafe fn local_u8_prefix_wide8(p : *const [u8; 16], s : *mut [u8; 16]) {
    use std::arch::x86_64::*;
    let mut x = wide::u8x16::from(*p);
    x += mem::transmute::<_, wide::u8x16>(_mm_slli_si128(mem::transmute(x), 4));
    x += mem::transmute::<_, wide::u8x16>(_mm_slli_si128(mem::transmute(x), 8));
    *s = x.into();
}
unsafe fn accumulate_u8_wide(dst : &mut [u8; 16], s : wide::u8x16) -> wide::u8x16 {
    let d : wide::u8x16 = mem::transmute(_mm_broadcast_ss(mem::transmute::<_, &f32>(&dst[3])));
    let mut x : wide::u8x16 = (*dst).into();
    x += s;
    *dst = x.into();
    s + d
}*/

use std::arch::x86_64::*;

// Mimics the _MM_SHUFFLE(z, y, x, w) macro, which is currently unstable
// https://shybovycha.github.io/2017/02/21/speeding-up-algorithms-with-sse.html
// Given those definitions, the call m3 = _mm_shuffle_ps(m1, m2, _MM_SHUFFLE(z, y, x, w))
// is equal to the formula m3 = (m2(z) << 6) | (m2(y) << 4) | (m1(x) << 2) | m1(w).
const fn mm_shuffle_mask(z : i32, y : i32, x : i32, w : i32) -> i32 {
    (z << 6) | (y << 4) | (x << 2) | w
}

unsafe fn packed_slice<T, const N : usize>(a : &[T]) -> &[[T;N]] {
    let n = a.len();
    assert!(n >= N);
    assert!(n % N == 0);
    std::mem::transmute::<_, &[[T; N]]>(std::slice::from_raw_parts(&a[0] as *const T, n / N))
}

unsafe fn packed_slice_mut<T, const N : usize>(a : &mut [T]) -> &mut [[T; N]] {
    let n = a.len();
    assert!(n >= N);
    assert!(n % N == 0);
    std::mem::transmute::<_, &mut [[T; N]]>(std::slice::from_raw_parts_mut(&mut a[0] as *mut T, n / N))
}

/*// Based on https://stackoverflow.com/questions/19494114/parallel-prefix-cumulative-sum-with-sse
mod sse {

    use core::arch::x86_64::*;

    unsafe fn scan_SSE(mut x : __m128) -> __m128 {
        x = _mm_add_ps(x, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(x), 4)));
        x = _mm_add_ps(x, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(x), 8)));
        x
    }

    // Store the prefix sum of a into the s vector (assumed of same length and length divisible by 4).
    pub unsafe fn prefix_sum_SSE(a : &[f32], s : &mut [f32]) {
        assert!(a.len() == s.len());
        assert!(a.len() % 4 == 0);
        let mut offset : __m128 = _mm_setzero_ps();
        for i in (0..a.len()).step_by(4) {

            // TODO segfault here
            let x : __m128 = _mm_load_ps(&a[i]);

            let out : __m128 = scan_SSE(x);

            let out = _mm_add_ps(out, offset);
            _mm_store_ps(&mut s[i] as *mut _, out);
            println!("here2");
            offset = _mm_shuffle_ps(out, out, super::mm_shuffle_mask(3, 3, 3, 3));
        }
    }

}

mod avx {

    /*use core::arch::x86_64::*;

     fn scan_AVX(x : __m256) -> __m256 {
        let (t0, t1) : (__m256, __m256) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        //shift1_AVX + add
        let t0 = _mm256_permute_ps(x, _MM_SHUFFLE(2, 1, 0, 3));
        let t1 = _mm256_permute2f128_ps(t0, t0, 41);
        let x = _mm256_add_ps(x, _mm256_blend_ps(t0, t1, 0x11));
        //shift2_AVX + add
        let t0 = _mm256_permute_ps(x, _MM_SHUFFLE(1, 0, 3, 2));
        let t1 = _mm256_permute2f128_ps(t0, t0, 41);
        let x = _mm256_add_ps(x, _mm256_blend_ps(t0, t1, 0x33));
        //shift3_AVX + add
        let x = _mm256_add_ps(x,_mm256_permute2f128_ps(x, x, 41));
        x
    }

    void prefix_sum_AVX(float *a, float *s, const int n) {
        let offset : __m256 = _mm256_setzero_ps();
        for (int i = 0; i < n; i += 8) {
            __m256 x = _mm256_loadu_ps(&a[i]);
            __m256 out = scan_AVX(x);
            out = _mm256_add_ps(out, offset);
            _mm256_storeu_ps(&s[i], out);
            //broadcast last element
            __m256 t0 = _mm256_permute2f128_ps(out, out, 0x11);
            offset = _mm256_permute_ps(t0, 0xff);
        }
    }*/

}*/

