use super::*;

/// Represents a 3x3 image region pattern.
pub enum Pattern {

    // A across where foreground matches the internal horizontal and vertical row,
    // and background matches the corners.
    Cross,

    // A full square matching the foreground. Background matches nothing.
    Square,

    // Matches only center, and none of the other neighboring pixels
    Dot,

    // VerticalBar

    // HorizontalBar

    // Matches center and corner points.
    // X
}

fn cross_match<F, B>(win : &Window<'_, u8>, scale : u16, fg : F, bg : B) -> bool
where
    F : Fn(u8)->bool + Copy + Clone,
    B : Fn(u8)->bool + Copy + Clone
{
    for i in (0..(win.height())).step_by(scale as usize) {
        if !fg(win[(i, win.width()/2)]) {
            return false;
        }
    }

    for j in (0..(win.width())).step_by(scale as usize) {
        if !fg(win[(win.height()/2, j)]) {
            return false;
        }
    }

    for i in [0, win.height()-1] {
        for j in [0, win.width()-1] {
            if !bg(win[(i, j)]) {
                return false;
            }
        }
    }

    true
}

fn dot_match<F, B>(win : &Window<'_, u8>, scale : u16, fg : F, bg : B) -> bool
where
    F : Fn(u8)->bool + Copy + Clone,
    B : Fn(u8)->bool + Copy + Clone
{
    if !fg(win[(win.height()/2, win.width()/2)]) {
        return false;
    }
    for j in (0..win.width()).step_by(scale as usize) {
        if !bg(win[(0, j)]) {
            return false;
        }
    }
    for j in (0..win.width()).step_by(scale as usize) {
        if !bg(win[(win.height()-1, j)]) {
            return false;
        }
    }
    if !bg(win[(win.height()/2, 0)]) || !bg(win[(win.height()/2, win.width()-1)]) {
        return false;
    }

    true
}

fn square_match<F, B>(win : &Window<'_, u8>, scale : u16, fg : F, bg : B) -> bool
where
    F : Fn(u8)->bool + Copy + Clone,
    B : Fn(u8)->bool + Copy + Clone
{
    for px in win.pixels(scale as usize){
        if !fg(*px) {
            return false;
        }
    }
    true
}

fn cross_patch(i : u16, j : u16, scale : u16, win : &Window<'_, u8>) -> Patch {
    let mut pxs = Vec::new();
    for pi in ((i-1)*scale)..(i+1)*scale {
        pxs.push((pi, j));
    }
    for pj in ((j-1)*scale)..(j+1)*scale {
        pxs.push((i, pj));
    }
    Patch {
        pxs,
        outer_rect : ((i-1)*scale as u16, (j-1)*scale as u16, (3*scale) as u16, (3*scale) as u16),
        color : 0,
        scale : scale as u16,
        img_height : win.height(),
        area : 5
    }
}

fn dot_patch(i : u16, j : u16, scale : u16, win : &Window<'_, u8>) -> Patch {
    Patch {
        pxs : vec![(i*scale, j*scale)],
        outer_rect : ((i-1)*scale as u16, (j-1)*scale as u16, scale as u16, scale as u16),
        color : 0,
        scale : scale as u16,
        img_height : win.height(),
        area : 1
    }
}

fn square_patch(i : u16, j : u16, scale : u16, win : &Window<'_, u8>) -> Patch {
    let mut pxs = Vec::new();
    for pi in ((i-1)*scale)..(i+1)*scale {
        for pj in ((j-1)*scale)..(j+1)*scale {
            pxs.push((pi, pj));
        }
    }
    Patch {
        pxs,
        outer_rect : ((i-1)*scale as u16, (j-1)*scale as u16, (3*scale) as u16, (3*scale) as u16),
        color : 0,
        scale : scale as u16,
        img_height : win.height(),
        area : 9
    }
}

pub struct PatternSegmenter {

}

impl PatternSegmenter {

    pub fn new() -> Self {
        Self { }
    }

    /* Segments the image, from a foreground pixel comparison function and a
    background pixel comparison function. Returns patches for regions where a
    match happens. */
    pub fn segment<F, B>(
        &mut self,
        win : &Window<'_, u8>,
        pattern : Pattern,
        fg : F,
        bg : B,
        scale : u16
    ) -> Vec<Patch>
    where
        F : Fn(u8)->bool + Copy + Clone,
        B : Fn(u8)->bool + Copy + Clone
    {
        let mut patches : Vec<Patch> = Vec::new();
        let match_func = match pattern {
            Pattern::Cross => cross_match::<F, B>,
            Pattern::Square => square_match::<F, B>,
            Pattern::Dot => dot_match::<F, B>
        };
        let patch_func = match pattern {
            Pattern::Cross => cross_patch,
            Pattern::Square => square_patch,
            Pattern::Dot => dot_patch
        };
        for (i, j, px) in win.labeled_pixels::<usize, _>(scale as usize) {
            if i > 0 && j > 0 && i < (win.height() / scale as usize) - 1 && j < (win.width() / scale as usize) - 1 {
                let sub_win = win.sub_window(
                    ((i as usize-1)*scale as usize, (j as usize-1)*scale as usize),
                    ((3*scale) as usize, (3*scale) as usize)
                ).unwrap();
                if match_func(&sub_win, scale, fg, bg) {
                    let new_patch = patch_func(i as u16, j as u16, scale, win);
                    let overlaps_last = if let Some(last) = patches.last() {
                        crate::shape::rect_overlaps(&last.outer_rect, &new_patch.outer_rect)
                    } else {
                        false
                    };
                    // if !overlaps_last {
                    patches.push(new_patch);
                    // }
                }
            }
        }

        // TODO join neighboring patches if they share a full matching side.

        patches
    }

}

