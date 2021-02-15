use opencv::core::*;
use opencv::prelude::Tracker;
use opencv::tracking::TrackerMOSSE;
use std::ffi;
use crate::image::cvutils;

pub struct CVTracker {

    // Also see opencv::tracking::{TrackerMIL, TrackerGOTURN}; BUT GOTURN is RGB-based
    mosse : Ptr<dyn TrackerMOSSE>,
    frame : usize,
    img_dim : (usize, usize),
    win_dim : (usize, usize)
}

impl CVTracker {

    pub fn new(img_dim : (usize, usize), win_dim : (usize, usize)) -> Self {
        Self{ 
            mosse : TrackerMOSSE::create().unwrap(),  
            img_dim, 
            win_dim,
            frame : 0
        }
    }
    
    pub fn reposition(&mut self, buf : &mut [u8], pos : (usize, usize)) -> Result<(), String> {
        let rect = Rect2d {
            x : pos.1 as f64, 
            y : pos.0 as f64, 
            width : self.win_dim.1 as f64, 
            height : self.win_dim.0 as f64
        };
        let mat = unsafe { cvutils::slice_to_mat(buf, self.img_dim.1, None) };
        self.mosse = TrackerMOSSE::create().unwrap();  
        let ans = self.mosse.init(
            &mat, 
            rect
        ).unwrap();
        assert!(ans);
        self.frame = 0;
        Ok(())
    }
    
    pub fn update(&mut self, buf : &mut [u8]) -> Option<(usize, usize)> {
        let mut rect = Rect2d{ x : 0.0, y : 0.0, width : 0.0, height : 0.0 };
        self.frame += 1;
        let mat = unsafe { cvutils::slice_to_mat(buf, self.img_dim.1, None) };
        let ans = self.mosse.update(
            &mat,
            &mut rect
        ).unwrap();
        if ans {
            Some(((rect.y as usize), rect.x as usize))
        } else {
            None
        }
    }
    
}


