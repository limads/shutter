/*
Calculates image profiles (1D signals over rows and columns).
*/

// use std::convert::AsRef;
use crate::image::*;

#[derive(Debug, Clone)]
pub struct Profile {
    pub row_sum : Vec<i32>,
    pub col_sum : Vec<i32>
}

impl Profile {

    pub fn calculate<S>(w : &Image<u8, S>) -> Self
    where
        S : Storage<u8>
    {
        let mut row_sum = Vec::new();
        let mut col_sum = Vec::new();

        /*for r in 0..w.as_ref().height() {
            row_sum.push(crate::global::sum::<_, f64>(&w.as_ref().sub_window((r, 0), (1, w.as_ref().width())).unwrap(), 1) as u32);
        }
        for c in 0..w.as_ref().width() {
            col_sum.push(crate::global::sum::<_, f64>(&w.as_ref().sub_window((0, c), (w.as_ref().height(), 1)).unwrap(), 1) as u32);
        }*/
        for r in 0..w.height() {
            row_sum.push(w.window((r, 0), (1, w.width())).unwrap().accum::<i32>());
        }
        for c in 0..w.width() {
            col_sum.push(w.window((0, c), (w.height(), 1)).unwrap().accum::<i32>());
        }

        Self { row_sum, col_sum }
    }

}


