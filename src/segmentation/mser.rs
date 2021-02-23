use std::ptr;

/// Maximally stable external regions
pub struct MSER {
    filt : *mut VLMserFilt
}

impl MSER {

    pub fn new(dims : (usize, usize)) -> Self {
        // TODO confirm row/col order
        let dims = [dims.0 as i32, dims.1 as i32];
        let mser = vl_mser_new (2, dims.as_ptr());    
    }
    
    fn process(&mut self, img : &[u8]) {
        vl_mser_process(self.filt, img.as_ptr())
    }
    
    fn regions(&mut self) {
        let regions : *const u32 = vl_mser_get_regions(self.filt as *const _);
    }
    
    fn ellipsis(&mut self) {
        vl_mser_ell_fit(self.filt);
        let ellipsis : *const f32 = vl_mser_get_ell(self.filt as *const _);	
    }
    
    fn statistics(&self) {
        let stats = vl_mser_get_stats(self.filt as *const _);
        /* Struct has the fields (all of which are i32):
        num_extremal
        num_abs_unstable 
        num_too_big
        num_too_small
        num_duplicates
        num_unstable*/
    }
    
}

impl Drop for MSER {
    fn drop(&mut self) {
        vl_mser_delete(self.filt);
    }
}

