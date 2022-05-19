use crate::image::*;
use std::iter::FromIterator;
use std::mem;

pub struct GrayHistogram([u32; 256]);

impl GrayHistogram {

    pub fn calculate(win : Window<'_, u8>) -> Self {

        // TODO Actually, hist is n_levels - 1 = 255, since position ix means a count of pixels w/ intensity <= px.
        let mut hist : [u32; 256] = [0; 256];

        #[cfg(feature="ipp")]
        unsafe {
            let mut n_levels = 256;
            let n_channels = 1;
            let (step, sz) = crate::image::ipputils::step_and_size_for_window(&win);
            let dtype = crate::foreign::ipp::ippi::IppDataType_ipp8u;
            let uniform_step = 1;
            let mut spec_sz = 0;
            let mut buf_sz = 0;
            let ans = crate::foreign::ipp::ippi::ippiHistogramGetBufferSize(
                dtype,
                sz,
                &n_levels as *const _,
                n_channels,
                uniform_step,
                &mut spec_sz as *mut _,
                &mut buf_sz as *mut _
            );
            assert!(ans == 0);
            let mut spec = Vec::from_iter((0..spec_sz).map(|_| 0u8 ));
            let mut hist_buffer = Vec::from_iter((0..buf_sz).map(|_| 0u8 ));

            let mut lower_lvl : f32 = 0.0;
            let mut upper_lvl : f32 = 255.0;
            let ans = crate::foreign::ipp::ippi::ippiHistogramUniformInit(
                dtype,
                &mut lower_lvl as *mut _,
                &mut upper_lvl as *mut _,
                &mut n_levels as *mut _,
                n_channels,
                mem::transmute(spec.as_mut_ptr())
            );
            assert!(ans == 0);

            let ans = crate::foreign::ipp::ippi::ippiHistogram_8u_C1R(
                win.as_ptr(),
                step,
                sz,
                hist.as_mut_ptr(),
                mem::transmute(spec.as_ptr()),
                hist_buffer.as_mut_ptr()
            );
            assert!(ans == 0);

            // let ans = crate::foreign::ipp::ippi::ippiHistogramGetLevels(spec.as_ptr(), )

            return Self(hist)
        }

        unimplemented!()
    }

}


