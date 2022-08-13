use std::env;
// use std::fs;
// use std::path::Path;

// Moved to volta. Perhaps just link to ippi here.
/*fn try_link_mkl() {
    if let Ok(_) = env::var("CARGO_FEATURE_MKL") {
        println!("cargo:rustc-link-lib=mkl_intel_lp64");
        println!("cargo:rustc-link-lib=mkl_intel_thread");
        println!("cargo:rustc-link-lib=mkl_core");
        println!("cargo:rustc-link-lib=pthread");
        println!("cargo:rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=iomp5");
    }

}*/

fn try_link_ippi() {
    // NOTE: /opt/intel/oneapi/ipp/latest/lib/intel64 should be in LD_LIBRARY_PATH, because the DLLs
    // will search be searched for the linked code as well (not only by cargo).
    if let Ok(_) = env::var("CARGO_FEATURE_IPP") {
        println!("cargo:rustc-link-search=/opt/intel/oneapi/ipp/latest/lib/intel64");
        println!("cargo:rustc-link-lib=ippcore");
        println!("cargo:rustc-link-lib=ippvm");
        println!("cargo:rustc-link-lib=ipps");
        println!("cargo:rustc-link-lib=ippi");
        println!("cargo:rustc-link-lib=ippcv");
    }
}

/*fn try_link_gsl() {
    if let Ok(_) = env::var("CARGO_FEATURE_GSL") {
        println!("cargo:rustc-link-lib=gsl");
        println!("cargo:rustc-link-lib=gslcblas");
        println!("cargo:rustc-link-lib=m");
    }
}*/

fn _try_link_vlfeat() {
    // if let Ok(_) = env::var("CARGO_FEATURE_VLFEATLIB") {
    //    println!("cargo:rustc-link-lib=vl");
    // }
}

fn main() {
    // try_link_vlfeat();
    // try_link_gsl();
    // try_link_mkl();
    try_link_ippi();
}
