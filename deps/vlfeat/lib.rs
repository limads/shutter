#![allow(dead_code)]
#![allow(mutable_transmutes)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(unused_assignments)]
#![allow(unused_mut)]
#![feature(asm)]
#![feature(c_variadic)]
#![feature(extern_types)]
#![feature(label_break_value)]
#![feature(register_tool)]
#![register_tool(c2rust)]

extern crate libc;
pub mod src {
pub mod aib;
pub mod array;
pub mod covdet;
pub mod dsift;
pub mod fisher;
pub mod generic;
pub mod getopt_long;
pub mod gmm;
pub mod hikmeans;
pub mod homkermap;
pub mod host;
pub mod ikmeans;
pub mod imopv;
pub mod kdtree;
pub mod kmeans;
pub mod lbp;
pub mod liop;
pub mod mathop;
pub mod mser;
pub mod pgm;
pub mod quickshift;
pub mod random;
pub mod rodrigues;
pub mod scalespace;
pub mod sift;
pub mod slic;
pub mod stringop;
pub mod svm;
pub mod svmdataset;
pub mod vlad;
} // mod src
