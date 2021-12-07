#![doc(html_logo_url = "https://raw.githubusercontent.com/limads/shutter/master/assets/logo.png")]
#![allow(unused)]

pub mod io;

// pub(crate) mod foreign;

// pub mod histogram;

// Scalar image operations
pub mod scalar;

// Binary image operations
pub mod binary;

// Low-level image features
pub mod feature;

pub mod image;

pub mod path;

// #[cfg(feature="opencvlib")]
// pub mod tracking;

// #[cfg(feature="opencvlib")]
// pub mod template;

// #[cfg(feature="opencvlib")]
// pub mod matching;

// #[cfg(feature="opencvlib")]
// pub mod flow;

#[cfg(feature="opencvlib")]
pub mod contour;

// pub mod edge;

// pub mod shape;

pub mod threshold;

// pub mod feature;

// pub mod filter;

// Defines operations on binary images.
#[cfg(feature="opencvlib")]
pub mod morphology;

#[cfg(feature="opencvlib")]
pub mod filter;

pub mod cluster;

// pub mod segmentation;

#[cfg(feature="opencvlib")]
pub mod geom;

// #[cfg(feature="opencvlib")]
// pub mod detection;

pub mod foreign;

#[cfg(feature="processor")]
pub mod processor;
