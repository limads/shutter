#![doc(html_logo_url = "https://raw.githubusercontent.com/limads/shutter/master/assets/logo.png")]
#![allow(unused)]

// pub(crate) mod foreign;

pub mod image;

pub mod path;

//#[cfg(feature="opencvlib")]
//pub mod tracking;

#[cfg(feature="opencvlib")]
pub mod template;

#[cfg(feature="opencvlib")]
pub mod matching;

#[cfg(feature="opencvlib")]
pub mod flow;

#[cfg(feature="opencvlib")]
pub mod contour;

#[cfg(feature="opencvlib")]
pub mod shape;

#[cfg(feature="opencvlib")]
pub mod threshold;

#[cfg(feature="opencvlib")]
pub mod morphology;

#[cfg(feature="opencvlib")]
pub mod filter;

pub mod cluster;

#[cfg(feature="vlfeat")]
pub mod segmentation;

#[cfg(feature="opencvlib")]
pub mod geom;

// #[cfg(feature="opencvlib")]
// pub mod detection;

pub mod foreign;

#[cfg(feature="processor")]
pub mod processor;
