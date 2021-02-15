#![doc(html_logo_url = "file:///home/diego/Software/shutter/assets/logo.png")]

#![allow(unused)]

// pub(crate) mod foreign;

pub mod image;

#[cfg(feature="opencvlib")]
pub mod tracking;

#[cfg(feature="opencvlib")]
pub mod template;

#[cfg(feature="opencvlib")]
pub mod matching;

#[cfg(feature="opencvlib")]
pub mod flow;

#[cfg(feature="opencvlib")]
pub mod threshold;


