#![allow(warnings)]

#![doc(html_logo_url = "https://raw.githubusercontent.com/limads/shutter/master/assets/logo.png")]

/// Common optical manipulations (calculation of focal depth, magnification, camera calibration, lens distortion correction),
pub mod optics;

pub mod draw;

pub mod warp;

pub mod edge;

pub mod convert;

pub mod resample;

pub mod motion;

pub mod stereo;

pub mod object;

pub mod graph;

/// Structures and algorithms for sparse binary image representation (in terms of graphs, ordered arrays, etc).
/// Offers Run-length encoding of binary image representation, and other alternative representations.
pub mod sparse;

// Contains algorithms to partition the images over homogeneous regions, with a criterion for
// homegeneity that is algorithm-specific. Contains dense and sparse data structures to represent
// image regions.
pub mod region;

// Binary image operations
pub mod binary;

// Pattern and texture analysis operations (segmentation, classification)
pub mod pattern;

/// Image-to-image grayscale operations (thresholding, segmentation)
pub mod gray;

/// Color image operations
pub mod color;

/// Point-wise operations (enhance, equalize, normalize, etc).
pub mod point;

/// Global operations (mean, max, avg)
pub mod global;

pub mod hist;

/// Local (non-filtering) operations (median, min, max)
pub mod local;

pub mod ffi;

pub mod io;

// pub(crate) mod foreign;

pub mod profile;

// Scalar image operations
pub mod scalar;

// Low-level image features
pub mod feature;

pub mod image;

pub mod path;

pub mod raster;

// #[cfg(feature="opencvlib")]
// pub mod tracking;

pub mod template;

// #[cfg(feature="opencvlib")]
// pub mod matching;

// #[cfg(feature="opencvlib")]
// pub mod flow;

#[cfg(feature="opencv")]
pub mod contour;

pub mod corner;

pub mod integral;

// pub mod edge;

// pub mod shape;

// pub mod threshold;

// pub mod feature;

// pub mod filter;

// Defines operations on binary images.
#[cfg(feature="opencv")]
pub mod morph;

// #[cfg(feature="opencv")]
// pub mod filter;

pub mod cluster;

// pub mod segmentation;

#[cfg(feature="opencv")]
pub mod geom;

// #[cfg(feature="opencvlib")]
// pub mod detection;

pub mod foreign;

#[cfg(feature="glib")]
#[cfg(feature="once_cell")]
#[cfg(feature="gstreamer")]
#[cfg(feature="gstreamer-base")]
#[cfg(feature="gstreamer-video")]
pub mod processor;

pub mod prelude {

    pub use super::image::*;

    pub use super::raster::*;

    pub use super::local::*;

    pub use super::convert::*;

    pub use super::resample::*;

    pub use super::draw::*;

}

/*#[cfg(feature="mlua")]
impl mlua::UserData for crate::image::Image<u8> {

    fn add_methods<'lua, M: mlua::UserDataMethods<'lua, Self>>(methods: &mut M) {

        use crate::image::*;

        methods.add_method("show", |_ : &mlua::Lua, this : &Image<u8>, _: ()| {
            this.show();
            Ok(())
        });

    }

}

#[cfg(feature="mlua")]
#[mlua::lua_module]
fn libshutter(lua : &mlua::Lua) -> mlua::Result<mlua::Table> {

    use mlua::{Table, Lua};
    use crate::image::*;

    let exports = lua.create_table()?;

    exports.set("open", lua.create_function(|_ : &Lua, path : String|->mlua::Result<Image<u8>> {
        let img = crate::io::decode_from_file(&path)
            .map_err(|e| mlua::Error::RuntimeError(format!("{}",e)) )?;
        Ok(img)
    })?)?;

    Ok(exports)
}*/


