#![allow(warnings)]

#![doc(html_logo_url = "https://raw.githubusercontent.com/limads/shutter/master/assets/logo.png")]

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

#[cfg(feature="opencv")]
pub mod contour;

// pub mod edge;

// pub mod shape;

pub mod threshold;

// pub mod feature;

// pub mod filter;

// Defines operations on binary images.
#[cfg(feature="opencv")]
pub mod morphology;

#[cfg(feature="opencv")]
pub mod filter;

pub mod cluster;

// pub mod segmentation;

#[cfg(feature="opencv")]
pub mod geom;

// #[cfg(feature="opencvlib")]
// pub mod detection;

pub mod foreign;

#[cfg(feature="gstreamer")]
#[cfg(feature="gstreamer-base")]
#[cfg(feature="gstreamer-video")]
pub mod processor;

#[cfg(feature="mlua")]
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
}


