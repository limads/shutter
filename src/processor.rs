use gstreamer as gst;
use gstreamer_base as gst_base;
use anyhow::Error;
use once_cell::sync::Lazy;
use std::cell::RefCell;
use gstreamer_video;
use gstreamer::{FlowError, FlowSuccess};
use gstreamer_base::BaseTransform;
use glib::subclass::TypeData;
use std::ptr::NonNull;
use gstreamer_base::BaseTransformExt;
use glib;
use glib::Cast;
use std::mem;
use glib::ObjectExt;
use gstreamer_video::VideoInfo;

// Based on the subclass example at https://github.com/sdroege/gstreamer-rs/blob/master/examples/src/bin/subclass.rs

/// This will be exported as ObjectRef, which is a wrapper around
/// ptr::NonNull<GObject>. GObject only implements new(), with_type() and new_generic().
glib::wrapper! {
    pub struct Processor(ObjectSubclass<imp::Processor>) @extends gst_base::BaseTransform, gst::Element, gst::Object;
}

/*const fn func_spec() -> glib::ParamSpec {
    glib::ParamSpec::uint64(
        "func",
        "func",
        "func",
        0,
        u64::max_value(),
        0,
        glib::ParamFlags::READWRITE,
    )
}

static PROPERTIES : [glib::ParamSpec; 1] = [func_spec()];*/

// let proc = Processor::new(myfunc)
// Where myfunc has in its body the retrieval of a static reference: OnceCell<RefCell<EyeTracker>>;
impl Processor {

    /// Returns a Gstreamer BaseTransform implementor which calls an arbitrary function pointer.
    /// If you need to preserve some state T across calls, consider declaring a one-time initialization static cell:
    /// static MY_STATE : OnceCell<Mutex<T>> = OnceCell::new();
    /// Which is used inside your function.
    pub fn create(f : fn(&mut[u8]) -> bool) -> Self {
        let proc : Self = glib::Object::new(&[("name", &Some("Processor"))]).unwrap();
        proc.set_property("func", &(f as u64)).unwrap();
        proc
    }
    
}

// The debug category we use below for our filter
pub static CAT: Lazy<gst::DebugCategory> = Lazy::new(|| {
    gst::DebugCategory::new(
        "Processor",
        gst::DebugColorFlags::empty(),
        Some("Processor"),
    )
});

mod imp {

    use super::*;
    use glib::{subclass::{self, object}};
    use gstreamer_base::subclass::prelude::BaseTransformImpl;
    use gstreamer::subclass::prelude::ElementImpl;
    use glib::subclass::prelude::ObjectSubclass;
    
    pub struct Processor{ processor : RefCell<Option<fn(&mut[u8])->bool>> }
    
    // This trait registers our type with the GObject object system and
    // provides the entry points for creating a new instance and setting
    // up the class data
    impl ObjectSubclass for Processor {
        const NAME: &'static str = "Processor";
        type Type = super::Processor;
        type ParentType = gst_base::BaseTransform;
        type Interfaces = ();
        type Instance = gst::subclass::ElementInstanceStruct<Self>;
        type Class = subclass::simple::ClassStruct<Self>;

        glib::object_subclass!();

        // Either implement new if no args are required; or with_class if args are required.
        fn new() -> Self {
            // Not really required, since we are instantiating our object from
            // Rust. We should implement this if we were to use some sort of FFI
            // that knows about our struct only via the GObject API.
            // let func = |data : &mut[u8]| -> bool { true };
            Self { processor : RefCell::new(None) }
        }
        
    }

    // Implementation of glib::Object virtual methods
    impl object::ObjectImpl for Processor {
    
        fn properties() -> &'static [glib::ParamSpec] {
            use once_cell::sync::Lazy;
            static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
                vec![glib::ParamSpec::uint64(
                    "func",
                    "func",
                    "func",
                    0,
                    u64::max_value(),
                    0,
                    glib::ParamFlags::READWRITE,
                )]
            });
            PROPERTIES.as_ref()
        }
        
        fn set_property(&self, obj: &Self::Type, id: usize, value: &glib::Value, pspec : &glib::ParamSpec) {
            // let prop = &PROPERTIES[id];
            match pspec.get_name() {
                "func" => {
                    // println!("setting calcfunc now");
                    if let Ok(val) = value.get::<u64>() {
                        // println!("cast from u64 ok!");
                        if let Some(val) = val {
                            unsafe {
                                // let fptr = mem::transmute::<u64, FloatMatrixFn>(val);
                                // let bx_ptr = Box::new(fptr);
                                // self.bind_func(bx_ptr);
                                // Perhaps just clone it here for safety.
                                if val == 0 {
                                    panic!("Invalid function pointer");
                                }
                                
                                let fn_ptr = mem::transmute::<u64, fn(&mut[u8])->bool>(val);
                                //if fn_ptr.is_null() {
                                //    panic!("Pointer to informed box is null");
                                //}
                                *(self.processor.borrow_mut()) = Some(fn_ptr);
                            }
                        } else {
                            println!("Could not get value");
                        }
                    } else {
                        println!("Invalid cast to u64");
                    }
                },
                _ => {
                    println!("Invalid property");
                }
            }
        }
        
    }

    // Implementation of gst::Element virtual methods
    impl ElementImpl for Processor {
        fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
            static ELEMENT_METADATA: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
                gst::subclass::ElementMetadata::new(
                    "General-purpose buffer byte processor",
                    "Filter/Video",
                    "General-purpose byte processor",
                    "Diego Lima",
                )
            });
            Some(&*ELEMENT_METADATA)
        }
        
        fn pad_templates() -> &'static [gst::PadTemplate] {
            static PAD_TEMPLATES: Lazy<Vec<gst::PadTemplate>> = Lazy::new(|| {
                // Create pad templates for our sink and source pad. These are later used for
                // actually creating the pads and beforehand already provide information to
                // GStreamer about all possible pads that could exist for this type.

                // On both of pads we can only handle F32 mono at any sample rate.
                let caps = gst::Caps::new_simple("video/x-raw",&[
                    ("format", &gstreamer_video::VideoFormat::Gray8.to_str()),
                    ("width", &gstreamer::IntRange::<i32>::new(0, 3000)),
                    ("height", &gstreamer::IntRange::<i32>::new(0, 3000)),
                    ("framerate", &gstreamer::FractionRange::new(
                        gstreamer::Fraction::new(0, 1), gstreamer::Fraction::new(3000, 1),),
                    ),],);

                vec![
                    // The src pad template must be named "src" for basetransform
                    // and specific a pad that is always there
                    gst::PadTemplate::new(
                        "src",
                        gst::PadDirection::Src,
                        gst::PadPresence::Always,
                        &caps,
                    )
                    .unwrap(),
                    // The sink pad template must be named "sink" for basetransform
                    // and specific a pad that is always there
                    gst::PadTemplate::new(
                        "sink",
                        gst::PadDirection::Sink,
                        gst::PadPresence::Always,
                        &caps,
                    )
                    .unwrap(),
                ]
            });
            PAD_TEMPLATES.as_ref()
        }

    }

    // Implementation of gst_base::BaseTransform virtual methods
    impl BaseTransformImpl for Processor {
        
        const MODE: gst_base::subclass::BaseTransformMode =
            gst_base::subclass::BaseTransformMode::AlwaysInPlace;
            
        const PASSTHROUGH_ON_SAME_CAPS: bool = false;
        
        const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

        // Consider implementing those two here
        /*fn get_unit_size(&self, _element: &Self::Type, caps: &gst::Caps) -> Option<usize> {
            let video_info = VideoInfo::from_caps(caps).ok();
            video_info.map(|info| info.size() as usize)
        }*/
        /*fn stop(&self, element: &Self::Type) -> Result<(), gst::ErrorMessage> {
        
            // Drop any required preserved state
            
            // self.history.lock().unwrap().clear();
            // gst_info!(CAT, obj: element, "Stopped");
            Ok(())
        }*/
        
        fn transform_ip(
            &self,
            element: &Self::Type,
            buf: &mut gst::BufferRef,
        ) -> Result<gst::FlowSuccess, gst::FlowError> {
            if let Ok(mut buf_8bit) = buf.map_writable() {
                if let Some(processor) = *self.processor.borrow() {
                    match (processor)(buf_8bit.as_mut_slice()) {
                        true => Ok(FlowSuccess::Ok),
                        false => Err(FlowError::Error)    
                    }
                } else {
                    println!("No function bound to processor");
                    Err(FlowError::Error)    
                }
            } else {
                println!("Could not map into a writable buffer");
                Err(FlowError::Error)
            }
        }
    }

    unsafe impl Send for Processor {}

    unsafe impl Sync for Processor {}

}




