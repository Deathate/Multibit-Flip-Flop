#![allow(clippy::wildcard_imports)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::default_trait_access)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::similar_names)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::struct_field_names)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::format_push_string)]
#![allow(clippy::missing_fields_in_debug)]

mod geometry;

pub mod util;
pub use util::*;

mod rtree;
use rtree::*;

mod bridge_py;
pub use bridge_py::*;

use vec_util::*;

mod file_writer;

mod helper;

use file_writer::*;

mod custom_trait;
use custom_trait::*;

mod class;
pub use class::*;

mod mbffg;
pub use mbffg::*;
