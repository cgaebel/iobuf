//! A contiguous region of bytes, useful for I/O operations.
//!
//! An Iobuf consists of:
//!
//!   - a buffer
//!   - limits   (a subrange of the buffer)
//!   - a window (a subrange of the limits)
//!
//! All iobuf operations are restricted to operate within the limits. Initially,
//! the window of an Iobuf is identical to the limits. If you have an `&mut` to
//! an Iobuf, you may change the window and limits. If you only have a `&`, you
//! may not. Similarly, if you have a `RWIobuf`, you may modify the data in the
//! buffer. If you have a `ROIobuf`, you may not.
//!
//! The limits can be `narrow`ed, but never widened. The window may be set to
//! any arbitrary subrange of the limits.
//!
//! Iobufs are cheap to `clone`, since the buffers are refcounted. Use this to
//! construct multiple views into the same data.
//!
//! To keep the struct small (24 bytes!), the maximum size of an Iobuf is 2 GB.
//! Please let me know if you need more than this. I have never before seen a
//! use case for Iobufs larger than 2 GB, and it gives us a 40% smaller struct
//! compared to supporting INT64_MAX bytes. The improved size allows much better
//! register/cache usage and faster moves, both of which are critical for
//! performance.
//!
//! Although this library is designed for efficiency, and hence gives you lots
//! of ways to omit bounds checks, that does not mean it's recommended you do.
//! They merely provide a way for you to manually bounds check a whole bunch
//! of data at once (with `check_range`), and then `peek` or `poke` out the data
//! you want. See the documentation for `check_range` for an example.
//!
//! To repeat: Do not omit bounds checks unless you've checked _very_ carefully
//! that they are redundant. This can cause terrifying security issues.

#![doc(html_root_url = "http://www.rust-ci.org/cgaebel/iobuf/doc/iobuf/")]

#![feature(phase)]
#![feature(unsafe_destructor)]

#![deny(missing_docs)]

#![no_std]

                                   extern crate alloc;
                                   extern crate collections;
             #[phase(plugin,link)] extern crate core;

// For tests only:
#[cfg(test)] #[phase(plugin,link)] extern crate std;
#[cfg(test)]                       extern crate test;

// NOTE: libcollections is only needed for `BufSpan`. If you don't need that
// functionality, feel free to hack this library to remove that file, and
// therefore the dependency.
//
// Also, when you do that, file an issue letting me know you had to do that.
// I'll use that as a cue to break out `BufSpan` into its own library.

pub use raw::{Prim, Allocator};
pub use iobuf::Iobuf;
pub use impls::{RWIobuf, ROIobuf, AROIobuf};
pub use ringbuf::IORingbuf;
pub use bufspan::{BufSpan, ByteIter, SpanIter, SpanMoveIter};

// https://github.com/rust-lang/rust/issues/18491#issuecomment-61293267
#[cfg(not(test))]
mod std { pub use core::fmt; }

mod raw;
mod iobuf;
mod impls;
mod ringbuf;
mod bufspan;
