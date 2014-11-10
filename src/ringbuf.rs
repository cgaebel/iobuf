use collections::slice::SlicePrelude;
use core::mem;
use core::str::StrPrelude;

use iobuf::Iobuf;
use impls::{RWIobuf, ROIobuf};

/// A ring buffer implemented with `Iobuf`s.
pub struct IORingbuf {
  /// The contents of the window is space for input to be put into. Therefore,
  /// initially full.
  i_buf: RWIobuf<'static>,
  /// The contents of the window are things needing to be output. Therefore,
  /// initally empty.
  o_buf: RWIobuf<'static>,
}

impl IORingbuf {
  /// Creates a new ring buffer, with room for `cap` bytes.
  pub fn new(cap: uint) -> IORingbuf {
    let left_size = cap / 2;
    let mut ret =
      IORingbuf {
        i_buf: RWIobuf::new(left_size),
        o_buf: RWIobuf::new(cap - left_size),
      };
    ret.o_buf.flip_lo(); // start with an empty o_buf.
    ret
  }

  /// Returns an Iobuf, whose window may be filled with new data. This acts as
  /// the "push" operations for the ringbuf.
  ///
  /// It is easy to get garbage data if using a clone of the returned Iobuf.
  /// This is not memory-unsafe, but should be avoided.
  #[inline(always)]
  pub fn push_buf(&mut self) -> &mut RWIobuf<'static> {
    &mut self.i_buf
  }

  /// Returns an Iobuf, whose window may be have data `consume`d out of it. This
  /// acts as the "pop" operation for the ringbuf.
  ///
  /// After emptying out the returned Iobuf, it is not necessarily true that
  /// the ringbuf is empty. To truly empty out the ringbuf, you must pop
  /// Iobufs in a loop until `is_empty` returns `true`.
  ///
  /// It is easy to get garbage data if using a clone of the returned Iobuf.
  /// This is not memory-unsafe, but should be avoided.
  #[inline]
  pub fn pop_buf(&mut self) -> &mut ROIobuf<'static> {
    if self.o_buf.is_empty() {
      self.i_buf.flip_lo();
      self.o_buf.reset();
      mem::swap(&mut self.i_buf, &mut self.o_buf);
    }
    // Clients should only be doing read-only operations into the iobuf, so
    // return a ROIobuf.
    unsafe { mem::transmute(&mut self.o_buf) }
  }

  /// `true` if there is no data to pop in the Iobuf.
  #[inline]
  pub fn is_empty(&self) -> bool {
    self.i_buf.cap() == self.i_buf.len() && self.o_buf.is_empty()
  }

  /// `true` if there is no room for new data in the Iobuf.
  #[inline(always)]
  pub fn is_full(&self) -> bool {
    self.i_buf.is_empty()
  }
}
