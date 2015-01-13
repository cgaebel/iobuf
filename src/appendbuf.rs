use std::fmt::{self, Show, Formatter};
use std::mem;
use std::num::Int;
use std::sync::Arc;

use raw::{Allocator, RawIobuf};
use iobuf::{Iobuf};
use impls::{AROIobuf};

/// append-only input buffer
///
/// This buffer is intended to act as a intermediate buffer to be `fill`ed by
/// incoming streaming data, such as sockets, files, or perhaps DB results
///
/// It has the unique feature of being able to break off reference counted
/// slices of data in its buffer that it has already written.
///
/// This invariant is enforced by allowing slices to be taken only from the
/// low side of the buffer, before the start of the window
///
/// Its primary interface is `fill`, which is the mechanism for appending data,
/// and atomic_slice, which will take a position
/// and a length and return a Result<AROIobuf, ()>
///
#[unsafe_no_drop_flag]
pub struct AppendBuf<'a> {
   raw: RawIobuf<'a>,
}

impl<'a> AppendBuf<'a> {
    /// Constructs a trivially empty Iobuf, limits and window are 0, and there's
    /// an empty backing buffer. This will not allocate.
    ///
    /// ```rust
    /// use iobuf::{AppendBuf,Iobuf};
    ///
    /// let mut b = AppendBuf::empty();
    ///
    /// assert_eq!(b.len(), 0);
    /// assert_eq!(b.cap(), 0);
    /// ```
    #[inline(always)]
    pub fn empty() -> AppendBuf<'static> {
        AppendBuf { raw: RawIobuf::empty() }
    }

    /// Constructs a new Iobuf with a buffer of size `len`, undefined contents,
    /// and the limits and window set to the full size of the buffer.
    ///
    /// The maximum length of an Iobuf is approximately 2 GB.
    ///
    /// ```rust
    /// use iobuf::{AppendBuf,Iobuf};
    ///
    /// let mut b = AppendBuf::new(10);
    ///
    /// assert_eq!(b.len(), 10);
    /// assert_eq!(b.cap(), 10);
    /// ```
    #[inline(always)]
    pub fn new(len: usize) -> AppendBuf<'static> {
        AppendBuf { raw: RawIobuf::new(len) }
    }

    /// Constructs a new Iobuf with a buffer of size `len`, undefined contents,
    /// and the limits and window set to the full range of the buffer. The memory
    /// will be allocated out of the given allocator, instead of the global heap.
    ///
    /// The maximum length of an Iobuf is approximately 2 GB.
    #[inline(always)]
    pub fn new_with_allocator(len: usize, allocator: Arc<Box<Allocator>>) -> AppendBuf<'static> {
        AppendBuf { raw: RawIobuf::new_with_allocator(len, allocator) }
    }

    /// Creates an AROIobuf as a slice of written buffer. This is space that preceeds
    /// the window in the buffer, or, more specifically, between the lo_min and lo offsets.
    /// This guarantees that the AROIobuf can be thought of as safely immutable while this
    /// buffer can continue to be `fill`ed and `poke`d. There are no operations for this buffer
    /// to reset the window to a lower position in the buffer.
    /// len is the number of bytes back from the start of the window where the slice begins
    ///  (and also the length of the slice)
    ///
    /// This function does not mutate the buffer, window, or limits.
    /// It just increases the refcount.
    ///
    /// ```rust
    /// #![allow(unstable)]
    /// use iobuf::{AppendBuf, Iobuf};
    ///
    ///   let mut buf = AppendBuf::new(24);
    ///   for i in b'A' .. b'X' + 1 {
    ///     buf.fill_be(i).unwrap();
    ///   }
    ///
    ///   let begin = buf.atomic_slice_from_begin(8).ok().expect("from_begin");
    ///   let middle = buf.atomic_slice_pos_from_end(16, 8).ok().expect("pos_from_end");
    ///   let end = buf.atomic_slice_from_end(8).ok().expect("from_end");
    ///   let meh = buf.atomic_slice_pos_from_begin(4, 8).ok().expect("pos_from_begin");
    ///
    ///   let b = unsafe { begin.as_window_slice() };
    ///   let m = unsafe { middle.as_window_slice() };
    ///   let e = unsafe { end.as_window_slice() };
    ///   let z = unsafe { meh.as_window_slice() };
    ///
    ///   assert_eq!(b, b"ABCDEFGH");
    ///   assert_eq!(m, b"IJKLMNOP");
    ///   assert_eq!(e, b"QRSTUVWX");
    ///   assert_eq!(z, b"EFGHIJKL");
    /// ```
    ///
    #[inline]
    pub fn atomic_slice_from_end(&self, len: u32) -> Result<AROIobuf, ()> {
      unsafe {
        let mut ret = self.raw.clone_atomic();
        if len > self.raw.lo() { return Err(()) }
        let lim = (self.raw.lo() - len,self.raw.lo()) ;
        try!(ret.expand_limits_and_window(lim, lim));
        Ok(mem::transmute(ret))
      }
    }

    /// Creates an AROIobuf as a slice of written buffer. This is space that preceeds
    /// the window in the buffer, or, more specifically, between the lo_min and lo offsets.
    /// This guarantees that the AROIobuf can be thought of as safely immutable while this
    /// buffer can continue to be `fill`ed and `poke`d. There are no operations for this buffer
    /// to reset the window to a lower position in the buffer.
    /// pos is the offset backwards from the beginning of the window
    /// len is the length of the slice
    ///
    /// This function does not mutate the buffer, window, or limits.
    /// It just increases the refcount.
    ///
    /// ```rust
    /// #![allow(unstable)]
    /// use iobuf::{AppendBuf, Iobuf};
    ///
    ///   let mut buf = AppendBuf::new(24);
    ///   for i in b'A' .. b'X' + 1 {
    ///     buf.fill_be(i).unwrap();
    ///   }
    ///
    ///   let begin = buf.atomic_slice_from_begin(8).ok().expect("from_begin");
    ///   let middle = buf.atomic_slice_pos_from_end(16, 8).ok().expect("pos_from_end");
    ///   let end = buf.atomic_slice_from_end(8).ok().expect("from_end");
    ///   let meh = buf.atomic_slice_pos_from_begin(4, 8).ok().expect("pos_from_begin");
    ///
    ///   let b = unsafe { begin.as_window_slice() };
    ///   let m = unsafe { middle.as_window_slice() };
    ///   let e = unsafe { end.as_window_slice() };
    ///   let z = unsafe { meh.as_window_slice() };
    ///
    ///   assert_eq!(b, b"ABCDEFGH");
    ///   assert_eq!(m, b"IJKLMNOP");
    ///   assert_eq!(e, b"QRSTUVWX");
    ///   assert_eq!(z, b"EFGHIJKL");
    /// ```
    ///
    #[inline]
    pub fn atomic_slice_pos_from_end(&self, pos: u32, len: u32) -> Result<AROIobuf, ()> {
      unsafe {
        let mut ret = self.raw.clone_atomic();
        if pos > self.raw.lo() { return Err(()) }
        let lim = (self.raw.lo() - pos, (self.raw.lo() - pos) + len);
        try!(ret.expand_limits_and_window(lim, lim));
        Ok(mem::transmute(ret))
      }
    }

    /// Creates an AROIobuf as a slice of written buffer. This is space that preceeds
    /// the window in the buffer, or, more specifically, between the lo_min and lo offsets.
    /// This guarantees that the AROIobuf can be thought of as safely immutable while this
    /// buffer can continue to be `fill`ed and `poke`d. There are no operations for this buffer
    /// to reset the window to a lower position in the buffer.
    /// pos is the point from the beginning of the buffer, or lo_min.
    /// len is the length
    ///
    /// This function does not mutate the buffer, window, or limits.
    /// It just increases the refcount.
    ///
    /// ```rust
    /// #![allow(unstable)]
    /// use iobuf::{AppendBuf, Iobuf};
    ///
    ///   let mut buf = AppendBuf::new(24);
    ///   for i in b'A' .. b'X' + 1 {
    ///     buf.fill_be(i).unwrap();
    ///   }
    ///
    ///   let begin = buf.atomic_slice_from_begin(8).ok().expect("from_begin");
    ///   let middle = buf.atomic_slice_pos_from_end(16, 8).ok().expect("pos_from_end");
    ///   let end = buf.atomic_slice_from_end(8).ok().expect("from_end");
    ///   let meh = buf.atomic_slice_pos_from_begin(4, 8).ok().expect("pos_from_begin");
    ///
    ///   let b = unsafe { begin.as_window_slice() };
    ///   let m = unsafe { middle.as_window_slice() };
    ///   let e = unsafe { end.as_window_slice() };
    ///   let z = unsafe { meh.as_window_slice() };
    ///
    ///   assert_eq!(b, b"ABCDEFGH");
    ///   assert_eq!(m, b"IJKLMNOP");
    ///   assert_eq!(e, b"QRSTUVWX");
    ///   assert_eq!(z, b"EFGHIJKL");
    /// ```
    ///
    #[inline]
    pub fn atomic_slice_from_begin(&self, len: u32) -> Result<AROIobuf, ()> {
      unsafe {
        let mut ret = self.raw.clone_atomic();
        let lim = (self.raw.lo_min(), self.raw.lo_min() + len);
        try!(ret.expand_limits_and_window(lim, lim));
        Ok(mem::transmute(ret))
      }
    }

    /// Creates an AROIobuf as a slice of written buffer. This is space that preceeds
    /// the window in the buffer, or, more specifically, between the lo_min and lo offsets.
    /// This guarantees that the AROIobuf can be thought of as safely immutable while this
    /// buffer can continue to be `fill`ed and `poke`d. There are no operations for this buffer
    /// to reset the window to a lower position in the buffer.
    /// pos is the point from the beginning of the buffer, or lo_min.
    /// len is the length
    ///
    /// This function does not mutate the buffer, window, or limits.
    /// It just increases the refcount.
    ///
    /// ```rust
    /// #![allow(unstable)]
    /// use iobuf::{AppendBuf, Iobuf};
    ///
    ///   let mut buf = AppendBuf::new(24);
    ///   for i in b'A' .. b'X' + 1 {
    ///     buf.fill_be(i).unwrap();
    ///   }
    ///
    ///   let begin = buf.atomic_slice_from_begin(8).ok().expect("from_begin");
    ///   let middle = buf.atomic_slice_pos_from_end(16, 8).ok().expect("pos_from_end");
    ///   let end = buf.atomic_slice_from_end(8).ok().expect("from_end");
    ///   let meh = buf.atomic_slice_pos_from_begin(4, 8).ok().expect("pos_from_begin");
    ///
    ///   let b = unsafe { begin.as_window_slice() };
    ///   let m = unsafe { middle.as_window_slice() };
    ///   let e = unsafe { end.as_window_slice() };
    ///   let z = unsafe { meh.as_window_slice() };
    ///
    ///   assert_eq!(b, b"ABCDEFGH");
    ///   assert_eq!(m, b"IJKLMNOP");
    ///   assert_eq!(e, b"QRSTUVWX");
    ///   assert_eq!(z, b"EFGHIJKL");
    /// ```
    ///
    #[inline]
    pub fn atomic_slice_pos_from_begin(&self, pos: u32, len: u32) -> Result<AROIobuf, ()> {
      unsafe {
        let mut ret = self.raw.clone_atomic();
        let lim = (self.raw.lo_min() + pos, self.raw.lo_min() + pos + len);
        try!(ret.expand_limits_and_window(lim, lim));
        Ok(mem::transmute(ret))
      }
    }

    /// Writes the bytes at a given offset from the beginning of the window, into
    /// the supplied buffer. Either the entire buffer is copied, or an error is
    /// returned because bytes outside of the window would be written.
    ///
    /// ```rust
    /// use iobuf::{RWIobuf,Iobuf};
    ///
    /// let data = &[ 1,2,3,4 ];
    ///
    /// let mut b = RWIobuf::new(10);
    ///
    /// assert_eq!(b.poke(0, data), Ok(()));
    /// assert_eq!(b.poke(3, data), Ok(()));
    /// assert_eq!(b.resize(7), Ok(()));
    /// assert_eq!(b.poke(4, data), Err(())); // no partial write, just failure
    ///
    /// let expected = [ 1,2,3,1,2,3,4 ];
    /// unsafe { assert_eq!(b.as_window_slice(), expected); }
    /// ```
    #[inline(always)]
    pub fn poke(&self, pos: u32, src: &[u8]) -> Result<(), ()> { self.raw.poke(pos, src) }

    /// Writes a big-endian primitive at a given offset from the beginning of the
    /// window.
    ///
    /// An error is returned if bytes outside of the window would be accessed.
    ///
    /// ```rust
    /// use iobuf::{RWIobuf,Iobuf};
    ///
    /// let mut b = RWIobuf::new(10);
    ///
    /// assert_eq!(b.poke_be(0, 0x0304u16), Ok(()));
    /// assert_eq!(b.poke_be(1, 0x0505u16), Ok(()));
    /// assert_eq!(b.poke_be(3, 0x06070809u32), Ok(()));
    ///
    /// assert_eq!(b.resize(7), Ok(()));
    ///
    /// let expected = [ 3,5,5,6,7,8,9 ];
    /// unsafe { assert_eq!(b.as_window_slice(), expected); }
    /// ```
    #[inline(always)]
    pub fn poke_be<T: Int>(&self, pos: u32, t: T) -> Result<(), ()> { self.raw.poke_be(pos, t) }

    /// Writes a little-endian primitive at a given offset from the beginning of
    /// the window.
    ///
    /// An error is returned if bytes outside of the window would be accessed.
    ///
    /// ```rust
    /// use iobuf::{RWIobuf,Iobuf};
    ///
    /// let mut b = RWIobuf::new(10);
    ///
    /// assert_eq!(b.poke_le(0, 0x0304u16), Ok(()));
    /// assert_eq!(b.poke_le(1, 0x0505u16), Ok(()));
    /// assert_eq!(b.poke_le(3, 0x06070809u32), Ok(()));
    ///
    /// assert_eq!(b.resize(7), Ok(()));
    ///
    /// unsafe { assert_eq!(b.as_window_slice(), [ 4, 5, 5, 9, 8, 7, 6 ]); }
    /// ```
    #[inline(always)]
    pub fn poke_le<T: Int>(&self, pos: u32, t: T) -> Result<(), ()> { self.raw.poke_le(pos, t) }

    /// Writes bytes from the supplied buffer, starting from the front of the
    /// window. Either the entire buffer is copied, or an error is returned
    /// because bytes outside the window were requested.
    ///
    /// After the bytes have been written, the window will be moved to no longer
    /// include then.
    ///
    /// ```rust
    /// use iobuf::{RWIobuf,Iobuf};
    ///
    /// let data = &[ 1, 2, 3, 4 ];
    ///
    /// let mut b = RWIobuf::new(10);
    ///
    /// assert_eq!(b.fill(data), Ok(()));
    /// assert_eq!(b.fill(data), Ok(()));
    /// assert_eq!(b.fill(data), Err(()));
    ///
    /// b.flip_lo();
    ///
    /// unsafe { assert_eq!(b.as_window_slice(), [ 1,2,3,4,1,2,3,4 ]); }
    /// ```
    #[inline(always)]
    pub fn fill(&mut self, src: &[u8]) -> Result<(), ()> { self.raw.fill(src) }

    /// Writes a big-endian primitive into the beginning of the window.
    ///
    /// After the primitive has been written, the window will be moved such that
    /// it is no longer included.
    ///
    /// An error is returned if bytes outside of the window were requested.
    ///
    /// ```rust
    /// use iobuf::{RWIobuf,Iobuf};
    ///
    /// let mut b = RWIobuf::new(10);
    ///
    /// assert_eq!(b.fill_be(0x12345678u32), Ok(()));
    /// assert_eq!(b.fill_be(0x11223344u32), Ok(()));
    /// assert_eq!(b.fill_be(0x54321123u32), Err(()));
    /// assert_eq!(b.fill_be(0x8877u16), Ok(()));
    ///
    /// b.flip_lo();
    ///
    /// unsafe { assert_eq!(b.as_window_slice(), [ 0x12, 0x34, 0x56, 0x78
    ///                                          , 0x11, 0x22, 0x33, 0x44
    ///                                          , 0x88, 0x77 ]); }
    /// ```
    #[inline(always)]
    pub fn fill_be<T: Int>(&mut self, t: T) -> Result<(), ()> { self.raw.fill_be(t) }

    /// Writes a little-endian primitive into the beginning of the window.
    ///
    /// After the primitive has been written, the window will be moved such that
    /// it is no longer included.
    ///
    /// An error is returned if bytes outside of the window were requested.
    ///
    /// ```rust
    /// use iobuf::{RWIobuf,Iobuf};
    ///
    /// let mut b = RWIobuf::new(10);
    ///
    /// assert_eq!(b.fill_le(0x12345678u32), Ok(()));
    /// assert_eq!(b.fill_le(0x11223344u32), Ok(()));
    /// assert_eq!(b.fill_le(0x54321123u32), Err(()));
    /// assert_eq!(b.fill_le(0x8877u16), Ok(()));
    ///
    /// b.flip_lo();
    ///
    /// unsafe { assert_eq!(b.as_window_slice(), [ 0x78, 0x56, 0x34, 0x12
    ///                                          , 0x44, 0x33, 0x22, 0x11
    ///                                          , 0x77, 0x88 ]); }
    /// ```
    #[inline(always)]
    pub fn fill_le<T: Int>(&mut self, t: T) -> Result<(), ()> { self.raw.fill_le(t) }

    /// Advances the lower bound of the window by `len`. `Err(())` will be
    /// returned if you advance past the upper bound of the window.
    ///
    /// ```rust
    /// use iobuf::{ROIobuf,Iobuf};
    ///
    /// let mut b = ROIobuf::from_str("hello");
    /// assert_eq!(b.advance(3), Ok(()));
    /// assert_eq!(b.advance(3), Err(()));
    /// unsafe { assert_eq!(b.as_window_slice(), b"lo"); }
    /// ```
    #[inline(always)]
    pub fn advance(&mut self, len: u32) -> Result<(), ()> { self.raw.advance(len) }

    /// Sets the window to the limits.
    ///
    /// "Take it to the limit..."
    ///
    /// NOTE: This can only work if the refcount on this buffer is 0
    ///
    /// ```rust
    /// use iobuf::{ROIobuf,Iobuf};
    ///
    /// let mut b = ROIobuf::from_str("hello");
    /// assert_eq!(b.resize(3), Ok(()));
    /// unsafe { assert_eq!(b.as_window_slice(), b"hel"); }
    /// assert_eq!(b.advance(2), Ok(()));
    /// unsafe { assert_eq!(b.as_window_slice(), b"l"); }
    /// b.reset();
    /// unsafe { assert_eq!(b.as_window_slice(), b"hello"); }
    /// ```
    #[inline]
    pub fn reset(&mut self) -> Result<(), ()> { unsafe {
        if self.raw.is_unique_atomic() {
            Ok(self.raw.reset())
        } else {
            Err(())
        }
    }}

    /// Returns the capacity of the current writing window
    #[inline(always)]
    pub fn len(&self) -> u32 { self.raw.len() }

    /// Returns the capacity of the entire buffer, written or not
    #[inline(always)]
    pub fn cap(&self) -> u32 { self.raw.cap() }

    /// Returns whether or not `len() == 0`.
    #[inline(always)]
    pub fn is_empty(&self) -> bool { self.raw.is_empty() }
}

impl<'a> Show for AppendBuf<'a> {
    #[inline]
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        self.raw.show(f, "append-only")
    }
}

#[unsafe_destructor]
impl<'a> Drop for AppendBuf<'a> {
  #[inline(always)]
  fn drop(&mut self) {
    unsafe { self.raw.drop_atomic() }
  }
}
