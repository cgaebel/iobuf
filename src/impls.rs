use std::fmt::{self, Show, Formatter};
use std::mem;
use std::num::Int;
use std::sync::Arc;


use raw::{Allocator, RawIobuf};
use iobuf::Iobuf;

/// Read-Only Iobuf
///
/// An `Iobuf` that cannot write into the buffer, but all read-only operations
/// are supported. It is possible to get a `RWIobuf` by performing a `deep_clone`
/// of the Iobuf, but this is extremely inefficient.
///
/// If your function only needs to do read-only operations on an Iobuf, consider
/// taking a generic `Iobuf` trait instead. That way, it can be used with either
/// a ROIobuf or a RWIobuf, generically.
#[unsafe_no_drop_flag]
pub struct ROIobuf<'a> {
  raw: RawIobuf<'a>,
}

#[test]
fn check_sane_roiobuf_size() {
    assert_eq!(mem::size_of::<ROIobuf<'static>>(), mem::size_of::<*mut u8>() + 16);
}

impl<'a> Clone for ROIobuf<'a> {
  #[inline(always)]
  fn clone(&self) -> ROIobuf<'a> {
    ROIobuf {
      raw: unsafe { self.raw.clone_nonatomic() },
    }
  }

  #[inline(always)]
  fn clone_from(&mut self, source: &ROIobuf<'a>) {
    unsafe { self.raw.clone_from_nonatomic(&source.raw) }
  }
}

#[unsafe_destructor]
impl<'a> Drop for ROIobuf<'a> {
  #[inline(always)]
  fn drop(&mut self) { unsafe { self.raw.drop_nonatomic() } }
}

/// Read-Write Iobuf
///
/// An `Iobuf` which can read and write into a buffer.
///
/// Iobufs may be `cloned` cheaply. When cloned, the data itself is shared and
/// refcounted, and a new copy of the limits and window is made. This can be
/// used to construct multiple views into the same buffer.
///
/// `poke` and `fill` write a value at a position relative to the start of
/// the window. Only `fill` advances the window by the amount written.
/// They are meant to be used with `try!`.
///
/// A suffix `_be` means the data will be read big-endian. A suffix `_le` means
/// the data will be read little-endian.
///
/// The `unsafe_` prefix means the function omits bounds checks. Misuse can
/// easily cause security issues. Be careful!
#[unsafe_no_drop_flag]
pub struct RWIobuf<'a> {
  raw: RawIobuf<'a>,
}

#[test]
fn check_sane_rwiobuf_size() {
    assert_eq!(mem::size_of::<RWIobuf<'static>>(), mem::size_of::<*mut u8>() + 16);
}

impl<'a> Clone for RWIobuf<'a> {
  #[inline(always)]
  fn clone(&self) -> RWIobuf<'a> {
    RWIobuf {
      raw: unsafe { self.raw.clone_nonatomic() },
    }
  }

  #[inline(always)]
  fn clone_from(&mut self, source: &RWIobuf<'a>) {
    unsafe { self.raw.clone_from_nonatomic(&source.raw) }
  }
}

#[unsafe_destructor]
impl<'a> Drop for RWIobuf<'a> {
  #[inline(always)]
  fn drop(&mut self) { unsafe { self.raw.drop_nonatomic() } }
}

/// Atomic Read-Only Iobuf
///
/// An `ROIobuf` which is safe to `Send` across tasks and `Share` with other tasks.
/// This atomically refcounts the buffer, which has a greater cost on `.clone()`,
/// but provides more flexibility to multithreaded consumers.
///
/// To create an `AROIobuf`, create a normal `Iobuf` and call `.atomic_read_only()`.
///
/// Below is an example of fill an Iobuf in one thread with the numbers 0x00 to
/// 0xFF, and consuming/validating these numbers in parallel in 4 other threads:
///
/// ```rust
/// #![allow(unstable)]
/// use iobuf::{RWIobuf, AROIobuf, Iobuf};
/// use std::sync::Future;
///
/// // Write the bytes 0x00 - 0xFF into an Iobuf.
/// fn fill(buf: &mut RWIobuf<'static>) -> Result<(), ()> {
///   for i in (0x00 .. 0x100) {
///     try!(buf.fill_be(i as u8));
///   }
///
///   Ok(())
/// }
///
/// // Validates the contents of buf are `idx`, `idx+1`, `idx+2`, ...
/// // until the buffer is exhausted.
/// fn check(buf: &mut AROIobuf, mut idx: u32) -> Result<(), ()> {
///   loop {
///     let b: u8 =
///       match buf.consume_be::<u8>() {
///         Err(()) => return Ok(()),
///         Ok(b) => b
///       };
///
///     if b as u32 == idx {
///       idx += 1;
///     } else {
///       return Err(())
///     }
///   }
/// }
///
/// let mut source_b = RWIobuf::new(256);
/// assert_eq!(fill(&mut source_b), Ok(()));
///
/// // Reset the Iobuf for reading.
/// source_b.flip_lo();
///
/// // We can still clone this buffer. It will be non-atomically refcounted for
/// // now.
/// {
///   let _ = source_b.clone();
/// }
///
/// // Now prepare it for sending to our 4 subtasks.
/// let shared_b: AROIobuf = source_b.atomic_read_only().unwrap();
///
/// let mut tasks = vec!();
///
/// for i in range(0u32, 4) {
///   // This clone modifies the AROIobuf's atomic refcount.
///   let mut b = shared_b.clone();
///   tasks.push(Future::spawn(move || {
///     let start = i*0x40;
///     assert_eq!(b.advance(start), Ok(()));
///     assert_eq!(b.resize(0x40), Ok(()));
///     assert_eq!(check(&mut b, start), Ok(()));
///     assert!(b.is_empty());
///
///     // This clone will atomically modify refcounts.
///     {
///       let _ = b.clone();
///     }
///   }));
/// }
///
/// for mut t in tasks.into_iter() {
///   t.get();
/// }
/// ```
#[unsafe_no_drop_flag]
pub struct AROIobuf {
  raw: RawIobuf<'static>,
}

unsafe impl Send for AROIobuf {}
unsafe impl Sync for AROIobuf {}

impl Clone for AROIobuf {
  #[inline(always)]
  fn clone(&self) -> AROIobuf { AROIobuf { raw: unsafe { self.raw.clone_atomic() } } }

  #[inline(always)]
  fn clone_from(&mut self, source: &AROIobuf) { unsafe { self.raw.clone_from_atomic(&source.raw) } }
}

impl Drop for AROIobuf {
  #[inline(always)]
  fn drop(&mut self) { unsafe { self.raw.drop_atomic() } }
}


/// A unique, immutable Iobuf.
///
/// If the refcount on an Iobuf is `1`, it can be made unique with `.unique()`.
/// This will allow sending across channels, and later conversion back to a
/// normal refcounted (atomically or non) Iobuf with zero overhead.
#[unsafe_no_drop_flag]
pub struct UniqueIobuf {
  raw: RawIobuf<'static>,
}

unsafe impl Send for UniqueIobuf {}
unsafe impl Sync for UniqueIobuf {}

impl UniqueIobuf {
  /// Safely converts a `UniqueIobuf` into a `ROIobuf`.
  #[inline(always)]
  pub fn read_only(self) -> ROIobuf<'static> {
    unsafe { mem::transmute(self) }
  }

  /// Safely converts a `UniqueIobuf` into a `RWIobuf`.
  #[inline(always)]
  pub fn read_write(self) -> RWIobuf<'static> {
    unsafe { mem::transmute(self) }
  }

  /// Safely converts a `UniqueIobuf` into a `AROIobuf`.
  #[inline(always)]
  pub fn atomic_read_only(self) -> AROIobuf {
    unsafe { mem::transmute(self) }
  }
}

impl Drop for UniqueIobuf {
  #[inline(always)]
  fn drop(&mut self) { unsafe { self.raw.drop_nonatomic() } }
}

impl<'a> ROIobuf<'a> {
  /// Constructs a trivially empty Iobuf, limits and window are 0, and there's
  /// an empty backing buffer. This will not allocate.
  ///
  /// ```rust
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::empty();
  ///
  /// assert_eq!(b.cap(), 0);
  /// assert_eq!(b.len(), 0);
  /// ```
  #[inline(always)]
  pub fn empty() -> ROIobuf<'static> {
    ROIobuf { raw: RawIobuf::empty() }
  }

  /// Constructs an Iobuf with the same contents as a string. The limits and
  /// window will be initially set to cover the whole string.
  ///
  /// No copying or allocating will be done by this function.
  ///
  /// ```rust
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("hello");
  ///
  /// assert_eq!(b.cap(), 5);
  /// assert_eq!(b.len(), 5);
  /// unsafe { assert_eq!(b.as_window_slice(), b"hello"); }
  /// unsafe { assert_eq!(b.as_limit_slice(), b"hello"); }
  /// ```
  #[inline(always)]
  pub fn from_str(s: &'a str) -> ROIobuf<'a> {
    ROIobuf { raw: RawIobuf::from_str(s) }
  }

  /// Copies a `str` into a read-only Iobuf. The contents of the `str` will be
  /// copied, so prefer to use the other constructors whenever possible.
  ///
  /// ```rust
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str_copy("hello");
  ///
  /// assert_eq!(b.cap(), 5);
  /// assert_eq!(b.len(), 5);
  /// unsafe { assert_eq!(b.as_window_slice(), b"hello"); }
  /// unsafe { assert_eq!(b.as_limit_slice(), b"hello"); }
  /// ```
  #[inline(always)]
  pub fn from_str_copy(s: &str) -> ROIobuf<'static> {
    ROIobuf { raw: RawIobuf::from_str_copy(s) }
  }

  /// Copies a `str` into a read-only Iobuf, whose memory comes from the given
  /// allocator.
  #[inline(always)]
  pub fn from_str_copy_with_allocator(s: &str, allocator: Arc<Box<Allocator>>) -> ROIobuf<'static> {
    ROIobuf { raw: RawIobuf::from_str_copy_with_allocator(s, allocator) }
  }

  /// Copies the contents of a slice into a read-only Iobuf. The contents of the
  /// slice will be copied, so prefer to use the other constructors whenever
  /// possible.
  ///
  /// ```rust
  /// #![allow(unstable)]
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut v = vec!(1u8, 2, 3, 4, 5, 6);
  /// v.as_mut_slice()[1] = 20;
  ///
  /// let mut b = ROIobuf::from_slice_copy(&v[]);
  ///
  /// let expected = [ 1,20,3,4,5,6 ];
  /// unsafe { assert_eq!(b.as_window_slice(), &expected[]); }
  /// ```
  #[inline(always)]
  pub fn from_slice_copy(s: &[u8]) -> ROIobuf<'static> {
    ROIobuf { raw: RawIobuf::from_slice_copy(s) }
  }

  /// Copies a byte vector into a new read-only Iobuf, whose memory comes from
  /// the given allocator.
  #[inline(always)]
  pub fn from_slice_copy_with_allocator(s: &[u8], allocator: Arc<Box<Allocator>>) -> ROIobuf<'static> {
    ROIobuf { raw: RawIobuf::from_slice_copy_with_allocator(s, allocator) }
  }

  /// Constructs an Iobuf from a slice. The Iobuf will not copy the slice
  /// contents, and therefore their lifetimes will be linked.
  ///
  /// ```rust
  /// #![allow(unstable)]
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let s = [1,2,3,4];
  ///
  /// let mut b = ROIobuf::from_slice(&s[]);
  ///
  /// assert_eq!(b.advance(1), Ok(()));
  ///
  /// assert_eq!(s[1], 2); // we can still use the slice!
  /// assert_eq!(b.peek_be(1), Ok(0x0304u16)); // ...and the Iobuf!
  /// ```
  #[inline(always)]
  pub fn from_slice(s: &'a [u8]) -> ROIobuf<'a> {
    ROIobuf { raw: RawIobuf::from_slice(s) }
  }
}

impl<'a> RWIobuf<'a> {
  /// Constructs a trivially empty Iobuf, limits and window are 0, and there's
  /// an empty backing buffer. This will not allocate.
  ///
  /// ```rust
  /// use iobuf::{RWIobuf,Iobuf};
  ///
  /// let mut b = RWIobuf::empty();
  ///
  /// assert_eq!(b.len(), 0);
  /// assert_eq!(b.cap(), 0);
  /// ```
  #[inline(always)]
  pub fn empty() -> RWIobuf<'static> {
    RWIobuf { raw: RawIobuf::empty() }
  }

  /// Constructs a new Iobuf with a buffer of size `len`, undefined contents,
  /// and the limits and window set to the full size of the buffer.
  ///
  /// The maximum length of an Iobuf is approximately 2 GB.
  ///
  /// ```rust
  /// use iobuf::{RWIobuf,Iobuf};
  ///
  /// let mut b = RWIobuf::new(10);
  ///
  /// assert_eq!(b.len(), 10);
  /// assert_eq!(b.cap(), 10);
  /// ```
  #[inline(always)]
  pub fn new(len: usize) -> RWIobuf<'static> {
    RWIobuf { raw: RawIobuf::new(len) }
  }

  /// Constructs a new Iobuf with a buffer of size `len`, undefined contents,
  /// and the limits and window set to the full range of the buffer. The memory
  /// will be allocated out of the given allocator, instead of the global heap.
  ///
  /// The maximum length of an Iobuf is approximately 2 GB.
  #[inline(always)]
  pub fn new_with_allocator(len: usize, allocator: Arc<Box<Allocator>>) -> RWIobuf<'static> {
    RWIobuf { raw: RawIobuf::new_with_allocator(len, allocator) }
  }

  /// Copies a `str` into a writeable Iobuf. The contents of the `str` will be
  /// copied, so prefer to use the non-copying constructors whenever possible.
  ///
  /// ```rust
  /// use iobuf::{RWIobuf,Iobuf};
  ///
  /// let mut b = RWIobuf::from_str_copy("hello");
  ///
  /// b.poke_be(1, b'4').unwrap();
  ///
  /// assert_eq!(b.len(), 5);
  /// assert_eq!(b.cap(), 5);
  /// unsafe { assert_eq!(b.as_window_slice(), b"h4llo"); }
  /// unsafe { assert_eq!(b.as_limit_slice(), b"h4llo"); }
  /// ```
  #[inline(always)]
  pub fn from_str_copy(s: &str) -> RWIobuf<'static> {
    RWIobuf { raw: RawIobuf::from_str_copy(s) }
  }

  /// Copies a `str` into a writeable Iobuf, whose memory comes from the given
  /// allocator.
  #[inline(always)]
  pub fn from_str_copy_with_allocator(s: &str, allocator: Arc<Box<Allocator>>) -> RWIobuf<'static> {
    RWIobuf { raw: RawIobuf::from_str_copy_with_allocator(s, allocator) }
  }

  /// Constructs an Iobuf from a slice. The Iobuf will not copy the slice
  /// contents, and therefore their lifetimes will be linked.
  ///
  /// ```rust
  /// use iobuf::{RWIobuf,Iobuf};
  ///
  /// let mut s = [1,2,3,4];
  ///
  /// {
  ///   let mut b = RWIobuf::from_slice(s.as_mut_slice());
  ///
  ///   assert_eq!(b.advance(1), Ok(()));
  ///   assert_eq!(b.peek_be(1), Ok(0x0304u16)); // ...and the Iobuf!
  ///   assert_eq!(b.poke_be(1, 100u8), Ok(()));
  /// }
  ///
  /// // We can still use the slice, but only because of the braces above.
  /// assert_eq!(s[2], 100);
  /// ```
  #[inline(always)]
  pub fn from_slice(s: &'a mut [u8]) -> RWIobuf<'a> {
    RWIobuf { raw: RawIobuf::from_slice(s) }
  }

  /// Copies a byte vector into a new, writeable Iobuf. The contents of the
  /// slice will be copied, so prefer to use the other constructors whenever
  /// possible.
  ///
  /// ```rust
  /// #![allow(unstable)]
  /// use iobuf::{RWIobuf,Iobuf};
  ///
  /// let mut v = vec!(1u8, 2, 3, 4, 5, 6, 10);
  /// v.as_mut_slice()[1] = 20;
  ///
  /// let mut b = RWIobuf::from_slice_copy(&v[]);
  ///
  /// let expected = [ 1,20,3,4,5,6,10 ];
  /// unsafe { assert_eq!(b.as_window_slice(), expected); }
  /// ```
  #[inline(always)]
  pub fn from_slice_copy(s: &[u8]) -> RWIobuf<'static> {
    RWIobuf { raw: RawIobuf::from_slice_copy(s) }
  }

  /// Copies a byte vector into a new writeable Iobuf, whose memory comes from
  /// the given allocator.
  #[inline(always)]
  pub fn from_slice_copy_with_allocator(s: &[u8], allocator: Arc<Box<Allocator>>) -> RWIobuf<'static> {
    RWIobuf { raw: RawIobuf::from_slice_copy_with_allocator(s, allocator) }
  }

  /// Reads the data in the window as a mutable slice. Note that since `&mut`
  /// in rust really means `&unique`, this function lies. There can exist
  /// multiple slices of the same data. Therefore, this function is unsafe.
  ///
  /// It may only be used safely if you ensure that the data in the iobuf never
  /// interacts with the slice, as they pointe to the same data. `peek`ing or
  /// `poke`ing the slice returned from this function is a big no-no.
  ///
  /// ```rust
  /// #![allow(unstable)]
  /// use iobuf::{RWIobuf, Iobuf};
  ///
  /// let mut s = [1,2,3];
  ///
  /// {
  ///   let mut b = RWIobuf::from_slice(&mut s[]);
  ///
  ///   assert_eq!(b.advance(1), Ok(()));
  ///   unsafe { b.as_mut_window_slice()[1] = 30; }
  /// }
  ///
  /// let expected = [ 1,2,30 ];
  /// assert_eq!(s, &expected[]);
  /// ```
  #[inline(always)]
  pub unsafe fn as_mut_window_slice<'b>(&'b self) -> &'b mut [u8] {
    self.raw.as_mut_window_slice()
  }

  /// Reads the data in the window as a mutable slice. Note that since `&mut`
  /// in rust really means `&unique`, this function lies. There can exist
  /// multiple slices of the same data. Therefore, this function is unsafe.
  ///
  /// It may only be used safely if you ensure that the data in the iobuf never
  /// interacts with the slice, as they pointe to the same data. `peek`ing or
  /// `poke`ing the slice returned from this function is a big no-no.
  ///
  /// ```rust
  /// #![allow(unstable)]
  /// use iobuf::{RWIobuf, Iobuf};
  ///
  /// let mut s = [1,2,3];
  ///
  /// {
  ///   let mut b = RWIobuf::from_slice(&mut s[]);
  ///
  ///   assert_eq!(b.advance(1), Ok(()));
  ///   unsafe { b.as_mut_limit_slice()[1] = 20; }
  /// }
  ///
  /// assert_eq!(s, &[1,20,3][]);
  /// ```
  #[inline(always)]
  pub unsafe fn as_mut_limit_slice<'b>(&'b self) -> &'b mut [u8] {
    self.raw.as_mut_limit_slice()
  }

  /// Gets a read-only copy of this Iobuf. This is a very cheap operation, as
  /// the backing buffers are shared. This can be useful for interfacing with
  /// code that only accepts read-only Iobufs.
  ///
  /// In general, ROIobuf should never be used as a function parameter. If
  /// read-only acceess is all that is required, take a generic `<T: Iobuf>`.
  ///
  /// ```rust
  /// use iobuf::{ROIobuf,RWIobuf,Iobuf};
  ///
  /// let mut s = [1,2,3,4];
  ///
  /// let rwb: RWIobuf = RWIobuf::from_slice(s.as_mut_slice());
  ///
  /// // write some data into rwb.
  ///
  /// let rb: ROIobuf = rwb.read_only();
  ///
  /// // now do read-only ops.
  /// assert_eq!(rb.len(), 4);
  /// ```
  #[inline(always)]
  pub fn read_only(&self) -> ROIobuf<'a> {
    ROIobuf { raw: unsafe { self.raw.clone_nonatomic() } }
  }

  /// Copies data from the window to the lower limit fo the iobuf and sets the
  /// window to range from the end of the copied data to the upper limit. This
  /// is typically called after a series of `Consume`s to save unread data and
  /// prepare for the next series of `Fill`s and `flip_lo`s.
  ///
  /// ```rust
  /// use std::result::Result::{self,Ok};
  /// use iobuf::{RWIobuf,Iobuf};
  ///
  /// // A header, saying how many shorts will follow. Unfortunately, our buffer
  /// // isn't big enough for all the shorts! Assume the rest will be sent in a
  /// // later packet.
  /// let mut s = [ 0x02, 0x11, 0x22, 0x33 ];
  /// let mut b = RWIobuf::from_slice(s.as_mut_slice());
  ///
  /// #[derive(Eq, PartialEq, Show)]
  /// enum ParseState {
  ///   NeedMore(u16), // sum so far
  ///   Done(u16),     // final sum
  /// };
  ///
  /// // Returns a pair of the sum of shorts seen so far, and `true` if we're
  /// // finally done parsing. The sum will be partial if parsing is incomplete.
  /// fn parse(b: &mut RWIobuf) -> Result<ParseState, ()> {
  ///   let len: u8 = try!(b.consume_be());
  ///   let mut sum = 0u16;
  ///
  ///   for _ in (0u8 .. len) {
  ///     unsafe {
  ///       if b.len() < 2 {
  ///         b.compact();
  ///         return Ok(ParseState::NeedMore(sum));
  ///       }
  ///       sum += b.unsafe_consume_be();
  ///     }
  ///   }
  ///
  ///   Ok(ParseState::Done(sum))
  /// }
  ///
  /// assert_eq!(parse(&mut b), Ok(ParseState::NeedMore(0x1122)));
  /// assert_eq!(b.len(), 3);
  /// assert_eq!(b.cap(), 4);
  /// assert_eq!(b.peek_be(0), Ok(0x11u8));
  /// ```
  #[inline(always)]
  pub fn compact(&mut self) { self.raw.compact() }

  /// Writes the bytes at a given offset from the beginning of the window, into
  /// the supplied buffer. Either the entire buffer is copied, or an error is
  /// returned because bytes outside of the window would be written.
  ///
  /// ```rust
  /// #![allow(unstable)]
  /// use iobuf::{RWIobuf,Iobuf};
  ///
  /// let data = [ 1,2,3,4 ];
  ///
  /// let mut b = RWIobuf::new(10);
  ///
  /// assert_eq!(b.poke(0, &data[]), Ok(()));
  /// assert_eq!(b.poke(3, &data[]), Ok(()));
  /// assert_eq!(b.resize(7), Ok(()));
  /// assert_eq!(b.poke(4, &data[]), Err(())); // no partial write, just failure
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
  /// #![allow(unstable)]
  /// use iobuf::{RWIobuf,Iobuf};
  ///
  /// let data = [ 1, 2, 3, 4 ];
  ///
  /// let mut b = RWIobuf::new(10);
  ///
  /// assert_eq!(b.fill(&data[]), Ok(()));
  /// assert_eq!(b.fill(&data[]), Ok(()));
  /// assert_eq!(b.fill(&data[]), Err(()));
  ///
  /// b.flip_lo();
  ///
  /// unsafe { assert_eq!(b.as_window_slice(), &[ 1,2,3,4,1,2,3,4 ][]); }
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

  /// Writes the bytes at a given offset from the beginning of the window, into
  /// the supplied buffer. It is undefined behavior to write outside the iobuf
  /// window.
  ///
  /// ```rust
  /// use iobuf::{RWIobuf,Iobuf};
  ///
  /// let data = [ 1,2,3,4 ];
  ///
  /// let mut b = RWIobuf::new(10);
  ///
  /// unsafe {
  ///   b.check_range_fail(1, 7);
  ///
  ///   b.unsafe_advance(1);
  ///   b.narrow();
  ///
  ///   b.unsafe_poke(0, &data);
  ///   b.unsafe_poke(3, &data);
  ///   b.unsafe_advance(7);
  /// }
  ///
  /// b.flip_lo();
  ///
  /// unsafe { assert_eq!(b.as_window_slice(), [ 1,2,3,1,2,3,4 ]); }
  /// ```
  #[inline(always)]
  pub unsafe fn unsafe_poke(&self, pos: u32, src: &[u8]) { self.raw.unsafe_poke(pos, src) }

  /// Writes a big-endian primitive at a given offset from the beginning of the
  /// window. It is undefined behavior to write outside the iobuf window.
  ///
  /// ```rust
  /// use iobuf::{RWIobuf,Iobuf};
  ///
  /// let mut b = RWIobuf::new(10);
  ///
  /// unsafe {
  ///   b.check_range_fail(0, 7);
  ///
  ///   b.unsafe_poke_be(0, 0x0304u16);
  ///   b.unsafe_poke_be(1, 0x0505u16);
  ///   b.unsafe_poke_be(3, 0x06070809u32);
  /// }
  ///
  /// assert_eq!(b.resize(7), Ok(()));
  ///
  /// unsafe { assert_eq!(b.as_window_slice(), [ 3, 5, 5, 6, 7, 8, 9 ]); }
  /// ```
  #[inline(always)]
  pub unsafe fn unsafe_poke_be<T: Int>(&self, pos: u32, t: T) { self.raw.unsafe_poke_be(pos, t) }

  /// Writes a little-endian primitive at a given offset from the beginning of
  /// the window. It is undefined behavior to write outside the iobuf window.
  ///
  /// ```rust
  /// use iobuf::{RWIobuf,Iobuf};
  ///
  /// let mut b = RWIobuf::new(10);
  ///
  /// unsafe {
  ///   b.check_range_fail(0, 7);
  ///
  ///   b.unsafe_poke_le(0, 0x0304u16);
  ///   b.unsafe_poke_le(1, 0x0505u16);
  ///   b.unsafe_poke_le(3, 0x06070809u32);
  /// }
  ///
  /// assert_eq!(b.resize(7), Ok(()));
  ///
  /// unsafe { assert_eq!(b.as_window_slice(), [ 4, 5, 5, 9, 8, 7, 6 ]); }
  /// ```
  #[inline(always)]
  pub unsafe fn unsafe_poke_le<T: Int>(&self, pos: u32, t: T) { self.raw.unsafe_poke_le(pos, t) }

  /// Writes bytes from the supplied buffer, starting from the front of the
  /// window. It is undefined behavior to write outside the iobuf window.
  ///
  /// After the bytes have been written, the window will be moved to no longer
  /// include then.
  ///
  /// ```rust
  /// #![allow(unstable)]
  /// use iobuf::{RWIobuf,Iobuf};
  ///
  /// let data = [ 1, 2, 3, 4 ];
  ///
  /// let mut b = RWIobuf::new(10);
  ///
  /// unsafe {
  ///   b.check_range_fail(0, 8);
  ///
  ///   b.unsafe_fill(&data[]);
  ///   b.unsafe_fill(&data[]);
  /// }
  ///
  /// b.flip_lo();
  ///
  /// unsafe { assert_eq!(b.as_window_slice(), [ 1,2,3,4,1,2,3,4 ]); }
  /// ```
  #[inline(always)]
  pub unsafe fn unsafe_fill(&mut self, src: &[u8]) { self.raw.unsafe_fill(src) }

  /// Writes a big-endian primitive into the beginning of the window. It is
  /// undefined behavior to write outside the iobuf window.
  ///
  /// After the primitive has been written, the window will be moved such that
  /// it is no longer included.
  ///
  /// ```rust
  /// use iobuf::{RWIobuf,Iobuf};
  ///
  /// let mut b = RWIobuf::new(10);
  ///
  /// unsafe {
  ///   b.check_range_fail(0, 10);
  ///
  ///   b.unsafe_fill_be(0x12345678u32);
  ///   b.unsafe_fill_be(0x11223344u32);
  ///   // b.unsafe_fill_be(0x54321123u32); DO NOT DO THIS. Undefined behavior.
  ///   b.unsafe_fill_be(0x8877u16);
  /// }
  ///
  /// b.flip_lo();
  ///
  /// unsafe { assert_eq!(b.as_window_slice(), [ 0x12, 0x34, 0x56, 0x78
  ///                                          , 0x11, 0x22, 0x33, 0x44
  ///                                          , 0x88, 0x77 ]); }
  /// ```
  #[inline(always)]
  pub unsafe fn unsafe_fill_be<T: Int>(&mut self, t: T) { self.raw.unsafe_fill_be(t) }

  /// Writes a little-endian primitive into the beginning of the window. It is
  /// undefined behavior to write outside the iobuf window.
  ///
  /// After the primitive has been written, the window will be moved such that
  /// it is no longer included.
  ///
  /// ```rust
  /// use iobuf::{RWIobuf,Iobuf};
  ///
  /// let mut b = RWIobuf::new(10);
  ///
  /// unsafe {
  ///   b.check_range_fail(0, 10);
  ///
  ///   b.unsafe_fill_le(0x12345678u32);
  ///   b.unsafe_fill_le(0x11223344u32);
  ///   // b.unsafe_fill_le(0x54321123u32); DO NOT DO THIS. Undefined behavior.
  ///   b.unsafe_fill_le(0x8877u16);
  /// }
  ///
  /// b.flip_lo();
  ///
  /// unsafe { assert_eq!(b.as_window_slice(), [ 0x78, 0x56, 0x34, 0x12
  ///                                          , 0x44, 0x33, 0x22, 0x11
  ///                                          , 0x77, 0x88 ]); }
  /// ```
  #[inline(always)]
  pub unsafe fn unsafe_fill_le<T: Int>(&mut self, t: T) { self.raw.unsafe_fill_le(t) }
}

impl AROIobuf {
  /// Stops atomically reference counting a unique buffer. This method returns
  /// `Ok` if the `AROIobuf` is the last of its kind, and `Err` if it's not.
  ///
  /// ```rust
  /// use iobuf::{AROIobuf, ROIobuf, Iobuf};
  ///
  /// let buf: ROIobuf<'static> = ROIobuf::from_str_copy("hello, world!");
  /// let a_buf: AROIobuf = buf.atomic_read_only().unwrap();
  /// unsafe { assert_eq!(a_buf.as_window_slice(), b"hello, world!"); }
  ///
  /// let buf: ROIobuf<'static> = a_buf.read_only().unwrap();
  /// // TA-DA!
  /// unsafe { assert_eq!(buf.as_window_slice(), b"hello, world!"); }
  /// ```
  #[inline(always)]
  pub fn read_only(self) -> Result<ROIobuf<'static>, AROIobuf> {
    unsafe {
      if self.raw.is_unique_atomic() {
        Ok(mem::transmute(self))
      } else {
        Err(self)
      }
    }
  }

  /// Stops atomically reference counting a unique buffer. This method returns
  /// `Ok` if the `AROIobuf` is the last of its kind, and `Err` if it's not.
  ///
  /// ```rust
  /// use iobuf::{AROIobuf, ROIobuf, RWIobuf, Iobuf};
  ///
  /// let buf: ROIobuf<'static> = ROIobuf::from_str_copy("hello, world!");
  /// let a_buf: AROIobuf = buf.atomic_read_only().unwrap();
  /// unsafe { assert_eq!(a_buf.as_window_slice(), b"hello, world!"); }
  ///
  /// let buf: RWIobuf<'static> = a_buf.read_write().unwrap();
  /// // TA-DA!
  /// unsafe { assert_eq!(buf.as_window_slice(), b"hello, world!"); }
  /// ```
  #[inline(always)]
  pub fn read_write(self) -> Result<RWIobuf<'static>, AROIobuf> {
    unsafe {
      if self.raw.is_unique_atomic() {
        Ok(mem::transmute(self))
      } else {
        Err(self)
      }
    }
  }
}

impl<'a> Iobuf for ROIobuf<'a> {
  #[inline(always)]
  fn deep_clone(&self) -> RWIobuf<'static> { RWIobuf { raw: self.raw.deep_clone() } }

  #[inline(always)]
  fn deep_clone_with_allocator(&self, allocator: Arc<Box<Allocator>>) -> RWIobuf<'static> {
    RWIobuf { raw: self.raw.deep_clone_with_allocator(allocator) }
  }

  #[inline(always)]
  fn unique(self) -> Result<UniqueIobuf, ROIobuf<'a>> {
    unsafe {
      if self.raw.is_unique_nonatomic() {
        Ok(mem::transmute(self))
      } else {
        Err(self)
      }
    }
  }

  #[inline(always)]
  fn atomic_read_only(self) -> Result<AROIobuf, ROIobuf<'a>> {
    unsafe {
      if self.raw.is_unique_nonatomic() {
        Ok(mem::transmute(self))
      } else {
        Err(self)
      }
    }
  }

  #[inline(always)]
  fn len(&self) -> u32 { self.raw.len() }

  #[inline(always)]
  fn cap(&self) -> u32 { self.raw.cap() }

  #[inline(always)]
  fn is_empty(&self) -> bool { self.raw.is_empty() }

  #[inline(always)]
  unsafe fn as_window_slice<'b>(&'b self) -> &'b [u8] { self.raw.as_window_slice() }

  #[inline(always)]
  unsafe fn as_limit_slice<'b>(&'b self) -> &'b [u8] { self.raw.as_limit_slice() }

  #[inline(always)]
  fn sub_window(&mut self, pos: u32, len: u32) -> Result<(), ()> { self.raw.sub_window(pos, len) }

  #[inline(always)]
  fn sub_window_from(&mut self, pos: u32) -> Result<(), ()> { self.raw.sub_window_from(pos) }

  #[inline(always)]
  fn sub_window_to(&mut self, len: u32) -> Result<(), ()> { self.raw.sub_window_to(len) }

  #[inline(always)]
  unsafe fn unsafe_sub_window(&mut self, pos: u32, len: u32) { self.raw.unsafe_sub_window(pos, len) }

  #[inline(always)]
  unsafe fn unsafe_sub_window_from(&mut self, pos: u32) { self.raw.unsafe_sub_window_from(pos) }

  #[inline(always)]
  unsafe fn unsafe_sub_window_to(&mut self, len: u32) { self.raw.unsafe_sub_window_to(len) }

  #[inline(always)]
  fn sub(&mut self, pos: u32, len: u32) -> Result<(), ()> { self.raw.sub(pos, len) }

  #[inline(always)]
  fn sub_from(&mut self, pos: u32) -> Result<(), ()> { self.raw.sub_from(pos) }

  #[inline(always)]
  fn sub_to(&mut self, len: u32) -> Result<(), ()> { self.raw.sub_to(len) }

  #[inline(always)]
  unsafe fn unsafe_sub(&mut self, pos: u32, len: u32) { self.raw.unsafe_sub(pos, len) }

  #[inline(always)]
  unsafe fn unsafe_sub_from(&mut self, pos: u32) { self.raw.unsafe_sub_from(pos) }

  #[inline(always)]
  unsafe fn unsafe_sub_to(&mut self, len: u32) { self.raw.unsafe_sub_to(len) }

  #[inline(always)]
  fn set_limits_and_window(&mut self, limits: (u32, u32), window: (u32, u32)) -> Result<(), ()> { self.raw.set_limits_and_window(limits, window) }

  #[inline(always)]
  fn narrow(&mut self) { self.raw.narrow() }

  #[inline(always)]
  fn advance(&mut self, len: u32) -> Result<(), ()> { self.raw.advance(len) }

  #[inline(always)]
  unsafe fn unsafe_advance(&mut self, len: u32) { self.raw.unsafe_advance(len) }

  #[inline(always)]
  fn extend(&mut self, len: u32) -> Result<(), ()> { self.raw.extend(len) }

  #[inline(always)]
  unsafe fn unsafe_extend(&mut self, len: u32) { self.raw.unsafe_extend(len) }

  #[inline(always)]
  fn is_extended_by<Buf: Iobuf>(&self, other: &Buf) -> bool { unsafe { self.raw.is_extended_by(other.as_raw()) } }

  #[inline(always)]
  fn extend_with<Buf: Iobuf>(&mut self, other: &Buf) -> Result<(), ()> { unsafe { self.raw.extend_with(other.as_raw()) } }

  #[inline(always)]
  fn resize(&mut self, len: u32) -> Result<(), ()> { self.raw.resize(len) }

  #[inline(always)]
  unsafe fn unsafe_resize(&mut self, len: u32) { self.raw.unsafe_resize(len) }

  #[inline(always)]
  fn split_at(&self, pos: u32) -> Result<(ROIobuf<'a>, ROIobuf<'a>), ()> {
    self.raw.split_at_nonatomic(pos).map(|(a, b)| (ROIobuf { raw: a }, ROIobuf { raw: b }))
  }

  #[inline(always)]
  unsafe fn unsafe_split_at(&self, pos: u32) -> (ROIobuf<'a>, ROIobuf<'a>) {
    let (a, b) = self.raw.unsafe_split_at_nonatomic(pos);
    (ROIobuf { raw: a }, ROIobuf { raw: b })
  }

  #[inline(always)]
  fn split_start_at(&mut self, pos: u32) -> Result<ROIobuf<'a>, ()> {
    self.raw.split_start_at_nonatomic(pos).map(|b| ROIobuf { raw: b })
  }

  #[inline(always)]
  unsafe fn unsafe_split_start_at(&mut self, pos: u32) -> ROIobuf<'a> {
    ROIobuf { raw: self.raw.unsafe_split_start_at_nonatomic(pos) }
  }

  #[inline(always)]
  fn rewind(&mut self) { self.raw.rewind() }

  #[inline(always)]
  fn reset(&mut self) { self.raw.reset() }

  #[inline(always)]
  fn flip_lo(&mut self) { self.raw.flip_lo() }

  #[inline(always)]
  fn flip_hi(&mut self) { self.raw.flip_hi() }

  #[inline(always)]
  fn lo_space(&self) -> u32 { self.raw.lo_space() }

  #[inline(always)]
  fn hi_space(&self) -> u32 { self.raw.hi_space() }

  #[inline(always)]
  fn peek(&self, pos: u32, dst: &mut [u8]) -> Result<(), ()> { self.raw.peek(pos, dst) }
  #[inline(always)]
  fn peek_be<T: Int>(&self, pos: u32) -> Result<T, ()> { self.raw.peek_be(pos) }
  #[inline(always)]
  fn peek_le<T: Int>(&self, pos: u32) -> Result<T, ()> { self.raw.peek_le(pos) }

  #[inline(always)]
  fn consume(&mut self, dst: &mut [u8]) -> Result<(), ()> { self.raw.consume(dst) }
  #[inline(always)]
  fn consume_be<T: Int>(&mut self) -> Result<T, ()> { self.raw.consume_be::<T>() }
  #[inline(always)]
  fn consume_le<T: Int>(&mut self) -> Result<T, ()> { self.raw.consume_le::<T>() }

  #[inline(always)]
  fn check_range(&self, pos: u32, len: u32) -> Result<(), ()> { self.raw.check_range_u32(pos, len) }

  #[inline(always)]
  fn check_range_usize(&self, pos: u32, len: usize) -> Result<(), ()> { self.raw.check_range_usize(pos, len) }

  #[inline(always)]
  fn check_range_fail(&self, pos: u32, len: u32) { self.raw.check_range_u32_fail(pos, len) }

  #[inline(always)]
  fn check_range_usize_fail(&self, pos: u32, len: usize) { self.raw.check_range_usize_fail(pos, len) }

  #[inline(always)]
  unsafe fn unsafe_peek(&self, pos: u32, dst: &mut [u8]) { self.raw.unsafe_peek(pos, dst) }
  #[inline(always)]
  unsafe fn unsafe_peek_be<T: Int>(&self, pos: u32) -> T { self.raw.unsafe_peek_be(pos) }
  #[inline(always)]
  unsafe fn unsafe_peek_le<T: Int>(&self, pos: u32) -> T { self.raw.unsafe_peek_le(pos) }

  #[inline(always)]
  unsafe fn unsafe_consume(&mut self, dst: &mut [u8]) { self.raw.unsafe_consume(dst) }
  #[inline(always)]
  unsafe fn unsafe_consume_be<T: Int>(&mut self) -> T { self.raw.unsafe_consume_be::<T>() }
  #[inline(always)]
  unsafe fn unsafe_consume_le<T: Int>(&mut self) -> T { self.raw.unsafe_consume_le::<T>() }

  #[inline(always)]
  unsafe fn as_raw<'b>(&'b self) -> &RawIobuf<'b> { mem::transmute(&self.raw) }

  #[inline(always)]
  fn ptr(&self) -> *mut u8 { self.raw.ptr() }
  #[inline(always)]
  fn is_owned(&self) -> bool { self.raw.is_owned() }
  #[inline(always)]
  fn lo_min(&self) -> u32 { self.raw.lo_min() }
  #[inline(always)]
  fn lo(&self) -> u32 { self.raw.lo() }
  #[inline(always)]
  fn hi(&self) -> u32 { self.raw.hi() }
  #[inline(always)]
  fn hi_max(&self) -> u32 { self.raw.hi_max() }
}

impl Iobuf for AROIobuf {
  #[inline(always)]
  fn deep_clone(&self) -> RWIobuf<'static> {
    RWIobuf { raw: self.raw.deep_clone() }
  }

  #[inline(always)]
  fn deep_clone_with_allocator(&self, allocator: Arc<Box<Allocator>>) -> RWIobuf<'static> {
    RWIobuf { raw: self.raw.deep_clone_with_allocator(allocator) }
  }

  #[inline(always)]
  fn unique(self) -> Result<UniqueIobuf, AROIobuf> {
    unsafe {
      if self.raw.is_unique_atomic() {
        Ok(mem::transmute(self))
      } else {
        Err(self)
      }
    }
  }

  #[inline(always)]
  fn atomic_read_only(self) -> Result<AROIobuf, AROIobuf> {
    Ok(self)
  }

  #[inline(always)]
  fn len(&self) -> u32 { self.raw.len() }

  #[inline(always)]
  fn cap(&self) -> u32 { self.raw.cap() }

  #[inline(always)]
  fn is_empty(&self) -> bool { self.raw.is_empty() }

  #[inline(always)]
  unsafe fn as_window_slice<'b>(&'b self) -> &'b [u8] { self.raw.as_window_slice() }

  #[inline(always)]
  unsafe fn as_limit_slice<'b>(&'b self) -> &'b [u8] { self.raw.as_limit_slice() }

  #[inline(always)]
  fn sub_window(&mut self, pos: u32, len: u32) -> Result<(), ()> { self.raw.sub_window(pos, len) }

  #[inline(always)]
  fn sub_window_from(&mut self, pos: u32) -> Result<(), ()> { self.raw.sub_window_from(pos) }

  #[inline(always)]
  fn sub_window_to(&mut self, len: u32) -> Result<(), ()> { self.raw.sub_window_to(len) }

  #[inline(always)]
  unsafe fn unsafe_sub_window(&mut self, pos: u32, len: u32) { self.raw.unsafe_sub_window(pos, len) }

  #[inline(always)]
  unsafe fn unsafe_sub_window_from(&mut self, pos: u32) { self.raw.unsafe_sub_window_from(pos) }

  #[inline(always)]
  unsafe fn unsafe_sub_window_to(&mut self, len: u32) { self.raw.unsafe_sub_window_to(len) }

  #[inline(always)]
  fn sub(&mut self, pos: u32, len: u32) -> Result<(), ()> { self.raw.sub(pos, len) }

  #[inline(always)]
  fn sub_from(&mut self, pos: u32) -> Result<(), ()> { self.raw.sub_from(pos) }

  #[inline(always)]
  fn sub_to(&mut self, len: u32) -> Result<(), ()> { self.raw.sub_to(len) }

  #[inline(always)]
  unsafe fn unsafe_sub(&mut self, pos: u32, len: u32) { self.raw.unsafe_sub(pos, len) }

  #[inline(always)]
  unsafe fn unsafe_sub_from(&mut self, pos: u32) { self.raw.unsafe_sub_from(pos) }

  #[inline(always)]
  unsafe fn unsafe_sub_to(&mut self, len: u32) { self.raw.unsafe_sub_to(len) }

  #[inline(always)]
  fn set_limits_and_window(&mut self, limits: (u32, u32), window: (u32, u32)) -> Result<(), ()> { self.raw.set_limits_and_window(limits, window) }

  #[inline(always)]
  fn narrow(&mut self) { self.raw.narrow() }

  #[inline(always)]
  fn advance(&mut self, len: u32) -> Result<(), ()> { self.raw.advance(len) }

  #[inline(always)]
  unsafe fn unsafe_advance(&mut self, len: u32) { self.raw.unsafe_advance(len) }

  #[inline(always)]
  fn extend(&mut self, len: u32) -> Result<(), ()> { self.raw.extend(len) }

  #[inline(always)]
  unsafe fn unsafe_extend(&mut self, len: u32) { self.raw.unsafe_extend(len) }

  #[inline(always)]
  fn is_extended_by<Buf: Iobuf>(&self, other: &Buf) -> bool { unsafe { self.raw.is_extended_by(other.as_raw()) } }

  #[inline(always)]
  fn extend_with<Buf: Iobuf>(&mut self, other: &Buf) -> Result<(), ()> { unsafe { self.raw.extend_with(other.as_raw()) } }

  #[inline(always)]
  fn resize(&mut self, len: u32) -> Result<(), ()> { self.raw.resize(len) }

  #[inline(always)]
  unsafe fn unsafe_resize(&mut self, len: u32) { self.raw.unsafe_resize(len) }

  #[inline(always)]
  fn split_at(&self, pos: u32) -> Result<(AROIobuf, AROIobuf), ()> {
    self.raw.split_at_atomic(pos).map(
      |(a, b)| (AROIobuf { raw: a },
                AROIobuf { raw: b }))
  }

  #[inline(always)]
  unsafe fn unsafe_split_at(&self, pos: u32) -> (AROIobuf, AROIobuf) {
    let (a, b) = self.raw.unsafe_split_at_atomic(pos);
    (AROIobuf { raw: a },
     AROIobuf { raw: b })
  }

  #[inline(always)]
  fn split_start_at(&mut self, pos: u32) -> Result<AROIobuf, ()> {
    self.raw.split_start_at_atomic(pos).map(
      |b| AROIobuf { raw: b })
  }

  #[inline(always)]
  unsafe fn unsafe_split_start_at(&mut self, pos: u32) -> AROIobuf {
    AROIobuf { raw: self.raw.unsafe_split_start_at_atomic(pos) }
  }

  #[inline(always)]
  fn rewind(&mut self) { self.raw.rewind() }

  #[inline(always)]
  fn reset(&mut self) { self.raw.reset() }

  #[inline(always)]
  fn flip_lo(&mut self) { self.raw.flip_lo() }

  #[inline(always)]
  fn flip_hi(&mut self) { self.raw.flip_hi() }

  #[inline(always)]
  fn lo_space(&self) -> u32 { self.raw.lo_space() }

  #[inline(always)]
  fn hi_space(&self) -> u32 { self.raw.hi_space() }

  #[inline(always)]
  fn peek(&self, pos: u32, dst: &mut [u8]) -> Result<(), ()> { self.raw.peek(pos, dst) }
  #[inline(always)]
  fn peek_be<T: Int>(&self, pos: u32) -> Result<T, ()> { self.raw.peek_be(pos) }
  #[inline(always)]
  fn peek_le<T: Int>(&self, pos: u32) -> Result<T, ()> { self.raw.peek_le(pos) }

  #[inline(always)]
  fn consume(&mut self, dst: &mut [u8]) -> Result<(), ()> { self.raw.consume(dst) }
  #[inline(always)]
  fn consume_be<T: Int>(&mut self) -> Result<T, ()> { self.raw.consume_be::<T>() }
  #[inline(always)]
  fn consume_le<T: Int>(&mut self) -> Result<T, ()> { self.raw.consume_le::<T>() }

  #[inline(always)]
  fn check_range(&self, pos: u32, len: u32) -> Result<(), ()> { self.raw.check_range_u32(pos, len) }

  #[inline(always)]
  fn check_range_usize(&self, pos: u32, len: usize) -> Result<(), ()> { self.raw.check_range_usize(pos, len) }

  #[inline(always)]
  fn check_range_fail(&self, pos: u32, len: u32) { self.raw.check_range_u32_fail(pos, len) }

  #[inline(always)]
  fn check_range_usize_fail(&self, pos: u32, len: usize) { self.raw.check_range_usize_fail(pos, len) }

  #[inline(always)]
  unsafe fn unsafe_peek(&self, pos: u32, dst: &mut [u8]) { self.raw.unsafe_peek(pos, dst) }
  #[inline(always)]
  unsafe fn unsafe_peek_be<T: Int>(&self, pos: u32) -> T { self.raw.unsafe_peek_be(pos) }
  #[inline(always)]
  unsafe fn unsafe_peek_le<T: Int>(&self, pos: u32) -> T { self.raw.unsafe_peek_le(pos) }

  #[inline(always)]
  unsafe fn unsafe_consume(&mut self, dst: &mut [u8]) { self.raw.unsafe_consume(dst) }
  #[inline(always)]
  unsafe fn unsafe_consume_be<T: Int>(&mut self) -> T { self.raw.unsafe_consume_be::<T>() }
  #[inline(always)]
  unsafe fn unsafe_consume_le<T: Int>(&mut self) -> T { self.raw.unsafe_consume_le::<T>() }

  #[inline(always)]
  unsafe fn as_raw<'b>(&'b self) -> &RawIobuf<'b> { mem::transmute(&self.raw) }

  #[inline(always)]
  fn ptr(&self) -> *mut u8 { self.raw.ptr() }
  #[inline(always)]
  fn is_owned(&self) -> bool { self.raw.is_owned() }
  #[inline(always)]
  fn lo_min(&self) -> u32 { self.raw.lo_min() }
  #[inline(always)]
  fn lo(&self) -> u32 { self.raw.lo() }
  #[inline(always)]
  fn hi(&self) -> u32 { self.raw.hi() }
  #[inline(always)]
  fn hi_max(&self) -> u32 { self.raw.hi_max() }
}

impl<'a> Iobuf for RWIobuf<'a> {
  #[inline(always)]
  fn deep_clone(&self) -> RWIobuf<'static> { RWIobuf { raw: self.raw.deep_clone() } }

  #[inline(always)]
  fn deep_clone_with_allocator(&self, allocator: Arc<Box<Allocator>>) -> RWIobuf<'static> {
    RWIobuf { raw: self.raw.deep_clone_with_allocator(allocator) }
  }

  #[inline(always)]
  fn unique(self) -> Result<UniqueIobuf, RWIobuf<'a>> {
    unsafe {
      if self.raw.is_unique_atomic() {
        Ok(mem::transmute(self))
      } else {
        Err(self)
      }
    }
  }

  #[inline(always)]
  fn atomic_read_only(self) -> Result<AROIobuf, RWIobuf<'a>> {
    unsafe {
      if self.raw.is_unique_nonatomic() {
        Ok(mem::transmute(self))
      } else {
        Err(self)
      }
    }
  }

  #[inline(always)]
  fn len(&self) -> u32 { self.raw.len() }

  #[inline(always)]
  fn cap(&self) -> u32 { self.raw.cap() }

  #[inline(always)]
  fn is_empty(&self) -> bool { self.raw.is_empty() }

  #[inline(always)]
  unsafe fn as_window_slice<'b>(&'b self) -> &'b [u8] { self.raw.as_window_slice() }

  #[inline(always)]
  unsafe fn as_limit_slice<'b>(&'b self) -> &'b [u8] { self.raw.as_limit_slice() }

  #[inline(always)]
  fn sub_window(&mut self, pos: u32, len: u32) -> Result<(), ()> { self.raw.sub_window(pos, len) }

  #[inline(always)]
  fn sub_window_from(&mut self, pos: u32) -> Result<(), ()> { self.raw.sub_window_from(pos) }

  #[inline(always)]
  fn sub_window_to(&mut self, len: u32) -> Result<(), ()> { self.raw.sub_window_to(len) }

  #[inline(always)]
  unsafe fn unsafe_sub_window(&mut self, pos: u32, len: u32) { self.raw.unsafe_sub_window(pos, len) }

  #[inline(always)]
  unsafe fn unsafe_sub_window_from(&mut self, pos: u32) { self.raw.unsafe_sub_window_from(pos) }

  #[inline(always)]
  unsafe fn unsafe_sub_window_to(&mut self, len: u32) { self.raw.unsafe_sub_window_to(len) }

  #[inline(always)]
  fn sub(&mut self, pos: u32, len: u32) -> Result<(), ()> { self.raw.sub(pos, len) }

  #[inline(always)]
  fn sub_from(&mut self, pos: u32) -> Result<(), ()> { self.raw.sub_from(pos) }

  #[inline(always)]
  fn sub_to(&mut self, len: u32) -> Result<(), ()> { self.raw.sub_to(len) }

  #[inline(always)]
  unsafe fn unsafe_sub(&mut self, pos: u32, len: u32) { self.raw.unsafe_sub(pos, len) }

  #[inline(always)]
  unsafe fn unsafe_sub_from(&mut self, pos: u32) { self.raw.unsafe_sub_from(pos) }

  #[inline(always)]
  unsafe fn unsafe_sub_to(&mut self, len: u32) { self.raw.unsafe_sub_to(len) }

  #[inline(always)]
  fn set_limits_and_window(&mut self, limits: (u32, u32), window: (u32, u32)) -> Result<(), ()> { self.raw.set_limits_and_window(limits, window) }

  #[inline(always)]
  fn narrow(&mut self) { self.raw.narrow() }

  #[inline(always)]
  fn advance(&mut self, len: u32) -> Result<(), ()> { self.raw.advance(len) }

  #[inline(always)]
  unsafe fn unsafe_advance(&mut self, len: u32) { self.raw.unsafe_advance(len) }

  #[inline(always)]
  fn extend(&mut self, len: u32) -> Result<(), ()> { self.raw.extend(len) }

  #[inline(always)]
  unsafe fn unsafe_extend(&mut self, len: u32) { self.raw.unsafe_extend(len) }

  #[inline(always)]
  fn is_extended_by<Buf: Iobuf>(&self, other: &Buf) -> bool { unsafe { self.raw.is_extended_by(other.as_raw()) } }

  #[inline(always)]
  fn extend_with<Buf: Iobuf>(&mut self, other: &Buf) -> Result<(), ()> { unsafe { self.raw.extend_with(other.as_raw()) } }

  #[inline(always)]
  fn resize(&mut self, len: u32) -> Result<(), ()> { self.raw.resize(len) }

  #[inline(always)]
  unsafe fn unsafe_resize(&mut self, len: u32) { self.raw.unsafe_resize(len) }

  #[inline(always)]
  fn split_at(&self, pos: u32) -> Result<(RWIobuf<'a>, RWIobuf<'a>), ()> {
    self.raw.split_at_nonatomic(pos).map(|(a, b)| (RWIobuf { raw: a }, RWIobuf { raw: b }))
  }

  #[inline(always)]
  unsafe fn unsafe_split_at(&self, pos: u32) -> (RWIobuf<'a>, RWIobuf<'a>) {
    let (a, b) = self.raw.unsafe_split_at_nonatomic(pos);
    (RWIobuf { raw: a }, RWIobuf { raw: b })
  }

  #[inline(always)]
  fn split_start_at(&mut self, pos: u32) -> Result<RWIobuf<'a>, ()> {
    self.raw.split_start_at_nonatomic(pos).map(|b| RWIobuf { raw: b })
  }

  #[inline(always)]
  unsafe fn unsafe_split_start_at(&mut self, pos: u32) -> RWIobuf<'a> {
    RWIobuf { raw: self.raw.unsafe_split_start_at_nonatomic(pos) }
  }

  #[inline(always)]
  fn rewind(&mut self) { self.raw.rewind() }

  #[inline(always)]
  fn reset(&mut self) { self.raw.reset() }

  #[inline(always)]
  fn flip_lo(&mut self) { self.raw.flip_lo() }

  #[inline(always)]
  fn flip_hi(&mut self) { self.raw.flip_hi() }

  #[inline(always)]
  fn lo_space(&self) -> u32 { self.raw.lo_space() }

  #[inline(always)]
  fn hi_space(&self) -> u32 { self.raw.hi_space() }

  #[inline(always)]
  fn peek(&self, pos: u32, dst: &mut [u8]) -> Result<(), ()> { self.raw.peek(pos, dst) }
  #[inline(always)]
  fn peek_be<T: Int>(&self, pos: u32) -> Result<T, ()> { self.raw.peek_be(pos) }
  #[inline(always)]
  fn peek_le<T: Int>(&self, pos: u32) -> Result<T, ()> { self.raw.peek_le(pos) }

  #[inline(always)]
  fn consume(&mut self, dst: &mut [u8]) -> Result<(), ()> { self.raw.consume(dst) }
  #[inline(always)]
  fn consume_be<T: Int>(&mut self) -> Result<T, ()> { self.raw.consume_be::<T>() }
  #[inline(always)]
  fn consume_le<T: Int>(&mut self) -> Result<T, ()> { self.raw.consume_le::<T>() }

  #[inline(always)]
  fn check_range(&self, pos: u32, len: u32) -> Result<(), ()> { self.raw.check_range_u32(pos, len) }

  #[inline(always)]
  fn check_range_usize(&self, pos: u32, len: usize) -> Result<(), ()> { self.raw.check_range_usize(pos, len) }

  #[inline(always)]
  fn check_range_fail(&self, pos: u32, len: u32) { self.raw.check_range_u32_fail(pos, len) }

  #[inline(always)]
  fn check_range_usize_fail(&self, pos: u32, len: usize) { self.raw.check_range_usize_fail(pos, len) }

  #[inline(always)]
  unsafe fn unsafe_peek(&self, pos: u32, dst: &mut [u8]) { self.raw.unsafe_peek(pos, dst) }
  #[inline(always)]
  unsafe fn unsafe_peek_be<T: Int>(&self, pos: u32) -> T { self.raw.unsafe_peek_be(pos) }
  #[inline(always)]
  unsafe fn unsafe_peek_le<T: Int>(&self, pos: u32) -> T { self.raw.unsafe_peek_le(pos) }

  #[inline(always)]
  unsafe fn unsafe_consume(&mut self, dst: &mut [u8]) { self.raw.unsafe_consume(dst) }
  #[inline(always)]
  unsafe fn unsafe_consume_be<T: Int>(&mut self) -> T { self.raw.unsafe_consume_be::<T>() }
  #[inline(always)]
  unsafe fn unsafe_consume_le<T: Int>(&mut self) -> T { self.raw.unsafe_consume_le::<T>() }

  #[inline(always)]
  unsafe fn as_raw<'b>(&'b self) -> &'b RawIobuf<'b> { mem::transmute(&self.raw) }

  #[inline(always)]
  fn ptr(&self) -> *mut u8 { self.raw.ptr() }
  #[inline(always)]
  fn is_owned(&self) -> bool { self.raw.is_owned() }
  #[inline(always)]
  fn lo_min(&self) -> u32 { self.raw.lo_min() }
  #[inline(always)]
  fn lo(&self) -> u32 { self.raw.lo() }
  #[inline(always)]
  fn hi(&self) -> u32 { self.raw.hi() }
  #[inline(always)]
  fn hi_max(&self) -> u32 { self.raw.hi_max() }
}

impl<'a> Show for ROIobuf<'a> {
  #[inline]
  fn fmt(&self, f: &mut Formatter) -> fmt::Result {
    self.raw.show(f, "read-only")
  }
}

impl<'a> Show for RWIobuf<'a> {
  #[inline]
  fn fmt(&self, f: &mut Formatter) -> fmt::Result {
    self.raw.show(f, "read-write")
  }
}

impl Show for AROIobuf {
  #[inline]
  fn fmt(&self, f: &mut Formatter) -> fmt::Result {
    self.raw.show(f, "atomic read-only")
  }
}

impl Show for UniqueIobuf {
  #[inline]
  fn fmt(&self, f: &mut Formatter) -> fmt::Result {
    self.raw.show(f, "unique")
  }
}

#[cfg(never)]
mod test {
  use impls::AROIobuf;
  use iobuf::Iobuf;

  use quickcheck::{quickcheck, Arbitrary, Gen, Shrinker};

    impl Arbitrary for AROIobuf {
      fn arbitrary<G: Gen>(g: &mut G) -> AROIobuf {
        let data: Vec<u8> = Arbitrary::arbitrary(g);

        assert!(data.len() < u32::MAX as usize);

        let (a, b, c, d): (u32, u32, u32, u32) = Arbitrary::arbitrary(g);

        let hi_max = a % (data.len() as u32 + 1);
        let hi     = b % (hi_max + 1);
        let lo     = c % (hi + 1);
        let lo_min = d % (lo + 1);

        let mut buf = ROIobuf::from_slice_copy(data[]);
        buf.set_limits_and_window((lo_min, hi_max), (lo, hi)).unwrap();

        buf.atomic_read_only().unwrap()
      }

      fn shrink(&self) -> Box<Shrinker<AROIobuf>+'static> {
        let mut v: Vec<AROIobuf> = vec!();

        // explore every possible subset of limits and bounds
        for hi_max in range(0, self.raw.hi_max()) {
          for hi in range(0, min(hi_max, self.raw.hi())) {
            for lo in range(self.raw.lo() + 1, hi + 1) {
              for lo_min in range(self.raw.lo_min() + 1, lo + 1) {
                let mut new_buf: AROIobuf = (*self).clone();
                new_buf.set_limits_and_window((lo_min, hi_max), (lo, hi)).unwrap();
                v.push(new_buf);
              }
            }
          }
        }

        box v.into_iter()
      }
    }


  #[test]
  fn prop_valid_hi() {
    fn test_hi(v: AROIobuf) -> bool {
         v.hi() <= v.hi_max()
    }
    quickcheck(test_hi as fn(AROIobuf) -> bool)
  }

  #[test]
  fn prop_valid_bounds() {
    fn test_bounds(v: AROIobuf) -> bool {
      v.lo() <= v.hi()
    }
    quickcheck(test_bounds as fn(AROIobuf) -> bool)
  }

  #[test]
  fn prop_valid_lo() {
    fn test_lo(v: AROIobuf) -> bool {
      v.lo_min() <= v.lo()
    }
    quickcheck(test_lo as fn(AROIobuf) -> bool)
  }
}
