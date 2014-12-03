use alloc::arc::Arc;
use alloc::boxed::Box;

use core::clone::Clone;
use core::fmt::Show;
use core::result::Result;

use raw::{Prim, Allocator, RawIobuf};
use impls::{AROIobuf, RWIobuf};

/// Input/Output Buffer
///
/// Have your functions take a generic IObuf when they don't modify the buffer
/// contents. This allows them to be used with both `ROIobuf`s and `RWIobuf`s.
///
/// `peek` accesses a value at a position relative to the start of the
/// window without advancing, and is meant to be used with `try!`. Its dual,
/// `poke`, is only implemented for `RWIobuf`, since it needs to write into the
/// buffer.
///
/// `consume` accesses a value at the beginning of the window, and advances the
/// window to no longer cover include it. Its dual, `fill`, is only implemented
/// for `RWIobuf`, since it needs to write into the buffer.
///
/// A suffix `_be` means the data will be read big-endian. A suffix `_le` means
/// the data will be read little-endian.
///
/// The `unsafe_` prefix means the function omits bounds checks. Misuse can
/// easily cause security issues. Be careful!
pub trait Iobuf: Clone + Show {
  /// Copies the data byte-by-byte in the Iobuf into a new, writeable Iobuf.
  /// The new Iobuf and the old Iobuf will not share storage.
  ///
  /// ```
  /// use iobuf::{RWIobuf,Iobuf};
  ///
  /// let mut s = [ 1, 2 ];
  ///
  /// let mut b = RWIobuf::from_slice(&mut s);
  /// let mut c = b.deep_clone();
  ///
  /// assert_eq!(b.poke_be(0, 0u8), Ok(()));
  /// assert_eq!(c.peek_be::<u8>(0), Ok(1u8));
  /// ```
  fn deep_clone(&self) -> RWIobuf<'static>;

  /// Copies the data byte-by-byte in the Iobuf into a new, writable Iobuf.
  /// The new Iobuf will have storage allocated out of `allocator`, and will not
  /// share the buffer with the original Iobuf.
  fn deep_clone_with_allocator(&self, allocator: Arc<Box<Allocator>>) -> RWIobuf<'static>;

  /// Returns `Ok` if the Iobuf is the last to reference the underlying data,
  /// and upgrades it to an `AROIobuf` which can be sent over channels and
  /// `Arc`ed with impunity. This is extremely useful in situations where Iobufs
  /// are created and written in one thread, and consumed in another.
  ///
  /// Only Iobufs which were originally allocated on the heap (for example, with
  /// a `_copy` constructor or `RWIobuf::new`) may be converted to an `AROIobuf`.
  ///
  /// Returns `Err` if the buffer is not the last to reference the underlying
  /// data. If this case is hit, the buffer passed by value is returned by value.
  ///
  /// ```
  /// use iobuf::{RWIobuf,Iobuf};
  ///
  /// let b = RWIobuf::from_str_copy("hello, world");
  /// assert!(b.atomic_read_only().is_ok());
  ///
  /// let b = RWIobuf::from_str_copy("hi");
  /// let c = b.clone();
  /// assert!(b.atomic_read_only().is_err());
  /// let d = c.clone();
  /// assert!(d.atomic_read_only().is_err());
  /// assert!(c.atomic_read_only().is_ok());
  /// ```
  fn atomic_read_only(self) -> Result<AROIobuf, Self>;

  /// Returns the size of the window.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("Hello");
  ///
  /// assert_eq!(b.len(), 5);
  /// assert_eq!(b.advance(2), Ok(()));
  /// assert_eq!(b.len(), 3);
  /// ```
  fn len(&self) -> u32;

  /// Returns the size of the limits. The capacity of an iobuf can be reduced
  /// via `narrow`.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("Hello");
  ///
  /// assert_eq!(b.cap(), 5);
  /// assert_eq!(b.advance(2), Ok(()));
  /// assert_eq!(b.cap(), 5);
  ///
  /// b.narrow();
  /// assert_eq!(b.cap(), 3);
  /// ```
  fn cap(&self) -> u32;

  /// `true` if `len() == 0`.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// assert!(ROIobuf::from_str("").is_empty());
  /// assert!(!ROIobuf::from_str("a").is_empty());
  /// ```
  fn is_empty(&self) -> bool;

  /// Reads the data in the window as an immutable slice. Note that `Peek`s
  /// and `Poke`s into the iobuf will change the contents of the slice, even
  /// though it advertises itself as immutable. Therefore, this function is
  /// `unsafe`.
  ///
  /// To use this function safely, you must manually ensure that the slice never
  /// interacts with the same Iobuf. If you take a slice of an Iobuf, you can
  /// immediately poke it into itself. This is unsafe, and undefined. However,
  /// it can be safely poked into a _different_ iobuf without issue.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// unsafe {
  ///   let mut b = ROIobuf::from_str("hello");
  ///   assert_eq!(b.as_window_slice(), b"hello");
  ///   assert_eq!(b.advance(2), Ok(()));
  ///   assert_eq!(b.as_window_slice(), b"llo");
  /// }
  /// ```
  unsafe fn as_window_slice<'b>(&'b self) -> &'b [u8];

  /// Reads the data in the limits as an immutable slice. Note that `Peek`s
  /// and `Poke`s into the iobuf will change the contents of the slice, even
  /// though it advertises itself as immutable. Therefore, this function is
  /// `unsafe`.
  ///
  /// To use this function safely, you must manually ensure that the slice never
  /// interacts with the same Iobuf. If you take a slice of an Iobuf, you can
  /// immediately poke it into itself. This is unsafe, and undefined. However,
  /// it can be safely poked into a _different_ iobuf without issue.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// unsafe {
  ///   let mut b = ROIobuf::from_str("hello");
  ///   assert_eq!(b.as_limit_slice(), b"hello");
  ///   assert_eq!(b.advance(2), Ok(()));
  ///   assert_eq!(b.as_limit_slice(), b"hello");
  ///   b.narrow();
  ///   assert_eq!(b.as_limit_slice(), b"llo");
  /// }
  /// ```
  unsafe fn as_limit_slice<'b>(&'b self) -> &'b [u8];

  /// Changes the Iobuf's bounds to the subrange specified by `pos` and `len`,
  /// which must lie within the current window.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("hello");
  /// assert_eq!(b.sub_window(1, 3), Ok(()));
  /// unsafe { assert_eq!(b.as_window_slice(), b"ell") };
  /// // The limits are unchanged. If you just want them to match the bounds, use
  /// // `sub` and friends.
  /// b.reset();
  /// unsafe { assert_eq!(b.as_window_slice(), b"hello") };
  /// ```
  ///
  /// If your position and length do not lie in the current window, you will get
  /// an error.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("hello");
  /// assert_eq!(b.advance(2), Ok(()));
  /// assert_eq!(b.sub_window(0, 5), Err(())); // boom
  /// ```
  ///
  /// If you want to slice from the start, use `sub_to`:
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("hello");
  /// assert_eq!(b.sub_window_to(3), Ok(()));
  /// unsafe { assert_eq!(b.as_window_slice(), b"hel") };
  /// ```
  ///
  /// If you want to slice to the end, use `sub_from`:
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("hello");
  /// assert_eq!(b.sub_window_from(2), Ok(()));
  /// unsafe { assert_eq!(b.as_window_slice(), b"llo") };
  /// ```
  fn sub_window(&mut self, pos: u32, len: u32) -> Result<(), ()>;

  /// Changes the Iobuf's bounds to start at `pos`, and go to the end of the
  /// current window.
  fn sub_window_from(&mut self, pos: u32) -> Result<(), ()>;

  /// Changes the Iobuf's bounds to extend for only `len` bytes.
  ///
  /// This is the same as `resize`, but might make more semantic sense at the
  /// call site depending on context.
  fn sub_window_to(&mut self, len: u32) -> Result<(), ()>;

  /// The same as `sub_window`, but no bounds checks are performed. You should
  /// probably just use `sub_window`.
  unsafe fn unsafe_sub_window(&mut self, pos: u32, len: u32);

  /// The same as `sub_window_from`, but no bounds checks are performed. You
  /// should probably just use `sub_window_from`.
  unsafe fn unsafe_sub_window_from(&mut self, pos: u32);

  /// The same as `sub_window_to`, but no bounds checks are performed. You
  /// should probably just use `sub_window_to`.
  unsafe fn unsafe_sub_window_to(&mut self, pos: u32);

  /// Changes the Iobuf's limits and bounds to the subrange specified by
  /// `pos` and `len`, which must lie within the current window.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("hello");
  /// assert_eq!(b.sub(1, 3), Ok(()));
  /// unsafe { assert_eq!(b.as_window_slice(), b"ell") };
  ///
  /// // The limits are changed, too! If you just want to change the bounds, use
  /// // `sub_window` and friends.
  /// b.reset();
  /// unsafe { assert_eq!(b.as_window_slice(), b"ell") };
  /// ```
  ///
  /// If your position and length do not lie in the current window, you will get
  /// an error.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("hello");
  /// assert_eq!(b.advance(2), Ok(()));
  /// assert_eq!(b.sub(0, 5), Err(())); // boom
  /// ```
  ///
  /// If you want to slice from the start, use `sub_to`:
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("hello");
  /// assert_eq!(b.sub_to(3), Ok(()));
  /// unsafe { assert_eq!(b.as_window_slice(), b"hel") };
  /// ```
  ///
  /// If you want to slice to the end, use `sub_from`:
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("hello");
  /// assert_eq!(b.sub_from(2), Ok(()));
  /// unsafe { assert_eq!(b.as_window_slice(), b"llo") };
  /// ```
  fn sub(&mut self, pos: u32, len: u32) -> Result<(), ()>;

  /// Changes the Iobuf's limits and bounds to start from `pos` and extend to
  /// the end of the current window.
  fn sub_from(&mut self, pos: u32) -> Result<(), ()>;

  /// Changes the Iobuf's limits and bounds to start at the beginning of the
  /// current window, and extend for `len` bytes.
  fn sub_to(&mut self, len: u32) -> Result<(), ()>;

  /// The same as `sub`, but no bounds checks are performed. You should probably
  /// just use `sub`.
  unsafe fn unsafe_sub(&mut self, pos: u32, len: u32);

  /// The same as `sub_from`, but no bounds checks are performed. You should
  /// probably just use `sub_from`.
  unsafe fn unsafe_sub_from(&mut self, pos: u32);

  /// The same as `sub_to`, but no bounds checks are performed. You should
  /// probably just use `sub_to`.
  unsafe fn unsafe_sub_to(&mut self, len: u32);

  /// Overrides the existing limits and window of the Iobuf, returning `Err(())`
  /// if attempting to widen either of them.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("hello");
  /// assert_eq!(b.set_limits_and_window((1, 3), (2, 3)), Ok(()));
  /// assert_eq!(b.cap(), 2);
  /// assert_eq!(b.len(), 1);
  /// // trying to shrink the limits...
  /// assert_eq!(b.set_limits_and_window((1, 4), (2, 2)), Err(()));
  /// // trying to shrink the window...
  /// assert_eq!(b.set_limits_and_window((1, 3), (2, 4)), Err(()));
  /// ```
  fn set_limits_and_window(&mut self, limits: (u32, u32), window: (u32, u32)) -> Result<(), ()>;

  /// Sets the limits to the current window.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("hello");
  /// assert_eq!(b.advance(2), Ok(()));
  /// b.narrow();
  /// assert_eq!(b.cap(), 3);
  /// ```
  fn narrow(&mut self);

  /// Advances the lower bound of the window by `len`. `Err(())` will be
  /// returned if you advance past the upper bound of the window.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("hello");
  /// assert_eq!(b.advance(3), Ok(()));
  /// assert_eq!(b.advance(3), Err(()));
  /// unsafe { assert_eq!(b.as_window_slice(), b"lo"); }
  /// ```
  fn advance(&mut self, len: u32) -> Result<(), ()>;

  /// Advances the lower bound of the window by `len`. No bounds checking will
  /// be performed.
  ///
  /// A common pattern with `unsafe_advance` is to consolidate multiple bounds
  /// checks into one. In this example, O(n) bounds checks are consolidated into
  /// O(1) bounds checks:
  ///
  /// ```
  /// use std::mem;
  /// use std::result::{Result,Ok};
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let data = [2, 0x12, 0x34, 0x56, 0x78];
  /// let mut b = ROIobuf::from_slice(&data);
  ///
  /// fn parse<B: Iobuf>(b: &mut B) -> Result<u16, ()> {
  ///   let num_shorts: u8 = try!(b.consume_be());
  ///   let short_size = mem::size_of::<u16>() as u32;
  ///   let num_bytes = num_shorts as u32 * short_size;
  ///
  ///   unsafe {
  ///     try!(b.check_range(0, num_bytes));
  ///
  ///     let mut sum = 0u16;
  ///
  ///     for i in range(0, num_shorts as u32).map(|x| x * short_size) {
  ///       sum += b.unsafe_peek_be(i);
  ///     }
  ///
  ///     b.unsafe_advance(num_bytes);
  ///
  ///     Ok(sum)
  ///   }
  /// }
  ///
  /// assert_eq!(parse(&mut b), Ok(0x1234 + 0x5678));
  /// assert_eq!(b.len(), 0);
  /// ```
  ///
  /// Alternatively, you could use `unsafe_consume` in a similar, arguably
  /// clearer way:
  ///
  /// ```
  /// use std::mem;
  /// use std::result::{Result,Ok};
  /// use iobuf::{ROIobuf,Iobuf};
  /// let data = [2, 0x12, 0x34, 0x56, 0x78];
  /// let mut b = ROIobuf::from_slice(&data);
  ///
  /// fn parse<B: Iobuf>(b: &mut B) -> Result<u16, ()> {
  ///   let num_shorts: u8 = try!(b.consume_be());
  ///   let short_size = mem::size_of::<u16>() as u32;
  ///   let num_bytes = num_shorts as u32 * short_size;
  ///
  ///   unsafe {
  ///     try!(b.check_range(0, num_bytes));
  ///
  ///     let mut sum = 0u16;
  ///
  ///     for _ in range(0, num_shorts) {
  ///       sum += b.unsafe_consume_be();
  ///     }
  ///
  ///     Ok(sum)
  ///   }
  /// }
  ///
  /// assert_eq!(parse(&mut b), Ok(0x1234 + 0x5678));
  /// assert_eq!(b.len(), 0);
  /// ```
  unsafe fn unsafe_advance(&mut self, len: u32);

  /// Advances the upper bound of the window by `len`. `Err(())` will be
  /// returned if you advance past the upper limit.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("hello");
  /// b.resize(2).unwrap();
  /// assert_eq!(b.extend(1), Ok(()));
  /// unsafe { assert_eq!(b.as_window_slice(), b"hel"); }
  /// assert_eq!(b.extend(3), Err(()));
  /// unsafe { assert_eq!(b.as_window_slice(), b"hel"); }
  /// ```
  fn extend(&mut self, len: u32) -> Result<(), ()>;

  /// Advances the upper bound of the window by `len`. No bounds checking will
  /// be performed.
  unsafe fn unsafe_extend(&mut self, len: u32);

  /// Returns `true` if the `other` Iobuf's window is the region directly after
  /// our window. This does not inspect the buffer -- it only compares raw
  /// pointers.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut a = ROIobuf::from_str_copy("hello");
  /// let mut b = a.clone();
  /// let mut c = a.clone();
  /// let mut d = ROIobuf::from_str_copy("hello");
  ///
  /// assert_eq!(a.sub_window_to(2), Ok(()));
  ///
  /// // b actually IS an extension of a.
  /// assert_eq!(b.sub_window_from(2), Ok(()));
  /// assert_eq!(a.is_extended_by(&b), true);
  ///
  /// // a == "he", b == "lo", it's missing the "l", therefore not an extension.
  /// assert_eq!(c.sub_window_from(3), Ok(()));
  /// assert_eq!(b.is_extended_by(&a), false);
  ///
  /// // Different allocations => not an extension.
  /// assert_eq!(d.sub_window_from(2), Ok(()));
  /// assert_eq!(a.is_extended_by(&d), false);
  /// ```
  fn is_extended_by<Buf: Iobuf>(&self, other: &Buf) -> bool;

  /// Attempts to extend an Iobuf with the contents of another Iobuf. If this
  /// Iobuf's window is not the region directly before the other Iobuf's window,
  /// no extension will be performed and `Err(())` will be returned. If the
  /// operation was successful, `Ok(())` will be returned.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut a = ROIobuf::from_str_copy("hello");
  /// let mut b = a.clone();
  /// let mut c = a.clone();
  /// let mut d = ROIobuf::from_str_copy("hello");
  ///
  /// assert_eq!(a.sub_window_to(2), Ok(()));
  ///
  /// // Different allocations => not an extension.
  /// assert_eq!(d.sub_window_from(2), Ok(()));
  /// assert_eq!(a.extend_with(&d), Err(()));
  ///
  /// // b actually IS an extension of a.
  /// assert_eq!(b.sub_window_from(2), Ok(()));
  /// assert_eq!(a.extend_with(&b), Ok(()));
  /// unsafe {
  ///   assert_eq!(a.as_window_slice(), b"hello");
  /// }
  ///
  /// assert_eq!(a.sub_window_to(2), Ok(()));
  ///
  /// // a == "he", b == "lo", it's missing the "l", therefore not an extension.
  /// assert_eq!(c.sub_window_from(3), Ok(()));
  /// assert_eq!(b.extend_with(&a), Err(()));
  /// ```
  fn extend_with<Buf: Iobuf>(&mut self, other: &Buf) -> Result<(), ()>;

  /// Sets the length of the window, provided it does not exceed the limits.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("hello");
  /// assert_eq!(b.resize(3), Ok(()));
  /// assert_eq!(b.peek_be(2), Ok(b'l'));
  /// assert_eq!(unsafe { b.as_window_slice() }, b"hel");
  /// assert_eq!(b.peek_be::<u8>(3), Err(()));
  /// assert_eq!(b.advance(1), Ok(()));
  /// assert_eq!(b.resize(5), Err(()));
  /// ```
  fn resize(&mut self, len: u32) -> Result<(), ()>;

  /// Sets the length of the window. No bounds checking will be performed.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("hello");
  ///
  /// unsafe {
  ///   assert_eq!(b.check_range(1, 3), Ok(()));
  ///   assert_eq!(b.peek_be(1), Ok(b'e'));
  ///   assert_eq!(b.peek_be(2), Ok(b'l'));
  ///   assert_eq!(b.peek_be(3), Ok(b'l'));
  ///   b.unsafe_resize(4); // safe, since we already checked it.
  /// }
  /// ```
  unsafe fn unsafe_resize(&mut self, len: u32);

  /// Splits an Iobuf around an index.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("helloworld");
  ///
  /// match b.split_at(5) {
  ///   Err(())    => panic!("This won't happen."),
  ///   Ok((c, d)) => unsafe {
  ///     assert_eq!(c.as_window_slice(), b"hello");
  ///     assert_eq!(d.as_window_slice(), b"world");
  ///   }
  /// }
  ///
  /// match b.split_at(0) {
  ///   Err(())    => panic!("This won't happen, either."),
  ///   Ok((c, d)) => unsafe {
  ///     assert_eq!(c.as_window_slice(), b"");
  ///     assert_eq!(d.as_window_slice(), b"helloworld");
  ///   }
  /// }
  ///
  /// match b.split_at(10000) {
  ///   Ok(_)   => panic!("This won't happen!"),
  ///   Err(()) => unsafe { assert_eq!(b.as_window_slice(), b"helloworld"); },
  /// }
  /// ```
  fn split_at(&self, pos: u32) -> Result<(Self, Self), ()>;

  /// Like `split_at`, but does not perform bounds checking.
  unsafe fn unsafe_split_at(&self, pos: u32) -> (Self, Self);

  /// Splits out the start of an Iobuf at an index.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("helloworld");
  ///
  /// match b.split_start_at(5) {
  ///   Err(()) => panic!("This won't happen."),
  ///   Ok(c)   => unsafe {
  ///     assert_eq!(b.as_window_slice(), b"world");
  ///     assert_eq!(c.as_window_slice(), b"hello");
  ///   }
  /// }
  ///
  /// match b.split_start_at(0) {
  ///   Err(()) => panic!("This won't happen, either."),
  ///   Ok(c)   => unsafe {
  ///     assert_eq!(b.as_window_slice(), b"world");
  ///     assert_eq!(c.as_window_slice(), b"");
  ///   }
  /// }
  ///
  /// match b.split_start_at(10000) {
  ///   Ok(_)   => panic!("This won't happen!"),
  ///   Err(()) => unsafe { assert_eq!(b.as_window_slice(), b"world"); },
  /// }
  /// ```
  fn split_start_at(&mut self, pos: u32) -> Result<Self, ()>;

  /// Like `split_start_at`, but does not perform bounds checking.
  unsafe fn unsafe_split_start_at(&mut self, pos: u32) -> Self;

  /// Sets the lower bound of the window to the lower limit.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("hello");
  ///
  /// assert_eq!(b.advance(2), Ok(()));
  /// assert_eq!(b.resize(2), Ok(()));
  ///
  /// unsafe { assert_eq!(b.as_window_slice(), b"ll"); }
  ///
  /// b.rewind();
  ///
  /// unsafe { assert_eq!(b.as_window_slice(), b"hell"); }
  /// ```
  fn rewind(&mut self);

  /// Sets the window to the limits.
  ///
  /// "Take it to the limit..."
  ///
  /// ```
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
  fn reset(&mut self);

  /// Sets the window to range from the lower limit to the lower bound of the
  /// old window. This is typically called after a series of `Fill`s, to
  /// reposition the window in preparation to `Consume` the newly written data.
  ///
  /// If the area of the limits is denoted with `[ ]` and the area of the window
  /// is denoted with `x`, then the `flip_lo` looks like:
  ///
  /// `Before: [       xxxx  ]`
  ///
  /// `After:  [xxxxxxx      ]`
  ///
  /// ```
  /// use iobuf::{RWIobuf,Iobuf};
  ///
  /// let mut b = RWIobuf::new(4);
  ///
  /// assert_eq!(b.fill_be(1u8), Ok(()));
  /// assert_eq!(b.fill_be(2u8), Ok(()));
  /// assert_eq!(b.fill_be(3u8), Ok(()));
  /// assert_eq!(b.len(), 1);
  ///
  /// b.flip_lo();
  ///
  /// assert_eq!(b.consume_be(), Ok(0x0102u16));
  /// assert_eq!(b.consume_be(), Ok(0x03u8));
  /// assert!(b.is_empty());
  /// ```
  fn flip_lo(&mut self);

  /// Sets the window to range from the upper bound of the old window to the
  /// upper limit. This is a dual to `flip_lo`, and is typically called when the
  /// data in the current (narrowed) window has been processed and the window
  /// needs to be positioned over the remaining data in the buffer.
  ///
  /// If the area of the limits is denoted with `[ ]` and the area of the window
  /// is denoted with `x`, then the `flip_lo` looks like:
  ///
  /// `Before: [       xxxx  ]`
  ///
  /// `After:  [           xx]`
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("hello");
  ///
  /// assert_eq!(b.resize(3), Ok(()));
  ///
  /// assert_eq!(b.consume_be(), Ok(b'h'));
  /// assert_eq!(b.consume_be(), Ok(b'e'));
  /// assert_eq!(b.consume_be(), Ok(b'l'));
  /// assert!(b.is_empty());
  ///
  /// b.flip_hi();
  ///
  /// assert!(!b.is_empty());
  /// assert_eq!(b.consume_be(), Ok(b'l'));
  /// assert_eq!(b.consume_be(), Ok(b'o'));
  /// assert!(b.is_empty());
  /// ```
  fn flip_hi(&mut self);

  /// Returns the number of bytes between the lower limit and the lower bound.
  ///
  /// ```
  /// use iobuf::{RWIobuf,Iobuf};
  ///
  /// let mut b = RWIobuf::new(100);
  ///
  /// assert_eq!(b.advance(20), Ok(()));
  /// assert_eq!(b.lo_space(), 20);
  /// b.flip_lo();
  /// assert_eq!(b.lo_space(), 0);
  /// ```
  fn lo_space(&self) -> u32;

  /// Returns the number of bytes between the upper bound and the upper limit.
  ///
  /// ```
  /// use iobuf::{RWIobuf,Iobuf};
  ///
  /// let mut b = RWIobuf::new(100);
  ///
  /// assert_eq!(b.resize(20), Ok(()));
  /// assert_eq!(b.hi_space(), 80);
  /// b.flip_hi();
  /// assert_eq!(b.hi_space(), 0);
  /// ```
  fn hi_space(&self) -> u32;

  /// Reads the bytes at a given offset from the beginning of the window, into
  /// the supplied buffer. Either the entire buffer is filled, or an error is
  /// returned because bytes outside of the window were requested.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  /// use std::iter::AdditiveIterator;
  ///
  /// let data = [ 0x01, 0x02, 0x03, 0x04 ];
  ///
  /// let mut b = ROIobuf::from_slice(&data);
  /// let mut tgt4 = [ 0x00, 0x00, 0x00, 0x00 ];
  /// let mut tgt3 = [ 0x00, 0x00, 0x00 ];
  ///
  /// assert_eq!(b.peek(0, &mut tgt4), Ok(()));
  /// assert_eq!(tgt4.iter().map(|&x| x).sum(), 10);
  /// assert_eq!(b.peek(1, &mut tgt3), Ok(()));
  /// assert_eq!(tgt3.iter().map(|&x| x).sum(), 9);
  /// assert_eq!(b.peek(1, &mut tgt4), Err(()));
  /// ```
  fn peek(&self, pos: u32, dst: &mut [u8]) -> Result<(), ()>;

  /// Reads a big-endian primitive at a given offset from the beginning of the
  /// window.
  ///
  /// An error is returned if bytes outside of the window were requested.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let data = [ 0x01, 0x02, 0x03, 0x04 ];
  /// let mut b = ROIobuf::from_slice(&data);
  ///
  /// assert_eq!(b.advance(1), Ok(()));
  ///
  /// assert_eq!(b.peek_be(0), Ok(0x0203u16));
  /// assert_eq!(b.peek_be(1), Ok(0x0304u16));
  /// assert_eq!(b.peek_be::<u16>(2), Err(()));
  /// ```
  fn peek_be<T: Prim>(&self, pos: u32) -> Result<T, ()>;

  /// Reads a little-endian primitive at a given offset from the beginning of
  /// the window.
  ///
  /// An error is returned if bytes outside of the window were requested.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let data = [ 0x01, 0x02, 0x03, 0x04 ];
  /// let mut b = ROIobuf::from_slice(&data);
  ///
  /// assert_eq!(b.advance(1), Ok(()));
  ///
  /// assert_eq!(b.peek_le(0), Ok(0x0302u16));
  /// assert_eq!(b.peek_le(1), Ok(0x0403u16));
  /// assert_eq!(b.peek_le::<u16>(2), Err(()));
  /// ```
  fn peek_le<T: Prim>(&self, pos: u32) -> Result<T, ()>;

  /// Reads bytes, starting from the front of the window, into the supplied
  /// buffer. Either the entire buffer is filled, or an error is returned
  /// because bytes outside the window were requested.
  ///
  /// After the bytes have been read, the window will be moved to no longer
  /// include then.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  /// use std::iter::AdditiveIterator;
  ///
  /// let data = [ 0x01, 0x02, 0x03, 0x04 ];
  ///
  /// let mut b = ROIobuf::from_slice(&data);
  /// let mut tgt3 = [ 0x00, 0x00, 0x00 ];
  /// let mut tgt1 = [ 0x00 ];
  ///
  /// assert_eq!(b.consume(&mut tgt3), Ok(()));
  /// assert_eq!(tgt3.iter().map(|&x| x).sum(), 6);
  /// assert_eq!(b.consume(&mut tgt3), Err(()));
  /// assert_eq!(b.consume(&mut tgt1), Ok(()));
  /// assert_eq!(tgt1[0], 4);
  /// ```
  fn consume(&mut self, dst: &mut [u8]) -> Result<(), ()>;

  /// Reads a big-endian primitive from the beginning of the window.
  ///
  /// After the primitive has been read, the window will be moved such that it
  /// is no longer included.
  ///
  /// An error is returned if bytes outside of the window were requested.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let data = [ 0x01, 0x02, 0x03, 0x04 ];
  /// let mut b = ROIobuf::from_slice(&data);
  ///
  /// assert_eq!(b.advance(1), Ok(()));
  ///
  /// assert_eq!(b.consume_be(), Ok(0x0203u16));
  /// assert_eq!(b.consume_be::<u16>(), Err(()));
  /// assert_eq!(b.consume_be(), Ok(0x04u8));
  /// ```
  fn consume_be<T: Prim>(&mut self) -> Result<T, ()>;

  /// Reads a little-endian primitive from the beginning of the window.
  ///
  /// After the primitive has been read, the window will be moved such that it
  /// is no longer included.
  ///
  /// An error is returned if bytes outside of the window were requested.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let data = [ 0x01, 0x02, 0x03, 0x04 ];
  /// let mut b = ROIobuf::from_slice(&data);
  ///
  /// assert_eq!(b.advance(1), Ok(()));
  ///
  /// assert_eq!(b.consume_le(), Ok(0x0302u16));
  /// assert_eq!(b.consume_le::<u16>(), Err(()));
  /// assert_eq!(b.consume_le(), Ok(0x04u8));
  /// ```
  fn consume_le<T: Prim>(&mut self) -> Result<T, ()>;

  /// Returns an `Err(())` if the `len` bytes, starting at `pos`, are not all
  /// in the window. To be used with the `try!` macro.
  ///
  /// Make sure you use this in conjunction with the `unsafe` combinators. It
  /// is recommended you minimize your bounds checks by doing it once with
  /// `check_range`, followed by unsafe accesses. If you do unsafe accesses
  /// without a `check_range`, there will likely be reliability and security
  /// issues with your application.
  ///
  /// Below is a correct usage of `check_range` to minimize bounds checks:
  ///
  /// ```
  /// use std::result::{Result,Ok};
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// // [ number of byte buffers, size of first byte buffer, ...bytes, etc. ]
  /// let data = [ 0x02, 0x02, 0x55, 0x66, 0x03, 0x11, 0x22, 0x33 ];
  /// let mut b = ROIobuf::from_slice(&data);
  ///
  /// // Returns the sum of the bytes, omitting as much bounds checking as
  /// // possible while still maintaining safety.
  /// fn parse<B: Iobuf>(b: &mut B) -> Result<uint, ()> {
  ///   let mut sum = 0u;
  ///
  ///   let num_buffers: u8 = try!(b.consume_be());
  ///
  ///   for _ in range(0, num_buffers) {
  ///     let len: u8 = try!(b.consume_be());
  ///
  ///     unsafe {
  ///       try!(b.check_range(0, len as u32));
  ///
  ///       for _ in range(0, len) {
  ///         sum += b.unsafe_consume_be::<u8>() as uint;
  ///       }
  ///     }
  ///   }
  ///
  ///   Ok(sum)
  /// }
  ///
  /// assert_eq!(parse(&mut b), Ok(0x55 + 0x66 + 0x11 + 0x22 + 0x33));
  /// ```
  fn check_range(&self, pos: u32, len: u32) -> Result<(), ()>;

  /// The same as `check_range`, but with a `uint` length. If you're checking
  /// the range of something which might overflow an `i32`, use this version
  /// instead of `check_range`.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("hello");
  ///
  /// assert_eq!(b.check_range_uint(1u32, 5u), Err(()));
  /// ```
  fn check_range_uint(&self, pos: u32, len: uint) -> Result<(), ()>;

  /// The same as `check_range`, but fails if the bounds check returns `Err(())`.
  ///
  /// ```should_fail
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("hello");
  ///
  /// b.check_range_fail(1, 5); // boom.
  /// ```
  fn check_range_fail(&self, pos: u32, len: u32);

  /// The same as `check_range_uint`, but fails if the bounds check returns
  /// `Err(())`.
  ///
  /// ```should_fail
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("hello");
  ///
  /// b.check_range_uint_fail(1u32, 5u); // boom.
  /// ```
  fn check_range_uint_fail(&self, pos: u32, len: uint);

  /// Reads the bytes at a given offset from the beginning of the window, into
  /// the supplied buffer. It is undefined behavior to read outside the iobuf
  /// window.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let data = [1,2,3,4,5,6];
  /// let mut b = ROIobuf::from_slice(&data);
  ///
  /// let mut dst = [0, ..4];
  ///
  /// unsafe {
  ///   // one range check, instead of two!
  ///   assert_eq!(b.check_range(0, 5), Ok(()));
  ///
  ///   assert_eq!(b.unsafe_peek_be::<u8>(0), 1u8);
  ///   b.unsafe_peek(1, &mut dst);
  /// }
  ///
  /// let expected = [ 2u8, 3, 4, 5 ];
  /// assert_eq!(dst.as_slice(), expected.as_slice());
  /// ```
  unsafe fn unsafe_peek(&self, pos: u32, dst: &mut [u8]);

  /// Reads a big-endian primitive at a given offset from the beginning of the
  /// window. It is undefined behavior to read outside the iobuf window.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let data = [1,2,3,4,5,6];
  /// let mut b = ROIobuf::from_slice(&data);
  ///
  /// unsafe {
  ///   assert_eq!(b.check_range(0, 6), Ok(()));
  ///   let x: u16 = b.unsafe_peek_be(0);
  ///   let y: u32 = b.unsafe_peek_be(2);
  ///   b.unsafe_advance(6);
  ///
  ///   let z: u32 = x as u32 + y;
  ///   assert_eq!(z, 0x0102 + 0x03040506);
  /// }
  /// ```
  unsafe fn unsafe_peek_be<T: Prim>(&self, pos: u32) -> T;

  /// Reads a little-endian primitive at a given offset from the beginning of
  /// the window. It is undefined behavior to read outside the iobuf window.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let data = [1,2,3,4,5,6];
  /// let mut b = ROIobuf::from_slice(&data);
  ///
  /// unsafe {
  ///   assert_eq!(b.check_range(0, 6), Ok(()));
  ///   let x: u16 = b.unsafe_peek_le(0);
  ///   let y: u32 = b.unsafe_peek_le(2);
  ///   b.unsafe_advance(6);
  ///
  ///   let z: u32 = x as u32 + y;
  ///   assert_eq!(z, 0x0201 + 0x06050403);
  /// }
  /// ```
  unsafe fn unsafe_peek_le<T: Prim>(&self, pos: u32) -> T;

  /// Reads bytes, starting from the front of the window, into the supplied
  /// buffer. After the bytes have been read, the window will be moved to no
  /// longer include then.
  ///
  /// It is undefined behavior to request bytes outside the iobuf window.
  unsafe fn unsafe_consume(&mut self, dst: &mut [u8]);

  /// Reads a big-endian primitive at the beginning of the window.
  ///
  /// After the primitive has been read, the window will be moved such that it
  /// is no longer included.
  ///
  /// It is undefined behavior if bytes outside the window are requested.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let data = [ 0x01, 0x02, 0x03, 0x04 ];
  /// let mut b = ROIobuf::from_slice(&data);
  ///
  /// assert_eq!(b.advance(1), Ok(()));
  ///
  /// unsafe {
  ///   assert_eq!(b.check_range(0, 3), Ok(()));
  ///   assert_eq!(b.unsafe_consume_be::<u16>(), 0x0203u16);
  ///   assert_eq!(b.unsafe_consume_be::<u8>(), 0x04u8);
  /// }
  /// ```
  unsafe fn unsafe_consume_be<T: Prim>(&mut self) -> T;

  /// Reads a little-endian primitive at the beginning of the window.
  ///
  /// After the primitive has been read, the window will be moved such that it
  /// is no longer included.
  ///
  /// It is undefined behavior if bytes outside of the window are requested.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let data = [ 0x01, 0x02, 0x03, 0x04 ];
  /// let mut b = ROIobuf::from_slice(&data);
  ///
  /// assert_eq!(b.advance(1), Ok(()));
  ///
  /// unsafe {
  ///   assert_eq!(b.check_range(0, 3), Ok(()));
  ///   assert_eq!(b.unsafe_consume_le::<u16>(), 0x0302u16);
  ///   assert_eq!(b.unsafe_consume_le::<u8>(), 0x04u8);
  /// }
  /// ```
  unsafe fn unsafe_consume_le<T: Prim>(&mut self) -> T;

  /// For internal use only.
  unsafe fn as_raw<'a>(&self) -> &RawIobuf<'a>;

  /// Gets a pointer to the start of the internal backing buffer. This is
  /// extremely low level, and it is not recommended you use this interface.
  fn ptr(&self) -> *mut u8;

  /// Returns `true` if the Iobuf points to owned memory (i.e. has to do a
  /// refcount modification on `clone` or `drop`) or borrowed memory.
  fn is_owned(&self) -> bool;

  /// Returns an index into the buffer returned by `ptr` that represents the
  /// inclusive lower bound of the limits.
  fn lo_min(&self) -> u32;

  /// Returns an index into the buffer returned by `ptr` that represents the
  /// inclusive lower bound of the window.
  fn lo(&self) -> u32;

  /// Returns an index into the buffer returned by `ptr` that represents the
  /// exclusive upper bound of the window.
  fn hi(&self) -> u32;

  /// Returns an index into the buffer returned by `ptr` that represents the
  /// exclusive upper bound of the limits.
  fn hi_max(&self) -> u32;

}
