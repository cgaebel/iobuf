//! A contiguous region of bytes, useful for I/O operations.
//!
//! An Iobuf consists of:
//!
//!   - buffer
//!   - limits (a subrange of the buffer)
//!   - window (a subrange of the limits)
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

#![license = "MIT"]

use std::cell::UnsafeCell;
use std::fmt::{Formatter,FormatError,Show};
use std::iter;
use std::mem;
use std::num::Zero;
use std::ptr;
use std::raw;
use std::rc::Rc;
use std::result::{Result,Ok,Err};

/// A generic, over all built-in number types. Think of it as [u,i][8,16,32,64].
///
/// `Prim` is intentionally not implemented for `int` and `uint`, since these
/// have no portable representation. Are they 32 or 64 bits? No one knows.
pub trait Prim
  : Copy
  + Zero
  + Shl<uint, Self>
  + Shr<uint, Self>
  + BitOr<Self, Self>
  + BitAnd<Self, Self>
  + FromPrimitive
  + ToPrimitive
{}

impl Prim for i8  {}
impl Prim for u8  {}
impl Prim for i16 {}
impl Prim for u16 {}
impl Prim for i32 {}
impl Prim for u32 {}
impl Prim for i64 {}
impl Prim for u64 {}

/// Both owned vectors of memory and slices of memory are supported.
/// In the case of a `ROIobuf`, although there is mutable slice access, it is
/// never used.
enum MaybeOwnedBuffer<'a> {
  OwnedBuffer(Vec<u8>),
  BorrowedBuffer(&'a mut [u8]),
}

impl<'a> MaybeOwnedBuffer<'a> {
  #[inline]
  unsafe fn as_slice(&self) -> &[u8] {
    match *self {
      OwnedBuffer(ref v)    => v.as_slice(),
      BorrowedBuffer(ref s) => s.as_slice(),
    }
  }

  #[inline]
  fn as_mut_slice(&mut self) -> &mut [u8] {
    match *self {
      OwnedBuffer(ref mut v)    => v.as_mut_slice(),
      BorrowedBuffer(ref mut s) => s.as_mut_slice(),
    }
  }

  #[inline]
  fn len(&self) -> uint {
    match *self {
      OwnedBuffer(ref v)    => v.len(),
      BorrowedBuffer(ref s) => s.len(),
    }
  }
}

/// By factoring out the range check, we prevent rustc from emitting a ton of
/// formatting code in our tight, little functions.
#[cold]
fn bad_range(pos: uint, len: uint) {
  fail!("Iobuf got invalid range: pos={}, len={}", pos, len);
}

/// A `RawIobuf` is the representation of both a `RWIobuf` and a `ROIobuf`.
/// It is very cheap to clone, as the backing buffer is shared and refcounted.
#[deriving(Clone)]
struct RawIobuf<'a> {
  buf:    Rc<UnsafeCell<MaybeOwnedBuffer<'a>>>,
  lo_min: uint,
  lo:     uint,
  hi:     uint,
  hi_max: uint,
}

impl<'a> RawIobuf<'a> {
  fn of_buf<'a>(buf: MaybeOwnedBuffer<'a>) -> RawIobuf<'a> {
    let len = buf.len();
    RawIobuf {
      buf: Rc::new(UnsafeCell::new(buf)),
      lo_min: 0,
      lo:     0,
      hi:     len,
      hi_max: len,
    }
  }

  fn new(len: uint) -> RawIobuf<'static> {
    RawIobuf::of_buf(OwnedBuffer(Vec::from_elem(len, 0u8)))
  }

  fn from_str<'a>(s: &'a str) -> RawIobuf<'a> {
    unsafe {
      let bytes: &mut [u8] = mem::transmute(s.as_bytes());
      RawIobuf::of_buf(BorrowedBuffer(bytes))
    }
  }

  fn from_vec(v: Vec<u8>) -> RawIobuf<'static> {
    RawIobuf::of_buf(OwnedBuffer(v))
  }

  fn from_slice<'a>(s: &'a [u8]) -> RawIobuf<'a> {
    unsafe {
      let mut_buf: &mut [u8] = mem::transmute(s);
      RawIobuf::of_buf(BorrowedBuffer(mut_buf))
    }
  }

  #[inline(always)]
  fn check_range(&self, pos: uint, len: uint) -> Result<(), ()> {
    if pos + len <= self.len() {
      Ok(())
    } else {
      Err(())
    }
  }

  #[inline(always)]
  fn check_range_fail(&self, pos: uint, len: uint) {
    match self.check_range(pos, len) {
      Ok(()) => {},
      Err(()) => bad_range(pos, len),
    }
  }

  #[inline]
  fn sub(&mut self, pos: uint, len: uint) -> Result<(), ()> {
    unsafe {
      try!(self.check_range(pos, len));
      Ok(self.unsafe_sub(pos, len))
    }
  }

  #[inline]
  unsafe fn unsafe_sub(&mut self, pos: uint, len: uint) {
    let lo      = self.lo + pos;
    let hi      = lo + len;
    self.lo_min = lo;
    self.lo     = lo;
    self.hi     = hi;
    self.hi_max = hi;
  }

  /// Both the limits and the window are [lo, hi).
  #[inline]
  fn set_limits_and_window(&mut self, limits: (uint, uint), window: (uint, uint)) -> Result<(), ()> {
    let (new_lo_min, new_hi_max) = limits;
    let (new_lo, new_hi) = window;
    if new_hi_max < new_lo_min  { return Err(()); }
    if new_hi     < new_lo      { return Err(()); }
    if new_lo_min < self.lo_min { return Err(()); }
    if new_hi_max > self.hi_max { return Err(()); }
    if new_lo     < self.lo     { return Err(()); }
    if new_hi     > self.hi     { return Err(()); }
    self.lo_min = new_lo_min;
    self.lo     = new_lo;
    self.hi     = new_hi;
    self.hi_max = new_hi_max;
    Ok(())
  }

  #[inline(always)]
  fn len(&self) -> uint {
    self.hi - self.lo
  }

  #[inline(always)]
  fn cap(&self) -> uint {
    self.hi_max - self.lo_min
  }

  #[inline(always)]
  fn is_empty(&self) -> bool {
    self.hi == self.lo
  }

  #[inline(always)]
  fn narrow(&mut self) {
    self.lo_min = self.lo;
    self.hi_max = self.hi;
  }

  #[inline(always)]
  fn advance(&mut self, len: uint) -> Result<(), ()> {
    unsafe {
      try!(self.check_range(0, len));
      self.unsafe_advance(len);
      Ok(())
    }
  }

  #[inline(always)]
  unsafe fn unsafe_advance(&mut self, len: uint) {
    self.lo += len;
  }

  #[inline(always)]
  fn resize(&mut self, len: uint) -> Result<(), ()> {
    let new_hi = self.lo + len;
    if new_hi > self.hi_max { return Err(()) }
    self.hi = new_hi;
    Ok(())
  }

  #[inline(always)]
  unsafe fn unsafe_resize(&mut self, len: uint) {
    self.hi = self.lo + len;
  }

  #[inline(always)]
  fn rewind(&mut self) {
    self.lo = self.lo_min;
  }

  #[inline(always)]
  fn reset(&mut self) {
    self.lo = self.lo_min;
    self.hi = self.hi_max;
  }

  #[inline(always)]
  fn flip_lo(&mut self) {
    self.hi = self.lo;
    self.lo = self.lo_min;
  }

  #[inline(always)]
  fn flip_hi(&mut self) {
    self.lo = self.hi;
    self.hi = self.hi_max;
  }

  #[inline(always)]
  fn compact(&mut self) {
    unsafe {
      let len = self.len();
      let s: raw::Slice<u8> = mem::transmute((*self.buf.get()).as_mut_slice());
      ptr::copy_memory(
        s.data as *mut u8,
        s.data.offset(self.lo as int) as *const u8,
        len);
      self.lo = self.lo_min + len;
      self.hi = self.hi_max;
    }
  }

  #[inline]
  unsafe fn as_slice(&self) -> &[u8] {
    (*self.buf.get()).as_slice().slice(self.lo, self.hi)
  }

  #[inline]
  unsafe fn as_mut_slice(&mut self) -> &mut [u8] {
    (*self.buf.get()).as_mut_slice().mut_slice(self.lo, self.hi)
  }

  #[inline(always)]
  fn peek(&self, pos: uint, dst: &mut [u8]) -> Result<(), ()> {
    unsafe {
      try!(self.check_range(pos, dst.len()));
      Ok(self.unsafe_peek(pos, dst))
    }
  }

  #[inline(always)]
  fn peek_be<T: Prim>(&self, pos: uint) -> Result<T, ()> {
    unsafe {
      try!(self.check_range(pos, mem::size_of::<T>()));
      Ok(self.unsafe_peek_be::<T>(pos))
    }
  }

  #[inline(always)]
  fn peek_le<T: Prim>(&self, pos: uint) -> Result<T, ()> {
    unsafe {
      try!(self.check_range(pos, mem::size_of::<T>()));
      Ok(self.unsafe_peek_le::<T>(pos))
    }
  }

  #[inline(always)]
  fn poke(&self, pos: uint, src: &[u8]) -> Result<(), ()> {
    unsafe {
      try!(self.check_range(pos, src.len()));
      Ok(self.unsafe_poke(pos, src))
    }
  }

  #[inline(always)]
  fn poke_be<T: Prim>(&self, pos: uint, t: T) -> Result<(), ()> {
    unsafe {
      try!(self.check_range(pos, mem::size_of::<T>()));
      Ok(self.unsafe_poke_be(pos, t))
    }
  }

  #[inline(always)]
  fn poke_le<T: Prim>(&self, pos: uint, t: T) -> Result<(), ()> {
    unsafe {
      try!(self.check_range(pos, mem::size_of::<T>()));
      Ok(self.unsafe_poke_le(pos, t))
    }
  }

  #[inline(always)]
  fn fill(&mut self, src: &[u8]) -> Result<(), ()> {
    unsafe {
      try!(self.check_range(0, src.len()));
      Ok(self.unsafe_fill(src))
    }
  }

  #[inline(always)]
  fn fill_be<T: Prim>(&mut self, t: T) -> Result<(), ()> {
    unsafe {
      try!(self.check_range(0, mem::size_of::<T>()));
      Ok(self.unsafe_fill_be(t))
    }
  }

  #[inline(always)]
  fn fill_le<T: Prim>(&mut self, t: T) -> Result<(), ()> {
    unsafe {
      try!(self.check_range(0, mem::size_of::<T>()));
      Ok(self.unsafe_fill_le(t)) // ok, unsafe fillet? om nom.
    }
  }

  #[inline(always)]
  fn consume(&mut self, dst: &mut [u8]) -> Result<(), ()> {
    unsafe {
      try!(self.check_range(0, dst.len()));
      Ok(self.unsafe_consume(dst))
    }
  }

  #[inline(always)]
  fn consume_le<T: Prim>(&mut self) -> Result<T, ()> {
    unsafe {
      try!(self.check_range(0, mem::size_of::<T>()));
      Ok(self.unsafe_consume_le())
    }
  }

  #[inline(always)]
  fn consume_be<T: Prim>(&mut self) -> Result<T, ()> {
    unsafe {
      try!(self.check_range(0, mem::size_of::<T>()));
      Ok(self.unsafe_consume_be())
    }
  }

  #[inline(always)]
  unsafe fn get_at<T: Prim>(&self, idx: uint) -> T {
    FromPrimitive::from_u8(
      *(*self.buf.get()).as_slice().unsafe_get(self.lo + idx))
    .unwrap()
  }

  #[inline(always)]
  unsafe fn set_at<T: Prim>(&self, idx: uint, val: T) {
    (*self.buf.get()).as_mut_slice().unsafe_set(self.lo + idx, val.to_u8().unwrap())
  }

  unsafe fn unsafe_peek(&self, pos: uint, dst: &mut [u8]) {
    let dst: raw::Slice<u8> = mem::transmute(dst);
    let (dst, len) = (dst.data as *mut u8, dst.len);
    let src: &[u8] = (*self.buf.get()).as_slice().slice_from(self.lo+pos);
    let src: raw::Slice<u8> = mem::transmute(src);
    let src = src.data;
    ptr::copy_memory(dst, src, len);
  }

  unsafe fn unsafe_peek_be<T: Prim>(&self, pos: uint) -> T {
    let bytes = mem::size_of::<T>();
    let mut x: T = Zero::zero();

    for i in iter::range(0, bytes) {
      x = self.get_at::<T>(pos+i) | (x << 8);
    }

    x
  }

  unsafe fn unsafe_peek_le<T: Prim>(&self, pos: uint) -> T {
    let bytes = mem::size_of::<T>();
    let mut x: T = Zero::zero();

    for i in iter::range(0, bytes) {
      x = (x >> 8) | (self.get_at::<T>(pos+i) << ((bytes - 1)* 8));
    }

    x
  }

  unsafe fn unsafe_poke(&self, pos: uint, src: &[u8]) {
    for (i, &src) in src.iter().enumerate() {
      self.set_at(pos+i, src);
    }
  }

  unsafe fn unsafe_poke_be<T: Prim>(&self, pos: uint, t: T) {
    let bytes = mem::size_of::<T>();

    let msk: T = FromPrimitive::from_u8(0xFFu8).unwrap();

    for i in iter::range(0, bytes) {
      self.set_at(pos+i, t << ((bytes-i-1)*8) & msk);
    }
  }

  unsafe fn unsafe_poke_le<T: Prim>(&self, pos: uint, t: T) {
    let bytes = mem::size_of::<T>();

    let msk: T = FromPrimitive::from_u8(0xFFu8).unwrap();

    for i in iter::range(0, bytes) {
      self.set_at(pos+i, t << (i*8) & msk);
    }
  }

  #[inline(always)]
  unsafe fn unsafe_fill(&mut self, src: &[u8]) {
    self.unsafe_poke(0, src);
    self.lo += src.len();
  }

  #[inline(always)]
  unsafe fn unsafe_fill_be<T: Prim>(&mut self, t: T) {
    self.unsafe_poke_be(0, t);
    self.lo += mem::size_of::<T>();
  }

  #[inline(always)]
  unsafe fn unsafe_fill_le<T: Prim>(&mut self, t: T) {
    self.unsafe_poke_le(0, t);
    self.lo += mem::size_of::<T>();
  }

  #[inline(always)]
  unsafe fn unsafe_consume(&mut self, dst: &mut [u8]) {
    self.unsafe_peek(0, dst);
    self.lo += dst.len();
  }

  #[inline(always)]
  unsafe fn unsafe_consume_le<T: Prim>(&mut self) -> T {
    let ret = self.unsafe_peek_le::<T>(0);
    self.lo += mem::size_of::<T>();
    ret
  }

  #[inline(always)]
  unsafe fn unsafe_consume_be<T: Prim>(&mut self) -> T {
    let ret = self.unsafe_peek_be::<T>(0);
    self.lo += mem::size_of::<T>();
    ret
  }

  fn show_hex(&self, f: &mut Formatter, half_line: &[u8])
      -> Result<(), FormatError> {
    for &x in half_line.iter() {
      try!(write!(f, "{:02x} ", x));
    }
    Ok(())
  }

  fn show_ascii(&self, f: &mut Formatter, half_line: &[u8])
      -> Result<(), FormatError> {
     for &x in half_line.iter() {
       let c = if x >= 32 && x < 126 { x as char } else { '.' };
       try!(write!(f, "{}", c));
     }
     Ok(())
  }

  fn show_line(&self, f: &mut Formatter, line_number: uint, chunk: &[u8])
      -> Result<(), FormatError> {

    if      self.len() <= 1u <<  8 { try!(write!(f, "0x{:02x}",  line_number * 8)) }
    else if self.len() <= 1u << 16 { try!(write!(f, "0x{:04x}",  line_number * 8)) }
    else if self.len() <= 1u << 24 { try!(write!(f, "0x{:06x}",  line_number * 8)) }
    else if self.len() <= 1u << 32 { try!(write!(f, "0x{:08x}",  line_number * 8)) }
    else if self.len() <= 1u << 40 { try!(write!(f, "0x{:010x}", line_number * 8)) }
    else if self.len() <= 1u << 48 { try!(write!(f, "0x{:012x}", line_number * 8)) }
    else if self.len() <= 1u << 56 { try!(write!(f, "0x{:014x}", line_number * 8)) }
    else                           { try!(write!(f, "0x{:016x}", line_number * 8)) }

    try!(write!(f, ":  "));

    let chunk_len = chunk.len();

    let (left_slice, right_slice) =
      if chunk_len >= 4 {
        (chunk.slice(0, 4), Some(chunk.slice_from(4)))
      } else {
        (chunk, None)
      };

    try!(self.show_hex(f, left_slice));
    try!(write!(f, "  "));
    try!(self.show_ascii(f, left_slice));
    try!(write!(f, "  "));
    match right_slice {
      None => {},
      Some(right_slice) => {
        try!(self.show_ascii(f, right_slice));
        try!(write!(f, "  "));
        try!(self.show_hex(f, right_slice));
      }
    }

    write!(f, "\n")
  }

  fn show(&self, f: &mut Formatter, ty: &str) -> Result<(), FormatError> {
    try!(write!(f, "{} IObuf, raw length={}, limits=[{},{}), bounds=[{},{})\n",
                ty, unsafe { (*self.buf.get()).as_slice().len() }, self.lo_min, self.hi_max, self.lo, self.hi));

    if self.lo == self.hi { return write!(f, "<empty buffer>"); }

    let b = unsafe { self.as_slice() };

    for (i, c) in b.chunks(8).enumerate() {
      try!(self.show_line(f, i, c));
    }

    Ok(())
  }
}

/// Have your functions take a generic IObuf whenever they don't modify the
/// window or bounds. This way, the functions can be used with both `ROIobuf`s
/// and `RWIobuf`s.
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
  /// Returns the size of the window.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("Hello");
  /// assert_eq!(b.advance(2), Ok(()));
  /// assert_eq!(b.len(), 3);
  /// ```
  fn len(&self) -> uint;

  /// Returns the size of the limits subrange. The capacity of an iobuf can be
  /// reduced via `narrow`.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("Hello");
  /// assert_eq!(b.advance(2), Ok(()));
  /// assert_eq!(b.cap(), 5);
  /// ```
  fn cap(&self) -> uint;

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
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// unsafe {
  ///   let mut b = ROIobuf::from_str("hello");
  ///   assert_eq!(b.as_slice(), b"hello");
  ///   assert_eq!(b.advance(2), Ok(()));
  ///   assert_eq!(b.as_slice(), b"llo");
  /// }
  /// ```
  unsafe fn as_slice(&self) -> &[u8];

  /// Changes the iobuf's limits and bounds to the subrange specified by
  /// `pos` and `len`, which must lie within the current window.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("hello");
  /// assert_eq!(b.sub(1, 3), Ok(()));
  /// unsafe { assert_eq!(b.as_slice(), b"ell") };
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
  /// If you want to slice from the start, set `pos` to `0`.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("hello");
  /// assert_eq!(b.sub(0, 3), Ok(()));
  /// unsafe { assert_eq!(b.as_slice(), b"hel") };
  /// ```
  ///
  /// If you want to slice to the end, set `len` to `self.len() - pos`.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("hello");
  /// let len = b.len() - 2; // makes borrowck happy
  /// assert_eq!(b.sub(2, len), Ok(()));
  /// unsafe { assert_eq!(b.as_slice(), b"llo") };
  /// ```
  fn sub(&mut self, pos: uint, len: uint) -> Result<(), ()>;

  /// The same as `sub`, but no bounds checks are performed. You should probably
  /// just use `sub`.
  unsafe fn unsafe_sub(&mut self, pos: uint, len: uint);

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
  fn set_limits_and_window(&mut self, limits: (uint, uint), window: (uint, uint)) -> Result<(), ()>;

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
  /// ```
  fn advance(&mut self, len: uint) -> Result<(), ()>;

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
  ///   let num_bytes: uint = num_shorts as uint * mem::size_of::<u16>();
  ///   unsafe {
  ///     try!(b.check_range(0, num_bytes));
  ///
  ///     let mut sum = 0u16;
  ///
  ///     for i in range(0, num_shorts as uint).map(|x| x * mem::size_of::<u16>()) {
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
  ///   let num_bytes: uint = num_shorts as uint * mem::size_of::<u16>();
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
  unsafe fn unsafe_advance(&mut self, len: uint);

  /// Sets the length of the window, provided it does not exceed the limits.
  fn resize(&mut self, len: uint) -> Result<(), ()>;

  /// Sets the length of the window. No bounds checking will be performed.
  unsafe fn unsafe_resize(&mut self, len: uint);

  /// Sets the lower bound of the window to the lower limit.
  fn rewind(&mut self);

  /// Sets the window to the limits.
  ///
  /// "Take it to the limit..."
  fn reset(&mut self);

  /// Sets the window to range from the lower limit to the lower bound of the
  /// old window. This is typically called after a series of `Fill`s, to
  /// reposition the window in preparation to `Consume` the newly written data.
  ///
  /// If the area of the limits is denoted with `[]` and the area of the window
  /// is denoted with `x`, then the `flip_lo` looks like:
  ///
  /// `Before: [       xxxx  ]`
  ///
  /// `After:  [xxxxxxx      ]`
  fn flip_lo(&mut self);

  /// Sets the window to range from the upper bound of the old window to the
  /// upper limit. This is a dual to `flip_lo`, and is typically called when the
  /// data in the current (narrowed) window has been processed and the window
  /// needs to be positioned over the remaining data in the buffer.
  ///
  /// If the area of the limits is denoted with `[]` and the area of the window
  /// is denoted with `x`, then the `flip_lo` looks like:
  ///
  /// `Before: [       xxxx  ]`
  ///
  /// `After:  [           xx]`
  fn flip_hi(&mut self);

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
  fn peek(&self, pos: uint, dst: &mut [u8]) -> Result<(), ()>;

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
  fn peek_be<T: Prim>(&self, pos: uint) -> Result<T, ()>;

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
  fn peek_le<T: Prim>(&self, pos: uint) -> Result<T, ()>;

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

  /// Reads a big-endian primitive at a given offset from the beginning of the
  /// window.
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

  /// Reads a little-endian primitive at a given offset from the beginning of
  /// the window.
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
  ///   unsafe {
  ///     let mut sum = 0u;
  ///
  ///     let num_buffers: u8 = try!(b.consume_be());
  ///     for _ in range(0, num_buffers) {
  ///       let len: u8 = try!(b.consume_be());
  ///       try!(b.check_range(0, len as uint));
  ///       for _ in range(0, len) {
  ///         sum += b.unsafe_consume_be::<u8>() as uint;
  ///       }
  ///     }
  ///
  ///     Ok(sum)
  ///   }
  /// }
  ///
  /// assert_eq!(parse(&mut b), Ok(0x55 + 0x66 + 0x11 + 0x22 + 0x33));
  /// ```
  fn check_range(&self, pos: uint, len: uint) -> Result<(), ()>;

  /// The same as `check_range`, but fails if the bounds check returns `Err(())`.
  fn check_range_fail(&self, pos: uint, len: uint);

  unsafe fn unsafe_peek(&self, pos: uint, dst: &mut [u8]);
  unsafe fn unsafe_peek_be<T: Prim>(&self, pos: uint) -> T;
  unsafe fn unsafe_peek_le<T: Prim>(&self, pos: uint) -> T;

  unsafe fn unsafe_consume(&mut self, dst: &mut [u8]);
  unsafe fn unsafe_consume_be<T: Prim>(&mut self) -> T;
  unsafe fn unsafe_consume_le<T: Prim>(&mut self) -> T;
}

/// An `Iobuf` that cannot write into the buffer, but all read-only operations
/// are supported. It is possible to get a `RWIobuf` by performing a `deep_clone`
/// of the Iobuf, but this is extremely inefficient.
///
/// If your function only needs to do read-only operations on an Iobuf, consider
/// taking a generic `Iobuf` trait instead. That way, it can be used with either
/// a ROIobuf or a RWIobuf, generically.
#[deriving(Clone)]
pub struct ROIobuf<'a> {
  raw: RawIobuf<'a>,
}

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
#[deriving(Clone)]
pub struct RWIobuf<'a> {
  raw: RawIobuf<'a>,
}

impl<'a> ROIobuf<'a> {
  #[inline]
  pub fn new(len: uint) -> ROIobuf<'static> {
    ROIobuf { raw: RawIobuf::new(len) }
  }

  #[inline]
  pub fn from_str<'a>(s: &'a str) -> ROIobuf<'a> {
    ROIobuf { raw: RawIobuf::from_str(s) }
  }

  #[inline]
  pub fn from_vec(v: Vec<u8>) -> ROIobuf<'static> {
    ROIobuf { raw: RawIobuf::from_vec(v) }
  }

  #[inline]
  pub fn from_slice<'a>(s: &'a [u8]) -> ROIobuf<'a> {
    ROIobuf { raw: RawIobuf::from_slice(s) }
  }

  #[inline]
  pub fn deep_clone(&self) -> RWIobuf<'static> {
    unsafe {
      RWIobuf {
        raw: RawIobuf {
          buf:
            Rc::new(UnsafeCell::new(OwnedBuffer(
              match *self.raw.buf.get() {
                OwnedBuffer(ref v) => v.clone(),
                BorrowedBuffer(ref s) => FromIterator::from_iter(s.iter().map(|&x| x)),
              }))),
          lo_min: self.raw.lo_min,
          lo:     self.raw.lo,
          hi:     self.raw.hi,
          hi_max: self.raw.hi_max,
        }
      }
    }
  }
}

impl<'a> RWIobuf<'a> {
  #[inline]
  pub fn new(len: uint) -> RWIobuf<'static> {
    RWIobuf { raw: RawIobuf::new(len) }
  }

  #[inline]
  pub fn from_str<'a>(s: &'a mut str) -> RWIobuf<'a> {
    RWIobuf { raw: RawIobuf::from_str(s) }
  }

  #[inline]
  pub fn from_vec(v: Vec<u8>) -> RWIobuf<'static> {
    RWIobuf { raw: RawIobuf::from_vec(v) }
  }

  #[inline]
  pub fn from_slice<'a>(s: &'a mut [u8]) -> RWIobuf<'a> {
    RWIobuf { raw: RawIobuf::from_slice(s) }
  }

  /// Get a read-only copy of this Iobuf. This is a very cheap operation, as the
  /// backing buffers are shared.
  #[inline]
  pub fn read_only(&self) -> ROIobuf<'a> {
    ROIobuf { raw: self.raw.clone() }
  }

  /// Copies data from the window to the lower limit fo the iobuf and sets the
  /// window to range from the end of the copied data to the upper limit. This
  /// is typically called after a series of `Consume`s to save unread data and
  /// prepare for the next series of `Fill`s and `flip_lo`s.
  #[inline(always)]
  pub fn compact(&mut self) { self.raw.compact() }

  /// Reads the data in the window as a mutable slice. Note that since `&mut`
  /// in rust really means `&unique`, this function lies. There can exist
  /// multiple slices of the same data. Therefore, this function is unsafe.
  #[inline(always)]
  pub unsafe fn as_mut_slice(&mut self) -> &mut [u8] { self.raw.as_mut_slice() }

  #[inline(always)]
  pub fn poke(&self, pos: uint, src: &[u8]) -> Result<(), ()> { self.raw.poke(pos, src) }
  #[inline(always)]
  pub fn poke_be<T: Prim>(&self, pos: uint, t: T) -> Result<(), ()> { self.raw.poke_be(pos, t) }
  #[inline(always)]
  pub fn poke_le<T: Prim>(&self, pos: uint, t: T) -> Result<(), ()> { self.raw.poke_le(pos, t) }

  #[inline(always)]
  pub fn fill(&mut self, src: &[u8]) -> Result<(), ()> { self.raw.fill(src) }
  #[inline(always)]
  pub fn fill_be<T: Prim>(&mut self, t: T) -> Result<(), ()> { self.raw.fill_be(t) }
  #[inline(always)]
  pub unsafe fn fill_le<T: Prim>(&mut self, t: T) -> Result<(), ()> { self.raw.fill_le(t) }

  #[inline(always)]
  pub unsafe fn unsafe_poke(&self, pos: uint, src: &[u8]) { self.raw.unsafe_poke(pos, src) }
  #[inline(always)]
  pub unsafe fn unsafe_poke_be<T: Prim>(&self, pos: uint, t: T) { self.raw.unsafe_poke_be(pos, t) }
  #[inline(always)]
  pub unsafe fn unsafe_poke_le<T: Prim>(&self, pos: uint, t: T) { self.raw.unsafe_poke_le(pos, t) }

  #[inline(always)]
  pub unsafe fn unsafe_fill(&mut self, src: &[u8]) { self.raw.unsafe_fill(src) }
  #[inline(always)]
  pub unsafe fn unsafe_fill_be<T: Prim>(&mut self, t: T) { self.raw.unsafe_fill_be(t) }
  #[inline(always)]
  pub unsafe fn unsafe_fill_le<T: Prim>(&mut self, t: T) { self.raw.unsafe_fill_le(t) }
}

impl<'a> Iobuf for ROIobuf<'a> {
  #[inline(always)]
  fn len(&self) -> uint { self.raw.len() }

  #[inline(always)]
  fn cap(&self) -> uint { self.raw.cap() }

  #[inline(always)]
  fn is_empty(&self) -> bool { self.raw.is_empty() }

  #[inline(always)]
  unsafe fn as_slice(&self) -> &[u8] { self.raw.as_slice() }

  #[inline(always)]
  fn sub(&mut self, pos: uint, len: uint) -> Result<(), ()> { self.raw.sub(pos, len) }

  #[inline(always)]
  unsafe fn unsafe_sub(&mut self, pos: uint, len: uint) { self.raw.unsafe_sub(pos, len) }

  #[inline(always)]
  fn set_limits_and_window(&mut self, limits: (uint, uint), window: (uint, uint)) -> Result<(), ()> { self.raw.set_limits_and_window(limits, window) }

  #[inline(always)]
  fn narrow(&mut self) { self.raw.narrow() }

  #[inline(always)]
  fn advance(&mut self, len: uint) -> Result<(), ()> { self.raw.advance(len) }

  #[inline(always)]
  unsafe fn unsafe_advance(&mut self, len: uint) { self.raw.unsafe_advance(len) }

  #[inline(always)]
  fn resize(&mut self, len: uint) -> Result<(), ()> { self.raw.resize(len) }

  #[inline(always)]
  unsafe fn unsafe_resize(&mut self, len: uint) { self.raw.unsafe_resize(len) }

  #[inline(always)]
  fn rewind(&mut self) { self.raw.rewind() }

  #[inline(always)]
  fn reset(&mut self) { self.raw.reset() }

  #[inline(always)]
  fn flip_lo(&mut self) { self.raw.flip_lo() }

  #[inline(always)]
  fn flip_hi(&mut self) { self.raw.flip_hi() }

  #[inline(always)]
  fn peek(&self, pos: uint, dst: &mut [u8]) -> Result<(), ()> { self.raw.peek(pos, dst) }
  #[inline(always)]
  fn peek_be<T: Prim>(&self, pos: uint) -> Result<T, ()> { self.raw.peek_be(pos) }
  #[inline(always)]
  fn peek_le<T: Prim>(&self, pos: uint) -> Result<T, ()> { self.raw.peek_le(pos) }

  #[inline(always)]
  fn consume(&mut self, dst: &mut [u8]) -> Result<(), ()> { self.raw.consume(dst) }
  #[inline(always)]
  fn consume_be<T: Prim>(&mut self) -> Result<T, ()> { self.raw.consume_be::<T>() }
  #[inline(always)]
  fn consume_le<T: Prim>(&mut self) -> Result<T, ()> { self.raw.consume_le::<T>() }

  #[inline(always)]
  fn check_range(&self, pos: uint, len: uint) -> Result<(), ()> { self.raw.check_range(pos, len) }

  #[inline(always)]
  fn check_range_fail(&self, pos: uint, len: uint) { self.raw.check_range_fail(pos, len) }

  #[inline(always)]
  unsafe fn unsafe_peek(&self, pos: uint, dst: &mut [u8]) { self.raw.unsafe_peek(pos, dst) }
  #[inline(always)]
  unsafe fn unsafe_peek_be<T: Prim>(&self, pos: uint) -> T { self.raw.unsafe_peek_be(pos) }
  #[inline(always)]
  unsafe fn unsafe_peek_le<T: Prim>(&self, pos: uint) -> T { self.raw.unsafe_peek_le(pos) }

  #[inline(always)]
  unsafe fn unsafe_consume(&mut self, dst: &mut [u8]) { self.raw.unsafe_consume(dst) }
  #[inline(always)]
  unsafe fn unsafe_consume_be<T: Prim>(&mut self) -> T { self.raw.unsafe_consume_be::<T>() }
  #[inline(always)]
  unsafe fn unsafe_consume_le<T: Prim>(&mut self) -> T { self.raw.unsafe_consume_le::<T>() }
}

impl<'a> Iobuf for RWIobuf<'a> {
  #[inline(always)]
  fn len(&self) -> uint { self.raw.len() }

  #[inline(always)]
  fn cap(&self) -> uint { self.raw.cap() }

  #[inline(always)]
  fn is_empty(&self) -> bool { self.raw.is_empty() }

  #[inline(always)]
  unsafe fn as_slice(&self) -> &[u8] { self.raw.as_slice() }

  #[inline(always)]
  fn sub(&mut self, pos: uint, len: uint) -> Result<(), ()> { self.raw.sub(pos, len) }

  #[inline(always)]
  unsafe fn unsafe_sub(&mut self, pos: uint, len: uint) { self.raw.unsafe_sub(pos, len) }

  #[inline(always)]
  fn set_limits_and_window(&mut self, limits: (uint, uint), window: (uint, uint)) -> Result<(), ()> { self.raw.set_limits_and_window(limits, window) }

  #[inline(always)]
  fn narrow(&mut self) { self.raw.narrow() }

  #[inline(always)]
  fn advance(&mut self, len: uint) -> Result<(), ()> { self.raw.advance(len) }

  #[inline(always)]
  unsafe fn unsafe_advance(&mut self, len: uint) { self.raw.unsafe_advance(len) }

  #[inline(always)]
  fn resize(&mut self, len: uint) -> Result<(), ()> { self.raw.resize(len) }

  #[inline(always)]
  unsafe fn unsafe_resize(&mut self, len: uint) { self.raw.unsafe_resize(len) }

  #[inline(always)]
  fn rewind(&mut self) { self.raw.rewind() }

  #[inline(always)]
  fn reset(&mut self) { self.raw.reset() }

  #[inline(always)]
  fn flip_lo(&mut self) { self.raw.flip_lo() }

  #[inline(always)]
  fn flip_hi(&mut self) { self.raw.flip_hi() }

  #[inline(always)]
  fn peek(&self, pos: uint, dst: &mut [u8]) -> Result<(), ()> { self.raw.peek(pos, dst) }
  #[inline(always)]
  fn peek_be<T: Prim>(&self, pos: uint) -> Result<T, ()> { self.raw.peek_be(pos) }
  #[inline(always)]
  fn peek_le<T: Prim>(&self, pos: uint) -> Result<T, ()> { self.raw.peek_le(pos) }

  #[inline(always)]
  fn consume(&mut self, dst: &mut [u8]) -> Result<(), ()> { self.raw.consume(dst) }
  #[inline(always)]
  fn consume_be<T: Prim>(&mut self) -> Result<T, ()> { self.raw.consume_be::<T>() }
  #[inline(always)]
  fn consume_le<T: Prim>(&mut self) -> Result<T, ()> { self.raw.consume_le::<T>() }

  #[inline(always)]
  fn check_range(&self, pos: uint, len: uint) -> Result<(), ()> { self.raw.check_range(pos, len) }

  #[inline(always)]
  fn check_range_fail(&self, pos: uint, len: uint) { self.raw.check_range_fail(pos, len) }

  #[inline(always)]
  unsafe fn unsafe_peek(&self, pos: uint, dst: &mut [u8]) { self.raw.unsafe_peek(pos, dst) }
  #[inline(always)]
  unsafe fn unsafe_peek_be<T: Prim>(&self, pos: uint) -> T { self.raw.unsafe_peek_be(pos) }
  #[inline(always)]
  unsafe fn unsafe_peek_le<T: Prim>(&self, pos: uint) -> T { self.raw.unsafe_peek_le(pos) }

  #[inline(always)]
  unsafe fn unsafe_consume(&mut self, dst: &mut [u8]) { self.raw.unsafe_consume(dst) }
  #[inline(always)]
  unsafe fn unsafe_consume_be<T: Prim>(&mut self) -> T { self.raw.unsafe_consume_be::<T>() }
  #[inline(always)]
  unsafe fn unsafe_consume_le<T: Prim>(&mut self) -> T { self.raw.unsafe_consume_le::<T>() }
}

impl<'a> Show for ROIobuf<'a> {
  fn fmt(&self, f: &mut Formatter) -> Result<(), FormatError> {
    self.raw.show(f, "read-only")
  }
}

impl<'a> Show for RWIobuf<'a> {
  fn fmt(&self, f: &mut Formatter) -> Result<(), FormatError> {
    self.raw.show(f, "read-write")
  }
}
