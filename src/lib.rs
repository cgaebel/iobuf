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
//!
//! To keep the struct small (24 bytes!), the maximum size of an Iobuf is 2 GB.
//! Please let me know if you need more than this. I have never before seen a
//! use case for Iobufs larger than 2 GB, and it gives us a 40% smaller struct
//! compared to support INT64_MAX bytes. The improved size allows much better
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

#![license = "MIT"]

#![feature(phase)]
#![feature(unsafe_destructor)]
#![no_std]

extern crate alloc;
extern crate collections;

#[phase(plugin, link)]
extern crate core;

#[cfg(test)] #[phase(plugin,link)] extern crate std;

#[cfg(test)] extern crate native;
#[cfg(test)] extern crate test;

use alloc::heap;

use collections::string::String;
use collections::vec::Vec;
use core::clone::Clone;
use core::collections::Collection;
use core::fmt::{Formatter,FormatError,Show};
use core::kinds::Copy;
use core::kinds::marker::{ContravariantLifetime, NoCopy, NoSend, NoSync};
use core::iter;
use core::iter::Iterator;
use core::mem;
use core::num::{Zero, FromPrimitive, ToPrimitive};
use core::ops::{Drop, Shl, Shr, BitOr, BitAnd};
use core::option::{Option, Some, None};
use core::ptr;
use core::ptr::RawPtr;
use core::raw;
use core::result::{Result,Ok,Err};
use core::slice::{ImmutableSlice, AsSlice};
use core::str::StrSlice;
use core::u32;
use core::uint;

// https://github.com/rust-lang/rust/issues/18491#issuecomment-61293267
#[cfg(not(test))]
mod std { pub use core::fmt; }

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

/// The biggest Iobuf supported, to allow us to have a small RawIobuf struct.
/// By limiting the buffer sizes, we can bring the struct down from 40 bytes to
/// 24 bytes -- a 40% reduction. This frees up precious cache and registers for
/// the actual processing.
static MAX_BUFFER_LEN: uint = 0x7FFF_FFFF;

// By factoring out the calls to `panic!`, we prevent rustc from emitting a ton
// of formatting code in our tight, little functions, and also help guide
// inlining.

#[cold]
fn bad_range(pos: u64, len: u64) -> ! {
  panic!("Iobuf got invalid range: pos={}, len={}", pos, len)
}

#[cold]
fn buffer_too_big(actual_size: uint) -> ! {
  panic!("Tried to create an Iobuf that's too big: {} bytes. Max size = {}",
         actual_size, MAX_BUFFER_LEN)
}

/// A `RawIobuf` is the representation of both a `RWIobuf` and a `ROIobuf`.
/// It is very cheap to clone, as the backing buffer is shared and refcounted.
#[unsafe_no_drop_flag]
struct RawIobuf<'a> {
  // The -1nd size_of<uint>() bytes of `buf` is the refcount.
  // The -2st size_of<uint>() bytes of `buf` is the length of the allocation.
  // Starting at `buf` is the raw data itself.
  buf:    *mut u8,
  // If the highest bit of this is set, `buf` is owned and the data before the
  // pointer is valid. If it is not set, then the buffer wasn't allocated by us:
  // it's owned by someone else. Therefore, there's no header, and no need to
  // deallocate or refcount.
  lo_min_and_owned_bit: u32,
  lo:     u32,
  hi:     u32,
  hi_max: u32,
  lifetm: ContravariantLifetime<'a>,
  nocopy: NoCopy,
  nosend: NoSend,
  nosync: NoSync,
}

impl<'a> Clone for RawIobuf<'a> {
  #[inline]
  fn clone(&self) -> RawIobuf<'a> {
    self.inc_ref_count();

    RawIobuf {
      buf:    self.buf,
      lo_min_and_owned_bit: self.lo_min_and_owned_bit,
      lo:     self.lo,
      hi:     self.hi,
      hi_max: self.hi_max,
      lifetm: self.lifetm,
      nocopy: NoCopy,
      nosend: NoSend,
      nosync: NoSync,
    }
  }
}

/// The bitmask to get the "is the buffer owned" bit.
static OWNED_MASK: u32 = 1u32 << (u32::BITS - 1);

#[unsafe_destructor]
impl<'a> Drop for RawIobuf<'a> {
  fn drop(&mut self) {
    unsafe {
      match self.dec_ref_count() {
        None => {},
        Some(bytes_allocated) =>
          heap::deallocate(
            self.buf.offset(
              -2 * (uint::BYTES as int)),
            bytes_allocated,
            mem::align_of::<uint>()),
      }
    }

    self.buf    = ptr::null_mut();
    self.lo_min_and_owned_bit = 0;
    self.lo     = 0;
    self.hi     = 0;
    self.hi_max = 0;
  }
}

impl<'a> RawIobuf<'a> {
  fn new(len: uint) -> RawIobuf<'static> {
    unsafe {
      if len > MAX_BUFFER_LEN {
          buffer_too_big(len);
      }

      let data_len = 2*uint::BYTES + len;

      let buf: *mut u8 = heap::allocate(data_len, mem::align_of::<uint>());

      let allocated_len: *mut uint = buf as *mut uint;
      let ref_count: *mut uint = buf.offset(uint::BYTES as int) as *mut uint;

      *allocated_len = data_len;
      *ref_count     = 1;

      let buf: *mut u8 = buf.offset(2 * (uint::BYTES as int));

      RawIobuf {
        buf:    buf,
        lo_min_and_owned_bit: OWNED_MASK,
        lo:     0,
        hi:     len as u32,
        hi_max: len as u32,
        lifetm: ContravariantLifetime,
        nocopy: NoCopy,
        nosend: NoSend,
        nosync: NoSync,
      }
    }
  }

  #[inline(always)]
  fn empty() -> RawIobuf<'static> {
    RawIobuf {
      buf:    ptr::null_mut(),
      lo_min_and_owned_bit: 0,
      lo:     0,
      hi:     0,
      hi_max: 0,
      lifetm: ContravariantLifetime,
      nocopy: NoCopy,
      nosend: NoSend,
      nosync: NoSync,
    }
  }

  #[inline(always)]
  fn lo_min(&self) -> u32 {
    self.lo_min_and_owned_bit & !OWNED_MASK
  }

  #[inline(always)]
  fn set_lo_min(&mut self, new_value: u32) {
    if cfg!(debug) {
      if new_value > MAX_BUFFER_LEN as u32 {
        panic!("new lo_min out of range (max = 0x7FFF_FFFF): {:X}", new_value);
      }
    }
    self.lo_min_and_owned_bit &= OWNED_MASK;
    self.lo_min_and_owned_bit |= new_value;
  }

  #[inline(always)]
  fn is_owned(&self) -> bool {
    self.lo_min_and_owned_bit & OWNED_MASK != 0
  }

  #[inline(always)]
  fn ref_count(&self) -> Option<*mut uint> {
    unsafe {
      if self.is_owned() {
        Some(self.buf.offset(-(uint::BYTES as int)) as *mut uint)
      } else {
        None
      }
    }
  }

  #[inline(always)]
  fn amount_allocated(&self) -> Option<*mut uint> {
    unsafe {
      if self.is_owned() {
        Some(self.buf.offset(-2 * (uint::BYTES as int)) as *mut uint)
      } else {
        None
      }
    }
  }

  #[inline(always)]
  fn inc_ref_count(&self) {
    match self.ref_count() {
      Some(dst) => unsafe { *dst += 1; },
      None      => {},
    }
  }

  /// Returns `Some(bytes_allocated)` if the buffer needs to be freed.
  #[inline]
  fn dec_ref_count(&self) -> Option<uint> {
    match self.ref_count() {
      None => None,
      Some(ref_count) => unsafe {
        *ref_count -= 1;
        if *ref_count == 0 {
          self.amount_allocated().map(|p| *p)
        } else {
          None
        }
      }
    }
  }

  #[inline(always)]
  fn from_str<'a>(s: &'a str) -> RawIobuf<'a> {
    RawIobuf::from_slice(s.as_bytes())
  }

  #[inline(always)]
  fn from_string(s: String) -> RawIobuf<'static> {
    RawIobuf::from_vec(s.into_bytes())
  }

  #[inline(always)]
  fn from_vec(v: Vec<u8>) -> RawIobuf<'static> {
    unsafe {
      let b = RawIobuf::new(v.len());
      let s: raw::Slice<u8> = mem::transmute(v.as_slice());
      ptr::copy_nonoverlapping_memory(b.buf, s.data, s.len);
      b
    }
  }

  #[inline(always)]
  fn from_slice<'a>(s: &'a [u8]) -> RawIobuf<'a> {
    unsafe {
      let s_slice: raw::Slice<u8> = mem::transmute(s);
      let ptr = s_slice.data as *mut u8;
      let len = s_slice.len;

      if len > MAX_BUFFER_LEN {
        buffer_too_big(len);
      }

      RawIobuf {
        buf:    ptr,
        lo_min_and_owned_bit: 0,
        lo:     0,
        hi:     len as u32,
        hi_max: len as u32,
        lifetm: ContravariantLifetime,
        nocopy: NoCopy,
        nosend: NoSend,
        nosync: NoSync,
      }
    }
  }

  #[inline]
  fn deep_clone(&self) -> RawIobuf<'static> {
    unsafe {
      let my_data = self.as_raw_limit_slice();

      let mut b = RawIobuf::new(my_data.len);

      b.lo = self.lo;
      b.hi = self.hi;

      ptr::copy_memory(b.buf, my_data.data, my_data.len);

      b
    }
  }

  #[inline(always)]
  unsafe fn as_raw_limit_slice(&self) -> raw::Slice<u8> {
    raw::Slice {
      data: self.buf.offset(self.lo_min() as int) as *const u8,
      len:  self.cap() as uint,
    }
  }

  #[inline(always)]
  unsafe fn as_limit_slice<'b>(&'b self) -> &'b [u8] {
    mem::transmute(self.as_raw_limit_slice())
  }

  #[inline(always)]
  unsafe fn as_limit_slice_mut<'b>(&'b self) -> &'b mut [u8] {
    mem::transmute(self.as_raw_limit_slice())
  }

  #[inline(always)]
  unsafe fn as_raw_window_slice(&self) -> raw::Slice<u8> {
    raw::Slice {
      data: self.buf.offset(self.lo as int) as *const u8,
      len:  self.len() as uint,
    }
  }

  #[inline(always)]
  unsafe fn as_window_slice<'b>(&'b self) -> &'b [u8] {
    mem::transmute(self.as_raw_window_slice())
  }

  #[inline(always)]
  unsafe fn as_window_slice_mut<'b>(&'b self) -> &'b mut [u8] {
    mem::transmute(self.as_raw_window_slice())
  }

  #[inline(always)]
  fn check_range(&self, pos: u64, len: u64) -> Result<(), ()> {
    if pos + len <= self.len() as u64 {
      Ok(())
    } else {
      Err(())
    }
  }

  #[inline(always)]
  fn check_range_u32(&self, pos: u32, len: u32) -> Result<(), ()> {
    self.check_range(pos as u64, len as u64)
  }

  #[inline(always)]
  fn check_range_uint(&self, pos: u32, len: uint) -> Result<(), ()> {
    self.check_range(pos as u64, len as u64)
  }

  #[inline(always)]
  fn check_range_u32_fail(&self, pos: u32, len: u32) {
    match self.check_range_u32(pos, len) {
      Ok(()) => {},
      Err(()) => bad_range(pos as u64, len as u64),
    }
  }

  #[inline(always)]
  fn check_range_uint_fail(&self, pos: u32, len: uint) {
    match self.check_range_uint(pos, len) {
      Ok(())  => {},
      Err(()) => bad_range(pos as u64, len as u64),
    }
  }

  #[inline(always)]
  fn debug_check_range_u32(&self, pos: u32, len: u32) {
    if cfg!(debug) {
      self.check_range_u32_fail(pos, len);
    }
  }

  #[inline(always)]
  fn debug_check_range_uint(&self, pos: u32, len: uint) {
    if cfg!(debug) {
      self.check_range_uint_fail(pos, len);
    }
  }

  #[inline(always)]
  fn sub_window(&mut self, pos: u32, len: u32) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_u32(pos, len));
      Ok(self.unsafe_sub_window(pos, len))
    }
  }

  #[inline(always)]
  fn sub_window_from(&mut self, pos: u32) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_u32(pos, 0));
      Ok(self.unsafe_sub_window_from(pos))
    }
  }

  #[inline(always)]
  fn sub_window_to(&mut self, len: u32) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_u32(0, len));
      Ok(self.unsafe_sub_window_to(len))
    }
  }

  #[inline(always)]
  unsafe fn unsafe_sub_window(&mut self, pos: u32, len: u32) {
    self.debug_check_range_u32(pos, len);
    self.unsafe_resize(pos);
    self.flip_hi();
    self.unsafe_resize(len);
  }

  #[inline(always)]
  unsafe fn unsafe_sub_window_from(&mut self, pos: u32) {
    self.debug_check_range_u32(pos, 0);
    self.unsafe_resize(pos);
    self.flip_hi();
  }

  #[inline(always)]
  unsafe fn unsafe_sub_window_to(&mut self, len: u32) {
    self.debug_check_range_u32(0, len);
    self.unsafe_resize(len)
  }

  #[inline(always)]
  fn sub(&mut self, pos: u32, len: u32) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_u32(pos, len));
      Ok(self.unsafe_sub(pos, len))
    }
  }

  #[inline(always)]
  fn sub_from(&mut self, pos: u32) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_u32(pos, 0));
      Ok(self.unsafe_sub_from(pos))
    }
  }

  #[inline(always)]
  fn sub_to(&mut self, len: u32) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_u32(0, len));
      Ok(self.unsafe_sub_to(len))
    }
  }

  #[inline(always)]
  unsafe fn unsafe_sub(&mut self, pos: u32, len: u32) {
    self.debug_check_range_u32(pos, len);
    self.unsafe_sub_window(pos, len);
    self.narrow();
  }

  #[inline(always)]
  unsafe fn unsafe_sub_from(&mut self, pos: u32) {
    self.debug_check_range_u32(pos, 0);
    self.unsafe_sub_window_from(pos);
    self.narrow();
  }

  #[inline(always)]
  unsafe fn unsafe_sub_to(&mut self, len: u32) {
    self.debug_check_range_u32(0, len);
    self.unsafe_sub_window_to(len);
    self.narrow();
  }

  /// Both the limits and the window are [lo, hi).
  #[inline]
  fn set_limits_and_window(&mut self, limits: (u32, u32), window: (u32, u32)) -> Result<(), ()> {
    let (new_lo_min, new_hi_max) = limits;
    let (new_lo, new_hi) = window;
    let lo_min = self.lo_min();
    if new_hi_max < new_lo_min  { return Err(()); }
    if new_hi     < new_lo      { return Err(()); }
    if new_lo_min < lo_min      { return Err(()); }
    if new_hi_max > self.hi_max { return Err(()); }
    if new_lo     < self.lo     { return Err(()); }
    if new_hi     > self.hi     { return Err(()); }
    self.set_lo_min(new_lo_min);
    self.lo     = new_lo;
    self.hi     = new_hi;
    self.hi_max = new_hi_max;
    Ok(())
  }

  #[inline(always)]
  fn len(&self) -> u32 {
    self.hi - self.lo
  }

  #[inline(always)]
  fn cap(&self) -> u32 {
    self.hi_max - self.lo_min()
  }

  #[inline(always)]
  fn is_empty(&self) -> bool {
    self.hi == self.lo
  }

  #[inline(always)]
  fn narrow(&mut self) {
    let lo = self.lo;
    self.set_lo_min(lo);
    self.hi_max = self.hi;
  }

  #[inline(always)]
  fn advance(&mut self, len: u32) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_u32(0, len));
      self.unsafe_advance(len);
      Ok(())
    }
  }

  #[inline(always)]
  unsafe fn unsafe_advance(&mut self, len: u32) {
    self.debug_check_range_u32(0, len);
    self.lo += len;
  }

  #[inline(always)]
  fn extend(&mut self, len: u32) -> Result<(), ()> {
    unsafe {
      let hi     = self.hi     as u64;
      let hi_max = self.hi_max as u64;
      let new_hi = hi + len    as u64;

      if new_hi > hi_max {
        Err(())
      } else {
        Ok(self.unsafe_extend(len))
      }
    }
  }

  #[inline(always)]
  unsafe fn unsafe_extend(&mut self, len: u32) {
    if cfg!(debug) {
      let hi     = self.hi     as u64;
      let hi_max = self.hi_max as u64;
      let new_hi = hi + len    as u64;

      if new_hi > hi_max {
        bad_range(new_hi, 0);
      }

    }
    self.hi += len;
  }

  #[inline(always)]
  fn is_extended_by_raw<'b>(&self, other: &RawIobuf<'b>) -> bool {
    unsafe {
      self.buf.offset(self.hi as int) == other.buf.offset(other.lo as int)
    }
  }

  #[inline(always)]
  fn is_extended_by_ro<'b>(&self, other: &ROIobuf<'b>) -> bool {
    self.is_extended_by_raw(&other.raw)
  }

  #[inline(always)]
  fn is_extended_by_rw<'b>(&self, other: &RWIobuf<'b>) -> bool {
    self.is_extended_by_raw(&other.raw)
  }

  #[inline(always)]
  fn resize(&mut self, len: u32) -> Result<(), ()> {
    let new_hi = self.lo + len;
    if new_hi > self.hi_max { return Err(()) }
    self.hi = new_hi;
    Ok(())
  }

  #[inline(always)]
  unsafe fn unsafe_resize(&mut self, len: u32) {
    self.debug_check_range_u32(0, len);
    self.hi = self.lo + len;
  }

  #[inline(always)]
  fn rewind(&mut self) {
    self.lo = self.lo_min();
  }

  #[inline(always)]
  fn reset(&mut self) {
    self.lo = self.lo_min();
    self.hi = self.hi_max;
  }

  #[inline(always)]
  fn flip_lo(&mut self) {
    self.hi = self.lo;
    self.lo = self.lo_min();
  }

  #[inline(always)]
  fn flip_hi(&mut self) {
    self.lo = self.hi;
    self.hi = self.hi_max;
  }

  fn compact(&mut self) {
    unsafe {
      let len = self.len();
      let lo_min = self.lo_min();
      ptr::copy_memory(
        self.buf.offset(lo_min as int),
        self.buf.offset(self.lo as int) as *const u8,
        len as uint);
      self.lo = lo_min + len;
      self.hi = self.hi_max;
    }
  }

  #[inline(always)]
  fn peek(&self, pos: u32, dst: &mut [u8]) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_uint(pos, dst.len()));
      Ok(self.unsafe_peek(pos, dst))
    }
  }

  #[inline(always)]
  fn peek_be<T: Prim>(&self, pos: u32) -> Result<T, ()> {
    unsafe {
      try!(self.check_range_u32(pos, mem::size_of::<T>() as u32));
      Ok(self.unsafe_peek_be::<T>(pos))
    }
  }

  #[inline(always)]
  fn peek_le<T: Prim>(&self, pos: u32) -> Result<T, ()> {
    unsafe {
      try!(self.check_range_u32(pos, mem::size_of::<T>() as u32));
      Ok(self.unsafe_peek_le::<T>(pos))
    }
  }

  #[inline(always)]
  fn poke(&self, pos: u32, src: &[u8]) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_uint(pos, src.len()));
      Ok(self.unsafe_poke(pos, src))
    }
  }

  #[inline(always)]
  fn poke_be<T: Prim>(&self, pos: u32, t: T) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_u32(pos, mem::size_of::<T>() as u32));
      Ok(self.unsafe_poke_be(pos, t))
    }
  }

  #[inline(always)]
  fn poke_le<T: Prim>(&self, pos: u32, t: T) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_u32(pos, mem::size_of::<T>() as u32));
      Ok(self.unsafe_poke_le(pos, t))
    }
  }

  #[inline(always)]
  fn fill(&mut self, src: &[u8]) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_uint(0, src.len()));
      Ok(self.unsafe_fill(src))
    }
  }

  #[inline(always)]
  fn fill_be<T: Prim>(&mut self, t: T) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_u32(0, mem::size_of::<T>() as u32));
      Ok(self.unsafe_fill_be(t))
    }
  }

  #[inline(always)]
  fn fill_le<T: Prim>(&mut self, t: T) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_u32(0, mem::size_of::<T>() as u32));
      Ok(self.unsafe_fill_le(t)) // Ok, unsafe fillet? om nom.
    }
  }

  #[inline(always)]
  fn consume(&mut self, dst: &mut [u8]) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_uint(0, dst.len()));
      Ok(self.unsafe_consume(dst))
    }
  }

  #[inline(always)]
  fn consume_le<T: Prim>(&mut self) -> Result<T, ()> {
    unsafe {
      try!(self.check_range_u32(0, mem::size_of::<T>() as u32));
      Ok(self.unsafe_consume_le())
    }
  }

  #[inline(always)]
  fn consume_be<T: Prim>(&mut self) -> Result<T, ()> {
    unsafe {
      try!(self.check_range_u32(0, mem::size_of::<T>() as u32));
      Ok(self.unsafe_consume_be())
    }
  }

  #[inline(always)]
  unsafe fn get_at<T: Prim>(&self, pos: u32) -> T {
    self.debug_check_range_u32(pos, 1);
    FromPrimitive::from_u8(
      ptr::read(self.buf.offset((self.lo + pos) as int) as *const u8))
      .unwrap()
  }

  #[inline(always)]
  unsafe fn set_at<T: Prim>(&self, pos: u32, val: T) {
    self.debug_check_range_u32(pos, 1);
    ptr::write(
      self.buf.offset((self.lo + pos) as int),
      val.to_u8().unwrap())
  }

  #[inline]
  unsafe fn unsafe_peek(&self, pos: u32, dst: &mut [u8]) {
    let len = dst.len();
    self.debug_check_range_uint(pos, len);

    let dst: raw::Slice<u8> = mem::transmute(dst);

    ptr::copy_nonoverlapping_memory(
      dst.data as *mut u8,
      self.buf.offset((self.lo + pos) as int) as *const u8,
      len);
  }

  unsafe fn unsafe_peek_be<T: Prim>(&self, pos: u32) -> T {
    let bytes = mem::size_of::<T>() as u32;
    self.debug_check_range_u32(pos, bytes);

    let mut x: T = Zero::zero();

    for i in iter::range(0, bytes) {
      x = self.get_at::<T>(pos+i) | (x << 8);
    }

    x
  }

  unsafe fn unsafe_peek_le<T: Prim>(&self, pos: u32) -> T {
    let bytes = mem::size_of::<T>() as u32;
    self.debug_check_range_u32(pos, bytes);

    let mut x: T = Zero::zero();

    for i in iter::range(0, bytes) {
      x = (x >> 8) | (self.get_at::<T>(pos+i) << ((bytes - 1) * 8) as uint);
    }

    x
  }

  #[inline]
  unsafe fn unsafe_poke(&self, pos: u32, src: &[u8]) {
    let len = src.len();
    self.debug_check_range_uint(pos, len);

    let src: raw::Slice<u8> = mem::transmute(src);

    ptr::copy_nonoverlapping_memory(
      self.buf.offset((self.lo + pos) as int),
      src.data as *const u8,
      len);
  }

  unsafe fn unsafe_poke_be<T: Prim>(&self, pos: u32, t: T) {
    let bytes = mem::size_of::<T>() as u32;
    self.debug_check_range_u32(pos, bytes);

    let msk: T = FromPrimitive::from_u8(0xFFu8).unwrap();

    for i in iter::range(0, bytes) {
      self.set_at(pos+i, (t >> ((bytes-i-1)*8) as uint) & msk);
    }
  }

  unsafe fn unsafe_poke_le<T: Prim>(&self, pos: u32, t: T) {
    let bytes = mem::size_of::<T>() as u32;
    self.debug_check_range_u32(pos, bytes);

    let msk: T = FromPrimitive::from_u8(0xFFu8).unwrap();

    for i in iter::range(0, bytes) {
      self.set_at(pos+i, (t >> (i*8) as uint) & msk);
    }
  }

  #[inline(always)]
  unsafe fn unsafe_fill(&mut self, src: &[u8]) {
    self.debug_check_range_uint(0, src.len());
    self.unsafe_poke(0, src);
    self.lo += src.len() as u32;
  }

  #[inline(always)]
  unsafe fn unsafe_fill_be<T: Prim>(&mut self, t: T) {
    let bytes = mem::size_of::<T>() as u32;
    self.debug_check_range_u32(0, bytes);
    self.unsafe_poke_be(0, t);
    self.lo += bytes;
  }

  #[inline(always)]
  unsafe fn unsafe_fill_le<T: Prim>(&mut self, t: T) {
    let bytes = mem::size_of::<T>() as u32;
    self.debug_check_range_u32(0, bytes);
    self.unsafe_poke_le(0, t);
    self.lo += bytes;
  }

  #[inline(always)]
  unsafe fn unsafe_consume(&mut self, dst: &mut [u8]) {
    self.debug_check_range_uint(0, dst.len());
    self.unsafe_peek(0, dst);
    self.lo += dst.len() as u32;
  }

  #[inline(always)]
  unsafe fn unsafe_consume_le<T: Prim>(&mut self) -> T {
    let bytes = mem::size_of::<T>() as u32;
    self.debug_check_range_u32(0, bytes);
    let ret = self.unsafe_peek_le::<T>(0);
    self.lo += bytes;
    ret
  }

  #[inline(always)]
  unsafe fn unsafe_consume_be<T: Prim>(&mut self) -> T {
    let bytes = mem::size_of::<T>() as u32;
    self.debug_check_range_u32(0, bytes);
    let ret = self.unsafe_peek_be::<T>(0);
    self.lo += bytes;
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

    if      self.len() <= 1 <<  8 { try!(write!(f, "0x{:02x}",  line_number * 8)) }
    else if self.len() <= 1 << 16 { try!(write!(f, "0x{:04x}",  line_number * 8)) }
    else if self.len() <= 1 << 24 { try!(write!(f, "0x{:06x}",  line_number * 8)) }
    else                          { try!(write!(f, "0x{:08x}",  line_number * 8)) }

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
    try!(write!(f, "{} IObuf, limits=[{},{}), bounds=[{},{})\n",
                ty, self.lo_min(), self.hi_max, self.lo, self.hi));

    if self.lo == self.hi { return write!(f, "<empty buffer>"); }

    let b = unsafe { self.as_window_slice() };

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
  /// Copies the data byte-by-byte in the Iobuf into a new, writeable Iobuf.
  /// The new Iobuf and the old Iobuf will not share storage.
  ///
  /// ```
  /// use iobuf::{RWIobuf,Iobuf};
  ///
  /// let mut s = [ 1, 2 ];
  ///
  /// let mut b = RWIobuf::from_slice(&mut s);
  ///
  /// let mut c = b.deep_clone();
  ///
  /// assert_eq!(b.poke_be(0, 0u8), Ok(()));
  /// assert_eq!(c.peek_be::<u8>(0), Ok(1u8));
  /// ```
  fn deep_clone(&self) -> RWIobuf<'static>;

  /// Returns the size of the window.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("Hello");
  /// assert_eq!(b.advance(2), Ok(()));
  /// assert_eq!(b.len(), 3);
  /// ```
  fn len(&self) -> u32;

  /// Returns the size of the limits subrange. The capacity of an iobuf can be
  /// reduced via `narrow`.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("Hello");
  /// assert_eq!(b.advance(2), Ok(()));
  /// assert_eq!(b.cap(), 5);
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
  ///   let num_bytes = num_shorts as u32 * mem::size_of::<u16>() as u32;
  ///
  ///   unsafe {
  ///     try!(b.check_range(0, num_bytes));
  ///
  ///     let mut sum = 0u16;
  ///
  ///     for i in range(0, num_shorts as u32)
  ///                .map(|x| x * mem::size_of::<u16>() as u32) {
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
  ///   let num_bytes = num_shorts as u32 * mem::size_of::<u16>() as u32;
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
  /// let mut a = ROIobuf::from_string("hello".to_string());
  /// let mut b = a.clone();
  /// let mut c = a.clone();
  /// let mut d = ROIobuf::from_string("hello".to_string());
  ///
  /// assert_eq!(a.sub_window_to(2), Ok(()));
  ///
  /// // b actually IS an extension of a.
  /// assert_eq!(b.sub_window_from(2), Ok(()));
  /// assert_eq!(a.is_extended_by_ro(&b), true);
  ///
  /// // a == "he", b == "lo", it's missing the "l", therefore not an extension.
  /// assert_eq!(c.sub_window_from(3), Ok(()));
  /// assert_eq!(b.is_extended_by_ro(&a), false);
  ///
  /// // Different allocations => not an extension.
  /// assert_eq!(d.sub_window_from(2), Ok(()));
  /// assert_eq!(a.is_extended_by_ro(&d), false);
  /// ```
  fn is_extended_by_ro<'a>(&self, other: &ROIobuf<'a>) -> bool;

  /// The same as `is_extended_by_ro`, but the `other` buf is writable. They
  /// both work the same, so just use whatever works for the type you have.
  fn is_extended_by_rw<'a>(&self, other: &RWIobuf<'a>) -> bool;

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

  /// Sets the lower bound of the window to the lower limit.
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
  /// assert_eq!(unsafe { b.as_window_slice() }, b"hel");
  /// assert_eq!(b.advance(2), Ok(()));
  /// assert_eq!(unsafe { b.as_window_slice() }, b"l");
  /// b.reset();
  /// assert_eq!(unsafe { b.as_window_slice() }, b"hello");
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
}

/// An `Iobuf` that cannot write into the buffer, but all read-only operations
/// are supported. It is possible to get a `RWIobuf` by performing a `deep_clone`
/// of the Iobuf, but this is extremely inefficient.
///
/// If your function only needs to do read-only operations on an Iobuf, consider
/// taking a generic `Iobuf` trait instead. That way, it can be used with either
/// a ROIobuf or a RWIobuf, generically.
pub struct ROIobuf<'a> {
  raw: RawIobuf<'a>,
}

impl<'a> Clone for ROIobuf<'a> {
  fn clone(&self) -> ROIobuf<'a> {
    ROIobuf {
      raw: self.raw.clone()
    }
  }
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
pub struct RWIobuf<'a> {
  raw: RawIobuf<'a>,
}

impl<'a> Clone for RWIobuf<'a> {
  fn clone(&self) -> RWIobuf<'a> {
    RWIobuf {
      raw: self.raw.clone()
    }
  }
}

impl<'a> ROIobuf<'a> {
  /// Constructs a trivially empty Iobuf, limits and window are 0, and there's
  /// an empty backing buffer. Unfortunately, that backing buffer is refcounted,
  /// so this still needs an allocation.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::empty();
  ///
  /// assert_eq!(b.cap(), 0);
  /// assert_eq!(b.len(), 0);
  /// ```
  #[inline]
  pub fn empty() -> ROIobuf<'static> {
    ROIobuf { raw: RawIobuf::empty() }
  }

  /// Constructs an Iobuf with the same contents as a string. The limits and
  /// window will be initially set to cover the whole string.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_str("hello");
  ///
  /// assert_eq!(b.cap(), 5);
  /// assert_eq!(b.len(), 5);
  /// unsafe { assert_eq!(b.as_window_slice(), b"hello"); }
  /// unsafe { assert_eq!(b.as_limit_slice(), b"hello"); }
  /// ```
  #[inline]
  pub fn from_str<'a>(s: &'a str) -> ROIobuf<'a> {
    ROIobuf { raw: RawIobuf::from_str(s) }
  }

  /// Directly converts a string into a read-only Iobuf. The Iobuf will take
  /// ownership of the string, therefore there will be no copying.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut b = ROIobuf::from_string("hello".into_string());
  ///
  /// assert_eq!(b.cap(), 5);
  /// assert_eq!(b.len(), 5);
  /// unsafe { assert_eq!(b.as_window_slice(), b"hello"); }
  /// unsafe { assert_eq!(b.as_limit_slice(), b"hello"); }
  /// ```
  #[inline]
  pub fn from_string(s: String) -> ROIobuf<'static> {
    ROIobuf { raw: RawIobuf::from_string(s) }
  }

  /// Directly converts a byte vector into a read-only Iobuf. The Iobuf will
  /// take ownership of the vector, therefore there will be no copying.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let mut v = vec!(1u8, 2, 3, 4, 5, 6);
  /// v.as_mut_slice()[1] = 20;
  ///
  /// let mut b = ROIobuf::from_vec(v);
  ///
  /// let expected = [ 1,20,3,4,5,6 ];
  /// unsafe { assert_eq!(b.as_window_slice(), expected.as_slice()); }
  /// ```
  #[inline]
  pub fn from_vec(v: Vec<u8>) -> ROIobuf<'static> {
    ROIobuf { raw: RawIobuf::from_vec(v) }
  }

  /// Construclts an Iobuf from a slice. The Iobuf will not copy the slice
  /// contents, and therefore their lifetimes will be linked.
  ///
  /// ```
  /// use iobuf::{ROIobuf,Iobuf};
  ///
  /// let s = [1,2,3,4];
  ///
  /// let mut b = ROIobuf::from_slice(s.as_slice());
  ///
  /// assert_eq!(b.advance(1), Ok(()));
  ///
  /// assert_eq!(s[1], 2); // we can still use the slice!
  /// assert_eq!(b.peek_be(1), Ok(0x0304u16)); // ...and the Iobuf!
  /// ```
  #[inline]
  pub fn from_slice<'a>(s: &'a [u8]) -> ROIobuf<'a> {
    ROIobuf { raw: RawIobuf::from_slice(s) }
  }
}

impl<'a> RWIobuf<'a> {
  /// Constructs a trivially empty Iobuf, limits and window are 0, and there's
  /// an empty backing buffer. Unfortunately, that backing buffer is refcounted,
  /// so this still needs an allocation.
  ///
  /// ```
  /// use iobuf::{RWIobuf,Iobuf};
  ///
  /// let mut b = RWIobuf::empty();
  ///
  /// assert_eq!(b.len(), 0);
  /// assert_eq!(b.cap(), 0);
  /// ```
  #[inline]
  pub fn empty() -> RWIobuf<'static> {
    RWIobuf { raw: RawIobuf::empty() }
  }

  /// Constructs a new Iobuf with a buffer of size `len`, undefined contents,
  /// and the limits and window set to the full size of the buffer.
  ///
  /// ```
  /// use iobuf::{RWIobuf,Iobuf};
  ///
  /// let mut b = RWIobuf::new(10);
  ///
  /// assert_eq!(b.len(), 10);
  /// assert_eq!(b.cap(), 10);
  /// ```
  #[inline(always)]
  pub fn new(len: uint) -> RWIobuf<'static> {
    RWIobuf { raw: RawIobuf::new(len) }
  }

  /// Directly converts a string into a writeable Iobuf. The Iobuf will take
  /// ownership of the string, therefore there will be no copying.
  ///
  /// ```
  /// use iobuf::{RWIobuf,Iobuf};
  ///
  /// let mut b = RWIobuf::from_string("hello".into_string());
  ///
  /// b.poke_be(1, b'4').unwrap();
  ///
  /// assert_eq!(b.len(), 5);
  /// assert_eq!(b.cap(), 5);
  /// unsafe { assert_eq!(b.as_window_slice(), b"h4llo"); }
  /// unsafe { assert_eq!(b.as_limit_slice(), b"h4llo"); }
  /// ```
  #[inline(always)]
  pub fn from_string(s: String) -> RWIobuf<'static> {
    RWIobuf { raw: RawIobuf::from_string(s) }
  }

  /// Directly converts a byte vector into a writeable Iobuf. The Iobuf will
  /// take ownership of the vector, therefore there will be no copying.
  ///
  /// ```
  /// use iobuf::{RWIobuf,Iobuf};
  ///
  /// let mut v = vec!(1u8, 2, 3, 4, 5, 6, 10);
  /// v.as_mut_slice()[1] = 20;
  ///
  /// let mut b = RWIobuf::from_vec(v);
  ///
  /// let expected = [ 1,20,3,4,5,6,10 ];
  /// unsafe { assert_eq!(b.as_window_slice(), expected.as_slice()); }
  /// ```
  #[inline(always)]
  pub fn from_vec(v: Vec<u8>) -> RWIobuf<'static> {
    RWIobuf { raw: RawIobuf::from_vec(v) }
  }

  /// Construclts an Iobuf from a slice. The Iobuf will not copy the slice
  /// contents, and therefore their lifetimes will be linked.
  ///
  /// ```
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
  pub fn from_slice<'a>(s: &'a mut [u8]) -> RWIobuf<'a> {
    RWIobuf { raw: RawIobuf::from_slice(s) }
  }

  /// Reads the data in the window as a mutable slice. Note that since `&mut`
  /// in rust really means `&unique`, this function lies. There can exist
  /// multiple slices of the same data. Therefore, this function is unsafe.
  ///
  /// It may only be used safely if you ensure that the data in the iobuf never
  /// interacts with the slice, as they point to the same data. `peek`ing or
  /// `poke`ing the slice returned from this function is a big no-no.
  ///
  /// ```
  /// use iobuf::{RWIobuf, Iobuf};
  ///
  /// let mut s = [1,2,3];
  ///
  /// {
  ///   let mut b = RWIobuf::from_slice(&mut s);
  ///
  ///   assert_eq!(b.advance(1), Ok(()));
  ///   unsafe { b.as_window_slice_mut()[1] = 30; }
  /// }
  ///
  /// let expected = [ 1,2,30 ];
  /// assert_eq!(s.as_slice(), expected.as_slice());
  /// ```
  #[inline(always)]
  pub unsafe fn as_window_slice_mut<'b>(&'b self) -> &'b mut [u8] {
    self.raw.as_window_slice_mut()
  }

  /// Reads the data in the window as a mutable slice. Note that since `&mut`
  /// in rust really means `&unique`, this function lies. There can exist
  /// multiple slices of the same data. Therefore, this function is unsafe.
  ///
  /// It may only be used safely if you ensure that the data in the iobuf never
  /// interacts with the slice, as they point to the same data. `peek`ing or
  /// `poke`ing the slice returned from this function is a big no-no.
  ///
  /// ```
  /// use iobuf::{RWIobuf, Iobuf};
  ///
  /// let mut s = [1,2,3];
  ///
  /// {
  ///   let mut b = RWIobuf::from_slice(&mut s);
  ///
  ///   assert_eq!(b.advance(1), Ok(()));
  ///   unsafe { b.as_limit_slice_mut()[1] = 20; }
  /// }
  ///
  /// assert_eq!(s.as_slice(), [1,20,3].as_slice());
  /// ```
  #[inline(always)]
  pub unsafe fn as_limit_slice_mut<'b>(&'b self) -> &'b mut [u8] {
    self.raw.as_limit_slice_mut()
  }

  /// Gets a read-only copy of this Iobuf. This is a very cheap operation, as
  /// the backing buffers are shared. This can be useful for interfacing with
  /// code that only accepts read-only Iobufs.
  ///
  /// In general, ROIobuf should never be used as a function parameter. If
  /// read-only acceess is all that is required, take a generic `<T: Iobuf>`.
  ///
  /// ```
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
    ROIobuf { raw: self.raw.clone() }
  }

  /// Copies data from the window to the lower limit fo the iobuf and sets the
  /// window to range from the end of the copied data to the upper limit. This
  /// is typically called after a series of `Consume`s to save unread data and
  /// prepare for the next series of `Fill`s and `flip_lo`s.
  ///
  /// ```
  /// use std::iter::range;
  /// use std::result::{Result,Ok};
  /// use iobuf::{RWIobuf,Iobuf};
  ///
  /// // A header, saying how many shorts will follow. Unfortunately, our buffer
  /// // isn't big enough for all the shorts! Assume the rest will be sent in a
  /// // later packet.
  /// let mut s = [ 0x02, 0x11, 0x22, 0x33 ];
  /// let mut b = RWIobuf::from_slice(s.as_mut_slice());
  ///
  /// #[deriving(Eq, PartialEq, Show)]
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
  ///   for _ in range(0u8, len) {
  ///     unsafe {
  ///       if b.len() < 2 {
  ///         b.compact();
  ///         return Ok(NeedMore(sum));
  ///       }
  ///       sum += b.unsafe_consume_be();
  ///     }
  ///   }
  ///
  ///   Ok(Done(sum))
  /// }
  ///
  /// assert_eq!(parse(&mut b), Ok(NeedMore(0x1122)));
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
  /// ```
  /// use iobuf::{RWIobuf,Iobuf};
  ///
  /// let data = [ 1,2,3,4 ];
  ///
  /// let mut b = RWIobuf::new(10);
  ///
  /// assert_eq!(b.poke(0, data.as_slice()), Ok(()));
  /// assert_eq!(b.poke(3, data.as_slice()), Ok(()));
  /// assert_eq!(b.resize(7), Ok(()));
  /// assert_eq!(b.poke(4, data.as_slice()), Err(())); // no partial write, just failure
  ///
  /// let expected = [ 1,2,3,1,2,3,4 ];
  /// unsafe { assert_eq!(b.as_window_slice(), expected.as_slice()); }
  /// ```
  #[inline(always)]
  pub fn poke(&self, pos: u32, src: &[u8]) -> Result<(), ()> { self.raw.poke(pos, src) }

  /// Writes a big-endian primitive at a given offset from the beginning of the
  /// window.
  ///
  /// An error is returned if bytes outside of the window would be accessed.
  ///
  /// ```
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
  /// unsafe { assert_eq!(b.as_window_slice(), expected.as_slice()); }
  /// ```
  #[inline(always)]
  pub fn poke_be<T: Prim>(&self, pos: u32, t: T) -> Result<(), ()> { self.raw.poke_be(pos, t) }

  /// Writes a little-endian primitive at a given offset from the beginning of
  /// the window.
  ///
  /// An error is returned if bytes outside of the window would be accessed.
  ///
  /// ```
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
  /// unsafe { assert_eq!(b.as_window_slice(), [ 4, 5, 5, 9, 8, 7, 6 ].as_slice()); }
  /// ```
  #[inline(always)]
  pub fn poke_le<T: Prim>(&self, pos: u32, t: T) -> Result<(), ()> { self.raw.poke_le(pos, t) }

  /// Writes bytes from the supplied buffer, starting from the front of the
  /// window. Either the entire buffer is copied, or an error is returned
  /// because bytes outside the window were requested.
  ///
  /// After the bytes have been written, the window will be moved to no longer
  /// include then.
  ///
  /// ```
  /// use iobuf::{RWIobuf,Iobuf};
  ///
  /// let data = [ 1, 2, 3, 4 ];
  ///
  /// let mut b = RWIobuf::new(10);
  ///
  /// assert_eq!(b.fill(data.as_slice()), Ok(()));
  /// assert_eq!(b.fill(data.as_slice()), Ok(()));
  /// assert_eq!(b.fill(data.as_slice()), Err(()));
  ///
  /// b.flip_lo();
  ///
  /// unsafe { assert_eq!(b.as_window_slice(), [ 1,2,3,4,1,2,3,4 ].as_slice()); }
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
  /// ```
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
  ///                                          , 0x88, 0x77 ].as_slice()); }
  /// ```
  #[inline(always)]
  pub fn fill_be<T: Prim>(&mut self, t: T) -> Result<(), ()> { self.raw.fill_be(t) }

  /// Writes a little-endian primitive into the beginning of the window.
  ///
  /// After the primitive has been written, the window will be moved such that
  /// it is no longer included.
  ///
  /// An error is returned if bytes outside of the window were requested.
  ///
  /// ```
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
  ///                                          , 0x77, 0x88 ].as_slice()); }
  /// ```
  #[inline(always)]
  pub fn fill_le<T: Prim>(&mut self, t: T) -> Result<(), ()> { self.raw.fill_le(t) }

  /// Writes the bytes at a given offset from the beginning of the window, into
  /// the supplied buffer. It is undefined behavior to write outside the iobuf
  /// window.
  ///
  /// ```
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
  ///   b.unsafe_poke(0, data);
  ///   b.unsafe_poke(3, data);
  ///   b.unsafe_advance(7);
  /// }
  ///
  /// b.flip_lo();
  ///
  /// unsafe { assert_eq!(b.as_window_slice(), [ 1,2,3,1,2,3,4 ].as_slice()); }
  /// ```
  #[inline(always)]
  pub unsafe fn unsafe_poke(&self, pos: u32, src: &[u8]) { self.raw.unsafe_poke(pos, src) }

  /// Writes a big-endian primitive at a given offset from the beginning of the
  /// window. It is undefined behavior to write outside the iobuf window.
  ///
  /// ```
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
  /// unsafe { assert_eq!(b.as_window_slice(), [ 3, 5, 5, 6, 7, 8, 9 ].as_slice()); }
  /// ```
  #[inline(always)]
  pub unsafe fn unsafe_poke_be<T: Prim>(&self, pos: u32, t: T) { self.raw.unsafe_poke_be(pos, t) }

  /// Writes a little-endian primitive at a given offset from the beginning of
  /// the window. It is undefined behavior to write outside the iobuf window.
  ///
  /// ```
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
  /// unsafe { assert_eq!(b.as_window_slice(), [ 4, 5, 5, 9, 8, 7, 6 ].as_slice()); }
  /// ```
  #[inline(always)]
  pub unsafe fn unsafe_poke_le<T: Prim>(&self, pos: u32, t: T) { self.raw.unsafe_poke_le(pos, t) }

  /// Writes bytes from the supplied buffer, starting from the front of the
  /// window. It is undefined behavior to write outside the iobuf window.
  ///
  /// After the bytes have been written, the window will be moved to no longer
  /// include then.
  ///
  /// ```
  /// use iobuf::{RWIobuf,Iobuf};
  ///
  /// let data = [ 1, 2, 3, 4 ];
  ///
  /// let mut b = RWIobuf::new(10);
  ///
  /// unsafe {
  ///   b.check_range_fail(0, 8);
  ///
  ///   b.unsafe_fill(data.as_slice());
  ///   b.unsafe_fill(data.as_slice());
  /// }
  ///
  /// b.flip_lo();
  ///
  /// unsafe { assert_eq!(b.as_window_slice(), [ 1,2,3,4,1,2,3,4 ].as_slice()); }
  /// ```
  #[inline(always)]
  pub unsafe fn unsafe_fill(&mut self, src: &[u8]) { self.raw.unsafe_fill(src) }

  /// Writes a big-endian primitive into the beginning of the window. It is
  /// undefined behavior to write outside the iobuf window.
  ///
  /// After the primitive has been written, the window will be moved such that
  /// it is no longer included.
  ///
  /// ```
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
  ///                                          , 0x88, 0x77 ].as_slice()); }
  /// ```
  #[inline(always)]
  pub unsafe fn unsafe_fill_be<T: Prim>(&mut self, t: T) { self.raw.unsafe_fill_be(t) }

  /// Writes a little-endian primitive into the beginning of the window. It is
  /// undefined behavior to write outside the iobuf window.
  ///
  /// After the primitive has been written, the window will be moved such that
  /// it is no longer included.
  ///
  /// ```
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
  ///                                          , 0x77, 0x88 ].as_slice()); }
  /// ```
  #[inline(always)]
  pub unsafe fn unsafe_fill_le<T: Prim>(&mut self, t: T) { self.raw.unsafe_fill_le(t) }
}

impl<'a> Iobuf for ROIobuf<'a> {
  #[inline(always)]
  fn deep_clone(&self) -> RWIobuf<'static> { RWIobuf { raw: self.raw.deep_clone() } }

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
  fn is_extended_by_ro<'a>(&self, other: &ROIobuf<'a>) -> bool { self.raw.is_extended_by_ro(other) }

  #[inline(always)]
  fn is_extended_by_rw<'a>(&self, other: &RWIobuf<'a>) -> bool { self.raw.is_extended_by_rw(other) }

  #[inline(always)]
  fn resize(&mut self, len: u32) -> Result<(), ()> { self.raw.resize(len) }

  #[inline(always)]
  unsafe fn unsafe_resize(&mut self, len: u32) { self.raw.unsafe_resize(len) }

  #[inline(always)]
  fn rewind(&mut self) { self.raw.rewind() }

  #[inline(always)]
  fn reset(&mut self) { self.raw.reset() }

  #[inline(always)]
  fn flip_lo(&mut self) { self.raw.flip_lo() }

  #[inline(always)]
  fn flip_hi(&mut self) { self.raw.flip_hi() }

  #[inline(always)]
  fn peek(&self, pos: u32, dst: &mut [u8]) -> Result<(), ()> { self.raw.peek(pos, dst) }
  #[inline(always)]
  fn peek_be<T: Prim>(&self, pos: u32) -> Result<T, ()> { self.raw.peek_be(pos) }
  #[inline(always)]
  fn peek_le<T: Prim>(&self, pos: u32) -> Result<T, ()> { self.raw.peek_le(pos) }

  #[inline(always)]
  fn consume(&mut self, dst: &mut [u8]) -> Result<(), ()> { self.raw.consume(dst) }
  #[inline(always)]
  fn consume_be<T: Prim>(&mut self) -> Result<T, ()> { self.raw.consume_be::<T>() }
  #[inline(always)]
  fn consume_le<T: Prim>(&mut self) -> Result<T, ()> { self.raw.consume_le::<T>() }

  #[inline(always)]
  fn check_range(&self, pos: u32, len: u32) -> Result<(), ()> { self.raw.check_range_u32(pos, len) }

  #[inline(always)]
  fn check_range_uint(&self, pos: u32, len: uint) -> Result<(), ()> { self.raw.check_range_uint(pos, len) }

  #[inline(always)]
  fn check_range_fail(&self, pos: u32, len: u32) { self.raw.check_range_u32_fail(pos, len) }

  #[inline(always)]
  fn check_range_uint_fail(&self, pos: u32, len: uint) { self.raw.check_range_uint_fail(pos, len) }

  #[inline(always)]
  unsafe fn unsafe_peek(&self, pos: u32, dst: &mut [u8]) { self.raw.unsafe_peek(pos, dst) }
  #[inline(always)]
  unsafe fn unsafe_peek_be<T: Prim>(&self, pos: u32) -> T { self.raw.unsafe_peek_be(pos) }
  #[inline(always)]
  unsafe fn unsafe_peek_le<T: Prim>(&self, pos: u32) -> T { self.raw.unsafe_peek_le(pos) }

  #[inline(always)]
  unsafe fn unsafe_consume(&mut self, dst: &mut [u8]) { self.raw.unsafe_consume(dst) }
  #[inline(always)]
  unsafe fn unsafe_consume_be<T: Prim>(&mut self) -> T { self.raw.unsafe_consume_be::<T>() }
  #[inline(always)]
  unsafe fn unsafe_consume_le<T: Prim>(&mut self) -> T { self.raw.unsafe_consume_le::<T>() }
}

impl<'a> Iobuf for RWIobuf<'a> {
  #[inline(always)]
  fn deep_clone(&self) -> RWIobuf<'static> { RWIobuf { raw: self.raw.deep_clone() } }

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
  fn is_extended_by_ro<'a>(&self, other: &ROIobuf<'a>) -> bool { self.raw.is_extended_by_ro(other) }

  #[inline(always)]
  fn is_extended_by_rw<'a>(&self, other: &RWIobuf<'a>) -> bool { self.raw.is_extended_by_rw(other) }

  #[inline(always)]
  fn resize(&mut self, len: u32) -> Result<(), ()> { self.raw.resize(len) }

  #[inline(always)]
  unsafe fn unsafe_resize(&mut self, len: u32) { self.raw.unsafe_resize(len) }

  #[inline(always)]
  fn rewind(&mut self) { self.raw.rewind() }

  #[inline(always)]
  fn reset(&mut self) { self.raw.reset() }

  #[inline(always)]
  fn flip_lo(&mut self) { self.raw.flip_lo() }

  #[inline(always)]
  fn flip_hi(&mut self) { self.raw.flip_hi() }

  #[inline(always)]
  fn peek(&self, pos: u32, dst: &mut [u8]) -> Result<(), ()> { self.raw.peek(pos, dst) }
  #[inline(always)]
  fn peek_be<T: Prim>(&self, pos: u32) -> Result<T, ()> { self.raw.peek_be(pos) }
  #[inline(always)]
  fn peek_le<T: Prim>(&self, pos: u32) -> Result<T, ()> { self.raw.peek_le(pos) }

  #[inline(always)]
  fn consume(&mut self, dst: &mut [u8]) -> Result<(), ()> { self.raw.consume(dst) }
  #[inline(always)]
  fn consume_be<T: Prim>(&mut self) -> Result<T, ()> { self.raw.consume_be::<T>() }
  #[inline(always)]
  fn consume_le<T: Prim>(&mut self) -> Result<T, ()> { self.raw.consume_le::<T>() }

  #[inline(always)]
  fn check_range(&self, pos: u32, len: u32) -> Result<(), ()> { self.raw.check_range_u32(pos, len) }

  #[inline(always)]
  fn check_range_uint(&self, pos: u32, len: uint) -> Result<(), ()> { self.raw.check_range_uint(pos, len) }

  #[inline(always)]
  fn check_range_fail(&self, pos: u32, len: u32) { self.raw.check_range_u32_fail(pos, len) }

  #[inline(always)]
  fn check_range_uint_fail(&self, pos: u32, len: uint) { self.raw.check_range_uint_fail(pos, len) }

  #[inline(always)]
  unsafe fn unsafe_peek(&self, pos: u32, dst: &mut [u8]) { self.raw.unsafe_peek(pos, dst) }
  #[inline(always)]
  unsafe fn unsafe_peek_be<T: Prim>(&self, pos: u32) -> T { self.raw.unsafe_peek_be(pos) }
  #[inline(always)]
  unsafe fn unsafe_peek_le<T: Prim>(&self, pos: u32) -> T { self.raw.unsafe_peek_le(pos) }

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

#[test]
fn peek_be() {
  let s = [1,2,3,4];
  let b = ROIobuf::from_slice(&s);
  assert_eq!(b.peek_be(0), Ok(0x01020304u32));
}

#[test]
fn peek_le() {
  let s = [1,2,3,4];
  let b = ROIobuf::from_slice(&s);
  assert_eq!(b.peek_le(0), Ok(0x04030201u32));
}

#[test]
fn poke_be() {
  let b = RWIobuf::new(4);
  assert_eq!(b.poke_be(0, 0x01020304u32), Ok(()));
  let expected = [ 1,2,3,4 ];
  unsafe { assert_eq!(b.as_window_slice(), expected.as_slice()); }
}

#[test]
fn poke_le() {
  let b = RWIobuf::new(4);
  assert_eq!(b.poke_le(0, 0x01020304u32), Ok(()));
  let expected = [ 4,3,2,1 ];
  unsafe { assert_eq!(b.as_window_slice(), expected.as_slice()); }
}
