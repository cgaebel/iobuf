use alloc::heap;

use core::nonzero::NonZero;

use std::fmt::{self, Formatter};
use std::io;
use std::marker::PhantomData;
use std::mem;
use std::ptr;
use std::raw::{self, Repr};
use std::i32;
use std::u32;
use std::sync::Arc;
use std::sync::atomic::{self, AtomicUsize, Ordering};

use intlike::{IntLike, from_be, from_le, to_be, to_le};

#[cfg(target_pointer_width = "64")]
const TARGET_WORD_SIZE: usize = 64;

#[cfg(target_pointer_width = "32")]
const TARGET_WORD_SIZE: usize = 32;

/// All Iobuf data is aligned to 16 bytes. This allows people to write more
/// efficient SIMD code on Iobufs.
const DATA_ALIGNMENT: usize = 16;

/// The biggest Iobuf supported, to allow us to have a small RawIobuf struct.
/// By limiting the buffer sizes, we can bring the struct down from 40 bytes to
/// 24 bytes -- a 40% reduction. This frees up precious cache and registers for
/// the actual processing.
const MAX_BUFFER_LEN: usize = i32::MAX as usize - ALLOCATION_HEADER_SIZE;

/// The bitmask to get the "is the buffer owned" bit.
const OWNED_MASK: u32 = 1 << (u32::BITS - 1);

/// Used to provide custom memory to Iobufs, instead of just using the heap.
pub trait Allocator: Sync + Send {
  /// Allocates `len` bytes of memory, with an alignment of `align`.
  fn allocate(&self, len: usize, align: usize) -> *mut u8;

  /// Deallocates memory allocated by `allocate`.
  fn deallocate(&self, ptr: NonZero<*mut u8>, len: usize, align: usize);
}

struct AllocationHeader {
  allocator: Option<NonZero<*mut ()>>,
  allocation_length: usize,
  refcount: usize,
  _pad: usize,
}

// Needed because size_of isn't compile-time.
const ALLOCATION_HEADER_SIZE: usize = 4*TARGET_WORD_SIZE/8;

#[test]
fn correct_header_size() {
  assert_eq!(ALLOCATION_HEADER_SIZE, mem::size_of::<AllocationHeader>());
  assert!(ALLOCATION_HEADER_SIZE % DATA_ALIGNMENT == 0);
  assert!(mem::align_of::<AllocationHeader>() <= DATA_ALIGNMENT);
}

#[test]
fn result_of_boxed_string() {
  assert_eq!(mem::size_of::<Result<(), Box<String>>>(),
             mem::size_of::<           Box<String> >());
}

/// Rust has the wrong parameter order.
/// > http://internals.rust-lang.org/t/memcpy-is-backwards/1797
#[inline(always)]
unsafe fn memcpy<T>(dst: *mut T, src: *const T, n: usize) {
  ptr::copy_nonoverlapping(src, dst, n)
}

/// Rust has the wrong parameter order.
/// > http://internals.rust-lang.org/t/memcpy-is-backwards/1797
#[inline(always)]
unsafe fn memmove<T>(dst: *mut T, src: *const T, n: usize) {
  ptr::copy(src, dst, n)
}

impl AllocationHeader {
  #[inline]
  fn allocate(&self, len: usize) -> *mut u8 {
    unsafe {
      match self.allocator {
        None => heap::allocate(len, DATA_ALIGNMENT),
        Some(allocator) => {
          let allocator: &Arc<Box<Allocator>> = mem::transmute(&*allocator);
          allocator.allocate(len, DATA_ALIGNMENT)
        }
      }
    }
  }

  #[inline(always)]
  unsafe fn nonatomic_refcount(&self) -> usize {
    self.refcount
  }

  #[inline(always)]
  unsafe fn atomic_refcount<'a>(&'a self) -> &'a AtomicUsize {
    mem::transmute(&self.refcount)
  }

  #[inline(always)]
  unsafe fn inc_ref_count_atomic(&mut self) {
    self.atomic_refcount().fetch_add(1, Ordering::Relaxed);
  }

  #[inline(always)]
  unsafe fn inc_ref_count_nonatomic(&mut self) {
    self.refcount += 1;
  }

  #[inline]
  fn deallocator(&self) -> Deallocator {
    unsafe {
      Deallocator {
        allocation_length: self.allocation_length,
        allocator:
          match self.allocator {
            None => None,
            Some(allocator) => mem::transmute(*allocator),
          },
      }
    }
  }

  #[inline]
  #[must_use]
  unsafe fn dec_ref_count_nonatomic(&mut self) -> Result<(), ()> {
    debug_assert!(self.refcount != 0);
    self.refcount -= 1;
    err_if(self.refcount == 0)
  }

  #[inline]
  #[must_use]
  unsafe fn dec_ref_count_atomic(&mut self) -> Result<(), ()> {
    if self.atomic_refcount().fetch_sub(1, Ordering::Release) == 1 {
      atomic::fence(Ordering::Acquire);
      Err(())
    } else {
      Ok(())
    }
  }

  // Keep this out of line to allow inlining of the drop glue.
  #[cold]
  unsafe fn deallocate(&self, buf: NonZero<*mut u8>) {
    self.deallocator().deallocate(buf)
  }
}

/// Each leaf of this enum is tagged with the allocation length listed in the
/// header.
struct Deallocator {
  allocation_length: usize,
  allocator: Option<Arc<Box<Allocator>>>,
}

impl Deallocator {
  fn deallocate(self, ptr: NonZero<*mut u8>) {
    unsafe {
      let ptr: *mut u8 = ptr.offset(-(ALLOCATION_HEADER_SIZE as isize));
      match self.allocator {
        None =>
          heap::deallocate(ptr, self.allocation_length, DATA_ALIGNMENT),
        Some(alloc) =>
          alloc.deallocate(NonZero::new(ptr), self.allocation_length, DATA_ALIGNMENT),
      }
    }
  }
}

// By factoring out the calls to `panic!`, we prevent rustc from emitting a ton
// of formatting code in our tight, little functions, and also help guide
// inlining.

#[cold]
fn bad_range(pos: u64, len: u64) -> ! {
  panic!("Iobuf got invalid range: pos={}, len={}", pos, len)
}

#[cold]
fn buffer_too_big(actual_size: usize) -> ! {
  panic!("Tried to create an Iobuf that's too big: {} bytes. Max size = {}",
         actual_size, MAX_BUFFER_LEN)
}

#[cold]
fn allocator_returned_null() -> ! {
  panic!("Iobuf's allocator returned NULL. Out of memory?");
}

#[cold]
fn improperly_aligned_data(ptr: *mut u8) -> ! {
  panic!("{:?} is not aligned to {} bytes.", ptr, DATA_ALIGNMENT);
}

#[cold]
pub fn write_failed() -> io::Result<usize> {
  Err(io::Error::new(io::ErrorKind::Other, "No space left in iobuf."))
}

#[inline(always)]
fn err_if(cond: bool) -> Result<(), ()> {
  if cond { Err(()) } else { Ok(()) }
}

/// A `RawIobuf` is the representation of both a `RWIobuf` and a `ROIobuf`.
/// It is very cheap to clone, as the backing buffer is shared and refcounted.
pub struct RawIobuf<'a> {
  // Starting at `buf` is the raw data itself.
  // If the buf was allocated by us (i.e. the owned bit is set), the bytes
  // immediately preceding represent the allocation header. See the `header`
  // function.
  buf:    NonZero<*mut u8>,
  // If the highest bit of this is set, `buf` is owned and the data before the
  // pointeer is valid. If it is not set, then the buffer wasn't allocated by us:
  // it's owned by someone else. Therefore, there's no header, and no need to
  // deallocate or refcount.
  lo_min_and_owned_bit: u32,
  lo:     u32,
  hi:     u32,
  hi_max: u32,
  lifetm: PhantomData<&'a ()>,
}

/// Make sure the compiler doesn't resize this to something silly.
#[test]
fn check_sane_raw_size() {
  assert_eq!(mem::size_of::<RawIobuf>(), mem::size_of::<*mut u8>() + 16);
}

impl<'a> RawIobuf<'a> {
  pub fn new_impl(
      len:       usize,
      allocator: Option<NonZero<*mut ()>>) -> RawIobuf<'static> {
    unsafe {
      if len > MAX_BUFFER_LEN {
        buffer_too_big(len);
      }

      let data_len = ALLOCATION_HEADER_SIZE + len;

      let allocation_header =
        AllocationHeader {
          allocator:         allocator,
          allocation_length: data_len,
          refcount:          1,
          _pad:              0,
        };

      let buf = allocation_header.allocate(data_len);

      if buf.is_null() {
        allocator_returned_null();
      }

      ptr::write(buf as *mut AllocationHeader, allocation_header);

      let buf: *mut u8 = buf.offset(ALLOCATION_HEADER_SIZE as isize);

      if buf as usize % DATA_ALIGNMENT != 0 {
        improperly_aligned_data(buf);
      }

      RawIobuf {
        buf:    NonZero::new(buf),
        lo_min_and_owned_bit: OWNED_MASK,
        lo:     0,
        hi:     len as u32,
        hi_max: len as u32,
        lifetm: PhantomData,
      }
    }
  }

  #[inline]
  pub fn new(len: usize) -> RawIobuf<'static> {
    RawIobuf::new_impl(len, None)
  }

  #[inline]
  pub fn new_with_allocator(len: usize, allocator: Arc<Box<Allocator>>) -> RawIobuf<'static> {
    unsafe {
      let allocator: *mut () = mem::transmute(allocator);
      RawIobuf::new_impl(len, Some(NonZero::new(allocator)))
    }
  }

  #[cold]
  fn inv_fail(&self, msg: &str) -> Box<String> {
    Box::new(format!("{}: Iobuf {{ buf: {:?}, owned: {}, lo_min: {}, lo: {}, hi: {}, hi_max: {} }}",
      msg,
      *self.buf, self.is_owned(), self.lo_min(), self.lo(), self.hi(), self.hi_max()))
  }

  /// Checks a single invariant condition, returning a boxed (for size savings)
  /// error in the case of failure, with a descriptive message.
  #[inline]
  pub fn inv_check(&self, is_valid: bool, msg: &str) -> Result<(), Box<String>> {
    if is_valid {
      Ok(())
    } else {
      Err(self.inv_fail(msg))
    }
  }

  pub fn invariant(&self) -> Result<(), Box<String>> {
    try!(self.inv_check(self.lo_min() <= self.hi_max(),
      "The end of the limits must be after the start of the limits"));

    try!(self.inv_check(self.lo() <= self.hi(),
      "The end of the window must be after the start of the window"));

    try!(self.inv_check(self.lo_min() <= self.lo(),
      "The start of the window must be after the start of the limits"));

    try!(self.inv_check(self.hi() <= self.hi_max(),
      "The end of the limits must be after the end of the window"));

    Ok(())
  }

  #[cfg(not(debug))]
  #[inline(always)]
  fn debug_check_invariants<T>(&self, t: T) -> T {
    t
  }

  #[cfg(debug)]
  fn debug_check_invariants<T>(&self, t: T) -> T {
    match self.invariant() {
      Ok(())   => t,
      Err(msg) => panic!(&*msg)
    }
  }

  #[inline]
  pub fn empty() -> RawIobuf<'static> {
    RawIobuf {
      buf:    unsafe { NonZero::new(heap::EMPTY as *mut u8) },
      lo_min_and_owned_bit: 0,
      lo:     0,
      hi:     0,
      hi_max: 0,
      lifetm: PhantomData,
    }
  }

  #[inline]
  unsafe fn nonatomic_inc_ref_count(&self) {
    match self.header() {
      None => {},
      Some(h) => h.inc_ref_count_nonatomic(),
    }
  }

  #[inline]
  unsafe fn atomic_inc_ref_count(&self) {
    match self.header() {
      None => {},
      Some(h) => h.inc_ref_count_atomic(),
    }
  }

  #[inline]
  unsafe fn nonatomic_dec_ref_count(&self) {
    let buf = self.buf;

    match self.header() {
      None => {},
      Some(h) => {
        if h.dec_ref_count_nonatomic().is_err() {
          h.deallocate(buf);
        }
      }
    }
  }

  #[inline]
  unsafe fn atomic_dec_ref_count(&self) {
    let buf = self.buf;

    match self.header() {
      None => {},
      Some(h) => {
        if h.dec_ref_count_atomic().is_err() {
          h.deallocate(buf);
        }
      }
    }
  }

  #[inline]
  pub unsafe fn clone_atomic(&self) -> RawIobuf<'a> {
    self.atomic_inc_ref_count();

    RawIobuf {
      buf:    self.buf,
      lo_min_and_owned_bit: self.lo_min_and_owned_bit,
      lo:     self.lo,
      hi:     self.hi,
      hi_max: self.hi_max,
      lifetm: PhantomData,
    }
  }

  // Keep this out of line to guide inlining.
  unsafe fn clone_from_fix_nonatomic_refcounts(&self, source: &Self) {
    source.nonatomic_inc_ref_count();
    self.nonatomic_dec_ref_count();
  }

  // Keep this out of line to guide inlining.
  unsafe fn clone_from_fix_atomic_refcounts(&self, source: &Self) {
    source.atomic_inc_ref_count();
    self.atomic_dec_ref_count();
  }

  #[inline]
  pub unsafe fn clone_from_atomic(&mut self, source: &Self) {
    if self.ptr()      != source.ptr()
    || self.is_owned() != source.is_owned() {
      self.clone_from_fix_atomic_refcounts(source);
    }

    self.buf    = source.buf;
    self.lo_min_and_owned_bit = source.lo_min_and_owned_bit;
    self.lo     = source.lo;
    self.hi     = source.hi;
    self.hi_max = source.hi_max;

    self.debug_check_invariants(())
  }

  #[inline]
  pub unsafe fn clone_nonatomic(&self) -> Self {
    self.nonatomic_inc_ref_count();

    RawIobuf {
      buf:    self.buf,
      lo_min_and_owned_bit: self.lo_min_and_owned_bit,
      lo:     self.lo,
      hi:     self.hi,
      hi_max: self.hi_max,
      lifetm: PhantomData,
    }
  }

  #[inline]
  pub unsafe fn clone_from_nonatomic(&mut self, source: &Self) {
    if self.ptr()      != source.ptr()
    || self.is_owned() != source.is_owned() {
      self.clone_from_fix_nonatomic_refcounts(source);
    }

    self.buf    = source.buf;
    self.lo_min_and_owned_bit = source.lo_min_and_owned_bit;
    self.lo     = source.lo;
    self.hi     = source.hi;
    self.hi_max = source.hi_max;

    self.debug_check_invariants(())
  }

  #[inline]
  pub unsafe fn drop_atomic(&mut self) {
    // TODO: Remove this when zeroing drop is removed.
    if self.lo_min_and_owned_bit == mem::POST_DROP_U32 { return; }

    self.atomic_dec_ref_count();

    // Reset the owned bit, to prevent double-frees when drop reform lands.
    self.lo_min_and_owned_bit = 0;
  }

  #[inline]
  pub unsafe fn drop_nonatomic(&mut self) {
    // TODO: Remove this when zeroing drop is removed.
    if self.lo_min_and_owned_bit == mem::POST_DROP_U32 { return; }

    self.nonatomic_dec_ref_count();

    // Reset the owned bit, to prevent double-frees when drop reform lands.
    self.lo_min_and_owned_bit = 0;
  }

  #[inline]
  pub fn lo_min(&self) -> u32 {
    self.lo_min_and_owned_bit & !OWNED_MASK
  }

  #[inline]
  pub fn set_lo_min(&mut self, new_value: u32) {
    if cfg!(debug) {
      if new_value > MAX_BUFFER_LEN as u32 {
        panic!("new lo_min out of range (max = {:X}): {:X}", MAX_BUFFER_LEN, new_value);
      }
    }
    self.lo_min_and_owned_bit &= OWNED_MASK;
    self.lo_min_and_owned_bit |= new_value;

    self.debug_check_invariants(())
  }

  #[inline]
  pub fn is_owned(&self) -> bool {
    self.lo_min_and_owned_bit & OWNED_MASK != 0
  }

  #[inline]
  pub fn header(&self) -> Option<&mut AllocationHeader> {
    unsafe {
      if self.is_owned() {
        Some(mem::transmute(
          self.buf.offset(-(ALLOCATION_HEADER_SIZE as isize))))
      } else {
        None
      }
    }
  }

  #[inline]
  pub fn from_str(s: &'a str) -> Self {
    RawIobuf::from_slice(s.as_bytes())
  }

  #[inline]
  pub fn from_str_copy(s: &str) -> RawIobuf<'static> {
    RawIobuf::from_slice_copy(s.as_bytes())
  }

  #[inline]
  pub fn from_str_copy_with_allocator(s: &str, allocator: Arc<Box<Allocator>>) -> RawIobuf<'static> {
    RawIobuf::from_slice_copy_with_allocator(s.as_bytes(), allocator)
  }

  #[inline]
  pub fn from_slice(s: &'a [u8]) -> Self {
    unsafe {
      let len = s.repr().len;

      if len > MAX_BUFFER_LEN {
        buffer_too_big(len);
      }

      RawIobuf {
        buf:    NonZero::new(s.repr().data as *mut u8),
        lo_min_and_owned_bit: 0,
        lo:     0,
        hi:     len as u32,
        hi_max: len as u32,
        lifetm: PhantomData,
      }
    }
  }

  #[inline]
  pub fn from_slice_copy(s: &[u8]) -> RawIobuf<'static> {
    unsafe {
      let b = RawIobuf::new(s.len());
      let s = s.repr();
      memcpy(*b.buf, s.data, s.len);
      b
    }
  }

  #[inline]
  pub fn from_slice_copy_with_allocator(s: &[u8], allocator: Arc<Box<Allocator>>) -> RawIobuf<'static> {
    unsafe {
      let s = s.repr();
      let b = RawIobuf::new_with_allocator(s.len, allocator);
      memcpy(*b.buf, s.data, s.len);
      b
    }
  }

  #[inline]
  pub fn deep_clone(&self) -> RawIobuf<'static> {
    unsafe {
      let mut b = RawIobuf::from_slice_copy(self.as_limit_slice());

      let lo_min = self.lo_min();

      b.lo = self.lo - lo_min;
      b.hi = self.hi - lo_min;

      b
    }
  }

  #[inline]
  pub fn deep_clone_with_allocator(&self, allocator: Arc<Box<Allocator>>) -> RawIobuf<'static> {
    unsafe {
      let mut b = RawIobuf::from_slice_copy_with_allocator(self.as_limit_slice(), allocator);

      let lo_min = self.lo_min();

      b.lo = self.lo - lo_min;
      b.hi = self.hi - lo_min;

      b
    }
  }

  #[inline]
  pub unsafe fn is_unique_nonatomic(&self) -> bool {
    match self.header() {
      Some(ref header) if header.nonatomic_refcount() == 1 => true,
      _ => false,
    }
  }

  #[inline]
  pub unsafe fn is_unique_atomic(&self) -> bool {
    match self.header() {
      Some(ref header) if header.atomic_refcount().load(Ordering::SeqCst) == 1 => true,
      _ => false,
    }
  }

  #[inline]
  pub unsafe fn as_raw_limit_slice(&self) -> raw::Slice<u8> {
    raw::Slice {
      data: self.buf.offset(self.lo_min() as isize),
      len:  self.cap() as usize,
    }
  }

  #[inline]
  pub unsafe fn as_limit_slice<'b>(&'b self) -> &'b [u8] {
    mem::transmute(self.as_raw_limit_slice())
  }

  #[inline]
  pub unsafe fn as_mut_limit_slice<'b>(&'b self) -> &'b mut [u8] {
    mem::transmute(self.as_raw_limit_slice())
  }

  #[inline]
  pub unsafe fn as_raw_window_slice(&self) -> raw::Slice<u8> {
    raw::Slice {
      data: self.buf.offset(self.lo as isize),
      len:  self.len() as usize,
    }
  }

  #[inline]
  pub unsafe fn as_window_slice<'b>(&'b self) -> &'b [u8] {
    mem::transmute(self.as_raw_window_slice())
  }

  #[inline]
  pub unsafe fn as_mut_window_slice<'b>(&'b self) -> &'b mut [u8] {
    mem::transmute(self.as_raw_window_slice())
  }

  #[inline]
  pub fn check_range(&self, pos: u64, len: u64) -> Result<(), ()> {
    err_if(pos + len > self.len() as u64)
  }

  #[inline]
  pub fn check_range_u32(&self, pos: u32, len: u32) -> Result<(), ()> {
    self.check_range(pos as u64, len as u64)
  }

  #[inline]
  pub fn check_range_usize(&self, pos: u32, len: usize) -> Result<(), ()> {
    self.check_range(pos as u64, len as u64)
  }

  #[inline]
  pub fn check_range_u32_fail(&self, pos: u32, len: u32) {
    if self.check_range_u32(pos, len).is_err() {
      bad_range(pos as u64, len as u64);
    }
  }

  #[inline]
  pub fn check_range_usize_fail(&self, pos: u32, len: usize) {
    if self.check_range_usize(pos, len).is_err() {
      bad_range(pos as u64, len as u64);
    }
  }

  #[inline]
  pub fn debug_check_range_u32(&self, pos: u32, len: u32) {
    if cfg!(debug) {
      self.check_range_u32_fail(pos, len);
    }
  }

  #[inline]
  pub fn debug_check_range_usize(&self, pos: u32, len: usize) {
    if cfg!(debug) {
      self.check_range_usize_fail(pos, len);
    }
  }

  #[inline]
  pub fn sub_window(&mut self, pos: u32, len: u32) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_u32(pos, len));
      Ok(self.unsafe_sub_window(pos, len))
    }
  }

  #[inline]
  pub fn sub_window_from(&mut self, pos: u32) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_u32(pos, 0));
      Ok(self.unsafe_sub_window_from(pos))
    }
  }

  #[inline]
  pub fn sub_window_to(&mut self, len: u32) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_u32(0, len));
      Ok(self.unsafe_sub_window_to(len))
    }
  }

  #[inline]
  pub unsafe fn unsafe_sub_window(&mut self, pos: u32, len: u32) {
    self.debug_check_range_u32(pos, len);
    self.unsafe_sub_window_from(pos);
    self.unsafe_sub_window_to(len);
  }

  #[inline]
  pub unsafe fn unsafe_sub_window_from(&mut self, pos: u32) {
    self.debug_check_range_u32(pos, 0);
    self.unsafe_resize(pos);
    self.flip_hi();
  }

  #[inline]
  pub unsafe fn unsafe_sub_window_to(&mut self, len: u32) {
    self.debug_check_range_u32(0, len);
    self.unsafe_resize(len)
  }

  #[inline]
  pub fn sub(&mut self, pos: u32, len: u32) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_u32(pos, len));
      Ok(self.unsafe_sub(pos, len))
    }
  }

  #[inline]
  pub fn sub_from(&mut self, pos: u32) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_u32(pos, 0));
      Ok(self.unsafe_sub_from(pos))
    }
  }

  #[inline]
  pub fn sub_to(&mut self, len: u32) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_u32(0, len));
      Ok(self.unsafe_sub_to(len))
    }
  }

  #[inline]
  pub unsafe fn unsafe_sub(&mut self, pos: u32, len: u32) {
    self.debug_check_range_u32(pos, len);
    self.unsafe_sub_window(pos, len);
    self.narrow();
  }

  #[inline]
  pub unsafe fn unsafe_sub_from(&mut self, pos: u32) {
    self.debug_check_range_u32(pos, 0);
    self.unsafe_sub_window_from(pos);
    self.narrow();
  }

  #[inline]
  pub unsafe fn unsafe_sub_to(&mut self, len: u32) {
    self.debug_check_range_u32(0, len);
    self.unsafe_sub_window_to(len);
    self.narrow();
  }

  /// Both the limits and the window are [lo, hi).
  #[inline]
  pub fn set_limits_and_window(&mut self, limits: (u32, u32), window: (u32, u32)) -> Result<(), ()> {
    let (new_lo_min, new_hi_max) = limits;
    let (new_lo, new_hi) = window;
    let lo_min = self.lo_min();

    try!(err_if(new_hi_max < new_lo_min));
    try!(err_if(new_hi     < new_lo));
    try!(err_if(new_lo_min < lo_min));
    try!(err_if(new_hi_max > self.hi_max));
    try!(err_if(new_lo     < self.lo));
    try!(err_if(new_hi     > self.hi));

    self.set_lo_min(new_lo_min);
    self.lo     = new_lo;
    self.hi     = new_hi;
    self.hi_max = new_hi_max;

    Ok(self.debug_check_invariants(()))
  }

  /// Both the limits and the window are [lo, hi).
  #[inline]
  pub fn expand_limits_and_window(&mut self, limits: (u32, u32), window: (u32, u32)) -> Result<(), ()> {
    let (new_lo_min, new_hi_max) = limits;
    let (new_lo, new_hi) = window;
    let lo_min = self.lo_min();

    try!(err_if(new_hi_max < new_lo_min));
    try!(err_if(new_hi     < new_lo));
    try!(err_if(new_lo_min < lo_min));
    try!(err_if(new_hi_max > self.hi_max));

    self.set_lo_min(new_lo_min);
    self.lo     = new_lo;
    self.hi     = new_hi;
    self.hi_max = new_hi_max;

    Ok(self.debug_check_invariants(()))
  }

  #[inline]
  pub fn len(&self) -> u32 {
    self.hi - self.lo
  }

  #[inline]
  pub fn cap(&self) -> u32 {
    self.hi_max - self.lo_min()
  }

  #[inline]
  pub fn is_empty(&self) -> bool {
    self.hi == self.lo
  }

  #[inline]
  pub fn narrow(&mut self) {
    let lo = self.lo;
    self.set_lo_min(lo);
    self.hi_max = self.hi;
    self.debug_check_invariants(())
  }

  #[inline]
  pub fn advance(&mut self, len: u32) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_u32(0, len));
      self.unsafe_advance(len);
      Ok(())
    }
  }

  #[inline]
  pub unsafe fn unsafe_advance(&mut self, len: u32) {
    self.debug_check_range_u32(0, len);
    self.lo += len;
    self.debug_check_invariants(())
  }

  #[inline]
  pub fn extend(&mut self, len: u32) -> Result<(), ()> {
    unsafe {
      let hi     = self.hi     as u64;
      let hi_max = self.hi_max as u64;
      let new_hi = hi + len    as u64;

      try!(err_if(new_hi > hi_max));

      Ok(self.unsafe_extend(len))
    }
  }

  #[inline]
  pub unsafe fn unsafe_extend(&mut self, len: u32) {
    if cfg!(debug) {
      let hi     = self.hi     as u64;
      let hi_max = self.hi_max as u64;
      let new_hi = hi + len    as u64;

      if new_hi > hi_max {
        bad_range(new_hi, 0);
      }
    }
    self.hi += len;

    self.debug_check_invariants(())
  }

  #[inline]
  pub fn is_extended_by<'b>(&self, other: &RawIobuf<'b>) -> bool {
    unsafe {
      self.buf.offset(self.hi as isize) == other.buf.offset(other.lo as isize)
         // check_range, but with `cap()` instead of `len()`.
      && self.hi as u64 + other.len() as u64 <= self.hi_max as u64
    }
  }

  #[inline]
  pub fn extend_with<'b>(&mut self, other: &RawIobuf<'b>) -> Result<(), ()> {
    unsafe {
      try!(err_if(!self.is_extended_by(other)));
      Ok(self.unsafe_extend(other.len()))
    }
  }

  #[inline]
  pub fn resize(&mut self, len: u32) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_u32(0, len));
      Ok(self.unsafe_resize(len))
    }
  }

  #[inline]
  pub unsafe fn unsafe_resize(&mut self, len: u32) {
    self.debug_check_range_u32(0, len);
    self.hi = self.lo + len;
    self.debug_check_invariants(())
  }

  #[inline]
  pub fn split_at_nonatomic(&self, pos: u32) -> Result<(Self, Self), ()> {
    unsafe {
      try!(self.check_range_u32(pos, 0));
      Ok(self.unsafe_split_at_nonatomic(pos))
    }
  }

  #[inline]
  pub unsafe fn unsafe_split_at_nonatomic(&self, pos: u32) -> (Self, Self) {
    self.debug_check_range_u32(pos, 0);
    let mut ret = ((*self).clone_nonatomic(), (*self).clone_nonatomic());
    ret.0.unsafe_resize(pos);
    ret.1.unsafe_advance(pos);
    ret
  }

  #[inline]
  pub fn split_start_at_nonatomic(&mut self, pos: u32) -> Result<Self, ()> {
    unsafe {
      try!(self.check_range_u32(pos, 0));
      Ok(self.unsafe_split_start_at_nonatomic(pos))
    }
  }

  #[inline]
  pub unsafe fn unsafe_split_start_at_nonatomic(&mut self, pos: u32) -> Self {
    self.debug_check_range_u32(pos, 0);
    let mut ret = (*self).clone_nonatomic();
    ret.unsafe_resize(pos);
    self.unsafe_advance(pos);
    ret
  }

  #[inline]
  pub fn split_at_atomic(&self, pos: u32) -> Result<(Self, Self), ()> {
    unsafe {
      try!(self.check_range_u32(pos, 0));
      Ok(self.unsafe_split_at_atomic(pos))
    }
  }

  #[inline]
  pub unsafe fn unsafe_split_at_atomic(&self, pos: u32) -> (Self, Self) {
    self.debug_check_range_u32(pos, 0);
    let mut ret = ((*self).clone_atomic(), (*self).clone_atomic());
    ret.0.unsafe_resize(pos);
    ret.1.unsafe_advance(pos);
    ret
  }

  #[inline]
  pub fn split_start_at_atomic(&mut self, pos: u32) -> Result<Self, ()> {
    unsafe {
      try!(self.check_range_u32(pos, 0));
      Ok(self.unsafe_split_start_at_atomic(pos))
    }
  }

  #[inline]
  pub unsafe fn unsafe_split_start_at_atomic(&mut self, pos: u32) -> Self {
    self.debug_check_range_u32(pos, 0);
    let mut ret = (*self).clone_atomic();
    ret.unsafe_resize(pos);
    self.unsafe_advance(pos);
    ret
  }

  #[inline]
  pub fn rewind(&mut self) {
    self.lo = self.lo_min();
    self.debug_check_invariants(())
  }

  #[inline]
  pub fn reset(&mut self) {
    self.lo = self.lo_min();
    self.hi = self.hi_max;
    self.debug_check_invariants(())
  }

  #[inline]
  pub fn flip_lo(&mut self) {
    self.hi = self.lo;
    self.lo = self.lo_min();
    self.debug_check_invariants(())
  }

  #[inline]
  pub fn flip_hi(&mut self) {
    self.lo = self.hi;
    self.hi = self.hi_max;
    self.debug_check_invariants(())
  }

  #[inline]
  pub fn lo_space(&self) -> u32 {
    self.lo - self.lo_min()
  }

  #[inline]
  pub fn hi_space(&self) -> u32 {
    self.hi_max - self.hi
  }

  #[inline]
  pub fn compact(&mut self) {
    unsafe {
      let len    = self.len();
      let lo_min = self.lo_min();

      memmove(
        self.buf.offset(lo_min as isize),
        self.buf.offset(self.lo as isize),
        len as usize);

      self.lo = lo_min + len;
      self.hi = self.hi_max;

      self.debug_check_invariants(())
    }
  }

  #[inline]
  pub fn peek(&self, pos: u32, dst: &mut [u8]) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_usize(pos, dst.len()));
      Ok(self.unsafe_peek(pos, dst))
    }
  }

  #[inline]
  pub fn peek_be<T: IntLike>(&self, pos: u32) -> Result<T, ()> {
    unsafe {
      try!(self.check_range_u32(pos, mem::size_of::<T>() as u32));
      Ok(self.unsafe_peek_be::<T>(pos))
    }
  }

  #[inline]
  pub fn peek_le<T: IntLike>(&self, pos: u32) -> Result<T, ()> {
    unsafe {
      try!(self.check_range_u32(pos, mem::size_of::<T>() as u32));
      Ok(self.unsafe_peek_le::<T>(pos))
    }
  }

  #[inline]
  pub fn poke(&self, pos: u32, src: &[u8]) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_usize(pos, src.len()));
      Ok(self.unsafe_poke(pos, src))
    }
  }

  #[inline]
  pub fn poke_be<T: IntLike>(&self, pos: u32, t: T) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_u32(pos, mem::size_of::<T>() as u32));
      Ok(self.unsafe_poke_be(pos, t))
    }
  }

  #[inline]
  pub fn poke_le<T: IntLike>(&self, pos: u32, t: T) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_u32(pos, mem::size_of::<T>() as u32));
      Ok(self.unsafe_poke_le(pos, t))
    }
  }

  #[inline]
  pub fn fill(&mut self, src: &[u8]) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_usize(0, src.len()));
      Ok(self.unsafe_fill(src))
    }
  }

  #[inline]
  pub fn fill_be<T: IntLike>(&mut self, t: T) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_u32(0, mem::size_of::<T>() as u32));
      Ok(self.unsafe_fill_be(t))
    }
  }

  #[inline]
  pub fn fill_le<T: IntLike>(&mut self, t: T) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_u32(0, mem::size_of::<T>() as u32));
      Ok(self.unsafe_fill_le(t)) // Ok, unsafe fillet? om nom.
    }
  }

  #[inline]
  pub fn consume(&mut self, dst: &mut [u8]) -> Result<(), ()> {
    unsafe {
      try!(self.check_range_usize(0, dst.len()));
      Ok(self.unsafe_consume(dst))
    }
  }

  #[inline]
  pub fn consume_le<T: IntLike>(&mut self) -> Result<T, ()> {
    unsafe {
      try!(self.check_range_u32(0, mem::size_of::<T>() as u32));
      Ok(self.unsafe_consume_le())
    }
  }

  #[inline]
  pub fn consume_be<T: IntLike>(&mut self) -> Result<T, ()> {
    unsafe {
      try!(self.check_range_u32(0, mem::size_of::<T>() as u32));
      Ok(self.unsafe_consume_be())
    }
  }

  #[inline]
  pub unsafe fn unsafe_peek(&self, pos: u32, dst: &mut [u8]) {
    let len = dst.len();
    self.debug_check_range_usize(pos, len);

    let dst: raw::Slice<u8> = mem::transmute(dst);

    memcpy(
      dst.data as *mut u8,
      self.buf.offset((self.lo + pos) as isize),
      len);
  }

  #[inline]
  unsafe fn unsafe_peek_raw<T: IntLike>(&self, pos: u32) -> T {
    let len = mem::size_of::<T>();
    self.debug_check_range_usize(pos, len);

    let mut dst: T = mem::uninitialized();

    memcpy(
      &mut dst as *mut T as *mut u8,
      self.buf.offset((self.lo + pos) as isize),
      len);

    dst
  }

  #[inline]
  pub unsafe fn unsafe_peek_be<T: IntLike>(&self, pos: u32) -> T {
    from_be(self.unsafe_peek_raw(pos))
  }

  #[inline]
  pub unsafe fn unsafe_peek_le<T: IntLike>(&self, pos: u32) -> T {
    from_le(self.unsafe_peek_raw(pos))
  }

  #[inline]
  pub unsafe fn unsafe_poke(&self, pos: u32, src: &[u8]) {
    let len = src.len();
    self.debug_check_range_usize(pos, len);

    let src: raw::Slice<u8> = mem::transmute(src);

    memcpy(
      self.buf.offset((self.lo + pos) as isize),
      src.data,
      len);
  }

  #[inline]
  unsafe fn unsafe_poke_raw<T: IntLike>(&self, pos: u32, t: T) {
    let len = mem::size_of::<T>();
    self.debug_check_range_usize(pos, len);

    memcpy(
      self.buf.offset((self.lo + pos) as isize),
      &t as *const T as *const u8,
      len);
  }

  #[inline]
  pub unsafe fn unsafe_poke_be<T: IntLike>(&self, pos: u32, t: T) {
    self.unsafe_poke_raw(pos, to_be(t))
  }

  #[inline]
  pub unsafe fn unsafe_poke_le<T: IntLike>(&self, pos: u32, t: T) {
    self.unsafe_poke_raw(pos, to_le(t))
  }

  #[inline]
  pub unsafe fn unsafe_fill(&mut self, src: &[u8]) {
    self.debug_check_range_usize(0, src.len());
    self.unsafe_poke(0, src);
    self.lo += src.len() as u32;
    self.debug_check_invariants(())
  }

  #[inline]
  pub unsafe fn unsafe_fill_be<T: IntLike>(&mut self, t: T) {
    let bytes = mem::size_of::<T>() as u32;
    self.debug_check_range_u32(0, bytes);
    self.unsafe_poke_be(0, t);
    self.lo += bytes;
    self.debug_check_invariants(())
  }

  #[inline]
  pub unsafe fn unsafe_fill_le<T: IntLike>(&mut self, t: T) {
    let bytes = mem::size_of::<T>() as u32;
    self.debug_check_range_u32(0, bytes);
    self.unsafe_poke_le(0, t);
    self.lo += bytes;
    self.debug_check_invariants(())
  }

  #[inline]
  pub unsafe fn unsafe_consume(&mut self, dst: &mut [u8]) {
    self.debug_check_range_usize(0, dst.len());
    self.unsafe_peek(0, dst);
    self.lo += dst.len() as u32;
    self.debug_check_invariants(())
  }

  #[inline]
  pub unsafe fn unsafe_consume_le<T: IntLike>(&mut self) -> T {
    let bytes = mem::size_of::<T>() as u32;
    self.debug_check_range_u32(0, bytes);
    let ret = self.unsafe_peek_le::<T>(0);
    self.lo += bytes;
    self.debug_check_invariants(ret)
  }

  #[inline]
  pub unsafe fn unsafe_consume_be<T: IntLike>(&mut self) -> T {
    let bytes = mem::size_of::<T>() as u32;
    self.debug_check_range_u32(0, bytes);
    let ret = self.unsafe_peek_be::<T>(0);
    self.lo += bytes;
    self.debug_check_invariants(ret)
  }

  #[inline(always)]
  pub fn ptr(&self) -> NonZero<*mut u8> {
    self.buf
  }

  #[inline(always)]
  pub fn lo(&self) -> u32 {
    self.lo
  }

  #[inline(always)]
  pub fn hi(&self) -> u32 {
    self.hi
  }

  #[inline(always)]
  pub fn hi_max(&self) -> u32 {
    self.hi_max
  }

  #[cold]
  fn show_hex(&self, f: &mut Formatter, half_line: &[u8])
      -> fmt::Result {
    for &x in half_line.iter() {
      try!(write!(f, "{:02x} ", x));
    }
    Ok(())
  }

  #[cold]
  fn show_ascii(&self, f: &mut Formatter, half_line: &[u8])
      -> fmt::Result {
    for &x in half_line.iter() {
      let c = if x >= 32 && x < 126 { x as char } else { '.' };
      try!(write!(f, "{}", c));
    }
    Ok(())
  }

  #[cold]
  fn show_line(&self, f: &mut Formatter, line_number: usize, chunk: &[u8])
      -> fmt::Result {

    if      self.len() <= 1 <<  8 { try!(write!(f, "0x{:02x}",  line_number * 8)) }
    else if self.len() <= 1 << 16 { try!(write!(f, "0x{:04x}",  line_number * 8)) }
    else if self.len() <= 1 << 24 { try!(write!(f, "0x{:06x}",  line_number * 8)) }
    else                          { try!(write!(f, "0x{:08x}",  line_number * 8)) }

    try!(write!(f, ":  "));

    let chunk_len = chunk.len();

    let (left_slice, right_slice) =
      if chunk_len >= 4 {
        (&chunk[0..4], Some(&chunk[4..]))
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

  #[cold]
  pub fn show(&self, f: &mut Formatter, ty: &str) -> fmt::Result {
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

#[test]
fn peek_be() {
  use iobuf::Iobuf;
  use impls::ROIobuf;

  let s = [1,2,3,4];
  let b = ROIobuf::from_slice(&s);
  assert_eq!(b.peek_be(0), Ok(0x01020304u32));
}

#[test]
fn peek_le() {
  use iobuf::Iobuf;
  use impls::ROIobuf;

  let s = [1,2,3,4];
  let b = ROIobuf::from_slice(&s);
  assert_eq!(b.peek_le(0), Ok(0x04030201u32));
}

#[test]
fn poke_be() {
  use iobuf::Iobuf;
  use impls::RWIobuf;

  let b = RWIobuf::new(4);
  assert_eq!(b.poke_be(0, 0x01020304u32), Ok(()));
  let expected = [ 1,2,3,4 ];
  unsafe { assert_eq!(b.as_window_slice(), &expected[..]); }
}

#[test]
fn poke_le() {
  use iobuf::Iobuf;
  use impls::RWIobuf;

  let b = RWIobuf::new(4);
  assert_eq!(b.poke_le(0, 0x01020304u32), Ok(()));
  let expected = [ 4,3,2,1 ];
  unsafe { assert_eq!(b.as_window_slice(), &expected[..]); }
}


#[test]
fn peek_be_u8() {
  use iobuf::Iobuf;
  use impls::ROIobuf;

  let b = ROIobuf::from_str("abc");
  assert_eq!(b.peek_be(0), Ok(b'a'));
}

#[test]
fn over_resize() {
  use iobuf::Iobuf;
  use impls::RWIobuf;
  let mut b = RWIobuf::new(1024);
  assert_eq!(b.advance(512), Ok(()));
  assert_eq!(b.resize(0x7FFF_FFFF), Err(()));
}

#[test]
#[should_panic]
fn create_huge_iobuf() {
  use impls::RWIobuf;
  RWIobuf::new(0x8000_0000);
}

#[test]
fn check_large_range_pos() {
  use impls::RWIobuf;
  use iobuf::Iobuf;
  let b = RWIobuf::new(100);
  unsafe { assert_eq!(b.as_raw().check_range(0x8000_0000, 0), Err(())); }
}

#[test]
fn check_large_range_len() {
  use impls::RWIobuf;
  use iobuf::Iobuf;
  let b = RWIobuf::new(100);
  unsafe { assert_eq!(b.as_raw().check_range(0, 0x8000_0000), Err(())); }
}

#[test]
fn test_allocator() {
  use impls::RWIobuf;
  use self::Allocator;

  struct MyAllocator;

  impl Allocator for MyAllocator {
    fn allocate(&self, size: usize, align: usize) -> *mut u8 {
      unsafe { ::alloc::heap::allocate(size, align) }
    }

    fn deallocate(&self, ptr: NonZero<*mut u8>, len: usize, align: usize) {
      unsafe { ::alloc::heap::deallocate(*ptr, len, align) }
    }
  }

  RWIobuf::new_with_allocator(1000, Arc::new(Box::new(MyAllocator)));
}
