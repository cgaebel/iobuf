use std::fmt;
use std::iter;
use std::mem;
use std::num;
use std::num::Zero;
use std::ptr;
use std::raw;
use std::rc::Rc;
use std::result;

enum MaybeOwnedBuffer<'a> {
  OwnedBuffer(Vec<u8>),
  BorrowedBuffer(&'a mut [u8]),
}

impl<'a> MaybeOwnedBuffer<'a> {
  #[inline]
  fn as_slice(&self) -> &[u8] {
    match self {
      &OwnedBuffer(ref v)    => v.as_slice(),
      &BorrowedBuffer(ref s) => s.as_slice(),
    }
  }

  #[inline]
  fn as_mut_slice(&self) -> &mut [u8] {
    unsafe {
      match self {
        &OwnedBuffer(ref v) => {
          let mut_v: &mut Vec<u8> = mem::transmute(v);
          mut_v.as_mut_slice()
        },
        &BorrowedBuffer(ref s) => {
          let mut_s: &mut &mut [u8] = mem::transmute(s);
          mut_s.as_mut_slice()
        },
      }
    }
  }
}

#[cold]
fn bad_range(pos: uint, len: uint) {
  fail!("Iobuf got invalid range: pos={}, len={}", pos, len);
}

#[deriving(Clone)]
struct RawIobuf<'a> {
  buf:    Rc<MaybeOwnedBuffer<'a>>,
  lo_min: uint,
  lo:     uint,
  hi:     uint,
  hi_max: uint,
}

impl<'a> RawIobuf<'a> {
  fn new(len: uint) -> RawIobuf<'static> {
    RawIobuf {
      buf: Rc::new(OwnedBuffer(Vec::from_elem(len, 0u8))),
      lo_min: 0,
      lo:     0,
      hi:     len,
      hi_max: len,
    }
  }

  fn from_str<'a>(s: &'a str) -> RawIobuf<'a> {
    unsafe {
      let bytes: &mut [u8] = mem::transmute(s.as_bytes());
      RawIobuf {
        buf: Rc::new(BorrowedBuffer(bytes)),
        lo_min: 0,
        lo:     0,
        hi:     s.len(),
        hi_max: s.len(),
      }
    }
  }

  fn from_vec(v: Vec<u8>) -> RawIobuf<'static> {
    let len = v.len();
    RawIobuf {
      buf: Rc::new(OwnedBuffer(v)),
      lo_min: 0,
      lo:     0,
      hi:     len,
      hi_max: len,
    }
  }

  fn from_slice<'a>(s: &'a [u8]) -> RawIobuf<'a> {
    unsafe {
      let len = s.len();
      let mut_buf: &mut [u8] = mem::transmute(s);
      RawIobuf {
        buf: Rc::new(BorrowedBuffer(mut_buf)),
        lo_min: 0,
        lo:     0,
        hi:     len,
        hi_max: len,
      }
    }
  }

  #[inline(always)]
  fn check_range(&self, pos: uint, len: uint) {
    if pos + len <= self.len() {
      bad_range(pos, len);
    }
  }

  #[inline]
  fn sub(&mut self, pos: uint, len: uint) {
    self.check_range(pos, len);
    self.hi     = self.lo + len;
    self.hi_max = self.hi;
    self.lo     = self.lo + pos;
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
  fn advance(&mut self, len: uint) {
    unsafe {
      self.check_range(0, len);
      self.unsafe_advance(len);
    }
  }

  #[inline(always)]
  unsafe fn unsafe_advance(&mut self, len: uint) {
    self.lo += len;
  }

  #[inline(always)]
  fn resize(&mut self, len: uint) {
    unsafe {
      self.check_range(0, len);
      self.unsafe_resize(len);
    }
  }

  #[inline(always)]
  unsafe fn unsafe_resize(&mut self, len: uint) {
    self.hi += len;
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
      let s: raw::Slice<u8> = mem::transmute(self.buf.as_mut_slice());
      ptr::copy_memory(
        s.data as *mut u8,
        s.data.offset(self.lo as int) as *const u8,
        self.len());
    }
  }

  #[inline]
  fn as_slice(&self) -> &[u8] {
    self.buf.as_slice().slice(self.lo, self.hi)
  }

  #[inline]
  fn as_mut_slice(&mut self) -> &mut [u8] {
    self.buf.as_mut_slice().mut_slice(self.lo, self.hi)
  }

  // I need [Peek, Poke, Fill, Consume] x ([u8] + [u16, u32, u64] x [le, be]) x [(), unsafe_]
  // That's 56 functions :(

  #[inline(always)]
  fn peek_be<T: Copy + Zero + Shl<uint, T> + BitOr<T, T> + FromPrimitive>(&self) -> T {
    unsafe {
      self.check_range(0, mem::size_of::<T>());
      self.unsafe_peek_be::<T>()
    }
  }

  #[inline(always)]
  fn peek_le<T: Copy + Zero + Shl<uint, T> + Shr<uint, T> + BitOr<T, T> + FromPrimitive>(&self) -> T {
    unsafe {
      self.check_range(0, mem::size_of::<T>());
      self.unsafe_peek_le::<T>()
    }
  }

  #[inline(always)]
  fn poke_be<T: Copy + Shl<uint, T> + BitAnd<T, T> + FromPrimitive + ToPrimitive>(&self, t: T) {
    unsafe {
      self.check_range(0, mem::size_of::<T>());
      self.unsafe_poke_be(t)
    }
  }

  #[inline(always)]
  fn poke_le<T: Copy + Shl<uint, T> + BitAnd<T, T> + FromPrimitive + ToPrimitive>(&self, t: T) {
    unsafe {
      self.check_range(0, mem::size_of::<T>());
      self.unsafe_poke_le(t)
    }
  }

  #[inline(always)]
  fn fill_be<T: Copy + Shl<uint, T> + BitAnd<T, T> + FromPrimitive + ToPrimitive>(&mut self, t: T) {
    unsafe {
      self.check_range(0, mem::size_of::<T>());
      self.unsafe_fill_be(t)
    }
  }

  #[inline(always)]
  fn fill_le<T: Copy + Shl<uint, T> + BitAnd<T, T> + FromPrimitive + ToPrimitive>(&mut self, t: T) {
    unsafe {
      self.check_range(0, mem::size_of::<T>());
      self.unsafe_fill_le(t) // unsafe fillet? om nom.
    }
  }

  #[inline(always)]
  unsafe fn get_at<T: FromPrimitive>(&self, idx: uint) -> T {
    FromPrimitive::from_u8(
      *self.buf.as_slice().unsafe_get(self.lo + idx))
    .unwrap()
  }

  #[inline(always)]
  unsafe fn set_at<T: ToPrimitive>(&self, idx: uint, val: T) {
    self.buf.as_mut_slice().unsafe_set(self.lo + idx, val.to_u8().unwrap())
  }

  unsafe fn unsafe_peek_be<T: Copy + Zero + Shl<uint, T> + BitOr<T, T> + FromPrimitive>(&self) -> T {
    let bytes = mem::size_of::<T>();
    let mut x: T = Zero::zero();

    for i in iter::range(0, bytes) {
      x = self.get_at::<T>(i) | (x << 8);
    }

    x
  }

  unsafe fn unsafe_peek_le<T: Copy + Zero + Shl<uint, T> + Shr<uint, T> + BitOr<T, T> + FromPrimitive>(&self) -> T {
    let bytes = mem::size_of::<T>();
    let mut x: T = Zero::zero();

    for i in iter::range(0, bytes) {
      x = (x >> 8) | (self.get_at::<T>(i) << ((bytes - 1)* 8));
    }

    x
  }

  unsafe fn unsafe_poke_be<T: Copy + Shl<uint, T> + BitAnd<T, T> + FromPrimitive + ToPrimitive>(&self, t: T) {
    let bytes = mem::size_of::<T>();

    let msk: T = FromPrimitive::from_u8(0xFFu8).unwrap();

    for i in iter::range(0, bytes) {
      self.set_at(i, t << ((bytes-i-1)*8) & msk);
    }
  }

  unsafe fn unsafe_poke_le<T: Copy + Shl<uint, T> + BitAnd<T, T> + FromPrimitive + ToPrimitive>(&self, t: T) {
    let bytes = mem::size_of::<T>();

    let msk: T = FromPrimitive::from_u8(0xFFu8).unwrap();

    for i in iter::range(0, bytes) {
      self.set_at(i, t << (i*8) & msk);
    }
  }

  #[inline(always)]
  unsafe fn unsafe_fill_be<T: Copy + Shl<uint, T> + BitAnd<T, T> + FromPrimitive + ToPrimitive>(&mut self, t: T) {
    self.unsafe_poke_be(t);
    self.lo += mem::size_of::<T>();
  }

  #[inline(always)]
  unsafe fn unsafe_fill_le<T: Copy + Shl<uint, T> + BitAnd<T, T> + FromPrimitive + ToPrimitive>(&mut self, t: T) {
    self.unsafe_poke_le(t);
    self.lo += mem::size_of::<T>();
  }

  #[inline(always)]
  unsafe fn unsafe_consume_le<T: Copy + Zero + Shl<uint, T> + Shr<uint, T> + BitOr<T, T> + FromPrimitive>(&mut self) -> T {
    self.lo += mem::size_of::<T>();
    self.unsafe_peek_le::<T>()
  }

  #[inline(always)]
  unsafe fn unsafe_consume_be<T: Copy + Zero + Shl<uint, T> + BitOr<T, T> + FromPrimitive>(&mut self) -> T {
    self.lo += mem::size_of::<T>();
    self.unsafe_peek_be::<T>()
  }

  fn show_hex(&self, f: &mut fmt::Formatter, half_line: &[u8])
      -> result::Result<(), fmt::FormatError> {
    for &x in half_line.iter() {
      try!(write!(f, "{:02x} ", x));
    }
    result::Ok(())
  }

  fn show_ascii(&self, f: &mut fmt::Formatter, half_line: &[u8])
      -> result::Result<(), fmt::FormatError> {
     for &x in half_line.iter() {
       let c = if x >= 32 && x < 126 { x as char } else { '.' };
       try!(write!(f, "{}", c));
     }
     result::Ok(())
  }

  fn show_line(&self, f: &mut fmt::Formatter, line_number: uint, chunk: &[u8])
      -> result::Result<(), fmt::FormatError> {

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

  fn show(&self, f: &mut fmt::Formatter, ty: &str) -> result::Result<(), fmt::FormatError> {
    try!(write!(f, "{} IObuf, raw length={}, limits=[{},{}), bounds=[{},{})\n",
                ty, self.buf.as_slice().len(), self.lo_min, self.hi_max, self.lo, self.hi));

    if self.lo == self.hi { return write!(f, "<empty buffer>"); }

    let b = self.buf.as_slice().slice(self.lo, self.hi);

    for (i, c) in b.chunks(8).enumerate() {
      try!(self.show_line(f, i, c));
    }

    result::Ok(())
  }
}

pub trait Iobuf: Clone + fmt::Show {
  // read-only methods on iobufs go here.
}

#[deriving(Clone)]
pub struct ROIobuf<'a> {
  raw: RawIobuf<'a>,
}

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
}

impl<'a> Iobuf for ROIobuf<'a> {
  // ...
}

impl<'a> Iobuf for RWIobuf<'a> {
  // ...
}

impl<'a> fmt::Show for ROIobuf<'a> {
  fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::FormatError> {
    self.raw.show(f, "read-only")
  }
}

impl<'a> fmt::Show for RWIobuf<'a> {
  fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::FormatError> {
    self.raw.show(f, "read-write")
  }
}
