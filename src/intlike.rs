use std::mem;

/// 8-64 bit integer types. Not floating point.
///
/// This used to be in the standard library, but was pulled out into an external
/// library that adds support for arbitrary-sized integers, rationals, complex
/// numbers, and this trait. It also has dependencies on rustc-serialize and rand,
/// which is totally unnecessary. Therefore, I've pulled the trait's features
/// that I need into here.
pub trait IntLike {}

impl IntLike for i8    {}
impl IntLike for i16   {}
impl IntLike for i32   {}
impl IntLike for i64   {}
impl IntLike for isize {}
impl IntLike for u8    {}
impl IntLike for u16   {}
impl IntLike for u32   {}
impl IntLike for u64   {}
impl IntLike for usize {}

#[inline(always)]
pub unsafe fn from_be<T: IntLike>(x: T) -> T {
  match mem::size_of::<T>() {
    1 => mem::transmute_copy(&u8::from_be(mem::transmute_copy(&x))),
    2 => mem::transmute_copy(&u16::from_be(mem::transmute_copy(&x))),
    4 => mem::transmute_copy(&u32::from_be(mem::transmute_copy(&x))),
    8 => mem::transmute_copy(&u64::from_be(mem::transmute_copy(&x))),
    n => bad_int_like(n),
  }
}

#[inline(always)]
pub unsafe fn from_le<T: IntLike>(x: T) -> T {
  match mem::size_of::<T>() {
    1 => mem::transmute_copy(&u8::from_le(mem::transmute_copy(&x))),
    2 => mem::transmute_copy(&u16::from_le(mem::transmute_copy(&x))),
    4 => mem::transmute_copy(&u32::from_le(mem::transmute_copy(&x))),
    8 => mem::transmute_copy(&u64::from_le(mem::transmute_copy(&x))),
    n => bad_int_like(n),
  }
}

#[inline(always)]
pub unsafe fn to_be<T: IntLike>(x: T) -> T {
  match mem::size_of::<T>() {
    1 => mem::transmute_copy(&u8::to_be(mem::transmute_copy(&x))),
    2 => mem::transmute_copy(&u16::to_be(mem::transmute_copy(&x))),
    4 => mem::transmute_copy(&u32::to_be(mem::transmute_copy(&x))),
    8 => mem::transmute_copy(&u64::to_be(mem::transmute_copy(&x))),
    n => bad_int_like(n),
  }
}

#[inline(always)]
pub unsafe fn to_le<T: IntLike>(x: T) -> T {
  match mem::size_of::<T>() {
    1 => mem::transmute_copy(&u8::to_le(mem::transmute_copy(&x))),
    2 => mem::transmute_copy(&u16::to_le(mem::transmute_copy(&x))),
    4 => mem::transmute_copy(&u32::to_le(mem::transmute_copy(&x))),
    8 => mem::transmute_copy(&u64::to_le(mem::transmute_copy(&x))),
    n => bad_int_like(n),
  }
}

#[cold]
fn bad_int_like(n: usize) -> ! {
  panic!("Tried to serialize an integer of size {} into an Iobuf.
          Iobufs only support power of two sizes of 8-64 bytes, inclusive.", n)
}
