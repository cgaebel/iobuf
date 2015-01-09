use std::cmp::Ordering;
use std::fmt;
use std::intrinsics::move_val_init;
use std::iter::{self, order, FromIterator, AdditiveIterator};
use std::mem;
use std::num::ToPrimitive;
use std::option;
use std::slice;
use std::vec;

use iobuf::Iobuf;

use BufSpan::{Empty, One, Many};
use SpanIter::{Opt, Lot};
use SpanMoveIter::{MoveOpt, MoveLot};

/// A span over potentially many Iobufs. This is useful as a "string" type where
/// the contents of the string can come from multiple IObufs, and you want to
/// avoid copying the buffer contents unnecessarily.
///
/// As an optimization, pushing an Iobuf that poisizes to data immediately after
/// the range represented by the last Iobuf pushed will result in just expanding
/// the held Iobuf's range. This prevents allocating lots of unnecessary
/// isizeermediate buffers, while still maisizeaining the illusion of "pushing lots
/// of buffers" while incrementally parsing.
///
/// A `BufSpan` is isizeernally represented as either an `Iobuf` or a `Vec<Iobuf>`,
/// depending on how many different buffers were used.
pub enum BufSpan<Buf> {
  /// A span over 0 bytes.
  Empty,
  /// A single span over one range.
  One (Buf),
  /// A span over several backing Iobufs.
  Many(Vec<Buf>),
}

impl<Buf: Iobuf> Clone for BufSpan<Buf> {
  #[inline]
  fn clone(&self) -> BufSpan<Buf> {
    match *self {
      Empty       => Empty,
      One(ref b)  => One ((*b).clone()),
      Many(ref v) => Many((*v).clone()),
    }
  }
}

impl<Buf: Iobuf> fmt::Show for BufSpan<Buf> {
  fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
    let mut first_time = true;

    for b in self.iter() {
      if !first_time {
        try!(write!(f, "\n"));
      } else {
        first_time = false;
      }

      try!(b.fmt(f));
    }

    Ok(())
  }
}

impl<Buf: Iobuf> FromIterator<Buf> for BufSpan<Buf> {
  #[inline]
  fn from_iter<I: Iterator<Item=Buf>>(iterator: I) -> BufSpan<Buf> {
    let mut ret = BufSpan::new();
    ret.extend(iterator);
    ret
  }
}

impl<Buf: Iobuf> Extend<Buf> for BufSpan<Buf> {
  #[inline]
  fn extend<I: Iterator<Item=Buf>>(&mut self, mut iterator: I) {
    for x in iterator {
      self.push(x);
    }
  }
}

impl<Buf: Iobuf> BufSpan<Buf> {
  /// Creates a new, empty `Bufspan`.
  ///
  /// ```rust
  /// use iobuf::{BufSpan, ROIobuf};
  ///
  /// let s: BufSpan<ROIobuf<'static>> = BufSpan::new();
  /// assert!(s.is_empty());
  /// ```
  #[inline]
  pub fn new() -> BufSpan<Buf> {
    BufSpan::Empty
  }

  /// Creates a new `BufSpan` from an Iobuf.
  ///
  /// ```rust
  /// use iobuf::{BufSpan, ROIobuf};
  ///
  /// let s = BufSpan::from_buf(ROIobuf::from_slice(b"hello"));
  /// assert_eq!(s.iter().count(), 1);
  /// assert_eq!(s.count_bytes(), 5);
  /// ```
  #[inline]
  pub fn from_buf(b: Buf) -> BufSpan<Buf> {
    if b.is_empty() { Empty } else { One(b) }
  }

  /// Returns `true` iff the span is over an empty range.
  ///
  /// ```rust
  /// use iobuf::{BufSpan, Iobuf, ROIobuf};
  ///
  /// let mut s = BufSpan::new();
  ///
  /// assert!(s.is_empty());
  ///
  /// s.push(ROIobuf::from_str(""));
  /// assert!(s.is_empty());
  ///
  /// s.push(ROIobuf::from_str("hello, world!"));
  /// assert!(!s.is_empty());
  /// ```
  #[inline]
  pub fn is_empty(&self) -> bool {
    match *self {
      Empty => true,
      _     => false,
    }
  }

  /// The fast path during pushing -- either fills in the first buffer, or
  /// extends an existing one.
  ///
  /// Returns `None` if the fast path was taken and nothing more needs to be
  /// done. Returns `Some` if we need to do a slow push.
  #[inline]
  fn try_to_extend(&mut self, b: Buf) -> Option<Buf> {
    if b.len() == 0 { return None; }
    if self.is_empty() {
      // EFFICIENCY HACK: We know we're empty, so we can drop without running
      // drop glue. rustc isn't smart enough to figure this out. This will
      // stop all drop calls in this function, leaving any dropping that might
      // have to happen to `slow_push`.
      unsafe {
        move_val_init(self, One(b));
        return None;
      }
    }

    match *self {
      Empty => unreachable!(),
      One(ref mut b0) => {
        match b0.extend_with(&b) {
          Ok (()) => return None,
          Err(()) => return Some(b),
        }
      }
      Many(_) => return Some(b),
    }
  }

  /// Appends a buffer to a `BufSpan`. If the buffer is an extension of the
  /// previously pushed buffer, the range will be extended. Otherwise, the new
  /// non-extension buffer will be added to the end of a vector.
  ///
  /// ```rust
  /// use iobuf::{BufSpan, Iobuf, ROIobuf};
  ///
  /// let mut s = BufSpan::new();
  ///
  /// s.push(ROIobuf::from_str("he"));
  /// s.push(ROIobuf::from_str("llo"));
  ///
  /// assert_eq!(s.count_bytes() as usize, "hello".len());
  /// assert_eq!(s.iter().count(), 2);
  ///
  /// let mut b0 = ROIobuf::from_str(" world");
  /// let mut b1 = b0.clone();
  ///
  /// b0.resize(2).unwrap();
  /// b1.advance(2).unwrap();
  ///
  /// s.push(b0);
  /// s.push(b1);
  ///
  /// // b0 and b1 are immediately after each other, and from the same buffer,
  /// // so get merged isizeo one Iobuf.
  /// assert_eq!(s.count_bytes() as usize, "hello world".len());
  /// assert_eq!(s.iter().count(), 3);
  /// ```
  #[inline(always)]
  pub fn push(&mut self, b: Buf) {
    match self.try_to_extend(b) {
      None    => {},
      Some(b) => self.slow_push(b),
    }
  }

  /// The slow path during a push. This is only taken if a `BufSpan` must span
  /// multiple backing buffers.
  #[cold]
  fn slow_push(&mut self, b: Buf) {
    match *self {
      Empty  => unreachable!(),
      One(_) => {},
      Many(ref mut v) => unsafe {
        let last_pos = v.len() - 1;
        match v.get_unchecked_mut(last_pos).extend_with(&b) {
          Ok (()) => {},
          Err(()) => v.push(b),
        }
        return;
      }
    }

    // Need to upgrade from a `One` isizeo a `Many`. This requires replacement.
    let this = mem::replace(self, Empty);
    // We know that we're empty, therefore no drop glue needs to be run.
    unsafe {
      move_val_init(self,
        match this {
          One(b0) => {
            let mut v = Vec::with_capacity(2);
            v.push(b0);
            v.push(b);
            Many(v)
          },
          _ => unreachable!(),
        })
    }
  }

  /// Returns an iterator over references to the buffers inside the `BufSpan`.
  #[inline]
  pub fn iter<'a>(&'a self) -> SpanIter<'a, Buf> {
    match *self {
      Empty       => Opt(None.into_iter()),
      One (ref b) => Opt(Some(b).into_iter()),
      Many(ref v) => Lot(v.as_slice().iter()),
    }
  }

  /// Returns a moving iterator over the buffers inside the `BufSpan`.
  #[inline]
  pub fn isizeo_iter(self) -> SpanMoveIter<Buf> {
    match self {
      Empty   => MoveOpt(None.into_iter()),
      One (b) => MoveOpt(Some(b).into_iter()),
      Many(v) => MoveLot(v.into_iter()),
    }
  }

  /// Returns an iterator over the bytes in the `BufSpan`.
  #[inline]
  pub fn iter_bytes<'a>(&'a self) -> ByteIter<'a, Buf> {
    #[inline]
    fn iter_buf_<B: Iobuf>(buf: &B) -> slice::Iter<u8> {
        unsafe { buf.as_window_slice().iter() }
    }

    #[inline]
    fn deref_u8_(x: &u8) -> u8 { *x }

    let iter_buf : fn(&Buf) -> slice::Iter<u8> = iter_buf_;
    let deref_u8 : fn(&u8) -> u8 = deref_u8_;

    self.iter().flat_map(iter_buf).map(deref_u8)
  }

  /// Returns `true` iff the bytes in this `BufSpan` are the same as the bytes
  /// in the other `BufSpan`.
  ///
  /// ```rust
  /// use iobuf::{BufSpan, ROIobuf, RWIobuf};
  ///
  /// let a = BufSpan::from_buf(ROIobuf::from_str("hello"));
  /// let b = BufSpan::from_buf(RWIobuf::from_str_copy("hello"));
  ///
  /// assert!(a.byte_equal(&b));
  ///
  /// let mut c = BufSpan::from_buf(ROIobuf::from_str("hel"));
  /// c.push(ROIobuf::from_str("lo"));
  ///
  /// assert!(a.byte_equal(&c)); assert!(c.byte_equal(&a));
  ///
  /// let d = BufSpan::from_buf(ROIobuf::from_str("helo"));
  /// assert!(!a.byte_equal(&d));
  /// ```
  #[inline]
  pub fn byte_equal<Buf2: Iobuf>(&self, other: &BufSpan<Buf2>) -> bool {
    self.count_bytes_cmp(other.count_bytes() as usize) == Ordering::Equal
    && self.iter_bytes().zip(other.iter_bytes()).all(|(a, b)| a == b)
  }

  /// A more efficient version of byte_equal, specialized to work exclusively on
  /// slices.
  ///
  /// ```rust
  /// use iobuf::{BufSpan, ROIobuf};
  ///
  /// let a = BufSpan::from_buf(ROIobuf::from_str("hello"));
  ///
  /// assert!(a.byte_equal_slice(b"hello"));
  /// assert!(!a.byte_equal_slice(b"helo"));
  /// ```
  #[inline]
  pub fn byte_equal_slice(&self, other: &[u8]) -> bool {
    self.count_bytes_cmp(other.len()) == Ordering::Equal
    && self.iter_bytes().zip(other.iter()).all(|(a, &b)| a == b)
  }

  /// Counts the number of bytes this `BufSpan` is over. This is
  /// `O(self.iter().len())`.
  ///
  /// ```rust
  /// use iobuf::{BufSpan, ROIobuf};
  ///
  /// let mut a = BufSpan::from_buf(ROIobuf::from_str("hello"));
  /// a.push(ROIobuf::from_str(" "));
  /// a.push(ROIobuf::from_str("world"));
  ///
  /// assert_eq!(a.count_bytes(), 11); // iterates over the pushed buffers.
  /// ```
  #[inline]
  pub fn count_bytes(&self) -> u32 {

    // `self.iter().map(|b| b.len()).sum()` would be shorter, but I like to
    // specialize for the much more common case of empty or singular `BufSpan`s.
    match *self {
      Empty       => 0,
      One (ref b) => b.len(),
      Many(ref v) => v.iter().map(|b| b.len()).sum(),
    }
  }

  /// Compares the number of bytes in this span with another number, returning
  /// how they compare. This is more efficient than calling `count_bytes` and
  /// comparing that result, since we might be able to avoid iterating over all
  /// the buffers.
  ///
  /// ```rust
  /// use std::cmp::Ordering;
  /// use iobuf::{BufSpan, ROIobuf};
  ///
  /// let mut a = BufSpan::from_buf(ROIobuf::from_str("hello"));
  /// a.push(ROIobuf::from_str(" "));
  /// a.push(ROIobuf::from_str("world"));
  ///
  /// assert_eq!(a.count_bytes_cmp(0), Ordering::Greater);
  /// assert_eq!(a.count_bytes_cmp(11), Ordering::Equal);
  /// assert_eq!(a.count_bytes_cmp(9001), Ordering::Less);
  /// ```
  #[inline]
  pub fn count_bytes_cmp(&self, other: usize) -> Ordering {
    let mut other =
      match other.to_u32() {
        None        => return Ordering::Less,
        Some(other) => other,
      };

    match *self {
      Empty       => 0.cmp(&other),
      One (ref b) => b.len().cmp(&other),
      Many(ref v) => {
        for b in v.iter() {
          let len = b.len();
          if len > other { return Ordering::Greater }
          other -= len;
        }
        if other == 0 { Ordering::Equal }
        else          { Ordering::Less  }
      }
    }
  }

  /// Extends this span to include the range denoted by another span.
  ///
  /// ```rust
  /// use iobuf::{BufSpan, ROIobuf};
  ///
  /// let mut a = BufSpan::from_buf(ROIobuf::from_str("hello"));
  /// a.push(ROIobuf::from_str(" "));
  /// let mut b = BufSpan::from_buf(ROIobuf::from_str("world"));
  /// b.push(ROIobuf::from_str("!!!"));
  ///
  /// a.append(b);
  ///
  /// assert!(a.byte_equal_slice(b"hello world!!!"));
  /// ```
  #[inline]
  pub fn append(&mut self, other: BufSpan<Buf>) {
    if self.is_empty() {
      *self = other;
    } else {
      self.extend(other.isizeo_iter())
    }
  }

  /// Returns `true` if the span begins with the given bytes.
  ///
  /// ```rust
  /// use iobuf::{BufSpan, ROIobuf};
  ///
  /// let mut a = BufSpan::from_buf(ROIobuf::from_str("hello"));
  /// a.push(ROIobuf::from_str(" "));
  /// a.push(ROIobuf::from_str("world!"));
  ///
  /// assert!(a.starts_with(b""));
  /// assert!(a.starts_with(b"hel"));
  /// assert!(a.starts_with(b"hello "));
  /// assert!(a.starts_with(b"hello wor"));
  /// assert!(a.starts_with(b"hello world!"));
  ///
  /// assert!(!a.starts_with(b"goodbye"));
  /// ```
  #[inline]
  pub fn starts_with(&self, other: &[u8]) -> bool {
    if self.count_bytes_cmp(other.len()) == Ordering::Less { return false }
    self.iter_bytes().zip(other.iter()).all(|(a, b)| a == *b)
  }

  /// Returns `true` if the span ends with the given bytes.
  ///
  /// ```rust
  /// use iobuf::{BufSpan, ROIobuf};
  ///
  /// let mut a = BufSpan::from_buf(ROIobuf::from_str("hello"));
  /// a.push(ROIobuf::from_str(" "));
  /// a.push(ROIobuf::from_str("world!"));
  ///
  /// assert!(a.ends_with(b""));
  /// assert!(a.ends_with(b"!"));
  /// assert!(a.ends_with(b"rld!"));
  /// assert!(a.ends_with(b"lo world!"));
  /// assert!(a.ends_with(b"hello world!"));
  ///
  /// assert!(!a.ends_with(b"goodbye"));
  /// ```
  #[inline]
  pub fn ends_with(&self, other: &[u8]) -> bool {
    if self.count_bytes_cmp(other.len()) == Ordering::Less { return false }
    self.iter_bytes().rev().zip(other.iter().rev()).all(|(a, b)| a == *b)
  }
}

impl<Buf: Iobuf> PartialEq for BufSpan<Buf> {
    fn eq(&self, other: &BufSpan<Buf>) -> bool {
        self.byte_equal(other)
    }
}

impl<Buf: Iobuf> Eq for BufSpan<Buf> {}

impl<Buf: Iobuf> PartialOrd for BufSpan<Buf> {
    fn partial_cmp(&self, other: &BufSpan<Buf>) -> Option<Ordering> {
        order::partial_cmp(self.iter_bytes(), other.iter_bytes())
    }
}

impl<Buf: Iobuf> Ord for BufSpan<Buf> {
    fn cmp(&self, other: &BufSpan<Buf>) -> Ordering {
        order::cmp(self.iter_bytes(), other.iter_bytes())
    }
}

/// An iterator over the bytes in a `BufSpan`.
pub type ByteIter<'a, Buf> =
  iter::Map<&'a u8, u8,
    iter::FlatMap<&'a Buf, &'a u8,
      SpanIter<'a, Buf>,
      slice::Iter<'a, u8>,
      fn(&Buf) -> slice::Iter<u8>>,
    fn(&u8) -> u8>;

/// An iterator over references to buffers inside a `BufSpan`.
pub enum SpanIter<'a, Buf: 'a> {
  /// An optional item to iterate over.
  Opt(option::IntoIter<&'a Buf>),
  /// A lot of items to iterate over.
  Lot(slice::Iter<'a, Buf>),
}

impl<'a, Buf: Iobuf> Iterator for SpanIter<'a, Buf> {
  type Item = &'a Buf;

  #[inline(always)]
  fn next(&mut self) -> Option<&'a Buf> {
    // I'm couting on this match getting lifted out of the loop with
    // loop-invariant code motion.
    match *self {
      Opt(ref mut iter) => iter.next(),
      Lot(ref mut iter) => iter.next(),
    }
  }

  #[inline(always)]
  fn size_hint(&self) -> (usize, Option<usize>) {
    match *self {
      Opt(ref iter) => iter.size_hint(),
      Lot(ref iter) => iter.size_hint(),
    }
  }
}

impl<'a, Buf: Iobuf> DoubleEndedIterator for SpanIter<'a, Buf> {
  #[inline(always)]
  fn next_back(&mut self) -> Option<&'a Buf> {
    // I'm couting on this match getting lifted out of the loop with
    // loop-invariant code motion.
    match *self {
      Opt(ref mut iter) => iter.next_back(),
      Lot(ref mut iter) => iter.next_back(),
    }
  }
}

impl<'a, Buf: Iobuf> ExactSizeIterator for SpanIter<'a, Buf> {}

/// A moving iterator over buffers inside a `BufSpan`.
pub enum SpanMoveIter<Buf> {
  /// An optional item to iterate over.
  MoveOpt(option::IntoIter<Buf>),
  /// A lot of items to iterate over.
  MoveLot(vec::IntoIter<Buf>),
}

impl<Buf: Iobuf> Iterator for SpanMoveIter<Buf> {
  type Item = Buf;

  #[inline(always)]
  fn next(&mut self) -> Option<Buf> {
    // I'm couting on this match getting lifted out of the loop with
    // loop-invariant code motion.
    match *self {
      MoveOpt(ref mut iter) => iter.next(),
      MoveLot(ref mut iter) => iter.next(),
    }
  }

  #[inline(always)]
  fn size_hint(&self) -> (usize, Option<usize>) {
    match *self {
      MoveOpt(ref iter) => iter.size_hint(),
      MoveLot(ref iter) => iter.size_hint(),
    }
  }
}

impl<Buf: Iobuf> DoubleEndedIterator for SpanMoveIter<Buf> {
  #[inline(always)]
  fn next_back(&mut self) -> Option<Buf> {
    // I'm couting on this match getting lifted out of the loop with
    // loop-invariant code motion.
    match *self {
      MoveOpt(ref mut iter) => iter.next_back(),
      MoveLot(ref mut iter) => iter.next_back(),
    }
  }
}

impl<Buf: Iobuf> ExactSizeIterator for SpanMoveIter<Buf> {}

#[cfg(test)]
mod bench {
  use test::{black_box, Bencher};
  use super::super::iobuf::Iobuf;
  use super::super::impls::{ROIobuf, RWIobuf};
  use super::BufSpan;
  use std::iter::range;

  #[bench]
  fn create_roiobuf(b: &mut Bencher) {
    b.iter(|| {
      let buf = ROIobuf::from_str_copy("hello, world!");
      black_box(buf);
    })
  }

  #[bench]
  fn test_none_to_one(b: &mut Bencher) {
    b.iter(|| {
      let mut buf = BufSpan::new();
      buf.push(ROIobuf::from_str_copy("hello, world!"));
      black_box(buf);
    })
  }

  #[bench]
  fn test_none_to_one_with_copy(b: &mut Bencher) {
    b.iter(|| {
      let mut buf = BufSpan::new();
      let to_push = ROIobuf::from_str_copy("hello, world!");
      buf.push(to_push);
      black_box(buf);
    })
  }

  #[bench]
  fn test_none_to_many(b: &mut Bencher) {
    b.iter(|| {
      let mut buf = BufSpan::new();
      buf.push(ROIobuf::from_str_copy("hello "));
      buf.push(ROIobuf::from_str_copy("world!"));
      black_box(buf);
    })
  }

  #[bench]
  fn extend_1k_iobuf_0(b: &mut Bencher) {
    b.iter(|| {
      let source = RWIobuf::new(1024);
      let mut i = 0u32;
      for _ in range(0u32, 1000) {
        unsafe { source.unsafe_poke_be(i, b'a'); }
        i += 1;
      }
      let mut source = source.read_only();

      let mut dst = BufSpan::new();

      for _ in range(0u32, 1000) {
        unsafe {
          let (start, end) = source.unsafe_split_at(1);
          dst.push(start);
          source = end;
        }
      }

      black_box(dst);
    })
  }

  #[bench]
  fn extend_1k_iobuf_1(b: &mut Bencher) {
    b.iter(|| {
      let source = RWIobuf::new(1024);
      let mut i = 0u32;
      for _ in range(0u32, 1000) {
        unsafe { source.unsafe_poke_be(i, b'a'); }
        i += 1;
      }
      let mut source = source.read_only();

      let mut dst = BufSpan::new();

      for _ in range(0u32, 1000) {
        unsafe {
          let start = source.unsafe_split_start_at(1);
          dst.push(start);
        }
      }

      black_box(dst);
    })
  }

  #[bench]
  fn extend_1k_iobuf_2(b: &mut Bencher) {
    let source = RWIobuf::new(1024);
    let mut i = 0u32;
    for _ in range(0u32, 500) {
      unsafe { source.unsafe_poke_be(i, b'a'); }
      i += 1;
    }

    i = 500;
    for _ in range(500u32, 1000) {
      unsafe { source.unsafe_poke_be(i, b'b'); }
      i += 1;
    }

    b.iter(|| {
      let mut source = source.read_only();

      let mut dst_a = BufSpan::new();
      let mut dst_b = BufSpan::new();
      let mut other = BufSpan::new();

      for _ in range(0u32, 1000) {
        unsafe {
          let first_letter = source.unsafe_split_start_at(1);

          match first_letter.unsafe_peek_be(0) {
            b'a' => dst_a.push(first_letter),
            b'b' => dst_b.push(first_letter),
              _  => other.push(first_letter),
          }
        }
      }

      black_box((dst_a, dst_b, other));
    })
  }

  #[bench]
  fn extend_1k_iobuf_3(b: &mut Bencher) {
    let source = RWIobuf::new(1024);
    let mut i = 0u32;
    for _ in range(0u32, 500) {
      unsafe { source.unsafe_poke_be(i, b'a'); }
      i += 1;
    }

    let mut i = 500;
    for _ in range(500u32, 1000) {
      unsafe { source.unsafe_poke_be(i, b'b'); }
      i += 1;
    }

    b.iter(|| {
      let mut source = source.read_only();

      let mut dst_a = BufSpan::new();
      let mut dst_b = BufSpan::new();
      let mut other = BufSpan::new();

      for _ in range(0u32, 1000) {
        unsafe {
          let first_letter = source.unsafe_split_start_at(1);

          let to_push = match first_letter.unsafe_peek_be(0) {
            b'a' => &mut dst_a,
            b'b' => &mut dst_b,
              _  => &mut other,
          };
          to_push.push(first_letter);
        }
      }

      black_box((dst_a, dst_b, other));
    })
  }

  #[bench]
  fn clone_and_drop(b: &mut Bencher) {
    let patient_zero = RWIobuf::new(1024);
    b.iter(|| {
      let clone = patient_zero.clone();
      black_box(clone);
    })
  }
}
