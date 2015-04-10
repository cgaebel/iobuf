use std::cmp::Ordering;
use std::fmt::{self, Formatter, Debug};
use std::intrinsics::{assume, move_val_init};
use std::iter::{self, order, IntoIterator, FromIterator};
use std::mem;
use std::option;
use std::slice;
use std::vec;

use iobuf::Iobuf;

use BufSpan::{Empty, One, Many};
use SpanIter::{Opt, Lot};
use SpanMoveIter::{MoveOpt, MoveLot};

#[cold]
fn bytes_in_vbuf<Buf: Iobuf>(v: &[Buf]) -> usize {
  v.into_iter().map(|b| b.len() as usize).sum()
}

#[test]
fn test_bytes_in_vbuf() {
  use impls::ROIobuf;

  let bs = [
    ROIobuf::from_str("1234"),
    ROIobuf::from_str("5678"),
    ROIobuf::from_str("9"),
  ];

  assert_eq!(bytes_in_vbuf(&bs), 9);
  assert_eq!(bytes_in_vbuf(&bs[1..]), 5);
  assert_eq!(bytes_in_vbuf(&bs[2..]), 1);
}

#[cold]
fn count_bytes_cmp_vbuf<Buf: Iobuf>(v: &[Buf], mut other: usize) -> Ordering {
  for b in v {
    let len = b.len() as usize;
    if len > other { return Ordering::Greater }
    other -= len;
  }
  if other == 0 { Ordering::Equal }
  else          { Ordering::Less  }
}

#[test]
fn test_count_bytes_cmp_vbuf() {
  use impls::ROIobuf;

  let bs = [
    ROIobuf::from_str("1234"),
    ROIobuf::from_str("5678"),
    ROIobuf::from_str("9"),
  ];

  assert_eq!(count_bytes_cmp_vbuf(&bs, 0 ), Ordering::Greater);
  assert_eq!(count_bytes_cmp_vbuf(&bs, 10), Ordering::Less);
  assert_eq!(count_bytes_cmp_vbuf(&bs, 9 ), Ordering::Equal);
}

#[cold]
fn byte_equal_slice_vbuf<Buf: Iobuf>(v: &[Buf], mut other: &[u8]) -> bool {
  if count_bytes_cmp_vbuf(v, other.len()) != Ordering::Equal {
    return false;
  }

  unsafe {
    for b in v {
      let b_as_slice = b.as_window_slice();
      assume(other.len() >= b_as_slice.len());
      let (start, new_other) = other.split_at(b_as_slice.len());
      if b_as_slice != start { return false; }
      other = new_other;
    }
  }

  true
}

#[test]
fn test_byte_equal_slice_vbuf() {
  use impls::ROIobuf;

  let bs = [
    ROIobuf::from_str("1234"),
    ROIobuf::from_str("5678"),
    ROIobuf::from_str("9"),
  ];

  assert!(byte_equal_slice_vbuf(&bs, b"123456789"));
  assert!(!byte_equal_slice_vbuf(&bs, b"123456780"));
  assert!(!byte_equal_slice_vbuf(&bs, b"987654321"));
  assert!(!byte_equal_slice_vbuf(&bs, b"12345678"));
  assert!(!byte_equal_slice_vbuf(&bs, b"23456789"));
}

#[inline]
fn byte_equal_buf_buf<Buf1: Iobuf, Buf2: Iobuf>(x: &Buf1, y: &Buf2) -> bool {
  unsafe {
    x.as_window_slice() == y.as_window_slice()
  }
}

#[inline]
fn byte_equal_buf_vbuf<Buf1: Iobuf, Buf2: Iobuf>(x: &Buf1, y: &[Buf2]) -> bool {
  unsafe { byte_equal_slice_vbuf(y, x.as_window_slice()) }
}

#[cold]
fn byte_equal_vbuf_vbuf<Buf1: Iobuf, Buf2: Iobuf>(x: &[Buf1], y: &[Buf2]) -> bool {
  if count_bytes_cmp_vbuf(x, bytes_in_vbuf(y)) != Ordering::Equal {
    return false;
  }

  unsafe {
    x.into_iter().flat_map(|b| b.as_window_slice().into_iter())
    .zip(y.iter().flat_map(|b| b.as_window_slice().into_iter()))
    .all(|(x, y)| x == y)
  }
}

#[test]
fn test_byte_equal_vbuf_vbuf() {
  use impls::ROIobuf;

  let b0 = [
    ROIobuf::from_str("1234"),
    ROIobuf::from_str("5678"),
    ROIobuf::from_str("9"),
  ];

  let b1 = [
    ROIobuf::from_str("12"),
    ROIobuf::from_str("34567"),
    ROIobuf::from_str("89"),
  ];

  let b2 = [
    ROIobuf::from_str("123456789"),
  ];

  let b3 = [
    ROIobuf::from_str("123456780"),
  ];

  let b4 = [
    ROIobuf::from_str("11111111111111"),
  ];

  assert!(byte_equal_vbuf_vbuf(&b0, &b1));
  assert!(byte_equal_vbuf_vbuf(&b1, &b0));

  assert!(byte_equal_vbuf_vbuf(&b0, &b2));
  assert!(byte_equal_vbuf_vbuf(&b2, &b0));

  assert!(byte_equal_vbuf_vbuf(&b1, &b2));
  assert!(byte_equal_vbuf_vbuf(&b2, &b1));

  assert!(!byte_equal_vbuf_vbuf(&b0, &b3));
  assert!(!byte_equal_vbuf_vbuf(&b1, &b3));
  assert!(!byte_equal_vbuf_vbuf(&b2, &b3));

  assert!(!byte_equal_vbuf_vbuf(&b0, &b4));
  assert!(!byte_equal_vbuf_vbuf(&b1, &b4));
  assert!(!byte_equal_vbuf_vbuf(&b2, &b4));

  assert!(!byte_equal_vbuf_vbuf(&b3, &b4));
}

/// `true` if `v` starts with `other`
#[cold]
fn starts_with_vbuf<Buf: Iobuf>(v: &[Buf], mut other: &[u8]) -> bool {
  for b in v {
    let b = unsafe { b.as_window_slice() };
    match b.len().cmp(&other.len()) {
      Ordering::Greater => return b.starts_with(other),
      Ordering::Equal   => return b == other,
      Ordering::Less    => {
        let (start, new_other) = other.split_at(b.len());
        if b != start { return false; }
        other = new_other;
      }
    }
  }

  // Walked through all of `v`. If `other` is empty, `v` == `other`, and
  // therefore, `v` starts with `other`.
  other.is_empty()
}

#[test]
fn test_starts_with_vbuf() {
  use impls::ROIobuf;

  let b0 = [
    ROIobuf::from_str("1234"),
    ROIobuf::from_str("5678"),
    ROIobuf::from_str("9"),
  ];

  assert!(starts_with_vbuf(&b0, b"123456789"));
  assert!(starts_with_vbuf(&b0, b"12345678"));
  assert!(starts_with_vbuf(&b0, b""));
  assert!(starts_with_vbuf(&b0, b"12345"));

  assert!(!starts_with_vbuf(&b0, b"123456780"));
  assert!(!starts_with_vbuf(&b0, b"1234567890"));
  assert!(!starts_with_vbuf(&b0, b"123450789"));

  assert!(!starts_with_vbuf(&b0, b"2"));
}

#[cold]
fn ends_with_vbuf<Buf: Iobuf>(v: &[Buf], mut other: &[u8]) -> bool {
  for b in v.into_iter().rev() {
    let b = unsafe { b.as_window_slice() };
    match b.len().cmp(&other.len()) {
      Ordering::Greater => return b.ends_with(other),
      Ordering::Equal   => return b == other,
      Ordering::Less    => {
        let (new_other, end) = other.split_at(other.len() - b.len());
        if b != end { return false; }
        other = new_other;
      }
    }
  }

  // Walked through all of `v`. If `other` is empty, `v` == `other`, and
  // therefore, `v` ends with `other`.
  other.is_empty()
}

#[test]
fn test_ends_with_vbuf() {
  use impls::ROIobuf;

  let b0 = [
    ROIobuf::from_str("1234"),
    ROIobuf::from_str("5678"),
    ROIobuf::from_str("9"),
  ];

  assert!(ends_with_vbuf(&b0, b"123456789"));
  assert!(ends_with_vbuf(&b0, b"23456789"));
  assert!(ends_with_vbuf(&b0, b"3456789"));
  assert!(ends_with_vbuf(&b0, b"456789"));
  assert!(ends_with_vbuf(&b0, b"56789"));
  assert!(ends_with_vbuf(&b0, b"6789"));
  assert!(ends_with_vbuf(&b0, b"9"));
  assert!(ends_with_vbuf(&b0, b""));

  assert!(!ends_with_vbuf(&b0, b"1234567890"));
  assert!(!ends_with_vbuf(&b0, b"023456789"));
  assert!(!ends_with_vbuf(&b0, b"123456780"));
  assert!(!ends_with_vbuf(&b0, b"123450789"));
  assert!(!ends_with_vbuf(&b0, b"987654321"));
}

#[inline]
fn cmp_buf_buf<Buf: Iobuf>(bx: &Buf, by: &Buf) -> Ordering {
  unsafe {
    order::cmp(
      bx.as_window_slice().into_iter(),
      by.as_window_slice().into_iter())
  }
}

#[cold]
fn cmp_buf_vec<Buf: Iobuf>(b: &Buf, v: &[Buf]) -> Ordering {
  let mut b = unsafe { b.as_window_slice() };

  for x in v {
    let x = unsafe { x.as_window_slice() };

    if b.len() >= x.len() {
      let (start, new_b) = b.split_at(x.len());

      match order::cmp(start.into_iter(), x.into_iter()) {
        Ordering::Equal => { b = new_b; }
        order => return order,
      }
    } else {
      return order::cmp((&b).into_iter(), x.into_iter());
    }
  }

  if b.is_empty() { Ordering::Equal } else { Ordering::Greater }
}

#[test]
fn test_cmp_buf_vec() {
  use impls::ROIobuf;

  let b0 = [
    ROIobuf::from_str("1234"),
    ROIobuf::from_str("5678"),
    ROIobuf::from_str("9"),
  ];

  assert_eq!(cmp_buf_vec(&ROIobuf::from_str("123456789"), &b0), Ordering::Equal);

  assert_eq!(cmp_buf_vec(&ROIobuf::from_str("12345678"), &b0), Ordering::Less);
  assert_eq!(cmp_buf_vec(&ROIobuf::from_str("1234567890"), &b0), Ordering::Greater);

  assert_eq!(cmp_buf_vec(&ROIobuf::from_str("023456789"), &b0), Ordering::Less);
  assert_eq!(cmp_buf_vec(&ROIobuf::from_str("223456789"), &b0), Ordering::Greater);
}

#[cold]
fn cmp_vec_vec<Buf: Iobuf>(vx: &BufSpan<Buf>, vy: &BufSpan<Buf>) -> Ordering {
  order::cmp(vx.iter_bytes(), vy.iter_bytes())
}

/// A span over potentially many Iobufs. This is useful as a "string" type where
/// the contents of the string can come from multiple IObufs, and you want to
/// avoid copying the buffer contents unnecessarily.
///
/// As an optimization, pushing an Iobuf that points to data immediately after
/// the range represented by the last Iobuf pushed will result in just expanding
/// the held Iobuf's range. This prevents allocating lots of unnecessary
/// intermediate buffers, while still maintaining the illusion of "pushing lots
/// of buffers" while incrementally parsing.
///
/// A `BufSpan` is internally represented as either an `Iobuf` or a `Vec<Iobuf>`,
/// depending on how many different buffers were used.
#[derive(Clone)]
pub enum BufSpan<Buf> {
  /// A span over 0 bytes.
  Empty,
  /// A single span over one range.
  One (Buf),
  /// A span over several backing Iobufs.
  Many(Vec<Buf>),
}

impl<Buf: Iobuf> Debug for BufSpan<Buf> {
  fn fmt(&self, f: &mut Formatter) -> fmt::Result {
    let mut first_time = true;

    for b in self {
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
  fn from_iter<T>(iterator: T) -> Self
      where T: IntoIterator<Item=Buf> {
    let mut ret = BufSpan::new();
    ret.extend(iterator);
    ret
  }
}

impl<Buf: Iobuf> Extend<Buf> for BufSpan<Buf> {
  #[inline]
  fn extend<I: IntoIterator<Item=Buf>>(&mut self, iterator: I) {
    for x in iterator {
      self.push(x);
    }
  }
}

impl<Buf: Iobuf> IntoIterator for BufSpan<Buf> {
  type Item = Buf;
  type IntoIter = SpanMoveIter<Buf>;

  #[inline]
  fn into_iter(self) -> SpanMoveIter<Buf> {
    match self {
      Empty   => MoveOpt(None.into_iter()),
      One (b) => MoveOpt(Some(b).into_iter()),
      Many(v) => MoveLot(v.into_iter()),
    }
  }
}

impl<'a, Buf: Iobuf> IntoIterator for &'a BufSpan<Buf> {
  type Item = &'a Buf;
  type IntoIter = SpanIter<'a, Buf>;

  #[inline]
  fn into_iter(self) -> SpanIter<'a, Buf> {
    match *self {
      Empty       => Opt(None.into_iter()),
      One (ref b) => Opt(Some(b).into_iter()),
      Many(ref v) => Lot(v.into_iter()),
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
  pub fn new() -> Self {
    BufSpan::Empty
  }

  /// Creates a new `BufSpan` from an Iobuf.
  ///
  /// ```rust
  /// use iobuf::{BufSpan, ROIobuf};
  /// use std::iter::IntoIterator;
  ///
  /// let s = BufSpan::from_buf(ROIobuf::from_slice(b"hello"));
  /// assert_eq!((&s).into_iter().count(), 1);
  /// assert_eq!(s.count_bytes(), 5);
  /// ```
  #[inline]
  pub fn from_buf(b: Buf) -> Self {
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
  /// use std::iter::IntoIterator;
  ///
  /// let mut s = BufSpan::new();
  ///
  /// s.push(ROIobuf::from_str("he"));
  /// s.push(ROIobuf::from_str("llo"));
  ///
  /// assert_eq!(s.count_bytes() as usize, "hello".len());
  /// assert_eq!((&s).into_iter().count(), 2);
  ///
  /// let mut b0 = ROIobuf::from_str(" world");
  /// let mut b1 = b0.clone();
  ///
  /// assert_eq!(b0.resize(2), Ok(()));
  /// assert_eq!(b1.advance(2), Ok(()));
  ///
  /// s.push(b0);
  /// s.push(b1);
  ///
  /// // b0 and b1 are immediately after each other, and from the same buffer,
  /// // so get merged into one Iobuf.
  /// assert_eq!(s.count_bytes() as usize, "hello world".len());
  /// assert_eq!((&s).into_iter().count(), 3);
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

    // Need to upgrade from a `One` into a `Many`. This requires replacement.
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

    self.into_iter().flat_map(iter_buf).map(deref_u8)
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
    match (self, other) {
      (&Empty      , &Empty      ) => true,
      (&Empty      ,     _       ) => false,
      (    _       , &Empty      ) => false,
      (&One (ref x), &One (ref y)) => byte_equal_buf_buf(x, y),
      (&One (ref x), &Many(ref y)) => byte_equal_buf_vbuf(x, y),
      (&Many(ref x), &One (ref y)) => byte_equal_buf_vbuf(y, x),
      (&Many(ref x), &Many(ref y)) => byte_equal_vbuf_vbuf(x, y),
    }
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
    match *self {
      Empty       => other.is_empty(),
      One (ref b) => unsafe { b.as_window_slice() == other },
      Many(ref v) => byte_equal_slice_vbuf(v, other),
    }
  }

  /// Counts the number of bytes this `BufSpan` is over. This is
  /// `O(self.into_iter().len())`.
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
  pub fn count_bytes(&self) -> usize {
    // `self.into_iter().map(|b| b.len()).sum()` would be shorter, but I like to
    // specialize for the much more common case of empty or singular `BufSpan`s.
    match *self {
      Empty       => 0,
      One (ref b) => b.len() as usize,
      Many(ref v) => bytes_in_vbuf(v),
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
    match *self {
      Empty       => 0.cmp(&other),
      One (ref b) => (b.len() as usize).cmp(&other),
      Many(ref v) => count_bytes_cmp_vbuf(v, other),
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
  pub fn append(&mut self, other: Self) {
    if self.is_empty() {
      unsafe { move_val_init(self, other) }
    } else {
      self.extend(other.into_iter())
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
    match *self {
      Empty => other.is_empty(),
      One(ref b) => unsafe { b.as_window_slice().starts_with(other) },
      Many(ref v) => starts_with_vbuf(v, other),
    }
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
    match *self {
      Empty       => other.is_empty(),
      One (ref b) => unsafe { b.as_window_slice().ends_with(other) },
      Many(ref v) => ends_with_vbuf(v, other),
    }
  }
}

impl<Buf: Iobuf> PartialEq for BufSpan<Buf> {
  #[inline]
  fn eq(&self, other: &Self) -> bool {
    self.byte_equal(other)
  }
}

impl<Buf: Iobuf> Eq for BufSpan<Buf> {}

impl<Buf: Iobuf> PartialOrd for BufSpan<Buf> {
  #[inline]
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}

impl<Buf: Iobuf> Ord for BufSpan<Buf> {
  #[inline]
  fn cmp(&self, other: &Self) -> Ordering {
    match (self, other) {
      (&Empty, &Empty) => Ordering::Equal,
      (&Empty,    _  ) => Ordering::Less,
      (  _   , &Empty) => Ordering::Greater,
      (&One (ref bx), &One (ref by)) => cmp_buf_buf(bx, by),
      (&One (ref bx), &Many(ref vy)) => cmp_buf_vec(bx, vy),
      (&Many(ref vx), &One (ref by)) => cmp_buf_vec(by, vx).reverse(),
      (&Many(   _  ), &Many(   _  )) => cmp_vec_vec(self, other),
    }
  }
}

/// An iterator over the bytes in a `BufSpan`.
pub type ByteIter<'a, Buf> =
  iter::Map<
    iter::FlatMap<
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
      for _ in 32..1000 {
        unsafe { source.unsafe_poke_be(i, b'a'); }
        i += 1;
      }
      let mut source = source.read_only();

      let mut dst = BufSpan::new();

      for _ in 0..1000 {
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
      for _ in 0..1000 {
        unsafe { source.unsafe_poke_be(i, b'a'); }
        i += 1;
      }
      let mut source = source.read_only();

      let mut dst = BufSpan::new();

      for _ in 0..1000 {
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
    for _ in 0..500 {
      unsafe { source.unsafe_poke_be(i, b'a'); }
      i += 1;
    }

    i = 500;
    for _ in 500..1000 {
      unsafe { source.unsafe_poke_be(i, b'b'); }
      i += 1;
    }

    b.iter(|| {
      let mut source = source.read_only();

      let mut dst_a = BufSpan::new();
      let mut dst_b = BufSpan::new();
      let mut other = BufSpan::new();

      for _ in 0..1000 {
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
    for _ in 0..500 {
      unsafe { source.unsafe_poke_be(i, b'a'); }
      i += 1;
    }

    let mut i = 500;
    for _ in 500..1000 {
      unsafe { source.unsafe_poke_be(i, b'b'); }
      i += 1;
    }

    b.iter(|| {
      let mut source = source.read_only();

      let mut dst_a = BufSpan::new();
      let mut dst_b = BufSpan::new();
      let mut other = BufSpan::new();

      for _ in 0..1000 {
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
