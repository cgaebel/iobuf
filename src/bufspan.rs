use collections::slice::{mod, AsSlice, SlicePrelude};
use collections::vec::{mod, Vec};
use core::clone::Clone;
use core::cmp::{Eq, PartialEq, Ord, PartialOrd, Ordering};
use core::fmt;
use core::mem;
use core::iter::{mod, order, Extend, AdditiveIterator, Iterator, FromIterator};
use core::iter::{DoubleEndedIterator, ExactSize};
use core::option::{mod, Some, None, Option};
use core::result::{Ok, Err};

use iobuf::Iobuf;

use BufSpan::{Empty, One, Many};
use SpanIter::{Opt, Lot};
use SpanMoveIter::{MoveOpt, MoveLot};

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
      }

      try!(b.fmt(f));

      first_time = false;
    }

    Ok(())
  }
}

impl<Buf: Iobuf> FromIterator<Buf> for BufSpan<Buf> {
  #[inline]
  fn from_iter<T: Iterator<Buf>>(iterator: T) -> BufSpan<Buf> {
    let mut ret = BufSpan::new();
    ret.extend(iterator);
    ret
  }
}

impl<Buf: Iobuf> Extend<Buf> for BufSpan<Buf> {
  #[inline]
  fn extend<T: Iterator<Buf>>(&mut self, mut iterator: T) {
    for x in iterator {
      self.push(x);
    }
  }
}

impl<Buf: Iobuf> BufSpan<Buf> {
  /// Creates a new, empty `Bufspan`.
  ///
  /// ```
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
  /// ```
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
  /// ```
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
      BufSpan::Empty => true,
      _              => false,
    }
  }

  /// The fast path during pushing -- either fills in the first buffer, or
  /// extends an existing one.
  ///
  /// Returns `None` if the fast path was taken and nothing more needs to be
  /// done. Returns `Some` if we need to do a slow push.
  fn try_to_extend(&mut self, b: Buf) -> Option<Buf> {
    let len = b.len();
    if len == 0 { return None; }

    match *self {
      Empty => {},
      One(ref mut b0) => {
        unsafe {
          if b0.is_extended_by(&b) {
            b0.unsafe_extend(len);
            return None;
          } else {
            return Some(b); // Upgrade the `One` into a `Many` in the slow path.
          }
        }
      }
      Many(ref mut v) => {
        // I wish this wouldn't unwind, and just abort... :(
        let last = v.last_mut().unwrap();
        unsafe {
          if last.is_extended_by(&b) {
            last.unsafe_extend(len);
            return None;
          } else {
            return Some(b);
          }
        }
      }
    }

    // Handle this case in the fast path, instead of leaving it for slow_push
    // to clean up.
    *self = One(b);
    None
  }

  /// Appends a buffer to a `BufSpan`. If the buffer is an extension of the
  /// previously pushed buffer, the range will be extended. Otherwise, the new
  /// non-extension buffer will be added to the end of a vector.
  ///
  /// ```
  /// use iobuf::{BufSpan, Iobuf, ROIobuf};
  ///
  /// let mut s = BufSpan::new();
  ///
  /// s.push(ROIobuf::from_str("he"));
  /// s.push(ROIobuf::from_str("llo"));
  ///
  /// assert_eq!(s.count_bytes() as uint, "hello".len());
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
  /// // so get merged into one Iobuf.
  /// assert_eq!(s.count_bytes() as uint, "hello world".len());
  /// assert_eq!(s.iter().count(), 3);
  /// ```
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
    let this = mem::replace(self, Empty);
    *self =
      match this {
        Empty   => One(b),
        One(b0) => {
          let mut v = Vec::with_capacity(2);
          v.push(b0);
          v.push(b);
          Many(v)
        },
        Many(mut bs) => { bs.push(b); Many(bs) }
      };
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
  pub fn into_iter(self) -> SpanMoveIter<Buf> {
    match self {
      Empty   => MoveOpt(None.into_iter()),
      One (b) => MoveOpt(Some(b).into_iter()),
      Many(v) => MoveLot(v.into_iter()),
    }
  }

  /// Returns an iterator over the bytes in the `BufSpan`.
  #[inline]
  pub fn iter_bytes<'a>(&'a self) -> ByteIter<'a, Buf> {
    self.iter()
        .flat_map(|buf| unsafe { buf.as_window_slice().iter() })
        .map(|&b| b)
  }

  /// Returns `true` iff the bytes in this `BufSpan` are the same as the bytes
  /// in the other `BufSpan`.
  ///
  /// ```
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
    self.count_bytes() == other.count_bytes()
    && self.iter_bytes().zip(other.iter_bytes()).all(|(a, b)| a == b)
  }

  /// A more efficient version of byte_equal, specialized to work exclusively on
  /// slices.
  ///
  /// ```
  /// use iobuf::{BufSpan, ROIobuf};
  ///
  /// let a = BufSpan::from_buf(ROIobuf::from_str("hello"));
  ///
  /// assert!(a.byte_equal_slice(b"hello"));
  /// assert!(!a.byte_equal_slice(b"helo"));
  /// ```
  #[inline]
  pub fn byte_equal_slice(&self, other: &[u8]) -> bool {
    self.count_bytes() as uint == other.len()
    && self.iter_bytes().zip(other.iter()).all(|(a, &b)| a == b)
  }

  /// Counts the number of bytes this `BufSpan` is over. This is
  /// `O(self.iter().len())`.
  ///
  /// ```
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

  /// Extends this span to include the range denoted by another span.
  ///
  /// ```
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
      self.extend(other.into_iter())
    }
  }

  /// Returns `true` if the span begins with the given bytes.
  ///
  /// ```
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
    if (self.count_bytes() as uint) < other.len() { return false }

    self.iter_bytes().zip(other.iter()).all(|(a, b)| a == *b)
  }

  /// Returns `true` if the span ends with the given bytes.
  ///
  /// ```
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
    if (self.count_bytes() as uint) < other.len() { return false }

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
  iter::Map<'static, &'a u8, u8,
    iter::FlatMap<'static, &'a Buf,
      SpanIter<'a, Buf>,
      slice::Items<'a, u8>>>;

/// An iterator over references to buffers inside a `BufSpan`.
pub enum SpanIter<'a, Buf: 'a> {
  /// An optional item to iterate over.
  Opt(option::Item<&'a Buf>),
  /// A lot of items to iterate over.
  Lot(slice::Items<'a, Buf>),
}

impl<'a, Buf: Iobuf> Clone for SpanIter<'a, Buf> {
  #[inline(always)]
  fn clone(&self) -> SpanIter<'a, Buf> {
    match *self {
      Opt(ref iter) => Opt((*iter).clone()),
      Lot(ref iter) => Lot((*iter).clone()),
    }
  }
}

impl<'a, Buf: Iobuf> Iterator<&'a Buf> for SpanIter<'a, Buf> {
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
  fn size_hint(&self) -> (uint, Option<uint>) {
    match *self {
      Opt(ref iter) => iter.size_hint(),
      Lot(ref iter) => iter.size_hint(),
    }
  }
}

impl<'a, Buf: Iobuf> DoubleEndedIterator<&'a Buf> for SpanIter<'a, Buf> {
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

impl<'a, Buf: Iobuf> ExactSize<&'a Buf> for SpanIter<'a, Buf> {}

/// A moving iterator over buffers inside a `BufSpan`.
pub enum SpanMoveIter<Buf> {
  /// An optional item to iterate over.
  MoveOpt(option::Item<Buf>),
  /// A lot of items to iterate over.
  MoveLot(vec::MoveItems<Buf>),
}

impl<Buf: Iobuf> Iterator<Buf> for SpanMoveIter<Buf> {
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
  fn size_hint(&self) -> (uint, Option<uint>) {
    match *self {
      MoveOpt(ref iter) => iter.size_hint(),
      MoveLot(ref iter) => iter.size_hint(),
    }
  }
}

impl<Buf: Iobuf> DoubleEndedIterator<Buf> for SpanMoveIter<Buf> {
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

impl<Buf: Iobuf> ExactSize<Buf> for SpanMoveIter<Buf> {}
