use collections::slice::{mod, AsSlice, SlicePrelude};
use collections::vec::{mod, Vec};
use core::mem;
use core::iter::{mod, AdditiveIterator, Iterator};
use core::option::{mod, Some, None, Option};

use iobuf::Iobuf;

/// A span over potentially many Iobufs. This is useful as a "string" type where
/// the contents of the string can potentially come from multiple IObufs, and
/// you want to avoid copying.
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
  Empty,
  One (Buf),
  Many(Vec<Buf>),
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
    Empty
  }

  /// Creates a new `BufSpan` from a slice of bytes.
  ///
  /// ```
  /// use iobuf::{BufSpan, Iobuf, ROIobuf};
  /// use std::iter::AdditiveIterator;
  ///
  /// let s = BufSpan::from_buf(ROIobuf::from_slice(b"hello"));
  /// assert_eq!(s.iter().count(), 1);
  /// assert_eq!(s.iter().map(|b| b.len()).sum(), 5);
  /// ```
  #[inline]
  pub fn from_buf(b: Buf) -> BufSpan<Buf> {
    One(b)
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
      Empty       => true,
      One (ref b) => b.is_empty(),
      Many(ref v) => v.iter().all(|b| b.is_empty()),
    }
  }

  /// The fast path during pushing -- either fills in the first buffer, or
  /// extends an existing one.
  ///
  /// Returns `None` if the fast path was taken and nothing more needs to be
  /// done. Returns `Some` if we need to do a slow push.
  #[inline]
  fn try_to_extend(&mut self, b: Buf) -> Option<Buf> {
    match *self {
      Empty => {},
      One(ref mut b0) => {
        unsafe {
          if b0.is_extended_by(&b) {
            b0.unsafe_extend(b.len());
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
            last.unsafe_extend(b.len());
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
  #[inline]
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
  /// let b = BufSpan::from_buf(RWIobuf::from_string("hello".into_string()));
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
}

/// An iterator over the bytes in a `BufSpan`.
pub type ByteIter<'a, Buf> =
  iter::Map<'static, &'a u8, u8,
    iter::FlatMap<'static, &'a Buf,
      SpanIter<'a, Buf>,
      slice::Items<'a, u8>>>;

/// An iterator over references to buffers inside a `BufSpan`.
pub enum SpanIter<'a, Buf: 'a> {
  Opt(option::Item<&'a Buf>),
  Lot(slice::Items<'a, Buf>),
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
}

/// A moving iterator over buffers inside a `BufSpan`.
pub enum SpanMoveIter<Buf> {
  MoveOpt(option::Item<Buf>),
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
}
