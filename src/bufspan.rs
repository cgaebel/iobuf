use collections::slice::{mod, AsSlice, SlicePrelude};
use collections::vec::{mod, Vec};
use core::mem;
use core::iter::Iterator;
use core::option::{mod, Some, None, Option};

use iobuf::Iobuf;

/// A span over potentially many Iobufs. This is useful as a "string" type where
/// the contents of the string can potentially come from multiple IObufs, and
/// you want to avoid copying.
///
/// A `BufSpan` is internally represented as either an `Iobuf` or a `Vec<Iobuf>`,
/// depending on how many different buffers were used.
pub enum BufSpan<Buf> {
  Empty,
  One (Buf),
  Many(Vec<Buf>),
}

impl<Buf: Iobuf> BufSpan<Buf> {
  #[inline]
  pub fn new() -> BufSpan<Buf> {
    Empty
  }

  #[inline]
  pub fn push(&mut self, b: Buf) {
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

  pub fn iter<'a>(&'a self) -> SpanIter<'a, Buf> {
    match *self {
      Empty       => Opt(None.into_iter()),
      One (ref b) => Opt(Some(b).into_iter()),
      Many(ref v) => Lot(v.as_slice().iter()),
    }
  }

  pub fn into_iter(self) -> SpanMoveIter<Buf> {
    match self {
      Empty   => MoveOpt(None.into_iter()),
      One (b) => MoveOpt(Some(b).into_iter()),
      Many(v) => MoveLot(v.into_iter()),
    }
  }
}

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

pub enum SpanMoveIter<Buf> {
  MoveOpt(option::Item<Buf>),
  MoveLot(vec::MoveItems<Buf>),
}

impl<Buf: Iobuf> Iterator<Buf> for SpanMoveIter<Buf> {
  #[inline(always)]
  fn next(&mut self) -> Option<Buf> {
    match *self {
      MoveOpt(ref mut iter) => iter.next(),
      MoveLot(ref mut iter) => iter.next(),
    }
  }
}
