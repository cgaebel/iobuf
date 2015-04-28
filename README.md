Iobuf
=====

A contiguous region of bytes, useful for I/O operations.

[![Build Status](https://travis-ci.org/cgaebel/iobuf.svg?branch=master)](https://travis-ci.org/cgaebel/iobuf)

An Iobuf consists of:

  - a buffer
  - limits   (a subrange of the buffer)
  - a window (a subrange of the limits)

All iobuf operations are restricted to operate within the limits. Initially,
the window of an Iobuf is identical to the limits. If you have an `&mut` to
an Iobuf, you may change the window and limits. If you only have a `&`, you
may not. Similarly, if you have a `RWIobuf`, you may modify the data in the
buffer. If you have a `ROIobuf`, you may not.

The limits can be `narrow`ed, but never widened. The window may be set to
any arbitrary subrange of the limits.

Iobufs are cheap to `clone`, since the buffers are refcounted. Use this to
construct multiple views into the same data.

This started out as a direct port of Jane Street Core's
[Iobuf](https://github.com/janestreet/core/blob/master/lib/iobuf.mli) module,
but has evolved into much more.

Documentation
-------------

See the very thorough [API Docs](https://cgaebel.github.io/iobuf/).
