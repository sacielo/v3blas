# Implementation guidelines: C++ template variant

Status: **shelved for the primary target.** OpenBLAS does not compile C++
sources, so `OpenBLAS.md` is the active plan. This document
keeps the C++ path as a guideline for any target BLAS library that does
accept C++ in-tree (and as the record of the design decision).

## When this variant applies

A target library that (a) compiles C++ in its build and (b) would accept
C++ extension sources. If not both, use the target's own genericity
mechanism instead (see `OpenBLAS.md` for the preprocessor
path). The API spec, kernel set, and coding principles are the same —
they live in `../../README.md` and the current `../kernels/` spec; only
the *genericity mechanism* differs.

## Design guidelines

- **One generic type:** a `v3<T>` struct of three component pointers plus
  `n`, instantiated for `float`, `double`, complex float, complex double
  — same s/d/c/z roles as BLAS. One type, four instantiations, one kernel
  body per op. (The C analogue cannot exist — no templates — which is
  why the C path uses the preprocessor and the ABI never mentions the
  struct.)
- **One template body per family**, instantiated four times per op;
  symbol-level prefixes only (`s3_`, `d3_`, `c3_`, `z3_`) — precision must
  not appear in the body.
- **C facade:** C callers get four plain typedefs generated from one
  macro pattern (`{ T *x,*y,*z; int n; }` → `s3v`, `d3v`, `c3v`, `z3v`),
  and thin wrapper functions that translate struct-in → flat-pointers →
  call the template kernel. The public ABI stays the flat-pointer,
  Fortran-compatible form defined in `../../README.md` — the struct is a
  calling convenience, never the ABI.
- **Layer 0 needs no struct and no template argument struct at all:**
  its operands are plain 1D arrays; a `template <typename T>` free
  function per kernel, same four instantiations.
- Complex instantiations use the library's complex type or a
  two-`FLOAT` pair; semantics per spec: bilinear, no conjugation in v1.
- Inner loops: same FMA-shape rules as the spec (`std::fma` where a
  multiply feeds an add; plain mul/div elsewhere). The compiler's own
  vectorizer is the v1 SIMD story; intrinsics only as a later per-arch
  addition.

## Why it is shelved

1. The primary target (OpenBLAS) is C-only: its build never invokes a
   C++ compiler. The preprocessor idiom is strictly better there — it is
   the library's own mechanism for `saxpy`.
2. The flat-pointer ABI decision (see `../../README.md`, Public API) removed
   the one thing the template design was buying at the API level: the
   single generic struct. What remains of its value (one body, type
   safety at the call site) is caller-side sugar and can be a header
   over the C symbols later.
