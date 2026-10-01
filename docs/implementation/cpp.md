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

- **One generic descriptor:** `v3<T>` = `{ T *x, *y, *z; long n; }` —
  the same single-pointer fat view the spec defines (SoA components),
  instantiated for `float`, `double`, complex float, complex double.
  One type, four instantiations; `create`/`wrap`/`slice`/`free` as
  inline templates mirroring the C constructors. One kernel body per
  op.
- **One template body per family**, instantiated four times per op;
  symbol-level prefixes only (`s3_`, `d3_`, `c3_`, `z3_`) — precision must
  not appear in the body.
- **C facade:** the C descriptor typedefs (`s3v`, `d3v`, `c3v`, `z3v`)
  *are* the layout (one macro over `{ T *x,*y,*z; long n; }`); the
  template is the same shape. Layer-1 public entries take the
  descriptor pointer — one argument per vector; internal kernels stay
  flat-pointer, entries unwrap (exactly as in `OpenBLAS.md`).
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
2. The descriptor ABI (see `../../README.md`, Public API) is a plain C
   struct reachable through a macro, so the template's one remaining
   API-level asset — the single generic struct — is matched by the C
   path itself. What survives of the template's value (one body, type
   safety at the call site) is caller-side sugar over the C symbols.
