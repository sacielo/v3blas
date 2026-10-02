# Porting v3blas to OpenBLAS

Target: **OpenBLAS `v0.3.34`**, pinned in `.gitmodules` at
`e0166008be8e466242aa76b2ff75ce3f0fbf574a` as a shallow submodule. Moving to a
newer release means re-basing the patch.

This document is how to land the kernel set in `README.md`. It contains no API
contract: signatures, handles, strides, validation, versioning and the status
mechanism are all in the README, because they are CBLAS's business and another
implementation must honour them too. Where this document and the README
disagree about the API, the README wins and this document is wrong.

Every path and symbol below was read from the pinned tree.

## The model we mirror

`axpy` is the operation to copy, and it is three layers:

1. **Generic kernel body**, `kernel/arm/axpy.c` — portable C against type
   macros, per-element work in an unrolled streaming loop. Registered for four
   precisions at once: `kernel/x86_64/KERNEL.generic:102-105` maps
   `SAXPYKERNEL` and `DAXPYKERNEL` to `../arm/axpy.c`, `CAXPYKERNEL` and
   `ZAXPYKERNEL` to `../arm/zaxpy.c`.
2. **Complex sibling**, `kernel/arm/zaxpy.c` — `FLOAT` becomes the real
   component, arrays are flat interleaved (`x[ix]`, `x[ix+1]`,
   `inc_x2 = 2*inc_x`), the scalar arrives as `da_r, da_i`, and
   `#if !defined(CONJ)` picks the conjugation convention. Our v1 is always the
   no-CONJ path, so that switch stays dormant.
3. **Public entry**, `interface/axpy.c` — the Fortran-ABI `NAME` and the
   CBLAS-style `CNAME`, edge cases, then one call into the kernel through an
   `_k` macro (`AXPYU_K` → `DAXPYU_K` → `daxpy_k`, resolved via
   `common_macro.h` + `common_d.h`; `_k` prototypes live in
   `common_level1.h`).

Per-precision boilerplate is *build glue*, not hand-written:
`GenerateNamedObjects` (`cmake/utils.cmake:305`) emits a two-line wrapper per
precision (`#define CNAME`, `#define DOUBLE`, `#include` the body).

Note what `interface/axpy.c:74` does — `if (n <= 0) return;` — and what it does
**not** do: validate, or report anything. There is no error-status mechanism
anywhere in the tree (`CBLAS_PARAM` is absent from `cblas.h` and `common.h`).
That is exactly why the README defines ours.

## Files to add

| new file | contains |
|---|---|
| `kernel/generic/v3.c` | Layer-1 real bodies → `s3_*`, `d3_*` |
| `kernel/generic/zv3.c` | Layer-1 complex bodies → `c3_*`, `z3_*` |
| `kernel/generic/ew.c` | Layer-0 real bodies → `s1_*`, `d1_*` |
| `kernel/generic/zew.c` | Layer-0 complex bodies → `c1_*`, `z1_*` |
| `interface/v3.c`, `interface/ew.c` | public entries, both `NAME` and `CNAME` |
| `include/v3blas.h` | handle typedefs, constructors, status, user prototypes |

Appended, not rewritten: `_k` prototypes in `common_level1.h`; registration in
`kernel/Makefile.L1` (legacy build) **and** `cmake/kernel.cmake`
(`SetDefaultL1`, `cmake/kernel.cmake:37`). Both registration paths must carry
the change — that is part of the patch, not a follow-up.

The four body files are named `v3`/`z v3`/`ew`/`z ew` rather than per-op
because each holds one family. Real and complex are separate files, following
`axpy.c`/`zaxpy.c`.

## `_k` ABI — ours, deliberately not stock's

Stock's is `int daxpy_k(BLASLONG, BLASLONG, BLASLONG, double, double *, …)`,
called as `AXPYU_K(n, 0, 0, alpha, x, incx, y, incy, NULL, 0)`. The leading
zeros and the trailing `NULL, 0` are dummy slots that let one signature serve
the threaded dispatch. We do not need them, so we do not carry them — four dead
arguments per call across 19 bodies × 4 precisions is not worth mimicking.

```c
/* Layer 1 */
int d3cross_k(BLASLONG n, BLASLONG inc,
              const T *x, const T *y, const T *z, T *cx, T *cy, T *cz);
/* Layer 0 */
int d1had_k(BLASLONG n, BLASLONG incx, BLASLONG incy, BLASLONG incz,
            const T *x, const T *y, T *z);
```

Flat, `cinc` left derived inside the body (it is only ever `3 * inc`), pointers
only — **a kernel body never sees a handle**. The `_k` name is kept because it
is how OpenBLAS organises every other kernel, and it is what `TARGET`
overrides and `KERNEL.generic` registration point at.

## Entry structure

Each public entry does, in order:

1. **Validate** (README §Validation) — `n == 0` returns immediately with the
   status untouched; every other fault sets `V3BLAS_PARAM`. Complete before any
   store, so a rejected call leaves the output as it was.
2. **Handle the stride** — `inc == 1` → the wide `*_CORE` body; otherwise a
   scalar stepping loop `x[i*inc]`, and for `inc < 0` start at
   `x + (n-1)*|inc|`. Layer 0 additionally takes `inc == 0` as broadcast.
3. **Split and dispatch** — see below.

## Threading

`interface/axpy.c` already sets the precedent: `nthreads = num_cpu_avail(1)`,
forced to 1 when `n <= MULTI_THREAD_MINIMAL`, then dispatch. We take both from
there — `num_cpu_avail(int)` (`common_thread.h:143`, which honours
`openblas_set_num_threads` and OpenBLAS's OpenMP mode) and `MULTI_THREAD_MINIMAL`
— so v3blas adds no knob of its own and inherits the host's threshold.

The spawn is `exec_blas(BLASLONG num_cpu, blas_param_t *param, void *buffer)`
(`common_thread.h:198`), **not** `blas_level1_thread`
(`common_thread.h:206`). The latter takes three operand slots, one scalar, and
a cast function pointer, which fits `d3cross` and `d3had` and fits none of
`d3axpby`, `d3crossdot` or the three forks — three mechanisms for nineteen
bodies means three places for an SMP bug. `exec_blas` has no arity limit: we
define one param struct per family, holding the kernel id plus the argument
tuple, and hand the spawned function that single `void *`.

Split `n` into contiguous chunks, never interleaved, so each thread's
prefetcher gets a linear stream. No work stealing: at N = 10⁸ over 16 threads
each chunk is ~150 MB, so imbalance is noise. Thread *k*'s range need not start
on a vector-width boundary; each thread pays a misaligned prologue, invisible
on a memory-bound kernel. `blas_cpu_number` (`common_thread.h:54`) is the count
the rest of the library already consults.

## What the patch must NOT do

- No new library, no new `.so`, no new CMake target — additions land in the
  existing `libblas`.
- No changes to existing routines; only *added* files plus appended lines
  (`common_level1.h`, `kernel/Makefile.L1`, `cmake/kernel.cmake`). A clean,
  mergeable, append-only patch.
- No C++.

## Build & test loop

1. Branch `v3blas` inside `subm/openblas/`.
2. Build with `make` first. CMake is flagged incomplete and experimental
   upstream, and is not installed here — the CMake registration still ships in
   the patch, but the Makefile path is the one that gets tested.
3. `nm -D lib*.so` → exactly the 120 symbols of README acceptance criterion 1.
4. `tests/test_v3.c` — pure C, links `-lblas -lm` only, run at every thread
   count the library offers.
5. `git -C subm/openblas format-patch` → `patch/0001-*.patch`. Regenerated from
   the branch, never hand-edited.

OpenBLAS is not yet built in this environment, so none of the above has run.
