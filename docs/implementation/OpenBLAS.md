# Implementation guidelines: OpenBLAS (primary target)

Status: **primary target.** All mechanism facts below were verified
against an OpenBLAS checkout in `subm/openblas/` (master @ 539ca18,
shallow; that commit is the submodule pin).
This document says *how to code* the kernel set (enumerated in
`../kernels/spellform.md`) onto OpenBLAS; it contains no code — the coding happens on
a branch inside `subm/openblas/`. Per deliberate design, this guide never names
individual kernels: it codes "the kernel set" generically, so a new
kernel variant never invalidates it.

Hard constraint: **plain C only.** OpenBLAS does not compile C++ sources.
Its own genericity mechanism is the preprocessor (CNAME/FLOAT), and we
use exactly that — the same path `saxpy` itself takes from one C file to
four precision symbols.

## The verified saxpy model (what we mirror)

Three layers, one operation:

1. **Kernel body** — one portable C file written against type macros,
   `kernel/arm/axpy.c`. It defines only `CNAME` and `FLOAT`-typed code;
   the same file is registered for two precisions at once
   (`SAXPYKERNEL` *and* `DAXPYKERNEL` = `../arm/axpy.c` in
   `kernel/x86_64/KERNEL.generic`; identical `ifndef`-guarded defaults
   in legacy `kernel/Makefile.L1`). The per-element work sits in
   `AXPY_CORE`, an unrolled streaming loop. SIMD variants are separate
   files (`kernel/x86_64/daxpy.c` + `daxpy_microk_*.c`) selected by
   `KERNEL.<CORE>` overrides — i.e. **generic body now, per-arch SIMD is
   purely additive later.**
2. **Complex sibling** — `kernel/arm/zaxpy.c`: `FLOAT` becomes the real
   component; arrays are flat interleaved re/im (`x[ix]`, `x[ix+1]`,
   `inc_x2 = 2*inc_x`); the scalar arrives as the pair `da_r, da_i`;
   `#if !defined(CONJ)` selects the conjugation convention. Our v1 is
   always the no-CONJ path. The `z`-prefix-on-the-file convention marks
   the complex version (`zaxpy.c`, `zscal.c`, `zrot.c`).
3. **Public entry + glue** — `interface/axpy.c` defines the Fortran-ABI
   entry `NAME` (`daxpy_`: every argument a pointer) and the CBLAS-style
   `CNAME` (`daxpy`: values), handles edge cases, and calls the kernel
   via an `_k` macro (`AXPYU_K → DAXPYU_K → daxpy_k` through
   `common_macro.h` + `common_d.h`; `_k` prototypes in
   `common_level1.h`). The per-precision type loop is *build glue*:
   CMake's `GenerateNamedObjects` (`cmake/utils.cmake`) auto-generates a
   two-line wrapper per precision (`#define CNAME daxpy_k`, `#define
   DOUBLE`, `#include` the body) — nothing to hand-write. Default kernel
   registration: `SetDefaultL1` in `cmake/kernel.cmake` (CMake build) and
   the `ifndef` blocks in `kernel/Makefile.L1` (legacy build). Both must
   be patched.

## Coding guidelines for our patch

### Families and files

Four generic bodies, each compiled twice by the build glue:

| new file | contains | precisions produced |
|----------|----------|---------------------|
| `kernel/generic/v3.c`  | Layer-1 real bodies   | `s3_*`, `d3_*` |
| `kernel/generic/zv3.c` | Layer-1 complex bodies | `c3_*`, `z3_*` |
| `kernel/generic/ew.c`  | Layer-0 real bodies   | `s*`, `d*` |
| `kernel/generic/zew.c` | Layer-0 complex bodies | `c*`, `z*` |

plus:

- `interface/v3.c`, `interface/ew.c` — public entries (pattern:
  `interface/axpy.c`),
- `include/d3.h`, `include/ew.h` — user prototypes (flat pointers;
  optional per-precision struct typedef sugar for C callers),
- appended `_k` prototypes in `common_level1.h`,
- registration: one `ifndef` block in `kernel/Makefile.L1` +
  `SetDefaultL1` fallback lines in `cmake/kernel.cmake`, mapping each
  `<P><OP>KERNEL` name to our generic files.

### Body rules

- Real and complex are separate files (axpy.c/zaxpy.c precedent): the
  complex body works on `FLOAT` re/im pairs; the real body on plain
  `FLOAT`.
- v1 kernels take the public argument shape (n + scalars + array
  pointers, contiguous). Do **not** copy `daxpy_k`'s 10-slot dummy ABI —
  those slots exist for the SMP dispatch we don't use in v1. When SMP
  arrives, entries adopt the `blas_level1_thread` pattern and the ABI
  grows then.
- One `*_CORE`-style inner loop per kernel: unrolled streaming pass, no
  tiling (memory-bound by design). FMA shape per the spec's inner-loop
  table (`fma(a,b,c)` wherever a multiply feeds an add; pure
  scales/chains/divisions stay plain — the mul/div floor).
- Cross-fusing kernels (a cross feeding a contraction): cross in
  registers, contraction in registers, intermediate never stored.
- Complex: flat interleaved, scaled kernels take `(a_r, a_i)`; never
  conjugate; keep the `#if !defined(CONJ)`-style switch dormant so the
  Hermitian variant is a later, additive decision.
- Entries: edge cases per the spec (n ≤ 0; full-RHS-scaled kernels with
  a = 0 → store zeros); then one call into the `_k` kernel.

### Naming inside the tree

- Symbol form: `<precision-letter>[3]<spec-name>` — trailing `_` for
  the Fortran entry, no underscore for the C name, `_k` suffix for the
  internal kernel, exactly as stock BLAS relates `daxpy_`/`daxpy`/
  `daxpy_k`.
- Layer 1 carries the `3` marker (prefixes s3/d3/c3/z3); Layer 0 uses
  plain BLAS prefixes (s/d/c/z) — v3 ops stay distinguishable at link
  level and never collide with stock symbols.
- Op names are not restated here; they come from the kernel spec
  (`../kernels/spellform.md`). Header include guard + prototype blocks mirror
  `common_level1.h` style.

### What the patch must NOT do

- No new library, no new `.so`, no new CMake target — additions land in
  the existing `libblas`.
- No changes to existing routines; only *added* files plus appended
  lines (`common_level1.h`, `Makefile.L1`, `cmake/kernel.cmake`). A
  clean, mergeable, append-only patch.
- No C++.

## Host repo layout (this folder)

    v3blas/
    ├── .git                      # repo
    ├── README.md                 # the core intro (main page) + bug log
    ├── docs/kernels/<scheme>.md  # swappable kernel spec, one file per scheme
    ├── docs/implementation/OpenBLAS.md   # this file
    ├── docs/implementation/cpp.md
    ├── subm/openblas/            # submodule → OpenBLAS (pinned commit)
    ├── patch/0001-*.patch      # git -C subm/openblas format-patch of branch v3blas
    ├── tests/
    │   ├── test_v3.c             # links ONLY -lblas -lm
    │   └── Makefile
    └── Makefile                  # submodule init → patch apply → build

Develop on a branch inside the `subm/openblas/` checkout; `patch/` is
regenerated from it, never hand-edited. The repo hosts docs + patch +
tests + glue — no duplicated source. (`subm/openblas/` is registered as
the submodule, pinned to the current master head; it is a *shallow*
clone — `git -C subm/openblas fetch --unshallow` if full history is
ever needed locally.)

## Build & test loop

1. Branch `v3blas` inside `subm/openblas/`; code the four bodies + two interface
   files + headers + registration lines.
2. Build (`make` legacy and/or CMake config — both registration paths
   must work).
3. `nm -D libblas.so` check → the complete symbol set of the kernel
   spec (it states its own counts), beside stock `saxpy_`.
4. `tests/test_v3.c`: pure C, links `-lblas -lm` only; four precisions;
   random + edge N (0, 1, odd, unaligned); the kernel spec's test
   identities.
5. `git -C subm/openblas format-patch` → `patch/0001-*.patch`.

## OpenBLAS-specific open items

- Pinned submodule ref (currently a shallow clone of master).
- v1 SIMD scope: portable-C-only first patch (draft assumption); AVX2
  later as a `KERNEL.HASWELL`-style override, no API change.
- Confirm with maintainers when PR-ing: placement of new portable L1
  bodies (`kernel/generic/` vs `kernel/arm/`-style cross-reference —
  precedent shows x86_64 happily including `../arm/axpy.c`), and whether
  both build systems must carry the registration in the PR (assumed yes).
