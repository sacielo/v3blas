# Implementation guidelines: OpenBLAS (primary target)

Status: **primary target.** The submodule is pinned to the release tag
**`v0.3.34`** (commit `c0827a716473bd61d3e8fa44c25184d370400267`), recorded
in `.gitmodules`. Pinning to a tag rather than a branch means the v3blas
patch has to be re-based when moving to a newer release.

**Verification status: unverified.** The mechanism facts below were read
from the public OpenBLAS tree and are believed correct, but
`subm/openblas/` has never been initialized or built in this repo, so
nothing here has been checked against a compiling checkout. Claims that
need confirmation before the patch is written are marked **⚠
unverified**; the first real build closes them.

This document says *how to code* the kernel set (enumerated in
`../kernels/v1set.md`) onto OpenBLAS; it contains no code — the coding happens on
a branch inside `subm/openblas/`. Per deliberate design, this guide names
only the mechanisms, not individual kernels: it codes "the kernel set"
generically, so a new kernel variant never invalidates it.

Hard constraint: **plain C only.** OpenBLAS does not compile C++ sources.
Its own genericity mechanism is the preprocessor (CNAME/FLOAT), and we
use exactly that — the same path `saxpy` itself takes from one C file to
four precision symbols.

## The saxpy model (what we mirror)

**⚠ unverified** — read from the public tree, not confirmed against a
build at v0.3.34. Path names in particular may have moved.

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
- `include/d3.h`, `include/ew.h` — the handle typedefs (`s3v`…`z3v`:
  `{ T *x,*y,*z; blaslong n, inc, cinc; }`; `s1v`…`z1v`:
  `{ T *p; blaslong n, inc; }`), constructor prototypes
  (`v3_create`/`v3_wrap`/`v3_wrap_component`/`v3_component`/`v3_pitch`/
  `v3_destroy`), and user prototypes for both layers,
- appended `_k` prototypes in `common_level1.h`,
- registration: one `ifndef` block in `kernel/Makefile.L1` +
  `SetDefaultL1` fallback lines in `cmake/kernel.cmake`, mapping each
  `<P><OP>KERNEL` name to our generic files.

### Body rules

- Real and complex are separate files (axpy.c/zaxpy.c precedent): the
  complex body works on `FLOAT` re/im pairs; the real body on plain
  `FLOAT`.
- The `_k` kernel ABI stays **flat**: explicit `n` + `inc` + scalars +
  component pointers. Public entries unwrap the `v3`/`v1` handles
  (length and stride taken from the handle, all vector operands
  checked for agreement once) and call flat kernels — kernel bodies
  never see a handle.
- **Threading lives in `interface/`, and `_k` kernels stay
  single-threaded forever.** An entry splits `n` into contiguous chunks, one
  per thread, and calls the `_k` kernel per range. Do **not** copy
  `daxpy_k`'s 10-slot dummy ABI — those slots exist for OpenBLAS's own SMP
  dispatch and we do not want them. Our split needs no ABI growth at all,
  because a `_k` call already takes `n` and a base pointer: chunk *k* is just
  `_k(n_k, inc, base + offset_k, …)`.
- **⚠ unverified — mechanism name to confirm at v0.3.34.** The library's
  existing entry-threaded ops use a `blas_level1_thread`-style path
  (`common_macro.h` / `common_thread.h`). Whether that macro is the right
  thing to reuse for an elementwise L1 op, or whether a hand-rolled static
  chunk loop calling `_k` per range is simpler and safer, is a patch-time
  call. The *contract* is fixed either way: threads in the interface, `_k`
  single-threaded, split on `n`. A hand-rolled loop has one advantage worth
  weighing — it makes the chunk boundaries and the single-threaded threshold
  explicit and auditable, where a generic macro hides both.
- **Chunk, never interleave.** Each thread gets a contiguous range so its
  prefetcher sees a linear stream. A static split with no work stealing is
  sufficient: at N = 10⁸ over 16 threads each chunk is ~150 MB, so imbalance
  is noise. Thread *k*'s range need not start on a vector-width boundary; each
  thread pays a misaligned prologue, which is invisible on a
  memory-bound kernel.
- **One threshold policy, both layers.** Below a per-op element count the
  entry runs single-threaded, because fork overhead exceeds the work. The
  number is measured per op per build, not fixed by the spec (README
  §Deferred). Layer 1 splits the handle's `n`; Layer 0 splits its `n`
  argument. Same code path.
- **No thread-count knob of our own.** Read the count the library already
  exposes (`openblas_set_num_threads` / `OPENBLAS_NUM_THREADS`; internally
  the global the entries already consult). Introducing a second control next
  to the first is a wart that never goes away, and the premise is landing
  inside `libblas`.
- One `*_CORE`-style inner loop per kernel: unrolled streaming pass, no
  tiling (memory-bound by design). FMA shape per the spec's inner-loop
  table (`fma(a,b,c)` wherever a multiply feeds an add; pure
  scales/chains/divisions stay plain — the mul/div floor).
- Cross-fusing kernels (a cross feeding a contraction): cross in
  registers, contraction in registers, intermediate never stored.
- Complex: flat interleaved, scaled kernels take `(a_r, a_i)`; never
  conjugate; keep the `#if !defined(CONJ)`-style switch dormant so the
  Hermitian variant is a later, additive decision.
- Entries: unwrap descriptors (n from the descriptor, consistency of
  all vector operands checked once), edge cases per the spec (n ≤ 0;
  full-RHS-scaled kernels with a = 0 → store zeros); then one flat call
  into the `_k` kernel.
- Allocator (`v3_create`): one block of `v3_pitch(n)` elements, where
  `v3_pitch(n) = ((3n + 7) / 8) * 8` — **fixed 8, never the build's vector
  width.** This is a spec decision, not a tuning knob, and it deliberately
  overrides the obvious choice: rounding to the widest register makes the
  pitch a function of the build, so a buffer wrapped by one build can be
  misaligned in another, and `v3_pitch` can return *fewer* bytes than the
  buffer actually holds. Fixed 8 makes pitch a stable ABI fact, gives
  64-byte alignment for `double` at any `n`, and bounds the waste at 7
  elements total. **Consequence for this guide: the unrolled width is now a
  free choice** (§SIMD below), because nothing outside `v3_create` depends on
  it any more. A fourth component would buy nothing: `y` lands at `base + n`
  whether there are three or four, so the constraint is `n`, not the component
  count. `inc = 1`. **The pitch stays inside the allocator** — it is not a
  handle field and must never reach a kernel — except that `v3_pitch(n)` is
  public, so a caller who `malloc`s and wraps matches it exactly.
- **No tail padding, no store past `n`** (spec principle 6): every inner
  loop is a wide unrolled body plus a scalar tail. The inter-component
  pitch above is *not* slack — it is never written, and it does not relax
  this rule. There is no alignment/flag test to select a fast path, because
  the entry point cannot distinguish a created vector from a wrapped one.
  Reallocating this later would be an ABI break, so it is settled now.
- **Strides**: the `_k` ABI takes `inc` alongside `n`. The core is
  written once for `inc == 1` (the `*_CORE` unrolled body) and the
  entry branches on `inc` exactly as `daxpy` does — `inc == 1` → core,
  otherwise a scalar stepping loop `x[i*inc]`. Vectorized strided
  variants are out of scope for v1 (see README §Deferred).
- **SIMD: plain portable C for v1, no intrinsics.** Inner loops are
  scalar-typed and hand-unrolled 4×, letting the compiler's auto-vectorizer
  emit whatever the build target enables. There are no `#include
  <immintrin.h>` files, no per-arch bodies, and no runtime dispatch of our
  own — OpenBLAS's existing `TARGET` mechanism already selects the
  compiler flags and the CPU the binary is built for, so a per-arch
  override remains available later as an additive change to *this* file.
  **Consequence for the spec's acceptance criterion 4:** a portable build
  will not reach 90% of streaming bandwidth on AVX-512 hardware, so that
  number is per-implementation, not a property of the kernel set. The
  honest v1 target is "no slower than the hand-written scalar reference
  the compiler itself produces", with the ratio *measured* and recorded
  here rather than asserted in the spec. A later implementation that adds
  intrinsics can carry a bandwidth number; this one does not.

### Naming inside the tree

- Symbol form: `<precision-letter><width><op>` — trailing `_` for
  the Fortran entry, no underscore for the C name, `_k` suffix for the
  internal kernel, exactly as stock BLAS relates `daxpy_`/`daxpy`/
  `daxpy_k`.
- Width is in the symbol: `s3`/`d3`/`c3`/`z3` for Layer 1,
  `s1`/`d1`/`c1`/`z1` for Layer 0. **The `1` is not optional for Layer
  0** — stock BLAS already exports `dscal`, `daxpy`, `daxpby` and `dnrm2`,
  each with a different signature, and stock `dnrm2` is a *reduction* to a
  scalar where ours is elementwise. C has no overloading, so a collision
  here would link cleanly and run the wrong function. Nothing in stock
  BLAS begins `<precision>1`.
- Op names are not restated here; they come from the kernel spec
  (`../kernels/v1set.md`). Header include guard + prototype blocks mirror
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
   identities — **re-run at every thread count the library offers**, since a
   split on `n` must not change an output bit, and the canaries must hold
   after threaded calls too.
5. `git -C subm/openblas format-patch` → `patch/0001-*.patch`.

## OpenBLAS-specific open items

- ~~Pinned submodule ref~~ — **resolved: `v0.3.34`**, recorded in
  `.gitmodules`. Moving to a newer release means re-basing the patch.
- v1 SIMD scope: **resolved — plain portable C, no intrinsics** (see §Body
  rules). AVX2 remains available later as a `KERNEL.HASWELL`-style
  override, additive and with no API change.
- **Initialize and build `subm/openblas/` at v0.3.34**, then close the
  **⚠ unverified** marks above. Until then every path name here
  (`kernel/generic/`, `kernel/Makefile.L1`, `cmake/kernel.cmake`,
  `common_level1.h`, `interface/axpy.c`) is a claim from reading the tree,
  not a fact about the pin.
- Confirm with maintainers when PR-ing: placement of new portable L1
  bodies (`kernel/generic/` vs `kernel/arm/`-style cross-reference —
  precedent shows x86_64 happily including `../arm/axpy.c`), and whether
  both build systems must carry the registration in the PR (assumed yes).
