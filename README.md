# v3blas: Element-wise 3-Vector and Scalar-Array Extensions to BLAS

**DRAFT — working document under iteration. Not final.**

This is the stable introduction: motivation, public-API rules, coding
principles. The kernel-level spec — naming grammar as applied, legend,
op tables, inner-loop shapes, per-op argument forms, composed
operations, test identities — lives in **`docs/kernels/`**, one file
per scheme (current: `spellform.md`), and is the part expected to
iterate (try a new kernel scheme = add a file). Everything here is
kernel-set-agnostic by design. The bug log at the end records
corrections made against earlier drafts of this project.

## Motivation

Every MHD, CFD, and electromagnetics code re-implements the same few-line
element-wise cross/dot/Hadamard on N grid points. This is the "60 years of
physicists" problem. The operations are memory-bound, same roofline class as
SAXPY (arithmetic intensity well below 1). They belong in BLAS.

BLAS is frozen (1991 spec, Level 3). The C++ standard (P1673/P1674) is debating
element-wise ops and hasn't shipped. The implementations (OpenBLAS, oneMKL)
can add extensions *now* without waiting for the standard. `axpby` has been
in OpenBLAS for 15+ years and is still not in the BLAS standard.

## Use case (the beachhead)

MHD time loop, per grid point:

    F = (∇×B) × B / μ₀       ← Lorentz force
    S = (E × B) / μ₀         ← Poynting flux
    Q = (∇×B)·(∇×B) / σ      ← Joule heating
    h = B · (∇×B)            ← helicity density

All element-wise once `∇×B` exists (the stencil that produces it is out of
scope — it needs neighbor values, not this API). All memory-bound.
N = 10⁸–10¹⁰ grid points. Shared operands (`B` feeds `F`, `S` and the next
step; `∇×B` feeds `F` and `Q`) are why inputs must never be destroyed (see
Design principle 3). The compositions over the kernel set are spelled out in
`docs/kernels/spellform.md` §Composed operations.

## Design principles

1. **Generic, not per-precision.** One generic body per family, four
   instantiations. Precision lives only in the symbol prefix
   (`s3_`, `d3_`, `c3_`, `z3_`), exactly as
   `saxpy`/`daxpy`/`caxpy`/`zaxpy` differ. *How* one body becomes four
   precisions (preprocessor, template, …) is an implementation question —
   see the `docs/implementation/` documents.
2. **The name is the formula.** Kernel names spell out their mathematics in
   a fixed one-letter grammar (applied in `docs/kernels/spellform.md` §Naming system).
   No domain names, no implementation names
   (no "fma" anywhere — FMA is a property of the compiled inner loop, not
   of the API).
3. **Pure-first.** Inputs are `const`, never modified. Outputs are
   write-only, caller-provided, fresh. In-place forms exist only as explicit
   sibling kernels, named so that the in-place nature is visible in the
   name. (BLAS's 1979 in-place convention saves nothing — traffic is
   identical — and it destroys shared operands, which the beachhead cannot
   afford.)
4. **Zero reductions in v1.** Every kernel is a single streaming pass: no
   partial sums, no thread sync, no atomics. A global sum, if ever needed,
   is a pointwise-dot kernel + a stock `ddot` on its result.
5. **FMA-shaped inner loops.** Every `±` in the math is absorbed into the
   last multiply per element; no standalone add the FMA unit could fuse;
   intermediates (e.g. a cross feeding a contraction) stay in registers.
   Scalar arguments and in-place accumulation exist in the API *because*
   they let the math stay in registers and cut memory traffic.

## Public API (wrapper definitions)

Every entry is **flat-pointer, BLAS/Fortran-ABI**: all arguments passed by
reference, `n` an int, arrays pointers, scalars pointers. No struct in the
ABI, no allocation, no new library: the symbols land inside `libblas` next
to `saxpy`/`daxpy` and link with `-lblas` alone. (A per-precision struct
typedef may be offered in the header as pure C-caller sugar; it changes
nothing linkable.)

**Symbol form:** `<precision><name>_`, precision ∈ {s, d, c, z}. Layer 1
names carry the `3` marker (`d3zeaxvy_`); Layer 0 names do not (`dters_`).
C-callers use the same symbols without the trailing underscore, exactly as
for stock BLAS.

**Argument order is mechanical — the name is the signature:** `n` first,
then the remaining operands in order of first appearance on the
right-hand side, left to right; scalars appear where their term spells
them; a fresh output comes last; in-place outputs reuse their single
pointer. The c/z precisions take complex scalars (a pair of reals) in
the same slots. (The per-op enumeration for the current kernel set:
`docs/kernels/spellform.md` §Per-op argument forms.)

**Wrapper (entry) behavior.** The public entry handles the edge cases —
`n ≤ 0` → return; for kernels whose *entire* RHS is `a·(…)`, `a = 0` →
store zeros and return (kernels with extra terms get no such shortcut) —
and then runs the kernel: one streaming pass over `n` elements. v1:
single-threaded, contiguous only (see Deferred). The mechanics of entry,
kernel body, and build registration are per target library — see
`docs/implementation/`.

## Coding principles (memory model)

- No allocation in the API. Caller owns all pointers (same as BLAS): each
  call is a *view* over caller arrays, owning no state — same role as `lda`
  in `dgemm`. On GPU backends the pointers are device pointers (same as
  cuBLAS); the user allocates via their platform's allocator.
- Inputs are `const`. Outputs are write-only, fresh, caller-provided; the
  in-place siblings additionally read their LHS operand (per-element, so
  aliasing is safe by construction).
- Single streaming pass; no partial sums, no thread sync, no atomics (v1).
- FMA shape: every multiply that feeds an add in the formula is written as
  `fma(a,b,c)`; sums of products accumulate through the FMA; pure scales,
  product chains, and division operands stay plain mul/div (the floor —
  nothing to fold).
- Intermediates stay in registers (e.g. a cross feeding a contraction).
- v1 complex: flat interleaved re/im elements; **no conjugation anywhere**
  (pointwise self-products in c/z precisions are bilinear, not Hermitian).
- v1: contiguous component arrays only; a *sub-run* (e.g. a halo-padded row
  interior) is expressed by pointing into the array with a shorter `n` —
  same mechanism as BLAS subvectors. Strided subsequences: v2.

## Layer split (why two layers, one patch)

Layer 1 (v3) holds the ops that use the 3-vector *structure*: cross and
dot contractions need the three-component grouping. Layer 0 (plain 1D,
symbols without `3`) holds the structure-blind half of the algebra —
useful to any code doing element-wise work, and what makes field
coefficients and mixed-component products expressible. Layer-1 kernels
are self-contained fused passes (register fusion) and do **not** call
Layer-0 kernels: no internal ordering dependency, no patch split. The
two layer definitions and their counts: `docs/kernels/spellform.md`.

## Deferred (stash / TODO)

- **Multi-output, shared-operand kernels** (naming for multi-LHS formulas
  to be devised when revisited):
  - `cross2b`:  c1 = x×y, c2 = x×w            (shared x; w = 4th vector)
  - `cross3b`:  c1 = x×y, c2 = x×w, c3 = x×?  (needs a 5th-vector letter)
  - `dot3cross` (old doc's fused dot+cross)
- **Strides (v2 — requirement confirmed: halo-padded and staggered
  storage exist in target codes).** v1 works on dense runs only, but a
  *sub-run* is already expressible in v1 by pointer offset + shorter `n`
  (a call is a view — same as BLAS subvectors via `x + (i-1)*incx`);
  e.g. row interiors of a halo-padded 2D grid are processed by per-row
  calls, no copies. What v1 cannot do in one call: a *strided*
  subsequence (`inc ≠ 1`) — a column of padded rows, even/odd
  sub-lattices, interleaved components. v2 design: one extra `long inc`
  pointer argument (applied to all three components); kernel = entry
  branch exactly like `daxpy` — `inc == 1` → the fast SIMD loop, else
  scalar stepping loop. A single `inc` expresses 1D arithmetic
  progressions only; a 2D interior of per-row-halo storage is a set of
  dense runs, handled by per-row calls. Staggered grids: each component
  is dense in its own array (stagger is positional, not memory spacing),
  so SoA staggered storage is served by v1.
- **Complex conjugation conventions** (v1: no conjugation anywhere)
- **Reductions** (global dot/norm; compose with stock `ddot` meanwhile)
- **Fresh 4-vector forms** (e.g. `w = x⊙y⊙z` into a fresh `w`; only the
  in-place 3-distinct forms are provided)
- **Fortran operator module** (`E ^ B`, `E * B` from the old doc) —
  deferred past v1.

## Acceptance criteria

1. `nm -D` on the built library shows the full symbol set of the current
   kernel spec (`docs/kernels/spellform.md`) — every kernel in all four precisions,
   alongside `saxpy`/`daxpy`.
2. A short C test compiles and links with `-lblas -lm` and nothing else.
3. Tests pass for all four precisions, random and edge N (0, 1, odd,
   unaligned).
4. Microbenchmark shows each kernel at memory-bandwidth roofline.
5. The kernel-set identities of `docs/kernels/spellform.md` pass.

## Implementations

The spec above is implementation-agnostic — and the implementation
guides deliberately never reference individual kernel names; they code
"the kernel set" generically. Each target BLAS library gets its own
guidelines document:

- **`docs/implementation/OpenBLAS.md`** — the primary target. Plain C, the
  library's own CNAME/FLOAT preprocessor idiom (verified against OpenBLAS
  master: that is exactly how `saxpy` reaches four precisions from one
  C file). Also defines the host-repo layout (submodule + patch + tests).
- **`docs/implementation/cpp.md`** — C++ template variant: one struct, one
  template body per op. Kept for targets that accept C++ sources;
  OpenBLAS does not compile C++, so it is not the primary path.
- (future: `docs/implementation/clblast.md`, `docs/implementation/onemkl.md`, … — each maps
  the same kernel spec (`docs/kernels/spellform.md`) onto one library's genericity
  mechanism.)

## Corrections to the first draft (bug log)

1. `d3v/s3v/c3v/z3v` four structs → one generic body per family;
   precision is a symbol-level prefix only.
2. `fma3/fma3m/divfma/ratfma/sqr/sqrm/triple` → formula-spelled names;
   `divfma` eliminated outright (dividing by a scalar = multiplying by its
   reciprocal).
3. `lorentz`/`poynting` removed from the API (compositions of `zeaxvy`);
   the old claim "axpby is literally d3_fma3" corrected: the 3-vector
   `axpby` is `yeaxpby`.
4. `cross2`'s Lagrange framing was backwards — see `rexvyxdvy` in
   `docs/kernels/spellform.md`.
5. Global-reduction `dot3`/`norm2` removed; the v1 API has zero reductions
   (all pure streaming).
6. Stride support removed (it contradicted the stride-less struct);
   contiguous-only in v1, v2 adds `inc`.
7. In-place-by-default (1979 BLAS convention) rejected: pure-first;
   in-place is an explicit, name-visible sibling.
8. Savings table in the old doc is arithmetically correct once the unit is
   defined: **1 read = 1 component array** (8 GB at N = 10⁹).
9. Complex semantics made explicit: no conjugation in v1.
10. Mid-session idea to cut the Hadamard 3-kernels and split into two
    patches (0000/0001) is **reversed**: all v3 kernels are kept, the
    structure-blind family also gets 1D (Layer-0, no-`3`) versions, and
    everything ships in one patch.
11. **Notation re-foundation.** The dot-product glyph moves `s` → `d`
    (freeing `s`); pure scalars `a b c`, scalar arrays `r s t (u)`,
    3-vectors `x y z (w)`. Layer 0 re-spelled in scalar-array letters only
    (no `x y z`): old `zexy` → `ters`, `zeaxyz` → `tearst`. Layer-1 dot
    kernels → `rexdy`/`rexdx`/`rexvydz`/`rexvyxdvy`.
12. **In-place division doubled.** Pure division stays single (`zexoy`);
    in-place division ships as mirror pairs (LHS in denominator / LHS in
    numerator): `yeaxoy`/`yeayox`, `zeaxyoz`/`zeaxzoy`, and their Layer-0
    twins `searos`/`seasor`, `tearsot`/`teartos`.
13. **Implementation split.** The C++ template design moved out of the
    core spec into `docs/implementation/cpp.md` (it is shelved for the primary
    target: OpenBLAS does not compile C++). The active plan is plain C
    through OpenBLAS's own CNAME/FLOAT preprocessor idiom, documented in
    `docs/implementation/OpenBLAS.md` after verification against an OpenBLAS
    checkout. The ABI is flat pointers either way; the v3 struct survives
    only as optional header sugar.
14. **Kernel split.** Kernel descriptions + legend moved out of this
    introduction into `docs/kernels/spellform.md` (the file meant to iterate); the
    argument-order rule replaced its per-op tables here (the rule is
    mechanical: the name is the signature); the implementation guides
    were scrubbed of individual kernel names — they reference "the
    kernel set" only.
15. **Naming-system split.** The letter space, spelling rules, and
    reading examples moved to `docs/kernels/spellform.md` — grammar and vocabulary
    evolve with the kernel set, so they ship in the swappable file; the
    introduction keeps only the invariant principle (§Design
    principles 2).

## Open items (not yet decided)

1. ~~**Primary API language**~~ — **resolved: plain C**, the preprocessor
   is the template engine (see `docs/implementation/OpenBLAS.md`); the C++ variant
   lives on in `docs/implementation/cpp.md` for C++-accepting targets.
2. **Submodule pin** — OpenBLAS chosen (the doc's existing target,
   `axpby` precedent, C, realistic PR path); pinned ref TBD.
3. **v1 SIMD scope** — portable C only for the first patch vs portable C
   + one AVX2 path (per-arch override makes SIMD additive later either
   way; draft assumes portable-C-only first).
4. ~~**Unapproved additions**~~ — resolved: keep all; the structure-blind
   family additionally ships 1D (Layer-0) versions in the same patch.
5. ~~**Fortran operator module**~~ — resolved: deferred past v1 (moved to
   Deferred).
