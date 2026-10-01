# v3blas: Element-wise 3-Vector and Scalar-Array Extensions to BLAS

**DRAFT — working document under iteration. Not final.**

This is the stable introduction: motivation, public-API rules, coding
principles. The kernel-level spec — names, op tables, argument forms,
composed operations, test identities — lives in
**`docs/kernels/v1set.md`**, and is the part expected to iterate. Two
earlier naming schemes are kept for the record and are superseded:
`docs/kernels/spellform.md` (a frozen 41-body letter spelling) and
`docs/kernels/letterspace.md` (the rejected letter grammar, §Corrections
21). Everything here is kernel-set-agnostic by design. The bug log at
the end records corrections made against earlier drafts of this
project.

> **Start here next time.** The kernel set is **19 bodies / 76 symbols** —
> ten Layer-1 (`d3*`), six Layer-0 (`d1*`), three forks — and is settled
> pending review. Three things do not exist yet and block acceptance:
> `tests/`, the top-level `Makefile`, and an initialized `subm/openblas/`.
> **§Open items → OPEN carries fourteen labelled questions (A1–A5, B1–B4,
> C1–C4, D1); read that table before changing anything.** Nothing in this
> repo has ever been built or run.

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

    F = (∇∧B) ∧ B / μ₀       ← Lorentz force
    S = (E ∧ B) / μ₀         ← Poynting flux
    Q = (∇∧B)·(∇∧B) / σ      ← Joule heating
    h = B · (∇∧B)            ← helicity density

All element-wise once `∇∧B` exists (the stencil that produces it is out of
scope — it needs neighbor values, not this API). All memory-bound.
N = 10⁸–10¹⁰ grid points. Shared operands (`B` feeds `F`, `S` and the next
step; `∇∧B` feeds `F` and `Q`) are why inputs must never be destroyed (see
Design principle 3). The compositions over the kernel set are spelled out in
`docs/kernels/v1set.md` §Composed operations.

**These four are the whole of v1's motivation, and they are pure vector
algebra** — no component is ever named:

```c
d3crossscal(&curlB, &B, 1/mu0, &F);        /* F = (∇∧B)∧B / μ₀  — Lorentz  */
d3crossscal(&E,    &B, 1/mu0, &S);        /* S = (E∧B)   / μ₀   — Poynting */
d3dotxy_dotxz(&curlB, &curlB, &B, &Q0, &h);  /* Q₀ = (∇∧B)·(∇∧B), h = B·(∇∧B) */
d1axpby(Q0.n, 1/sigma, &Q0.p, Q0.inc, 0, &Q0.p, Q0.inc, &Q.p, Q.inc);
```

Each `μ₀⁻¹` and `σ⁻¹` is folded into a scalar argument, so every term is one
kernel and one pass — and `Q`/`h` share `∇∧B` via the fork, so the whole
beachhead is four calls. The F/S pair is two calls and *cannot* be one: they
share `B` as the second operand, and cross is not commutative (§The v1
kernel set).

Field coefficients stay expressible inside Layer 1 without reaching for a
component: `d3had(&rho, &v, &rho_v)` — a scalar array against a vector is a
cross-layer *shape*, not a component access. What genuinely is out of scope
is anything component-shaped — `ρ·v.x²` is one entry of a second-order
tensor, the same reason `∇∧B` is (see §Deferred).

### The v1 kernel set

**19 bodies, 76 symbols, both layers.** Small enough to be obviously correct,
complete enough to run the motivating loop with no Layer-0 detour. Names
follow BLAS convention and nothing else — see Design principle 2 for what
replaced "the name is the formula".

**Layer 1 — 3-vectors.** Ten bodies. The `3` after the precision letter names
the width, and the width is fixed: there is no length argument.

| symbol | mathematics | earns its place |
|---|---|---|
| `d3scal` | z = a·x | |
| `d3axpy` | y = a·x + y | accumulate form of the above |
| `d3axpby` | z = a·x + b·y | every RK/IMEX stage of the time loop |
| `d3had` | z = x.*y | field coefficient, ρ·v |
| `d3cross` | z = x∧y | the cross itself |
| `d3crossscal` | z = a·(x∧y) | **F, S** — scalar folded in, cross never stored |
| `d3dot` | r = x·y | **h, Q** |
| `d3norm2` | z = x·x | |
| `d3crossdot` | r = (x∧y)·z | α-effect, cross stays in registers |
| `d3crossnorm2` | r = ‖x∧y‖² | Alfvén speed, cross stays in registers |

Composites name the **inner op first**: `cross` then `dot` is `(x∧y)·z`. So
`d3crossdot` also spells `x·(y∧z)` — dot is commutative, the arguments
permute, and no second symbol is needed. `d3crossnorm2` is `d3dot` applied
to a cross with itself, `(x∧y)·(x∧y)`, so it is the chained form of an
existing op rather than a new one. `d3norm2` is `d3dot(x,x)` spelled
shorter; it ships because it reads better at the call site, and it is four
symbols of admitted duplication.

**Forks** — two outputs, one body, one symbol. `_` separates the clauses and a
clause spells the operands it uses, so the shared operand is written once.
**All three follow one rule: the first operand is shared, the second varies.**

| symbol | replaces | shape |
|---|---|---|
| `d3crossxy_crossxz` | `d3cross` + `d3cross` | c₁ = x∧y, c₂ = x∧z — the div-B shape (v∧B, v∧E) |
| `d3crossxy_dotxy` | `d3cross` + `d3dot` | c = x∧y, r = x·y |
| `d3dotxy_dotxz` | `d3dot` + `d3dot` | r₁ = x·y, r₂ = x·z — **Q, h** |

`d3dotxy_dotxz` closes the beachhead: with x = ∇∧B, y = ∇∧B, z = B it gives
Q₀ = x·y and h = x·z in one pass. Passing `x` and `y` as the same handle is
legal and costs nothing — the clauses are independent and the shared operand
is simply read twice from cache.

**The F/S pair provably cannot be a fork, and that is worth knowing.** F and S
share `B`, but as the *second* operand: F = ∇∧B ∧ B and S = E ∧ B. A fork
shares the first operand, and cross is not commutative, so `x∧y` and `y∧x`
cannot be swapped to fit the shape. That is the difference between the two
pairs, and it is not a naming accident.

Register pressure is what caps this at two. A cross has three live component
expressions, so two cross clauses need six and a three-way fan would need
nine — more than the sixteen vector registers of an x86-64 target. That is
the whole reason wider forks are deferred rather than merely unwritten.

**Layer 0 — scalar arrays.** Six bodies. The `1` names the width for the same
reason `3` does, and it is **mandatory here, not decorative**: stock BLAS
already exports `dscal`, `daxpy`, `daxpby` and `dnrm2`, each with a different
signature and — for `dnrm2` — the opposite semantics (stock reduces to a
scalar, ours is elementwise). Reusing those symbols would link cleanly and
run the wrong function, since C has no overloading. Nothing in stock BLAS
begins `<precision>1`, so `d1*` is collision-free.

| symbol | mathematics | stock equivalent |
|---|---|---|
| `d1scal` | z = a·x | `dscal` (in place) |
| `d1axpy` | y = a·x + y | `daxpy` (in place) |
| `d1axpby` | z = a·x + b·y | OpenBLAS `daxpby` (in place) |
| `d1had` | z = x.*y | none |
| `d1nrm2` | z = ‖x‖² | `dnrm2` (a *reduction* to a scalar) |
| `d1nrm` | z = √‖x‖² | none |

Layer 0 ships in v1 — not for its own sake, but because it is what composes a
3-vector out of stock-shaped calls on its components (see §Vector handles),
and because a caller holding only plain arrays should not have to reach for
Layer 1 at all.

**Counting:** 10 Layer-1 bodies + 6 Layer-0 bodies + 3 forks = 19 bodies;
× 4 precisions = 76 symbols. Forks count once because both clauses share one
body.

**Not in v1:** element-wise divide, a constant 3-vector operand class (the
`u v t` block of the rejected letter grammar — a single `n = 1` vector is too
much machinery for the value it adds, and a caller who needs one wraps a
length-1 array itself), conjugation, global reductions, and any shape needing
a fourth component.

## Design principles

1. **Generic, not per-precision.** One generic body per family, four
   instantiations. Precision lives only in the symbol prefix (`s3`, `d3`,
   `c3`, `z3` and `s1`…`z1`), exactly as `saxpy`/`daxpy`/`caxpy`/`zaxpy`
   differ. *How* one body becomes four precisions (preprocessor, template,
   …) is an implementation question — see the `docs/implementation/`
   documents.
2. **The name is a label; the formula is the table row.** The earlier
   principle "the name is the formula" — a fixed one-letter grammar in which
   a symbol parses as its own mathematics — is **withdrawn** (§Corrections
   21). It failed on its own terms: a letter grammar has a large typo
   surface, and the failure mode is not a compile error but a *valid but
   different* kernel name, which nothing catches. Reading a symbol told you
   the mathematics, but nobody could write it reliably, and the writing was
   the actual work.

   What replaces it is a smaller set of rules, all of them checkable:

   | part | means | precedent |
   |---|---|---|
   | `s` `d` `c` `z` | precision | stock BLAS |
   | `1` `3` | operand width | `?gemm3m` is the only digit BLAS uses, and never there |
   | `ax` `axpby` | a fused scalar multiply, `axpy`-style | stock |
   | `cross` `dot` `had` `norm` | the element operation | stock `dot` |
   | juxtaposition | composition, **inner op first** — `crossdot` is `(x∧y)·z` | |
   | `_` | fork separator, one clause per output | |

   Rules: no domain names, no implementation names (no "fma" anywhere — FMA
   is a property of the compiled inner loop, not of the API); a name is
   never composed into another name; and **no symbol may collide with stock
   BLAS**, since the point is to land inside `libblas` and C has no
   overloading to fall back on.
3. **Pure-first, and in-place is not a kernel.** Inputs are `const`,
   never modified. The output is always the last argument, a handle, and
   write-only. Overwriting an input is done by passing its handle
   again as the output — a call-site spelling, not a sibling routine.
   (BLAS's 1979 in-place convention exists because BLAS passes *pointers*:
   a pure form then needs a temporary, and the temporary costs a full
   extra pass. With handles the temporary is free and the traffic is
   identical, so the distinction has nothing left to buy.) Shared
   operands are why inputs must never be destroyed silently — a caller
   that wants an overwrite asks for it.
4. **Zero reductions in v1.** Every kernel is a single streaming pass: no
   partial sums, no thread sync, no atomics. A global sum, if ever needed,
   is a pointwise kernel + a stock `ddot` on its result. This is what makes
   SMP free: a split on `n` has no combine step, so threading costs nothing
   in the kernel body (see §Coding principles).
5. **FMA-shaped where it is free — never at the API's expense.** A multiply
   that feeds an add is written `fma(a,b,c)` so the add rides the FMA unit;
   intermediates (a cross feeding a contraction) stay in registers. But
   **FMA shape never shapes the API.** An earlier draft argued that a pure
   scale was illegitimate because it has no add to fuse into, and concluded
   the scale kernel should carry a vector offset — and then that a scalar
   offset should be its default. Both conclusions were driven by an inner-loop
   property, not by a caller, and they made the API worse: a scale is a
   scale, and it ships as `d3scal`/`d1scal` with no extra term.
6. **No write past `n`, ever.** A kernel writes exactly the `n` elements
   the handle names: a wide unrolled loop plus a scalar tail. There is
   no hidden padding and no "store full registers past the end" fast path
   — see §Vector handles for why that is not implementable safely.

## Public API (wrapper definitions)

No allocation inside the calls, no new library: the symbols land inside
`libblas` next to `saxpy`/`daxpy` and link with `-lblas` alone.

**The two layers have deliberately different argument shapes**, because they
answer different questions.

*Layer 1* — one argument per operand, the operand being a pointer to a
descriptor (§Vector handles). Length and stride travel inside the
descriptor, so neither appears in the signature:

```c
void d3cross    (const v3 *x, const v3 *y, v3 *c);
void d3crossscal(const v3 *x, const v3 *y, T a, v3 *c);
void d3dot      (const v3 *x, const v3 *y, v1 *r);
void d3crossxy_crossxz(const v3 *x, const v3 *y, const v3 *z, v3 *c1, v3 *c2);
```

*Layer 0* — flat stock CBLAS, so a caller who has nothing but plain arrays
needs no glue at all:

```c
void d1axpby(blaslong n, T a, const T *x, blaslong incx,
             T b, const T *y, blaslong incy, T *z, blaslong incz);
```

**Handles pass by pointer, not by value.** Three reasons, in order of weight:

- **ABI stability.** By value, adding a field to `v3` changes the mangled
  signature of every symbol and breaks every already-compiled caller. By
  pointer it changes nothing. This is the same reasoning that fixed `n` at a
  fixed width (§Corrections 18), one level up.
- **Immutability.** `const v3 *` means the kernel provably cannot reseat
  `x`, `y` or `z`: the handle the caller passed is the handle that got used.
  Const *members* were rejected instead — they would delete struct
  assignment, and the output handle must be writable.
- **Precedent.** CBLAS passes pointers, `const` on inputs, output last.

The per-call cost is O(1) against O(n) work, so this is never a speed
question.

**The output is always the last argument and is always a descriptor** — a
`v3` when the kernel produces a 3-vector, a `v1` when it produces a scalar
array (`d3dot` writes a `v1`; `d3cross` writes a `v3`). Output handles are
non-`const`. In-place is a **call-site spelling, not a kernel**: pass an
input handle again as the output (`d3cross(&x, &y, &x)`). Per-element
aliasing is safe by construction, and the output's length and stride are
taken from whichever operand supplies them when the handles disagree —
checked once at entry.

Argument order is mechanical: operands in order of first appearance in the
mathematics, left to right; scalars appear where their term sits. The c/z
precisions take complex scalars (a pair of reals) in the same slots.
(Per-op enumeration: `docs/kernels/v1set.md` §Argument forms.)

**Index width.** Layer 1 counts in `int64_t` (`blaslong`) throughout,
because the target problem size (N = 10⁸–10¹⁰) exceeds the 2³¹ reach of
BLAS's own `blasint` — at N = 10¹⁰ a 32-bit index silently wraps, which is a
correctness bug, not a portability nit. Layer 0 follows the host BLAS
convention (`blasint`) so that its calls remain drop-in compatible with
stock BLAS symbols; a Layer-0 call on a run longer than 2³¹−1 is out of
contract and must be split by the caller into ≤2³¹-element chunks. That
asymmetry is deliberate and is the only one in the API.

**Symbol form:** `<precision><width><op>`, precision ∈ {s, d, c, z}, width
∈ {1, 3}. Trailing `_` for the Fortran entry, no underscore for the C name,
exactly as stock BLAS relates `daxpy_`/`daxpy`. **No symbol may equal a
stock one** — see §Design principles 2 for why `d1*` rather than `d*`.

### Vector handles (layout & constructors)

**Layout: SoA, not AoS.** A 3-vector is three parallel component arrays —
kernels sweep components, so each component array must be unit stride in the
dense case. The components need not be adjacent to *each other*; what makes
them one vector is the descriptor. There are two, and they are the same idea
at two widths:

```c
typedef struct { T *x, *y, *z; blaslong n, inc, cinc; } v3;  /* 48 B, s3v/d3v/c3v/z3v */
typedef struct { T *p;            blaslong n, inc;      } v1;  /* 24 B, s1v/d1v/c1v/z1v */
```

- `n` — length in elements. `inc` — element stride (`inc = 1` is dense).
- `cinc` — **`3 * inc`, stored rather than derived.** The stride of one
  *component* of this vector is `cinc`, not `inc`, and that is the single
  easiest field in the API to get wrong by hand. Storing it means callers
  never compute it, and the value cannot drift out of sync with `inc`
  because `create`/`wrap` set both.

All three integer fields are `blaslong`: the descriptor is public ABI, so
`long` is not admissible (8 bytes on LP64, 4 on ILP32, and the library must
agree with the caller's compiler about the layout). `inc` ships populated in
v1 — strides are supported now, not reserved (§Corrections 6). There is
deliberately **no `pitch` field**: the distance between components is an
allocation-time fact no kernel has use for.

**What a `v3` means depends on its stride — and that is the whole of the
layer split at the type level.**

| handle | meaning | accepted by |
|---|---|---|
| `v3` with `inc == 1` | a field of 3-vectors | Layer 1 |
| `v3` with `inc == k` | three strided scalar arrays, one per component | Layer 0 only |
| `v1` | one scalar array | Layer 0 |

Layer 1 **requires `inc == 1`**; there is no unit-stride-only caveat, it is
part of what a 3-vector means. Layer 0 accepts any `inc`. So the types
compose in one direction only, and a `v3` built by `v3_wrap_component` (which
sets `inc = cinc` of its parent) is a Layer-0 handle and nothing else.

**Constructors** (plain C, one header; all are views that own nothing):

```c
v3      *v3_create(blaslong n);                              /* the only malloc in the API */
v3      *v3_wrap(v3 *out, T *base, blaslong n, blaslong inc);
v3      *v3_wrap_component(v3 *out, const v3 *v, int k);
v1       v3_component(const v3 *v, int k);
blaslong  v3_pitch(blaslong n);
void     v3_destroy(v3 *v);
```

- **`v3_create(n)`** — build from nothing: one block of `v3_pitch(n)`
  elements carved into three components, `inc = 1`. Caller frees with
  `v3_destroy`. This is the **only** function in the API that calls an
  allocator, which is what makes the backend swappable — see below.
- **`v3_wrap(out, base, n, inc)`** — aggregate a vector from an existing
  interleaved 3N block, zero-copy, arbitrary stride. No allocation, so no
  matching destroy; the caller owns `base`.
- **`v3_wrap_component(out, v, k)`** — one component of `v`, as a `v3` whose
  three pointers all name the same element stream (`x + k`, `y + k`,
  `z + k`) and whose `inc` is the parent's `cinc`. For Layer-1-shaped
  operations on a component.
- **`v3_component(v, k)`** — one component as a `v1`, for Layer 0. This is
  the ordinary path:

  ```c
  v1 Bx = v3_component(&B, 0);   /* p = B.x, n = B.n, inc = B.cinc */
  ```

- **`v3_pitch(n)`** — the rounded pitch, `((3n + 7) / 8) * 8`. **Public**,
  because a caller who allocates with `malloc` and wraps the result must
  match it.

**Pitch is a fixed 8, forever, and that is deliberate.** The obvious design —
round up to the build's widest vector register — makes the pitch a function
of the build's SIMD width, which means a buffer wrapped by one build can be
misaligned in another, and `v3_pitch` can even return *fewer* bytes than the
buffer was allocated with. Pinning it to 8 turns it into a stable ABI fact:
alignment is never worse than 8 elements (64 B for `double`, one cache line),
the waste is bounded at 7 elements total, and the unrolled width becomes a
private choice inside the core rather than something a caller's allocation
depends on.

**The allocator lives in exactly one function, and it is swappable.** The
kernels are pure address arithmetic; `v3_create` is the only thing that calls
`malloc`. When the backend changes — `malloc_shared` for a host/device
unified allocation, or a platform allocator, same as cuBLAS — one function
changes and nothing else notices, because the descriptor holds pointers and
no kernel ever dereferences through anything but those. A GPU-resident field
wraps the same way and its component handles carry the residency with them.

**A `wrap`-ed vector gets whatever alignment the caller's pointers have**, so
the scalar tail path is always present, and **no hidden padding ever
exists.** The earlier scheme (`create` carves a per-element padded slot so
kernels may store full registers past `n`) is **withdrawn** (§Corrections
17). It was unsound: a `wrap`-ed vector is caller memory and may be a slice
of a larger array, so a store past `n` corrupts caller data; a slice of a
`create`-ed vector has no slack by construction; and `inc` makes it worse,
since with `inc = 3` the end of a component array is nowhere near the end of
the block. An entry point also cannot tell a padded vector from a wrapped
one, so no alignment test can select a fast path — alignment says nothing
about slack. Cost of the fix is one scalar tail loop per component,
invisible on a memory-bound kernel, which is the only kind in v1. The
invariant is checked directly instead: guard canaries after every buffer
(Acceptance criteria 3). The inter-component pitch is *not* slack — it is
never written and does not relax this rule.

**Components are reachable, and deliberately so.** The earlier rule — a
3-vector is a unit of algebra and the caller's vocabulary cannot reach inside
one (§Corrections 19) — is **withdrawn** (§Corrections 22). It was coherent
and it cost more than it bought: private components meant there was *no*
Layer-0 composition of a 3-vector at all, and the four componentwise kernels
in the earlier v1 set existed only because that composition was impossible.

What replaces it is that reaching a component is a **constructor**, never a
kernel, and it returns a handle rather than a value:

```c
v1 Bx = v3_component(&B, 0);
d1axpby(B.n, +1, v3_component(&v,2).p, v.cinc, -1, Bx.p, Bx.inc, F.x, F.cinc);
fwrite(Bx.p, B.n, sizeof *Bx.p, fp);          /* I/O on the caller's own memory */
```

So the 3-vector layer and the BLAS layer compose in both directions, and the
MHD loop can be written either way — pure vector algebra, or stock-shaped
calls on components when the components are what you already have:

```c
/* div B, entirely Layer 0 */
v1 Bx = v3_component(&B,0), By = v3_component(&B,1), Bz = v3_component(&B,2);
v1 vx = v3_component(&v,0), vy = v3_component(&v,1), vz = v3_component(&v,2);
d1axpby(B.n, +1, vz.p, vz.inc, -1, By.p, By.inc, F.x, F.cinc);   /* F += v_z·B_y − v_y·B_z */
d1axpby(B.n, -1, vx.p, vx.inc, +1, Bz.p, Bz.inc, F.x, F.cinc);   /* F += −v_x·B_z + v_z·B_x */
```

The members are public, so `B->y` is also legal and is a legitimate escape
hatch for host code that already knows its layout — but it comes with the
stride trap, since a component's stride is `B->cinc`, not `B->inc`, and
nothing type-checks that. `v3_component` is the spelling that cannot be got
wrong.

**`*_create` is for scratch and intermediates.** Anything whose numbers the
caller wants, it wraps on its own buffers first. That keeps the
created-vs-wrapped asymmetry gone: the library never allocates storage whose
contents the caller must read back through an accessor it does not have.

Projection (`d3cross(x, y, c)` with a unit `y`) is admitted on its merits as
an operation, not as a back door to component access — and now that it *is*
an access, it has nothing left to smuggle.

**Wrapper (entry) behavior.** The public entry handles the edge cases —
`n ≤ 0` → return; for kernels whose *entire* RHS is `a·(…)`, `a = 0` → store
zeros and return (kernels with extra terms get no such shortcut) — checks
that all operand handles agree on `n` and `inc`, and then runs the kernel:
one streaming pass over `n` elements. v1: single-threaded. The mechanics of
entry, kernel body, and build registration are per target library — see
`docs/implementation/`.

## Coding principles (memory model)

- The only allocation is explicit and lives in exactly one function:
  `v3_create`/`v3_destroy`, caller-side (on GPU backends through the
  platform's allocator — same as cuBLAS). Everything else is a view:
  descriptors own no state, same role as `lda` in `dgemm`. `v3_create`
  yields scratch; results a caller wants to read come back in buffers the
  caller wrapped (see §Vector handles).
- Inputs are `const`. The output is always the last argument, a descriptor
  (`v3` for a 3-vector, `v1` for a scalar array), and write-only.
  **In-place is a call-site spelling, not a kernel**: pass an input handle
  again as the output (`d3cross(&X, &Y, &X)`). Traffic is identical either
  way — a pure form's write and an in-place form's read are the same word —
  and per-element aliasing is safe by construction. BLAS needs separate
  in-place routines because it passes *pointers*, where a pure form needs a
  temporary and the temporary costs a full extra pass; descriptors make the
  temporary free, so the distinction dissolves.
- Single streaming pass; no partial sums, no thread sync, no atomics (v1).
- **SMP is in v1, and it is the biggest lever in the spec.** These kernels are
  bandwidth-bound, and one core pulls on the order of 10–20 GB/s against a
  socket's 200–400. A single-threaded v3blas therefore delivers a few percent
  of what the hardware can do, and no amount of inner-loop tuning closes that
  gap. Everything about the kernel design is already SMP-ready: principle 4
  gives zero reductions, so a split on `n` has no combine step, no partial
  sums, no atomics, no sync inside the kernel — and forks are no different,
  since each output element depends on one element of each input.
- **The parallelism axis is the element index, never the component.** Three
  components is not three-way parallelism: splitting by component moves the
  same bytes to three threads and caps the thread count at 3 on a machine
  with 16 cores. It is also *impossible* for the fused kernels —
  `d3crossdot` holds a whole cross in registers, and a component-split thread
  cannot compute one. Element-splitting has neither problem. What 3
  components do cost is three concurrent streams per core instead of one, so
  3× the prefetcher/TLB pressure — mild on modern cores, and an argument for
  a modest unroll.
- **Threading lives in the interface layer only, and `_k` kernels stay
  single-threaded forever.** An entry splits `n` into contiguous chunks — one
  per thread, never interleaved, so each thread's prefetcher gets a linear
  stream — and calls the `_k` kernel per range. A static split with no work
  stealing suffices: at N = 10⁸ across 16 threads each chunk is ~150 MB, so
  imbalance is noise. Below a per-op element threshold the entry runs
  single-threaded, because fork overhead exceeds the work.
- **Both layers, one mechanism.** Layer 1 splits the handle's `n`; Layer 0
  splits its `n` argument. Same code path, same threshold policy.
- **Thread count comes from the host library, never from us.** v3blas
  introduces no knob of its own; it reads the count the library already
  exposes (`openblas_set_num_threads` / `OPENBLAS_NUM_THREADS`). A second
  thread-count control next to the first one is a wart that never goes away,
  and the whole premise is landing inside `libblas`.
- **Known cost:** thread *k*'s range does not start on a vector-width
  boundary, so each thread pays a misaligned prologue. Invisible on a
  memory-bound kernel.
- **FMA shape, where it is free.** A multiply that feeds an add in the
  mathematics is written `fma(a,b,c)`, so the add rides the FMA unit and the
  multiply is free; sums of products accumulate through it. Pure scales,
  product chains and division operands stay plain mul/div. This is a
  property of the compiled inner loop, **not a reason to shape the API**:
  `d3scal` and `d1scal` are bare multiplies because a scale is a scale, and
  an earlier draft of this spec tried to argue the opposite and was wrong.
- Intermediates stay in registers (e.g. a cross feeding a contraction).
- v1 complex: flat interleaved re/im elements; **no conjugation anywhere**
  (pointwise self-products in c/z precisions are bilinear, not Hermitian).
- **Strides.** `inc` is in the handle for both layers. Layer 1 requires
  `inc == 1` — that is what a 3-vector means (§Vector handles) — and a call
  with `inc != 1` is rejected rather than silently mis-executed. Layer 0
  accepts any `inc`, and its entry branches exactly like `daxpy`: `inc == 1`
  → the wide unrolled body, otherwise a scalar stepping loop. **Invariant:
  `inc` must never reach the inner loop**; a contiguous core that takes `inc`
  as a parameter would cost more than every handle indirection combined.
  Vectorized strided execution is deferred.

## Layer split (why two layers, one patch)

Layer 1 (v3, width 3) holds the ops that use the 3-vector *structure*: cross
and dot contractions need the three-component grouping. Layer 0 (plain 1D,
width 1) holds the structure-blind half of the algebra — useful to any code
doing element-wise work, and what makes field coefficients and
mixed-component products expressible. Layer-1 kernels are self-contained
fused passes (register fusion) and do **not** call Layer-0 kernels: no
internal ordering dependency, no patch split. The two layer definitions and
their counts: `docs/kernels/v1set.md`.

**The split is a theorem, not a preference.** Layer-0 ops are
*component-separable* — the output's `c` component is a function of the input
`c` components alone — and any composition of component-separable ops stays
component-separable. So a formula is reachable from Layer 0 **iff** each
output component depends only on the corresponding input components. Cross
and dot violate exactly that, which is why they, and only they, require
Layer 1. Routing `z = x∧y` through Layer 0 costs ~3× the memory traffic
*and* two temporaries of length N; `r = x·y` costs the same. The six-call
div-B expansion in `docs/kernels/v1set.md` §Composed operations is that
theorem written out at N.

**But the theorem does not mean the layers are closed off from each other,
and v1 ships both.** A component view (`v3_component`) plus Layer 0 *can*
spell anything the theorem says it can — it is just slower, by the factor the
theorem names. So the split is a statement about the cheapest spelling, not
about what is expressible, and `v3_component` is the seam where the two meet.
This is why Layer 0 is in v1 rather than deferred: without it the "cheapest
spelling" argument has no fallback for a caller who already holds components.

A field coefficient is a cross-layer *shape*, not a component reach: `d3had`
takes a scalar array against a vector.

## Deferred (stash / TODO)

- **Forks beyond two outputs.** v1 ships the two in §The v1 kernel set. Still
  deferred: three-way and wider fans, and any fork mixing a 3-vector output
  with more than one scalar-array output. The cap is register pressure, not
  imagination — two cross clauses need six live component expressions, a
  three-way needs nine, and x86-64 has sixteen vector registers.
- **Strided kernels beyond the scalar stepping loop.** `inc` is in v1
  (§Vector handles), so every kernel accepts a stride and the entry branches
  exactly like `daxpy`: `inc == 1` → the wide loop, else scalar stepping.
  What is deferred is *vectorized* strided execution (gather/shuffle paths
  for `inc = 2, 3, 4`) — correctness first, throughput for strides later. A
  single `inc` expresses 1D arithmetic progressions only, so a 2D interior
  of per-row-halo storage is a set of calls (one per row or per column)
  rather than one call; staggered grids need nothing extra, since stagger is
  positional and each component stays dense in its own array.
- **Element-wise divide, and the other Layer-0 shape gaps.** v1's Layer 0 is
  a working set, not a complete element-wise algebra: divide, `min`/`max`,
  and anything needing a reciprocal table are all later patches.
- **A constant 3-vector operand class.** The rejected letter grammar reserved
  `u v t` for single `n = 1` vectors. Deferred to v2 as too much machinery
  for the value it adds — a caller who needs one wraps a length-1 array with
  `v3_wrap`, which costs a line.
- **SMP threading: promoted into v1.** Both layers, interface-layer only,
  reusing the host library's thread count — see §Coding principles. What
  remains undecided is the per-op element threshold below which an entry
  stays single-threaded, which is a measured quantity per op and per build,
  not a spec constant.
- **Vectorized strided execution.** `inc` is in v1 and Layer 0 accepts any
  `inc`, but the wide core runs at `inc == 1` only; other strides take the
  scalar stepping loop. With SMP on top, the strided path can be threaded
  too (split on `n` as usual) — but it is not vectorized, so it will not
  reach the same bandwidth, and for a bandwidth-bound library that gap is
  the reason it stays last.
- **Complex conjugation conventions** (v1: no conjugation anywhere)
- **Reductions** (global dot/norm; compose with stock `ddot` meanwhile)
- **A width other than 1 or 3.** The width is in the symbol name, which is a
  promise that 1 and 3 are the only widths. A 4-vector (spacetime EM) would
  need `d4*` and a third handle type.
- **Fortran operator module** (`E ^ B`, `E * B` from the old doc) —
  deferred past v1.

## Acceptance criteria

1. `nm -D` on the built library shows exactly the v1 kernel set of §The v1
   kernel set — 19 bodies × 4 precisions = 76 symbols, no more — plus the
   descriptor constructors (see §Vector handles). `saxpy`/`daxpy` remain for
   the tests' baseline
   comparisons. A symbol that appears here and not in the spec is a defect,
   and so is its absence. **No v3blas symbol may equal a stock BLAS symbol**
   — checkable, and the reason Layer 0 is spelled `d1*`.
2. A short C test compiles and links against **the build under test** and
   nothing else beyond `-lm`. Until some BLAS ships these symbols, "the
   build under test" is the local OpenBLAS build, addressed by path or
   `rpath` — *not* by a bare `-lblas`, which on most distributions
   resolves to the system provider (reference/netlib BLAS) and tests the
   wrong library. The test itself stays BLAS-ABI-only, so that the day a
   real BLAS ships the symbols the same source links unchanged.
3. Tests pass for all four precisions; random and edge `N` (0, 1, odd,
   unaligned base pointers, non-unit `inc`); and **no test writes outside
   its buffers** — every buffer is allocated at exact size with guard
   canaries on both sides, checked after every kernel call. That is how
   principle 6 is enforced rather than merely asserted. Every identity runs
   at **every thread count the host library offers**, not just 1: a split on
   `n` must not change a single output bit, and canaries are checked after
   every threaded call too.
4. Bandwidth, as a measurable target rather than the word "roofline". At
   `N` large enough that the working set exceeds LLC (≥ 2²⁶ elements per
   component) each kernel sustains a **recorded fraction of a copy loop
   over the same footprint**, with that fraction stated per implementation
   and re-measured whenever a kernel or a build target changes. The
   measurement, not the number, is the criterion: the spec cannot fix a
   percentage, because the achievable rate is a property of the target's
   vector width and code generation, not of the kernel set. What *is*
   fixed here: no kernel may be slower than the equivalent scalar-C
   reference at any `N`, and multi-output kernels must move strictly
   fewer bytes than the sequence of kernels they replace, by the amount
   the spec claims. At small `N` (≤ 2¹⁶) no kernel is slower than that
   reference either — there they are latency- and dispatch-bound, and
   "roofline" is meaningless.
5. The kernel-set identities pass in all four precisions: every entry in
   `docs/kernels/v1set.md` §Tests holds for every kernel of §The v1 kernel
   set, including the structural ones (guard canaries, stride transparency,
   purity, edge `n`, bilinear-not-Hermitian complex). Two more, specific to
   the handle design: **`v3_pitch` is stable** — the same `n` yields the same
   pitch regardless of build or vector width — and **a Layer-1 call with
   `inc != 1` is rejected**, not silently mis-executed.

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
  the same kernel spec (`docs/kernels/v1set.md`) onto one library's genericity
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
6. ~~Stride support removed~~ — **reversed.** Deferring `inc` to v2 was an
   ABI break: the descriptor is public, so a field added later is a layout
   change every already-compiled caller would miss. `inc` ships in v1, and
   the entry branches like `daxpy` (`inc == 1` → wide loop, else scalar
   stepping). Only *vectorized* strided execution is deferred.
7. ~~In-place rejected as name-visible siblings~~ — **superseded twice.**
   First the siblings were dropped as pointless (traffic is identical, and
   a descriptor makes the temporary free, so `d3cxy(&X,&Y,&X)` covers
   both). Then the output letter left the name entirely, which makes an
   in-place *kernel* structurally unrepresentable: there is nothing in a
   name left to distinguish it.
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
16. **Layout decision: SoA behind a single-pointer descriptor.**
    Component arrays are contiguous (kernels sweep components; AoS
    stride-3 lanes would tax every SIMD loop), but a vector is still
    one argument: a pointer to a per-precision view
    `{x, y, z, n, inc}` with `create`/`wrap`/`slice`/`free`
    constructors. Layer 1 drops the `n` argument (length lives in the
    descriptor); Layer 0 keeps the flat BLAS shape; internal `_k`
    kernels stay flat-pointer — entries unwrap descriptors. Supersedes
    the earlier "flat pointers, struct is sugar" API stance for Layer 1.
17. **Hidden tail padding withdrawn, inter-component pitch added.** The
    "quaternion" padded slot per element (item 16) is unsound: a `wrap`-ed
    vector is caller memory, a `slice` near a parent's end has no slack by
    construction, and `inc` puts the tail nowhere near the block's end. An
    entry also cannot tell a padded vector from a wrapped one, so no
    alignment test can select the fast path. Every kernel writes exactly
    `n` elements: wide body plus scalar tail (principle 6), enforced by
    guard canaries in the tests. What replaced it for alignment is a
    different object: **three** components, not four, at a pitch rounded
    up to the vector width. A fourth component buys no alignment, since
    `y` lands at `base + N` either way — the constraint is `N`, not the
    component count. The pitch is never written and stays inside the
    allocator.
18. **`long n` → fixed width.** The descriptor is public ABI: `long` is 8
    bytes on LP64 and 4 on ILP32, so the library and the caller's compiler
    can disagree on the layout. Layer 1 is `int64_t` throughout (the
    target N = 10⁸–10¹⁰ exceeds the 2³¹ reach of BLAS's own `blasint`);
    Layer 0 keeps `blasint` so its calls stay drop-in compatible with
    stock BLAS symbols.
19. **Components are private to the caller.** A 3-vector is a unit of
    algebra, not three loose arrays. Kernels mix components internally
    (that is what cross and dot *are*); the caller's vocabulary has no
    way to reach inside one. Results come back in buffers the caller
    wrapped, so I/O and per-component reductions need no getter — and
    `*_create` is reserved for scratch, which removes the
    created-vs-wrapped asymmetry entirely. `xdv` with `v = (1,0,0)` is
    not an extraction back door.
20. **Naming system re-founded** (`docs/kernels/letterspace.md`, a
    sibling proposal file — `spellform.md` is left intact as the current
    scheme). The output letter and the `=` glyph are gone: a name is now
    the bare formula (`cxy`, not `zexvy`), the output is the last
    argument, and the letter space is blocked into uniform blocks —
    `a b e` scalars, `t u v` constant 3-vectors, `q r s` scalar arrays,
    `w x y z` 3-vectors, glyphs `c d m o p` and fork separator `f`.
21. **Letter grammar withdrawn — superseded by item 22.** The letter space
    of items 4, 11 and 20 is **rejected**. It was consistently re-derived
    and consistently mis-spelled, by both author and reviewer: the failure
    mode is the problem, not carelessness. A letter grammar has a typo
    surface proportional to its size, and a typo produces a *valid but
    different* kernel name (`trrmss` for `terrmss`) rather than a compile
    error. Reading a symbol told you the mathematics; writing one reliably
    did not work, and writing was the actual job. Kept for the record in
    `docs/kernels/letterspace.md`; nothing in the spec depends on it.
22. **Naming re-founded on BLAS convention** (`docs/kernels/v1set.md`).
    Design principle 2 is replaced: *the name is a label, the formula is
    the table row, names follow BLAS conventions and nothing else.* The
    grammar is now five checkable rules — precision letter, width digit,
    `ax`/`axpby` for a fused scalar multiply, stock stems for the element
    operations, juxtaposition for composition with the **inner op first**
    (`d3crossdot` is `(x∧y)·z`), `_` for forks with one clause per output.
    Composites that collapse get no symbol: `d3crossdot` also spells
    `x·(y∧z)` by commutativity. A new rule falls out of the old library:
    **no symbol may equal a stock BLAS symbol**, which is why Layer 0 is
    `d1*` and not `d*` (`dscal`, `daxpy`, `daxpby` all exist with
    different signatures, and stock `dnrm2` is a *reduction* where ours is
    elementwise — C has no overloading, so the collision would link cleanly
    and run the wrong function).
23. **The constant 3-vector class is out of v1.** The `u v t` block (a
    single `n = 1` 3-vector operand) is deferred to v2: too much machinery
    for the value it adds, and a caller who wants one wraps a length-1 array.
24. **Layer 0 promoted into v1, and both layers re-signed.** Layer 0 is no
    longer deferred — with components reachable (§Corrections 22 of the
    handle story, below) the two layers compose in both directions, and a
    caller with plain arrays should not have to reach for Layer 1. Layer 1
    takes `const v3 *` handles **by pointer**; Layer 0 takes **flat stock
    CBLAS** arguments, so a real BLAS caller needs no glue at all. Handles
    by value were rejected: they would make a future field addition an ABI
    break for every symbol, and `const v3 *` also makes reseating impossible
    by construction. Output handle type is `v3` for a 3-vector result, `v1`
    for a scalar-array result. The set is now 19 bodies / 76 symbols.
     Register pressure caps forks at two outputs (§The v1 kernel set).
25. **`cinc` added to the handle; pitch pinned to 8.** A component's stride
    is `3 * inc`, and it is the field easiest to get wrong by hand, so it is
    *stored* rather than derived. The inter-component pitch was also
    re-decided: rounding to the build's widest vector register makes the
    pitch a function of the build, so a buffer wrapped by one build can be
    misaligned in another and `v3_pitch` can return fewer bytes than the
    buffer holds. Fixed 8 makes it a stable ABI fact and demotes the unrolled
    width to a private choice inside the core. `v3_pitch` is **public**, so a
    caller who `malloc`s and wraps matches it.
26. **Components are reachable — item 19 reversed.** "A 3-vector is a unit of
    algebra and the caller's vocabulary cannot reach inside one" is
    **withdrawn**. It was coherent and it cost more than it bought: private
    components meant there was no Layer-0 composition of a 3-vector at all,
    and four of the thirteen v1 bodies existed only to work around that.
    Reaching a component is now a **constructor** (`v3_component` → a `v1`,
    `v3_wrap_component` → a `v3`), never a kernel and never a getter, and it
    returns a handle rather than a value. I/O and per-component reductions
    still need no API of ours — they are stock calls on the caller's own
    memory. The members stay public, so `B->y` is a legal escape hatch, with
    the stride trap (`B->cinc`, not `B->inc`) that `v3_component` exists to
    remove.
27. **Parallelism axis settled: the element index, never the component.**
    Three components is not three-way parallelism. These kernels are
    bandwidth-bound below arithmetic intensity 1, so component-splitting
    moves the same bytes, caps the thread count at 3, and is outright
    impossible for the fused bodies — `d3crossdot` holds a whole cross in
    registers and a component-split thread cannot form one.
28. **SMP promoted into v1, both layers.** It was deferred twice on the
    grounds that v1 should stay "obviously correct". That was the wrong
    trade: one core pulls ~10–20 GB/s against a socket's 200–400, so a
    single-threaded v3blas delivers a few percent of the hardware's
    capability and no inner-loop tuning closes that gap. The kernel design
    was already SMP-ready — principle 4's zero reductions means a split on
    `n` has no combine step, so there are no atomics, no partial sums and no
    sync in any kernel body, and forks are no exception. Threading therefore
    stays **entirely in the interface layer**: entries split `n` into
    contiguous chunks and call the `_k` kernel per range, and the `_k`
    kernels remain single-threaded forever. Both layers use the same path
    (Layer 1 splits the handle's `n`, Layer 0 its `n` argument).
    **v3blas introduces no thread-count knob of its own** — it reads the one
    the host library already exposes (`openblas_set_num_threads` /
    `OPENBLAS_NUM_THREADS`), because a second control next to the first is a
    wart that never goes away. Criterion 3 now runs every identity at every
    thread count, since a split must not change an output bit.
29. **`d3dotxy_dotxz` added; the fork rule is now uniform.** All three forks
    share the **first** operand and vary the second — `crossxy_crossxz`,
    `crossxy_dotxy`, `dotxy_dotxz`. The new one closes the beachhead: with
    x = ∇∧B, y = ∇∧B, z = B it returns Q₀ and h in one pass, so x and y are
    the same handle, which is legal and free. The corollary is worth
    recording: **the F/S pair provably cannot be a fork**, because those two
    share `B` as the *second* operand of a non-commutative cross. That is a
    property of the mathematics, not a naming choice.

## Open items

### OPEN — carry these into the next session

| # | question | why it matters | where |
|---|---|---|---|
| **A1** | **Is `v3_pitch` really fixed at 8, or should it follow the build after all?** It was pinned to 8 so the pitch is a stable ABI fact independent of SIMD width. Cost: up to 7 wasted elements per allocation, and a 4-wide build gets 8-alignment it cannot use. Confirm this is worth it now that SMP is in — at N=10⁸ the waste is nothing, so the argument may have shifted. | ABI stability vs wasted bandwidth | §Vector handles |
| **A2** | **The per-op single-thread threshold.** Below what element count does an entry stay single-threaded? Deliberately unspecified — it is measured per op per build. Needs a measurement plan, not a number. | Threading overhead vs work | §Coding principles |
| **A3** | **Reuse OpenBLAS's `blas_level1_thread`, or hand-roll a static chunk loop?** Flagged ⚠ unverified against the pin. The contract is fixed either way (threads in the interface, `_k` single-threaded, split on `n`); only the mechanism is open. My lean is hand-rolled, because it makes chunk boundaries and the threshold auditable. | Patch structure | `OpenBLAS.md` §Body rules |
| **A4** | **Is `blas_cpu_number`/`openblas_set_num_threads` the right count to read**, or does the entry need something else (affinity, NUMA node)? | SMP correctness | `OpenBLAS.md` |
| **A5** | **Handle ABI freeze.** `v3` = `{ T *x,*y,*z; blaslong n, inc, cinc; }`, `v1` = `{ T *p; blaslong n, inc; }`. Passing by pointer was chosen *so* a field can be appended later without breaking compiled callers — which only holds if these are treated as frozen. Needs explicit sign-off before the first patch. | Irreversible once shipped | §Vector handles |
| **B1** | **`dnrm2` is the awkward name.** `d1nrm2` is elementwise (writes an array); stock `dnrm2` reduces to a scalar. Distinct symbol so it links, but the same word with opposite contracts in one header is a reading trap. Rename (e.g. `d1sqr`) or accept? | API clarity | §The v1 kernel set |
| **B2** | **Scalar argument position is not uniform.** `d3scal(x, a, z)` follows stock (`dscal`), while `d3axpby(x, a, y, b, z)` follows first-appearance. The real rule is written down but the exception is real. Should `d3scal` be `(x, a, z)` or `(a, x, z)` for internal consistency? | API consistency | `v1set.md` §Argument forms |
| **B3** | **`d3norm2` duplicates `d3dot(x,x)`.** Admitted as 4 symbols for readability. Keep or cut? | Symbol budget | §The v1 kernel set |
| **B4** | **`Q` and `h` need `d3dotxy_dotxz` with x and y being the same handle.** Legal and free, but a fork that takes an aliased pair deserves an explicit test rather than an incidental one — one is in `v1set.md` §Tests. Anything else aliasing-sensitive? | Test coverage | `v1set.md` §Tests |
| **C1** | **No `tests/` directory and no top-level `Makefile` exist.** They are described in prose only, and acceptance criterion 2's `rpath` requirement has no home. This is the biggest gap between the spec and reality. | Blocks acceptance | `OpenBLAS.md` §Host repo layout |
| **C2** | **`subm/openblas/` is never initialized or built.** Every ⚠ unverified mark in `OpenBLAS.md` — including the whole `blas_level1_thread` question A3 — closes only on a real build at `v0.3.34`. | Blocks the patch | `.gitmodules` |
| **C3** | **Allocation policy for `v3_create`.** `malloc` for v1; the shared/device path is designed-for but unspecified. Which backends are actually in scope for v1? | Design intent vs reality | §Vector handles |
| **C4** | **Sub-ranges.** `v3_wrap` takes a base + n; there is no `slice`. Deferred implicitly, never stated. Is a sub-run view (halo interiors) needed in v1, or is `base + k*inc` enough? | API surface | §Vector handles |
| **D1** | **Suppress or mark the superseded scheme files?** `spellform.md` and `letterspace.md` are banner-marked but still tracked. Delete, or keep as design history? | Repo hygiene | `docs/kernels/` |

### Resolved

1. ~~**Primary API language**~~ — **plain C**, the preprocessor is the
   template engine (see `docs/implementation/OpenBLAS.md`); the C++ variant
   lives on in `docs/implementation/cpp.md` for C++-accepting targets.
2. ~~**Submodule pin**~~ — **OpenBLAS, release tag `v0.3.34`** (commit
   `c0827a7`), recorded in `.gitmodules`. Chosen for the `axpby` precedent
   (an extension added to frozen BLAS, which is the whole argument), plain C,
   and a realistic PR path. A tag rather than a branch means the patch is
   re-based when moving to a newer release. **Not yet initialized** — see C2.
3. ~~**v1 SIMD scope**~~ — **plain portable C**, hand-unrolled 4×, no
   intrinsics, no per-arch bodies, no dispatch of our own. A
   per-implementation decision; `OpenBLAS.md` records the consequence, that
   acceptance criterion 4 can no longer fix a bandwidth percentage, because a
   portable build's achievable rate is a property of the target and not of the
   kernel set. Since the pitch is pinned, the unroll width is now free —
   nothing outside `v3_create` depends on it.
4. ~~**Unapproved additions**~~ — **reversed twice.** The structure-blind
   family (Layer 0) is *no longer deferred*: v1 ships both layers, because
   components are now reachable (§Corrections 26) and the two layers are meant
   to compose.
5. ~~**Fortran operator module**~~ — deferred past v1 (§Deferred).
6. ~~**Minimal kernel subset**~~ — **re-opened twice, now 19 bodies / 76
   symbols, both layers**: ten Layer-1 bodies, six Layer-0, three forks.
   Growth is a patch to the relevant layer, never a silent addition.
7. ~~**Threading policy**~~ — **SMP is in v1, both layers.** Element index
   only; interface-layer only; `_k` kernels single-threaded; thread count
   read from the host library, no knob of our own. Threshold left open as A2.
8. ~~**Component privacy**~~ — **reversed** (§Corrections 26). Components are
   reachable through a constructor, never a getter.
9. ~~**Letter grammar**~~ — **rejected** (§Corrections 21–22).
