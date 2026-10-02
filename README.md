# v3blas — element-wise 3-vector and scalar-array kernels for BLAS

Every MHD, CFD and electromagnetics code re-implements the same few-line
element-wise cross / dot / Hadamard on N grid points. The operations are
memory-bound, same roofline class as SAXPY, arithmetic intensity well below 1.
They belong in BLAS, and BLAS implementations can take extensions now.

The MHD time loop this exists for, per grid point:

    F = (∇∧B) ∧ B / μ₀     Lorentz force
    S = E ∧ B / μ₀          Poynting flux
    Q = (∇∧B)·(∇∧B) / σ    Ohmic heating
    h = B·(∇∧B)            helicity

```c
d3crossscal(&curlB, &B, 1/mu0, &F);           /* F = (∇∧B)∧B / μ₀  */
d3crossscal(&E,    &B, 1/mu0, &S);           /* S = (E∧B)   / μ₀  */
d3dotxy_dotxz(&curlB, &curlB, &B, &Q0, &h);   /* Q₀, h — one pass  */
d1axpby(Q0.n, 1/sigma, &Q0.p, Q0.inc, 0, &Q0.p, Q0.inc, &Q.p, Q.inc);
```

---

## Kernel set — 19 bodies, 76 symbols

### Layer 1 — 3-vectors (10)

| symbol | mathematics | |
|---|---|---|
| `d3scal` | z = a·x | |
| `d3axpy` | y = a·x + y | |
| `d3axpby` | z = a·x + b·y | RK/IMEX stages |
| `d3had` | z = x.*y | field coefficient ρ·v |
| `d3cross` | z = x∧y | |
| `d3crossscal` | z = a·(x∧y) | F, S |
| `d3dot` | r = x·y | h, Q |
| `d3sqr` | z = x·x | |
| `d3crossdot` | r = (x∧y)·z | α-effect |
| `d3crosssqr` | r = (x∧y)·(x∧y) | Alfvén speed |

Composites name the inner op first: `crossdot` is `(x∧y)·z`, and it also
spells `x·(y∧z)` by commutativity. `d3crosssqr` is `d3dot` on a cross with
itself. `d3sqr` is `d3dot(x,x)` spelled shorter; the duplication is admitted.

### Forks (3)

Two outputs, one body, one symbol. `_` separates clauses; a clause spells its
operands. All three share the first operand and vary the second.

| symbol | shape |
|---|---|
| `d3crossxy_crossxz` | c₁ = x∧y, c₂ = x∧z — div B |
| `d3crossxy_dotxy` | c = x∧y, r = x·y |
| `d3dotxy_dotxz` | r₁ = x·y, r₂ = x·z — Q, h |

With x = y = ∇∧B and z = B, `d3dotxy_dotxz` gives Q₀ and h in one pass; passing
the same handle twice is legal and free.

F and S share `B` as the *second* operand of a non-commutative cross, so no
fork shape can hold both.

Forks cap at two outputs: a cross has three live component expressions, so two
clauses need six and three would need nine, over x86-64's sixteen vector
registers.

### Layer 0 — scalar arrays (6)

| symbol | mathematics | stock equivalent |
|---|---|---|
| `d1scal` | z = a·x | `dscal` (in place) |
| `d1axpy` | y = a·x + y | `daxpy` (in place) |
| `d1axpby` | z = a·x + b·y | `daxpby` (in place) |
| `d1had` | z = x.*y | none |
| `d1sqr` | z = x·x | `dnrm2` (a reduction) |
| `d1sqrt` | z = √(x·x) | none |

### Composition

A name is never composed into another name. Write two calls and name the
temporary:

```c
v3 *t = v3blas_create(n);
d3cross(&v, &B, t);
d3crosssqr(&curlB, &B, &Q0);
v3blas_destroy(t);
```

---

## Naming

| part | means |
|---|---|
| `s` `d` `c` `z` | precision |
| `1` `3` | operand width |
| `ax` `axpby` | fused scalar multiply |
| `cross` `dot` `had` `sqr` `sqrt` | element operation |
| juxtaposition | composition, inner op first |
| `_` | fork separator |

- **No symbol may equal a stock BLAS symbol.** Hence Layer 0 is `d1*`: stock
  `dscal`, `daxpy`, `daxpby` exist with different signatures and C has no
  overloading, so a collision links cleanly and runs the wrong function.
- **No `nrm` stem.** Stock `dnrm2`/`dnrm` are reductions to a real scalar,
  Hermitian in c/z. Ours would be elementwise and bilinear, so `sqr`/`sqrt`
  claim only what is true.
- No domain names, no implementation names, **no `fma` anywhere**.
- Anything without a precision letter takes the `v3blas_` prefix.
- Trailing `_` for the Fortran entry, none for the C name.

---

## Handles

```c
typedef struct { T *x, *y, *z; blaslong n, inc, cinc; } v3;  /* 48 B */
typedef struct { T *p;            blaslong n, inc;      } v1;  /* 24 B */
```

`cinc` is `3 * inc`, **stored rather than derived** — a component's stride is
`cinc`, not `inc`, and it is the field easiest to get wrong by hand. There is
no `pitch` field: the distance between components is an allocation-time fact no
kernel uses.

```c
v3      *v3blas_create(blaslong n);                    /* the only allocation */
v3      *v3blas_wrap(v3 *out, T *base, blaslong n, blaslong inc);
v3      *v3blas_wrap_component(v3 *out, const v3 *v, int k);
v1       v3blas_component(const v3 *v, int k);
blaslong  v3blas_pitch(blaslong n);                    /* ((3n + 7) / 8) * 8 */
void     v3blas_destroy(v3 *v);
```

`v3blas_create` is for scratch and intermediates; anything the caller reads
back comes in a buffer the caller wrapped. Everything else is a view and owns
nothing.

Pitch is fixed at 8 forever, so it is a stable ABI fact independent of the
build's vector width. `v3blas_pitch` is public so a caller who `malloc`s and
wraps matches it.

Components are reachable through constructors, never kernels:

```c
v1 Bx = v3blas_component(&B, 0);   /* p = B.x, n = B.n, inc = B.cinc */

/* div B, entirely Layer 0 */
d1axpby(B.n, +1, vz.p, vz.inc, -1, By.p, By.inc, F.x, F.cinc);
d1axpby(B.n, -1, vx.p, vx.inc, +1, Bz.p, Bz.inc, F.x, F.cinc);
```

### The layer split

Layer 1 requires `inc == 1`; Layer 0 accepts any `inc`. Composition runs one
way only.

The split is a theorem. Layer-0 ops are component-separable — the output's c
component is a function of the input c components alone — and compositions of
component-separable ops stay component-separable. So a formula is reachable
from Layer 0 **iff** each output component depends only on the corresponding
input components. Cross and dot violate exactly that, which is why they, and
only they, need Layer 1.

The layers are not closed off from each other. A component view plus Layer 0
spells anything the theorem allows, at ~3× the traffic — the six-call div-B
expansion above is one `d3cross` written out. The split is about the cheapest
spelling, not about what is expressible.

---

## C API

Symbols land inside `libblas` beside `saxpy`/`daxpy` and link with `-lblas`
alone.

```c
/* Layer 1 — one argument per operand; n and inc live in the handle */
void d3scal     (const v3 *x, T a, v3 *z);
void d3axpy     (const v3 *x, T a, v3 *y);
void d3axpby    (const v3 *x, T a, const v3 *y, T b, v3 *z);
void d3had      (const v3 *x, const v3 *y, v3 *z);
void d3cross    (const v3 *x, const v3 *y, v3 *c);
void d3crossscal(const v3 *x, const v3 *y, T a, v3 *c);
void d3dot      (const v3 *x, const v3 *y, v1 *r);
void d3sqr      (const v3 *x, v3 *z);
void d3crossdot (const v3 *x, const v3 *y, const v3 *z, v1 *r);
void d3crosssqr (const v3 *x, const v3 *y, v1 *r);

void d3crossxy_crossxz(const v3 *x, const v3 *y, const v3 *z, v3 *c1, v3 *c2);
void d3crossxy_dotxy  (const v3 *x, const v3 *y, v3 *c, v1 *r);
void d3dotxy_dotxz    (const v3 *x, const v3 *y, const v3 *z, v1 *r1, v1 *r2);

/* Layer 0 — flat CBLAS */
void d1scal (blaslong n, T a, const T *x, blaslong incx, T *z, blaslong incz);
void d1axpy (blaslong n, T a, const T *x, blaslong incx, T *y, blaslong incy);
void d1axpby(blaslong n, T a, const T *x, blaslong incx,
             T b, const T *y, blaslong incy, T *z, blaslong incz);
void d1had  (blaslong n, const T *x, blaslong incx,
             const T *y, blaslong incy, T *z, blaslong incz);
void d1sqr  (blaslong n, const T *x, blaslong incx, T *z, blaslong incz);
void d1sqrt (blaslong n, const T *x, blaslong incx, T *z, blaslong incz);
```

- Handles pass by pointer, so appending a field is not an ABI break. Inputs are
  `const`.
- **The output is always the last argument and is always a handle**, typed to
  the result: `v3` for a 3-vector, `v1` for a scalar array. A dot has no three
  components to write, so `d3dot` writes a `v1`.
- **In-place is a call-site spelling**: pass an input handle again as the output
  (`d3cross(&x, &y, &x)`). Per-element aliasing is safe by construction.
- **Argument order**: operands in order of first appearance, left to right;
  scalars sit where their term sits. c/z take complex scalars as a real pair.
- **Index width**: Layer 1 is `int64_t` throughout, because N = 10⁸–10¹⁰
  overflows a 32-bit index. Layer 0 follows the host `blasint` so its calls stay
  drop-in compatible; a Layer-0 call longer than 2³¹−1 must be split by the
  caller.

### Validation

| condition | behaviour |
|---|---|
| `n == 0` | quick return, no error, no writes |
| `n < 0` | parameter error |
| NULL handle or array | parameter error |
| `inc < 1` | parameter error |
| Layer 1 `inc != 1` | parameter error |
| operands disagree on `n`, `inc`, or `cinc != 3 * inc` | parameter error |

Validation completes before any store, so a rejected call leaves the output
untouched. Errors go to the host library's global error status.

`v3blas_create` is the exception: it returns `NULL` on failure rather than
setting the status. `v3blas_destroy(NULL)` is a no-op. Every backend honours
that — return `NULL`, never abort.

Entries are reentrant. Concurrent calls on distinct handles are safe, including
from inside a region the caller already threaded. Two calls writing one output
handle are the caller's data race, as in stock BLAS.

### Versioning

| change | version |
|---|---|
| new kernel body, constructor, or test | minor |
| field appended to `v3`/`v1` | minor (safe only because handles pass by pointer) |
| change to an existing symbol's signature or semantics | major |
| change to a handle's field order, width, or the meaning of `cinc` | major |
| precision added (`c`, `z`) | major — it changes what existing symbols mean |

### Operator headers (optional C++ API)

A second API where every convenience the C API refuses goes:

```cpp
v3<double> F = cross(curlB, B) * (1/mu0);
v1<double> h = dot(B, curlV);
```

It owes the C API nothing backwards: `v3` in place, RAII, temporaries,
strided views, `norm2`/`abs2`/`conj_dot`, throw or `status` instead of a
global. Every C-side restriction must be something C cannot express or a real
ambiguity this layer cannot paper over.

---

## Coding guidelines

- The only allocation is `v3blas_create`/`v3blas_destroy`. Descriptors own no
  state, like `lda` in `dgemm`.
- **One streaming pass, no reductions.** No partial sums, no atomics, no sync
  inside a kernel. A global sum is a pointwise kernel plus a stock `ddot`.
- **No write past `n`, ever.** A wide unrolled loop plus a scalar tail. No
  padding and no "store full registers past the end", because a wrapped vector
  is caller memory and may be a slice of a larger array. The inter-component
  pitch is not slack.
- **`inc` never reaches the inner loop.** The entry branches on it exactly as
  `daxpy` does: `inc == 1` → the wide body, else a scalar stepping loop.
- **SMP is in scope.** Threading lives in the public entry layer only; kernel
  bodies stay single-threaded. An entry splits `n` into contiguous chunks — one
  per thread, never interleaved — and calls the kernel per range. A static
  split suffices; below a per-op element threshold the entry runs
  single-threaded. Both layers use the same path: Layer 1 splits the handle's
  `n`, Layer 0 its `n` argument.
  **The parallelism axis is the element index, never the component.** Splitting
  by component moves the same bytes, caps threads at 3, and is impossible for
  the fused kernels — `d3crossdot` holds a whole cross in registers and a
  component-split thread cannot form one.
  Thread count comes from the host library; v3blas adds no knob of its own.
- **FMA shape is an inner-loop property only.** `fma(a,b,c)` where a multiply
  feeds an add; plain mul/div elsewhere. Intermediates stay in registers. It
  never shapes the API — a scale is a bare multiply.
- **Complex is flat interleaved and bilinear**, with no conjugation anywhere.
- One generic body per family, four instantiations; precision lives only in the
  symbol prefix.

---

## Not in v1

- Forks beyond two outputs, and forks mixing a 3-vector output with more than
  one scalar-array output.
- Vectorized strided execution. `inc` is in the handle, so every kernel accepts
  a stride; only the vectorized path is missing.
- Element-wise divide, `min`/`max`, reciprocal-table ops.
- A constant 3-vector operand class. Wrap a length-1 array instead.
- Conjugation. `d3dot` is bilinear.
- Reductions. Compose with stock `ddot`.
- A width other than 1 or 3.
- A Fortran operator module.

---

## Acceptance criteria

1. `nm -D` shows exactly the 76 symbols plus the constructors, and no v3blas
   symbol equals a stock BLAS symbol.
2. A C test links against the build under test and nothing beyond `-lm` —
   addressed by path or `rpath`, never a bare `-lblas`. The test source stays
   BLAS-ABI-only.
3. All four precisions pass; random and edge `n` (0, 1, odd, unaligned,
   non-unit `inc`); **no write outside any buffer**, with guard canaries on both
   sides of every exact-size buffer checked after every call, at **every thread
   count** — a split on `n` must not move a single output bit, and a chunk
   boundary that overruns is the same canary bug.
4. At `N` past LLC (≥ 2²⁶ elements per component) each kernel sustains a
   **recorded fraction of a copy loop over the same footprint**, restated
   whenever a kernel or build target changes. Fixed here: no kernel is slower
   than the equivalent scalar-C reference at any `n`, and multi-output kernels
   move strictly fewer bytes than the sequence they replace.
5. `v3blas_pitch` is stable across builds, and a Layer-1 call with `inc != 1`
   is rejected rather than mis-executed.

### Identities to test

`≡` algebraic, `≈` within rounding. **s/d only** means false in c/z, where
products are bilinear.

| | |
|---|---|
| cross is antisymmetric | `d3cross(x,y,c)`; `d3cross(y,x,c2)` → `c2 ≡ −c` |
| self-cross vanishes | `d3cross(x,x,c)` → `c ≡ 0` |
| cross scales | `d3cross(x,y,c)` ≡ `d3cross(x,w,c)` |
| dot is symmetric | `d3dot(x,y,r)` ≡ `d3dot(y,x,r)` |
| self-dot non-negative | **s/d only** — `d3sqr(x,z)` → `z[i] ≥ 0` |
| cross ⊥ its factors | **s/d only** — `d3dot(x,c,r)` → `r ≡ 0` after `d3cross(x,y,c)` |
| crosssqr is dot of cross with itself | `d3crosssqr(x,y,r)` ≡ `d3cross(x,y,c)`; `d3dot(c,c,r)` |
| crossdot absorbs a permutation | `d3crossdot(x,y,z,r)` ≡ `d3crossdot(y,x,z,r)` |
| scale by zero | `d3scal(x,0,z)` → `z ≡ 0` |
| scale is the axpby case | `d3scal(x,a,z)` ≡ `d3axpby(x,a,y,1,z)`, `y ≡ z` |
| axpy accumulates | `d3axpy(x,a,y)` ≡ `d3axpby(x,a,y,1,y)` |
| had is commutative | `d3had(x,y,z)` ≡ `d3had(y,x,z)` |
| sqrt is the root of sqr | `d1sqrt(x,z)` ≡ `d1sqr(x,w)`; `z[i] ≈ √w[i]` |
| Layer 0 ≡ Layer 1 at width 1 | `d1axpby(n,a,x,1,b,y,1,z,1)` ≡ `d3axpby` |
| a fork ≡ its two calls | `d3crossxy_crossxz(x,y,z,c1,c2)` ≡ `d3cross(x,y,c1)`; `d3cross(x,z,c2)` |
| clauses are independent | permuting `y`/`z` swaps `r1`/`r2`, changing neither |
| shared operands are read, not consumed | inputs compare equal to their pre-call copies |
| inputs are never written | every input buffer compares equal after the call |
| Layer 1 rejects `inc != 1` | an `inc = 3` handle passed to `d3cross` fails |
| operand disagreement fails | mismatched `n` is a parameter error; `c` is untouched |
| a split changes nothing | all of the above hold at every thread count |
| complex is bilinear | `c3cross`/`c3dot` match the expanded real-pair formula |

---

## Implementations

One working document per target library under `docs/implementation/`, used one
at a time. The kernel set above is the spec; those documents are how to land it
in a particular tree.
