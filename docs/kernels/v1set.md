# The v1 kernel set

The authoritative kernel-level spec: names, mathematics, argument forms,
the fork rule, composed operations, and the identities the tests must check.
Everything structural — handles, layers, principles, acceptance criteria —
lives in `README.md`; this file is the part expected to iterate.

Superseded schemes, kept for the record and referenced by nothing:
`spellform.md` (a frozen 41-body letter spelling) and `letterspace.md` (the
letter grammar, rejected — README §Corrections 21).

## Naming rules

Five rules, all checkable. There is no grammar to parse and no letter space to
remember; the only hard requirement is that **no symbol may equal a stock BLAS
symbol**, because the point is to land inside `libblas` and C has no
overloading.

| part | means | example |
|---|---|---|
| `s` `d` `c` `z` | precision | stock |
| `1` `3` | operand width | `d3cross`, `d1scal` |
| `ax` `axpby` | a fused scalar multiply | stock `axpy`, `axpby` |
| `cross` `dot` `had` `norm` | the element operation | stock `dot` |
| juxtaposition | composition, **inner op first** | `crossdot` is `(x∧y)·z` |
| `_` | fork separator, one clause per output | `crossxy_dotxy` |

Two consequences worth stating because they save symbols:

- **Composites that collapse do not get symbols.** `d3crossdot` also spells
  `x·(y∧z)` because dot is commutative — permute the arguments. There is no
  mirror kernel for cross, which is not commutative, and no
  `crossdotdotxy` family.
- **Chained forms are not new mathematics.** `d3crossnorm2` is
  `(x∧y)·(x∧y)` — `d3dot` applied to a cross with itself. It exists because
  the intermediate stays in registers, not because the formula is new.
  `d3norm2` is `d3dot(x,x)` spelled shorter; it ships because it reads better
  at the call site, and the duplication is admitted.

No domain names (`lorentz`, `poynting`), no implementation names (`fma`
anywhere — FMA is a property of the compiled inner loop, not of the API).

## Layer 1 — 3-vectors

Ten bodies. Width is fixed at 3 and named in the symbol; there is no length
argument, it lives in the handle (`README.md` §Vector handles).

| symbol | mathematics | output handle | earns its place |
|---|---|---|---|
| `d3scal` | z = a·x | `v3` | |
| `d3axpy` | y = a·x + y | `v3` | accumulate form of the above |
| `d3axpby` | z = a·x + b·y | `v3` | every RK/IMEX stage of the time loop |
| `d3had` | z = x.*y | `v3` | field coefficient, ρ·v |
| `d3cross` | z = x∧y | `v3` | the cross itself |
| `d3crossscal` | z = a·(x∧y) | `v3` | **F, S** — scalar folded in, cross never stored |
| `d3dot` | r = x·y | `v1` | **h, Q** |
| `d3norm2` | z = x·x | `v3` | |
| `d3crossdot` | r = (x∧y)·z | `v1` | α-effect, cross stays in registers |
| `d3crossnorm2` | r = ‖x∧y‖² | `v1` | Alfvén speed, cross stays in registers |

**The output handle type is not decorative.** `d3dot` writes a scalar array,
so its output is a `v1`, not a `v3`. This is forced, not chosen: a dot has no
3 components to write. The general rule is *the output is the last argument,
and its handle type matches what the kernel produces* — `v3` for a 3-vector,
`v1` for a scalar array.

## Layer 0 — scalar arrays

Six bodies, flat stock CBLAS arguments. Four exist in BLAS under a similar
name and **must not reuse the symbol** — `dscal`, `daxpy`, `daxpby` all have
different signatures (in place, ours out of place) and stock `dnrm2` is a
*reduction* to a scalar where ours is elementwise. C would link one of these
cleanly and run the wrong function.

| symbol | mathematics | stock equivalent |
|---|---|---|
| `d1scal` | z = a·x | `dscal` (in place) |
| `d1axpy` | y = a·x + y | `daxpy` (in place) |
| `d1axpby` | z = a·x + b·y | OpenBLAS `daxpby` (in place) |
| `d1had` | z = x.*y | none |
| `d1nrm2` | z = ‖x‖² | `dnrm2` (a *reduction* to a scalar) |
| `d1nrm` | z = √‖x‖² | none |

The `1` is not decorative either: it says the operand width, exactly as `3`
does, and nothing in stock BLAS begins `<precision>1`.

## Forks

Two outputs, one body, one symbol. `_` separates clauses and a clause spells
the operands it uses, so a shared operand is written **once**:

| symbol | replaces | outputs |
|---|---|---|
| `d3crossxy_crossxz` | `d3cross` + `d3cross` | c₁ = x∧y, c₂ = x∧z — the div-B shape (v∧B, v∧E) |
| `d3crossxy_dotxy` | `d3cross` + `d3dot` | c = x∧y, r = x·y |
| `d3dotxy_dotxz` | `d3dot` + `d3dot` | r₁ = x·y, r₂ = x·z — **Q, h** |

A fork is one body and one symbol, and it saves a pass over whatever the
clauses share. **All three share the first operand and vary the second** — one
rule, not three special cases.

`d3dotxy_dotxz` closes the beachhead. With x = ∇∧B, y = ∇∧B, z = B it
returns Q₀ = x·y and h = x·z in one pass, so `x` and `y` are the same handle.
That is legal and free: the clauses are independent, and the shared operand is
simply read twice from cache rather than once from memory and once again
after a round trip.

**The F/S pair provably cannot be a fork.** F = ∇∧B ∧ B and S = E ∧ B share
`B`, but as the *second* operand. A fork shares the first operand and cross is
not commutative, so `x∧y` and `y∧x` cannot be swapped to fit the shape. Worth
knowing, because it is the difference between the two pairs and it is not a
naming accident.

Register pressure is what caps this at two outputs: a cross has three live
component expressions, so two cross clauses need six and a three-way fan
would need nine — more than the sixteen vector registers of an x86-64 target.
That is why wider forks are deferred rather than merely unwritten.

## Argument forms

Layer 1 carries neither `n` nor `inc` — both live in the handle. Handles pass
by pointer with `const` on inputs:

```c
void d3scal      (const v3 *x, T a, v3 *z);
void d3axpy      (const v3 *x, T a, v3 *y);
void d3axpby     (const v3 *x, T a, const v3 *y, T b, v3 *z);
void d3had       (const v3 *x, const v3 *y, v3 *z);
void d3cross     (const v3 *x, const v3 *y, v3 *c);
void d3crossscal (const v3 *x, const v3 *y, T a, v3 *c);
void d3dot       (const v3 *x, const v3 *y, v1 *r);
void d3norm2     (const v3 *x, v3 *z);
void d3crossdot  (const v3 *x, const v3 *y, const v3 *z, v1 *r);
void d3crossnorm2(const v3 *x, const v3 *y, v1 *r);
void d3crossxy_crossxz(const v3 *x, const v3 *y, const v3 *z, v3 *c1, v3 *c2);
void d3crossxy_dotxy  (const v3 *x, const v3 *y, v3 *c, v1 *r);
void d3dotxy_dotxz    (const v3 *x, const v3 *y, const v3 *z, v1 *r1, v1 *r2);
```

**Scalar argument position is not uniform, and that is a real wart.** In
`axpby` the scalar sits immediately before its own operand, so `d1scal`
would have to be `d1scal(a, x, z)` to match `d1axpby(a, x, b, y, z)` — but
stock `dscal` is `(n, a, x, incx)` and stock `daxpy` is `(n, a, x, incx, y,
incy)`, both with the scalar *second*. Matching the neighbours that a caller
already knows beats matching an internal rule, so the standalone scales take
`(x, a, z)`:

| symbol | signature | follows |
|---|---|---|
| `d1scal` | `(x, a, z)` | stock `dscal`/`daxpy` — scalar second |
| `d3scal` | `(x, a, z)` | `d3axpy`, which is `(x, a, y)` |
| `d1axpy` | `(n, a, x, incx, y, incy)` | stock exactly, except out of place |
| `d3axpy` | `(x, a, y)` | stock, except out of place |
| `d1axpby` | `(n, a, x, incx, b, y, incy, z, incz)` | stock order, plus a third operand |
| `d3axpby` | `(x, a, y, b, z)` | each scalar before its operand |
| `d3crossscal` | `(x, y, a, c)` | `d3axpby` order: scalars after their operands |

The rule that actually holds is: **each scalar follows the operands of the
term it belongs to**, which is the same first-appearance ordering everything
else uses. `d3crossscal`'s `a` multiplies the whole `x∧y`, so it comes last;
`d3axpby`'s `a` multiplies `x` alone, so it comes after `x`.

Layer 0 is flat stock CBLAS, so a caller holding only plain arrays needs no
glue:

```c
void d1scal (blaslong n, T a, const T *x, blaslong incx, T *z, blaslong incz);
void d1axpy (blaslong n, T a, const T *x, blaslong incx, T *y, blaslong incy);
void d1axpby(blaslong n, T a, const T *x, blaslong incx,
             T b, const T *y, blaslong incy, T *z, blaslong incz);
void d1had  (blaslong n, const T *x, blaslong incx,
             const T *y, blaslong incy, T *z, blaslong incz);
void d1nrm2 (blaslong n, const T *x, blaslong incx, T *z, blaslong incz);
void d1nrm  (blaslong n, const T *x, blaslong incx, T *z, blaslong incz);
```

All Layer-1 forms require `inc == 1`. Layer 0 accepts any `inc`.

## Composed operations

A name is never composed into another name. A caller who needs a sequence
writes two calls and names the temporary:

```c
v3 *t = v3_create(n);
d3cross(&v, &B, t);              /* t = v∧B          */
d3crossdot(&curlB, &B, &h);      /* h = (∇∧B)·B      */
d3crossnorm2(&curlB, &B, &Q0);   /* Q0 = ‖∇∧B‖²      */
d1axpby(Q0.n, 1/sigma, &Q0.p, Q0.inc, 0, &Q0.p, Q0.inc, &Q.p, Q.inc);
v3_destroy(t);
```

Were composition a symbol-level operator, the library would owe its users the
closure of the vocabulary under composition — an unbounded symbol set, which
contradicts acceptance criterion 1.

The beachhead (§Use case) in full — four calls, one pass each:

```c
d3crossscal(&curlB, &B, 1/mu0, &F);          /* F = (∇∧B)∧B / μ₀  */
d3crossscal(&E,    &B, 1/mu0, &S);          /* S = (E∧B)   / μ₀   */
d3dotxy_dotxz(&curlB, &curlB, &B, &Q0, &h);  /* Q0, h — one pass   */
d1axpby(Q0.n, 1/sigma, &Q0.p, Q0.inc, 0, &Q0.p, Q0.inc, &Q.p, Q.inc);
```

Every `μ₀⁻¹` and `σ⁻¹` is folded into a scalar argument rather than costing a
pass of its own. The F/S pair is two calls and provably cannot be one — they
share `B` as the second operand of a non-commutative cross (§Forks).

The same loop written entirely in Layer 0, over components, with no Layer-1
symbol at all:

```c
v1 Bx = v3_component(&B,0), By = v3_component(&B,1), Bz = v3_component(&B,2);
v1 vx = v3_component(&v,0), vy = v3_component(&v,1), vz = v3_component(&v,2);
v1 E  = v3_component(&E,0), Ey = v3_component(&E,1), Ez = v3_component(&E,2);

/* div B = v∧B */
d1axpby(n, +1, vz.p, vz.inc, -1, By.p, By.inc, Bx.p, Bx.inc);
d1axpby(n, -1, vx.p, vx.inc, +1, Bz.p, Bz.inc, Bx.p, Bx.inc);
d1axpby(n, +1, vy.p, vy.inc, -1, Bz.p, Bz.inc, By.p, By.inc);
d1axpby(n, -1, vz.p, vz.inc, +1, Bx.p, Bx.inc, By.p, By.inc);
d1axpby(n, +1, vz.p, vz.inc, -1, Bx.p, Bx.inc, Bz.p, Bz.inc);
d1axpby(n, -1, vy.p, vy.inc, +1, By.p, By.inc, Bz.p, Bz.inc);
```

That is six passes where `d3cross` is one. It is not the v1 recommendation —
it is the demonstration that the two layers compose, which is what
`v3_component` exists for and what motivated Layer 0 shipping in v1 at all.

## Tests

The identity each kernel must satisfy, in all four precisions. These are not
correctness tests of the implementation — a kernel computing the wrong formula
still satisfies most of them — they are the algebraic properties the
*definition* claims, and a violation means the body or the spec is wrong.

Notation: `‖·‖` componentwise, `≡` algebraic identity (all `n`, all inputs),
`≈` within rounding.

**Cross and dot**

| property | identity |
|---|---|
| cross is antisymmetric | `d3cross(x,y,c)` then `d3cross(y,x,c2)` → `c2 ≡ −c` |
| cross is not commutative | the two above are distinct; no mirror kernel exists |
| self-cross vanishes | `d3cross(x,x,c)` → `c ≡ 0` |
| cross scales | `d3cross(x,y,c)` ≡ `d3cross(x,w,c)`; `d3crossscal(x,w,1/a,d)` → `d ≡ c/a` |
| dot is symmetric | `d3dot(x,y,r)` ≡ `d3dot(y,x,r)` |
| self-dot is non-negative | `d3norm2(x,z)` → `z[i] ≥ 0`, `= 0` iff `x[i] = 0` |
| cross is orthogonal to its factors | `d3cross(x,y,c)` then `d3dot(x,c,r)` → `r ≡ 0`; likewise `d3dot(y,c,r)` |
| crossnorm2 is dot of cross with itself | `d3crossnorm2(x,y,r)` ≡ `d3cross(x,y,c)`; `d3dot(c,c,r)` |
| crossdot absorbs a permutation | `d3crossdot(x,y,z,r)` ≡ `d3crossdot(y,x,z,r)` |

**Componentwise**

| property | identity |
|---|---|
| scale by zero gives zero | `d3scal(x,0,z)` → `z ≡ 0` |
| scale is the axpby case | `d3scal(x,a,z)` ≡ `d3axpby(x,a,y,1,z)` with `y ≡ z` |
| axpy accumulates | `d3axpy(x,a,y)` ≡ `d3axpby(x,a,y,1,y)` |
| axpby with `a=1, b=0` copies | ≡ copy |
| axpby with `a=0` scales | `d3axpby(x,0,y,b,z)` ≡ `d3scal(y,b,z)` |
| had is commutative | `d3had(x,y,z)` ≡ `d3had(y,x,z)` |
| had against a constant | `d3had(one,x,z)` ≡ copy, with `one[i] = 1` |
| had against zero | `d3had(zero,x,z)` → `z ≡ 0` |
| nrm2 is nrm squared | `d1nrm(x,z)` ≡ `d1nrm2(x,w)`; `z[i] ≈ √w[i]` |
| Layer 0 ≡ Layer 1 at width 1 | `d1axpby(n,a,x,1,b,y,1,z,1)` ≡ `d3axpby` on the same data |

**Forks** — the only identities not inherited from stock BLAS, and the reason
to test them at all:

| property | identity |
|---|---|
| fork ≡ its two separate calls | `d3crossxy_crossxz(x,y,z,c1,c2)` ≡ `d3cross(x,y,c1)`; `d3cross(x,z,c2)` |
| a split changes nothing | every identity above holds at **every thread count** the host library offers, not just 1 — a split on `n` must not move a single output bit |
| likewise for the mixed fork | `d3crossxy_dotxy(x,y,c,r)` ≡ `d3cross(x,y,c)`; `d3dot(x,y,r)` |
| likewise for the dot/dot fork | `d3dotxy_dotxz(x,y,z,r1,r2)` ≡ `d3dot(x,y,r1)`; `d3dot(x,z,r2)` |
| a clause may reuse the shared operand | `d3dotxy_dotxz(x,x,y,r1,r2)` → `r1 ≡ r2 ≡ ‖x‖²` |
| the clauses are independent | permuting `y` and `z` swaps `r1` and `r2`, changing neither value |
| shared operands are read, not consumed | after the fork, `x`, `y`, `z` compare equal to their pre-call copies |

**Handles** — specific to this design, and the ones that would otherwise pass
silently:

| property | check |
|---|---|
| `v3_pitch` is stable | the same `n` returns the same pitch regardless of build or vector width |
| `cinc == 3 * inc` | for every constructed handle, including wrapped and component views |
| Layer 1 rejects `inc != 1` | a `v3` built with `inc = 3` passed to `d3cross` fails, rather than silently mis-executing |
| `v3_component` agrees with member access | `v3_component(&B,k).p == B.x + k`, `.inc == B.cinc` |
| a created vector is writable and destroyed cleanly | `v3_create` then `v3_destroy`, no leak, no double free |

**Structurally, for every kernel** — no exceptions, and independent of the
identities above:

| property | check |
|---|---|
| no write past `n` | guard canaries on **both sides** of every exact-size buffer, checked after **every** kernel call (README criterion 3) — including at every thread count, since a chunk boundary that overruns is the SMP version of the same bug |
| stride is transparent | `inc = 1` and `inc = k` results agree element-for-element, for Layer 0 |
| pure-first | inputs compare equal to their pre-call copies |
| edge `n` | `n = 0`, `1`, odd, and `n` not a multiple of the vector width; also `n` straddling the pitch boundary (`n` and `v3_pitch(n)/3`) |
| unaligned base pointers | a handle whose `x` is deliberately misaligned takes the scalar tail and still agrees |
| complex | bilinear, not Hermitian: `d3cross`/`d3dot` in c/z agree with the expanded real-pair formula; no conjugation anywhere |
| in-place | `d3cross(x,y,x)` ≡ `d3cross(x,y,c)` into fresh memory |