> **REJECTED — kept for the record only.** Nothing in the spec depends on
> this file. The letter grammar below was consistently re-derived and
> consistently mis-spelled, by both author and reviewer, and the failure mode
> is the reason: a typo in a letter grammar produces a *valid but different*
> kernel name rather than a compile error. The current kernel set is
> **`v1set.md`**, named on BLAS convention instead (README §Corrections 21–22).

## Legend
### Variables
| letters | class | argument type |
|---|---|---|
| `a` `b` `e` | pure scalars | `T` |
| `q` `r` `s` | scalar arrays | `T *` |
| `u` `v` `t` | 1-sized 3-vectors | `d3v *` |
| `x` `y` `z` `w` | long 3-vectors | `d3v *` |

### Operands
| glyph | operation | math | commutative |
|---|---|---|---|
| *adjacency* | hadamard element-wise product | .* | yes |
| `c` | cross | ∧ | **no** |
| `d` | dot product | · | yes |
| `p` | addition | + |  yes |
| `m` | subtraction | - | **no** |
| `_` | fork separator (§Forks) | N/A |

## The v1 names
| name | mathematics | output type |
| core |---|---|
| `axcy`  | a*x∧y | T3* |
| `xcycz` | x∧y∧z | T3* |
| `xdypa` | x·y + a | T*  |
| `xdxpa` | x·x + a  | T* |
| `xvydz` | x∧y·z | v3* |
| `xvydxvy` | r = ‖x∧y‖² |  |
| hadamard |---|---|
| `xypa` | x.*y+a | T3* |
| `axy` | z = a·(x.*y) | T3* |
| `axpby` | y = a·x + b·y | `(a, &x, b, &y)` |
| `axpbyez` | y = a·x + b·y +ez | `(a, &x, b, &y)` |
| `rx` | z = r.*x | `(&r, &x, &z)` |

`rx` is the only way to scale a vector by a field: ρ·v is `rx` with
r = ρ.

### Forks

`_` separates clauses, **one clause per output**, and each clause names
only the operands it uses. Shared operands are passed once as arguments,
not repeated in each clause — so a fork is one body and one symbol, and
it saves a pass over whatever the clauses share.

| name | mathematics | beachhead |
|---|---|---|
| `axvy_axvy` | z₁ = a·(x∧y), z₂ = a·(x′∧y) | `d3axvy_axvy(&curlB, &E, &B, 1/mu0, &F, &S)` |
| `xdx_xdy` | r₁ = x·x, r₂ = x·y | `d3xdx_xdy(&curlB, &B, &Q, &h)` |

Traffic: `axvy_axvy` 18 → 15 mem/elt (shares B and a), `xdx_xdy`
11 → 8 mem/elt (shares ∇∧B).

## Tests

The identity each kernel must satisfy, in all four precisions. These are
not correctness tests of the implementation — a kernel that computes the
wrong formula still satisfies most of them — they are the algebraic
properties the *definition* claims, and a violation means the body or the
spec is wrong.

Notation: `‖·‖` componentwise, `≡` algebraic identity (all `n`, all
inputs), `≈` within rounding.

**Cross and dot**

| property | identity |
|---|---|
| cross is antisymmetric | `cxy(&x,&y,&z)` ≡ `cxy(&y,&x,&z)`, `z = −z_old` |
| cross is not commutative | the two above are distinct; no mirror kernel exists |
| self-cross vanishes | `cxy(&x,&x,&z)` → `‖z‖ = 0` |
| cross scales | `cxy(a,&x,&y,&z)` ≡ `axy(a, &x, &y, &w); cxy(&w,&y,&z)` |
| dot is symmetric | `xdy(&x,&y,&r)` ≡ `xdy(&y,&x,&r)` |
| self-dot is non-negative | `xdx(&x,&r)` → `r[i] ≥ 0`, `= 0` iff `x[i] = 0` |
| cross is orthogonal to its factors | `xdy(&x, cross_out)` → 0 |
| nested cross | `cxycz(&x,&y,&z,&w)` ≡ expand, and `w` has the triple-product form |

**Componentwise**

| property | identity |
|---|---|
| scale is idempotent on a unit | `axy(a=1, &x, &y, &z)` ≡ `xy(&x,&y,&z)` |
| scale by zero gives zero | `axy(a=0, …)` → `‖z‖ = 0` |
| scalar scale commutes | `axy(a,&x,&y,&z)` ≡ `axy(b,&x,&y,&w); axy(a/b, &w,·,·)` when `b ≠ 0` |
| axpby with `a=1, b=0` copies | `axpby(1,&x,0,&y)` ≡ copy |
| axpby with `a=0` scales | `axpby(0,&x,b,&y)` ≡ `axy(b,·,&y,&tmp); axpby(·,·,·,&tmp)` |
| `rx` against a constant | `rx(&one, &x, &z)` ≡ copy, with `one[i] = 1` |
| `rx` against zero | `rx(&zero, &x, &z)` → `‖z‖ = 0` |

**Forks** — the only identities not inherited from stock BLAS, and the
reason to test them at all:

| property | identity |
|---|---|
| fork ≡ its two separate calls | `axvy_axvy(&x,&w,&y,a,&z1,&z2)` ≡ `axvy(a,&x,&y,&z1); axvy(a,&w,&y,&z2)` |
| likewise for the contraction fork | `xdx_xdy(&x,&y,&r1,&r2)` ≡ `xdx(&x,&r1); xdy(&x,&y,&r2)` |
| shared operands are read, not consumed | after the fork, `x`, `y`, `w` compare equal to their pre-call contents |

**Structurally, for every kernel** — no exceptions, and independent of
the identities above:

| property | check |
|---|---|
| no write past `n` | guard canaries on **both sides** of every exact-size buffer, checked after **every** kernel call (README criterion 3) |
| stride is transparent | `inc = 1` and `inc = k` results agree element-for-element |
| pure-first | inputs compare equal to their pre-call copies |
| edge `n` | `n = 0`, `1`, odd, and `n` that is not a multiple of the vector width |
| complex | bilinear, not Hermitian: `cxy`/`xdy` in c/z agree with the expanded real-pair formula; no conjugation anywhere |

## Composition is caller-side

A name is never composed into another name. A caller who needs a
sequence writes two calls and names the temporary, as in the particle
Lorentz force:

```c
d3cxy(&v, &B, &t);          /* t = v∧B        */
d3axpby(&t, &E, q, q, &F);  /* F = q·t + q·E  */
```

Were `_` a symbol-level operator, the library would owe its users the
closure of the grammar under composition — an unbounded symbol set,
which contradicts acceptance criterion 1.
