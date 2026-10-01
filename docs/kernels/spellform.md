# Kernel scheme `spellform` — Layer 1 (3-vectors) + Layer 0 (1D scalar arrays)

**Scheme: `spellform`** — kernel names *spell their formula* in
one-letter glyphs (§Naming system). This folder holds one file per
scheme; trying a different scheme = adding a file here, nothing else
references op names. This file enumerates the current ops — naming
grammar as applied, legend, kernel tables, inner-loop shapes, per-op
argument forms, composed operations, test identities. The stable
introduction (motivation, API rules, coding principles) lives in
`../../README.md`.

## Naming system

### Letter space

```
a b c       pure scalars (kernel inputs)
            e = equals    p = plus    m = minus
            o = over (÷)  v = vector product (×)    d = dot product (·)
            (juxtaposition = Hadamard product, element-wise ⊙)
r s t (u)   scalar arrays (arrays of scalars; pointwise 1D)
x y z (w)   3-vectors
```

The six glyph letters (e p m o v d) are disjoint from every variable pool
above. All-lowercase → valid Fortran symbols as-is.

### Rules

- **Name = formula, spelled left to right:** `[lhs] e [term] (p|m) [term] …`
  A term is a product, written by juxtaposition: `ax` = a·x, `xy` = x⊙y,
  `xvy` = x×y, `xdy` = x·y, `xoy` = x÷y.
- **Lettering by kind and position.** Distinct variables of each kind are
  lettered by first appearance on the right-hand side, left to right, from
  that kind's alphabet:
    3-vectors     → x, y, z, (w)
    scalar arrays → r, s, t, (u)
  A *fresh* output (absent from the RHS) takes the next unused letter of its
  kind; an *in-place* output keeps the letter it already has. The LHS is the
  output's letter — `x`/`y`/`z` for a 3-vector, `r`/`s`/`t` for a scalar array.
- **In-place is visible, not flagged:** the LHS letter reappears as a term
  (`yeaxpby` ends in `y`; `zeaxy` does not).
- **Chained same-glyph reads left-to-right:** `x v y v z` = `(x×y)×z`.
- **No fresh 4-vector kernels.** Formulas with 4 distinct vectors are
  provided only in their in-place 3-distinct form.
- **In-place division comes in mirror pairs.** A pure division needs no
  mirror (`zexoy` with swapped arguments is the reverse quotient), but an
  in-place division pins the LHS, so the two operand orders are distinct
  kernels: LHS in the denominator (`yeaxoy`: y = a·x÷y) vs LHS in the
  numerator (`yeayox`: y = a·y÷x). 3-operand form: the LHS slides into the
  numerator, displacing the last factor to the denominator
  (`zeaxyoz` ↔ `zeaxzoy`).
- **Precision prefix** `s3/d3/c3/z3` is prepended at the symbol level:
  the double-precision `zexvy` links as `d3zexvy`.

### Reading examples

    yeaxpby   =  y e ax p by           y = a·x + b·y
    zexvy     =  z e x v y             z = x × y
    zeaxypz   =  z e axy p z           z = a·(x⊙y) + z
    rexvydz   =  r e (x v y) d z       r[i] = (x[i]×y[i])·z[i]
    rexvyxdvy =  r e (x v y) d (x v y) r[i] = ‖x[i]×y[i]‖²
    searpbs   =  s e ar p bs           s = a·r + b·s       (Layer 0)
    tearspt   =  t e ars p t           t = a·(r⊙s) + t     (Layer 0)

## Legend

⊙ = Hadamard (element-wise product), × = vector product, · = dot
product, ÷ = element-wise quotient. `a·x` with scalar `a` means
componentwise scaling (free; it is not a kernel op). Memory column
(`mem/e`): loads+stores of component words per element (1 v3 vector =
3 words). The letters, glyphs, and spelling rules that generated these
names: §Naming system above.

## Layer 1 — the v3 kernel set (25 kernels → 100 symbols)

### 1-vector (x*)

| name   | formula      | mem/e | note                    |
|--------|--------------|-------|-------------------------|
| `xeax` | x = a·x      | 3+3   | the v3 `scal`           |

### 2-vector (y*)

| name       | formula         | mem/e | note                        |
|------------|-----------------|-------|-----------------------------|
| `yeax`     | y = a·x         | 3+3   | scaled copy                 |
| `yeaxpby`  | y = a·x + b·y   | 6+3   | the v3 `axpby`; time-integrator state update |
| `yexx`     | y = x⊙x         | 3+3   | per-component square        |
| `yeaxxpy`  | y = a·(x⊙x) + y | 6+3   | scaled square, accumulate   |
| `yeaxxmy`  | y = a·(x⊙x) − y | 6+3   | same, minus                 |
| `yeaxoy`   | y = a·x ÷ y     | 6+3   | in-place divide, LHS in denominator |
| `yeayox`   | y = a·y ÷ x     | 6+3   | in-place divide, LHS in numerator   |

### 3-vector, pure (z*)

| name        | formula        | mem/e | note                        |
|-------------|----------------|-------|-----------------------------|
| `zexy`      | z = x⊙y        | 6+3   | Hadamard                    |
| `zeaxy`     | z = a·(x⊙y)    | 6+3   | scaled Hadamard             |
| `zexvy`     | z = x×y        | 6+3   | cross                       |
| `zeaxvy`    | z = a·(x×y)    | 6+3   | **the beachhead op**: `F=(∇×B)×B/μ₀` and `S=(E×B)/μ₀` are both this |
| `zexoy`     | z = x÷y        | 6+3   | element-wise quotient       |
| `zexxpyy`   | z = x⊙x + y⊙y  | 6+3   | sum of squares              |
| `zexxmxyy`  | z = x⊙x − y⊙y  | 6+3   | difference of squares       |

### 3-vector, in-place (z*)

| name       | formula         | mem/e | note                        |
|------------|-----------------|-------|-----------------------------|
| `zeaxypz`  | z = a·(x⊙y) + z | 9+3   | Hadamard multiply-accumulate|
| `zeaxymz`  | z = a·(x⊙y) − z | 9+3   | same, minus                 |
| `zeaxyoz`  | z = a·(x⊙y) ÷ z | 9+3   | in-place divide, LHS in denominator |
| `zeaxzoy`  | z = a·(x⊙z) ÷ y | 9+3   | mirror: LHS in numerator    |
| `zeaxyz`   | z = a·(x⊙y)⊙z   | 9+3   | elementwise 3-chain         |
| `zexvyvz`  | z = (x×y)×z     | 9+3   | vector triple, left-to-right|

### Scalar-array outputs (r*)

| name         | formula               | mem/e | note                          |
|--------------|-----------------------|-------|-------------------------------|
| `rexdy`       | r[i] = x[i]·y[i]      | 6+1   | pointwise dot (E·B, v·B, B·ω) |
| `rexdx`       | r[i] = x[i]·x[i]      | 3+1   | pointwise ‖·‖²                |
| `rexvydz`     | r[i] = (x[i]×y[i])·z[i] | 9+1 | scalar triple (helicity, α-effect) |
| `rexvyxdvy`   | r[i] = (x[i]×y[i])·(x[i]×y[i]) | 6+1 | ‖x×y‖² pointwise, **cross formed in registers** |

All 25: arithmetic intensity ≈ 0.2–0.3 flops/word — deep memory-bound,
same roofline class as SAXPY.

### Inner loops (FMA shape, per component)

    xeax:      t = a*x.c
    yeaxpby:   y.c = fma(b, y.c, a*x.c)          1 mul + 1 FMA
    zexvy:     z.x = fma(x.y, y.z, -(x.z*y.y))   2 instr, not 3
    zeaxvy:    2 mul + 1 FMA
    zexxpyy:   z.c = fma(x.c, x.c, y.c*y.c)      1 mul + 1 FMA
    zeaxypz:   z.c = fma(a*x.c, y.c, z.c)        1 mul + 1 FMA
    rexdy:     r = x.x*y.x + x.y*y.y + x.z*y.z   3 mul + 2 FMA
    rexvyxdvy: c = x×y (registers); r = c·c      2R+1W total
    zexvyvz:   two crosses, first in registers

### `rexvyxdvy` (the redesigned `cross2`)

Evaluated directly, not via Lagrange: form the cross in registers,
self-dot in registers, never store the intermediate.

| strategy                            | mem/elt      |
|-------------------------------------|--------------|
| direct (this)                       | 6 loads + 1 store |
| Lagrange ‖x‖²‖y‖² − (x·y)²         | 12 loads + 1 store — only wins if the three factors are live elsewhere |
| naive (cross to memory, then norm)  | 6 loads + 4 stores |

The Lagrange identity is kept in this doc only as a reuse-case note. The
cleverness is register fusion, not the identity.

## Layer 0 — 1D element-wise kernels (same patch, no `3` prefix)

The structure-blind half of the algebra, as plain 1D-array ops. These
take **scalar arrays only** (`r s t`) — no `x y z` appears in a Layer-0
name — and the symbol carries **no `3`**: the double element-wise
product `t = r⊙s` links as `dters`, not `d3…`. This is the
"standard-BLAS extension" layer — useful to any code doing element-wise
work, not just 3-vectors, and it is what makes field coefficients and
mixed-component products expressible (see Composed operations).

Ships in the **same single patch** as Layer 1. The Layer-1 v3 kernels are
self-contained fused passes (register fusion) and do **not** call the
Layer-0 kernels, so there is no internal ordering dependency and "do the
3-kernels build on the 1-kernels?" is a non-question — either implementation
is possible without a patch split.

### 1-array (r*)

| name   | formula  | note                           |
|--------|----------|--------------------------------|
| `rear` | r = a·r  | STOCK `dscal` — not re-added   |

### 2-array (s*)

| name      | formula         | note                            |
|-----------|-----------------|---------------------------------|
| `sear`    | s = a·r         | scaled copy (new)               |
| `searpbs` | s = a·r + b·s   | STOCK `daxpby` — not re-added   |
| `serr`    | s = r⊙r         | per-element square (new)        |
| `searrps` | s = a·(r⊙r) + s | scaled square, accumulate       |
| `searrms` | s = a·(r⊙r) − s | same, minus                     |
| `searos`  | s = a·r ÷ s     | in-place divide, LHS in denominator |
| `seasor`  | s = a·s ÷ r     | in-place divide, LHS in numerator   |

### 3-array (t*)

| name      | formula          | note                            |
|-----------|------------------|---------------------------------|
| `ters`    | t = r⊙s          | element-wise product (new)      |
| `tears`   | t = a·(r⊙s)      | scaled element-wise product     |
| `teros`   | t = r÷s          | element-wise quotient           |
| `terrsps` | t = r⊙r + s⊙s    | sum of squares                  |
| `trrmss`  | t = r⊙r − s⊙s    | difference of squares           |
| `tearspt` | t = a·(r⊙s) + t  | element-wise multiply-accumulate|
| `tearsmt` | t = a·(r⊙s) − t  | same, minus                     |
| `tearsot` | t = a·(r⊙s) ÷ t  | in-place divide, LHS in denominator |
| `teartos` | t = a·(r⊙t) ÷ s  | mirror: LHS in numerator        |
| `tearst`  | t = a·(r⊙s)⊙t    | element-wise 3-chain            |

**16 new** Layer-0 kernels (the 18 structure-blind ops minus the 2 that
are stock `dscal`/`daxpby`) × 4 precisions = **64 symbols**. No `d` (dot)
or `v` (cross) glyph appears in any Layer-0 name: contractions use the
3-vector structure and live in Layer 1.

**Combined (this variant): 100 (Layer 1) + 64 (Layer 0) = 164 linkable
symbols.**

## Per-op argument forms

Rule (from `../../README.md` §Public API): operands in order of first
appearance on the RHS, left to right; scalars where their term spells
them; a fresh output last; in-place outputs reuse their single pointer.
Layer-1 vector args are **descriptor pointers** (`s3v*/d3v*/c3v*/z3v*`)
and there is **no `n`** — the descriptor carries the length, which also
sizes plain scalar-array outputs. Layer-0 args are plain array pointers
with `n` first, BLAS-style. c/z precisions take complex scalars (pairs
of reals) in the same slots.

Layer 1 (capital = `d3v *`; lowercase `r` = plain array, sized by the
descriptors' `n`):

| args         | kernels |
|--------------|---------|
| (a, X)       | `xeax` |
| (a, X, b, Y) | `yeaxpby` |
| (a, X, Y)    | `yeax`, `yeaxxpy`, `yeaxxmy`, `yeaxoy`, `yeayox` |
| (X, Y)       | `yexx` |
| (X, r)       | `rexdx` |
| (X, Y, Z)      | `zexy`, `zexvy`, `zexoy`, `zexxpyy`, `zexxmxyy`, `zexvyvz` |
| (a, X, Y, Z)   | `zeaxy`, `zeaxvy`, `zeaxypz`, `zeaxymz`, `zeaxyoz`, `zeaxzoy`, `zeaxyz` |
| (X, Y, r)      | `rexdy`, `rexvyxdvy` |
| (X, Y, Z, r)   | `rexvydz` |

Layer 0 (plain arrays; `n` first):

| args            | kernels |
|-----------------|---------|
| (n, a, r)       | `rear` |
| (n, a, r, b, s) | `searpbs` |
| (n, a, r, s)    | `sear`, `serr`, `searrps`, `searrms`, `searos`, `seasor` |
| (n, r, s, t)    | `ters`, `teros`, `terrsps`, `trrmss` |
| (n, a, r, s, t) | `tears`, `tearspt`, `tearsmt`, `tearsot`, `teartos`, `tearst` |

## Composed operations (no domain names in the API)

The beachhead composes from the generic set (vector args are
descriptors, shown as `&name`; Layer-0 calls keep their `n`):

    F = (∇×B)×B/μ₀      d3zeaxvy(&curlB, &B, &mu0i, &F)       one call
    S = (E×B)/μ₀        d3zeaxvy(&E, &B, &mu0i, &S)           one call
    Q = ‖∇×B‖²/σ        d3rexdx(&curlB, q);  dscal(n, &sigmai, q)
    h = B·(∇×B)         d3rexdy(&B, &curlB, h)
    particle Lorentz    t = v×B;  F = q·t + q·E
                        d3zexvy(&v, &B, &t);  d3yeaxpby(&t, &E, &q, &q, &F)

Field coefficients and mixed-component products are structure-blind, so
they live in Layer 0 (a coefficient that is itself a *field*, and
products of *different* components, are plain 1D-array ops with no v3
form):

    rho*v.x²    (flux/stress diagonal)  dserr(n, v.x, tmp);     dters(n, rho, tmp, out)
    rho*v.x*v.y (off-diagonal)          dters(n, v.x, v.y, tmp); dters(n, rho, tmp, out)

The old doc's `d3_lorentz`/`d3_poynting` are removed from the API: they
were physics branding on `zeaxvy`. Shared-operand fusions for the
two-cross case are stashed (`../../README.md` §Deferred).

## Test identities

- `rexvyxdvy` vs Lagrange: ‖x×y‖² = ‖x‖²‖y‖² − (x·y)² (cross-kernel
  consistency, using `rexdx`/`rexdy`)
- `rexvydz` cyclic invariance: (x×y)·z = (y×z)·x = (z×x)·y
- `zexvyvz` vs identity: (x×y)×z = (x·z)·y − (y·z)·x
- in-place vs pure equivalence: `zeaxypz` on a copy == `zeaxy`-then-add
- bilinearity spot checks for the `a`-scaled families
- complex precisions: pure bilinear (no conjugation) — verify explicitly
