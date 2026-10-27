# FYS4480/FYS9480 – Week 44, October 29–30, 2026

**Many-body perturbation theory: reference states, projection methods and energy corrections**

Lecturers: Cecilie Glittum and Morten Hjorth-Jensen

- Thursday October 29: a reference determinant and its excitations, the exact energy projected onto it, projection operators and the reduced resolvent, Brillouin–Wigner versus Rayleigh–Schrödinger, and the first three orders
- Friday October 30: the second-order energy from Wick's theorem, a two-level benchmark through fourth order, and every contribution to the energy through fourth order and to the wave operator through third

Reading: the slides; lecture notes chapter 9, sections 9.1–9.6; Shavitt and Bartlett, chapter 2; Szabo and Ostlund, chapter 6. Code: `BookPrograms/chapter09/mbpt.py`; notebook `week44.ipynb`.

Exercises: `Exercises/ExercisesWeek44.tex` (reproduced on the last two slides); answers on the week-45 slides.

## Plan for week 44

### Thursday October 29, 2 × 45 min: reference states, projection methods and systematic orders (sections 9.1–9.4)
- A reference determinant, its 1p1h and 2p2h excitations, and the exact energy projected onto the reference: only singles and doubles couple directly
- A two-state problem solved by hand: the orders we must recover
- Intermediate normalisation, two energy differences that must be kept apart, P and Q, the reduced resolvent, an exact equation for Q|Ψ₀⟩ and its Neumann series
- Two choices of starting energy: Brillouin–Wigner (ω = E₀, implicit) and Rayleigh–Schrödinger (ω = ε₀, explicit)
- A resolvent identity, the resummation in Q space, the energy-dependent effective Hamiltonian, and the secular equation recovered
- Rayleigh–Schrödinger order by order: E⁽¹⁾, E⁽²⁾, and the normalisation subtraction in E⁽³⁾

### Friday October 30, 2 × 45 min: the second-order energy and a fourth-order check (sections 9.5–9.6)
- First-order direct and exchange terms; which intermediate determinants contribute; the perturbation normal-ordered, H_I = E⁽¹⁾ + U_N + V_N
- Singles: the coherent mean-field sum. Doubles: Wick contractions, the counting factor, the leading amplitude. The complete E⁽²⁾
- A purely off-diagonal two-level benchmark: Brillouin–Wigner exact with two vertices, Rayleigh–Schrödinger through fourth order, only the folded term survives; the radius of convergence |2λ/Δ| = 1
- The list: every contribution to the energy through fourth order and to the wave operator through third order

## Exercises, week 44

1. Let H = H₀ + V with ⟨φ₀|ψ₀⟩ ≠ 0 and non-degenerate ground states. Show E₀ − ε₀ = ⟨φ₀|V|ψ₀⟩/⟨φ₀|ψ₀⟩; show P = |φ₀⟩⟨φ₀| and Q = 1 − P are idempotent; derive the expansions of |ψ₀⟩ and E₀ in powers of [Q/(z − H₀)](z − E₀ + V) for arbitrary z; discuss z = E₀ (Brillouin–Wigner) and z = ε₀ (Rayleigh–Schrödinger) and compare the first terms.
2. Two fermions in the pair orbitals |m, −m⟩ of a single j-shell with ⟨m, −m|v|m′, −m′⟩ = −G: show from the Brillouin–Wigner expansion that E₀ = −(j + 1/2)G; confirm it by diagonalisation; apply Rayleigh–Schrödinger theory and discuss the differences.

## Next week (week 45)

- Diagram rules and their derivation (chapter 9, sections 9.6–9.12): the perturbation drawn as crosses and interaction lines, particle and hole lines, the energy denominator as a cut between vertices; the rules (sums, a denominator per cut, ½ per pair of equivalent lines, the sign (−1)^(h+l)) derived from Wick's theorem; first-order, first-order-wave-function and second-order diagrams matched to this week's formulas; third order, unlinked diagrams and the linked-diagram theorem; the four models of chapter 9 order by order, size extensivity, and how far the series can be trusted.
- Reading: chapter 9, sections 9.6–9.12; Shavitt and Bartlett, chapters 4–5. The week-44 exercises are discussed in the exercise session and answered on the week-45 slides.
