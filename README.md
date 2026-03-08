# Corrected Schweighart–Sedwick Satellite Relative Motion Model

**Reproduction of:** Traub, Neubert, Ingrillini (2025)
*"Corrected Closed-Form Solutions to the Schweighart–Sedwick Satellite Relative Motion Model Including Differential Drag"*
Acta Astronautica 234, 742–753. DOI: [10.1016/j.actaastro.2025.05.019](https://doi.org/10.1016/j.actaastro.2025.05.019)

---

## What This Paper Does

The Schweighart–Sedwick (SS) equations model the relative motion between two spacecraft in a J₂-perturbed (Earth oblateness) orbit, expressed in the Local-Vertical Local-Horizontal (LVLH) frame. The paper identifies and corrects a previously unnoticed mathematical error in the widely-used closed-form analytical solutions by Shao et al. for the SS equations when a **tangential differential force** (δfᵧ ≠ 0) is applied (e.g., differential aerodynamic drag).

### The Bug
When computing the particular solution to the forced SS equations, Shao et al. used the Hill–Clohessy–Wiltshire (HCW) secular coefficient:

```
Q_shao = 2·δfᵧ / n           ← WRONG for J₂ ≠ 0 (c ≠ 1)
```

The correct SS coefficient is:

```
Q_traub = 2n·δfᵧ / ω²_SS     ← CORRECT
```

where ω²_SS = n²(4c + 2 − 5c²) and c is the SS oblateness coefficient. This error causes a **secular (linearly growing) error** in the along-track separation δy that is proportional to (c − 1) × δfᵧ × t.

---

## Physics Background

### Coordinate System
The LVLH (L-frame) is centered on the **chief** spacecraft:
- **x̂** — radial (away from Earth center)
- **ŷ** — along-track (direction of motion)
- **ẑ** — cross-track (completes right-hand system)

### The SS Equations (in-plane)
```
δẍ − 2n·δẏ − (5c²−2)n²·δx = δfₓ    [radial]
δÿ + 2cn·δẋ              = δfᵧ    [along-track]
```

### The SS Coefficient c
```
c = sqrt(1 + (3·J₂·R²_E)/(2p²) · (1 − 1.5·sin²i))
```
- c = 1 when J₂ = 0 (recovers standard HCW equations)
- c ≈ 1.001–1.002 for typical LEO orbits

---

## Repository Structure

```
Schweightart-Sedwick Satellite/
├── main.py                        # Run this — generates all 9 figures + validation
│
├── dynamics/
│   ├── constants.py               # Physical constants (μ, R_E, J₂)
│   ├── ss_model.py                # SS coefficient c, mean motion n, ω_SS
│   ├── numerical_integrator.py    # DOP853 ground-truth integration (rtol=1e-12)
│   ├── analytical_shao.py         # Original (incorrect) Shao et al. solutions
│   └── analytical_traub.py        # Corrected Traub et al. solutions
│
├── figures/
│   ├── fig1_geometry.py           # Formation geometry schematic
│   ├── fig2_decomposition.py      # Secular + periodic state decomposition
│   ├── fig3_noforce.py            # Validation: no differential force
│   ├── fig4_radial.py             # Validation: radial force only
│   ├── fig5_tangential.py         # Validation: tangential force only ← KEY FIGURE
│   ├── fig6_combined.py           # Validation: combined forces
│   ├── fig7_errors.py             # Error magnitudes on log scale
│   ├── fig8_parametric.py         # Error scaling with |c−1| (J₂ effect)
│   └── fig9_phase_plane.py        # Phase plane under constant forces
│
├── validation/
│   └── verify_CW_reduction.py     # 5 automated correctness tests
│
└── outputs/                       # All generated figures (PDF + PNG)
```

---

## Running the Code

### Requirements
- Python 3.x (Anaconda recommended)
- NumPy, SciPy, Matplotlib

### Run
```bash
cd "Schweightart-Sedwick Satellite"
/Users/rishinalem/anaconda3/bin/python main.py
```

All 9 figures are saved to `outputs/` as both PDF and PNG. Validation test results are printed to the console.

---

## Simulation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| a | 6871 km | Semi-major axis (500 km altitude) |
| e | 0 | Eccentricity (circular orbit) |
| i | 98° | Inclination (sun-synchronous) |
| n | ~1.098 × 10⁻³ rad/s | Chief mean motion |
| T | ~5726 s (~95 min) | Orbital period |
| c | ~1.00145 | SS oblateness coefficient |
| δx₀ | 100 m | Initial radial separation |
| δẏ₀ | −2cn·δx₀ | Bounded orbit condition |
| δfᵧ | 1×10⁻⁷ m/s² | Differential tangential force |
| t_final | 5T | Integration duration |

---

## Output Figures

| File | Description |
|------|-------------|
| `fig01_formation_geometry` | LVLH frame schematic with chief, deputy, ρ vector |
| `fig02_state_decomposition` | δx and δy split into secular + periodic components |
| `fig03_validation_noforce` | Numerical vs Shao vs Traub — no force (all agree) |
| `fig04_validation_radial` | Numerical vs Shao vs Traub — radial force (all agree) |
| **`fig05_validation_tangential`** | **KEY: Shao secular error vs Traub accuracy** |
| `fig06_validation_combined` | Combined radial + tangential forces |
| `fig07_error_comparison` | Log-scale error magnitude vs time |
| `fig08_parametric_study` | Shao error ∝ \|c−1\| (linear on log-log) |
| `fig09_phase_plane` | Phase plane trajectories under ±δfᵧ |

---

## Validation Tests

Five automated tests verify correctness:

1. **CW Reduction** — Setting c=1 exactly, Traub solutions must match standard HCW formulas to machine precision
2. **No-force agreement** — Both solutions agree with numerical integration
3. **Radial force only** — Shao and Traub agree (bug only affects δfᵧ terms)
4. **Tangential force, c=1** — Traub and Shao are identical (error ∝ c−1, vanishes at c=1)
5. **Error scaling** — Shao δy error scales linearly with |c−1| (slope ≈ 1 on log-log)

---

## Key Result

After 5 orbital periods with δfᵧ = 10⁻⁷ m/s² (tangential force only):

| Solution | δx error | δy error |
|----------|----------|----------|
| Shao et al. (incorrect) | ~0.03 m | **~0.32 m** (secular growth) |
| Traub et al. (corrected) | ~10⁻¹⁰ m | ~10⁻¹⁰ m (machine precision) |

The Shao error grows as `Δ ≈ 2·δfᵧ·t·n·(c−1)·(5c+1) / ω²_SS`, linearly in time and proportional to J₂.
