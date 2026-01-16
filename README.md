## Reproducing the numerical demonstrations

The scripts in src/ reproduce the numerical results shown in the
"Numerical demonstrations" subsection of the paper "Scalable multi-qubit optimal control with k-body decomposable targets" by B. Baran et. al..

We consider two three-qubit entangling gates:

- A *diagonal gate*, approximately exp(i π/4 Z ⊗ Z ⊗ Z) (ZZZ)
- A *non-diagonal gate*, approximately exp(i π/4 X ⊗ Z ⊗ Z) (XZZ)

To regenerate the figures:

```bash
# Diagonal gate (ZZZ)
python -m src.reproduce_numerics --gate diagonal

# Non-diagonal gate (XZZ)
python -m src.reproduce_numerics --gate nondiagonal