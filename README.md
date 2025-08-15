# AI Project – KNN Classifier & Genetic Algorithm Regression

This project implements and compares two AI approaches:

1. **K-Nearest Neighbour (KNN) Classifier**
2. **Regression with a Genetic Algorithm (GA)**

It was developed as part of **LZSCC.361 – Coursework Part II**.

---

## 📌 Overview

### **Task 1 – KNN Classifier**
- Implements a **KNN classifier** with Euclidean and Manhattan distances.
- Chooses the optimal **k** value by evaluating classification error on training and validation sets.
- **Tie-breaking rule**: if k is even and there is a tie, label `1` is predicted.
- Euclidean distance achieved the lowest validation error with **k = 3**.

**Best results (Euclidean):**
- Training error: `0.01667`
- Validation error: `0.03519`

---

### **Task 2 – Genetic Algorithm Regression**
- Implements a **continuous-valued GA** to fit coefficients `[a0, a1, a2]` for regression.
- **Encoding**: real-valued genes for continuous optimization.
- **Objective function**: Mean Squared Error (MSE).
- **Selection**: Roulette wheel (fitness = 1/error).
- **Crossover**: Intermediate arithmetic crossover (p = 0.75).
- **Mutation**: Adaptive mutation (p = 0.02, increases on stagnation).
- **Population**: size 10 with 3 elites per generation.
- **Termination**: best fitness < 2.98.

**Best results:**
- Achieved lower classification errors than KNN in some runs, but with variable runtime (5–220 generations, worst ~54s).
- More exploration of the solution space than KNN.

---

## 📊 Comparative Analysis
| Method | Train Error | Validation Error | Speed |
|--------|------------|------------------|-------|
| KNN (k=3, Euclidean) | 0.01667 | 0.03519 | Fast |
| GA Regression | Variable (can be better) | Variable (can be better) | Slower, variable |

**Conclusion:**
- **KNN** → fast, reliable for consistent accuracy.
- **GA** → better potential accuracy but slower and less predictable.

---


