# Machine Learning Invariants of Tensors

This repository contains the implementation accompanying the paper:

> **Machine Learning Invariants of Tensors**  
> Athithan Elamaran, Christian Ferko, Sterling Scarlett  
> arXiv:2512.23750v1

The project introduces a data-driven framework for discovering functionally independent tensor invariants using graph enumeration, random tensor sampling, and numerical linear algebra. :contentReference[oaicite:0]{index=0}

---

## Paper

The paper proposes an algorithm that:

- Enumerates tensor contraction graphs (tensor networks)
- Generates random numerical tensor realizations
- Evaluates scalar contractions
- Detects linear and polynomial dependencies numerically
- Identifies minimal generating sets of invariants

The main case study analyzes an antisymmetric 3-form \(H_{\mu\nu\rho}\) in six dimensions and finds **five independent invariants**. :contentReference[oaicite:1]{index=1}

---

## Overview

Given a tensor representation, the algorithm:

1. Enumerates inequivalent contraction graphs
2. Samples random tensors
3. Computes scalar contractions
4. Uses SVD / nullspace analysis to identify relations
5. Removes dependent invariants
6. Iterates until the invariant basis stabilizes

This provides a numerical approach to studying invariant theory problems that are often analytically difficult.

---

## Features

- Tensor-network / graph-based invariant construction
- Automatic detection of dependent invariants
- Numerical discovery of syzygies
- Support for antisymmetric tensors
- Relation extraction between invariant bases
- Case study implementations for:
  - Trace variables
  - Hodge-dual variables
  - Spinor variables

---
