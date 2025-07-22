# Fair Archetypal Analysis

## Algorithms

The proposed algorithms are implemented in Python, leveraging popular libraries such as NumPy, scikit-learn, and PyTorch. The core methods for fair archetypal analysis are provided in the Jupyter notebooks included in this repository. These algorithms aim to perform archetypal analysis while ensuring fairness with respect to sensitive attributes.

- **Archetypal Analysis**: Decomposes data into convex combinations of extremal points (archetypes).
- **Fairness Constraints**: Methods are adapted to mitigate bias and ensure fair representation across groups.

## Experiments

A set of experiments is provided to demonstrate the effectiveness and fairness of the proposed algorithms. All experiments can be found in the Jupyter notebooks in this repository.

### Synthetic data

- **exemple.ipynb**: Demonstrates the basic archetypal analysis algorithm on synthetic datasets.
- **exemple_multiple.ipynb**: Explores the behavior of the algorithm under multiple synthetic scenarios.
- **exemple_moons.ipynb**: Applies the algorithm to the "moons" synthetic dataset, commonly used for clustering and fairness evaluation.

### Real data

- **ansur.ipynb**: Applies fair archetypal analysis to the ANSUR dataset, a real-world anthropometric dataset, to evaluate fairness and interpretability in practical scenarios.

---

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aleixalcacer/JA-fair-aa.git
   cd JA-fair-aa
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebooks:**
   Open any of the provided `.ipynb` files in Jupyter Notebook or JupyterLab to reproduce the experiments.


## Citation

If you use this code or algorithms in your research, please cite:

```bibtex
@misc{alcacer2025incorporatingfairnessconstraintsarchetypal,
      title={Incorporating Fairness Constraints into Archetypal Analysis}, 
      author={Aleix Alcacer and Irene Epifanio},
      year={2025},
      eprint={2507.12021},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2507.12021}, 
}
```
