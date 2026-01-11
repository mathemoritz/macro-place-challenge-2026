# External Dependencies

This directory contains external code dependencies as git submodules.

## MacroPlacement (TILOS)

**Source**: [TILOS-AI-Institute/MacroPlacement](https://github.com/TILOS-AI-Institute/MacroPlacement)

**Purpose**: Ground truth evaluation using their sophisticated congestion model.

### What We Use

We only use the **PlacementCost evaluator** from this repository:

```
MacroPlacement/
└── CodeElements/
    └── Plc_client/
        ├── plc_client_os.py      ← Main evaluator class
        └── plc_client_os.pyx     ← Cython version (optional)
```

**Key functionality**:
- `PlacementCost` class for evaluation
- Wirelength, density, and congestion cost computation
- Grid-based placement representation
- Net routing simulation (2-pin, 3-pin, multi-pin)

### What We Don't Use

The MacroPlacement repo contains many other components we don't need:

- ❌ `Testcases/` - Circuit benchmarks (we use our own tensor format)
- ❌ `ExperimentalData/` - Research results and flow data
- ❌ `Flows/` - Full physical design flows (Cadence, SKY130, etc.)
- ❌ `CodeElements/SimulatedAnnealing/` - Placement algorithms (we provide our own)
- ❌ `CodeElements/FDPlacement/` - Force-directed placement
- ❌ `CodeElements/Clustering/` - Netlist clustering
- ❌ `Docs/OurProgress/` - Research documentation

### Size Optimization

The full MacroPlacement repository is ~500MB due to testcases and experimental data.

**What we actually need**: ~5MB (just the Plc_client code)

If you want to minimize download size, you can use a shallow clone:

```bash
git submodule update --init --depth 1 external/MacroPlacement
```

Or use sparse checkout to only get the Plc_client directory:

```bash
git -C external/MacroPlacement config core.sparseCheckout true
echo "CodeElements/Plc_client/*" >> external/MacroPlacement/.git/info/sparse-checkout
git submodule update --force --checkout external/MacroPlacement
```

### Dependencies

The PlacementCost evaluator requires:
- `absl-py` - Abseil Python library (Google's Python utilities)
- `matplotlib` - For visualization (optional, only if using plotting features)
- `numpy` - Numerical operations

Install with:
```bash
pip install -r external/requirements-tilos.txt
```

### License

The MacroPlacement repository is licensed under BSD 3-Clause License.
See `external/MacroPlacement/LICENSE` for details.

### Attribution

When using the TILOS evaluator, please cite their paper:

```bibtex
@inproceedings{kahng2023assessment,
  title={Assessment of Reinforcement Learning for Macro Placement},
  author={Kahng, Andrew B and others},
  booktitle={ISPD},
  year={2023}
}
```

## Why Use a Submodule?

Instead of copying their code or requiring separate installation, we use a git submodule because:

1. **Automatic updates**: Easy to pull in bug fixes and improvements
2. **Proper attribution**: Clear provenance of external code
3. **Reduced maintenance**: Don't need to maintain a fork
4. **Optional dependency**: Users can skip it if they only want the approximate evaluator
5. **Version pinning**: Locked to specific commit for reproducibility

## Troubleshooting

### Submodule not initialized

```bash
git submodule update --init external/MacroPlacement
```

### Import errors

Make sure TILOS dependencies are installed:
```bash
pip install -r external/requirements-tilos.txt
```

### No module named 'absl'

```bash
pip install absl-py
```

### Competition still works without it!

If you don't initialize the submodule, the competition will automatically fall back to the approximate evaluator. The TILOS evaluator is **only needed for official/ground-truth scoring**.
