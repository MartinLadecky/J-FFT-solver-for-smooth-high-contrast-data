# Jacobi-accelerated FFT-based solver for smooth high-contrast data

[![arXiv](https://img.shields.io/badge/arXiv-2508.02613-b31b1b.svg)](https://arxiv.org/abs/2508.02613)

## Description

This repository contains the supplementary code and data for the paper:

> **Jacobi-accelerated FFT-based solver for smooth high-contrast data**  
> Martin Ladecký, Ivana Pultarová, François Bignonnet, Indre Jödicke, Jan Zeman, Lars Pastewka  
> *Preprint*, 2026  
> arXiv: [2508.02613](https://arxiv.org/abs/2508.02613)

## Repository Structure

```
├── README.md               # This file
├── LICENSE                 # License information
├── requirements.txt        # Python dependencies
├── notebooks/              # Jupyter notebooks
│   ├── introduction_conductivity.ipynb
│   ├── introduction_elasticity.ipynb
│   ├── sharp_vs_smooth_data.ipynb
│   ├── sharp_vs_smooth_data_variable_jacobi.ipynb
│   ├── mesh_size_dependency.ipynb
│   ├── computational_time.ipynb
│   ├── power_law_elasticity.ipynb
│   └── power_law_elasticity_de_Geus.ipynb
├── data/                   # Microstructure data (.npy files)
└── figures/                # Generated figures
```

## Requirements
 - See `requirements.txt` for package dependencies
 

## Usage

The notebooks demonstrate the performance of Green, Jacobi, and Green-Jacobi preconditioners:

1. `notebooks/introduction_conductivity.ipynb` - Application to 2D steady-state conductivity.
2. `notebooks/introduction_elasticity.ipynb` - Application to 2D linear elasticity.
3. `notebooks/sharp_vs_smooth_data.ipynb` - Comparison using auxetic microstructure with sharp vs. smooth transitions.
4. `notebooks/sharp_vs_smooth_data_variable_jacobi.ipynb` - Investigation of Jacobi fallback stability at extreme contrasts.
5. `notebooks/mesh_size_dependency.ipynb` - Evaluation of PCG iteration count vs. grid size for sharp and smooth geometries.
6. `notebooks/computational_time.ipynb` - Computational time comparison across different preconditioners and problem sizes.
7. `notebooks/power_law_elasticity.ipynb` - Nonlinear power-law elasticity problems with preconditioner performance analysis.
8. `notebooks/power_law_elasticity_de_Geus.ipynb` - Power-law elasticity examples adapted from de Geus methodology.

## Data

The `data/` directory contains auxetic microstructure geometries in `.npy` format (64x64 and 1024x1024 resolutions) used in the notebooks, as well as several cosine-based geometries.

## Citation

If you use this code, please cite:

```bibtex
@misc{ladecký2026jacobiacceleratedfftbasedsolversmooth,
      title={Jacobi-accelerated FFT-based solver for smooth high-contrast data}, 
      author={Martin Ladecký and Ivana Pultarová and François Bignonnet and Indre Jödicke and Jan Zeman and Lars Pastewka},
      year={2026},
      eprint={2508.02613},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/2508.02613}, 
}

## Acknowledgments

This code was partially adapted/motivated by:

- **GooseFFT** by T.W.J. de Geus and J. Vondřejc
- Original source: https://github.com/tdegeus/GooseFFT
- Original license: MIT License, Copyright (c) 2016 T.W.J. de Geus
- de Geus et al., Comput. Methods Appl. Mech. Eng., 2017, 318:412–430
- https://doi.org/10.1016/j.cma.2016.12.032

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Martin Ladecký - [martin.ladecky@imtek.uni-freiburg.de](mailto:martin.ladecky@imtek.uni-freiburg.de)  
Martin Ladecký - [m.ladecky@gmail.com](mailto:m.ladecky@gmail.com)

## Funding

This development has received funding from the European Commission (Marie Sklodowska-Curie Fellowship 101106585 — microFFTTO).


