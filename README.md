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
│   └── mesh_size_dependency.ipynb
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
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Martin Ladecký - [martin.ladecky@imtek.uni-freiburg.de](mailto:martin.ladecky@imtek.uni-freiburg.de)  
Martin Ladecký - [m.ladecky@gmail.com](mailto:m.ladecky@gmail.com)
