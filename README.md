# Probabilistic Downscaling of Climate Variables

Project with Colloquium (MA8114)  at TUM: Probabilistic Downscaling of Climate Variables Using Denoising Diffusion Probabilistic Models

Supervisor: Prof. Dr. Rüdiger Westermann (Chair of Computer Graphics and Visualization)\
Advisor: Kevin Höhlein (Chair of Computer Graphics and Visualization)

---

Downscaling combines methods that are used to infer high-resolution information from 
low-resolution climate variables. We approach this problem as an image super-resolution 
task and employ Denoising Diffusion Probabilistic Model to generate finer-scale variables 
conditioned on coarse-scale information. Experiments are conducted on WeatherBench dataset 
by analysing temperature at 2 m height above the surface variable. See the final report [here](https://github.com/davitpapikyan/Probabilistic-Downscaling-of-Climate-Variables/blob/main/report.pdf).

![](results/reverse_diffusion_steps.jpg?raw=true)

---

## References

- Liangwei Jiang (2021) Image-Super-Resolution-via-Iterative-Refinement [[Source code](https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement#readme)]
- Song et al. (2021) Score-Based Generative Modeling through Stochastic Differential Equations [[Source code](https://github.com/yang-song/score_sde_pytorch)]
- Stephan Rasp, Peter D. Dueben, Sebastian Scher, Jonathan A. Weyn, Soukayna Mouatadid, and Nils Thuerey, 2020. WeatherBench: A benchmark dataset for data-driven weather forecasting. arXiv: [WeatherBench: A benchmark dataset for data-driven weather forecasting
](https://arxiv.org/abs/2002.00469)
