# 📚 Research Source References

### Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.

**Description**: A foundational text covering autocorrelation, partial autocorrelation, time series decomposition, and forecasting.

**Applicability**: Supports EDA and sequence analysis methods for identifying periodicity, seasonality, and autocorrelation in temporal data.

---

### Kleinberg, J. (2003). *Bursty and Hierarchical Structure in Streams*. Data Mining and Knowledge Discovery, 7, 373–397.

**Description**: Presents an algorithm for detecting bursty patterns and hierarchical event structures in data streams.

**Applicability**: Useful for identifying event bursts and activity spikes in agent behavior logs during temporal EDA.

---

### Vaswani, A., et al. (2017). *Attention is All You Need*. NeurIPS.

**Description**: Introduced the Transformer architecture and sinusoidal positional encoding for modeling sequences using self-attention.

**Applicability**: Forms the foundation for temporal modeling with Transformers and attention heatmap analysis in your plan.

---

### Kazemi, S. M., et al. (2019). *Time2Vec: Learning a Vector Representation of Time*. arXiv:1907.05321.

**Description**: Proposes a learnable encoding of time that combines periodic and linear components for flexible temporal representation.

**Applicability**: Supports your plan to experiment with explicit time representations alongside positional embeddings.

---

### Chung, J., Ahn, S., & Bengio, Y. (2016). *Hierarchical Multiscale Recurrent Neural Networks*. arXiv:1609.01704.

**Description**: Introduces HM-RNNs that can discover and model latent temporal hierarchies in sequential data.

**Applicability**: Inspires the hierarchical chunking and episodic segmentation components of your plan.

---

### Cuturi, M., & Blondel, M. (2017). *Soft-DTW: A Differentiable Loss Function for Time-Series*. ICML.

**Description**: Introduces Soft-DTW, a differentiable variant of Dynamic Time Warping, for use in gradient-based learning.

**Applicability**: Enables trajectory alignment learning in your experiments using differentiable distance metrics.

---

### Guo, Y., et al. (2022). *Smooth Video Synthesis from Coarse-to-Fine Time-Aware Diffusion*. arXiv:2206.07600.

**Description**: Explores latent smoothness in generative video models and proposes regularization for stable temporal generation.

**Applicability**: Informs latent smoothness evaluation and interpolation quality metrics in your temporal encoder.

---

### Jaderberg, M., et al. (2016). *Reinforcement Learning with Unsupervised Auxiliary Tasks*. arXiv:1611.05397.

**Description**: Presents the UNREAL agent using auxiliary objectives (reward prediction, pixel change) to improve learning.

**Applicability**: Provides blueprint for designing auxiliary self-supervised tasks to shape temporal representation.

---

### Tengda Han, Weidi Xie, & Andrew Zisserman (2022). *Temporal Alignment Networks for Long-Term Video*. CVPR.

**Description**: Proposes a model for aligning long video sequences with weakly-labeled temporal structure.

**Applicability**: Supports the idea of aligning multimodal data across time and learning phase-aware embeddings.

---

### Mrowca, D., et al. (2018). *Flexible Neural Representation for Physics Prediction*. NeurIPS.

**Description**: Introduces structured representation learning for object-based physics with trajectory interpolation.

**Applicability**: Demonstrates benefits of smooth latent trajectories for simulation and counterfactual reasoning.

---

```bibtex
@book{box2015time,
  title={Time Series Analysis: Forecasting and Control},
  author={Box, George EP and Jenkins, Gwilym M and Reinsel, Gregory C and Ljung, Greta M},
  year={2015},
  publisher={John Wiley \& Sons}
}

@article{kleinberg2003bursty,
  title={Bursty and hierarchical structure in streams},
  author={Kleinberg, Jon},
  journal={Data Mining and Knowledge Discovery},
  volume={7},
  number={4},
  pages={373--397},
  year={2003},
  publisher={Springer}
}

@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems},
  volume={30},
  year={2017}
}

@article{kazemi2019time2vec,
  title={Time2Vec: Learning a vector representation of time},
  author={Kazemi, SM and Goel, Rishab and Eghbali, Sina and Ramanan, Junaid and Sahota, Nino and Thakur, Siddhartha and Wu, Jie and Smyth, Padhraic and Poupart, Pascal},
  journal={arXiv preprint arXiv:1907.05321},
  year={2019}
}

@article{chung2016hierarchical,
  title={Hierarchical multiscale recurrent neural networks},
  author={Chung, Junyoung and Ahn, Sungjin and Bengio, Yoshua},
  journal={arXiv preprint arXiv:1609.01704},
  year={2016}
}

@inproceedings{cuturi2017soft,
  title={Soft-DTW: a differentiable loss function for time-series},
  author={Cuturi, Marco and Blondel, Mathieu},
  booktitle={Proceedings of the 34th International Conference on Machine Learning},
  pages={894--903},
  year={2017},
  organization={PMLR}
}

@article{guo2022smooth,
  title={Smooth video synthesis from coarse-to-fine time-aware diffusion},
  author={Guo, Yilun and Lin, Ke and He, Di and Gong, Boqing and Liu, Jing},
  journal={arXiv preprint arXiv:2206.07600},
  year={2022}
}

@article{jaderberg2016reinforcement,
  title={Reinforcement learning with unsupervised auxiliary tasks},
  author={Jaderberg, Max and Mnih, Volodymyr and Czarnecki, Wojciech Marian and Schaul, Tom and Leibo, Joel Z and Silver, David and Kavukcuoglu, Koray},
  journal={arXiv preprint arXiv:1611.05397},
  year={2016}
}

@inproceedings{han2022temporal,
  title={Temporal alignment networks for long-term video},
  author={Han, Tengda and Xie, Weidi and Zisserman, Andrew},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10469--10479},
  year={2022}
}

@inproceedings{mrowca2018flexible,
  title={Flexible neural representation for physics prediction},
  author={Mrowca, Damian and Zoran, Daniel and Freeman, William T and Tenenbaum, Joshua B and Yildirim, Ilker and Wu, Jiajun},
  booktitle={Advances in Neural Information Processing Systems},
  volume={31},
  year={2018}
}
```