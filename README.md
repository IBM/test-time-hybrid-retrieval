# Multimodal Hybrid Retrieval with Guided Query Refinement (GQR)

This repository contains the official code for the paper:

**Guided Query Refinement: Multimodal Hybrid Retrieval with Test-Time Optimization**, *Uzan et al., ICLR 2026*  [arXiv:2510.05038](https://arxiv.org/abs/2510.05038)

Guided Query Refinement (GQR) is a **test-time optimization** method for multimodal retrieval.  
It improves a *primary* retriever by refining its query representation using similarity signals from a *complementary* retriever, enabling stronger hybrid retrieval without additional training.

---

## 📂 Project Structure

The repository is organized into two main components:

1. **GQR-Tutorial**  
   A lightweight walkthrough introducing the retrieval setting, baselines, and the GQR method.  
   We recommend starting here to build intuition for the approach.

2. **paper-repro**  
   Code, configurations, and scripts used to reproduce the experiments and results reported in the paper.

---

## 📜 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{uzan2026guidedqueryrefinement,
  title     = {Guided Query Refinement: Multimodal Hybrid Retrieval with Test-Time Optimization},
  author    = {Uzan, Omri and Yehudai, Asaf and Pony, Roi and Shnarch, Eyal and Gera, Ariel},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2510.05038}
}
