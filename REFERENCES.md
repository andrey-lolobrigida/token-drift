# References

## Checked against our results

- Kawin Ethayarajh. 2019. **How Contextual are Contextualized Word Representations?
  Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings.** EMNLP-IJCNLP 2019,
  pages 55–65. [ACL Anthology](https://aclanthology.org/D19-1006/)
  — Anisotropy by layer. **Reproduces**, GPT-2 almost point for point (FINDINGS §4).

- Elena Voita, Rico Sennrich, Ivan Titov. 2019. **The Bottom-up Evolution of
  Representations in the Transformer: A Study with Machine Translation and Language
  Modeling Objectives.** EMNLP-IJCNLP 2019, pages 4396–4406.
  [ACL Anthology](https://aclanthology.org/D19-1448/)
  — Frequent tokens' representations change more across layers. **Doesn't reproduce**,
  context-free or corpus-averaged (FINDINGS §5, §11).

## Background reading (not compared against)

- Emily Cheng, Diego Doimo, Corentin Kervadec, Iuri Macocco, Lei Yu, Alessandro Laio,
  Marco Baroni. 2025. **Emergence of a High-Dimensional Abstraction Phase in Language
  Transformers.** ICLR 2025. [OpenReview](https://openreview.net/forum?id=0fD3iIBhlV)

- Karthik Viswanathan, Yuri Gardinazzi, Giada Panerai, Alberto Cazzaniga, Matteo
  Biagetti. 2026. **The Intrinsic Dimension of Prompts in Internal Representations of
  Large Language Models.** TMLR. [OpenReview](https://openreview.net/forum?id=rBEgNAslpY)
  (earlier arXiv versions: *The Geometry of Tokens in Internal Representations of Large
  Language Models*)

## Models, data and tools

- Stella Biderman, Hailey Schoelkopf, Quentin Anthony, Herbie Bradley, Kyle O'Brien, Eric
  Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, Aviya
  Skowron, Lintang Sutawika, Oskar van der Wal. 2023. **Pythia: A Suite for Analyzing Large
  Language Models Across Training and Scaling.** ICML 2023.
  [arXiv:2304.01373](https://arxiv.org/abs/2304.01373)
  — The model (`EleutherAI/pythia-70m`) and its public training checkpoints (v2).

- Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever. 2019.
  **Language Models are Unsupervised Multitask Learners.** OpenAI technical report.
  — GPT-2 small, the second model.

- Leo Gao, Stella Biderman, Sid Black, et al. 2020. **The Pile: An 800GB Dataset of Diverse
  Text for Language Modeling.** [arXiv:2101.00027](https://arxiv.org/abs/2101.00027)
  — v1 text: [`NeelNanda/pile-10k`](https://huggingface.co/datasets/NeelNanda/pile-10k) (the
  first 10k Pile documents); probe runs: shards of
  [`EleutherAI/the_pile_deduplicated`](https://huggingface.co/datasets/EleutherAI/the_pile_deduplicated).

- [Project Gutenberg](https://www.gutenberg.org/): the 10 moral-philosophy books of the probe
  runs (ids in `configs/pythia70m_probe.yaml`).

- Simon Kornblith, Mohammad Norouzi, Honglak Lee, Geoffrey Hinton. 2019. **Similarity of
  Neural Network Representations Revisited.** ICML 2019.
  [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)
  — CKA, the whole-matrix similarity score.

- Leland McInnes, John Healy, James Melville. 2018. **UMAP: Uniform Manifold Approximation
  and Projection for Dimension Reduction.** [arXiv:1802.03426](https://arxiv.org/abs/1802.03426)
  — The 2D maps; `AlignedUMAP` from [`umap-learn`](https://github.com/lmcinnes/umap) keeps the
  flipbook frames comparable.
