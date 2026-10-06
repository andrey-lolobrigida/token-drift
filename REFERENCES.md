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
