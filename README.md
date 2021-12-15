## Sentence Representation Learning with Transformers

This framework provides implementations of our models developped for sentence representation learning.  The following publications are implemented in this repo,

- VaSCL [Virtual Augmentation Supported Contrastive Learning of Sentence Representations](https://arxiv.org/abs/2110.08552) (Coming SOON!) 

- PairSupCon  [Pairwise Supervised Contrastive Learning of Sentence Representations (EMNLP 2021)](https://aclanthology.org/2021.emnlp-main.467/)

- SCCL [Supporting Clustering with Contrastive Learning](https://aclanthology.org/2021.naacl-main.427.pdf). (A contrastive learning supported text clustering framework, which can be leveraged for learning both dense representations or categorical representations. Checkout our implementations here https://github.com/amazon-research/sccl)




If you find this repository helpful, feel free to cite the associated publications:

```bibtex 
@inproceedings{zhang-etal-2021-pairwise,
title = "Pairwise Supervised Contrastive Learning of Sentence Representations",
author = "Zhang, Dejiao  and
  Li, Shang-Wen  and
  Xiao, Wei  and
  Zhu, Henghui  and
  Nallapati, Ramesh  and
  Arnold, Andrew O.  and
  Xiang, Bing",
booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
month = nov,
year = "2021",
address = "Online and Punta Cana, Dominican Republic",
publisher = "Association for Computational Linguistics",
url = "https://aclanthology.org/2021.emnlp-main.467",
pages = "5786--5798",
abstract = "Many recent successes in sentence representation learning have been achieved by simply fine-tuning on the Natural Language Inference (NLI) datasets with triplet loss or siamese loss. Nevertheless, they share a common weakness: sentences in a contradiction pair are not necessarily from different semantic categories. Therefore, optimizing the semantic entailment and contradiction reasoning objective alone is inadequate to capture the high-level semantic structure. The drawback is compounded by the fact that the vanilla siamese or triplet losses only learn from individual sentence pairs or triplets, which often suffer from bad local optima. In this paper, we propose PairSupCon, an instance discrimination based approach aiming to bridge semantic entailment and contradiction understanding with high-level categorical concept encoding. We evaluate PairSupCon on various downstream tasks that involve understanding sentence semantics at different granularities. We outperform the previous state-of-the-art method with 10{\%}{--}13{\%} averaged improvement on eight clustering tasks, and 5{\%}{--}6{\%} averaged improvement on seven semantic textual similarity (STS) tasks."}
````



```bibtex
@article{zhang2021virtual,
  title={Virtual Augmentation Supported Contrastive Learning of Sentence Representations},
  author={Zhang, Dejiao and Xiao, Wei and Zhu, Henghui and Ma, Xiaofei and Arnold, Andrew O},
  journal={arXiv preprint arXiv:2110.08552},
  year={2021}
}
```



Contact person: [Dejiao Zhang](https://www.amazon.science/author/deijao-zhang), [dejiaoz@amazon.com](dejiaoz@amazon.com)




## License

This project is licensed under the Apache-2.0 License.

