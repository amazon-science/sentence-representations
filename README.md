## Sentence Representations Learning with Transformers

This framework provides implementations of our models developed for sentence representation learning.  The following publications are implemented in this repo,

- DSE [Learning Dialogue Representations from Consecutive Utterances (NAACL 2022)](https://www.amazon.science/publications/learning-dialogue-representations-from-consecutive-utterances). Checkout our implementations [here](https://github.com/amazon-research/dse)

- VaSCL [Virtual Augmentation Supported Contrastive Learning of Sentence Representations (Findings of ACL 2022)](https://arxiv.org/abs/2110.08552)  

- PairSupCon  [Pairwise Supervised Contrastive Learning of Sentence Representations (EMNLP 2021)](https://aclanthology.org/2021.emnlp-main.467/)

- SCCL [Supporting Clustering with Contrastive Learning (NAACL 2021)](https://aclanthology.org/2021.naacl-main.427.pdf). (A contrastive learning supported text clustering approach, which can be leveraged for learning both dense and categorical representations. Checkout our implementation [here](https://github.com/amazon-research/sccl))

Our checkpoints can be loaded from [HuggingFace Model Hub.](https://huggingface.co/aws-ai)



If you find this repository helpful, feel free to cite the associated publications:
```bibtex
@inproceedings{zhang-etal-2022-virtual,
    title = "Virtual Augmentation Supported Contrastive Learning of Sentence Representations",
    author = "Zhang, Dejiao  and
      Xiao, Wei  and
      Zhu, Henghui  and
      Ma, Xiaofei  and
      Arnold, Andrew",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-acl.70",
    doi = "10.18653/v1/2022.findings-acl.70",
    pages = "864--876",
    abstract = "Despite profound successes, contrastive representation learning relies on carefully designed data augmentations using domain-specific knowledge. This challenge is magnified in natural language processing, where no general rules exist for data augmentation due to the discrete nature of natural language. We tackle this challenge by presenting a Virtual augmentation Supported Contrastive Learning of sentence representations (VaSCL). Originating from the interpretation that data augmentation essentially constructs the neighborhoods of each training instance, we, in turn, utilize the neighborhood to generate effective data augmentations. Leveraging the large training batch size of contrastive learning, we approximate the neighborhood of an instance via its K-nearest in-batch neighbors in the representation space. We then define an instance discrimination task regarding the neighborhood and generate the virtual augmentation in an adversarial training manner. We access the performance of VaSCL on a wide range of downstream tasks and set a new state-of-the-art for unsupervised sentence representation learning.",
}
```


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



Contact person: [Dejiao Zhang](https://www.amazon.science/author/deijao-zhang), [dejiaoz@amazon.com](dejiaoz@amazon.com)




## License

This project is licensed under the Apache-2.0 License.

