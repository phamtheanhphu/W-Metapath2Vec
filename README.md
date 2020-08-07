# W-Metapath2Vec

This is the source code of *W-Metapath2Vec* which is a topic-driven meta-path-based network representation learning technique for the content-based heterogeneous information network (C-HIN).

Our contributions in the *W-Metapath2Vec* model are mainly inspired from previous work of Yuxiao Dong et al. with the proposed Metapath2Vec model (2017) in this paper:
```
Dong, Y., Chawla, N. V., & Swami, A. (2017, August). metapath2vec: Scalable representation learning for heterogeneous networks. In Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining (pp. 135-144).
```

## Requirements
- Python >= 3.6
- Gensim >= 3.8.3
- NetworkX >= 2.3
- Numpy >= 1.17

## Dataset usage
- The DBLP bibliographic network (https://dblp.uni-trier.de/) is the main experimental dataset. 
- The topic distributions of papers are extracted from the papers' abstracts (retrieved from Aminer repository: https://www.aminer.org/) by using the LDA topic modelling of the Scikit-Learn library (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html) 

## Citing
If you find W-Metapath2Vec algorithm is useful in your researches, please cite the following paper:

    @article{pham2019w,
      title={W-MetaPath2Vec: The topic-driven meta-path-based model for large-scaled content-based heterogeneous information network representation learning},
      author={Pham, Phu and Do, Phuc},
      journal={Expert Systems with Applications},
      volume={123},
      pages={328--344},
      year={2019},
      publisher={Elsevier}
    }

## Miscellaneous

Please send any question you might have about the code and/or the algorithm to <phamtheanhphu@gmail.com>.