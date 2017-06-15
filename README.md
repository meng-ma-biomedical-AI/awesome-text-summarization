### Extractive Text Summarization
1. Shashi Narayan, Nikos Papasarantopoulos, Mirella Lapata, Shay B. Cohen. [Neural Extractive Summarization with Side Information](https://arxiv.org/abs/1704.04530). 2017.

### Abstractive Text Summarization
1. Alexander M. Rush, Sumit Chopra, Jason Weston. [A Neural Attention Model for Abstractive Sentence Summarization](https://arxiv.org/abs/1509.00685). EMNLP, 2015. The source code in LUA Torch7 is [NAMAS](https://github.com/facebook/NAMAS).
   * They use sequence-to-sequence encoder-decoder LSTM with attention.
   * Ihey use the first sentence of a document. The source document is quite small (about 1 paragraph or ~500 words in the training dataset of Gigaword) and the produced output is also very short (about 75 characters). It remains an open challenge to scale up these limits - to produce longer summaries over multi-paragraph text input (even good LSTM models with attention models fall victim to vanishing gradients when the input sequences become longer than a few hundred items).
   * The evaluation method used for automatic summarization has traditionally been the ROUGE metric - which has been shown to correlate well with human judgment of summary quality, but also has a known tendency to encourage "extractive" summarization - so that using ROUGE as a target metric to optimize will lead a summarizer towards a copy-paste behavior of the input instead of the hoped-for reformulation type of summaries.
1. Peter Liu and Xin Pan. [Sequence-to-Sequence with Attention Model for Text Summarization](https://research.googleblog.com/2016/08/text-summarization-with-tensorflow.html). 2016. The source code in Python is [textsum](https://github.com/tensorflow/models/tree/master/textsum).
   * They use sequence-to-sequence encoder-decoder LSTM with attention and bidirectional neural net.
   * They use the first 2 sentences of a document with a limit at 120 words.
   * The scores achieved by Google’s *textsum* are 42.57 ROUGE-1 and 23.13 ROUGE-2.
1. Ramesh Nallapati, Bowen Zhou, Cicero Nogueira dos santos, Caglar Gulcehre, Bing Xiang. [Abstractive Text Summarization Using Sequence-to-Sequence RNNs and Beyond](https://arxiv.org/abs/1602.06023). 2016.
   * They use GRU with attention and bidirectional neural net.
   * They use the first 2 sentences of a documnet with a limit at 120 words.
   * They use the [Large vocabulary trick (LVT)](https://arxiv.org/abs/1412.2007) of Jean et al. 2014, which means when you decode, use only the words that appear in the source - this reduces perplexity. But then you lose the capability to do "abstractive" summary. So they do "vocabulary expansion" by adding a layer of "word2vec nearest neighbors" to the words in the input.
   * Feature rich encoding - they add TF*IDF and Named Entity types to the word embeddings (concatenated) to the encodings of the words - this adds to the encoding dimensions that reflect "importance" of the words.
   * The most interesting of all is what they call the "Switching Generator/Pointer" layer. In the decoder, they add a layer that decides to either generate a new word based on the context / previously generated word (usual decoder) or copy a word from the input (that is - add a pointer to the input). They learn when to do Generate vs. Pointer and when it is a Pointer which word of the input to Point to.
1. Konstantin Lopyrev. [Generating News Headlines with Recurrent Neural Networks](https://arxiv.org/abs/1512.01712). 2015. The source code in Python is [headlines](https://github.com/udibr/headlines).
1. Sumit Chopra, Alexander M. Rush and Michael Auli. [Abstractive Sentence Summarization with Attentive Recurrent Neural Networks](http://harvardnlp.github.io/papers/naacl16_summary.pdf). NAACL, 2016.
1. Jianpeng Cheng, Mirella Lapata. [Neural Summarization by Extracting Sentences and Words](https://arxiv.org/abs/1603.07252)". ACL, 2016.
   * This paper use attention as a mechanism for identifying the best sentences to extract, and then go beyond that to generate an abstractive summary.
1. Romain Paulus, Caiming Xiong, Richard Socher. [A Deep Reinforced Model for Abstractive Summarization](https://metamind.io/static/pdf/deep-reinforced-model-arxiv-v1.pdf). 2017.
1. Shibhansh Dohare, Harish Karnick. [Text Summarization using Abstract Meaning Representation](https://arxiv.org/abs/1706.01678). 	arXiv:1706.01678, 2017.

### Text Summarization
  - Ryang, Seonggi, and Takeshi Abekawa. "[Framework of automatic text summarization using reinforcement learning](http://dl.acm.org/citation.cfm?id=2390980)." In Proceedings of the 2012 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning, pp. 256-265. Association for Computational Linguistics, 2012. [not neural-based methods]
  - King, Ben, Rahul Jha, Tyler Johnson, Vaishnavi Sundararajan, and Clayton Scott. "[Experiments in Automatic Text Summarization Using Deep Neural Networks](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.459.8775&rep=rep1&type=pdf)." Machine Learning (2011).
  - Liu, Yan, Sheng-hua Zhong, and Wenjie Li. "[Query-Oriented Multi-Document Summarization via Unsupervised Deep Learning](http://www.aaai.org/ocs/index.php/AAAI/AAAI12/paper/view/5058/5322)." AAAI. 2012.
  - Rioux, Cody, Sadid A. Hasan, and Yllias Chali. "[Fear the REAPER: A System for Automatic Multi-Document Summarization with Reinforcement Learning](http://emnlp2014.org/papers/pdf/EMNLP2014075.pdf)." In EMNLP, pp. 681-690. 2014.[not neural-based methods]
  - PadmaPriya, G., and K. Duraiswamy. "[An Approach For Text Summarization Using Deep Learning Algorithm](http://thescipub.com/PDF/jcssp.2014.1.9.pdf)." Journal of Computer Science 10, no. 1 (2013): 1-9.
  - Denil, Misha, Alban Demiraj, and Nando de Freitas. "[Extraction of Salient Sentences from Labelled Documents](http://arxiv.org/abs/1412.6815)." arXiv preprint arXiv:1412.6815 (2014).
  - Kågebäck, Mikael, et al. "[Extractive summarization using continuous vector space models](http://www.aclweb.org/anthology/W14-1504)." Proceedings of the 2nd Workshop on Continuous Vector Space Models and their Compositionality (CVSC)@ EACL. 2014.
  - Denil, Misha, Alban Demiraj, Nal Kalchbrenner, Phil Blunsom, and Nando de Freitas. "[Modelling, Visualising and Summarising Documents with a Single Convolutional Neural Network](http://arxiv.org/abs/1406.3830)." arXiv preprint arXiv:1406.3830 (2014).
  - Cao, Ziqiang, Furu Wei, Li Dong, Sujian Li, and Ming Zhou. "[Ranking with Recursive Neural Networks and Its Application to Multi-document Summarization](http://gana.nlsde.buaa.edu.cn/~lidong/aaai15-rec_sentence_ranking.pdf)." (AAAI'2015).
  - Fei Liu, Jeffrey Flanigan, Sam Thomson, Norman Sadeh, and Noah A. Smith. "[Toward Abstractive Summarization Using Semantic Representations](http://www.cs.cmu.edu/~nasmith/papers/liu+flanigan+thomson+sadeh+smith.naacl15.pdf)." NAACL 2015
  - Wenpeng Yin， Yulong Pei. "Optimizing Sentence Modeling and Selection for Document Summarization." IJCAI 2015
  - He, Zhanying, Chun Chen, Jiajun Bu, Can Wang, Lijun Zhang, Deng Cai, and Xiaofei He. "[Document Summarization Based on Data Reconstruction](http://cs.nju.edu.cn/zlj/pdf/AAAI-2012-He.pdf)." In AAAI. 2012.
  - Liu, He, Hongliang Yu, and Zhi-Hong Deng. "[Multi-Document Summarization Based on Two-Level Sparse Representation Model](http://www.cis.pku.edu.cn/faculty/system/dengzhihong/papers/AAAI%202015_Multi-Document%20Summarization%20Based%20on%20Two-Level%20Sparse%20Representation%20Model.pdf)." In Twenty-Ninth AAAI Conference on Artificial Intelligence. 2015.
  - Jin-ge Yao, Xiaojun Wan, Jianguo Xiao. "[Compressive Document Summarization via Sparse Optimization](http://ijcai.org/Proceedings/15/Papers/198.pdf)." IJCAI 2015
  - Piji Li, Lidong Bing, Wai Lam, Hang Li, and Yi Liao. "[Reader-Aware Multi-Document Summarization via Sparse Coding](http://arxiv.org/abs/1504.07324)." IJCAI 2015.
  - Hu, Baotian, Qingcai Chen, and Fangze Zhu. "[LCSTS: a large scale chinese short text summarization dataset](http://arxiv.org/abs/1506.05865)." arXiv preprint arXiv:1506.05865 (2015).
  - Gulcehre, Caglar, Sungjin Ahn, Ramesh Nallapati, Bowen Zhou, and Yoshua Bengio. "[Pointing the Unknown Words](http://arxiv.org/abs/1603.08148)." arXiv preprint arXiv:1603.08148 (2016).
  - Jiatao Gu, Zhengdong Lu, Hang Li, Victor O.K. Li. "[Incorporating Copying Mechanism in Sequence-to-Sequence Learning](http://arxiv.org/abs/1603.06393)." ACL. (2016)
  - Zhang, Jianmin, Jin-ge Yao, and Xiaojun Wan. "[Toward constructing sports news from live text commentary](http://www.icst.pku.edu.cn/lcwm/wanxj/files/acl16_sports.pdf)." In Proceedings of ACL. 2016.
  - Ziqiang Cao, Wenjie Li, Sujian Li, Furu Wei. "[AttSum: Joint Learning of Focusing and Summarization with Neural Attention](http://arxiv.org/abs/1604.00125)".  arXiv:1604.00125 (2016)
  - Ayana, Shiqi Shen, Zhiyuan Liu, Maosong Sun. "[Neural Headline Generation with Sentence-wise Optimization](http://arxiv.org/abs/1604.01904)". arXiv:1604.01904 (2016)
  - Kikuchi, Yuta, Graham Neubig, Ryohei Sasano, Hiroya Takamura, and Manabu Okumura. "[Controlling Output Length in Neural Encoder-Decoders](https://arxiv.org/abs/1609.09552)." arXiv preprint arXiv:1609.09552 (2016).
  - Qian Chen, Xiaodan Zhu, Zhenhua Ling, Si Wei and Hui Jiang. "[Distraction-Based Neural Networks for Document Summarization](https://arxiv.org/abs/1610.08462)." IJCAI 2016.
  - Wang, Lu, and Wang Ling. "[Neural Network-Based Abstract Generation for Opinions and Arguments](http://www.ccs.neu.edu/home/luwang/papers/NAACL2016.pdf)." NAACL 2016.
  - Yishu Miao, Phil Blunsom. "[Language as a Latent Variable: Discrete Generative Models for Sentence Compression](http://arxiv.org/abs/1609.07317)." EMNLP 2016.
  - Takase, Sho, Jun Suzuki, Naoaki Okazaki, Tsutomu Hirao, and Masaaki Nagata. "[Neural headline generation on abstract meaning representation](https://www.aclweb.org/anthology/D/D16/D16-1112.pdf)." EMNLP, pp. 1054-1059. 2016.
  - Hongya Song, Zhaochun Ren, Piji Li, Shangsong Liang, Jun Ma, and Maarten de Rijke. [Summarizing Answers in Non-Factoid Community Question-Answering](http://dl.acm.org/citation.cfm?id=3018704). In WSDM 2017: The 10th International Conference on Web Search and Data Mining, 2017.
  - Wenyuan Zeng, Wenjie Luo, Sanja Fidler, Raquel Urtasun. "[Efficient Summarization with Read-Again and Copy Mechanism](https://arxiv.org/abs/1611.03382)." arXiv preprint arXiv:1611.03382 (2016).
  - Piji Li, Zihao Wang, Wai Lam, Zhaochun Ren, Lidong Bing. "[Salience Estimation via Variational Auto-Encoders for Multi-Document Summarization](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14613)". In AAAI, 2017.
  - Ramesh Nallapati, Feifei Zhai, Bowen Zhou. [SummaRuNNer: A Recurrent Neural Network based Sequence Model for Extractive Summarization of Documents](https://arxiv.org/abs/1611.04230). In AAAI, 2017.
  - Ramesh Nallapati, Bowen Zhou, Mingbo Ma. "[Classify or Select: Neural Architectures for Extractive Document Summarization](https://arxiv.org/abs/1611.04244)." arXiv preprint arXiv:1611.04244 (2016).
  - Suzuki, Jun, and Masaaki Nagata. "[Cutting-off Redundant Repeating Generations for Neural Abstractive Summarization](http://www.aclweb.org/anthology/E17-2047)." EACL 2017 (2017): 291.
  - Jiwei Tan and Xiaojun Wan. [Abstractive Document Summarization with a Graph-Based Attentional Neural Model](). ACL, 2017.
  - Preksha Nema, Mitesh M. Khapra, Balaraman Ravindran and Anirban Laha. [Diversity driven attention model for query-based abstractive summarization](). ACL,2017
  - Abigail See, Peter J. Liu and Christopher D. Manning. [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368). ACL, 2017.
  - Qingyu Zhou, Nan Yang, Furu Wei and Ming Zhou. [Selective Encoding for Abstractive Sentence Summarization](https://arxiv.org/abs/1704.07073). ACL, 2017
  - Maxime Peyrard and Judith Eckle-Kohler. [Supervised Learning of Automatic Pyramid for Optimization-Based Multi-Document Summarization](). ACL, 2017.

### Opinion Summarization
1. Ganesan, Kavita A., Zhai ChengXiang, and Han Jiawei. [Opinosis: A Graph Based Approach to Abstractive Summarization of Highly Redundant Opinions](http://kavita-ganesan.com/opinosis). Proceedings of the 23rd International Conference on Computational Linguistics (COLING '10), 2010.
1. Wu, Haibing, Yiwei Gu, Shangdi Sun, and Xiaodong Gu. [Aspect-based Opinion Summarization with Convolutional Neural Networks](http://arxiv.org/abs/1511.09128). arXiv preprint arXiv:1511.09128 (2015).
1. Irsoy, Ozan, and Claire Cardie. [Opinion Mining with Deep Recurrent Neural Networks](http://anthology.aclweb.org/D/D14/D14-1080.pdf). In EMNLP, pp. 720-728. 2014.
1. Piji Li, Zihao Wang, Zhaochun Ren, Lidong Bing, Wai Lam. [Neural Rating Regression with Abstractive Tips Generation for Recommendation](). In SIGIR, pp xx-xx. 2017.

### Reading Comprehension
 - Hermann, Karl Moritz, Tomas Kocisky, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom. "[Teaching machines to read and comprehend](http://papers.nips.cc/paper/5945-teaching-machines-to-read-and-comprehend)." In Advances in Neural Information Processing Systems, pp. 1693-1701. 2015.
 - Hill, Felix, Antoine Bordes, Sumit Chopra, and Jason Weston. "[The Goldilocks Principle: Reading Children's Books with Explicit Memory Representations](http://arxiv.org/abs/1511.02301)." arXiv preprint arXiv:1511.02301 (2015).
 - Kadlec, Rudolf, Martin Schmid, Ondrej Bajgar, and Jan Kleindienst. "[Text Understanding with the Attention Sum Reader Network](http://arxiv.org/abs/1603.01547)." arXiv preprint arXiv:1603.01547 (2016).
 - Chen, Danqi, Jason Bolton, and Christopher D. Manning. "[A thorough examination of the cnn/daily mail reading comprehension task](http://arxiv.org/abs/1606.02858)." arXiv preprint arXiv:1606.02858 (2016).
 - Dhingra, Bhuwan, Hanxiao Liu, William W. Cohen, and Ruslan Salakhutdinov. "[Gated-Attention Readers for Text Comprehension](http://arxiv.org/abs/1606.01549)." arXiv preprint arXiv:1606.01549 (2016).
 - Sordoni, Alessandro, Phillip Bachman, and Yoshua Bengio. "[Iterative Alternating Neural Attention for Machine Reading](http://arxiv.org/abs/1606.02245)." arXiv preprint arXiv:1606.02245 (2016).
 - Trischler, Adam, Zheng Ye, Xingdi Yuan, and Kaheer Suleman. "[Natural Language Comprehension with the EpiReader](http://arxiv.org/abs/1606.02270)." arXiv preprint arXiv:1606.02270 (2016).
 - Yiming Cui, Zhipeng Chen, Si Wei, Shijin Wang, Ting Liu, Guoping Hu. "[Attention-over-Attention Neural Networks for Reading Comprehension](http://arxiv.org/abs/1607.04423)." arXiv preprint arXiv:1607.04423 (2016).
 - Yiming Cui, Ting Liu, Zhipeng Chen, Shijin Wang, Guoping Hu. "[Consensus Attention-based Neural Networks for Chinese Reading Comprehension](https://arxiv.org/abs/1607.02250)." arXiv preprint arXiv:1607.02250 (2016).
 - Daniel Hewlett, Alexandre Lacoste, Llion Jones, Illia Polosukhin, Andrew Fandrianto, Jay Han, Matthew Kelcey and David Berthelot. "[WIKIREADING: A Novel Large-scale Language Understanding Task over Wikipedia](http://www.aclweb.org/anthology/P/P16/P16-1145.pdf)." ACL (2016). pp. 1535-1545.
  - Minghao Hu, Yuxing Peng, Xipeng Qiu. "[Mnemonic Reader for Machine Comprehension](https://arxiv.org/abs/1705.02798)." arXiv:1705.02798 (2017).
  - Wenhui Wang, Nan Yang, Furu Wei, Baobao Chang and Ming Zhou. "[R-NET: Machine Reading Comprehension with Self-matching Networks](https://www.microsoft.com/en-us/research/publication/mcr/)." ACL (2017).

### Sentence Modelling
  - Kalchbrenner, Nal, Edward Grefenstette, and Phil Blunsom. "[A convolutional neural network for modelling sentences](http://arxiv.org/abs/1404.2188)." arXiv preprint arXiv:1404.2188 (2014).
  - Kim, Yoon. "[Convolutional neural networks for sentence classification](http://arxiv.org/abs/1408.5882)." arXiv preprint arXiv:1408.5882 (2014).
  - Le, Quoc V., and Tomas Mikolov. "[Distributed representations of sentences and documents](http://arxiv.org/abs/1405.4053)." arXiv preprint arXiv:1405.4053 (2014).
  - Yang, Zichao, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, and Eduard Hovy. "[Hierarchical Attention Networks for Document Classification](http://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)." In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. 2016.
