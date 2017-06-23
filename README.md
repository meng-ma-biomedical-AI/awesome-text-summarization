### Corpus
1. [Opinosis dataset](http://kavita-ganesan.com/opinosis-opinion-dataset) contains 51 articles. Each article is about a product’s feature, like iPod’s Battery Life, etc. and is a collection of reviews by customers who purchased that product. Each article in the dataset has 5 manually written “gold” summaries. Usually the 5 gold summaries are different but they can also be the same text repeated 5 times.
1. [DUC](http://duc.nist.gov/)
1. [English Gigaword](https://catalog.ldc.upenn.edu/LDC2003T05): English Gigaword was produced by Linguistic Data Consortium (LDC).
1. [CNN and Daily Mail](http://cs.nyu.edu/~kcho/DMQA/) or [github](https://github.com/deepmind/rc-data).
   * *CNN* contains the documents and accompanying questions from the news articles of CNN. There are approximately 90k documents and 380k questions.
   * *Daily Mail* contains the documents and accompanying questions from the news articles of Daily Mail. There are approximately 197k documents and 879k questions.
1. [Processed CNN and Daily Mail](https://github.com/danqi/rc-cnn-dailymail) datasets are just simply concatenation of all data instances and keeping document, question and answer only for their inputs.
1. [Large Scale Chinese Short Text Summarization Dataset（LCSTS）](http://icrc.hitsz.edu.cn/Article/show/139.html): This corpus is constructed from the Chinese microblogging website SinaWeibo. It consists of over 2 million real Chinese short texts with short summaries given by the writer of each text.

### Text Summarization Software
1. [sumy](https://github.com/miso-belica/sumy) is a simple library and command line utility for extracting summary from HTML pages or plain texts. The package also contains simple evaluation framework for text summaries. Implemented summarization methods are *Luhn*, *Edmundson*, *LSA*, *LexRank*, *TextRank*, *SumBasic* and *KL-Sum*.
1. [TextRank4ZH](https://github.com/letiantian/TextRank4ZH) implements the *TextRank* algorithm to extract key words/phrases and text summarization
in Chinese. It is written in Python.
1. [snownlp](https://github.com/isnowfy/snownlp) is python library for processing Chinese text.
1. [PKUSUMSUM](https://github.com/PKULCWM/PKUSUMSUM) is an integrated toolkit for automatic document summarization. It supports single-document, multi-document and topic-focused multi-document summarizations, and a variety of summarization methods have been implemented in the toolkit. It supports Western languages (e.g. English) and Chinese language.
1. [fnlp](https://github.com/FudanNLP/fnlp) is a toolkit for Chinese natural language processing.

### Word/Sentence Representation
1.  [N-Grams](https://lagunita.stanford.edu/c4x/Engineering/CS-224N/asset/slp4.pdf)
1. Yoshua Bengio, Réjean Ducharme, Pascal Vincent and Christian Jauvin. [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf). 2003.
   * They proposed to fight the curse of dimensionality by learning a distributed representation for words which allows each training sentence to inform the model about an exponential number of semantically neighboring sentences.
1. Ryan Kiros, Yukun Zhu, Ruslan Salakhutdinov, Richard S. Zemel, Antonio Torralba, Raquel Urtasun and Sanja Fidler. [Skip-Thought Vectors](https://arxiv.org/abs/1506.06726). 2015. The source code in Python is [skip-thoughts](https://github.com/ryankiros/skip-thoughts).

#### word2vec
1. [Word2Vec Resources](http://mccormickml.com/2016/04/27/word2vec-resources/): This is a post with links to and descriptions of word2vec tutorials, papers, and implementations.

### Extractive Text Summarization
1. H. P. Luhn. [The automatic creation of literature abstracts](http://courses.ischool.berkeley.edu/i256/f06/papers/luhn58.pdf). IBM Journal of Research and Development, 1958. Luhn's method is as follows:
   1. Ignore Stopwords: Common words (known as stopwords) are ignored.
   1. Determine Top Words: The most often occuring words in the document are counted up.
   1. Select Top Words: A small number of the top words are selected to be used for scoring.
   1. Select Top Sentences: Sentences are scored according to how many of the top words they contain. The top four sentences are selected for the summary.
1. H. P. Edmundson. [New Methods in Automatic Extracting](http://courses.ischool.berkeley.edu/i256/f06/papers/edmonson69.pdf). Journal of the Association for Computing Machinery, 1969.
1. David M. Blei, Andrew Y. Ng and Michael I. Jordan. [Latent Dirichlet Allocation](http://ai.stanford.edu/~ang/papers/jair03-lda.pdf). Journal of Machine Learning Research, 2003. The source code in Python is [sklearn.decomposition.LatentDirichletAllocation](http://scikit-learn.org/dev/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html). Reimplement Luhn's algorithm, but with topics instead of words and applied to several documents instead of one.
   1. Train LDA on all products of a certain type (e.g. all the books)
   1. Treat all the reviews of a particular product as one document, and infer their topic distribution
   1. Infer the topic distribution for each sentence
   1. For each topic that dominates the reviews of a product, pick some sentences that are themselves dominated by that topic.
1. David M. Blei. [Probabilistic Topic Models](http://www.cs.columbia.edu/~blei/papers/Blei2012.pdf). Communications of the ACM, 2012.
1. Rada Mihalcea and Paul Tarau. [TextRank: Bringing Order into Texts](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf). ACL, 2004. The source code in Python is [pytextrank](https://github.com/ceteri/pytextrank). `pytextrank` works in four stages, each feeding its output to the next:
   * Part-of-Speech Tagging and lemmatization are performed for every sentence in the document.
   * Key phrases are extracted along with their counts, and are normalized.
   * Calculates a score for each sentence by approximating jaccard distance between the sentence and key phrases.
   * Summarizes the document based on most significant sentences and key phrases.
1. Federico Barrios, Federico López, Luis Argerich and Rosa Wachenchauzer. [Variations of the Similarity Function of TextRank for Automated Summarization](https://arxiv.org/abs/1602.03606). 2016. The source code in Python is [gensim.summarization](http://radimrehurek.com/gensim/). Gensim's summarization only works for English for now, because the text is pre-processed so that stop words are removed and the words are stemmed, and these processes are language-dependent. TextRank works as follows:
   * Pre-process the text: remove stop words and stem the remaining words.
   * Create a graph where vertices are sentences.
   * Connect every sentence to every other sentence by an edge. The weight of the edge is how similar the two sentences are.
   * Run the PageRank algorithm on the graph.
   * Pick the vertices(sentences) with the highest PageRank score.
1. [TextTeaser](https://github.com/MojoJolo/textteaser) uses basic summarization features and build from it. Those features are:
   * Title feature is used to score the sentence with the regards to the title. It is calculated as the count of words which are common to title of the document and sentence.
   * Sentence length is scored depends on how many words are in the sentence. TextTeaser defined a constant “ideal” (with value 20), which represents the ideal length of the summary, in terms of number of words. Sentence length is calculated as a normalized distance from this value.
   * Sentence position is where the sentence is located. I learned that introduction and conclusion will have higher score for this feature.
   * Keyword frequency is just the frequency of the words used in the whole text in the bag-of-words model (after removing stop words).
1. Güneş Erkan and Dragomir R. Radev. [LexRank: Graph-based Lexical Centrality as Salience in Text Summarization](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume22/erkan04a-html/erkan04a.html). 2004.
   * LexRank uses IDF-modified Cosine as the similarity measure between two sentences. This similarity is used as weight of the graph edge between two sentences. LexRank also incorporates an intelligent post-processing step which makes sure that top sentences chosen for the summary are not too similar to each other.
1. [Latent Semantic Analysis(LSA) Tutorial](https://technowiki.wordpress.com/2011/08/27/latent-semantic-analysis-lsa-tutorial/).
1. Josef Steinberger and Karel Jezek. [Using Latent Semantic Analysis in Text Summarization and Summary Evaluation](http://www.kiv.zcu.cz/~jstein/publikace/isim2004.pdf). Proc. ISIM’04, 2004.
1. Josef Steinberger, Massimo Poesio, Mijail A Kabadjov and Karel Ježek. [Two uses of anaphora resolution in summarization](http://www.sensei-conversation.eu/wp-content/uploads/files/IPMpaper_official.pdf). Information Processing & Management, 2007.
1. Josef Steinberger and Karel Ježek. [Text summarization and singular value decomposition](https://www.researchgate.net/profile/Karel_Jezek2/publication/226424326_Text_Summarization_and_Singular_Value_Decomposition/links/57233c1308ae586b21d87e66/Text-Summarization-and-Singular-Value-Decomposition.pdf). International Conference on Advances in Information Systems, 2004.
1. Ani Nenkova and Kathleen McKeown. [Automatic summarization](https://www.cis.upenn.edu/~nenkova/1500000015-Nenkova.pdf).
Foundations and Trend in Information Retrieval, 2011. [The slides](https://www.fosteropenscience.eu/sites/default/files/pdf/2932.pdf) are also available.
1. Shashi Narayan, Nikos Papasarantopoulos, Mirella Lapata, Shay B. Cohen. [Neural Extractive Summarization with Side Information](https://arxiv.org/abs/1704.04530). 2017.

### Abstractive Text Summarization
1. Alexander M. Rush, Sumit Chopra, Jason Weston. [A Neural Attention Model for Abstractive Sentence Summarization](https://arxiv.org/abs/1509.00685). EMNLP, 2015. The source code in LUA Torch7 is [NAMAS](https://github.com/facebook/NAMAS).
   * They use sequence-to-sequence encoder-decoder LSTM with attention.
   * They use the first sentence of a document. The source document is quite small (about 1 paragraph or ~500 words in the training dataset of Gigaword) and the produced output is also very short (about 75 characters). It remains an open challenge to scale up these limits - to produce longer summaries over multi-paragraph text input (even good LSTM models with attention models fall victim to vanishing gradients when the input sequences become longer than a few hundred items).
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
1. Jiwei Li, Minh-Thang Luong and Dan Jurafsky. [A Hierarchical Neural Autoencoder for Paragraphs and Documents](https://arxiv.org/abs/1506.01057). ACL 2015. The source code in Matlab is [Hierarchical-Neural-Autoencoder](https://github.com/jiweil/Hierarchical-Neural-Autoencoder).
1. Sumit Chopra, Alexander M. Rush and Michael Auli. [Abstractive Sentence Summarization with Attentive Recurrent Neural Networks](http://harvardnlp.github.io/papers/naacl16_summary.pdf). NAACL, 2016.
1. Jianpeng Cheng, Mirella Lapata. [Neural Summarization by Extracting Sentences and Words](https://arxiv.org/abs/1603.07252). ACL, 2016.
   * This paper uses attention as a mechanism for identifying the best sentences to extract, and then go beyond that to generate an abstractive summary.
1. Romain Paulus, Caiming Xiong, Richard Socher. [A Deep Reinforced Model for Abstractive Summarization](https://metamind.io/static/pdf/deep-reinforced-model-arxiv-v1.pdf). 2017.
1. Shibhansh Dohare, Harish Karnick. [Text Summarization using Abstract Meaning Representation](https://arxiv.org/abs/1706.01678). 2017.

### Text Summarization
1. Eduard Hovy and Chin-Yew Lin. [Automated text summarization and the summarist system](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/tipster-proc-hovy-lin-final.pdf). In Proceedings of a Workshop on Held at Baltimore, Maryland, ACL, 1998.
1. Eduard Hovy and Chin-Yew Lin. [Automated Text Summarization in SUMMARIST](https://www.isi.edu/natural-language/people/hovy/papers/98hovylin-summarist.pdf). In Advances in Automatic Text Summarization, 1999.
1. Dipanjan Das and Andre F.T. Martins. [A survey on automatic text summarization](https://wtlab.um.ac.ir/images/e-library/text_summarization/A%20Survey%20on%20Automatic%20Text%20Summarization.pdf). Technical report, CMU, 2007
1. J. Leskovec, L. Backstrom, J. Kleinberg. [Meme-tracking and the Dynamics of the News Cycle](http://www.memetracker.org). ACM SIGKDD Intl. Conf. on Knowledge Discovery and Data Mining, 2009.
1. Ryang, Seonggi, and Takeshi Abekawa. "[Framework of automatic text summarization using reinforcement learning](http://dl.acm.org/citation.cfm?id=2390980)." In Proceedings of the 2012 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning, pp. 256-265. Association for Computational Linguistics, 2012. [not neural-based methods]
1. King, Ben, Rahul Jha, Tyler Johnson, Vaishnavi Sundararajan, and Clayton Scott. "[Experiments in Automatic Text Summarization Using Deep Neural Networks](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.459.8775&rep=rep1&type=pdf)." Machine Learning (2011).
1. Liu, Yan, Sheng-hua Zhong, and Wenjie Li. "[Query-Oriented Multi-Document Summarization via Unsupervised Deep Learning](http://www.aaai.org/ocs/index.php/AAAI/AAAI12/paper/view/5058/5322)." AAAI. 2012.
1. Rioux, Cody, Sadid A. Hasan, and Yllias Chali. "[Fear the REAPER: A System for Automatic Multi-Document Summarization with Reinforcement Learning](http://emnlp2014.org/papers/pdf/EMNLP2014075.pdf)." In EMNLP, pp. 681-690. 2014.[not neural-based methods]
1. PadmaPriya, G., and K. Duraiswamy. "[An Approach For Text Summarization Using Deep Learning Algorithm](http://thescipub.com/PDF/jcssp.2014.1.9.pdf)." Journal of Computer Science 10, no. 1 (2013): 1-9.
1. Denil, Misha, Alban Demiraj, and Nando de Freitas. "[Extraction of Salient Sentences from Labelled Documents](http://arxiv.org/abs/1412.6815)." arXiv preprint arXiv:1412.6815 (2014).
1. Kågebäck, Mikael, et al. "[Extractive summarization using continuous vector space models](http://www.aclweb.org/anthology/W14-1504)." Proceedings of the 2nd Workshop on Continuous Vector Space Models and their Compositionality (CVSC)@ EACL. 2014.
1. Denil, Misha, Alban Demiraj, Nal Kalchbrenner, Phil Blunsom, and Nando de Freitas. "[Modelling, Visualising and Summarising Documents with a Single Convolutional Neural Network](http://arxiv.org/abs/1406.3830)." arXiv preprint arXiv:1406.3830 (2014).
1. Cao, Ziqiang, Furu Wei, Li Dong, Sujian Li, and Ming Zhou. "[Ranking with Recursive Neural Networks and Its Application to Multi-document Summarization](http://gana.nlsde.buaa.edu.cn/~lidong/aaai15-rec_sentence_ranking.pdf)." (AAAI'2015).
1. Fei Liu, Jeffrey Flanigan, Sam Thomson, Norman Sadeh, and Noah A. Smith. "[Toward Abstractive Summarization Using Semantic Representations](http://www.cs.cmu.edu/~nasmith/papers/liu+flanigan+thomson+sadeh+smith.naacl15.pdf)." NAACL 2015
1. Wenpeng Yin， Yulong Pei. "Optimizing Sentence Modeling and Selection for Document Summarization." IJCAI 2015
1. He, Zhanying, Chun Chen, Jiajun Bu, Can Wang, Lijun Zhang, Deng Cai, and Xiaofei He. "[Document Summarization Based on Data Reconstruction](http://cs.nju.edu.cn/zlj/pdf/AAAI-2012-He.pdf)." In AAAI. 2012.
1. Liu, He, Hongliang Yu, and Zhi-Hong Deng. "[Multi-Document Summarization Based on Two-Level Sparse Representation Model](http://www.cis.pku.edu.cn/faculty/system/dengzhihong/papers/AAAI%202015_Multi-Document%20Summarization%20Based%20on%20Two-Level%20Sparse%20Representation%20Model.pdf)." In Twenty-Ninth AAAI Conference on Artificial Intelligence. 2015.
1. Jin-ge Yao, Xiaojun Wan and Jianguo Xiao. [Compressive Document Summarization via Sparse Optimization](http://ijcai.org/Proceedings/15/Papers/198.pdf). IJCAI, 2015.
1. Piji Li, Lidong Bing, Wai Lam, Hang Li, and Yi Liao. "[Reader-Aware Multi-Document Summarization via Sparse Coding](http://arxiv.org/abs/1504.07324)." IJCAI 2015.
1. Xiaojun Wan, Yansong Feng and Weiwei Sun. [Automatic Text Generation: Research Progress and Future Trends](http://www.icst.pku.edu.cn/lcwm/wanxj/files/TextGenerationSurvey.pdf). Book Chapter in CCF 2014-2015 Annual Report on Computer Science and Technology in China (In Chinese), 2015.
1. Gulcehre, Caglar, Sungjin Ahn, Ramesh Nallapati, Bowen Zhou, and Yoshua Bengio. "[Pointing the Unknown Words](http://arxiv.org/abs/1603.08148)." arXiv preprint arXiv:1603.08148 (2016).
1. Jiatao Gu, Zhengdong Lu, Hang Li, Victor O.K. Li. [Incorporating Copying Mechanism in Sequence-to-Sequence Learning](https://arxiv.org/abs/1603.06393). ACL. (2016)
   * They addressed an important problem in sequence-to-sequence (Seq2Seq) learning referred to as copying, in which certain segments in the input sequence are selectively replicated in the output sequence. In this paper, they incorporated copying into neural network-based Seq2Seq learning and propose a new model called CopyNet with encoder-decoder structure. CopyNet can nicely integrate the regular way of word generation in the decoder with the new copying mechanism which can choose sub-sequences in the input sequence and put them at proper places in the output sequence.
1. Jianmin Zhang, Jin-ge Yao and Xiaojun Wan. [Toward constructing sports news from live text commentary](http://www.icst.pku.edu.cn/lcwm/wanxj/files/acl16_sports.pdf). In Proceedings of ACL, 2016.
1. Ziqiang Cao, Wenjie Li, Sujian Li, Furu Wei. "[AttSum: Joint Learning of Focusing and Summarization with Neural Attention](http://arxiv.org/abs/1604.00125)".  arXiv:1604.00125 (2016)
1. Ayana, Shiqi Shen, Yu Zhao, Zhiyuan Liu and Maosong Sun. [Neural Headline Generation with Sentence-wise Optimization](https://arxiv.org/abs/1604.01904). 2016.
1. Ayana, Shiqi Shen, Zhiyuan Liu and Maosong Sun. [Neural Headline Generation with Minimum Risk Training](https://128.84.21.199/abs/1604.01904v1). 2016.
1. Kikuchi, Yuta, Graham Neubig, Ryohei Sasano, Hiroya Takamura, and Manabu Okumura. "[Controlling Output Length in Neural Encoder-Decoders](https://arxiv.org/abs/1609.09552)." arXiv preprint arXiv:1609.09552 (2016).
1. Qian Chen, Xiaodan Zhu, Zhenhua Ling, Si Wei and Hui Jiang. "[Distraction-Based Neural Networks for Document Summarization](https://arxiv.org/abs/1610.08462)." IJCAI 2016.
1. Wang, Lu, and Wang Ling. "[Neural Network-Based Abstract Generation for Opinions and Arguments](http://www.ccs.neu.edu/home/luwang/papers/NAACL2016.pdf)." NAACL 2016.
1. Yishu Miao, Phil Blunsom. "[Language as a Latent Variable: Discrete Generative Models for Sentence Compression](http://arxiv.org/abs/1609.07317)." EMNLP 2016.
1. Takase, Sho, Jun Suzuki, Naoaki Okazaki, Tsutomu Hirao, and Masaaki Nagata. "[Neural headline generation on abstract meaning representation](https://www.aclweb.org/anthology/D/D16/D16-1112.pdf)." EMNLP, pp. 1054-1059. 2016.
1. Hongya Song, Zhaochun Ren, Piji Li, Shangsong Liang, Jun Ma, and Maarten de Rijke. [Summarizing Answers in Non-Factoid Community Question-Answering](http://dl.acm.org/citation.cfm?id=3018704). In WSDM 2017: The 10th International Conference on Web Search and Data Mining, 2017.
1. Wenyuan Zeng, Wenjie Luo, Sanja Fidler, Raquel Urtasun. "[Efficient Summarization with Read-Again and Copy Mechanism](https://arxiv.org/abs/1611.03382)." arXiv preprint arXiv:1611.03382 (2016).
1. Piji Li, Zihao Wang, Wai Lam, Zhaochun Ren, Lidong Bing. "[Salience Estimation via Variational Auto-Encoders for Multi-Document Summarization](https://aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14613)". In AAAI, 2017.
1. Ramesh Nallapati, Feifei Zhai, Bowen Zhou. [SummaRuNNer: A Recurrent Neural Network based Sequence Model for Extractive Summarization of Documents](https://arxiv.org/abs/1611.04230). In AAAI, 2017.
1. Ramesh Nallapati, Bowen Zhou, Mingbo Ma. "[Classify or Select: Neural Architectures for Extractive Document Summarization](https://arxiv.org/abs/1611.04244)." arXiv preprint arXiv:1611.04244 (2016).
1. Suzuki, Jun, and Masaaki Nagata. "[Cutting-off Redundant Repeating Generations for Neural Abstractive Summarization](http://www.aclweb.org/anthology/E17-2047)." EACL 2017 (2017): 291.
1. Jiwei Tan and Xiaojun Wan. [Abstractive Document Summarization with a Graph-Based Attentional Neural Model](). ACL, 2017.
1. Preksha Nema, Mitesh M. Khapra, Balaraman Ravindran and Anirban Laha. [Diversity driven attention model for query-based abstractive summarization](). ACL,2017
1. Abigail See, Peter J. Liu and Christopher D. Manning. [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368). ACL, 2017.
1. Qingyu Zhou, Nan Yang, Furu Wei and Ming Zhou. [Selective Encoding for Abstractive Sentence Summarization](https://arxiv.org/abs/1704.07073). ACL, 2017
1. Maxime Peyrard and Judith Eckle-Kohler. [Supervised Learning of Automatic Pyramid for Optimization-Based Multi-Document Summarization](). ACL, 2017.
1. Jin-ge Yao, Xiaojun Wan and Jianguo Xiao. [Recent Advances in Document Summarization](http://www.icst.pku.edu.cn/lcwm/wanxj/files/summ_survey_draft.pdf). KAIS, survey paper, 2017.
1. Pranay Mathur, Aman Gill and Aayush Yadav. [Text Summarization in Python: Extractive vs. Abstractive techniques revisited](https://rare-technologies.com/text-summarization-in-python-extractive-vs-abstractive-techniques-revisited/#text_summarization_in_python). 2017.
   * They compared modern extractive methods like LexRank, LSA, Luhn and Gensim’s existing TextRank summarization module on the [Opinosis dataset](http://kavita-ganesan.com/opinosis-opinion-dataset) of 51 (article, summary) pairs. They also had a try with an abstractive technique using Tensorflow’s algorithm [textsum](https://github.com/tensorflow/models/tree/master/textsum), but didn’t obtain good results due to its extremely high hardware demands (7000 GPU hours).

### Chinese Text Summarization
1. Mao Song Sun. [Natural Language Processing Based on Naturally Annotated Web Resources](http://www.thunlp.org/site2/images/stories/files/2011_zhongwenxinxixuebao_sms.pdf). Journal of Chinese Information Processing, 2011.
1. Baotian Hu, Qingcai Chen and Fangze Zhu. [LCSTS: A Large Scale Chinese Short Text Summarization Dataset](https://arxiv.org/abs/1506.05865). 2015.
   * They constructed a large-scale Chinese short text summarization dataset constructed from the Chinese microblogging website Sina Weibo, which is released to [the public](http://icrc.hitsz.edu.cn/Article/show/139.html). Then they performed GRU-based encoder-decoder method on it to generate summary. They took the whole short text as one sequence, this may not be very reasonable, because most of short texts contain several sentences.
   * LCSTS contains 2,400,591 (short text, summary) pairs as the training set and 1,106  pairs as the test set.
   * All the models are trained on the GPUs tesla M2090 for about one week.
   * The results show that the RNN with context outperforms RNN without context on both character and word based input.
   * Moreover, the performances of the character-based input outperform the word-based input.

### Evaluation Metrics
1. Chin-Yew Lin and Eduard Hovy. [Automatic Evaluation of Summaries Using N-gram
Co-Occurrence Statistics](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/naacl2003.pdf). In Proceedings of the Human Technology Conference 2003 (HLT-NAACL-2003).
1. Chin-Yew Lin. [Rouge: A package for automatic evaluation of summaries](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/07/was2004.pdf). Workshop on Text Summarization Branches Out, Post-Conference Workshop of ACL 2004.
1. Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. [BLEU: a Method for Automatic Evaluation of Machine Translation](http://www.aclweb.org/anthology/P02-1040.pdf).

### Opinion Summarization
1. Kavita Ganesan, ChengXiang Zhai and Jiawei Han. [Opinosis: A Graph Based Approach to Abstractive Summarization of Highly Redundant Opinions](http://kavita-ganesan.com/opinosis). Proceedings of COLING '10, 2010.
1. Kavita Ganesan, ChengXiang Zhai and Evelyne Viegas. [Micropinion Generation: An Unsupervised Approach to Generating Ultra-Concise Summaries of Opinions](http://kavita-ganesan.com/micropinion-generation). WWW'12, 2012.
1. Kavita Ganesan. [Opinion Driven Decision Support System (ODSS)](http://kavita-ganesan.com/phd-thesis). PhD Thesis, University of Illinois at Urbana-Champaign, 2013.
1. Haibing Wu, Yiwei Gu, Shangdi Sun and Xiaodong Gu. [Aspect-based Opinion Summarization with Convolutional Neural Networks](https://arxiv.org/abs/1511.09128). 2015.
1. Ozan Irsoy and Claire Cardie. [Opinion Mining with Deep Recurrent Neural Networks](https://www.cs.cornell.edu/~oirsoy/files/emnlp14drnt.pdf). In EMNLP, 2014.

### Reading Comprehension
1. Karl Moritz Hermann, Tomas Kocisky, Edward Grefenstette, Lasse Espeholt, Will Kay, Mustafa Suleyman, and Phil Blunsom. [Teaching machines to read and comprehend](http://papers.nips.cc/paper/5945-teaching-machines-to-read-and-comprehend). NIPS, 2015. The source code in Python is [DeepMind-Teaching-Machines-to-Read-and-Comprehend](https://github.com/thomasmesnard/DeepMind-Teaching-Machines-to-Read-and-Comprehend).
1. Hill, Felix, Antoine Bordes, Sumit Chopra, and Jason Weston. "[The Goldilocks Principle: Reading Children's Books with Explicit Memory Representations](http://arxiv.org/abs/1511.02301)." arXiv preprint arXiv:1511.02301 (2015).
1. Kadlec, Rudolf, Martin Schmid, Ondrej Bajgar, and Jan Kleindienst. "[Text Understanding with the Attention Sum Reader Network](http://arxiv.org/abs/1603.01547)." arXiv preprint arXiv:1603.01547 (2016).
1. Danqi Chen, Jason Bolton and Christopher D. Manning. [A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task](https://arxiv.org/abs/1606.02858). ACL, 2016. The source code in Python is [rc-cnn-dailymail](https://github.com/danqi/rc-cnn-dailymail).
1. Dhingra, Bhuwan, Hanxiao Liu, William W. Cohen, and Ruslan Salakhutdinov. "[Gated-Attention Readers for Text Comprehension](http://arxiv.org/abs/1606.01549)." arXiv preprint arXiv:1606.01549 (2016).
1. Sordoni, Alessandro, Phillip Bachman, and Yoshua Bengio. "[Iterative Alternating Neural Attention for Machine Reading](http://arxiv.org/abs/1606.02245)." arXiv preprint arXiv:1606.02245 (2016).
1. Trischler, Adam, Zheng Ye, Xingdi Yuan, and Kaheer Suleman. "[Natural Language Comprehension with the EpiReader](http://arxiv.org/abs/1606.02270)." arXiv preprint arXiv:1606.02270 (2016).
1. Yiming Cui, Zhipeng Chen, Si Wei, Shijin Wang, Ting Liu, Guoping Hu. "[Attention-over-Attention Neural Networks for Reading Comprehension](http://arxiv.org/abs/1607.04423)." arXiv preprint arXiv:1607.04423 (2016).
1. Yiming Cui, Ting Liu, Zhipeng Chen, Shijin Wang, Guoping Hu. "[Consensus Attention-based Neural Networks for Chinese Reading Comprehension](https://arxiv.org/abs/1607.02250)." arXiv preprint arXiv:1607.02250 (2016).
1. Daniel Hewlett, Alexandre Lacoste, Llion Jones, Illia Polosukhin, Andrew Fandrianto, Jay Han, Matthew Kelcey and David Berthelot. "[WIKIREADING: A Novel Large-scale Language Understanding Task over Wikipedia](http://www.aclweb.org/anthology/P/P16/P16-1145.pdf)." ACL (2016). pp. 1535-1545.
1. Minghao Hu, Yuxing Peng, Xipeng Qiu. "[Mnemonic Reader for Machine Comprehension](https://arxiv.org/abs/1705.02798)." arXiv:1705.02798 (2017).
1. Wenhui Wang, Nan Yang, Furu Wei, Baobao Chang and Ming Zhou. "[R-NET: Machine Reading Comprehension with Self-matching Networks](https://www.microsoft.com/en-us/research/publication/mcr/)." ACL (2017).

### Sentence Modelling
1. Kalchbrenner, Nal, Edward Grefenstette, and Phil Blunsom. "[A convolutional neural network for modelling sentences](http://arxiv.org/abs/1404.2188)." arXiv preprint arXiv:1404.2188 (2014).
1. Kim, Yoon. "[Convolutional neural networks for sentence classification](http://arxiv.org/abs/1408.5882)." arXiv preprint arXiv:1408.5882 (2014).
1. Le, Quoc V., and Tomas Mikolov. "[Distributed representations of sentences and documents](http://arxiv.org/abs/1405.4053)." arXiv preprint arXiv:1405.4053 (2014).
1. Yang, Zichao, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, and Eduard Hovy. "[Hierarchical Attention Networks for Document Classification](http://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)." In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. 2016.
