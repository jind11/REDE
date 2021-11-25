# REDE

This is the source code for the paper: [Jin, Di, et al. "Towards Zero and Few-shot Knowledge-seeking Turn Detection in Task-orientated Dialogue Systems." ACL ConvAI Workshop (2021).](https://aclanthology.org/2021.nlp4convai-1.27/) If you use the code, please cite the paper:

```
@inproceedings{jin-etal-2021-towards,
    title = "Towards Zero and Few-shot Knowledge-seeking Turn Detection in Task-orientated Dialogue Systems",
    author = "Jin, Di  and
      Gao, Shuyang  and
      Kim, Seokhwan  and
      Liu, Yang  and
      Hakkani-Tur, Dilek",
    booktitle = "Proceedings of the 3rd Workshop on Natural Language Processing for Conversational AI",
    month = nov,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.nlp4convai-1.27",
    pages = "281--288",
    abstract = "Most prior work on task-oriented dialogue systems is restricted to supporting domain APIs. However, users may have requests that are out of the scope of these APIs. This work focuses on identifying such user requests. Existing methods for this task mainly rely on fine-tuning pre-trained models on large annotated data. We propose a novel method, REDE, based on adaptive representation learning and density estimation. REDE can be applied to zero-shot cases, and quickly learns a high-performing detector with only a few shots by updating less than 3K parameters. We demonstrate REDE{'}s competitive performance on DSTC9 data and our newly collected test set.",
}
```

## Data
We have provided all data needed in the folder of "data/kmdm". Here are brief descriptions of each set:
- OODRemovedtrain.tsv: all non-knowledge-seeking turns in the train set
- OODtrain.tsv: all knowledge-seeking turns in the train set
- eval.tsv: validation set containing both non-knowledge-seeking and knowledge-seeking turns
- test.tsv: test set containing both non-knowledge-seeking and knowledge-seeking turns
- test_tripadvisor.tsv: contrast test set obtained from Tripadvisor forum (please refer to the paper for details of data collection)

## Prerequisites:
Required packages are listed in the requirements.txt file:
```
pip install -r requirements.txt
```

## How to use
* Run the following commands to download the sentence embedding models parameters:

```
gdown --id 1mOJX1jVeOEImGPKX1voAA8pf-cc60iVC
unzip sentence_embedders.zip
```

* Run this command to obtain sentence embeddings for each set:

```
sh ./sentence_embedder.sh
```

* Run this command to obtain knowledge-seeking turn detection performance for each evaluation set:

```
python density-estimation.py bgm
```

&nbsp;&nbsp;&nbsp;&nbsp;You may replace "bgm" with other kinds of density estimation models such as "gmm" or "ocsvm". But in general, "bgm" works the best.

&nbsp;&nbsp;&nbsp;&nbsp;After running this command, you should expect to obtain the following results:

&nbsp;&nbsp;&nbsp;&nbsp;Val results:  {'P': defaultdict(<class 'list'>, {767: [0.9902512185976753]}), 'R': defaultdict(<class 'list'>, {767: [0.988028432472877]}), 'F1': defaultdict(<class 'list'>, {767: [0.9891385767790263]})}

&nbsp;&nbsp;&nbsp;&nbsp;Test results:  {'P': defaultdict(<class 'list'>, {767: [0.9805057955742887]}), 'R': defaultdict(<class 'list'>, {767: [0.939424533064109]}), 'F1': defaultdict(<class 'list'>, {767: [0.9595256509409642]})}

&nbsp;&nbsp;&nbsp;&nbsp;Contrast results:  {'P': defaultdict(<class 'list'>, {767: [0.8942307692307693]}), 'R': defaultdict(<class 'list'>, {767: [0.9043760129659644]}), 'F1': defaultdict(<class 'list'>, {767: [0.8992747784045126]})}

&nbsp;&nbsp;&nbsp;&nbsp;Subjective questions results:  {'P': defaultdict(<class 'list'>, {767: [1.0]}), 'R': defaultdict(<class 'list'>, {767: [0.9382763975155279]}), 'F1': defaultdict(<class 'list'>, {767: [0.9681554175846184]})}
