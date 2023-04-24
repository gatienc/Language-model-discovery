# TPE ISSD

## gatien CHENU, Emna Kriaa



# 1. Introduction
Les modèles de langages sont des modèles statistiques qui permettent de prédire la probabilité d'apparition d'un mot dans un texte à partir des mots qui le précèdent. Ces modèles sont utilisés pour la génération de texte, la correction orthographique, la traduction automatique, la détection de la langue d'un texte, etc.
Ils ont subies un fort essor ces dernières années, notamment grâce à l'augmentation des données disponibles et à l'augmentation des capacités de calcul. Les modèles de langages sont des modèles de réseaux de neurones profonds, qui sont des modèles d'apprentissage automatique. Ils sont donc très sensibles aux données d'entrées, et à la manière dont elles sont préparées.

# les orientations du sujet,
au tout départ l'idée était d'écrire un tout petit modèle de langages utilisant les émojis , ou alors utilisait une mini langue ( langue de jeux vidéos) pour découvrir et mettre en application notre découverte des modèles de langages.

dans un premier temps, nous avons suivi [ce tutoriel](https://towardsdatascience.com/word2vec-with-pytorch-implementing-original-paper-2cd7040120b0)

## les modèles de vectorization entrainés

| Nom      | Créateur | Année     |
| :---        |    :----:   |          ---: |
| [Word2Vec](https://arxiv.org/pdf/1301.3781.pdf)   |   Google      |    2013   |
|[GloVe](https://nlp.stanford.edu/projects/glove/) |Stanford| 2014|
|[FastText](https://arxiv.org/pdf/1607.04606.pdf)| Facebook| 2015-2018|
|[ELMo](https://arxiv.org/pdf/1802.05365.pdf)| AllenNLP| 2018|

## modèle de langange
| Nom      | Créateur | Année     |
|[BERT](https://arxiv.org/pdf/1810.04805.pdf)| Google| 2019|
|[GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)| OpenAI| 2018|
|[XLNet](https://arxiv.org/pdf/1906.08237.pdf)| Google| 2019|
|[RoBERTa](https://arxiv.org/pdf/1907.11692.pdf)| Facebook| 2019|
|[GPT-3](https://arxiv.org/pdf/2005.14165.pdf)| OpenAI| 2020|
|[GPT-4](https://arxiv.org/pdf/2303.08774.pdf)| OpenAI| 2021|




GPT : Generative Pre-trained Transformer

papier emblématique : [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

mais les Generative Pre-trained model ne sont pas des modèles de word2-vec


# 2. Emoji2Vec

## mise en contexte
les emoji sont codé au format unicode , et sont donc des caractères comme les autres. On peut donc les coder comme des mots, et les utiliser dans un modèle de langages. Maleureusement les emoji sont très peu utilisé dans les datasets de textes. Nous avons cherché les endroits où pouvait être le plus utiliser les émojis, c'est évidemment dans les messages privés et les réseaux sociaux. 
malheureusement les messages privés sont garder précieusement et sur les réseaux sociaux l'utilisation des émojis est très différente de l'utilisation des mots. 
les émojis ne sont pas utilisé comme des mots, mais plutôt comme des ponctuations.

Nous avons eu l'idée de réaliser la vectorization de chaque émoji en utilisant leur description. 
La publication emoji2vec a bien entraîné seulement sur la description des emojis , ce qui est un énorme bien comparé à leur utilisation réelle.
Exemple : chez les jeunes : 😭 ou encore 💀 
Peut représenter :" mort de rire " ce qui n'est pas le cas dans un modèle avec seulement les descriptions pour l'utilisation "initiale" des emojis , selon emoji2vec, 10 % des tweets comprennent un emoji
De plus l'utilisation des emojis sur twitter différent des autres utilisations sur d'autres services.

# note et pensé

dans une IA de complétion de texte , les tokens peuvent être des mots, des caractères, des syllabes, des émojis, etc. 
il est important aussi qu'elle ne puisse être que des fractions de mots, car cela permet de générer des mots inconnus: 
[à 9 min ](https://www.youtube.com/watch?v=Sv5OLj2nVAQ)

d'où viennent les mots inconnus ? peuvent-t'il venir d'autre part que le prompt de départ ? est-ce que ce serait utile de créer des tokens temporaires pour les mots inconnus ? 

## - le pré-traitement des données 
je remarque sur le [word2vec français](http://nlp.polytechnique.fr/word2vec) qu'il y a beaucoup de mot mal traité et qui aurait dû être regroupé en un seul token: 
- exemple: @google -> google

ou plus dur:
 - twttr-> twitter

il y a aussi des liens dans le dataset 
- exemple : http://wordpress.org/?v=3.5

il y a aussi les mots tel que : adblock, adblockers, adblocks 

il faudrait trouver un moyen de les regrouper en un seul token, ou alors de les traiter différemment.

alors pour traiter au mieux ses données on peut tout d'abord réaliser de l'exploration , par exemple afficher les données contenant des caractères spéciaux


# dataset 
Nous avons donc cherché un dataset de tweets, et nous avons trouvé [celui-ci](https://www.kaggle.com/kazanova/sentiment140) qui contient 1,6 millions de tweets.

# bibliographie:

- page wikipédia des émojis: https://fr.wikipedia.org/wiki/Emoji

- Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf). In Proceedings of Workshop at ICLR, 2013.

- Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean. [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546.pdf). In Proceedings of NIPS, 2013.

- Tomas Mikolov, Wen-tau Yih, and Geoffrey Zweig. [Linguistic Regularities in Continuous Space Word Representations](https://aclanthology.org/N13-1090.pdf). In Proceedings of NAACL HLT, 2013.

- Ben Eisner, Tim Rocktäschel, Isabelle Augenstein, Matko Bošnjak, Sebastian Riedel [emoji2vec: Learning emoji representations from their description](https://arxiv.org/pdf/1609.08359.pdf) (2016)


https://www.mdpi.com/2078-2489/11/1/24
papier sur la pertincence de la ponctuation dans les modèles de langages:
intéressant car on peut voir  les émojis comme de la ponctuation


Attention Is All You Need 
papier sur l'arrivée des transformer mais c'est pas vraiment dans le contexte de notre TPE , car cela permet de mettre en contexte dans une phrase

