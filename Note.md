# TPE ISSD

## gatien CHENU, Emna Kriaa



# 1. Introduction
Les modèles de langages sont des modèles statistiques qui permettent de prédire la probabilité d'apparition d'un mot dans un texte à partir des mots qui le précèdent. Ces modèles sont utilisés pour la génération de texte, la correction orthographique, la traduction automatique, la détection de la langue d'un texte, etc.
Ils ont subies un fort essor ces dernières années, notamment grâce à l'augmentation des données disponibles et à l'augmentation des capacités de calcul. Les modèles de langages sont des modèles de réseaux de neurones profonds, qui sont des modèles d'apprentissage automatique. Ils sont donc très sensibles aux données d'entrées, et à la manière dont elles sont préparées.

# les orientations du sujet,
au tout départ l'idée était d'écrire un tout petit modèle de langages utilisant les émojis , ou alors utilisait une mini langue ( langue de jeux vidéos) pour découvrir et mettre en application notre découverte des modèles de langages.

dans un premier temps, nous avons suivi [ce tutoriel](https://towardsdatascience.com/word2vec-with-pytorch-implementing-original-paper-2cd7040120b0)



# note et pensé

dans une IA de complétion de texte , les tokens peuvent être des mots, des caractères, des syllabes, des émojis, etc. 
il est important aussi qu'elle ne puisse être que des fractions de mots, car cela permet de générer des mots inconnus: 
[à 9 min ](https://www.youtube.com/watch?v=Sv5OLj2nVAQ)

d'où viennent les mots inconnus ? peuvent-t'il venir d'autre part que le prompt de départ ? est-ce que ce serait utile de créer des tokens temporaires pour les mots inconnus ? 