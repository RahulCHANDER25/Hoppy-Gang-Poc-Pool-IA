# Pacman Learning Project

Nous avons eu l'idée de faire ce projet car on avait bien aimé le jour 5 de la piscine et que le Pacman ressemblait à un défi atteignable et sympa à réaliser.

Au début nous avons utilisé le QLearning, nous nous sommes donc rendu compte que c'était une mauvaise idée car le QLearning gère mal les ennemies qui bougent, nous avons donc voulu apprendre le DeepQLearning pour que nôtre ![Pac-man AI] est un apprentissage amélioré. Cependant l'apprentissage de cet algorithme a été dur. Nous avons eu donc peu de temps pour entrainer notre Pacman à gagner et ne jamais mourir ceci explique le fait que notre Pacman mange les fantômes. Nous voulons donc après le rush l'entrainer un max pour le rendre plus réactif aux fantômes.

# Prérequis

Activez un environnement conda pour installer gym

Vous devez avoir gym version 0.21.0 ainsi que certains modules pour accéder à l'environnement du Pacman

pip install gym[atari,accept-rom-license]==0.21.0

Il faut aussi installer torch

pip install torch

# Lancer le project

./play

Cela va lancer le Pac-man avec notre qtable pour montrer comment il fonctionne.
Vous aurez accès à la fin de la partie à un graphique montrant les évolutions des rewards au cours de la partie.

# Conclusion

Merci a l'équipe pour ce projet (le team en question) :

Rahul Chander
Tiphaine Bertone
Georgios Kypriadis
Baptiste Avert