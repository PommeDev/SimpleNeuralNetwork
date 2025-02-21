#include "calcul_erreur.hpp"

// https://fr.wikipedia.org/wiki/Perceptron_multicouche
//but faire une descente de gradient sur MSE
//calculer y = sigma(Wx + B)
// W poid ; B biais ; sigma fonction d'activation.
// divise par 2 pour simplifier la dérivation


long double MSE(RowVectorld y,RowVectorld e){
    Index n = e.size();
    long double acc = 0;
    for(Index i =0;i<n;i++){
        long double temp = (y[i] - e[i]);
        acc += (temp * temp);
    }
    return acc/(2*n);
}


long double MAE(RowVectorld y,RowVectorld e){
    Index n = e.size();
    long double acc = 0;
    for(Index i =0;i<n;i++){
        long double temp = (y[i] - e[i]);
        acc += abs(temp);
    }
    return acc/(2*n);
}


long double crossEntropy(const Matrixld& y, const Matrixld& e) {
    // Vérifier que S et Y ont la même taille

    long double loss = 0.0;

    // Calcul de l'entropie croisée
    for (int i = 0; i < y.rows(); ++i) {
        for (int j = 0; j < y.cols(); ++j) {
            if (e(i, j) == 1) {
                loss -= e(i, j) * log(y(i, j)); // Seul l'élément correspondant à la classe vraie compte
            }
        }
    }

    return loss / y.rows(); // Diviser par le nombre d'exemples dans le batch pour avoir la moyenne
}