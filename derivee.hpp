#ifndef DER_DEFINED
#define DER_DEFINED
#include "type.hpp"
#include "activation_fonction_matrix.hpp"


inline Matrixld dsigmoid(const Matrixld& x) {
    return ((sigmoid_m(x).array() * (static_cast<long double>(1.0) - sigmoid_m(x).array())).matrix()).eval();
}


inline Matrixld dReLU(const Matrixld& x){
    Matrixld dReLU = (x.array() > 0).cast<long double>() + (x.array() <= 0).cast<long double>() * EPS;
    return dReLU;
}


inline Matrixld dSoftMax(const Matrixld& S) {
    // S est un vecteur de probabilité prédite par Softmax (de dimension n)
    // La dérivée de Softmax est la matrice Jacobienne, de taille n x n

    Index n = S.size();  // Nombre d'éléments dans S
    Matrixld jacobian(n, n);  // Matrice Jacobienne n x n

    // Calcul de la matrice Jacobienne
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                jacobian(i, j) = S(i) * (1 - S(i));  // Dérivée diagonale
            } else {
                jacobian(i, j) = -S(i) * S(j);  // Dérivée hors diagonale
            }
        }
    }

    return jacobian;
}


#endif
