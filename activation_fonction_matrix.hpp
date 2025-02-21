#ifndef AFM_DEFINED
#define AFM_DEFINED
#include "type.hpp"
#include "activation_fonction_dim1.hpp"



inline Matrixld sigmoid_m(const Matrixld& x) {
    Matrixld temp(x);
    for (Index i=0;i<x.rows();i++){
        for (Index j =0;j<x.cols();j++){
            temp(i,j) = sigmoid(x(i,j));
        }
    }
    return temp;
}


inline Matrixld ReLU_m(const Matrixld& x){
    Matrixld temp(x);
    for (Index i=0;i<x.rows();i++){
        for (Index j =0;j<x.cols();j++){
            temp(i,j) = ReLU(x(i,j));
        }
    }
    return temp;
}

/* Calcul softmax */
inline Matrixld SoftMax(const Matrixld& x) {
    // Soustraction du max de chaque ligne pour la stabilité numérique
    Matrixld stabilized_x = x;
    for (int i = 0; i < x.rows(); ++i) {
        long double row_max = x.row(i).maxCoeff();  // Trouver le max de la ligne
        stabilized_x.row(i) = x.row(i).array() - row_max;
    }

    // Calcul de l'exponentielle
    Matrixld exp_x = stabilized_x.array().exp();

    // Somme des exponentielles
    Matrixld sum_exp(x.rows(), 1);
    for (int i = 0; i < x.rows(); ++i) {
        sum_exp(i, 0) = exp_x.row(i).sum();
    }

    // Normalisation
    Matrixld result(x.rows(), x.cols());
    for (int i = 0; i < x.rows(); ++i) {
        result.row(i) = exp_x.row(i) / sum_exp(i, 0);
    }

    return result;
}


Matrixld SoftMax2(const Matrixld& x);


#endif