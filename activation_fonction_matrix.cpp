#include "activation_fonction_matrix.hpp"


using Matrixld = Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic>;
using VectorXld = Eigen::Matrix<long double, Eigen::Dynamic, 1>;

Matrixld SoftMax2(const Matrixld& x) {
    // Soustraction du max pour la stabilité numérique
    VectorXld max_per_row = x.rowwise().maxCoeff();  // Vecteur des max par ligne
    Matrixld stabilized_x = x;
    
    // Soustraction explicite du max avec broadcasting
    for (int i = 0; i < x.rows(); ++i) {
        stabilized_x.row(i) -= Matrixld::Constant(1, x.cols(), max_per_row(i));
    }

    // Exponentielle élément par élément
    Matrixld exp_x = stabilized_x.array().exp();

    // Somme des exponentielles par ligne
    VectorXld sum_exp = exp_x.rowwise().sum();

    // Normalisation (division élément par élément avec broadcasting manuel)
    Matrixld result = exp_x;
    for (int i = 0; i < x.rows(); ++i) {
        result.row(i) /= sum_exp(i);
    }

    return result;
}


