#include "gradients.hpp"


Matrixld gradient_W_sigmoid_MSE(const Matrixld& y, const Matrixld& error,const  Matrixld& x){
     return ((error.array() * y.array() * (static_cast<long double>(1.0) - y.array())).matrix().transpose() * x).eval();
}

RowVectorld gradient_B_sigmoid_MSE(const Matrixld& y, const Matrixld& error) {
    return RowVectorld((error.array() * y.array() * (static_cast<long double>(1.0) - y.array())).colwise().sum());
}

//On utilise Leaked ReLU avec eps = EPS
Matrixld gradient_W_ReLU(const Matrixld& y, const Matrixld& error, const Matrixld& x) {
    Matrixld dReLU = (y.array() > 0).cast<long double>() + (y.array() <= 0).cast<long double>() * EPS;  // Gradient de ReLU : 1 si y > 0, sinon 0
    return ((error.array() * dReLU.array()).matrix().transpose() * x).eval();
}


RowVectorld gradient_B_ReLU(const Matrixld& y, const Matrixld& error) {
    Matrixld dReLU = (y.array() > 0).cast<long double>() + (y.array() <= 0).cast<long double>() * EPS;  // Gradient de ReLU : 1 si y > 0, sinon 0
    return ((error.array() * dReLU.array()).colwise().sum()).eval();
}


Matrixld gradient_W_softmax(const Matrixld& y [[maybe_unused]], const Matrixld& error, const Matrixld& x) {
    return (error.transpose() * x).eval();
}


RowVectorld gradient_B_softmax(const Matrixld& y [[maybe_unused]],const Matrixld& error) {
    return RowVectorld(error.colwise().sum());
}


