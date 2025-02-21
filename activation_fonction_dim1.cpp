#include "activation_fonction_dim1.hpp"


long double sigmoid(long double x){
    return 1/(1+exp2l(-x));
}

//leaked ReLU
long double ReLU(long double x){
    return __max(EPS*x,x);
}

long double softplus(long double x){
    return log10l(1+exp2l(x));
}

long double tanh2(long double x){
    return (exp2l(2*x)-1)/(exp2l(2*x)+1);
}

