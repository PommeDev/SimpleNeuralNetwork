#include "activation_fonction_dim_n.hpp"


RowVectorld sigmoid_n(const RowVectorld& x) {
    RowVectorld temp(x.size());
    for (int i = 0; i < x.size(); i++) {
        temp(i) = sigmoid(x(i));
    }
    return temp;
}


RowVectorld ReLU_n(const RowVectorld& x){
        RowVectorld temp(x.size());
    int i = 0;
    for (long double y : x){
        temp(i) = ReLU(y);
        i++;
    }
    return temp;
}
RowVectorld softplus_n(const RowVectorld& x){
        RowVectorld temp(x.size());
    int i = 0;
    for (long double y : x){
        temp(i) = softplus(y);
        i++;
    }
    return temp;

}

RowVectorld tanh_n(const RowVectorld& x){
        RowVectorld temp(x.size());
    int i = 0;
    for (long double y : x){
        temp(i) = tanh2(y);
        i++;
    }
    return temp;
}