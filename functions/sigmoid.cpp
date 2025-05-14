#include "type.hpp"
#include "activation_fonction_dim1.hpp"

class Sigmoid{

    public:
    
    Sigmoid(){};

    Matrixld forward(const Matrixld& x){
        xi = x;
        Matrixld temp(x);
        for (Index i=0;i<x.rows();i++){
            for (Index j =0;j<x.cols();j++){
                temp(i,j) = sigmoid(x(i,j));
            }
        }
        return temp;
    }

    Matrixld backward(const Matrixld& xb_chapeau){
        return dsigmoid(xi)*xb_chapeau;
    }


    private:
    Matrixld xi;

    Matrixld dsigmoid(const Matrixld& x){
        return ((forward(x).array() * (static_cast<long double>(1.0) - forward(x).array())).matrix()).eval();
    }


};

