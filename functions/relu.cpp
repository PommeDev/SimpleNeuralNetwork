#include "type.hpp"
#include "activation_fonction_dim1.hpp"

class ReLu{

    public:
    
    ReLu(){};

    Matrixld forward(const Matrixld& x){
        xi = x;
        Matrixld temp(x);
        for (Index i=0;i<x.rows();i++){
            for (Index j =0;j<x.cols();j++){
                temp(i,j) = ReLU(x(i,j));
            }
        }
        return temp;

    }

    Matrixld backward(const Matrixld& xb_chapeau){
        return dReLu(xi)*xb_chapeau;
    }


    private:
    Matrixld xi;

    Matrixld dReLu(const Matrixld& x){
        Matrixld dReLU = (x.array() > 0).cast<long double>() + (x.array() <= 0).cast<long double>() * 0;
        return dReLU;
    }


};

