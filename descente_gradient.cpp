
#include "descente_gradient.hpp"


RowVectorld calcul_couche_i(
    const function<RowVectorld(const RowVectorld&)> activation_function,
    const Matrixld& W,
    const RowVectorld& B,
    const RowVectorld& x
){
        //calcul de y = sigma(Wx + B)
        RowVectorld y;
        y = activation_function((W * x.transpose()).transpose() + B);
        return y;

}

/* Marche qu'avec sigmoid */
/* J(W,B)=−(1/m)*​∑​[yi​log(y^​i​)+(1−yi​)log(1−y^​i​)] */

void calcul_gradient( 
    long double alpha, //learning rate
    int iter_max,
    long double eps, //precision (condition d'arrêt)  
    const Matrixld& X, 
    const RowVectorld& Y, 
    Matrixld& W, 
    RowVectorld& B,
    const function<RowVectorld(const RowVectorld&)> activation_function
){
    Matrixld Z_m(W.rows(),X.cols());
    Matrixld E_m(Y.rows(),X.cols());
    int nb_iteration = 0;
    Index m = X.cols();
    Matrixld grad_W(W.rows(),W.cols());
    RowVectorld grad_B(B.rows(),B.cols());

    do{
        for (Index i = 0;i<m;i++){
            RowVectorld Z = calcul_couche_i(activation_function,W,B,(X.col(i)).transpose());
            Z_m.col(i) = Z.transpose();
            RowVectorld E = Z - Y.col(i);
            E_m.col(i) = E.transpose();
        }
        
        grad_W = (1/static_cast<long double>(m))*(X*E_m.transpose());
        grad_B = (1/static_cast<long double>(m))*(E_m.rowwise().sum());
 
        W = W - alpha*grad_W.transpose();
        B = B - alpha*grad_B;
        nb_iteration ++;
    }while(nb_iteration < iter_max && grad_W.norm() > eps && grad_B.norm() > eps);

}