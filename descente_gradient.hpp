#include "type.hpp"
#include <cstdlib>
#include <functional>
#include <iostream>

using namespace std;

//calculer y = sigma(Wx + B)
// W poid ; B biais ; sigma fonction d'activation.

RowVectorld calcul_couche_i(
    const function<RowVectorld(const RowVectorld&)> activation_function,
    const Matrixld& W,
    const RowVectorld& B,
    const RowVectorld& x
);
//calcul de y = sigma(Wx + B) , si couche précedente n neuronne et actuel m alors W : (m,n) , x dim n , B dim m ;

//marche pour une couche
void calcul_gradient( 
    long double alpha, //learning rate
    int iter_max,
    long double eps, //precision (condition d'arrêt)  
    const Matrixld& X, //ensemble des exemples pour tester
    const RowVectorld& Y, 
    Matrixld& W, 
    RowVectorld& B,
    const function<RowVectorld(const RowVectorld&)> activation_function
);