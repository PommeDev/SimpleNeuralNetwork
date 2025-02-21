#include <math.h>
#include <stdio.h>
#include "type.hpp"

long double MSE(RowVectorld y,RowVectorld e);
/* Erreur quadratique moyenne*/

long double MAE(RowVectorld y,RowVectorld e);
/* Erreur moyenne absolue */

long double crossEntropy(const Matrixld& y, const Matrixld& e);
