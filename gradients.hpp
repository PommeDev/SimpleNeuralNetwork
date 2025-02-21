#include "type.hpp"

Matrixld gradient_W_sigmoid_MSE(const Matrixld& y, const Matrixld& error, const Matrixld& x);

RowVectorld gradient_B_sigmoid_MSE(const Matrixld& y, const Matrixld& error);


Matrixld gradient_W_ReLU(const Matrixld& y, const Matrixld& error, const Matrixld& x);

RowVectorld gradient_B_ReLU(const Matrixld& y, const Matrixld& error);


Matrixld gradient_W_softmax_CE(const Matrixld& y, const Matrixld& error, const Matrixld& x);

RowVectorld gradient_B_softmax_CE(const Matrixld& y, const Matrixld& error);