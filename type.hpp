#include "Eigen/Dense"


using namespace Eigen;

#ifndef ACTIVATIONS_ENUM
#define ACTIVATIONS_ENUM
enum class Activation{
    Sigmoid,
    ReLU,
    SoftPlus,
    Tanh
};
#endif


#ifndef MLD_DEFINED
#define MLD_DEFINED
typedef Matrix<long double, Dynamic, Dynamic> Matrixld;
#endif
#ifndef RWLD_DEFINED
#define RWLD_DEFINED
typedef Matrix<long double, 1, Dynamic> RowVectorld;
#endif

#ifndef EPS_DEFINED
#define EPS_DEFINED
extern long double EPS;
#endif