#include "type.hpp"
#include <random>
#include <cstdlib>

using namespace std;

class Linear{

    public:
    Linear(int input_size,int output_size){
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> dist(0.0, 1.0);
        W = (Matrixld::NullaryExpr(output_size, input_size, [&]() { return dist(gen); }));
        B = (RowVectorld::NullaryExpr(output_size, [&]() { return dist(gen); }));

    };

    Linear(int input_size,int output_size,long double alpha){
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> dist(0.0, 1.0); // Distribution normale pour Xavier/He init
        W = (Matrixld::NullaryExpr(output_size, input_size, [&]() { return dist(gen); }));
        B = (RowVectorld::NullaryExpr(output_size, [&]() { return dist(gen); }));

    };

    Matrixld forward(const Matrixld& x){
        xi = x;
        return W*x + B.transpose();
    }

    Matrixld backward(const Matrixld& xb_chapeau){
        W -= alpha*xb_chapeau*xi.transpose();
        B -= alpha*(xb_chapeau.colwise().sum()).transpose();
        return W.transpose()*xb_chapeau;
    }


    private:
    Matrixld W;
    RowVectorld B;
    Matrixld xi;
    long double alpha = 0.01;


};
