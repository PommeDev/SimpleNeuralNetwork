#include "useful.hpp"


string toString(Matrixld mat){
    std::ostringstream oss;
    oss << "Matrix(" << mat.rows() << "x" << mat.cols() << "):\n";
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            oss << mat(i, j) << " ";
        }
        oss << "\n";
    }
    return oss.str();
}

