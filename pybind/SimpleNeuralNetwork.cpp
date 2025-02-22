#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include <pybind11/functional.h> 
#include "Network.hpp"
#include "useful.hpp"

namespace py = pybind11;
using namespace pybind11::literals;



PYBIND11_MODULE(SimpleNeuralNetwork, m){
    py::class_<Network>(m, "Network", "Base class for NeuralNetwork, use it to create a multi-layer perceptrons with a softmax end layer")
        .def(py::init<vector<int>,vector<function<Matrixld(const Matrixld&)>>, vector<function<Matrixld(const Matrixld&)>>>(), "nodes"_a, "activations"_a, "derivatives"_a, "A complex construcotr @Deprecated")
        .def(py::init<vector<int>,vector<function<Matrixld(const Matrixld&)>>,vector<function<Matrixld(const Matrixld&)>> ,function<long double(const Matrixld&,const Matrixld&)>>(),"nodes"_a, "activations"_a, "derivatives"_a, "cost_function"_a ,"@Deprecated")
        .def(py::init<vector<int>,vector<function<Matrixld(const Matrixld&)>>,vector<function<Matrixld(const Matrixld&)>>,function<long double(const Matrixld&,const Matrixld&)>,bool>(),"nodes"_a, "activations"_a, "derivatives"_a, "cost_function"_a, "same_start"_a, "@Deprecated")
        .def(py::init<>(), "Base constructor with this, you have to use add_layer, add_enter and add_exit methods")
        .def("afficher_cmd", &Network::afficher_cmd, "Display on shell all layer and Weights")
        .def("predict", &Network::predict, "X"_a, "Return the answer predict by Network")
        .def("train_full", &Network::train_full, "X"_a,"E"_a,"alpha"_a,"eps"_a, "max_iter"_a, "affichage"_a = false, "Train the Network, with X enter and E waiting result")
        .def("train_full_batch", &Network::train_full_batch, "X"_a,"E"_a,"alpha"_a,"eps"_a, "max_iter"_a, "affichage"_a = false, "Train the Network, with Xs batch enter and Es batch waiting result")
        .def("better_train_full", &Network::better_train_full, "X"_a,"E"_a,"alpha"_a,"eps"_a, "max_iter"_a, "affichage"_a = false, "ratio"_a = 100, "Train the Network, with X enter and E waiting result, ratio it's how many time we write time remaining")
        .def("better_train_full_batch", &Network::better_train_full_batch, "Xs"_a,"Es"_a,"alpha"_a,"eps"_a, "max_iter"_a, "affichage"_a = false, "ratio"_a = 100, "Train the Network, with Xs batch enter and Es batch waiting result, ratio it's how many time we write time remaining")
        .def("train_1_gen", &Network::train_1_gen, "X"_a,"E"_a,"alpha"_a,"eps"_a, "max_iter"_a, "affichage"_a, "Forwardpropagation then backpropagation")
        .def("__str__", &Network::toString)
        .def("to_json", &Network::to_json, "Convertit l'état du réseau en JSON.")
        .def("from_json", &Network::from_json, "j"_a, "Charge l'état du réseau depuis un objet JSON.")
        .def("save_to_file", &Network::save_to_file, "filename"_a, "Sauvegarde le réseau dans un fichier JSON.")
        .def("load_from_file", &Network::load_from_file, "filename"_a, "Charge le réseau depuis un fichier JSON.")
        .def("ajoute_couche", &Network::ajoute_couche, "neuronnes"_a, "activation"_a, "derivee"_a, 
             "Ajoute une couche avec un nombre spécifique de neurones et des fonctions d'activation personnalisées.")
        .def("add_layer", &Network::add_layer, "neuronnes"_a, "activation"_a, 
             "Ajoute une couche avec le nombre de neurones donné et une fonction d'activation prédéfinie.")
        .def("add_enter", &Network::add_enter, "neuronnes"_a, 
             "Définit la couche d'entrée avec le nombre de neurones spécifié.")
        .def("add_exit", &Network::add_exit, "neuronnes"_a, "activation"_a, 
             "Ajoute une couche de sortie avec un nombre de neurones et une activation définis.")
        .def("init_train", &Network::init_train, 
             "alpha"_a, "eps"_a, "max_iter"_a, "affichage_gen"_a, "fonction_cout"_a, 
             "Initialise l'entraînement avec les hyperparamètres spécifiés et une fonction de coût personnalisée.")
        .def("get_couche", &Network::getCouche, "Retourne le nombre de couches du réseau.")
        .def("get_epoch", &Network::getEpoch, "Retourne le nombre d'époques (epochs) réalisées.")
        .def("get_neuronnes", &Network::getNeuronnes, "Retourne un vecteur contenant le nombre de neurones par couche.")
        .def("get_weigths", &Network::getWeigths, "Retourne les poids du réseau sous forme de matrice.")
        .def("get_bias", &Network::getBias, "Retourne les biais sous forme de vecteur.")
        ;


    py::enum_<Activation>(m, "Activation", py::arithmetic())
        .value("Relu", Activation::ReLU, "Fonction d'activation ReLU")
        .value("Sigmoid", Activation::Sigmoid, "Fonction d'activation Sigmoïde")
        .export_values();


    m.def("Sigmoid", &sigmoid_m, "x"_a, "Apply Sigmoid to each coefficient of x");
    m.def("ReLU", &ReLU_m, "x"_a, "Apply leaked ReLU to each coeff of x");
    m.def("SoftMax", &SoftMax, "x"_a, "Apply SoftMax to each coeff of x");
    
    m.def("dSigmoid", &dsigmoid, "x"_a, "Apply dSigmoid to each coefficient of x");
    m.def("dReLU", &dReLU, "x"_a, "Apply leaked dReLU to each coeff of x");
    m.def("dSoftMax", &dSoftMax, "x"_a, "Apply dSoftMax to each coeff of x");
    m.def("CrossEntropy", &crossEntropy, "y"_a, "e"_a, "Applye corss entropy to y and e, y: Result give, e: Result wanted");

    py::class_<Matrixld>(m, "Matrixld")
    .def(py::init<int, int>(), py::arg("rows"), py::arg("cols"))
    .def("rows", &Matrixld::rows)
    .def("cols", &Matrixld::cols)
    .def("data", [](Matrixld& mat) {
        return py::buffer_info(
            mat.data(),
            sizeof(long double),
            py::format_descriptor<long double>::format(),
            2,
            { mat.rows(), mat.cols() },
            { sizeof(long double) * mat.cols(), sizeof(long double) }
        );
    })
    .def("__getitem__", [](const Matrixld& mat, std::pair<int, int> idx) {
        return mat(idx.first, idx.second);
    })
    .def("__setitem__", [](Matrixld& mat, std::pair<int, int> idx, long double value) {
        mat(idx.first, idx.second) = value;
    });


    py::class_<RowVectorld>(m, "RowVectorld")
    .def(py::init<int>(), py::arg("size"))
    .def("size", &RowVectorld::size)
    .def("data", [](RowVectorld& vec) {
        return py::buffer_info(
            vec.data(),
            sizeof(long double),
            py::format_descriptor<long double>::format(),
            1,
            { vec.size() },
            { sizeof(long double) }
        );
    })
    .def("__getitem__", [](const RowVectorld& vec, int idx) {
        return vec(idx);
    })
    .def("__setitem__", [](RowVectorld& vec, int idx, long double value) {
        vec(idx) = value;
    });


    m.def("toString", &toString, "mat"_a, "affiche la matrice mat");

}



