#include "reseau.hpp"
#include <random>
#include <sstream>
#include "calcul_erreur.hpp"



void Network::init_params(){
    // Initialisation aléatoire des poids et biais
    random_device rd;
    mt19937 gen(rd());
    normal_distribution<double> dist(0.0, 1.0); // Distribution normale pour Xavier/He init

    for (size_t i = 1; i < couche; i++) { // On commence à 1 car on regarde les connexions entre les couches
        Ws.push_back(Matrixld::NullaryExpr(neuronnes[i], neuronnes[i - 1], [&]() { return dist(gen); }));
        Bs.push_back(RowVectorld::NullaryExpr(neuronnes[i], [&]() { return dist(gen); }));
    }
}


void Network::init_fixe_params(){

    for (size_t i = 1; i < couche; i++) { // On commence à 1 car on regarde les connexions entre les couches
        Ws.push_back(Matrixld::NullaryExpr(neuronnes[i], neuronnes[i - 1], [&]() { return 1.0; }));
        Bs.push_back(RowVectorld::NullaryExpr(neuronnes[i], [&]() { return 0; }));
    }
}


Network::Network(
    size_t couche,
    long double alpha,
    long double eps,
    vector<int> neuronnes,
    vector<function<Matrixld(const Matrixld&)>> activations,
    vector<function<Matrixld(const Matrixld&)>> derivees,
    vector<function<Matrixld(const Matrixld&,const Matrixld&,const Matrixld&)>> formule_gradient_W,
    vector<function<RowVectorld(const Matrixld&,const Matrixld&)>> formule_gradient_B
    ):couche(couche),generation(0),alpha(alpha),max_iter(10000),eps(eps),neuronnes(neuronnes), activations(activations),derivees(derivees),formule_gradient_W(formule_gradient_W),formule_gradient_B(formule_gradient_B){
            init_params();
            need_to_train = true;
            fonction_cout = crossEntropy;
}


Network::Network(
        size_t couche,
        long double alpha,
        long double eps,
        int max_iter,
        vector<int> neuronnes,
        vector<function<Matrixld(const Matrixld&)>> activations,
        vector<function<Matrixld(const Matrixld&)>> derivees,
        vector<function<Matrixld(const Matrixld&,const Matrixld&,const Matrixld&)>> formule_gradient_W,
        vector<function<RowVectorld(const Matrixld&,const Matrixld&)>> formule_gradient_B,
        function<long double(const Matrixld&,const Matrixld&)> fonction_cout
        ):couche(couche),generation(0),alpha(alpha),max_iter(max_iter),eps(eps),neuronnes(neuronnes), activations(activations), derivees(derivees),formule_gradient_W(formule_gradient_W),formule_gradient_B(formule_gradient_B),fonction_cout(fonction_cout){
            init_params();
            need_to_train = true;
}

Network::Network(
        size_t couche,
        long double alpha,
        long double eps,
        int max_iter,
        vector<int> neuronnes,
        vector<function<Matrixld(const Matrixld&)>> activations,
        vector<function<Matrixld(const Matrixld&)>> derivees,
        vector<function<Matrixld(const Matrixld&,const Matrixld&,const Matrixld&)>> formule_gradient_W,
        vector<function<RowVectorld(const Matrixld&,const Matrixld&)>> formule_gradient_B,
        function<long double(const Matrixld&,const Matrixld&)> fonction_cout,
        bool same_start // Si true alors initialise toujous les poids et biais pareils.
        ):couche(couche),generation(0),alpha(alpha),max_iter(max_iter),eps(eps),neuronnes(neuronnes), activations(activations), derivees(derivees),formule_gradient_W(formule_gradient_W),formule_gradient_B(formule_gradient_B),fonction_cout(fonction_cout)
    {
        need_to_train = true;
        if(same_start){
            init_fixe_params();
        }else{
            init_params();
        }
}

Network::Network():generation(0){}


void Network::train_1_gen(Matrixld X,Matrixld E){
    /*Propagation avant */
    
    vector<Matrixld> Result_couches(couche);
    Matrixld Y = X;
    for (size_t i = 0;i<couche-1;++i){ //pour chaque couche
        Matrixld Z;
        Z = ((Ws[i]*Y.transpose()).colwise()+Bs[i].transpose());
        Z.transposeInPlace();
        Y = activations[i](Z);
        Result_couches[i] = Y;
        
    }
    
    
    vector<Matrixld> errorpropag(couche-1);
    errorpropag[couche-2] = Result_couches[couche-2] - E.transpose(); // pour sofmax pas besoin de changer
    vector<Matrixld> dW(couche-1);
    vector<RowVectorld> dB(couche-1);
    
    
    for (size_t i = couche-2;i>0;i--){
        

        dW[i] = Result_couches[i-1].transpose()*errorpropag[i]; // E : Valeur voulu , X : valeur initial , Y : Valeur de la couche

        
        dB[i] = errorpropag[i].colwise().sum();

        if (i > 0){
           
            errorpropag[i-1] = ((Ws[i].transpose()*errorpropag[i].transpose()).array()*(derivees[i-1](Result_couches[i-1]).transpose()).array()).matrix().transpose();
        }
    }
    

    dW[0] = X.transpose() * errorpropag[0];

    dB[0] = errorpropag[0].colwise().sum();

    
    
    for (size_t l = 0; l < couche-1; l++) {


        Ws[l] -= alpha * dW[l].transpose();


        Bs[l] -= alpha * dB[l];
    }



    //alpha = 0.99*alpha; //ajustement de alpha à tester
    generation++;
    if (affichage_gen)   cout << "Géneration : " << generation << " -> Done" << endl;
    
    if (affichage_gen)  cout <<"Perte : " << fonction_cout(Result_couches[couche-2],E.transpose()) <<endl;
    need_to_train = fonction_cout(Result_couches[couche-2],E.transpose()) > eps; // A changer pour prendre en compte la fonction cout
    //need_to_train = (-(E.transpose().array() * Result_couches[couche-2].array().log() + (1 - E.transpose().array()) * (1.0 - Result_couches[couche-2].array()).log()).mean()) > eps;
}


void Network::train_full(Matrixld X,Matrixld E){ //Prendre en compte eps
    auto debut = high_resolution_clock::now(); // Capture le temps de début

    int iteration = 0;
    cout << "Debut" << endl;
    while(iteration < max_iter){
        if(need_to_train){
            train_1_gen(X,E);
            iteration ++;
        }else{
            break;
        }

    }

    auto fin = high_resolution_clock::now();
    duration<double> temps = fin - debut; // Durée en secondes
    cout << "Entrainement terminé en " << temps.count() << " secondes" << endl;

}

void Network::train_full_batch(vector<Matrixld> X, vector<Matrixld> E){
    auto debut = high_resolution_clock::now();
    cout << "Début" << endl;
    int iteration = 0;
    while(iteration < max_iter){
        if(need_to_train){
            for(size_t i =0;i<X.size();i++){
                train_1_gen(X[i],E[i]);
            }
            
            iteration ++;
        }
    }

    auto fin = high_resolution_clock::now();
    duration<double> temps = fin - debut; // Durée en secondes
    cout << "Entrainement terminé en " << temps.count() << " secondes" << endl;

}



Matrixld Network::predict(Matrixld X){
    Matrixld Y = X;
    for (size_t i = 0;i<couche-1;i++){
        
        Y = activations[i](Ws[i]*Y.transpose()+Bs[i].transpose()).transpose();
        
    }

    return SoftMax(Y);

}

void Network::afficher_cmd(){
    size_t i =0;
    for (Matrixld y:Ws){
        cout << "Couche : " << i+1 <<endl;
        cout << y << " + " << Bs[i] <<endl;
        i++;
    }
}




// Sérialisation en JSON
//PB de précision surmetn vérifier
json Network::to_json() const {
    json j;
    j["couche"] = couche;
    j["generation"] = generation;
    j["alpha"] = std::to_string(alpha);  // Converti en string
    j["max_iter"] = max_iter;
    j["eps"] = std::to_string(eps);      // Converti en string
    j["neuronnes"] = neuronnes;
    j["affichage_gen"] = affichage_gen;
    j["need_to_train"] = need_to_train;

    // Conversion des matrices Eigen en vecteurs JSON (avec conversion en string)
    for (const auto& W : Ws) {
        json W_json;
        for (Index i = 0; i < W.rows(); ++i) {
            std::vector<std::string> row;
            for (Index j = 0; j < W.cols(); ++j) {
                std::ostringstream oss;
                oss.precision(20);  // Définit la précision maximale
                oss << W(i, j);
                row.push_back(oss.str());
            }
            W_json.push_back(row);
        }
        j["Ws"].push_back(W_json);
    }

    for (const auto& B : Bs) {
        std::vector<std::string> b_vec;
        for (int i = 0; i < B.cols(); ++i) {
            std::ostringstream oss;
            oss.precision(20);
            oss << B(0, i);
            b_vec.push_back(oss.str());
        }
        j["Bs"].push_back(b_vec);
    }

    return j;
}

// Désérialisation depuis JSON
void Network::from_json(const json& j) {
    couche = j.at("couche").get<size_t>();
    generation = j.at("generation").get<int>();
    alpha = stold(j.at("alpha").get<string>());
    max_iter = j.at("max_iter").get<int>();
    eps = stold(j.at("eps").get<string>());
    neuronnes = j.at("neuronnes").get<vector<int>>();
    affichage_gen = j.at("affichage_gen").get<bool>();
    need_to_train = j.at("need_to_train").get<bool>();

    Ws.clear();
    for (const auto& W_json : j.at("Ws")) {
        size_t rows = W_json.size();
        size_t cols = W_json[0].size();
        Matrixld W(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                W(i, j) = stold(W_json[i][j].get<string>());
        Ws.push_back(W);
    }

    Bs.clear();
    for (const auto& B_json : j.at("Bs")) {
        RowVectorld B(1, B_json.size());
        for (size_t i = 0; i < B_json.size(); ++i)
            B(0, i) = stold(B_json[i].get<string>());
        Bs.push_back(B);
    }
}

// Sauvegarde dans un fichier JSON
void Network::save_to_file(const string& filename) const {
    ofstream file(filename);
    file << to_json().dump(4); // Beautify JSON
    file.close();
}

// Chargement depuis un fichier JSON
void Network::load_from_file(const string& filename) {
    ifstream file(filename);
    json j;
    file >> j;
    from_json(j);
    file.close();
}

void Network::afficher_graphique() {
    cout << "AAA" << endl;
}


void Network::ajoute_couche(int neuronnes, function<Matrixld(const Matrixld&)> activation, function<Matrixld(const Matrixld&)> derivee){
    couche++;
    this->neuronnes.push_back(neuronnes);
    this->activations.push_back(activation);
    this->derivees.push_back(derivee);
}

void Network::init_train(long double alpha, long double eps, int max_iter, bool affichage_gen, function<long double(const Matrixld&,const Matrixld&)> fonction_cout){
    this->alpha =alpha;
    this->affichage_gen = affichage_gen;
    this->eps = eps;
    this->max_iter = max_iter;
    this->fonction_cout = fonction_cout;
    init_params();
}

