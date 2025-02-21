#include "type.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>
#include "nlohmann/json.hpp"
#include <chrono>
#include "activation_fonction_matrix.hpp"

using json = nlohmann::json;
using namespace std;
using namespace chrono;

//Le bon nom pour génération est epoch, plus ce reseau marche que pour softmax

class Network{
    public:
        size_t couche; // Le nombre de couche du réseau
        int generation; // La générations actuelles
        long double alpha; // taux d'apprentissage
        int max_iter; // Le nombre d'itération max
        long double eps; // La limite sur la norme des gradients
        vector<int> neuronnes; // Le nombre de neuronnes par couches
        vector<Matrixld> Ws; // Une liste des poids pour chaque couche
        vector<RowVectorld> Bs; // Une liste des biais pour chaque couche
        vector<function<Matrixld(const Matrixld&)>> activations; // Une liste des fonctions d'activations
        vector<function<Matrixld(const Matrixld&)>> derivees; //Les dérivées des fonctions d'activations
        vector<function<Matrixld(const Matrixld&,const Matrixld&,const Matrixld&)>> formule_gradient_W; // Une liste des fonctions pour calculer les gradients pour chaque couche
        vector<function<RowVectorld(const Matrixld&,const Matrixld&)>> formule_gradient_B;
        function<long double(const Matrixld&,const Matrixld&)> fonction_cout; //La fonction de cout utilisée
        bool affichage_gen = false;
        bool need_to_train;

    Network(
        size_t couche,
        long double alpha,
        long double eps,
        vector<int> neuronnes,
        vector<function<Matrixld(const Matrixld&)>> activations,
        vector<function<Matrixld(const Matrixld&)>> derivees,
        vector<function<Matrixld(const Matrixld&,const Matrixld&,const Matrixld&)>> formule_gradient_W,
        vector<function<RowVectorld(const Matrixld&,const Matrixld&)>> formule_gradient_B
        );

    Network(
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
        );

    Network(
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
        );/* Créé toujours, avec les mêmes poids initiaux, ou stochastique */


    Network();

    void afficher_cmd();

    void afficher_graphique();

    Matrixld predict(Matrixld X);
    
    /* E les résultats attendue , X les données pour l'entrainement, entraine jusqu'au nombre maximum d'itération ou selon eps */
    void train_full(Matrixld X,Matrixld E);
    
    /* Train l'ensemble du jeu de données en batch de taille diff*/
    void train_full_batch(vector<Matrixld> X, vector<Matrixld> E);

    /* entraine qu'une génération, pour un batch qui contient tout l'ensemble d'entrainement */
    void train_1_gen(Matrixld X,Matrixld E); 

    // Sérialisation en JSON
    json to_json() const; 

    // Désérialisation depuis JSON
    void from_json(const json& j);

    // Sauvegarde dans un fichier JSON
    void save_to_file(const string& filename) const;

    // Chargement depuis un fichier JSON
    void load_from_file(const string& filename);

    void ajoute_couche(int neuronnes, function<Matrixld(const Matrixld&)> activation, function<Matrixld(const Matrixld&)> derivee);

    void init_train(
        long double alpha,
        long double eps,
        int max_iter,
        bool affichage_gen,
        function<long double(const Matrixld&,const Matrixld&)> fonction_cout
    );

    private:
        void init_params();
        void init_fixe_params();
        
};


