#include "type.hpp"
#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>
#include "nlohmann/json.hpp"
#include <chrono>
#include "activation_fonction_matrix.hpp"
#include "derivee.hpp"
#include "calcul_erreur.hpp"

using json = nlohmann::json;
using namespace std;
using namespace chrono;

//ce reseau marche que pour softmax

class Network{
    public:
        

        Network(
            vector<int> neuronnes,
            vector<function<Matrixld(const Matrixld&)>> activations,
            vector<function<Matrixld(const Matrixld&)>> derivees
            );

        Network(
            vector<int> neuronnes,
            vector<function<Matrixld(const Matrixld&)>> activations,
            vector<function<Matrixld(const Matrixld&)>> derivees,
            function<long double(const Matrixld&,const Matrixld&)> fonction_cout
            );

        Network(
            vector<int> neuronnes,
            vector<function<Matrixld(const Matrixld&)>> activations,
            vector<function<Matrixld(const Matrixld&)>> derivees,
            function<long double(const Matrixld&,const Matrixld&)> fonction_cout,
            bool same_start // Si true alors initialise toujous les poids et biais pareils.
            );/* Créé toujours, avec les mêmes poids initiaux, ou stochastique */


        Network();

        void afficher_cmd();

        void afficher_graphique();

        Matrixld predict(Matrixld X);
        
        /* E les résultats attendue , X les données pour l'entrainement, entraine jusqu'au nombre maximum d'itération ou selon eps */
        void train_full(Matrixld X,Matrixld E, long double alpha, long double eps, int max_iter, bool affichage = false);
        
        /* Train l'ensemble du jeu de données en batch de taille diff*/
        void train_full_batch(vector<Matrixld> X, vector<Matrixld> E, long double alpha, long double eps, int max_iter, bool affichage = false);

        /* entraine qu'une génération, pour un batch qui contient tout l'ensemble d'entrainement */
        void train_1_gen(Matrixld X,Matrixld E, long double alpha, long double eps,bool affichage);

        void better_train_full(Matrixld X,Matrixld E, long double alpha, long double eps, int max_iter, bool affichage = false, int ratio = 100);
        
        /* Train l'ensemble du jeu de données en batch de taille diff*/
        void better_train_full_batch(vector<Matrixld> X, vector<Matrixld> E, long double alpha, long double eps, int max_iter, bool affichage = false, int ratio = 100);

        // Sérialisation en JSON
        json to_json() const; 

        // Désérialisation depuis JSON
        void from_json(const json& j);

        // Sauvegarde dans un fichier JSON
        void save_to_file(const string& filename) const;

        // Chargement depuis un fichier JSON
        void load_from_file(const string& filename);

        void ajoute_couche(int neuronnes, function<Matrixld(const Matrixld&)> activation, function<Matrixld(const Matrixld&)> derivee);

        void add_layer(int neuronnes, Activation activation);
        void add_enter(int neuronnes);
        void add_exit(int neuronnes, Activation activation);


        void init_train(
            long double alpha,
            long double eps,
            int max_iter,
            bool affichage_gen,
            function<long double(const Matrixld&,const Matrixld&)> fonction_cout
        );

        inline size_t getCouche(){return couche;}
        inline int getEpoch(){return epoch;}
        inline vector<int> getNeuronnes(){return neuronnes;}
        inline vector<Matrixld> getWeigths(){return Ws;}
        inline vector<RowVectorld> getBias(){return Bs;}


        inline string toString() const {
            ostringstream oss;
            oss <<"Network : couches" << couche;
            return oss.str();  
        }

    private:
        bool has_enter = false;
        bool has_exit = false;
        void init_params();
        void init_fixe_params();
        size_t couche; // Le nombre de couche du réseau
        int epoch; // La générations actuelles
        vector<int> neuronnes; // Le nombre de neuronnes par couches
        vector<Matrixld> Ws; // Une liste des poids pour chaque couche
        vector<RowVectorld> Bs; // Une liste des biais pour chaque couche
        vector<function<Matrixld(const Matrixld&)>> activations; // Une liste des fonctions d'activations
        vector<function<Matrixld(const Matrixld&)>> derivees; //Les dérivées des fonctions d'activations
        function<long double(const Matrixld&,const Matrixld&)> fonction_cout; //La fonction de cout utilisée
        bool need_to_train;

        
};


