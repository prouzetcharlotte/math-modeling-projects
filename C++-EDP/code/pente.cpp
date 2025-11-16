#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <fstream>
using namespace::std;

// Constantes
const double C = 1.0;
const double D = 1.0;
const int Nt = 50;
const double L = 1.0;

// Fonction linspace comme numpy
std::vector<double> linspace(double start, double end, int N) {
    std::vector<double> vect(N);
    double pas = (end - start) / (N - 1);
    for (int i = 0; i < N; ++i) vect[i] = start + i * pas;
    return vect;
}

/*Cas_test_1
double u_exacte(double x, double t) {
    return std::cos(M_PI * x) * (1 + t);
}

double f(double x, double t) {
    return std::cos(M_PI * x) + D*(std::pow(M_PI,2)*std::cos(M_PI*x)*(1+t)) - C*(M_PI*std::sin(M_PI*x)*(1+t));
}
*/
//Cas_test_2
double u_exacte(double x, double t) {
    return std::sin(M_PI*x)*std::exp(-t);
}

// Terme source
double f(double x, double t) {
    return -std::sin(M_PI*x)*std::exp(-t)+D*(M_PI*M_PI)*std::sin(M_PI*x)*std::exp(-t)+C*M_PI*std::cos(M_PI*x)*std::exp(-t);
}

/*Cas_test_3
double u_exacte(double x, double t) {
    return (std::sin(M_PI * x) + std::cos(2*M_PI*x)) * std::exp(-(M_PI*M_PI)*t);
}

// Terme source
double f(double x, double t) {
    return -(M_PI*M_PI)*(std::sin(M_PI*x)+std::cos(2*M_PI*x))*std::exp(-(M_PI*M_PI)*t)+
    D*(M_PI*M_PI)*(std::sin(M_PI*x)+4*std::cos(2*M_PI*x))*std::exp(-(M_PI*M_PI)*t) + 
    C*((M_PI * std::cos(M_PI * x) -2*M_PI*std::sin(2*M_PI*x))*std::exp(-(M_PI*M_PI)*t));
}*/

// Calcul de l'erreur L_inf pour un dt donne
double calcul_ln_erreur(double dt, double dt_max) {
    // Nx calculé comme en Python, en incluant le point final
    int Nx = static_cast<int>(L / std::sqrt(2*D*dt_max)) + 1;
    double T = Nt * dt;

    std::vector<double> x = linspace(0.0, L, Nx);
    std::vector<double> t(Nt); //Nt valeur initiale
    for(int n=0; n<Nt; ++n) t[n] = n*dt;

    std::vector<std::vector<double>> u_calcule(Nx, std::vector<double>(Nt, 0.0));

    // Condition initiale
    for(int i=0; i<Nx; ++i) u_calcule[i][0] = u_exacte(x[i],0.0);

    double v = C*dt / (x[1]-x[0]);
    double gamma = D*dt / std::pow(x[1]-x[0],2);
    assert(gamma <= 0.5);
    assert(std::abs(v) <= 1.0);

    // Boucle en temps
    for(int n=0; n<Nt-1; ++n){
        for(int i=1; i<Nx-1; ++i){
            double du = (C>=0) ? (u_calcule[i][n]-u_calcule[i-1][n])
                                : (u_calcule[i+1][n]-u_calcule[i][n]);
            u_calcule[i][n+1] = u_calcule[i][n] + gamma*(u_calcule[i+1][n]-2*u_calcule[i][n]+u_calcule[i-1][n])
                                 - v*du + f(x[i], t[n])*dt;
        }
        /*//Conditions aux limites Cas_test_1
        u_calcule[0][n+1] = 1 + t[n+1];
        u_calcule[Nx-1][n+1] = -(1 + t[n+1]);*/
        //Conditions aux limites Cas_test_2
        u_calcule[0][n+1] = 0;
        u_calcule[Nx-1][n+1] = 0;

        /*//Conditions aux limites Cas_test_3
        u_calcule[0][n+1]=std::exp(-(M_PI*M_PI) * t[n+1]);
        u_calcule[Nx-1][n+1]=std::exp(-(M_PI*M_PI) * t[n+1]);*/
    }

    // Calcul de l'erreur L_inf finale
    double norme_Linf = 0.0;
    for(int i=0; i<Nx; ++i){
        double e = std::abs(u_calcule[i][Nt-1] - u_exacte(x[i],T));
        if(e > norme_Linf) norme_Linf = e;
    }
    return norme_Linf;
}

// Régression linéaire simple y = a*x + b
std::pair<double,double> regression_lineaire(const std::vector<double>& X, const std::vector<double>& Y){
    int n = X.size();
    double sumX=0, sumY=0, sumXY=0, sumX2=0;
    for(int i=0;i<n;i++){
        sumX += X[i];
        sumY += Y[i];
        sumXY += X[i]*Y[i];
        sumX2 += X[i]*X[i];
    }
    double slope = (n*sumXY - sumX*sumY)/(n*sumX2 - sumX*sumX);
    double intercept = (sumY - slope*sumX)/n;
    return {slope, intercept};
}

int main() {
    std::vector<double> dt_values = {1.0/100000, 1.0/200000, 1.0/400000};
    double dt_max = *std::max_element(dt_values.begin(), dt_values.end());

    std::vector<double> E;
    for(auto dt: dt_values){
        double err = calcul_ln_erreur(dt, dt_max);
        E.push_back(err);
    }

    std::cout << "dt\tErreur L_inf" << std::endl;
    for(size_t i=0;i<E.size();++i) std::cout << dt_values[i] << "\t" << E[i] << std::endl;

    // Logarithme
    std::vector<double> ln_dt, ln_E;
    for(size_t i=0;i<E.size();++i){
        ln_dt.push_back(std::log(dt_values[i]));
        ln_E.push_back(std::log(E[i]));
    }

    auto [slope, intercept] = regression_lineaire(ln_dt, ln_E);
    std::cout << "Pente p (regression lineaire) : " << slope << std::endl;

    //Plot
    std::ofstream file("pente.dat");
    for(size_t i = 0; i < ln_dt.size(); ++i){
        file << ln_dt[i] << "\t" << ln_E[i] << "\n";
    }
    file.close();
    std::cout << "Fichier pente.dat cree." << std::endl;

    return 0;
}

