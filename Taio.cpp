#include <iostream>
#include <fstream>
#include <string>
#include <windows.h>


using namespace std;
struct Graph {
    int v;
    int** adjacencyMatrix;

    Graph(int v) {
        this->v = v;
        adjacencyMatrix = new int* [v];
        for (int i = 0; i < v; i++) {
            adjacencyMatrix[i] = new int[v]();
        }
    }

    ~Graph() {
        for (int i = 0; i < v; i++) {
            delete[] adjacencyMatrix[i];
        }
        delete[] adjacencyMatrix;
    }

    void print() const {
        std::cout << "Liczba wierzcholkow: " << v << "\n";
        std::cout << "Macierz sasiedztwa:\n";
        for (int i = 0; i < v; i++) {
            for (int j = 0; j < v; j++) {
                std::cout << adjacencyMatrix[i][j] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
};

Graph** readGraphsFromFile(const std::string& filename, int& numGraphs) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Nie mozna otworzyc pliku: " + filename);
    }

    file >> numGraphs;

    Graph** graphs = new Graph * [numGraphs];

    for (int g = 0; g < numGraphs; g++) {
        int v;
        file >> v;


        graphs[g] = new Graph(v);

        for (int i = 0; i < v; i++) {
            for (int j = 0; j < v; j++) {
                file >> graphs[g]->adjacencyMatrix[i][j];
            }
        }

    }

    file.close();
    return graphs;
}

void deleteGraphs(Graph** graphs, int numGraphs) {
    for (int i = 0; i < numGraphs;i++) {
        delete graphs[i];
    }
    delete[] graphs;
}

int main() {
    const string filename = "dane.txt";
    int numGraphs;

        Graph** graphs = readGraphsFromFile(filename, numGraphs);

        std::cout << "Wczytano " << numGraphs << " grafow:\n";
        for (int i = 0; i < numGraphs; i++) {
            graphs[i]->print();
        }

        deleteGraphs(graphs, numGraphs);

    return 0;
}
