#include <iostream>
#include <fstream>
#include <string>
#include <windows.h>
#include <tuple>
#include <vector>
#include <algorithm>

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

    bool isHamiltonianCycle(const vector<int>& permutation, int** adjacencyMatrix) const {
        for (int i = 0; i < v; ++i) {
            int from = permutation[i];
            int to = permutation[(i + 1) % v];
            if (adjacencyMatrix[from][to] == 0) {
                return false;
            }
        }
        return true;
    }

    int countMissingEdges(const vector<int>& permutation, int** adjacencyMatrix) const {
        int missingEdges = 0;
        for (int i = 0; i < v; ++i) {
            int from = permutation[i];
            int to = permutation[(i + 1) % v];
            if (adjacencyMatrix[from][to] == 0) {
                ++missingEdges;
            }
        }
        return missingEdges;
    }

    std::tuple<int, int> findHamiltonianExtension_exact() const {
        vector<int> permutation(v);
        for (int i = 0; i < v; ++i) {
            permutation[i] = i;
        }

        int minEdgesToAdd = INT_MAX;
        vector<int> bestPermutation;

        do {
            int missingEdges = countMissingEdges(permutation, adjacencyMatrix);
            if (missingEdges < minEdgesToAdd) {
                minEdgesToAdd = missingEdges;
                bestPermutation = permutation;
            }
        } while (std::next_permutation(permutation.begin(), permutation.end()));

        int** adjacencyMatrixCopy = new int* [v];
        for (int i = 0; i < v; i++) {
            adjacencyMatrixCopy[i] = new int[v];
            for (int j = 0; j < v; j++) {
                adjacencyMatrixCopy[i][j] = adjacencyMatrix[i][j];
            }
        }

        for (int i = 0; i < v; ++i) {
            int from = bestPermutation[i];
            int to = bestPermutation[(i + 1) % v];
            adjacencyMatrixCopy[from][to] = 1;
        }

        int hamiltonianCycles = 0;

        do {
            if (isHamiltonianCycle(permutation, adjacencyMatrixCopy)) {
                ++hamiltonianCycles;
            }
        } while (next_permutation(permutation.begin(), permutation.end()));

        hamiltonianCycles /= v;

        for (int i = 0; i < v; i++) {
            delete[] adjacencyMatrixCopy[i];
        }
        delete[] adjacencyMatrixCopy;

        return std::make_tuple(minEdgesToAdd, hamiltonianCycles);
    }

    std::tuple<int, std::vector<int>> nearestNeighbor(int start) const {
        vector<int> path;
        path.push_back(start);

        vector<bool> visited(v, false);
        visited[start] = true;
        int current = start;
        int missingEdges = 0;

        for (int i = 1; i < v; ++i) {
            int next = -1;
            for (int j = 0; j < v; ++j) {
                if (!visited[j] && (next == -1 || adjacencyMatrix[current][j] > adjacencyMatrix[current][next])) {
                    next = j;
                }
            }

            if (adjacencyMatrix[current][next] == 0) {
                ++missingEdges;
            }

            visited[next] = true;
            path.push_back(next);
            current = next;
        }

        if (adjacencyMatrix[current][start] == 0) {
            ++missingEdges;
        }

        return std::make_tuple(missingEdges, path);
    }

    std::tuple<int, int> findHamiltonianExtension_approx() const {
        int minMissingEdges = INT_MAX;
        std::vector<int> hamiltonianPath;

        for (int start = 0; start < v; ++start) {
            auto result = nearestNeighbor(start);
            int missingEdges = std::get<0>(result);
            if (minMissingEdges > missingEdges) {
                minMissingEdges = missingEdges;
                hamiltonianPath = std::get<1>(result);
            }
        }

        // TODO: Approximate number of hamiltionian cycles
        return std::make_tuple(minMissingEdges, 1);
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
    for (int i = 0; i < numGraphs; i++) {
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

        auto a = graphs[i]->findHamiltonianExtension_exact();
        std::cout << "Algorytm dokladny: minimalne rozszerzenie: " << std::get<0>(a) << ", ilosc cykli hamiltona: " << std::get<1>(a) << std::endl;

        auto b = graphs[i]->findHamiltonianExtension_approx();
        std::cout << "Algorytm aproksymujacy: minimalne rozszerzenie: " << std::get<0>(b) << ", ilosc cykli hamiltona: " << std::get<1>(b) << std::endl;
    }

    deleteGraphs(graphs, numGraphs);

    return 0;
}
