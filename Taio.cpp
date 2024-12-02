#include <iostream>
#include <fstream>
#include <string>
#include <windows.h>
#include <tuple>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <set>
#include <random>

using namespace std;
struct Graph {
    int v;
    int e;
    int** adjacencyMatrix;

    Graph(int v) {
        this->v = v;
        this->e = 0;
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
        std::cout << "Liczba krawedzi: " << e << "\n";
        std::cout << "Rozmiar grafu: " << graphSize() << "\n";
        std::cout << "Macierz sasiedztwa:\n";
        for (int i = 0; i < v; i++) {
            for (int j = 0; j < v; j++) {
                std::cout << adjacencyMatrix[i][j] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }

    int graphSize() const {
        return e + v;
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

    int estimateUniqueHamiltonianCycles(int startVertex, int iterations, std::set<std::vector<int>>& uniqueHamiltonianCycles, int** adjacencyMatrix) {
        std::random_device rd;
        std::mt19937 gen(rd());

        for (int i = 0; i < iterations; ++i) {
            vector<int> path;
            vector<bool> visited(v, false);
            path.push_back(startVertex);
            visited[startVertex] = true;

            int current = startVertex;
            for (int j = 1; j < v; ++j) {
                vector<int> candidates;
                for (int k = 0; k < v; ++k) {
                    if (!visited[k] && adjacencyMatrix[current][k] == 1) {
                        candidates.push_back(k);
                    }
                }

                if (candidates.empty()) break;

                uniform_int_distribution<> dist(0, candidates.size() - 1);
                int next = candidates[dist(gen)];

                path.push_back(next);
                visited[next] = true;
                current = next;
            }

            if (path.size() == v && adjacencyMatrix[current][startVertex] == 1) {
                uniqueHamiltonianCycles.insert(path);
            }
        }

        return uniqueHamiltonianCycles.size();
    }

    std::tuple<int, int> findHamiltonianExtension_approx() {
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

        int** adjacencyMatrixCopy = new int* [v];
        for (int i = 0; i < v; i++) {
            adjacencyMatrixCopy[i] = new int[v];
            for (int j = 0; j < v; j++) {
                adjacencyMatrixCopy[i][j] = adjacencyMatrix[i][j];
            }
        }

        for (int i = 0; i < v; ++i) {
            int from = hamiltonianPath[i];
            int to = hamiltonianPath[(i + 1) % v];
            adjacencyMatrixCopy[from][to] = 1;
        }

        std::set<std::vector<int>> uniqueCycles;
        uniqueCycles.insert(hamiltonianPath);

        int cycles = estimateUniqueHamiltonianCycles(hamiltonianPath.front(), graphSize()*graphSize(), uniqueCycles, adjacencyMatrixCopy);

        return std::make_tuple(minMissingEdges, cycles);
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
        int v, e = 0;
        file >> v;

        graphs[g] = new Graph(v);

        for (int i = 0; i < v; i++) {
            for (int j = 0; j < v; j++) {
                file >> graphs[g]->adjacencyMatrix[i][j];
                e += graphs[g]->adjacencyMatrix[i][j];
            }
        }

        graphs[g]->e = e;
    }

    file.close();
    return graphs;
}

Graph createRandomGraph(int v, double edgeProbability) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    Graph g(v);
    int e = 0;
    for (int i = 0; i < v; i++) {
        for (int j = 0; j < v; j++) {
            if (i == j) {
                continue;
            }
            g.adjacencyMatrix[i][j] = dis(gen) <= edgeProbability;
            e += g.adjacencyMatrix[i][j];
        }
    }
    g.e = e;
    return g;
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

    std::cout << "/n";

    Graph g = createRandomGraph(10, 0.7);
    g.print();
    auto a = g.findHamiltonianExtension_exact();
    std::cout << "Algorytm dokladny: minimalne rozszerzenie: " << std::get<0>(a) << ", ilosc cykli hamiltona: " << std::get<1>(a) << std::endl;

    auto b = g.findHamiltonianExtension_approx();
    std::cout << "Algorytm aproksymujacy: minimalne rozszerzenie: " << std::get<0>(b) << ", ilosc cykli hamiltona: " << std::get<1>(b) << std::endl;
    deleteGraphs(graphs, numGraphs);

    return 0;
}
