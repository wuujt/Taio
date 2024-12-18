#include <iostream>
#include <fstream>
#include <string>
#include <windows.h>
#include <vector>
#include <algorithm>
#include <climits>
#include <math.h>
#include <vector>
#include <queue>
#include <set>
#include <unordered_set>
#include <chrono>
#include <cstdlib>
#include <tuple>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <set>
#include <random>

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

    void addEdge(int u, int v) {
        adjacencyMatrix[u][v] = 1;
    }

    int calculateEdges() {
        int e = 0;
        for (int i = 0; i < v; i++) {
            for (int j = 0; j < v; j++) {
                e += adjacencyMatrix[i][j];
            }
        }
        return e;
    }

    vector<int> calculateDegrees() {
        vector<int> degrees(v, 0);
        for (int i = 0; i < v; i++) {
            for (int j = 0; j < v; j++) {
                degrees[i] += adjacencyMatrix[i][j];
            }
        }
        return degrees;
    }

    void print() const {
        cout << "Number of vertices: " << v << endl;
        cout << "Adjacency Matrix:" << endl;
        for (int i = 0; i < v; i++) {
            for (int j = 0; j < v; j++) {
                cout << adjacencyMatrix[i][j] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    int graphSize() {
        return calculateEdges() + v;
    }

};

//console_helpers
int handleOptions(int argc, char* argv[]);
void handleMetric(Graph& g1, Graph& g2, bool approx);
void handleCycles(Graph& graph, bool approx);
void handleHamilton(Graph& graph, bool approx);

//distance
int calculateDistance(Graph& G1, Graph& G2);
int approximateDistance(Graph& G1, Graph& G2);

//max_cycle
pair<int, pair<int, set<vector<int>>>> findMaxCycle(Graph& graph);
void backtrackingMaxCycle(Graph& graph, int currentVertex, int startVertex,
    set<int>& visited, int cycleLength,
    int& maxCycleLength, int& maxCycleCount,
    vector<int>& currentCycle, set<vector<int>>& allCycles);

//max_cycle_approx
pair<int, pair<int, vector<vector<int>>>> findLongestCyclesApproximation(const Graph& graph);
vector<pair<int, int>> getSpanningForest(const Graph& graph);
vector<pair<int, int>> getExtraEdges(const Graph& graph, const vector<pair<int, int>>& spanningForest);
void dfs(int node, const Graph& graph, vector<bool>& visited, vector<pair<int, int>>& spanningForest);
vector<int> findPath(int start, int end, const vector<pair<int, int>>& spanningForest, int vertices);

//hamiltonian_extension
bool isHamiltonianCycle(const vector<int>& permutation, Graph& g);
int countMissingEdges(const vector<int>& permutation, Graph& g);
tuple<int, int, set<vector<int>>, vector<vector<int>>> findHamiltonianExtension_exact(Graph& g);

//hamiltonian_extension_approx
tuple<int, vector<int>> nearestNeighbor(int start, Graph& g);
int estimateUniqueHamiltonianCycles(int startVertex, int iterations, set<vector<int>>& uniqueHamiltonianCycles, Graph& g);
tuple<int, int, set<vector<int>>, vector<vector<int>>> findHamiltonianExtension_approx(Graph& g);

//utils
Graph** readGraphsFromFile(const string& filename, int& numGraphs);
void deleteGraphs(Graph** graphs, int numGraphs);
vector<int> normalizeCycle(const vector<int>& cycle);
Graph extendGraph(Graph& G, int newSize);
void printCycles(const vector<vector<int>>& cycles);
void printUsage();
void runTests();


Graph generateRandomGraph(int n) {
    Graph graph(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j) {
                graph.adjacencyMatrix[i][j] = rand() % 2;
            }
        }
    }
    return graph;
}

template <typename Func>
double measureExecutionTime(Func function, Graph& graph1, Graph& graph2) {
    auto start = chrono::high_resolution_clock::now();
    function(graph1, graph2);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

void testFunctions(int n,
    int (*function1)(Graph&, Graph&),
    int (*function2)(Graph&, Graph&)) {

    Graph graph1 = generateRandomGraph(n);
    Graph graph2 = generateRandomGraph(n);
    cout << "Graph size: " << n << endl;

    double time1 = measureExecutionTime(function1, graph1, graph2);
    cout << "Execution time for function1: " << time1 << " seconds" << endl;

    double time2 = measureExecutionTime(function2, graph1, graph2);
    cout << "Execution time for function2: " << time2 << " seconds" << endl;
    cout << endl;
}

template <typename Func, typename... Args>
double measureExecutionTime2(Func function, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();
    function(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

// Flexible test function using templates
template <typename Func1, typename Func2>
void testFunctions2(int n,
    Func1 function1,
    Func2 function2)
{
    // Generate two random graphs
    Graph graph1 = generateRandomGraph(n);
    Graph graph2 = generateRandomGraph(n);
    cout << "Graph size: " << n << endl;
    double time1 = measureExecutionTime2(function1, graph1);
    cout << "Execution time for function1: " << time1 << " seconds" << endl;

    double time2 = measureExecutionTime2(function2, graph1);
    cout << "Execution time for function2: " << time2 << " seconds" << endl;
    cout << endl;

}

template <typename Func1>
void testOneFunction(int n,
    Func1 function1)
{
    Graph graph1 = generateRandomGraph(n);
    cout << "Graph size: " << n << endl;
    double time1 = measureExecutionTime2(function1, graph1);
    cout << "Execution time for function: " << time1 << " seconds" << endl;
    cout << endl;
}

void testOneFunctionBetween2Graphs(int n, int (*function1)(Graph&, Graph&))
{
    Graph graph1 = generateRandomGraph(n);
    Graph graph2 = generateRandomGraph(n);
    cout << "Graph size: " << n << endl;
    double time1 = measureExecutionTime(function1, graph1, graph2);
    cout << "Execution time for function: " << time1 << " seconds" << endl;
    cout << endl;
}

int main(int argc, char* argv[]) {


    if (argc == 2 && string(argv[1]) == "test") {
        runTests();
        return 0;
    }
    if (argc >= 4) {
        handleOptions(argc, argv);
        return 0;
    }
    else {
        printUsage();
        return 1;
    }

    // string filename = "dane.txt";
    // if (argc > 1) {
    //     filename = argv[1];
    // }
    // int numGraphs;

    // //for (int i = 2; i < 20; i = i++)
    //   //  testFunctions2(i, findMaxCycle, findLongestCyclesApproximation);

    // //for (int i = 2; i < 20; i = i++)
    //  //   testFunctions(i, calculateDistance, approximateDistance);

    //// runHamiltonTests();

    // Graph** graphs = readGraphsFromFile(filename, numGraphs);

    // cout << "Wczytano " << numGraphs << " grafow:" << endl;
    // for (int i = 0; i < numGraphs; i++) {
    //     graphs[i]->print();
    // }
    // if (numGraphs > 1)
    // {
    //     cout << "Distance: " << calculateDistance(*graphs[0], *graphs[1]) << endl;
    //     cout << "Approx distance: " << approximateDistance(*graphs[0], *graphs[1]) << endl;
    // }
    // cout << endl;
    // pair<int, pair<int, vector<vector<int>>>> approxMaxCycles = findLongestCyclesApproximation(*graphs[0]);
    // pair<int, pair<int, set<vector<int>>>> maxCycles = findMaxCycle(*graphs[0]);

    // int cycleLength = maxCycles.first;
    // int numberOfMaxCycles = maxCycles.second.first;
    // set<vector<int>> cycles = maxCycles.second.second;

    // cout << "Max cycle:" << endl;
    // cout << "Length of longest cycle " << cycleLength << endl;
    // cout << "Number of cycles: " << numberOfMaxCycles << endl;
    // cout << "Cycles: " << endl;
    // for (const auto& cycle : cycles) {
    //     for (int node : cycle) {
    //         cout << node << " ";
    //     }
    //     cout << endl;
    // }
    // cout << endl;
    // cycleLength = approxMaxCycles.first;
    // numberOfMaxCycles = approxMaxCycles.second.first;
    // vector<vector<int>> cycles2 = approxMaxCycles.second.second;

    // cout << "Approx max cycle:" << endl;
    // cout << "Length of longest cycle " << cycleLength << endl;
    // cout << "Number of cycles: " << numberOfMaxCycles << endl;
    // cout << "Cycles:" << endl;
    // for (const auto& cycle : cycles2) {
    //     for (int node : cycle) {
    //         cout << node << " ";
    //     }
    //     cout << endl;
    // }


    // auto hamiltonianExtension = findHamiltonianExtension_exact(*graphs[0]);
    // auto hamiltonianExtension_approx = findHamiltonianExtension_approx(*graphs[0]);
    // cout << "Hamiltonian extension:" << endl;
    // cout << "Numer of edges to add " << get<0>(hamiltonianExtension) << endl;
    // cout << "Number of hamiltonian cycles " << get<1>(hamiltonianExtension) << endl;

    // cout << "Hamiltonian extension approx:" << endl;
    // cout << "Numer of edges to add " << get<0>(hamiltonianExtension_approx) << endl;
    // cout << "Number of hamiltonian cycles " << get<1>(hamiltonianExtension_approx) << endl;

    // deleteGraphs(graphs, numGraphs);

    //return 0;
}

void printUsage() {
    cout << "Usage:" << endl;
    cout << "./Taio.exe test" << endl;
    cout << "to run performance tests on random graphs or:" << endl;
    cout << "./Taio.exe [filename] distance [graph_index1] [graph_index2] [approx]" << endl;
    cout << "./Taio.exe [filename] [function] [graph_index] [approx]" << endl;
    cout << "Options for [function]: cycles or hamilton" << endl;
    cout << "[approx]: Leave empty for exact function or use 'approx' for an approximate version" << endl;
    cout << "Examples: " << endl;
    cout << "./Taio.exe dane.txt cycles 1" << endl;
    cout << "./Taio.exe dane.txt hamilton 2 approx" << endl;
}

void runTests() {
    cout << "Running random tests:" << endl;
    cout << endl << "Running tests for finding max cycle: function 1 is exact, function 2 is approximation" << endl;
    for (int i = 2; i < 10; i = i++)
        testFunctions2(i, findMaxCycle, findLongestCyclesApproximation);
    cout << endl << "Running tests for finding distance: function 1 is exact, function 2 is approximation" << endl;
    for (int i = 2; i < 10; i = i++)
        testFunctions(i, calculateDistance, approximateDistance);
    cout << endl << "Running tests for finding hamilton extension: function 1 is exact, function 2 is approximation" << endl;
    for (int i = 2; i < 10; i = i++)
        testFunctions2(i, findHamiltonianExtension_exact, findHamiltonianExtension_approx);
    cout << endl << "Running tests for approximating finding max cycles:" << endl;
    for (int i = 4; i < 40; i += 4)
        testOneFunction(i, findLongestCyclesApproximation);
    cout << endl << "Running tests for approximating finding distance:" << endl;
    for (int i = 4; i < 40; i += 4)
        testOneFunctionBetween2Graphs(i, approximateDistance);
    cout << endl << "Running tests for approximating finding hamiltonian extension:" << endl;
    for (int i = 4; i < 40; i += 4)
        testOneFunction(i, findHamiltonianExtension_approx);

}
#pragma region console_helpers

int handleOptions(int argc, char* argv[]) {
    string filename = argv[1];
    string function = argv[2];
    int numGraphs;

    Graph** graphs = readGraphsFromFile(filename, numGraphs);

    try {
        if (function == "distance") {
            int index1 = std::stoi(argv[3]);
            int index2 = std::stoi(argv[4]);
            Graph graph1 = *graphs[index1];
            Graph graph2 = *graphs[index2];
            bool approx = false;
            if (argc == 6 && string(argv[5]) == "approx") {
                approx = true;
                cout << "Approx ";
            }
            handleMetric(graph1, graph2, approx);
        }
        else {
            int index = std::stoi(argv[3]);
            Graph graph = *graphs[index];

            if (function == "cycles") {

                bool approx = false;
                if (argc == 5 && string(argv[4]) == "approx") {
                    approx = true;
                    cout << "Approx ";
                }
                handleCycles(graph, approx);
            }
            else if (function == "hamilton") {
                bool approx = false;
                if (argc == 5 && string(argv[4]) == "approx") {
                    approx = true;
                    cout << "Approx ";
                }
                handleHamilton(graph, approx);
            }
            else {
                std::cerr << "Unknown function: " << function << std::endl;
                printUsage();
                deleteGraphs(graphs, numGraphs);
                return 1;
            }
        }

    }
    catch (const std::exception& e) {
        cerr << "Error: " << e.what() << endl;
        printUsage();
        deleteGraphs(graphs, numGraphs);
        return 1;
    }

    return 0;
}

void handleMetric(Graph& g1, Graph& g2, bool approx) {
    cout << "Distance between graphs: " << endl;
    cout << "First graph:" << endl;
    g1.print();
    cout << "Second graph:" << endl;
    g2.print();
    auto start = std::chrono::high_resolution_clock::now();
    int distance = approx ? approximateDistance(g1, g2) : calculateDistance(g1, g2);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double time = elapsed.count();
    cout << "Execution time: " << time << " seconds" << endl;
    cout << "Distance: " << distance << endl;
}

void handleCycles(Graph& graph, bool approx) {
    cout << "Max cycles:" << endl;
    cout << "Graph:" << endl;
    graph.print();
    auto start = std::chrono::high_resolution_clock::now();
    pair<int, pair<int, vector<vector<int>>>> approxMaxCycles;
    pair<int, pair<int, set<vector<int>>>> maxCycles;
    if (approx) {
        approxMaxCycles = findLongestCyclesApproximation(graph);
    }
    else {
        maxCycles = findMaxCycle(graph);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double time = elapsed.count();
    cout << "Execution time: " << time << " seconds" << endl;

    if (approx) {
        int cycleLength = approxMaxCycles.first;
        int numberOfMaxCycles = approxMaxCycles.second.first;
        vector<vector<int>> cycles = approxMaxCycles.second.second;

        cout << "Length of longest cycle " << cycleLength << endl;
        cout << "Number of cycles: " << numberOfMaxCycles << endl;
        cout << "Cycles: " << endl;
        for (const auto& cycle : cycles) {
            for (int node : cycle) {
                cout << node << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
    else {
        int cycleLength = maxCycles.first;
        int numberOfMaxCycles = maxCycles.second.first;
        set<vector<int>> cycles = maxCycles.second.second;

        cout << "Length of longest cycle " << cycleLength << endl;
        cout << "Number of cycles: " << numberOfMaxCycles << endl;
        cout << "Cycles: " << endl;
        for (const auto& cycle : cycles) {
            for (int node : cycle) {
                cout << node << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
}

void handleHamilton(Graph& graph, bool approx) {

    cout << "Hamiltonian extension:" << endl;
    cout << "Graph:" << endl;
    graph.print();
    auto start = std::chrono::high_resolution_clock::now();
    auto hamiltonianExtension = approx ? findHamiltonianExtension_approx(graph) : findHamiltonianExtension_exact(graph);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    double time = elapsed.count();
    cout << "Execution time: " << time << " seconds" << endl;
    cout << "Edges to add: " << endl;
    auto& extension = get<3>(hamiltonianExtension);
    for (const auto& row : extension) {
        for (const auto& value : row) {
            cout << value << " ";
}
        cout << endl;
    }
    cout << "Numer of edges to add: " << get<0>(hamiltonianExtension) << endl;
    cout << "Number of hamiltonian cycles: " << get<1>(hamiltonianExtension) << endl;
    cout << "List of hamiltonian cycles: " << endl;
    auto& vec = get<2>(hamiltonianExtension);
    for (const auto& cycle : vec) {
        for (int i = 0; i < cycle.size(); i++) {
            cout << cycle[i] << " ";
        }
        cout << cycle[0];
        cout << endl;
    }
    cout << endl;
}

#pragma endregion

#pragma region distance
int calculateDistance(Graph& G1, Graph& G2) {
    int n1 = G1.v;
    int n2 = G2.v;
    int n = max(n1, n2);

    Graph extendedG1 = extendGraph(G1, n);
    Graph extendedG2 = extendGraph(G2, n);

    vector<int> permutation(n);
    for (int i = 0; i < n; i++) permutation[i] = i;

    int minDistance = INT_MAX;

    do {
        int** transformedMatrix = new int* [n];
        for (int i = 0; i < n; i++) {
            transformedMatrix[i] = new int[n]();
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i < n2 && j < n2) {
                    transformedMatrix[permutation[i]][permutation[j]] = extendedG2.adjacencyMatrix[i][j];
                }
            }
        }
        int diff = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (extendedG1.adjacencyMatrix[i][j] != transformedMatrix[i][j]) diff++;
            }
        }

        minDistance = min(minDistance, diff);

        for (int i = 0; i < n; i++) {
            delete[] transformedMatrix[i];
        }
        delete[] transformedMatrix;

    } while (next_permutation(permutation.begin(), permutation.end()));


    return minDistance;
}

int approximateDistance(Graph& G1, Graph& G2) {
    int n1 = G1.v;
    int n2 = G2.v;
    int n = max(n1, n2);

    Graph extendedG1 = extendGraph(G1, n);
    Graph extendedG2 = extendGraph(G2, n);

    vector<int> degreesG1 = extendedG1.calculateDegrees();
    vector<int> degreesG2 = extendedG2.calculateDegrees();

    vector<int> orderG1(n), orderG2(n);
    for (int i = 0; i < n; i++) {
        orderG1[i] = i;
        orderG2[i] = i;
    }

    sort(orderG1.begin(), orderG1.end(), [&degreesG1](int a, int b) {
        return degreesG1[a] > degreesG1[b];
        });

    sort(orderG2.begin(), orderG2.end(), [&degreesG2](int a, int b) {
        return degreesG2[a] > degreesG2[b];
        });

    vector<int> mapping(n);
    for (int i = 0; i < n; i++) {
        mapping[orderG1[i]] = orderG2[i];
    }

    int differences = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (extendedG1.adjacencyMatrix[mapping[i]][mapping[j]] != extendedG2.adjacencyMatrix[i][j]) {
                differences++;
            }
        }
    }

    return differences;
}

#pragma endregion
#pragma region max_cycle

pair<int, pair<int, set<vector<int>>>> findMaxCycle(Graph& graph) {
    int maxCycleLength = 0;
    int maxCycleCount = 0;
    set<vector<int>> allCycles;

    for (int vertex = 0; vertex < graph.v; ++vertex) {
        set<int> visited;
        vector<int> currentCycle;
        backtrackingMaxCycle(graph, vertex, vertex, visited, 0, maxCycleLength,
            maxCycleCount, currentCycle, allCycles);
    }

    return { allCycles.size() > 0 ? maxCycleLength + 1 : 0, {allCycles.size(), allCycles} };
}

void backtrackingMaxCycle(Graph& graph, int currentVertex, int startVertex,
    set<int>& visited, int cycleLength,
    int& maxCycleLength, int& maxCycleCount,
    vector<int>& currentCycle, set<vector<int>>& allCycles) {

    visited.insert(currentVertex);
    currentCycle.push_back(currentVertex);

    for (int neighbor = 0; neighbor < graph.v; ++neighbor) {
        if (graph.adjacencyMatrix[currentVertex][neighbor] == 1) {
            if (neighbor == startVertex && cycleLength >=1) {
                vector<int> normalizedCycle = normalizeCycle(currentCycle);
                if (cycleLength > maxCycleLength) {
                    maxCycleLength = cycleLength;
                    maxCycleCount = 1;
                    allCycles.clear();
                    normalizedCycle.push_back(normalizedCycle[0]);
                    allCycles.insert(normalizedCycle);
                }
                else if (cycleLength == maxCycleLength) {
                    normalizedCycle.push_back(normalizedCycle[0]);
                    ++maxCycleCount;
                    allCycles.insert(normalizedCycle);
                }
            }
            else if (visited.find(neighbor) == visited.end()) {
                backtrackingMaxCycle(graph, neighbor, startVertex, visited,
                    cycleLength + 1, maxCycleLength, maxCycleCount,
                    currentCycle, allCycles);
            }
        }
    }

    visited.erase(currentVertex);
    currentCycle.pop_back();
}

#pragma endregion

#pragma region max_cycle_approx

pair<int, pair<int, vector<vector<int>>>> findLongestCyclesApproximation(const Graph& graph) {
    vector<pair<int, int>> spanningForest = getSpanningForest(graph);
    vector<pair<int, int>> extraEdges = getExtraEdges(graph, spanningForest);

    vector<vector<int>> cycles;
    int longestLength = 0;

    for (const auto& edge : extraEdges) {
        int u = edge.first, v = edge.second;

        vector<int> path = findPath(v, u, spanningForest, graph.v);

        if (!path.empty()) {
            path.push_back(v);
            cycles.push_back(path);
            longestLength = max(longestLength, (int)path.size());
        }
    }

    vector<vector<int>> longestCycles;
    for (const auto& cycle : cycles) {
        if (cycle.size() == longestLength) {
            longestCycles.push_back(cycle);
        }
    }

    return { longestLength - 1,{longestCycles.size(),longestCycles} };
}
vector<pair<int, int>> getSpanningForest(const Graph& graph) {
    vector<bool> visited(graph.v, false);
    vector<pair<int, int>> spanningForest;

    for (int i = 0; i < graph.v; i++) {
        if (!visited[i]) {
            dfs(i, graph, visited, spanningForest);
        }
    }
    return spanningForest;
}

vector<pair<int, int>> getExtraEdges(const Graph& graph, const vector<pair<int, int>>& spanningForest) {
    set<pair<int, int>> forestEdges;
    for (const auto& edge : spanningForest) {
        forestEdges.insert(edge);
    }

    vector<pair<int, int>> extraEdges;

    for (int i = 0; i < graph.v; i++) {
        for (int j = 0; j < graph.v; j++) {
            if (graph.adjacencyMatrix[i][j] == 1) {
                if (forestEdges.find({ i, j }) == forestEdges.end()) {
                    extraEdges.push_back({ i, j });
                }
            }
        }
    }

    return extraEdges;
}



void dfs(int node, const Graph& graph, vector<bool>& visited, vector<pair<int, int>>& spanningForest) {
    visited[node] = true;
    for (int i = 0; i < graph.v; i++) {
        if (graph.adjacencyMatrix[node][i] == 1 && !visited[i]) {
            spanningForest.push_back({ node, i });
            dfs(i, graph, visited, spanningForest);
        }
    }
}

vector<int> findPath(int start, int end, const vector<pair<int, int>>& spanningForest, int vertices) {
    vector<vector<int>> treeAdj(vertices);
    for (const auto& edge : spanningForest) {
        treeAdj[edge.first].push_back(edge.second);
    }

    vector<int> parent(vertices, -1);
    queue<int> q;
    q.push(start);
    parent[start] = start;

    while (!q.empty()) {
        int node = q.front();
        q.pop();

        if (node == end) break;

        for (int neighbor : treeAdj[node]) {
            if (parent[neighbor] == -1) {
                parent[neighbor] = node;
                q.push(neighbor);
            }
        }
    }

    vector<int> path;
    if (parent[end] == -1) return path;

    for (int at = end; at != start; at = parent[at]) {
        path.push_back(at);
    }
    path.push_back(start);
    reverse(path.begin(), path.end());

    return path;
}

#pragma endregion

#pragma region hamiltonian_extension
bool isHamiltonianCycle(const vector<int>& permutation, Graph& g) {
    for (int i = 0; i < g.v; ++i) {
        int from = permutation[i];
        int to = permutation[(i + 1) % g.v];
        if (g.adjacencyMatrix[from][to] == 0) {
            return false;
        }
    }
    return true;
}

int countMissingEdges(const vector<int>& permutation, Graph& g) {
    int missingEdges = 0;
    for (int i = 0; i < g.v; ++i) {
        int from = permutation[i];
        int to = permutation[(i + 1) % g.v];
        if (g.adjacencyMatrix[from][to] == 0) {
            ++missingEdges;
        }
    }
    return missingEdges;
}

tuple<int, int, set<vector<int>>, vector<vector<int>>> findHamiltonianExtension_exact(Graph& g) {
    set<vector<int>> hamiltonianCycles;
    vector<int> permutation(g.v);
    for (int i = 0; i < g.v; ++i) {
        permutation[i] = i;
    }

    int minEdgesToAdd = INT_MAX;
    vector<int> bestPermutation;

    do {
        int missingEdges = countMissingEdges(permutation, g);
        if (missingEdges < minEdgesToAdd) {
            minEdgesToAdd = missingEdges;
            bestPermutation = permutation;
        }
    } while (std::next_permutation(permutation.begin(), permutation.end()));

    Graph extendedGraph(g.v);
    for (int i = 0; i < g.v; i++) {
        for (int j = 0; j < g.v; j++) {
            extendedGraph.adjacencyMatrix[i][j] = g.adjacencyMatrix[i][j];
        }
    }

    vector<vector<int>> minimumExtension(g.v, vector<int>(g.v, 0));
    for (int i = 0; i < g.v; ++i) {
        int from = bestPermutation[i];
        int to = bestPermutation[(i + 1) % extendedGraph.v];
        extendedGraph.adjacencyMatrix[from][to] = 1;
        if (g.adjacencyMatrix[from][to] == 0) {
            minimumExtension[from][to] = 1;
    }
    }

    int hamiltonianCyclesNum = 0;

    vector<int> currentPath;

    do {
        if (isHamiltonianCycle(permutation, extendedGraph)) {
            ++hamiltonianCyclesNum;
            currentPath = permutation;
            rotate(currentPath.begin(), find(currentPath.begin(), currentPath.end(), 0), currentPath.end());
            hamiltonianCycles.insert(currentPath);

        }
    } while (next_permutation(permutation.begin(), permutation.end()));

    hamiltonianCyclesNum /= g.v;

    return std::make_tuple(minEdgesToAdd, hamiltonianCyclesNum, hamiltonianCycles, minimumExtension);
}
#pragma endregion

#pragma region hamiltonian_extension_approx
tuple<int, vector<int>> nearestNeighbor(int start, Graph& g) {
    vector<int> path;
    path.push_back(start);

    vector<bool> visited(g.v, false);
    visited[start] = true;
    int current = start;
    int missingEdges = 0;

    for (int i = 1; i < g.v; ++i) {
        int next = -1;
        for (int j = 0; j < g.v; ++j) {
            if (!visited[j] && (next == -1 || g.adjacencyMatrix[current][j] > g.adjacencyMatrix[current][next])) {
                next = j;
            }
        }

        if (g.adjacencyMatrix[current][next] == 0) {
            ++missingEdges;
        }

        visited[next] = true;
        path.push_back(next);
        current = next;
    }

    if (g.adjacencyMatrix[current][start] == 0) {
        ++missingEdges;
    }

    return std::make_tuple(missingEdges, path);
}
int estimateUniqueHamiltonianCycles(int startVertex, int iterations, set<vector<int>>& uniqueHamiltonianCycles, Graph& g) {
    std::random_device rd;
    std::mt19937 gen(rd());

    for (int i = 0; i < iterations; ++i) {
        vector<int> path;
        vector<bool> visited(g.v, false);
        path.push_back(startVertex);
        visited[startVertex] = true;

        int current = startVertex;
        for (int j = 1; j < g.v; ++j) {
            vector<int> candidates;
            for (int k = 0; k < g.v; ++k) {
                if (!visited[k] && g.adjacencyMatrix[current][k] == 1) {
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

        if (path.size() == g.v && g.adjacencyMatrix[current][startVertex] == 1) {
            uniqueHamiltonianCycles.insert(path);
        }
    }

    return uniqueHamiltonianCycles.size();
}
tuple<int, int, set<vector<int>>, vector<vector<int>>> findHamiltonianExtension_approx(Graph& g) {
    int minMissingEdges = INT_MAX;
    vector<int> hamiltonianPath;

    for (int start = 0; start < g.v; ++start) {
        auto result = nearestNeighbor(start, g);
        int missingEdges = get<0>(result);
        if (minMissingEdges > missingEdges) {
            minMissingEdges = missingEdges;
            hamiltonianPath = get<1>(result);
        }
    }

    Graph extendedGraph(g.v);
    for (int i = 0; i < g.v; i++) {
        for (int j = 0; j < g.v; j++) {
            extendedGraph.adjacencyMatrix[i][j] = g.adjacencyMatrix[i][j];
        }
    }
    vector<vector<int>> minimumExtension(g.v, vector<int>(g.v, 0));
    for (int i = 0; i < g.v; ++i) {
        int from = hamiltonianPath[i];
        int to = hamiltonianPath[(i + 1) % extendedGraph.v];
        extendedGraph.adjacencyMatrix[from][to] = 1;
        if (g.adjacencyMatrix[from][to] == 0) {
            minimumExtension[from][to] = 1;
        }
    }

    set<vector<int>> uniqueCycles;
    uniqueCycles.insert(hamiltonianPath);

    int cycles = estimateUniqueHamiltonianCycles(hamiltonianPath.front(), g.v, uniqueCycles, extendedGraph);

    return std::make_tuple(minMissingEdges, cycles, uniqueCycles, minimumExtension);
}
#pragma endregion

#pragma region utils

Graph** readGraphsFromFile(const string& filename, int& numGraphs) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Nie mozna otworzyc pliku: " + filename);
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

vector<int> normalizeCycle(const vector<int>& cycle) {
    int minVertex = *min_element(cycle.begin(), cycle.end());

    int minIndex = find(cycle.begin(), cycle.end(), minVertex) - cycle.begin();

    vector<int> normalizedCycle(cycle.begin() + minIndex, cycle.end());
    normalizedCycle.insert(normalizedCycle.end(), cycle.begin(), cycle.begin() + minIndex);

    return normalizedCycle;
}

Graph extendGraph(Graph& G, int newSize) {

    Graph newGraph = Graph(newSize);

    int** newMatrix = new int* [newSize];
    for (int i = 0; i < newSize; i++) {
        newMatrix[i] = new int[newSize]();
        for (int j = 0; j < newSize; j++) {
            if (i < G.v && j < G.v) {
                newMatrix[i][j] = G.adjacencyMatrix[i][j];

            }
            else
                newMatrix[i][j] = 0;
        }
    }
    newGraph.adjacencyMatrix = newMatrix;
    return newGraph;
}

void printCycles(const vector<vector<int>>& cycles) {
    for (const auto& cycle : cycles) {
        for (int node : cycle) {
            cout << node << " ";
        }
        cout << endl;
    }
}


#pragma endregion
