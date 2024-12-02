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


};

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


//utils
Graph** readGraphsFromFile(const string& filename, int& numGraphs);
void deleteGraphs(Graph** graphs, int numGraphs);
vector<int> normalizeCycle(const vector<int>& cycle);
Graph extendGraph(Graph& G, int newSize);
void printCycles(const vector<vector<int>>& cycles);


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

int main(int argc, char* argv[]) {

    string filename = "dane.txt";
    if (argc > 1) {
        filename = argv[1];
    }
    int numGraphs;

    for (int i = 2; i < 20; i = i++)
        testFunctions2(i, findMaxCycle, findLongestCyclesApproximation);

    for (int i = 2; i < 20; i = i++)
        testFunctions(i, calculateDistance, approximateDistance);

    Graph** graphs = readGraphsFromFile(filename, numGraphs);

    cout << "Wczytano " << numGraphs << " grafow:" << endl;
    for (int i = 0; i < numGraphs; i++) {
        graphs[i]->print();
    }
    if (numGraphs > 1)
    {
        cout << "Dinstance: " << calculateDistance(*graphs[0], *graphs[1]) << endl;
        cout << "Approx distance: " << approximateDistance(*graphs[0], *graphs[1]) << endl;
    }
    cout << endl;
    pair<int, pair<int, vector<vector<int>>>> approxMaxCycles = findLongestCyclesApproximation(*graphs[0]);
    pair<int, pair<int, set<vector<int>>>> maxCycles = findMaxCycle(*graphs[0]);

    int cycleLength = maxCycles.first;
    int numberOfMaxCycles = maxCycles.second.first;
    set<vector<int>> cycles = maxCycles.second.second;

    cout << "Max cycle:" << endl;
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
    cycleLength = approxMaxCycles.first;
    numberOfMaxCycles = approxMaxCycles.second.first;
    vector<vector<int>> cycles2 = approxMaxCycles.second.second;

    cout << "Approx max cycle:" << endl;
    cout << "Length of longest cycle " << cycleLength << endl;
    cout << "Number of cycles: " << numberOfMaxCycles << endl;
    cout << "Cycles:" << endl;
    for (const auto& cycle : cycles2) {
        for (int node : cycle) {
            cout << node << " ";
        }
        cout << endl;
    }

    return 0;
}
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
                    transformedMatrix[permutation[i]][j] = extendedG2.adjacencyMatrix[i][j];
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
            if (extendedG1.adjacencyMatrix[mapping[i]][j] != extendedG2.adjacencyMatrix[i][j]) {
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

    return { maxCycleLength + 2 , {allCycles.size(), allCycles} };
}

void backtrackingMaxCycle(Graph& graph, int currentVertex, int startVertex,
    set<int>& visited, int cycleLength,
    int& maxCycleLength, int& maxCycleCount,
    vector<int>& currentCycle, set<vector<int>>& allCycles) {

    visited.insert(currentVertex);
    currentCycle.push_back(currentVertex);

    for (int neighbor = 0; neighbor < graph.v; ++neighbor) {
        if (graph.adjacencyMatrix[currentVertex][neighbor] == 1) {
            if (neighbor == startVertex && cycleLength > 2) {
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
        vector<int> path = findPath(u, v, spanningForest, graph.v);

        path.push_back(u);

        cycles.push_back(path);
        longestLength = max(longestLength, (int)path.size());
    }

    vector<vector<int>> longestCycles;
    for (const auto& cycle : cycles) {
        if (cycle.size() == longestLength) {
            longestCycles.push_back(cycle);
        }
    }

    return { longestLength,{longestCycles.size(),longestCycles} };
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
        forestEdges.insert({ edge.second, edge.first });
    }

    vector<pair<int, int>> extraEdges;

    for (int i = 0; i < graph.v; i++) {
        for (int j = i + 1; j < graph.v; j++) {
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
        treeAdj[edge.second].push_back(edge.first);
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
