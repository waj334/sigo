package builder

import (
	"fmt"
	"golang.org/x/tools/go/ssa"
)

type Graph struct {
	nodes   map[*ssa.Package]bool
	adjList map[*ssa.Package][]*ssa.Package
}

func NewGraph() *Graph {
	return &Graph{
		nodes:   make(map[*ssa.Package]bool),
		adjList: make(map[*ssa.Package][]*ssa.Package),
	}
}

func (g *Graph) AddEdge(src, dest *ssa.Package) {
	g.nodes[src] = true
	g.nodes[dest] = true
	g.adjList[src] = append(g.adjList[src], dest)
}

func (g *Graph) AddGlobalDependency(dependency *ssa.Package) {
	for node := range g.nodes {
		// Do not add a self-loop
		if node == dependency {
			continue
		}
		// Check if node already has a direct edge to dependency
		hasEdge := false
		for _, neighbor := range g.adjList[node] {
			if neighbor == dependency {
				hasEdge = true
				break
			}
		}
		// If not, add the edge
		if !hasEdge {
			g.AddEdge(node, dependency)
		}
	}
}

func (g *Graph) dfs(node *ssa.Package, visited, recStack map[*ssa.Package]bool, sortedNodes *[]*ssa.Package, cycleNodes *[]*ssa.Package) bool {
	visited[node] = true
	recStack[node] = true

	for _, neighbor := range g.adjList[node] {
		if recStack[neighbor] {
			*cycleNodes = append(*cycleNodes, neighbor)
			return true
		}
		if !visited[neighbor] && g.dfs(neighbor, visited, recStack, sortedNodes, cycleNodes) {
			*cycleNodes = append(*cycleNodes, neighbor)
			return true
		}
	}

	recStack[node] = false
	*sortedNodes = append(*sortedNodes, node)
	return false
}

func (g *Graph) Buckets() ([][]*ssa.Package, error) {
	visited := make(map[*ssa.Package]bool)
	recStack := make(map[*ssa.Package]bool)
	sortedNodes := make([]*ssa.Package, 0)

	for node := range g.nodes {
		if !visited[node] {
			cycleNodes := make([]*ssa.Package, 0)
			if g.dfs(node, visited, recStack, &sortedNodes, &cycleNodes) {
				return nil, fmt.Errorf("graph contains a cycle involving nodes: %v", cycleNodes)
			}
		}
	}

	buckets := make([][]*ssa.Package, 0)
	for i := len(sortedNodes) - 1; i >= 0; i-- {
		bucket := []*ssa.Package{sortedNodes[i]}
		buckets = append([][]*ssa.Package{bucket}, buckets...)
	}

	return buckets, nil
}
