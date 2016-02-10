package org.platanios.learn.graph;

import org.platanios.utilities.TriFunction;

import java.util.HashSet;
import java.util.Set;

/**
 * @param   <V> Vertex content type.
 * @param   <E> Edge content type.
 *
 * @author Emmanouil Antonios Platanios
 */
public class Graph<V, E> {
    private final Set<Vertex<V, E>> vertices = new HashSet<>();
    private final Set<Edge<V, E>> edges = new HashSet<>();

    public Graph() {
    }

    public Graph<V, E> addVertex(Vertex<V, E> vertex) {
        vertices.add(vertex);
        return this;
    }

    public Graph<V, E> addEdge(Edge<V, E> edge) {
        vertices.add(edge.getSourceVertex());
        vertices.add(edge.getDestinationVertex());
        edges.add(edge);
        return this;
    }

    public Graph<V, E> addEdge(Vertex<V, E> sourceVertex, Vertex<V, E> destinationVertex) {
        vertices.add(sourceVertex);
        vertices.add(destinationVertex);
        edges.add(new Edge<>(sourceVertex, destinationVertex));
        return this;
    }

    public Graph<V, E> addEdge(Vertex<V, E> sourceVertex, Vertex<V, E> destinationVertex, E content) {
        vertices.add(sourceVertex);
        vertices.add(destinationVertex);
        edges.add(new Edge<>(sourceVertex, destinationVertex, content));
        return this;
    }

    public Set<Vertex<V, E>> getVertices() {
        return vertices;
    }

    public int getNumberOfVertices() {
        return vertices.size();
    }

    public Set<Edge<V, E>> getEdges() {
        return edges;
    }

    public int getNumberOfEdges() {
        return edges.size();
    }

    public void computeVerticesUpdatedContent(TriFunction<V, Set<Edge<V, E>>, Set<Edge<V, E>>, V> computeFunction) {
        vertices.parallelStream().forEach(vertex -> vertex.computeUpdatedContent(computeFunction));
    }

    public void computeEdgesUpdatedContent(TriFunction<E, Vertex<V, E>, Vertex<V, E>, E> computeFunction) {
        edges.parallelStream().forEach(edge -> edge.computeUpdatedContent(computeFunction));
    }

    public void computeUpdatedContent(TriFunction<V, Set<Edge<V, E>>, Set<Edge<V, E>>, V> vertexComputeFunction,
                                      TriFunction<E, Vertex<V, E>, Vertex<V, E>, E> edgeComputeFunction) {
        computeVerticesUpdatedContent(vertexComputeFunction);
        computeEdgesUpdatedContent(edgeComputeFunction);
    }

    public void updateVerticesContent() {
        vertices.parallelStream().forEach(Vertex::updateContent);
    }

    public void updateEdgesContent() {
        edges.parallelStream().forEach(Edge::updateContent);
    }
}
