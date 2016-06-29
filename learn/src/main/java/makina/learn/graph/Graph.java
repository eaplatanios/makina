package makina.learn.graph;

import java.util.HashSet;
import java.util.Set;
import java.util.function.Function;

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
        vertices.add(edge.sourceVertex());
        vertices.add(edge.destinationVertex());
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

    public Set<Vertex<V, E>> vertices() {
        return vertices;
    }

    public int numberOfVertices() {
        return vertices.size();
    }

    public Set<Edge<V, E>> edges() {
        return edges;
    }

    public int numberOfEdges() {
        return edges.size();
    }

    public void computeVerticesUpdatedContent(Function<Vertex<V, E>, V> updateFunction) {
        vertices.parallelStream().forEach(vertex -> vertex.computeUpdatedContent(updateFunction));
    }

    public void computeEdgesUpdatedContent(Function<Edge<V, E>, E> updateFunction) {
        edges.parallelStream().forEach(edge -> edge.computeUpdatedContent(updateFunction));
    }

    public void updateContent(Function<Vertex<V, E>, V> vertexUpdateFunction,
                              Function<Edge<V, E>, E> edgeUpdateFunction) {
        computeVerticesUpdatedContent(vertexUpdateFunction);
        computeEdgesUpdatedContent(edgeUpdateFunction);
    }

    public void updateVerticesContent() {
        vertices.parallelStream().forEach(Vertex::updateContent);
    }

    public void updateEdgesContent() {
        edges.parallelStream().forEach(Edge::updateContent);
    }
}
