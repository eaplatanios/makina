package makina.learn.graph;

import java.util.function.Function;

/**
 * @param   <V> Vertex content type.
 * @param   <E> Edge content type.
 *
 * @author Emmanouil Antonios Platanios
 */
public class Edge<V, E> {
    private final Vertex<V, E> sourceVertex;
    private final Vertex<V, E> destinationVertex;

    private E content;
    private E updatedContent;

    public Edge(Vertex<V, E> sourceVertex, Vertex<V, E> destinationVertex) {
        this(sourceVertex, destinationVertex, null);
    }

    public Edge(Vertex<V, E> sourceVertex, Vertex<V, E> destinationVertex, E content) {
        this.sourceVertex = sourceVertex;
        this.destinationVertex = destinationVertex;
        this.content = content;
        sourceVertex.addOutgoingEdge(this);
        destinationVertex.addIncomingEdge(this);
    }

    public Vertex<V, E> sourceVertex() {
        return sourceVertex;
    }

    public Vertex<V, E> destinationVertex() {
        return destinationVertex;
    }

    public E content() {
        return content;
    }

    public E updatedContent() {
        return updatedContent;
    }

    public void computeUpdatedContent(Function<Edge<V, E>, E> computeFunction) {
        updatedContent = computeFunction.apply(this);
    }

    public void updateContent() {
        content = updatedContent;
        updatedContent = null;
    }
}
