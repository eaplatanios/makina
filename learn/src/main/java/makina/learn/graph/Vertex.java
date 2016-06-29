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
public class Vertex<V, E> {
    private final Set<Edge<V, E>> incomingEdges = new HashSet<>();
    private final Set<Edge<V, E>> outgoingEdges = new HashSet<>();

    private int numberOfIncomingEdges;
    private int numberOfOutgoingEdges;
    private V content;
    private V updatedContent;

    public Vertex() {
        this(null);
    }

    public Vertex(V content) {
        this.content = content;
        numberOfIncomingEdges = 0;
        numberOfOutgoingEdges = 0;
    }

    protected Vertex<V, E> addIncomingEdge(Edge<V, E> edge) {
        if (edge.destinationVertex() != this)
            throw new IllegalArgumentException("The destination of the provided edge is not the current vertex.");
        incomingEdges.add(edge);
        numberOfIncomingEdges++;
        return this;
    }

    protected Vertex<V, E> addOutgoingEdge(Edge<V, E> edge) {
        if (edge.sourceVertex() != this)
            throw new IllegalArgumentException("The source of the provided edge is not the current vertex.");
        outgoingEdges.add(edge);
        numberOfOutgoingEdges++;
        return this;
    }

    public Set<Edge<V, E>> incomingEdges() {
        return incomingEdges;
    }

    public Set<Edge<V, E>> outgoingEdges() {
        return outgoingEdges;
    }

    public int numberOfIncomingEdges() {
        return numberOfIncomingEdges;
    }

    public int numberOfOutgoingEdges() {
        return numberOfOutgoingEdges;
    }

    public V content() {
        return content;
    }

    public V updatedContent() {
        return updatedContent;
    }

    public void computeUpdatedContent(Function<Vertex<V, E>, V> updateFunction) {
        updatedContent = updateFunction.apply(this);
    }

    public void updateContent() {
        content = updatedContent;
        updatedContent = null;
    }
}
