package makina.learn.graph;

import org.junit.Assert;
import org.junit.Test;

import java.util.Random;

/**
 * @author Emmanouil Antonios Platanios
 */
public class PageRankAlgorithmTest {
    private final Random random = new Random();

    @Test
    public void testThreeUnlinkedPages() {
        Graph<PageRankAlgorithm.VertexContentType, Void> graph = new Graph<>();
        Vertex<PageRankAlgorithm.VertexContentType, Void> pageA = new Vertex<>(new PageRankAlgorithm.VertexContentType(0, random.nextDouble()));
        Vertex<PageRankAlgorithm.VertexContentType, Void> pageB = new Vertex<>(new PageRankAlgorithm.VertexContentType(1, random.nextDouble()));
        Vertex<PageRankAlgorithm.VertexContentType, Void> pageC = new Vertex<>(new PageRankAlgorithm.VertexContentType(2, random.nextDouble()));
        graph.addVertex(pageA);
        graph.addVertex(pageB);
        graph.addVertex(pageC);
        PageRankAlgorithm pageRankAlgorithm =
                new PageRankAlgorithm.Builder<>(graph)
                        .dampingFactor(0.85)
                        .maximumNumberOfIterations(1000)
                        .loggingLevel(4)
                        .build();
        pageRankAlgorithm.computeRanks();
        Assert.assertEquals(0.05, pageA.content().getRank(), 1e-2);
        Assert.assertEquals(0.05, pageB.content().getRank(), 1e-2);
    }

    @Test
    public void testThreeFullyLinkedPages() {
        Graph<PageRankAlgorithm.VertexContentType, Void> graph = new Graph<>();
        Vertex<PageRankAlgorithm.VertexContentType, Void> pageA = new Vertex<>(new PageRankAlgorithm.VertexContentType(0, random.nextDouble()));
        Vertex<PageRankAlgorithm.VertexContentType, Void> pageB = new Vertex<>(new PageRankAlgorithm.VertexContentType(1, random.nextDouble()));
        Vertex<PageRankAlgorithm.VertexContentType, Void> pageC = new Vertex<>(new PageRankAlgorithm.VertexContentType(2, random.nextDouble()));
        graph.addVertex(pageA);
        graph.addVertex(pageB);
        graph.addVertex(pageC);
        graph.addEdge(pageA, pageB);
        graph.addEdge(pageB, pageA);
        graph.addEdge(pageA, pageC);
        graph.addEdge(pageC, pageA);
        graph.addEdge(pageB, pageC);
        graph.addEdge(pageC, pageB);
        PageRankAlgorithm pageRankAlgorithm =
                new PageRankAlgorithm.Builder<>(graph)
                        .dampingFactor(0.85)
                        .maximumNumberOfIterations(1000)
                        .loggingLevel(4)
                        .build();
        pageRankAlgorithm.computeRanks();
        Assert.assertEquals(0.33, pageA.content().getRank(), 1e-2);
        Assert.assertEquals(0.33, pageB.content().getRank(), 1e-2);
        Assert.assertEquals(0.33, pageC.content().getRank(), 1e-2);
    }

    @Test
    public void testThreePartiallyLinkedPages() {
        Graph<PageRankAlgorithm.VertexContentType, Void> graph = new Graph<>();
        Vertex<PageRankAlgorithm.VertexContentType, Void> pageA = new Vertex<>(new PageRankAlgorithm.VertexContentType(0, random.nextDouble()));
        Vertex<PageRankAlgorithm.VertexContentType, Void> pageB = new Vertex<>(new PageRankAlgorithm.VertexContentType(1, random.nextDouble()));
        Vertex<PageRankAlgorithm.VertexContentType, Void> pageC = new Vertex<>(new PageRankAlgorithm.VertexContentType(2, random.nextDouble()));
        graph.addVertex(pageA);
        graph.addVertex(pageB);
        graph.addVertex(pageC);
        graph.addEdge(pageA, pageB);
        graph.addEdge(pageB, pageA);
        graph.addEdge(pageA, pageC);
        graph.addEdge(pageC, pageA);
        PageRankAlgorithm pageRankAlgorithm =
                new PageRankAlgorithm.Builder<>(graph)
                        .dampingFactor(0.85)
                        .maximumNumberOfIterations(1000)
                        .loggingLevel(4)
                        .build();
        pageRankAlgorithm.computeRanks();
        Assert.assertEquals(0.487, pageA.content().getRank(), 1e-3);
        Assert.assertEquals(0.257, pageB.content().getRank(), 1e-3);
        Assert.assertEquals(0.257, pageC.content().getRank(), 1e-3);
    }
}
