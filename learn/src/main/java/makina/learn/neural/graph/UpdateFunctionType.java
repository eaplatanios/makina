package makina.learn.neural.graph;

import makina.learn.graph.Vertex;

import java.util.Set;

/**
 * @author Emmanouil Antonios Platanios
 */
public enum UpdateFunctionType {
    SIGMOID {
        @Override
        public <E> DeepGraph.UpdateFunction<DeepGraph.VertexContent, E> getFunction(
                int featureVectorsSize, Set<Vertex<DeepGraph.VertexContent, E>> vertices
        ) {
            return new SigmoidUpdateFunction<>(featureVectorsSize, vertices);
        }
    },
    UNDIRECTED_SIGMOID {
        @Override
        public <E> DeepGraph.UpdateFunction<DeepGraph.VertexContent, E> getFunction(
                int featureVectorsSize, Set<Vertex<DeepGraph.VertexContent, E>> vertices
        ) {
            return new UndirectedSigmoidUpdateFunction<>(featureVectorsSize, vertices);
        }
    },
    GRU {
        @Override
        public <E> DeepGraph.UpdateFunction<DeepGraph.VertexContent, E> getFunction(
                int featureVectorsSize, Set<Vertex<DeepGraph.VertexContent, E>> vertices
        ) {
            return new GRUUpdateFunction<>(featureVectorsSize, vertices);
        }
    },
    UNDIRECTED_GRU {
        @Override
        public <E> DeepGraph.UpdateFunction<DeepGraph.VertexContent, E> getFunction(
                int featureVectorsSize, Set<Vertex<DeepGraph.VertexContent, E>> vertices
        ) {
            return new UndirectedGRUUpdateFunction<>(featureVectorsSize, vertices);
        }
    };

    public abstract <E> DeepGraph.UpdateFunction<DeepGraph.VertexContent, E> getFunction(
            int featureVectorsSize, Set<Vertex<DeepGraph.VertexContent, E>> vertices
    );
}
