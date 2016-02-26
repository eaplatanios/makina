package org.platanios.learn.neural.graph;

import org.platanios.learn.graph.Vertex;

import java.util.Set;

/**
 * @author Emmanouil Antonios Platanios
 */
public enum FeatureVectorFunctionType {
    SIGMOID {
        @Override
        public <E> GraphRecursiveNeuralNetwork.FeatureVectorFunction<GraphRecursiveNeuralNetwork.VertexContentType, E> getFunction(int featureVectorsSize, Set<Vertex<GraphRecursiveNeuralNetwork.VertexContentType, E>> vertices) {
            return new SigmoidFeatureVectorFunction<>(featureVectorsSize, vertices);
        }
    },
    UNDIRECTED_SIGMOID {
        @Override
        public <E> GraphRecursiveNeuralNetwork.FeatureVectorFunction<GraphRecursiveNeuralNetwork.VertexContentType, E> getFunction(int featureVectorsSize, Set<Vertex<GraphRecursiveNeuralNetwork.VertexContentType, E>> vertices) {
            return new UndirectedSigmoidFeatureVectorFunction<>(featureVectorsSize, vertices);
        }
    },
    GRU {
        @Override
        public <E> GraphRecursiveNeuralNetwork.FeatureVectorFunction<GraphRecursiveNeuralNetwork.VertexContentType, E> getFunction(int featureVectorsSize, Set<Vertex<GraphRecursiveNeuralNetwork.VertexContentType, E>> vertices) {
            return new GRUFeatureVectorFunction<>(featureVectorsSize, vertices);
        }
    },
    UNDIRECTED_GRU {
        @Override
        public <E> GraphRecursiveNeuralNetwork.FeatureVectorFunction<GraphRecursiveNeuralNetwork.VertexContentType, E> getFunction(int featureVectorsSize, Set<Vertex<GraphRecursiveNeuralNetwork.VertexContentType, E>> vertices) {
            return new UndirectedGRUFeatureVectorFunction<>(featureVectorsSize, vertices);
        }
    };

    public abstract <E> GraphRecursiveNeuralNetwork.FeatureVectorFunction<GraphRecursiveNeuralNetwork.VertexContentType, E> getFunction(
            int featureVectorsSize,
            Set<Vertex<GraphRecursiveNeuralNetwork.VertexContentType, E>> vertices
    );
}
