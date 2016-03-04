package org.platanios.learn.neural.graph;

import org.platanios.learn.graph.Edge;
import org.platanios.learn.graph.Vertex;
import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;
import org.platanios.learn.neural.network.Network;
import org.platanios.learn.neural.network.NetworkBuilder;

import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @author Emmanouil Antonios Platanios
 */
public class SigmoidFeatureVectorFunction<E>
        extends GraphRecursiveNeuralNetwork.FeatureVectorFunction<GraphRecursiveNeuralNetwork.VertexContentType, E> {
    private final Map<Integer, Network> networks = new ConcurrentHashMap<>();
    private final Random random = new Random();

    private final int featureVectorsSize;
    private final int parametersVectorSize;

    public SigmoidFeatureVectorFunction(int featureVectorsSize,
                                        Set<Vertex<GraphRecursiveNeuralNetwork.VertexContentType, E>> vertices) {
        this.featureVectorsSize = featureVectorsSize;
        parametersVectorSize = 3 * (featureVectorsSize * featureVectorsSize) + featureVectorsSize;
        for (Vertex<GraphRecursiveNeuralNetwork.VertexContentType, E> vertex : vertices) {
            NetworkBuilder networkBuilder = new NetworkBuilder();
            int phiLayerId = networkBuilder.addInputLayer(featureVectorsSize, "phi");
            int phiInLayerId = networkBuilder.addInputLayer(featureVectorsSize, "phi_in");
            int phiOutLayerId = networkBuilder.addInputLayer(featureVectorsSize, "phi_out");
            int phiHiddenLayerId = networkBuilder.addFullyConnectedLayer(phiLayerId, featureVectorsSize, "W_phi", "b");
            int phiInHiddenLayerId = networkBuilder.addFullyConnectedLayer(phiInLayerId, featureVectorsSize, "W_phi_in");
            int phiOutHiddenLayerId = networkBuilder.addFullyConnectedLayer(phiOutLayerId, featureVectorsSize, "W_phi_out");
            int additionHiddenLayerId = networkBuilder.addAdditionLayer(phiHiddenLayerId, phiInHiddenLayerId, phiOutHiddenLayerId);
            int sigmoidLayerId = networkBuilder.addSigmoidLayer(true, additionHiddenLayerId);
            networks.put(vertex.getContent().getId(), networkBuilder.build());
        }
    }

    @Override
    public int getParametersVectorSize() {
        return parametersVectorSize;
    }

    @Override
    public Vector getInitialParametersVector() {
        Vector initialParametersVector = Vectors.dense(parametersVectorSize);
        int parameterVectorIndex = 0;
        for (int layerIndex = 0; layerIndex < 3; layerIndex++)
            for (int i = 0; i < featureVectorsSize; i++)
                for (int j = 0; j < featureVectorsSize; j++)
                    initialParametersVector.set(parameterVectorIndex++, (random.nextDouble() - 0.5) * 2 * 1.0 / Math.sqrt(featureVectorsSize));
        return initialParametersVector;
    }

    @Override
    public Vector value(Vector parameters, Vertex<GraphRecursiveNeuralNetwork.VertexContentType, E> vertex, int step) {
//        Vector incomingFeatureVectorsSum = Vectors.dense(featureVectorsSize);
//        for (Edge<GraphRecursiveNeuralNetwork.VertexContentType, E> incomingEdge : vertex.getIncomingEdges()) {
//            GraphRecursiveNeuralNetwork.VertexContentType vertexContent = incomingEdge.getSourceVertex().getContent();
//            incomingFeatureVectorsSum.addInPlace(vertexContent.featureVectors[step]);
//        }
//        Vector outgoingFeatureVectorsSum = Vectors.dense(featureVectorsSize);
//        for (Edge<GraphRecursiveNeuralNetwork.VertexContentType, E> outgoingEdge : vertex.getOutgoingEdges()) {
//            GraphRecursiveNeuralNetwork.VertexContentType vertexContent = outgoingEdge.getDestinationVertex().getContent();
//            outgoingFeatureVectorsSum.addInPlace(vertexContent.featureVectors[step]);
//        }
        Network network = networks.get(vertex.getContent().getId());
        network.set("phi", vertex.getContent().featureVectors[step]);
        network.set("phi_in", vertex.getContent().getIncomingFeatureVectorsSum(step));
        network.set("phi_out", vertex.getContent().getOutgoingFeatureVectorsSum(step));
        network.set("W_phi", parameters.get(0, featureVectorsSize * featureVectorsSize - 1));
        network.set("W_phi_in", parameters.get(featureVectorsSize * featureVectorsSize, 2 * featureVectorsSize * featureVectorsSize - 1));
        network.set("W_phi_out", parameters.get(2 * featureVectorsSize * featureVectorsSize, 3 * featureVectorsSize * featureVectorsSize - 1));
        network.set("b", parameters.get(3 * featureVectorsSize * featureVectorsSize, 3 * featureVectorsSize * featureVectorsSize + featureVectorsSize - 1));
        return network.value();
    }

    @Override
    public Matrix gradient(Vector parameters, Vertex<GraphRecursiveNeuralNetwork.VertexContentType, E> vertex, int step) {
//        Vector incomingFeatureVectorsSum = Vectors.dense(featureVectorsSize);
//        for (Edge<GraphRecursiveNeuralNetwork.VertexContentType, E> incomingEdge : vertex.getIncomingEdges()) {
//            GraphRecursiveNeuralNetwork.VertexContentType vertexContent = incomingEdge.getSourceVertex().getContent();
//            incomingFeatureVectorsSum.addInPlace(vertexContent.featureVectors[step]);
//        }
//        Vector outgoingFeatureVectorsSum = Vectors.dense(featureVectorsSize);
//        for (Edge<GraphRecursiveNeuralNetwork.VertexContentType, E> outgoingEdge : vertex.getOutgoingEdges()) {
//            GraphRecursiveNeuralNetwork.VertexContentType vertexContent = outgoingEdge.getDestinationVertex().getContent();
//            outgoingFeatureVectorsSum.addInPlace(vertexContent.featureVectors[step]);
//        }
        Network network = networks.get(vertex.getContent().getId());
        network.set("phi", vertex.getContent().featureVectors[step]);
        network.set("phi_in", vertex.getContent().getIncomingFeatureVectorsSum(step));
        network.set("phi_out", vertex.getContent().getOutgoingFeatureVectorsSum(step));
        network.set("W_phi", parameters.get(0, featureVectorsSize * featureVectorsSize - 1));
        network.set("W_phi_in", parameters.get(featureVectorsSize * featureVectorsSize, 2 * featureVectorsSize * featureVectorsSize - 1));
        network.set("W_phi_out", parameters.get(2 * featureVectorsSize * featureVectorsSize, 3 * featureVectorsSize * featureVectorsSize - 1));
        network.set("b", parameters.get(3 * featureVectorsSize * featureVectorsSize, 3 * featureVectorsSize * featureVectorsSize + featureVectorsSize - 1));
        Matrix[] networkGradients = network.gradient("W_phi", "W_phi_in", "W_phi_out", "b");
        Matrix gradient = Matrix.zeros(featureVectorsSize, parametersVectorSize);
        gradient.setSubMatrix(0, featureVectorsSize - 1,
                              0, featureVectorsSize * featureVectorsSize - 1,
                              networkGradients[0]);
        gradient.setSubMatrix(0, featureVectorsSize - 1,
                              featureVectorsSize * featureVectorsSize, 2 * featureVectorsSize * featureVectorsSize - 1,
                              networkGradients[1]);
        gradient.setSubMatrix(0, featureVectorsSize - 1,
                              2 * featureVectorsSize * featureVectorsSize, 3 * featureVectorsSize * featureVectorsSize - 1,
                              networkGradients[2]);
        gradient.setSubMatrix(0, featureVectorsSize - 1,
                              3 * featureVectorsSize * featureVectorsSize, 3 * featureVectorsSize * featureVectorsSize + featureVectorsSize - 1,
                              networkGradients[3]);
        return gradient;
    }

    @Override
    public Matrix featureVectorGradient(Vector parameters,
                                        Vertex<GraphRecursiveNeuralNetwork.VertexContentType, E> vertex,
                                        Vertex<GraphRecursiveNeuralNetwork.VertexContentType, E> differentiatingVertex,
                                        int step) {
//        Vector incomingFeatureVectorsSum = Vectors.dense(featureVectorsSize);
//        for (Edge<GraphRecursiveNeuralNetwork.VertexContentType, E> incomingEdge : vertex.getIncomingEdges()) {
//            GraphRecursiveNeuralNetwork.VertexContentType vertexContent = incomingEdge.getSourceVertex().getContent();
//            incomingFeatureVectorsSum.addInPlace(vertexContent.featureVectors[step]);
//        }
//        Vector outgoingFeatureVectorsSum = Vectors.dense(featureVectorsSize);
//        for (Edge<GraphRecursiveNeuralNetwork.VertexContentType, E> outgoingEdge : vertex.getOutgoingEdges()) {
//            GraphRecursiveNeuralNetwork.VertexContentType vertexContent = outgoingEdge.getDestinationVertex().getContent();
//            outgoingFeatureVectorsSum.addInPlace(vertexContent.featureVectors[step]);
//        }
        Network network = networks.get(differentiatingVertex.getContent().getId());
        network.set("phi", vertex.getContent().featureVectors[step]);
        network.set("phi_in", vertex.getContent().getIncomingFeatureVectorsSum(step));
        network.set("phi_out", vertex.getContent().getOutgoingFeatureVectorsSum(step));
        network.set("W_phi", parameters.get(0, featureVectorsSize * featureVectorsSize - 1));
        network.set("W_phi_in", parameters.get(featureVectorsSize * featureVectorsSize, 2 * featureVectorsSize * featureVectorsSize - 1));
        network.set("W_phi_out", parameters.get(2 * featureVectorsSize * featureVectorsSize, 3 * featureVectorsSize * featureVectorsSize - 1));
        network.set("b", parameters.get(3 * featureVectorsSize * featureVectorsSize, 3 * featureVectorsSize * featureVectorsSize + featureVectorsSize - 1));
        if (differentiatingVertex.getContent().id == vertex.getContent().id) {
            return network.gradient("phi");
        } else {
            boolean incomingVertex = false;
            for (Edge<GraphRecursiveNeuralNetwork.VertexContentType, E> edge : vertex.getIncomingEdges())
                if (edge.getSourceVertex().getContent().id == differentiatingVertex.getContent().id) {
                    incomingVertex = true;
                    break;
                }
            if (incomingVertex) {
                return network.gradient("phi_in");
            } else {
                boolean outgoingVertex = false;
                for (Edge<GraphRecursiveNeuralNetwork.VertexContentType, E> edge : vertex.getOutgoingEdges())
                    if (edge.getDestinationVertex().getContent().id == differentiatingVertex.getContent().id) {
                        outgoingVertex = true;
                        break;
                    }
                if (outgoingVertex) {
                    return network.gradient("phi_out");
                } else {
                    return Matrix.zeros(featureVectorsSize, featureVectorsSize);
                }
            }
        }
    }
}
