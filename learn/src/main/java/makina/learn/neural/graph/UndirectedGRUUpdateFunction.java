package makina.learn.neural.graph;

import makina.learn.graph.Edge;
import makina.learn.graph.Vertex;
import makina.learn.neural.network.Network;
import makina.learn.neural.network.NetworkBuilder;
import makina.math.matrix.Matrix;
import makina.math.matrix.Vector;
import makina.math.matrix.Vectors;

import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @author Emmanouil Antonios Platanios
 */
public class UndirectedGRUUpdateFunction<E>
        extends DeepGraph.UpdateFunction<DeepGraph.VertexContent, E> {
    private final Map<Integer, Network> networks = new ConcurrentHashMap<>();
    private final Random random = new Random();

    private final int featureVectorsSize;
    private final int parametersVectorSize;

    public UndirectedGRUUpdateFunction(int featureVectorsSize,
                                       Set<Vertex<DeepGraph.VertexContent, E>> vertices) {
        this.featureVectorsSize = featureVectorsSize;
        parametersVectorSize = 6 * (featureVectorsSize * featureVectorsSize) + 3 * featureVectorsSize;
        for (Vertex<DeepGraph.VertexContent, E> vertex : vertices) {
            NetworkBuilder networkBuilder = new NetworkBuilder();
            int phiLayerId = networkBuilder.addInputLayer(featureVectorsSize, "phi");
            int phiInLayerId = networkBuilder.addInputLayer(featureVectorsSize, "phi_neighbors");
            int rhoPhiHiddenLayerId = networkBuilder.addFullyConnectedLayer(phiLayerId, featureVectorsSize, "rho_W_phi", "rho_b");
            int rhoPhiNeighborsHiddenLayerId = networkBuilder.addFullyConnectedLayer(phiInLayerId, featureVectorsSize, "rho_W_phi_neighbors");
            int rhoAdditionHiddenLayerId = networkBuilder.addAdditionLayer(rhoPhiHiddenLayerId, rhoPhiNeighborsHiddenLayerId);
            int rhoSigmoidHiddenLayerId = networkBuilder.addSigmoidLayer(rhoAdditionHiddenLayerId);
            int zPhiHiddenLayerId = networkBuilder.addFullyConnectedLayer(phiLayerId, featureVectorsSize, "z_W_phi", "z_b");
            int zPhiNeighborsHiddenLayerId = networkBuilder.addFullyConnectedLayer(phiInLayerId, featureVectorsSize, "z_W_phi_neighbors");
            int zAdditionHiddenLayerId = networkBuilder.addAdditionLayer(zPhiHiddenLayerId, zPhiNeighborsHiddenLayerId);
            int zSigmoidHiddenLayerId = networkBuilder.addSigmoidLayer(zAdditionHiddenLayerId);
            int hTildePhiMultiplicationHiddenLayerId = networkBuilder.addElementwiseMultiplicationLayer(rhoSigmoidHiddenLayerId, phiLayerId);
            int hTildeRhoPhiHiddenLayerId = networkBuilder.addFullyConnectedLayer(hTildePhiMultiplicationHiddenLayerId, featureVectorsSize, "h_tilde_W_phi", "h_tilde_b");
            int hTildePhiNeighborsHiddenLayerId = networkBuilder.addFullyConnectedLayer(phiInLayerId, featureVectorsSize, "h_tilde_W_phi_neighbors");
            int hTildeAdditionHiddenLayerId = networkBuilder.addAdditionLayer(hTildeRhoPhiHiddenLayerId, hTildePhiNeighborsHiddenLayerId);
            int hTildeTanhHiddenLayerId = networkBuilder.addTanhLayer(hTildeAdditionHiddenLayerId);
            int hZPhiMultiplicationHiddenLayerId = networkBuilder.addElementwiseMultiplicationLayer(zSigmoidHiddenLayerId, phiLayerId);
            int constantOneLayerId = networkBuilder.addConstantLayer(Vectors.ones(featureVectorsSize));
            int zSubtractionHiddenLayerId = networkBuilder.addSubtractionLayer(constantOneLayerId, zSigmoidHiddenLayerId);
            int hZHTildeMultiplicationHiddenLayerId = networkBuilder.addElementwiseMultiplicationLayer(zSubtractionHiddenLayerId, hTildeTanhHiddenLayerId);
            int hAdditionHiddenLayerId = networkBuilder.addAdditionLayer(true, hZPhiMultiplicationHiddenLayerId, hZHTildeMultiplicationHiddenLayerId);
            networks.put(vertex.content().id(), networkBuilder.build());
        }
    }

    @Override
    public int parametersSize() {
        return parametersVectorSize;
    }

    @Override
    public Vector initialParameters() {
        Vector initialParametersVector = Vectors.dense(parametersVectorSize);
        int parameterVectorIndex = 0;
        for (int layerIndex = 0; layerIndex < 6; layerIndex++)
            for (int i = 0; i < featureVectorsSize; i++)
                for (int j = 0; j < featureVectorsSize; j++)
                    initialParametersVector.set(parameterVectorIndex++, (random.nextDouble() - 0.5) * 2 * 1.0 / Math.sqrt(featureVectorsSize));
        return initialParametersVector;
    }

    @Override
    public Vector value(Vector parameters, Vertex<DeepGraph.VertexContent, E> vertex, int step) {
        Vector neighborFeatureVectorsSum = Vectors.dense(featureVectorsSize);
        neighborFeatureVectorsSum.addInPlace(vertex.content().incomingFeaturesSum(step));
        neighborFeatureVectorsSum.addInPlace(vertex.content().outgoingFeaturesSum(step));
//        for (Edge<GraphRecursiveNeuralNetwork.VertexContentType, E> incomingEdge : vertex.incomingEdges()) {
//            GraphRecursiveNeuralNetwork.VertexContentType vertexContent = incomingEdge.sourceVertex().content();
//            neighborFeatureVectorsSum.addInPlace(vertexContent.features[step]);
//        }
//        for (Edge<GraphRecursiveNeuralNetwork.VertexContentType, E> outgoingEdge : vertex.outgoingEdges()) {
//            GraphRecursiveNeuralNetwork.VertexContentType vertexContent = outgoingEdge.destinationVertex().content();
//            neighborFeatureVectorsSum.addInPlace(vertexContent.features[step]);
//        }
        Network network = networks.get(vertex.content().id());
        network.set("phi", vertex.content().features[step]);
        network.set("phi_neighbors", neighborFeatureVectorsSum);
        network.set("rho_W_phi", parameters.get(0, featureVectorsSize * featureVectorsSize - 1));
        network.set("rho_W_phi_neighbors", parameters.get(featureVectorsSize * featureVectorsSize, 2 * featureVectorsSize * featureVectorsSize - 1));
        network.set("z_W_phi", parameters.get(2 * featureVectorsSize * featureVectorsSize, 3 * featureVectorsSize * featureVectorsSize - 1));
        network.set("z_W_phi_neighbors", parameters.get(3 * featureVectorsSize * featureVectorsSize, 4 * featureVectorsSize * featureVectorsSize - 1));
        network.set("h_tilde_W_phi", parameters.get(4 * featureVectorsSize * featureVectorsSize, 5 * featureVectorsSize * featureVectorsSize - 1));
        network.set("h_tilde_W_phi_neighbors", parameters.get(5 * featureVectorsSize * featureVectorsSize, 6 * featureVectorsSize * featureVectorsSize - 1));
        network.set("rho_b", parameters.get(6 * featureVectorsSize * featureVectorsSize, 6 * featureVectorsSize * featureVectorsSize + featureVectorsSize - 1));
        network.set("z_b", parameters.get(6 * featureVectorsSize * featureVectorsSize + featureVectorsSize, 6 * featureVectorsSize * featureVectorsSize + 2 * featureVectorsSize - 1));
        network.set("h_tilde_b", parameters.get(6 * featureVectorsSize * featureVectorsSize + 2 * featureVectorsSize, 6 * featureVectorsSize * featureVectorsSize + 3 * featureVectorsSize - 1));
        return network.value();
    }

    @Override
    public Matrix gradient(Vector parameters, Vertex<DeepGraph.VertexContent, E> vertex, int step) {
        Vector neighborFeatureVectorsSum = Vectors.dense(featureVectorsSize);
        neighborFeatureVectorsSum.addInPlace(vertex.content().incomingFeaturesSum(step));
        neighborFeatureVectorsSum.addInPlace(vertex.content().outgoingFeaturesSum(step));
//        for (Edge<GraphRecursiveNeuralNetwork.VertexContentType, E> incomingEdge : vertex.incomingEdges()) {
//            GraphRecursiveNeuralNetwork.VertexContentType vertexContent = incomingEdge.sourceVertex().content();
//            neighborFeatureVectorsSum.addInPlace(vertexContent.features[step]);
//        }
//        for (Edge<GraphRecursiveNeuralNetwork.VertexContentType, E> outgoingEdge : vertex.outgoingEdges()) {
//            GraphRecursiveNeuralNetwork.VertexContentType vertexContent = outgoingEdge.destinationVertex().content();
//            neighborFeatureVectorsSum.addInPlace(vertexContent.features[step]);
//        }
        Network network = networks.get(vertex.content().id());
        network.set("phi", vertex.content().features[step]);
        network.set("phi_neighbors", neighborFeatureVectorsSum);
        network.set("rho_W_phi", parameters.get(0, featureVectorsSize * featureVectorsSize - 1));
        network.set("rho_W_phi_neighbors", parameters.get(featureVectorsSize * featureVectorsSize, 2 * featureVectorsSize * featureVectorsSize - 1));
        network.set("z_W_phi", parameters.get(2 * featureVectorsSize * featureVectorsSize, 3 * featureVectorsSize * featureVectorsSize - 1));
        network.set("z_W_phi_neighbors", parameters.get(3 * featureVectorsSize * featureVectorsSize, 4 * featureVectorsSize * featureVectorsSize - 1));
        network.set("h_tilde_W_phi", parameters.get(4 * featureVectorsSize * featureVectorsSize, 5 * featureVectorsSize * featureVectorsSize - 1));
        network.set("h_tilde_W_phi_neighbors", parameters.get(5 * featureVectorsSize * featureVectorsSize, 6 * featureVectorsSize * featureVectorsSize - 1));
        network.set("rho_b", parameters.get(6 * featureVectorsSize * featureVectorsSize, 6 * featureVectorsSize * featureVectorsSize + featureVectorsSize - 1));
        network.set("z_b", parameters.get(6 * featureVectorsSize * featureVectorsSize + featureVectorsSize, 6 * featureVectorsSize * featureVectorsSize + 2 * featureVectorsSize - 1));
        network.set("h_tilde_b", parameters.get(6 * featureVectorsSize * featureVectorsSize + 2 * featureVectorsSize, 6 * featureVectorsSize * featureVectorsSize + 3 * featureVectorsSize - 1));
        Matrix[] networkGradients = network.gradient("rho_W_phi", "rho_W_phi_neighbors",
                                                     "z_W_phi", "z_W_phi_neighbors",
                                                     "h_tilde_W_phi", "h_tilde_W_phi_neighbors",
                                                     "rho_b", "z_b", "h_tilde_b");
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
                              3 * featureVectorsSize * featureVectorsSize, 4 * featureVectorsSize * featureVectorsSize - 1,
                              networkGradients[3]);
        gradient.setSubMatrix(0, featureVectorsSize - 1,
                              4 * featureVectorsSize * featureVectorsSize, 5 * featureVectorsSize * featureVectorsSize - 1,
                              networkGradients[4]);
        gradient.setSubMatrix(0, featureVectorsSize - 1,
                              5 * featureVectorsSize * featureVectorsSize, 6 * featureVectorsSize * featureVectorsSize - 1,
                              networkGradients[5]);
        gradient.setSubMatrix(0, featureVectorsSize - 1,
                              6 * featureVectorsSize * featureVectorsSize, 6 * featureVectorsSize * featureVectorsSize + featureVectorsSize - 1,
                              networkGradients[6]);
        gradient.setSubMatrix(0, featureVectorsSize - 1,
                              6 * featureVectorsSize * featureVectorsSize + featureVectorsSize, 6 * featureVectorsSize * featureVectorsSize + 2 * featureVectorsSize - 1,
                              networkGradients[7]);
        gradient.setSubMatrix(0, featureVectorsSize - 1,
                              6 * featureVectorsSize * featureVectorsSize + 2 * featureVectorsSize, 6 * featureVectorsSize * featureVectorsSize + 3 * featureVectorsSize - 1,
                              networkGradients[8]);
        return gradient;
    }

    @Override
    public Matrix featuresGradient(Vector parameters,
                                   Vertex<DeepGraph.VertexContent, E> vertex,
                                   Vertex<DeepGraph.VertexContent, E> differentiatingVertex,
                                   int step) {
        Vector neighborFeatureVectorsSum = Vectors.dense(featureVectorsSize);
        neighborFeatureVectorsSum.addInPlace(vertex.content().incomingFeaturesSum(step));
        neighborFeatureVectorsSum.addInPlace(vertex.content().outgoingFeaturesSum(step));
//        for (Edge<GraphRecursiveNeuralNetwork.VertexContentType, E> incomingEdge : vertex.incomingEdges()) {
//            GraphRecursiveNeuralNetwork.VertexContentType vertexContent = incomingEdge.sourceVertex().content();
//            neighborFeatureVectorsSum.addInPlace(vertexContent.features[step]);
//        }
//        for (Edge<GraphRecursiveNeuralNetwork.VertexContentType, E> outgoingEdge : vertex.outgoingEdges()) {
//            GraphRecursiveNeuralNetwork.VertexContentType vertexContent = outgoingEdge.destinationVertex().content();
//            neighborFeatureVectorsSum.addInPlace(vertexContent.features[step]);
//        }
        Network network = networks.get(differentiatingVertex.content().id());
        network.set("phi", vertex.content().features[step]);
        network.set("phi_neighbors", neighborFeatureVectorsSum);
        network.set("rho_W_phi", parameters.get(0, featureVectorsSize * featureVectorsSize - 1));
        network.set("rho_W_phi_neighbors", parameters.get(featureVectorsSize * featureVectorsSize, 2 * featureVectorsSize * featureVectorsSize - 1));
        network.set("z_W_phi", parameters.get(2 * featureVectorsSize * featureVectorsSize, 3 * featureVectorsSize * featureVectorsSize - 1));
        network.set("z_W_phi_neighbors", parameters.get(3 * featureVectorsSize * featureVectorsSize, 4 * featureVectorsSize * featureVectorsSize - 1));
        network.set("h_tilde_W_phi", parameters.get(4 * featureVectorsSize * featureVectorsSize, 5 * featureVectorsSize * featureVectorsSize - 1));
        network.set("h_tilde_W_phi_neighbors", parameters.get(5 * featureVectorsSize * featureVectorsSize, 6 * featureVectorsSize * featureVectorsSize - 1));
        network.set("rho_b", parameters.get(6 * featureVectorsSize * featureVectorsSize, 6 * featureVectorsSize * featureVectorsSize + featureVectorsSize - 1));
        network.set("z_b", parameters.get(6 * featureVectorsSize * featureVectorsSize + featureVectorsSize, 6 * featureVectorsSize * featureVectorsSize + 2 * featureVectorsSize - 1));
        network.set("h_tilde_b", parameters.get(6 * featureVectorsSize * featureVectorsSize + 2 * featureVectorsSize, 6 * featureVectorsSize * featureVectorsSize + 3 * featureVectorsSize - 1));
        if (differentiatingVertex.content().id == vertex.content().id) {
            return network.gradient("phi");
        } else {
            boolean incomingVertex = false;
            for (Edge<DeepGraph.VertexContent, E> edge : vertex.incomingEdges())
                if (edge.sourceVertex().content().id == differentiatingVertex.content().id) {
                    incomingVertex = true;
                    break;
                }
            if (incomingVertex) {
                return network.gradient("phi_neighbors");
            } else {
                boolean outgoingVertex = false;
                for (Edge<DeepGraph.VertexContent, E> edge : vertex.outgoingEdges())
                    if (edge.destinationVertex().content().id == differentiatingVertex.content().id) {
                        outgoingVertex = true;
                        break;
                    }
                if (outgoingVertex) {
                    return network.gradient("phi_neighbors");
                } else {
                    return Matrix.zeros(featureVectorsSize, featureVectorsSize);
                }
            }
        }
    }
}
