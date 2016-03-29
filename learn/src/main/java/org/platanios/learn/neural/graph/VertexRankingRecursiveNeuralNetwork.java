package org.platanios.learn.neural.graph;

import org.platanios.learn.graph.Graph;
import org.platanios.math.matrix.Matrix;
import org.platanios.math.matrix.Vector;
import org.platanios.math.matrix.VectorNorm;
import org.platanios.math.matrix.Vectors;
import org.platanios.learn.neural.activation.SigmoidFunction;
import org.platanios.learn.neural.network.Network;
import org.platanios.learn.neural.network.NetworkBuilder;

import java.util.Map;
import java.util.Random;

/**
 * @author Emmanouil Antonios Platanios
 */
public class VertexRankingRecursiveNeuralNetwork<E> extends GraphRecursiveNeuralNetwork<E> {

    public VertexRankingRecursiveNeuralNetwork(int featureVectorsSize,
                                               int outputVectorSize,
                                               int maximumNumberOfSteps,
                                               Graph<GraphRecursiveNeuralNetwork.VertexContentType, E> graph,
                                               Map<Integer, Vector> trainingData,
                                               FeatureVectorFunctionType featureVectorFunctionType) {
        super(featureVectorsSize,
              outputVectorSize,
              maximumNumberOfSteps,
              graph,
              trainingData,
              featureVectorFunctionType.getFunction(featureVectorsSize, graph.getVertices()),
              new NetworkOutputFunction(featureVectorsSize, outputVectorSize),
              new L2NormLossFunction());
    }

    private static class InnerProductOutputFunction extends GraphRecursiveNeuralNetwork.OutputFunction {
        private final Random random = new Random();
        private final SigmoidFunction activationFunction = new SigmoidFunction();

        private final int featureVectorsSize;
        private final int outputVectorSize;
        private final int parametersVectorSize;

        private InnerProductOutputFunction(int featureVectorsSize, int outputVectorSize) {
            this.featureVectorsSize = featureVectorsSize;
            this.outputVectorSize = outputVectorSize;
            this.parametersVectorSize = (featureVectorsSize + 1) * outputVectorSize;
        }

        @Override
        public int getParametersVectorSize() {
            return parametersVectorSize;
        }

        @Override
        public Vector getInitialParametersVector() {
            Vector initialParametersVector = Vectors.dense(parametersVectorSize);
            int parameterVectorIndex = 0;
            for (int i = 0; i < outputVectorSize; i++)
                for (int j = 0; j < featureVectorsSize; j++)
                    initialParametersVector.set(parameterVectorIndex++,
                                                (random.nextDouble() - 0.5) * 2 * 1.0 / Math.sqrt(outputVectorSize));
            return initialParametersVector;
        }

        @Override
        public Vector value(Vector featureVector, Vector parameters) {
            Vector value = Vectors.dense(outputVectorSize);
            int parameterVectorIndex = 0;
            for (int i = 0; i < outputVectorSize; i++)
                for (int j = 0; j < featureVectorsSize; j++)
                    value.set(i, value.get(i) + parameters.get(parameterVectorIndex++) * featureVector.get(j));
            for (int i = 0; i < outputVectorSize; i++)
                value.set(i, value.get(i) + parameters.get(parameterVectorIndex++));
            return activationFunction.value(value);
        }

        @Override
        public Matrix featureVectorGradient(Vector featureVector, Vector parameters) {
            Vector value = Vectors.dense(outputVectorSize);
            Matrix gradient = new Matrix(outputVectorSize, featureVectorsSize);
            int parameterVectorIndex = 0;
            for (int i = 0; i < outputVectorSize; i++)
                for (int j = 0; j < featureVectorsSize; j++) {
                    value.set(i, value.get(i) + parameters.get(parameterVectorIndex) * featureVector.get(j));
                    gradient.setElement(i, j, parameters.get(parameterVectorIndex++));
                }
            for (int i = 0; i < outputVectorSize; i++)
                value.set(i, value.get(i) + parameters.get(parameterVectorIndex++));
            return activationFunction.gradient(value).multiply(gradient);
        }

        @Override
        public Matrix parametersGradient(Vector featureVector, Vector parameters) {
            Vector value = Vectors.dense(outputVectorSize);
            Matrix gradient = new Matrix(outputVectorSize, parametersVectorSize);
            int parameterVectorIndex = 0;
            for (int i = 0; i < outputVectorSize; i++)
                for (int j = 0; j < featureVectorsSize; j++) {
                    value.set(i, value.get(i) + parameters.get(parameterVectorIndex) * featureVector.get(j));
                    gradient.setElement(i, parameterVectorIndex++, featureVector.get(j));
                }
            for (int i = 0; i < outputVectorSize; i++) {
                value.set(i, value.get(i) + parameters.get(parameterVectorIndex));
                gradient.setElement(i, parameterVectorIndex++, 1);
            }
            return activationFunction.gradient(value).multiply(gradient);
        }
    }

    private static class NetworkOutputFunction extends GraphRecursiveNeuralNetwork.OutputFunction {
        private final Random random = new Random();

        private final Network network;
        private final int featureVectorsSize;
        private final int outputVectorSize;
        private final int parametersVectorSize;

        private NetworkOutputFunction(int featureVectorsSize, int outputVectorSize) {
            this.featureVectorsSize = featureVectorsSize;
            this.outputVectorSize = outputVectorSize;
            parametersVectorSize = 2 * featureVectorsSize * (featureVectorsSize + outputVectorSize + 1) + outputVectorSize;
            NetworkBuilder networkBuilder = new NetworkBuilder();
            int inputLayerId = networkBuilder.addInputLayer(featureVectorsSize, "input");
            int hiddenLayerId = networkBuilder.addFullyConnectedLayer(inputLayerId, 2 * featureVectorsSize, "W_hidden", "b_hidden");
            int activationLayerId = networkBuilder.addSigmoidLayer(hiddenLayerId);
            int outputLayerId = networkBuilder.addFullyConnectedLayer(true, activationLayerId, outputVectorSize, "W_out", "b_out");
            network = networkBuilder.build();
        }

        @Override
        public int getParametersVectorSize() {
            return parametersVectorSize;
        }

        @Override
        public Vector getInitialParametersVector() {
            Vector initialParametersVector = Vectors.dense(parametersVectorSize);
            int parameterVectorIndex = 0;
            for (int i = 0; i < 2 * featureVectorsSize; i++)
                for (int j = 0; j < featureVectorsSize; j++)
                    initialParametersVector.set(parameterVectorIndex++,
                                                (random.nextDouble() - 0.5) * 2 * 1.0 / Math.sqrt(2 * featureVectorsSize));
            for (int i = 0; i < outputVectorSize; i++)
                for (int j = 0; j < 2 * featureVectorsSize; j++)
                    initialParametersVector.set(parameterVectorIndex++,
                                                (random.nextDouble() - 0.5) * 2 * 1.0 / Math.sqrt(2 * outputVectorSize));
            return initialParametersVector;
        }

        @Override
        public synchronized Vector value(Vector featureVector, Vector parameters) {
            network.set("input", featureVector);
            network.set("W_hidden", parameters.get(0, 2 * featureVectorsSize * featureVectorsSize - 1));
            network.set("W_out", parameters.get(2 * featureVectorsSize * featureVectorsSize, 2 * featureVectorsSize * (featureVectorsSize + outputVectorSize) - 1));
            network.set("b_hidden", parameters.get(2 * featureVectorsSize * (featureVectorsSize + outputVectorSize), 2 * featureVectorsSize * (featureVectorsSize + outputVectorSize + 1) - 1));
            network.set("b_out", parameters.get(2 * featureVectorsSize * (featureVectorsSize + outputVectorSize + 1), 2 * featureVectorsSize * (featureVectorsSize + outputVectorSize + 1) + outputVectorSize - 1));
            return network.value();
        }

        @Override
        public synchronized Matrix featureVectorGradient(Vector featureVector, Vector parameters) {
            network.set("input", featureVector);
            network.set("W_hidden", parameters.get(0, 2 * featureVectorsSize * featureVectorsSize - 1));
            network.set("W_out", parameters.get(2 * featureVectorsSize * featureVectorsSize, 2 * featureVectorsSize * (featureVectorsSize + outputVectorSize) - 1));
            network.set("b_hidden", parameters.get(2 * featureVectorsSize * (featureVectorsSize + outputVectorSize), 2 * featureVectorsSize * (featureVectorsSize + outputVectorSize + 1) - 1));
            network.set("b_out", parameters.get(2 * featureVectorsSize * (featureVectorsSize + outputVectorSize + 1), 2 * featureVectorsSize * (featureVectorsSize + outputVectorSize + 1) + outputVectorSize - 1));
            return network.gradient("input");
        }

        @Override
        public synchronized Matrix parametersGradient(Vector featureVector, Vector parameters) {
            network.set("input", featureVector);
            network.set("W_hidden", parameters.get(0, 2 * featureVectorsSize * featureVectorsSize - 1));
            network.set("W_out", parameters.get(2 * featureVectorsSize * featureVectorsSize, 2 * featureVectorsSize * (featureVectorsSize + outputVectorSize) - 1));
            network.set("b_hidden", parameters.get(2 * featureVectorsSize * (featureVectorsSize + outputVectorSize), 2 * featureVectorsSize * (featureVectorsSize + outputVectorSize + 1) - 1));
            network.set("b_out", parameters.get(2 * featureVectorsSize * (featureVectorsSize + outputVectorSize + 1), 2 * featureVectorsSize * (featureVectorsSize + outputVectorSize + 1) + outputVectorSize - 1));
            Matrix[] networkGradients = network.gradient("W_hidden", "W_out", "b_hidden", "b_out");
            Matrix gradient = Matrix.zeros(outputVectorSize, parametersVectorSize);
            gradient.setSubMatrix(0, outputVectorSize - 1,
                                  0, 2 * featureVectorsSize * featureVectorsSize - 1,
                                  networkGradients[0]);
            gradient.setSubMatrix(0, outputVectorSize - 1,
                                  2 * featureVectorsSize * featureVectorsSize, 2 * featureVectorsSize * (featureVectorsSize + outputVectorSize) - 1,
                                  networkGradients[1]);
            gradient.setSubMatrix(0, outputVectorSize - 1,
                                  2 * featureVectorsSize * (featureVectorsSize + outputVectorSize), 2 * featureVectorsSize * (featureVectorsSize + outputVectorSize + 1) - 1,
                                  networkGradients[2]);
            gradient.setSubMatrix(0, outputVectorSize - 1,
                                  2 * featureVectorsSize * (featureVectorsSize + outputVectorSize + 1), 2 * featureVectorsSize * (featureVectorsSize + outputVectorSize + 1) + outputVectorSize - 1,
                                  networkGradients[3]);
            return gradient;
        }
    }

    private static class L2NormLossFunction extends GraphRecursiveNeuralNetwork.LossFunction {
        @Override
        public double value(Vector networkOutput, Vector correctOutput) {
            return networkOutput.sub(correctOutput).norm(VectorNorm.L2_SQUARED);
        }

        @Override
        public Vector gradient(Vector networkOutput, Vector correctOutput) {
            return networkOutput.sub(correctOutput).mult(2);
        }
    }
}
