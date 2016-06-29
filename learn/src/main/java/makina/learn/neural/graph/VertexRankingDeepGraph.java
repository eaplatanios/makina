package makina.learn.neural.graph;

import makina.learn.neural.network.Network;
import makina.learn.graph.Graph;
import makina.learn.neural.activation.SigmoidFunction;
import makina.learn.neural.network.NetworkBuilder;
import makina.math.matrix.Matrix;
import makina.math.matrix.Vector;
import makina.math.matrix.VectorNorm;
import makina.math.matrix.Vectors;

import java.util.Random;

/**
 * @author Emmanouil Antonios Platanios
 */
public class VertexRankingDeepGraph<E> extends DeepGraph<E> {

    public VertexRankingDeepGraph(int featuresSize,
                                  int outputsSize,
                                  int maximumNumberOfSteps,
                                  Graph<VertexContent, E> graph,
                                  UpdateFunctionType updateFunctionType) {
        super(featuresSize,
              outputsSize,
              maximumNumberOfSteps,
              graph,
              updateFunctionType.getFunction(featuresSize, graph.vertices()),
              new NetworkOutputFunction(featuresSize, outputsSize),
              new L2NormLossFunction());
    }

    private static class InnerProductOutputFunction extends DeepGraph.OutputFunction {
        private final Random random = new Random();
        private final SigmoidFunction activationFunction = new SigmoidFunction();

        private final int featuresSize;
        private final int outputsSize;
        private final int parametersSize;

        private InnerProductOutputFunction(int featuresSize, int outputsSize) {
            this.featuresSize = featuresSize;
            this.outputsSize = outputsSize;
            this.parametersSize = (featuresSize + 1) * outputsSize;
        }

        @Override
        public int parametersSize() {
            return parametersSize;
        }

        @Override
        public Vector initialParameters() {
            Vector initialParameters = Vectors.dense(parametersSize);
            int parameterIndex = 0;
            for (int i = 0; i < outputsSize; i++)
                for (int j = 0; j < featuresSize; j++)
                    initialParameters.set(parameterIndex++,
                                                (random.nextDouble() - 0.5) * 2 * 1.0 / Math.sqrt(outputsSize));
            return initialParameters;
        }

        @Override
        public Vector value(Vector features, Vector parameters) {
            Vector value = Vectors.dense(outputsSize);
            int parameterIndex = 0;
            for (int i = 0; i < outputsSize; i++)
                for (int j = 0; j < featuresSize; j++)
                    value.set(i, value.get(i) + parameters.get(parameterIndex++) * features.get(j));
            for (int i = 0; i < outputsSize; i++)
                value.set(i, value.get(i) + parameters.get(parameterIndex++));
            return activationFunction.value(value);
        }

        @Override
        public Matrix featuresGradient(Vector features, Vector parameters) {
            Vector value = Vectors.dense(outputsSize);
            Matrix gradient = new Matrix(outputsSize, featuresSize);
            int parameterIndex = 0;
            for (int i = 0; i < outputsSize; i++)
                for (int j = 0; j < featuresSize; j++) {
                    value.set(i, value.get(i) + parameters.get(parameterIndex) * features.get(j));
                    gradient.setElement(i, j, parameters.get(parameterIndex++));
                }
            for (int i = 0; i < outputsSize; i++)
                value.set(i, value.get(i) + parameters.get(parameterIndex++));
            return activationFunction.gradient(value).multiply(gradient);
        }

        @Override
        public Matrix parametersGradient(Vector features, Vector parameters) {
            Vector value = Vectors.dense(outputsSize);
            Matrix gradient = new Matrix(outputsSize, parametersSize);
            int parameterIndex = 0;
            for (int i = 0; i < outputsSize; i++)
                for (int j = 0; j < featuresSize; j++) {
                    value.set(i, value.get(i) + parameters.get(parameterIndex) * features.get(j));
                    gradient.setElement(i, parameterIndex++, features.get(j));
                }
            for (int i = 0; i < outputsSize; i++) {
                value.set(i, value.get(i) + parameters.get(parameterIndex));
                gradient.setElement(i, parameterIndex++, 1);
            }
            return activationFunction.gradient(value).multiply(gradient);
        }
    }

    private static class NetworkOutputFunction extends DeepGraph.OutputFunction {
        private final Random random = new Random();

        private final Network network;
        private final int featuresSize;
        private final int outputsSize;
        private final int parametersSize;

        private NetworkOutputFunction(int featuresSize, int outputsSize) {
            this.featuresSize = featuresSize;
            this.outputsSize = outputsSize;
            parametersSize = 2 * featuresSize * (featuresSize + outputsSize + 1) + outputsSize;
            NetworkBuilder networkBuilder = new NetworkBuilder();
            int inputLayerId = networkBuilder.addInputLayer(featuresSize, "input");
            int hiddenLayerId = networkBuilder.addFullyConnectedLayer(inputLayerId, 2 * featuresSize, "W_hidden", "b_hidden");
            int activationLayerId = networkBuilder.addSigmoidLayer(hiddenLayerId);
            int outputLayerId = networkBuilder.addFullyConnectedLayer(true, activationLayerId, outputsSize, "W_out", "b_out");
            network = networkBuilder.build();
        }

        @Override
        public int parametersSize() {
            return parametersSize;
        }

        @Override
        public Vector initialParameters() {
            Vector initialParameters = Vectors.dense(parametersSize);
            int parameterIndex = 0;
            for (int i = 0; i < 2 * featuresSize; i++)
                for (int j = 0; j < featuresSize; j++)
                    initialParameters.set(parameterIndex++,
                                                (random.nextDouble() - 0.5) * 2 * 1.0 / Math.sqrt(2 * featuresSize));
            for (int i = 0; i < outputsSize; i++)
                for (int j = 0; j < 2 * featuresSize; j++)
                    initialParameters.set(parameterIndex++,
                                                (random.nextDouble() - 0.5) * 2 * 1.0 / Math.sqrt(2 * outputsSize));
            return initialParameters;
        }

        @Override
        public synchronized Vector value(Vector features, Vector parameters) {
            network.set("input", features);
            network.set("W_hidden", parameters.get(0, 2 * featuresSize * featuresSize - 1));
            network.set("W_out", parameters.get(2 * featuresSize * featuresSize, 2 * featuresSize * (featuresSize + outputsSize) - 1));
            network.set("b_hidden", parameters.get(2 * featuresSize * (featuresSize + outputsSize), 2 * featuresSize * (featuresSize + outputsSize + 1) - 1));
            network.set("b_out", parameters.get(2 * featuresSize * (featuresSize + outputsSize + 1), 2 * featuresSize * (featuresSize + outputsSize + 1) + outputsSize - 1));
            return network.value();
        }

        @Override
        public synchronized Matrix featuresGradient(Vector features, Vector parameters) {
            network.set("input", features);
            network.set("W_hidden", parameters.get(0, 2 * featuresSize * featuresSize - 1));
            network.set("W_out", parameters.get(2 * featuresSize * featuresSize, 2 * featuresSize * (featuresSize + outputsSize) - 1));
            network.set("b_hidden", parameters.get(2 * featuresSize * (featuresSize + outputsSize), 2 * featuresSize * (featuresSize + outputsSize + 1) - 1));
            network.set("b_out", parameters.get(2 * featuresSize * (featuresSize + outputsSize + 1), 2 * featuresSize * (featuresSize + outputsSize + 1) + outputsSize - 1));
            return network.gradient("input");
        }

        @Override
        public synchronized Matrix parametersGradient(Vector features, Vector parameters) {
            network.set("input", features);
            network.set("W_hidden", parameters.get(0, 2 * featuresSize * featuresSize - 1));
            network.set("W_out", parameters.get(2 * featuresSize * featuresSize, 2 * featuresSize * (featuresSize + outputsSize) - 1));
            network.set("b_hidden", parameters.get(2 * featuresSize * (featuresSize + outputsSize), 2 * featuresSize * (featuresSize + outputsSize + 1) - 1));
            network.set("b_out", parameters.get(2 * featuresSize * (featuresSize + outputsSize + 1), 2 * featuresSize * (featuresSize + outputsSize + 1) + outputsSize - 1));
            Matrix[] networkGradients = network.gradient("W_hidden", "W_out", "b_hidden", "b_out");
            Matrix gradient = Matrix.zeros(outputsSize, parametersSize);
            gradient.setSubMatrix(0, outputsSize - 1,
                                  0, 2 * featuresSize * featuresSize - 1,
                                  networkGradients[0]);
            gradient.setSubMatrix(0, outputsSize - 1,
                                  2 * featuresSize * featuresSize, 2 * featuresSize * (featuresSize + outputsSize) - 1,
                                  networkGradients[1]);
            gradient.setSubMatrix(0, outputsSize - 1,
                                  2 * featuresSize * (featuresSize + outputsSize), 2 * featuresSize * (featuresSize + outputsSize + 1) - 1,
                                  networkGradients[2]);
            gradient.setSubMatrix(0, outputsSize - 1,
                                  2 * featuresSize * (featuresSize + outputsSize + 1), 2 * featuresSize * (featuresSize + outputsSize + 1) + outputsSize - 1,
                                  networkGradients[3]);
            return gradient;
        }
    }

    private static class L2NormLossFunction extends DeepGraph.LossFunction {
        @Override
        public double value(Vector networkOutput, Vector observedOutput) {
            return networkOutput.sub(observedOutput).norm(VectorNorm.L2_SQUARED);
        }

        @Override
        public Vector gradient(Vector networkOutput, Vector observedOutput) {
            return networkOutput.sub(observedOutput).mult(2);
        }
    }
}
