package org.platanios.learn.neural.graph;

import org.platanios.learn.graph.Graph;
import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.VectorNorm;
import org.platanios.learn.math.matrix.Vectors;

import java.util.Map;

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
              new InnerProductOutputFunction(featureVectorsSize, outputVectorSize),
              new L2NormLossFunction());
    }

    private static class InnerProductOutputFunction extends GraphRecursiveNeuralNetwork.OutputFunction {
        private final int featureVectorsSize;
        private final int outputVectorSize;
        private final int parametersVectorSize;

        private InnerProductOutputFunction(int featureVectorsSize, int outputVectorSize) {
            this.featureVectorsSize = featureVectorsSize;
            this.outputVectorSize = outputVectorSize;
            this.parametersVectorSize = featureVectorsSize * outputVectorSize;
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
                    initialParametersVector.set(parameterVectorIndex++, 1.0);
            return initialParametersVector;
        }

        @Override
        public Vector value(Vector featureVector, Vector parameters) {
            Vector value = Vectors.dense(outputVectorSize);
            int parameterVectorIndex = 0;
            for (int i = 0; i < outputVectorSize; i++)
                for (int j = 0; j < featureVectorsSize; j++)
                    value.set(i, value.get(i) + parameters.get(parameterVectorIndex++) * featureVector.get(j));
            return value;
        }

        @Override
        public Matrix featureVectorGradient(Vector featureVector, Vector parameters) {
            Matrix gradient = new Matrix(outputVectorSize, featureVectorsSize);
            int parameterVectorIndex = 0;
            for (int i = 0; i < outputVectorSize; i++)
                for (int j = 0; j < featureVectorsSize; j++)
                    gradient.setElement(i, j, parameters.get(parameterVectorIndex++));
            return gradient;
        }

        @Override
        public Matrix parametersGradient(Vector featureVector, Vector parameters) {
            Matrix gradient = new Matrix(outputVectorSize, parametersVectorSize);
            int parameterVectorIndex = 0;
            for (int i = 0; i < outputVectorSize; i++)
                for (int j = 0; j < featureVectorsSize; j++)
                    gradient.setElement(i, parameterVectorIndex++, featureVector.get(j));
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
