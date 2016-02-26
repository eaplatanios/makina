package org.platanios.learn.neural.graph;

import org.platanios.learn.graph.Graph;
import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;
import org.platanios.learn.neural.activation.SigmoidFunction;

import java.util.Map;

/**
 * @author Emmanouil Antonios Platanios
 */
public class VertexClassificationRecursiveNeuralNetwork<E> extends GraphRecursiveNeuralNetwork<E> {

    public VertexClassificationRecursiveNeuralNetwork(int featureVectorsSize,
                                                      int outputVectorSize,
                                                      int maximumNumberOfSteps,
                                                      Graph<VertexContentType, E> graph,
                                                      Map<Integer, Vector> trainingData,
                                                      FeatureVectorFunctionType featureVectorFunctionType) {
        super(featureVectorsSize,
              outputVectorSize,
              maximumNumberOfSteps,
              graph,
              trainingData,
              featureVectorFunctionType.getFunction(featureVectorsSize, graph.getVertices()),
              new SigmoidOutputFunction(featureVectorsSize, outputVectorSize),
              new CrossEntropyLossFunction());
    }

    private static class SigmoidOutputFunction extends OutputFunction {
        private final int featureVectorsSize;
        private final int outputVectorSize;
        private final int parametersVectorSize;

        private SigmoidOutputFunction(int featureVectorsSize, int outputVectorSize) {
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
                    initialParametersVector.set(parameterVectorIndex++, 1.0 / featureVectorsSize);
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
            return SigmoidFunction.value(value);
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
            return SigmoidFunction.gradient(value).multiply(gradient);
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
            return SigmoidFunction.gradient(value).multiply(gradient);
        }
    }

    private static class CrossEntropyLossFunction extends LossFunction {
        @Override
        public double value(Vector networkOutput, Vector correctOutput) {
            double value = 0;
            for (Vector.VectorElement element : correctOutput)
                if (element.value() >= 0.5)
                    value -= Math.log(networkOutput.get(element.index()));
                else
                    value -= Math.log(1 - networkOutput.get(element.index()));
            return value;
        }

        @Override
        public Vector gradient(Vector networkOutput, Vector correctOutput) {
            Vector gradient = Vectors.build(networkOutput.size(), networkOutput.type());
            for (Vector.VectorElement element : correctOutput) {
                double output = networkOutput.get(element.index());
                if (element.value() >= 0.5 && output == 0.0) // TODO: Fix this.
                    gradient.set(element.index(), -Double.MAX_VALUE);
                else if (element.value() < 0.5 && output == 1.0)
                    gradient.set(element.index(), Double.MAX_VALUE);
                else if (output > 0.0 && output < 1.0)
                    gradient.set(element.index(), (output - element.value()) / (output * (1 - output)));
            }
            return gradient;
        }
    }
}
