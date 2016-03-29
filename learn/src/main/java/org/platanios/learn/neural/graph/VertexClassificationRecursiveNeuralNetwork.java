package org.platanios.learn.neural.graph;

import org.platanios.learn.graph.Graph;
import org.platanios.math.matrix.Matrix;
import org.platanios.math.matrix.Vector;
import org.platanios.math.matrix.Vectors;
import org.platanios.learn.neural.activation.SigmoidFunction;
import org.platanios.learn.neural.activation.SoftmaxFunction;

import java.util.Map;
import java.util.Random;

/**
 * @author Emmanouil Antonios Platanios
 */
public class VertexClassificationRecursiveNeuralNetwork<E> extends GraphRecursiveNeuralNetwork<E> {

    public VertexClassificationRecursiveNeuralNetwork(int featureVectorsSize,
                                                      int outputVectorSize,
                                                      int maximumNumberOfSteps,
                                                      boolean binaryClassification,
                                                      Graph<VertexContentType, E> graph,
                                                      Map<Integer, Vector> trainingData,
                                                      FeatureVectorFunctionType featureVectorFunctionType) {
        super(featureVectorsSize,
              outputVectorSize,
              maximumNumberOfSteps,
              graph,
              trainingData,
              featureVectorFunctionType.getFunction(featureVectorsSize, graph.getVertices()),
              new ClassificationOutputFunction(featureVectorsSize, outputVectorSize),
              new CrossEntropyLossFunction(binaryClassification));
    }

    private static class ClassificationOutputFunction extends OutputFunction {
        private final Random random = new Random();

        private final int featureVectorsSize;
        private final int outputVectorSize;
        private final int parametersVectorSize;

        private ClassificationOutputFunction(int featureVectorsSize, int outputVectorSize) {
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
            if (outputVectorSize == 1)
                return SigmoidFunction.value(value);
            else
                return SoftmaxFunction.value(value);
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
            if (outputVectorSize == 1)
                return SigmoidFunction.gradient(value).multiply(gradient);
            else
                return SoftmaxFunction.gradient(value).multiply(gradient);
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
            if (outputVectorSize == 1)
                return SigmoidFunction.gradient(value).multiply(gradient);
            else
                return SoftmaxFunction.gradient(value).multiply(gradient);
        }
    }

    private static class CrossEntropyLossFunction extends LossFunction {
        private final boolean binaryClassification;

        private CrossEntropyLossFunction(boolean binaryClassification) {
            this.binaryClassification = binaryClassification;
        }

        @Override
        public double value(Vector networkOutput, Vector correctOutput) {
            double value = 0;
            if (binaryClassification) {
                for (Vector.VectorElement element : correctOutput)
                    if (element.value() >= 0.5)
                        value -= Math.log(networkOutput.get(element.index()));
                    else
                        value -= Math.log(1 - networkOutput.get(element.index()));
            } else {
                if (networkOutput.get((int) correctOutput.get(0)) > 0)
                    value -= Math.log(networkOutput.get((int) correctOutput.get(0)));
                else
                    value += Double.MAX_VALUE;
            }
            return value;
        }

        @Override
        public Vector gradient(Vector networkOutput, Vector correctOutput) {
            Vector gradient = Vectors.build(networkOutput.size(), networkOutput.type());
            if (binaryClassification) {
                for (Vector.VectorElement element : correctOutput) {
                    double output = networkOutput.get(element.index());
                    if (element.value() >= 0.5 && output == 0.0) // TODO: Fix this.
                        gradient.set(element.index(), -Double.MAX_VALUE);
                    else if (element.value() < 0.5 && output == 1.0)
                        gradient.set(element.index(), Double.MAX_VALUE);
                    else if (output > 0.0 && output < 1.0)
                        gradient.set(element.index(), (output - element.value()) / (output * (1 - output)));
                }
            } else {
                double output = networkOutput.get((int) correctOutput.get(0));
                if (output == 0.0)
                    gradient.set((int) correctOutput.get(0), -Double.MAX_VALUE);
                else if (output > 0.0 && output <= 1.0)
                    gradient.set((int) correctOutput.get(0), -1.0 / output);
            }
            return gradient;
        }
    }
}
