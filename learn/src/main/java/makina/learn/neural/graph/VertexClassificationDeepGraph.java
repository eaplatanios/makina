package makina.learn.neural.graph;

import makina.learn.graph.Graph;
import makina.learn.neural.activation.SigmoidFunction;
import makina.learn.neural.activation.SoftmaxFunction;
import makina.math.matrix.Matrix;
import makina.math.matrix.Vector;
import makina.math.matrix.Vectors;

import java.util.Random;

/**
 * @author Emmanouil Antonios Platanios
 */
public class VertexClassificationDeepGraph<E> extends DeepGraph<E> {

    public VertexClassificationDeepGraph(int featuresSize,
                                         int outputsSize,
                                         int maximumNumberOfSteps,
                                         boolean binaryClassification,
                                         Graph<VertexContent, E> graph,
                                         UpdateFunctionType updateFunctionType) {
        super(featuresSize,
              outputsSize,
              maximumNumberOfSteps,
              graph,
              updateFunctionType.getFunction(featuresSize, graph.vertices()),
              new ClassificationOutputFunction(featuresSize, outputsSize),
              new CrossEntropyLossFunction(binaryClassification));
    }

    private static class ClassificationOutputFunction extends OutputFunction {
        private final Random random = new Random();

        private final int featuresSize;
        private final int outputsSize;
        private final int parametersSize;

        private ClassificationOutputFunction(int featuresSize, int outputsSize) {
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
                    initialParameters.set(parameterIndex++, (random.nextDouble() - 0.5) * 2 * 1.0 / Math.sqrt(outputsSize));
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
            if (outputsSize == 1)
                return SigmoidFunction.value(value);
            else
                return SoftmaxFunction.value(value);
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
            if (outputsSize == 1)
                return SigmoidFunction.gradient(value).multiply(gradient);
            else
                return SoftmaxFunction.gradient(value).multiply(gradient);
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
            if (outputsSize == 1)
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
        public double value(Vector networkOutput, Vector observedOutput) {
            double value = 0;
            if (binaryClassification) {
                for (Vector.Element element : observedOutput)
                    if (element.value() >= 0.5)
                        value -= Math.log(networkOutput.get(element.index()));
                    else
                        value -= Math.log(1 - networkOutput.get(element.index()));
            } else {
                if (networkOutput.get((int) observedOutput.get(0)) > 0)
                    value -= Math.log(networkOutput.get((int) observedOutput.get(0)));
                else
                    value += Double.MAX_VALUE;
            }
            return value;
        }

        @Override
        public Vector gradient(Vector networkOutput, Vector observedOutput) {
            Vector gradient = Vectors.build(networkOutput.size(), networkOutput.type());
            if (binaryClassification) {
                for (Vector.Element element : observedOutput) {
                    double output = networkOutput.get(element.index());
                    if (element.value() >= 0.5 && output == 0.0) // TODO: Fix this.
                        gradient.set(element.index(), -Double.MAX_VALUE);
                    else if (element.value() < 0.5 && output == 1.0)
                        gradient.set(element.index(), Double.MAX_VALUE);
                    else if (output > 0.0 && output < 1.0)
                        gradient.set(element.index(), (output - element.value()) / (output * (1 - output)));
                }
            } else {
                double output = networkOutput.get((int) observedOutput.get(0));
                if (output == 0.0)
                    gradient.set((int) observedOutput.get(0), -Double.MAX_VALUE);
                else if (output > 0.0 && output <= 1.0)
                    gradient.set((int) observedOutput.get(0), -1.0 / output);
            }
            return gradient;
        }
    }
}
