package org.platanios.learn.neural.graph;

import org.platanios.learn.graph.Edge;
import org.platanios.learn.graph.Graph;
import org.platanios.learn.graph.Vertex;
import org.platanios.learn.math.matrix.*;
import org.platanios.learn.neural.activation.ActivationFunction;
import org.platanios.learn.neural.activation.SigmoidFunction;
import org.platanios.learn.optimization.QuasiNewtonSolver;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.function.DerivativesApproximation;
import org.platanios.learn.optimization.function.NonSmoothFunctionException;
import org.platanios.learn.optimization.linesearch.BacktrackingLineSearch;
import org.platanios.utilities.Arrays;

import java.util.Map;

/**
 * @author Emmanouil Antonios Platanios
 */
public class SimpleGraphRNN<E> {
    private final int featureVectorsSize;
    private final GraphRecursiveNeuralNetwork<E> graphRNN;
    private final FeatureVectorFunction<E> featureVectorFunction;
    private final InnerProductOutputFunction outputFunction;
    private final ObjectiveFunction objectiveFunction;

    private Vector outputFunctionParameters;
    private Vector featureVectorFunctionParameters;

    public SimpleGraphRNN(int featureVectorsSize,
                          int maximumNumberOfSteps,
                          Graph<GraphRecursiveNeuralNetwork.VertexContentType, E> graph,
                          Map<Integer, Vector> trainingData) {
        this.featureVectorsSize = featureVectorsSize;
        featureVectorFunction = new FeatureVectorFunction<>(featureVectorsSize);
        outputFunction = new InnerProductOutputFunction();
        graphRNN = new GraphRecursiveNeuralNetwork<>(
                featureVectorsSize,
                1,
                maximumNumberOfSteps,
                graph,
                trainingData,
                featureVectorFunction,
                outputFunction,
                new L2NormLossFunction(),
//                Vectors.dense(featureVectorFunction.getParametersVectorSize()),
                featureVectorFunction.getInitialParametersVector(),
                Vectors.dense(featureVectorsSize)
        );
        objectiveFunction = new ObjectiveFunction();
    }

    public Graph<GraphRecursiveNeuralNetwork.VertexContentType, E> getGraph() {
        return graphRNN.getGraph();
    }

    public void trainNetwork() {
        Vector initialPoint = Vectors.dense(featureVectorsSize + featureVectorFunction.getParametersVectorSize());
        for (Vector.VectorElement element : featureVectorFunction.getInitialParametersVector())
            initialPoint.set(element.index() + featureVectorsSize, element.value());
        BacktrackingLineSearch lineSearch = new BacktrackingLineSearch(objectiveFunction, 0.9, 0.5);
        lineSearch.setInitialStepSize(1.0);
        QuasiNewtonSolver solver =
                new QuasiNewtonSolver.Builder(objectiveFunction, initialPoint)
                        .method(QuasiNewtonSolver.Method.BROYDEN_FLETCHER_GOLDFARB_SHANNO)
//                        .method(QuasiNewtonSolver.Method.LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO)
//                        .m(10)
                        .lineSearch(lineSearch)
                        .gradientTolerance(1e-10)
//                        .lineSearch(new NoLineSearch(0.1))
                        .loggingLevel(5)
                        .build();
//        NonlinearConjugateGradientSolver solver =
//                new NonlinearConjugateGradientSolver.Builder(objectiveFunction,
//                                                             DenseVector.generateRandomVector(featureVectorsSize + featureVectorFunction.getParametersVectorSize()))
//                        .method(NonlinearConjugateGradientSolver.Method.FLETCHER_RIEVES_POLAK_RIBIERE)
//                        .restartMethod(NonlinearConjugateGradientSolver.RestartMethod.GRADIENTS_ORTHOGONALITY_CHECK)
//                        .loggingLevel(5)
//                        .build();
//        RPropSolver solver =
//                new RPropSolver.Builder(objectiveFunction,
//                                        DenseVector.generateRandomVector(featureVectorsSize + featureVectorFunction.getParametersVectorSize()))
//                        .lineSearch(lineSearch)
//                        .checkForPointConvergence(true)
//                        .checkForObjectiveConvergence(true)
//                        .loggingLevel(5)
//                        .build();
        Vector solution = solver.solve();
        outputFunctionParameters = solution.get(0, featureVectorsSize - 1);
        featureVectorFunctionParameters = solution.get(featureVectorsSize, solution.size() - 1);
        graphRNN.setOutputFunctionParameters(outputFunctionParameters);
        graphRNN.setFeatureVectorFunctionParameters(featureVectorFunctionParameters);
    }

    public void performForwardPass() {
        graphRNN.performForwardPass();
    }

    public void resetGraph() {
        graphRNN.resetGraph();
    }

    public double getOutputForVertex(Vertex<GraphRecursiveNeuralNetwork.VertexContentType, E> vertex) {
        return outputFunction.value(vertex.getContent().getFeatureVector(), outputFunctionParameters).get(0);
    }

    public boolean checkDerivative(double tolerance) {
        try {
            DerivativesApproximation derivativesApproximation =
                    new DerivativesApproximation(objectiveFunction, DerivativesApproximation.Method.CENTRAL_DIFFERENCE);
            Vector point = DenseVector.generateRandomVector(featureVectorsSize + featureVectorFunction.getParametersVectorSize());
            double[] actualResult = derivativesApproximation.approximateGradient(point).getDenseArray();
            double[] expectedResult = objectiveFunction.getGradient(point).getDenseArray();
            return Arrays.equals(actualResult, expectedResult, tolerance);
        } catch (NonSmoothFunctionException e) {
            return false;
        }
    }

    private class ObjectiveFunction extends AbstractFunction {
        private Vector oldPoint = Vectors.dense(featureVectorsSize + featureVectorFunction.getParametersVectorSize());

        @Override
        protected double computeValue(Vector point) {
            checkPoint(point);
            return graphRNN.getLossFunctionValue();
        }

        @Override
        protected Vector computeGradient(Vector point) {
            checkPoint(point);
            Vector gradient = Vectors.dense(featureVectorsSize + featureVectorFunction.getParametersVectorSize());
            gradient.set(0, featureVectorsSize - 1, graphRNN.getOutputFunctionParametersGradient());
            gradient.set(featureVectorsSize, point.size() - 1, graphRNN.getFeatureVectorFunctionParametersGradient());
            return gradient;
        }

        private void checkPoint(Vector point) {
            if (!Arrays.equals(oldPoint.getDenseArray(), point.getDenseArray(), 1e-5)) {
                graphRNN.setOutputFunctionParameters(point.get(0, featureVectorsSize - 1));
                graphRNN.setFeatureVectorFunctionParameters(point.get(featureVectorsSize, point.size() - 1));
            }
        }
    }

    public static class FeatureVectorFunction<E> extends GraphRecursiveNeuralNetwork.FeatureVectorFunction<GraphRecursiveNeuralNetwork.VertexContentType, E> {
//        private final ActivationFunction activationFunction = new LeakyRectifiedLinearFunction.Builder().build();
        private final ActivationFunction activationFunction = new SigmoidFunction();

        private final int featureVectorsSize;
        private final int parametersVectorSize;

        public FeatureVectorFunction(int featureVectorsSize) {
            this.featureVectorsSize = featureVectorsSize;
            parametersVectorSize = 3 * (featureVectorsSize * featureVectorsSize) + featureVectorsSize;
        }

        public int getParametersVectorSize() {
            return parametersVectorSize;
        }

        public Vector getInitialParametersVector() {
            Vector initialParametersVector = Vectors.dense(parametersVectorSize);
            int parameterVectorIndex = 0;
            for (int i = 0; i < featureVectorsSize; i++)
                for (int j = 0; j < featureVectorsSize; j++)
                    if (i == j)
                        initialParametersVector.set(parameterVectorIndex++, 1);
                    else
                        parameterVectorIndex++;
            return initialParametersVector;
        }

        @Override
        public Vector value(Vector parameters, Vertex<GraphRecursiveNeuralNetwork.VertexContentType, E> vertex, int step) {
            Vector value = Vectors.dense(featureVectorsSize);
            int parameterVectorIndex = 0;
            for (int i = 0; i < featureVectorsSize; i++)
                for (int j = 0; j < featureVectorsSize; j++)
                    value.set(i, value.get(i) + parameters.get(parameterVectorIndex++) * vertex.getContent().featureVectors[step].get(j));
            Vector incomingFeatureVectorsSum = Vectors.dense(featureVectorsSize);
            for (Edge<GraphRecursiveNeuralNetwork.VertexContentType, E> incomingEdge : vertex.getIncomingEdges()) {
                GraphRecursiveNeuralNetwork.VertexContentType vertexContent = incomingEdge.getSourceVertex().getContent();
                incomingFeatureVectorsSum.addInPlace(vertexContent.featureVectors[step]);
            }
            for (int i = 0; i < featureVectorsSize; i++)
                for (int j = 0; j < featureVectorsSize; j++)
                    value.set(i, value.get(i) + parameters.get(parameterVectorIndex++) * incomingFeatureVectorsSum.get(j));
            Vector outgoingFeatureVectorsSum = Vectors.dense(featureVectorsSize);
            for (Edge<GraphRecursiveNeuralNetwork.VertexContentType, E> outgoingEdge : vertex.getOutgoingEdges()) {
                GraphRecursiveNeuralNetwork.VertexContentType vertexContent = outgoingEdge.getDestinationVertex().getContent();
                outgoingFeatureVectorsSum.addInPlace(vertexContent.featureVectors[step]);
            }
            for (int i = 0; i < featureVectorsSize; i++)
                for (int j = 0; j < featureVectorsSize; j++)
                    value.set(i, value.get(i) + parameters.get(parameterVectorIndex++) * outgoingFeatureVectorsSum.get(j));
            for (int i = 0; i < featureVectorsSize; i++)
                value.set(i, value.get(i) + parameters.get(parameterVectorIndex++));
            return activationFunction.getValue(value);
        }

        @Override
        public Matrix gradient(Vector parameters, Vertex<GraphRecursiveNeuralNetwork.VertexContentType, E> vertex, int step) {
            Vector value = Vectors.dense(featureVectorsSize);
            Matrix gradient = new Matrix(featureVectorsSize, parameters.size());
            int parameterVectorIndex = 0;
            for (int i = 0; i < featureVectorsSize; i++)
                for (int j = 0; j < featureVectorsSize; j++) {
                    value.set(i, value.get(i) + parameters.get(parameterVectorIndex) * vertex.getContent().featureVectors[step].get(j));
                    gradient.setElement(i, parameterVectorIndex++, vertex.getContent().featureVectors[step].get(j));
                }
            Vector incomingFeatureVectorsSum = Vectors.dense(featureVectorsSize);
            for (Edge<GraphRecursiveNeuralNetwork.VertexContentType, E> incomingEdge : vertex.getIncomingEdges()) {
                GraphRecursiveNeuralNetwork.VertexContentType vertexContent = incomingEdge.getSourceVertex().getContent();
                incomingFeatureVectorsSum.addInPlace(vertexContent.featureVectors[step]);
            }
            for (int i = 0; i < featureVectorsSize; i++)
                for (int j = 0; j < featureVectorsSize; j++) {
                    value.set(i, value.get(i) + parameters.get(parameterVectorIndex) * incomingFeatureVectorsSum.get(j));
                    gradient.setElement(i, parameterVectorIndex++, incomingFeatureVectorsSum.get(j));
                }
            Vector outgoingFeatureVectorsSum = Vectors.dense(featureVectorsSize);
            for (Edge<GraphRecursiveNeuralNetwork.VertexContentType, E> outgoingEdge : vertex.getOutgoingEdges()) {
                GraphRecursiveNeuralNetwork.VertexContentType vertexContent = outgoingEdge.getDestinationVertex().getContent();
                outgoingFeatureVectorsSum.addInPlace(vertexContent.featureVectors[step]);
            }
            for (int i = 0; i < featureVectorsSize; i++)
                for (int j = 0; j < featureVectorsSize; j++) {
                    value.set(i, value.get(i) + parameters.get(parameterVectorIndex) * outgoingFeatureVectorsSum.get(j));
                    gradient.setElement(i, parameterVectorIndex++, outgoingFeatureVectorsSum.get(j));
                }
            for (int i = 0; i < featureVectorsSize; i++) {
                value.set(i, value.get(i) + parameters.get(parameterVectorIndex));
                gradient.setElement(i, parameterVectorIndex++, 1);
            }
            return activationFunction.getGradient(value).multiply(gradient);
        }

        @Override
        public Matrix featureVectorGradient(Vector parameters,
                                            Vertex<GraphRecursiveNeuralNetwork.VertexContentType, E> vertex,
                                            Vertex<GraphRecursiveNeuralNetwork.VertexContentType, E> differentiatingVertex,
                                            int step) {
            Matrix gradient = new Matrix(featureVectorsSize, featureVectorsSize);
            if (differentiatingVertex.getContent().id == vertex.getContent().id) {
                int parameterVectorIndex = 0;
                for (int i = 0; i < featureVectorsSize; i++)
                    for (int j = 0; j < featureVectorsSize; j++)
                        gradient.setElement(i, j, parameters.get(parameterVectorIndex++));
            } else {
                boolean incomingVertex = false;
                for (Edge<GraphRecursiveNeuralNetwork.VertexContentType, E> edge : vertex.getIncomingEdges())
                    if (edge.getSourceVertex().getContent().id == differentiatingVertex.getContent().id) {
                        incomingVertex = true;
                        break;
                    }
                if (incomingVertex) {
                    int parameterVectorIndex = featureVectorsSize * featureVectorsSize;
                    for (int i = 0; i < featureVectorsSize; i++)
                        for (int j = 0; j < featureVectorsSize; j++)
                            gradient.setElement(i, j, parameters.get(parameterVectorIndex++));
                } else {
                    boolean outgoingVertex = false;
                    for (Edge<GraphRecursiveNeuralNetwork.VertexContentType, E> edge : vertex.getOutgoingEdges())
                        if (edge.getDestinationVertex().getContent().id == differentiatingVertex.getContent().id) {
                            outgoingVertex = true;
                            break;
                        }
                    if (outgoingVertex) {
                        int parameterVectorIndex = 2 * (featureVectorsSize * featureVectorsSize);
                        for (int i = 0; i < featureVectorsSize; i++)
                            for (int j = 0; j < featureVectorsSize; j++)
                                gradient.setElement(i, j, parameters.get(parameterVectorIndex++));
                    }
                }
            }
            return activationFunction.getGradient(value(parameters, vertex, step)).multiply(gradient);
        }
    }

    public static class InnerProductOutputFunction extends GraphRecursiveNeuralNetwork.OutputFunction {
        @Override
        public Vector value(Vector featureVector, Vector parameters) {
            return Vectors.dense(featureVector.inner(parameters));
        }

        @Override
        public Matrix featureVectorGradient(Vector featureVector, Vector parameters) {
            Matrix gradient = new Matrix(1, parameters.size());
            gradient.setRow(0, parameters);
            return gradient;
        }

        @Override
        public Matrix parametersGradient(Vector featureVector, Vector parameters) {
            Matrix gradient = new Matrix(1, featureVector.size());
            gradient.setRow(0, featureVector);
            return gradient;
        }
    }

    public static class L2NormLossFunction extends GraphRecursiveNeuralNetwork.LossFunction {
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
