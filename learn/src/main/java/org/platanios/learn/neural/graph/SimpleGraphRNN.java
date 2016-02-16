package org.platanios.learn.neural.graph;

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import org.platanios.learn.graph.Edge;
import org.platanios.learn.graph.Graph;
import org.platanios.learn.graph.Vertex;
import org.platanios.learn.math.matrix.*;
import org.platanios.learn.neural.activation.ActivationFunction;
import org.platanios.learn.neural.activation.SigmoidFunction;
import org.platanios.learn.neural.network.*;
import org.platanios.learn.optimization.QuasiNewtonSolver;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.function.DerivativesApproximation;
import org.platanios.learn.optimization.function.NonSmoothFunctionException;
import org.platanios.learn.optimization.linesearch.BacktrackingLineSearch;
import org.platanios.utilities.Arrays;

import java.util.List;
import java.util.Map;

/**
 * @author Emmanouil Antonios Platanios
 */
public class SimpleGraphRNN<E> {
    private final int featureVectorsSize;
    private final int outputVectorSize;
    private final GraphRecursiveNeuralNetwork<E> graphRNN;
    private final FeatureVectorFunction<E> featureVectorFunction;
    private final InnerProductOutputFunction outputFunction;
    private final ObjectiveFunction objectiveFunction;

    private Vector outputFunctionParameters;
    private Vector featureVectorFunctionParameters;

    public SimpleGraphRNN(int featureVectorsSize,
                          int outputVectorSize,
                          int maximumNumberOfSteps,
                          Graph<GraphRecursiveNeuralNetwork.VertexContentType, E> graph,
                          Map<Integer, Vector> trainingData) {
        this.featureVectorsSize = featureVectorsSize;
        this.outputVectorSize = outputVectorSize;
        featureVectorFunction = new FeatureVectorFunction<>(featureVectorsSize);
        outputFunction = new InnerProductOutputFunction(featureVectorsSize, outputVectorSize);
        graphRNN = new GraphRecursiveNeuralNetwork<>(
                featureVectorsSize,
                outputVectorSize,
                maximumNumberOfSteps,
                graph,
                trainingData,
                featureVectorFunction,
                outputFunction,
                new L2NormLossFunction(),
//                Vectors.dense(featureVectorFunction.getParametersVectorSize()),
                featureVectorFunction.getInitialParametersVector(),
                outputFunction.getInitialParametersVector()
        );
        objectiveFunction = new ObjectiveFunction();
    }

    public Graph<GraphRecursiveNeuralNetwork.VertexContentType, E> getGraph() {
        return graphRNN.getGraph();
    }

    public boolean checkDerivative(double tolerance) {
        try {
            graphRNN.randomizeGraph();
            DerivativesApproximation derivativesApproximation =
                    new DerivativesApproximation(objectiveFunction, DerivativesApproximation.Method.CENTRAL_DIFFERENCE, 1e-4);
            Vector point = DenseVector.generateRandomVector(outputFunction.getParametersVectorSize() + featureVectorFunction.getParametersVectorSize());
            double[] actualResult = derivativesApproximation.approximateGradient(point).getDenseArray();
            double[] expectedResult = objectiveFunction.getGradient(point).getDenseArray();
            graphRNN.resetGraph();
            return Arrays.equals(actualResult, expectedResult, tolerance);
        } catch (NonSmoothFunctionException e) {
            return false;
        }
    }

    public void trainNetwork() {
        Vector initialPoint = Vectors.dense(outputFunction.getParametersVectorSize() + featureVectorFunction.getParametersVectorSize());
        for (Vector.VectorElement element : outputFunction.getInitialParametersVector())
            initialPoint.set(element.index(), element.value());
        for (Vector.VectorElement element : featureVectorFunction.getInitialParametersVector())
            initialPoint.set(element.index() + featureVectorsSize, element.value());
        BacktrackingLineSearch lineSearch = new BacktrackingLineSearch(objectiveFunction, 0.5, 0.5);
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
        outputFunctionParameters = solution.get(0, outputFunction.getParametersVectorSize() - 1);
        featureVectorFunctionParameters = solution.get(outputFunction.getParametersVectorSize(), solution.size() - 1);
        graphRNN.setOutputFunctionParameters(outputFunctionParameters);
        graphRNN.setFeatureVectorFunctionParameters(featureVectorFunctionParameters);
    }

    public void performForwardPass() {
        graphRNN.performForwardPass();
    }

    public void resetGraph() {
        graphRNN.resetGraph();
    }

    public Vector getOutputForVertex(Vertex<GraphRecursiveNeuralNetwork.VertexContentType, E> vertex) {
        return outputFunction.value(vertex.getContent().getFeatureVector(), outputFunctionParameters);
    }

    private class ObjectiveFunction extends AbstractFunction {
        private Vector oldPoint = Vectors.dense(0);

        @Override
        protected double computeValue(Vector point) {
            checkPoint(point);
            return graphRNN.getLossFunctionValue();
        }

        @Override
        protected Vector computeGradient(Vector point) {
            checkPoint(point);
            Vector gradient = Vectors.dense(outputFunction.getParametersVectorSize() + featureVectorFunction.getParametersVectorSize());
            gradient.set(0, outputFunction.getParametersVectorSize() - 1, graphRNN.getOutputFunctionParametersGradient());
            gradient.set(outputFunction.getParametersVectorSize(), point.size() - 1, graphRNN.getFeatureVectorFunctionParametersGradient());
            return gradient;
        }

        private void checkPoint(Vector point) {
            if (!Arrays.equals(oldPoint.getDenseArray(), point.getDenseArray(), 1e-5)) {
                graphRNN.setOutputFunctionParameters(point.get(0, outputFunction.getParametersVectorSize() - 1));
                graphRNN.setFeatureVectorFunctionParameters(point.get(outputFunction.getParametersVectorSize(), point.size() - 1));
                oldPoint = point;
            }
        }
    }

    public static class FeatureVectorFunction<E> extends GraphRecursiveNeuralNetwork.FeatureVectorFunction<GraphRecursiveNeuralNetwork.VertexContentType, E> {
        //        private final ActivationFunction activationFunction = new LeakyRectifiedLinearFunction.Builder().build();
        private final ActivationFunction activationFunction = new SigmoidFunction();
        private final Network network;
        private final Variable phi;
        private final Variable phiIn;
        private final Variable phiOut;
        private final Variable newPhi;

        private final int featureVectorsSize;
        private final int parametersVectorSize;

        private State networkState;

        public FeatureVectorFunction(int featureVectorsSize) {
            this.featureVectorsSize = featureVectorsSize;
            parametersVectorSize = 3 * (featureVectorsSize * featureVectorsSize) + featureVectorsSize;
            InputLayer phi = Layers.input(featureVectorsSize);
            InputLayer phiIn = Layers.input(featureVectorsSize);
            InputLayer phiOut = Layers.input(featureVectorsSize);
            Layer phiHiddenLayer = Layers.fullyConnected(phi, phi.outputSize(), "W_phi", "b");
            Layer phiInHiddenLayer = Layers.fullyConnected(phiIn, phiIn.outputSize(), "W_phi_in");
            Layer phiOutHiddenLayer = Layers.fullyConnected(phiOut, phiOut.outputSize(), "W_phi_out");
            Layer additionHiddenLayer = Layers.addition(phiHiddenLayer, phiInHiddenLayer, phiOutHiddenLayer);
            Layer sigmoidLayer = Layers.sigmoidActivation(additionHiddenLayer);
            Layer outputLayer = Layers.output(sigmoidLayer);
            network = new Network.Builder(Lists.newArrayList(phi, phiIn, phiOut))
                    .addLayer(phiHiddenLayer)
                    .addLayer(phiInHiddenLayer)
                    .addLayer(phiOutHiddenLayer)
                    .addLayer(additionHiddenLayer)
                    .addLayer(sigmoidLayer)
                    .addLayer(outputLayer)
                    .build();
            this.phi = phi.inputVariable();
            this.phiIn = phiIn.inputVariable();
            this.phiOut = phiOut.inputVariable();
            this.newPhi = sigmoidLayer.outputVariable();
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
            for (int j = 0; j < featureVectorsSize; j++)
                for (int i = 0; i < featureVectorsSize; i++)
                    value.set(i, value.get(i) + parameters.get(parameterVectorIndex++) * vertex.getContent().featureVectors[step].get(j));
            Vector incomingFeatureVectorsSum = Vectors.dense(featureVectorsSize);
            for (Edge<GraphRecursiveNeuralNetwork.VertexContentType, E> incomingEdge : vertex.getIncomingEdges()) {
                GraphRecursiveNeuralNetwork.VertexContentType vertexContent = incomingEdge.getSourceVertex().getContent();
                incomingFeatureVectorsSum.addInPlace(vertexContent.featureVectors[step]);
            }
            for (int j = 0; j < featureVectorsSize; j++)
                for (int i = 0; i < featureVectorsSize; i++)
                    value.set(i, value.get(i) + parameters.get(parameterVectorIndex++) * incomingFeatureVectorsSum.get(j));
            Vector outgoingFeatureVectorsSum = Vectors.dense(featureVectorsSize);
            for (Edge<GraphRecursiveNeuralNetwork.VertexContentType, E> outgoingEdge : vertex.getOutgoingEdges()) {
                GraphRecursiveNeuralNetwork.VertexContentType vertexContent = outgoingEdge.getDestinationVertex().getContent();
                outgoingFeatureVectorsSum.addInPlace(vertexContent.featureVectors[step]);
            }
            for (int j = 0; j < featureVectorsSize; j++)
                for (int i = 0; i < featureVectorsSize; i++)
                    value.set(i, value.get(i) + parameters.get(parameterVectorIndex++) * outgoingFeatureVectorsSum.get(j));
            for (int i = 0; i < featureVectorsSize; i++)
                value.set(i, value.get(i) + parameters.get(parameterVectorIndex++));
//            return activationFunction.getValue(computeValue);
            value = activationFunction.getValue(value);


            networkState = new State(Sets.union(network.variables(), network.parameters()));
            networkState.set(phi, vertex.getContent().featureVectors[step]);
            networkState.set(phiIn, incomingFeatureVectorsSum);
            networkState.set(phiOut, outgoingFeatureVectorsSum);
            networkState.set("W_phi", parameters.get(0, featureVectorsSize * featureVectorsSize - 1));
            networkState.set("W_phi_in", parameters.get(featureVectorsSize * featureVectorsSize, 2 * featureVectorsSize * featureVectorsSize - 1));
            networkState.set("W_phi_out", parameters.get(2 * featureVectorsSize * featureVectorsSize, 3 * featureVectorsSize * featureVectorsSize - 1));
            networkState.set("b", parameters.get(3 * featureVectorsSize * featureVectorsSize, 3 * featureVectorsSize * featureVectorsSize + featureVectorsSize - 1));
            Vector networkValue = network.value(networkState);

            return value;
        }

        @Override
        public Matrix gradient(Vector parameters, Vertex<GraphRecursiveNeuralNetwork.VertexContentType, E> vertex, int step) {
            Vector value = Vectors.dense(featureVectorsSize);
            Matrix gradient = new Matrix(featureVectorsSize, parameters.size());
            int parameterVectorIndex = 0;
            for (int j = 0; j < featureVectorsSize; j++)
                for (int i = 0; i < featureVectorsSize; i++) {
                    value.set(i, value.get(i) + parameters.get(parameterVectorIndex) * vertex.getContent().featureVectors[step].get(j));
                    gradient.setElement(i, parameterVectorIndex++, vertex.getContent().featureVectors[step].get(j));
                }
            Vector incomingFeatureVectorsSum = Vectors.dense(featureVectorsSize);
            for (Edge<GraphRecursiveNeuralNetwork.VertexContentType, E> incomingEdge : vertex.getIncomingEdges()) {
                GraphRecursiveNeuralNetwork.VertexContentType vertexContent = incomingEdge.getSourceVertex().getContent();
                incomingFeatureVectorsSum.addInPlace(vertexContent.featureVectors[step]);
            }
            for (int j = 0; j < featureVectorsSize; j++)
                for (int i = 0; i < featureVectorsSize; i++) {
                    value.set(i, value.get(i) + parameters.get(parameterVectorIndex) * incomingFeatureVectorsSum.get(j));
                    gradient.setElement(i, parameterVectorIndex++, incomingFeatureVectorsSum.get(j));
                }
            Vector outgoingFeatureVectorsSum = Vectors.dense(featureVectorsSize);
            for (Edge<GraphRecursiveNeuralNetwork.VertexContentType, E> outgoingEdge : vertex.getOutgoingEdges()) {
                GraphRecursiveNeuralNetwork.VertexContentType vertexContent = outgoingEdge.getDestinationVertex().getContent();
                outgoingFeatureVectorsSum.addInPlace(vertexContent.featureVectors[step]);
            }
            for (int j = 0; j < featureVectorsSize; j++)
                for (int i = 0; i < featureVectorsSize; i++) {
                    value.set(i, value.get(i) + parameters.get(parameterVectorIndex) * outgoingFeatureVectorsSum.get(j));
                    gradient.setElement(i, parameterVectorIndex++, outgoingFeatureVectorsSum.get(j));
                }
            for (int i = 0; i < featureVectorsSize; i++) {
                value.set(i, value.get(i) + parameters.get(parameterVectorIndex));
                gradient.setElement(i, parameterVectorIndex++, 1);
            }
//            return activationFunction.getGradient(value).multiply(gradient);
            gradient = activationFunction.getGradient(value).multiply(gradient);


            networkState = new State(Sets.union(network.variables(), network.parameters()));
            networkState.set(phi, vertex.getContent().featureVectors[step]);
            networkState.set(phiIn, incomingFeatureVectorsSum);
            networkState.set(phiOut, outgoingFeatureVectorsSum);
            networkState.set("W_phi", parameters.get(0, featureVectorsSize * featureVectorsSize - 1));
            networkState.set("W_phi_in", parameters.get(featureVectorsSize * featureVectorsSize, 2 * featureVectorsSize * featureVectorsSize - 1));
            networkState.set("W_phi_out", parameters.get(2 * featureVectorsSize * featureVectorsSize, 3 * featureVectorsSize * featureVectorsSize - 1));
            networkState.set("b", parameters.get(3 * featureVectorsSize * featureVectorsSize, 3 * featureVectorsSize * featureVectorsSize + featureVectorsSize - 1));
            List<Matrix> networkGradients = network.gradient(networkState, Lists.newArrayList(
                    Variables.get("W_phi"),
                    Variables.get("W_phi_in"),
                    Variables.get("W_phi_out"),
                    Variables.get("b")
            ));
            Matrix networkGradient = Matrix.zeros(featureVectorsSize, parametersVectorSize);
            networkGradient.setSubMatrix(0, featureVectorsSize - 1,
                                         0, featureVectorsSize * featureVectorsSize - 1,
                                         networkGradients.get(0));
            networkGradient.setSubMatrix(0, featureVectorsSize - 1,
                                         featureVectorsSize * featureVectorsSize, 2 * featureVectorsSize * featureVectorsSize - 1,
                                         networkGradients.get(1));
            networkGradient.setSubMatrix(0, featureVectorsSize - 1,
                                         2 * featureVectorsSize * featureVectorsSize, 3 * featureVectorsSize * featureVectorsSize - 1,
                                         networkGradients.get(2));
            networkGradient.setSubMatrix(0, featureVectorsSize - 1,
                                         3 * featureVectorsSize * featureVectorsSize, 3 * featureVectorsSize * featureVectorsSize + featureVectorsSize - 1,
                                         networkGradients.get(3));
            return gradient;
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
        private final int featureVectorsSize;
        private final int outputVectorSize;
        private final int parametersVectorSize;

        public InnerProductOutputFunction(int featureVectorsSize, int outputVectorSize) {
            this.featureVectorsSize = featureVectorsSize;
            this.outputVectorSize = outputVectorSize;
            this.parametersVectorSize = featureVectorsSize * outputVectorSize;
        }

        public int getParametersVectorSize() {
            return parametersVectorSize;
        }

        public Vector getInitialParametersVector() {
            Vector initialParametersVector = Vectors.dense(parametersVectorSize);
            int parameterVectorIndex = 0;
            for (int i = 0; i < outputVectorSize; i++)
                for (int j = 0; j < featureVectorsSize; j++)
                    initialParametersVector.set(parameterVectorIndex++, 1 / featureVectorsSize);
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
