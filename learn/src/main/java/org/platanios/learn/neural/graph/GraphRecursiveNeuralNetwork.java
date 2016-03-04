package org.platanios.learn.neural.graph;

import org.platanios.learn.graph.Edge;
import org.platanios.learn.graph.Graph;
import org.platanios.learn.graph.Vertex;
import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;
import org.platanios.learn.optimization.QuasiNewtonSolver;
import org.platanios.learn.optimization.function.AbstractFunction;
import org.platanios.learn.optimization.function.DerivativesApproximation;
import org.platanios.learn.optimization.function.NonSmoothFunctionException;
import org.platanios.learn.optimization.linesearch.BacktrackingLineSearch;
import org.platanios.utilities.ArrayUtilities;

import java.util.Map;

/**
 * @author Emmanouil Antonios Platanios
 */
public class GraphRecursiveNeuralNetwork<E> {
    protected final int featureVectorsSize;
    protected final int outputVectorSize;
    protected final int maximumNumberOfSteps;
    protected final Graph<VertexContentType, E> graph;
    protected final Map<Integer, Vector> trainingData;
    protected final ObjectiveFunction objectiveFunction;
    protected final FeatureVectorFunction<VertexContentType, E> featureVectorFunction;
    protected final OutputFunction outputFunction;
    protected final LossFunction lossFunction;

    private boolean needsForwardPass = true;
    private boolean needsBackwardPass = true;

    protected Vector featureVectorFunctionParameters;
    protected Vector outputFunctionParameters;

    private double lossFunctionValue;
    private Vector featureVectorFunctionParametersGradient;
    private Vector outputFunctionParametersGradient;

    public GraphRecursiveNeuralNetwork(int featureVectorsSize,
                                       int outputVectorSize,
                                       int maximumNumberOfSteps,
                                       Graph<VertexContentType, E> graph,
                                       Map<Integer, Vector> trainingData,
                                       FeatureVectorFunction<VertexContentType, E> featureVectorFunction,
                                       OutputFunction outputFunction,
                                       LossFunction lossFunction) {
        this.featureVectorsSize = featureVectorsSize;
        this.outputVectorSize = outputVectorSize;
        this.maximumNumberOfSteps = maximumNumberOfSteps;
        this.graph = graph;
        this.trainingData = trainingData;
        this.featureVectorFunction = featureVectorFunction;
        this.outputFunction = outputFunction;
        this.lossFunction = lossFunction;
        this.featureVectorFunctionParameters = featureVectorFunction.getInitialParametersVector();
        this.outputFunctionParameters = outputFunction.getInitialParametersVector();
        objectiveFunction = new ObjectiveFunction();
        resetGraph();
    }

    public Graph<VertexContentType, E> getGraph() {
        return graph;
    }

    public Vector getOutputForVertex(Vertex<GraphRecursiveNeuralNetwork.VertexContentType, E> vertex) {
        return outputFunction.value(vertex.getContent().getFeatureVector(), outputFunctionParameters);
    }

    public Vector getFeatureVectorFunctionParameters() {
        return featureVectorFunctionParameters;
    }

    public void setFeatureVectorFunctionParameters(Vector featureVectorFunctionParameters) {
        this.featureVectorFunctionParameters = featureVectorFunctionParameters;
        if (!needsForwardPass || !needsBackwardPass)
            resetGraph();
    }

    public Vector getOutputFunctionParameters() {
        return outputFunctionParameters;
    }

    public void setOutputFunctionParameters(Vector outputFunctionParameters) {
        this.outputFunctionParameters = outputFunctionParameters;
        if (!needsForwardPass || !needsBackwardPass)
            resetGraph();
    }

    public double lossFunctionValue() {
        if (needsForwardPass)
            performForwardPass();
        return lossFunctionValue;
    }

    public Vector getFeatureVectorFunctionParametersGradient() {
        if (needsForwardPass)
            performForwardPass();
        if (needsBackwardPass)
            performBackwardPass();
        return featureVectorFunctionParametersGradient;
    }

    public Vector getOutputFunctionParametersGradient() {
        if (needsForwardPass)
            performForwardPass();
        if (needsBackwardPass)
            performBackwardPass();
        return outputFunctionParametersGradient;
    }

    public void resetGraph() {
        graph.computeVerticesUpdatedContent(this::resetVertexComputeFunction);
        graph.updateVerticesContent();
        graph.computeVerticesUpdatedContent(this::neighborsSumVertexComputeFunction);
        needsForwardPass = true;
        needsBackwardPass = true;
    }

    private VertexContentType resetVertexComputeFunction(Vertex<VertexContentType, E> vertex) {
        Vector[] featureVectors = new Vector[maximumNumberOfSteps + 1];
        featureVectors[0] = Vectors.dense(featureVectorsSize);  // TODO: Change the feature vectors initial computeValue.
        return new VertexContentType(vertex.getContent().id, 0, featureVectors, null);
    }

//    public void randomizeGraph() {
//        graph.computeVerticesUpdatedContent(this::randomizeVertexComputeFunction);
//        graph.updateVerticesContent();
//        needsForwardPass = true;
//        needsBackwardPass = true;
//    }
//
//    private VertexContentType randomizeVertexComputeFunction(Vertex<VertexContentType, E> vertex) {
//        Vector[] featureVectors = new Vector[maximumNumberOfSteps + 1];
//        featureVectors[0] = Vectors.random(featureVectorsSize);  // TODO: Change the feature vectors initial computeValue.
//        return new VertexContentType(vertex.getContent().id, 0, featureVectors, null);
//    }

    public void performForwardPass() {
        lossFunctionValue = 0;
        for (int step = 0; step < maximumNumberOfSteps; step++) {
            graph.computeVerticesUpdatedContent(this::forwardVertexComputeFunction);
            graph.updateVerticesContent();
            graph.computeVerticesUpdatedContent(this::neighborsSumVertexComputeFunction);
        }
        needsForwardPass = false;
    }

    private VertexContentType neighborsSumVertexComputeFunction(Vertex<VertexContentType, E> vertex) {
        Vector incomingFeatureVectorsSum = Vectors.dense(featureVectorsSize);
        for (Edge<GraphRecursiveNeuralNetwork.VertexContentType, E> incomingEdge : vertex.getIncomingEdges()) {
            GraphRecursiveNeuralNetwork.VertexContentType vertexContent = incomingEdge.getSourceVertex().getContent();
            incomingFeatureVectorsSum.addInPlace(vertexContent.featureVectors[vertexContent.currentStep]);
        }
        Vector outgoingFeatureVectorsSum = Vectors.dense(featureVectorsSize);
        for (Edge<GraphRecursiveNeuralNetwork.VertexContentType, E> outgoingEdge : vertex.getOutgoingEdges()) {
            GraphRecursiveNeuralNetwork.VertexContentType vertexContent = outgoingEdge.getDestinationVertex().getContent();
            outgoingFeatureVectorsSum.addInPlace(vertexContent.featureVectors[vertexContent.currentStep]);
        }
        vertex.getContent().setIncomingFeatureVectorsSum(incomingFeatureVectorsSum, vertex.getContent().currentStep);
        vertex.getContent().setOutgoingFeatureVectorsSum(outgoingFeatureVectorsSum, vertex.getContent().currentStep);
        return null;
    }

    private VertexContentType forwardVertexComputeFunction(Vertex<VertexContentType, E> vertex) {
        VertexContentType vertexContent = vertex.getContent();
        vertexContent.featureVectors[vertexContent.currentStep + 1] =
                featureVectorFunction.value(featureVectorFunctionParameters, vertex, vertexContent.currentStep);
        if (vertexContent.currentStep == maximumNumberOfSteps - 1) {
            Vector correctOutput = trainingData.getOrDefault(vertexContent.id, null);
            if (correctOutput != null)
                synchronized (this) {
                    lossFunctionValue += lossFunction.value(
                            outputFunction.value(vertex.getContent().featureVectors[vertexContent.currentStep + 1],
                                                 outputFunctionParameters),
                            correctOutput
                    );
                }
        }
        return new VertexContentType(
                vertexContent.id,
                vertexContent.currentStep + 1,
                vertexContent.featureVectors,
                vertexContent.incomingFeatureVectorsSum,
                vertexContent.outgoingFeatureVectorsSum,
                null
        );
    }

    public void performBackwardPass() {
        outputFunctionParametersGradient = Vectors.dense(outputFunctionParameters.size());
        featureVectorFunctionParametersGradient = Vectors.dense(featureVectorFunctionParameters.size());
        for (int step = maximumNumberOfSteps; step > 0; step--) {
            graph.computeVerticesUpdatedContent(this::backwardVertexComputeFunction);
            graph.updateVerticesContent();
        }
        needsBackwardPass = false;
    }

    private VertexContentType backwardVertexComputeFunction(Vertex<VertexContentType, E> vertex) {
        VertexContentType vertexContent = vertex.getContent();
        Vector featureVectorGradient;
        if (vertexContent.currentStep == maximumNumberOfSteps) {
            Vector lossFunctionGradient = Vectors.dense(outputVectorSize);
            Vector correctOutput = trainingData.getOrDefault(vertexContent.id, null);
            if (correctOutput != null)
                lossFunctionGradient = lossFunction.gradient(
                        outputFunction.value(vertexContent.featureVectors[vertexContent.currentStep],
                                             outputFunctionParameters),
                        correctOutput
                );
            synchronized (this) {
                outputFunctionParametersGradient.addInPlace(lossFunctionGradient.transMult(
                        outputFunction.parametersGradient(vertexContent.featureVectors[vertexContent.currentStep],
                                                          outputFunctionParameters)
                ));
            }
            featureVectorGradient = lossFunctionGradient.transMult(
                    outputFunction.featureVectorGradient(vertexContent.featureVectors[vertexContent.currentStep],
                                                         outputFunctionParameters)
            );
        } else {
            featureVectorGradient = vertexContent.featureVectorGradient.transMult(
                    featureVectorFunction.featureVectorGradient(featureVectorFunctionParameters,
                                                                vertex,
                                                                vertex,
                                                                vertexContent.currentStep)
            );
            for (Edge<VertexContentType, E> incomingEdge : vertex.getIncomingEdges())
                featureVectorGradient.addInPlace(
                        incomingEdge.getSourceVertex().getContent().featureVectorGradient.transMult(
                                featureVectorFunction.featureVectorGradient(featureVectorFunctionParameters,
                                                                            incomingEdge.getSourceVertex(),
                                                                            vertex,
                                                                            vertexContent.currentStep)
                        )
                );
            for (Edge<VertexContentType, E> outgoingEdge : vertex.getOutgoingEdges())
                featureVectorGradient.addInPlace(
                        outgoingEdge.getDestinationVertex().getContent().featureVectorGradient.transMult(
                                featureVectorFunction.featureVectorGradient(featureVectorFunctionParameters,
                                                                            outgoingEdge.getDestinationVertex(),
                                                                            vertex,
                                                                            vertexContent.currentStep)
                        )
                );
        }
        synchronized (this) {
            featureVectorFunctionParametersGradient.addInPlace(featureVectorGradient.transMult(
                    featureVectorFunction.gradient(featureVectorFunctionParameters, vertex, vertexContent.currentStep - 1)
            ));
        }
        return new VertexContentType(
                vertexContent.id,
                vertexContent.currentStep - 1,
                vertexContent.featureVectors,
                vertexContent.incomingFeatureVectorsSum,
                vertexContent.outgoingFeatureVectorsSum,
                featureVectorGradient
        );
    }

    public boolean checkDerivative(double tolerance) {
        try {
            resetGraph();
            DerivativesApproximation derivativesApproximation =
                    new DerivativesApproximation(objectiveFunction, DerivativesApproximation.Method.CENTRAL_DIFFERENCE);
            Vector point = Vectors.random(outputFunction.getParametersVectorSize() + featureVectorFunction.getParametersVectorSize());
            double[] actualResult = derivativesApproximation.approximateGradient(point).getDenseArray();
            double[] expectedResult = objectiveFunction.getGradient(point).getDenseArray();
            resetGraph();
            return ArrayUtilities.equals(actualResult, expectedResult, tolerance);
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
//                        .lineSearch(lineSearch)
//                        .gradientTolerance(1e-10)
                        .checkForObjectiveConvergence(true)
                        .objectiveChangeTolerance(1e-6)
                        .checkForGradientConvergence(true)
                        .gradientTolerance(1e-6)
                        .maximumNumberOfIterations(1000)
                        .maximumNumberOfFunctionEvaluations(10000)
                        .loggingLevel(5)
                        .build();
//        NonlinearConjugateGradientSolver solver =
//                new NonlinearConjugateGradientSolver.Builder(objectiveFunction, initialPoint)
//                        .method(NonlinearConjugateGradientSolver.Method.FLETCHER_RIEVES_POLAK_RIBIERE)
//                        .restartMethod(NonlinearConjugateGradientSolver.RestartMethod.GRADIENTS_ORTHOGONALITY_CHECK)
//                        .loggingLevel(5)
//                        .build();
//        GradientDescentSolver solver =
//                new GradientDescentSolver.Builder(objectiveFunction, initialPoint)
//                        .lineSearch(new NoLineSearch(10, 0.75))
//                        .gradientTolerance(1e-10)
//                        .loggingLevel(5)
//                        .build();
//        RPropSolver solver =
//                new RPropSolver.Builder(objectiveFunction, initialPoint)
//                        .lineSearch(new NoLineSearch(10, 0.75))
//                        .checkForPointConvergence(true)
//                        .checkForObjectiveConvergence(true)
//                        .loggingLevel(5)
//                        .build();
        Vector solution = solver.solve();
        outputFunctionParameters = solution.get(0, outputFunction.getParametersVectorSize() - 1);
        featureVectorFunctionParameters = solution.get(outputFunction.getParametersVectorSize(), solution.size() - 1);
        setOutputFunctionParameters(outputFunctionParameters);
        setFeatureVectorFunctionParameters(featureVectorFunctionParameters);
    }

    public static class VertexContentType {
        protected int id;
        protected int currentStep;                // k
        protected Vector[] featureVectors;        // φ(1),...,φ(k)
        protected Vector[] incomingFeatureVectorsSum;
        protected Vector[] outgoingFeatureVectorsSum;
        protected Vector featureVectorGradient;   // dL / dφ(k)

        public VertexContentType(int id,
                                 int currentStep,
                                 Vector[] featureVectors,
                                 Vector featureVectorGradient) {
            this.id = id;
            this.currentStep = currentStep;
            this.featureVectors = featureVectors;
            if (featureVectors != null) {
                incomingFeatureVectorsSum = new Vector[featureVectors.length];
                outgoingFeatureVectorsSum = new Vector[featureVectors.length];
            } else {
                incomingFeatureVectorsSum = null;
                outgoingFeatureVectorsSum = null;
            }
            this.featureVectorGradient = featureVectorGradient;
        }

        public VertexContentType(int id,
                                 int currentStep,
                                 Vector[] featureVectors,
                                 Vector[] incomingFeatureVectorsSum,
                                 Vector[] outgoingFeatureVectorsSum,
                                 Vector featureVectorGradient) {
            this.id = id;
            this.currentStep = currentStep;
            this.featureVectors = featureVectors;
            this.incomingFeatureVectorsSum = incomingFeatureVectorsSum;
            this.outgoingFeatureVectorsSum = outgoingFeatureVectorsSum;
            this.featureVectorGradient = featureVectorGradient;
        }

        public VertexContentType setIncomingFeatureVectorsSum(Vector incomingFeatureVectorSum, int step) {
            this.incomingFeatureVectorsSum[step] = incomingFeatureVectorSum;
            return this;
        }

        public VertexContentType setOutgoingFeatureVectorsSum(Vector outgoingFeatureVectorSum, int step) {
            this.outgoingFeatureVectorsSum[step] = outgoingFeatureVectorSum;
            return this;
        }

        public int getId() {
            return id;
        }

        public Vector getFeatureVector() {
            return featureVectors[currentStep];
        }

        public Vector[] getIncomingFeatureVectorsSum() {
            return incomingFeatureVectorsSum;
        }

        public Vector[] getOutgoingFeatureVectorsSum() {
            return outgoingFeatureVectorsSum;
        }

        public Vector getIncomingFeatureVectorsSum(int step) {
            return incomingFeatureVectorsSum[step];
        }

        public Vector getOutgoingFeatureVectorsSum(int step) {
            return outgoingFeatureVectorsSum[step];
        }
    }

    private class ObjectiveFunction extends AbstractFunction {
        private Vector oldPoint = Vectors.dense(0);

        @Override
        protected double computeValue(Vector point) {
            checkPoint(point);
            return lossFunctionValue();
        }

        @Override
        protected Vector computeGradient(Vector point) {
            checkPoint(point);
            Vector gradient = Vectors.dense(outputFunction.getParametersVectorSize() + featureVectorFunction.getParametersVectorSize());
            gradient.set(0, outputFunction.getParametersVectorSize() - 1, getOutputFunctionParametersGradient());
            gradient.set(outputFunction.getParametersVectorSize(), point.size() - 1, getFeatureVectorFunctionParametersGradient());
            return gradient;
        }

        private void checkPoint(Vector point) {
            if (!ArrayUtilities.equals(oldPoint.getDenseArray(), point.getDenseArray(), 1e-10)) {
                setOutputFunctionParameters(point.get(0, outputFunction.getParametersVectorSize() - 1));
                setFeatureVectorFunctionParameters(point.get(outputFunction.getParametersVectorSize(), point.size() - 1));
                oldPoint = point;
            }
        }
    }

    public abstract static class FeatureVectorFunction<VertexContentType, E> {
        public abstract int getParametersVectorSize();
        public abstract Vector getInitialParametersVector();
        public abstract Vector value(Vector parameters, Vertex<VertexContentType, E> vertex, int step);
        public abstract Matrix gradient(Vector parameters, Vertex<VertexContentType, E> vertex, int step);
        public abstract Matrix featureVectorGradient(Vector parameters,
                                                     Vertex<VertexContentType, E> vertex,
                                                     Vertex<VertexContentType, E> differentiatingVertex,
                                                     int step);
    }

    public abstract static class OutputFunction {
        public abstract int getParametersVectorSize();
        public abstract Vector getInitialParametersVector();
        public abstract Vector value(Vector featureVector, Vector parameters);
        public abstract Matrix featureVectorGradient(Vector featureVector, Vector parameters);
        public abstract Matrix parametersGradient(Vector featureVector, Vector parameters);
    }

    public abstract static class LossFunction {
        public abstract double value(Vector networkOutput, Vector correctOutput);
        public abstract Vector gradient(Vector networkOutput, Vector correctOutput);
    }
}
