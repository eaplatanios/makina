package makina.learn.neural.graph;

import makina.learn.graph.Graph;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import makina.learn.graph.Edge;
import makina.learn.graph.Vertex;
import makina.math.matrix.Matrix;
import makina.math.matrix.Vector;
import makina.math.matrix.Vectors;
import makina.optimization.QuasiNewtonSolver;
import makina.optimization.function.AbstractFunction;
import makina.optimization.function.DerivativesApproximation;
import makina.optimization.function.NonSmoothFunctionException;
import makina.optimization.linesearch.BacktrackingLineSearch;
import makina.utilities.ArrayUtilities;

import java.util.Map;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DeepGraph<E> {
    private final Logger logger = LogManager.getFormatterLogger("Deep Graph");

    protected final int featuresSize;
    protected final int outputsSize;
    protected final int maximumNumberOfSteps;
    protected final Graph<VertexContent, E> graph;
    protected final ObjectiveFunction objectiveFunction;
    protected final UpdateFunction<VertexContent, E> updateFunction;
    protected final OutputFunction outputFunction;
    protected final LossFunction lossFunction;

    private boolean needsForwardPass = true;
    private boolean needsBackwardPass = true;
    private boolean training = false;

    protected Map<Integer, Vector> trainingData;
    protected Vector updateFunctionParameters;
    protected Vector outputFunctionParameters;

    private double loss;
    private Vector updateFunctionParametersGradient;
    private Vector outputFunctionParametersGradient;

    public DeepGraph(int featuresSize,
                     int outputsSize,
                     int maximumNumberOfSteps,
                     Graph<VertexContent, E> graph,
                     UpdateFunction<VertexContent, E> updateFunction,
                     OutputFunction outputFunction,
                     LossFunction lossFunction) {
        this.featuresSize = featuresSize;
        this.outputsSize = outputsSize;
        this.maximumNumberOfSteps = maximumNumberOfSteps;
        this.graph = graph;
        this.updateFunction = updateFunction;
        this.outputFunction = outputFunction;
        this.lossFunction = lossFunction;
        this.updateFunctionParameters = updateFunction.initialParameters();
        this.outputFunctionParameters = outputFunction.initialParameters();
        objectiveFunction = new ObjectiveFunction();
        resetGraph();
    }

    public Graph<VertexContent, E> graph() {
        return graph;
    }

    public Vector output(Vertex<VertexContent, E> vertex) {
        forwardPass();
        return outputFunction.value(vertex.content().features(), outputFunctionParameters);
    }

    public Vector updateFunctionParameters() {
        return updateFunctionParameters;
    }

    public void updateFunctionParameters(Vector updateFunctionParameters) {
        this.updateFunctionParameters = updateFunctionParameters;
        if (!needsForwardPass || !needsBackwardPass)
            resetGraph();
    }

    public Vector outputFunctionParameters() {
        return outputFunctionParameters;
    }

    public void outputFunctionParameters(Vector outputFunctionParameters) {
        this.outputFunctionParameters = outputFunctionParameters;
        if (!needsForwardPass || !needsBackwardPass)
            resetGraph();
    }

    private double lossFunctionValue() {
        forwardPass();
        return loss;
    }

    private Vector updateFunctionParametersGradient() {
        forwardPass();
        backwardPass();
        return updateFunctionParametersGradient;
    }

    private Vector outputFunctionParametersGradient() {
        forwardPass();
        backwardPass();
        return outputFunctionParametersGradient;
    }

    public void resetGraph() {
        graph.computeVerticesUpdatedContent(this::resetVertexUpdateFunction);
        graph.updateVerticesContent();
        graph.computeVerticesUpdatedContent(this::neighborsSumVertexUpdateFunction);
        needsForwardPass = true;
        needsBackwardPass = true;
    }

    private VertexContent resetVertexUpdateFunction(Vertex<VertexContent, E> vertex) {
        Vector[] featureVectors = new Vector[maximumNumberOfSteps + 1];
        featureVectors[0] = Vectors.dense(featuresSize);  // TODO: Change the feature vectors initial computeValue.
        return new VertexContent(vertex.content().id, 0, featureVectors, null);
    }

//    public void randomizeGraph() {
//        graph.updateVerticesContent(this::randomizeVertexComputeFunction);
//        graph.updateVerticesContent();
//        needsForwardPass = true;
//        needsBackwardPass = true;
//    }
//
//    private VertexContentType randomizeVertexComputeFunction(Vertex<VertexContentType, E> vertex) {
//        Vector[] features = new Vector[maximumNumberOfSteps + 1];
//        features[0] = Vectors.random(featuresSize);  // TODO: Change the feature vectors initial computeValue.
//        return new VertexContentType(vertex.content().id, 0, features, null);
//    }

    private void forwardPass() {
        if (!needsForwardPass)
            return;
        loss = 0;
        for (int step = 0; step < maximumNumberOfSteps; step++) {
            graph.computeVerticesUpdatedContent(this::forwardVertexUpdateFunction);
            graph.updateVerticesContent();
            graph.computeVerticesUpdatedContent(this::neighborsSumVertexUpdateFunction);
        }
        needsForwardPass = false;
    }

    private VertexContent neighborsSumVertexUpdateFunction(Vertex<VertexContent, E> vertex) {
//        double samplingProbability = 0.1 * Math.sqrt(1.0 / vertex.incomingEdges().size());
        Vector incomingFeatureVectorsSum = Vectors.dense(featuresSize);
        for (Edge<VertexContent, E> incomingEdge : vertex.incomingEdges()) {
//            if (training && Math.random() > samplingProbability)
//                continue;
            VertexContent vertexContent = incomingEdge.sourceVertex().content();
            incomingFeatureVectorsSum.addInPlace(vertexContent.features[vertexContent.currentStep]);
        }
//        samplingProbability = 0.1 * Math.sqrt(1.0 / vertex.outgoingEdges().size());
        Vector outgoingFeatureVectorsSum = Vectors.dense(featuresSize);
        for (Edge<VertexContent, E> outgoingEdge : vertex.outgoingEdges()) {
//            if (training && Math.random() > samplingProbability)
//                continue;
            VertexContent vertexContent = outgoingEdge.destinationVertex().content();
            outgoingFeatureVectorsSum.addInPlace(vertexContent.features[vertexContent.currentStep]);
        }
        vertex.content().incomingFeaturesSum(incomingFeatureVectorsSum, vertex.content().currentStep);
        vertex.content().outgoingFeaturesSum(outgoingFeatureVectorsSum, vertex.content().currentStep);
        return null;
    }

    private VertexContent forwardVertexUpdateFunction(Vertex<VertexContent, E> vertex) {
        VertexContent vertexContent = vertex.content();
        vertexContent.features[vertexContent.currentStep + 1] =
                updateFunction.value(updateFunctionParameters, vertex, vertexContent.currentStep);
        if (vertexContent.currentStep == maximumNumberOfSteps - 1 && training) {
            Vector observedOutput = trainingData.getOrDefault(vertexContent.id, null);
            if (observedOutput != null)
                synchronized (this) {
                    loss += lossFunction.value(
                            outputFunction.value(vertex.content().features[vertexContent.currentStep + 1],
                                                 outputFunctionParameters),
                            observedOutput
                    );
                }
        }
        return new VertexContent(
                vertexContent.id,
                vertexContent.currentStep + 1,
                vertexContent.features,
                vertexContent.incomingFeaturesSum,
                vertexContent.outgoingFeaturesSum,
                null
        );
    }

    private void backwardPass() {
        if (!needsBackwardPass)
            return;
        outputFunctionParametersGradient = Vectors.dense(outputFunctionParameters.size());
        updateFunctionParametersGradient = Vectors.dense(updateFunctionParameters.size());
        for (int step = maximumNumberOfSteps; step > 0; step--) {
            graph.computeVerticesUpdatedContent(this::backwardVertexUpdateFunction);
            graph.updateVerticesContent();
        }
        needsBackwardPass = false;
    }

    private VertexContent backwardVertexUpdateFunction(Vertex<VertexContent, E> vertex) {
        VertexContent content = vertex.content();
        Vector featureVectorGradient;
        if (content.currentStep == maximumNumberOfSteps) {
            Vector lossFunctionGradient = Vectors.dense(outputsSize);
            Vector observedOutput = trainingData.getOrDefault(content.id, null);
            if (observedOutput != null)
                lossFunctionGradient = lossFunction.gradient(
                        outputFunction.value(content.features[content.currentStep], outputFunctionParameters),
                        observedOutput
                );
            synchronized (this) {
                outputFunctionParametersGradient.addInPlace(lossFunctionGradient.transMult(
                        outputFunction.parametersGradient(content.features[content.currentStep], outputFunctionParameters)
                ));
            }
            featureVectorGradient = lossFunctionGradient.transMult(outputFunction.featuresGradient(content.features[content.currentStep], outputFunctionParameters));
        } else {
            featureVectorGradient = content.featuresGradient.transMult(
                    updateFunction.featuresGradient(updateFunctionParameters, vertex, vertex, content.currentStep)
            );
            for (Edge<VertexContent, E> incomingEdge : vertex.incomingEdges())
                featureVectorGradient.addInPlace(
                        incomingEdge.sourceVertex().content().featuresGradient.transMult(
                                updateFunction.featuresGradient(updateFunctionParameters,
                                                                incomingEdge.sourceVertex(),
                                                                vertex,
                                                                content.currentStep)
                        )
                );
            for (Edge<VertexContent, E> outgoingEdge : vertex.outgoingEdges())
                featureVectorGradient.addInPlace(
                        outgoingEdge.destinationVertex().content().featuresGradient.transMult(
                                updateFunction.featuresGradient(updateFunctionParameters,
                                                                outgoingEdge.destinationVertex(),
                                                                vertex,
                                                                content.currentStep)
                        )
                );
        }
        synchronized (this) {
            updateFunctionParametersGradient.addInPlace(featureVectorGradient.transMult(
                    updateFunction.gradient(updateFunctionParameters, vertex, content.currentStep - 1)
            ));
        }
        return new VertexContent(
                content.id,
                content.currentStep - 1,
                content.features,
                content.incomingFeaturesSum,
                content.outgoingFeaturesSum,
                featureVectorGradient
        );
    }

    private boolean checkGradient(double tolerance) {
        try {
            resetGraph();
            DerivativesApproximation derivativesApproximation =
                    new DerivativesApproximation(objectiveFunction, DerivativesApproximation.Method.CENTRAL_DIFFERENCE);
            Vector point = Vectors.random(outputFunction.parametersSize() + updateFunction.parametersSize());
            double[] actualResult = derivativesApproximation.approximateGradient(point).getDenseArray();
            double[] expectedResult = objectiveFunction.getGradient(point).getDenseArray();
            resetGraph();
            return ArrayUtilities.equals(actualResult, expectedResult, tolerance);
        } catch (NonSmoothFunctionException e) {
            return false;
        }
    }

    public void train(Map<Integer, Vector> trainingData) {
        this.trainingData = trainingData;
        needsForwardPass = true;
        needsBackwardPass = true;
        training = true;
        if (!checkGradient(1e-5))
            logger.warn("The gradient is not the same as the one obtained by the method of finite differences.");
        Vector initialPoint = Vectors.dense(outputFunction.parametersSize() + updateFunction.parametersSize());
        for (Vector.Element element : outputFunction.initialParameters())
            initialPoint.set(element.index(), element.value());
        for (Vector.Element element : updateFunction.initialParameters())
            initialPoint.set(element.index() + outputFunction.parametersSize(), element.value());
        BacktrackingLineSearch lineSearch = new BacktrackingLineSearch(objectiveFunction, 0.5, 0.5);
        lineSearch.setInitialStepSize(1.0);
        QuasiNewtonSolver solver =
                new QuasiNewtonSolver.Builder(objectiveFunction, initialPoint)
//                        .method(QuasiNewtonSolver.Method.BROYDEN_FLETCHER_GOLDFARB_SHANNO)
                        .method(QuasiNewtonSolver.Method.LIMITED_MEMORY_BROYDEN_FLETCHER_GOLDFARB_SHANNO)
                        .m(10)
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
////                        .lineSearch(new NoLineSearch(10, 0.75))
//                        .lineSearch(lineSearch)
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
        outputFunctionParameters = solution.get(0, outputFunction.parametersSize() - 1);
        updateFunctionParameters = solution.get(outputFunction.parametersSize(), solution.size() - 1);
        outputFunctionParameters(outputFunctionParameters);
        updateFunctionParameters(updateFunctionParameters);
        training = false;
    }

    public static class VertexContent {
        protected int id;
        protected Vector[] features;        // φ(1),...,φ(k)
        int currentStep;                    // k
        Vector[] incomingFeaturesSum;
        Vector[] outgoingFeaturesSum;
        Vector featuresGradient;            // dL / dφ(k)

        public VertexContent(int id, int currentStep, Vector[] features, Vector featuresGradient) {
            this.id = id;
            this.currentStep = currentStep;
            this.features = features;
            if (features != null) {
                incomingFeaturesSum = new Vector[features.length];
                outgoingFeaturesSum = new Vector[features.length];
            } else {
                incomingFeaturesSum = null;
                outgoingFeaturesSum = null;
            }
            this.featuresGradient = featuresGradient;
        }

        public VertexContent(int id,
                             int currentStep,
                             Vector[] features,
                             Vector[] incomingFeaturesSum,
                             Vector[] outgoingFeaturesSum,
                             Vector featuresGradient) {
            this.id = id;
            this.currentStep = currentStep;
            this.features = features;
            this.incomingFeaturesSum = incomingFeaturesSum;
            this.outgoingFeaturesSum = outgoingFeaturesSum;
            this.featuresGradient = featuresGradient;
        }

        VertexContent incomingFeaturesSum(Vector incomingFeaturesSum, int step) {
            this.incomingFeaturesSum[step] = incomingFeaturesSum;
            return this;
        }

        VertexContent outgoingFeaturesSum(Vector outgoingFeaturesSum, int step) {
            this.outgoingFeaturesSum[step] = outgoingFeaturesSum;
            return this;
        }

        public int id() {
            return id;
        }

        public Vector features() {
            return features[currentStep];
        }

        Vector[] incomingFeaturesSum() {
            return incomingFeaturesSum;
        }

        Vector[] outgoingFeaturesSum() {
            return outgoingFeaturesSum;
        }

        Vector incomingFeaturesSum(int step) {
            return incomingFeaturesSum[step];
        }

        Vector outgoingFeaturesSum(int step) {
            return outgoingFeaturesSum[step];
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
            Vector gradient = Vectors.dense(outputFunction.parametersSize() + updateFunction.parametersSize());
            gradient.set(0, outputFunction.parametersSize() - 1, outputFunctionParametersGradient());
            gradient.set(outputFunction.parametersSize(), point.size() - 1, updateFunctionParametersGradient());
            return gradient;
        }

        private void checkPoint(Vector point) {
            if (!ArrayUtilities.equals(oldPoint.getDenseArray(), point.getDenseArray(), 1e-10)) {
                outputFunctionParameters(point.get(0, outputFunction.parametersSize() - 1));
                updateFunctionParameters(point.get(outputFunction.parametersSize(), point.size() - 1));
                oldPoint = point;
            }
        }
    }

    public abstract static class UpdateFunction<VertexContentType, E> {
        public abstract int parametersSize();
        public abstract Vector initialParameters();
        public abstract Vector value(Vector parameters, Vertex<VertexContentType, E> vertex, int step);
        public abstract Matrix gradient(Vector parameters, Vertex<VertexContentType, E> vertex, int step);
        public abstract Matrix featuresGradient(Vector parameters,
                                                Vertex<VertexContentType, E> vertex,
                                                Vertex<VertexContentType, E> differentiatingVertex,
                                                int step);
    }

    public abstract static class OutputFunction {
        public abstract int parametersSize();
        public abstract Vector initialParameters();
        public abstract Vector value(Vector features, Vector parameters);
        public abstract Matrix featuresGradient(Vector features, Vector parameters);
        public abstract Matrix parametersGradient(Vector features, Vector parameters);
    }

    public abstract static class LossFunction {
        public abstract double value(Vector networkOutput, Vector observedOutput);
        public abstract Vector gradient(Vector networkOutput, Vector observedOutput);
    }
}
