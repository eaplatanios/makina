package org.platanios.learn.neural.graph;

import org.platanios.learn.graph.Edge;
import org.platanios.learn.graph.Graph;
import org.platanios.learn.graph.Vertex;
import org.platanios.learn.math.matrix.Matrix;
import org.platanios.learn.math.matrix.Vector;
import org.platanios.learn.math.matrix.Vectors;

import java.util.Map;

/**
 * @author Emmanouil Antonios Platanios
 */
public class GraphRecursiveNeuralNetwork<E> {
    private final int featureVectorsSize;
    private final int outputVectorSize;
    private final int maximumNumberOfSteps;
    private final Graph<VertexContentType, E> graph;
    private final Map<Integer, Vector> trainingData;
    private final FeatureVectorFunction<VertexContentType, E> featureVectorFunction;
    private final OutputFunction outputFunction;
    private final LossFunction lossFunction;

    private boolean needsForwardPass = true;
    private boolean needsBackwardPass = true;

    private Vector featureVectorFunctionParameters;
    private Vector outputFunctionParameters;

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
                                       LossFunction lossFunction,
                                       Vector featureVectorFunctionParameters,
                                       Vector outputFunctionParameters) {
        this.featureVectorsSize = featureVectorsSize;
        this.outputVectorSize = outputVectorSize;
        this.maximumNumberOfSteps = maximumNumberOfSteps;
        this.graph = graph;
        this.trainingData = trainingData;
        this.featureVectorFunction = featureVectorFunction;
        this.outputFunction = outputFunction;
        this.lossFunction = lossFunction;
        this.featureVectorFunctionParameters = featureVectorFunctionParameters;
        this.outputFunctionParameters = outputFunctionParameters;
        resetGraph();
    }

    public Graph<VertexContentType, E> getGraph() {
        return graph;
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

    public double getLossFunctionValue() {
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
        needsForwardPass = true;
        needsBackwardPass = true;
    }

    private VertexContentType resetVertexComputeFunction(Vertex<VertexContentType, E> vertex) {
        Vector[] featureVectors = new Vector[maximumNumberOfSteps];
        featureVectors[0] = Vectors.dense(featureVectorsSize);  // TODO: Change the feature vectors initial value.
        return new VertexContentType(vertex.getContent().id, 0, featureVectors, null);
    }

    public void performForwardPass() {
        lossFunctionValue = 0;
        for (int step = 0; step < maximumNumberOfSteps - 1; step++) {
            graph.computeVerticesUpdatedContent(this::forwardVertexComputeFunction);
            graph.updateVerticesContent();
        }
        needsForwardPass = false;
    }

    private VertexContentType forwardVertexComputeFunction(Vertex<VertexContentType, E> vertex) {
        VertexContentType vertexContent = vertex.getContent();
        vertexContent.featureVectors[vertexContent.currentStep + 1] =
                featureVectorFunction.value(featureVectorFunctionParameters, vertex, vertexContent.currentStep);
        if (vertexContent.currentStep == maximumNumberOfSteps - 2) {
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
                null
        );
    }

    public void performBackwardPass() {
        outputFunctionParametersGradient = Vectors.dense(outputFunctionParameters.size());
        featureVectorFunctionParametersGradient = Vectors.dense(featureVectorFunctionParameters.size());
        for (int step = maximumNumberOfSteps - 1; step > 0; step--) {
            graph.computeVerticesUpdatedContent(this::backwardVertexComputeFunction);
            graph.updateVerticesContent();
        }
        needsBackwardPass = false;
    }

    private VertexContentType backwardVertexComputeFunction(Vertex<VertexContentType, E> vertex) {
        VertexContentType vertexContent = vertex.getContent();
        Vector featureVectorGradient;
        if (vertexContent.currentStep == maximumNumberOfSteps - 1) {
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
            featureVectorGradient =
                    featureVectorFunction.featureVectorGradient(featureVectorFunctionParameters,
                                                                vertex,
                                                                vertex,
                                                                vertexContent.currentStep)
                            .multiply(vertexContent.featureVectorGradient);
            for (Edge<VertexContentType, E> incomingEdge : vertex.getIncomingEdges())
                featureVectorGradient.addInPlace(
                        featureVectorFunction.featureVectorGradient(featureVectorFunctionParameters,
                                                                    incomingEdge.getSourceVertex(),
                                                                    vertex,
                                                                    vertexContent.currentStep)
                                .multiply(incomingEdge.getSourceVertex().getContent().featureVectorGradient)
                );
            for (Edge<VertexContentType, E> outgoingEdge : vertex.getOutgoingEdges())
                featureVectorGradient.addInPlace(
                        featureVectorFunction.featureVectorGradient(featureVectorFunctionParameters,
                                                                    outgoingEdge.getDestinationVertex(),
                                                                    vertex,
                                                                    vertexContent.currentStep)
                                .multiply(outgoingEdge.getDestinationVertex().getContent().featureVectorGradient)
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
                featureVectorGradient
        );
    }

    public static class VertexContentType {
        protected int id;
        protected int currentStep;                // k
        protected Vector[] featureVectors;        // φ(1),...,φ(k)
        protected Vector featureVectorGradient;   // dL / dφ(k)

        public VertexContentType(int id,
                                 int currentStep,
                                 Vector[] featureVectors,
                                 Vector featureVectorGradient) {
            this.id = id;
            this.currentStep = currentStep;
            this.featureVectors = featureVectors;
            this.featureVectorGradient = featureVectorGradient;
        }

        public int getId() {
            return id;
        }

        public Vector getFeatureVector() {
            return featureVectors[currentStep];
        }
    }

    public abstract static class FeatureVectorFunction<VertexContentType, E> {
        public abstract Vector value(Vector parameters, Vertex<VertexContentType, E> vertex, int step);
        public abstract Matrix gradient(Vector parameters, Vertex<VertexContentType, E> vertex, int step);
        public abstract Matrix featureVectorGradient(Vector parameters,
                                                     Vertex<VertexContentType, E> vertex,
                                                     Vertex<VertexContentType, E> differentiatingVertex,
                                                     int step);
    }

    public abstract static class OutputFunction {
        public abstract Vector value(Vector featureVector, Vector parameters);
        public abstract Matrix featureVectorGradient(Vector featureVector, Vector parameters);
        public abstract Matrix parametersGradient(Vector featureVector, Vector parameters);
    }

    public abstract static class LossFunction {
        public abstract double value(Vector networkOutput, Vector correctOutput);
        public abstract Vector gradient(Vector networkOutput, Vector correctOutput);
    }
}
