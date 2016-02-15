package org.platanios.learn.graph;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * @author Emmanouil Antonios Platanios
 */
public class PageRankAlgorithm<E> {
    private final Logger logger = LogManager.getFormatterLogger("PageRank");

    private final Graph<VertexContentType, E> graph;
    private final double dampingFactor;
    private final double dampingTerm;
    private final int maximumNumberOfIterations;
    private final boolean checkForRankConvergence;
    private final double rankChangeTolerance;
    private final int loggingLevel;

    private int currentIteration = 0;
    private double lastMeanAbsoluteRankChange = Double.MAX_VALUE;

    public static class Builder<E> {
        private final Graph<VertexContentType, E> graph;

        private double dampingFactor = 0.85;
        private int maximumNumberOfIterations = 30;
        private boolean checkForRankConvergence = true;
        private double rankChangeTolerance = 1e-6;
        private int loggingLevel = 0;

        public Builder(Graph<VertexContentType, E> graph) {
            this.graph = graph;
        }

        public Builder dampingFactor(double dampingFactor) {
            this.dampingFactor = dampingFactor;
            return this;
        }

        public Builder maximumNumberOfIterations(int maximumIterations) {
            this.maximumNumberOfIterations = maximumIterations;
            return this;
        }

        public Builder checkForRankConvergence(boolean checkForRankConvergence) {
            this.checkForRankConvergence = checkForRankConvergence;
            return this;
        }

        public Builder rankChangeTolerance(double rankChangeTolerance) {
            this.rankChangeTolerance = rankChangeTolerance;
            return this;
        }

        public Builder loggingLevel(int loggingLevel) {
            this.loggingLevel = loggingLevel;
            return this;
        }

        public PageRankAlgorithm<E> build() {
            return new PageRankAlgorithm<>(this);
        }
    }

    private PageRankAlgorithm(Builder<E> builder) {
        graph = builder.graph;
        dampingFactor = builder.dampingFactor;
        dampingTerm = (1.0 - dampingFactor) / graph.getNumberOfVertices();
        maximumNumberOfIterations = builder.maximumNumberOfIterations;
        checkForRankConvergence = builder.checkForRankConvergence;
        rankChangeTolerance = builder.rankChangeTolerance;
        loggingLevel = builder.loggingLevel;
    }

    public void computeRanks() {
        while (!checkTerminationConditions()) {
            performIterationUpdates();
            currentIteration++;
            if ((loggingLevel == 1 && currentIteration % 1000 == 0)
                    || (loggingLevel == 2 && currentIteration % 100 == 0)
                    || (loggingLevel == 3 && currentIteration % 10 == 0)
                    || loggingLevel > 3)
                logIteration();
        }
    }

    private boolean checkTerminationConditions() {
        return currentIteration >= maximumNumberOfIterations
                || checkForRankConvergence && lastMeanAbsoluteRankChange <= rankChangeTolerance;
    }

    private void performIterationUpdates() {
        graph.computeVerticesUpdatedContent(this::vertexComputeFunction);
        graph.updateVerticesContent();
        if (checkForRankConvergence) {
            lastMeanAbsoluteRankChange = 0.0;
            for (Vertex<VertexContentType, E> vertex : graph.getVertices())
                lastMeanAbsoluteRankChange += vertex.getContent().lastAbsoluteChange;
            lastMeanAbsoluteRankChange /= graph.getNumberOfVertices();
        }
    }

    private void logIteration() {
        if (checkForRankConvergence)
            logger.info("Iteration #: %10d | Mean Absolute Rank Change: %20s", currentIteration, lastMeanAbsoluteRankChange);
        else
            logger.info("Iteration #: %10d", currentIteration);
    }

    private VertexContentType vertexComputeFunction(Vertex<VertexContentType, E> vertex) {
        double oldRank = vertex.getContent().rank;
        double newRank = 0.0;
        for (Edge<VertexContentType, E> incomingEdge : vertex.getIncomingEdges())
            newRank += incomingEdge.getSourceVertex().getContent().rank / incomingEdge.getSourceVertex().getNumberOfOutgoingEdges();
        newRank *= dampingFactor;
        newRank += dampingTerm;
        return new VertexContentType(vertex.getContent().id,
                                     newRank,
                                     checkForRankConvergence ? Math.abs(newRank - oldRank) : 0.0);
    }

    public static class VertexContentType {
        private final int id;
        private final double rank;
        private final double lastAbsoluteChange;

        public VertexContentType(int id, double rank) {
            this(id, rank, Double.MAX_VALUE);
        }

        public VertexContentType(int id, double rank, double lastAbsoluteChange) {
            this.id = id;
            this.rank = rank;
            this.lastAbsoluteChange = lastAbsoluteChange;
        }

        public int getId() {
            return id;
        }

        public double getRank() {
            return rank;
        }

        public double getLastAbsoluteChange() {
            return lastAbsoluteChange;
        }
    }
}
