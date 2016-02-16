package org.platanios.learn.graph;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * @author Emmanouil Antonios Platanios
 */
public class HITSAlgorithm<E> {
    private final Logger logger = LogManager.getFormatterLogger("HITS");

    private final Graph<VertexContentType, E> graph;
    private final int maximumNumberOfIterations;
    private final boolean checkForScoresConvergence;
    private final double scoresChangeTolerance;
    private final int loggingLevel;

    private int currentIteration = 0;
    private double lastMeanAbsoluteRankChange = Double.MAX_VALUE;
    private double authorityNormalizationConstant = 0.0;
    private double hubNormalizationConstant = 0.0;

    public static class Builder<E> {
        private final Graph<VertexContentType, E> graph;

        private int maximumNumberOfIterations = 30;
        private boolean checkForScoresConvergence = true;
        private double scoresChangeTolerance = 1e-6;
        private int loggingLevel = 0;

        public Builder(Graph<VertexContentType, E> graph) {
            this.graph = graph;
        }

        public Builder maximumNumberOfIterations(int maximumIterations) {
            this.maximumNumberOfIterations = maximumIterations;
            return this;
        }

        public Builder checkForScoresConvergence(boolean checkForScoresConvergence) {
            this.checkForScoresConvergence = checkForScoresConvergence;
            return this;
        }

        public Builder scoresChangeTolerance(double scoresChangeTolerance) {
            this.scoresChangeTolerance = scoresChangeTolerance;
            return this;
        }

        public Builder loggingLevel(int loggingLevel) {
            this.loggingLevel = loggingLevel;
            return this;
        }

        public HITSAlgorithm<E> build() {
            return new HITSAlgorithm<>(this);
        }
    }

    private HITSAlgorithm(Builder<E> builder) {
        graph = builder.graph;
        maximumNumberOfIterations = builder.maximumNumberOfIterations;
        checkForScoresConvergence = builder.checkForScoresConvergence;
        scoresChangeTolerance = builder.scoresChangeTolerance;
        loggingLevel = builder.loggingLevel;
    }

    public void computeScores() {
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
                || checkForScoresConvergence && lastMeanAbsoluteRankChange <= scoresChangeTolerance;
    }

    // TODO: Can be made more efficient by calling the update vertices content method only once.
    private void performIterationUpdates() {
        graph.computeVerticesUpdatedContent(this::vertexComputeFunction);
        graph.updateVerticesContent();
        authorityNormalizationConstant = Math.sqrt(authorityNormalizationConstant);
        hubNormalizationConstant = Math.sqrt(hubNormalizationConstant);
        graph.computeVerticesUpdatedContent(this::normalizationVertexComputeFunction);
        graph.updateVerticesContent();
        authorityNormalizationConstant = 0.0;
        hubNormalizationConstant = 0.0;
        if (checkForScoresConvergence) {
            lastMeanAbsoluteRankChange = 0.0;
            for (Vertex<VertexContentType, E> vertex : graph.getVertices())
                lastMeanAbsoluteRankChange += vertex.getContent().lastAbsoluteChange;
            lastMeanAbsoluteRankChange /= graph.getNumberOfVertices();
        }
    }

    private void logIteration() {
        if (checkForScoresConvergence)
            logger.info("Iteration #: %10d | Mean Absolute Rank Change: %20s", currentIteration, lastMeanAbsoluteRankChange);
        else
            logger.info("Iteration #: %10d", currentIteration);
    }

    private VertexContentType vertexComputeFunction(Vertex<VertexContentType, E> vertex) {
        double oldAuthorityScore = vertex.getContent().authorityScore;
        double oldHubScore = vertex.getContent().hubScore;
        double newAuthorityScore = 0.0;
        double newHubScore = 0.0;
        for (Edge<VertexContentType, E> incomingEdge : vertex.getIncomingEdges())
            newAuthorityScore += incomingEdge.getSourceVertex().getContent().hubScore;
        for (Edge<VertexContentType, E> outgoingEdge : vertex.getOutgoingEdges())
            newHubScore += outgoingEdge.getDestinationVertex().getContent().authorityScore;
        authorityNormalizationConstant += newAuthorityScore * newAuthorityScore;
        hubNormalizationConstant += newHubScore * newHubScore;
        return new VertexContentType(vertex.getContent().id,
                                     newAuthorityScore,
                                     newHubScore,
                                     checkForScoresConvergence ?
                                             Math.abs(newAuthorityScore - oldAuthorityScore)
                                                     + Math.abs(newHubScore - oldHubScore)
                                             : 0.0);
    }

    private VertexContentType normalizationVertexComputeFunction(Vertex<VertexContentType, E> vertex) {
        return new VertexContentType(vertex.getContent().id,
                                     vertex.getContent().authorityScore / authorityNormalizationConstant,
                                     vertex.getContent().hubScore / hubNormalizationConstant,
                                     vertex.getContent().lastAbsoluteChange);
    }

    public static class VertexContentType {
        private final int id;
        private final double authorityScore;
        private final double hubScore;
        private final double lastAbsoluteChange;

        public VertexContentType(int id, double authorityScore, double hubScore) {
            this(id, authorityScore, hubScore, Double.MAX_VALUE);
        }

        public VertexContentType(int id, double authorityScore, double hubScore, double lastAbsoluteChange) {
            this.id = id;
            this.authorityScore = authorityScore;
            this.hubScore = hubScore;
            this.lastAbsoluteChange = lastAbsoluteChange;
        }

        public int getId() {
            return id;
        }

        public double getAuthorityScore() {
            return authorityScore;
        }

        public double getHubScore() {
            return hubScore;
        }

        public double getLastAbsoluteChange() {
            return lastAbsoluteChange;
        }
    }
}
