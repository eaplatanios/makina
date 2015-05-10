package org.platanios.experiment.psl;

import com.google.common.collect.ImmutableList;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.learn.optimization.ConsensusAlternatingDirectionsMethodOfMultipliersSolver;

import java.util.*;

/**
 * Class for
 */
public class RandomWalkSampler<T> implements ConsensusAlternatingDirectionsMethodOfMultipliersSolver.SubProblemSelector {

    private final Map<T, GraphEdges> entityToGraph;
    private final List<T> graphSampleOrigins;
    private final List<Integer> graphSampleCursorOriginIndices;
    private final List<GraphEdges> graphSampleCursor;
    private final Random random;
    private final double restartProbability;
    private final double sampleProbablity;
    private final Map<Integer, PredicateInformation> predicateInformation;
    private final Map<Integer, Integer> internalToExternalIds;
    private HashSet<Integer> predicateIdsToUpdate;
    private final TermPredicateIdGetter predicateIdGetter;
    private static final int maxSampleAttempts = 20;
    private final boolean logSampleCounts;
    private int[] sampleCounts;
    private static final Logger logger = LogManager.getLogger("Classification / Training");
    // for debugging
    public ProbabilisticSoftLogicPredicateManager predicateManager = null;
    public Map<Integer, ProbabilisticSoftLogicProblem.Rule> termToRule = null;

    private RandomWalkSampler(Builder builder) {

        this.random = builder.random;
        this.entityToGraph = builder.entityToGraph;
        this.graphSampleOrigins = builder.originEntities;
        this.graphSampleCursor = new ArrayList<>();
        this.graphSampleCursorOriginIndices = new ArrayList<>();
        this.restartProbability = builder.restartProbability;
        this.sampleProbablity = builder.sampleProbability;
        this.predicateInformation = builder.predicateInformation;
        this.internalToExternalIds = builder.internalToExternalIds;
        this.predicateIdGetter = builder.predicateIdGetter;
        this.predicateIdsToUpdate = new HashSet<>();
        this.logSampleCounts = builder.logSampleCounts;
        this.sampleCounts = null;

    }

    public int[] selectSubProblems(ConsensusAlternatingDirectionsMethodOfMultipliersSolver solver) {

        // choose which seed to start each cursor from
        for (int iCursor = this.graphSampleCursorOriginIndices.size(); iCursor < solver.numberOfSubProblemSamples; ++iCursor) {
            int iSeed = random.nextInt(this.graphSampleOrigins.size());
            this.graphSampleCursorOriginIndices.add(iSeed);
            this.graphSampleCursor.add(this.entityToGraph.get(this.graphSampleOrigins.get(iSeed)));
        }

        if (solver.getCurrentIteration() > 0) {

            if (solver.getCurrentIteration() == 5000) {
                // get the top 100
                PriorityQueue<Map.Entry<Integer, Integer>> priorityQueue = new PriorityQueue<>(
                        this.sampleCounts.length,
                        (element1, element2) -> (int) Math.signum(element2.getValue() - element1.getValue())
                );

                for (int indexCount = 0; indexCount < this.sampleCounts.length; ++indexCount) {
                    priorityQueue.add(new AbstractMap.SimpleEntry<>(indexCount, this.sampleCounts[indexCount]));
                }

                for (int indexCount = 0; indexCount < 100; ++indexCount) {

                    Map.Entry<Integer, Integer> sampleCount = priorityQueue.poll();
                    int[] subProblemPredicateIds = this.predicateIdGetter.getInternalPredicateIds(sampleCount.getKey());
                    StringBuilder sb = new StringBuilder();
                    for (int predicateId : subProblemPredicateIds) {
                        int externalId = this.internalToExternalIds.get(predicateId);
                        if (predicateManager != null) {
                            ProbabilisticSoftLogicProblem.Predicate predicate = predicateManager.getPredicateFromId(externalId);
                            sb.append(predicate.toString());
                        } else {
                            sb.append(externalId);
                        }
                        sb.append(" ");
                    }
                    sb.append(";");
                    if (this.termToRule != null) {
                        ProbabilisticSoftLogicProblem.Rule rule = this.termToRule.get(sampleCount.getKey());
                        sb.append(rule.toString());
                    }
                    logger.info("RankedSample" + indexCount + " (" + sampleCount.getKey() + ", " + sampleCount.getValue() + "): " + sb.toString());
                }
            }

            for (int predicateId : predicateIdsToUpdate) {
                double value = solver.currentPoint.get(predicateId);
                int externalId = this.internalToExternalIds.get(predicateId);
                PredicateInformation predicateInformation = this.predicateInformation.get(externalId);
                predicateInformation.Edge.EdgeWeight = value;
            }
        }

        predicateIdsToUpdate.clear();

        int[] sampledSubProblems = new int[solver.numberOfSubProblemSamples];

        for (int i = 0; i < solver.numberOfSubProblemSamples; ++i) {

            boolean isSampled = false;
            for (int sampleAttempt = 0; sampleAttempt < maxSampleAttempts && !isSampled; ++sampleAttempt) {
                // first choose whether to restart
                double randRestart = this.random.nextDouble();
                if (randRestart < this.restartProbability) {
                    this.graphSampleCursor.set(i, this.entityToGraph.get(this.graphSampleOrigins.get(this.graphSampleCursorOriginIndices.get(i))));
                } else {
                    boolean isSampleOnStep = this.random.nextDouble() < this.sampleProbablity;
                    Map.Entry<Integer, GraphEdges> step = this.graphSampleCursor.get(i).step();
                    if (step != null) { // step can be null for nodes with no outgoing edges, keep looping so there is a chance to restart
                        this.graphSampleCursor.set(i, step.getValue());
                        if (isSampleOnStep) {
                            int predicateId = step.getKey();
                            List<Integer> terms = this.predicateInformation.get(predicateId).Terms;
                            if (terms.size() > 0) {
                                int indexIntoTerms = random.nextInt(terms.size());
                                sampledSubProblems[i] = terms.get(indexIntoTerms);
                                int[] subProblemPredicateIds = this.predicateIdGetter.getInternalPredicateIds(sampledSubProblems[i]);
                                if (this.logSampleCounts) {
                                    if (this.sampleCounts == null) {
                                        this.sampleCounts = new int[solver.getNumberOfTerms()];
                                    }
                                    ++this.sampleCounts[sampledSubProblems[i]];
                                }
                                for (int subProblemPredicateId : subProblemPredicateIds) {
                                    this.predicateIdsToUpdate.add(subProblemPredicateId);
                                }
                                isSampled = true;
                            }
                        }
                    }
                }
            }

            if (!isSampled) {// this could happen if we are in a bad part of the graph that has observed edges everywhere

                // remove this seed since it seems bad
                this.graphSampleOrigins.remove(this.graphSampleCursorOriginIndices.get(i));

                int iSeed = random.nextInt(this.graphSampleOrigins.size());
                this.graphSampleCursorOriginIndices.set(i, iSeed);
                this.graphSampleCursor.set(i, this.entityToGraph.get(this.graphSampleOrigins.get(iSeed)));

                sampledSubProblems[i] = this.random.nextInt(solver.getNumberOfTerms());

            }
        }

        return sampledSubProblems;

    }

    private static class PredicateInformation {

        public GraphEdges.Edge Edge;
        public final List<Integer> Terms;

        public PredicateInformation(List<Integer> terms) {
            this.Terms = terms;
        }

    }

    private static class GraphEdges {

        public class Edge {

            public double EdgeWeight;
            public final GraphEdges EndNode;

            public Edge(double weight, GraphEdges endNode) {
                this.EdgeWeight = weight;
                this.EndNode = endNode;
            }

        }

        private final HashMap<Integer, Edge> edges = new HashMap<>();
        private final Random random;

        public GraphEdges(Random random) {
            this.random = random;
        }

        public Edge addEdge(double weight, GraphEdges endNode, int edgeId) {
            if (this.edges.containsKey(edgeId)) {
                throw new UnsupportedOperationException("Cannot add node twice");
            }
            Edge result = new Edge(weight, endNode);
            this.edges.put(edgeId, result);
            return result;
        }


        public Map.Entry<Integer, GraphEdges> step() {
            double rand = this.random.nextDouble();
            double accum = 0;
            for (Map.Entry<Integer, Edge> edgeEntry : this.edges.entrySet()) {
                accum += edgeEntry.getValue().EdgeWeight;
                if (accum >= rand) {
                    return new AbstractMap.SimpleEntry<>(edgeEntry.getKey(), edgeEntry.getValue().EndNode);
                }
            }
            return null;
        }

    }

    public interface TermPredicateIdGetter {
        int[] getInternalPredicateIds(int term);
    }

    public static class Builder<T> {

        private Random random;
        private HashMap<T, GraphEdges> entityToGraph = new HashMap<>();
        private List<T> originEntities = new ArrayList<>();
        private final double restartProbability;
        private final double sampleProbability;
        private Map<Integer, Integer> internalToExternalIds;
        private Map<Integer, PredicateInformation> predicateInformation;
        private TermPredicateIdGetter predicateIdGetter;
        public boolean logSampleCounts;

        public Builder(
                Map<Integer, List<Integer>> predicateIdToTerms,
                Map<Integer, Integer> internalToExternalIds,
                TermPredicateIdGetter predicateIdGetter,
                double restartProbability,
                double sampleProbability,
                Random random) {
            this.random = random;

            this.restartProbability = restartProbability;
            this.sampleProbability = sampleProbability;

            this.internalToExternalIds = internalToExternalIds;
            this.predicateIdGetter = predicateIdGetter;
            this.predicateInformation = new HashMap<>(predicateIdToTerms.size());
            predicateIdToTerms.forEach((Integer id, List<Integer> terms) -> this.predicateInformation.put(id, new PredicateInformation(terms)));
        }

        // allows the same entity to be added multiple times
        public Builder<T> addOriginEntity(T origin) {
            this.originEntities.add(origin);
            return this;
        }

        public Builder<T> addUnobservedEdge(int id, T startNode, T endNode) {
            return internalAdd(id, 1, startNode, endNode); // use 1 for unobserved edge weight
        }

        public Builder<T> addObservedEdge(int id, double weight, T startNode, T endNode) {
            return internalAdd(id, weight, startNode, endNode);
        }

        private Builder<T> internalAdd(int id, double weight, T startNodeKey, T endNodeKey) {
            GraphEdges startNode = entityToGraph.getOrDefault(startNodeKey, null);
            if (startNode == null) {
                startNode = new GraphEdges(random);
                entityToGraph.put(startNodeKey, startNode);
            }

            GraphEdges endNode = entityToGraph.getOrDefault(endNodeKey, null);
            if (endNode == null) {
                endNode = new GraphEdges(this.random);
                entityToGraph.put(endNodeKey, endNode);
            }

            GraphEdges.Edge edge = startNode.addEdge(weight, endNode, id);
            PredicateInformation information = this.predicateInformation.getOrDefault(id, null);
            if (information == null) { //observed predicates might not be in the graph yet
                information = new PredicateInformation(new ArrayList<>());
                this.predicateInformation.put(id, information);
            }
            information.Edge = edge;
            return this;
        }

        public RandomWalkSampler<T> build() {
            return new RandomWalkSampler<>(this);
        }

    }

}