package org.platanios.experiment.psl;

import com.google.common.collect.ImmutableList;
import org.platanios.learn.optimization.ConsensusAlternatingDirectionsMethodOfMultipliersSolver;

import java.util.*;

/**
 * Class for
 */
public class RandomWalkSampler<T> implements ConsensusAlternatingDirectionsMethodOfMultipliersSolver.SubProblemSelector {

    private final Map<T, GraphEdges> entityToGraph;
    private final List<T> graphSampleOrigins;
    private final List<GraphEdges> graphSampleCursor;
    private final Random random;
    private final double restartProbability;
    private final double sampleProbablity;
    private final Map<Integer, PredicateInformation> predicateInformation;
    private final Map<Integer, Integer> internalToExternalIds;
    private HashSet<Integer> predicateIdsToUpdate;
    private final TermPredicateIdGetter predicateIdGetter;
    private static final int maxSampleAttempts = 20;

    private RandomWalkSampler(Builder builder) {

        this.random = builder.random;
        this.entityToGraph = builder.entityToGraph;
        this.graphSampleOrigins = builder.originEntities;
        this.graphSampleCursor = new ArrayList<>(this.graphSampleOrigins.size());
        this.graphSampleOrigins.forEach((entity) -> this.graphSampleCursor.add(this.entityToGraph.get(entity)));
        this.restartProbability = builder.restartProbability;
        this.sampleProbablity = builder.sampleProbability;
        this.predicateInformation = builder.predicateInformation;
        this.internalToExternalIds = builder.internalToExternalIds;
        this.predicateIdGetter = builder.predicateIdGetter;
        this.predicateIdsToUpdate = new HashSet<>();

    }

    public int[] selectSubProblems(ConsensusAlternatingDirectionsMethodOfMultipliersSolver solver) {

        if (solver.getCurrentIteration() > 0) {
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

            int iSeed = random.nextInt(this.graphSampleCursor.size());

            boolean isSampled = false;
            for (int sampleAttempt = 0; sampleAttempt < maxSampleAttempts && !isSampled; ++sampleAttempt) {
                // first choose whether to restart, sample or step
                double rand = this.random.nextDouble();
                if (rand < this.restartProbability) {
                    this.graphSampleCursor.set(iSeed, this.entityToGraph.get(this.graphSampleOrigins.get(iSeed)));
                } else {
                    boolean isSampleOnStep = rand < (this.restartProbability + this.sampleProbablity);
                    Map.Entry<Integer, GraphEdges> step = this.graphSampleCursor.get(iSeed).step();
                    if (step != null) { // step can be null for nodes with no outgoing edges, keep looping so there is a chance to restart
                        this.graphSampleCursor.set(iSeed, step.getValue());
                        if (isSampleOnStep) {
                            int predicateId = step.getKey();
                            List<Integer> terms = this.predicateInformation.get(predicateId).Terms;
                            if (terms.size() > 0) {
                                int indexIntoTerms = random.nextInt(terms.size());
                                sampledSubProblems[i] = terms.get(indexIntoTerms);
                                int[] subProblemPredicateIds = this.predicateIdGetter.getInternalPredicateIds(sampledSubProblems[i]);
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
                this.graphSampleOrigins.remove(iSeed);
                this.graphSampleCursor.remove(iSeed);
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

            if (this.sampleProbability + this.restartProbability >= 1) {
                throw new UnsupportedOperationException("Sample Probability + Restart Probability must be less than 1");
            }

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