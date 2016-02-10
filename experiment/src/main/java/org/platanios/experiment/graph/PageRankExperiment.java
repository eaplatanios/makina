package org.platanios.experiment.graph;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.platanios.learn.graph.Graph;
import org.platanios.learn.graph.PageRankAlgorithm;
import org.platanios.learn.graph.Vertex;

import java.util.HashMap;
import java.util.Map;

/**
 * @author Emmanouil Antonios Platanios
 */
public class PageRankExperiment {
    private final Logger logger = LogManager.getFormatterLogger("PageRank Experiment");
    private final Map<Integer, Vertex<PageRankAlgorithm.VertexContentType, Void>> vertexIndexesMap = new HashMap<>();

    private final String dataSetName;
    private final Graph<PageRankAlgorithm.VertexContentType, Void> graph;

    public PageRankExperiment(DataSets.DataSet dataSet) {
        dataSetName = dataSet.getName();
        logger.info("Number of vertices: " + dataSet.getVertexIndices().size());
        logger.info("Number of edges: " + dataSet.getEdges().size());
        graph = new Graph<>();
        for (int vertexIndex : dataSet.getVertexIndices())
            vertexIndexesMap.put(vertexIndex, new Vertex<>(new PageRankAlgorithm.VertexContentType(0.0)));
        for (DataSets.Edge edge : dataSet.getEdges())
            graph.addEdge(vertexIndexesMap.get(edge.getSourceVertexIndex()), vertexIndexesMap.get(edge.getDestinationVertexIndex()));
        logger.info("Loaded graph for the " + dataSetName + " data set.");
    }

    public void runPageRank() {
        logger.info("Running PageRank for the " + dataSetName + " data set.");
        PageRankAlgorithm pageRankAlgorithm =
                new PageRankAlgorithm.Builder<>(graph)
                        .dampingFactor(0.85)
                        .maximumNumberOfIterations(1000)
                        .checkForRankConvergence(false)
                        .loggingLevel(2)
                        .build();
        pageRankAlgorithm.computeRanks();
        logger.info("Finished running PageRank for the " + dataSetName + " data set.");
    }

    public static void main(String[] args) {
        PageRankExperiment experiment = new PageRankExperiment(DataSets.loadPokecDataSet("/Users/Anthony/Downloads/Pokec Graph Data Set/soc-pokec-relationships.txt"));
//        PageRankExperiment experiment = new PageRankExperiment(DataSets.loadTwitterSocialCirclesDataSet("/Users/Anthony/Downloads/Twitter Social Circles Data Set/twitter_combined.txt"));
//        PageRankExperiment experiment = new PageRankExperiment(DataSets.loadPokecDataSet("/Users/Anthony/Downloads/soc-Epinions1.txt"));
        experiment.runPageRank();
    }
}
