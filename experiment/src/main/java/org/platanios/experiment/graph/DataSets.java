package org.platanios.experiment.graph;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.Set;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DataSets {
    public static DataSet loadTwitterSocialCirclesDataSet(String filePath) {
        DataSet dataSet = new DataSet("Twitter Social Circles");
        try {
            Files.newBufferedReader(Paths.get(filePath)).lines().forEach(line -> {
                String[] lineParts = line.split(" ");
                dataSet.addEdge(Integer.parseInt(lineParts[0]), Integer.parseInt(lineParts[1]));
            });
        } catch (IOException e) {
            throw new IllegalStateException("An exception occured while processing the Twitter data set file.");
        }
        return dataSet;
    }

    public static DataSet loadPokecDataSet(String filePath) {
        DataSet dataSet = new DataSet("Pokec");
        try {
            Files.newBufferedReader(Paths.get(filePath)).lines().forEach(line -> {
                String[] lineParts = line.split("\t");
                dataSet.addEdge(Integer.parseInt(lineParts[0]), Integer.parseInt(lineParts[1]));
            });
        } catch (IOException e) {
            throw new IllegalStateException("An exception occured while processing the Pokec data set file.");
        }
        return dataSet;
    }

    public static class DataSet {
        private final Set<Integer> vertexIndices = new HashSet<>();
        private final Set<Edge> edges = new HashSet<>();

        private final String name;

        private DataSet(String name) {
            this.name = name;
        }

        private void addVertex(int vertexIndex) {
            vertexIndices.add(vertexIndex);
        }

        private void addEdge(int sourceVertexIndex, int destinationVertexIndex) {
            vertexIndices.add(sourceVertexIndex);
            vertexIndices.add(destinationVertexIndex);
            edges.add(new Edge(sourceVertexIndex, destinationVertexIndex));
        }

        public Set<Integer> getVertexIndices() {
            return vertexIndices;
        }

        public Set<Edge> getEdges() {
            return edges;
        }

        public String getName() {
            return name;
        }
    }

    public static class Edge {
        private final int sourceVertexIndex;
        private final int destinationVertexIndex;

        public Edge(int sourceVertexIndex, int destinationVertexIndex) {
            this.sourceVertexIndex = sourceVertexIndex;
            this.destinationVertexIndex = destinationVertexIndex;
        }

        public int getSourceVertexIndex() {
            return sourceVertexIndex;
        }

        public int getDestinationVertexIndex() {
            return destinationVertexIndex;
        }
    }
}
