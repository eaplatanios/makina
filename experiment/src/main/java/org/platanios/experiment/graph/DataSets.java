package org.platanios.experiment.graph;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
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

    public static LabeledDataSet loadLabeledDataSet(String folderPath) {
        String[] folderPathParts = folderPath.split("/");
        String name = folderPathParts[folderPathParts.length - 1];
        LabeledDataSet dataSet = new LabeledDataSet(name);
        try {
            Files.newBufferedReader(Paths.get(folderPath + "/vertices.txt")).lines().forEach(line -> {
                String[] lineParts = line.split("\t");
                dataSet.addVertex(Integer.parseInt(lineParts[0]));
                if (lineParts.length > 1)
                    dataSet.addVertexLabel(Integer.parseInt(lineParts[0]), Integer.parseInt(lineParts[1]));
            });
            Files.newBufferedReader(Paths.get(folderPath + "/edges.txt")).lines().forEach(line -> {
                String[] lineParts = line.split("\t");
                dataSet.addEdge(Integer.parseInt(lineParts[0]), Integer.parseInt(lineParts[1]));
            });
        } catch (IOException e) {
            throw new IllegalStateException("An exception occured while loading the " + name + " data set.");
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

        void addVertex(int vertexIndex) {
            vertexIndices.add(vertexIndex);
        }

        void addEdge(int sourceVertexIndex, int destinationVertexIndex) {
            vertexIndices.add(sourceVertexIndex);
            vertexIndices.add(destinationVertexIndex);
            edges.add(new Edge(sourceVertexIndex, destinationVertexIndex));
        }

        Set<Integer> getVertexIndices() {
            return vertexIndices;
        }

        Set<Edge> getEdges() {
            return edges;
        }

        String getName() {
            return name;
        }
    }

    public static class LabeledDataSet extends DataSet {
        private final Map<Integer, Integer> vertexLabels = new HashMap<>();

        private LabeledDataSet(String name) {
            super(name);
        }

        void addVertexLabel(int vertexIndex, int label) {
            vertexLabels.put(vertexIndex, label);
        }

        int getVertexLabel(int vertexIndex) {
            return vertexLabels.get(vertexIndex);
        }

        Map<Integer, Integer> getVertexLabels() {
            return vertexLabels;
        }
    }

    public static class Edge {
        private final int sourceVertexIndex;
        private final int destinationVertexIndex;

        private Edge(int sourceVertexIndex, int destinationVertexIndex) {
            this.sourceVertexIndex = sourceVertexIndex;
            this.destinationVertexIndex = destinationVertexIndex;
        }

        int getSourceVertexIndex() {
            return sourceVertexIndex;
        }

        int getDestinationVertexIndex() {
            return destinationVertexIndex;
        }
    }
}
