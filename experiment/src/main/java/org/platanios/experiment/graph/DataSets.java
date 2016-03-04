package org.platanios.experiment.graph;

import com.google.common.base.Objects;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DataSets {
    public static DataSet loadUnlabeledDataSet(String filePath, boolean directed) {
        String[] filePathParts = filePath.split("/");
        String name = filePathParts[filePathParts.length - 1].split("\\.")[0];
        DataSet dataSet = new DataSet(name);
        try {
            Files.newBufferedReader(Paths.get(filePath)).lines().forEach(line -> {
                String[] lineParts = line.split("\t");
                if (directed || !dataSet.getEdges().contains(new Edge(Integer.parseInt(lineParts[1]), Integer.parseInt(lineParts[0]))))
                    dataSet.addEdge(Integer.parseInt(lineParts[0]), Integer.parseInt(lineParts[1]));
            });
        } catch (IOException e) {
            throw new IllegalStateException("An exception occurred while processing the " + name + " data set.");
        }
        return dataSet;
    }

    public static LabeledDataSet loadLabeledDataSet(String folderPath, boolean directed) {
        String[] folderPathParts = folderPath.split("/");
        String name = folderPathParts[folderPathParts.length - 1];
        LabeledDataSet dataSet = new LabeledDataSet(name);
        Map<Integer, Integer> classIndexes = new HashMap<>();
        AtomicInteger numberOfClasses = new AtomicInteger(0);
        try {
            Files.newBufferedReader(Paths.get(folderPath + "/vertices.txt")).lines().forEach(line -> {
                String[] lineParts = line.split("\t");
                dataSet.addVertex(Integer.parseInt(lineParts[0]));
                if (lineParts.length > 1) {
                    if (!classIndexes.containsKey(Integer.parseInt(lineParts[1])))
                        classIndexes.put(Integer.parseInt(lineParts[1]), numberOfClasses.getAndIncrement());
                    dataSet.addVertexLabel(Integer.parseInt(lineParts[0]), classIndexes.get(Integer.parseInt(lineParts[1])));
                }
            });
            Files.newBufferedReader(Paths.get(folderPath + "/edges.txt")).lines().forEach(line -> {
                String[] lineParts = line.split("\t");
                if (directed || !dataSet.getEdges().contains(new Edge(Integer.parseInt(lineParts[1]), Integer.parseInt(lineParts[0]))))
                    dataSet.addEdge(Integer.parseInt(lineParts[0]), Integer.parseInt(lineParts[1]));
            });
        } catch (IOException e) {
            throw new IllegalStateException("An exception occurred while loading the " + name + " data set.");
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
        private final Map<Integer, Integer> vertexStatistics = new HashMap<>();

        private LabeledDataSet(String name) {
            super(name);
        }

        void addVertexLabel(int vertexIndex, int label) {
            vertexLabels.put(vertexIndex, label);
            if (!vertexStatistics.containsKey(label))
                vertexStatistics.put(label, 0);
            vertexStatistics.put(label, vertexStatistics.get(label) + 1);
        }

        int getVertexLabel(int vertexIndex) {
            return vertexLabels.get(vertexIndex);
        }

        Map<Integer, Integer> getVertexLabels() {
            return vertexLabels;
        }

        Map<Integer, Integer> getVertexStatistics() {
            return vertexStatistics;
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

        @Override
        public boolean equals(Object other) {
            if (this == other)
                return true;
            if (other == null || getClass() != other.getClass())
                return false;

            Edge that = (Edge) other;

            return Objects.equal(sourceVertexIndex, that.sourceVertexIndex)
                    && Objects.equal(destinationVertexIndex, that.destinationVertexIndex);
        }

        @Override
        public int hashCode() {
            return Objects.hashCode(sourceVertexIndex, destinationVertexIndex);
        }
    }
}
