package makina.learn.classification;

import makina.math.matrix.Vector;

import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class TrainingData {
    private List<Entry> data;

    public TrainingData(List<Entry> data) {
        this.data = data;
    }

    public List<Entry> getData() {
        return data;
    }

    public static class Entry {
        public final Vector features;
        public final int label;

        public Entry(Vector features, int label) {
            this.features = features;
            this.label = label;
        }
    }
}
