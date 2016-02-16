package org.platanios.learn.neural.network;

import org.platanios.learn.math.matrix.Vector;

import java.util.ArrayList;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
public class NetworkLearner {
    private final Network network;
    private final State state;

    public NetworkLearner(Network network) {
        this.network = network;
        this.state = new State();
    }

    // TODO: Implement this.
    public Network train(TrainingData data) {
        return null;
    }

    public static class TrainingData {
        private final List<Vector> inputs = new ArrayList<>();
        private final List<Vector> outputs = new ArrayList<>();

        private final int inputSize;
        private final int outputSize;

        public TrainingData(int inputSize, int outputSize) {
            this.inputSize = inputSize;
            this.outputSize = outputSize;
        }

        public TrainingData addExample(Vector input, Vector output) {
            if (input.size() != inputSize)
                throw new IllegalArgumentException("The input vector size must be the same for all training data.");
            if (output.size() != outputSize)
                throw new IllegalArgumentException("The output vector size must be the same for all training data.");
            inputs.add(input);
            outputs.add(output);
            return this;
        }
    }
}
