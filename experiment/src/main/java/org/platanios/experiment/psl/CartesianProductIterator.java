package org.platanios.experiment.psl;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * Created by Dan on 4/26/2015.
 * Iterates over all combinations of items in each of the lists
 */
public class CartesianProductIterator<T> implements Iterable<List<T>> {

    public CartesianProductIterator(List<List<T>> lists) {
        this.lists = lists;
    }

    public Iterator<List<T>> iterator() {
        return new IteratorState();
    }

    private List<List<T>> lists;

    private class IteratorState implements Iterator<List<T>> {

        public IteratorState() {
            int virtualLengthTemp = 1;
            for (int i = 0; i < CartesianProductIterator.this.lists.size(); ++i) {
                virtualLengthTemp *= CartesianProductIterator.this.lists.get(i).size();
            }
            this.virtualLength = virtualLengthTemp;
            this.virtualIndex = 0;
        }

        private int virtualIndex;
        private final int virtualLength;
        private int[] indices = new int[CartesianProductIterator.this.lists.size()];

        public boolean hasNext() {
            return this.virtualIndex < this.virtualLength;
        }

        public List<T> next() {

            ++this.virtualIndex;
            if(this.virtualIndex == 1) {
                return this.getResult();
            }

            for (int i = 0; i < this.indices.length; ++i) {
                ++this.indices[i];
                if (this.indices[i] < CartesianProductIterator.this.lists.get(i).size()) {
                    return this.getResult();
                } else {
                    this.indices[i] = 0;
                }
            }

            throw new UnsupportedOperationException("Beyond the end");
        }

        private List<T> getResult() {
            ArrayList<T> result = new ArrayList<>();
            for (int j = 0; j < this.indices.length; ++j) {
                result.add(CartesianProductIterator.this.lists.get(j).get(this.indices[j]));
            }
            return result;
        }
    }

}
