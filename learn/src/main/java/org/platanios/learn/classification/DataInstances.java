package org.platanios.learn.classification;

import org.platanios.learn.math.matrix.Vector;

import java.util.List;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DataInstances {
    public static <T extends Vector, S> List<DataInstance<T, S>> getSingleViewDataInstances(
            List<MultiViewDataInstance<T, S>> multiViewDataInstances,
            int view
    ) {
        return multiViewDataInstances
                .stream()
                .map(multiViewDataInstance -> multiViewDataInstance.getSingleViewDataInstance(view))
                .collect(Collectors.toList());
    }
}
