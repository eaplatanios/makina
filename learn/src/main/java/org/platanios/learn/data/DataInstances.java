package org.platanios.learn.data;

import org.platanios.learn.math.matrix.Vector;

import java.util.List;
import java.util.stream.Collectors;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DataInstances {
    public static <T extends Vector> List<? extends DataInstance<T>> getSingleViewDataInstances(
            List<? extends MultiViewDataInstance<T>> multiViewDataInstances,
            int view
    ) {
        return multiViewDataInstances
                .stream()
                .map(multiViewDataInstance -> multiViewDataInstance.getSingleViewDataInstance(view))
                .collect(Collectors.toList());
    }
}
