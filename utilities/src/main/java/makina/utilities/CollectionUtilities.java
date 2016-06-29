package makina.utilities;

import java.util.Comparator;
import java.util.Map;
import java.util.TreeMap;

/**
 * @author Emmanouil Antonios Platanios
 */
public class CollectionUtilities {
    public static <K, V extends Comparable<V>> Map<K, V> sortByValue(final Map<K, V> map) {
        Comparator<K> valueComparator = (key1, key2) -> {
            int compare = map.get(key2).compareTo(map.get(key1));
            if (compare == 0) return 1;
            else return compare;
        };
        Map<K, V> sortedByValues = new TreeMap<>(valueComparator);
        sortedByValues.putAll(map);
        return sortedByValues;
    }
}
