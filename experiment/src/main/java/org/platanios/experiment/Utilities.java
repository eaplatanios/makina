package org.platanios.experiment;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Utilities {
    public static <K, V> void writeMap(Map<K, V> map, String filename) {
        try {
            FileOutputStream fos = new FileOutputStream(filename);
            ObjectOutputStream oos = new ObjectOutputStream(fos);
            oos.writeObject(map);
            oos.close();
            fos.close();
        } catch (IOException e) {
            System.out.println("An exception was thrown while trying to write a map.");
            e.printStackTrace();
        }
    }

    @SuppressWarnings("unchecked")
    public static <K, V> Map<K, V> readMap(String filename) {
        Map<K, V> map = new HashMap<>();
        try {
            FileInputStream fis = new FileInputStream(filename);
            ObjectInputStream ois = new ObjectInputStream(fis);
            map = (Map<K, V>) ois.readObject();
            ois.close();
            fis.close();
        } catch (IOException|ClassNotFoundException e) {
            System.out.println("An exception was thrown while trying to read a map.");
            e.printStackTrace();
        }
        return map;
    }
}
