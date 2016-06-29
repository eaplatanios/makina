package makina.learn.classification;

import makina.utilities.UnsafeSerializationUtilities;

import java.io.IOException;
import java.io.InputStream;

/**
 * @author Emmanouil Antonios Platanios
 */
public class Classifiers {
    public static Classifier read(InputStream inputStream) throws IOException {
        ClassifierType classifierType = ClassifierType.values()[UnsafeSerializationUtilities.readInt(inputStream)];
        return classifierType.read(inputStream, false);
    }
}
