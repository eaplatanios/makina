package org.platanios.learn.classification;

import org.platanios.learn.math.matrix.Vector;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.io.ObjectInputStream;
import java.io.Serializable;
import java.sql.Connection;

/**
 * @author Emmanouil Antonios Platanios.
 */
public enum StorageType {
    IN_MEMORY {
        @Override
        public <T extends Vector, R extends Serializable> Storage<T, R> build(ObjectInputStream inputStream) {
            return new InMemoryStorage<T, R>().loadFeatureMap(inputStream);
        }

        @Override
        public <T extends Vector, R extends Serializable> Storage<T, R> build(Connection databaseConnection) {
            throw new NotImplementedException();
        }
    },
    MARIADB {
        @Override
        public <T extends Vector, R extends Serializable> Storage<T, R> build(ObjectInputStream inputStream) {
            throw new NotImplementedException();
        }

        @Override
        public <T extends Vector, R extends Serializable> Storage<T, R> build(Connection databaseConnection) {
            throw new NotImplementedException();
        }
    };

    public abstract <T extends Vector, R extends Serializable> Storage<T, R> build(ObjectInputStream inputStream);
    public abstract <T extends Vector, R extends Serializable> Storage<T, R> build(Connection databaseConnection);
}
