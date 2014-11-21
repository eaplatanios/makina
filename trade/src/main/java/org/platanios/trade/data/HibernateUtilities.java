package org.platanios.trade.data;

import org.hibernate.SessionFactory;
import org.hibernate.boot.registry.StandardServiceRegistryBuilder;
import org.hibernate.cfg.Configuration;
import org.hibernate.service.ServiceRegistry;

/**
 * @author Emmanouil Antonios Platanios
 */
class HibernateUtilities {
    private static SessionFactory sessionFactory = buildSessionFactory();

    protected static SessionFactory buildSessionFactory() {
        Configuration configuration = new Configuration().configure();
//        Configuration configuration =
//                new Configuration()
//                        .setProperty("hibernate.connection.driver_class", "org.mariadb.jdbc.Driver")
//                        .setProperty("hibernate.connection.url", "jdbc:mariadb://localhost")
//                        .setProperty("hibernate.connection.username", "root")
//                        .setProperty("hibernate.connection.password", "")
//                        .setProperty("hibernate.connection.pool_size", "1")
//                        .setProperty("hibernate.dialect", "org.hibernate.dialect.MySQLDialect")
//                        .setProperty("hibernate.current_session_context_class", "thread")
//                        .setProperty("hibernate.show_sql", "false")
//                        .setProperty("hibernate.id.new_generator_mappings", "true")
//                        .setProperty("hibernate.hbm2ddl.auto", "update")
//                        .addAnnotatedClass(DailyStockData.class)
//                        .addAnnotatedClass(DataVendor.class)
//                        .addAnnotatedClass(Exchange.class)
//                        .addAnnotatedClass(Industry.class)
//                        .addAnnotatedClass(IndustryGroup.class)
//                        .addAnnotatedClass(Sector.class)
//                        .addAnnotatedClass(Stock.class)
//                        .addAnnotatedClass(SubIndustry.class);
        ServiceRegistry serviceRegistry =
                new StandardServiceRegistryBuilder().applySettings(configuration.getProperties()).build();
        sessionFactory = configuration.buildSessionFactory(serviceRegistry);
        return sessionFactory;
    }

    protected static SessionFactory getSessionFactory() {
        return sessionFactory;
    }
}
