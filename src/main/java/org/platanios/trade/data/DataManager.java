package org.platanios.trade.data;

import org.hibernate.Session;

import java.sql.Timestamp;
import java.util.Date;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DataManager {
    public static class ExchangeBuilder {
        private final String code;
        private final String name;

        private String city = null;
        private String country = null;
        private String currency = null;

        public ExchangeBuilder(String code, String name) {
            this.code = code;
            this.name = name;
        }

        public ExchangeBuilder city(String city) {
            this.city = city;
            return this;
        }

        public ExchangeBuilder country(String country) {
            this.country = country;
            return this;
        }

        public ExchangeBuilder currency(String currency) {
            this.currency = currency;
            return this;
        }

        protected Exchange build() {
            Exchange exchange = new Exchange();
            exchange.setCode(code);
            exchange.setName(name);
            exchange.setCity(city);
            exchange.setCountry(country);
            exchange.setCurrency(currency);
            Timestamp currentDateTime = new Timestamp(new Date().getTime());
            exchange.setDatetimeCreated(currentDateTime);
            exchange.setDatetimeUpdated(currentDateTime);
            return exchange;
        }
    }

    private void createAndStoreExchange(ExchangeBuilder exchangeBuilder) {
        Session session = HibernateUtilities.getSessionFactory().getCurrentSession();
        session.beginTransaction();
        session.save(exchangeBuilder.build());
        session.getTransaction().commit();
    }

    public void initializeDatabase() {
        createAndStoreExchange(new ExchangeBuilder("N", "NYSE").city("New York").country("USA").currency("USD"));
        createAndStoreExchange(new ExchangeBuilder("A", "NYSE MKT").city("New York").country("USA").currency("USD"));
        createAndStoreExchange(new ExchangeBuilder("R", "NYSE Arca").city("New York").country("USA").currency("USD"));
        createAndStoreExchange(new ExchangeBuilder("Q", "NASDAQ").city("New York").country("USA").currency("USD"));
        createAndStoreExchange(new ExchangeBuilder("X", "Other").city("New York").country("USA").currency("USD"));
    }

    public static void main(String[] args) {
        DataManager dataManager = new DataManager();
        dataManager.initializeDatabase();
    }
}
