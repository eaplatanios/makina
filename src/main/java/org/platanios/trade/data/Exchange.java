package org.platanios.trade.data;

import javax.persistence.*;
import javax.validation.constraints.NotNull;
import java.sql.Timestamp;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
@Entity
@Table(name = "Exchanges", schema = "", catalog = "trade")
public class Exchange {
    private long id;
    private String code;
    private String name;
    private String city;
    private String country;
    private String currency;
    private Timestamp datetimeCreated;
    private Timestamp datetimeUpdated;
    private List<Stock> stocksTraded;

    @Id
    @GeneratedValue
    @Column(name = "id")
    public long getId() {
        return id;
    }

    private void setId(long id) {
        this.id = id;
    }

    @Basic
    @Column(name = "code")
    @NotNull
    public String getCode() {
        return code;
    }

    public void setCode(String code) {
        this.code = code;
    }

    @Basic
    @Column(name = "name")
    @NotNull
    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    @Basic
    @Column(name = "city")
    public String getCity() {
        return city;
    }

    public void setCity(String city) {
        this.city = city;
    }

    @Basic
    @Column(name = "country")
    public String getCountry() {
        return country;
    }

    public void setCountry(String country) {
        this.country = country;
    }

    @Basic
    @Column(name = "currency")
    public String getCurrency() {
        return currency;
    }

    public void setCurrency(String currency) {
        this.currency = currency;
    }

    @Basic
    @Column(name = "datetime_created")
    @NotNull
    public Timestamp getDatetimeCreated() {
        return datetimeCreated;
    }

    public void setDatetimeCreated(Timestamp datetimeCreated) {
        this.datetimeCreated = datetimeCreated;
    }

    @Basic
    @Column(name = "datetime_updated")
    @NotNull
    public Timestamp getDatetimeUpdated() {
        return datetimeUpdated;
    }

    public void setDatetimeUpdated(Timestamp datetimeUpdated) {
        this.datetimeUpdated = datetimeUpdated;
    }

    @OneToMany(mappedBy = "primaryExchange", cascade = CascadeType.ALL, orphanRemoval = true)
    public List<Stock> getStocksTraded() {
        return stocksTraded;
    }

    private void setStocksTraded(List<Stock> stocksTraded) {
        this.stocksTraded = stocksTraded;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;

        Exchange that = (Exchange) o;

        if (id != that.id)
            return false;
        if (!code.equals(that.code))
            return false;
        if (!name.equals(that.name))
            return false;
        if (city != null ? !city.equals(that.city) : that.city != null)
            return false;
        if (country != null ? !country.equals(that.country) : that.country != null)
            return false;
        if (currency != null ? !currency.equals(that.currency) : that.currency != null)
            return false;
        if (!datetimeCreated.equals(that.datetimeCreated))
            return false;
        if (!datetimeUpdated.equals(that.datetimeUpdated))
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = (int) (id ^ (id >>> 32));
        result = 31 * result + code.hashCode();
        result = 31 * result + name.hashCode();
        result = 31 * result + (city != null ? city.hashCode() : 0);
        result = 31 * result + (country != null ? country.hashCode() : 0);
        result = 31 * result + (currency != null ? currency.hashCode() : 0);
        result = 31 * result + datetimeCreated.hashCode();
        result = 31 * result + datetimeUpdated.hashCode();
        return result;
    }
}
