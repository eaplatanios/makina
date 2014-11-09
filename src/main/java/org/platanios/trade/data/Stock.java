package org.platanios.trade.data;

import javax.persistence.*;
import javax.validation.constraints.NotNull;
import java.sql.Timestamp;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
@Entity
@Table(name = "Stocks", schema = "", catalog = "trade")
public class Stock {
    private long id;
    private String cusipId;
    private String cikId;
    private String tickerSymbol;
    private Exchange primaryExchange;
    private String name;
    private SubIndustry subIndustry;
    private Timestamp dateTimeCreated;
    private Timestamp dateTimeUpdated;
    private List<DailyStockData> dailyData;

    @Id
    @GeneratedValue(strategy = GenerationType.AUTO)
    @Column(name = "id")
    public long getId() {
        return id;
    }

    private void setId(long id) {
        this.id = id;
    }

    @Basic
    @Column(name = "cusip_id", unique = true)
    public String getCusipId() {
        return cusipId;
    }

    public void setCusipId(String cusipId) {
        this.cusipId = cusipId;
    }

    @Basic
    @Column(name = "cik_id", unique = true)
    public String getCikId() {
        return cikId;
    }

    public void setCikId(String cikId) {
        this.cikId = cikId;
    }

    @Basic
    @Column(name = "ticker_symbol")
    @NotNull
    public String getTickerSymbol() {
        return tickerSymbol;
    }

    public void setTickerSymbol(String tickerSymbol) {
        this.tickerSymbol = tickerSymbol;
    }

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "primary_exchange_id")
    @NotNull
    public Exchange getPrimaryExchange() {
        return primaryExchange;
    }

    public void setPrimaryExchange(Exchange primaryExchange) {
        this.primaryExchange = primaryExchange;
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

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "sub_industry_id")
    public SubIndustry getSubIndustry() {
        return subIndustry;
    }

    public void setSubIndustry(SubIndustry subIndustry) {
        this.subIndustry = subIndustry;
    }

    @Basic
    @Column(name = "date_time_created")
    @NotNull
    public Timestamp getDateTimeCreated() {
        return dateTimeCreated;
    }

    public void setDateTimeCreated(Timestamp dateTimeCreated) {
        this.dateTimeCreated = dateTimeCreated;
    }

    @Basic
    @Column(name = "date_time_updated")
    @NotNull
    public Timestamp getDateTimeUpdated() {
        return dateTimeUpdated;
    }

    public void setDateTimeUpdated(Timestamp dateTimeUpdated) {
        this.dateTimeUpdated = dateTimeUpdated;
    }

    @OneToMany(mappedBy = "stock", cascade = CascadeType.ALL, orphanRemoval = true)
    public List<DailyStockData> getDailyData() {
        return dailyData;
    }

    private void setDailyData(List<DailyStockData> dailyData) {
        this.dailyData = dailyData;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;

        Stock that = (Stock) o;

        if (id != that.id)
            return false;
        if (cusipId != null ? !cusipId.equals(that.cusipId) : that.cusipId != null)
            return false;
        if (cikId != null ? !cikId.equals(that.cikId) : that.cikId != null)
            return false;
        if (!tickerSymbol.equals(that.tickerSymbol))
            return false;
        if (!primaryExchange.equals(that.primaryExchange))
            return false;
        if (!name.equals(that.name))
            return false;
        if (subIndustry != null ? !subIndustry.equals(that.subIndustry) : that.subIndustry != null)
            return false;
        if (!dateTimeCreated.equals(that.dateTimeCreated))
            return false;
        if (!dateTimeUpdated.equals(that.dateTimeUpdated))
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = (int) (id ^ (id >>> 32));
        result = 31 * result + (cusipId != null ? cusipId.hashCode() : 0);
        result = 31 * result + (cikId != null ? cikId.hashCode() : 0);
        result = 31 * result + tickerSymbol.hashCode();
        result = 31 * result + primaryExchange.hashCode();
        result = 31 * result + name.hashCode();
        result = 31 * result + (subIndustry != null ? subIndustry.hashCode() : 0);
        result = 31 * result + dateTimeCreated.hashCode();
        result = 31 * result + dateTimeUpdated.hashCode();
        return result;
    }
}
