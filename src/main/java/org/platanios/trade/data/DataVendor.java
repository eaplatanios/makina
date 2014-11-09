package org.platanios.trade.data;

import javax.persistence.*;
import javax.validation.constraints.NotNull;
import javax.validation.constraints.Pattern;
import java.sql.Timestamp;

/**
 * @author Emmanouil Antonios Platanios
 */
@Entity
@Table(name = "DataVendors", schema = "", catalog = "trade")
public class DataVendor {
    private long id;
    private String name;
    private String abbreviation;
    private String websiteUrl;
    private String supportEmail;
    private Timestamp datetimeCreated;
    private Timestamp datetimeUpdated;

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
    @Column(name = "name")
    @NotNull
    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    @Basic
    @Column(name = "abbreviation")
    @NotNull
    public String getAbbreviation() {
        return abbreviation;
    }

    public void setAbbreviation(String abbreviation) {
        this.abbreviation = abbreviation;
    }

    @Basic
    @Column(name = "website_url")
    public String getWebsiteUrl() {
        return websiteUrl;
    }

    public void setWebsiteUrl(String websiteUrl) {
        this.websiteUrl = websiteUrl;
    }

    @Basic
    @Column(name = "support_email")
    @Pattern(regexp = "[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\\."
            + "[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@"
            + "(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\\.)+[a-z0-9]"
            + "(?:[a-z0-9-]*[a-z0-9])?",
            message = "{invalid.email}")
    public String getSupportEmail() {
        return supportEmail;
    }

    public void setSupportEmail(String supportEmail) {
        this.supportEmail = supportEmail;
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

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;

        DataVendor that = (DataVendor) o;

        if (id != that.id)
            return false;
        if (!name.equals(that.name))
            return false;
        if (!abbreviation.equals(that.abbreviation))
            return false;
        if (websiteUrl != null ? !websiteUrl.equals(that.websiteUrl) : that.websiteUrl != null)
            return false;
        if (supportEmail != null ? !supportEmail.equals(that.supportEmail) : that.supportEmail != null)
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
        result = 31 * result + name.hashCode();
        result = 31 * result + abbreviation.hashCode();
        result = 31 * result + (websiteUrl != null ? websiteUrl.hashCode() : 0);
        result = 31 * result + (supportEmail != null ? supportEmail.hashCode() : 0);
        result = 31 * result + datetimeCreated.hashCode();
        result = 31 * result + datetimeUpdated.hashCode();
        return result;
    }
}
