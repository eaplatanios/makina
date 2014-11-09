package org.platanios.trade.data;

import javax.persistence.*;
import javax.validation.constraints.NotNull;
import java.sql.Timestamp;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
@Entity
@Table(name = "IndustryGroups", catalog = "trade")
public class IndustryGroup {
    private long id;
    private String gicsId;
    private Sector sector;
    private String name;
    private Timestamp dateTimeCreated;
    private Timestamp dateTimeUpdated;
    private List<Industry> industries;

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
    @Column(name = "gics_id", unique = true)
    @NotNull
    public String getGicsId() {
        return gicsId;
    }

    public void setGicsId(String gicsId) {
        this.gicsId = gicsId;
    }

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "sector_id")
    @NotNull
    public Sector getSector() {
        return sector;
    }

    public void setSector(Sector sector) {
        this.sector = sector;
    }

    @Basic
    @Column(name = "name", unique = true)
    @NotNull
    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
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

    @OneToMany(mappedBy = "industryGroup", cascade = CascadeType.ALL, orphanRemoval = true)
    public List<Industry> getIndustries() {
        return industries;
    }

    private void setIndustries(List<Industry> industries) {
        this.industries = industries;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;

        IndustryGroup that = (IndustryGroup) o;

        if (id != that.id)
            return false;
        if (!gicsId.equals(that.gicsId))
            return false;
        if (!sector.equals(that.sector))
            return false;
        if (!name.equals(that.name))
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = (int) (id ^ (id >>> 32));
        result = 31 * result + gicsId.hashCode();
        result = 31 * result + sector.hashCode();
        result = 31 * result + name.hashCode();
        return result;
    }
}
