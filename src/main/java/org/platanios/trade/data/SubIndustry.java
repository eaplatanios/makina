package org.platanios.trade.data;

import javax.persistence.*;
import javax.validation.constraints.NotNull;

/**
 * @author Emmanouil Antonios Platanios
 */
@Entity
@Table(name = "SubIndustries", schema = "", catalog = "trade")
public class SubIndustry {
    private long id;
    private String gicsId;
    private Industry industry;
    private String name;
    private String description;

    @Id
    @Column(name = "id")
    public long getId() {
        return id;
    }

    private void setId(long id) {
        this.id = id;
    }

    @Basic
    @Column(name = "gics_id")
    @NotNull
    public String getGicsId() {
        return gicsId;
    }

    public void setGicsId(String gicsId) {
        this.gicsId = gicsId;
    }

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "industry_id")
    @NotNull
    public Industry getIndustry() {
        return industry;
    }

    public void setIndustry(Industry industry) {
        this.industry = industry;
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
    @Column(name = "description")
    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;

        SubIndustry that = (SubIndustry) o;

        if (id != that.id)
            return false;
        if (!gicsId.equals(that.gicsId))
            return false;
        if (!industry.equals(that.industry))
            return false;
        if (!name.equals(that.name))
            return false;
        if (description != null ? !description.equals(that.description) : that.description != null)
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = (int) (id ^ (id >>> 32));
        result = 31 * result + gicsId.hashCode();
        result = 31 * result + industry.hashCode();
        result = 31 * result + name.hashCode();
        result = 31 * result + (description != null ? description.hashCode() : 0);
        return result;
    }
}
