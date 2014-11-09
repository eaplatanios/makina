package org.platanios.trade.data;

import javax.persistence.*;
import javax.validation.constraints.NotNull;
import java.util.List;

/**
 * @author Emmanouil Antonios Platanios
 */
@Entity
@Table(name = "Sectors", schema = "", catalog = "trade")
public class Sector {
    private long id;
    private String gicsId;
    private String name;
    private List<IndustryGroup> industryGroups;

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

    @Basic
    @Column(name = "name")
    @NotNull
    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    @OneToMany(mappedBy = "sector", cascade = CascadeType.ALL, orphanRemoval = true)
    public List<IndustryGroup> getIndustryGroups() {
        return industryGroups;
    }

    private void setIndustryGroups(List<IndustryGroup> industryGroups) {
        this.industryGroups = industryGroups;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o)
            return true;
        if (o == null || getClass() != o.getClass())
            return false;

        Sector that = (Sector) o;

        if (id != that.id)
            return false;
        if (!gicsId.equals(that.gicsId))
            return false;
        if (!name.equals(that.name))
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result = (int) (id ^ (id >>> 32));
        result = 31 * result + gicsId.hashCode();
        result = 31 * result + name.hashCode();
        return result;
    }
}
