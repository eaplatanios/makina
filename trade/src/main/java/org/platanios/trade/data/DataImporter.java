package org.platanios.trade.data;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Properties;
import java.util.stream.Stream;

/**
 * @author Emmanouil Antonios Platanios
 */
public class DataImporter {
    private static final Logger logger = LogManager.getLogger("Data Importer");
    private static final Properties properties = new Properties();
    private static final String gicsDataFile;
    static {
        try {
            properties.load(ClassLoader.getSystemResourceAsStream("DataImporter.properties"));
            gicsDataFile = properties.getProperty("gicsDataPath");
        } catch (IOException e) {
            logger.error("Could not load the data importer properties.", e);
            throw new RuntimeException(e);
        }
    }

    public static void importGICSData() {
        try (Stream<String> lines = Files.lines(Paths.get(gicsDataFile), Charset.defaultCharset())) {
            Iterator<String> linesIterator = lines.iterator();
            List<Sector> sectors = new ArrayList<>();
            List<IndustryGroup> industryGroups = new ArrayList<>();
            List<Industry> industries = new ArrayList<>();
            List<SubIndustry> subIndustries = new ArrayList<>();
            while (linesIterator.hasNext()) {
                String[] lineParts = linesIterator.next().split("\t");
                if (!lineParts[0].equals(""))
                    sectors.add(new Sector.Builder(Integer.parseInt(lineParts[0]), lineParts[1]).build());
                if (!lineParts[2].equals(""))
                    industryGroups.add(new IndustryGroup.Builder(Integer.parseInt(lineParts[2]),
                                                                 sectors.get(sectors.size() - 1),
                                                                 lineParts[3]).build());
                if (!lineParts[4].equals(""))
                    industries.add(new Industry.Builder(Integer.parseInt(lineParts[4]),
                                                        industryGroups.get(industryGroups.size() - 1),
                                                        lineParts[5]).build());
                subIndustries.add(new SubIndustry.Builder(Integer.parseInt(lineParts[6]),
                                                          industries.get(industries.size() - 1),
                                                          lineParts[7]).description(lineParts[8]).build());
            }
            DataManager.addSectors(sectors);
            DataManager.addIndustryGroups(industryGroups);
            DataManager.addIndustries(industries);
            DataManager.addSubIndustries(subIndustries);
        } catch (IOException e) {
            logger.error("Could not load the data importer properties.", e);
            throw new RuntimeException(e);
        }
    }
}
