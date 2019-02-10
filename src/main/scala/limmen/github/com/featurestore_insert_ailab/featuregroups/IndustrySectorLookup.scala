package limmen.github.com.feature_engineering_spark.featuregroup

import org.apache.log4j.{ Level, LogManager, Logger }
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.monotonically_increasing_id
import io.hops.util.Hops
import scala.collection.JavaConversions._
import collection.JavaConverters._
import org.apache.spark.sql.Row

/**
 * Contains logic for computing the industry_sector_lookup featuregroup
 */
object IndustrySectorLookup {

  /**
   * Computes the featuregroup
   *
   * @param input path to the input dataset to read (csv)
   * @param output name of the output featuregroup table
   * @param version version of the featuregroup
   * @param partitions number of spark partitions to parallelize the compute on
   * @param logger spark logger
   * @param create boolean flag whether to create new featuregroup or insert into an existing ones
   * @param jobId the id of the hopsworks job
   * @param keepInMemory if true the lookup map is kept in memory and not serialized to Hive
   */
  def computeFeatures(spark: SparkSession, input: String, featuregroupName: String,
    version: Int, partitions: Int, log: Logger, create: Boolean, jobId: Int, keepInMemory: Boolean = false): Map[String, Long] = {
    log.info(s"Running computeFeatures for featuregroup: ${featuregroupName}")
    val rawDf = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(input).repartition(partitions)
    val industrySectors = rawDf.select("industry_sector").distinct
    val industrySectorsWithIndex = industrySectors.withColumn("id", monotonically_increasing_id())
    log.info("Extracted industry sectors and mapped to ids:")
    log.info(industrySectorsWithIndex.show(5))
    log.info("Schema: \n" + industrySectorsWithIndex.printSchema)
    if (keepInMemory) {
      val industrySectorLookupList: Array[Row] = industrySectorsWithIndex.collect
      val industrySectorLookupMap = industrySectorLookupList.map((row: Row) => {
        row.getAs[String]("industry_sector") -> row.getAs[Long]("id")
      }).toMap
      return industrySectorLookupMap
    }
    val featurestore = Hops.getProjectFeaturestore
    val descriptiveStats = true
    val featureCorr = false
    val featureHistograms = true
    val clusterAnalysis = false
    val statColumns = List[String]().asJava
    val numBins = 20
    val corrMethod = "pearson"
    val numClusters = 5
    val description = "lookup table for id to industry_sector, used when converting from numeric to categorical representation and vice verse"
    val primaryKey = "id"
    val dependencies = List[String](input).asJava
    if (create) {
      log.info(s"Creating featuregroup $featuregroupName version $version in featurestore $featurestore")
      Hops.createFeaturegroup(spark, industrySectorsWithIndex, featuregroupName, featurestore, version, description,
        jobId, dependencies, primaryKey, descriptiveStats, featureCorr, featureHistograms, clusterAnalysis,
        statColumns, numBins, corrMethod, numClusters)
      log.info(s"Creation of featuregroup $featuregroupName complete")
    } else {
      log.info(s"Inserting into featuregroup $featuregroupName version $version in featurestore $featurestore")
      Hops.insertIntoFeaturegroup(spark, industrySectorsWithIndex, featuregroupName, featurestore, version, "overwrite",
        descriptiveStats, featureCorr, featureHistograms, clusterAnalysis, statColumns, numBins, corrMethod,
        numClusters)
      log.info(s"Insertion into featuregroup $featuregroupName complete")
    }
    return null
  }
}
