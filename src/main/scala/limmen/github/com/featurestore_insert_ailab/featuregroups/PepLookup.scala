package limmen.github.com.feature_engineering_spark.featuregroup

import org.apache.log4j.{ Level, LogManager, Logger }
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.monotonically_increasing_id
import io.hops.util.Hops
import org.apache.spark.sql.Row
import scala.collection.JavaConversions._
import collection.JavaConverters._
import org.apache.spark.sql.Row

/**
 * Contains logic for computing the pep_lookup featuregroup
 */
object PepLookup {

  /**
   * Computes the featuregroup
   *
   * @param spark the spark session
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
    version: Int, partitions: Int, log: Logger, create: Boolean, jobId: Int, keepInMemory: Boolean = false): Map[Boolean, Long] = {
    log.info(s"Running computeFeatures for featuregroup: ${featuregroupName}")
    val rawDf = spark.read.format("csv").option("header", "true").load(input).repartition(partitions)
    import spark.implicits._
    val peps = rawDf.select("pep").distinct
    val pepsWithIndex = peps.withColumn("id", monotonically_increasing_id())
    if (keepInMemory) {
      val pepLookupList: Array[Row] = pepsWithIndex.collect
      val pepLookupMap = pepLookupList.map((row: Row) => {
        val pep = row.getAs[String]("pep")
        pep match {
          case "True" => true -> row.getAs[Long]("id")
          case "False" => false -> row.getAs[Long]("id")
        }
      }).toMap
      return pepLookupMap
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
    val description = "lookup table for id to pep type, used when converting from numeric to categrorical representation and vice verse"
    val primaryKey = "id"
    val dependencies = List[String](input).asJava
    if (create) {
      log.info(s"Creating featuregroup $featuregroupName version $version in featurestore $featurestore")
      Hops.createFeaturegroup(spark, pepsWithIndex, featuregroupName, featurestore, version, description,
        jobId, dependencies, primaryKey, descriptiveStats, featureCorr, featureHistograms, clusterAnalysis,
        statColumns, numBins, corrMethod, numClusters)
      log.info(s"Creation of featuregroup $featuregroupName complete")
    } else {
      log.info(s"Inserting into featuregroup $featuregroupName version $version in featurestore $featurestore")
      Hops.insertIntoFeaturegroup(spark, pepsWithIndex, featuregroupName, featurestore, version, "overwrite",
        descriptiveStats, featureCorr, featureHistograms, clusterAnalysis, statColumns, numBins, corrMethod,
        numClusters)
      log.info(s"Insertion into featuregroup $featuregroupName complete")
    }
    return null
  }
}
