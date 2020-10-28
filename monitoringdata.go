package monitoringdata

//Author : Vishwanathan Raman
//
//EmailId : datasigntist@gmail.com
//
//Last Modified Date : 18-Oct-2020
//
//Description:
//This module has been developed as an utility for MLOPs to monitor the data at source for any changes.
//More often the data that is being used score the model turns out to be very different from the data that is being used to train the model.
//As a result the model gives subpar results. Hence its important to monitor the scoring data periodically to see if there are any changes in the data.
//If there are changes then the respective team has to be notified so that the model can be tweaked or retrained.
//
//The module does the following and returns it as a map object having the sections for OriginalData, CurrentData and
//StabilityIndex
//
//1) Builds the Elementary Statistics on the individual datasets
//2) Calculates the Stability Index against each feature
//
//It accepts 2 inputs, the first being the contents of the original data and second being the current data.
//The term original data refers to the data that was used for building the machine learning model.
//The term current data refers to the data that is being used for scoring against the machine learning model.
//The main program should use the ReadAll method to read all the data at once and pass in as the values to the function
//GetStatisticsOnData
//
//It auto classifies the type of the data (Categorical or Continuous or Discrete) based on runtime obeservation.
//The current version scans only 1 row of data to determine its type.
//In future a more robust sampling method will be implemented.
//
//Assumption:
//1) The first row in the Data is Column Headers. This is key as the ColumnHeaders are internally used as keys to reecord the observations
//
//2) In the observed data if the number of unique items is less than 5% of the overall data then its marked as Categorical by default
//
//3) It is assumed that the structure of the data both original and current are the same therefore conclusions related to the type of the data is based on the original data which is then applied on the current data.
//

import (
	"strconv"
	"strings"

	"sort"

	"gonum.org/v1/gonum/stat"

	"math"
)

type dataStatistics struct {
	Name                     string
	Datatype                 string
	Mean                     float64
	Median                   float64
	Q1                       float64
	Q3                       float64
	IQR                      float64
	STDEV                    float64
	Max                      float64
	Min                      float64
	UniqueItemsCount         int
	UniqueItems              interface{}
	UniqueItemsCountValues   []float64
	VariableType             string
	DistType                 string
	MissingDataCount         int
	QuantileDataDistribution []float64
	UniqueItemCountAndValues map[string]int
}

var colDatacollectionStructOriginal map[string]dataStatistics
var colDatacollectionStructCurrent map[string]dataStatistics

// GetStatisticsOnData function does the following
// It takes as input the original training data and the current data
// The main program should use the ReadAll method to read all the data at once and pass in as the values
func GetStatisticsOnData(originalRecord [][]string, currentRecord [][]string) map[string]interface{} {

	columnNumbers := len(originalRecord[0])
	colNames := make([]string, columnNumbers)
	totalNumberofRecordsInOriginal := float64(len(originalRecord) - 1)
	totalNumberofRecordsInCurrent := float64(len(originalRecord) - 1)
	dataSummaryAndStatistics := make(map[string]interface{}, 0)

	/*
		colDatacollectionStructOriginalChannel := make(chan map[string]dataStatistics)
		colDatacollectionStructCurrentChannel := make(chan map[string]dataStatistics)
	*/

	for i := 0; i < columnNumbers; i++ {
		colNames[i] = originalRecord[0][i]
	}

	colDatacollectionStructOriginal = processAndCollectStatsOnData(colNames, originalRecord, true)
	colDatacollectionStructCurrent = processAndCollectStatsOnData(colNames, currentRecord, false)

	/*
		go processAndCollectStatsOnDataThroughChannels(colNames, originalRecord, true, colDatacollectionStructOriginalChannel)
		go processAndCollectStatsOnDataThroughChannels(colNames, originalRecord, false, colDatacollectionStructCurrentChannel)

		colDatacollectionStructOriginal = <-colDatacollectionStructOriginalChannel
		colDatacollectionStructCurrent = <-colDatacollectionStructCurrentChannel
	*/

	stabilityIndexValues := calculateOverallStabilityIndex(colNames, totalNumberofRecordsInOriginal, totalNumberofRecordsInCurrent)

	dataSummaryAndStatistics["OriginalData"] = colDatacollectionStructOriginal
	dataSummaryAndStatistics["CurrentData"] = colDatacollectionStructCurrent
	dataSummaryAndStatistics["StabilityIndexValues"] = stabilityIndexValues

	return dataSummaryAndStatistics
}

func calculateOverallStabilityIndex(colNames []string, totalNumberofRecordsInOriginal float64, totalNumberofRecordsInCurrent float64) map[string]float64 {

	stabilityIndexValues := make(map[string]float64, len(colNames))

	for _, colName := range colNames {
		if colDatacollectionStructOriginal[colName].DistType == "Non Unique" {
			if colDatacollectionStructOriginal[colName].VariableType == "Continuous" {
				psi := calculatePopulationStabilityIndexContinuous(colDatacollectionStructOriginal[colName].QuantileDataDistribution, colDatacollectionStructCurrent[colName].QuantileDataDistribution, totalNumberofRecordsInOriginal, totalNumberofRecordsInCurrent)
				stabilityIndexValues[colName] = psi
			} else if colDatacollectionStructOriginal[colName].VariableType == "Categorical" || colDatacollectionStructOriginal[colName].VariableType == "Discrete" {
				psi := calculatePopulationStabilityIndexCategorical(colDatacollectionStructOriginal[colName].UniqueItemCountAndValues, colDatacollectionStructCurrent[colName].UniqueItemCountAndValues, totalNumberofRecordsInOriginal, totalNumberofRecordsInCurrent)
				stabilityIndexValues[colName] = psi
			}
		}
	}

	return stabilityIndexValues

}

func calculatePopulationStabilityIndexCategorical(originalData map[string]int, currentData map[string]int, totalOriginal float64, totalCurrent float64) float64 {

	var psi float64

	for strKey, data := range currentData {
		actual := float64(originalData[strKey]) / float64(totalOriginal)
		expected := float64(data) / float64(totalCurrent)
		psi = psi + ((actual - expected) * math.Log(actual/expected))
	}

	return psi
}

func calculatePopulationStabilityIndexContinuous(originalData []float64, currentData []float64, totalOriginal float64, totalCurrent float64) float64 {

	var psi float64

	for i := 0; i < len(originalData); i++ {
		actual := originalData[i] / totalOriginal
		expected := currentData[i] / totalCurrent
		psi = psi + ((actual - expected) * math.Log(actual/expected))
	}

	return psi
}

func processAndCollectStatsOnDataThroughChannels(colNames []string, record [][]string, isOriginal bool, c chan map[string]dataStatistics) {
	colDatacollectionStruct := make(map[string]dataStatistics)

	totalNumberofRecords := len(record) - 1
	columnNumbers := len(record[0])
	colDataTypes := make(map[string]string)

	for i := 0; i < columnNumbers; i++ {
		colNames[i] = record[0][i]
	}

	for i := 0; i < columnNumbers; i++ {
		if isNumeric(record[1][i]) {
			if isInt(record[1][i]) {
				colDataTypes[colNames[i]] = "int64"
			} else {
				colDataTypes[colNames[i]] = "float64"
			}
		} else {
			colDataTypes[colNames[i]] = "string"
		}
	}

	for col := 0; col < columnNumbers; col++ {

		var tempColDataValues []float64
		var tempColDataValuesString []string
		var missingDataCount int

		if colDataTypes[colNames[col]] != "string" {
			for row := 1; row <= totalNumberofRecords; row++ {

				if strings.TrimSpace(record[row][col]) == "" {
					missingDataCount = missingDataCount + 1
				}
				tempColDataValues = append(tempColDataValues, convertToFloat64(record[row][col]))
			}
			retDS := getStatsOnDataFloat64(colNames[col], tempColDataValues, colDataTypes[colNames[col]], totalNumberofRecords, missingDataCount, isOriginal)

			colDatacollectionStruct[colNames[col]] = retDS

		} else {
			for row := 1; row <= totalNumberofRecords; row++ {

				if strings.TrimSpace(record[row][col]) == "" {
					record[row][col] = strings.TrimSpace(record[row][col])
				}

				tempColDataValuesString = append(tempColDataValuesString, record[row][col])
			}

			retDS := getStatsOnDataString(colNames[col], tempColDataValuesString, colDataTypes[colNames[col]], totalNumberofRecords)

			colDatacollectionStruct[colNames[col]] = retDS

		}
	}

	c <- colDatacollectionStruct

	close(c)

}

func processAndCollectStatsOnData(colNames []string, record [][]string, isOriginal bool) map[string]dataStatistics {

	colDatacollectionStruct := make(map[string]dataStatistics)

	totalNumberofRecords := len(record) - 1
	columnNumbers := len(record[0])
	colDataTypes := make(map[string]string)

	for i := 0; i < columnNumbers; i++ {
		colNames[i] = record[0][i]
	}

	for i := 0; i < columnNumbers; i++ {
		if isNumeric(record[1][i]) {
			if isInt(record[1][i]) {
				colDataTypes[colNames[i]] = "int64"
			} else {
				colDataTypes[colNames[i]] = "float64"
			}
		} else {
			colDataTypes[colNames[i]] = "string"
		}
	}

	for col := 0; col < columnNumbers; col++ {

		var tempColDataValues []float64
		var tempColDataValuesString []string
		var missingDataCount int

		if colDataTypes[colNames[col]] != "string" {
			for row := 1; row <= totalNumberofRecords; row++ {

				if strings.TrimSpace(record[row][col]) == "" {
					missingDataCount = missingDataCount + 1
				}
				tempColDataValues = append(tempColDataValues, convertToFloat64(record[row][col]))
			}
			retDS := getStatsOnDataFloat64(colNames[col], tempColDataValues, colDataTypes[colNames[col]], totalNumberofRecords, missingDataCount, isOriginal)

			colDatacollectionStruct[colNames[col]] = retDS

		} else {
			for row := 1; row <= totalNumberofRecords; row++ {

				if strings.TrimSpace(record[row][col]) == "" {
					record[row][col] = strings.TrimSpace(record[row][col])
				}

				tempColDataValuesString = append(tempColDataValuesString, record[row][col])
			}

			retDS := getStatsOnDataString(colNames[col], tempColDataValuesString, colDataTypes[colNames[col]], totalNumberofRecords)

			colDatacollectionStruct[colNames[col]] = retDS

		}
	}

	return colDatacollectionStruct

}

func countValuesInRange(data []float64, Q1 float64, Q2 float64, Q3 float64, ignoreZeroValues bool) []float64 {

	countedValuesInQuantile := make([]float64, 0)

	dataCountQ1 := 0.0
	dataCountQ2 := 0.0
	dataCountQ3 := 0.0
	dataCountQ4 := 0.0

	for _, num := range data {

		if ignoreZeroValues && num == 0 {
			continue
		}

		if num <= Q1 {
			dataCountQ1 = dataCountQ1 + 1
		} else if num > Q1 && num <= Q2 {
			dataCountQ2 = dataCountQ2 + 1
		} else if num > Q2 && num <= Q3 {
			dataCountQ3 = dataCountQ3 + 1
		} else if num > Q3 {
			dataCountQ4 = dataCountQ4 + 1
		}
	}

	countedValuesInQuantile = append(countedValuesInQuantile, dataCountQ1, dataCountQ2, dataCountQ3, dataCountQ4)

	return countedValuesInQuantile

}

func countUniqueItems(data []float64) ([]float64, []interface{}, int) {

	dataCount := make(map[float64]int)
	uniqueItems := make([]interface{}, 0)
	uniqueItemsCountValues := make([]float64, 0)

	for _, num := range data {
		dataCount[num] = dataCount[num] + 1
	}

	for key, countValue := range dataCount {
		uniqueItems = append(uniqueItems, key)
		uniqueItemsCountValues = append(uniqueItemsCountValues, float64(countValue))
	}

	return uniqueItemsCountValues, uniqueItems, len(dataCount)
}

func countUniqueItemsString(data []string) ([]float64, []interface{}, int, int, map[string]int) {

	dataCount := make(map[string]int)
	uniqueItems := make([]interface{}, 0)
	uniqueItemsCountValues := make([]float64, 0)

	for _, num := range data {
		dataCount[num] = dataCount[num] + 1
	}

	for key, countValue := range dataCount {
		uniqueItems = append(uniqueItems, key)
		uniqueItemsCountValues = append(uniqueItemsCountValues, float64(countValue))
	}

	missingValueCount := dataCount[""]

	return uniqueItemsCountValues, uniqueItems, len(dataCount), missingValueCount, dataCount
}

func isDiscrete(s string) bool {
	_, err := strconv.ParseFloat(s, 64)
	return err == nil
}

func isNumeric(s string) bool {
	_, err := strconv.ParseFloat(s, 64)
	return err == nil
}

func isInt(s string) bool {
	_, err := strconv.ParseInt(s, 10, 64)
	return err == nil
}

func convertToInt64(s string) int64 {
	value, _ := strconv.ParseInt(s, 10, 64)
	return value
}

func convertToFloat64(s string) float64 {
	value, _ := strconv.ParseFloat(s, 64)
	return value
}

func getStatsOnDataString(name string, data []string, datatype string, totalNumberofRecords int) dataStatistics {

	var ds dataStatistics
	var distType string

	uniqueItemsCountValues, uniqueItems, countedUniqueItems, missingValueCount, dataCount := countUniqueItemsString(data)

	if (float64(countedUniqueItems) / float64(totalNumberofRecords)) > 0.9 {
		uniqueItemsCountValues = nil
		uniqueItems = nil
		distType = "Unique"
		dataCount = nil
	} else {
		distType = "Non Unique"

	}
	ds = dataStatistics{
		Name:                     name,
		Datatype:                 datatype,
		UniqueItemsCount:         countedUniqueItems,
		UniqueItems:              uniqueItems,
		VariableType:             "Categorical",
		UniqueItemsCountValues:   uniqueItemsCountValues,
		DistType:                 distType,
		MissingDataCount:         missingValueCount,
		UniqueItemCountAndValues: dataCount,
	}
	return ds
}

func getStatsOnDataFloat64(name string, data []float64, datatype string, totalNumberofRecords int, missingDataCount int, isOriginal bool) dataStatistics {

	// Making an assumption if the number of unique items is less than 5% of the overall data then
	// its Categorical

	var ds dataStatistics
	var Q1, Q2, Q3 float64

	sort.Float64s(data)

	uniqueItemsCountValues, uniqueItems, countedUniqueItems := countUniqueItems(data)

	if countedUniqueItems == totalNumberofRecords {
		ds = dataStatistics{
			Name:             name,
			Datatype:         datatype,
			UniqueItemsCount: countedUniqueItems,
			VariableType:     "Continuous",
			DistType:         "Unique",
		}
	} else if float64(countedUniqueItems) > 0.02*float64(totalNumberofRecords) {

		if isOriginal {
			Q1 = stat.Quantile(0.25, 4, data, nil)
			Q2 = stat.Quantile(0.50, 4, data, nil)
			Q3 = stat.Quantile(0.75, 4, data, nil)
		} else {
			Q1 = colDatacollectionStructOriginal[name].Q1
			Q2 = colDatacollectionStructOriginal[name].Median
			Q3 = colDatacollectionStructOriginal[name].Q3
		}
		ignoreZeroValues := true

		quantileDataDistribution := countValuesInRange(data, Q1, Q2, Q3, ignoreZeroValues)

		ds = dataStatistics{
			Name:                     name,
			Datatype:                 datatype,
			Mean:                     stat.Mean(data, nil),
			Q1:                       Q1,
			Q3:                       Q3,
			IQR:                      Q3 - Q1,
			Median:                   Q2,
			STDEV:                    stat.StdDev(data, nil),
			Max:                      stat.Quantile(1, 4, data, nil),
			Min:                      stat.Quantile(0, 4, data, nil),
			UniqueItemsCount:         countedUniqueItems,
			UniqueItems:              uniqueItems,
			VariableType:             "Continuous",
			UniqueItemsCountValues:   uniqueItemsCountValues,
			DistType:                 "Non Unique",
			MissingDataCount:         missingDataCount,
			QuantileDataDistribution: quantileDataDistribution,
		}
	} else {
		ds = dataStatistics{
			Name:                   name,
			Datatype:               datatype,
			UniqueItemsCount:       countedUniqueItems,
			UniqueItems:            uniqueItems,
			VariableType:           "Discrete",
			UniqueItemsCountValues: uniqueItemsCountValues,
			DistType:               "Non Unique",
			MissingDataCount:       missingDataCount,
		}
	}

	return ds
}
