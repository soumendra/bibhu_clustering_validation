# bibhu_clustering_validation

## Goal

* Predict energy consumption of households in Jaipur municipal boundary.
    - energy consumption = annual units consumed (we can use either exact units consumed or intervals of units consumed as the target variable)
* Build archetypes (clusters) of electricity consumption patterns in households

## Data dictionary

* Variables of interest (that have been captured)
    - building_age: (intervals) age of the building
        - above 40 years
        - 20 to 40 years
        - below 20 years
    - household_size: (integer) no of people depending on an electricity meter (proxy kitchen)
    - income_class: (categorical) income groups are coming from Pradhanmantri Awas Yojna (PMAY)
        - Group 1: 0-3 LPA
        - Group 2: 3-6 LPA
        - Group 3: 6-12 LPA
        - Group 4: 12-18 LPA
    - builtup_area: (numeric) total builtup area in sq meter
    - bhk: (int) number of bedrooms
    - equipment_load: (numeric) sum of wattage of major equipments in the household (9 major equipments defined, capturing 90% of load according to existing literature)

### Suggested features

* household_size/builtup_area
* equipment_load/builtup_area


# To Do

1. Literature review of predictive models (Bibhu)
     - Size of datasets (no of features, no of datapoints)
     - Is the data publicly available
     - Target values predicted
     - Algorithms used to train models
     - What features usually proved to be significant predictors
2. Literature review of clustering models (Bibhu)


