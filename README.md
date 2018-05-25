
# Investigating Income Inequality, Corruption, Health and Literacy

### Author: David Oguara
** May 2018 **

## Table of Contents
<ul>
<li><a href="#intro">Introduction</a></li>
<li><a href="#wrangling">Data Wrangling</a></li>
<li><a href="#eda">Exploratory Data Analysis</a></li>
<li><a href="#conclusions">Conclusions</a></li>
</ul>

<a id='intro'></a>
## Introduction

For this project, we will be investigating income inequality, corruption perception, literacy and health. We will consider which countries are the best and worst in each of these metrics, and seek to uncover relationships between these metrics within countries of the world.


#### Dataset Selection

Gapminder data on literacy, poverty and inequality, and incomes and growth has been selected for use in this analysis. This data comes from multiple original sources, including World Bank, UNESCO, and Transparency International. 

For consistency, we have selected a five-year period, 2007 to 2011, as the active window for investigation. Where gapminder is missing data within this window, we have sourced data directly from the source organization. 

This analysis is not focused on trends over time; rather, it seeks to identify relationships among variables related to socio-economic indicators. Therefore, in most cases, averages over this five-year window are used to investigate these correlations. Five years is seen as short enough to minimise changes in trendline for each variable, and long enough that gaps in data can be filled from alternate years.

CSV files and their sources are listed below:

>**Dataset** / Provider / Filename csv

>**Corruption Perception Index (CPI)** / Transparency International / corruption_2008_2009.csv, corruption_2007_2010_2011.csv

>**Infant Mortality Rate (rate per 1000 births)** / Various sources / infant.csv

>**Literacy rate, adult total (% of people ages 15 and above)** / UNESCO / literacy.csv

>**Inequality index (Gini)** / The World Bank (WEBSITE) / inequality.csv

>**Poverty (% people below USD1.90 a day)** / The World Bank / poverty_2usd.csv

>**GDP/capita (USD, inflation-adjusted)** / The World Bank / gdp.csv

#### Investigation Questions

The following questions will be investigated in this analysis.

**Q1. Which five countries have the highest and lowest income inequality?**

**Q2. Is there a relationship between income inequality and corruption?**

**Q3. Is GDP per capita a good predictor of income inequality?**

**Q4. Does a correlation exist between literacy and health?**

The packages inported below are required to conduct this analysis.


```python
#Import packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline

```

<a id='wrangling'></a>
## Data Wrangling

In this section, we will load the data, inspect it for fitness, and then perform any necessary cleaning and trimming operations to get the data ready for analysis.

### General Properties

**Corruption Perception Index (CPI) - Data Inspection**

From Transparency International website https://www.transparency.org/cpi2011/results: 

>The Corruption Perception Index ranks countries/territories based on how corrupt their public sector is perceived to be. A country/territory’s score indicates the perceived level of public sector corruption on a scale of 0 - 10, where 0 means that a country is perceived as highly corrupt and 10 means that a country is perceived as very clean.


```python
# Import Gapminder cpi data

df_corr_a = pd.read_csv('corruption_2008_2009.csv')
df_corr_a.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>2008</th>
      <th>2009</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>New Zealand</td>
      <td>9.4</td>
      <td>9.3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Denmark</td>
      <td>9.3</td>
      <td>9.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sweden</td>
      <td>9.2</td>
      <td>9.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Singapore</td>
      <td>9.2</td>
      <td>9.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Switzerland</td>
      <td>9.0</td>
      <td>8.7</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check dimensions

df_corr_a.shape
```




    (180, 3)




```python
df_corr_a.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 180 entries, 0 to 179
    Data columns (total 3 columns):
    Country    180 non-null object
    2008       180 non-null float64
    2009       176 non-null float64
    dtypes: float64(2), object(1)
    memory usage: 4.3+ KB
    

Gapminder data only contains 2008 and 2009 data. Note that this data is stored in table format, with years as columns. It will need to be converted to long format, with years in a single column. 

Further, our analysis requires 2007 to 2011 data. We have sourced the missing years data from Transparency International website, https://www.transparency.org/research/cpi/overview


```python
# Import missing years data

df_corr_b = pd.read_csv('corruption_2007_2010_2011.csv')
df_corr_b.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>cpi_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Denmark</td>
      <td>2007</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Finland</td>
      <td>2007</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>New Zealand</td>
      <td>2007</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Singapore</td>
      <td>2007</td>
      <td>9.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sweden</td>
      <td>2007</td>
      <td>9.3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Confirm data for desired years is present

df_corr_b.year.value_counts()
```




    2011    183
    2007    180
    2010    178
    Name: year, dtype: int64




```python
# Inspect data types

df_corr_b.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 541 entries, 0 to 540
    Data columns (total 3 columns):
    country      541 non-null object
    year         541 non-null int64
    cpi_score    541 non-null float64
    dtypes: float64(1), int64(1), object(1)
    memory usage: 12.8+ KB
    

Data for 2007, 2010, 2011 is stored in ideal long format. Tasks for this dataset will be: 
- transform gapminder subset data into long format
- check and convert data types as needed
- combine gapminder data with manually downloaded missing years data.

Let us proceed to import and inspect the other datasets and identify any data-cleansing tasks.

#### Infant Mortality Rate (rate per 1000 births) - Data Inspection

According to a 2003 paper published in Journal of Epidiemology and Public Health:

>The infant mortality rate (IMR), defined as the number of deaths in children under 1 year of age per 1000 live births in the same year, has in the past been regarded as a highly sensitive (proxy) measure of population health.2 This reflects the apparent association between the causes of infant mortality and other factors that are likely to influence the health status of whole populations such as their economic development, general living conditions, social well being, rates of illness, and the quality of the environment. 

Source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1732453/

**We have therefore selected infant mortality rate as a measure of whole-country population health for this investigation.**

Data has been sourced from Gapminder.


```python
# Import infant_mortality data
df_infant = pd.read_csv('infant.csv')
df_infant.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Infant mortality rate</th>
      <th>1800</th>
      <th>1801</th>
      <th>1802</th>
      <th>1803</th>
      <th>1804</th>
      <th>1805</th>
      <th>1806</th>
      <th>1807</th>
      <th>1808</th>
      <th>...</th>
      <th>2006</th>
      <th>2007</th>
      <th>2008</th>
      <th>2009</th>
      <th>2010</th>
      <th>2011</th>
      <th>2012</th>
      <th>2013</th>
      <th>2014</th>
      <th>2015</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Abkhazia</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>82.3</td>
      <td>80.4</td>
      <td>78.6</td>
      <td>76.8</td>
      <td>75.1</td>
      <td>73.4</td>
      <td>71.7</td>
      <td>69.9</td>
      <td>68.1</td>
      <td>66.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Akrotiri and Dhekelia</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albania</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>17.4</td>
      <td>16.7</td>
      <td>16.0</td>
      <td>15.4</td>
      <td>14.8</td>
      <td>14.3</td>
      <td>13.8</td>
      <td>13.3</td>
      <td>12.9</td>
      <td>12.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Algeria</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>27.6</td>
      <td>26.4</td>
      <td>25.3</td>
      <td>24.3</td>
      <td>23.5</td>
      <td>22.8</td>
      <td>22.4</td>
      <td>22.1</td>
      <td>22.0</td>
      <td>21.9</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 217 columns</p>
</div>



#### Literacy rate, adult total (% of people ages 15 and above) - Data Inspection

Adult literacy rate is the percentage of people aged 15 years and above who can read and write. Data is almost entirely collected by UNESCO Institute for Statistis on behalf of UNESCO, and is collated by mostly surveys wherein participants self-declare. The global literacy rate for adults 15 years and older is 86% as at 2015.

Our dataset will investigate average literacy from 2007 to 2011, and identify any relationships to other metrics like health, income inequality and corruption perception. 


```python
# Import literacy data
df_literacy = pd.read_csv('literacy.csv')
df_literacy.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Adult (15+) literacy rate (%). Total</th>
      <th>1975</th>
      <th>1976</th>
      <th>1977</th>
      <th>1978</th>
      <th>1979</th>
      <th>1980</th>
      <th>1981</th>
      <th>1982</th>
      <th>1983</th>
      <th>...</th>
      <th>2002</th>
      <th>2003</th>
      <th>2004</th>
      <th>2005</th>
      <th>2006</th>
      <th>2007</th>
      <th>2008</th>
      <th>2009</th>
      <th>2010</th>
      <th>2011</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>18.157681</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>39.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albania</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>95.93864</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>96.845299</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>69.8735</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>72.648679</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Andorra</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Angola</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>70.362420</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>



#### Inequality index (Gini) - Data Inspection

World Bank website, https://data.worldbank.org/indicator/SI.POV.GINI explains the GINI index:
 
> Gini index measures the extent to which the distribution of income (or, in some cases, consumption expenditure) among individuals or households within an economy deviates from a perfectly equal distribution. A Lorenz curve plots the cumulative percentages of total income received against the cumulative number of recipients, starting with the poorest individual or household. The Gini index measures the area between the Lorenz curve and a hypothetical line of absolute equality, expressed as a percentage of the maximum area under the line. Thus a Gini index of 0 represents perfect equality, while an index of 100 implies perfect inequality.

Gapminder data only goes to 2010. For convenience, GINI data 2007 to 2011 is downloaded from  the Word Bank website above. This will allow us to match time period of data in other comparison variables.


```python
# Import inequality index data data
df_inequality = pd.read_csv('inequality.csv')
df_inequality.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country Name</th>
      <th>1960</th>
      <th>1961</th>
      <th>1962</th>
      <th>1963</th>
      <th>1964</th>
      <th>1965</th>
      <th>1966</th>
      <th>1967</th>
      <th>1968</th>
      <th>...</th>
      <th>2008</th>
      <th>2009</th>
      <th>2010</th>
      <th>2011</th>
      <th>2012</th>
      <th>2013</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aruba</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Angola</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>42.7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albania</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Andorra</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 59 columns</p>
</div>



#### Poverty (% people below USD1.90 a day)


According to World Bank website https://data.worldbank.org/indicator/SI.POV.DDAY:

> Poverty headcount ratio at USD1.90 a day is the percentage of the population living on less than USD1.90 a day at 2011 international prices. As a result of revisions in PPP exchange rates, poverty rates for individual countries cannot be compared with poverty rates reported in earlier editions.

Given the above, we will select and use data for 2011 only


```python
# Import poverty index data data
df_poverty = pd.read_csv('poverty_below1_90usd.csv')
df_poverty.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country Name</th>
      <th>1960</th>
      <th>1961</th>
      <th>1962</th>
      <th>1963</th>
      <th>1964</th>
      <th>1965</th>
      <th>1966</th>
      <th>1967</th>
      <th>1968</th>
      <th>...</th>
      <th>2008</th>
      <th>2009</th>
      <th>2010</th>
      <th>2011</th>
      <th>2012</th>
      <th>2013</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aruba</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Angola</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>30.1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albania</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>0.4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Andorra</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 59 columns</p>
</div>




```python
# Select only 2011 data
df_poverty = df_poverty.loc[:,['Country Name','2011']]
df_poverty.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country Name</th>
      <th>2011</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aruba</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Angola</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albania</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Andorra</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



#### GDP/capita (USD, inflation-adjusted)

According to Investopedia website:

>Per capita GDP is a measure of the total output of a country that takes the gross domestic product (GDP) and divides it by the number of people in that country. The per capita GDP is especially useful when comparing one country to another, because it shows the relative performance of the countries. A rise in per capita GDP signals growth in the economy and tends to reflect an increase in productivity.

Read more: Per Capita GDP https://www.investopedia.com/terms/p/per-capita-gdp.asp#ixzz5ER0jDGOi 

GDP per capita data is sourced entirely from Gapminder 


```python
# Import gdp per capita data
df_gdp = pd.read_csv('gdp.csv')
df_gdp.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Income per person (fixed 2000 US$)</th>
      <th>1960</th>
      <th>1961</th>
      <th>1962</th>
      <th>1963</th>
      <th>1964</th>
      <th>1965</th>
      <th>1966</th>
      <th>1967</th>
      <th>1968</th>
      <th>...</th>
      <th>2002</th>
      <th>2003</th>
      <th>2004</th>
      <th>2005</th>
      <th>2006</th>
      <th>2007</th>
      <th>2008</th>
      <th>2009</th>
      <th>2010</th>
      <th>2011</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Abkhazia</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Akrotiri and Dhekelia</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albania</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>1313.722725</td>
      <td>1381.040832</td>
      <td>1454.022854</td>
      <td>1525.723589</td>
      <td>1594.495067</td>
      <td>1681.613910</td>
      <td>1804.419415</td>
      <td>1857.352947</td>
      <td>1915.424459</td>
      <td>1965.707230</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Algeria</td>
      <td>1280.384828</td>
      <td>1085.414612</td>
      <td>855.947986</td>
      <td>1128.41578</td>
      <td>1170.323896</td>
      <td>1215.015783</td>
      <td>1127.614288</td>
      <td>1200.558225</td>
      <td>1291.863983</td>
      <td>...</td>
      <td>1871.921986</td>
      <td>1971.512803</td>
      <td>2043.135713</td>
      <td>2115.186028</td>
      <td>2124.957754</td>
      <td>2155.485231</td>
      <td>2173.787903</td>
      <td>2192.703976</td>
      <td>2231.980246</td>
      <td>2255.225482</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 53 columns</p>
</div>



We have now imported all the data we need for our analysis. During this process we have made a number of observations about the data, and identified initial actions that will need to be carried out to bring the data into a fit state for analysis.

### Data Cleaning

These are our observations on imported data, and the cleaning actions that will need to be performed.

 - Data is held in table format and will need to be converted to long format, so that years appear in a single column
 - Some data will need to be combined, particularly corruption perception data
 - Since we only require 2007 to 2011, data will need to be trimmed to this window
 - Data types will need to be checked and converted as needed
 - The column names for the metrics we wish to investigate will need to be renamed for ease of referencing
 
 Let us now proceed to carry out these cleaning operations.

#### Corruption Perception Index (CPI) - Data Cleaning


```python
# Transpose Gapminder 2008/2009 data into long format, using pd.melt() function

df_corr_a = pd.melt(df_corr_a, id_vars='Country', var_name='year', value_name='cpi_score')
```


```python
# Rename columns to lowercase to match both datasets
# Recall that df_corr_a holds 2008 & 2009 data, and df_corr_b contains data for missing years (2007, 2010, 2011)

df_corr_a.rename(columns=lambda x: x.lower(), inplace=True);
df_corr_b.rename(columns=lambda x: x.lower().replace(' ', '_'), inplace=True);
```


```python
# Check both corruption_perception dataset columns are aligned

df_corr_a.columns == df_corr_b.columns
```




    array([ True,  True,  True])




```python
# Combine corruption_perception datasets into single dataframe

df_corruption = df_corr_a.append(df_corr_b, ignore_index=True)
```


```python
# Check number of records in combined dataframe

df_corruption.shape
```




    (901, 3)




```python
# Confirm all years data present

df_corruption.year.value_counts()
```




    2011    183
    2007    180
    2009    180
    2008    180
    2010    178
    Name: year, dtype: int64




```python
# Check datatypes

df_corruption.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 901 entries, 0 to 900
    Data columns (total 3 columns):
    country      901 non-null object
    year         901 non-null object
    cpi_score    897 non-null float64
    dtypes: float64(1), object(2)
    memory usage: 21.2+ KB
    


```python
# Inspect 'NaN' records
df_corruption[pd.isnull(df_corruption).any(axis=1)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>cpi_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>201</th>
      <td>Saint Lucia</td>
      <td>2009</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>210</th>
      <td>Saint Vincent and the Grenadines</td>
      <td>2009</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>219</th>
      <td>Brunei Darussalam</td>
      <td>2009</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>255</th>
      <td>Suriname</td>
      <td>2009</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Inpsect 'year' object

type(df_corruption['year'][0])
```




    str



Observation: 'year' appears to be string object and will need to be converted to integer. Also, NaN values will need to be removed


```python
# convert year to int, remove 'NaN' records
df_corruption.year = pd.to_numeric(df_corruption.year, errors='coerce')
df_corruption.dropna(inplace=True)
```


```python
# Check data types and dataframe size

df_corruption.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 897 entries, 0 to 900
    Data columns (total 3 columns):
    country      897 non-null object
    year         897 non-null int64
    cpi_score    897 non-null float64
    dtypes: float64(1), int64(1), object(1)
    memory usage: 28.0+ KB
    


```python
df_corruption.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>cpi_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>New Zealand</td>
      <td>2008</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Denmark</td>
      <td>2008</td>
      <td>9.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sweden</td>
      <td>2008</td>
      <td>9.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Singapore</td>
      <td>2008</td>
      <td>9.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Switzerland</td>
      <td>2008</td>
      <td>9.0</td>
    </tr>
  </tbody>
</table>
</div>



We have now cleaned the 'corruption perception' dataset and brought it to a fit state for further work. Cleaning actions we successfully performed include:

- Transposing 2008/2009 data into long format
- Renaming columns
- Combining 2008/2009 data with 2007/2010/2011 data
- Converted data types
- Removed NaN records

We will proceed to clean the next dataset, infant mortality rate.

#### Infant Mortality Rate (rate per 1000 births) - Data Cleaning


```python
# Rename columns and Transpose data to long format
df_infant.rename(columns=lambda x: x.replace('Infant mortality rate', 'country'), inplace=True)
df_infant = pd.melt(df_infant, id_vars='country', var_name='year', value_name='infant_mortality_rate')
```


```python
# Filter data down to desired years, 2007 to 2011
y1 = '2007'
y2 = '2011'
df_infant = df_infant.query('year >= @y1 and year <= @y2')
```


```python
# convert year to int and infant_mortality_rate to float, remove 'NAs', reset indexes 
df_infant.year = pd.to_numeric(df_infant.year, errors='coerce')
df_infant.infant_mortality_rate = pd.to_numeric(df_infant.infant_mortality_rate, errors='coerce')
df_infant.dropna(inplace=True)
df_infant.reset_index(drop=True, inplace=True)
df_infant.shape
```




    (983, 3)




```python
# Check records per year
df_infant.year.value_counts()
```




    2008    199
    2007    199
    2011    195
    2010    195
    2009    195
    Name: year, dtype: int64




```python
#Check data types, NaNs
df_infant.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 983 entries, 0 to 982
    Data columns (total 3 columns):
    country                  983 non-null object
    year                     983 non-null int64
    infant_mortality_rate    983 non-null float64
    dtypes: float64(1), int64(1), object(1)
    memory usage: 23.1+ KB
    


```python
df_infant.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>infant_mortality_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>2007</td>
      <td>80.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albania</td>
      <td>2007</td>
      <td>16.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>2007</td>
      <td>26.4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Andorra</td>
      <td>2007</td>
      <td>2.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Angola</td>
      <td>2007</td>
      <td>117.1</td>
    </tr>
  </tbody>
</table>
</div>



Infant mortality rate data is now cleaned.

#### Literacy rate, adult total (% of people ages 15 and above) - Data Cleaning


```python
# Rename columns and Transpose data to long format

df_literacy.rename(columns=lambda x: x.replace('Adult (15+) literacy rate (%). Total', 'country'), inplace=True)
df_literacy = pd.melt(df_literacy, id_vars='country', var_name='year', value_name='literacy_rate')
df_literacy.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>literacy_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>1975</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albania</td>
      <td>1975</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>1975</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Andorra</td>
      <td>1975</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Angola</td>
      <td>1975</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Filter data down to desired years, 2006 to 2010
y1 = '2007'
y2 = '2011'
df_literacy = df_literacy.query('year >= @y1 and year <= @y2')
```


```python
# convert year to int and literacy_rate to float, remove 'NAs', reset indexes 

df_literacy.year = pd.to_numeric(df_literacy.year, errors='coerce')
df_literacy.literacy_rate = pd.to_numeric(df_literacy.literacy_rate, errors='coerce')
df_literacy.dropna(inplace=True)
df_literacy.reset_index(drop=True, inplace=True)
df_literacy.shape
```




    (206, 3)




```python
# Check records per year
df_literacy.year.value_counts()
```




    2011    84
    2010    35
    2007    32
    2009    28
    2008    27
    Name: year, dtype: int64




```python
# Check data types, null values
df_literacy.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 206 entries, 0 to 205
    Data columns (total 3 columns):
    country          206 non-null object
    year             206 non-null int64
    literacy_rate    206 non-null float64
    dtypes: float64(1), int64(1), object(1)
    memory usage: 4.9+ KB
    


```python
df_literacy.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>literacy_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Azerbaijan</td>
      <td>2007</td>
      <td>99.601906</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bolivia</td>
      <td>2007</td>
      <td>90.743470</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Brazil</td>
      <td>2007</td>
      <td>90.009370</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Burkina Faso</td>
      <td>2007</td>
      <td>28.729214</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cameroon</td>
      <td>2007</td>
      <td>70.679938</td>
    </tr>
  </tbody>
</table>
</div>



Literacy rate data is now cleaned.

#### Inequality index (Gini) - Data Cleaning


```python
# Rename columns and Transpose data to long format
df_inequality.rename(columns=lambda x: x.replace('Country Name', 'country'), inplace=True)
df_inequality = pd.melt(df_inequality, id_vars='country', var_name='year', value_name='gini_index')
df_inequality.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>gini_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aruba</td>
      <td>1960</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>1960</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Angola</td>
      <td>1960</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albania</td>
      <td>1960</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Andorra</td>
      <td>1960</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Filter data down to desired years, 2007 to 2011
y1 = '2007'
y2 = '2011'
df_inequality = df_inequality.query('year >= @y1 and year <= @y2')
```


```python
# Remove 'NaN' values, reset indexes, convert year to int and gini_index to float
df_inequality.dropna(inplace=True)
df_inequality.reset_index(drop=True, inplace=True)
df_inequality.year = pd.to_numeric(df_inequality.year, errors='coerce')
df_inequality.gini_index = pd.to_numeric(df_inequality.gini_index, errors='coerce')
df_inequality.shape
```




    (376, 3)




```python
# Check records per year
df_inequality.year.value_counts()
```




    2010    83
    2009    76
    2011    74
    2007    72
    2008    71
    Name: year, dtype: int64




```python
# Check data types and null values
df_inequality.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 376 entries, 0 to 375
    Data columns (total 3 columns):
    country       376 non-null object
    year          376 non-null int64
    gini_index    376 non-null float64
    dtypes: float64(1), int64(1), object(1)
    memory usage: 8.9+ KB
    


```python
df_inequality.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>gini_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Argentina</td>
      <td>2007</td>
      <td>46.3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Armenia</td>
      <td>2007</td>
      <td>31.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Austria</td>
      <td>2007</td>
      <td>30.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Belgium</td>
      <td>2007</td>
      <td>29.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Bulgaria</td>
      <td>2007</td>
      <td>36.1</td>
    </tr>
  </tbody>
</table>
</div>



Income inequality dataset is now cleaned and ready for further work.

#### Poverty (% people below USD1.90 a day) - Data Cleaning


```python
# Rename columns and Transpose data to long format
df_poverty.rename(columns=lambda x: x.replace('Country Name', 'country'), inplace=True)
df_poverty = pd.melt(df_poverty, id_vars='country', var_name='year', value_name='poverty_index')
df_poverty.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>poverty_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Aruba</td>
      <td>2011</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>2011</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Angola</td>
      <td>2011</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albania</td>
      <td>2011</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Andorra</td>
      <td>2011</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Remove 'NaN' values, reset indexes, convert year to int and poverty_index to float
df_poverty.dropna(inplace=True)
df_poverty.reset_index(drop=True, inplace=True)
df_poverty.year = pd.to_numeric(df_poverty.year, errors='coerce')
df_poverty.poverty_index = pd.to_numeric(df_poverty.poverty_index, errors='coerce')
df_poverty.shape
```




    (90, 3)




```python
# Confirm only 2011 data present
df_poverty.year.value_counts()
```




    2011    90
    Name: year, dtype: int64




```python
# Check data types, null values
df_poverty.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 90 entries, 0 to 89
    Data columns (total 3 columns):
    country          90 non-null object
    year             90 non-null int64
    poverty_index    90 non-null float64
    dtypes: float64(1), int64(1), object(1)
    memory usage: 2.2+ KB
    


```python
df_poverty.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>poverty_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Argentina</td>
      <td>2011</td>
      <td>0.9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Armenia</td>
      <td>2011</td>
      <td>2.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Austria</td>
      <td>2011</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Belgium</td>
      <td>2011</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Benin</td>
      <td>2011</td>
      <td>53.1</td>
    </tr>
  </tbody>
</table>
</div>



We have now completed cleaning of the poverty_index dataset.

#### GDP/capita (USD, inflation-adjusted) - Data Cleaning**

We will now perform cleaning on the GDP dataset


```python
# Rename columns and Transpose data to long format
df_gdp.rename(columns=lambda x: x.replace('Income per person (fixed 2000 US$)', 'country'), inplace=True)
df_gdp = pd.melt(df_gdp, id_vars='country', var_name='year', value_name='gdp_per_capita')
df_gdp.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Abkhazia</td>
      <td>1960</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>1960</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Akrotiri and Dhekelia</td>
      <td>1960</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albania</td>
      <td>1960</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Algeria</td>
      <td>1960</td>
      <td>1280.384828</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Filter data down to desired years, 2007 to 2011
y1 = '2007'
y2 = '2011'
df_gdp = df_gdp.query('year >= @y1 and year <= @y2')
```


```python
# Remove 'NaN' values, reset indexes, convert year to int and gdp_per_capita to float
df_gdp.dropna(inplace=True)
df_gdp.reset_index(drop=True, inplace=True)
df_gdp.year = pd.to_numeric(df_gdp.year, errors='coerce')
df_gdp.gdp_per_capita = pd.to_numeric(df_gdp.gdp_per_capita, errors='coerce')
df_gdp.shape
```




    (930, 3)




```python
# Check records per year
df_gdp.year.value_counts()
```




    2007    193
    2008    191
    2009    189
    2010    182
    2011    175
    Name: year, dtype: int64




```python
# Check data types and nulls

df_gdp.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 930 entries, 0 to 929
    Data columns (total 3 columns):
    country           930 non-null object
    year              930 non-null int64
    gdp_per_capita    930 non-null float64
    dtypes: float64(1), int64(1), object(1)
    memory usage: 21.9+ KB
    


```python
df_gdp.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Albania</td>
      <td>2007</td>
      <td>1681.61391</td>
    </tr>
  </tbody>
</table>
</div>



GDP per capita data is now cleaned.

At this stage, we have cleaned the single-variable data sets. We shall now comine them into a single dataframe, to allow for comparisons across variable

#### Combined Data

In the cell below, we will combine all individual datasets into a single dataframe.


```python
# Combine individual datasets into a single dataframe

df = pd.merge(df_corruption, df_infant, on=['country', 'year'], how='outer')
df = pd.merge(df, df_literacy, on=['country', 'year'], how='outer')
df = pd.merge(df, df_inequality, on=['country', 'year'], how='outer')
df = pd.merge(df, df_poverty, on=['country', 'year'], how='outer')
df = pd.merge(df, df_gdp, on=['country', 'year'], how='outer')
```

Let us inspect the combined data.


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>cpi_score</th>
      <th>infant_mortality_rate</th>
      <th>literacy_rate</th>
      <th>gini_index</th>
      <th>poverty_index</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>New Zealand</td>
      <td>2008</td>
      <td>9.4</td>
      <td>5.3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15011.18385</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Denmark</td>
      <td>2008</td>
      <td>9.3</td>
      <td>3.6</td>
      <td>NaN</td>
      <td>25.2</td>
      <td>NaN</td>
      <td>32320.10054</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sweden</td>
      <td>2008</td>
      <td>9.2</td>
      <td>2.6</td>
      <td>NaN</td>
      <td>26.8</td>
      <td>NaN</td>
      <td>32798.73425</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Singapore</td>
      <td>2008</td>
      <td>9.2</td>
      <td>2.2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30131.61718</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Switzerland</td>
      <td>2008</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>33.8</td>
      <td>NaN</td>
      <td>39324.73112</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (1104, 8)




```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1104 entries, 0 to 1103
    Data columns (total 8 columns):
    country                  1104 non-null object
    year                     1104 non-null int64
    cpi_score                897 non-null float64
    infant_mortality_rate    983 non-null float64
    literacy_rate            206 non-null float64
    gini_index               376 non-null float64
    poverty_index            90 non-null float64
    gdp_per_capita           930 non-null float64
    dtypes: float64(6), int64(1), object(1)
    memory usage: 77.6+ KB
    


```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>cpi_score</th>
      <th>infant_mortality_rate</th>
      <th>literacy_rate</th>
      <th>gini_index</th>
      <th>poverty_index</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1104.000000</td>
      <td>897.000000</td>
      <td>983.000000</td>
      <td>206.000000</td>
      <td>376.000000</td>
      <td>90.000000</td>
      <td>930.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2009.009058</td>
      <td>4.014158</td>
      <td>28.860905</td>
      <td>84.283146</td>
      <td>36.582979</td>
      <td>9.026667</td>
      <td>8418.930436</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.437078</td>
      <td>2.088395</td>
      <td>26.159773</td>
      <td>17.152699</td>
      <td>8.551911</td>
      <td>15.310287</td>
      <td>13276.734101</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2007.000000</td>
      <td>1.000000</td>
      <td>1.760000</td>
      <td>25.307745</td>
      <td>23.700000</td>
      <td>0.000000</td>
      <td>97.910183</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2008.000000</td>
      <td>2.500000</td>
      <td>7.750000</td>
      <td>76.014593</td>
      <td>30.175000</td>
      <td>0.200000</td>
      <td>672.367236</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2009.000000</td>
      <td>3.300000</td>
      <td>18.300000</td>
      <td>91.845252</td>
      <td>33.950000</td>
      <td>1.200000</td>
      <td>2503.908898</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2010.000000</td>
      <td>5.100000</td>
      <td>45.400000</td>
      <td>96.178175</td>
      <td>42.325000</td>
      <td>8.425000</td>
      <td>9783.327358</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2011.000000</td>
      <td>9.500000</td>
      <td>120.500000</td>
      <td>99.998262</td>
      <td>63.400000</td>
      <td>54.200000</td>
      <td>108111.212800</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check records per year
df.year.value_counts()
```




    2011    233
    2007    228
    2010    217
    2008    217
    2009    209
    Name: year, dtype: int64




```python
# Check for duplicated values
df.duplicated().value_counts()
```




    False    1104
    dtype: int64




```python
df.country.value_counts()
```




    Pakistan                          5
    Luxembourg                        5
    Sudan                             5
    Bosnia and Herzegovina            5
    Angola                            5
    North Korea                       5
    Belize                            5
    Malawi                            5
    Russia                            5
    Cameroon                          5
    Canada                            5
    Dominican Republic                5
    Estonia                           5
    Nepal                             5
    Switzerland                       5
    Slovak Republic                   5
    Tanzania                          5
    Mexico                            5
    Finland                           5
    United States                     5
    Fiji                              5
    Vietnam                           5
    France                            5
    Chad                              5
    Afghanistan                       5
    Belarus                           5
    Turkey                            5
    St. Vincent and the Grenadines    5
    Iceland                           5
    Armenia                           5
                                     ..
    Kuweit                            1
    Cayman Islands                    1
    Channel Islands                   1
    Europe & Central Asia             1
    IDA total                         1
    Viet Nam                          1
    Congo  Republic                   1
    Moldovaa                          1
    Gambia, The                       1
    Low income                        1
    High income                       1
    Congo-Brazzaville                 1
    South Asia                        1
    USA                               1
    Sub-Saharan Africa                1
    Korea, Dem. Rep.                  1
    Lao PDR                           1
    Central African Rep.              1
    Democratic Republic of Congo      1
    Upper middle income               1
    Low & middle income               1
    Korea (North)                     1
    Brunei Darussalam                 1
    East Asia & Pacific               1
    Cabo Verde                        1
    Aruba                             1
    Czech Republik                    1
    Middle East & North Africa        1
    Yemen, Rep.                       1
    Congo, Republic                   1
    Name: country, Length: 261, dtype: int64



We have made a number of observations on this combined dataframe.

Our variables where correctly combined into a single dataframe, there appear to be no duplicates, and the data types are correct.

However, we observe that some country names are not consistently spelt, for example, 'Democratic Republic of Congo' and 'Democratic Republic of the Congo' likely refer to the same country. Similarly, 'Russia' and 'Russian Federation' both appear in the data.

Also, some records appear to be for non-state entities, for example, 'High income' and 'Latin America & Carribean' and 'Fragile and conflict affected situations'.

We also spot a row for 'Kuweit' and speculate there may be a separate row for 'Kuwait'.


```python
#Check for occurence of 'Kuwait' or 'Kuweit'

a1 = 'kuweit'
a2 = 'kuwait'

df.query('country.str.lower() == @a1 or country.str.lower() == @a2')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>cpi_score</th>
      <th>infant_mortality_rate</th>
      <th>literacy_rate</th>
      <th>gini_index</th>
      <th>poverty_index</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>65</th>
      <td>Kuwait</td>
      <td>2008</td>
      <td>4.1</td>
      <td>9.7</td>
      <td>93.906206</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>25308.08492</td>
    </tr>
    <tr>
      <th>242</th>
      <td>Kuwait</td>
      <td>2009</td>
      <td>4.5</td>
      <td>9.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23116.34061</td>
    </tr>
    <tr>
      <th>415</th>
      <td>Kuweit</td>
      <td>2007</td>
      <td>4.3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>589</th>
      <td>Kuwait</td>
      <td>2010</td>
      <td>4.5</td>
      <td>9.2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23114.58667</td>
    </tr>
    <tr>
      <th>768</th>
      <td>Kuwait</td>
      <td>2011</td>
      <td>4.6</td>
      <td>8.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24286.11580</td>
    </tr>
    <tr>
      <th>908</th>
      <td>Kuwait</td>
      <td>2007</td>
      <td>NaN</td>
      <td>9.8</td>
      <td>93.664185</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>25100.02810</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check for occurence of 'Russia' or 'Russian Federation'

a3 = 'russia'
a4 = 'russian federation'

df.query('country.str.lower() == @a3 or country.str.lower() == @a4')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>cpi_score</th>
      <th>infant_mortality_rate</th>
      <th>literacy_rate</th>
      <th>gini_index</th>
      <th>poverty_index</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>146</th>
      <td>Russia</td>
      <td>2008</td>
      <td>2.2</td>
      <td>11.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3043.665599</td>
    </tr>
    <tr>
      <th>322</th>
      <td>Russia</td>
      <td>2009</td>
      <td>2.1</td>
      <td>10.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2806.414830</td>
    </tr>
    <tr>
      <th>500</th>
      <td>Russia</td>
      <td>2007</td>
      <td>2.3</td>
      <td>12.4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2888.847355</td>
    </tr>
    <tr>
      <th>697</th>
      <td>Russia</td>
      <td>2010</td>
      <td>2.1</td>
      <td>10.3</td>
      <td>99.684267</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2928.005033</td>
    </tr>
    <tr>
      <th>861</th>
      <td>Russia</td>
      <td>2011</td>
      <td>2.4</td>
      <td>9.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3054.727742</td>
    </tr>
    <tr>
      <th>1058</th>
      <td>Russian Federation</td>
      <td>2007</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>42.3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1061</th>
      <td>Russian Federation</td>
      <td>2008</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.6</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1063</th>
      <td>Russian Federation</td>
      <td>2009</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>39.8</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1068</th>
      <td>Russian Federation</td>
      <td>2010</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>39.5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1069</th>
      <td>Russian Federation</td>
      <td>2011</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>39.7</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



As suspected, 'Kuwait' and 'Kuweit' both occur in the data, as do records against both 'Russia' and 'Russian Federation'

We therefore judge that a manual process of cleaning 'country' needs to be performed outside python, and the cleaned data then re-integrated into our analysis data.

To accomplish this, we shall export a unique list of countries, manually create a mapping table containing valid countries with names spelt correctly, and then import this back into our dataframe.

This edit will also afford us the opportunity to augment country data with Region categorization, to pose and answer additional questions. Regional classifications have been obtained from the website below.

https://meta.wikimedia.org/wiki/List_of_countries_by_regional_classification

Important to stress fixing errors in country names cannot be performed programatically.


```python
# This snippet exports the original country list to a csv file
pd.DataFrame.to_csv(pd.DataFrame(df.country.value_counts()),'country_list.csv')
#pd.DataFrame(df.country.value_counts())

```

After performing offline cleaning of the country list and adding of 'Region' information, we are now ready to import and integrate this list into our dataset.

Filename holding our cleaned country list is: 'country_list_clean.csv'


```python
country_list = pd.read_csv('country_list_clean.csv')
country_list.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>country_fixed</th>
      <th>region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Uzbekistan</td>
      <td>Uzbekistan</td>
      <td>CIS</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tuvalu</td>
      <td>Tuvalu</td>
      <td>Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Estonia</td>
      <td>Estonia</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Papua New Guinea</td>
      <td>Papua New Guinea</td>
      <td>Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Slovakia</td>
      <td>Slovakia</td>
      <td>Europe</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check that same number of records exist in both country lists

print('Dimensions - Original Country List: {}\nDimensions - Cleaned Country List: {}'.
      format(pd.DataFrame(df.country.value_counts()).shape, country_list.shape))
```

    Dimensions - Original Country List: (261, 1)
    Dimensions - Cleaned Country List: (261, 3)
    


```python
# Import columns into analysis dataframe
df = pd.merge(df, country_list, how='left', left_on='country', right_on='country')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>cpi_score</th>
      <th>infant_mortality_rate</th>
      <th>literacy_rate</th>
      <th>gini_index</th>
      <th>poverty_index</th>
      <th>gdp_per_capita</th>
      <th>country_fixed</th>
      <th>region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>New Zealand</td>
      <td>2008</td>
      <td>9.4</td>
      <td>5.3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15011.18385</td>
      <td>New Zealand</td>
      <td>Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Denmark</td>
      <td>2008</td>
      <td>9.3</td>
      <td>3.6</td>
      <td>NaN</td>
      <td>25.2</td>
      <td>NaN</td>
      <td>32320.10054</td>
      <td>Denmark</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Sweden</td>
      <td>2008</td>
      <td>9.2</td>
      <td>2.6</td>
      <td>NaN</td>
      <td>26.8</td>
      <td>NaN</td>
      <td>32798.73425</td>
      <td>Sweden</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Singapore</td>
      <td>2008</td>
      <td>9.2</td>
      <td>2.2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>30131.61718</td>
      <td>Singapore</td>
      <td>Asia &amp; Pacific</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Switzerland</td>
      <td>2008</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>33.8</td>
      <td>NaN</td>
      <td>39324.73112</td>
      <td>Switzerland</td>
      <td>Europe</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Take a copy of the dataframe, and perfom cleaning in this copy
df_clean = df.copy()
```

One of the actions we performed in building a clean country list was to identify non-country records in our original data. These 'countries' have been marked as 'DELETE' in the country_fixed column of our cleaned data, so as to easily identify them for removal.

We will now display and then remove these records.


```python
# List records for non-country entities, which have been identified with "DELETE" 
#   in 'country_fixed' column

df_clean.query('country_fixed == "DELETE"')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>cpi_score</th>
      <th>infant_mortality_rate</th>
      <th>literacy_rate</th>
      <th>gini_index</th>
      <th>poverty_index</th>
      <th>gdp_per_capita</th>
      <th>country_fixed</th>
      <th>region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1070</th>
      <td>East Asia &amp; Pacific</td>
      <td>2011</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.6</td>
      <td>NaN</td>
      <td>DELETE</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1072</th>
      <td>Fragile and conflict affected situations</td>
      <td>2011</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>36.9</td>
      <td>NaN</td>
      <td>DELETE</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1073</th>
      <td>High income</td>
      <td>2011</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.6</td>
      <td>NaN</td>
      <td>DELETE</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1074</th>
      <td>IDA total</td>
      <td>2011</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>34.1</td>
      <td>NaN</td>
      <td>DELETE</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1075</th>
      <td>Latin America &amp; Caribbean</td>
      <td>2011</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.6</td>
      <td>NaN</td>
      <td>DELETE</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1076</th>
      <td>Low income</td>
      <td>2011</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>50.5</td>
      <td>NaN</td>
      <td>DELETE</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1077</th>
      <td>Lower middle income</td>
      <td>2011</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>19.6</td>
      <td>NaN</td>
      <td>DELETE</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1078</th>
      <td>Low &amp; middle income</td>
      <td>2011</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>16.4</td>
      <td>NaN</td>
      <td>DELETE</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1079</th>
      <td>Middle East &amp; North Africa</td>
      <td>2011</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.7</td>
      <td>NaN</td>
      <td>DELETE</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1080</th>
      <td>South Asia</td>
      <td>2011</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>20.1</td>
      <td>NaN</td>
      <td>DELETE</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1081</th>
      <td>Sub-Saharan Africa</td>
      <td>2011</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>44.9</td>
      <td>NaN</td>
      <td>DELETE</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1082</th>
      <td>Upper middle income</td>
      <td>2011</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.7</td>
      <td>NaN</td>
      <td>DELETE</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1083</th>
      <td>World</td>
      <td>2011</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>13.8</td>
      <td>NaN</td>
      <td>DELETE</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Remove records

df_clean = df_clean.query('country_fixed != "DELETE"')
```


```python
# Check records no longer exist

df_clean.query('country_fixed =="DELETE"')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>cpi_score</th>
      <th>infant_mortality_rate</th>
      <th>literacy_rate</th>
      <th>gini_index</th>
      <th>poverty_index</th>
      <th>gdp_per_capita</th>
      <th>country_fixed</th>
      <th>region</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



Deletion of non-country records was successful.

Next, let's look at instances where same countries have different spellings. Examples we spotted earlier are 'Kuwait vs Kuweit', and 'Russia vs Russian Federation'. Our manual check uncovered many more. In cells below however, let us display name anomalies for Kuwait and Russian Federation.


```python
# Using earlier example of 'Kuwait' or 'Kuweit'

a1 = 'kuweit'
a2 = 'kuwait'

df.query('country.str.lower() == @a1 or country.str.lower() == @a2')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>cpi_score</th>
      <th>infant_mortality_rate</th>
      <th>literacy_rate</th>
      <th>gini_index</th>
      <th>poverty_index</th>
      <th>gdp_per_capita</th>
      <th>country_fixed</th>
      <th>region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>65</th>
      <td>Kuwait</td>
      <td>2008</td>
      <td>4.1</td>
      <td>9.7</td>
      <td>93.906206</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>25308.08492</td>
      <td>Kuwait</td>
      <td>Arab States</td>
    </tr>
    <tr>
      <th>242</th>
      <td>Kuwait</td>
      <td>2009</td>
      <td>4.5</td>
      <td>9.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23116.34061</td>
      <td>Kuwait</td>
      <td>Arab States</td>
    </tr>
    <tr>
      <th>415</th>
      <td>Kuweit</td>
      <td>2007</td>
      <td>4.3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Kuwait</td>
      <td>Arab States</td>
    </tr>
    <tr>
      <th>589</th>
      <td>Kuwait</td>
      <td>2010</td>
      <td>4.5</td>
      <td>9.2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23114.58667</td>
      <td>Kuwait</td>
      <td>Arab States</td>
    </tr>
    <tr>
      <th>768</th>
      <td>Kuwait</td>
      <td>2011</td>
      <td>4.6</td>
      <td>8.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24286.11580</td>
      <td>Kuwait</td>
      <td>Arab States</td>
    </tr>
    <tr>
      <th>908</th>
      <td>Kuwait</td>
      <td>2007</td>
      <td>NaN</td>
      <td>9.8</td>
      <td>93.664185</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>25100.02810</td>
      <td>Kuwait</td>
      <td>Arab States</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Using the earlier seen 'Russia' vs 'Russian Federation' example

a3 = 'russia'
a4 = 'russian federation'
df_clean.query('country.str.lower() == @a3 or country.str.lower() == @a4')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>cpi_score</th>
      <th>infant_mortality_rate</th>
      <th>literacy_rate</th>
      <th>gini_index</th>
      <th>poverty_index</th>
      <th>gdp_per_capita</th>
      <th>country_fixed</th>
      <th>region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>146</th>
      <td>Russia</td>
      <td>2008</td>
      <td>2.2</td>
      <td>11.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3043.665599</td>
      <td>Russian Federation</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>322</th>
      <td>Russia</td>
      <td>2009</td>
      <td>2.1</td>
      <td>10.9</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2806.414830</td>
      <td>Russian Federation</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>500</th>
      <td>Russia</td>
      <td>2007</td>
      <td>2.3</td>
      <td>12.4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2888.847355</td>
      <td>Russian Federation</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>697</th>
      <td>Russia</td>
      <td>2010</td>
      <td>2.1</td>
      <td>10.3</td>
      <td>99.684267</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2928.005033</td>
      <td>Russian Federation</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>861</th>
      <td>Russia</td>
      <td>2011</td>
      <td>2.4</td>
      <td>9.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3054.727742</td>
      <td>Russian Federation</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>1058</th>
      <td>Russian Federation</td>
      <td>2007</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>42.3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Russian Federation</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>1061</th>
      <td>Russian Federation</td>
      <td>2008</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Russian Federation</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>1063</th>
      <td>Russian Federation</td>
      <td>2009</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>39.8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Russian Federation</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>1068</th>
      <td>Russian Federation</td>
      <td>2010</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>39.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Russian Federation</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>1069</th>
      <td>Russian Federation</td>
      <td>2011</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>39.7</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>Russian Federation</td>
      <td>Europe</td>
    </tr>
  </tbody>
</table>
</div>



Column 'country_fixed' contains the correct country names. 

However, our comparison metrics are split across variants of country spellings in 'country' column. How many rows are duplicated in this way?


```python
# Number of rows with duplicates on 'country_fixed' and 'year' columns

df_clean.duplicated(['country_fixed','year']).value_counts()
```




    False    1023
    True       68
    dtype: int64



We have 68 duplicates.

Let us illustrate a single instance of this problem, using Russia data for 2008.


```python
# Show records for Russia, 2008

y = int(2008)
df_clean.query('country_fixed.str.lower() == @a4 and year == @y')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>cpi_score</th>
      <th>infant_mortality_rate</th>
      <th>literacy_rate</th>
      <th>gini_index</th>
      <th>poverty_index</th>
      <th>gdp_per_capita</th>
      <th>country_fixed</th>
      <th>region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>146</th>
      <td>Russia</td>
      <td>2008</td>
      <td>2.2</td>
      <td>11.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3043.665599</td>
      <td>Russian Federation</td>
      <td>Europe</td>
    </tr>
    <tr>
      <th>1061</th>
      <td>Russian Federation</td>
      <td>2008</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>41.6</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Russian Federation</td>
      <td>Europe</td>
    </tr>
  </tbody>
</table>
</div>



As observed earlier, records for Russia in year 2008 are split across two rows, 146 and 1061. All metrics for Russia in year 2008 need to appear in a single row. This principle applies to all similarly duplicated records.  

We will use groupby() to consolidate values, using 'country_fixed', 'year' and 'region' as keys, and reset index to restore keys to columns.


```python
# Groupby to consolidate values
df_clean = pd.DataFrame(df_clean.groupby(['country_fixed','year','region']).sum())

#reset index, to restore country, year and region to columns
df_clean.reset_index(level=df_clean.index.names, inplace=True)
df_clean.head(6)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country_fixed</th>
      <th>year</th>
      <th>region</th>
      <th>cpi_score</th>
      <th>infant_mortality_rate</th>
      <th>literacy_rate</th>
      <th>gini_index</th>
      <th>poverty_index</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>2007</td>
      <td>Asia &amp; Pacific</td>
      <td>1.8</td>
      <td>80.4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>2008</td>
      <td>Asia &amp; Pacific</td>
      <td>1.3</td>
      <td>78.6</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>2009</td>
      <td>Asia &amp; Pacific</td>
      <td>1.4</td>
      <td>76.8</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>2010</td>
      <td>Asia &amp; Pacific</td>
      <td>1.4</td>
      <td>75.1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>2011</td>
      <td>Asia &amp; Pacific</td>
      <td>1.5</td>
      <td>73.4</td>
      <td>39.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Albania</td>
      <td>2007</td>
      <td>Europe</td>
      <td>2.9</td>
      <td>16.7</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1681.61391</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check to confirm "Russian Federation" record is now correctly displayed

y = int(2008)
df_clean.query('country_fixed.str.lower() == @a4 and year == @y')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country_fixed</th>
      <th>year</th>
      <th>region</th>
      <th>cpi_score</th>
      <th>infant_mortality_rate</th>
      <th>literacy_rate</th>
      <th>gini_index</th>
      <th>poverty_index</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>745</th>
      <td>Russian Federation</td>
      <td>2008</td>
      <td>Europe</td>
      <td>2.2</td>
      <td>11.6</td>
      <td>0.0</td>
      <td>41.6</td>
      <td>0.0</td>
      <td>3043.665599</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check if any duplicates on 'country_fixed' and 'year' columns

df_clean.duplicated(['country_fixed','year']).value_counts()
```




    False    1021
    dtype: int64



We sucessfully used groupby() to ensure all metrics for each year, country and region combination appear on a single row, and then removed the resulting duplicated. We now have correct unique records per country and year. We have confirmed this by checking the record for Russia in 2008.

Let us now rename the country_fixed column, and proceed with further checks.


```python
# Rename 'country_fixed' to 'country'

df_clean.rename(columns=lambda x: x.replace('country_fixed', 'country'), inplace=True)
df_clean.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>year</th>
      <th>region</th>
      <th>cpi_score</th>
      <th>infant_mortality_rate</th>
      <th>literacy_rate</th>
      <th>gini_index</th>
      <th>poverty_index</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>2007</td>
      <td>Asia &amp; Pacific</td>
      <td>1.8</td>
      <td>80.4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_clean.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>cpi_score</th>
      <th>infant_mortality_rate</th>
      <th>literacy_rate</th>
      <th>gini_index</th>
      <th>poverty_index</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
      <td>1021.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2008.980411</td>
      <td>3.522331</td>
      <td>27.786748</td>
      <td>17.005218</td>
      <td>13.472282</td>
      <td>0.541528</td>
      <td>7668.565432</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.416848</td>
      <td>2.373268</td>
      <td>26.243390</td>
      <td>34.703443</td>
      <td>18.399647</td>
      <td>4.362516</td>
      <td>12895.942581</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2007.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2008.000000</td>
      <td>2.200000</td>
      <td>7.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>498.570158</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2009.000000</td>
      <td>3.000000</td>
      <td>17.300000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2134.037162</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2010.000000</td>
      <td>4.700000</td>
      <td>43.900000</td>
      <td>0.000000</td>
      <td>31.500000</td>
      <td>0.000000</td>
      <td>8151.712950</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2011.000000</td>
      <td>11.000000</td>
      <td>120.500000</td>
      <td>99.998262</td>
      <td>63.400000</td>
      <td>54.200000</td>
      <td>108111.212800</td>
    </tr>
  </tbody>
</table>
</div>



Observe that counts on all columns are 1021. Values that were null in original datasets are now held as zeros, resulting in incorrect counts. Incorrect counts on columns will cause wrong values for mean, min, max and other aggregations.

We will need to change the zeros back to null.


```python
# Replace all zeros with NaN, so our aggregations (count, mean, sum, min) are accurate

df_clean = df_clean.replace(0.0, np.nan)
df_clean.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>cpi_score</th>
      <th>infant_mortality_rate</th>
      <th>literacy_rate</th>
      <th>gini_index</th>
      <th>poverty_index</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1021.000000</td>
      <td>894.000000</td>
      <td>983.000000</td>
      <td>206.000000</td>
      <td>376.000000</td>
      <td>57.000000</td>
      <td>930.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2008.980411</td>
      <td>4.022707</td>
      <td>28.860905</td>
      <td>84.283146</td>
      <td>36.582979</td>
      <td>9.700000</td>
      <td>8418.930436</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.416848</td>
      <td>2.101977</td>
      <td>26.159773</td>
      <td>17.152699</td>
      <td>8.551911</td>
      <td>16.006952</td>
      <td>13276.734101</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2007.000000</td>
      <td>1.000000</td>
      <td>1.760000</td>
      <td>25.307745</td>
      <td>23.700000</td>
      <td>0.100000</td>
      <td>97.910183</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2008.000000</td>
      <td>2.500000</td>
      <td>7.750000</td>
      <td>76.014593</td>
      <td>30.175000</td>
      <td>0.500000</td>
      <td>672.367236</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2009.000000</td>
      <td>3.300000</td>
      <td>18.300000</td>
      <td>91.845252</td>
      <td>33.950000</td>
      <td>1.800000</td>
      <td>2503.908898</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2010.000000</td>
      <td>5.100000</td>
      <td>45.400000</td>
      <td>96.178175</td>
      <td>42.325000</td>
      <td>7.900000</td>
      <td>9783.327358</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2011.000000</td>
      <td>11.000000</td>
      <td>120.500000</td>
      <td>99.998262</td>
      <td>63.400000</td>
      <td>54.200000</td>
      <td>108111.212800</td>
    </tr>
  </tbody>
</table>
</div>



Obeserving the output in the last two cells, we note that the 'count' aggregation now correctly shows the number of non-zero values in each column. Also, min, mean and other aggregations are now correctly calculated.

We will now create a version of our dataset that contains average of each measure by country. This will be used for most of the following analysis. Why have we taken this decision?

Within the five year window of our dataset, countries will have missing data across years. Taking an average of yearly observations by country allows us to run comparisons among countries with reduced concern over availability of data in a given year.


```python
# Create dataframe of average observations by country. 
df_avg = df_clean.drop('year', axis=1)
df_avg = pd.DataFrame(df_avg.groupby(['country','region']).mean())

# reset indexes and restore 'country' and 'region' columns
df_avg.reset_index(level=df_avg.index.names, inplace=True)

df_avg.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>region</th>
      <th>cpi_score</th>
      <th>infant_mortality_rate</th>
      <th>literacy_rate</th>
      <th>gini_index</th>
      <th>poverty_index</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>Asia &amp; Pacific</td>
      <td>1.48</td>
      <td>76.86</td>
      <td>39.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albania</td>
      <td>Europe</td>
      <td>3.16</td>
      <td>15.44</td>
      <td>96.391969</td>
      <td>30.00</td>
      <td>NaN</td>
      <td>1844.903592</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>Arab States</td>
      <td>2.90</td>
      <td>24.46</td>
      <td>NaN</td>
      <td>27.60</td>
      <td>0.5</td>
      <td>2201.836568</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Andorra</td>
      <td>Europe</td>
      <td>NaN</td>
      <td>2.60</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>21719.572490</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Angola</td>
      <td>Africa</td>
      <td>1.98</td>
      <td>112.08</td>
      <td>70.362420</td>
      <td>42.70</td>
      <td>NaN</td>
      <td>611.714745</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Antigua and Barbuda</td>
      <td>South/Latin America</td>
      <td>NaN</td>
      <td>8.22</td>
      <td>98.950000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>11817.136463</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Argentina</td>
      <td>South/Latin America</td>
      <td>2.92</td>
      <td>13.38</td>
      <td>97.858770</td>
      <td>44.00</td>
      <td>0.9</td>
      <td>10321.740247</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Armenia</td>
      <td>CIS</td>
      <td>2.70</td>
      <td>17.02</td>
      <td>99.568170</td>
      <td>29.56</td>
      <td>2.2</td>
      <td>1391.495500</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Aruba</td>
      <td>South/Latin America</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>96.822640</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Australia</td>
      <td>Asia &amp; Pacific</td>
      <td>8.70</td>
      <td>4.20</td>
      <td>NaN</td>
      <td>35.05</td>
      <td>NaN</td>
      <td>25092.326418</td>
    </tr>
  </tbody>
</table>
</div>



'Averages' dataframe was successfully created.

Let us inspect this dataframe, to identify any anomalies and plan any corrective actions. 


```python
df_avg.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 211 entries, 0 to 210
    Data columns (total 8 columns):
    country                  211 non-null object
    region                   211 non-null object
    cpi_score                185 non-null float64
    infant_mortality_rate    199 non-null float64
    literacy_rate            140 non-null float64
    gini_index               134 non-null float64
    poverty_index            57 non-null float64
    gdp_per_capita           193 non-null float64
    dtypes: float64(6), object(2)
    memory usage: 13.3+ KB
    


```python
df_avg.mean()
```




    cpi_score                   4.025874
    infant_mortality_rate      28.644548
    literacy_rate              82.733571
    gini_index                 38.365933
    poverty_index               9.700000
    gdp_per_capita           9170.833436
    dtype: float64




```python
df_avg.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cpi_score</th>
      <th>infant_mortality_rate</th>
      <th>literacy_rate</th>
      <th>gini_index</th>
      <th>poverty_index</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>185.000000</td>
      <td>199.000000</td>
      <td>140.000000</td>
      <td>134.000000</td>
      <td>57.000000</td>
      <td>193.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.025874</td>
      <td>28.644548</td>
      <td>82.733571</td>
      <td>38.365933</td>
      <td>9.700000</td>
      <td>9170.833436</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.090305</td>
      <td>26.047186</td>
      <td>18.664967</td>
      <td>8.469815</td>
      <td>16.006952</td>
      <td>14625.856835</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.930000</td>
      <td>25.307745</td>
      <td>24.540000</td>
      <td>0.100000</td>
      <td>103.104815</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.460000</td>
      <td>7.677500</td>
      <td>71.369112</td>
      <td>31.875000</td>
      <td>0.500000</td>
      <td>731.926135</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.240000</td>
      <td>17.680000</td>
      <td>90.599093</td>
      <td>36.340000</td>
      <td>1.800000</td>
      <td>2581.767415</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.140000</td>
      <td>44.650000</td>
      <td>97.713798</td>
      <td>43.550000</td>
      <td>7.900000</td>
      <td>10558.767324</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.380000</td>
      <td>112.080000</td>
      <td>99.998262</td>
      <td>63.200000</td>
      <td>54.200000</td>
      <td>103885.246787</td>
    </tr>
  </tbody>
</table>
</div>



We notice there are null values in the metrics columns, with the worst being 'poverty_index' with only 57 valid records in our dataset of 211 records.

**The paucity of values against 'poverty' suggests our ability to run comparisons across countries in this column will be limited. We will therefore remove the 'poverty_index' column from our df_avg dataset.**


```python
# Remove 'poverty_index' column

df_avg.drop('poverty_index', axis=1, inplace=True)
df_avg.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 211 entries, 0 to 210
    Data columns (total 7 columns):
    country                  211 non-null object
    region                   211 non-null object
    cpi_score                185 non-null float64
    infant_mortality_rate    199 non-null float64
    literacy_rate            140 non-null float64
    gini_index               134 non-null float64
    gdp_per_capita           193 non-null float64
    dtypes: float64(5), object(2)
    memory usage: 11.6+ KB
    

Null values in other columns will need to be filled in. Our approach will be to use the 'Region' averages to fill in missing values. 


```python
# Calculate and inspect Region mean values 

df_avg.groupby(['region']).mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cpi_score</th>
      <th>infant_mortality_rate</th>
      <th>literacy_rate</th>
      <th>gini_index</th>
      <th>gdp_per_capita</th>
    </tr>
    <tr>
      <th>region</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Africa</th>
      <td>2.945891</td>
      <td>61.859545</td>
      <td>64.777314</td>
      <td>45.027143</td>
      <td>1256.879586</td>
    </tr>
    <tr>
      <th>Arab States</th>
      <td>3.478000</td>
      <td>30.746667</td>
      <td>81.398935</td>
      <td>33.212500</td>
      <td>7493.379077</td>
    </tr>
    <tr>
      <th>Asia &amp; Pacific</th>
      <td>3.890270</td>
      <td>27.043810</td>
      <td>81.383965</td>
      <td>37.144133</td>
      <td>6875.002351</td>
    </tr>
    <tr>
      <th>CIS</th>
      <td>2.380000</td>
      <td>27.272000</td>
      <td>99.603976</td>
      <td>30.420000</td>
      <td>1361.668615</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>5.789512</td>
      <td>5.607500</td>
      <td>98.149657</td>
      <td>31.779065</td>
      <td>20446.367688</td>
    </tr>
    <tr>
      <th>North America</th>
      <td>7.990000</td>
      <td>5.660000</td>
      <td>NaN</td>
      <td>37.225000</td>
      <td>42977.916206</td>
    </tr>
    <tr>
      <th>South/Latin America</th>
      <td>3.983229</td>
      <td>18.504306</td>
      <td>92.565855</td>
      <td>48.778125</td>
      <td>5744.894997</td>
    </tr>
  </tbody>
</table>
</div>



We observe that with only one exception, we can obtain region mean for all metrics. These region mean values will be used to fill in the missing values.

Literacy rate appears to be missing for the entire North American region. We will fill this in by using the overall or global literacy rate mean. 

First, however, we will now fill in the missing values using region averages where available.


```python
# Temporary dataframe with missing values filled in with region averages 

df_avg_fill = df_avg.groupby('region').transform(lambda x: x.fillna(x.mean()))
df_avg_fill.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cpi_score</th>
      <th>infant_mortality_rate</th>
      <th>literacy_rate</th>
      <th>gini_index</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.480000</td>
      <td>76.86</td>
      <td>39.000000</td>
      <td>37.144133</td>
      <td>6875.002351</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.160000</td>
      <td>15.44</td>
      <td>96.391969</td>
      <td>30.000000</td>
      <td>1844.903592</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.900000</td>
      <td>24.46</td>
      <td>81.398935</td>
      <td>27.600000</td>
      <td>2201.836568</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.789512</td>
      <td>2.60</td>
      <td>98.149657</td>
      <td>31.779065</td>
      <td>21719.572490</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.980000</td>
      <td>112.08</td>
      <td>70.362420</td>
      <td>42.700000</td>
      <td>611.714745</td>
    </tr>
  </tbody>
</table>
</div>



We will now replace columns in our 'averages' dataset with columns from our 'filled-in-values' dataset


```python
# Copy filled-in values into our working 'averages' dataset

df_avg.cpi_score = df_avg_fill.cpi_score
df_avg.infant_mortality_rate = df_avg_fill.infant_mortality_rate
df_avg.literacy_rate = df_avg_fill.literacy_rate
df_avg.gini_index = df_avg_fill.gini_index
df_avg.gdp_per_capita = df_avg_fill.gdp_per_capita
```

Using a temporary dataframe, we obtained corrected metrics columns and then used these fixed columns to replace columns in our dataset. With the exception of literacy_rate for North America, all null values should now be filled.

Let us now inspect our working dataset.


```python
df_avg.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>region</th>
      <th>cpi_score</th>
      <th>infant_mortality_rate</th>
      <th>literacy_rate</th>
      <th>gini_index</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>Asia &amp; Pacific</td>
      <td>1.480000</td>
      <td>76.86</td>
      <td>39.000000</td>
      <td>37.144133</td>
      <td>6875.002351</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albania</td>
      <td>Europe</td>
      <td>3.160000</td>
      <td>15.44</td>
      <td>96.391969</td>
      <td>30.000000</td>
      <td>1844.903592</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>Arab States</td>
      <td>2.900000</td>
      <td>24.46</td>
      <td>81.398935</td>
      <td>27.600000</td>
      <td>2201.836568</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Andorra</td>
      <td>Europe</td>
      <td>5.789512</td>
      <td>2.60</td>
      <td>98.149657</td>
      <td>31.779065</td>
      <td>21719.572490</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Angola</td>
      <td>Africa</td>
      <td>1.980000</td>
      <td>112.08</td>
      <td>70.362420</td>
      <td>42.700000</td>
      <td>611.714745</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_avg.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 211 entries, 0 to 210
    Data columns (total 7 columns):
    country                  211 non-null object
    region                   211 non-null object
    cpi_score                211 non-null float64
    infant_mortality_rate    211 non-null float64
    literacy_rate            208 non-null float64
    gini_index               211 non-null float64
    gdp_per_capita           211 non-null float64
    dtypes: float64(5), object(2)
    memory usage: 11.6+ KB
    

As observed earlier, 'literacy_rate' has some null values. Let us see details of those records.


```python
# List records with Null

df_avg[pd.isnull(df_avg).any(axis=1)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>region</th>
      <th>cpi_score</th>
      <th>infant_mortality_rate</th>
      <th>literacy_rate</th>
      <th>gini_index</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>Bermuda</td>
      <td>North America</td>
      <td>7.99</td>
      <td>5.66</td>
      <td>NaN</td>
      <td>37.225</td>
      <td>65455.868678</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Canada</td>
      <td>North America</td>
      <td>8.78</td>
      <td>4.94</td>
      <td>NaN</td>
      <td>33.700</td>
      <td>25781.976322</td>
    </tr>
    <tr>
      <th>201</th>
      <td>United States</td>
      <td>North America</td>
      <td>7.20</td>
      <td>6.38</td>
      <td>NaN</td>
      <td>40.750</td>
      <td>37695.903618</td>
    </tr>
  </tbody>
</table>
</div>



As decided earlier, we will obtain the global mean of literacy_rate and assign this to North American countries.


```python
# We are interested in the mean of Literacy rate

df_avg.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cpi_score</th>
      <th>infant_mortality_rate</th>
      <th>literacy_rate</th>
      <th>gini_index</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>211.000000</td>
      <td>211.000000</td>
      <td>208.000000</td>
      <td>211.000000</td>
      <td>211.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.088697</td>
      <td>27.883904</td>
      <td>84.767862</td>
      <td>39.064161</td>
      <td>8890.097876</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.002674</td>
      <td>25.568050</td>
      <td>16.496973</td>
      <td>7.960935</td>
      <td>14024.659745</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.930000</td>
      <td>25.307745</td>
      <td>24.540000</td>
      <td>103.104815</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.610000</td>
      <td>7.550000</td>
      <td>81.383965</td>
      <td>33.035000</td>
      <td>878.625684</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.480000</td>
      <td>17.780000</td>
      <td>92.182549</td>
      <td>37.144133</td>
      <td>3036.536292</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.170000</td>
      <td>43.290000</td>
      <td>98.149657</td>
      <td>45.027143</td>
      <td>9054.292336</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.380000</td>
      <td>112.080000</td>
      <td>99.998262</td>
      <td>63.200000</td>
      <td>103885.246787</td>
    </tr>
  </tbody>
</table>
</div>



All columns now appear to have values, with exception of literacy rate. As observed earlier, literacy rate is missing values for 3 countries in North America.

To remedy this, we will use the global literacy rate average, 84.7, for these North American countries.


```python
# Fill in missing literacy rate values using global average

df_avg.literacy_rate = df_avg.literacy_rate.fillna(df_avg.literacy_rate.mean())
```

Let us confirm that the average literacy rate value has replaced the null North American values.


```python
df_avg.query('region == "North America"')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>region</th>
      <th>cpi_score</th>
      <th>infant_mortality_rate</th>
      <th>literacy_rate</th>
      <th>gini_index</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>20</th>
      <td>Bermuda</td>
      <td>North America</td>
      <td>7.99</td>
      <td>5.66</td>
      <td>84.767862</td>
      <td>37.225</td>
      <td>65455.868678</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Canada</td>
      <td>North America</td>
      <td>8.78</td>
      <td>4.94</td>
      <td>84.767862</td>
      <td>33.700</td>
      <td>25781.976322</td>
    </tr>
    <tr>
      <th>201</th>
      <td>United States</td>
      <td>North America</td>
      <td>7.20</td>
      <td>6.38</td>
      <td>84.767862</td>
      <td>40.750</td>
      <td>37695.903618</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_avg[pd.isnull(df_avg).any(axis=1)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>region</th>
      <th>cpi_score</th>
      <th>infant_mortality_rate</th>
      <th>literacy_rate</th>
      <th>gini_index</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



Missing literacy rate for North America was correctly updated.

One final check of our data, to confirm all cleaning actions have been successfully carried out.


```python
df_avg.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 211 entries, 0 to 210
    Data columns (total 7 columns):
    country                  211 non-null object
    region                   211 non-null object
    cpi_score                211 non-null float64
    infant_mortality_rate    211 non-null float64
    literacy_rate            211 non-null float64
    gini_index               211 non-null float64
    gdp_per_capita           211 non-null float64
    dtypes: float64(5), object(2)
    memory usage: 11.6+ KB
    


```python
df_avg.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cpi_score</th>
      <th>infant_mortality_rate</th>
      <th>literacy_rate</th>
      <th>gini_index</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>211.000000</td>
      <td>211.000000</td>
      <td>211.000000</td>
      <td>211.000000</td>
      <td>211.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.088697</td>
      <td>27.883904</td>
      <td>84.767862</td>
      <td>39.064161</td>
      <td>8890.097876</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.002674</td>
      <td>25.568050</td>
      <td>16.378714</td>
      <td>7.960935</td>
      <td>14024.659745</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.930000</td>
      <td>25.307745</td>
      <td>24.540000</td>
      <td>103.104815</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.610000</td>
      <td>7.550000</td>
      <td>81.383965</td>
      <td>33.035000</td>
      <td>878.625684</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.480000</td>
      <td>17.780000</td>
      <td>91.559827</td>
      <td>37.144133</td>
      <td>3036.536292</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.170000</td>
      <td>43.290000</td>
      <td>98.149657</td>
      <td>45.027143</td>
      <td>9054.292336</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.380000</td>
      <td>112.080000</td>
      <td>99.998262</td>
      <td>63.200000</td>
      <td>103885.246787</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_avg.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>region</th>
      <th>cpi_score</th>
      <th>infant_mortality_rate</th>
      <th>literacy_rate</th>
      <th>gini_index</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>Asia &amp; Pacific</td>
      <td>1.480000</td>
      <td>76.86</td>
      <td>39.000000</td>
      <td>37.144133</td>
      <td>6875.002351</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albania</td>
      <td>Europe</td>
      <td>3.160000</td>
      <td>15.44</td>
      <td>96.391969</td>
      <td>30.000000</td>
      <td>1844.903592</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>Arab States</td>
      <td>2.900000</td>
      <td>24.46</td>
      <td>81.398935</td>
      <td>27.600000</td>
      <td>2201.836568</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Andorra</td>
      <td>Europe</td>
      <td>5.789512</td>
      <td>2.60</td>
      <td>98.149657</td>
      <td>31.779065</td>
      <td>21719.572490</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Angola</td>
      <td>Africa</td>
      <td>1.980000</td>
      <td>112.08</td>
      <td>70.362420</td>
      <td>42.700000</td>
      <td>611.714745</td>
    </tr>
  </tbody>
</table>
</div>



Cleaning now appears complete, missing values appear to have been filled in.

What have we done so far? We have :
   
   - Consolidated our data from multiple files / sources into a single data set
   - Removed data for non-country entities
   - Resolved naming discrepancies, and combined observations where needed
   - Augmented our data by adding 'region' classification
   - Created a version of our dataset that holds the five-year average of observations in each metric
   - Identified missing values
   - Removed 'Poverty' metric from our dataset, and re-framed our questions, due to large number of missing values
   - Used Region averages to fill in missing values, and a global average where regional average was not available
   - Checked and confirmed all cleaning actions successsful and complete. 

As a result of these steps, our data is now ready for exploratory analysis.

Before we dive into exploratory analysis however, let us take a quick overview of distributions within our metrics, followed by a brief description of the range of values in each.


```python
df_avg.hist(figsize=(15,15));
```


![png](output_168_0.png)


**A brief reminder of the columns, what they measure and the value range.**


**CPI Score:** Ranks countries/territories based on how corrupt their public sector is perceived to be. 0 means that a country is perceived as highly corrupt and 10 means that a country is perceived as very clean.


**GDP per Capita:** Per capita GDP is a measure of the total output of a country that takes the gross domestic product (GDP) and divides it by the number of people in that country. Higher values for per capita GDP signals high productivity withiin a country. Values are in USD.


**GINI Index:** Measures the extent to which the distribution of income among individuals or households within an economy deviates from a perfectly equal distribution. A Gini index of 0 represents perfect equality, while an index of 100 implies perfect inequality.

**Infant Mortality Rate (IMR):** IMR has been defined as the number of deaths in children under 1 year of age per 1000 live births in the same year.

**Adult Literacy Rate:** Adult literacy rate is the percentage of people aged 15 years and above who can read and write.


<a id='eda'></a>
## Exploratory Data Analysis

Let us now explore our data, using our research questions as guide.

### Q1. Which five countries have the worst and best measures of income inequality?

A Gini index of 0 represents perfect equality, while an index of 100 implies perfect inequality.



```python
# Quick view of distribution of GINI data

df_avg.gini_index.hist()
plt.title('Spread and Count of GINI Index Values')
plt.ylabel('Frequency')
plt.xlabel('GINI Index');
```


![png](output_174_0.png)



```python
df_avg.gini_index.mean(), df_avg.gini_index.median()
```




    (39.064160699836485, 37.144133333333336)



Data appears to skew to the right, implying that if we split the data around the middle point, we will find there are more countries with high income inequality (low values in the index) than there are with low inequality. Let us identify the five highest and lowest. 

The next cell will identify the 5 highest and 5 lowest, and combine the results into a single dataframe.


```python
# Lowest income inequality
l = df_avg.gini_index.drop_duplicates().nsmallest(5)
df_lowest_gini = df_avg.query('gini_index == @l')
df_lowest_gini_t = df_lowest_gini.drop(['cpi_score', 'infant_mortality_rate', 'literacy_rate', 'gdp_per_capita'], axis=1)
df_lowest_gini_t['gini_type'] = 'Lowest inequality'

# Highest income inequality
h = df_avg.gini_index.drop_duplicates().nlargest(5)
df_highest_gini = df_avg.query('gini_index == @h')
df_highest_gini_t = df_highest_gini.drop(['cpi_score', 'infant_mortality_rate', 'literacy_rate', 'gdp_per_capita'], axis=1)
df_highest_gini_t['gini_type'] = 'Highest inequality'

# Combine lowest and Highest GINI
df_gini_lh_t = df_lowest_gini_t.append(df_highest_gini_t)
df_gini_lh_t
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>region</th>
      <th>gini_index</th>
      <th>gini_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>51</th>
      <td>Czech Republic</td>
      <td>Europe</td>
      <td>26.30</td>
      <td>Lowest inequality</td>
    </tr>
    <tr>
      <th>141</th>
      <td>Norway</td>
      <td>Europe</td>
      <td>26.26</td>
      <td>Lowest inequality</td>
    </tr>
    <tr>
      <th>166</th>
      <td>Slovakia</td>
      <td>Europe</td>
      <td>26.34</td>
      <td>Lowest inequality</td>
    </tr>
    <tr>
      <th>167</th>
      <td>Slovenia</td>
      <td>Europe</td>
      <td>24.54</td>
      <td>Lowest inequality</td>
    </tr>
    <tr>
      <th>198</th>
      <td>Ukraine</td>
      <td>CIS</td>
      <td>25.66</td>
      <td>Lowest inequality</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Botswana</td>
      <td>Africa</td>
      <td>60.50</td>
      <td>Highest inequality</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Central African Republic</td>
      <td>Africa</td>
      <td>56.20</td>
      <td>Highest inequality</td>
    </tr>
    <tr>
      <th>130</th>
      <td>Namibia</td>
      <td>Africa</td>
      <td>61.00</td>
      <td>Highest inequality</td>
    </tr>
    <tr>
      <th>170</th>
      <td>South Africa</td>
      <td>Africa</td>
      <td>63.20</td>
      <td>Highest inequality</td>
    </tr>
    <tr>
      <th>209</th>
      <td>Zambia</td>
      <td>Africa</td>
      <td>55.60</td>
      <td>Highest inequality</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot five highest and five lowest on a horizontal bar chart

df_gini_lh_t.sort_values(by=['gini_index'], inplace=True)
ax = df_gini_lh_t.plot(kind='barh')
ax.set_yticklabels(df_gini_lh_t.country)

ax.set_title('Top 5 and Bottom 5 GINI Countries')

# set individual bar labels
for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width(), i.get_y(), str(round((i.get_width()), 2)));
    
# Credit for annotations: http://robertmitchellv.com/blog-bar-chart-annotations-pandas-mpl.html
```


![png](output_178_0.png)


All five worst inequality scores are in Africa, while countries with best (lowest) inequality scores are in Europe. Let's use the 'region' calssification to see if this trend extends beyong the five highest and lowest.


```python
# GINI index by region 

df_avg.groupby('region')['gini_index'].mean()
```




    region
    Africa                 45.027143
    Arab States            33.212500
    Asia & Pacific         37.144133
    CIS                    30.420000
    Europe                 31.779065
    North America          37.225000
    South/Latin America    48.778125
    Name: gini_index, dtype: float64




```python
# Groupby 'region' in the main dataset, and plot a histogram on the value count

df_avg_mean = df_avg.groupby('region')['gini_index'].mean()
df_avg_mean.sort_values(inplace=True)
ax = df_avg_mean.plot(kind='barh')

ax.set_title('Average Income Inequality Score (GINI) by Region')
ax.set_xlabel('GINI Index')
ax.set_ylabel('Region')

for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width(), i.get_y(), str(round((i.get_width()), 2)));
```


![png](output_181_0.png)


Surprisingly, although Africa has the five worst countries with income inequality, the region with worst average income equality appears to be South America. Let us try to understand why by comparing inequality among three regions: Africa, Europe and South America


```python
# Create masks
africa = df_avg.region == 'Africa'
europe = df_avg.region == 'Europe'
s_america = df_avg.region == 'South/Latin America'

# Filter and plot
df_avg.gini_index[africa].hist(alpha=0.5, bins=5, label='Africa', figsize=(10,6))
df_avg.gini_index[europe].hist(alpha=0.5, bins=5, label='Europe', figsize=(10,6))
df_avg.gini_index[s_america].hist(alpha=0.5, bins=5, label='S. America', figsize=(10,6))
#plt.figure(figsize=(15,15))
plt.title('GINI Index Distribution in Africa, Europe and South America')
plt.ylabel('Count of Observations')
plt.xlabel('GINI Index Bins')

plt.legend();
```


![png](output_183_0.png)


The above histogram helps us understand our earlier finding. The GINI observations in South America are tightly clustered between 44 and 55. With Africa on the other hand, observations are spread across a broad value range. A few African countries appear to have low levels of income inequality (low GINI indexes), and these countries push Africa to a superior average score relative to South America. 

Lastly, let us identify these unusual African countries with low income inequality scores. 


```python
# Top five African countries with best income inequality measures

al = df_avg[africa].gini_index.drop_duplicates().nsmallest(5)
df_africa_lowest_gini = df_avg[africa].query('gini_index == @al')
df_africa_lowest_gini_t = df_africa_lowest_gini.drop(['cpi_score', 'infant_mortality_rate', 'literacy_rate', 'gdp_per_capita'], axis=1)
df_africa_lowest_gini_t['gini_type'] = 'Lowest inequality'
df_africa_lowest_gini_t.sort_values(by='gini_index')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>region</th>
      <th>gini_index</th>
      <th>gini_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>159</th>
      <td>Sao Tome and Principe</td>
      <td>Africa</td>
      <td>30.8</td>
      <td>Lowest inequality</td>
    </tr>
    <tr>
      <th>115</th>
      <td>Mali</td>
      <td>Africa</td>
      <td>33.0</td>
      <td>Lowest inequality</td>
    </tr>
    <tr>
      <th>62</th>
      <td>Ethiopia</td>
      <td>Africa</td>
      <td>33.2</td>
      <td>Lowest inequality</td>
    </tr>
    <tr>
      <th>164</th>
      <td>Sierra Leone</td>
      <td>Africa</td>
      <td>34.0</td>
      <td>Lowest inequality</td>
    </tr>
    <tr>
      <th>137</th>
      <td>Niger</td>
      <td>Africa</td>
      <td>34.4</td>
      <td>Lowest inequality</td>
    </tr>
  </tbody>
</table>
</div>



So our observation in relation to this question is that **the top five countries with lowest income inequality are in Europe. On the other hand, the top 5 countries with highest income inquality are in Africa.** However, a few Afrian countries seem to have admirably high levels of fair income distribution, so that on average, South America, not Africa, appears to have the worst income inequality.

### Q2. Is there a relationship between income inequality and corruption?

**Important Note:**

**With GINI, 0 is perfect income distribution and 100 is totally imperfect income distribution**

**With CPI, 0 is very corrupt and 10 is very clean**


```python
# Create dataframe with only these two columns

df_q2 = df_avg.drop(['infant_mortality_rate', 'literacy_rate', 'gdp_per_capita'], axis=1)

df_q2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 211 entries, 0 to 210
    Data columns (total 4 columns):
    country       211 non-null object
    region        211 non-null object
    cpi_score     211 non-null float64
    gini_index    211 non-null float64
    dtypes: float64(2), object(2)
    memory usage: 6.7+ KB
    

In earlier exploration, we have looked closely at the income inequality (GINI) data. Let us now examine data in the 'corruption perception' column.

As a reminder, CPI Score ranks countries based on how corrupt their public sector is perceived to be. **0 means that a country is perceived as highly corrupt and 10 means that a country is perceived as very clean.**


```python
#Plot  frequency distribution of cpi score

df_q2.cpi_score.plot(kind='hist')
plt.title('Frequency Distribution of CPI Score')
plt.xlabel('CPI Score');
```


![png](output_191_0.png)



```python
df_q2.cpi_score.describe()
```




    count    211.000000
    mean       4.088697
    std        2.002674
    min        1.000000
    25%        2.610000
    50%        3.480000
    75%        5.170000
    max        9.380000
    Name: cpi_score, dtype: float64



We note that the CPI distribution is skewed to the right, implying that a larger number of countries score lower than the average corruption perception.

Let us see how this differs by region.


```python
# List all regions

df_q2.region.drop_duplicates()
```




    0          Asia & Pacific
    1                  Europe
    2             Arab States
    4                  Africa
    5     South/Latin America
    7                     CIS
    20          North America
    Name: region, dtype: object




```python
# Create masks

asia = df_q2.region == 'Asia & Pacific'
europe = df_q2.region == 'Europe'
arab = df_q2.region == 'Arab States'
africa = df_q2.region == 'Africa'
s_america = df_q2.region == 'South/Latin America'
cis = df_q2.region == 'CIS'
n_america = df_q2.region == 'North America'
```


```python
# Filter dataframe by masks and plot histogram

df_q2.cpi_score[asia].hist(alpha=0.5, bins=2, label='Asia', figsize=(15,8))
df_q2.cpi_score[europe].hist(alpha=0.5, bins=2, label='Europe', figsize=(15,8))
df_q2.cpi_score[arab].hist(alpha=0.5, bins=2, label='Arab States', figsize=(15,8))
df_q2.cpi_score[africa].hist(alpha=0.5, bins=2, label='Africa', figsize=(15,8))
df_q2.cpi_score[s_america].hist(alpha=0.5, bins=2, label='S. America', figsize=(10,8))
df_q2.cpi_score[cis].hist(alpha=0.5, bins=2, label='CIS', figsize=(15,8))
df_q2.cpi_score[n_america].hist(alpha=0.5, bins=2, label='N. America', figsize=(15,8))

# Set title and label axes
plt.title('Frequency Distribution of CPI Score by Region')
plt.xlabel('CPI Score')
plt.ylabel('Frequency')

plt.legend();
```


![png](output_196_0.png)


Some interesting observations here regarding CPI Score.

Most countries in Africa, Asia, North America have poor scores on corruption perception, whereas Europe appears to have a small marority of countries with higher than average scores.

Africa in particular appears to have most countries scoring below the global average.

Given that this is a measure of perception as opposed to presence of corruption (which is near impossible to objectively measure), it can be pointed out that a population dissatisfied with its standard of living is likely to percieve it's public servants as corrupt. Obviously this perception will be further heightened when public sector inefficiency is combined with ostentatious living of public officials, and where the justice system does not act consistently and promptly against accused corrupt officers.



Let us now examine the data to see if a relationship exists between income inequality and corruption perception.

Recall that CPI goes from zero to ten, where zero means a country is percieved as highly corrupt. GINI goes from zero to 100, where 100 denotes perfect income inequality, that is, one person earns all income. So the measures are not travelling in the same direction. Careful interpretation and careful use of terms will therefore be required.


```python
# Plot a scatter diagram, using adjusted cpi scores

df_q2.plot(x='gini_index', y='cpi_score', kind='scatter', figsize=(8,8))
plt.title('Income Inequality (GINI) and Corruption Perception Index');
```


![png](output_200_0.png)



```python
df_q2.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cpi_score</th>
      <th>gini_index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>211.000000</td>
      <td>211.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.088697</td>
      <td>39.064161</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.002674</td>
      <td>7.960935</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>24.540000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.610000</td>
      <td>33.035000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.480000</td>
      <td>37.144133</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.170000</td>
      <td>45.027143</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.380000</td>
      <td>63.200000</td>
    </tr>
  </tbody>
</table>
</div>



From above, we observe a very weak negative correlation between income inequality and corruption. **The data therefore is inconclusive with regards to a relationship between income inequality and corruption perception.** 

We will nevertheless seek to confirm this graphical observation by computing the correlation coefficient of these two data columns.


```python
q2_correl = df_q2.gini_index.corr(df_q2.cpi_score)
print('Correlation coefficient between GINI and CPI: {}'.format(q2_correl))
```

    Correlation coefficient between GINI and CPI: -0.22182758050583504
    

Negative value of -0.2 suggests a very weak negative correlation between income inequality and corruption perception, in line with visual observation.

### Q3. Is GDP per capita a good predictor of income inequality?


```python
# Create dataframe with only these two columns

df_q3 = df_avg.drop(['cpi_score', 'infant_mortality_rate', 'literacy_rate'], axis=1)
df_q3.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 211 entries, 0 to 210
    Data columns (total 4 columns):
    country           211 non-null object
    region            211 non-null object
    gini_index        211 non-null float64
    gdp_per_capita    211 non-null float64
    dtypes: float64(2), object(2)
    memory usage: 6.7+ KB
    


```python
df_q3.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>region</th>
      <th>gini_index</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>Asia &amp; Pacific</td>
      <td>37.144133</td>
      <td>6875.002351</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albania</td>
      <td>Europe</td>
      <td>30.000000</td>
      <td>1844.903592</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>Arab States</td>
      <td>27.600000</td>
      <td>2201.836568</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Andorra</td>
      <td>Europe</td>
      <td>31.779065</td>
      <td>21719.572490</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Angola</td>
      <td>Africa</td>
      <td>42.700000</td>
      <td>611.714745</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_q3.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gini_index</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>211.000000</td>
      <td>211.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>39.064161</td>
      <td>8890.097876</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.960935</td>
      <td>14024.659745</td>
    </tr>
    <tr>
      <th>min</th>
      <td>24.540000</td>
      <td>103.104815</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>33.035000</td>
      <td>878.625684</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>37.144133</td>
      <td>3036.536292</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>45.027143</td>
      <td>9054.292336</td>
    </tr>
    <tr>
      <th>max</th>
      <td>63.200000</td>
      <td>103885.246787</td>
    </tr>
  </tbody>
</table>
</div>



We have earlier examined GINI data and observed its right-ward skew, confirmed by the median being lower than the mean.

Looking at the gdp_per_capita data, we also observe the median to be significantly below the mean, suggesting there are outliers with large values and the distribution is skewed heavily to the right. 

Let us use a histogram to check this.


```python
df_q3.gdp_per_capita.hist()
plt.title('Frequency Distribution of GDP Per Capita')
plt.ylabel('Frequency')
plt.xlabel('GDP Per Capita USD');

```


![png](output_210_0.png)


We can quickly look to see which countries have the highest GDP Per Capita, to better understand the chart.


```python
# Highest GDP Per Capita
gdp_high = df_q3.gdp_per_capita.drop_duplicates().nlargest(5)
df_largest_gdp = df_q3.query('gdp_per_capita == @gdp_high')
df_largest_gdp.sort_values(by='gdp_per_capita', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>region</th>
      <th>gini_index</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>123</th>
      <td>Monaco</td>
      <td>Europe</td>
      <td>31.779065</td>
      <td>103885.246787</td>
    </tr>
    <tr>
      <th>106</th>
      <td>Liechtenstein</td>
      <td>Europe</td>
      <td>31.779065</td>
      <td>82380.816170</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Bermuda</td>
      <td>North America</td>
      <td>37.225000</td>
      <td>65455.868678</td>
    </tr>
    <tr>
      <th>108</th>
      <td>Luxembourg</td>
      <td>Europe</td>
      <td>31.500000</td>
      <td>53424.811696</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Channel Islands</td>
      <td>Europe</td>
      <td>31.779065</td>
      <td>45226.031700</td>
    </tr>
  </tbody>
</table>
</div>



Monaco is the source of the large outlier value, and this is supported by other publicly available data sources, such as CIA World Factbook:

https://www.cia.gov/library/publications/the-world-factbook/rankorder/2004rank.html

One other point of concern is the suggestion, from the mean and the 25th percentile, that a significant number of countries have very low GDP per capita values. It will be instructive to see the countries with the ten lowest values, and possibly also see the distribution spread by region 


```python
# Lowest GDP per capita

gdp_low = df_q3.gdp_per_capita.drop_duplicates().nsmallest(10)
df_smallest_gdp = df_q3.query('gdp_per_capita in @gdp_low')
df_smallest_gdp.sort_values(by='gdp_per_capita')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>region</th>
      <th>gini_index</th>
      <th>gdp_per_capita</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>43</th>
      <td>Congo, Dem. Rep.</td>
      <td>Africa</td>
      <td>45.027143</td>
      <td>103.104815</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Burundi</td>
      <td>Africa</td>
      <td>45.027143</td>
      <td>136.974269</td>
    </tr>
    <tr>
      <th>60</th>
      <td>Eritrea</td>
      <td>Africa</td>
      <td>45.027143</td>
      <td>152.838384</td>
    </tr>
    <tr>
      <th>76</th>
      <td>Guinea-Bissau</td>
      <td>Africa</td>
      <td>50.700000</td>
      <td>160.105497</td>
    </tr>
    <tr>
      <th>112</th>
      <td>Malawi</td>
      <td>Africa</td>
      <td>45.500000</td>
      <td>172.459054</td>
    </tr>
    <tr>
      <th>137</th>
      <td>Niger</td>
      <td>Africa</td>
      <td>34.400000</td>
      <td>175.920115</td>
    </tr>
    <tr>
      <th>62</th>
      <td>Ethiopia</td>
      <td>Africa</td>
      <td>33.200000</td>
      <td>203.703223</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Central African Republic</td>
      <td>Africa</td>
      <td>56.200000</td>
      <td>228.850633</td>
    </tr>
    <tr>
      <th>104</th>
      <td>Liberia</td>
      <td>Africa</td>
      <td>36.500000</td>
      <td>247.134621</td>
    </tr>
    <tr>
      <th>111</th>
      <td>Madagascar</td>
      <td>Africa</td>
      <td>42.400000</td>
      <td>249.453045</td>
    </tr>
  </tbody>
</table>
</div>



All ten countries with lowest GDP per capita are in Africa, with the lowest value being Democratic Republic of Congo, showing **GDP per capita of USD103**. Contrasting this with Monaco at over **USD103,000** or even with war-torn Afghanistan at **USD6,875** is a shocking illustration of differences in standards of living across the world.

Let us see the average GDP Per Capita by region.


```python
# GDP Per Capita by Region

df_q3_gdp_region = df_q3.groupby('region')['gdp_per_capita'].mean()
df_q3_gdp_region.sort_values(inplace=True)

ax = df_q3_gdp_region.plot(kind='barh')

ax.set_title('GDP Per Capita by Region')
ax.set_xlabel('GDP Per Capita USD')
ax.set_ylabel('Region')

for i in ax.patches:
    # get_width pulls left or right; get_y pushes up or down
    ax.text(i.get_width(), i.get_y(), str(round((i.get_width()), 2)));
```


![png](output_217_0.png)


Chart above illustrates the huge discrepancies in national wealth and productivity, given that GDP Per Capita is gross national income divided by population.

North America has by far the highest values, most likely driven by the huge USD19 trillion GDP of the USA.

One important point to add is that the GDP Per Capita figure is not adjusted for cost of living differences. Cost of living comparison website https://www.numbeo.com/cost-of-living/ for example, estimates that consumer prices including rent are 40% lower in South Africa compared to USA. This number will be higher for poorer African countries, suggesting that living conditions may not be as stark as the raw difference in GDP per capita suggests.

We will now draw a scatter plot to check for a relationship between GINI index and GDP per capita


```python
# Plot a scatter diagram, using adjusted cpi scores

df_q3.plot(x='gini_index', y='gdp_per_capita', kind='scatter', figsize=(8,8))
plt.title('Income Inequality (GINI) and GDP per Capita')
plt.ylabel('GDP Per Capita USD')
plt.xlabel('Income Inequality (GINI)');
```


![png](output_220_0.png)


From scatter plot above, there appears to be a weak negative correlation between income inequality and GDP per capita. This suggests that **the lower gdp per capita in a given country, the greater its income inequality is likely to be.** Put another way, **richer countries tend to have a fairer income distribution among their citizens.** 

Again, this association is weak and is not considered causative, merely correlative.

We will confirm this obervation by calculating the correlation coefficient.


```python
q3_correl = df_q3.gini_index.corr(df_q3.gdp_per_capita)
```


```python
print('Correlation coefficient between income inequality and GDP per Capita: {}'.format(q3_correl))
```

    Correlation coefficient between income inequality and GDP per Capita: -0.3038398435955632
    

Negative coefficient of -0.3 confirms the visual observation.

### Q4. Does a correlation exist between literacy and health?

In this section, we seek to understand if any correlation exists between literacy and population health. This question is relevant because, for example, it will be useful for public policy if it can be shown that a literate population is more likely to be healthy.

As a reminder:

**Adult Literacy Rate:** Adult literacy rate is the percentage of people aged 15 years and above who can read and write.

**Infant Mortality Rate (IMR):** IMR has been defined as the number of deaths in children under 1 year of age per 1000 live births in the same year.

Let us examine the data.


```python
# Create dataframe with only these relevant columns

df_q4 = df_avg.drop(['cpi_score', 'gdp_per_capita', 'gini_index'], axis=1)
df_q4.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 211 entries, 0 to 210
    Data columns (total 4 columns):
    country                  211 non-null object
    region                   211 non-null object
    infant_mortality_rate    211 non-null float64
    literacy_rate            211 non-null float64
    dtypes: float64(2), object(2)
    memory usage: 6.7+ KB
    


```python
df_q4.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>region</th>
      <th>infant_mortality_rate</th>
      <th>literacy_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>Asia &amp; Pacific</td>
      <td>76.86</td>
      <td>39.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Albania</td>
      <td>Europe</td>
      <td>15.44</td>
      <td>96.391969</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Algeria</td>
      <td>Arab States</td>
      <td>24.46</td>
      <td>81.398935</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Andorra</td>
      <td>Europe</td>
      <td>2.60</td>
      <td>98.149657</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Angola</td>
      <td>Africa</td>
      <td>112.08</td>
      <td>70.362420</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_q4.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>infant_mortality_rate</th>
      <th>literacy_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>211.000000</td>
      <td>211.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>27.883904</td>
      <td>84.767862</td>
    </tr>
    <tr>
      <th>std</th>
      <td>25.568050</td>
      <td>16.378714</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.930000</td>
      <td>25.307745</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7.550000</td>
      <td>81.383965</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>17.780000</td>
      <td>91.559827</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>43.290000</td>
      <td>98.149657</td>
    </tr>
    <tr>
      <th>max</th>
      <td>112.080000</td>
      <td>99.998262</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Histogram of Literacy Rate

df_q4.literacy_rate.plot(kind='hist')
plt.title('Frequency Distribution of Adult Literacy Rate')
plt.xlabel('Literacy Rate %');
```


![png](output_230_0.png)


Literacy rate is the percentage of adults aged 15 and over who can read and write. 

While it's heartening to observe that most countries seem to have a high percentage of educated adults, we observe that in a small number of countries, there are disturbingly high numbers of adults wo are unable to read and write. 

Let us quickly try to dentify these countries. 


```python
# Copy dataframe and remove unneeded column
literacy_nsmall = df_q4.copy()
literacy_nsmall = literacy_nsmall.drop('infant_mortality_rate', axis=1)

# Filter to 10 lowest literacy rate countries
literacy_nsmall_list = literacy_nsmall.literacy_rate.drop_duplicates().nsmallest(10)
literacy_nsmall = literacy_nsmall.query('literacy_rate in @literacy_nsmall_list')

# Sort dataframe
literacy_nsmall.sort_values('literacy_rate', inplace=True)
literacy_nsmall
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>region</th>
      <th>literacy_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>75</th>
      <td>Guinea</td>
      <td>Africa</td>
      <td>25.307745</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Burkina Faso</td>
      <td>Africa</td>
      <td>28.729214</td>
    </tr>
    <tr>
      <th>115</th>
      <td>Mali</td>
      <td>Africa</td>
      <td>32.270483</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Chad</td>
      <td>Africa</td>
      <td>35.391470</td>
    </tr>
    <tr>
      <th>62</th>
      <td>Ethiopia</td>
      <td>Africa</td>
      <td>38.995982</td>
    </tr>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>Asia &amp; Pacific</td>
      <td>39.000000</td>
    </tr>
    <tr>
      <th>104</th>
      <td>Liberia</td>
      <td>Africa</td>
      <td>42.941084</td>
    </tr>
    <tr>
      <th>164</th>
      <td>Sierra Leone</td>
      <td>Africa</td>
      <td>43.283100</td>
    </tr>
    <tr>
      <th>161</th>
      <td>Senegal</td>
      <td>Africa</td>
      <td>49.695127</td>
    </tr>
    <tr>
      <th>128</th>
      <td>Mozambique</td>
      <td>Africa</td>
      <td>50.583811</td>
    </tr>
  </tbody>
</table>
</div>



The ten countries with lowest literacy rates are overwhelmingly in Africa, with Guinea, Burkina Faso and Mali as the lowest three at 25%, 29%, and 32%.

Let us perform the same quick check of countries with hightest literacy rates.


```python
# Copy dataframe and remove unneeded column
literacy_nlarge = df_q4.copy()
literacy_nlarge = literacy_nlarge.drop('infant_mortality_rate', axis=1)

# Filter to 10 highest literacy rate countries
literacy_nlarge_list = literacy_nlarge.literacy_rate.drop_duplicates().nlargest(10)
literacy_nlarge = literacy_nlarge.query('literacy_rate in @literacy_nlarge_list')

# Sort dataframe
literacy_nlarge.sort_values(by='literacy_rate', ascending=False, inplace=True)
literacy_nlarge
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>region</th>
      <th>literacy_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>140</th>
      <td>North Korea</td>
      <td>Asia &amp; Pacific</td>
      <td>99.998262</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Cuba</td>
      <td>South/Latin America</td>
      <td>99.834250</td>
    </tr>
    <tr>
      <th>61</th>
      <td>Estonia</td>
      <td>Europe</td>
      <td>99.796890</td>
    </tr>
    <tr>
      <th>101</th>
      <td>Latvia</td>
      <td>Europe</td>
      <td>99.784240</td>
    </tr>
    <tr>
      <th>68</th>
      <td>Georgia</td>
      <td>CIS</td>
      <td>99.732470</td>
    </tr>
    <tr>
      <th>94</th>
      <td>Kazakhstan</td>
      <td>CIS</td>
      <td>99.732411</td>
    </tr>
    <tr>
      <th>150</th>
      <td>Poland</td>
      <td>Europe</td>
      <td>99.730190</td>
    </tr>
    <tr>
      <th>198</th>
      <td>Ukraine</td>
      <td>CIS</td>
      <td>99.718740</td>
    </tr>
    <tr>
      <th>185</th>
      <td>Tajikistan</td>
      <td>CIS</td>
      <td>99.707060</td>
    </tr>
    <tr>
      <th>107</th>
      <td>Lithuania</td>
      <td>Europe</td>
      <td>99.703550</td>
    </tr>
  </tbody>
</table>
</div>



Some surprise appearances are North Korea, Cuba and Kazakhstan.

We should take a similar look at our infant mortality rate data: distribution, ten highest and ten lowest. After taking an overview of the data in this way, we will finally attempt to answer the research question.

Recall that infant mortality rate is the number of deaths in children under 1 year of age per 1000 live births, and this measure is seen as a good indicator of population health.

Let us start by looking at the distribution of infant mortality rates.


```python
df_q4.infant_mortality_rate.plot(kind='hist')
plt.title('Distribution of Infant Mortality Rates')
plt.ylabel('Frequency')
plt.xlabel('Infant Mortality Rate');
```


![png](output_238_0.png)


Data is skewed to right, with most countries having low IMR. Unfortunately, however, a few countries have high IMR values. We will take a look at the ten highest and ten lowest IMR countries.


```python
# Copy dataframe and remove unneeded column
imr_nlarge = df_q4.copy()
imr_nlarge.drop('literacy_rate', axis=1, inplace=True)

# Filter to 10 highest imr
imr_nlarge_list = imr_nlarge.infant_mortality_rate.drop_duplicates().nlargest(10)
imr_nlarge = imr_nlarge.query('infant_mortality_rate in @imr_nlarge_list')

# Sort dataframe
imr_nlarge.sort_values(by='infant_mortality_rate', ascending=False, inplace=True)
imr_nlarge
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>region</th>
      <th>infant_mortality_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>Angola</td>
      <td>Africa</td>
      <td>112.08</td>
    </tr>
    <tr>
      <th>164</th>
      <td>Sierra Leone</td>
      <td>Africa</td>
      <td>111.54</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Central African Republic</td>
      <td>Africa</td>
      <td>103.48</td>
    </tr>
    <tr>
      <th>169</th>
      <td>Somalia</td>
      <td>Arab States</td>
      <td>99.80</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Chad</td>
      <td>Africa</td>
      <td>95.08</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Congo, Dem. Rep.</td>
      <td>Africa</td>
      <td>87.00</td>
    </tr>
    <tr>
      <th>115</th>
      <td>Mali</td>
      <td>Africa</td>
      <td>85.30</td>
    </tr>
    <tr>
      <th>138</th>
      <td>Nigeria</td>
      <td>Africa</td>
      <td>84.44</td>
    </tr>
    <tr>
      <th>59</th>
      <td>Equatorial Guinea</td>
      <td>Africa</td>
      <td>81.34</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Cote d'Ivoire</td>
      <td>Africa</td>
      <td>79.26</td>
    </tr>
  </tbody>
</table>
</div>



All ten countries with highest IMR are in Africa, including Somalia which our region classification shows as an Arab State. Let us also look at the ten countries with lowest IMR. 


```python
# Copy dataframe and remove unneeded column
imr_small = df_q4.copy()
imr_small.drop('literacy_rate', axis=1, inplace=True)

# Filter to 10 lowest imr
imr_small_list = imr_small.infant_mortality_rate.drop_duplicates().nsmallest(10)
imr_small = imr_small.query('infant_mortality_rate in @imr_small_list')

# Sort dataframe
imr_small.sort_values(by='infant_mortality_rate', inplace=True)
imr_small
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>region</th>
      <th>infant_mortality_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>106</th>
      <td>Liechtenstein</td>
      <td>Europe</td>
      <td>1.93</td>
    </tr>
    <tr>
      <th>82</th>
      <td>Iceland</td>
      <td>Europe</td>
      <td>1.98</td>
    </tr>
    <tr>
      <th>108</th>
      <td>Luxembourg</td>
      <td>Europe</td>
      <td>2.06</td>
    </tr>
    <tr>
      <th>165</th>
      <td>Singapore</td>
      <td>Asia &amp; Pacific</td>
      <td>2.22</td>
    </tr>
    <tr>
      <th>92</th>
      <td>Japan</td>
      <td>Asia &amp; Pacific</td>
      <td>2.44</td>
    </tr>
    <tr>
      <th>181</th>
      <td>Sweden</td>
      <td>Europe</td>
      <td>2.54</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Andorra</td>
      <td>Europe</td>
      <td>2.60</td>
    </tr>
    <tr>
      <th>64</th>
      <td>Finland</td>
      <td>Europe</td>
      <td>2.62</td>
    </tr>
    <tr>
      <th>141</th>
      <td>Norway</td>
      <td>Europe</td>
      <td>2.74</td>
    </tr>
    <tr>
      <th>167</th>
      <td>Slovenia</td>
      <td>Europe</td>
      <td>2.86</td>
    </tr>
  </tbody>
</table>
</div>



The ten lowest IMR countries are shown above. The metric suggests high levels of populations health for these countries.

We will now investigate if any relationship exists between adult literacy and health.


```python
# Single shot view of distributions and correlations

pd.plotting.scatter_matrix(df_q4);
```


![png](output_244_0.png)



```python
# Scatter plot between literacy and IMR

q4_1 = df_q4.plot(kind='scatter', x='literacy_rate', y='infant_mortality_rate')
q4_1.set_title('Literacy and Infant Mortality Rate')
q4_1.set_xlabel('Literacy Rate')
q4_1.set_ylabel('Infant Mortality Rate');
```


![png](output_245_0.png)


There appears to be a strong negative correlation between literacy and infant mortality. **The higher the rate of adult literacy, the fewer infant deaths are recorded, indicating a healthier populations are associated with literate populations.**

Let us confirm this by calculating the correlation coefficient.


```python
# Calculate correlation coefficient between literacy and infant mortality

df_q4.literacy_rate.corr(df_q4.infant_mortality_rate)
```




    -0.7603129044114025



High negative coefficient of -0.76 confirms the visual observation.

It must be repeated that this merely indicates a correlation and does not confirm causation. 

<a id='conclusions'></a>
## Conclusions

We will now summarize the findings from our data exploration, performed under the guidance of four questions.

It must be stressed that the main tool of analysis here was descriptive statistics. And even that was limited to mainly correlation. This analysis therefore makes no claims about a causal relation between any of the measures used.

Nevetheless, this analysis has explored multiple socio-economic indicators across countries of the world. For each indicator, we have identified global and regional frequency distributions, as well as top and bottom countries. So even though the research questions focused on correlation-type questions, our journey through the exploration process exposed some interesting learnings. 

For example, our exploration showed that while we find the worst cases of income inequality in Africa, on average by region it is South America that shows the worst levels.

We also found, surprisingly for me, that North Korea, Cuba and Kazakhstan are in the top six most literate countries in the world.

These insights and correlations exposed in this project can serve as a useful entry point into deeper analysis of the research questions posed here.

### Question 1 Findings: Which five countries have the highest and lowest income inequality?

Our observation in relation to this question is that the five countries with lowest income inequality are in Europe. They are Slovenia, Ukraine, Norway, Czech Republic, Slovakia.

On the other hand, the five countries with highest income inequality are in Africa. They are South Africa, Namibia, Botswana, Central African Republic, and Zambia. 

However, a few Afrian countries seem to have admirably high levels of fair income distribution, so that on average, South America, not Africa, appears to have the worst income inequality.

### Question 2 Findings: Is there a relationship between income inequality and corruption?

Our observation is that a very weak negative correlation exists between income inequality and corruption. **The data therefore is inconclusive with regards to a relationship between income inequality and corruption perception.** 


### Question 3 Findings: Is GDP per capita a good predictor of income inequality?

From analysis of our data, it appears there is a negative correlation between income inequality and GDP per capita. This suggests that **the lower the gdp per capita in a country, the greater its income inequality is likely to be.** Put another way, **richer countries tend to have a fairer income distribution among their citizens.** 

Again, this association is not considered to be causative, merely correlative.

### Question 4 Findings: Does a correlation exist between literacy and health?

There appears to be a strong negative correlation between literacy and infant mortality. **This implies that the higher the rate of adult literacy, the fewer infant deaths are recorded, or put another way, healthy populations are likely to be literate.**

## Resources

### Coding resources

- Python For Data Analysis 2nd Edition, Wes Mckinney, O'REILLY, ISBN: 9781491957660
- https://stackoverflow.com/
- http://robertmitchellv.com/blog-bar-chart-annotations-pandas-mpl.html
- https://matplotlib.org/api/
- http://pandas.pydata.org/pandas-docs/stable/
- https://morphocode.com/pandas-cheat-sheet/



### Definitions and Additional Information on metrics
- https://www.cia.gov/library/publications/the-world-factbook/rankorder/2004rank.html
- https://www.transparency.org/cpi2011/results
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1732453/
- https://data.worldbank.org/indicator/SI.POV.GINI 
- https://meta.wikimedia.org/wiki/List_of_countries_by_regional_classification
- https://www.numbeo.com/cost-of-living/


