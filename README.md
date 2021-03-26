# Data_mining_6103_p1
1 Data collection and pre-processing 
	1.1 data source
    I used three data sets. They are all coming from World Bank, which ensure the reliability of all the data. The first data set gives the percent of the military spending on GDP for all the countries. The second and third one shows the gdp and population worldwide. 
	
  1.2 select partial data
    Since I only want to focus on 10 countries and 8 years, which is from 2010 to 2017, I need to subset three data sets. 
	
  1.3 Checking missing values
    Then I checked missing values in three data sets. Fortunately, there is no missing value.
	
  1.4 Merge Data sets
    For convience, I merged three data sets into one. 
	
  1.5 Calculate total military spending
    Since we only know the percent of the military spending on gdp and total gdp, we are able to calculate the total military spending in dollar to make the further analysis easier. 


2 Analysis
	2.1 Compare the military data to that country’s GDP
    I draw the scatter plot with one linear regression line to compare the military data to gdp for every country. As you can see, the military total spending has positive relationship between gdp, which means the increase in military spending means increase in gdp. However, there is only one exception. For united states the military spending has negative relationship with gdp. ???

	2.2 Compare the overall military spending of the all 10 countries in absolute and percentages
    Then I draw the PIE graph to compare the overall military spending of 10 countries in 8 years. The result is pretty consistent. For other countries, the United State spends the most on the military. The second position is China. 

	2.3 Compare the per person military spending to the per person GDP in absolute and percentages
    In order to compare them, we need to create three new columns. per person military spending can be derived by using military spending divided by population. per person GDP should be using the gdp divided by population. Then, we use per person military spending to divide per person gdp. FROM THE BAR GRAPH, We find the ratio between per person military spending to the per person gdp is biggest for Saudi Arabia in every year. 

	2.4 Single out the fastest growing countries in military spending in absolute and percentage
    In order to figure out the fastest growing countries in military spending, firstly, I want to draw the military spending for each country during 2010-2017. From the line graph, we can predict the growth rate in military spending is biggest. In order to verify our prediction, then, we calculate the growth rate from 2010 to 2017 for each country by the formula here. Finally, we draw a bar graph to show the growth rate for each country. As you can see, China has biggest growth rate among other countries. 

In conclusion, from our analysis, we find the correlation of military expenditures and economic growth presented by gdp is positive. The United State spends the most on the military. the ratio between per person military spending to the per person gdp is biggest for Saudi Arabia in every year. China has biggest growth rate among other countries.

