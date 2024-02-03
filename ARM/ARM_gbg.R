####################################################
###
### Association Rule Mining Examples
### This example will use the Apriori Alg.
### 
### Code courtesy of Dr. Ami Gates, CU Boulder 
####################################################
## To perform Association Rule Mining, transaction 
## data is needed. 
## MANY libraries will also be required... 
## 
## To perform association rule mining, you must have
## transaction data AND you must create a datafile
## that presents that data in whatever way your library
## expects. I will use *basket* here.
##
## 

## TO GET this code to work and to get arulesViz to work - 
## you will have to carefully and patiently do the following:

## DO these installs once
## install.packages("arules")
## For arulesViz to work on R version 3.5.x, you will
## need to first go through an installation of RTools. 
## See HOW TO above.
## Next - once the RTools exe has been downloaded and installed
## per the instructions, then, do these install.packages here in RStudio:
## install.packages("TSP")
## install.packages("data.table")
## NOTE: If you are asked if you want to INSTALL FROM SOURCE - click YES!
# install.packages("arulesViz", dependencies = TRUE)
## IMPORTANT ## arules ONLY grabs rules with ONE item on the right
## install.packages("sp")
## NOTE R V3.5.0 does not use the older
## datasets packages
## install.packages("datasets.load") - not used here
## install.packages("ggplot2") - not used here

## install.packages("dplyr", dependencies = TRUE)
## install.packages("purrr", dependencies = TRUE)
## install.packages("devtools", dependencies = TRUE)
## install.packages("tidyr")
library(viridis)
library(arules)
library(TSP)
library(data.table)
#library(ggplot2)
#library(Matrix)
library(tcltk)
library(dplyr)
#library(devtools)
library(purrr)
library(tidyr)
## DO THIS ONCE
## FIRST - you MUST register and log into github
## install_github("mhahsler/arulesViz")
## RE: https://github.com/mhahsler/arulesViz

##############
## IF YOUR CODE BREAKS - TRY THIS
##
## Error in length(obj) : Method length not implemented for class rules 
## DO THIS: 
## (1) detach("package:arulesViz", unload=TRUE)
## (2) detach("package:arules", unload=TRUE)
## (3) library(arules)
## (4) library(arulesViz)
###################################################################

## To see if you have tcltk run this on the console...
# capabilities()["tcltk"]
library(arulesViz)
library(RColorBrewer)

## YOUR working dir goes here...
setwd("C:/Users/casey/OneDrive/Documents/MSDS_Courses/Spring_2023/Machine_Learning/ARM")

# -------------------------------------------------------------------------------- #
## LOAD in  and inspect data

gbg <- read.transactions("prepped_data/gbg_prepped_for_ARM.csv",
                           rm.duplicates = FALSE, 
                           format = "basket",  ##if you use "single" also use cols=c(1,2)
                           sep=",",  ## csv file
                           cols=NULL) ## The dataset has no row numbers
inspect(gbg)

# --------------------------------------------------------------------------------- #
##Use apriori to get the RULES
Frules = arules::apriori(gbg, parameter = list(support=.35, 
                                                 confidence=.5, minlen=2))
inspect(Frules)

# --------------------------------------------------------------------------------- #
## ANALYZE the RULES

## Plot of which items are most frequent
itemFrequencyPlot(gbg, topN=10, type="absolute")

## Sort rules by a measure such as conf, sup, or lift
SortedRules <- sort(Frules, by="confidence", decreasing=TRUE)
inspect(SortedRules[1:10])
(summary(SortedRules))

## Selecting or targeting specific rules with 'Under' in the left hand side
UnderRules <- apriori(data=gbg,parameter = list(supp=.0005, conf=.6, minlen=2),
                     appearance = list(default='lhs', rhs="Under"),
                     control=list(verbose=FALSE))
UnderRules <- sort(UnderRules, decreasing=TRUE, by="confidence")
inspect(UnderRules[1:10])

## Selecting or targeting specific rules with 'Over' in the left hand side
OverRules <- apriori(data=gbg,parameter = list(supp=.0005, conf=.55, minlen=2),
                       appearance = list(default="lhs", rhs="Over"),
                       control=list(verbose=FALSE))
OverRules <- sort(OverRules, decreasing=TRUE, by="confidence")
inspect(OverRules[1:8])

# Write the rules to a text file
write(OverRules[1:8], file='OverRules.txt', sep= ", ", quote=FALSE)

# --------------------------------------------------------------------------- #
## Visualize RULES

subrules <- head(sort(UnderRules, by="lift"),10)
plot(subrules)

#plot(subrules, method="graph", engine="interactive")
Graph <- plot(OverRules[1:10], method="graph", engine="htmlwidget",
     title = 'Top 10 Rules Based on Confidence (lhs="Over")')
Graph

# Save the widget as an html file
htmlwidgets::saveWidget(widget = Graph,
           file = "Over_Confidence.html",
           selfcontained = TRUE)

