

pacman::p_load(tidyverse, reticulate, reshape2, caret, fastDummies, MASS, doParallel, 
               ggpubr, RANN, mice, e1071, mda, caretEnsemble)


data = py$set1

data$response = as.factor(ifelse(data$MAKE == "FORD", 1, 0))

### What's our response?

pie = data  %>%
  mutate(response = as.numeric(response)) %>%
  group_by(response) %>%
  summarise(weight = sum(response/response)/nrow(data), name = ifelse(weight > 0.5, "FORD", "Other manufacturers"), lab.ypos = cumsum(weight) - 0.5*weight)

ggplot(pie, aes(x = "", y = weight, fill = name)) +
  geom_bar(width = 1, stat = "identity", color = "white") +
  coord_polar("y", start = 0)+
  geom_text(aes(label=round(weight * 100,digits=3), y = lab.ypos, label = weight), color = "white")+
  scale_fill_manual(values = c("blue3", "#868686FF")) +
  theme_void() +
  labs(title="Proportion of Ford trucks", 
       subtitle="21.2% of the registered truck crashes are FORD trucks",
       caption="Pere Fuster - UC3M")


### Specifying factors and numeric variables 

data$time = as.factor(ifelse(data$CRASH_HOUR == 5 | data$CRASH_HOUR == 6 | data$CRASH_HOUR == 7 | data$CRASH_HOUR == 8 , "Night Morning", 
                             ifelse(data$CRASH_HOUR == 9 | data$CRASH_HOUR == 10 | data$CRASH_HOUR == 11, "Day Morning",
                                    ifelse(data$CRASH_HOUR == 12 | data$CRASH_HOUR == 13 | data$CRASH_HOUR == 14, "Early afternoon",
                                           ifelse(data$CRASH_HOUR == 15 | data$CRASH_HOUR == 16 | data$CRASH_HOUR == 17, "Late afternoon",
                                                  ifelse(data$CRASH_HOUR == 18 | data$CRASH_HOUR == 19 | data$CRASH_HOUR == 20, "Evening", "Night"))))))

data$CRASH_DAY_OF_WEEK = as.factor(data$CRASH_DAY_OF_WEEK) 
data$CRASH_MONTH = as.factor(data$CRASH_MONTH) 

x = rep(0, nrow(data))
for (i in 1:nrow(data)){
  if (grepl("2014", data$CRASH_DATE_x[i], fixed=TRUE)){
    x[i] = 2014}
  else if (grepl("2015", data$CRASH_DATE_x[i], fixed=TRUE)){
    x[i] = 2015}
  else if (grepl("2016", data$CRASH_DATE_x[i], fixed=TRUE)){
    x[i] = 2016}
  else if (grepl("2017", data$CRASH_DATE_x[i], fixed=TRUE)){
    x[i] = 2017}
  else if (grepl("2018", data$CRASH_DATE_x[i], fixed=TRUE)){
    x[i] = 2018}
  else if (grepl("2019", data$CRASH_DATE_x[i], fixed=TRUE)){
    x[i] = 2019}
  else if (grepl("2020", data$CRASH_DATE_x[i], fixed=TRUE)){
    x[i] = 2020}
} # There are no dates previous to 2015

data$CAR_AGE = x - data$VEHICLE_YEAR

# I received the error below and VEHICLE_DEFECT_NONE has also been removed
#> init = mice(ford, maxit=5) 
#Error in str2lang(x) : <text>:1:169: unexpected symbol
#1: UNIT_TYPE_DRIVER ~ 0+UNIT_TYPE_DRIVERLESS+UNIT_TYPE_PARKED+VEHICLE_DEFECT_NONE+VEHICLE_DEFECT_OTHER+VEHICLE_USE_COMMERCIAL - MULTI-UNIT+VEHICLE_USE_COMMERCIAL - SINGLE UNIT
# I realised this variable makes sense to have is as parked or not 
#table(categorical$UNIT_TYPE)
#
#ISABLED VEHICLE              DRIVER          DRIVERLESS NON-CONTACT VEHICLE              PARKED 
#1               16592                 135                   8                1771

# For interpretatibility, 
data$PARKED = as.factor(ifelse(data$UNIT_TYPE == "PARKED", 1, 0))



















