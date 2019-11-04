#Survival Analysis HW 1

#Importing Data
library(sas7bdat)
HurricaneDF <- read.sas7bdat("C:/Users/17708/Downloads/Homework1_SA/hurricane.sas7bdat")

#Querying to create dfs
library(sqldf)
Survivors <- sqldf("select * from HurricaneDF where survive = 1")
Fail1 <- sqldf("select * from HurricaneDF where survive = 0 and reason = 1")
Fail2 <- sqldf("select * from HurricaneDF where survive = 0 and reason = 2")
Fail3 <- sqldf("select * from HurricaneDF where survive = 0 and reason = 3")
Fail4 <- sqldf("select * from HurricaneDF where survive = 0 and reason = 4")

#Determining proportions
(316/770)*100
(115/770)*100
(112/770)*100
(111/770)*100
(116/770)*100

#Determining avg fail time
mean(Fail1$hour)
mean(Fail2$hour)
mean(Fail3$hour)
mean(Fail4$hour)

#One-Way ANOVA DF
AnovaDF <- sqldf("select hour,reason from HurricaneDF where reason > 0")

library(dplyr)
group_by(AnovaDF, reason) %>%
  summarise(
    count = n(),
    mean = mean(hour, na.rm = TRUE),
    sd = sd(hour, na.rm = TRUE)
  )

#Levene's Test for Equal Vars
library(car)
leveneTest(hour ~ factor(reason), data = AnovaDF)

#One-Way ANOVA
res.aov <- aov(hour ~ factor(reason), data = AnovaDF)

summary(res.aov)

#Tukey
TukeyHSD(res.aov)

#QQ PLot for Normality
plot(res.aov, 2)

# Survival Function
install.packages("survival")
install.packages("survminer")

library(survival)
library(survminer)
Hurricane_surv <- Surv(time = HurricaneDF$hour, event = HurricaneDF$survive == 0)

Hurricane_km <- survfit(Hurricane_surv ~ 1, data = HurricaneDF)
summary(Hurricane_km)
plot(Hurricane_km, main = "Survival Function", xlab = "Hour", ylab = "Survival Probability")

#Survival Probs for all types together 
ggsurvplot(Hurricane_km, data = HurricaneDF, conf.int = TRUE, palette = "purple",
           xlab = "Hour", ylab = "Survival Probability", legend = "none",
           break.y.by = 0.1)

#Survival Probs by type
survdiff(Hurricane_surv ~ reason, rho = 0, data = HurricaneDF)

Hurricane_strat <- survfit(Hurricane_surv ~ reason, data = HurricaneDF)
summary(Hurricane_strat)
ggsurvplot(Hurricane_strat, data = HurricaneDF, palette="hue",
           xlab = "Hour", ylab = "Survival Probability", break.y.by = 0.1,
           legend.title = "Reason", legend.labs = c("0", "1", "2", "3", "4"))

#Hazard Probs for all types together 
Hurricane_km$hp <- Hurricane_km$n.event/Hurricane_km$n.risk
Hurricane_haz <- merge(data.frame(time = seq(1,48,1)), data.frame(time = Hurricane_km$time, hp = Hurricane_km$hp), by = "time", all = TRUE)
Hurricane_haz[is.na(Hurricane_haz) == TRUE] <- 0

plot(y = Hurricane_haz$hp, x = Hurricane_haz$time, main = "Hazard Probability Function", xlab = "Tenure", ylab = "Hazard Probability",
     type = 'l')

ggsurvplot(Hurricane_km, data = HurricaneDF, fun = "cumhaz", conf.int = TRUE, palette = "purple",
           xlab = "Hour", ylab = "Cumulative Hazard", legend = "none")

#Hazard Probs by Type
ggsurvplot(Hurricane_strat, data = HurricaneDF, fun = "cumhaz",  palette = "hue",
           xlab = "Hour", ylab = "Cumulative Hazard", legend = "none")

#Pairwise Comparisons 
pairwise_survdiff(Surv(time = hour, event = survive == 0) ~ reason, data = HurricaneDF, rho = 0)

