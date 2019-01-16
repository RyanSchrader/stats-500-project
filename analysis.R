# initial setup
install.packages("glmnet")
install.packages("hydroGOF")
install.packages("Metrics")
install.packages("faraway")
install.packages("leaps")
require(glmnet)
require(hydroGOF)
require(Metrics)
require(faraway)

df = read.csv("scraped_dataset_all_samples.csv")

df$timedelta = as.integer(difftime("2015-11-27", df$date))
df$hour_posted = as.integer((as.integer(substring(as.character(df$time), 1, 2)) + 1*(as.integer(substring(as.character(df$time), 4, 5))>29)) %% 24)
df$month_posted = as.integer(substring(as.character(df$date), 6, 7))  
df$content_has_imgs = df$num_imgs > 1
df$content_has_videos = df$num_videos > 0
df$is_weekend = (df$weekday_is_Sat == 1 | df$weekday_is_Sun == 1)

cols.dont.want <-c("url", "date", "time", "author", "shares_facebook", "shares_google_plus", "shares_linked_in", "shares_pinterest", "shares_stumble_upon", "shares_twitter")
df <- df[, ! names(df) %in% cols.dont.want, drop = F]

round(cor(df[,5:49]),2)
lmod <- lm(shares_total ~ ., data=df)
x <- model.matrix(lmod)[,-1]
e <- eigen(t(x) %*% x)
e$values
sqrt(e$val[1]/e$val)
vif(x)

names_keep <- c("timedelta", "month_posted", "hour_posted", "title_num_words", "content_num_words", "num_hrefs", "num_self_hrefs", "content_has_imgs", "content_has_videos", "num_topics", "data_channel_is_business", "data_channel_is_social.media", "is_weekend", "global_sentiment_polarity", "global_sentiment_subjectivity", "avg_positive_polarity", "avg_negative_polarity", "title_sentiment_polarity", "title_sentiment_subjectivity", "shares_total")

df = df[,(names(df) %in% names_keep)]

### MODEL SELECTION
# model selection: backwards, forwards, stepwise selection
newdata <- na.omit(df)
null = lm(shares_total ~ 1, data=newdata)
full = lm(shares_total ~ ., data=newdata)
backwardselmodel = step(full, data=newdata, direction="backward")
null= lm(shares_total ~ 1, data=df)
full = lm(shares_total ~ ., data=df)
forwardselmodel = step(null, scope=list(lower=null, upper=full), direction="forward")
model_stepwise = step(lmod, df, direction="both")

# model selection: AIC
require(leaps)
b <- regsubsets(shares_total ~ ., df, nvmax=19)
rs <- summary(b)
num_samples = length(df[,1])
AIC <- num_samples*log(rs$rss/num_samples) + (2:20)*2
plot(AIC ~ I(1:19), ylab="AIC", xlab="Number of Predictors")
which.min(AIC) 
  #12
rs$which 
  # resulting model: title_num_words + content_num_words + num_hrefs + num_self_hrefs + num_topics + data_channel_is_business + global_sentiment_subjectivity + avg_positive_polarity + avg_negative_polarity + timedelta + content_has_imgs + content_has_videos

# adjusted R^2
plot(2:20, rs$adjr2, xlab="Number of Parameters", ylab="Adjusted R-squared")
which.max(rs$adjr2)
rs$which[which.max(rs$adjr2),]

# Cp statistic
plot(2:20, rs$cp, xlab="Number of Parameters", ylab="Cp Statistic")
abline(0,1)
### END MODEL SELECTION

### TRANSFORMATIONS
# boxcox
require(MASS)
dfdata <- as.data.frame.matrix(df)
lmodbc <- lm((shares_total+1) ~ timedelta + title_num_words + num_hrefs + num_self_hrefs + content_has_imgs + content_has_videos + num_topics + global_sentiment_subjectivity + avg_positive_polarity + avg_negative_polarity + content_num_words, dfdata)
boxcox(lmodbc,plotit=T,lambda=seq(-0.3,0.3,by=0.1))

#logtrans
logtrans(lmodbc,plotit=TRUE,alpha=seq(-min(df$shares_total+1)+0.001,0,by=0.01))
### END TRANSFORMATIONS

### COLLINEARITY CHECK OF FINAL MODEL
lmod<-lm(log(shares_total+1)~timedelta+title_num_words+content_num_words
         +num_hrefs+num_self_hrefs+conten_has_imgs
         +conten_has_videos+num_topics
         +global_sentiment_subjectivity+avg_positive_polarity
         +avg_negative_polarity+I(num_hrefs*content_num_words),df)
summary(lmod)
x<-model.matrix(lmod)[,-1]
round(cor(x),2)

e<-eigen(t(scale(x))%*%scale(x))
e$val
sqrt(e$val[1]/e$val)

e<-eigen(t(x)%*%x)
e$val
sqrt(e$val[1]/e$val)
vif(x)
###

# FINAL MODEL (merging of all selected by above measures): log(shares_total+1) ~ timedelta + title_num_words + num_hrefs + num_self_hrefs + content_has_imgs + content_has_videos + num_topics + global_sentiment_subjectivity + avg_positive_polarity + avg_negative_polarity + content_num_words + (num_hrefs*content_num_words)

### DIAGNOSTICS
# diagnostics on final model
lmod <- lm(log(shares_total+1) ~ timedelta + title_num_words + num_hrefs + num_self_hrefs + content_has_imgs + content_has_videos + num_topics + global_sentiment_subjectivity + avg_positive_polarity + avg_negative_polarity + content_num_words + (num_hrefs*content_num_words), df)

# diagnostics: examining constancy of variance
plot(fitted(lmod), residuals(lmod), xlab="Fitted", ylab="Residuals", ylim=c(-10,10))
abline(h=0)
plot(fitted(lmod), sqrt(abs(residuals(lmod))), xlab="Fitted", ylab=expression(sqrt(hat(epsilon))), ylim=c(0,250))

# diagnostics: examining normality of model
qqnorm(residuals(lmod), ylab="Residuals", main="")
qqline(residuals(lmod))
qqnorm(rstandard(lmod))
abline(0,1)

# diagnostics: examining correlation of errors
n <- length(residuals(lmod))
plot(tail(residuals(lmod),n-1) ~ head(residuals(lmod),n-1), xlab=expression(hat(epsilon)[i]),ylab=expression(hat(epsilon)[i+1]))
abline(h=0,v=0,col=grey(0.75))

# Influential Points
cook<-cooks.distance(lmod)
halfnorm(cook,3,ylab="Cook's distances")

lmodi<-lm(log(shares_total+1)~timedelta+title_num_words+content_num_words
          +num_hrefs+num_self_hrefs+content_has_imgs
          +content_has_videos+num_topics
          +global_sentiment_subjectivity+avg_positive_polarity
          +avg_negative_polarity+I(num_hrefs*content_num_words),df,subset=(cook<max(cook)))
summary(lmodi)

# Leverage Points
hatv<-hatvalues(lmod)
head(hatv)
sum(hatv)

halfnorm(hatv,ylab="Leverages")

qqnorm(rstandard(lmod))
abline(0,1)

# Outliers and ANOVA
stud<-rstudent(lmod)
stud[which.max(abs(stud))]
qt(.05/(39092*2),39079)
lmodj<-lm(log(shares_total+1)~timedelta+title_num_words+content_num_words
          +num_hrefs+num_self_hrefs+content_has_imgs
          +content_has_videos+num_topics
          +global_sentiment_subjectivity+avg_positive_polarity
          +avg_negative_polarity+I(num_hrefs*content_num_words),df,subset=(abs(stud)<4.843742))
summary(lmodj)
table(abs(stud)>4.843742)
which(abs(stud)>4.843742)

lmodh<-lm(log(shares_total+1)~timedelta+title_num_words+content_num_words
          +num_hrefs+num_self_hrefs+content_has_imgs
          +content_has_videos+num_topics
          +global_sentiment_subjectivity+avg_positive_polarity
          +avg_negative_polarity,df,subset=(abs(stud)<4.843742))
summary(lmodh)
anova(lmodh,lmodj)
### END DIAGNOSTICS

### CROSS-VALIDATION OF MODELS
# example linear model cv code
require(Metrics)

#linear model: 19 predictors (regular)
df_trash = df[,names(df) %in% names_keep]
k <- 10
MSEs = NULL
for (i in 1:k) {
     ind <- seq(i-1, nrow(df_trash), by=10)
     df_training = df_trash[-ind,]
     df_test = df_trash[ind,]
     lmod <- lm(shares_total ~ ., df_training)
     MSEs[i] = rmse(predict(lmod, df_test), df_test$shares_total)
}
avg_test_RMSE = sum(MSEs)/k
avg_test_RMSE

#linear model: 11 predictors (regular)
k <- 10
MSEs = NULL
for (i in 1:k) {
  ind <- seq(i-1, nrow(df), by=10)
  df_training = df[-ind,]
  df_test = df[ind,]
  lmod <- lm(shares_total ~ timedelta + title_num_words + num_hrefs + num_self_hrefs + content_has_imgs + content_has_videos + num_topics + global_sentiment_subjectivity + avg_positive_polarity + avg_negative_polarity + content_num_words, df_training)
  MSEs[i] = rmse(predict(lmod, df_test), df_test$shares_total)
}
avg_test_RMSE = sum(MSEs)/k
avg_test_RMSE

#linear model: 11 predictors (regular, transform)
k <- 10
MSEs = NULL
for (i in 1:k) {
  ind <- seq(i-1, nrow(df), by=10)
  df_training = df[-ind,]
  df_test = df[ind,]
  lmod <- lm(log(shares_total+1) ~ timedelta + title_num_words + num_hrefs + num_self_hrefs + content_has_imgs + content_has_videos + num_topics + global_sentiment_subjectivity + avg_positive_polarity + avg_negative_polarity + content_num_words, df_training)
  MSEs[i] = rmse(predict(lmod, df_test), log(df_test$shares_total+1))
}
avg_test_RMSE = sum(MSEs)/k
avg_test_RMSE

#linear model: 12 predictors (regular, transform, interaction term)
k <- 10
MSEs = NULL
for (i in 1:k) {
  ind <- seq(i-1, nrow(df), by=10)
  df_training = df[-ind,]
  df_test = df[ind,]
  lmod <- lm(log(shares_total+1) ~ timedelta + title_num_words + num_hrefs + num_self_hrefs + content_has_imgs + content_has_videos + num_topics + global_sentiment_subjectivity + avg_positive_polarity + avg_negative_polarity + content_num_words + (num_hrefs*content_num_words), df_training)
  MSEs[i] = rmse(predict(lmod, df_test), log(df_test$shares_total+1))
}
avg_test_RMSE = sum(MSEs)/k
avg_test_RMSE

#linear model: 12 predictors (regular, transform, interaction term, omit biggest outlier)
df_omit_outlier <- df[-c(8883),]
k <- 10
MSEs = NULL
for (i in 1:k) {
  ind <- seq(i-1, nrow(df_omit_outlier), by=10)
  df_training = df_omit_outlier[-ind,]
  df_test = df_omit_outlier[ind,]
  lmod <- lm(log(shares_total+1) ~ timedelta + title_num_words + num_hrefs + num_self_hrefs + content_has_imgs + content_has_videos + num_topics + global_sentiment_subjectivity + avg_positive_polarity + avg_negative_polarity + content_num_words + (num_hrefs*content_num_words), df_training)
  MSEs[i] = rmse(predict(lmod, df_test), log(df_test$shares_total+1))
}
avg_test_RMSE = sum(MSEs)/k
avg_test_RMSE

# ridge regression
require(glmnet)
k <- 10
MSEs = NULL
for (i in 1:k) {
  ind <- seq(i-1, nrow(df), by=10)
  df_training = df[-ind,]
  df_test = df[ind,]
  
  x = model.matrix(log(shares_total+1) ~ timedelta + title_num_words + num_hrefs + num_self_hrefs + content_has_imgs + content_has_videos + num_topics + global_sentiment_subjectivity + avg_positive_polarity + avg_negative_polarity + content_num_words + (num_hrefs*content_num_words), df)[,-15]
  y=log(df$shares_total+1)
  set.seed(1)
  y.test=y[ind]
  grid = 10^seq(10,-2,length=100)
  ridge.mod=glmnet(x[-ind,],y[-ind],alpha=0,lambda=grid,thresh=1e-12)
  set.seed(1)
  cv.out=cv.glmnet(x[-ind,],y[-ind],alpha=0)
  bestlam=cv.out$lambda.min
  ridge.pred=predict(ridge.mod,s=bestlam,newx=x[ind,])
  MSEs[i] = rmse(ridge.pred, y.test)
}
avg_test_RMSE = sum(MSEs)/k
avg_test_RMSE

# lasso regression
require(glmnet)
k <- 10
MSEs = NULL
for (i in 1:k) {
  ind <- seq(i-1, nrow(df), by=10)
  df_training = df[-ind,]
  df_test = df[ind,]
  
  x = model.matrix(log(shares_total+1) ~ timedelta + title_num_words + num_hrefs + num_self_hrefs + content_has_imgs + content_has_videos + num_topics + global_sentiment_subjectivity + avg_positive_polarity + avg_negative_polarity + content_num_words + (num_hrefs*content_num_words), df)[,-15]
  y=log(df$shares_total+1)
  set.seed(1)
  y.test=y[ind]
  grid = 10^seq(10,-2,length=100)
  lasso.mod=glmnet(x[-ind,],y[-ind],alpha=1,lambda=grid)
  set.seed(1)
  cv.out=cv.glmnet(x[-ind,],y[-ind],alpha=1)
  bestlam=cv.out$lambda.min
  lasso.pred=predict(lasso.mod,s=bestlam,newx=x[ind,])
  MSEs[i] = rmse(lasso.pred, y.test)
}
avg_test_RMSE = sum(MSEs)/k
avg_test_RMSE

# robust regression
require(MASS)
k <- 10
MSEs = NULL
for (i in 1:k) {
  ind <- seq(i-1, nrow(df), by=10)
  df_training = df[-ind,]
  df_test = df[ind,]
  rlmod <- rlm(log(shares_total+1) ~ timedelta + title_num_words + num_hrefs + num_self_hrefs + content_has_imgs + content_has_videos + num_topics + global_sentiment_subjectivity + avg_positive_polarity + avg_negative_polarity + content_num_words + (num_hrefs*content_num_words), df_training)
  MSEs[i] = rmse(predict(rlmod, df_test), log(df_test$shares_total+1))
}
avg_test_RMSE = sum(MSEs)/k
avg_test_RMSE

# generalized least squares
require(nlme)
k <- 10
MSEs = NULL
for (i in 1:k) {
  ind <- seq(i-1, nrow(df), by=10)
  df_training = df[-ind,]
  df_test = df[ind,]
  glsmod <- gls(log(shares_total+1) ~ timedelta + title_num_words + num_hrefs + num_self_hrefs + content_has_imgs + content_has_videos + num_topics + global_sentiment_subjectivity + avg_positive_polarity + avg_negative_polarity + content_num_words + (num_hrefs*content_num_words), df_training)
  MSEs[i] = rmse(predict(glsmod, df_test), log(df_test$shares_total+1))
}
avg_test_RMSE = sum(MSEs)/k
avg_test_RMSE

# logistic regression
popularity=cut(df$shares_total, br=c(0,2500,1595000), labels = c("not popular","popular"))
k <- 10
df$popularity <- 1*(popularity=="popular")
accuracies = NULL
for (i in 1:k) {
  ind <- seq(i-1, nrow(df), by=10)
  df_training <- df[-ind,]
  df_test <- df[ind,]
  logisticmodel <- glm(formula = popularity ~ timedelta + title_num_words + num_hrefs + num_self_hrefs + content_has_imgs + content_has_videos + num_topics + global_sentiment_subjectivity + avg_positive_polarity + avg_negative_polarity + content_num_words + (num_hrefs*content_num_words), family = binomial, data=df_training)
  pred = predict(logisticmodel, newdata=df_test, type="response")
  pred.popular = round(pred)
  pred.table = table(pred.popular, df_test$popularity)
  accuracy = (pred.table[1]+pred.table[4])/(pred.table[1]+pred.table[2]+pred.table[3]+pred.table[4])
  accuracies[i] = accuracy
}
avg_test_accuracy_Logistics = sum(accuracies)/k
accuracies
avg_test_accuracy_Logistics

# linear model for a specific market segment
df_all = read.csv("scraped_dataset_all_samples.csv")
df_all$timedelta = as.integer(difftime("2015-11-27", df_all$date))
df_all$hour_posted = as.integer((as.integer(substring(as.character(df_all$time), 1, 2)) + 1*(as.integer(substring(as.character(df_all$time), 4, 5))>29)) %% 24)
df_all$month_posted = as.integer(substring(as.character(df_all$date), 6, 7))  
df_all$content_has_imgs = df_all$num_imgs > 1
df_all$content_has_videos = df_all$num_videos > 0
df_all$is_weekend = (df_all$weekday_is_Sat == 1 | df_all$weekday_is_Sun == 1)
k <- 10
MSEs = NULL
for (i in 1:k) {
  ind <- seq(i-1, nrow(df_all[df_all$data_channel_is_world==1,]), by=10)
  df_training = df_all[df_all$data_channel_is_world==1,][-ind,]
  df_test = df_all[df_all$data_channel_is_world==1,][ind,]
  lmod <- lm(log(shares_total+1) ~ timedelta + title_num_words + num_hrefs + num_self_hrefs + content_has_imgs + content_has_videos + num_topics + global_sentiment_subjectivity + avg_positive_polarity + avg_negative_polarity + content_num_words + (num_hrefs*content_num_words), df_training)
  MSEs[i] = rmse(predict(lmod, df_test), log(df_test$shares_total+1))
}
avg_test_RMSE = sum(MSEs)/k
avg_test_RMSE
### END OF CROSS-VALIDATION OF MODELS
