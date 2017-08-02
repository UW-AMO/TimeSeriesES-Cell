df_paper <- read.csv("DataPaper/TwitterDownsampledRed.csv")

ts_twitter = ts(df_paper['TwitterDownsampled'], frequency = 131)

holtwintersR <- HoltWinters(ts_twitter)
plot(holtwintersR$fitted)

p <- predict(holtwintersR, 600, prediction.interval = TRUE, level = 0.95)
plot(holtwintersR, p)