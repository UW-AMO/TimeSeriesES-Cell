df_paper <- read.csv("DataPaper/df_synthetic_paper.csv")

ts_clean = ts(df_paper['ts_outlier'], frequency = 50)

holtwintersR <- HoltWinters(ts_clean)
plot(holtwintersR$fitted)

p <- predict(holtwintersR, 500, prediction.interval = TRUE, level = 0.95)
plot(holtwintersR, p)