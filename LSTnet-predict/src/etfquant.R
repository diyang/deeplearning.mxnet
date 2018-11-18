require(RMySQL)
require(mxnet)
require(abind)
require(quantmod)
setwd('~/Documents/deeplearning.mxnet/LSTnet-predict/src/')
source("./lstnet_train.R")

data.preparation <- function(data, 
                             seq.len, 
                             horizon = 3, 
                             splits = 0.8, 
                             batch.size = 128,
                             max.records = NULL,
                             normalise = FALSE)
{
  x <- apply(data, 2, as.matrix)
  x.shape <- dim(x)
  if ( !is.null(max.records) && max.records <= x.shape[1]){
    x <- x[1:max.records,]
    x.shape[1] <- max.records
  }
  
  sigmoid.norm <- function(x){
    return(x/sqrt(1+(x**2)))
  }
  #normalisation process
  if(normalise){
    x.norm <- apply(x, 2, sigmoid.norm)
    x <- x.norm
  }
  
  # initialise data tensor container
  x_ts <- array(rep(0,(x.shape[1]-seq.len)*seq.len*x.shape[2]), dim=c(x.shape[1]-seq.len, seq.len, x.shape[2]))
  y_ts <- array(rep(0,(x.shape[1]-seq.len)))
  
  for(n in 1:x.shape[1])
  {
    if(n < seq.len)
      next
    else if((n+horizon) > x.shape[1])
      next
    else{
      y_n <- x[n+horizon,1]
      x_n <- x[(n-seq.len+1):n,]
    }
    x_ts[n-seq.len+1,,] <- x_n
    y_ts[n-seq.len+1] <- y_n
  }
  
  #split into training, validating, and testing data
  x_ts.shape <- dim(x_ts)
  y_ts.shape <- dim(y_ts)
  
  train_samples <- floor(floor(x_ts.shape[1]*splits)/batch.size)*batch.size
  valid_samples <- floor(floor(x_ts.shape[1] - train_samples)/batch.size)*batch.size
  #test_samples <- floor((x_ts.shape[1] - train_samples - valid_samples)/batch.size)*batch.size
  
  #build iterator to feed batches to network
  x.train <- mx.nd.array(x_ts[1:train_samples,,])
  x.valid <- mx.nd.array(x_ts[(train_samples+1):(train_samples+valid_samples),,])
  #x.test  <- mx.nd.array(x_ts[(train_samples+valid_samples+1):(train_samples+valid_samples+test_samples),,])
  
  x.train <- mx.nd.transpose(x.train, axes = c(1, 0, 2))
  x.valid <- mx.nd.transpose(x.valid, axes = c(1, 0, 2))
  #x.test <- mx.nd.transpose(x.test, axes = c(1, 0, 2))
  
  
  y.train <- matrix(y_ts[1:train_samples], 1, train_samples)
  y.valid <- matrix(y_ts[(train_samples+1):(train_samples+valid_samples)], 1, valid_samples)
  #y.test  <- matrix(y_ts[(train_samples+valid_samples+1):(train_samples+valid_samples+test_samples)], 1, test_samples)
  
  #y.train <- mx.nd.transpose(y.train, axes = c(0, 1))
  #y.valid <- mx.nd.transpose(y.valid, axes = c(0, 1))
  #y.test <- mx.nd.transpose(y.test, axes = c(0, 1))
  
  
  train <- list(data=as.array(x.train), label=y.train)
  valid <- list(data=as.array(x.valid), label=y.valid)
  #test  <- list(data=as.array(x.test),  label=y.test)
  if(dim(valid$data)[3] < batch.size){
    valid <- NULL
  }
  train_data <- list(train=train, valid=valid)#, test=test)
  
  return (train_data)
}

add.asx <- function(code){
  return(paste(code, 'AX', sep = '.'))
}

seq.len <- 120
horizon <- 1
max.records <- NULL
splits <- 0.7
batch.size<-256
seasonal.period <- 24
time.interval <- 1
filter.list <- c(5, 15, 30)
num.filter <- 100
dropout <-0.0
num.rnn.layer <- 1
learning.rate <- 0.5
lr_scheduler <- mx.lr_scheduler.FactorScheduler(step = 50, factor=0.5, stop_factor_lr = 1e-08)
wd <- 0.000
clip_gradient<-5
ctx <- mx.cpu()
initializer <- mx.init.uniform(0.01)
optimizer <- 'sgd'
output.size <- 1

mydb = dbConnect(MySQL(), user='root', password='bvus93Gz6e', dbname='mydb', host='localhost')
rs = dbSendQuery(mydb, "SELECT * FROM mydb.ASXETFINDEX where benchmark like '%Nasdaq%';")
content = fetch(rs, n=-1)
content$code <- sapply(content$code, add.asx)
etf.codes <- as.array(content$code)
getSymbols(etf.codes)
etf.list <- list()
train <- list()
valid <- list()
counter <- 1
for(code in etf.codes){
  ETF <- get(code)
  ETF <- na.omit(ETF)
  ETF.Cl <- Cl(ETF)
  ETF.Cl.BB <- BBands(ETF.Cl)
  ETF.Cl.MACD <- MACD(ETF.Cl)
  indices <- !is.na(ETF.Cl.MACD$signal)
  
  ETF.Cl.diff <- ETF.Cl[indices,] - 2*lag(ETF.Cl[indices,], 1) + lag(ETF.Cl[indices,], 2)
  ETF.dn.diff <- ETF.Cl[indices,] - ETF.Cl.BB$dn[indices,]
  ETF.up.diff <- ETF.Cl.BB$up[indices,] - ETF.Cl[indices,]
  ETF.ma.diff <- ETF.Cl[indices,] - ETF.Cl.BB$mavg[indices,]
  ETF.mcad.diff <-ETF.Cl.MACD$signal[indices,] - ETF.Cl.MACD$macd[indices,]
  
  days <- dim(ETF.Cl.diff)[1]
  etf.comprehensive <- as.data.frame(merge(ETF.Cl.diff[3:days,], ETF.dn.diff[3:days,], ETF.up.diff[3:days,], ETF.ma.diff[3:days,], ETF.mcad.diff[3:days,]))
  etf.list[[code]] <- ETF
  
  if(dim(etf.comprehensive)[1] < seq.len){
    next()
  }
  
  iter.data <- data.preparation(data = etf.comprehensive, 
                                seq.len = seq.len, 
                                horizon = horizon, 
                                splits = splits, 
                                batch.size = batch.size)
  
  if(counter == 1){
    train[["data"]] <- iter.data$train$data
    train[["label"]]<- iter.data$train$label
    
    valid[["data"]] <- iter.data$valid$data
    valid[["label"]]<- iter.data$valid$label
  }else{
    train[["data"]] <- abind(train[["data"]], iter.data$train$data, along = 3)
    train[["label"]]<- cbind(train[["label"]], iter.data$train$label)
    
    valid[["data"]] <- abind(valid[["data"]], iter.data$valid$data, along = 3)
    valid[["label"]]<- cbind(valid[["label"]], iter.data$valid$label)
  }
  counter <- counter+1
}

iter.batch <- list(train=train, valid=valid)

lstnet.model <- mx.lstnet(data = iter.batch,
                          seq.len = seq.len,
                          batch.size = batch.size,
                          ctx = mx.ctx.default(),
                          initializer = mx.init.uniform(0.01),
                          seasonal.period = seasonal.period,
                          time.interval = time.interval,
                          filter.list = filter.list,
                          num.filter = num.filter,
                          num.epoch = 5000,
                          output.size = 1,
                          learning.rate = learning.rate,
                          wd=wd,
                          clip_graident = clip_graident,
                          optimizer = optimizer,
                          dropout = dropout,
                          num.rnn.layer = num.rnn.layer,
                          lr_scheduler = lr_scheduler,
                          type='attn')