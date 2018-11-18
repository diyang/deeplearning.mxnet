require(quantmod)
require(mxnet)
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
  x <- as.matrix(sapply(data, as.numeric))
  x.shape <- dim(x)
  if ( !is.null(max.records) && max.records <= x.shape[1]){
    x <- x[1:max.records,]
    x.shape[1] <- max.records
  }
  
  #normalisation process
  if(normalise){
    x.norm <- mx.nd.array(x)
    x.norm.close <- mx.nd.array(x[,1:4])
    x.norm.cl.max <- as.array(mx.nd.max(x.norm.close))
    x.norm.cl.min <- as.array(mx.nd.min(x.norm.close))
    
    x.norm.macd <- mx.nd.array(x[,5:6])
    x.norm.macd.max <- as.array(mx.nd.max(x.norm.macd))
    x.norm.macd.min <- as.array(mx.nd.min(x.norm.macd))
    
    x.norm.min <- matrix(c(x.norm.cl.min,x.norm.cl.min,x.norm.cl.min,x.norm.cl.min,x.norm.macd.min,x.norm.macd.min), 1, 6)
    x.norm.max <- matrix(c(x.norm.cl.max,x.norm.cl.max,x.norm.cl.max,x.norm.cl.max,x.norm.macd.max,x.norm.macd.max), 1, 6)
    
    x.norm.max <- mx.nd.array(x.norm.max)
    x.norm.min <- mx.nd.array(x.norm.min)
    
    x.numerator <- mx.nd.broadcast.minus(x.norm, x.norm.min)
    x.denominator <- x.norm.max - x.norm.min
    x.norm <- mx.nd.broadcast.div(x.numerator, x.denominator)
    x.norm <- (x.norm - 0.5)*2
    x <- as.array(x.norm)
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

getSymbols(c('NDQ.AX'))
NDQ.AX <- na.omit(NDQ.AX)
NDQ.AX.Cl <- Cl(NDQ.AX)
NDQ.AX.BB <- BBands(NDQ.AX.Cl)
NDQ.AX.MACD <- MACD(NDQ.AX.Cl)
indices <- !is.na(NDQ.AX.MACD$signal)

#1st order difference
NDQ.AX.Cl.clean <- NDQ.AX.Cl[indices,]
days <- dim(NDQ.AX.Cl.clean)[1]

NDQ.AX.Cl.diff <- NDQ.AX.Cl.clean - 2*lag(NDQ.AX.Cl.clean, 1) + lag(NDQ.AX.Cl.clean, 2)
NDQ.AX.Cl_bn.clean <- NDQ.AX.Cl[indices,] - NDQ.AX.BB$dn[indices,]
NDQ.AX.Cl_up.clean <- NDQ.AX.Cl[indices,] - NDQ.AX.BB$up[indices,]
NDQ.AX.Cl_ma.clean <- NDQ.AX.Cl[indices,] - NDQ.AX.BB$mavg[indices,]
NDQ.AX.sig_mcad.clean <- NDQ.AX.MACD[indices,]$signal - NDQ.AX.MACD[indices,]$macd
NDQ.AX.comprehensive <- merge(NDQ.AX.Cl.diff[3:days,], NDQ.AX.Cl_bn.clean[3:days,], NDQ.AX.Cl_ma.clean[3:days,], NDQ.AX.Cl_up.clean[3:days,], NDQ.AX.sig_mcad.clean[3:days,])
NDQ.AX.matrix <- apply(NDQ.AX.comprehensive, 2, as.matrix)

seq.len <- 260
horizon <- 1
max.records <- NULL
splits <- 1.0
batch.size<-32
seasonal.period <- 24
time.interval <- 1
filter.list <- c(5, 15, 30)
num.filter <- 20
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

iter.data <- data.preparation(data = NDQ.AX.comprehensive, 
                              seq.len = seq.len, 
                              horizon = horizon, 
                              splits = splits, 
                              batch.size = batch.size)

lstnet.model <- mx.lstnet(data = iter.data,
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