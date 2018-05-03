# Author: Di YANG
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

require(mxnet)
source("./lstnet_train.R")

data.preparation <- function(data, 
                             seq.len, 
                             horizon = 3, 
                             splits = c(0.6, 0.2), 
                             batch_size = 128,
                             max.records = NULL)
{
  x <- as.matrix(sapply(data, as.numeric))
  x.shape <- dim(x)
  if ( !is.null(max.records) && max.records <= x.shape[1]){
    x <- x[1:max.records,]
    x.shape[1] <- max.records
  }
  
  # initialise data tensor container
  x_ts <- array(rep(0,(x.shape[1]-seq.len)*seq.len*x.shape[2]), dim=c(x.shape[1]-seq.len, seq.len, x.shape[2]))
  y_ts <- array(rep(0,(x.shape[1]-seq.len)*x.shape[2]), dim=c(x.shape[1]-seq.len, x.shape[2]))
  
  for(n in 1:x.shape[1])
  {
    if(n < seq.len)
      next
    else if((n+horizon) > x.shape[1])
      next
    else{
      y_n <- x[n+horizon,]
      x_n <- x[(n-seq.len+1):n,]
    }
    x_ts[n-seq.len+1,,] <- x_n
    y_ts[n-seq.len+1,] <- y_n
  }
  
  #split into training, validating, and testing data
  x_ts.shape <- dim(x_ts)
  y_ts.shape <- dim(y_ts)
  
  train_samples <- floor(floor(x_ts.shape[1]*splits[1])/batch_size)*batch_size
  valid_samples <- floor(floor(x_ts.shape[1]*splits[2])/batch_size)*batch_size
  test_samples <- floor((x_ts.shape[1] - train_samples - valid_samples)/batch_size)*batch_size
  
  #build iterator to feed batches to network
  x.train <- mx.nd.array(x_ts[1:train_samples,,])
  x.valid <- mx.nd.array(x_ts[(train_samples+1):(train_samples+valid_samples),,])
  x.test  <- mx.nd.array(x_ts[(train_samples+valid_samples+1):(train_samples+valid_samples+test_samples),,])
  
  x.train <- mx.nd.transpose(x.train, axes = c(1, 0, 2))
  x.valid <- mx.nd.transpose(x.valid, axes = c(1, 0, 2))
  x.test <- mx.nd.transpose(x.test, axes = c(1, 0, 2))
  
  
  y.train <- mx.nd.array(y_ts[1:train_samples,])
  y.valid <- mx.nd.array(y_ts[(train_samples+1):(train_samples+valid_samples),])
  y.test  <- mx.nd.array(y_ts[(train_samples+valid_samples+1):(train_samples+valid_samples+test_samples),])
  
  y.train <- mx.nd.transpose(y.train, axes = c(0, 1))
  y.valid <- mx.nd.transpose(y.valid, axes = c(0, 1))
  y.test <- mx.nd.transpose(y.test, axes = c(0, 1))
  
  
  train <- list(data=as.array(x.train), label=as.array(y.train))
  valid <- list(data=as.array(x.valid), label=as.array(y.valid))
  test  <- list(data=as.array(x.test),  label=as.array(y.test))
  
  train_data <- list(train=train, valid=valid, test=test)
  
  return (train_data)
}


seq.len <- 24*7
horizon <- 3
max.records <- NULL
splits <- c(0.6,0.2)
batch.size<-128
seasonal.period <- 24
time.interval <- 1
filter.list <- c(6, 12, 18)
num.filter <- 100
dropout <-0.2
num.rnn.layer <- 1
learning.rate <- 0.01
wd <- 0.00001
clip_gradient<-TRUE
optimiser <- 'sgd'

data <- read.csv('../data/electricity.txt', header=FALSE, sep=",")

iter.data <- data.preparation(data, 
                              seq.len = seq.len, 
                              horizon = horizon, 
                              splits = splits, 
                              batch_size = batch.size,
                              max.records = max.records)

lstnet.sym <- mx.lstnet(data = iter.data,
                        seq.len = seq.len,
                        batch.size = batch.size,
                        ctx = mx.ctx.default(),
                        initializer = mx.init.uniform(0.01),
                        seasonal.period = seasonal.period,
                        time.interval = time.interval,
                        filter.list = filter.list,
                        num.filter = num.filter,
                        num.epoch = 10,
                        learning.rate = learning.rate,
                        wd=wd,
                        clip_graident = clip_graident,
                        optimizer = optimiser,
                        dropout = dropout,
                        num.rnn.layer = num.rnn.layer)