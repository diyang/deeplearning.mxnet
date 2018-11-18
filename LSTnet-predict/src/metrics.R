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

specify_decimal <- function(x, k) trimws(format(round(x, k), nsmall=k))

rse <- function(label, pred)
{
  label.vec <- mx.nd.reshape(label, shape=c(1,-1))
  pred.vec  <- mx.nd.reshape(pred,  shape=c(1,-1))
  pred.mean <- mx.nd.mean(pred.vec)
  
  numerator   <- mx.nd.sqrt(mx.nd.sum(mx.nd.square(label.vec - pred.vec)))
  denominator <- mx.nd.sqrt(mx.nd.sum(mx.nd.square((mx.nd.broadcast.minus(pred.vec, pred.mean)))))
  return(as.array(numerator/denominator))
}

rae <- function(label, pred)
{
  label.vec <- mx.nd.reshape(label, shape=c(1,-1))
  pred.vec  <- mx.nd.reshape(pred,  shape=c(1,-1))
  pred.mean <- mx.nd.mean(pred.vec)
  
  numerator <- mx.nd.sum(mx.nd.abs(label.vec - pred.vec))
  denominator <- mx.nd.sum(mx.nd.abs(mx.nd.broadcast.minus(pred.vec, pred.mean)))

  return(as.array(numerator/denominator))
}

corr <- function(label, pred)
{
  label.shape <- dim(label)
  label.mean <- mx.nd.mean(label, axis=0)
  label.mean <- mx.nd.reshape(label.mean, shape=c(label.shape[1], 1))
  
  pred.shape <- dim(pred)
  pred.mean <- mx.nd.mean(pred, axis=0)
  pred.mean <- mx.nd.reshape(pred.mean, shape=c(label.shape[1], 1))
  
  numerator1 <- mx.nd.broadcast.minus(label, label.mean)
  numerator2 <- mx.nd.broadcast.minus(pred,  pred.mean)
  numerator <- mx.nd.sum((numerator1 * numerator2), axis=0)
  denominator <- mx.nd.sqrt(mx.nd.sum(mx.nd.square(numerator1), axis=0) * mx.nd.sum(mx.nd.square(numerator2), axis=0))
  denominator <- denominator + (mx.nd.ones.like(denominator)*0.001)
  
  corr <- mx.nd.mean(numerator/denominator)
  
  return(as.array(corr))
}

rmse <- function(label, pred)
{
  rmse <- mx.nd.sqrt(mx.nd.mean(mx.nd.square(label - pred)))
  return(as.array(rmse))
}


mx.lstnet.evaluation <- function(){
  init <- function(){
    state <- list()
    state[['RAE']] <- 0
    state[['RSE']] <- 0
    state[['CORR']] <- 0
    state[['RMSE']] <- 0
    state[['batch']] <- 0
    return(state)
  }
  
  update <- function(pred, label, state)
  {
    if(!is.mx.ndarray(pred)){
      pred <- mx.nd.array(pred)
    }
    
    if(!is.mx.ndarray(label)){
      label <- mx.nd.array(label)
    }
    
    state[['RAE']]  <- state[['RAE']] + rae(label = label, pred = pred)
    state[['RSE']]  <- state[['RSE']] + rse(label = label, pred = pred)
    state[['CORR']] <- state[['CORR']] +corr(label = label, pred = pred)
    state[['RMSE']]  <-state[['RMSE']] + rmse(label = label, pred = pred)
    state[['batch']] <- state[['batch']] + 1
    return(state)
  }
  
  get <- function(state){
    res <-list()
    res[['RAE']] <- specify_decimal(state[['RAE']]/state[['batch']], 6)
    res[['RSE']] <- specify_decimal(state[['RSE']]/state[['batch']], 6)
    res[['CORR']]<- specify_decimal(state[['CORR']]/state[['batch']], 6)
    res[['RMSE']]<- specify_decimal(state[['RMSE']]/state[['batch']], 6)
    return(res)
  }
  ret <- list(init=init, update=update, get=get)
  return(ret)
}