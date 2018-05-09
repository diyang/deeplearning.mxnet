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

mse <- function(label, pred)
{
  mse <- mx.nd.mean(mx.nd.square(label - pred))
  return(as.array(mse))
}

evaluate <- function(pred, label)
{
  pred <- mx.nd.array(pred)
  label <- mx.nd.array(label)

  eva.all <-list()
  eva.all[['RAE']]  <- rae(label = label, pred = pred)
  eva.all[['RSE']]  <- rse(label = label, pred = pred)
  eva.all[['CORR']] <- corr(label = label, pred = pred)
  eva.all[['MSE']] <- mse(label = label, pred = pred)
  return(eva.all)
}

