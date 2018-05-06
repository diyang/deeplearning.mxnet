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

require('mxnet')

gru.cell <- function(num.hidden, indata, prev.state, param, seqidx, layeridx, dropout=0){
  if (dropout > 0 && layeridx > 1)
    indata <- mx.symbol.Dropout(data=indata, p=dropout)
  
  i2h <- mx.symbol.FullyConnected(data=indata,
                                  weight=param$gates.i2h.weight,
                                  bias=param$gates.i2h.bias,
                                  num.hidden=num.hidden * 2,
                                  name=paste0("t", seqidx, ".l", layeridx, ".gates.i2h"))
  h2h <- mx.symbol.FullyConnected(data=prev.state$h,
                                  weight=param$gates.h2h.weight,
                                  bias=param$gates.h2h.bias,
                                  num.hidden=num.hidden * 2,
                                  name=paste0("t", seqidx, ".l", layeridx, ".gates.h2h"))
  gates <- i2h + h2h
  slice.gates <- mx.symbol.SliceChannel(gates, num.outputs=2,
                                        name=paste0("t", seqidx, ".l", layeridx, ".gru.slice"))
  update.gate <- mx.symbol.Activation(slice.gates[[1]], act.type="sigmoid")
  reset.gate <- mx.symbol.Activation(slice.gates[[2]], act.type="sigmoid")
  
  htrans.i2h <- mx.symbol.FullyConnected(data=indata,
                                         weight=param$trans.i2h.weight,
                                         bias=param$trans.i2h.bias,
                                         num.hidden=num.hidden,
                                         name=paste0("t", seqidx, ".l", layeridx, ".trans.i2h"))
  h.after.reset <- prev.state$h * reset.gate
  htrans.h2h <- mx.symbol.FullyConnected(data=h.after.reset,
                                         weight=param$trans.h2h.weight,
                                         bias=param$trans.h2h.bias,
                                         num.hidden=num.hidden,
                                         name=paste0("t", seqidx, ".l", layeridx, ".trans.h2h"))
  h.trans <- htrans.i2h + htrans.h2h
  h.trans.active <- mx.symbol.Activation(h.trans, act.type="relu")
  next.h <- prev.state$h + update.gate * (h.trans.active - prev.state$h)
  return (list(h=next.h))
}

lstm.cell <- function(num.hidden, indata, prev.state, param, seqidx, layeridx, dropout=0){
  if(dropout > 0 && layeridx > 1)
    indata <- mx.symbol.Dropout(data=indata, p=dropout)
  
  i2h <- mx.symbol.FullyConnected(data=indata,
                                  weight=param$i2h.weight,
                                  bias=param$i2h.bias,
                                  num.hidden=num.hidden * 4,
                                  name=paste0("t", seqidx, ".l", layeridx, ".i2h"))
  
  # make sure if prev state is avaliable
  if(!is.null(prev.state)){
    h2h <- mx.symbol.FullyConnected(data=prev.state$h,
                                    weight=param$h2h.weight,
                                    bias=param$h2h.bias,
                                    num.hidden = num.hidden * 4,
                                    name=paste0("t", seqidx, ".l", layeridx, ".h2h"))
    gates <- i2h+h2h
  }else{
    gates <- i2h
  }
  
  
  slice.gates <- mx.symbol.SliceChannel(gates, num.outputs=4,
                                        name=paste0("t", seqidx, ".l", layeridx, ".lstm.slice"))
  
  in.gate <- mx.symbol.Activation(slice.gates[[1]], act.type='sigmoid')
  in.transform <- mx.symbol.Activation(slice.gates[[2]], act.type="tanh")
  forget.gate <- mx.symbol.Activation(slice.gates[[3]], act.type="sigmoid")
  out.gate <- mx.symbol.Activation(slice.gates[[4]], act.type="sigmoid")
  
  if(!is.null(prev.state)){
    next.c <- (forget.gate * prev.state$c) + (in.gate * in.transform)
  }else{
    next.c <- in.gate * in.transform
  }
  
  next.h <- out.gate * mx.symbol.Activation(next.c, act.type="tanh")
  
  return (list(c=next.c, h=next.h))
}

rnn.skip.unroll<-function(data, 
                     num.rnn.layer=1,
                     seq.len,
                     num.hidden,
                     seasonal.period,
                     dropout=0,
                     config="gru")
{
  param.cells <- list()
  last.states <- list()
  for( i in 1:num.rnn.layer){
    if(config == "gru"){
      param.cells[[i]] <- list(gates.i2h.weight = mx.symbol.Variable(paste0("l", i, ".gates.i2h.weight")),
                               gates.i2h.bias = mx.symbol.Variable(paste0("l", i, ".gates.i2h.bias")),
                               gates.h2h.weight = mx.symbol.Variable(paste0("l", i, ".gates.h2h.weight")),
                               gates.h2h.bias = mx.symbol.Variable(paste0("l", i, ".gates.h2h.bias")),
                               
                               trans.i2h.weight = mx.symbol.Variable(paste0("l", i, ".trans.i2h.weight")),
                               trans.i2h.bias = mx.symbol.Variable(paste0("l", i, ".trans.i2h.bias")),
                               trans.h2h.weight = mx.symbol.Variable(paste0("l", i, ".trans.h2h.weight")),
                               trans.h2h.bias = mx.symbol.Variable(paste0("l", i, ".trans.h2h.bias")))
      state <- list(h=mx.symbol.Variable(paste0("l", i, ".gru.init.h")))
    }else{
      param.cells[[i]] <- list(i2h.weight = mx.symbol.Variable(paste0("l", i, ".i2h.weight")),
                               i2h.bias = mx.symbol.Variable(paste0("l", i, ".i2h.bias")),
                               h2h.weight = mx.symbol.Variable(paste0("l", i, ".h2h.weight")),
                               h2h.bias = mx.symbol.Variable(paste0("l", i, ".h2h.bias")))
      state <- list(c=mx.symbol.Variable(paste0("l", i, ".lstm.init.c")),
                    h=mx.symbol.Variable(paste0("l", i, ".lstm.init.h")))
    }
    last.states[[i]] <- state
  }
  
  data_seq_slice = mx.symbol.SliceChannel(data=data, num_outputs=seq.len, axis=2, squeeze_axis=1)
  
  last.hidden <- list()
  #it's a queue
  seasonal.states <- list()
  for (seqidx in 1:seq.len){
    hidden <- data_seq_slice[[seqidx]]
    # stack lstm
    if(seqidx <= seasonal.period){
      for (i in 1:num.rnn.layer){
        dropout <- ifelse(i==1, 0, dropout)
        prev.state <- last.states[[i]]
        
        if(config == "gru"){
          next.state <- gru.cell(num.hidden,
                                 indata = hidden,
                                 prev.state = prev.state,
                                 param = param.cells[[i]],
                                 seqidx = seqidx,
                                 layeridx = i,
                                 dropout = dropout)
        }else{
          next.state <- lstm.cell(num.hidden,
                                  indata = hidden,
                                  prev.state = prev.state,
                                  param = param.cells[[i]],
                                  seqidx = seqidx,
                                  layeridx = i,
                                  dropout = dropout)
        }
        hidden <- next.state$h
        last.states[[i]] <- next.state
      }
      seasonal.states <- c(seasonal.states, last.states)
    }else{
      for (i in 1:num.rnn.layer){
        dropout <- ifelse(i==1, 0, dropout)
        prev.state <- seasonal.states[[1]]
        seasonal.states <- seasonal.states[-1]
        if(config == "gru"){
          next.state <- gru.cell(num.hidden,
                                 indata = hidden,
                                 prev.state = prev.state,
                                 param = param.cells[[i]],
                                 seqidx = seqidx,
                                 layeridx = i,
                                 dropout = dropout)
        }else{
          next.state <- lstm.cell(num.hidden,
                                  indata = hidden,
                                  prev.state = prev.state,
                                  param = param.cells[[i]],
                                  seqidx = seqidx,
                                  layeridx = i,
                                  dropout = dropout)
        }
        hidden <- next.state$h
        last.states[[i]] <- next.state
      }
      seasonal.states <- c(seasonal.states, last.states)
    }
    
    # Aggeregate outputs from each timestep
    last.hidden <- c(last.hidden, hidden)
  }
  list.all <- list(outputs = last.hidden, last.states = last.states)
  
  return(list.all)
}

rnn.unroll <- function(data, 
                            num.rnn.layer=1,
                            seq.len,
                            num.hidden,
                            dropout=0,
                            config="gru")
{
  param.cells <- list()
  last.states <- list()
  for( i in 1:num.rnn.layer){
    if(config == "gru"){
      param.cells[[i]] <- list(gates.i2h.weight = mx.symbol.Variable(paste0("l", i, ".gates.i2h.weight")),
                               gates.i2h.bias = mx.symbol.Variable(paste0("l", i, ".gates.i2h.bias")),
                               gates.h2h.weight = mx.symbol.Variable(paste0("l", i, ".gates.h2h.weight")),
                               gates.h2h.bias = mx.symbol.Variable(paste0("l", i, ".gates.h2h.bias")),
                               
                               trans.i2h.weight = mx.symbol.Variable(paste0("l", i, ".trans.i2h.weight")),
                               trans.i2h.bias = mx.symbol.Variable(paste0("l", i, ".trans.i2h.bias")),
                               trans.h2h.weight = mx.symbol.Variable(paste0("l", i, ".trans.h2h.weight")),
                               trans.h2h.bias = mx.symbol.Variable(paste0("l", i, ".trans.h2h.bias")))
      state <- list(h=mx.symbol.Variable(paste0("l", i, ".gru.init.h")))
    }else{
      param.cells[[i]] <- list(i2h.weight = mx.symbol.Variable(paste0("l", i, ".i2h.weight")),
                               i2h.bias = mx.symbol.Variable(paste0("l", i, ".i2h.bias")),
                               h2h.weight = mx.symbol.Variable(paste0("l", i, ".h2h.weight")),
                               h2h.bias = mx.symbol.Variable(paste0("l", i, ".h2h.bias")))
      state <- list(c=mx.symbol.Variable(paste0("l", i, ".lstm.init.c")),
                    h=mx.symbol.Variable(paste0("l", i, ".lstm.init.h")))
    }
    last.states[[i]] <- state
  }
  
  data_seq_slice = mx.symbol.SliceChannel(data=data, num_outputs=seq.len, axis=2, squeeze_axis=1)
  
  last.hidden <- list()
  for (seqidx in 1:seq.len){
    hidden <- data_seq_slice[[seqidx]]
    # stack lstm
    for (i in 1:num.rnn.layer){
      dropout <- ifelse(i==1, 0, dropout)
      prev.state <- last.states[[i]]
      
      if(config == "gru"){
        next.state <- gru.cell(num.hidden,
                               indata = hidden,
                               prev.state = prev.state,
                               param = param.cells[[i]],
                               seqidx = seqidx,
                               layeridx = i,
                               dropout = dropout)
      }else{
        next.state <- lstm.cell(num.hidden,
                                indata = hidden,
                                prev.state = prev.state,
                                param = param.cells[[i]],
                                seqidx = seqidx,
                                layeridx = i,
                                dropout = dropout)
      }
      hidden <- next.state$h
      last.states[[i]] <- next.state
    }
    
    # Aggeregate outputs from each timestep
    last.hidden <- c(last.hidden, hidden)
  }
  list.all <- list(outputs = last.hidden, last.states = last.states)
  
  return(list.all)
}

mx.rnn.lstnet.skip <- function(seasonal.period,
                               time.interval,
                               filter.list,
                               num.filter,
                               seq.len,
                               input.size,
                               batch.size,
                               num.rnn.layer = 1,
                               init.update = FALSE,
                               dropout = 0)
{
  data <- mx.symbol.Variable('data')
  label<- mx.symbol.Variable('label')
  
  #reshape data before applying convolutional layer
  conv_input <- mx.symbol.Reshape(data=data, shape=c(seq.len, input.size, 1, batch.size))
  
  #################
  # CNN component #
  #################
  output <- list()
  for(i in 1:length(filter.list)){
    padi <- mx.symbol.pad(data=conv_input, mode='constant', constant_value=0, pad_width=c(0,(filter.list[[i]]-1), 0,0, 0,0, 0,0))
    convi<- mx.symbol.Convolution(data=padi, kernel=c(filter.list[[i]], input.size), num_filter = num.filter)
    acti <- mx.symbol.Activation(data=convi, act_type='relu')
    trans<- mx.symbol.Reshape(mx.symbol.transpose(data=acti, axes=c(0,2,3,1)), shape=c(seq.len, num.filter, batch.size))
    output <- c(output, trans)
  }
  #stack CNN output along Axis-Z
  cnn.features <- mx.symbol.Concat(data=output, num.args = length(filter.list), dim = 1)
  if(dropout > 0){
    cnn.features <- mx.symbol.Dropout(cnn.features, p=dropout)
  }
  
  ################
  #   GRU RNN    #
  ################
  output.gru.bulk  <- rnn.unroll(data=cnn.features, num.rnn.layer=num.rnn.layer, seq.len=seq.len, num.hidden=num.filter*length(filter.list), dropout=dropout)
  output.gru <- output.gru.bulk$outputs
  rnn.features = output.gru[[length(output.gru)]]
  
  ##################
  # LSTM skip RNN  #
  ##################
  p <- ceiling(seasonal.period / time.interval)
  output.lstm.bulk <- rnn.skip.unroll(data=cnn.features, num.rnn.layer=num.rnn.layer, seq.len=seq.len, num.hidden=num.filter*length(filter.list), seasonal.period = p, dropout=dropout, config="lstm")
  output.lstm <- output.lstm.bulk$outputs
  output.lstm <- rev(output.lstm)
  output.last.lstm <- output.lstm[1:p]
  skip.rnn.features <- mx.symbol.Concat(data=output.last.lstm, num.args = p, dim=1)
  
  ##################
  # Autoregression #
  ##################
  time.series.slice <- mx.symbol.SliceChannel(data=data, num_outputs=input.size, axis= 1, squeeze_axis=1)
  auto.list <- list()
  for(i in 1:input.size){
    time.series <- time.series.slice[[i]]
    fc.ts <- mx.symbol.FullyConnected(data=time.series, num_hidden = 1)
    auto.list <- c(auto.list, fc.ts)
  }
  ar.output <- mx.symbol.Concat(data=auto.list, num.args = input.size, dim=1)
  
  ##################
  # Prediction Out #  
  ##################
  neural.componts <- mx.symbol.Concat(data = c(rnn.features, skip.rnn.features), num.args=2, dim = 1)
  neural.output <- mx.symbol.FullyConnected(data=neural.componts, num_hidden=input.size)
  model.output <- neural.output + ar.output
  loss.grad <- mx.symbol.LinearRegressionOutput(data=model.output, label=label, name='loss')
  
  #include last states of GRU and LSTM for updating
  if(init.update){
    gru.last.states  <- output.gru.bulk$last.states
    lstm.last.states <- output.lstm.bulk$last.states
    
    gru.unpack.h <- list()
    lstm.unpack.c <- list()
    lstm.unpack.h <- list()
    for (i in 1:num.rnn.layer) {
      #gru lastest state
      gru.state <- gru.last.states[[i]]
      gru.state  <- list(h=mx.symbol.BlockGrad(gru.state$h, name=paste0("l", i, ".gru.last.h" )))
      gru.unpack.h <- c(gru.unpack.h, gru.state$h)
      
      #lstm latest state
      lstm.state <- lstm.last.states[[i]]
      lstm.state <- list(c=mx.symbol.BlockGrad(lstm.state$c, name=paste0("l", i, ".lstm.last.c")),
                         h=mx.symbol.BlockGrad(lstm.state$h, name=paste0("l", i, ".lstm.last.h")))
      lstm.unpack.c <- c(lstm.unpack.c, lstm.state$c)
      lstm.unpack.h <- c(lstm.unpack.h, lstm.state$h)
    }
    
    #include everything
    list.all <- c(loss.grad, gru.unpack.h, lstm.unpack.c, lstm.unpack.h)
    return(mx.symbol.Group(list.all))
  }else{
    return(loss.grad)
  }
}

mx.rnn.lstnet.attn <- function(filter.list,
                               num.filter,
                               seq.len,
                               input.size,
                               batch.size,
                               num.rnn.layer = 1,
                               init.update = FALSE,
                               dropout = 0)
{
  data <- mx.symbol.Variable('data')
  label<- mx.symbol.Variable('label')
  
  #reshape data before applying convolutional layer
  conv_input <- mx.symbol.Reshape(data=data, shape=c(seq.len, input.size, 1, batch.size))
  
  #################
  # CNN component #
  #################
  output <- list()
  for(i in 1:length(filter.list)){
    padi <- mx.symbol.pad(data=conv_input, mode='constant', constant_value=0, pad_width=c(0,(filter.list[[i]]-1), 0,0, 0,0, 0,0))
    convi<- mx.symbol.Convolution(data=padi, kernel=c(filter.list[[i]], input.size), num_filter = num.filter)
    acti <- mx.symbol.Activation(data=convi, act_type='relu')
    trans<- mx.symbol.Reshape(mx.symbol.transpose(data=acti, axes=c(0,2,3,1)), shape=c(seq.len, num.filter, batch.size))
    output <- c(output, trans)
  }
  #stack CNN output along Axis-Z
  cnn.features <- mx.symbol.Concat(data=output, num.args = length(filter.list), dim = 1)
  if(dropout > 0){
    cnn.features <- mx.symbol.Dropout(cnn.features, p=dropout)
  }
  
  ################
  #   GRU RNN    #
  ################
  output.gru.bulk  <- rnn.unroll(data=cnn.features, num.rnn.layer=num.rnn.layer, seq.len=seq.len, num.hidden=num.filter*length(filter.list), dropout=dropout)
  output.gru <- output.gru.bulk$outputs
  output.gru <- rev(output.gru)
  rnn.features = output.gru[[1]]
  
  ######################
  # Temporal Attention #
  ######################
  attn.numerator.stack <- list()
  for(i in 1:seq.len){
    attn.input <- mx.symbol.concat(data = c(output.gru[[i]], rnn.features), num.args = 2, dim = 1)
    attn.numerator <- mx.symbol.FullyConnected(data=attn.input, num_hidden=1)
    attn.numerator <- mx.symbol.exp(attn.numerator)
    attn.numerator.stack <- c(attn.numerator.stack, attn.numerator)
  }
  attn.numerator.stack <- mx.symbol.Concat(data=attn.numerator.stack, num.args = seq.len, dim = 1)
  attn.denominator <- mx.symbol.sum(data = attn.numerator.stack, axis = 1)
  attn.denominator <- mx.symbol.Reshape(data=attn.denominator, shape=c(1,batch.size))
  attn.denominator.stack <- mx.symbol.broadcast_axis(data=attn.denominator, axis=1, size=seq.len)
  attn.output <- attn.numerator.stack / attn.denominator.stack
  
  c.t.stack <-list()
  attn.output.slice.broad <- list()
  attn.output.slice <- mx.symbol.SliceChannel(data = attn.output, num_outputs = seq.len, axis = 1)
  for(i in 1:seq.len){
    attn.output.slice.broad[[i]] <- mx.symbol.broadcast_axis(data = attn.output.slice[[i]], axis=1, size=num.filter*length(filter.list))
    c.t.stack[[i]]<-output.gru[[i]]*attn.output.slice.broad[[i]]
    c.t.stack[[i]]<-mx.symbol.Reshape(data=c.t.stack[[i]], shape=c(300, 128, 1) )
  }
  aaa <- mx.symbol.concat(c.t.stack, num.args = seq.len, dim = 2)
  c.t <- mx.symbol.zeros_like(rnn.features) 
  
  ##################
  # Autoregression #
  ##################
  time.series.slice <- mx.symbol.SliceChannel(data=data, num_outputs=input.size, axis= 1, squeeze_axis=1)
  auto.list <- list()
  for(i in 1:input.size){
    time.series <- time.series.slice[[i]]
    fc.ts <- mx.symbol.FullyConnected(data=time.series, num_hidden = 1)
    auto.list <- c(auto.list, fc.ts)
  }
  ar.output <- mx.symbol.Concat(data=auto.list, num.args = input.size, dim=1)
  
  ##################
  # Prediction Out #  
  ##################
  neural.componts <- mx.symbol.Concat(data = c(c.t, rnn.features), num.args=2, dim = 1)
  neural.output <- mx.symbol.FullyConnected(data=neural.componts, num_hidden=input.size)
  model.output <- neural.output + ar.output
  loss.grad <- mx.symbol.LinearRegressionOutput(data=model.output, label=label, name='loss')
  #include last states of GRU and LSTM for updating
  
  if(init.update){
    gru.last.states  <- output.gru.bulk$last.states
    
    gru.unpack.h <- list()
    for (i in 1:num.rnn.layer) {
      #gru lastest state
      gru.state <- gru.last.states[[i]]
      gru.state  <- list(h=mx.symbol.BlockGrad(gru.state$h, name=paste0("l", i, ".gru.last.h" )))
      gru.unpack.h <- c(gru.unpack.h, gru.state$h)
    }
    
    #include everything
    list.all <- c(loss.grad, gru.unpack.h)
    return(mx.symbol.Group(list.all))
  }else{
    return(loss.grad)
  }
}