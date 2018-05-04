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
source("./lstnet_model.R")
source("./metrics.R")

lstnet.input <- function(data, batch.begin, batch.size)
{
  data.batch <- list()
  data.batch[["data"]] <- mx.nd.array(data$data[,,batch.begin:(batch.begin+batch.size-1)])
  data.batch[["label"]] <- mx.nd.array(data$label[,batch.begin:(batch.begin+batch.size-1)])
  return(data.batch)
}

is.param.name <- function(name) { 
  return (grepl('weight$', name) || grepl('bias$', name) ||  
            grepl('gamma$', name) || grepl('beta$', name) ) 
} 

mx.model.init.params <- function(symbol, input.shape, initializer, ctx) { 
  if (!is.mx.symbol(symbol)) 
    stop("symbol need to be MXSymbol") 
  
  slist <- symbol$infer.shape(input.shape) 
  
  if (is.null(slist)) 
    stop("Not enough information to get shapes") 
  arg.params <- mx.init.create(initializer, slist$arg.shapes, ctx, skip.unknown=TRUE) 
  aux.params <- mx.init.create(initializer, slist$aux.shapes, ctx, skip.unknown=FALSE) 
  return(list(arg.params=arg.params, aux.params=aux.params)) 
}

train.lstnet <- function(model,
                         train.data,
                         valid.data,
                         num.epoch,
                         learning.rate,
                         wd,
                         clip_gradient = TRUE,
                         init.update = FALSE,
                         optimiser = 'sgd')
{
  train.batch.shape <- dim(train.data$data)
  valid.batch.shape <- dim(valid.data$data)
  cat(paste0("Training with train.shape=(", paste0(train.batch.shape, collapse=","), ")"), "\n")
  cat(paste0("Validating with valid.shape=(", paste0(valid.batch.shape, collapse=","), ")"), "\n")
  
  m <- model
  seq.len <- m$seq.len
  batch.size <- m$batch.size
  num.rnn.layer <- m$num.rnn.layer
  num.filter <- m$num.filter
  input.size <- m$input.size
  
  opt <- mx.opt.create(optimiser, learning.rate = learning.rate,
                       wd = wd,
                       rescale.grad = (1/batch.size),
                       clip_gradient=clip_gradient)
  updater <- mx.opt.get.updater(opt, m$lsetnet.exec$ref.arg.arrays)
  
  for(epoch in 1:num.epoch)
  {
    ##################
    # batch training #
    ##################
    
    #initialise first hidden states of GRU and LSTM, and c state of LSTM
    init.states <- list()
    for(i in 1:num.rnn.layer){
      init.states[[paste0("l", i, ".gru.init.h")]] <- mx.nd.zeros(c(num.filter*length(filter.list), batch.size))
      init.states[[paste0("l", i, ".lstm.init.c")]] <- mx.nd.zeros(c(num.filter*length(filter.list), batch.size))
      init.states[[paste0("l", i, ".lstm.init.h")]] <- mx.nd.zeros(c(num.filter*length(filter.list), batch.size))
    }
    mx.exec.update.arg.arrays(m$lstnet.exec, init.states, match.name = TRUE)
    
    # batching start ...
    tic <- Sys.time()
    batch.counter <- 1
    for(begin in seq(1, train.batch.shape[[3]], batch.size)){
      train.input <- lstnet.input(train.data, begin, batch.size)
      mx.exec.update.arg.arrays(m$lstnet.exec, train.input, match.name = TRUE)
      mx.exec.forward(m$lstnet.exec, is.train = TRUE)
      
      # getting output of network for training evaluation purpose
      train.output <- as.array(m$lstnet.exec$outputs[['loss_output']])
      
      eval.batch.res <- evaluate(pred=train.output, label=as.array(train.input$label))
      cat(paste0("Epoch [", epoch, "] Batch [", batch.counter, "] : RAE:", 
                 eval.batch.res[['RAE']], "; RSE: ", eval.batch.res[['RSE']], "; CORR: ", eval.batch.res[['CORR']]," \n"))
      
      if(begin == 1){
        train.output.stack <- train.output
      }else{
        train.output.stack <- cbind(train.output.stack, train.output)
      }
      mx.exec.backward(m$lstnet.exec)
        
      #update GRU and LSTM states
      if(init.update){
        init.states <- list()
        for (i in 1:num.rnn.layer) {
          init.states[[paste0("l", i, ".gru.init.c")]] <- m$rnn.exec$outputs[[paste0("l", i, ".gru.last.c_output")]]
          init.states[[paste0("l", i, ".lstm.init.c")]] <- m$rnn.exec$outputs[[paste0("l", i, ".lstm.last.c_output")]]
          init.states[[paste0("l", i, ".lstm.init.h")]] <- m$rnn.exec$outputs[[paste0("l", i, ".lstm.last.h_output")]]
        }
        mx.exec.update.arg.arrays(m$lstnet.exec, init.states, match.name=TRUE)
      }
      batch.counter <- batch.counter + 1
    }
    toc <- Sys.time()
    
    eval.epoch.res <- evaluate(pred=train.output.stack, label=train.data$label)
    cat(paste0("Epoch [", epoch, "] Train: Time: ", as.numeric(toc - tic, units="secs"), " sec; RAE:", 
               eval.epoch.res[['RAE']], "; RSE: ", eval.epoch.res[['RSE']], "; CORR: ", eval.epoch.res[['CORR']]," \n"))
    
    ####################
    # batch validating #
    ####################
    init.states <- list()
    for(i in 1:num.rnn.layer){
      init.states[[paste0("l", i, ".gru.init.h")]] <- mx.nd.zeros(c(num.filter*length(filter.list), batch.size))
      init.states[[paste0("l", i, ".lstm.init.c")]] <- mx.nd.zeros(c(num.filter*length(filter.list), batch.size))
      init.states[[paste0("l", i, ".lstm.init.h")]] <- mx.nd.zeros(c(num.filter*length(filter.list), batch.size))
    }
    mx.exec.update.arg.arrays(m$lstnet.exec, init.states, match.name = TRUE)

    for(begin in seq(1, valid.batch.shape[[3]], batch.size)){
      valid.input <- lstnet.input(valid.data, begin, batch.size)
      mx.exec.update.arg.arrays(m$lstnet.exec, valid.input, match.name = TRUE)
      mx.exec.forward(m$lstnet.exec, is.train = FALSE)
      
      # getting output of network for validating evaluation purpose
      valid.output <- as.array(m$lstnet.exec$outputs[['loss_output']])
      if(begin == 1){
        valid.output.stack <- valid.output
      }else{
        valid.output.stack <- cbind(valid.output.stack, valid.output)
      }
      
      #update GRU and LSTM states 
      if(init.update){
        init.states <- list()
        for (i in 1:num.rnn.layer) {
          init.states[[paste0("l", i, ".gru.init.c")]] <- m$rnn.exec$outputs[[paste0("l", i, ".gru.last.c_output")]]
          init.states[[paste0("l", i, ".lstm.init.c")]] <- m$rnn.exec$outputs[[paste0("l", i, ".lstm.last.c_output")]]
          init.states[[paste0("l", i, ".lstm.init.h")]] <- m$rnn.exec$outputs[[paste0("l", i, ".lstm.last.h_output")]]
        }
        mx.exec.update.arg.arrays(m$lstnet.exec, init.states, match.name=TRUE)
      }
    }
    
    valid.res <- evaluate(pred=valid.output.stack, label=valid.data$label)
    cat(paste0("Epoch [", epoch, "] Validation; RAE:", valid.res[['RAE']], 
               "; RSE: ", valid.res[['RSE']], "; CORR: ", valid.res[['CORR']]," \n"))
    
  }
}

lstnet.setup.model <- function(lstnet.sym, 
                               ctx = mx.ctx.default(), 
                               num.rnn.layer = 1,
                               num.filter,
                               filter.list,
                               seq.len,
                               batch.size,
                               input.size,
                               init.states.name,
                               initializer=mx.init.uniform(0.01),
                               dropout = 0)
{
  arg.names <- lstnet.sym$arguments
  input.shape <- list()
  for (name in arg.names) {
    if (name %in% init.states.name) {
      input.shape[[name]] <- c(num.filter*length(filter.list), batch.size)
    }
    if (grepl('data$', name) ) {
      input.shape[[name]] <- c(seq.len, input.size, batch.size)
    }
    else if (grepl('label$', name) ){
      input.shape[[name]] <- c(input.size, batch.size)
    }
  }
  
  params <- mx.model.init.params(symbol = lstnet.sym, input.shape = input.shape, initializer = initializer, ctx = ctx)
 
  args <- input.shape
  args$symbol <- lstnet.sym
  args$ctx <- ctx
  args$grad.req <- 'add'
  lstnet.exec <- do.call(mx.simple.bind, args)
  
  mx.exec.update.arg.arrays(lstnet.exec, params$arg.params, match.name = TRUE)
  mx.exec.update.aux.arrays(lstnet.exec, params$aux.params, match.name = TRUE)
  
  grad.arrays <- list()
  for (name in names(lstnet.exec$ref.grad.arrays)) {
    if (is.param.name(name))
      grad.arrays[[name]] <- lstnet.exec$ref.arg.arrays[[name]]*0
  }
  mx.exec.update.grad.arrays(lstnet.exec, grad.arrays, match.name=TRUE)
  
  return (list(lstnet.exec=lstnet.exec, 
               symbol=lstnet.sym,
               num.rnn.layer=num.rnn.layer, 
               num.filter=num.filter,
               filter.list = filter.list,
               seq.len=seq.len, 
               batch.size=batch.size,
               input.size=input.size))
}

mx.lstnet <- function(#input data:
                      data,
                      #dataframe strct. params:
                      seq.len,
                      batch.size,
                      #setup params:
                      ctx = mx.ctx.default(),
                      initializer = mx.init.uniform(0.01),
                      #network struct. params:
                      seasonal.period,
                      time.interval = 1,
                      filter.list,
                      num.filter,
                      num.rnn.layer = 1,
                      dropout = 0,
                      #training params:
                      num.epoch,
                      learning.rate = 0.1,
                      wd,
                      init.update = FALSE,
                      clip_graident = TRUE,
                      optimizer='sgd'
                      )
{
  input.size <- dim(data$train$data)[2]
  
  lstnet.sym <- mx.rnn.lstnet(seasonal.period = seasonal.period,
                              time.interval = time.interval,
                              filter.list = filter.list,
                              num.filter = num.filter,
                              seq.len = seq.len,
                              input.size = input.size,
                              batch.size = batch.size,
                              num.rnn.layer = num.rnn.layer,
                              init.update = init.update,
                              dropout = dropout)
  
  init.lstm.states.c <- lapply(1:num.rnn.layer, function(i) {
    state.c <- paste0("l", i, ".lstm.init.c")
    return (state.c)
  })
  init.lstm.states.h <- lapply(1:num.rnn.layer, function(i) {
    state.h <- paste0("l", i, ".lstm.init.h")
    return (state.h)
  })
  init.gru.states.h <- lapply(1:num.rnn.layer, function(i) {
    state.h <- paste0("l", i, ".gru.init.h")
    return (state.h)
  })
  init.states.name <- c(init.lstm.states.c, init.lstm.states.h, init.gru.states.h)
  
  model <- lstnet.setup.model(lstnet.sym = lstnet.sym,
                              ctx = ctx,
                              num.rnn.layer = num.rnn.layer,
                              num.filter = num.filter,
                              filter.list = filter.list,
                              seq.len = seq.len,
                              batch.size = batch.size,
                              input.size = input.size,
                              init.states.name = init.states.name,
                              initializer = initializer,
                              dropout = dropout)
  
  model <- train.lstnet(model,
                        train.data = data$train,
                        valid.data = data$valid,
                        num.epoch = num.epoch,
                        learning.rate = learning.rate,
                        wd = wd,
                        clip_gradient = clip_gradient,
                        init.update = init.update,
                        optimiser = optimiser)
  
  # change model into MXFeedForwardModel
  model <- list(symbol=model$symbol, arg.params=model$lstnet.exec$ref.arg.arrays, aux.params=model$lstnet.exec$ref.aux.arrays)
  return(structure(model, class="MXFeedForwardModel"))
  
}
