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
#source("./io.R")

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

# Initialize the data iter
mx.model.init.iter.rnn <- function(X, y, batch.size, is.train) {
  if (is.MXDataIter(X)) return(X)
  shape <- dim(data)
  if (is.null(shape)) {
    num.data <- length(X)
  } else {
    ndim <- length(shape)
    num.data <- shape[[ndim]]
  }
  if (is.null(y)) {
    if (is.train) stop("Need to provide parameter y for training with R arrays.")
    y <- c(1:num.data) * 0
  }
  
  batch.size <- min(num.data, batch.size)
  
  return(mx.io.arrayiter(X, y, batch.size=batch.size, shuffle=is.train))
}

is.MXDataIter <- function(x) {
  inherits(x, "Rcpp_MXNativeDataIter") ||
    inherits(x, "Rcpp_MXArrayDataIter")
}

# check data and translate data into iterator if data is array/matrix
check.data <- function(data, batch.size, is.train) {
  if (!is.null(data) && !is.list(data) && !is.mx.dataiter(data)) {
    stop("The dataset should be either a mx.io.DataIter or a R list")
  }
  if (is.list(data)) {
    if (is.null(data$data) || is.null(data$label)){
      stop("Please provide dataset as list(data=R.array, label=R.array)")
    }
    data <- mx.model.init.iter.rnn(data$data, data$label, batch.size=batch.size, is.train = is.train)
  }
  if (!is.null(data) && !data$iter.next()) {
    data$reset()
    if (!data$iter.next()) stop("Empty input")
  }
  return (data)
}



train.lstnet <- function(model,
                         train.data,
                         valid.data,
                         num.epoch,
                         learning.rate,
                         wd,
                         init.states.name,
                         clip_gradient = TRUE,
                         update.period,
                         optimizer = 'sgd')
{
  m <- model
  seq.len <- m$seq.len
  batch.size <- m$batch.size
  num.rnn.layer <- m$num.rnn.layer
  num.filter <- m$num.filter
  input.size <- m$input.size
  
  opt <- mx.opt.create(optimizer, learning.rate = learning.rate,
                       wd = wd,
                       rescale.grad = (1/batch.size),
                       clip_gradient=clip_gradient)
  #opt <- mx.opt.create(optimizer, rescale.grad=(1/batch.size), ...)
  updater <- mx.opt.get.updater(opt, m$lsetnet.exec$ref.arg.arrays)
  
  for(epoch in 1:num.epoch)
  {
    ##################
    # batch training #
    ##################
    #initialise first hidden states of GRU and LSTM, and c state of LSTM
    init.states <- list()
    for (name in init.states.name) {
      init.states[[name]] <- m$lstnet.exec$ref.arg.arrays[[name]]*0
    }
    mx.exec.update.arg.arrays(m$lstnet.exec, init.states, match.name = TRUE)
    
    # batching start ...
    tic <- Sys.time()
    train.data$reset()
    batch.counter <- 1
    while(train.data$iter.next()){
      # set rnn input
      train.input <- train.data$value()
      mx.exec.update.arg.arrays(m$lstnet.exec, train.input, match.name = TRUE)
      mx.exec.forward(m$lstnet.exec, is.train = TRUE)
      train.output <- as.array(m$lstnet.exec$outputs[['loss_output']])
      
      if(batch.counter == 1){
        train.output.stack <- train.output
      }else{
        train.output.stack <- cbind(train.output.stack, train.output)
      }
      mx.exec.backward(m$lstnet.exec)
      init.states <- list()
      for (name in init.states.name) {
        init.states[[name]] <- m$lstnet.exec$ref.arg.arrays[[name]]*0
      }
      mx.exec.update.arg.arrays(m$lstnet.exec, init.states, match.name=TRUE)
      # update epoch counter
      batch.counter <- batch.counter + 1
      if (batch.counter %% update.period == 0) {
        # the gradient of initial c and inital h should be zero
        init.grad <- list()
        for (name in init.states.name) {
          init.grad[[name]] <- m$lstnet.exec$ref.arg.arrays[[name]]*0
        }
        
        mx.exec.update.grad.arrays(m$lstnet.exec, init.grad, match.name=TRUE)
        #arg.blocks <- updater(m$lstnet.exec$ref.arg.arrays, m$lstnet.exec$ref.grad.arrays)
        #mx.exec.update.arg.arrays(m$lstnet.exec, arg.blocks, skip.null=TRUE)
        
        grad.arrays <- list()
        for (name in names(m$lstnet.exec$ref.grad.arrays)) {
          if (is.param.name(name))
            grad.arrays[[name]] <- m$lstnet.exec$ref.grad.arrays[[name]]*0
        }
        mx.exec.update.grad.arrays(m$lstnet.exec, grad.arrays, match.name=TRUE)
      }
      eval.batch.res <- evaluate(pred=train.output, label=as.array(train.input$label))
      cat(paste0("Epoch [", epoch, "] Batch [", batch.counter, "] : MSE:", eval.batch.res[['MSE']]," \n"))
    }
    train.data$reset()
    toc <- Sys.time()
    eval.epoch.res <- evaluate(pred=train.output.stack, label=train.data$label)
    cat(paste0("Epoch [", epoch, "] Train Time: ", as.numeric(toc - tic, units="secs"), " sec; MSE:", eval.epoch.res[['MSE']]," \n"))
    
    #cat(paste0("Epoch [", epoch, "] Train Time: ", as.numeric(toc - tic, units="secs"), " sec; RAE:", 
    #           eval.epoch.res[['RAE']], "; RSE: ", eval.epoch.res[['RSE']], "; CORR: ", eval.epoch.res[['CORR']]," \n"))
    
    ####################
    # batch validating #
    ####################
    if( !is.null(valid.data)){
      for (name in init.states.name) {
        init.states[[name]] <- m$lstnet.exec$ref.arg.arrays[[name]]*0
      }
      mx.exec.update.arg.arrays(m$lstnet.exec, init.states, match.name=TRUE)
      
      valid.data$reset()
      batch.counter <- 1
      while (eval.data$iter.next()) {
        # set rnn input
        rnn.input <- valid.data$value()
        mx.exec.update.arg.arrays(m$lstnet.exec, rnn.input, match.name=TRUE)
        mx.exec.forward(m$lstnet.exec, is.train=FALSE)
        valid.output <- as.array(m$lstnet.exec$outputs[['loss_output']])
        if(batch.counter == 1){
          valid.output.stack <- valid.output
        }else{
          valid.output.stack <- cbind(valid.output.stack, valid.output)
        }
        
        # transfer the states
        init.states <- list()
        for (name in init.states.name) {
          init.states[[name]] <- m$lstnet.exec$ref.arg.arrays[[name]]*0
        }
        mx.exec.update.arg.arrays(m$lstnet.exec, init.states, match.name=TRUE)
        batch.counter <- batch.counter+1
      }
      valid.data$reset()
      valid.res <- evaluate(pred=valid.output.stack, label=valid.data$label)
      cat(paste0("Epoch [", epoch, "] Validation; MSE:", valid.res[['MSE']]," \n"))
      #cat(paste0("Epoch [", epoch, "] Validation; RAE:", valid.res[['RAE']], 
      #           "; RSE: ", valid.res[['RSE']], "; CORR: ", valid.res[['CORR']]," \n"))
    }    
  }
  
  return(m)
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
                      clip_graident = TRUE,
                      optimizer='sgd',
                      type = 'skip'
                      )
{
  train.data <- check.data(data$train, batch.size, TRUE)
  valid.data <- check.data(data$valid, batch.size, FALSE)
  
  input.size <- dim(data$train$data)[2]
  if(type == 'skip'){
    lstnet.sym <- mx.rnn.lstnet.skip(seasonal.period = seasonal.period,
                                     time.interval = time.interval,
                                     filter.list = filter.list,
                                     num.filter = num.filter,
                                     seq.len = seq.len,
                                     input.size = input.size,
                                     batch.size = batch.size,
                                     num.rnn.layer = num.rnn.layer,
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
    
  }else{
    lstnet.sym <- mx.rnn.lstnet.attn(filter.list = filter.list,
                                     num.filter = num.filter,
                                     seq.len = seq.len,
                                     input.size = input.size,
                                     batch.size = batch.size,
                                     num.rnn.layer = num.rnn.layer,
                                     dropout = dropout)
    
    init.gru.states.h <- lapply(1:num.rnn.layer, function(i) {
      state.h <- paste0("l", i, ".gru.init.h")
      return (state.h)
    })
    init.states.name <- c(init.gru.states.h)
  }
  
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
  #if(FALSE){
  model <- train.lstnet(model,
                        train.data = train.data,
                        valid.data = valid.data,
                        num.epoch = num.epoch,
                        learning.rate = learning.rate,
                        wd = wd,
                        init.states.name = init.states.name,
                        clip_gradient = clip_gradient,
                        update.period =  1,
                        optimizer = optimizer)
  
  # change model into MXFeedForwardModel
  model <- list(symbol=model$symbol, arg.params=model$lstnet.exec$ref.arg.arrays, aux.params=model$lstnet.exec$ref.aux.arrays)
  return(structure(model, class="MXFeedForwardModel"))
  #}
  #return(model)
}