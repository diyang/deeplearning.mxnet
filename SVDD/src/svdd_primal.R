require(mxnet)
require(MASS)
library("plotrix")

is.param.name <- function(name) { 
  return (grepl('weight$', name) || grepl('bias$', name) ||  
            grepl('gamma$', name) || grepl('beta$', name) || grepl('param$', name)) 
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

update.closure <- function(optimizer, weight, grad, state.list) {
  ulist <- lapply(seq_along(weight), function(i) {
    if (!is.null(grad[[i]])) {
      optimizer$update(i, weight[[i]], grad[[i]], state.list[[i]])
    } else {
      return(NULL)
    }
  })
  # update state list, use mutate assignment
  state.list <<- lapply(ulist, function(x) {
    x$state
  })
  # return updated weight list
  weight.list <- lapply(ulist, function(x) {
    x$weight
  })
  return(weight.list)
}

svdd.model <- function(tau=5, c=1)
{
  data <- mx.symbol.Variable('data')
  r <- mx.symbol.Variable('param.r')
  a <- mx.symbol.Variable('param.a')
  r2 <- mx.symbol.square(r)
  radius <- mx.symbol.sum(mx.symbol.square(mx.symbol.broadcast_minus(data, a)), axis=1)
  phi <- mx.symbol.broadcast_minus(radius, r2)
  s.phi <- mx.symbol.log(mx.symbol.exp(phi*tau)+1)/tau
  sum.s <- mx.symbol.sum(s.phi)
  loss <- r2+sum.s*c
  loss.all <- mx.symbol.MakeLoss(loss, name='loss')
  return(loss.all)
}

svdd.model.setup <- function(ctx = mx.cpu(),
                             symbol,
                             batch.size,
                             input.size,
                             initializer = mx.init.uniform(0.01)){
  input.shape <- list()
  input.shape[['data']] <- c(input.size, batch.size)
  input.shape[['param.a']] <- c(input.size,1)
  input.shape[['param.r']] <- c(1)
  
  params <- mx.model.init.params(symbol = symbol, input.shape = input.shape, initializer = initializer, ctx = ctx)
  
  args <- input.shape
  args$symbol <- svdd.sym
  args$ctx <- ctx
  args$grad.req <- 'add'
  svdd.exec <- do.call(mx.simple.bind, args)
  
  mx.exec.update.arg.arrays(svdd.exec, params$arg.params, match.name = TRUE)
  mx.exec.update.aux.arrays(svdd.exec, params$aux.params, match.name = TRUE)
  
  grad.arrays <- list()
  for (name in names(svdd.exec$ref.grad.arrays)) {
    if (is.param.name(name))
      grad.arrays[[name]] <- svdd.exec$ref.arg.arrays[[name]]*0
  }
  mx.exec.update.grad.arrays(svdd.exec, grad.arrays, match.name=TRUE)
  
  return(list(svdd.exec=svdd.exec, 
              symbol=symbol,
              batch.size=batch.size))
}

svdd.train <- function(train.data,
                       model, 
                       learning.rate = 0.01,
                       epoch = 100,
                       wd = 0.0,
                       optimizer = 'sgd'){
  m <- model
  batch.size <- m$batch.size
  
  opt <- mx.opt.create(optimizer, learning.rate,wd,
                       rescale.grad = (1/batch.size))
  state.list <- lapply(seq_along(m$svdd.exec$ref.arg.arrays), function(i) {
    if (is.null(m$svdd.exec$ref.arg.arrays[[i]])) return(NULL)
    opt$create.state(i, m$svdd.exec$ref.arg.arrays[[i]])
  })
  init.state <- list()
  init.state[['param.r']] <- mx.nd.array(2)
  mx.exec.update.arg.arrays(m$svdd.exec, init.state, match.name = TRUE)
  
  for(epoch in 1:epoch)
  {
    # batching start ...
    train.data$reset()
    batch.counter <- 1
    cat(paste0('Training Epoch ', epoch, '\n'))
    while(train.data$iter.next()){
      # set rnn input
      train.input <- train.data$value()
      train.input$label <- NULL
      mx.exec.update.arg.arrays(m$svdd.exec, train.input, match.name = TRUE)
      mx.exec.forward(m$svdd.exec, is.train = TRUE)
      mx.exec.backward(m$svdd.exec)
      arg.blocks <- update.closure(optimizer = opt, weight = m$svdd.exec$ref.arg.arrays, 
                                   grad = m$svdd.exec$ref.grad.arrays, state.list = state.list)
      mx.exec.update.arg.arrays(m$svdd.exec, arg.blocks, skip.null=TRUE)
      
      # evaluate performance and display
      train.output <- m$svdd.exec$ref.outputs[['loss_output']]
      cat(paste0("Epoch [", epoch, "] Batch [", batch.counter, "] Trianing: LOSS:", as.array(train.output), " \n"))
      batch.counter <- batch.counter +1
    }
    train.data$reset()
  }
  return(m)
}

mu <- rep(0,2)
cov <- matrix(c(1,0,0,1), 2, 2)
x1 <- 0.3*mvrnorm(n = 100, mu, cov)
x2 <- 0.3*mvrnorm(n = 100, mu, cov)
x_train <- rbind(x1+1, x2+1)
x_train <- mx.nd.array(x_train)
x_train <- mx.nd.transpose(x_train, axes=c(0,1))
y <- c(1:200)*0
data <- mx.io.arrayiter(data=x_train, label=y, batch.size = 20, shuffle = TRUE)

svdd.sym <- svdd.model()
model <- svdd.model.setup(symbol = svdd.sym, batch.size = 20, input.size = 2)
model <- svdd.train(train.data = data, model=model, epoch=245, learning.rate = 1e-05)

sample.x <- as.array(x_train)[1,]
sample.y <- as.array(x_train)[2,]
radius <- as.array(model$svdd.exec$ref.arg.arrays$param.r)
centre <- as.array(model$svdd.exec$ref.arg.arrays$param.a)
plot(sample.x, sample.y, type='p', xlim=c(-3,4), ylim=c(-3,4), asp=1)
draw.circle(centre[1],centre[2],radius,nv=1000,border='red',col=NA,lty=1,lwd=1)
lines(centre[1], centre[2], type='p', col='red')