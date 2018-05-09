require(mxnet)
source("./lstnet_model.R")
source("./lstnet_train.R")

num.rnn.layer <- 1
seq.len <- 24*7
num.filter <- 100
filter.list <- c(6, 12, 18)
dropout <- 0
batch.size <- 128
input.size <- 321

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
rnn.features = output.gru[[seq.len]]

######################
# Temporal Attention #
######################
attn.numerator.stack <- list()
for(i in 1:seq.len){
  if(i == seq.len){
    attn.input <- mx.symbol.Reshape(data = rnn.features, shape=c((num.filter*length(filter.list)), batch.size, 1))
    attn.input <- mx.symbol.broadcast_axis(data=attn.input, axis=0, size=2)
    attn.input <- mx.symbol.Reshape(data = attn.input, shape=c(batch.size, (2*num.filter*length(filter.list))))
    attn.input <- mx.symbol.transpose(data = attn.input, axes=c(0,1))
  }else{
    attn.input <- mx.symbol.Concat(data = c(output.gru[[i]], rnn.features), num.args = 2, dim = 1)
  }
  attn.numerator <- mx.symbol.FullyConnected(data=attn.input, num_hidden=1)
  attn.numerator <- mx.symbol.exp(attn.numerator)+(mx.symbol.ones_like(attn.numerator)*0.001)
  attn.numerator.stack <- c(attn.numerator.stack, attn.numerator)
}
attn.numerators <- mx.symbol.Concat(data=attn.numerator.stack, num.args = seq.len, dim = 1)
attn.denominator <- mx.symbol.sum(data = attn.numerators, axis = 1)
attn.denominator <- mx.symbol.Reshape(data=attn.denominator, shape=c(1,batch.size))
attn.alphas <- mx.symbol.broadcast_div(attn.numerators, attn.denominator)

attn.check <- mx.symbol.sum(data = attn.alphas, axis = 1)

attn.layer.stack <- list()
attn.alpha.slice <- mx.symbol.SliceChannel(data = attn.alphas, num_outputs = seq.len, axis = 1)
for(i in 1:seq.len){
  attn.layer <- mx.symbol.broadcast_mul(output.gru[[i]], attn.alpha.slice[[i]])
  attn.layer <- mx.symbol.Reshape(data= attn.layer, shape=c(1, (num.filter*length(filter.list)), batch.size) )
  attn.layer.stack <- c(attn.layer.stack, attn.layer)
}
attn.layers <- mx.symbol.Concat(attn.layer.stack, num.args = seq.len, dim = 2)
attn.temp <- mx.symbol.sum_axis(data = attn.layers, axis = 2)

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
neural.componts <- mx.symbol.Concat(data = c(attn.temp, rnn.features), num.args=2, dim = 1)
neural.output <- mx.symbol.FullyConnected(data=neural.componts, num_hidden=input.size)
model.output <- neural.output + ar.output
loss.grad <- mx.symbol.LinearRegressionOutput(data=model.output, label=label, name='loss')

attn.temp.output <- mx.symbol.BlockGrad(attn.check, name='attn')
list.all <- list(loss.grad, attn.temp.output)
lstnet.sym <- mx.symbol.Group(list.all)


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

input.shape <- list()
input.shape[['data']] <- c(168, 321, 128)
input.shape[['l1.gru.init.h']] <- c(300, 128)
input.shape[['label']] <- c(321, 128)
initializer=mx.init.uniform(0.01)
ctx <- mx.cpu()
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

model <-list(lstnet.exec=lstnet.exec, 
             symbol=loss.grad,
             num.rnn.layer=num.rnn.layer, 
             num.filter=num.filter,
             filter.list = filter.list,
             seq.len=seq.len, 
             batch.size=batch.size,
             input.size=input.size)

train.data <- iter.data$train
valid.data <- iter.data$valid
learning.rate <- 0.001
wd <- 0.00
clip_gradient <- TRUE
init.update <- FALSE
optimiser <- 'sgd'
type <- 'attn'


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

init.states <- list()
if(type == 'skip'){
  for(i in 1:num.rnn.layer){
    init.states[[paste0("l", i, ".gru.init.h")]] <- mx.nd.zeros(c(num.filter*length(filter.list), batch.size))
    init.states[[paste0("l", i, ".lstm.init.c")]] <- mx.nd.zeros(c(num.filter*length(filter.list), batch.size))
    init.states[[paste0("l", i, ".lstm.init.h")]] <- mx.nd.zeros(c(num.filter*length(filter.list), batch.size))
  }
}else{
  for(i in 1:num.rnn.layer){
    init.states[[paste0("l", i, ".gru.init.h")]] <- mx.nd.zeros(c(num.filter*length(filter.list), batch.size))
  }
}
begin <- 1
mx.exec.update.arg.arrays(m$lstnet.exec, init.states, match.name = TRUE)
train.input <- lstnet.input(train.data, begin, batch.size)
mx.exec.update.arg.arrays(m$lstnet.exec, train.input, match.name = TRUE)
mx.exec.forward(m$lstnet.exec, is.train = FALSE)

# getting output of network for training evaluation purpose
train.output <- as.array(m$lstnet.exec$outputs[['attn_output']])