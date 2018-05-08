require(mxnet)
source("./lstnet_model.R")

num.rnn.layer <- 1
seq.len <- 24*7
num.filter <- 100
filter.list <- c(6, 12, 18)
dropout <- 0
batch.size <- 128

data <- mx.symbol.Variable('data')

output.gru.bulk  <- rnn.unroll(data=data, num.rnn.layer=num.rnn.layer, seq.len=seq.len, num.hidden=num.filter*length(filter.list), dropout=dropout)
output.gru <- output.gru.bulk$outputs
rnn.features = output.gru[[seq.len]]

attn.numerator.stack <- list()
for(i in 1:seq.len){
  if(i == seq.len){
    attn.input <- mx.symbol.Reshape(data = rnn.features, shape=c(300, 128, 1))
    attn.input <- mx.symbol.broadcast_axis(data=attn.input, axis=0, size=2)
    attn.input <- mx.symbol.Reshape(data = attn.input, shape=c(128,600))
    attn.input <- mx.symbol.transpose(data = attn.input, axes=c(0,1))
  }else{
    attn.input <- mx.symbol.Concat(data = c(output.gru[[i]], rnn.features), num.args = 2, dim = 1)
  }
  attn.numerator <- mx.symbol.FullyConnected(data=attn.input, num_hidden=1)
  attn.numerator <- mx.symbol.exp(attn.numerator)
  attn.numerator.stack <- c(attn.numerator.stack, attn.numerator)
}
attn.numerators <- mx.symbol.Concat(data=attn.numerator.stack, num.args = seq.len, dim = 1)
attn.denominator <- mx.symbol.sum(data = attn.numerators, axis = 1)
attn.denominator <- mx.symbol.Reshape(data=attn.denominator, shape=c(1,batch.size))
attn.alphas <- mx.symbol.broadcast_div(attn.numerators, attn.denominator)

attn.layer.stack <- list()
attn.alpha.slice <- mx.symbol.SliceChannel(data = attn.alphas, num_outputs = seq.len, axis = 1)
for(i in 1:seq.len){
  attn.layer <- mx.symbol.broadcast_mul(output.gru[[i]], attn.alpha.slice[[i]])
  attn.layer <- mx.symbol.Reshape(data= attn.layer, shape=c(1, 300, 128) )
  attn.layer.stack <- c(attn.layer.stack, attn.layer)
}
attn.layers <- mx.symbol.Concat(attn.layer.stack, num.args = seq.len, dim = 2)
attn.temp <- mx.symbol.sum_axis(data = attn.layers, axis = 2) 

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
input.shape[['data']] <- c(168, 300, 128)
input.shape[['l1.gru.init.h']] <- c(300, 128)
initializer=mx.init.uniform(0.01)
ctx <- mx.cpu()
params <- mx.model.init.params(symbol = attn.temp, input.shape = input.shape, initializer = initializer, ctx = ctx)
args <- input.shape
args$symbol <- attn.temp
args$ctx <- ctx
args$grad.req <- 'add'
attn.exec <- do.call(mx.simple.bind, args)

require(mxnet)
a <- mx.nd.array(matrix(c(1,2,3,4,5,6,7,8,9,10,11,12), 3, 4))
b <- mx.nd.array(matrix(c(2,4,6,8,10,12,14,16,18,20,22,24), 3, 4))
c <- mx.nd.stack(c(a,b))

d <- mx.nd.reshape(c, shape=c(4, 6))

e <- mx.nd.transpose(d, axes=c(0,1))

