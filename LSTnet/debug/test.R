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
attn.output.slice <- mx.symbol.SliceChannel(data = attn.numerator.stack, num_outputs = seq.len, axis = 1, squeeze_axis = 1)

#attn.output.slice.layer <- mx.symbol.ones_like(rnn.features)

for(i in 1:seq.len){
  #attn.output.slice[[i]] <- mx.symbol.Reshape(data=attn.output.slice[[i]], shape=c(1, 128) )
  attn.output.slice.broad[[i]] <- mx.symbol.broadcast_axis(data = attn.output.slice[[i]], axis=1, size=num.filter*length(filter.list))
  c.t.stack[[i]]<-output.gru[[i]]*attn.output.slice.broad[[i]]
  c.t.stack[[i]]<-mx.symbol.Reshape(data=c.t.stack[[i]], shape=c(300, 128, 1) )
}
c.t <- mx.symbol.concat(c.t.stack, num.args = seq.len, dim = 2)
#c.t <- mx.symbol.zeros_like(rnn.features) 

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
params <- mx.model.init.params(symbol = c.t, input.shape = input.shape, initializer = initializer, ctx = ctx)
args <- input.shape
args$symbol <- c.t
args$ctx <- ctx
args$grad.req <- 'add'
c.t.exec <- do.call(mx.simple.bind, args)