require(mxnet)

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

m1 <- mx.symbol.Variable('matrix1')
m2 <- mx.symbol.Variable('matrix2')
ct.stack<-list()
slice.broad <- list()
for(i in 1:168){
  slice.broad[[i]] <- mx.symbol.broadcast_axis(data = m2, axis=1, size=300)
  ct.stack[[i]] <- m1 * slice.broad[[i]]
  ct.stack[[i]]<-mx.symbol.Reshape(data=ct.stack[[i]], shape=c(300, 128, 1) )
}
ct <- mx.symbol.concat(ct.stack, num.args = 168, dim = 2) 


input.shape<-list()
input.shape[['matrix1']] <- c(300, 128)
input.shape[['matrix2']] <- c(1, 128)

initializer=mx.init.uniform(0.01)
ctx <- mx.cpu()
params <- mx.model.init.params(symbol = ct, input.shape = input.shape, initializer = initializer, ctx = ctx)
args <- input.shape
args$symbol <- ct
args$ctx <- ctx
args$grad.req <- 'add'
ct.exec <- do.call(mx.simple.bind, args)