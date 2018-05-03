require(mxnet)

mx.std <- function(label){
  label_vec <- as.array(mx.nd.reshape(label, shape=c(1,-1)))
  return(sd(label_vec))
}

mx.std.array <- function(label){
  label.shape <- dim(label)
  label.mean <- mx.nd.mean(label, axis=1)
  label.mean <- matrix(as.array(label.mean), 1, label.shape[2])
  label.mean <- mx.nd.array(label.mean)
  label_mean.broad <- mx.nd.broadcast.axis(label.mean, axis=1, size=label.shape[1])
  
  numerator <- mx.nd.sum.axis(mx.nd.square(label - label_mean.broad), axis=1)
  numerator <- matrix(as.array(numerator), 1, label.shape[2])
  numerator <- mx.nd.array(numerator)
  denominator <- mx.nd.array(matrix((label.shape[1]-1), 1, label.shape[2]))
  std_array <- mx.nd.sqrt(numerator/denominator)
  return(std_array)
}

rse <- function(label, pred)
{
  numerator   <- as.array(mx.nd.sqrt(mx.nd.mean(mx.nd.square(label - pred))))
  denominator <- mx.std(label)
  return(as.array(numerator/denominator))
}

rae <- function(label, pred)
{
  label.vec <- mx.nd.reshape(label, shape=c(1,-1))
  pred.vec  <- mx.nd.reshape(pred,  shape=c(1,-1))
  vec.length <- dim(label.vec)[2]
  
  label_mean <- mx.nd.mean(label.vec)
  label_mean <- mx.nd.array(matrix(as.array(label_mean), 1, 1))
  label_mean_broad <- mx.nd.broadcast.axis(label_mean, axis=0, size = vec.length)
  
  numerator <- mx.nd.mean(mx.nd.abs(label.vec - pred.vec))
  denominator <- mx.nd.mean(mx.nd.abs(label.vec - label_mean_broad))
  return(as.array(numerator/denominator))
}

corr <- function(label, pred)
{
  label.shape <- dim(label)
  label.mean <- mx.nd.mean(label, axis=1)
  label.mean <- matrix(as.array(label.mean), 1, label.shape[2])
  label.mean <- mx.nd.array(label.mean)
  label_mean.broad <- mx.nd.broadcast.axis(label.mean, axis=1, size=label.shape[1])
  
  pred.shape <- dim(pred)
  pred.mean <- mx.nd.mean(pred, axis=1)
  pred.mean <- matrix(as.array(pred.mean), 1, pred.shape[2])
  pred.mean <- mx.nd.array(pred.mean)
  pred_mean.broad <- mx.nd.broadcast.axis(pred.mean, axis=1, size=pred.shape[1])
  
  numerator1 <- label - label_mean.broad
  numerator2 <- pred - pred_mean.broad
  numerator <- mx.nd.mean((numerator1 * numerator2), axis=1)
  
  numerator <- matrix(as.array(numerator), 1, label.shape[2])
  numerator <- mx.nd.array(numerator)
  
  denominator <- mx.std.array(label) * mx.std.array(pred)
  
  corr <- mx.nd.mean(numerator/denominator)
  
  return(as.array(corr))
}

evaluate <- function(pred, label)
{
  pred <- mx.nd.array(pred)
  label <- mx.nd.array(label)
  
  eva.all <-list()
  eva.all[['RAE']]  <- rae(label = label, pred = pred)
  eva.all[['RSE']]  <- rse(label = label, pred = pred)
  eva.all[['CORR']] <- corr(label = label, pred = pred)
  return(eva.all)
}
  