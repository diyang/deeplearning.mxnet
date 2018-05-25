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
#
# Support Vector Data Description (SVDD) for anomaly detection,
# This SVDD is based on 
# Tax, David M.J. and Duin, Robert P.W. "Support Vector Data Description"
# in "Machine Learning" 2004, 54, 1, pp. 45-66
# https://doi.org/10.1023/B:MACH.0000008084.60811.49

svdd.error <- function(idx, smo){
  K <- smo$K
  alphas <- smo$alphas
  label <- smo$label
  error <- smo$R2 - (K[idx, idx] - 2*t(label*alphas)%*%K[,idx] + t(label*alphas)%*%K%*%(label*alphas))
  return(error)
}

svdd.smo <- function(data, 
                     label,
                     alphas = NULL,
                     C = 0.8, 
                     tolerant= 0.0001,
                     epsilon = 0.0001,
                     sigma = 1,
                     kernel='linear'){
  
  input.size <- dim(data)[1]
  batch.size <- dim(data)[2]
  
  if(kernel == 'linear'){
    K <- t(data)%*%data
  }else if(kernel == 'RBF'){
    K <- matrix(0, batch.size, batch.size)
    for(i in 1:batch.size){
      for(j in 1:batch.size){
        K[i,j] <- t(data[,i] - data[,j])%*%(data[,i]-data[,j])
        K[i,j] <- exp(K[i,j]/(-1*sigma**2))
      }
    }
  }
  if(is.null(alphas)){
    alphas <- runif(batch.size, min=0, max=100)
    alphas <- matrix(alphas/sum(alphas))
  }
  
  # set first element as support vector to start loop
  init.R2 <- K[1,1] -2*t(label*alphas)%*%K[,1] + t(label*alphas)%*%K%*%(label*alphas)
  error.cache <- matrix(0, batch.size, 1)
  for(i in 1:batch.size){
    error.cache[i] <- init.R2 - (K[i,i] - 2*t(label*alphas)%*%K[,i] + t(label*alphas)%*%K%*%(alphas*label))
  }
  
  return(list(input.size = dim(data)[1],
              batch.size = dim(data)[2],
              alphas = alphas,
              R2 = init.R2,
              error.cache = error.cache,
              K = K,
              data = data,
              label = label,
              C=C,
              tolerant = tolerant, 
              epsilon = epsilon))
}

smo.feedforward <- function(i1, i2, smo){
  alpha1 <- smo$alphas[i1]
  E1 <- smo$error.cache[i1]
  y1 <- smo$label[i1]
  
  alpha2 <- smo$alphas[i2]
  E2 <- smo$error.cache[i2]
  y2 <- smo$label[i2]
  
  s<-y1*y2
  
  if(y1 == y2){
    L = max(0, alpha1+alpha2-smo$C)
    H = min(smo$C, alpha1+alpha2)
  }else{
    L = max(0, alpha2-alpha1)
    H = min(smo$C, smo$C+alpha2-alpha1)
  }
  if(L==H){
    return(list(smo=smo, flag = FALSE))
  }
  #quadratic programming 
  a <- 2*smo$K[i1,i2] - smo$K[i1,i1] - smo$K[i2,i2]
  b <- 2*(alpha1+s*alpha2)*s*(smo$K[i1,i1] - smo$K[i1,i2]) + y2*(smo$K[i2,i2] - smo$K[i1,i1]) + y2*t(smo$alphas*smo$label)%*%(smo$K[,i1]-smo$K[,i2]) + s*alpha1*(smo$K[i1,i2] - smo$K[i1,i1]) + alpha2*(smo$K[i2,i2] - smo$K[i1,i2])
  if(a<0){
    alpha2.new <- b/(-2*a)
    #clipping back
    if(alpha2.new < L){
      alpha2.new <- L
    }else if(alpha2.new > H){
      alpha2.new <- H
    }
  }else{
    Lobj <- a*(L**2) + b*H
    Hobj <- a*(H**2) + b*H
    if(Lobj > (Hobj + smo$epsilon)){
      alpha2.new <- L
    }else if(Lobj <(Hobj - smo$epsilon)){
      alpha2.new <- H
    }else{
      alpha2.new <- alpha2
    }
  }
  
  if(abs(alpha2.new - alpha2) < smo$epsilon*(alpha2+alpha2.new+smo$epsilon)){
    return(list(smo=smo, flag = FALSE))
  }
  
  alpha1.new <- alpha1 - s*(alpha2.new - alpha2)
  if(alpha1.new < 0){
    alpha2.new <- alpha1.new+s*alpha2.new
    alpha1.new <- 0
  }else if(alpha1.new > smo$C){
    offset <- alpha1.new - smo$C
    alpha2.new <- alpha2.new+s*offset
    alpha1.new <- smo$C
  }
  # make sure R2 comply with K.K.T.
  smo$alphas[i1] <- alpha1.new
  smo$alphas[i2] <- alpha2.new
  if(alpha1.new >0 && alpha1.new < smo$C){
    R2.new <- smo$R2-svdd.error(i1, smo) 
  }else{
    if (alpha2.new >0 && alpha2.new < smo$C){
      R2.new <- smo$R2-svdd.error(i2, smo) 
    }else{
      R2.i1 <- smo$R2-svdd.error(i1, smo) 
      R2.i2 <- smo$R2-svdd.error(i2, smo)
      R2.new <- (R2.i1+R2.i2)/2
    }
  }
  smo$R2 <- R2.new
  #update error matrix
  for(i in 1:smo$batch.size){
    smo$error.cache[i] <- svdd.error(i, smo)
  }
  if(alpha1.new > 0 && alpha1.new < smo$C){
    smo$error.cache[i1] <- 0
  }else{
    if(alpha2.new >0 && alpha2.new < smo$C){
      smo$error.cache[i2] <- 0
    }
  }
  
  return(list(smo=smo, flag = TRUE))
}

svdd.smo.examine <- function(i1, smo){
  alpha1 <- smo$alphas[i1]
  E1 <- smo$error.cache[i1]
  y1 <- smo$label[i1]
  r1 <- y1*E1
  if((r1<(smo$tolerant*-1)&&alpha1<smo$C)||(r1>smo$tolerant&&alpha1>0)){
    # heuristic 1: find the max deltaE
    max_delta_E <- 0
    i2 <- 0
    for(i in 1:smo$batch.size){
      #if it's supposed to be a support vector
      if(smo$alphas[i] >0 && smo$alphas[i]<smo$C){
        if(i != i1){
          E2 <- smo$error.cache[i]
          delta_error <- abs(E2-E1)
          if(delta_error > max_delta_E){
            max_delta_E = delta_error
            i2 = i
          }
        }
      }
    }
    
    if(i2 != 0){
      smo.new <- smo.feedforward(i1,i2,smo)
      if(smo.new$flag == TRUE ){
        return(list(smo=smo.new$smo, num.changed=1))
      }
    }
    
    # heuristic 2: find the suitable i1 on border at random
    random.index <- sample(smo$batch.size, smo$batch.size)
    for(i in 1:smo$batch.size){
      idx <- random.index[i]
      if(smo$alphas[idx]>0 && smo$alphas[idx]<smo$C){
        if(idx != i1){
          i2 <- idx
          smo.new <- smo.feedforward(i1,i2,smo)
          if(smo.new$flag == TRUE ){
            return(list(smo=smo.new$smo, num.changed=1))
          }
        }
      }
    }
    
    # heuristic 3: find the suitable i1 at random on all alphas
    random.index <- sample(smo$batch.size, smo$batch.size)
    for(i in 1:smo$batch.size){
      idx <- random.index[i]
      if(idx != i1){
        i2 <- idx
        smo.new <- smo.feedforward(i1,i2,smo)
        if(smo.new$flag == TRUE ){
          return(list(smo=smo.new$smo, num.changed=1))
        }
      }
    }
  }
  return(list(smo=smo, num.changed=0))
}

svdd.smo.train <- function(smo, epoch.max=1000){
  epoch <- 1
  examine.all <- TRUE
  batch.size <- smo$batch.size
  num.changed <- 0
  error.stack <- list()
  while(epoch < epoch.max && num.changed > 0 || examine.all){
    num.changed <- 0
    if(examine.all){
      for(i in 1:batch.size){
        outputs <- svdd.smo.examine(i,smo)
        smo <- outputs$smo
        num.changed <- num.changed+outputs$num.changed
      }
    }else{
      for(i in 1:batch.size){
        if(smo$alphas[i] != 0 && smo$alphas[i] != smo$C){
          outputs <- svdd.smo.examine(i,smo)
          smo <- outputs$smo
          num.changed <- num.changed+outputs$num.changed
        }
      }
    }
    if(examine.all){
      examine.all <- FALSE
    }else if(num.changed == 0){
      examine.all <- TRUE
    }
    
    error <- diag(smo$K)%*%(smo$alphas*smo$label) - t(smo$alphas*smo$label)%*%smo$K%*%(smo$alphas*smo$label)
    error.stack <- c(error.stack, error)
    cat(paste0('epoch: ', epoch, '; error: ', error, '; num.change: ',num.changed,'\n'))
    epoch = epoch + 1
  }
  return(smo)
}