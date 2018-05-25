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

require(MASS)
require(plotrix)
setwd('~/Documents/deeplearning.mxnet/SVDD/src/')
source('./svdd_dual.R')

#making some dummy data
mu <- rep(0,2)
cov <- matrix(c(1,0,0,1), 2, 2)

#positive/negative dummy data in format input.size * batch.size
x.pos <- matrix(c(0.5066706, 1.3298224, 1.216042 , 1.087533 , 0.577895 , 1.557349 , 1.079452 , 1.166510 , 0.9161158 , 1.3927195 , 1.030383 , 1.292907 , 0.7323972 , 1.0062514 , 1.3570871 , 0.8118171 , 1.065125 , 1.226348 , 1.171299 , 1.049957), 2, 10)
x.neg <- t(mvrnorm(n = 2, mu, cov*3)+1)
x.all <- cbind(x.pos,x.neg)

#postive/negative dummy lable
label.pos <- matrix(rep(1,10))
label.neg <- matrix(rep(-1,2))
label <- rbind(label.pos, label.neg)

#making svdd.smo model
smo <- svdd.smo(data = x.all,
                label = label,
                alphas = NULL,    #NULL will generate intial alphas randomly
                C = 1000,
                tolerant = 0.0001,
                epsilon = 0.0001,
                sigma = 1,        #standard deviation for RBF kernel, defaultly 1
                kernel='linear'   #kernel: options: 'linear' 'RBF'
                ) 

#train svdd.smo model
outputs <- svdd.smo.train(smo, epoch.max = 150)

#centre
a <- x.all%*%(outputs$alphas*outputs$label)

#radius
r <- sqrt(outputs$R2)

#visualisation
plot(x.pos[1,], x.pos[2,], type='p', xlim=c(-2,4), ylim=c(-2,4), asp=1)
lines(x.neg[1,], x.neg[2,], type='p', col='blue')
draw.circle(a[1],a[2],r,nv=1000,border='red',col=NA,lty=1,lwd=1)
lines(a[1], a[2], type='p', col='red')