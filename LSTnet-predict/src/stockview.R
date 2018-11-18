require(RMySQL)
require(mxnet)
require(abind)
require(quantmod)


getSymbols('000001.SS')
SSE <- get('000001.SS')

chartSeries(SSE, subset='last 3 months')
addBBands(n = 20, sd = 2, ma = "SMA", draw = 'bands', on = -1)

SSE.Cl <- Cl(SSE)
SSE.Cl <- na.omit(SSE.Cl)
SSE.MACD <- MACD(SSE.Cl)

plot(last(SSE.MACD$macd, '3 months'))
lines(last(SSE.MACD$signal, '3 months'), col='red')



getSymbols('TLS.AX')
chartSeries(TLS.AX, subset='last 3 months')
addBBands(n = 20, sd = 2, ma = "SMA", draw = 'bands', on = -1)


TLS.Cl <- Cl(TLS.AX)
TLS.Cl <- na.omit(TLS.AX)
TLS.MACD <- MACD(TLS.AX$TLS.AX.Close)

TLS.AX.3months <- last(TLS.MACD, '3 months')

plot(TLS.AX.3months$macd)
lines(TLS.AX.3months$signal, col='red')