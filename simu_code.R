library(kernlab)
library(e1071)
library(ccvsvm)

DataGen3 = function(nn, nm, pp, p1, p2, mu, ro, sdn=NULL, means=NULL){
  if(!is.null(sdn) && is.null(means)) {
    set.seed(sdn)
    means = matrix(rnorm(nm*2*pp), nm*2, pp)
    # negative centers
    means[1:nm, seq(p1)] = means[1:nm, seq(p1)] + mu
    # positive centers
    means[(nm+1):(2*nm), seq(p1+1, p2)] = means[(nm+1):(2*nm), seq(p1+1, p2)] + mu
  }
  id_pos = (1:nn)[rbinom(nn, 1, 0.5)==1]
  size_pos = length(id_pos)
  size_neg = nn - size_pos
  y = rep(-1, nn)
  y[id_pos] = 1
  X = matrix(rnorm(nn*pp, 0, ro), nn, pp)
  ids = rep(NA, nn)
  ids[-id_pos] = sample(1:nm, size_neg, replace=TRUE)
  ids[id_pos] = sample(nm+1:nm, size_pos, replace=TRUE)
  X = X + means[ids, ]
  list(X=X, y=y)
}


stan = function(train, validation=NULL, test=NULL) {
  # standardize data set
  # Args:
  #   train:      original training set
  #   validation: original validation set
  #   test:       original test set
  # Returns:
  #   train:      standardized training set
  #   validation: standardized validation set
  #   test:       standardized test set
  train_m     = colMeans(train$X)
  std_train_x = t(apply(train$X, 1, function(x) x - train_m))  
  train_sd    = apply(std_train_x, 2, 
                function(x) sqrt(x %*% x / length(x)))
  train_sd[train_sd==0] = 1
  train$X = t(apply(std_train_x, 1, function(x) x / train_sd))  
  if (!is.null(validation)) validation$X = 
    scale(validation$X, center=train_m, scale=train_sd)
  if (!is.null(test)) test$X = 
    scale(test$X, center=train_m, scale=train_sd)
  rm.att = function(x) {
    attributes(x) = attributes(x)[c(1,2)]
    x
  } 
  train$X = rm.att(train$X)
  validation$X = rm.att(validation$X)
  test$X = rm.att(test$X)
  # returns:
  list(train=train, validation=validation, test=test)
}

tune.ksvm = function(train, kernel, cross=5,
    C.list = 2^seq(-10, 10, len=51)) {
  # tune C and kernel parameter for kernel svm
  #
  # Args:
  #   train:   training data
  #   kernel:  selected kernel
  #   cross:   number of folds in cross validation
  #   C.list:  candidate values of C
  # Returns:
  #   Cval:    best C
  #   sig:     best kernel parameter

  C.len = length(C.list)
  tuneres = matrix(NA, C.len, 3)
  tuneres[, 1] = C.list
  colnames(tuneres) = c("C", "err", "sigma")
  for(i in seq(C.len)) {
    log = capture.output(
      m_t <- ksvm(x=train$X, y=train$y, kernel=kernel,
        kpar="automatic", C=C.list[i], cross=cross, 
        prob.model=FALSE, scaled=FALSE, type='C-svc')
    )
    tuneres[i, 2] = m_t@cross
    tuneres[i, 3] = m_t@kernelf@kpar$sigma
  }
  bestres = tuneres[which.min(tuneres[,2]), ]
  Cval = bestres[1]
  sig = bestres[3]
  #returns:
  list(Cval=Cval, sig=sig)
}

p1 = p2 = pp/2
means = matrix(rnorm(nm*2*pp), nm*2, pp)
means[1:nm, seq(p1)] = means[1:nm, seq(p1)] + mu
means[(nm+1):(2*nm), seq(p1+1, p2)] = means[(nm+1):(2*nm), seq(p1+1, p2)] + mu
bigN = as.integer(100000)

objfun = function(alp, intcp, Kmat, y, Cval){
  fh = as.vector(intcp + Kmat %*% alp)
  xi = ifelse(1 - y * fh > 0, 1 - y * fh, 0)
  lam = 0.5/Cval/length(y)
  as.numeric(lam * alp %*% Kmat %*% alp + sum(xi) / length(y)) 
}


for (rs in 1:50){
  train = DataGen3(nn=Nobs, nm=10L, pp=pp, p1=p1, p2=p2, mu=mu, ro=ro, means=means)
  sig = sigest(train$X)
  kern = rbfdot(sigma=sig)
  test = DataGen3(nn=10000, nm=10L, pp=pp, p1=p1, p2=p2, mu=mu, ro=ro, means=means)

  stanres = stan(train, test=test)
  train = stanres$train
  test = stanres$test
  rm(stanres)
  nf = length(train$y)
  lambda = exp(seq(6, -6, len=50))

  set.seed(rs)
  start_dr = Sys.time()
  fit_dr = scsvm(train$X, train$y, kern=kern, lambda=lambda, intcpt=TRUE,
    isdr=TRUE, eps=1e-6, maxit=10000L, pred.loss="misclass")
  end_dr = Sys.time()
  tim_dr = as.numeric(difftime(end_dr, start_dr, units="secs"))
  err_dr = mean(predict(fit_dr, kern, train$X, test$X, s=fit_dr$lambda.min) != test$y)

  set.seed(rs)
  start_1071 = Sys.time()
  tc = tune.control(cross=nf)
  cv_1071 = tune.svm(x=train$X, y=factor(train$y), kernel="radial", gamma=sig,
    cost=0.5/lambda/length(train$y), tunecontrol=tc, scaled=FALSE)
  end_1071 = Sys.time()
  tim_1071 = as.numeric(difftime(end_1071, start_1071, units="secs"))
  err_1071 = mean(predict(cv_1071$best.model, test$X) != test$y)


  set.seed(rs)
  start_klab = Sys.time()
  cv_klab = tune.ksvm(train, kernel="rbfdot", cross=nf, C.list=0.5/lambda/length(train$y))
  md_klab = ksvm(x=train$X, y=train$y, kernel="rbfdot", kpar=list(sigma=sig), 
    C=cv_1071$best.model$cost, cross=0, prob.model=FALSE, scaled=FALSE, type='C-svc')
  end_klab = Sys.time()
  tim_klab = as.numeric(difftime(end_klab, start_klab, units="secs"))
  err_klab = mean(predict(md_klab, test$X) != test$y)


  obj_dr = objfun(fit_dr$alp[-1, match(fit_fl$lambda.min, fit_fl$lambda)], 
    fit_dr$alp[1, match(fit_fl$lambda.min, fit_fl$lambda)], Kmat, train$y, Cval)

  m2 = ksvm(x=train$X, y=train$y, kernel="rbfdot", kpar=list(sigma=sig), 
    C=Cval, cross=0, prob.model=FALSE, scaled=FALSE, type='C-svc')
  alp2 = rep(0, length(train$y))
  alp2[m2@SVindex] = m2@coef[[1]]
  m3 = svm(x=train$X, y=factor(train$y), kernel="radial", gamma=sig, 
    cost=Cval, scaled=FALSE)
  alp3 = rep(0, length(train$y))
  alp3[m3$index] = m3$coefs

  obj_klab = objfun(alp2, -m2@b, Kmat, train$y, Cval)
  sns = sign(m3$rho/m2@b)
  obj_1071 = objfun(sns*alp3, -sns*m3$rho, Kmat, train$y, Cval)

}
