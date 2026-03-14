#------------------------------------------------------------------------------#
# run_gpd_hier_mcmc.R
# Hierarchical GPD MCMC for wind speed exceedances
#------------------------------------------------------------------------------#

# 0) Load libraries
library(dplyr)
library(extraDistr)    # dgpd()
library(adaptMCMC)
library(matrixStats)   # rowLogSumExps

# 1) Data import and formatting
df <- read.csv("data_2024_wind.csv", stringsAsFactors = FALSE)
df$date <- as.Date(df$date)
stations <- unique(df$station_name)
seasons  <- unique(df$season)
J <- length(stations); S <- length(seasons)

lengths <- matrix(0, nrow = J, ncol = S,
                  dimnames = list(stations, seasons))
lambdas <- matrix(0, nrow = J, ncol = S,
                  dimnames = list(stations, seasons))
for (j in seq_len(J)) {
  for (s in seq_len(S)) {
    sub <- df %>%
      filter(station_name == stations[j], season == seasons[s]) %>%
      arrange(date)
    lengths[j, s] <- nrow(sub)
    lambdas[j, s] <- mean(sub$exceed95)
  }
}
T_max <- max(lengths)

x_arr <- array(0, dim = c(J, S, T_max))
R_arr <- array(0, dim = c(J, S, T_max))
for (j in seq_len(J)) {
  for (s in seq_len(S)) {
    sub <- df %>%
      filter(station_name == stations[j], season == seasons[s]) %>%
      arrange(date)
    N_js <- lengths[j, s]
    x_arr[j, s, 1:N_js] <- ifelse(sub$exceed95 == 1,
                                  sub$precip_mm - sub$p95,
                                  0)
    R_arr[j, s, 1:N_js] <- sub$exceed95
  }
}

# 2) Univariate GPD log-density
log_gpd_univariate <- function(r, x, sigma, xi, lam,
                               eps = 1e-6, eps_xi = 1e-6) {
  z_raw <- 1 + xi * x / sigma
  z     <- pmax(z_raw, eps)
  log_gpd <- -log(sigma) - (1/xi + 1) * log(z)
  log_exp <- -log(sigma) - x / sigma
  log_exceed    <- ifelse(abs(xi) < eps_xi, log_exp, log_gpd) + log(lam)
  log_nonexceed <- log(1 - lam + eps)
  ifelse(r == 1, log_exceed, log_nonexceed)
}

# 3) Bivariate logistic extreme log-density\log
log_bivar_extremes <- function(r1, r2, x1, x2,
                               sigma, xi, alpha, lam,
                               eps = 1e-6, eps_xi = 1e-6) {
  is_zero <- abs(xi) < eps_xi
  f1 <- pmax(1 + xi * x1 / sigma, eps)
  f2 <- pmax(1 + xi * x2 / sigma, eps)
  logf1 <- log(f1); logf2 <- log(f2)
  logZ_gpd1 <- (1/xi) * logf1 - log(lam)
  logZ_gpd2 <- (1/xi) * logf2 - log(lam)
  logZ_exp1 <- x1/sigma - log(lam)
  logZ_exp2 <- x2/sigma - log(lam)
  logZ1 <- ifelse(is_zero, logZ_exp1, logZ_gpd1)
  logZ2 <- ifelse(is_zero, logZ_exp2, logZ_gpd2)
  inv_alpha <- 1/alpha
  a1 <- -inv_alpha * logZ1
  a2 <- -inv_alpha * logZ2
  logM <- rowLogSumExps(cbind(a1, a2))
  M_alpha_log <- alpha * logM
  log_f_00 <- log1p(-pmin(exp(M_alpha_log), 1 - eps))
  logdZ_gpd1 <- logZ_gpd1 - log(sigma) - logf1
  logdZ_gpd2 <- logZ_gpd2 - log(sigma) - logf2
  logdZ_exp1 <- logZ_exp1 - log(sigma)
  logdZ_exp2 <- logZ_exp2 - log(sigma)
  logdZ1 <- ifelse(is_zero, logdZ_exp1, logdZ_gpd1)
  logdZ2 <- ifelse(is_zero, logdZ_exp2, logdZ_gpd2)
  log_dM_db <- -log(alpha + eps) + (-(inv_alpha + 1)) * logZ2 + logdZ2
  log_f_01 <- log(alpha + eps) + (alpha - 1) * logM + log_dM_db
  log_dM_da <- -log(alpha + eps) + (-(inv_alpha + 1)) * logZ1 + logdZ1
  log_f_10 <- log(alpha + eps) + (alpha - 1) * logM + log_dM_da
  log_f_11 <- log(alpha + eps) + log(abs(alpha - 1) + eps) +
    (alpha - 2) * logM + log_dM_da + log_dM_db
  cond00 <- (r1 == 0 & r2 == 0)
  cond01 <- (r1 == 0 & r2 == 1)
  cond10 <- (r1 == 1 & r2 == 0)
  ifelse(cond00, log_f_00,
         ifelse(cond01, log_f_01,
                ifelse(cond10, log_f_10, log_f_11)))
}

# 4) Log-posterior function
log_posterior <- function(z) {
  # unpack parameters
  g_sigma <- z[1:S]
  e_sigma <- z[(S+1):(S+J)]
  g_xi    <- z[(S+J+1):(2*S+J)]
  e_xi    <- z[(2*S+J+1):(2*S+2*J)]
  a_star  <- z[(2*S+2*J+1):(2*S+3*J)]
  
  # transform parameters
  g_sigma_f <- log1p(exp(g_sigma)); e_sigma_f <- log1p(exp(e_sigma))
  sigma <- outer(e_sigma_f, g_sigma_f, "+")
  g_xi_f    <- log1p(exp(g_xi)); e_xi_f <- log1p(exp(e_xi))
  xi_mat    <- outer(e_xi_f, g_xi_f, "+")
  alpha     <- 1/(1+exp(-a_star))
  
  # priors + Jacobians
  lp <- 0
  lp <- lp + sum(dt(g_sigma, df=5, log=TRUE))
  lp <- lp + sum(dt(e_sigma, df=5, log=TRUE))
  lp <- lp + sum(dt(g_xi,    df=2, log=TRUE))
  lp <- lp + sum(dt(e_xi,    df=2, log=TRUE))
  #lp <- lp + sum(log(1/(1+exp(-g_sigma))))
  #lp <- lp + sum(log(1/(1+exp(-e_sigma))))
  #lp <- lp + sum(log(1/(1+exp(-g_xi))))
  #lp <- lp + sum(log(1/(1+exp(-e_xi))))
  lp <- lp + sum(dbeta(alpha, 1, 1, log=TRUE))
  lp <- lp + sum(log(alpha*(1-alpha) + 1e-6))
  
  # likelihood for t=1
  lp <- lp + sum(sapply(seq_len(J), function(j)
    sapply(seq_len(S), function(s)
      log_gpd_univariate(
        r = R_arr[j,s,1], x = x_arr[j,s,1],
        sigma = sigma[j,s], xi = xi_mat[j,s], lam = lambdas[j,s]
      )
    )
  ))
  
  # likelihood for t=2...T_js
  for (j in seq_len(J)) for (s in seq_len(S)) {
    N_js <- lengths[j,s]
    if (N_js < 2) next
    for (t in 1:(N_js-1)) {
      r1 <- R_arr[j,s,t]; r2 <- R_arr[j,s,t+1]
      x1 <- x_arr[j,s,t]; x2 <- x_arr[j,s,t+1]
      lj <- log_bivar_extremes(r1,r2,x1,x2,
                               sigma[j,s], xi_mat[j,s],
                               alpha[j], lambdas[j,s])
      lmarg <- log_gpd_univariate(r1, x1,
                                  sigma[j,s], xi_mat[j,s],
                                  lambdas[j,s])
      lp <- lp + (lj - lmarg)
    }
  }
  lp
}

# 5) MCMC execution
burn_in <- 10000
n_iter  <- 400000
init_z  <- rep(0, 2*S + 3*J)
scale_z <- rep(0.2, 2*S + 3*J)

start_time <- Sys.time()
out_mcmc <- MCMC(
  p             = log_posterior,
  n             = n_iter,
  init          = init_z,
  scale         = scale_z,
  adapt         = TRUE,
  acc.rate      = 0.234,
  showProgressBar = TRUE,
  list          = TRUE
)
end_time <- Sys.time()
cat("Elapsed time:", end_time - start_time, "\n")

# 6) Save results
full_file_name = "gpd_wind_2024_5_2_full_mcmc.rds"
saveRDS(out_mcmc, full_file_name)
cat("file saved:", full_file_name, "\n")

burned <- out_mcmc$samples[-seq_len(burn_in), ]

burned_file_name = "gpd_wind_2024_5_2_burned_mcmc.rds"
saveRDS(burned, burned_file_name)
cat("file saved:", burned_file_name, "\n")
