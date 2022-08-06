source("code/26072022_Rscript Weather Netherlands.R")
library(glue)
library(ggtext)
library(scales)
library(tidyverse)
library(lubridate)

local_weather <- df
colnames(local_weather) <- c("date", "tmax", "tmin", "prcp")
local_weather <- local_weather %>% 
  mutate(prcp_cm = prcp/10,
         tmax_c = tmax/10)
local_weather <- local_weather[-c(2:3)]
local_weather <- local_weather[-2]
colnames(local_weather) <- c("date", "prcp", "tmax")