source("code/20220608_local_weather_ne.R")

pretty_labels <- c("prob_prcp" = "Probality of precipitation", 
                  "mean_prcp" = "Average amount of\nprecipitation by day (mm)",
                  "mean_event" = "Average amount of\nprecipitation by event (mm)")

Sys.setlocale("LC_TIME", "English")

today_month <- month(today())
today_day <- day(today())
today_date <- ymd(glue("2020-{today_month}-{today_day}"))


local_weather %>% 
  select(date, prcp) %>% 
  mutate(day = day(date),
         month = month(date),
         year = year(date)) %>% 
  drop_na(prcp) %>% 
  group_by(month, day) %>% 
  summarize(prob_prcp = mean(prcp > 0),
            mean_prcp = mean(prcp),
            mean_event = mean(prcp[prcp > 0]),
            .groups = "drop") %>% 
  mutate(date = ymd(glue("2020-{month}-{day}"))) %>% 
  select(-month, -day) %>% 
  pivot_longer(cols = c(prob_prcp, mean_prcp, mean_event)) %>% 
  mutate(name = factor(name, levels = c("prob_prcp", "mean_prcp", "mean_event"))) %>% 
  ggplot(aes(x=date, y=value, color = name)) +
  geom_vline(xintercept = today_date, color = "red", size = 1) +
  geom_line() +
  geom_hline(yintercept = 0, color = "white") +
  geom_smooth(se = FALSE) +
  facet_wrap(~name, ncol = 1, scales = "free_y", 
             strip.position = "left",
             labeller = labeller(name = pretty_labels)) +
  scale_y_continuous(limits = c(0,NA), expand = c(0,0)) +
  scale_x_date(date_breaks = "2 months", 
               date_labels = "%B") +
  coord_cartesian(clip = "off") +
  labs(x=NULL,
        y=NULL,
       title = "Probability and Average Precipitation in The Netherlands",
       #title = glue("Through {today_month} {today_day}, the precipitation in The Netherlands near De Bilt has a <span style = 'color: red'>probability</span> and <span style = 'color: red'>average</span> for {today_date}")) +
       subtitle = "The red vertical line is today (August 6th, 2022)") +
  theme(
    panel.background = element_rect(fill = "black"),
    strip.placement = "outside",
    strip.background = element_blank(),
    strip.text = element_text(color = "white", size = 12), 
    panel.grid = element_blank(),
    plot.background = element_rect(fill = "black", color = "black"),
    axis.line = element_line(),
    plot.title.position = "plot",
    plot.title = element_textbox_simple(margin=margin(b=10), size = 25, color = "white"),
    plot.subtitle = element_textbox_simple(margin=margin(b=10), size = 15, color = "white"),
    axis.title.y = element_text(color = "white"),
    axis.text = element_text(color = "white", size = 12),
    legend.background = element_rect(fill = "black"),
    legend.text = element_text(color = "white", size = 10),
    legend.title = element_text(color = "white"),
    legend.position = "bottom",
    legend.key = element_rect(fill = "black")
    

    
    
    
    
    
  )

ggsave("figures/prob_prcp_amount_ne.png", width = 7, height = 9)
