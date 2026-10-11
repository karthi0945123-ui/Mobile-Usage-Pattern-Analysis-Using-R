# ============================================================
# Mobile Usage Pattern Analysis - Shiny Dashboard
# Run:  shiny::runApp()   (data/user_behavior_dataset.csv must exist)
# ============================================================
pkgs <- c("shiny","dplyr","ggplot2","DT","randomForest")
for (p in pkgs) if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
invisible(lapply(pkgs, library, character.only = TRUE))

load_data <- function() {
  if (file.exists("models/clean_data.rds")) return(readRDS("models/clean_data.rds"))
  d <- read.csv("data/user_behavior_dataset.csv", stringsAsFactors = FALSE)
  names(d) <- c("UserID","DeviceModel","OS","AppUsageMin","ScreenOnHrs","BatteryDrain",
                "AppsInstalled","DataUsageMB","Age","Gender","BehaviorClass")
  d$DeviceModel <- factor(d$DeviceModel); d$OS <- factor(d$OS); d$Gender <- factor(d$Gender)
  d$BehaviorClass <- factor(paste0("C", d$BehaviorClass), levels = paste0("C", 1:5)); d
}
df <- load_data()
df$Cluster <- NULL
feat <- c("OS","AppUsageMin","ScreenOnHrs","BatteryDrain","AppsInstalled","DataUsageMB","Age","Gender")
rf <- if (file.exists("models/rf_behavior.rds")) readRDS("models/rf_behavior.rds") else
  randomForest(BehaviorClass ~ ., data = df[, c("BehaviorClass", feat)], ntree = 300)
num_vars <- c("AppUsageMin","ScreenOnHrs","BatteryDrain","AppsInstalled","DataUsageMB","Age")

ui <- navbarPage("Mobile Usage Pattern Analysis",
  tabPanel("Overview",
    fluidRow(
      column(3, wellPanel(h5("Total Users"),          h3(textOutput("n_users")))),
      column(3, wellPanel(h5("Avg Screen-On (hrs)"),  h3(textOutput("avg_screen")))),
      column(3, wellPanel(h5("Avg Battery Drain (mAh)"), h3(textOutput("avg_batt")))),
      column(3, wellPanel(h5("Avg Data Usage (MB)"),  h3(textOutput("avg_data"))))),
    fluidRow(column(6, plotOutput("class_plot")), column(6, plotOutput("os_plot")))),
  tabPanel("Explore",
    sidebarLayout(
      sidebarPanel(
        selectInput("os", "Operating System", c("All", levels(df$OS))),
        checkboxGroupInput("gender", "Gender", levels(df$Gender), selected = levels(df$Gender)),
        sliderInput("age", "Age range", min(df$Age), max(df$Age), c(min(df$Age), max(df$Age))),
        selectInput("xv", "X variable", num_vars, "ScreenOnHrs"),
        selectInput("yv", "Y variable", num_vars, "BatteryDrain")),
      mainPanel(plotOutput("scatter"), plotOutput("box")))),
  tabPanel("Usage Clusters",
    sidebarLayout(
      sidebarPanel(sliderInput("k", "Number of clusters (k)", 2, 6, 3, step = 1),
                   helpText("K-Means on scaled usage variables.")),
      mainPanel(plotOutput("clus_plot"), h4("Cluster averages"), tableOutput("clus_tbl")))),
  tabPanel("Predict Behavior Class",
    sidebarLayout(
      sidebarPanel(
        selectInput("p_os", "OS", levels(df$OS)),
        selectInput("p_gender", "Gender", levels(df$Gender)),
        numericInput("p_age", "Age", 30, min = 10, max = 90),
        numericInput("p_app", "App usage (min/day)", 200),
        numericInput("p_screen", "Screen-on time (hrs/day)", 5),
        numericInput("p_batt", "Battery drain (mAh/day)", 1500),
        numericInput("p_apps", "Apps installed", 50),
        numericInput("p_data", "Data usage (MB/day)", 1000),
        actionButton("go", "Predict", class = "btn-primary")),
      mainPanel(h3(textOutput("pred_class")), tableOutput("pred_prob")))),
  tabPanel("Data", DTOutput("tbl")))

server <- function(input, output, session) {
  output$n_users    <- renderText(nrow(df))
  output$avg_screen <- renderText(round(mean(df$ScreenOnHrs), 2))
  output$avg_batt   <- renderText(round(mean(df$BatteryDrain)))
  output$avg_data   <- renderText(round(mean(df$DataUsageMB)))
  output$class_plot <- renderPlot(ggplot(df, aes(BehaviorClass, fill = BehaviorClass)) +
    geom_bar() + ggtitle("Users per Behavior Class") + theme(legend.position = "none"))
  output$os_plot <- renderPlot(ggplot(df, aes(OS, fill = Gender)) +
    geom_bar(position = "dodge") + ggtitle("OS by Gender"))

  filt <- reactive({
    d <- df %>% filter(Gender %in% input$gender, Age >= input$age[1], Age <= input$age[2])
    if (input$os != "All") d <- d %>% filter(OS == input$os); d })
  output$scatter <- renderPlot({
    req(nrow(filt()) > 0)
    ggplot(filt(), aes(.data[[input$xv]], .data[[input$yv]], colour = BehaviorClass)) +
      geom_point(alpha = .6) + geom_smooth(method = "lm", se = FALSE, colour = "black") +
      ggtitle(paste(input$yv, "vs", input$xv)) })
  output$box <- renderPlot({
    req(nrow(filt()) > 0)
    ggplot(filt(), aes(BehaviorClass, .data[[input$yv]], fill = BehaviorClass)) +
      geom_boxplot() + theme(legend.position = "none") + ggtitle(paste(input$yv, "by class")) })

  cl <- reactive({
    cols <- c("AppUsageMin","ScreenOnHrs","BatteryDrain","DataUsageMB","AppsInstalled")
    set.seed(42); km <- kmeans(scale(df[, cols]), input$k, nstart = 25)
    d <- df; d$Cluster <- factor(km$cluster); d })
  output$clus_plot <- renderPlot(ggplot(cl(), aes(ScreenOnHrs, AppUsageMin, colour = Cluster)) +
    geom_point(alpha = .7) + ggtitle("Usage clusters"))
  output$clus_tbl <- renderTable(cl() %>% group_by(Cluster) %>%
    summarise(Users = n(), AppUsageMin = mean(AppUsageMin), ScreenOnHrs = mean(ScreenOnHrs),
              BatteryDrain = mean(BatteryDrain), DataUsageMB = mean(DataUsageMB),
              AppsInstalled = mean(AppsInstalled)))

  pred <- eventReactive(input$go, {
    nd <- data.frame(OS = factor(input$p_os, levels(df$OS)), AppUsageMin = input$p_app,
      ScreenOnHrs = input$p_screen, BatteryDrain = input$p_batt, AppsInstalled = input$p_apps,
      DataUsageMB = input$p_data, Age = input$p_age, Gender = factor(input$p_gender, levels(df$Gender)))
    list(cls = predict(rf, nd), prob = predict(rf, nd, type = "prob")) })
  output$pred_class <- renderText(paste("Predicted behavior class:", as.character(pred()$cls)))
  output$pred_prob  <- renderTable(data.frame(Class = colnames(pred()$prob),
                                              Probability = round(as.numeric(pred()$prob), 3)))
  output$tbl <- renderDT(df, options = list(pageLength = 10, scrollX = TRUE))
}
shinyApp(ui, server)
