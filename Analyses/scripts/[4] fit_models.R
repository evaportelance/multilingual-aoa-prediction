# CDI -> AoA models

fit_bglm <- function(df, max_steps = 200) {
  model <- arm::bayesglm(cbind(num_true, num_false) ~ age,
                         family = "binomial",
                         prior.mean = .3,
                         prior.scale = c(.01),
                         prior.mean.for.intercept = 0,
                         prior.scale.for.intercept = 2.5,
                         prior.df = 1,
                         data = df,
                         maxit = max_steps)
  intercept <- model$coefficients[["(Intercept)"]]
  slope <- model$coefficients[["age"]]
  tibble(intercept = intercept, slope = slope, aoa = -intercept / slope)
}

fit_aoas <- function(wb_data, max_steps = 200, min_aoa = 0, max_aoa = 72) {
  aoas <- wb_data |>
    mutate(num_false = total - num_true) |>
    nest(data = -c(language, measure, uni_lemma)) |>
    mutate(aoas = map(data, fit_bglm)) |>
    select(-data) |>
    unnest(aoas) |>
    filter(aoa >= min_aoa, aoa <= max_aoa)
}

# Word predictor models

make_predictor_formula <- function(predictors, lexcat_interactions = TRUE) {
  if (lexcat_interactions) {
    effs_lex <- paste(predictors, "lexical_category", sep = " * ")
    glue("aoa ~ {paste(effs_lex, collapse = ' + ')}") |> as.formula()
  } else {
    glue("aoa ~ {paste(predictors, collapse = ' + ')}") |> as.formula()
  }
}

fit_group_model <- function(predictors, group_data, lexcat_interactions = TRUE,
                            model_formula = NULL) {
  # discard predictors that data has no values for
  #predictors <- predictors |> discard(\(p) all(is.na(group_data[[p]])))
  if (is.null(model_formula)) {
    model_formula <- make_predictor_formula(predictors, lexcat_interactions)
  }
  lm(model_formula, group_data)
}

get_vifs <- function(model) {
  vif <- car::vif(model)
  as.data.frame(vif) |> rownames_to_column("predictor") |> rename_with(tolower)
}

fit_models <- function(predictors, predictor_data, lexcat_interactions = TRUE,
                       model_formula = NULL) {
  predictor_data |>
    nest(group_data = -c(language, measure)) |>
    mutate(model = group_data |>
             map(\(gd) fit_group_model(predictors, gd, lexcat_interactions,
                                       model_formula)),
           coefs = map(model, broom::tidy),
           stats = map(model, broom::glance),
           vifs = map(model, get_vifs))
}
