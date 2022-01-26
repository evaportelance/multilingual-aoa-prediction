##...................................
## Data prep -- scaling, imputation, etc
##...................................

# transform any column that starts with "count" by smoothing (add 1),
# normalizing, and log transforming, then rename "count_x" to "freq_x"
transform_counts <- function(childes_metrics, smooth = TRUE, normalize = TRUE,
                             log_transform = TRUE) {
  trans_metrics <- childes_metrics |> group_by(language)
  trans_funs <- c()
  if (smooth) trans_funs <- c(trans_funs, \(count) count + 1)
  if (normalize) trans_funs <- c(trans_funs, \(count) count / sum(count))
  if (log_transform) trans_funs <- c(trans_funs, \(count) log(count))

  for (fun in trans_funs) {
    trans_metrics <- trans_metrics |> mutate(across(starts_with("count"), fun))
  }

  trans_metrics |> ungroup() |>
    rename_with(\(col) str_replace(col, "count", "freq"), starts_with("count"))
}

residualize_col <- function(target_column, residualizing_column) {
  return(lm(target_column ~ residualizing_column)$residuals)
}

# residualize all columns that starts with "freq_" from the column "freq"
residualize_freqs <- function(childes_metrics) {
  childes_metrics |>
    mutate(across(starts_with("freq_"), partial(residualize_col, freq)))
}



## Imputation

fit_predictor <- function(pred, d, pred_sources) {
  xs <- pred_sources |> discard(\(s) pred %in% s) |> unlist()
  x_str <- xs |> paste(collapse = " + ")
  lm(as.formula(glue("{pred} ~ {x_str}")), data = d) |>
    broom::augment(newdata = d) |>
    select(uni_lemma, category, lexical_category, .fitted)
}

get_missing_data <- function(lang_data, predictors) {
  missing <- lang_data |>
    pivot_longer(cols = !!predictors, names_to = "predictor",
                 values_to = "value") |>
    mutate(missing = is.na(value)) |>
    select(-value) |>
    pivot_wider(names_from = predictor, values_from = missing)
  return(missing)
}

# start with predictors with the fewest NAs
# create a list that can be iterated over
get_predictor_order <- function(lang_data, predictors, max_steps) {
  predictor_order <- lang_data |>
    pivot_longer(cols = !!predictors, names_to = "predictor",
                 values_to = "value") |>
    group_by(predictor) |>
    summarise(num_na = sum(is.na(value))) |>
    filter(num_na != 0) |>
    arrange(num_na) |>
    pull(predictor)

  num_repeats <- max_steps %/% length(predictor_order) + 1
  return(rep(predictor_order, num_repeats)[0:max_steps])
}

get_imputation_seed <- function(lang_data, predictors) {
  imputation_data <- lang_data |>
    mutate_at(vars(!!predictors),
              funs(as.numeric(Hmisc::impute(., fun = "random"))))
  return(imputation_data)
}

# takes in a predictor from the list of predictors, pulls the imputation data at
# current status, returns a new imputation data

do_iterate_imputation <- function(pred_sources, imputation_data, missing) {
  prediction_list <- unlist(pred_sources)
  # iterates through the predictor list for that language
  for (pred in prediction_list) {
    imputation_fits <- fit_predictor(pred, imputation_data, pred_sources)
    imputation_data <- missing |>
      select(uni_lemma, lexical_category, category, !!pred) |>
      rename(missing = !!pred) |>
      right_join(imputation_data,
                 by = c("uni_lemma", "lexical_category", "category")) |>
      left_join(imputation_fits,
                by = c("uni_lemma", "lexical_category", "category")) |>
      # if the value is missing, replace it with the new value
      mutate_at(vars(pred), funs(if_else(is.na(missing), .fitted, .))) |>
      select(-.fitted, -missing)
  }
  return(imputation_data)
}


do_lang_imputation <- function(language, data, pred_sources, max_steps) {
  # if all the predictors are from one source, fix the pred_sources
  if (length(pred_sources) == 1) pred_sources <- unlist(pred_sources)
  predictors <- unlist(pred_sources)
  print(glue("Imputing {language} with {max_steps} steps..."))
  predictor_list <- get_predictor_order(data, predictors, max_steps)
  # what do we do if a language is missing a predictor entirely?
  missing_data <- get_missing_data(data, predictors)
  imputed_data <- get_imputation_seed(data, predictors)
  imputed_data <- do_iterate_imputation(pred_sources, imputed_data,
                                        missing_data)
  return(imputed_data)
}

do_full_imputation <- function(model_data, pred_sources, max_steps) {
  # restrict to the sources in pred_sources
  # catch cases where a predictor in the predictor set isn't in the data
  pred_sources <- map(pred_sources, \(ps) discard(ps, \(p) all(is.na(model_data[[p]]))))

  nested_data <- model_data |>
    select(language, uni_lemma, lexical_category, category,
           all_of(!!unlist(pred_sources))) |>
    distinct() |>
    group_by(language) |>
    nest()

  imputed_data <- nested_data |>
    mutate(imputed = map2(language, data, function(lang, dat) {
      do_lang_imputation(lang, dat, pred_sources, max_steps)
    }))

  imputed_data |>
    ungroup() |>
    select(language, imputed) |>
    unnest(imputed)
}


# scaling predictors
do_scaling <- function(model_data, predictors) {
  model_data |>
    group_by(language) |>
    mutate(across(all_of(predictors), \(x) as.numeric(scale(x))))
}


prep_lexcat <- function(predictor_data, uni_lemmas, ref_value) {
  lexical_categories <- uni_lemmas |>
    unnest(cols = "items") |>
   # filter(!lexical_class=="adverbs") |>
    distinct() |>
    # uni_lemmas with item in multiple different classes treated as "other"
    mutate(lexical_category = if_else(str_detect(lexical_class, ","), "other",
                                      lexical_class)) |>
    filter(!lexical_category=="other") |>
           # collapse v, adj, adv into one category
    mutate(lexical_category = lexical_category |> as_factor() |>
             fct_collapse("predicates" = c("verbs", "adjectives", "adverbs")) |>
             fct_relevel("nouns","predicates","function_words")) |>
    select(-lexical_class)
  lexical_categories$lexical_category <- relevel(lexical_categories$lexical_category, ref=ref_value)

  contrasts(lexical_categories$lexical_category) <- contr.sum

  predictor_data |>
    left_join(lexical_categories) |>
    filter(!is.na(uni_lemma))  |>
    filter(!is.na(lexical_category)) |>
    filter(!(language == "Russian" & category == "sounds"))
}




