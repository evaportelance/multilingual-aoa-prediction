pred_dir <- "data/predictors"

# most predictors are attached to unilemmas/underlying concepts
# (e.g. concreteness) and can be generalized across languages
map_predictor <- function(uni_lemmas, predictor, variable_mapping) {

  # takes in a csv of predictor measures that may or may not map to
  # uni_lemmas and maps them
  predictor_data <- read_csv(glue("{pred_dir}/{predictor}.csv"))
  replacements <- read_csv(glue("{pred_dir}/{predictor}_replace.csv"))
  predictors <- discard(names(variable_mapping), \(v) v == "word")

  # clean the predictors
  renamed_predictors <- predictor_data |>
    rename(all_of(variable_mapping)) |>
    select(names(variable_mapping)) |>
    group_by(word) |>
    # in case a word appears in the list twice
    summarize(across({{ predictors }}, mean))

  # TODO: What do we do about things like "chips" and "can (auxiliary)"
  # chips doesnt match to "chip" and "can" gets the measures for "can (object)"
  uni_lemma_predictors <- uni_lemmas |> left_join(replacements) |>
    # if there is a hand written replacement, use the replacement
    # otherwise default to the cleaned uni_lemma
    mutate(word = case_when(
      !is.na(replacement) & replacement != "" ~ replacement,
      TRUE ~ str_replace(uni_lemma, "\\s*\\([^\\)]+\\)", ""))
    ) |>
    select(-replacement) |>
    left_join(renamed_predictors) |>
    select(-word) |>
    group_by(language, uni_lemma) |>
    summarize(across({{ predictors }}, mean)) |>
    ungroup()

  return(uni_lemma_predictors)
}

# Example -----
# combined_data <- rbind(readRDS("data/wordbank/croatian.rds"),
#                        readRDS("data/wordbank/english_(american).rds"),
#                        readRDS("data/wordbank/spanish_(mexican).rds"),
#                        readRDS("data/wordbank/russian.rds"))
#
# babiness_csv <- "data/predictors/babiness/babiness.csv"
# babiness_replace_csv <- "data/predictors/babiness/babiness_replace.csv"
# babiness_map <- c(word = "word", babiness = "babyAVG")
#
# baby_unilemma <- map_predictor(combined_data, babiness_csv, babiness_replace_csv, babiness_map)
#
# valence_csv <- "data/predictors/valence/valence.csv"
# valence_replace_csv <- "data/predictors/valence/valence_replace.csv"
# valence_mapping <- c(word = "Word", valence = "V.Mean.Sum", arousal = "A.Mean.Sum")
#
# valence_unilemma <- map_predictor(combined_data, valence_csv, valence_replace_csv, valence_mapping)
#
# concreteness_csv <- "data/predictors/concreteness/concreteness.csv"
# concreteness_replace_csv <- "data/predictors/concreteness/concreteness_replace.csv"
# concreteness_map <- c(word = "Word", concreteness = "Conc.M")
#
# conctreteness_unilemma <- map_predictor(combined_data, concreteness_csv,
#                                         concreteness_replace_csv, concreteness_map)
# #TODO: Need to download the correct Russian dictionary http://espeak.sourceforge.net/data/
# lang_code_map <- list(
#   "Croatian" = "hr",
#   "Danish" = "da",
#   "English (American)" = "en-us",
#   "French (Quebec)" = "fr",
#   "Italian" = "it",
#   "Norwegian" = "no",
#   "Russian" = "ru",
#   "Spanish (Mexican)" = "es",
#   "Swedish" = "sv",
#   "Turkish" = "tr"
# )
#
# phonemes_unilemma <- map_phonemes(combined_data, lang_code_map)
#
# #something funky is happening with english arms - 2 phonemes instead of 3 because splitting at the underscore?
#
# combined_joined <- combined_data |> left_join(phonemes_unilemma) |> left_join(baby_unilemma) |> left_join(valence_unilemma) |> left_join(conctreteness_unilemma)
#
# save(combined_joined, file = "data/temp_saved_data/new_pipeline_uni_joined.rds")
