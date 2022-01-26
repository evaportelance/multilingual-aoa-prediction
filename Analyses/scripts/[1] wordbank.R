get_inst_admins <- function(language, form, exclude_longitudinal = TRUE) {
  message(glue("Getting administrations for {language} {form}..."))

  admins <- get_administration_data(language = language,
                                    form = form,
                                    original_ids = TRUE)

   if (is.na(unique(admins$original_id[1]))==TRUE){
    admins <- admins %>%
      select(-(original_id)) %>%
      rename(original_id = data_id)
  }

  if (exclude_longitudinal) {
    # take earliest administration for any child with multiple administrations
    admins <- admins |>
      mutate(source_group = str_replace(source_name, " \\(.*\\)", "")) |>
      group_by(source_group, original_id) |>
      slice_min(age, with_ties = FALSE) |>
      ungroup()

  }
 if (!("data_id" %in% colnames(admins))){
   admins <- admins %>%
   rename(data_id = original_id)
 }

  admins |> select(language, form, age, data_id)
}

get_inst_words <- function(language, form) {
  message(glue("Getting words for {language} {form}..."))
  get_item_data(language = language, form = form) |>
    filter(type == "word") |>
    select(language, form, lexical_class, category, uni_lemma, definition,
           item_id, num_item_id)
}

get_inst_data <- function(language, form, admins, items) {
  message(glue("Getting data for {language} {form}..."))

  if (language !="Portuguese (European)"){
  get_instrument_data(language = language,
                      form = form,
                      items = items$item_id,
                      administrations = admins,
                      iteminfo = items) |>
    mutate(produces = !is.na(value) & value == "produces",
           understands = !is.na(value) &
             (value == "understands" | value == "produces")) |>
    select(-value) |>
    pivot_longer(names_to = "measure", values_to = "value",
                 cols = c(produces, understands)) |>
    filter(measure == "produces" | form == "WG") |>
    select(-num_item_id) |>
    filter(!is.na(uni_lemma))
  }else{
    unil <- read_csv("data/wordbank/portuguese_unilemmas.csv") %>%
      select(uni_lemma, definition, category)
    get_instrument_data(language = language,
                        form = form,
                        items = items$item_id,
                        administrations = admins,
                        iteminfo = items) |>
      mutate(produces = !is.na(value) & value == "produces",
             understands = !is.na(value) &
               (value == "understands" | value == "produces")) |>
      select(-value) |>
      pivot_longer(names_to = "measure", values_to = "value",
                   cols = c(produces, understands)) |>
      filter(measure == "produces" | form == "WG") |>
      select(-num_item_id, -uni_lemma) |>
      left_join(unil) |>
      filter(!is.na(uni_lemma))
  }
}

collapse_inst_data <- function(inst_data) {
  message(glue("Collapsing data for {unique(inst_data$language)} {unique(inst_data$form)}..."))

  inst_uni_lemmas <- inst_data |>
    distinct(measure, uni_lemma, lexical_class, category, item_id, definition) |>
    group_by(uni_lemma) |>
    nest(items = c(lexical_class, category, item_id, definition))

  inst_data |>
    filter(!is.na(value)) |>
    # for each child and uni_lemma, collapse across items
    group_by(language, form, measure, uni_lemma, age, data_id) |>
    summarise(uni_value = any(value)) |>
    # for each age and uni_lemma, collapse across children
    group_by(language, form, measure, uni_lemma, age) |>
    summarise(num_true = sum(uni_value), total = n()) |>
    ungroup() |>
    left_join(inst_uni_lemmas)

}

combine_form_data <- function(inst_summaries) {
  inst_combined <- bind_rows(inst_summaries)
  inst_combined |>
    unnest(items) |>
    nest(items = -c(language, measure, uni_lemma, age, num_true, total)) |>
    group_by(language, measure, uni_lemma, age) |>
    summarise(num_true = sum(num_true), total = sum(total),
              items = list(bind_rows(items))) |>
    ungroup()
}

create_inst_data <- function(language, form) {
  inst_admins <- get_inst_admins(language, form)
  inst_words <- get_inst_words(language, form)
  get_inst_data(language, form, inst_admins, inst_words)
}

create_wb_data <- function(language, write = TRUE) {
  lang <- language # for filter name scope issues
  insts <- get_instruments()
  forms <- insts |> filter(language == lang) |> pull(form)
  if (length(forms) == 0) {
    message(glue("\tNo instruments found for language {lang}, skipping."))
    return()
  }

  lang_datas <- map(forms, partial(create_inst_data, language = language))
  lang_summaries <- map(lang_datas, collapse_inst_data)
  lang_summary <- combine_form_data(lang_summaries)

  if (write) {
    lang_label <- normalize_language(language)
    saveRDS(lang_summary, file = glue("{wb_path}/{lang_label}.rds"))
  }
  return(lang_summary)
}

load_wb_data <- function(languages, cache = TRUE) {
  wb_data <- map_df(languages, function(lang) {
    norm_lang <- normalize_language(lang)
    lang_file <- glue("{wb_path}/{norm_lang}.rds")
    if (file.exists(lang_file)) {
      message(glue("Loading cached Wordbank data for {lang}..."))
      lang_data <- readRDS(lang_file)
    } else {
      if (cache) {
        message(glue("No cached Wordbank data for {lang}, getting and caching data."))
        lang_data <- create_wb_data(lang)
      } else {
        message(glue("No cached Wordbank data for {lang}, skipping."))
        lang_data <- tibble()
      }
    }
    return(lang_data)
  })
  return(wb_data)
}

extract_uni_lemmas <- function(wb_data) {
  wb_data |>
    filter(!is.na(uni_lemma)) |>
    distinct(language, uni_lemma, items) |>
    unnest(items) |>
    select(-c(form,item_id)) |>
    distinct() |>
    nest(items = -c(language, uni_lemma))
}
