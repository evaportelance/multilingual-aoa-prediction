Sys.setenv(DICPATH = "resources/dicts")

wb_path <- "data/wordbank"
childes_path <- "data/childes"

lang_map <- read_csv("resources/language_map.csv")

convert_lang_childes <- function(lang) {
  lang_map |> filter(wordbank == lang) |> pull(childes)
}

convert_lang_stemmer <- function(lang, method = "snowball") {
  lang_map |> filter(wordbank == lang) |> pull(!!method)
}

normalize_language <- function(language) {
  language |> str_replace(" ", "_") |> str_to_lower()
}
