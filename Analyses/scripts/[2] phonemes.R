lang_code_map <- list(
  "Croatian" = "hr",
  "Danish" = "da",
  "English (American)" = "en-us",
  "French (Quebec)" = "fr",
  "Italian" = "it",
  "Norwegian" = "no",
  "Russian" = "ru",
  "Spanish (Mexican)" = "es",
  "Swedish" = "sv",
  "Turkish" = "tr"
)

get_ipa <- function(word, lang) {
  lang_code <- lang_code_map[[lang]]
  system2("espeak", args = c("--ipa=3", "-v", lang_code, "-q", paste0('"', word, '"')), 
          stdout=TRUE) %>%
    gsub("^ ", "", .) %>%
    gsub("[ˈˌ]", "", .)
}

get_phons <- function(words, lang) {
  words %>% map_chr(function(word) get_ipa(word, lang))
}

num_phons <- function(phon_words) {
  phon_words %>% map_dbl(function(phon_word) {
    phon_word %>%
      map_dbl(~.x %>% str_replace("r", "_r") %>%
                str_replace("l", "_l") %>%#add _ before r and l?
                str_split("[_ \\-]+") %>% unlist() %>%
                keep(nchar(.) > 0 & !grepl("\\(.*\\)", .x)) %>% length()) %>%
      mean()
  })
}

str_phons <- function(phon_words) {
  phon_words %>% map(function(phon_word) {
    phon_word %>%
      map_chr(~.x %>% str_replace("r", "_r") %>%
                str_replace("l", "_l") %>% 
                str_replace("ɹ", "_ɹ") %>% str_split("[_ \\-]+") %>% unlist() %>%
                keep(nchar(.) > 0 & !grepl("\\(.*\\)", .x)) %>%
                paste(collapse = ""))
  })
}

num_chars <- function(words) {
  map_dbl(words, ~gsub("[[:punct:]]", "", .x) %>% nchar() %>% mean())
}

#some predictors are sensitive to the word, not the uni-lemma. Eg pronounciation
#For these cases, we get the predictor by word and then average by uni-lemma (eg a vs an)

# https://github.com/mikabr/aoa-prediction/blob/67764a7a4dfdd743278b8a56d042d25723dbdec7/aoa_unified/aoa_loading/aoa_loading.Rmd#L339

# clean_words(c("dog", "dog / cat", "dog (animal)", "(a) dog", "dog*", "dog(go)", "(a)dog", " dog ", "Cat"))
clean_words <- function(word_set){
  word_set %>%
    # dog / doggo -> c("dog", "doggo")
    strsplit("/") %>% flatten_chr() %>%
    # dog (animal) | (a) dog
    strsplit(" \\(.*\\)|\\(.*\\) ") %>% flatten_chr() %>%
    # dog* | dog? | dog! | ¡dog! | dog's
    gsub("[*?!¡']", "", .) %>%
    # dog(go) | (a)dog
    map_if(
      # if "dog(go)"
      ~grepl("\\(.*\\)", .x),
      # replace with "dog" and "doggo"
      ~c(sub("\\(.*\\)", "", .x),
         sub("(.*)\\((.*)\\)", "\\1\\2", .x))
    ) %>%
    flatten_chr() %>%
    # trim
    gsub("^ +| +$", "", .) %>%
    keep(nchar(.) > 0) %>%
    tolower() %>%
    unique()
}

map_phonemes <- function(uni_lemmas) {
  #TODO: add bug check for a language missing from the language map
  #some words eg "cugino/a" need to be mapped to "cugino / cugina"
  fixed_words <- read_csv("data/predictors/phonemes/fixed_words.csv") %>%
    select(language, uni_lemma, definition, fixed_word) %>%
    filter(!is.na(uni_lemma), !is.na(fixed_word))
  
  uni_cleaned <- uni_lemmas %>%
    unnest(cols = "items") %>%
    # distinct(language, uni_lemma, definition) %>%
    left_join(fixed_words) %>%
    mutate(fixed_definition = ifelse(is.na(fixed_word), definition, fixed_word),
           cleaned_words = map(fixed_definition, clean_words)) %>%
    select(-fixed_word) %>%
    group_by(language) %>%
    #for each language, get the phonemes for each word 
    mutate(phons = map2(cleaned_words, language, ~get_phons(.x, .y)))
  
  fixed_phons <- read_csv("data/predictors/phonemes/fixed_phons.csv") %>%
    select(language, uni_lemma, definition, fixed_phon) %>%
    filter(!is.na(uni_lemma), !is.na(fixed_phon)) %>%
    mutate(fixed_phon = strsplit(fixed_phon, ", "))
  
  uni_phons_fixed <- uni_cleaned %>% 
    left_join(fixed_phons) %>%
    mutate(phons = if_else(map_lgl(fixed_phon, is.null), phons, fixed_phon),
           str_phons = str_phons(phons)) %>%
    select(-fixed_phon)
  
  # get lengths
  uni_lengths <- uni_phons_fixed %>% mutate(num_char = num_chars(cleaned_words),
                                            num_phon = num_phons(phons)) %>%
    group_by(language, uni_lemma) %>%
    summarize(num_chars = mean(num_char),
              num_phons = mean(num_phon))
  return(uni_lengths)
}
