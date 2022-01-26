
# Gets stems for a list of words in a given language.
# Uses Snowball by default
# Also allows for hunspell as an alternative method

stem <- function(words, language) {
  if (language != "Croatian" & language!= "Mandarin (Beijing)" & language!= "Mandarin (Taiwanese)"){
  lang_snowball <- convert_lang_stemmer(language, "snowball")
  lang_hunspell <- convert_lang_stemmer(language, "hunspell")
  if (lang_snowball %in% SnowballC::getStemLanguages()) {
    SnowballC::wordStem(words, lang_snowball)
  } else if (lang_hunspell %in% hunspell::list_dictionaries()) {
    map_chr(words, \(word) {
      stem <- hunspell::hunspell_stem(word, dictionary(lang_hunspell))[[1]]
      return(if (length(stem) == 0) word else stem[1])
    })
  }} else if (language == "Croatian"){
    chunk_size <- 1000
    word_chunks <- split(words, ceiling(seq_along(words) / chunk_size))
    map(word_chunks, function(word_chunk) {
      system2("python",
              args = c("scripts/croatian_stemmer/croatian.py", sprintf('"%s"', word_chunk)),
              stdout = TRUE)
    }) |> unlist()}
else if (language == "Mandarin (Beijing)"){
    return(words)
}
else if (language == "Mandarin (Taiwanese)"){
    return(words)
}
}
