# kxx_tidy_glm_text2vec.R
# 2018 Feb 19

# install.packages("stringr")
# install.packages("tokenizers")

library(stringr)
library(magrittr)
library(dplyr)
library(text2vec)

library(tidyverse)
library(magrittr)
library(text2vec)
library(tokenizers)
library(glmnet)
library(doParallel)
registerDoParallel(4)


dir="C:/__KAGGLE__/ToxicCommentClassification/kxx_tidy_glm_text2vec"
setwd(dir)

train <- read_csv("../data_2018_02_16/train.csv/train.csv") 
test <- read_csv("../data_2018_02_16/test.csv/test.csv") 
subm <- read_csv("../data_2018_02_16/sample_submission.csv/sample_submission.csv") 

tri <- 1:nrow(train)
targets <- c("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")

#---------------------------
cat("Basic preprocessing & stats...\n")
tr_te <- train %>% 
  select(-one_of(targets)) %>% 
  bind_rows(test) %>% 
  mutate(length = str_length(comment_text),
         ncap = str_count(comment_text, "[A-Z]"),
         ncap_len = ncap / length,
         nexcl = str_count(comment_text, fixed("!")),
         nquest = str_count(comment_text, fixed("?")),
         npunct = str_count(comment_text, "[[:punct:]]"),
         nword = str_count(comment_text, "\\w+"),
         nsymb = str_count(comment_text, "&|@|#|\\$|%|\\*|\\^"),
         nsmile = str_count(comment_text, "((?::|;|=)(?:-)?(?:\\)|D|P))")) %>% 
  select(-id) %T>% 
  glimpse()

#---------------------------
cat("Parsing comments...\n")
it <- tr_te %$%
  str_to_lower(comment_text) %>%
  str_replace_all("[^[:alpha:]]", " ") %>%
  str_replace_all("\\s+", " ") %>%
  itoken(tokenizer = tokenize_word_stems)

vectorizer <- create_vocabulary(it, ngram = c(1, 1), stopwords = stopwords("en")) %>%
  prune_vocabulary(term_count_min = 3, doc_proportion_max = 0.3, vocab_term_max = 4000) %>%
  vocab_vectorizer()

dtm <- create_dtm(it, vectorizer) %>% 
  normalize(norm = "l2")

#---------------------------
cat("Preparing data for glmnet...\n")
X <- tr_te %>% 
  select(-comment_text) %>% 
  sparse.model.matrix(~ . - 1, .) %>% 
  cbind(dtm)

X_test <- X[-tri, ]
X <- X[tri, ]

rm(tr_te, test, tri, it, vectorizer, dtm); gc()

#---------------------------
cat("Training glmnet & predicting...\n")
for (target in targets) {
  cat("\nFitting", target, "...\n")
  y <- factor(train[[target]])
  m_glm <- cv.glmnet(X, y, alpha = 0, family = "binomial", type.measure = "auc",
                     parallel = T, standardize = T, nfolds = 4, nlambda = 50)
  cat("\tAUC:", max(m_glm$cvm))
  subm[[target]] <- predict(m_glm, X_test, type = "response", s = "lambda.min")
}

#---------------------------
cat("Creating submission file...\n")
write_csv(subm, "subm_kxx_tidy_glm_text2vec.csv")
