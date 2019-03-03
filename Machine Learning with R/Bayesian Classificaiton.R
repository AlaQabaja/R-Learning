# Say we are having an email with four words feature vector. W1, W2, W3, W4. 
# An email with word W1, W2 comes and we need to know if it is spam 
#P(spam|W1 and W2 and negate W3 and negate W4) = P(W1 and W2 and negate W3 and negate W4 | spam) * P(spam)/P(W1 and W2 and negate W3 and negate W4)
# Since we assume indepdendent events, independent features (class conditional independence, events are independent so long as they are conditioned on the same class value)
# Because the denominator does not depend on the class, it is reated as a constant value and can be ignored for now 
# P(spam|W1 and W2 and W3 and W4) (likelyhood of spam) = P(W1|spam) * P(W2|spam) * P(W3|spam) * P(W4|spam) * P(spam)
# P(ham |W1 and W2 ....) = same as above but with ham
# final probability of spam = P(spam from above) / (P(spam) + P(ham))
# In other words, P(class level (spam) | Features) = P(Cl) * Product(P(Featurei | C))
# If a word never appears before, it will set the probability for spam to zero. This doesn't make sense
# This can be resolved by using the Laplace estimator to add a constant to each value in the frequency table
# for numeric features, use bins, that's put the numeric feature into pockets 
library(tm)
library(SnowballC)
library(e1071)
library(gmodels)
install.packages("e1071")
setwd("C:\\Development\\Machine Learning with R\\datasets")
sms_raw = read.csv("sms_spam.csv", stringsAsFactors = FALSE)
sms_raw$type = as.factor(sms_raw$type)
sms_corpus <- VCorpus(VectorSource(sms_raw$text))
print(sms_corpus)
inspect(sms_corpus[1:2])
as.character(sms_corpus[[1]])
lapply(sms_corpus[1:5], as.character)
sms_corpus_clean <- tm_map(sms_corpus, content_transformer(tolower))
lapply(sms_corpus_clean[1:5], as.character)
sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers)
sms_corpus_clean <- tm_map(sms_corpus_clean, removeWords, stopwords())
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation)
sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)
sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)
sms_dtm_train <- sms_dtm[1:4169,]
sms_dtm_test <- sms_dtm[4170:5559,]
sms_train_labels <- sms_raw[1:4169,]$type
sms_test_labels <- sms_raw[4170:5559,]$type
prop.table(table(sms_train_labels))
prop.table(table(sms_test_labels))
sms_freq_words <- findFreqTerms(sms_dtm_train, 5)

sms_dtm_train <- sms_dtm_train[,sms_freq_words]
sms_dtm_test <- sms_dtm_test[,sms_freq_words]

# Naive bays doesn't take numeric vectors, rather, a character based data 
convert_counts <- function(x){x = ifelse(x > 0 , "Yes", "No")}

sms_train <- apply(sms_dtm_train, 2, convert_counts)
sms_test <- apply(sms_dtm_test, 2, convert_counts)

# Build the model 
sms_classifier <- naiveBayes(sms_train, sms_train_labels, laplace = 1)
sms_Test_pred <- predict(sms_classifier, sms_test)
CrossTable(sms_Test_pred, sms_test_labels, prop.chisq = FALSE, prop.t = FALSE, dnn = c("predicted","actual"))
