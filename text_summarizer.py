
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

#Stop words mean words like for, is, to, the, etc. which are not important
#So stop words can be ignored from our passage along with punctuations

text= """Apple Inc. is an American multinational technology company headquartered in Cupertino, California, United States. 
Apple is the largest technology company by revenue (totaling US$365.8 billion in 2021) and, as of June 2022, Apple is the world's biggest company by market capitalization, the fourth-largest personal computer vendor by unit sales and second-largest mobile phone manufacturer. 
Apple is one of the Big Five American information technology companies, alongside Alphabet, Amazon, Meta, and Microsoft.
Apple is the largest technology company by revenue."""

def summarizer(rawText):
  stopwords=list(STOP_WORDS)        #Taking the stopwords into a list
  nlp=spacy.load('en_core_web_sm')
  doc=nlp(rawText)

  #Tokenising the text i.e taking all the words from text and putting them in list as tokens
  tokens=[token.text for token in doc]

  word_freq={}
  for word in doc:
    if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
      if word.text not in word_freq.keys():
        word_freq[word.text]=1
      else:
        word_freq[word.text]+=1

  max_freq=max(word_freq.values())

  #For normalising the word frequencies
  for word in word_freq.keys():
    word_freq[word]=word_freq[word]/max_freq

  #Tokenising the sentences in text
  sent_tokens=[sent for sent in doc.sents]

  sent_scores={}            #To set the sentence scores, for prioritising them
  for sent in sent_tokens:
    for word in sent:
      if word.text in word_freq.keys():
        if sent not in sent_scores.keys():
          sent_scores[sent]=word_freq[word.text]
        else:
          sent_scores[sent]+=word_freq[word.text]

  select_len= int(len(sent_tokens) * 0.3)     #To set the summary length (30%)

  #To get the summary with length as select_len from sent_scores :-
  summary=nlargest(select_len, sent_scores, key=sent_scores.get)       #nlargest will select top (select_len) no. of sentences from sent_scores 

  final_summary=[word.text for word in summary]
  summary=' '.join(final_summary)

  print("Length of the original text : ", len(rawText.split(' ')))
  print("Length of the summary : ", len(summary.split(' ')))
  return summary

summarizer(text)