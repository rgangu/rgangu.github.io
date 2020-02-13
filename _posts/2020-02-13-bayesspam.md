---
title: "Spam or Ham?"
date: 2019-12-25
categories: bayesian ml spam-detection classification naive-bayes
comments: true
---

This is a common, well-versed concept that I am re-exploring at this point, to reiterate some of the key concepts that surround it. 

### Kinda a bit of a TL:DR, but not really

 Spam is a major concern in today’s communication platforms, whether it be email, text messaging, or LinkedIn. Many modern mail clients have and continue to utilize Bayesian spam filtering techniques to screen out spam for the convenience of their users. Naive Bayes classifiers, a family of simple probabilistic classifiers based on applying Bayes’ theorem with strong independence assumptions between features, are utilized here in an effort to build out a probabilistic model in a supervised learning setting. The main paradigm used in terms of the feature set involves ‘bag of words’ features, a common representation used in natural language processing and information retrieval. The words in a dataset of text messages, already labeled, are used in terms of both correlation for feature construction, and then Bayes’ theorem will be applied to calculate the probabilities of a message being considered spam or not spam. Particularly, a multinomial event model is used in this case, with frequencies and document lengths being used as well.

### Introduction

By the year 1996, Bayesian algorithms were utilized in order to sort and filter email. Native Bayesian filters did not become popular until a later period of time, but multiple programs were built and released in 1998 in order to deal with the emerging issue of unwanted emails. Later on, these programs were released in a commercial context, in spam filters. In 2002, a American computer scientist by the name of Paul Graham worked on an approach where the false positive rate for detecting spam was greatly decreased – therefore, from that point on, naive Bayesian filters could be used by themselves as the sole spam filter in an email service. Today, many modern email clients and software utilize Bayesian spam filtering, and users can independently install them as well. This article explores the usage of naive Bayesian filters to detect spam, particularly in the context of text messaging. 

### Background/Theory You Maybe Should Know

To understand the context behind the design and implementation, we must start with discussing and introducing a few key concepts needed for the spam detection.

First, let us start with a corpus of words we will call ‘X’. Within this corpus ‘X’,  we look at the subsections of this corpus as ‘documents’. The corpus used in this particular article, shown later on, is a dataset of text messages. Each message within this dataset of text messages is represented as the bag of its words – we disregard the premises of grammar and word order, but we keep the frequency of words used. The frequency of words used in each document, within our large corpus of text messages, is an essential feature utilized later on for training our Bayesian classifier. Actually, the bag-of-words model is quite commonly used in document classification in classification as an engineered feature. After we transform our text data into a ‘bag of words’, we are able to compute a number of features to help us further characterize and describe the text. Particularly, we will actually normalize the term frequencies within a particular sentence – this is because some words, known as stopwords, such as articles in English or similar common words, could deviate the truth of what is actually considered an ‘important’ word in a document.

One popular way to normalize term frequencies is using a measure known as term frequency-inverse document frequency, or TF-IDF for short. This is a numerical measure that increases in proportion to the number of times that a particular word shows up in a document, but is additively adjusted for the fact that some words appear more frequently in a general context (such as ‘the’). The two components are term frequency and inverse-document frequency. ‘TF’ can be computed by dividing the number of times a particular term t appears in a document divided by the total number of terms in the given document. ‘IDF’ can be computed by finding the log of the division of the total number of documents and the number of documents with the previously mentioned term t in them. TF-IDF is then the product of TF and IDF.  

Now, how is this bag-of-words model used in the spam filtering?  A particular text message will be modeled as an unordered collection of words – labeled as a probability distribution representing ‘spam’ or ‘ham’ - which is legitimate messages. When classifying the message, the Bayesian spam filter would then use Bayes’ Theorem to determine which bag of words (the spam one or the ham one) that a message is more likely to be belonging to. 

Bayesian classifiers use Bayes’ theorem -  a popular mathematical formula that describes the probability of an event e based on the prior knowledge of conditions related to event e. 

It can be stated mathematically using the equation `P(A|B) = (P(B|A)P(A))/(P(B))`, where `P(A|B)` is the chance of A occurring given B is true, `P(B|A)` is the chance of B occurring given A to be true, and `P(A)` and `P(B)` being the chances of observing A and B respectively.  `P(A)` is the prior probability, and `P(A|B)` is the posterior probability.  `P(B|A)` can be called the likelihood, and `P(B)` can be called the evidence.  Now – when applying this to a classification context – we first begin by finding the probability of given set of inputs for all possible values of a class, and then use the output that has maximum probability. This tells us the classification. 

The Naive Bayes classifier used in this article, known as the multinomial Naive Bayes classifier, uses features that represent frequencies from which events have been generated by a multinomial distribution – this is most used for text classification like in this use case. 


### Design and Implementation

Firstly, let’s start with sourcing the data. The data used for this article is the 'SMS Spam Collection v.1' - which is a public set of text messages collected for spam research, with each message labeled as 'spam' or 'ham'.
It is available in the UCI Machine Learning Repository.

We start by importing some needed libraries, then take in data in the form of a flat text file. We iteratively loop through the text file, and then reformat it into a way that it can be usable with the pandas library – then we write to a .csv file. At this point, we have two columns in our new comma-separated values file – SMS, which is the text contained in a message, and ‘SPAM’, which is a binary classification of whether the aforementioned message is considered to be ‘spam’ or ‘ham’.  We perform some text preprocessing needed – specifically focusing on creating a new feature called ‘length’ to later be used in our classification model. We begin to define the Naive Bayes classifier.

Within this, we start by splitting the data into training and validation subsets, and then calculate the prior probabilities of the messages being spam, the conditional independence, as well as the probabilities of the messages being ham (both classes of the target variable ‘SPAM’). We then establish a count of the number of terms that are spam as well as ham – then looking as well at the frequencies of each word in the corpus as a whole. For each text message, if the probability of it being spam is higher than it being ‘ham’ (not spam), then it is classified as such. 

To verify the performance of the multinomial naive Bayes classifier, validation is performed on the batch – to assess the accuracy and view the resultant confusion matrix. 

The most common spam and ham words are then computed for the viewing convenience, and the model is saved. Then, the Naive Bayes classification model gets trained and tested accordingly. The key idea here is that by introducing the features of word frequency and text length, we add two new powerful features that improve performance of detecting if a message is spam or not so. 


### Results

For the Naive Bayes classifier,  the final model used was a multinomial Naive Bayes classifier with length of the document (text message) and frequency of words used factored in as features as well. We assess three different results here, in terms of train-test splits. Firstly, when we split the training and testing sets in an unconventional 50/50 nature, we get the following metrics when assessing the performance: the accuracy was 96.2%, and that 2,680 out of 2,786 predictions were correct in terms of classifying if a given text message was ‘spam’ or ‘ham’.

Next, we perform a more conventional 80/20 train-test split. 

We see that the accuracy was improved, and is now 96.77% - with 1,079 out of 1,115 predictions being correct. Lastly, we look at this from an evidential learning perspective – where we add testing data to the training subset, and then re-train and re-validate. What we see in the results is interesting.

The accuracy dramatically improves, and 5,508 out of 5,572 predictions were correct. 

### Conclusion

Using a multinomial Naive Bayes classifier, we were able to predict whether a given document (in this case, a text message) was spam or not spam, to a high degree of accuracy. We utilized the bag-of-words model to be able to extract the features of frequency and document length to supplement the labels provided via our dataset about the binary classification of the text message. With using a specific version of Naive Bayes, the multinomial model, we assumed a multinomial distribution for each of our features. We calculated prior probabilities and likelihood for each of our observations, and derived posterior probabilities from which the maximum probability was used to make the determination of the spam/ham classification. What is interesting is how that despite Naive Bayes makes the assumption of conditional independence  - something that is hardly ever true – we derived a very high prediction accuracy. Ultimately, multinomial Naive Bayes was used, because it explicitly models the word counts and adjusts the underlying calculations to deal with them. Today’s mail clients and communication platforms rely on these Bayesian algorithms to filter out irrelevant content for their users – and often to a high level of success as well. 

*The full code can be found at https://github.com/rgangu/cs445/blob/master/Identifying%20Spam%20in%20Texts%20using%20Naive%20Bayes%20Classification%20-%20%20Rohit%20Gangupantulu%20-%20CMPSC%20445.ipynb.*

Thank you, and sorry for the long read! Feel free to comment.
