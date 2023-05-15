# Phishing Detection over HTTP Traffic

With the rise of the internet and technology, fraudulent attacks have become the norm of today’s society, commonly, in the form of malicious URLs. These attacks have become very common in the form of malicious URLs. A malicious URL is a link created with the purpose of promoting scams, attacks, and frauds. By clicking on an infected URL, you can download ransomware, virus, trojan, or any other type of malware that will compromise your machine. Hence, the identification of these malicious URLs has become a very significant task in everyday life


Our proposal is to design a Phishing Attack detection system over HTTP traffic based on machine learning and deep learning. HTTP Traffic data consists of hundreds or thousands of features and usually these features contain many redundant features which will increase the complexity. To overcome this, we converted the HTTP traffic data into numerical vectors with the help of Word2Vec and implemented Term Frequency-Inverse Document Frequency (TF-IDF) to generate low dimensional weighted paragraph vectors to reduce the training complexity. Next, we employed oversampling techniques – Synthetic Minority Over Sampling Techniques (SMOTE) and Adaptive Synthetic Sampling Approach (ADASYN) to handle the imbalanced dataset. Then, we applied various classification models such as Logistic Regression, KMeans classifier, Cat Boosting classifier and 1D Convolutional Neural Networks (CNN). 


The above process was done on an artificial dataset, Malicious-URLs and tested using tool-generated HTTP Traffic data. Experimental results have shown us that balancing the dataset using SMOTE gives better FPR values and better accuracy was achieved on SMOTE-balanced dataset with accuracy 94.6%.
