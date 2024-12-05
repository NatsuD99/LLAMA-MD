import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from read_data import read_medical_chatbot_dataset
from preprocessing import preprocess_text
from matplotlib import pyplot as plt
from preprocessing import stop_words, stop_words_pattern
import warnings
warnings.filterwarnings("ignore")
topics_df = pd.DataFrame(index=['Description', 'Doctor', 'Patient'], columns=[f'Topic {i + 1}' for i in range(10)])
def lsa_topic_modelling(descriptions,column_name):
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(descriptions)
    # print("No.of topic = Length of TFIDF components")
    # LSA (Latent Semantic Analysis)
    lsa = TruncatedSVD(n_components=10, n_iter=100, random_state=42)
    lsa_topics = lsa.fit_transform(tfidf_matrix)
    terms = tfidf_vectorizer.get_feature_names_out()
    print("\nLSA Topics: ")
    # print("Components: ", lsa.components_)

    for i, comp in enumerate(lsa.components_):
        termsInComp = zip(terms, comp)
        sortedterms = sorted(termsInComp, key=lambda x: x[1], reverse=True)[:10]
        print("Topics %d:" % i)

        for term in sortedterms:
            print(term[0])
        # Visualize top words for each topic
        topic_terms = [term[0] for term in sortedterms]
        importance = [term[1] for term in sortedterms]
        topics_df.loc[column_name, f'Topic {i + 1}'] = topic_terms
        plt.figure()
        plt.barh(topic_terms, importance)
        plt.xlabel("Importance")
        plt.title(f"LSA Topic {i + 1} for '{column_name}'")
        plt.gca().invert_yaxis()
        plt.show()
        print(" ")


# read
df = read_medical_chatbot_dataset()

# Preprocess
df = preprocess_text(df)

# Perform LSA
for column_name in ['Description', 'Doctor', 'Patient']:
    lsa_topic_modelling(df[column_name].tolist(), column_name)

topics_df.to_excel('lsa_topics.xlsx', index=True)
# print(topics_df)