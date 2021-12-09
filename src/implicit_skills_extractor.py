import implicit_skill_finder
import utils
import pandas as pd
import numpy as np
from joblib import dump, load


def show_topics(vectorizer, lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords


text = ['Work on complex and extremely varied data sets from some of the world’s largest organisations to solve real world problems Develop data science products and solutions for clients as well as for our data science team Write highly optimized code to advance our internal Data Science Toolbox Work in a multi-disciplinary environment with specialists in machine learning, engineering and design Focus on modelling by working alongside the Data Engineering team Add real-world impact to your academic expertise, as you are encouraged to write papers and present at meetings and conferences should you wish Take part in R&D (video: R&D at QuantumBlack); attend conferences such as NIPS and ICML as well as data science retrospectives where you will have the opportunity to share and learn from your co-workers Work in one of the most advanced data science teams globally']
doc_ids, docs = implicit_skill_finder.find_similar_texts(text)
best_model = load(utils.MODEL_DIR)
tf_vectorizer = load(utils.VECTOR_DIR)
lda_output = load(utils.OUTPUT_DIR)
topic_keywords = show_topics(vectorizer=tf_vectorizer, lda_model=best_model, n_words=15)        
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]

#topicnames = ["Topic" + str(i) for i in range(best_model.n_components)]
#docnames = ["Doc" + str(i) for i in range(len(lda_output))]
#df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

df_document_topic = pd.DataFrame(np.round(lda_output, 2))
topic_pos = (-df_document_topic.values).argsort()
implicit_skills = set()
for i in doc_ids:
    for j in range(len(df_topic_keywords)):
        if topic_pos[i, j] > 0.3:
            implicit_skills.update(df_topic_keywords.iloc[j, :].apply(lambda x: x.strip()).to_list())

print(implicit_skills)

#dominant_topic = np.argmax(df_document_topic.values, axis=1)
#print(dominant_topic)
#for i in dominant_topic:
    #print(df_document_topic.iloc[i, dominant_topic[i]])


#pd.set_option("display.max_rows", None, "display.max_columns", None)
