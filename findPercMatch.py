from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
ps = PorterStemmer()

def get_word_set(text):
    return set(ps.stem(word) for word in word_tokenize(text) if word not in stop_words)


query = "engage whatin"

test = "engage the prognosis for survival"

def returnPercentageMatch(query,text):
    set_query = get_word_set(query)
    set_text = get_word_set(text)
    intersection = set_query & set_text
    return len(intersection)/len(set_query)


print(returnPercentageMatch(query,test))