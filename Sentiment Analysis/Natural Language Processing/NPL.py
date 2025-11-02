from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import nltk

# Download required resources
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# Input text
text = 'I am not a sentimential person but I believe in the ultility of sentiment analysis'

# Tokenization
tokenizer = TreebankWordTokenizer()
tokens = tokenizer.tokenize(text)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word) for word in tokens]

# Stemming (lowercase before stemming)
stemmer = PorterStemmer()
stemmed = [stemmer.stem(word.lower()) for word in tokens]

# Stop word removal
stop_words = set(stopwords.words('english'))
filtered = [word for word in stemmed if word not in stop_words]

# Output
print('Tokens:', tokens)
print('Lemmatized:', lemmatized)
print('Stemmed:', stemmed)
print('Filtered:', filtered)
