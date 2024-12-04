
# Preprocess the descriptions
import re
from nltk.stem import WordNetLemmatizer
from read_data import *
from nltk.corpus import stopwords
import nltk
import warnings
warnings.filterwarnings("ignore")
stop_words = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
    'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
    'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
    'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
    'now', 'taking', 'delayed', 'suggest', 'get', 'getting', 'taking','hello','hi','pain','high','tightness','shortness'
    ,'remedy','hard','treated','child', 'pregnant','symptom','test','pill','chance','treatment','cause','persistant','lower','indicate'
    ,'leg','left','right','query','hope','hello','hi','dr','know','regards','answered','questions','understand',
    'consult','let','assist','revert','online','gone','ask','advice','years','old','day','like','having',
    'could','side','take','last','feel','months','days','ive','im','mg','th','red','avoid','week','ago','year',
    'time','need','thanks','help','care','answer','hussain','shina','question','solved','welcome','good','concern','doctor',
    'worry','son','kid','night','daily','daughter','feel','got','month','problem','started','went','said','want',
    'week','really','rule','ray','doubt','wish','happy','deari','magic','course','case','shina','using','use','chance','accordingly','persist',
    'better'
]
# stop_words = stopwords.words('english')

stop_words_pattern = r'\b(' + '|'.join(stop_words) + r')\b'

lemmatizer = WordNetLemmatizer()


def preprocess_text(df):
    df = df.drop_duplicates()
    def clean_text(desc):

        desc = desc.lower()

        desc = re.sub(r'^q\. ', '', desc)

        desc = re.sub(r'\?$', '', desc)

        desc = desc.strip()

        desc = re.sub(r'[^a-z0-9\s.,!?]', '', desc)

        desc = re.sub(r'\s+', ' ', desc)

        desc = re.sub(r'\d+', '', desc)

        desc = re.sub(r'<.*?>', '', desc)

        desc = re.sub(r'http\S+|www\S+', '', desc)

        desc = re.sub(stop_words_pattern, '', desc)

        desc = re.sub(r'\s+', ' ', desc).strip()
        # Lemmatize the words
        desc = ' '.join([lemmatizer.lemmatize(word) for word in desc.split()])
        return desc

    df['Description'] = df['Description'].apply(clean_text)
    df['Doctor'] = df['Doctor'].apply(clean_text)
    df['Patient'] = df['Patient'].apply(clean_text)
    return df

df = read_medical_chatbot_dataset()


df = preprocess_text(df)
# print(df.head())