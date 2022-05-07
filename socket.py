import socket
import string
import nltk
nltk.download("stopwords")
nltk.download('punkt')
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

stop_words = set(stopwords.words('english'))
stop_words = stop_words.union(string.punctuation)
s = socket.socket()
host = "127.0.0.1"
port = 9996
s.bind((host, port))
s.listen(5)
print('waiting for connections')

while True:
    c, addr = s.accept()
    print("Connected with ", addr)
    path = input("Enter file Path: ")
    with open(path) as txt:
        content = txt.read()
    content = content.lower()        
    content_tokens = word_tokenize(content)
    filtered_sentence = [w for w in content_tokens if not w in stop_words]
    content = " ".join(filtered_sentence)

    c.send(bytes(content, "utf-8"))
    c.close()
