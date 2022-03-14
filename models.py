import pandas as pd
import re
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import balanced_accuracy_score
import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments



#Loading the data file and creation of a two list, one containing the labels and one the input
data = pd.read_csv("sample_data.csv")
text = data["text"].tolist()
label = data["label"].tolist()

#Preprocessing
def umlauts(text):
    text = text.replace('ä', 'ae')
    text = text.replace('ö', 'oe')
    text = text.replace('ü', 'ue')
    text = text.replace('Ä', 'Ae')
    text = text.replace('Ö', 'Oe')
    text = text.replace('Ü', 'Ue')
    text = text.replace('ß', 'ss')
    return text

for i in range(len(text)):
    #Remove numbers
    text[i] = re.sub(r'\d+','', text[i])
    #Replace umlauts
    text[i] = umlauts(text[i])
    #Remove non english characters
    text[i] = re.sub("([^\x00-\x7F])+",'',text[i])
    #Convert to lower case
    text[i]= text[i].lower()
    #Remove Punctuation
    text[i] = text[i].translate(str.maketrans('','', string.punctuation))
    #Remove white spaces
    text[i] = text[i].strip()

#Remove empty entries
text_new = []
label_new = []
for w,l in zip(text,label):
    if w != " " and w != '' and type(l) == str:
        text_new.append(w)
        label_new.append(l)
text = text_new
label = label_new

#Converting classes into integers
classes = []
for class_ in label:
    if class_ not in classes:
        classes.append(class_)

for i in range(len(label)):
    label[i] = classes.index(label[i])

#Train-Test-Split
text_train, text_test, y_train, y_test = train_test_split(text, label, test_size = 0.2)

#Model 1
#Feature Extraction
vec = TfidfVectorizer(max_features=10000)
fit = vec.fit(text_train)
features_train = fit.transform(text_train)
features_test = fit.transform(text_test)

#Train SVM Model
SVM= svm.SVC()
SVM.fit(features_train, y_train)

#Calculate balanced accuracy
y_pred = SVM.predict(features_test)
print(balanced_accuracy_score(y_test, y_pred))

#Model 2
model_name = "bert-base-uncased"
max_length = 512

tokenizer = BertTokenizerFast.from_pretrained(model_name)

train_encodings = tokenizer(text_train, truncation=True, padding=True, max_length=max_length)
test_encodings = tokenizer(text_test, truncation=True, padding=True, max_length=max_length)

class NewsGroupsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

# convert our tokenized data into a torch Dataset
train_dataset = NewsGroupsDataset(train_encodings, y_train)
valid_dataset = NewsGroupsDataset(test_encodings, y_test)

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=len(label))

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=20,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    load_best_model_at_end=True,     # load the best model when finished training (default metric is loss)
    logging_steps=400,               # log & save weights each logging_steps
    save_steps=400,
    evaluation_strategy="steps",     # evaluate each `logging_steps`
)

def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = balanced_accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=valid_dataset,          # evaluation dataset
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest
)

# train the model
trainer.train()
trainer.evaluate()