import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import re


def create_dataframe(file_path):
    counter = 0
    questions = []
    answers = []
    imgs    = []
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if counter%2==0: 
                questions.append(row[0])
                
                #extract images
                img =row[0].split()
                img = img[-2] #+ '.jpg'
                imgs.append(img)
            else:     
                answers.append(row[0])
            
            
            counter +=1
            
    d = {'Questions':questions,'Answer':answers, 'Images':imgs} 
    df = pd.DataFrame(d)      
    
    
    return df   





def save_data(filename, X):
    df = X
    df.to_csv(filename, encoding='utf-8', index=True)
    
    
    
    
    



file_path = '../../Work Folders/Documents/Reduced DAQUAR/test.txt'
df = create_dataframe(file_path)

target = df['Answer']


X_train, X_valid,  y_train, y_valid = train_test_split(df, target, test_size=0.07, random_state=42)


save_data('data/DAQUAR_Train.csv',X_train)
save_data('data/DAQUAR_Valid.csv',X_valid)







file_path = '../../Work Folders/Documents/Reduced DAQUAR/test.txt'
df = create_dataframe(file_path)
#Save test data
save_data('data/DAQUAR_Test.csv',df)


