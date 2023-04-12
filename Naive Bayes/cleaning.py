import csv
import re
import langdetect
import numpy as np
def preprocess_text(text, clean, stop_words):
    text = text.lower()
    value = re.sub('[^a-z ]', '', text) # remove non-alphabetic characters
    text = text.split()
    value =[word for word in text if word not in stop_words]
    if clean:
        value = ' '.join(value)
    return value

if __name__ == "__main__":
    stop_words = open('stopwords', 'r', encoding = "ISO-8859-1").readlines()
    stop_words = [line.rstrip('\n') for line in stop_words]
    stop_words = [line.replace(' ', '') for line in stop_words]
    improved_data = []
    index = 0
    csvfile =  open('steam_reviews.csv', newline='', encoding='utf8')
    reader = csv.reader(csvfile)

    file = open('english_cleaned.csv', mode='w', newline='')

    # Create a CSV writer object
    writer = csv.writer(file)

    # Write the header row
    writer.writerow(['Helpful', 'Review'])

    # Write some data rows


    for row in reader:
        index += 1
        #if index > 40000:
        #    print(1+"!")
        if(index %1000 == 0):
            print("Still working ", index)
        # do something with each row
        review = preprocess_text(row[6], True, stop_words)
        try:
            result = langdetect.detect(review)
            if(row[2].isnumeric and result == "en"):
                score = float(row[2]) + 1.0
                score = np.log(score)
                improved_data.append([score, review])
                writer.writerow([score, review])

        except:
            index += 0


