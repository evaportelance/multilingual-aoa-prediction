import csv

words_of_interest = set({})
with open("../../data/model-sets/aoa_word_list.csv", 'r') as word_list:
    for line in word_list:
        word_of_interest = line.partition('\t')[2]
        words_of_interest.add(word_of_interest[:-1])

if 'tiger' in words_of_interest:
    print(237947913487731249)


print(words_of_interest)

w_and_gest_of_interest = []
with open("../../data/wordbank-CDI-aoa-data/words_and_gestures_produces_item_data.csv", 'r') as w_and_gest:
    for line in w_and_gest:
        string = line.partition(',\"word\",')[0]
        word = (string.partition(',')[2]).lower()
        word = word.strip('\"')
        if word in words_of_interest:
            w_and_gest_of_interest.append([line.strip('\n').strip( '\"')])
print(w_and_gest_of_interest)
print('\n\n\n')

w_and_sent_of_interest = []
with open("../../data/wordbank-CDI-aoa-data/words_and_sentences_produces_item_data.csv", 'r') as w_and_sent:
    for line in w_and_sent:
        string = line.partition(',\"word\",')[0]
        word = string.partition(',')[2]
        word = word.strip('\"')
        if word in words_of_interest:
            w_and_sent_of_interest.append([line.strip('\n').strip('\"')])
print(w_and_sent_of_interest)

with open("../../data/wordbank-CDI-aoa-data/words_of_interest_words_and_gestures.csv",'w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerows(w_and_gest_of_interest)


with open("../../data/wordbank-CDI-aoa-data/words_of_interest_words_and_sentences.csv",'w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    csvwriter.writerows(w_and_sent_of_interest)
    
    
