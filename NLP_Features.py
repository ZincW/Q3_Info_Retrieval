import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
#----------------------------------------------------------#
#            Preprocessing Data                            #
#----------------------------------------------------------#
training_csv = '/Users/Zina Wang/git/NLP/train.csv'
test_csv = '/Users/Zina Wang/git/NLP/test.csv'
train_set = pd.read_csv(training_csv)
test_set = pd.read_csv(test_csv)
#----------------------------------------------------------#
#            Get Length of Ques                            #
#----------------------------------------------------------#
que1_length_dic = {}
que2_length_dic = {}
for i in range(0,len(train_set)):
    que1_length_dic[i] = len(train_set.loc[i]['question1'].split())
    que2_length_dic[i] = len(train_set.loc[i]['question2'].split())
    print i,que1_length_dic[i], que2_length_dic[i]
que1_length = pd.DataFrame({'Que1_Length':que1_length_dic})
que2_length = pd.DataFrame({'Que2_Length':que2_length_dic})
que_length = que1_length.join(que2_length)
#----------------------------------------------------------#
#       Relations bewteen Length and Duplication           #
#----------------------------------------------------------#
ratio = {}
for i in range(0,len(train_set)):
    ratio[i] = float(que2_length_dic[i])/ float(que1_length_dic[i])
    print i,ratio[i]
que_ratio = pd.DataFrame({'Ratio':ratio})
que_feature = que_length.join(que_ratio)
que_feature = que_feature.join(train_set['is_duplicate'])
#----------------------------------------------------------#
#                  Plot Our Result                         #
#----------------------------------------------------------#
non_duplicate = que_feature.loc[que_feature['is_duplicate'] == 0]
duplicate = que_feature.loc[que_feature['is_duplicate'] == 1]
np.percentile(duplicate['Ratio'],99)
np.percentile(non_duplicate['Ratio'],99)
np.mean(duplicate['Ratio'])
np.mean(non_duplicate['Ratio'])
fig = plt.figure()
plt.hist(duplicate['Ratio'])
plt.hist(non_duplicate['Ratio'])

counts, bins = np.histogram(non_duplicate['Ratio'], bins="auto",normed=True)
widths = [bins[i + 1] - bins[i] for i in range(0, len(bins) - 1)]
print widths
plt.bar(bins[1:], counts, width=widths)

#plt.plot(index,Ratio)
plt.boxplot(duplicate['Ratio'])
plt.savefig('non_duplicate_plot.png')