import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
#----------------------------------------------------------#
#            Preprocessing Data                            #
#----------------------------------------------------------#
training_csv = 'C:/Users/zinaw/git/Q3_Info_Retrieval/train.csv'
test_csv = 'C:/Users/zinaw/git/Q3_Info_Retrieval/test.csv'
train_set = pd.read_csv(training_csv)
test_set = pd.read_csv(test_csv)
# If the question is empty, convert it into empty string
# train_set.dropna(subset=['question1','question2'], inplace=True)
train_set = train_set.replace(np.nan, ' ', regex=True)
#----------------------------------------------------------#
#            Get Length of Ques                            #
#----------------------------------------------------------#
que1_length_dic = {}
que2_length_dic = {}
for i in range(0,len(train_set)):
    temp_len1 =  len(str(train_set.loc[i]['question1']).split())
    temp_len2 =  len(str(train_set.loc[i]['question2']).split())
    if temp_len1 == 0 or temp_len2 == 0:
        continue
    que1_length_dic[i] = temp_len1
    que2_length_dic[i] = temp_len2

    print(i,que1_length_dic[i], que2_length_dic[i])
que1_length = pd.DataFrame({'Que1_Length':que1_length_dic})
que2_length = pd.DataFrame({'Que2_Length':que2_length_dic})
que_length = que1_length.join(que2_length)
#----------------------------------------------------------#
#       Relations bewteen Length and Duplication           #
#----------------------------------------------------------#
ratio = {}
# for i in range(0,len(train_set)):
# for i in que1_length_dic:
ratio = np.divide(que1_length_dic, que2_length_dic)
print(ratio)
que_ratio = pd.DataFrame({'Ratio':ratio})
que_feature = que_length.join(que_ratio)
que_feature = que_feature.join(train_set['is_duplicate'])
#----------------------------------------------------------#
#                  Show Our Result                         #
#----------------------------------------------------------#
non_duplicate = que_feature.loc[que_feature['is_duplicate'] == 0]
duplicate = que_feature.loc[que_feature['is_duplicate'] == 1]
print('non_duplicate set shape is:',non_duplicate.shape, 'duplicate set shape is:',duplicate.shape)
print('99\% duplicate questions\' ratio is below:',np.percentile(duplicate['Ratio'],99))
print('99\% non_duplicate questions\' ratio is below:',np.percentile(non_duplicate['Ratio'],99))
duplicate_mean = np.mean(duplicate['Ratio'])
non_dup_mean = np.mean(non_duplicate['Ratio'])
print('mean ratio of duplicate question:', duplicate_mean,'mean ratio of non_duplicate question:', non_dup_mean)
fig = plt.figure()
plt.hist(duplicate['Ratio'])
plt.legend('Ratio of Duplicate Question')
plt.savefig('ratio_duplicate_.png')
plt.hist(non_duplicate['Ratio'])
plt.legend('Ratio of Non_Duplicate Question')
plt.savefig('ratio_non_duplicate_.png')

counts, bins = np.histogram(non_duplicate['Ratio'], bins="auto",normed=True)
widths = [bins[i + 1] - bins[i] for i in range(0, len(bins) - 1)]
plt.bar(bins[1:], counts, width=widths)
plt.savefig('bar_non_duplicate.png')

counts, bins = np.histogram(duplicate['Ratio'], bins="auto",normed=True)
widths = [bins[i + 1] - bins[i] for i in range(0, len(bins) - 1)]
plt.bar(bins[1:], counts, width=widths)
plt.savefig('bar_duplicate.png')

plt.boxplot(duplicate['Ratio'])
plt.savefig('boxplot_duplicate.png')
plt.boxplot(non_duplicate['Ratio'])
plt.savefig('boxplot_non_duplicate.png')
