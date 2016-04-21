from analysis.DataSet import DataSet
from util.EvalUtil import EvalUtil

class BasicAnalysis:
  # ==========================
  # @param dataset {DataSet}
  # @return [{Int}] the unique tokens in the dataset
  # ==========================
  def uniq_tokens(self, dataset):
    uniq = set()
    while dataset.hasNext():
      instance = dataset.nextInstance()
      for token in str(instance).split("|")[-1].split(","): # -1 is agnostic to test/train
        uniq.add(token)
    dataset.reset()
    return uniq
  
  # ==========================
  # @param dataset {DataSet}
  # @return [{Int}] the unique user ids in the dataset
  # ==========================
  def uniq_users(self, dataset):
    # TODO: Fill in your code here
    return set()

  # ==========================
  # @param dataset {DataSet}
  # @return {Int: [{Int}]} a mapping from age group to unique users ids
  #                        in the dataset
  # ==========================
  def uniq_users_per_age_group(self, dataset):
    if dataset.has_label: # if training, index assuming label
      useridx = 3
      ageidx = 5
    else: # if testing, indexes are 1 to the left
      useridx = 2
      ageidx = 4

    ages = range(7)
    usersByAge = [set() for _ in range(7)]

    # Separate the (unique) users by age group 
    while dataset.hasNext():
      instance = dataset.nextInstance()
      instance = str(instance).split("|")
      ageGroup = int(instance[ageidx])
      user = instance[useridx]
      usersByAge[ageGroup].add(user)
    
    # Count the users in each age group
    ageGroupCounts = {}
    for age,users in enumerate(usersByAge):
      ageGroupCounts[age]=len(users)

    dataset.reset()
    return ageGroupCounts

  # ==========================
  # @param dataset {DataSet}
  # @return {Double} the average CTR for a dataset
  # ==========================
  def average_ctr(self, dataset):
    totalCount = 0.
    clickCount = 0.
    while dataset.hasNext():
      instance = dataset.nextInstance()
      for token in str(instance).split("|")[0]: # should only be used on labeled data, so this index is ok
        totalCount += 1
        if int(token) == 1:
          clickCount +=1
    dataset.reset()
    return clickCount/totalCount



if __name__ == '__main__':
  print "Basic Analysis..."

  basic = BasicAnalysis()
  training_size = DataSet.TRAININGSIZE
  testing_size = DataSet.TESTINGSIZE

  rootpath="/Users/katiedoroschak/Documents/uwacademics/cs547/hw1/ClickPrediction/"
  training = DataSet(rootpath+"data/train.txt", True, training_size)
  testing = DataSet(rootpath+"data/test.txt", False, testing_size)

  # # calculates the average CTR for the training data
  # print "Average CTR for training data: ", basic.average_ctr(training)

  # # counts the number of unique tokens in the training set
  # uniq_tok_train = basic.uniq_tokens(training)
  # print len(uniq_tok_train), "unique tokens in the training set."
  
  # # counts the number of unique tokens in the testing set
  # uniq_tok_test = basic.uniq_tokens(testing)
  # print len(uniq_tok_test), "unique tokens in the testing set."

  # # count number of tokens appearing in both sets
  # print len(uniq_tok_train.intersection(uniq_tok_test)), "unique tokens appear in both datasets."

  # calculates unique users in each age group
  print basic.uniq_users_per_age_group(training), "unique users in each age group of the training set."
  print basic.uniq_users_per_age_group(testing), "unique users in each age group of the testing set."

