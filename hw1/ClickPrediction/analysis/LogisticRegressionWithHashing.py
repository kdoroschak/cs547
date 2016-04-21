import timeit
import math
import numpy as np
from analysis.DataSet import DataSet
from util.EvalUtil import EvalUtil
from util.HashUtil import HashUtil
import timeit
from matplotlib import pyplot as plt

class Weights:
  def __init__(self, featuredim):
    self.featuredim = featuredim
    self.w0 = self.w_age = self.w_gender = self.w_depth = self.w_position = 0
    # hashed feature weights
    self.w_hashed_features = [0.0 for _ in range(featuredim)]
    # to keep track of the access timestamp of feature weights.
    #   use this to do delayed regularization.
    self.access_time = {}

  def __str__(self):
    formatter = "{0:.2f}"
    string = ""
    string += "Intercept: " + formatter.format(self.w0) + "\n"
    string += "Depth: " + formatter.format(self.w_depth) + "\n"
    string += "Position: " + formatter.format(self.w_position) + "\n"
    string += "Gender: " + formatter.format(self.w_gender) + "\n"
    string += "Age: " + formatter.format(self.w_age) + "\n"
    string += "Hashed Feature: "
    string += " ".join([str(val) for val in self.w_hashed_features])
    string += "\n"
    return string

  # @return {Double} the l2 norm of this weight vector
  def l2_norm(self):
    l2 = self.w0 * self.w0 +\
          self.w_age * self.w_age +\
          self.w_gender * self.w_gender +\
          self.w_depth * self.w_depth +\
          self.w_position * self.w_position
    for w in self.w_hashed_features:
      l2 += w * w
    return math.sqrt(l2)


class LogisticRegressionWithHashing:
  # ==========================
  # Helper function to compute inner product w^Tx.
  # @param weights {Weights}
  # @param instance {DataInstance}
  # @return {Double}
  # ==========================
  def compute_weight_feature_product(self, weights, instance, hashed_features, sparse_features):
    product_normal_features = instance.depth*weights.w_depth + instance.position*weights.w_position + instance.gender*weights.w_gender + instance.age*weights.w_age
    # product_tokens = np.dot(hashed_features, np.array(weights.w_hashed_features).T)

    product_tokens = 0
    for token in sparse_features:
      try: 
        weights.w_hashed_features[token]
      except:
        print token
        print weights.w_hashed_features
      product_tokens += weights.w_hashed_features[token] 
      weights.w_hashed_features[token]
      product_tokens += weights.w_hashed_features[token] 
    weight_feature_product = product_normal_features + product_tokens + weights.w0

    return  weight_feature_product
  
  # ==========================
  # Apply delayed regularization to the weights corresponding to the given
  # tokens.
  # @param featureids {[Int]} list of feature ids
  # @param weights {Weights}
  # @param now {Int} current iteration
  # @param step {Double} step size
  # @param lambduh {Double} lambda
  # ==========================
  def perform_delayed_regularization(self, featureids, weights, now, step, lambduh):
    weights.w_age -= weights.w_age * lambduh * step
    weights.w_depth -= weights.w_depth * lambduh * step
    weights.w_gender -= weights.w_gender * lambduh * step
    weights.w_position -= weights.w_position * lambduh * step
    for feature in featureids:
      # print feature
      # print w.
      if now==weights.access_time[feature]:
        continue
      assert now > weights.access_time[feature] 
      weights.w_hashed_features[feature] *= (1-step*lambduh)**(now-weights.access_time[feature]-1)
      weights.access_time[feature] = now
    return weights

  def reduce_feature_space(self, tokens, dim):
    new_features = np.zeros(dim, dtype=int)
    sparse_features = []
    for token in tokens:
      h = HashUtil.hash_to_range(token, dim)
      idx = h % dim
      new_features[idx] += HashUtil.hash_to_sign(token)
      sparse_features.append(idx*HashUtil.hash_to_sign(token))
    return new_features, sparse_features
  
  # ==========================
  # Train the logistic regression model using the training data and the
  # hyperparameters. Return the weights, and record the cumulative loss.
  # @return {Weights} the final trained weights.
  # ==========================
  def train(self, dataset, dim, lambduh, step, avg_loss, personalized):
    weights = Weights(dim)
    avg_loss_out = []
    t = 0
    totalLoss = 0
    while dataset.hasNext():
      t += 1
      instance = dataset.nextInstance()

      hashed_features, sparse_features = self.reduce_feature_space(instance.tokens, dim)

      for token in sparse_features:
        if token not in weights.access_time:
          weights.w_hashed_features[token] = hashed_features[token]
          weights.access_time[token] = t

      # Your code: perform delayed regularization
      weights = self.perform_delayed_regularization(sparse_features, weights, t, step, lambduh)

      # Your code: predict the label, record the loss
      predictedLabel = self.predictLabel(weights, instance, hashed_features, sparse_features)
      trueLabel = instance.clicked
      totalLoss += self.loss(trueLabel, predictedLabel)
      averageLoss = totalLoss/t
      # if math.floor(t/100) == t/100. or t==1: # print every 100th value of the loss
        # print t, ":", averageLoss
      avg_loss_out.append(averageLoss)
      

      # Your code: compute w0 + <w, x>, and gradient
      # Your code: update weights along the negative gradient
      weights.w0 = weights.w0 + step * (trueLabel - predictedLabel) #/ dataset.size
      update = step * (trueLabel- predictedLabel) #/ dataset.size
      weights.w_depth     += update * instance.depth
      weights.w_position  += update * instance.position
      weights.w_gender    += update * instance.gender
      weights.w_age       += update * instance.age
      for token in sparse_features:
        weights.w_hashed_features[token] += update

    dataset.reset()
    return weights, avg_loss_out, hashed_features, sparse_features
  # ==========================
  # Use the weights to predict the CTR for a dataset.
  # @param weights {Weights}
  # @param dataset {DataSet}
  # @param personalized {Boolean}
  # ==========================
  def predict(self, weights, dataset, personalized, hashed_features, sparse_features):
    prediction = []
    t=0
    while dataset.hasNext():
      instance = dataset.nextInstance()
      tokenweights = 0
      tokenweights = weights.w_hashed_features*hashed_features
      for token in hashed_features:
        if token not in weights.w_hashed_features:
          weights.w_hashed_features[token] = 0
        tokenweights += weights.w_hashed_features[token]
      prediction.append(self.predictLabel(weights, instance, hashed_features, sparse_features))
      t+=1

    dataset.reset()
    return prediction

  def predictLabel(self, weights, instance, hashed_features, sparse_features):
    expTerm = math.exp(self.compute_weight_feature_product(weights, instance, hashed_features, sparse_features))
    label = 1 / (1 + expTerm)
    return 1-label
  
  def loss(self, actual, predicted):
    l = np.sum((actual - predicted)**2)
    return l
  
if __name__ == '__main__':
  print "Training Logistic Regression with Hashed Features..."
  training_size = DataSet.TRAININGSIZE
  testing_size = DataSet.TESTINGSIZE

  rootpath="/Users/katiedoroschak/Documents/uwacademics/cs547/hw1/ClickPrediction/"
  training = DataSet(rootpath+"data/train.txt", True, training_size)
  testing = DataSet(rootpath+"data/test.txt", False, testing_size)

  m = [101, 12277, 1573549]
  lr = LogisticRegressionWithHashing()

  fig,ax = plt.subplots()


  step = 0.01
  lambduh = 0.001

  for dim in m:
      print "Doing training/testing for lambda=", lambduh, "step=", step
      weights,avg_loss,hashed_features,sparse_features = lr.train(training, dim, lambduh, step, None, None)
      print "Results from training/testing for lambda=", lambduh, "step=", step
      print "L2 norm of weights", weights.l2_norm()

      
      ax.plot(range(len(avg_loss)), avg_loss)

      # Test
      prediction = lr.predict(weights, testing, None, hashed_features, sparse_features)

      average_CTR_baseline = 0.033655
      print "Average CTR for baseline:", average_CTR_baseline
      RMSE_baseline = EvalUtil.eval_baseline("data/test_label.txt", average_CTR_baseline)
      print "Baseline RMSE", RMSE_baseline

      average_CTR_predicted = sum(prediction)/len(prediction)
      print "Average CTR for predicted:", average_CTR_predicted
      RMSE_predicted_CTR = EvalUtil.eval("data/test_label.txt", prediction)
      print "RMSE for predicted CTR:", RMSE_predicted_CTR


  fig.savefig("plot.png")





