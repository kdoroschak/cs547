import math
import numpy as np
from analysis.DataSet import DataSet
from util.EvalUtil import EvalUtil
from analysis.BasicAnalysis import BasicAnalysis
from matplotlib import pyplot as plt

# This class represents the weights in the logistic regression model.
class Weights:
  def __init__(self):
    self.w0 = self.w_age = self.w_gender = self.w_depth = self.w_position = 0
    # token feature weights
    self.w_tokens = {}
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
    string += "Tokens: " + str(self.w_tokens) + "\n"
    return string
  
  # @return {Double} the l2 norm of this weight vector
  def l2_norm(self):
    l2 = self.w0 * self.w0 +\
          self.w_age * self.w_age +\
          self.w_gender * self.w_gender +\
          self.w_depth * self.w_depth +\
          self.w_position * self.w_position
    for w in self.w_tokens.values():
      l2 += w * w
    return math.sqrt(l2)
  
  # @return {Int} the l2 norm of this weight vector
  def l0_norm(self):
    return 4 + len(self.w_tokens)


class LogisticRegression:
  # ==========================
  # Helper function to compute inner product w^Tx.
  # @param weights {Weights}
  # @param instance {DataInstance}
  # @return {Double}
  # ==========================
  def compute_weight_feature_product(self, weights, instance):
    product_normal_features = instance.depth*weights.w_depth + instance.position*weights.w_position + instance.gender*weights.w_gender + instance.age*weights.w_age
    product_tokens = 0
    for token in instance.tokens:
      weights.w_tokens[token]
      product_tokens += weights.w_tokens[token] 
    weight_feature_product = product_normal_features + product_tokens + weights.w0
    return  weight_feature_product
  
  # ==========================
  # Apply delayed regularization to the weights corresponding to the given
  # tokens.
  # @param tokens {[Int]} list of tokens
  # @param weights {Weights}
  # @param now {Int} current iteration
  # @param step {Double} step size
  # @param lambduh {Double} lambda
  # ==========================
  def perform_delayed_regularization(self, tokens, weights, now, step, lambduh):
    weights.w_age -= weights.w_age * lambduh * step
    weights.w_depth -= weights.w_depth * lambduh * step
    weights.w_gender -= weights.w_gender * lambduh * step
    weights.w_position -= weights.w_position * lambduh * step
    for token in tokens:
      if now==weights.access_time[token]:
        continue
      assert now > weights.access_time[token] 
      weights.w_tokens[token] *= (1-step*lambduh)**(now-weights.access_time[token]-1)
      weights.access_time[token] = now
    return weights

  # ==========================
  # Train the logistic regression model using the training data and the
  # hyperparameters. Return the weights, and record the cumulative loss.
  # @return {Weights} the final trained weights.
  # ==========================
  def train(self, dataset, lambduh, step, avg_loss):
    weights = Weights()
    t = 0
    totalLoss = 0
    avg_loss_out = []
    while dataset.hasNext():
      t += 1
      instance = dataset.nextInstance()
      for token in instance.tokens:
        if token not in weights.w_tokens:
          weights.w_tokens[token] = 0
          weights.access_time[token] = t

      # Your code: perform delayed regularization
      weights = self.perform_delayed_regularization(instance.tokens, weights, t, step, lambduh)

      # Your code: predict the label, record the loss
      predictedLabel = self.predictLabel(weights, instance)
      trueLabel = instance.clicked
      totalLoss += self.loss(trueLabel, predictedLabel)
      avg_loss = totalLoss/t
      # if math.floor(t/100) == t/100.: # print every 100th value of the loss
        # print t, ":", avg_loss
      avg_loss_out.append(avg_loss)

      # Your code: compute w0 + <w, x>, and gradient
      # Your code: update weights along the negative gradient
      weights.w0 = weights.w0 + step * (trueLabel - predictedLabel) #/ dataset.size
      update = step * (trueLabel- predictedLabel) #/ dataset.size
      weights.w_depth     += update * instance.depth
      weights.w_position  += update * instance.position
      weights.w_gender    += update * instance.gender
      weights.w_age       += update * instance.age
      for token in instance.tokens:
        weights.w_tokens[token] += update

    weights = self.perform_delayed_regularization(weights.access_time.keys(), weights, t, step, lambduh)
    prediction = self.predict(weights, dataset)

    dataset.reset()
    return weights, avg_loss_out

  def predictLabel(self, weights, instance):
    expTerm = math.exp(self.compute_weight_feature_product(weights, instance))
    label = expTerm / (1 + expTerm)
    return label

  def loss(self, actual, predicted):
    l = np.sum((actual - predicted)**2)
    return l

  # ==========================
  # Use the weights to predict the CTR for a dataset.
  # @param weights {Weights}
  # @param dataset {DataSet}
  # ==========================
  def predict(self, weights, dataset):
    prediction = []
    t=0
    while dataset.hasNext():
      instance = dataset.nextInstance()
      tokenweights = 0
      for token in instance.tokens:
        if token not in weights.w_tokens:
          weights.w_tokens[token] = 0
      prediction.append(self.predictLabel(weights, instance))

      t+=1

    dataset.reset()
    return prediction
  
  
if __name__ == '__main__':
  # TODO: Fill in your code here
  print "Training Logistic Regression..."
  basic = BasicAnalysis()


  training_size = DataSet.TRAININGSIZE
  testing_size = DataSet.TESTINGSIZE

  rootpath="/Users/katiedoroschak/Documents/uwacademics/cs547/hw1/ClickPrediction/"
  training = DataSet(rootpath+"data/train.txt", True, training_size)
  testing = DataSet(rootpath+"data/test.txt", False, testing_size)


  fig,ax = plt.subplots()

  lr = LogisticRegression()
  # steps = [0.001, 0.01, 0.05]
  # lambduhs = [0]

  steps = [0.05]
  lambduhs = [0, 0.002, 0.004, 0.006, 0.008, 0.010, 0.012, 0.014]


  for step in steps:
    for lambduh in lambduhs:

      print "Doing training/testing for lambda=", lambduh, "step=", step
      weights, avg_loss = lr.train(training, lambduh, step, 0)
      print "Results from training/testing for lambda=", lambduh, "step=", step
      print "L2 norm of weights", weights.l2_norm()

      
      ax.plot(range(len(avg_loss)), avg_loss)

      # Test
      prediction = lr.predict(weights, testing)

      average_CTR_baseline = 0.033655
      print "Average CTR for baseline:", average_CTR_baseline
      RMSE_baseline = EvalUtil.eval_baseline("data/test_label.txt", average_CTR_baseline)
      print "Baseline RMSE", RMSE_baseline

      average_CTR_predicted = sum(prediction)/len(prediction)
      print "Average CTR for predicted:", average_CTR_predicted
      RMSE_predicted_CTR = EvalUtil.eval("data/test_label.txt", prediction)
      print "RMSE for predicted CTR:", RMSE_predicted_CTR


      if step==0.01:
        print "age:", weights.w_age
        print "depth:", weights.w_depth
        print "gender:", weights.w_gender
        print "position:", weights.w_position

  fig.savefig("plot.png")
