# MLp with automatic validation set
from keras.models import Sequential
from keras.layers import Dense
import numpy
from sklearn.model_selection import StratifiedKFold
import redis
import concurrent.futures

# fix seed
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
dataset = numpy.loadtxt('pima-indians-diabetes.csv', delimiter=',')

# split int input (X) and output (Y)
X = dataset[:, 0:8]
Y = dataset[:, 8]

# set up k-fold cv
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

# connect to redis
r = redis.Redis()

def trainCycle(train, test, i):
    print('starting training cycle %i/10...' % i)
    kf_model = Sequential()
    kf_model.add(Dense(12, input_dim=8, activation='relu'))
    kf_model.add(Dense(8, activation='relu'))
    kf_model.add(Dense(1, activation='sigmoid'))
    kf_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    kf_model.fit(X[train], Y[train], validation_data=(X[test], Y[test]), epochs=50, batch_size=10, verbose=0)
    scores = kf_model.evaluate(X[test], Y[test], verbose=0)
    r.rpush("tensorflow:multiprocessing:results", scores[1] * 100)

def trainAll(executor):
    i = 1
    for train, test in kfold.split(X, Y):
      executor.submit(trainCycle, train, test, i)
      i += 1

if __name__ == "__main__":
  r.delete("tensorflow:multiprocessing:results")
  r.delete("tensorflow:multiprocessing:current")
  r.set("tensorflow:multiprocessing:current", 0)
  with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
    trainAll(executor)
  scores = [float(x.decode('utf-8')) for x in r.lrange("tensorflow:multiprocessing:results", 0, -1)]
  print('Accuracy: %.2f%% (+/- %.2f%%)' % (numpy.mean(scores), numpy.std(scores)))
