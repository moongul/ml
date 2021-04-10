from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7,
                31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.4, 33.5,
                34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0,
                38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0,
                450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0,
                700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0,
                925.0, 975.0, 950.0]

plt.scatter(bream_length, bream_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

smelt_lenght = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2,
                12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4,
                12.2, 19.7, 19.9]

plt.scatter(smelt_lenght, smelt_weight)
plt.scatter(bream_length, bream_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

length = bream_length + smelt_lenght
weight = bream_weight + smelt_weight
fish_data = [[l, w] for l, w in zip(length, weight)]
fish_target = [1]*35 + [0]*14

kn = KNeighborsClassifier()
kn.fit(fish_data, fish_target)
kn.score(fish_data, fish_target)

kn.predict([[30, 700]])

import numpy as np
index = np.arange(49)
np.random.shuffle(index)

train_arr = np.array(fish_data)
target_arr = np.array(fish_target) 

train_data = train_arr[index[:35]]
train_target = target_arr[index[:35]]

test_data = train_arr[index[35:]]
test_target = target_arr[index[35:]]

plt.scatter(train_data[:,0], train_data[:,1])
plt.scatter(test_data[:,0], test_data[:,1])
plt.show()

kn = kn.fit(train_data, train_target)
kn.score(test_data, test_target)

distance, indexes = kn.kneighbors([[25,150]])

plt.scatter(train_data[:,0], train_data[:,1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_data[indexes, 0], train_data[indexes, 1], marker='D')
plt.xlim((0,1000))
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

mean = np.mean(train_data, axis=0)
std = np.std(train_data, axis=0)
print(mean, std)
train_scaled = (train_data - mean) / std
test_scaled = (test_data - mean) / std

new = ([25,150] - mean) / std

kn.fit(train_scaled, train_target)
kn.score(test_scaled, test_target)

kn.predict([new])

distance, indexes = kn.kneighbors([new])
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

from sklearn.model_selection import train_test_split
bream_length = np.array(bream_length)
bream_weight = np.array(bream_weight)
train_input, test_input, train_target, test_target = train_test_split(bream_length, bream_weight, test_size=0.2, random_state=42, shuffle=True)
train_input = train_input.reshape(-1,1)
test_input = test_input.reshape(-1,1)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error
knr = KNeighborsRegressor()
knr.n_neighbors = 7
knr.fit(train_input, train_target)
print(knr.score(train_input, train_target))
print(knr.score(test_input, test_target))

test_prediction = knr.predict(test_input)
mae = mean_absolute_error(test_target, test_prediction)
print(mae)

distance, indexes = knr.kneighbors([[39]])

plt.scatter(train_input, train_target)
plt.scatter(train_input[indexes], train_target[indexes],marker='D')
plt.show()

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_input, train_target)
print(lr.predict([[60]]), knr.predict([[60]]))
print(lr.coef_, lr.intercept_)

plt.scatter(train_input, train_target)
plt.plot([15,50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_])
plt.scatter(39, 903.4, marker='^')
plt.show()

print(lr.score(train_input, train_target), lr.score(test_input, test_target))

train_poly = np.column_stack((train_input**2, train_input))
test_poly = np.column_stack((test_input**2, test_input))

lr = LinearRegression()
lr.fit(train_poly, train_target)

print(lr.coef_, lr.intercept_)

point = np.arange(25,45)
plt.scatter(train_input, train_target)
plt.plot(point, -2.74*point**2 + 5.05*point + 1026.22)
plt.show()