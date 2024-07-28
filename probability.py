import numpy as np

def create_train_data():
    train_data = np.array([[ 'Sunny ', 'Hot','High', 'Weak', 'no'],
 ['Sunny ', 'Hot', 'High', 'Strong', 'no'],
 ['Overcast', 'Hot', 'High', 'Weak', 'yes'],
 ['Rain ', 'Mild', 'High', 'Weak', 'yes'],
 ['Rain ', 'Cool', 'Normal', 'Weak', 'yes'],
 ['Rain ', 'Cool', 'Normal', 'Strong', 'no'],
 ['Overcast', 'Cool', 'Normal', 'Strong', 'yes'],
 ['Overcast', 'Mild', 'High', 'Weak', 'no'],
 ['Sunny ', 'Cool', 'Normal', 'Weak', 'yes'],
 ['Rain ', 'Mild', 'Normal', 'Weak', 'yes']])
    return train_data

train_data = create_train_data()
print(train_data)

def compute_prior_probability(train_data):
    y_unique = ['no', 'yes']
    prior_probability = np.zeros(len(y_unique))
    for i in range(len(y_unique)):
        prior_probability[i] = np.sum(train_data[:, -1] == y_unique[i]) / len(train_data)
    return prior_probability

prior_probability = compute_prior_probability(train_data)
print("P(play tennis = No) = ", prior_probability[0])
print("P(play tennis = Yes) = ", prior_probability[1])

def compute_conditional_probability(train_data):
    y_unique = ['no', 'yes']
    conditional_probability = []
    list_x_name = []
    for i in range(0, train_data.shape[1]-1):
        x_unique = np.unique(train_data[:, i])
        list_x_name.append(x_unique)

        for j in range(len(y_unique)):
            conditional_probability1 = []
            for k in range(len(x_unique)):

                conditional_probability1.append(np.sum((train_data[:, i] == x_unique[k]) & (train_data[:, -1] == y_unique[j])) / np.sum(train_data[:, -1] == y_unique[j]))
            conditional_probability.append(conditional_probability1)
            conditional_probability1 = []


    return conditional_probability, list_x_name

train_data = create_train_data()
_, list_x_name = compute_conditional_probability(train_data)
print(list_x_name)
print("x1 = ",list_x_name[0])
print("x2 = ", list_x_name[1])
print("x3 = ", list_x_name[2])
print("x4 = ", list_x_name[3])
print(_)

def get_index_from_value(feature_name, list_features):
    return np.where(feature_name == list_features)[0]

_, list_x_name = compute_conditional_probability(train_data)
outlook = list_x_name[0]
print(get_index_from_value('Overcast', outlook))
print(get_index_from_value('Rain ', outlook))
print(get_index_from_value('Sunny ', outlook))

train_data = create_train_data ()
conditional_probability , list_x_name = compute_conditional_probability ( train_data )
# Compute P(" Outlook "=" Sunny "| Play Tennis "=" Yes ")
x1 = get_index_from_value (" Sunny ", list_x_name[0])

print(conditional_probability)

# Prediction
def train_naive_bayes(train_data):
    #Step 1: Calculate Prior probability
    _ = ['no', 'yes']
    prior_probability = compute_prior_probability(train_data)
    #Step 2: Calculate Conditional probability
    conditional_probability, list_x_name = compute_conditional_probability(train_data)

    return prior_probability, conditional_probability, list_x_name

train_naive_bayes(train_data)

def prediction_play_tennis(X, list_x_name, prior_probability, conditional_probability):

    x1 = get_index_from_value(X[0], list_x_name[0])
    x2 = get_index_from_value(X[1], list_x_name[1])
    x3 = get_index_from_value(X[2], list_x_name[2])
    x4 = get_index_from_value(X[3], list_x_name[3])

    p0 = 0
    p1 = 0
    print(x1[0])

    p0 = prior_probability[0] * conditional_probability[x1[0]][0] * conditional_probability[x2[0]][0] * conditional_probability[x3[0]][0] * conditional_probability[x4[0]][0]
    p1 = prior_probability[1] * conditional_probability[x1[0]][1] * conditional_probability[x2[0]][1] * conditional_probability[x3[0]][1] * conditional_probability[x4[0]][1]

    if p0 > p1:
        print("AD should not go")
    else:
        print("Ad should go!")




