import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


data1=pd.read_csv('C:/Users/user/Desktop/bishal/tenserflow/train (1).csv')
data2=pd.read_csv('C:/Users/user/Desktop/bishal/tenserflow/eval.csv')



y_train=data1.pop('survived')
y_eval=data2.pop('survived')

#print(data1.head())

numerical_columns=['age','fare','n_siblings_spouses','parch']
categorical_columns=['sex','class','deck','embark_town','alone']

#one_hot encoding for categorical columns
data1=pd.get_dummies(data1,columns=categorical_columns,dtype=np.float32)
data2=pd.get_dummies(data2,columns=categorical_columns,dtype=np.float32)

data2 = data2.reindex(columns=data1.columns, fill_value=0)

#print(data1.head())
#print("Data columns after one-hot encoding (train):", data1.columns)
scaler=StandardScaler()
data1_scaled = scaler.fit_transform(data1[numerical_columns])
data1_scaled = pd.DataFrame(data1_scaled, columns=numerical_columns)
data1[numerical_columns] = data1_scaled

data2_scaled = scaler.transform(data2[numerical_columns])
data2_scaled = pd.DataFrame(data2_scaled, columns=numerical_columns)
data2[numerical_columns] = data2_scaled


model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=(data1.shape[1],))
])

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),  # Stochastic Gradient Descent optimizer
              loss='mean_squared_error')  # Use MSE for regression tasks

history = model.fit(data1, y_train, epochs=100, batch_size=32, verbose=1, validation_split=0.2)

# Evaluate the model
loss = model.evaluate(data2, y_eval)
print("Evaluation loss (MSE):", loss)

# Make predictions on the evaluation data
predictions = model.predict(data2)

# Print the first few predictions
#print("Sample Predictions:", predictions[:10])
'''
# Optionally, convert predictions to a DataFrame for easier comparison with actual values
predictions_df = pd.DataFrame({
    'Actual': y_eval,
    'Predicted': predictions.flatten()  # Flatten to turn it into a 1D array
})

# Display the first few rows of the DataFrame
print(predictions_df.head())

'''
# === Get user input ===
def new_data():
    age = float(input("Enter age: "))
    fare = float(input("Enter fare: "))
    sex = input("Enter sex (male/female): ")
    pclass = input("Enter class (First/Second/Third): ")
    siblings = int(input("Enter number of siblings/spouses aboard: "))
    parch = int(input("Enter number of parents/children aboard: "))
    deck = input("Enter deck (A-G or unknown): ")
    town = input("Enter embark town (Southampton, Cherbourg, Queenstown): ")
    alone = input("Were they alone? (y/n): ")

    # === Create a DataFrame ===
    user_data = pd.DataFrame([{
        'age': age,
        'fare': fare,
        'sex': sex,
        'class': pclass,
        'n_siblings_spouses': siblings,
        'parch': parch,
        'deck': deck,
        'embark_town': town,
        'alone': alone
    }])

    # === One-hot encode and align ===
    user_data = pd.get_dummies(user_data, columns=categorical_columns, dtype=np.float32)
    user_data = user_data.reindex(columns=data1.columns, fill_value=0)

    # === Scale numerical columns ===
    user_data[numerical_columns] = scaler.transform(user_data[numerical_columns])

    # === Predict ===
    prediction = model.predict(user_data)
    predicted_class = (prediction > 0.5).astype(int)

    print("\nPrediction (probability of survival): {:.2f}".format(prediction[0][0]))
    print("Prediction: SURVIVED" if predicted_class[0][0] == 1 else "Prediction: DID NOT SURVIVE")

while(1):
    new_data()

