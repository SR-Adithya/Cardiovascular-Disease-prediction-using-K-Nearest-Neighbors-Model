import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sbn
import numpy as np

information = pd.read_csv("code and dataset/train.csv")

information.rename(columns = {'sex' : 'Gender', 'is_smoking' : 'Smoker', 'cigsPerDay' : 'Cigarette(s) per day',
                              'totChol' : 'Recent cholestrol level', 'sysBP' : 'Upper BP', 'diaBP' : 'Lower BP',
                              'TenYearCHD' : 'Cardiovascular disease exposure'}, inplace = True)

# obtain data information

print(information.describe())
print(information.dtypes)
print(information.isnull().sum())

# transform gender column yes or no to 1 or 0

information['Gender'].replace(('M','F'),(1,0), inplace=True)

# transform smoker column yes or no to 1 or 0

information['Smoker'].replace(('YES','NO'),(1,0), inplace=True)

# nan values to be filled with mode value for cigarettes per day

information['Cigarette(s) per day'] = information['Cigarette(s) per day'].fillna(
    information.loc[information['Smoker'] == 1, 'Cigarette(s) per day'].mean())

#print(information['Cigarette(s) per day'].isnull().sum())

# nan values to be filled with 0 value for BPMeds but not required for the input as of now

# nan values to be filled with mean values for Recent cholestrol level

information['Recent cholestrol level'] = information['Recent cholestrol level'].fillna(information['Recent cholestrol level'].mean())

# nan values to be filled with mean values for BMI

information['BMI'] = information['BMI'].fillna(information['BMI'].mean())

# nan values to be filled with mean values for heart rate

information['heartRate'] = information['heartRate'].fillna(information['heartRate'].mean())

# nan values to be filled with mean values for glucose

information['glucose'] = information['glucose'].fillna(information['glucose'].mean())

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Split the data into input and output for the model

x = information[['age','Gender','Smoker','Cigarette(s) per day','Upper BP','Lower BP','heartRate']]
y = information['Cardiovascular disease exposure']

# Data split into train and test for model to learn

learn_x, try_x, learn_y, try_y = train_test_split(x, y, test_size = 0.2, random_state = 0)


# Model learning

model = KNeighborsClassifier(n_neighbors=5)
model.fit(learn_x, learn_y)

# Model prediction

forecast_y = model.predict(try_x)

# metrics used by the model to support its prediction

print("\nConfusion Matrix of the KNN classifier model: ", confusion_matrix(try_y, forecast_y))
print("\nAccuracy Score of the KNN classifier model: ", accuracy_score(try_y, forecast_y))
print("\nClassification Report of the KNN classifier model: \n", classification_report(try_y, forecast_y))

# for graph visual of the model

#count plot
plt.figure(figsize=(10, 8))
ax = sbn.countplot(x='Cardiovascular disease exposure', data=information, palette='Spectral')
ax.set_facecolor('black')
plt.gcf().patch.set_facecolor('black')
ax.bar_label(ax.containers[0], color='white', fontsize=14)
ax.bar_label(ax.containers[1], color='white', fontsize=14)
plt.title("Count plot: Cardiovascular Disease Exposure", color='white', fontsize=14)
plt.xlabel("CVD Exposure Class", color='white', fontsize=14)
plt.ylabel("Individuals Count", color='white', fontsize=14)
ax.tick_params(colors='white', size=20)
legend_elements = [
    Patch(facecolor='none', edgecolor='none', label='0 = Healthy'),
    Patch(facecolor='none', edgecolor='none', label='1 = Cardiovascular Disease')
]
ax.legend(handles=legend_elements, loc='upper right', frameon=True, labelcolor='black', fontsize=12)
#plt.text(0.5, -0.15, "Class Labels: 0 = Healthy | 1 = Cardiovascular Disease Exposure",
#    ha='center', va='center', transform=ax.transAxes, color='white', fontsize=12)
ax.yaxis.grid(True, color='white', linestyle='--', linewidth=0.6, alpha=0.6)
plt.show()

#violin plot for age and heart rate
fig, axes = plt.subplots(1, 2, figsize=(10, 8))

# Set black background
fig.patch.set_facecolor('black')
for ax in axes:
    ax.set_facecolor('black')

# ---- Violin plot for AGE ----
sbn.violinplot(
    x='Cardiovascular disease exposure',
    y='age',
    data=information,
    palette='vlag',
    ax=axes[0]
)

axes[0].set_title("Age Distribution vs CVD Exposure", color='white', fontsize=14)
axes[0].set_xlabel("CVD Exposure Class", color='white', fontsize=12)
axes[0].set_ylabel("Age (Years)", color='white', fontsize=12)
axes[0].tick_params(colors='white', labelsize=11)
axes[0].grid(True, color='white', linestyle='--', linewidth=0.6, alpha=0.6)

# ---- Violin plot for HEART RATE ----
sbn.violinplot(
    x='Cardiovascular disease exposure',
    y='heartRate',
    data=information,
    palette='Spectral',
    ax=axes[1]
)

axes[1].set_title("Heart Rate Distribution vs CVD Exposure", color='white', fontsize=14)
axes[1].set_xlabel("CVD Exposure Class", color='white', fontsize=12)
axes[1].set_ylabel("Heart Rate (BPM)", color='white', fontsize=12)
axes[1].tick_params(colors='white', labelsize=11)
axes[1].grid(True, color='white', linestyle='--', linewidth=0.6, alpha=0.6)

# ---- Legend (shared) ----
legend_elements = [
    Patch(facecolor='none', edgecolor='none', label='0 = Healthy'),
    Patch(facecolor='none', edgecolor='none', label='1 = Cardiovascular Disease')
]

fig.legend(
    handles=legend_elements,
    loc='upper center',
    frameon=True,
    labelcolor='black',
    fontsize=12
)
plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.show()

# pairplot
plt.figure(figsize=(20,20))
pair = sbn.pairplot(
    information[['age','Gender','Smoker','Cigarette(s) per day','Cardiovascular disease exposure']],
    hue='Cardiovascular disease exposure'
)
new_labels = ['0 = Healthy ', '1 = CVD Exposure']
for text, label in zip(pair._legend.texts, new_labels):
    text.set_text(label)

pair._legend.set_title("Legend")
plt.show()

# pair plot 2
plt.figure(figsize=(20,20))
pair = sbn.pairplot(
    information[['age','Upper BP','Lower BP','heartRate','Cardiovascular disease exposure']],
    hue='Cardiovascular disease exposure'
)
new_labels = ['0 = Healthy ', '1 = CVD Exposure']
for text, label in zip(pair._legend.texts, new_labels):
    text.set_text(label)

pair._legend.set_title("Legend")
plt.show()
# model testing with user inputs

age = int(input("Enter person's age (in years): "))
gender = input("Enter Gender of the person (male/female): ").strip().lower()
Smoker = input("Does the person smoke? (yes/no): ").strip().lower()
CigperD = int(input("If yes kindly mention no of cigarette(s) per day, If not a smoker kindly enter 0: "))
UBP = int(input("Enter Upper Blood Pressure (mm/Hg): "))
LBP = int(input("Enter Lower Blood Pressure (mm/Hg): "))
heartrate = int(input("Enter Heart Rate beats per minute (BPM): "))

# input value validation and check gender and smoker inputs
if gender not in ['male','female']:
    print("Check again and kindly enter Male or Female")
elif Smoker not in ['yes','no']:
    print("Check again and kindly enter Yes or No")
elif age<=0 or CigperD<0 or UBP<=0 or LBP<=0 or heartrate<=0:
    print("Please check all the entries and kindly enter proper values")
else:
    # convert gender f or m to 0 or 1
    gender = 1 if gender == 'male' else 0
    
    # convert smoker yes or no to 0 or 1
    Smoker = 1 if Smoker == 'yes' else 0

# Model prediction for new input and it's outcome

    Exposure = model.predict([[age,gender,Smoker,CigperD,UBP,LBP,heartrate]])[0]
    print("Details entered as per the admin: \n")
    print(f"Age of person: {age} years\n")
    print(f"Gender of person: {'Male' if gender == 1 else 'Female'}\n")
    print(f"Smoker or non-smoker: {'Smoker' if Smoker == 1 else 'non-smoker'}\n")
    print(f"Cigarette(s) per day: {CigperD} per day\n")
    print(f"Upper Blood Pressure: {UBP} mmHg\n")
    print(f"Lower Blood Pressure: {LBP} mmHg\n")
    print(f"Person's Heart Rate: {heartrate} BPM\n")
    if Exposure == 1:
        print("Cardiovascular Disease exposure status: May have CHD")
    
# suggestions based on the prediction:
        print("\n⚠ Health Advisory: You may be at risk of Coronary Heart Disease.")
        print("👉 Suggestions:")
        print("-✅ Consult a cardiologist for further diagnosis and medication.")
        print("-❌ Avoid fat or oil foods. Maintain a balanced diet with low cholesterol.")
        print("-🚶‍ Try to include walking or exercise for at least 30 minutes daily.")
        print("-⬇️ Reduce smoking if you are a smoker.")
        print("-🖥️ Monitor blood pressure regularly and note the reading for doctor reference.\n")
    else:
        print("Cardiovascular Disease exposure status: Healthy")
        print("\n✅ You appear healthy, no immediate CHD symptoms detected.")
        print("👉 Suggestions to stay healthy:")
        print("-💪 Maintain a healthy lifestyle with regular exercise.")
        print("-❌ Avoid smoking and excessive alcohol.")
        print("-🖥️ Monitor your blood pressure and heart rate regularly.")
        print("-🕘 Go for periodic health checkups.\n")


prob = model.predict_proba([[age,gender,Smoker,CigperD,UBP,LBP,heartrate]])[0][1]

print(f"Predicted probability of CVD: {prob:.2f}")
