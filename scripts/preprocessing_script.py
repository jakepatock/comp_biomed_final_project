"""This is the preprocessing script that preprocess and generates the dataset from the raw MIMIC IV 3.1 dataset.
This file takes no parameters. 
"""

import pandas as pd 
from icdmappings import Mapper
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from pathlib import Path
import os

# Function that replicated figure one from Hempel et al.
def make_figure_one(icu_stays_with_data):
    # Count nans
    nan_counts = icu_stays_with_data.isna().sum()
    # Calc nans percentage
    missing_percentages = (nan_counts / len(icu_stays_with_data)) * 100

    # Important variables we want 
    vital_features = ['GCS - Eye Opening', 'GCS - Motor Response', 'GCS - Verbal Response', 'Heart Rate', 'O2 saturation pulseoxymetry', 'Respiratory Rate']
    lab_features = ['Anion Gap', 'Bicarbonate', 'Chloride', 'Creatinine', 'Glucose', 'Hematocrit', 'Hemoglobin', 'Magnesium', 'MCH', 'MCHC', 'MCV', 'Platelet Count', 'Potassium', 'RDW', 'Red Blood Cells', 'Sodium', 'Urea Nitrogen', 'White Blood Cells']

    # Assing missing found 
    vital_signs_missing = missing_percentages[vital_features]
    lab_signs_missing = missing_percentages[lab_features]

    # Ploting 
    plt.bar(vital_signs_missing.index, vital_signs_missing.values)

    plt.xlabel('Feature')
    plt.xticks(rotation=90)
    plt.ylabel('Percentage Missing')

    plt.show()

    plt.bar(lab_signs_missing.index, lab_signs_missing.values)

    plt.xlabel('Feature')
    plt.xticks(rotation=90)
    plt.ylabel('Percentage Missing')

    plt.show()

repo_dir = Path(__file__).parent.parent
dataset_dir = os.path.join(repo_dir, 'dataset')

mimic_dataset_dir = os.path.join(dataset_dir, 'mimic-iv-3.1', 'mimic-iv-3.1')

# Aggregation function we will use to sumarize multiple reading of same feature 
agg_function = 'mean'

# Load ICUs stays 
icu_stays = pd.read_csv(os.path.join(mimic_dataset_dir, r"icu\icustays.csv.gz"))

# Loading admissions data
admissions = pd.read_csv(os.path.join(mimic_dataset_dir, r"hosp\admissions.csv.gz"))

# Filtering people on if they died in the hospital 
# Joining on admission df to get info if they died in icu 
deathtime_df = icu_stays.merge(admissions[['subject_id', 'hadm_id', 'hospital_expire_flag', 'admission_type', 'admission_location']], on=['subject_id', 'hadm_id'], how='left')

# Loading patient information table
patients = pd.read_csv(os.path.join(mimic_dataset_dir, r"hosp\patients.csv.gz"))

# Adding demographic data (2.2. Feature Selection)
age_added = deathtime_df.merge(patients[['subject_id', 'gender', 'anchor_age', 'anchor_year']], on='subject_id', how='left')
# Conversion to datetime 
age_added['intime'] = pd.to_datetime(age_added['intime'])

# Calculating age (2.2. Feature Selection)
age_added['age'] = age_added['anchor_age'] + (age_added['intime'].dt.year - age_added['anchor_year'])
age_added = age_added.drop(columns=['anchor_age', 'anchor_year'])

# Loading diagnosis 
diagnoses = pd.read_csv(os.path.join(mimic_dataset_dir, r"hosp\diagnoses_icd.csv.gz"), dtype={ 'subject_id': 'int32', 'hadm_id': 'int32', 'seq_num': 'int16', 'icd_code': 'string', 'icd_version': 'int8'})

# Spliting the dataframe to get all the ICD9 codes 
icd9 = diagnoses[diagnoses['icd_version'] == 9].copy()
icd10 = diagnoses[diagnoses['icd_version'] == 10].copy()

# Init mapper object 
mapper = Mapper()

# For each ICD9 code map it to the icd10 code 
mapped_codes = []
for idx, value in enumerate(icd9['icd_code']):
    mapped_code = mapper.map(value, source='icd9', target='icd10')
    mapped_codes.append(mapped_code)
    
# Set the old icd9 codes to the ICD 10 codes and update code version 
icd9['icd_code'] = mapped_codes
icd9['icd_version'] = 10

# Cat the two df back together 
mapped_icd_codes = pd.concat([icd9, icd10])

# Mapping the code to the chaper 
def icd10_to_chapter_number(code):
    # Ensuring its a valid string 
    if not isinstance(code, str) or len(code) < 1:
        return 0

    # Cast to uppercase 
    code = code.upper()

    # If the code is no diagnosis return 0 
    if code == 'NODX':
        return 0

    # Get first chart for mapping to chapter 
    first_char = code[0]

    # Try to get a code with a letter and two numbers 
    try:
        chapter_extract = int(code[1:3])
    # If that fails it a code with a letter and one number 
    except ValueError:
        chapter_extract = int(code[1:2])


    # Map first letter / number to chapter number
    if first_char in 'AB':
        return 1
    elif first_char == 'C':
        return 2
    elif first_char == 'D':
        if chapter_extract < 50:
            return 2
        else:
            return 3
    elif first_char == 'E':
        return 4
    elif first_char == 'F':
        return 5
    elif first_char == 'G':
        return 6
    elif first_char == 'H':
        if chapter_extract <= 59:
            return 7 
        else:
            return 8 
    elif first_char == 'I':
        return 9
    elif first_char == 'J':
        return 10
    elif first_char == 'K':
        return 11
    elif first_char == 'L':
        return 12
    elif first_char == 'M':
        return 13
    elif first_char == 'N':
        return 14
    elif first_char == 'O':
        return 15
    elif first_char == 'P':
        return 16
    elif first_char == 'Q':
        return 17
    elif first_char == 'R':
        return 18
    elif first_char in 'ST':
        return 19
    elif first_char in 'VYZ':
        return 20
    elif first_char == 'Z':
        return 21
    elif first_char == 'U':
        return 22
    else:
        return 0

# Map icd 10 to chapter
chapters = []
for code in mapped_icd_codes['icd_code']:
    chapter = icd10_to_chapter_number(code)
    chapters.append(chapter)
mapped_icd_codes['diagnoses_chapters'] = chapters

# Grab All ICD codes 
grouped_icd_codes = mapped_icd_codes.groupby(['subject_id', 'hadm_id'])['diagnoses_chapters'].apply(list).reset_index()

# Merge these to the main ICU DF 
diagnoses_added = age_added.merge(grouped_icd_codes[['subject_id', 'hadm_id', 'diagnoses_chapters']], on=['subject_id', 'hadm_id'], how='left')
diagnoses_added

# Reorder to keep the added features strights 
diagnoses_added['gender'] = diagnoses_added.pop('gender')
diagnoses_added['diagnoses_chapters'] = diagnoses_added.pop('diagnoses_chapters')
diagnoses_added['first_careunit'] = diagnoses_added.pop('first_careunit')
diagnoses_added['admission_type'] = diagnoses_added.pop('admission_type')
diagnoses_added['admission_location'] = diagnoses_added.pop('admission_location')

# Loading the tables with the vital values 
vital_events = pd.read_csv(os.path.join(mimic_dataset_dir, r"icu\chartevents.csv.gz"), usecols=['subject_id', 'hadm_id', 'stay_id', 'itemid', 'charttime', 'valuenum'])

# Loading dim tables to map id to feature name
d_vital_items = pd.read_csv(os.path.join(mimic_dataset_dir, r"icu\d_items.csv.gz"))

# Specifying what features we are interested in 
vital_feature_names = ['Heart Rate', 'O2 saturation pulseoxymetry', 'Respiratory Rate', 'Temperature Fahrenheit', 'GCS - Eye Opening', 'GCS - Motor Response', 'GCS - Verbal Response']
# Grabing those dims 
d_vital_items = d_vital_items[d_vital_items['label'].isin(vital_feature_names)] 

# Mapping the events ids to the label 
vital_events = vital_events.merge(d_vital_items[['itemid', 'label']], on='itemid')

# Merging the vital events wtih the icu stays 
vitals_added = diagnoses_added.merge(vital_events, on=['subject_id', 'hadm_id', 'stay_id'], how='left')
# Convert the charttime to datatime
vitals_added['charttime'] = pd.to_datetime(vitals_added['charttime'])

# Calculating how long ago the intime was from the charttime 
time_diff = vitals_added['charttime'] - vitals_added['intime']

# Taking only vital signs that were within the first 24 hours of the ICU stay 
vitals_added = vitals_added[time_diff <= pd.Timedelta(days=1)]

# Aggregating all within the first 24 hours 
agg_vitals = vitals_added.groupby(['subject_id', 'hadm_id', 'stay_id', 'label'])['valuenum'].agg(agg_function).reset_index()

# Pivot the table to group by ['subject_id', 'hadm_id', 'stay_id'] and add new label columns with valuenum values 
pivot_vitals = agg_vitals.pivot_table(index=['subject_id', 'hadm_id', 'stay_id'], columns='label', values='valuenum').reset_index()

# Loading the tables with the lab values
lab_events = pd.read_csv(os.path.join(mimic_dataset_dir, r"hosp\labevents.csv.gz"), usecols=['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum'])

# Loading dim tables to map id to feature name
d_lab_items = pd.read_csv(os.path.join(mimic_dataset_dir, r"hosp\d_labitems.csv.gz"))

# Specifying what features we are interested in 
lab_feature_names = ['Anion Gap', 'Bicarbonate', 'Chloride', 'Creatinine', 'Glucose', 'Sodium', 'Magnesium', 'Potassium', 'Phosphate', 'Urea Nitrogen', 'Hematocrit', 'Hemoglobin', 'MCH', 'MCHC', 'MCV', 'RDW', 'Red Blood Cells', 'White Blood Cells', 'Platelet Count']
# Grabing those dims 
d_lab_items = d_lab_items[(d_lab_items['label'].isin(lab_feature_names))]

# Mapping the events ids to the label 
lab_events = lab_events.merge(d_lab_items[['itemid', 'label']], on='itemid')

# Merging the vital events wtih the icu stays 
labs_added = diagnoses_added.merge(lab_events, on=['subject_id', 'hadm_id'], how='left')
# Convert the charttime to datatime
labs_added['charttime'] = pd.to_datetime(labs_added['charttime'])

# Calculating how long ago the intime was from the charttime 
time_diff = labs_added['charttime'] - labs_added['intime']

# Taking only vital signs that were within the first 24 hours of the ICU stay (since we did not merge on the stay_id we need to ensure the reading was taken before the ICU admitance therefore time_diff > 0 days)
labs_added = labs_added[(time_diff > pd.Timedelta(days=0)) & (time_diff <= pd.Timedelta(days=1))]

# Aggregating all within the first 24 hours 
agg_labs = labs_added.groupby(['subject_id', 'hadm_id', 'stay_id', 'label'])['valuenum'].agg(agg_function).reset_index()

if isinstance(agg_function, list):
    agg_labs.columns = ['subject_id', 'hadm_id', 'stay_id', 'label'] + [f'{func}_valuenum' for func in agg_function]
    pivot_labs = pd.concat([agg_labs.pivot(index=['subject_id', 'hadm_id', 'stay_id'], columns='label', values=f'{func}_valuenum').add_suffix(f'_{func}') for func in agg_function], axis=1).reset_index()
else:
    # Group by ['subject_id', 'hadm_id', 'stay_id'], columns labels, value valuenum
    pivot_labs = agg_labs.pivot_table(index=['subject_id', 'hadm_id', 'stay_id'], columns='label', values='valuenum').reset_index()

# Reordering the features 
final_vitals_added = diagnoses_added.merge(pivot_vitals, on=['subject_id', 'hadm_id', 'stay_id'], how='left')
final_vitals_added['Heart Rate'] = final_vitals_added.pop('Heart Rate')
final_vitals_added['O2 saturation pulseoxymetry'] = final_vitals_added.pop('O2 saturation pulseoxymetry')
final_vitals_added['Respiratory Rate'] = final_vitals_added.pop('Respiratory Rate')
final_vitals_added['Temperature Fahrenheit'] = final_vitals_added.pop('Temperature Fahrenheit')
final_vitals_added['GCS - Eye Opening'] = final_vitals_added.pop('GCS - Eye Opening')
final_vitals_added['GCS - Motor Response'] = final_vitals_added.pop('GCS - Motor Response')
final_vitals_added['GCS - Verbal Response'] = final_vitals_added.pop('GCS - Verbal Response')

# Reordering the features 
icu_stays_with_data = final_vitals_added.merge(pivot_labs, on=['subject_id', 'hadm_id', 'stay_id'], how='left')
icu_stays_with_data['Anion Gap'] = icu_stays_with_data.pop('Anion Gap')
icu_stays_with_data['Bicarbonate'] = icu_stays_with_data.pop('Bicarbonate')
icu_stays_with_data['Chloride'] = icu_stays_with_data.pop('Chloride')
icu_stays_with_data['Creatinine'] = icu_stays_with_data.pop('Creatinine')
icu_stays_with_data['Glucose'] = icu_stays_with_data.pop('Glucose')
icu_stays_with_data['Sodium'] = icu_stays_with_data.pop('Sodium')
icu_stays_with_data['Magnesium'] = icu_stays_with_data.pop('Magnesium')
icu_stays_with_data['Potassium'] = icu_stays_with_data.pop('Potassium')
icu_stays_with_data['Phosphate'] = icu_stays_with_data.pop('Phosphate')
icu_stays_with_data['Urea Nitrogen'] = icu_stays_with_data.pop('Urea Nitrogen')
icu_stays_with_data['Hematocrit'] = icu_stays_with_data.pop('Hematocrit')
icu_stays_with_data['Hemoglobin'] = icu_stays_with_data.pop('Hemoglobin')
icu_stays_with_data['MCH'] = icu_stays_with_data.pop('MCH')
icu_stays_with_data['MCHC'] = icu_stays_with_data.pop('MCHC')
icu_stays_with_data['MCV'] = icu_stays_with_data.pop('MCV')
icu_stays_with_data['RDW'] = icu_stays_with_data.pop('RDW')
icu_stays_with_data['Red Blood Cells'] = icu_stays_with_data.pop('Red Blood Cells')
icu_stays_with_data['White Blood Cells'] = icu_stays_with_data.pop('White Blood Cells')
icu_stays_with_data['Platelet Count'] = icu_stays_with_data.pop('Platelet Count')

# Getting Column names 
icu_stays_with_data.columns

make_figure_one(icu_stays_with_data)

plt.figure(figsize=(8, 6))
box_plot_features = ['Temperature Fahrenheit', 'Chloride', 'Creatinine', 'Heart Rate', 'MCV', 'O2 saturation pulseoxymetry', 'Respiratory Rate']
units = ["°F", "mEq/L", "mg/dL", "bpm", "fL", "%", "breaths/min"]

ax = sns.boxplot(data=icu_stays_with_data[box_plot_features], palette="Set2", flierprops={'marker': '.', 'markersize': 8, 'markerfacecolor': 'black', 'alpha': 0.4})

labels = [f"{col}\n({unit})" for col, unit in zip(box_plot_features, units)] 
ax.set_xticklabels(labels, rotation=45, ha='right')
plt.ylabel("Value of the measurment")
plt.ylim(0, 250)
plt.show()


# Filtering people with icu stays in between 1 and 21 days (2.2. Feature Selection)
los_in_range = icu_stays_with_data[(1 <= icu_stays_with_data['los']) & (icu_stays_with_data['los'] < 21)]

# Filtering out people who died in the hospital 
leave_icu_alive = los_in_range[los_in_range['hospital_expire_flag'] == 0]

# Filtering based on Readmission to ICU within 2 days  
# Sorting based on subject_id, and intime 
leave_icu_alive = leave_icu_alive.sort_values(['subject_id', 'intime'])
# Grouping by the patient id, adding previous icu outtime
leave_icu_alive['prev_outtime'] = (leave_icu_alive.groupby('subject_id')['outtime'].shift(1))
leave_icu_alive['prev_outtime'] = pd.to_datetime(leave_icu_alive['prev_outtime'])

## Subtract this intime - previous outline to calc how many days it has been since they last left the ICU 
leave_icu_alive['time_since_last_out'] = leave_icu_alive['intime'] - leave_icu_alive['prev_outtime']

# If the time since they last were admited to the ICU is greater than two days, consider it no Readmission (2.2. Feature Selection)
no_readmission = leave_icu_alive[(leave_icu_alive['time_since_last_out'] > pd.Timedelta(days=2)) | (leave_icu_alive['time_since_last_out'].isna())]

# Dropping columns we do not want 
no_readmission = no_readmission.drop(columns=['subject_id', 'hadm_id', 'stay_id', 'last_careunit', 'intime', 'outtime', 'hospital_expire_flag',  'prev_outtime', 'time_since_last_out', 'Heart Rhythm'], errors='ignore')

# Drop missing Values 
complete_icu_stays = no_readmission.dropna()

# Filtering out 'Heart Rate' outside of [25, 225]
in_range_data = complete_icu_stays[complete_icu_stays['Heart Rate'].between(25, 225)]

# Filtering out 'O2 saturation pulseoxymetry' outside of [7, 40]
in_range_data = in_range_data[in_range_data['O2 saturation pulseoxymetry'].between(50, 120)]

# Filtering out Respiratory Rate outside of [7, 40]
in_range_data = in_range_data[in_range_data['Respiratory Rate'].between(7, 40)]

# Filtering out Respiratory Rate outside of [86, 113]
in_range_data = in_range_data[in_range_data['Temperature Fahrenheit'].between(86, 113)]

in_range_data.to_csv(os.path.join(dataset_dir, 'mean_dataset.csv'), index=False)

# Getting features in figure 1 
corr_df = in_range_data[['RDW', 'Urea Nitrogen', 'Creatinine', 'MCV', 'los', 'GCS - Motor Response', 'MCH', 'Chloride', 'Sodium', 'Bicarbonate', 'MCHC', 'Red Blood Cells', 'Hematocrit', 'Hemoglobin']].corr()

# Mask for uper tri 
mask_upper = np.triu(np.ones_like(corr_df, dtype=bool), k=1)

# In it fig
fig, ax = plt.subplots(figsize=(10, 8))

# --- Lower triangle: numeric heatmap ---
sns.heatmap(
    corr_df,
    mask=mask_upper,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    cbar=False,
    ax=ax,
    linewidths=0.5,
    annot_kws={"size": 8}
)

# --- Upper triangle: bubble size by correlation ---
norm = Normalize(vmin=-1, vmax=1)  # match the heatmap scale
cmap = cm.get_cmap("coolwarm")

for i in range(len(corr_df)):
    for j in range(i + 1, len(corr_df)):
        corr = corr_df.iloc[i, j]
        ax.scatter(
            j + 0.5, i + 0.5,
            s=abs(corr) * 1000,
            color=cmap(norm(corr)),
            alpha=0.6,
            edgecolor="white",
            linewidth=0.5
        )

# --- Aesthetics ---
ax.set_xticklabels(corr_df.columns, rotation=45, ha='right')
ax.set_yticklabels(corr_df.columns, rotation=0)
plt.tight_layout()
plt.show()


