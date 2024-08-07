# Standard libs
import os
import math
import csv
# Data 
import pandas as pd
import numpy as np
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic

# Configure seaborn styling
sns.set(style="whitegrid")

# Global
path = "data/clean/adult_cleaned_3.csv"
global data = pd.read_csv(path)

income_mapping = {'<=50K': 'Below 50k', '>50K': 'Above 50k'}
data.loc[:, 'salary_level'] = data['income'].map(income_mapping)

def age_salary():
  # Categorize salary into above and below $50,000 analyze via age
  print(data['income'].unique())
  income_mapping = {'<=50K': 'Below 50k', '>50K': 'Above 50k'}
  data['salary_level'] = data['income'].map(income_mapping)
  print(data['income'].unique())
  
  plt.figure(figsize=(10, 6))
  sns.histplot(data, x='age', hue='salary_level', multiple='stack')
  plt.title('Age Distribution by Salary Level')
  plt.xlabel('Age')
  plt.ylabel('Count')
  plt.show()

def age_gender_salary():
  # Create separate histograms for males and females to better visualize the distrbution for age v salary 
  g = sns.FacetGrid(data, col="sex", hue="salary_level", height=5, aspect=1.5, palette='coolwarm', margin_titles=True)
  g.map(sns.histplot, 'age', multiple='stack', bins=30)
  g.add_legend(title="Salary Level")
  g.set_axis_labels("Age", "Count")
  g.set_titles("Gender: {col_name}")
  plt.subplots_adjust(top=0.85)
  g.fig.suptitle('Age Distribution by Salary Level, Split by Gender')
  plt.show()

def gender_salary():
  # identify gender distribution by salary level
  plt.figure(figsize=(8, 5))
  sns.countplot(data=data, x='sex', hue='salary_level')
  plt.title('Gender Distribution by Salary Level')
  plt.xlabel('Gender')
  plt.ylabel('Count')
  plt.show()

def education_salary():
  # identify education level distribution trends for salary level
  plt.figure(figsize=(12, 6))
  sns.countplot(data=data, x='education', hue='salary_level', order=data['education'].value_counts().index)
  plt.title('Education Level Distribution by Salary Level')
  plt.xlabel('Education Level')
  plt.xticks(rotation=45)
  plt.ylabel('Count')
  plt.show()

def education_gender_salary():
  data['income_binary'] = data['income'].apply(lambda x: 1 if x == '>50K' else 0)
  education_gender_income = data.groupby(['education', 'sex'])['income_binary'].mean().reset_index()
  
  
  plt.figure(figsize=(14, 6))
  sns.barplot(x='education', y='income_binary', hue='sex', data=education_gender_income)
  plt.title('Proportion of Individuals Earning >50K by Education Level and Gender')
  plt.xlabel('Education Level')
  plt.ylabel('Proportion Earning >50K')
  plt.xticks(rotation=45)
  plt.legend(title='Gender')
  plt.tight_layout()
  plt.show()

def age_education_salary():
  plt.figure(figsize=(14, 8))
  sns.boxplot(data=data, x='education', y='age', hue='salary_level')
  plt.title('Age and Education Level Distribution by Salary Level')
  plt.xlabel('Education Level')
  plt.ylabel('Age')
  plt.xticks(rotation=45)
  plt.show()

def marital_status_salary_mosaic():
  # Create a mosaic plot associating marital status to earning less than or greater than 50K
  currently_married = ['Married-civ-spouse', 'Married-AF-spouse', 'Married-spouse-absent']
  data["binary-marital-status"] = data["marital-status"].apply(lambda x: "currently-married" if x in currently_married else "currently-unmarried")
  fig, ax = plt.subplots(figsize=(15, 10))
  mosaic(data, ["binary-marital-status", "salary"], title="Marital-Status and Earnings over 50K", ax=ax)
  plt.show()

def marital_status_gender_salary():
  temp = data.groupby(['binary-marital-status', 'sex'])['salary'].value_counts().to_frame(name='count').reset_index()
  # Calculate the total count for each group of binary-marital-status and sex
  group_totals = temp.groupby(['binary-marital-status', 'sex'])['count'].transform('sum')
  
  # Calculate the proportion of each count within its group
  temp['proportion'] = temp['count'] / group_totals
  
  # Grouped Bar Chart
  sns.catplot(
      data=temp, kind="bar",
      x="binary-marital-status", y="count", hue="salary",
      col="sex", ci=None, height=5, aspect=1, palette=sns.color_palette("hls")
  )
  plt.subplots_adjust(top=0.9)
  plt.suptitle('Count of Salary Levels by Marital Status and Sex')
  plt.show()

def marital_status_age_salary_heatmap():
  # Bin customer age data into 7 bins
  temp = data.copy()
  temp["age-bins"] = pd.cut(temp["age"], 7)
  # Create frame by aggregating salary count by age and marital status
  temp = temp.groupby(["age-bins", "binary-marital-status"])["salary"].value_counts().to_frame(name='count').reset_index()
  temp['age-bins'] = temp['age-bins'].apply(lambda x: f"Age {round(x.left)} to Age {round(x.right)}") # Pandas Interval object contains a left, and right
  # Calculate the total count for each group of binary-marital-status and sex
  group_totals = temp.groupby(['age-bins', 'binary-marital-status'])['count'].transform('sum')
  
  # Calculate the proportion of each count within its group
  temp['proportion'] = temp['count'] / group_totals
  
  # Pivot the data to get it into a format suitable for a heatmap
  heatmap_data = temp.pivot_table(index=['age-bins', 'binary-marital-status'], columns='salary', values='proportion', aggfunc='mean')
  
  # Create the heatmap
  plt.figure(figsize=(12, 8))
  sns.heatmap(heatmap_data, annot=True, cmap='rocket_r', fmt='.2f', linewidths=.5)
  
  # Set plot labels and title
  plt.title('Proportion Heatmap by Age Range, Marital Status, and Salary')
  plt.xlabel('Salary')
  plt.ylabel('Age and Marital Status')
  
  # Show the plot
  plt.show()

def categorical_descriptions_salary():
  # Group by every category and find the most common descriptions for workers making over 50K
  binned_customer_df = data.copy()
  # Bin the age and hours-per-week to cast a wider net
  binned_customer_df['age'] = pd.cut(binned_customer_df['age'], 7)
  binned_customer_df['hours-per-week'] = pd.cut(binned_customer_df['hours-per-week'], 8)
  binned_customer_df['age'] = binned_customer_df['age'].apply(lambda x: f"Age {round(x.left)} to Age {round(x.right)}") # Pandas Interval object contains a left, and right
  binned_customer_df['hours-per-week'] = binned_customer_df['hours-per-week'].apply(lambda x: f"{round(x.left)} to {round(x.right)} hours")
  
  # Compile the individuals description
  binned_customer_df['aggregated-desc'] = binned_customer_df.apply(lambda x: x['workclass'] + ' sector ' + x['education'] + ' ' + x['age'] + ' years old ' + x['binary-marital-status'] + ' ' + x['occupation'] + ' ' + x['race'] + ' ' + x['sex'] + ' working ' + x['hours-per-week'] + '/week ', axis=1)
  temp = binned_customer_df.groupby(['aggregated-desc'])['salary'].value_counts().to_frame(name='count').reset_index()
  
  # Calculate the total count for each group of binary-marital-status and sex
  temp['desc-total'] = temp.groupby(['aggregated-desc'])['count'].transform('sum')
  temp = temp[temp['salary'] == "<=50K"].sort_values('count', ascending=False).reset_index()
  
  # Initialize the matplotlib figure
  f, ax = plt.subplots(figsize=(6, 10))
  
  # Plot the total
  sns.set_color_codes("pastel")
  sns.barplot(x="desc-total", y="aggregated-desc", data=temp[:10],
              label="Total", color="g")
  
  # Plot the counts over 50K
  sns.set_color_codes("muted")
  sns.barplot(x="count", y="aggregated-desc", data=temp[:10],
              label="<=50K", color="seagreen")
  
  # Add a legend and informative axis label
  plt.title("Top 10 Categories Making <=$50K Salary")
  ax.legend(ncol=2, loc="lower right", frameon=True)
  ax.set(xlim=(0, temp.loc[:, 'desc-total'].max()), ylabel="",
         xlabel="")
  sns.despine(left=True, bottom=True)

def occupation_salary():
  plt.figure(figsize=(14, 7))
  sns.countplot(data=data, x='occupation', hue='salary_level', palette='viridis')
  plt.title('Occupation vs. Income Level')
  plt.xticks(rotation=90)
  plt.xlabel('Occupation')
  plt.ylabel('Count')
  plt.legend(title='Income Level')
  plt.tight_layout()
  plt.show()

def race_salary():
  plt.figure(figsize=(10, 6))
  sns.countplot(data=data, x='race', hue='salary_level', palette='viridis')
  plt.title('Race vs. Income Level')
  plt.xlabel('Race')
  plt.ylabel('Count')
  plt.legend(title='Income Level')
  plt.tight_layout()
  plt.show()

def occupation_race_education_salary():
  # Categorize races into "White" and "Non-White"
  data['race_category'] = data['race'].apply(lambda x: 'White' if x == 'White' else 'Non-White')
  
  # Categorize education levels into "college" and "non-college"
  college_education = ['Bachelors', 'Some-college', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', 'Masters', 'Doctorate']
  data['education_category'] = data['education'].apply(lambda x: 'College' if x in college_education else 'Non-College')
  
  # Create frame by aggregating salary count by occupation, race category, and education category
  temp = data.groupby(["occupation", "race_category", "education_category"])["salary_level"].value_counts().to_frame(name='count').reset_index()
  
  # Calculate the total count for each group of race_category, education_category, and occupation
  group_totals = temp.groupby(['occupation', 'race_category', 'education_category'])['count'].transform('sum')
  
  # Calculate the proportion of each count within its group
  temp['proportion'] = temp['count'] / group_totals
  
  # Pivot the data to get it into a format suitable for a heatmap
  heatmap_data = temp.pivot_table(index=['occupation', 'race_category', 'education_category'], columns='salary_level', values='proportion', aggfunc='mean')
  
  # Plot the heatmap
  plt.figure(figsize=(20, 15))
  sns.heatmap(heatmap_data, annot=True, cmap='viridis', fmt='.2f', linewidths=.5)
  
  # Set plot labels and title
  plt.title('Proportion of Income Levels by Occupation, Race Category, and Education (College vs Non-College)')
  plt.xlabel('Salary Level')
  plt.ylabel('Occupation, Race Category, and Education')
  plt.tight_layout()
  plt.show()

def occupation_race_education_salary_catplot():
  # Create frame by aggregating salary count by occupation, race category, and education category
  temp = data.groupby(["occupation", "race_category", "education_category"])["salary_level"].value_counts().to_frame(name='count').reset_index()
  
  # Calculate the total count for each group of race_category, education_category, and occupation
  group_totals = temp.groupby(['occupation', 'race_category', 'education_category'])['count'].transform('sum')
  
  # Calculate the proportion of each count within its group
  temp['proportion'] = temp['count'] / group_totals
  
  # Create side-by-side bar plots
  g = sns.catplot(data=temp, x='occupation', y='proportion', hue='salary_level', col='education_category', row='race_category', kind='bar', height=7, aspect=2.5, palette='viridis')
  
  # Adjust x-tick labels
  for ax in g.axes.flat:
      for label in ax.get_xticklabels():
          label.set_rotation(45)
          label.set_ha('right')
          label.set_fontsize(24)
  
  # Set titles and labels
  for ax in g.axes.flat:
      ax.set_title(ax.get_title(), fontsize=24)
      ax.set_xlabel(ax.get_xlabel(), fontsize=24)
      ax.set_ylabel(ax.get_ylabel(), fontsize=24)
  g.set_axis_labels("Occupation", "Proportion",fontsize=24)
  
  # Adjust layout and title
  plt.subplots_adjust(top=0.9)
  g.fig.suptitle('Proportion of Income Levels by Occupation, Race, and Education (College vs Non-College)', fontsize=24)
  plt.show()

def relation_education_num_salary():
    newdata = data.copy()
    column = "relationship"
    filtercolumn = "education-num"
        
    Q1 = newdata[filtercolumn].quantile(0.25)
    Q3 = newdata[filtercolumn].quantile(0.75)
    # Compute IQR
    IQR = Q3 - Q1
    # Determine the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Find outliers
    newdata = newdata[(newdata[filtercolumn] >= lower_bound) & (newdata[filtercolumn] <= upper_bound)]

    print(filtercolumn, " range is:", lower_bound, upper_bound)

    #print(f"unfilter---> {column} :\n {data[column].value_counts()}")
    #print(f"FilteredData---> {column} :\n {newdata[column].value_counts()}")
    
    plt.figure(figsize=(8, 4))
    sns.histplot(newdata, x=column, hue='salary', multiple='stack')
    plt.title(f'Bar Chart of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability if needed
    plt.show()
    
    plt.figure(figsize=(8, 4))
    sns.histplot(newdata, x=filtercolumn, hue='salary', multiple='stack')
    plt.title(f'Bar Chart of {filtercolumn}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability if needed
    plt.show()
  
    def relationship(value):
        if value in [" Husband", " Wife"]:
            return "Husband+Wife"
        else:
            return "Other"
        
    def education_level(value):
        if value < 9:
            return "Basic"
        elif 9 <= value < 13:
            return "Qualified"
        elif 13 <= value:
            return "Highly Qualified"

    newdata['education-level'] = newdata['education-num'].map(education_level)
    newdata['relationship-level'] = newdata['relationship'].map(relationship)

    fig = px.parallel_categories(newdata, dimensions=['relationship-level', 'education-level', 'salary'])
    fig.show()

def education_level_United_States():
  usa_data = data[data['native-country'] == 'United States of America']
  data['combined'] = data['sex'] + ' with ' + data['salary']
  grouped_data = usa_data.groupby(['education', 'combined']).size().reset_index(name='Count')
  plt.figure(figsize=(12, 8))
  sns.lineplot(x='education', y='Count', hue='combined', data=grouped_data, palette='viridis')
  
  # Adding titles and labels
  plt.title('Number of People by Education Level, Salary Category, and Gender in United States of America')
  plt.xlabel('Education Level')
  plt.ylabel('Number of People')
  
  # Rotate x-axis labels for better readability
  plt.xticks(rotation=45)
  
  plt.legend(title='Sex and Salary')
  plt.tight_layout()
  
  # Show plot
  plt.show()

def salary_native_country_choropleth():
  categorical_attributes = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"] # These attributes are categorical data
  world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

  name_mapping = {
      'United-States': 'United States of America',
      'Russia': 'Russian Federation',
      'Bolivia': 'Bolivia (Plurinational State of)',
      'Vietnam': 'Viet Nam',
  }
  
  data['native-country'] = data['native-country'].replace(name_mapping)
  
  filtered_data = data[data['salary'] == '>50K']
  
  aggregated_data = filtered_data.groupby('native-country')['salary'].count().reset_index()
  
  merged_data = world.merge(aggregated_data, how='left', left_on='name', right_on='native-country')
  
  merged_data['salary'] = merged_data['salary'].fillna(0)
  
  country_to_exclude = 'United States of America'
  
  exclude_df = merged_data[merged_data['name'] == country_to_exclude]
  
  excluded_data = merged_data[merged_data['name'] != country_to_exclude]
  
  added_data = excluded_data.groupby('name').sum()
  
  exclude_df['salary'] = 0
  combined_data = pd.concat([excluded_data, exclude_df])

  def create_choropleth(data, column, title, cmap='OrRd'):
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    data.plot(column=column, 
              cmap=cmap, 
              linewidth=0.8, 
              ax=ax, 
              edgecolor='0.8')
    vmin = data[column].min()
    vmax = data[column].max()
    bins = np.linspace(vmin, vmax, 5)  # Adjust the number of bins as needed
    labels = [f'{int(bins[i])} - {int(bins[i+1])}' for i in range(len(bins)-1)]
    
    cmap = plt.get_cmap(cmap)
    norm = plt.Normalize(vmin, vmax)
    patches = [mpatches.Patch(color=cmap(norm((bins[i] + bins[i + 1]) / 2)), label=labels[i]) for i in range(len(labels))]
    
    ax.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5), title='Legend Title', fontsize='small')
    plt.title(title)
    plt.show()
    
  create_choropleth(merged_data, 'salary', 'People earning more than 50K in the world')
  create_choropleth(combined_data, 'salary', 'People earning more than 50K excluding USA')

def main():
  # Plot age distribution according to salary outcomes
  age_salary()

  # Plot age distribution by salary seperated by gender
  age_gender_salary()

  # Plot gender by salary
  gender_salary()

  # Plot salary by education
  education_salary()

  # Plot salary by education and gender
  education_gender_salary()

  # Plot salary by age and education
  age_education_salary()

  # Mosaic Plot for marital status and salary
  marital_status_salary_mosaic()

  # Heat map showing age and marital status with salary
  marital_status_age_salary_heatmap()

  # Plot the top ten descriptions of people making less than the target
  categorical_descriptions_salary()

  # Plot occupation by salary outcomes
  occupation_salary()

  # Plot race by salary outcomes
  race_salary()

  # Plot salary by occupation race and education
  occupation_race_education_salary()

  # Categorical plot for occupation, race, education and salary
  occupation_race_education_salary_catplot()

  # Plots for education, relation and salary
  relation_education_num_salary()

  # Plot the education level distribution for the United States
  education_level_United_States()

  # Plot the choropleth map for salary by native country
  salary_native_country_choropleth()


if __name__ == "__main__":
  main()

