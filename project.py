from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import scipy.stats as stats
import warnings
import plotly.express as px
import plotly.graph_objects as go
warnings.filterwarnings('ignore')


path_fat_quantity = '/content/Fat_Supply_Quantity_Data.csv'
path_food_quantity = '/content/Food_Supply_Quantity_kg_Data.csv'
path_food_kcal = '/content/Food_Supply_kcal_Data.csv'
path_protein_quantity = '/content/Protein_Supply_Quantity_Data.csv'
path_food_description = '/content/Supply_Food_Data_Descriptions.csv'

"""# Actual code

###Q1: What are the top three factors contributing to people
getting COVID-19 in terms of diet?
By analyzing all the factors using R^2 (coefficient of determination)
and Correlation coefficients, this question can be answered by
comparing the top three most significant percentages.
"""

# read all data
df_food_supply = pd.read_csv(path_food_quantity)
num_row, num_col = df_food_supply.shape
df_food_supply_sp = df_food_supply.sample(n=136)

f, [ax1, ax2] = plt.subplots(nrows=1, ncols=2, figsize=(16, 20))

# find the pairwise correlation between all columns of the dataframe
corrs = df_food_supply.corr(method='pearson').round(3)
corrs = corrs[['Deaths', 'Confirmed', 'Recovered', 'Active']]

# set up color map
cmap = sns.diverging_palette(150, 275, s=80, l=55, as_cmap=True)

# draw a heatmap
sns.heatmap(ax=ax1, data=corrs, cmap=cmap, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, annot=True)


# test code
# find the pairwise correlation between all columns of the dataframe
corrs_sp = df_food_supply_sp.corr(method='pearson').round(3)
corrs_sp = corrs_sp[['Deaths', 'Confirmed', 'Recovered', 'Active']]

# set up color map
cmap_sp = sns.diverging_palette(150, 275, s=80, l=55, as_cmap=True)

# draw a heatmap
sns.heatmap(ax=ax2, data=corrs_sp, cmap=cmap_sp, vmax=1, vmin=-1, center=0,
            square=True, linewidths=.5, annot=True)
plt.title('Random Sample Correlation Map')

"""### Q2: Does intake of more fat or intake of more protein
help people get away from COVID-19?
We can compare the data from fat and protein intake and see the
difference in recovery rate to conclude this question.

"""

# read csv files
df_fat = pd.read_csv(path_fat_quantity)
df_protein = pd.read_csv(path_protein_quantity)

# plot the recovery rate over different countries
country_recover = px.bar(df_food_supply, x="Country", y="Recovered")
country_recover = country_recover.update_xaxes(
                  categoryorder="total descending")
country_recover.show()

# Fat Intake
# find top 5 countries with higher recovered rate
recover_top5 = list(df_food_supply.nlargest(5, 'Recovered')['Country'])
df_top1_fat = df_fat[df_fat['Country'] == recover_top5[0]]
df_top2_fat = df_fat[df_fat['Country'] == recover_top5[1]]
df_top3_fat = df_fat[df_fat['Country'] == recover_top5[2]]
df_top4_fat = df_fat[df_fat['Country'] == recover_top5[3]]
df_top5_fat = df_fat[df_fat['Country'] == recover_top5[4]]

recover_top5_fat = df_fat.nlargest(5, 'Recovered')

ingredients = ['Alcoholic Beverages', 'Animal Products', 'Animal fats',
               'Aquatic Products, Other', 'Cereals - Excluding Beer', 'Eggs',
               'Fish, Seafood', 'Fruits - Excluding Wine', 'Meat',
               'Milk - Excluding Butter', 'Offals', 'Oilcrops', 'Pulses',
               'Spices', 'Starchy Roots', 'Stimulants', 'Sugar Crops',
               'Sugar & Sweeteners', 'Treenuts', 'Vegetal Products',
               'Vegetable Oils', 'Vegetables', 'Miscellaneous']

# rearrange positions of figures
fig1 = make_subplots(
    rows=1, cols=5,
    specs=[[{"type": "pie"},
            {"type": "pie"},
            {"type": "pie"},
            {"type": "pie"},
            {"type": "pie"}]])

# draw pie charts about fat intakes of different countries
# Montenegro
fig1.add_trace(go.Pie(hole=.3,
                      labels=ingredients,
                      values=df_top1_fat[ingredients].mean().tolist()),
               row=1,
               col=1)
# Czechia
fig1.add_trace(go.Pie(hole=.3,
                      labels=ingredients,
                      values=df_top2_fat[ingredients].mean().tolist()),
               row=1,
               col=2)
# Luxembourg
fig1.add_trace(go.Pie(hole=.3,
                      labels=ingredients,
                      values=df_top3_fat[ingredients].mean().tolist()),
               row=1,
               col=3)
# Slovenia
fig1.add_trace(go.Pie(hole=.3,
                      labels=ingredients,
                      values=df_top4_fat[ingredients].mean().tolist()),
               row=1,
               col=4)
# Georgia
fig1.add_trace(go.Pie(hole=.3,
                      labels=ingredients,
                      values=df_top5_fat[ingredients].mean().tolist()),
               row=1,
               col=5)

# set up titles and height, width of the figures
fig1.update_layout(title_text='Fat intake distribution from each ingredient'
                   'of Countries with TOP5 Recovery Rate')

# Protein Intake
recover_top5 = list(df_food_supply.nlargest(5, 'Recovered')['Country'])
df_top1_pt = df_protein[df_protein['Country'] == recover_top5[0]]
df_top2_pt = df_protein[df_protein['Country'] == recover_top5[1]]
df_top3_pt = df_protein[df_protein['Country'] == recover_top5[2]]
df_top4_pt = df_protein[df_protein['Country'] == recover_top5[3]]
df_top5_pt = df_protein[df_protein['Country'] == recover_top5[4]]

recover_top5_protein = df_protein.nlargest(5, 'Recovered')

ingredients = ['Alcoholic Beverages', 'Animal Products', 'Animal fats',
               'Aquatic Products, Other', 'Cereals - Excluding Beer', 'Eggs',
               'Fish, Seafood', 'Fruits - Excluding Wine', 'Meat',
               'Milk - Excluding Butter', 'Offals', 'Oilcrops', 'Pulses',
               'Spices', 'Starchy Roots', 'Stimulants', 'Sugar Crops',
               'Sugar & Sweeteners', 'Treenuts', 'Vegetal Products',
               'Vegetable Oils', 'Vegetables', 'Miscellaneous']

# rearrange positions of figures
fig2 = make_subplots(
    rows=1, cols=5,
    specs=[[{"type": "pie"},
            {"type": "pie"},
            {"type": "pie"},
            {"type": "pie"},
            {"type": "pie"}]])

# Protein intakes
# Montenegro
fig2.add_trace(go.Pie(hole=.3, labels=ingredients,
               values=df_top1_pt[ingredients].mean().tolist()),
               row=1, col=1)
# Czechia
fig2.add_trace(go.Pie(hole=.3, labels=ingredients,
               values=df_top2_pt[ingredients].mean().tolist()),
               row=1, col=2)
# Luxembourg
fig2.add_trace(go.Pie(hole=.3, labels=ingredients,
               values=df_top3_pt[ingredients].mean().tolist()),
               row=1, col=3)
# Slovenia
fig2.add_trace(go.Pie(hole=.3, labels=ingredients,
               values=df_top4_pt[ingredients].mean().tolist()),
               row=1, col=4)
# Georgia
fig2.add_trace(go.Pie(hole=.3, labels=ingredients,
               values=df_top5_pt[ingredients].mean().tolist()),
               row=1, col=5)

# set up titles and height, width of the figures
fig2.update_layout(title_text='Protein intake distribution from each\
                   ingredient of Countries with TOP5 Recovery Rate')

# show the figures
fig1.show()
fig2.show()

df_fat.shape

# TOP 5 death rate
# plot the recovery rate over different countries
country_death = px.bar(df_food_supply, x="Country", y="Deaths")
country_death = country_death.update_xaxes(categoryorder="total descending")
country_death.show()

# Fat Intake
# find top 5 countries with higher recovered rate
deaths_top5_countries = list(df_food_supply.nlargest(5, 'Deaths')['Country'])
df_d_top1_fat = df_fat[df_fat['Country'] == deaths_top5_countries[0]]
df_d_top2_fat = df_fat[df_fat['Country'] == deaths_top5_countries[1]]
df_d_top3_fat = df_fat[df_fat['Country'] == deaths_top5_countries[2]]
df_d_top4_fat = df_fat[df_fat['Country'] == deaths_top5_countries[3]]
df_d_top5_fat = df_fat[df_fat['Country'] == deaths_top5_countries[4]]

recover_top5_fat = df_fat.nlargest(5, 'Deaths')

ingredients = ['Alcoholic Beverages', 'Animal Products', 'Animal fats',
               'Aquatic Products, Other', 'Cereals - Excluding Beer', 'Eggs',
               'Fish, Seafood', 'Fruits - Excluding Wine', 'Meat',
               'Milk - Excluding Butter', 'Offals', 'Oilcrops', 'Pulses',
               'Spices', 'Starchy Roots', 'Stimulants', 'Sugar Crops',
               'Sugar & Sweeteners', 'Treenuts', 'Vegetal Products',
               'Vegetable Oils', 'Vegetables', 'Miscellaneous']

# rearrange positions of figures
fig3 = make_subplots(
    rows=1, cols=5,
    specs=[[{"type": "pie"}, {"type": "pie"}, {"type": "pie"},
            {"type": "pie"}, {"type": "pie"}]])

# draw pie charts about fat intakes of different countries
# Belgium
fig3.add_trace(go.Pie(hole=.3,
                      labels=ingredients,
                      values=df_d_top1_fat[ingredients].mean().tolist()),
               row=1,
               col=1)
# Slovenia
fig3.add_trace(go.Pie(hole=.3,
                      labels=ingredients,
                      values=df_d_top2_fat[ingredients].mean().tolist()),
               row=1,
               col=2)
# United Kingdom
fig3.add_trace(go.Pie(hole=.3,
                      labels=ingredients,
                      values=df_d_top3_fat[ingredients].mean().tolist()),
               row=1,
               col=3)
# Czechia
fig3.add_trace(go.Pie(hole=.3,
                      labels=ingredients,
                      values=df_d_top4_fat[ingredients].mean().tolist()),
               row=1,
               col=4)
# Italy
fig3.add_trace(go.Pie(hole=.3,
                      labels=ingredients,
                      values=df_d_top5_fat[ingredients].mean().tolist()),
               row=1,
               col=5)

# set up titles and height, width of the figures
fig3.update_layout(title_text='Fat intake distribution\
                   from each ingredient of\
                   Countries with TOP5 Deaths Rate')

# Protein Intake
deaths_top5_countries = list(df_food_supply.nlargest(5, 'Deaths')['Country'])
df_d_top1_pt = df_protein[df_protein['Country'] == deaths_top5_countries[0]]
df_d_top2_pt = df_protein[df_protein['Country'] == deaths_top5_countries[1]]
df_d_top3_pt = df_protein[df_protein['Country'] == deaths_top5_countries[2]]
df_d_top4_pt = df_protein[df_protein['Country'] == deaths_top5_countries[3]]
df_d_top5_pt = df_protein[df_protein['Country'] == deaths_top5_countries[4]]

ingredients = ['Alcoholic Beverages', 'Animal Products', 'Animal fats',
               'Aquatic Products, Other', 'Cereals - Excluding Beer', 'Eggs',
               'Fish, Seafood', 'Fruits - Excluding Wine', 'Meat',
               'Milk - Excluding Butter', 'Offals', 'Oilcrops', 'Pulses',
               'Spices', 'Starchy Roots', 'Stimulants', 'Sugar Crops',
               'Sugar & Sweeteners', 'Treenuts', 'Vegetal Products',
               'Vegetable Oils', 'Vegetables', 'Miscellaneous']

# rearrange positions of figures
fig4 = make_subplots(
    rows=1, cols=5,
    specs=[[{"type": "pie"},
            {"type": "pie"},
            {"type": "pie"},
            {"type": "pie"},
            {"type": "pie"}]])

# Protein intakes
# Belgium
fig4.add_trace(go.Pie(hole=.3, labels=ingredients,
               values=df_d_top1_pt[ingredients].mean().tolist()),
               row=1, col=1)
# Slovenia
fig4.add_trace(go.Pie(hole=.3, labels=ingredients,
               values=df_d_top2_pt[ingredients].mean().tolist()),
               row=1, col=2)
# United Kingdom
fig4.add_trace(go.Pie(hole=.3, labels=ingredients,
               values=df_d_top3_pt[ingredients].mean().tolist()),
               row=1, col=3)
# Czechia
fig4.add_trace(go.Pie(hole=.3, labels=ingredients,
               values=df_d_top4_pt[ingredients].mean().tolist()),
               row=1, col=4)
# Italy
fig4.add_trace(go.Pie(hole=.3, labels=ingredients,
               values=df_d_top5_pt[ingredients].mean().tolist()),
               row=1, col=5)


# set up titles and height, width of the figures
fig4.update_layout(title_text='Protein intake distribution\
                   from each ingredient\
                   of Countries with TOP5 Death Rate')


# show the figures
fig3.show()
fig4.show()

# Test Code
# generate sample
df_food_supply_sp = df_food_supply.sample(n=136)
df_fat_sp = df_fat.sample(n=136)
df_protein_sp = df_protein.sample(n=136)

# plot the recovery rate over different countries
recover_sp = px.bar(df_food_supply_sp, x="Country", y="Recovered")
recover_sp = recover_sp.update_xaxes(categoryorder="total descending")
recover_sp.show()

# Fat Intake
# find top 5 countries with higher recovered rate
recover_top5_sp = list(df_food_supply_sp.nlargest(5, 'Recovered')['Country'])
df_top1_fat_sp = df_fat_sp[df_fat_sp['Country'] == recover_top5_sp[0]]
df_top2_fat_sp = df_fat_sp[df_fat_sp['Country'] == recover_top5_sp[1]]
df_top3_fat_sp = df_fat_sp[df_fat_sp['Country'] == recover_top5_sp[2]]
df_top4_fat_sp = df_fat_sp[df_fat_sp['Country'] == recover_top5_sp[3]]
df_top5_fat_sp = df_fat_sp[df_fat_sp['Country'] == recover_top5_sp[4]]

recover_top5_fat_sp = df_fat_sp.nlargest(5, 'Recovered')

ingredients = ['Alcoholic Beverages', 'Animal Products', 'Animal fats',
               'Aquatic Products, Other', 'Cereals - Excluding Beer', 'Eggs',
               'Fish, Seafood', 'Fruits - Excluding Wine', 'Meat',
               'Milk - Excluding Butter', 'Offals', 'Oilcrops', 'Pulses',
               'Spices', 'Starchy Roots', 'Stimulants', 'Sugar Crops',
               'Sugar & Sweeteners', 'Treenuts', 'Vegetal Products',
               'Vegetable Oils', 'Vegetables', 'Miscellaneous']

# rearrange positions of figures
fig1_sp = make_subplots(
    rows=1, cols=5,
    specs=[[{"type": "pie"},
            {"type": "pie"},
            {"type": "pie"},
            {"type": "pie"},
            {"type": "pie"}]])

# draw pie charts about fat intakes of different countries
# Montenegro
fig1_sp.add_trace(go.Pie(hole=.3,
                         labels=ingredients,
                         values=df_top1_fat_sp[ingredients].mean().tolist()),
                  row=1,
                  col=1)
# Czechia
fig1_sp.add_trace(go.Pie(hole=.3,
                         labels=ingredients,
                         values=df_top2_fat_sp[ingredients].mean().tolist()),
                  row=1,
                  col=2)
# Luxembourg
fig1_sp.add_trace(go.Pie(hole=.3,
                         labels=ingredients,
                         values=df_top3_fat_sp[ingredients].mean().tolist()),
                  row=1,
                  col=3)
# Slovenia
fig1_sp.add_trace(go.Pie(hole=.3,
                         labels=ingredients,
                         values=df_top4_fat_sp[ingredients].mean().tolist()),
                  row=1,
                  col=4)
# Georgia
fig1_sp.add_trace(go.Pie(hole=.3,
                         labels=ingredients,
                         values=df_top5_fat_sp[ingredients].mean().tolist()),
                  row=1,
                  col=5)

# set up titles and height, width of the figures
fig1_sp.update_layout(title_text='Fat intake distribution of Sample Test Data')

# Protein Intake
recover_top5_sp = list(df_food_supply_sp.nlargest(5, 'Recovered')['Country'])
df_top1_pt_sp = df_protein_sp[df_protein_sp['Country'] == recover_top5_sp[0]]
df_top2_pt_sp = df_protein_sp[df_protein_sp['Country'] == recover_top5_sp[1]]
df_top3_pt_sp = df_protein_sp[df_protein_sp['Country'] == recover_top5_sp[2]]
df_top4_pt_sp = df_protein_sp[df_protein_sp['Country'] == recover_top5_sp[3]]
df_top5_pt_sp = df_protein_sp[df_protein_sp['Country'] == recover_top5_sp[4]]

recover_top5_protein_sp = df_protein_sp.nlargest(5, 'Recovered')

ingredients = ['Alcoholic Beverages', 'Animal Products', 'Animal fats',
               'Aquatic Products, Other', 'Cereals - Excluding Beer', 'Eggs',
               'Fish, Seafood', 'Fruits - Excluding Wine', 'Meat',
               'Milk - Excluding Butter', 'Offals', 'Oilcrops', 'Pulses',
               'Spices', 'Starchy Roots', 'Stimulants', 'Sugar Crops',
               'Sugar & Sweeteners', 'Treenuts', 'Vegetal Products',
               'Vegetable Oils', 'Vegetables', 'Miscellaneous']

# rearrange positions of figures
fig2_sp = make_subplots(
    rows=1, cols=5,
    specs=[[{"type": "pie"},
            {"type": "pie"},
            {"type": "pie"},
            {"type": "pie"},
            {"type": "pie"}]])

# Protein intakes
# Montenegro
fig2_sp.add_trace(go.Pie(hole=.3,
                         labels=ingredients,
                         values=df_top1_pt_sp[ingredients].mean().tolist()),
                  row=1,
                  col=1)
# Czechia
fig2_sp.add_trace(go.Pie(hole=.3,
                         labels=ingredients,
                         values=df_top2_pt_sp[ingredients].mean().tolist()),
                  row=1,
                  col=2)
# Luxembourg
fig2_sp.add_trace(go.Pie(hole=.3,
                         labels=ingredients,
                         values=df_top3_pt_sp[ingredients].mean().tolist()),
                  row=1,
                  col=3)
# Slovenia
fig2_sp.add_trace(go.Pie(hole=.3,
                         labels=ingredients,
                         values=df_top4_pt_sp[ingredients].mean().tolist()),
                  row=1,
                  col=4)
# Georgia
fig2_sp.add_trace(go.Pie(hole=.3,
                         labels=ingredients,
                         values=df_top5_pt_sp[ingredients].mean().tolist()),
                  row=1,
                  col=5)

# set up titles and height, width of the figures
fig2_sp.update_layout(title_text='Protein intake distribution of'
                      ' Sample Test Data')

# show the figures
fig1_sp.show()
fig2_sp.show()

"""### Q3: Hypothesis test with 2 sample t test"""

# group the data to 2 gourps: high recovery rate/low recovery rate

# read the data
# u_random = True
df = pd.read_csv(path_food_kcal)
df.dropna(inplace=True)
# find the median
median_rec = df['Recovered'].median()
# split it to half and store it to different variable
high_data = df[df['Recovered'] > median_rec]
high_data['Undernourished'] = high_data['Undernourished'].replace('<2.5', 2.5)
high = high_data['Undernourished'].astype(float)
# randomly select 50 data set from high
high = high.sample(n=50)
print(high)
low_data = df[df['Recovered'] < median_rec]
low_data['Undernourished'] = low_data['Undernourished'].replace('<2.5', 2.5)
low = low_data['Undernourished'].astype(float)
# randomly select 50 data set from low
low = low.sample(n=50)
print(low)
# find mean and SD
high_mean = high.mean()
high_mean = "{:.4f}".format(high_mean)
low_mean = low.mean()
low_mean = "{:.4f}".format(low_mean)
high_sd = statistics.stdev(high)
high_sd = "{:.4f}".format(high_sd)
low_sd = statistics.stdev(low)
low_sd = "{:.4f}".format(low_sd)
n_high = len(high)
n_low = len(low)
# make a 3X2 table
data = np.array([['Mean(x̄)', high_mean, low_mean],
                 ['Standard Deviation(s)', high_sd, low_sd],
                 ['Data size(n)', n_high, n_low]])

# pass column names in the columns parameter
df = pd.DataFrame(data, columns=['Statistic',
                                 'High Undernourished',
                                 'Low Undernourished'])
print(df)

# place hypothesis
print("H_0: µ_high = µ_low")
print("H_A: µ_high ≠ µ_low")

# checking conditions
sns.set()

# Random Condition
print("The 50 sample uses Python Random Sample generator to selected from each\
group")

# Independent Condition
if len(high) >= 30 and len(low) >= 30:
    print(True)
else:
    print("Condition check failed, we cannot conduct a 2 sample t test")

# Normal Conditions
fig, axs = plt.subplots(ncols=2)

sns.histplot(data=high, kde=True, ax=axs[0])
sns.histplot(data=low, kde=True, ax=axs[1])
fig.tight_layout()
plt.show()

# calculate testing statistic
result = stats.ttest_ind(a=high, b=low, equal_var=True)

print(result)
