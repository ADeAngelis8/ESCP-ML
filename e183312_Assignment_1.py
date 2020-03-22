import pandas as pd
import numpy as np
from pandas import DataFrame

df= pd.read_csv('TitanicTrain.csv',sep=',')

#Total number of passangers
number_passangers = len(df)
print('1.1 Total number of passangers are: ',number_passangers)

#Number of passangers who survived
survived_passangers = len(df[df.Survived==1])
print('1.2 Number of survived passangers are: ',survived_passangers)

#Number of passangers who did not survive
not_survived = len(df[df.Survived==0])
print('1.3 Number of passangers who did not survive are: ',not_survived)

#Number of female passangers who survived
female_survived = len(df[(df.Survived == 1) & (df.Sex == 'female')])
female_died = len(df[(df.Survived == 0) & (df.Sex == 'female')])
print ('2.1 Number of females who survived: ',female_survived)
print('2.2 Number of females who did not survive: ',female_died)

#Number of children on titanic
number_children = len(df[df.Age<17])
print('3. Total number of children on board: ',number_children)

#Number of children who died that were on the ship
dead_children = len(df[(df.Survived == 1) & (df.Age < 17)])
print('4. Total number of children who did not survive: ',dead_children)

#People with families
df['family'] = df['SibSp'] + df['Parch']
people_with_Families = len(df[df.family != 0])
print('5. Number of people with families: ',people_with_Families)

#Ratio of male:female
sexGrouped = df.groupby('Sex')
males = sexGrouped.get_group('male')
females = sexGrouped.get_group('female')
print ("6. Ratio of males to females is: " + str(round(100*(len(males)/number_passangers))) + ":" + str(round(100*(len(females)/number_passangers))))

#Survival factors
#1. Children vs Adults
print('------------------------------------------------------')
print('Survival factors')
print('1: Survival - Children vs Adults')
no_age = number_passangers -(number_children + len(df[df.Age>16]))
adults = (df[df.Age>16])
children = df[df.Age<17]

survivedGrouped = df.groupby('Survived')
survivors = survivedGrouped.get_group(1)
deceased = survivedGrouped.get_group(0)

childrenSurvivors = children.merge(survivors, on='PassengerId', how='inner')
childrenDeceased = children.merge(deceased, on='PassengerId', how='inner')
adultSurvivors = adults.merge(survivors, on='PassengerId', how='inner')
adultDeceased = adults.merge(deceased, on='PassengerId', how='inner')

print('Number of children who boarded Titanic:', number_children)
print ("Number of children who survived: " + str(len(childrenSurvivors)))
print ("Number of children who died: " + str(len(childrenDeceased)))
print('Number of adults who boarded Titanic:', number_passangers-(number_children + no_age))
print ("Number of adults who survived: " + str(len(adultSurvivors)))
print ("Number of adults who died: " + str(len(adultDeceased)))

print ("Ratio of children survived:deceased is " + str(round(100*(len(childrenSurvivors)/number_children))) + ":" + str(round(100*(len(childrenDeceased)/number_children))))
print ("Ratio of adults survived:deceased is " + str(round(100*(len(adultSurvivors)/len(adults)))) + ":" + str(round(100*(len(adultDeceased)/len(adults)))))

print('------------------------------------------------------')

print('2: Survival - Male vs Female')
sexGrouped = df.groupby('Sex')
males = sexGrouped.get_group('male')
females = sexGrouped.get_group('female')

maleSurvivors = males.merge(survivors, on='PassengerId', how='inner')
maleDeceased = males.merge(deceased, on='PassengerId', how='inner')
femaleSurvivors = females.merge(survivors, on='PassengerId', how='inner')
femaleDeceased = females.merge(deceased, on='PassengerId', how='inner')
male= len(males)
female=len(females)

print ("Number of men who survived: " + str(len(maleSurvivors)))
print ("Number of men who died: " + str(len(maleDeceased)))
print ("Number of women who survived: " + str(len(femaleSurvivors)))
print ("Number of women who died: " + str(len(femaleDeceased)))

print ("Ratio of males survived:deceased is " + str(round(100*len(maleSurvivors)/male)) + ":" + str(round(100*(len(maleDeceased)/male))))
print ("Ratio of females survived:deceased is " + str(round(100*len(femaleSurvivors)/female)) + ":" + str(round(100*(len(femaleDeceased)/female))))

print('------------------------------------------------------')

print('3: Survival - Fare')
fares = df.Fare
survivorFares = survivors.Fare
deceasedFares = deceased.Fare
print ('Survivor fares:', survivorFares.describe())
print ('Deceased fares:', deceasedFares.describe())

print('------------------------------------------------------')

print('4: Survival - PClass')
print(df[['Pclass','Survived']].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived', ascending=False))

print('------------------------------------------------------')

print('5: Survival - Family')
print(df[['family','Survived']].groupby(['family'],as_index=False).mean().sort_values(by='Survived',ascending=False))

print('------------------------------------------------------')

print('6: Survival - Embarkment')
table = pd.crosstab(df['Survived'],df['Embarked'])
print (table)
