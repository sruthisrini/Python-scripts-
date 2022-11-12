#!/usr/bin/env python
# coding: utf-8

# # Calculating mean length of Site_IDs

# In[25]:


# importing the necessary packages
import pandas as pd
import os
import pyranges as pr
import xlsxwriter

# creating a new workbook
workbook = xlsxwriter.Workbook('Mean_K562.xlsx')

# adding a new sheet to the workbook
worksheet = workbook.add_worksheet()

# adding two column names 'Site_ID','Mean'
worksheet.write('A1', 'Site_ID')
worksheet.write('B1', 'Mean')

# iterating over the bed files in K562 folder
os.chdir(r'D:\University of Freiburg\Semester-4\project 2\K562')
for i,files in enumerate(os.listdir(r"D:\University of Freiburg\Semester-4\project 2\K562")):
    path=pr.get_example_path(r"D:\University of Freiburg\Semester-4\project 2\K562/"+str(files))

# reading the bed file
    bed_file=pr.read_bed(path)

# converting the bed file to csv for manipulating the data
    bed_file.to_csv(files+".csv")

# reading the bed file in csv format
    bed_file=pd.read_csv(files+".csv")
    
    filename=bed_file["Name"].unique()[0]

# identifying the "." in the "Name" column of the bed file
    if (filename==".") or (filename==""):
        print(files)

# adding numbers at the end of each site ID to make it unique
    modified_name=[]
    for index,name in enumerate(bed_file["Name"]):
        modified_name.append(name+"_"+str(index+1))
    
    bed_file["Name"]=modified_name
    
    bed_file.to_csv(filename+".csv")

# calculating the length of each site
    length=bed_file["End"]-bed_file["Start"]
    mean_value=length.mean()

# adding the site ID and mean to the worksheet
    worksheet.write('A'+str(i+2),filename)
    worksheet.write('B'+str(i+2),mean_value)

workbook.close()


# In[2]:


import pandas as pd
mean_K562_df=pd.read_excel(r'D:\University of Freiburg\Semester-4\project 2\K562\Mean_K562.xlsx')
mean_K562_df

