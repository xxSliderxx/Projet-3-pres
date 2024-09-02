import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import seaborn as sns
import re
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.decomposition import PCA
from io import BytesIO
import base64




def ana():
    @st.cache_data
    def Enedis():
        st.header("General information", divider='rainbow')
        st.markdown(" ")
        st.markdown("""
                - This graphic gives us an overview of total electricity consumption in 2 regions in 24 months. The purpose is not to compare the indicators of 2 areas but to observe and analyse the elements which could be important for our model after. 

                - The region HDF has 3,3 millions more inhabitants  than the region CVDL (2,39 times) but its total electricity consumption is 1.8 times higher than CVDL's.  In the other hand, HDF consums 7% less electricity for heating than CVDL. From this perspective, the size of house/ appartement, type of heating energy should be considered as very important factors in electricity consumption.                 
                                """)

        # Chart of total consumption
        
        Holiday = ['période 1','période 2','période 3','période 4','période 5','période 6','période 7','période 8','période 9']
        Vacances = {'période 1':'blue','période 2':'blue','période 3':'blue','période 4':'blue','période 5':'blue','période 6':'blue',
                        'période 7':'blue','période 8':'blue','période 9':'blue','période 20':'green','période 21':'green',
                        'période 22':'green','période 23':'green','période 24':'green','période 25':'green','période 26':'green',
                        'période 27':'green','période 28':'green','période 29':'green','période 30':'green','période 31':'green'}
        
        Jour_ferie = ['2022-04-18','2022-05-01','2022-05-08','2022-05-26','2022-06-06','2022-07-14','2022-08-15','2022-11-01','2022-11-11','2022-12-25',
                          '2023-01-01','2023-04-10','2023-05-01','2023-05-08','2023-05-29','2023-07-14','2023-08-15','2023-11-01','2023-11-11','2023-12-25',
                          '2024-01-01','2024-04-01']
        
        dg = pd.read_csv('dg.csv')
        
        
        
        # Set up a dictionary of font title
        font_title = {'family': 'sans-serif',
                                'color':  '#114b98',
                                'fontweight': 'bold'}
        
        
        fig, ax = plt.subplots(2,1,figsize =(12,12))
        ax1 =sns.barplot(data = dg, x = 'Région', y='Total énergie soutirée (MWh)',  errorbar=None,ax=ax[0])
        for container in ax1.containers:
                ax1.bar_label(container, label_type="center", fmt="{:.0f} MWH",
                                color='#ffee78', fontsize=12, fontweight ='bold')

        ax1.set_title("Average electricity consumption per day", fontdict=font_title, fontsize = 22)
        ax1.set_xlabel(" " )
        ax1.set_ylabel("Total consumption in MWh")
        
        ax2 =sns.barplot(data = dg, x = 'Région', y='Conso_1pt (KWh)',  errorbar=None,ax=ax[1])
        for container in ax2.containers:
                ax2.bar_label(container, label_type="center", fmt="{:.0f} kWH",
                                color='#ffee78', fontsize=12, fontweight ='bold')

        ax2.set_title("Average electricity consumption per day and per point", fontdict=font_title, fontsize = 22)
        ax2.set_xlabel(" " )
        ax2.set_ylabel("Total consumption per consumption point in kWh")
        
        
        
        st.pyplot(fig)
        
        return(dg,font_title)

    
    (dg,font_title) = Enedis()
    
    st.markdown("""
                - Even if consumption is quiet different from the two regions, the shape of the two curves is the same.
                
                - In January consumption is at his highest level for both regions and in August it's at his lowest level.
                
                - Over the two years the shape of the curves is repeated.
                
                """)
    
    dg['Date'] = pd.to_datetime(dg['Date'])
    
    fig,axs =plt.subplots(figsize=(12,7))
    ax = sns.lineplot(data = dg,x = 'Date',y = "Total énergie soutirée (MWh)",hue = 'Région')
    ax.set_xlabel(' ')
    ax.set_ylabel("Total consumption in MWH")
    ax.set_title('Consumption by date', fontdict=font_title, fontsize = 22)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right',bbox_to_anchor=(0.90,0.89),ncol=2)
    ax.get_legend().remove()
    st.pyplot(fig)  
    
    
       
    
    st.subheader("Evolution of consumption")
    

    
    
    
    dg['Date'] = dg['Date'].astype('str')
    season = dg.loc[(dg["Date"] >= "2023-03-21") & (dg["Date"] < "2024-03-01")]
    season['Date'] = pd.to_datetime(season['Date'])
    
    

    
    st.markdown("""
                       - Whatever the region chosen, consumption fluctuates enormously over time as we have seen previously.
                       - Through the differents seasons consumption is clearly different. Winter's consumption is the highest wheareas the Summer's one is the lowest.
                       - School holidays and public holidays have no influence on consumption.             
                       """)
    
    
    
    Vacances = {'période 1':'blue','période 2':'blue','période 3':'blue','période 4':'blue','période 5':'blue','période 6':'blue',
                    'période 7':'blue','période 8':'blue','période 9':'blue','période 20':'green','période 21':'green',
                    'période 22':'green','période 23':'green','période 24':'green','période 25':'green','période 26':'green',
                    'période 27':'green','période 28':'green','période 29':'green','période 30':'green','période 31':'green'}
    
    Jour_ferie = ['2022-04-18','2022-05-01','2022-05-08','2022-05-26','2022-06-06','2022-07-14','2022-08-15','2022-11-01','2022-11-11','2022-12-25',
                      '2023-01-01','2023-04-10','2023-05-01','2023-05-08','2023-05-29','2023-07-14','2023-08-15','2023-11-01','2023-11-11','2023-12-25',
                      '2024-01-01','2024-04-01']


    
    
    with st.container():
        choice = st.multiselect(
        'Choose a specification :',
        options=['Seasonal','Public holiday', 'School holiday'],
        default= 'Seasonal'
        )
        if choice == ['Seasonal']:

            palette = {"Spring": "pink", "Summer": "green",
                        "Autumn": "#fec44f", "Winter": "#3182bd"}
            fig,axs =plt.subplots(2,1,figsize=(12,12))
            fig.suptitle("Seasonal consumption per point in MWh", fontdict=font_title, fontsize = 22)
            ax1 = sns.lineplot(data = season[season['Code région']==24],x= 'Date',y='Total énergie soutirée (MWh)',hue="Season",ax =axs[0],palette = palette)
            ax2 = sns.lineplot(data = season[season['Code région']==32],x= 'Date',y='Total énergie soutirée (MWh)',hue="Season",ax =axs[1],palette = palette)
            ax1.set_title('Profile : Centre-Val de Loire',  loc='left')
            ax2.set_title('Profile : Hauts-de-France',  loc='left')
            ax1.set_xlabel( ' ')
            ax1.set_ylabel('Total consumption in MWh')
            ax2.set_xlabel( ' ')
            ax2.set_ylabel('Total consumption in MWh')
            handles, labels = ax1.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right',bbox_to_anchor=(0.90,0.91),ncol=4)
            ax1.get_legend().remove()
            ax2.get_legend().remove()
            st.pyplot(fig)



        elif choice == ['Public holiday']:
            tab1, tab2 = st.tabs(["Centre-Val de Loire", "Hauts-de-France"])
            fig,axs =plt.subplots(2,1,figsize=(12,12))
            fig.suptitle("Consumption evolution by time in MWh", fontdict=font_title, fontsize = 22)
            ax1 = sns.lineplot(data = season[season['Code région']==24],x= 'Date',y='Total énergie soutirée (MWh)',ax =axs[0],color = 'green')
            sns.scatterplot(data = season[(season['Date'].isin(Jour_ferie)) & (season['Code région']==24)],x = 'Date',y = 'Total énergie soutirée (MWh)',
                ax =axs[0],s = 80, color = 'black')
            ax2 = sns.lineplot(data = season[season['Code région']==32],x= 'Date',y='Total énergie soutirée (MWh)',ax =axs[1],color = 'green')
            sns.scatterplot(data = season[(season['Date'].isin(Jour_ferie)) & (season['Code région']==32)],x = 'Date',y = 'Total énergie soutirée (MWh)',
                ax =axs[1],s = 80, color = 'black')     
            ax2.set_title('Profile : Hauts-de-France',  loc='left')
            ax1.set_xlabel( ' ')
            ax1.set_ylabel('Total consumption in MWh')
            ax2.set_xlabel( ' ')
            ax2.set_ylabel('Total consumption in MWh')
            handles, labels = ax1.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right',bbox_to_anchor=(0.90,0.91),ncol=4)
            st.pyplot(fig)

        elif choice == ['School holiday']:

            fig,axs =plt.subplots(2,1,figsize=(12,12))
            fig.suptitle("Consumption evolution by time in MWh", fontdict=font_title, fontsize = 22)
            ax1 = sns.lineplot(data = season[season['Code région']==24],x= 'Date',y='Total énergie soutirée (MWh)',hue = 'Période',ax =axs[0],palette = Vacances,legend = False)
            ax2 = sns.lineplot(data = season[season['Code région']==32],x= 'Date',y='Total énergie soutirée (MWh)',hue = 'Période',ax =axs[1],palette = Vacances,legend = False)
            ax1.set_title('Profile : Centre-Val de Loire',  loc='left')
            ax2.set_title('Profile : Hauts-de-France',  loc='left')
            ax1.set_xlabel( ' ')
            ax1.set_ylabel('Total consumption in MWh')
            ax2.set_xlabel( ' ')
            ax2.set_ylabel('Total consumption in MWh')
            handles, labels = ax1.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right',bbox_to_anchor=(0.90,0.91),ncol=4)
            st.pyplot(fig)




        else:
            fig,axs =plt.subplots(2,1,figsize=(12,12))
            fig.suptitle("Consumption evolution by time in MWh", fontdict=font_title, fontsize = 22)
            ax1 = sns.lineplot(data = season[season['Code région']==24],x= 'Date',y='Total énergie soutirée (MWh)',hue = 'Période',ax =axs[0],palette = Vacances,legend = False)
            ax2 = sns.lineplot(data = season[season['Code région']==32],x= 'Date',y='Total énergie soutirée (MWh)',hue = 'Période',ax =axs[1],palette = Vacances,legend = False)
            sns.scatterplot(data = season[(season['Date'].isin(Jour_ferie)) & (season['Code région']==32)],x = 'Date',y = 'Total énergie soutirée (MWh)',
            ax =axs[1],s = 80, color = 'black')
            sns.scatterplot(data = season[(season['Date'].isin(Jour_ferie)) & (season['Code région']==24)],x = 'Date',y = 'Total énergie soutirée (MWh)',
            ax =axs[0],s = 80, color = 'black')
            ax1.set_title('Profile : Centre-Val de Loire',  loc='left')
            ax2.set_title('Profile : Hauts-de-France',  loc='left')
            ax1.set_xlabel( ' ')
            ax1.set_ylabel('Total consumption in MWh')
            ax2.set_xlabel( ' ')
            ax2.set_ylabel('Total consumption in MWh')
            handles, labels = ax1.get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right',bbox_to_anchor=(0.90,0.91),ncol=4)
            st.pyplot(fig)
            
       ### Boxplot distribution of consum by holidays ###
        
    st.markdown(""" The previouses charts and the folowing one show us electric consumption is quite equal along differents kind of days except for the school holidays. 
                Usually people travel during this period.
                
                """)
    #Copie = season.groupby(['Région','Day_type'])['Total énergie soutirée (MWh)'].mean().round(2).reset_index()
    #Copie['Energy (MWh)'] = Copie['Total énergie soutirée (MWh)']
#
    #tree_map = px.treemap(Copie, path=[px.Constant("France"),"Région", "Day_type"], 
    #                                    values = 'Energy (MWh)', color = 'Energy (MWh)',
    #                                    color_continuous_scale='blues')
#
    #tree_map.update_layout(margin = dict(t=50, l=25, r=25, b=25), 
    #                                    title=dict(text="AVG consumption per day by kind of day", font=dict(size=20,color='#144b98', family='sans-serif'),x=0.25))
    #tree_map.update_traces(textinfo="label+text+value")
#
    #st.plotly_chart(tree_map, theme="streamlit")

    palette_box = {"National Holidays":"#2c7fb8",
                  "Week-end": "#2ca25f", "School Holidays": "#fa9fb5",
                  "Normal Day": "#fee6ce"}
        
    

    fig,axs =plt.subplots(2,1,figsize=(12,12))
    fig.suptitle("AVG consumption per day by kind of day", fontdict=font_title, fontsize = 22)
    ax1 = sns.barplot(data = season[season['Code région']==24],x= 'Day_type',y='Total énergie soutirée (MWh)',ax =axs[0],hue = 'Day_type',palette = palette_box,errorbar = None)
    ax2 = sns.barplot(data = season[season['Code région']==32],x= 'Day_type',y='Total énergie soutirée (MWh)',ax =axs[1],hue = 'Day_type',palette = palette_box,errorbar = None)
    ax1.set_title('Profile : Centre-Val de Loire',  loc='left')
    ax1.set_ylabel(' ')
    ax2.set_ylabel(' ')
    ax1.set_xlabel(' ')
    ax2.set_xlabel(' ')
    ax2.set_title('Profile : Hauts-de-France',  loc='left')
    for i in ax1.containers:
        ax1.bar_label(i,fontsize=12,fmt="{:.0f}")
    for i in ax2.containers:
        ax2.bar_label(i, fontsize=12,fmt="{:.0f}")
    st.pyplot(fig)


        
        ### Scatter min/max temerature ###

    st.header("Correlation of weather's features", divider='rainbow') 
        
    st.markdown("Electricity consumption is clearly different depending on temperature. The more it's cold the more the consumption is important. Temperature is definitely an important factor of the electrical needs.")
        
    
        
    #labels = df_cvdl["season"].unique()
    df_concat = pd.read_csv('df_concat.csv')
    dl = df_concat[df_concat['Code région'] == 24]
    dr = df_concat[df_concat['Code région'] == 32]
    labels = df_concat['Season'].unique()
    tab1, tab2 = st.tabs(["Centre-Val de Loire", "Hauts-de-France"])

    with tab1:

            buttonsLabels = [dict(label = "All", method = "update",visible=True, args = [{'x' : [dl['MAX_TEMPERATURE_C']]},{'y' : [dl['TEMPERATURE_EVENING_C']]},
                                                                                {'color': [dl['Total énergie soutirée (MWh)']]},
                                                                        ]
                                                                        )]
            for label in labels:
                    buttonsLabels.append(dict(label = label,method = "update",visible = True,args = [{'x' : [dl.loc[dl['Season'] == label, "MAX_TEMPERATURE_C"]]},
                                                                                {'y' : [dl.loc[dl['Season'] == label, "TEMPERATURE_EVENING_C"]]},
                                                                                {'color' : [dl.loc[dl['Season'] == label, "Total énergie soutirée (MWh)"]]},
                                                                        ]
                                                                        ))

            
            fig1 = go.Figure(px.scatter(dl, x="MAX_TEMPERATURE_C", y="TEMPERATURE_EVENING_C",color="Total énergie soutirée (MWh)",hover_data= ["Total énergie soutirée (MWh)"],
                            labels={"MAX_TEMPERATURE_C": "Max Temperature",
                                    "TEMPERATURE_EVENING_C": "Evening Temperature",
                                    "Total énergie soutirée (MWh)": "Consum"
                                     },color_continuous_scale='turbo'),
                )

            fig1.update_layout(updatemenus = [dict(buttons = buttonsLabels, showactive = True)],
                    margin = dict(t=50, l=25, r=25, b=25),
                   title=dict(text="Daily consumption distribution by Evening/Max Temperature",font=dict(size=20)),
                  title_font=dict(size=22,family= 'sans-serif',
                                color =  '#114b98')                  )

            st.plotly_chart(fig1, theme="streamlit")         

    with tab2:

            buttonslist = [dict(label = "All", method = "update",visible=True, args = [{'x' : [dr['MAX_TEMPERATURE_C']]},{'y' : [dr['TEMPERATURE_EVENING_C']]},
                                                                                {'color': [dr['Total énergie soutirée (MWh)']]},
                                                                        ]
                                                                        )]
            for label in labels:
                    buttonslist.append(dict(label = label,method = "update",visible = True,args = [{'x' : [dr.loc[dr['Season'] == label, "MAX_TEMPERATURE_C"]]},
                                                                                {'y' : [dr.loc[dr['Season'] == label, "TEMPERATURE_EVENING_C"]]},
                                                                                {'color' : [dr.loc[dr['Season'] == label, "Total énergie soutirée (MWh)"]]},
                                                                        ]
                                                                        ))
                    fig2 = go.Figure(px.scatter(dr, x="MAX_TEMPERATURE_C", y="TEMPERATURE_EVENING_C",color="Total énergie soutirée (MWh)",hover_data= ["Total énergie soutirée (MWh)"],
                            labels={"MAX_TEMPERATURE_C": "Max Temperature",
                                    "TEMPERATURE_EVENING_C": "Evening Temperature",
                                    "Total énergie soutirée (MWh)": "Consum"
                                     },color_continuous_scale='turbo'),
                )
                    fig2.update_layout(updatemenus = [dict(buttons = buttonslist, showactive = True)],
                    margin = dict(t=50, l=25, r=25, b=25),
                   title=dict(text="Daily consumption distribution by Min/Max Temperature",font=dict(size=20)),
                  title_font=dict(size=22,family= 'sans-serif',
                                color =  '#114b98'),
                   )
            st.plotly_chart(fig2)  


        
    
    
    
    
    
    
    st.markdown(" In the following charts the consumption is split along it's rainy or not, snowy or not and humidity's levels.")
    st.markdown("""
                
                - With further analyses we can notice consumption is quite different for both regions depending on two of the three factors.
                - Rain is definitely not an important factor of consumption.
                - Snow and humidity are importants factors of consumption.
                
                """)

    ### Boxplot Rainy day consumption ###
    
    fig,axs =plt.subplots(2,1,figsize=(12,12))
    fig.suptitle("AVG consumption per day by rainy/non rainy day", fontdict=font_title, fontsize = 22)
    ax1 = sns.boxplot(data = df_concat[df_concat['Code région']==24],y= 'Rain',x='Total énergie soutirée (MWh)',ax =axs[0],hue = 'Rain')
    ax2 = sns.boxplot(data = df_concat[df_concat['Code région']==32],y= 'Rain',x='Total énergie soutirée (MWh)',ax =axs[1],hue = 'Rain')
    ax1.set_title('Profile : Centre-Val de Loire',  loc='left')
    ax1.set_ylabel(' ')
    ax1.set_xlabel('Total energy (MWh) ')
    ax2.set_ylabel(' ')
    ax2.set_xlabel('Total energy (MWh) ')
    ax2.set_title('Profile : Hauts-de-France',  loc='left')
    st.pyplot(fig)
    
    df_concat['Humidity'] = ""
    for i in range(len(df_concat)):
        if df_concat['HUMIDITY_MAX_PERCENT'][i] > 70:
                df_concat['Humidity'][i] = "Humidity > 70 %"
        else:
                df_concat['Humidity'][i] = "Suitable Humidity"
    
    fig,axs =plt.subplots(2,1,figsize=(12,12))
    fig.suptitle("AVG consumption per day by Humidity Levels", fontdict=font_title, fontsize = 22)
    ax1 = sns.boxplot(data = df_concat[df_concat['Code région']==24],y= 'Humidity',x='Total énergie soutirée (MWh)',ax =axs[0],hue = 'Humidity')
    ax2 = sns.boxplot(data = df_concat[df_concat['Code région']==32],y= 'Humidity',x='Total énergie soutirée (MWh)',ax =axs[1],hue = 'Humidity')
    ax1.set_title('Profile : Centre-Val de Loire',  loc='left')
    ax1.set_ylabel(' ')
    ax1.set_xlabel('Total energy (MWh) ')
    ax2.set_ylabel(' ')
    ax2.set_xlabel('Total energy (MWh) ')
    ax2.set_ylabel(' ')
    
    ax2.set_title('Profile : Hauts-de-France',  loc='left')
    st.pyplot(fig)
    
    df_concat['Date'] = pd.to_datetime(df_concat['Date'])
    df_concat['Snow'] = ""
    for i in range(len(df_concat)):
        if df_concat['TOTAL_SNOW_MM'][i] ==0:
                df_concat['Snow'][i] = "No snow"
        else:
                df_concat['Snow'][i] = "Snow"
    
    fig,axs =plt.subplots(2,1,figsize=(12,12))
    fig.suptitle("AVG consumption per day by snowy/non snowy day", fontdict=font_title, fontsize = 22)
    ax1 = sns.boxplot(data = df_concat[df_concat['Code région']==24],y= 'Snow',x='Total énergie soutirée (MWh)',ax =axs[0],hue = 'Snow')
    ax2 = sns.boxplot(data = df_concat[df_concat['Code région']==32],y= 'Snow',x='Total énergie soutirée (MWh)',ax =axs[1],hue = 'Snow')
    ax1.set_title('Profile : Centre-Val de Loire',  loc='left')
    ax1.set_ylabel(' ')
    ax1.set_xlabel('Total energy (MWh) ')
    ax2.set_ylabel(' ')
    ax2.set_xlabel('Total energy (MWh) ')
    ax2.set_ylabel(' ')
    ax2.set_title('Profile : Hauts-de-France',  loc='left')
    st.pyplot(fig)
    
    
    
 
 
def ML():
    ### Regroupement des fichiers pour avoir une matrice de travail
    df_concat = pd.read_csv('df_concat.csv')
    
    
    
    
    ### Création du model de ML
     
    X = df_concat[['TOTAL_SNOW_MM', 'Code région', 'HUMIDITY_MAX_PERCENT', 'Day_type_mod',
                                 'Season-modified','PRECIP_TOTAL_DAY_MM','MAX_TEMPERATURE_C'] ]
    y= df_concat['Total énergie soutirée (MWh)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, train_size=0.75)
    
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    
    modelDTR = DecisionTreeRegressor(max_depth= 6, min_samples_leaf= 5, min_samples_split= 15,random_state = 42)
    modelDTR.fit(X_train_scaled, y_train)
    
    ### demande à l'utilisateur de rentrer les paramètres
    
    st.header("Consumption prediction")
    
    
 
    st.write("In order to correctly predict consumption we require your help. It doesn't take too long but we need you to answer the following questions.")
    
    T_max = st.number_input('What is the max temperature ? ')
    
    Precip = st.number_input('What is the amount of precipitation ? ')
    
    Region = st.radio('Select your region',['Hauts-de-France','Centre-Val de Loire'])
    if Region == 'Hauts-de-France':
        code = 32
    else:
        code = 24
    
    Season = st.radio('What is the Season ? ',['Autumn','Winter','Spring','Summer'])
    
    if Season == 'Autumn':
            Season = 3
    elif Season == 'Winter':
            Season = 4
    elif Season == 'Spring':
            Season = 1
    else:
            Season = 2
            
    Day = st.radio('Is the predicted day on a special day ?',['Normal Day','Week-end','National Holiday','School Holiday'])
    
    if Day == 'Normal Day':
            Day = 0
    elif Day == 'Week-end':
            Day = 1
    elif Day == 'National Holiday':
            Day = 3
    else:
            Day = 2
            
            
    Snow = st.number_input('What is the amount of snow ? ')
    
    
    
    Humidity = st.number_input('What is the maximum humidity percentage ? ')
       
        
    ### prédiction la consommation
       
    Res = [Snow,code,Humidity,Day,Season,Precip,T_max] 
    Res = modelDTR.predict([Res])
    Res= round(Res[0],2)
    
    st.write("\nWith the previouses parameters the electric consumption will be", Res, 'MWh')
    
    df_concat['Région'] = df_concat['Code région'].replace({24: 'Centre-Val de Loire',32:'Hauts de France'})
    
    
    fig = px.violin(df_concat[(df_concat['Code région'] == code) &(df_concat['Day_type_mod'] == Day) & (df_concat['Season-modified'] == Season) & abs((df_concat['MAX_TEMPERATURE_C']- T_max)<20)& 
                        (abs(df_concat['PRECIP_TOTAL_DAY_MM']-Precip)<0.4)  &    (abs(df_concat['TOTAL_SNOW_MM']-Snow)<0.01 ) 
                     
                    ], y='Total énergie soutirée (MWh)', x='Région', 
                box=True,  
                  
                hover_data=df_concat.columns)  


    fig.update_layout(
        title=dict(
            text='Distribution of consumption',
            font=dict(size=20, color='#144b98', family='sans-serif'),x=0.35
        ),
        xaxis_title='',
        yaxis_title='Consumption in MWh'
    )
    
    
    st.plotly_chart(fig)
