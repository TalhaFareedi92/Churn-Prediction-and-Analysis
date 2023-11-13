import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff

import plotly.graph_objects as go




st.set_page_config(page_title="Telco Analysis", page_icon=":bar_chart", layout='wide')

st.markdown("<style>div.block-container{padding-top : 1rem; margin-top:0px;}</style>", unsafe_allow_html=True)



st.markdown(
    '''
   <style>
    .hover-effect {
        color: blue;
        background-color: #629386;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        border-radius: 20px;
        transition-property: background-color, color;
        cursor: pointer;
    }
    .hover-effect:hover {
        background-color:  #DA7B2B;
        color: white;
        
    }
    hr{
    color:white;
    padding:5px;

    }
    </style>
    ''',
    unsafe_allow_html=True
)
st.markdown(
    '''
    <style>
        h1 {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            color: #FFFF;
            text-transform: uppercase;
            text-shadow: 2px 2px 2px rgba(0, 0, 0, 0.3);
            transform: perspective(150px) rotateX(6deg);
        }
        h2{
        font-size :25px;
        
        }
    </style>
    ''',
    unsafe_allow_html=True
)

st.markdown(
    '''
    <style>
        h1 {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            color: #ffff;
            text-transform: uppercase;
            text-shadow: 2px 2px 2px rgba(0, 0, 0, 0.3);
            transform: perspective(200px) rotateX(10deg);
        }
        
        .change_content::after {
            content: '';
            font-weight: 100;
            animation: changetxt 4s infinite linear;
            background-color: #B37A48;
            color: white;
        }

        @keyframes changetxt {
            0% { content: "Telco Analysis"; }
            25% { content: "Churn Analysis"; }
            50% { content: "Predictions"; }
            75% { content: "C Matrix"; }
            100% { content: "Grphical Repr"; }
        }
    </style>
    '''
    , unsafe_allow_html=True
)




#Title

st.markdown('<h1 class="hover-effect">Come to the Dashboard Do   <span class="change_content"> </span></h1>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = df.drop(columns=['customerID'])
st.sidebar.header("Get Desire Relations and Prediction")
columns_without_churn = [col for col in df.columns if col != 'Churn']
Your_Team = st.sidebar.selectbox('Select column to relate to Churn', columns_without_churn)
if st.sidebar.button("Analyze"):
         selected_df = df.groupby([Your_Team, 'Churn']).size().reset_index(name='Count')
         fig1 = px.bar(selected_df, x=Your_Team, y='Count', color='Churn', barmode='group',
             labels={'Count': 'Customer Count'},
             title=f'Relationship between {Your_Team} and Churn')
         st.plotly_chart(fig1)

columns_without_churn_updated = [col for col in columns_without_churn if col != Your_Team]

snsteam = st.sidebar.selectbox('Select column to relate to Churn', columns_without_churn_updated)

if st.sidebar.button("Analyze:"):
               selected_df = df.groupby([Your_Team,snsteam, 'Churn']).size().reset_index(name='Count')
               fig2 = px.bar(selected_df,x=snsteam,y='Count',facet_col=Your_Team, barmode='group',labels={'Count : Customer Count'},         
                                title=f'Relationship between {Your_Team} , {snsteam}and Churn')
               
               st.plotly_chart(fig2)


columns_without_churn_updated2 = [col for col in columns_without_churn if col != Your_Team if col !=snsteam]
thirdteam= st.sidebar.selectbox('Select column to relate to Churn', columns_without_churn_updated2)
if st.sidebar.button("Analyzze"):
                      selected_df = df.groupby([Your_Team,snsteam,thirdteam, 'Churn']).size().reset_index(name='Count')
                      fig = px.bar(selected_df, x='Count', y=Your_Team, color=snsteam,
                      facet_col=thirdteam,
                       labels={'count': 'Customer Count'},
                       title=f'Relationship between {Your_Team} , {snsteam},{thirdteam},and Churn', width=900,height=300)
                    
                      st.plotly_chart(fig)

st.markdown('<br>',unsafe_allow_html=True)

       
if st.sidebar.button("Get Logistic Regreassion"):
        data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
        input = data.drop(['customerID','Churn','PaperlessBilling','PaymentMethod','OnlineBackup','DeviceProtection','StreamingTV', 'StreamingMovies'], axis=1)
        output  = data['Churn']
        from sklearn.preprocessing import LabelEncoder
        gender = LabelEncoder()
        Partner = LabelEncoder()
        SeniorCitizen = LabelEncoder()
        Dependents = LabelEncoder()
        tenure = LabelEncoder()
        PhoneServices = LabelEncoder()
        Multihelplines = LabelEncoder()
        Internet_services = LabelEncoder()
        Online_security = LabelEncoder()
        TechSupport = LabelEncoder()
        Contract = LabelEncoder()
        Monthly_charges = LabelEncoder()
        Total_Charges = LabelEncoder()
        input['gender'] = gender.fit_transform(input['gender'])
        input['SeniorCitizen'] = SeniorCitizen.fit_transform(input['SeniorCitizen'])
        input['Partner'] = Partner.fit_transform(input['Partner'])
        input['Dependents'] = Dependents.fit_transform(input['Dependents'])
        input['tenure'] = tenure.fit_transform(input['tenure'])
        input['PhoneService'] = PhoneServices.fit_transform(input['PhoneService'])
        input['MultipleLines'] = Multihelplines.fit_transform(input['MultipleLines'])
        input['InternetService'] = Internet_services.fit_transform(input['InternetService'])
        input['OnlineSecurity'] = Online_security.fit_transform(input['OnlineSecurity'])
        input['TechSupport'] = TechSupport.fit_transform(input['TechSupport'])
        input['Contract'] = Contract.fit_transform(input['Contract'])
        input['MonthlyCharges'] = Monthly_charges.fit_transform(input['MonthlyCharges'])
        input['TotalCharges'] = Total_Charges.fit_transform(input['TotalCharges'])
        from sklearn.model_selection import train_test_split
        X_train,X_test,Y_train,Y_test= train_test_split(input,output,test_size=0.2)
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit(X_train,Y_train)
        a = model.predict(X_test)
        b= model.score(X_test,Y_test)*100
        st.write("The Score Of Model is ",b)
        st.write("Because of the encoding of limmited dimension and test size the score is Low ")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(a, Y_test)
        st.markdown('<h2 class="hover-effect"  >Confusion Matrix is </h2>',unsafe_allow_html=True)
        import matplotlib.pyplot as plt
        import seaborn as sns

        heatmap_fig = ff.create_annotated_heatmap(z=cm, x=['Predicted 0', 'Predicted 1'],
                                          y=['Actual 0', 'Actual 1'], colorscale='YlGnBu', showscale=False)

        heatmap_fig.update_layout(
            xaxis=dict(title='Predicted'),
            yaxis=dict(title='Actual'),
            title='Confusion Matrix'
            )
        st.plotly_chart(heatmap_fig)
        









        


      
















col1,col2 = st.columns([2,2])
st.markdown("<br>", unsafe_allow_html=True)
with col1:
    st.markdown('<h2 class="hover-effect"  >Distribution Of Male and Female</h2>',unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    males = df[df['gender'] == 'Male'].shape[0]
    females = df[df['gender'] == 'Female'].shape[0]
    color_map = {
    'Male': 'sky-blue',
    'Female': '#FF6C00'
}

    gender_counts = pd.DataFrame({'Gender': ['Male', 'Female'], 'Count': [males, females]})

    fig_gender = px.pie(gender_counts, values='Count', names='Gender', hole=0.6, height=400,
                    color='Gender', color_discrete_map=color_map,
                    labels={'Count': 'Customer Count'},
                    title='Distribution of Male and Female Customers')

    fig_gender.add_annotation(
    text=f'Total M , F Counts',
    showarrow=False,
    font=dict(size=15),
    x=0.5,
    y=0.5
    )
    fig_gender.add_annotation(
    text=f'Total Male: {males}, Total Female: {females}',
    showarrow=False,
    font=dict(size=15),
    x=0.5,
    y=-0.25,
    yanchor='top'
    )
    st.write("Total Number of Males are",males)
    st.write("Total Number of Females are",females)
    st.plotly_chart(fig_gender)
with col2:
        st.markdown('<h2 class="hover-effect"  >Total Churn Yes Or No</h2>',unsafe_allow_html=True)

        churn=1869
        not_churn = 5174
        st.write("Total Number of churn clients",churn)
        st.write("Total Number of not churn clients",not_churn)
        churn_counts = df['Churn'].value_counts()

        fig = px.pie(churn_counts, values=churn_counts.values, names=churn_counts.index, 
             title=' Total Churn Counts', color_discrete_sequence=px.colors.qualitative.Set1,hole=0.6,height=400
             )

        st.plotly_chart(fig)
st.markdown("<br><br><hr>",unsafe_allow_html=True)

st.markdown("<h4 class='hover-effect'>Analysis based on Senior Citizens, Dependents, having partners and gender</h4>",unsafe_allow_html=True)
grouped_data = df.groupby(['gender', 'SeniorCitizen', 'Partner', 'Dependents']).size().reset_index(name='count')

color_map = {     
    '0_yes_no':'rgb(70,130,180)',
    '1_no_no': 'rgb(255,0,0)',
    '1_yes_yes': 'rgb(165,42,42)'
}

fig = px.bar(
    grouped_data, 
    x='count', 
    y='gender', 
    color='Partner',
    facet_col='Dependents',
    facet_row='SeniorCitizen',
    width=1200,
    color_discrete_map=color_map,
    labels={'count': 'customer count'},
    title='customer distribution based on gender, seniorcitizen, partner, and dependents'
)



st.plotly_chart(fig)
import plotly.graph_objects as go
st.markdown("<h4 class='hover-effect'>Analysis based on Senior Citizens, Dependents, having partners and gender on Plotly Go Chart</h4>",unsafe_allow_html=True)


grouped_data = df.groupby(['gender', 'SeniorCitizen', 'Partner', 'Dependents']).size().reset_index(name='count')

Senior_citizen = grouped_data[grouped_data['SeniorCitizen'] == 1]  # Assuming 'Yes' is encoded as 1
Not_senior_citizen = grouped_data[grouped_data['Partner'] == 'Yes']
Dependents = grouped_data[grouped_data['Dependents'] == 'Yes']

fig = go.Figure(data=[
    go.Bar(name='Senior Citizen', x=Senior_citizen['gender'], y=Senior_citizen['count'], marker=dict(color='blue')),
    go.Bar(name='Having Partner', x=Not_senior_citizen['gender'], y=Not_senior_citizen['count'], marker=dict(color='#DAA06D')),
    go.Bar(name='With Dependents', x=Dependents['gender'], y=Dependents['count'], marker=dict(color='red'))
])

fig.update_layout(
    barmode='group',
    title='Customer Distribution Based on Gender, SeniorCitizen, Partner, and Dependents',
    xaxis=dict(title='Gender'),
    yaxis=dict(title='Customer Count'),
    legend=dict(title='Category'),
    bargap=0.6, # Adjust the gap between bars

)

fig.add_annotation(
    text="Customer Counts",
    xref="paper", yref="paper",
    x=0.02, y=1.05,
    showarrow=False,
    font=dict(size=14)
)

fig.add_annotation(
    text="Customer Categories",
    xref="paper", yref="paper",
    x=0.98, y=1.05,
    showarrow=False,
    font=dict(size=14)
)


st.plotly_chart(fig)









st.markdown("<h5>left side is gender on Right side distriution among senior citizen or not in column wise Dependents and hue is partners</h5>",unsafe_allow_html=True)
st.markdown("<br><br><hr>",unsafe_allow_html=True)

col3 , col4 = st.columns([2,2])

df['loyal'] = df['tenure']>=12
df['New'] = df['tenure']<12
with col3:
     st.markdown("<h4 class='hover-effect'>Destribution of Customers HAving Tenure greater than 12 or not </h4>",unsafe_allow_html=True)
     st.markdown("<br>",unsafe_allow_html=True)

     loyal = df['loyal'].sum()
     New = df['New'].sum()
     st.write("Total Number of Loyal Clientss",loyal)
     st.write("Total Number of New clients",New)

     gender_counts = pd.DataFrame({'Type': ['New', 'Loyal'], 'Count': [New, loyal]})
     color_map = {'New': '#7FC3BC', 'Loyal': 'white'}
     color_map = {
    'Count_Loyal': '#DAA06D',  
    'Count_New': 'pink'     
}

     fig = px.bar(gender_counts, x='Type', y='Count', labels={'Count': 'Customer Count'}, title='Distribution of New and Loyal Customers',
             width=450,color_discrete_map=color_map,color='Type')
     st.plotly_chart(fig)   
     st.markdown("<h5>Loyal are those who have tenure more than 12months New have Tenure lower than 12 month</h5>",unsafe_allow_html=True)

with col4:
       st.markdown("<h4 class='hover-effect'> Checking Gender for loyal and New  </h4>",unsafe_allow_html=True)

       st.markdown("<br><br><br>",unsafe_allow_html=True)
       loyal_counts = df[df['loyal']].groupby('gender').size().reset_index(name='Count')
       new_counts = df[~df['loyal']].groupby('gender').size().reset_index(name='Count')

       gender_counts = pd.merge(loyal_counts, new_counts, on='gender', suffixes=('_Loyal', '_New'))

       color_map = {
    'Count_Loyal': '#DAA06D',  
    'Count_New': 'pink'     
}
       fig = px.bar(gender_counts, x='gender', y=['Count_Loyal', 'Count_New'],
             labels={'value': 'Customer Count', 'variable': 'Customer Type'},
             title='Distribution of Loyal and New Customers Based on Gender',barmode='group',
             color_discrete_map=color_map)

       st.plotly_chart(fig, use_container_width=True)

st.markdown("<br><br><hr>",unsafe_allow_html=True)
st.markdown("<h4 class='hover-effect' style='text-align:center;'>Now Churn Analysis</h4>", unsafe_allow_html=True)

st.markdown("<hr>",unsafe_allow_html=True)
col5,col6 = st.columns([2,2])

grouped_data = df.groupby(['loyal', 'New', 'Churn']).size().reset_index(name='count')
color_map = {
    ('Loyal', 'New', 'Yes'): 'pink',      
    ('Loyal', 'New', 'No'): 'red',       
    ('Loyal', 'Existing', 'Yes'): '#DAA06D',  
    ('Loyal', 'Existing', 'No'): '#FF0000',   
}
fig = px.bar(grouped_data, x='Churn', y='count', facet_col='loyal', color_discrete_sequence=px.colors.qualitative.Set2
)
fig.update_layout(
    title='Loyal And New  Churn Analysis',
    xaxis_title='Churn',
    yaxis_title='Count',
    yaxis=dict(title_standoff=0),
    font=dict(family="Arial", size=12),
   )
st.plotly_chart(fig)



st.markdown('<h1 class="hover-effect">Contract wise Churn Analyais</h1>',unsafe_allow_html=True)

grouped_data = df.groupby(['Contract', 'Churn']).size().reset_index(name='count')

churn_data = grouped_data[grouped_data['Churn'] == 'Yes']
retain_data = grouped_data[grouped_data['Churn'] == 'No']

fig = go.Figure(data=[
    go.Bar(name='Churn', x=churn_data['Contract'], y=churn_data['count']),
    go.Bar(name='Retain', x=retain_data['Contract'], y=retain_data['count'])
])

fig.update_layout(barmode='group', title='Contract to Churn')

st.plotly_chart(fig)


grouped_data = df.groupby(['TotalCharges', 'Churn']).size().reset_index(name='count')

fig = px.scatter(grouped_data, x='TotalCharges', y='count', color='Churn',
                 labels={'TotalCharges': 'Total Charges', 'count': 'Count', 'Churn': 'Churn'},
                 title='Churn Counts Based on Total Charges',
                 log_x=True,
                 width=1200,
                 hover_data=['TotalCharges', 'count'],  
                 color_discrete_sequence=px.colors.qualitative.Set2,
                 size='count',  
                 size_max=30)  

fig.update_xaxes(type='linear')
st.plotly_chart(fig)
col6,col7 = st.columns([2,2])

with col6:

     grouped_data = df.groupby(['TotalCharges', 'Churn']).size().reset_index(name='count')

     fig_yes = px.scatter(grouped_data[grouped_data['Churn'] == 'Yes'], x='TotalCharges', y='count',
                      labels={'TotalCharges': 'Total Charges', 'count': 'Count'},
                      title='Churn Counts (Yes) Based on Total Charges',
                      log_x=True,  
                      hover_data=['TotalCharges', 'count'],  
                      size='count',  
                      size_max=60) 

     fig_yes.update_xaxes(type='linear')

     fig_no = px.scatter(grouped_data[grouped_data['Churn'] == 'No'], x='TotalCharges', y='count',
                     labels={'TotalCharges': 'Total Charges', 'count': 'Count'},
                     title='Churn Counts (No) Based on Total Charges',
                     log_x=True,  
                     hover_data=['TotalCharges', 'count'], 
                     size='count',  
                     size_max=60) 

     fig_no.update_xaxes(type='linear')

  
     st.plotly_chart(fig_yes)
     st.plotly_chart(fig_no)
     




    


    


