import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import mysql.connector
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

# Load the data from environment variables
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
INSTANCE_CONNECTION_NAME = os.getenv("INSTANCE_CONNECTION_NAME")

# Use the Unix socket provided by Cloud SQL Proxy
unix_socket = f"/cloudsql/{INSTANCE_CONNECTION_NAME}"

conn = None  # Initialize conn *outside* the try block

try:
    conn = mysql.connector.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        unix_socket=unix_socket
    )
    cursor = conn.cursor()

    query = """
    SELECT * FROM diabetes;
    """
    cursor.execute(query)
    results = cursor.fetchall()

    # Get column names from the cursor description
    column_names = [description[0] for description in cursor.description]

    df = pd.DataFrame(results, columns=column_names)

    df.head()

except mysql.connector.Error as err:
    print(f"Database connection error: {err}")
    df = pd.DataFrame()  # Create an empty DataFrame if connection fails
finally:
    if conn and conn.is_connected():  # Check if connection exists before calling is_connected()
        cursor.close()
        conn.close()


# Logistic Regression Model - Separate features (X) and target (y)
feature_names = ['Glucose', 'BMI', 'Age']
X = df[feature_names].values  # All features included in the model
y = df.iloc[:, -1].values   # The last feature (target)

# 3. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # 80/20 split
)

# Feature Scaling (Important for Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit and transform training data
X_test = scaler.transform(X_test)      # Transform test data using the same scaler

# Test whether the pickle exists.
if os.path.isfile("logistic_regression.mdl"):
    # If the pickle exists, open it.
    infile = open("logistic_regression.mdl", 'rb')
    # Load the pickle.
    model = pickle.load(infile)
    # Close the pickle file.
    infile.close()
    # Output a message to state that the pickle was
    # successfully loaded.
    print("Loaded pickle")

else:
    # Train the Logistic Regression model
    model = LogisticRegression(max_iter=1000, solver='liblinear', C=10.0)
    model.fit(X_train, y_train)
    # Open a file to save the pickle.
    outfile = open("logistic_regression.mdl", "wb")
    # Store the model in the file.
    pickle.dump(model, outfile)
    # Close the pickle file.
    outfile.close()


# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

# Classification Report (precision, recall, F1-score)
class_report = classification_report(y_test, y_pred)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

class_names = np.unique(y_test)  # Or specify manually

df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

percentage = accuracy * 100
remaining = 100 - percentage

fig7_accuracy = {'Category': ['Accuracy', 'Remaining'], 'Value': [percentage, remaining]}

# Calculate the average number of pregnancies
average_pregnancies = round(df['Pregnancies'].mean(), 2)

# Calculate the Glucose level
average_glucose = round(df['Glucose'].mean(), 2)

# Calculate the average blood pressure level
average_bloodpressure = round(df['BloodPressure'].mean(), 2)

# Calculate the average Insulin level
average_insulin = round(df['Insulin'].mean(), 2)

# Calculate the average BMI level
average_bmi = round(df['BMI'].mean(), 2)

# Calculate the average Age
average_age = round(df['Age'].mean(), 2)

outcome_counts = df['Outcome'].value_counts().reset_index()
outcome_counts.columns = ['Outcome', 'Count']
outcome_counts['Outcome'] = outcome_counts['Outcome'].astype(str)

df_new = df.drop('PatientID', axis=1)

# Count zeros per column
zeros_per_column = (df_new == 0).sum()

# Convert the Series to a DataFrame for Plotly Express
zeros_df = zeros_per_column.reset_index()
zeros_df.columns = ['Features', 'Number of Zeros']

num_rows = len(df.axes[0])

age_counts = df_new['Age'].value_counts().sort_index()

df['Outcome'] = df['Outcome'].astype('category')

# Create the custom color scale for heatmap\matrix
custom_colorscale = px.colors.make_colorscale(['#0081A7', '#F07167'])

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Initial figures and dropdown options (handle empty DataFrame)
initial_x = df_new.columns[0] if not df_new.empty and len(df_new.columns) > 0 else None
initial_y = df_new.columns[1] if not df_new.empty and len(df_new.columns) > 1 else None

fig1 = px.pie(outcome_counts, names='Outcome', values='Count', title='Binary Feature Outcome',
              color="Outcome", color_discrete_sequence=['#0081A7', '#F07167'])

fig2 = px.scatter(df_new, x=initial_x, y=initial_y, title='Scatter-chart',
                  color_discrete_sequence=['#0081A7', '#F07167'])

fig3 = px.box(df_new, x=initial_x, title='Number of pregnancies',
              color_discrete_sequence=['#0081A7', '#F07167'])

fig4 = px.bar(zeros_df, x='Features', y='Number of Zeros', title='Number of Zeros per Feature',
              color_discrete_sequence=['#0081A7', '#F07167'])

fig5 = px.violin(df, x="Outcome", y="Age", color="Outcome", box=True, points="all",
                 title="Age Distribution by Diabetes Outcome",
                 color_discrete_sequence=['#0081A7', '#F07167'])

fig6 = px.imshow(df_cm, labels=dict(x="Predicted", y="Actual", color="Count"), x=df_cm.columns, y=df_cm.index,
                 color_continuous_scale=custom_colorscale, text_auto=True)
fig6.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual", xaxis=dict(tickangle=-45),
                   yaxis=dict(tickangle=0), coloraxis_showscale=False)

fig7 = px.pie(fig7_accuracy, values='Value', names='Category', title=f'Accuracy',
              color_discrete_sequence=['#0081A7', '#F07167'], hole=0.6)
fig7.update_layout(showlegend=False)
fig7.update_traces(textinfo='none')
fig7.add_annotation(text=f"<b>{percentage:.1f}%</b>", x=0.5, y=0.5, font=dict(size=28, family="Arial",
                    color="black"), showarrow=False)

dropdown_options = [{'label': col, 'value': col} for col in df_new.columns] if not df_new.empty else []

app.layout = html.Div(children=[
html.Div(className="container-fluid", children=[
    html.H1(children="Pima Indian Diabetes Data Analysis and Prediction Dashboard", className="p-5 rounded-lg shadow",
            style={"background-color": '#0092BA', "margin-bottom": "20px", "color": "white"}),
        html.Div(className="row card-container",  children=[
            html.Div(className="col-md-2", children=[
                dbc.Card(children=[
                    dbc.CardBody(children=[
                        html.H5("👶 Average number of Pregnancies",  className="card-title",
                                style={'filter': 'grayscale(100%)'}),
                        html.P(average_pregnancies, className="card-text",
                               style={'text-indent': '33px', 'font-size': '18px', 'font-weight': 'bold'}),
                    ])
                ], style={"height": "100%"})
            ]),
            html.Div(className="col-md-2", children=[
                dbc.Card(children=[
                    dbc.CardBody(children=[
                        html.H5("🧪 Average Glucose level", className="card-title",
                                style={'filter': 'grayscale(100%)'}),
                        html.P(average_glucose, className="card-text",
                               style={'text-indent': '33px', 'font-size': '18px', 'font-weight': 'bold'}),
                    ])
                ], style={"height": "100%"})
            ]),
            html.Div(className="col-md-2", children=[
                dbc.Card(children=[
                    dbc.CardBody(children=[
                        html.H5("❤ Average Blood Pressure level", className="card-title",
                                style={'filter': 'grayscale(100%)'}),
                        html.P(average_bloodpressure, className="card-text",
                               style={'text-indent': '33px', 'font-size': '18px', 'font-weight': 'bold'}),
                    ])
                ], style={"height": "100%"})
            ]),
            html.Div(className="col-md-2", children=[
                dbc.Card(children=[
                    dbc.CardBody(children=[
                        html.H5("🧪 Average Insulin level", className="card-title",
                                style={'filter': 'grayscale(100%)'}),
                        html.P(average_insulin, className="card-text",
                               style={'text-indent': '33px', 'font-size': '18px', 'font-weight': 'bold'}),
                    ])
                ], style={"height": "100%"})
            ]),
            html.Div(className="col-md-2", children=[
                dbc.Card(children=[
                    dbc.CardBody(children=[
                        html.H5("💪 Average BMI", className="card-title", style={'filter': 'grayscale(100%)'}),
                        html.P(average_bmi, className="card-text", style={'text-indent': '33px', 'font-size': '18px',
                                                                          'font-weight': 'bold'}),
                    ])
                ], style={"height": "100%"})
            ]),
            html.Div(className="col-md-2", children=[
                dbc.Card(children=[
                    dbc.CardBody(children=[
                        html.H5("⌛ Average Age", className="card-title", style={'filter': 'grayscale(100%)'}),
                        html.P(average_age, className="card-text", style={'text-indent': '33px', 'font-size': '18px',
                                                                          'font-weight': 'bold'}),
                    ])
                ], style={"height": "100%"})
            ]),
        ], style={"padding-bottom": "10px"}),

    html.Div(className="row card-container", children=[
        html.Div(className="col-md-3", children=[
            dbc.Card(children=[
                dbc.CardBody(children=[
                    dcc.Graph(figure=fig1),
                    html.P(f"Number of Data Points: {num_rows}", className="card-text", style={'text-align': 'center'}),
                ])
            ], style={"height": "100%"})
        ]),
        html.Div(className="col-md-3", children=[
            dbc.Card(children=[
                dbc.CardBody(children=[
                    dcc.Graph(figure=fig5),
                ])
            ], style={"height": "100%"})
        ]),
        html.Div(className="col-md-3", children=[
            dbc.Card(children=[
                dbc.CardBody(children=[
                    dcc.Graph(id='scatter-chart', figure=fig2),
                    dcc.Dropdown(
                        id='x-axis-dropdown2',
                        options=dropdown_options,
                        value=initial_x,
                        clearable=False
                    ),
                    dcc.Dropdown(
                        id='y-axis-dropdown2',
                        options=dropdown_options,
                        value=initial_y,
                        clearable=False
                    ),
                ])
            ], style={"height": "100%"})
        ]),
        html.Div(className="col-md-3", children=[
            dbc.Card(children=[
                dbc.CardBody(children=[
                    dcc.Graph(id='boxplot', figure=fig3),
                    dcc.Dropdown(
                        id='x-axis-dropdown1',
                        options=dropdown_options,
                        value=initial_x,
                        clearable=False
                    ),
                ])
            ], style={"height": "100%"})
        ]),
    ], style={"padding-bottom": "10px"}),
         html.Div(className="row card-container", children=[
         html.Div(className="col-md-3", children=[
                dbc.Card(children=[
                 dbc.CardBody(children=[
                    dcc.Graph(figure=fig6),
                ])
                   ], style={"height": "100%"})
        ]),
         html.Div(className="col-md-3", children=[
            dbc.Card(children=[
                dbc.CardBody(children=[
                    dcc.Graph(figure=fig7),
                ])
                   ], style={"height": "100%"})
         ]),
         html.Div(className="col-md-3", children=[
            dbc.Card(children=[
                dbc.CardBody(children=[
                                html.H5("Logistic Regression model Performance Summary",
                                        className="card-title"),
                                html.Pre(class_report, style={"height": "100%", "padding-top": "40px",
                                                              "padding-left": "30px", 'text-indent': '10px',
                                                              'font-size': '16px', 'font-weight': 'bold'})
                ])
            ], style={"height": "100%", "padding-top": "30px", "padding-left": "20px", 'text-indent': '60px',
                      'font-size': '20px', 'font-weight': 'bold'})
         ]),
            html.Div(className="col-md-3", children=[
                dbc.Card(children=[
                   dbc.CardBody(children=[
                       html.H5("Diabetes risk factors", style={"margin-top": "22px"}),
                       dbc.Form([
                           dbc.Row([  # Row for inputs
                               dbc.Col(dbc.CardGroup([
                                   dbc.Label("Age", html_for="Age", className="form-label fw-bold"),
                                   dcc.Input(type="text", id="Age", className="form-control input-with-border",
                                             placeholder="Age"),
                               ]), width=4),  # Adjust width as needed
                               dbc.Col(dbc.CardGroup([
                                   dbc.Label("BMI", html_for="BMI", className="form-label fw-bold"),
                                   dcc.Input(type="text", id="BMI", className="form-control input-with-border",
                                             placeholder="BMI"),
                               ]), width=4),  # Adjust width as needed
                               dbc.Col(dbc.CardGroup([
                                   dbc.Label("Glucose level", html_for="Glucose", className="form-label fw-bold"),
                                   dcc.Input(type="text", id="Glucose", className="form-control input-with-border",
                                             placeholder="Glucose level"),
                               ]), width=4),  # Adjust width as needed
                           ]),
                           dbc.Button("Submit", color="secondary", id="submit-button", n_clicks=0, className="mt-3"),
                           # Button below, mt-3 adds margin
                       ]),
                     dbc.Card(children=[
                      dbc.CardBody(children=[
                         html.Div(className="card-body", children=[
                             html.H5("Diabetes Risk Prediction", style={"margin-top": "20px"}),
                             html.P("A score of 0: Low risk.",
                                    style={"font-weight": "bold", "font-size": "16px"}),
                             html.P("A score of 1 suggests further evaluation",
                                    style={"font-weight": "bold", "font-size": "16px"}),
                             html.Br(),
                         ]),
                         html.Div(id="output-text", style={"font-weight": "bold", "font-size": "70px", "text-align": "center"}),  # To display output
                     ])
                   ], style={"height": "100%", "margin-left": "10px", "margin-top": "50px"})
                ]),
                ]),
            ]),
             ], style={"height": "100%"})
        ]),
], style={"padding-bottom": "10px"})


# Callback to update the scatter boxplot
@app.callback(
    Output('boxplot', 'figure'),
    Input('x-axis-dropdown1', 'value')
)
def update_boxplot(x_value):
    if x_value and not df.empty:
        fig = px.box(df, x=x_value, color_discrete_sequence=['#ea8c55', '#ea526f'])
        fig.update_layout(title=f"{x_value}")
        return fig
    return {}


# Callback to update the scatter plot
@app.callback(
    Output('scatter-chart', 'figure'),
    Input('x-axis-dropdown2', 'value'),
    Input('y-axis-dropdown2', 'value')
)
def update_scatter_chart(x_value, y_value):
    if x_value and y_value and not df.empty:
        fig = px.scatter(df, x=x_value, y=y_value, color_discrete_sequence=['#ea8c55', '#ea526f'])
        fig.update_layout(title=f"{x_value} vs {y_value}")
        return fig
    return {}


# Callback to calculate the probability for patient to have diabetes
@app.callback(
    Output("output-text", "children"),  # Output to the output area
    Input("submit-button", "n_clicks"),  # Triggered by button clicks
    State("Age", "value"),  # Get current values of inputs
    State("BMI", "value"),
    State("Glucose", "value"),
)
def update_output(n_clicks, age, bmi, glucose):
    if n_clicks > 0:  # Only update on button click
        new_data = np.array([[age, bmi, glucose]])
        new_data_scaled = scaler.transform(new_data)
        prediction = model.predict(new_data_scaled)
        output_text = (f"{prediction}")
        return output_text
    return ""  # Return empty string initially


if __name__ == '__main__':
    # Get the port number from the environment variable PORT (default to 8050 for local development)
    port = int(os.environ.get("PORT", 8050))
    debug_mode = os.environ.get("DASH_DEBUG_MODE", "False").lower() == "true"
    app.run_server(debug=debug_mode, port=port, host='0.0.0.0')
