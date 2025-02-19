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
feature_names = ['Glucose', 'BMI', 'Age', 'Pregnancies', 'DiabetesPedigreeFunction', 'Insulin'],
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

# Renaming one column
df = df.rename(columns={'DiabetesPedigreeFunction': 'DPF'})

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)
class_names = np.unique(y_test)  # Or specify manually
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)

# Classification Report (precision, recall, F1-score)
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report = df_report.drop('accuracy', axis=0)
df_report.insert(0, '', ["No diabetes", "Diabetes", "Macro avg", "Weighted avg"])

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
                         # dcc.Graph(figure=fig4),
                         html.Br(),
                         html.H5("Classification report"),
                         html.Br(),
                         dbc.Table.from_dataframe(df_report.round(2), class_name="table table-bordered table-responsive")
                     ])
                 ], style={"height": "100%", "margin": "0 auto"})
             ] ),
    html.Div(className="col-md-3", children=[
        dbc.Card(children=[
            dbc.CardBody(children=[
                    html.Br(),
                    html.H5( "Diabetes Prediction" ),  # More descriptive heading
                    html.Br(),
                    html.Br(),
                    dbc.Form( [
                        dbc.Row( [
                            dbc.Col(
                                dcc.Input( id="Age", type="text", className="form-control", placeholder="Age" ),
                                className="col" ),
                            dbc.Col(
                                dcc.Input( id="BMI", type="text", className="form-control", placeholder="BMI" ),
                                className="col" ),
                            dbc.Col( dcc.Input( id="Glucose", type="text", className="form-control",
                                                placeholder="Glucose" ), className="col" ),
                        ] ),
                        html.Br(),
                        dbc.Row( [
                            dbc.Col( dcc.Input( id="Pregnancies", type="text", className="form-control",
                                                placeholder="Pregnancies" ), className="col" ),
                            dbc.Col(
                                dcc.Input( id="DPF", type="text", className="form-control", placeholder="DPF" ),
                                className="col" ),
                            dbc.Col( dcc.Input( id="Insulin", type="text", className="form-control",
                                                placeholder="Insulin" ), className="col" ),
                        ] ),
                        html.Br(),
                        dbc.Button( "Submit", color="primary", className="btn btn-primary btn-lg btn-block",
                                    id="submit-button" ),
                    ] ),
                    html.Br(),
                    html.Br(),
                    html.Div( id='prediction-output', style={'margin-top': '10px', 'margin-left': "20px", 'font-weight': 'bold', 'font-size': '40px'}),
                    # Output area for prediction
                    html.Div( id='error-message', style={'color': 'red', 'margin-top': '5px'} ),
                    # Error message area
                    html.Div(id="output-text"),
                    html.Br(),
            ])
        ], style={"height": "100%"})
    ]),
], style={"padding-bottom": "10px"})
]),
]),


@app.callback(
    Output('boxplot', 'figure'),
    Input('x-axis-dropdown1', 'value')
)
def update_boxplot(x_value):
    if x_value and not df.empty and x_value in df.columns:  # Check if column exists
        fig = px.box(df, x=x_value, color_discrete_sequence=['#0081A7', '#F07167'])
        fig.update_layout(title=f"{x_value}")
        return fig
    return px.box(df) if not df.empty else {} # Return empty or default plot

@app.callback(
    Output('scatter-chart', 'figure'),
    Input('x-axis-dropdown2', 'value'),
    Input('y-axis-dropdown2', 'value')
)
def update_scatter_chart(x_value, y_value):
    if x_value and y_value and not df.empty and x_value in df.columns and y_value in df.columns:
        fig = px.scatter(df, x=x_value, y=y_value, color_discrete_sequence=['#0081A7', '#F07167'])
        fig.update_layout(title=f"{x_value} vs {y_value}")
        return fig
    return px.scatter(df) if not df.empty else {} # Return empty or default plot

@app.callback(
    Output('prediction-output', 'children'),
    Output('error-message', 'children'),
    Input('submit-button', 'n_clicks'),
    State('Age', 'value'),
    State('BMI', 'value'),
    State('Glucose', 'value'),
    State('Pregnancies', 'value'),
    State('DPF', 'value'),
    State('Insulin', 'value')
)
def update_prediction(n_clicks, age, bmi, glucose, pregnancies, dpf, insulin):
    if n_clicks is None:
        return "", ""

    error_message = ""
    try:
        # Improved input validation and conversion
        try:
            age = float(age) if age is not None else None
            bmi = float(bmi) if bmi is not None else None
            glucose = float(glucose) if glucose is not None else None
            pregnancies = int(pregnancies) if pregnancies is not None else None
            dpf = float(dpf) if dpf is not None else None
            insulin = float(insulin) if insulin is not None else None
        except ValueError:
            error_message = "Invalid input. Please enter numbers only."
            return "", error_message

        required_fields = {"Age": age, "BMI": bmi, "Glucose": glucose, 'Pregnancies': pregnancies, 'DPF': dpf, 'Insulin': insulin}
        missing_fields = [field for field, value in required_fields.items() if value is None]
        if missing_fields:
            error_message = f"The following fields are required: {', '.join(missing_fields)}."
            return "", error_message

        # Consistent feature order – VERY IMPORTANT!
        new_data = np.array([[glucose, bmi, age, pregnancies, dpf, insulin]])  # Correct order
        new_data_scaled = scaler.transform(new_data)
        prediction = model.predict(new_data_scaled)[0]

        output_text = f"Prediction: {'Diabetes' if prediction == 1 else 'No Diabetes'}"
        return output_text, ""

    except Exception as e:  # Catching a broader exception can be useful for debugging
        error_message = f"An error occurred: {str( e )}"  # More informative error message
        return "", error_message


if __name__ == '__main__':
    # Get the port number from the environment variable PORT (default to 8050 for local development)
    port = int(os.environ.get("PORT", 8050))
    debug_mode = os.environ.get("DASH_DEBUG_MODE", "False").lower() == "true"
    app.run_server(debug=debug_mode, port=port, host='0.0.0.0')
