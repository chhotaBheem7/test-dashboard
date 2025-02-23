import dash
from dash import html, dcc, Input, Output, State, callback_context
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
feature_names = ['Glucose', 'BMI', 'Age', 'Pregnancies', 'DiabetesPedigreeFunction', 'Insulin']
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
class_names = np.unique(y_test)
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

try:
    df_new = df.drop('PatientID', axis=1)
except KeyError:  # Handle the case where 'PatientID' is not present
    df_new = df.copy()

num_rows = len(df.axes[0])

age_counts = df_new['Age'].value_counts().sort_index()

df['Outcome'] = df['Outcome'].astype('category')

# Get predicted probabilities for the positive class
y_scores = model.predict_proba(X_test)[:, 1]

# Create the custom color scale for heatmap\matrix
custom_colorscale = px.colors.make_colorscale(['#0081A7', '#F07167'])

def safe_convert_inputs(input_dict: dict) -> (dict, str):
    safe_dict = {}
    error_message = ""
    for key, value in input_dict.items():
        if value is None or value == "":  # Handle empty strings
            safe_dict[key] = None
            continue
        try:
            if key == 'Pregnancies':
                safe_dict[key] = int(value)
            else:
                safe_dict[key] = float(value)
        except ValueError:
            error_message = "Invalid input. Please enter numbers only."
            break
    return safe_dict, error_message

# Dash app initialization
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Initialize dropdown options (handle empty DataFrame)
available_columns = df_new.columns.tolist() if not df_new.empty else []
dropdown_options = [{'label': col, 'value': col} for col in available_columns]

initial_x = available_columns[0] if available_columns else None  # Default x
initial_y = available_columns[1] if len(available_columns) > 1 else None  # Default y

# Create initial figures (empty or default if no data)
fig1 = px.pie(outcome_counts, names='Outcome', values='Count', color="Outcome", color_discrete_sequence=['#0081A7', '#F07167']) if not df.empty else {}
fig2 = px.scatter(df_new, x=initial_x, y=initial_y, color_discrete_sequence=['#0081A7', '#F07167']) if not df_new.empty else {}
fig3 = px.box(df_new, x=initial_x, color_discrete_sequence=['#0081A7', '#F07167']) if not df_new.empty else {}
fig5 = px.violin(df_new, x="Outcome", y=initial_y, color="Outcome", box=True, points="all", color_discrete_sequence=['#0081A7', '#F07167']) if not df_new.empty else {}
fig6 = px.imshow(df_cm, labels=dict(x="Predicted", y="Actual", color="Count"), x=df_cm.columns, y=df_cm.index, color_continuous_scale=custom_colorscale, text_auto=True) if not df_cm.empty else {}
fig6.update_layout(coloraxis_showscale=False)
fig7 = px.pie(fig7_accuracy, values='Value', names='Category', color_discrete_sequence=['#0081A7', '#F07167'], hole=0.6) if fig7_accuracy else {}
if fig7:
    fig7.update_layout(showlegend=False)
    fig7.update_traces(textinfo='none')
    fig7.add_annotation(text=f"<b>{percentage:.2f}%</b>", x=0.5, y=0.5, font=dict(size=26, family="Arial", color="black"), showarrow=False)

app.layout = html.Div(style={'backgroundColor': 'white'}, className="container-fluid", children=[
html.Div([
    dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    "Pima Diabetes Dashboard", className="navbar-brand_top px-3 fw-bold fs-4 Sticky top my-link my-link:hover", href="https://en.wikipedia.org/wiki/Akimel_O%27odham"
                ),
            ],
fluid=True,  # Make the container fluid to use up full width
            className="px-3" # Add padding to the container
        ),
        className="my-custom-navbar_top",
    ),
    html.Div(id="content", style={"margin-top": "10px"}),  # Prevent overlap
]),
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
            dbc.Card(children=[html.Div(className="card-header", style={"background-color": "#0081A7", "color": "white"}, children=['Binary Feature Outcome']),
                dbc.CardBody(children=[
                    dcc.Graph(figure=fig1),
                    html.P(f"Number of Data Points: {num_rows}", className="card-text", style={'text-align': 'center'}),
                ])
            ], style={"height": "100%"})
        ]),
        html.Div(className="col-md-3", children=[
            dbc.Card( children=[
                html.Div( className="card-header", style={"background-color": "#0081A7", "color": "white"},
                          children=["Predictor Features versus Diabetes Outcome"] ),
                dbc.CardBody( children=[
                    dcc.Graph( id='diabetes-distribution-chart' ),
                    dcc.Dropdown(
                        id='y-axis-dropdown1',
                        options=dropdown_options,
                        value=initial_y,
                        clearable=False
                    ),
                ] )
            ], style={"height": "100%"} )
        ] ),
        html.Div(className="col-md-3", children=[
            dbc.Card(children=[html.Div(className="card-header", style={"background-color": "#0081A7", "color": "white"}, children=["Relationship Between Variables"]),
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
            dbc.Card(children=[html.Div(className="card-header", style={"background-color": "#0081A7", "color": "white"}, children=["Number of pregnancies"]),
                dbc.CardBody(children=[
                    dcc.Graph(id='boxplot', figure=fig3),
                    dcc.Dropdown(
                        id='x-axis-dropdown3',
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
                dbc.Card(children=[html.Div(className="card-header", style={"background-color": "#0081A7", "color": "white"}, children=["Confusion Matrix"]),
                 dbc.CardBody(children=[
                    dcc.Graph(figure=fig6),
                ])
                   ], style={"height": "100%"})
        ]),
         html.Div(className="col-md-3", children=[
            dbc.Card(children=[html.Div(className="card-header", style={"background-color": "#0081A7", "color": "white"}, children=["Accuracy of the model"]),
                dbc.CardBody(children=[
                    dcc.Graph(figure=fig7),
                ])
                   ], style={"height": "100%"})
         ]),
             html.Div(className="col-md-3", children=[
                 dbc.Card(children=[html.Div(className="card-header", style={"background-color": "#0081A7", "color": "white"}, children=["Classification report"]),
                     dbc.CardBody(children=[
                         html.Br(),
                         html.Br(),
                         html.Br(),
                         dbc.Table.from_dataframe(df_report.round(2), class_name="table table-bordered table-responsive")
                     ])
                 ], style={"height": "100%", "margin": "0 auto"})
             ] ),
    html.Div(className="col-md-3", children=[
        dbc.Card(children=[
            html.Div(className="card-header", style={"background-color": "#0081A7", "color": "white"}, children=[ "Pima Diabetes Predictor" ]),
            dbc.CardBody(
                children=[
                    html.H4('Diabetes classification: '),
                    html.Div(
                        id='prediction_output',
                        style={
                            'height': '100px',
                            'margin-top': '30px',
                            'margin-left': '20px',
                            'font-weight': 'bold',
                            'font-size': '30px'
                        }
                    ),
                    html.Br(),
                    html.H5('Enter values in all fields below: '),
                    html.Br(),
                    dbc.Form(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Input( id="Age", type="text", className="form-control", placeholder="Age" ),
                                        className="col",
                                    ),
                                    dbc.Col(
                                        dcc.Input( id="BMI", type="text", className="form-control", placeholder="BMI" ),
                                        className="col",
                                    ),
                                    dbc.Col(
                                        dcc.Input( id="Glucose", type="text", className="form-control",
                                                   placeholder="Glucose" ),
                                        className="col",
                                    ),
                                ]
                            ),
                            html.Br(),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dcc.Input( id="Pregnancies", type="text", className="form-control",
                                                   placeholder="Pregnancies" ),
                                        className="col",
                                    ),
                                    dbc.Col(
                                        dcc.Input( id="DPF", type="text", className="form-control", placeholder="DPF" ),
                                        className="col",
                                    ),
                                    dbc.Col(
                                        dcc.Input( id="Insulin", type="text", className="form-control",
                                                   placeholder="Insulin" ),
                                        className="col",
                                    ),
                                ]
                            ),
                            html.Br(),
                            dbc.Row( [
                                dbc.Col(
                                    dbc.Button(
                                        "Submit",
                                        color="primary",
                                        className="btn btn-primary btn-lg",
                                        id="submit-button",
                                    ),
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Clear",
                                        color="primary",
                                        className="btn btn-primary btn-lg",
                                        id="clear-button",
                                    ),
                                ),
                            ] ),
                        ]
                    ),
                    html.Br(),
                    html.Div( id='error-message', style={'color': 'red', 'margin-top': '5px'} ),
                    html.Div( id="output-text" ),
                ]
            )
        ], style={"height": "100%", "margin": "0 auto"})
    ], style={"padding-bottom": "10px"} ),
    ]),

    html.Div( [
        dbc.Navbar(
            dbc.Container(
                [
                ],
                fluid=True,  # Make the container fluid to use up full width
                className="px-3"  # Add padding to the container
            ),
            className="my-custom-navbar_bottom",
        ),
        html.Div( id="content", style={"margin-bottom": "10px"} ),  # Prevent overlap
    ])
]),


@app.callback(
    Output( 'boxplot', 'figure',),
    Input( 'x-axis-dropdown3', 'value' ),
    prevent_initial_call=True  # add this line
)
def update_boxplot(x_value):
    if x_value and not df.empty and x_value in df.columns:  # Check if column exists
        fig = px.box( df, x=x_value, color_discrete_sequence=['#0081A7', '#F07167'] )
        fig.update_layout( title=f"{x_value}" )
        return fig
    return px.box( df ) if not df.empty else {}  # Return empty or default plot

@app.callback(
    Output( 'diabetes-distribution-chart', 'figure' ),
    Input( 'y-axis-dropdown1', 'value' ),
)
def update_age_distribution_chart(selected_y):  # Correct: Only selected_y as input
    fig = px.violin(
        df_new,  # Use the global df_new DataFrame directly
        x="Outcome",
        y=selected_y,
        color="Outcome",
        box=True,
        points="all",  # Consider points="outliers" or False for large datasets
        color_discrete_sequence=['#0081A7', '#F07167'],
        title=f"Distribution of {selected_y} by Outcome"  # Add a title
    )

    fig.update_layout(
        xaxis_title="Outcome",
        yaxis_title=selected_y,
    )

    return fig

@app.callback(
    Output('scatter-chart', 'figure'),
    Input('x-axis-dropdown2', 'value'),
    Input('y-axis-dropdown2', 'value')
)
def update_scatter_chart(x_value, y_value):
    if x_value and y_value and not df.empty and x_value in df.columns and y_value in df.columns:
        fig = px.scatter(df_new, x=x_value, y=y_value, color_discrete_sequence=['#0081A7', '#F07167'])
        fig.update_layout(title=f"{x_value} vs {y_value}")
        return fig
    return px.scatter(df_new) if not df.empty else {} # Return empty or default plot


outputs = [Output('Age', 'value'),
           Output('BMI', 'value'),
           Output('Glucose', 'value'),
           Output('Pregnancies', 'value'),
           Output('DPF', 'value'),
           Output('Insulin', 'value'),
           Output('prediction_output', 'children'),
           Output('error-message', 'children')]  # Define outputs ONCE

@app.callback(
    outputs,
    [Input('submit-button', 'n_clicks'), Input('clear-button', 'n_clicks')],
    [State(field, 'value') for field in ['Age', 'BMI', 'Glucose', 'Pregnancies', 'DPF', 'Insulin']]
)
def update_or_clear_inputs(submit_n_clicks, clear_n_clicks, *args):
    ctx = callback_context

    if not ctx.triggered:  # Initial load
        return ['', '', '', '', '', '', "", ""]  # Return empty strings for all outputs

    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == 'clear-button':
        return ['', '', '', '', '', '', "", ""]  # Clear all input fields and messages

    elif triggered_id == 'submit-button':
        if submit_n_clicks is None:  # Handle initial state when button hasn't been clicked
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update # No updates

        required_fields = ['Age', 'BMI', 'Glucose', 'Pregnancies', 'DPF', 'Insulin']
        input_dict = dict(zip(required_fields, args))
        safe_input_dict, error_message = safe_convert_inputs(input_dict)  # Assuming this function handles conversions and basic validation

        if error_message:
            return *args, "", error_message  # Return input values, empty prediction, and the error

        missing_fields = [field for field, value in safe_input_dict.items() if value is None or value == ""]
        if missing_fields:
            error_message = f"The following fields are required: {', '.join(missing_fields)}."
            return *args, "", error_message  # Return inputs, empty prediction, and error

        all_none = all(value is None for value in safe_input_dict.values())
        if all_none:
            error_message = "All input values are missing or invalid."
            return *args, "", error_message # Return inputs, empty prediction, and error

        new_data = np.array([[safe_input_dict[field] for field in required_fields]])

        try:
            new_data_scaled = scaler.transform(new_data)  # Make sure 'scaler' is defined and fitted
            prediction = model.predict(new_data_scaled)[0]  # Make sure 'model' is defined and loaded
            if prediction == 1:
                prediction_output = "Diabetic"
            else:
                prediction_output = "Non-Diabetic"
            return *args, prediction_output, ""  # Return inputs, prediction, and empty error message

        except ValueError as e:
            error_message = f"Scaling Error: {e}. Check input values and data types."
            return *args, "", error_message  # Return inputs, empty prediction, and the error
        except AttributeError as e:  # Catch potential AttributeError if scaler not fitted
            error_message = f"Error: {e}. Please make sure the model and scaler are correctly loaded and fitted."
            return *args, "", error_message # Return inputs, empty prediction, and the error
        except Exception as e: # Catch any other exceptions
            error_message = f"A prediction error occurred: {e}"
            return *args, "", error_message # Return inputs, empty prediction, and the error


if __name__ == '__main__':
    # Get the port number from the environment variable PORT (default to 8050 for local development)
    port = int(os.environ.get("PORT", 8050))
    debug_mode = os.environ.get("DASH_DEBUG_MODE", "False").lower() == "true"
    app.run_server(debug=debug_mode, port=port, host='0.0.0.0')
