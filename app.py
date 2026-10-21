from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("model.pkl")


# -----------------------------
# Home Page
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')


# -----------------------------
# Predict House Price
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():

    # Get property details
    area_sqft = float(request.form['area'])
    bedrooms = int(request.form['bedrooms'])
    floors = int(request.form['floors'])
    age = int(request.form['age'])

    location_text = request.form['location']
    water_text = request.form['water']
    parking_text = request.form['parking']
    nearby_text = request.form['nearby']

    # Convert area using the same scaling as training
    area = area_sqft / 1000

    # Encode features to match the training dataset
    location = 1 if location_text == "town" else 0
    water = 1 if water_text == "yes" else 0
    parking = 1 if parking_text == "yes" else 0
    nearby = 1 if nearby_text == "yes" else 0

    # Prepare input for the model
    features = [[
        area,
        bedrooms,
        floors,
        age,
        location,
        water,
        parking,
        nearby
    ]]

    # Predict price
    prediction = model.predict(features)[0]

    # Property category
    if prediction < 2000000:
        category = "Affordable"
    elif prediction < 5000000:
        category = "Standard"
    elif prediction < 8000000:
        category = "Premium"
    else:
        category = "Luxury"

    # Property grade
    score = 0

    if location == 1:
        score += 2

    if water == 1:
        score += 1

    if parking == 1:
        score += 1

    if nearby == 1:
        score += 1

    if age < 10:
        score += 2
    elif age < 20:
        score += 1

    if score >= 7:
        grade = "A"
    elif score >= 5:
        grade = "B"
    elif score >= 3:
        grade = "C"
    else:
        grade = "D"

    # Property suggestions
    suggestions = []

    if parking == 0:
        suggestions.append(
            "Adding parking space may improve the property's appeal."
        )

    if water == 0:
        suggestions.append(
            "A reliable water facility may improve the property's appeal."
        )

    if nearby == 0:
        suggestions.append(
            "Check access to schools, hospitals and public transportation."
        )

    if age > 20:
        suggestions.append(
            "Consider inspecting and renovating older parts of the house."
        )

    if location == 0:
        suggestions.append(
            "Compare prices of similar properties in the surrounding area."
        )

    if not suggestions:
        suggestions.append(
            "Review similar local property listings before making a decision."
        )

    # Demo placeholder only; not measured model accuracy
    confidence = 92

    # Render the separate result page
    return render_template(
        'result.html',
        prediction_text=f"₹ {prediction:,.0f}",
        confidence=confidence,
        category=category,
        grade=grade,
        suggestions=suggestions,
        area=area_sqft,
        bedrooms=bedrooms,
        floors=floors,
        age=age,
        location=location_text,
        water=water_text,
        parking=parking_text,
        nearby=nearby_text
    )


# -----------------------------
# Run Flask
# -----------------------------
if __name__ == '__main__':
    app.run(debug=True)