import pickle

model = pickle.load(open('house_model.pkl', 'rb'))

def prediction(data):
    encoded = {
        'area': data['area'],
        'bedrooms': data['bedrooms'],
        'bathrooms': data['bathrooms'],
        'stories': data['stories'],
        'mainroad': 1 if data['mainroad'] == 'yes' else 0,
        'guestroom': 1 if data['guestroom'] == 'yes' else 0,
        'basement': 1 if data['basement'] == 'yes' else 0,
        'hotwaterheating': 1 if data['hotwaterheating'] == 'yes' else 0,
        'airconditioning': 1 if data['airconditioning'] == 'yes' else 0,
        'parking': data['parking'],
        'prefarea': 1 if data['prefarea'] == 'yes' else 0,
        'furnishingstatus': {'furnished': 0, 'semi-furnished': 1, 'unfurnished': 2}[data['furnishingstatus']]
    }
    features = [encoded[col] for col in ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']]
    return model.predict([features])