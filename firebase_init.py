import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin import credentials, firestore

cred = credentials.Certificate("/secrets/firebase.json")
firebase_admin.initialize_app(cred)

# cred = credentials.Certificate("./amoria.json")
# firebase_admin.initialize_app(cred)

db = firestore.client()
