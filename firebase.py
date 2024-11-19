import firebase_admin
from firebase_admin import credentials,db
# initialize
def init_firebase_connection(jfile,URL):
    # cred = credentials.Certificate("secrets_dir/emotiondetection-c1441-firebase-adminsdk-igmnb-f0e4b0da40.json")
    cred = credentials.Certificate(jfile)
    firebase_admin.initialize_app(cred,{
        'databaseURL':URL
        })
def set_FirebaseRefrence(path):
    ref =  db.reference(path)
    return ref
# write
def set_Value(score,type,refObj): 
    refObj.set({'score':score,'type':type})
# read 
def read_Value(refObj):
    val = refObj.get()
    return val


