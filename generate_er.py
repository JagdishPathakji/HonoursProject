import zlib, base64, urllib.request

ptext = """
@startuml
!theme plain
hide circle
skinparam linetype ortho
skinparam class {
    BackgroundColor white
    BorderColor black
    ArrowColor black
}

entity "ISL_Vocabulary" as vocab {
  * Word : string <<PK>>
  --
  Video_URL : string
  Local_Path : string
  Category : string
}

entity "Gesture_Landmarks" as landmarks {
  * Row_ID : integer <<PK>>
  --
  * Word : string <<FK>>
  Frame_Number : integer
  X_Coords : float[]
  Y_Coords : float[]
  Z_Coords : float[]
}

entity "ML_Model" as model {
  * Model_ID : string <<PK>>
  --
  Model_Type : string
  File_Path : string
  Accuracy_Score : float
}

entity "Medical_Vector" as vector {
  * Vector_ID : string <<PK>>
  --
  Embedding_Array : float[]
  Text_Chunk : text
  Metadata : json
}

vocab ||..o{ landmarks : "HAS MANY"
model ||..o{ vocab : "PREDICTS"
@enduml
"""

z = zlib.compress(ptext.encode('utf-8'))
b = base64.b64encode(z[2:-4]).decode('utf-8')
b = b.translate(str.maketrans(
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/',
    '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_'
))
url = 'http://www.plantuml.com/plantuml/png/' + b
print('URL:', url)

req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response, open('er_diagram.png', 'wb') as out_file:
    out_file.write(response.read())

print('Downloaded er_diagram.png')
