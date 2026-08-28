import zlib, base64, urllib.request

ptext = """
@startuml
left to right direction
skinparam packageStyle rectangle
skinparam usecase {
    BackgroundColor white
    BorderColor black
    ArrowColor black
}
skinparam actor {
    BackgroundColor white
    BorderColor black
}

actor "Deaf User" as DU
actor "Hearing User" as HU
actor "System Administrator" as SA

rectangle "Lingua ISL System" {
  usecase "Converts Gestures to Text" as UC1
  usecase "Interact with Medical Chatbot" as UC2
  usecase "Convert Speech to ISL" as UC3
  usecase "Convert Text to ISL" as UC4
  usecase "Retrieve Medical Content" as UC5
  usecase "Process & Lemmatize Text" as UC6
  usecase "Fetch Landmark Coordinates" as UC7
  usecase "Manage Video Dictionary" as UC8
  usecase "Retrieve Classification Models" as UC9
}

DU --> UC1
DU --> UC2
HU --> UC3
HU --> UC4
SA --> UC8
SA --> UC9

UC2 .> UC5 : <<include>>
UC2 .> UC4 : <<include>>
UC3 .> UC4 : <<include>>
UC4 .> UC6 : <<include>>
UC4 .> UC7 : <<include>>
@enduml
"""

z = zlib.compress(ptext.encode('utf-8'))
b = base64.b64encode(z[2:-4]).decode('utf-8')
b = b.translate(str.maketrans(
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/',
    '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_'
))
url = 'http://www.plantuml.com/plantuml/svg/' + b
print('URL:', url)

req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response:
    svg_data = response.read().decode('utf-8')

html_content = f'''<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ background-color: white; padding: 50px; display: flex; justify-content: center; }}
        svg {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    {svg_data}
</body>
</html>'''

with open('usecase_diagram.html', 'w', encoding='utf-8') as out_file:
    out_file.write(html_content)

print('Downloaded usecase_diagram.html')
