import urllib.request
import urllib.parse

dot = """
graph ER {
    layout=neato;
    overlap=false;
    splines=true;
    
    node [shape=box, style=filled, fillcolor="#90EE90", fontcolor="black"];
    Vocab [label="ISL Vocabulary"];
    Landmarks [label="Landmarks"];
    Model [label="ML Model"];
    Vector [label="Medical Vector"];
    
    node [shape=ellipse, style=filled, fillcolor="pink", fontcolor="black"]; Word_PK [label="Word (PK)"]; Video_URL; Local_Path;
    node [shape=ellipse, style=filled, fillcolor="lightblue", fontcolor="black"]; Row_ID [label="Row ID (PK)"]; Frame_Num; Coordinates;
    node [shape=ellipse, style=filled, fillcolor="gold", fontcolor="black"]; Model_ID [label="Model ID (PK)"]; Accuracy; File_Path;
    node [shape=ellipse, style=filled, fillcolor="lightgrey", fontcolor="black"]; Vector_ID [label="Vector ID (PK)"]; Embedding; Text_Chunk;
    
    node [shape=diamond, style=filled, fillcolor="orange", fontcolor="black"]; contains; predicts;
    
    Vocab -- Word_PK;
    Vocab -- Video_URL;
    Vocab -- Local_Path;
    
    Landmarks -- Row_ID;
    Landmarks -- Frame_Num;
    Landmarks -- Coordinates;
    
    Model -- Model_ID;
    Model -- Accuracy;
    Model -- File_Path;
    
    Vector -- Vector_ID;
    Vector -- Embedding;
    Vector -- Text_Chunk;
    
    Vocab -- contains [label="1"];
    contains -- Landmarks [label="N"];
    
    Model -- predicts [label="1"];
    predicts -- Vocab [label="N"];
}
"""

url = 'https://quickchart.io/graphviz?graph=' + urllib.parse.quote(dot) + '&format=svg'
print("Downloading Chen ER Diagram as SVG/HTML...")
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

with open('er_diagram.html', 'w', encoding='utf-8') as out_file:
    out_file.write(html_content)
print("Done!")
