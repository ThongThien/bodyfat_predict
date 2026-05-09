from ontology.ontology_engine import run_ontology

result = run_ontology(
    height=170,
    weight=65,
    chest=92,
    abdomen=85,
    hip=95,
    predicted_bf=18.7
)

print(result)