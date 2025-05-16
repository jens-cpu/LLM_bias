import json

input_file = "persona.jsonl"
output_file = "persona_reduced.jsonl"

seen_personas = set()

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        try:
            data = json.loads(line)
            # Nutze "id" als eindeutige Persona-Identifikation
            persona_id = data.get("id")
            if persona_id is None:
                # Falls keine ID, nutze kompletten JSON-String als Key
                persona_id = json.dumps(data, sort_keys=True)

            if persona_id not in seen_personas:
                seen_personas.add(persona_id)
                outfile.write(line)
        except json.JSONDecodeError:
            continue

print(f"Fertig! {len(seen_personas)} einzigartige Personas in '{output_file}'.")
