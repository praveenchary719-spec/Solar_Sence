import sqlite3
import re

conn = sqlite3.connect('C:/Users/ayan3/.gemini/antigravity-ide/conversations/d4a85118-5e6e-4cf6-8b9b-c925254ab451.db')
cur = conn.cursor()
rows = cur.execute("SELECT idx, step_payload FROM steps ORDER BY idx ASC").fetchall()

for idx, sp in rows:
    # sp is bytes
    # find ascii/utf-8 strings of length >= 20
    text = sp.decode('utf-8', errors='ignore')
    # find user messages or tool calls
    matches = re.findall(r'[A-Za-z0-9_\-\s,\.\?\!\:\/\"\']{25,}', text)
    if 'USER_REQUEST' in text or 'USER_INPUT' in text or 'see the thing' in text.lower() or 'features' in text.lower() or 'not working' in text.lower():
        print(f"*** STEP {idx} MATCH ***")
        for m in matches:
            if any(k in m.lower() for k in ['feature', 'model', 'ensemble', 'working', 'antigravity', 'request', 'user', 'button', 'card', 'slider']):
                print("  ", m.strip())
