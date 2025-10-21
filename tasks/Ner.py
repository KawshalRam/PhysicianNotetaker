import json
import re
from typing import Dict, List, Optional
from collections import defaultdict

import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc

# ----------------- Optional Imports -----------------
try:
    from scispacy.linking import EntityLinker
    from scispacy.abbreviation import AbbreviationDetector
    SCISPACY_LINKING_AVAILABLE = True
except ImportError:
    SCISPACY_LINKING_AVAILABLE = False
    print("⚠️ SciSpaCy linker not available. Install via: pip install scispacy")

try:
    from transformers import pipeline
    from word2number import w2n
    BIOBERT_AVAILABLE = True
except Exception:
    BIOBERT_AVAILABLE = False
    print("⚠️ BioBERT NER unavailable. Install `transformers word2number` for full functionality.")


# ----------------- Helper Utilities -----------------
GENERIC_NOISE = {
    "yes", "no", "ok", "okay", "good", "morning", "afternoon", "evening",
    "hello", "hi", "hey", "patient", "doctor", "physician", "today",
    "feeling", "examination", "check", "diagnosed", "with"
}

RELEVANT_SEMANTIC_TYPES = {
    "T184", "T033",  # Signs/Symptoms
    "T047", "T048", "T191", "T046",  # Diseases/Disorders
    "T060", "T061", "T058", "T121", "T200"  # Procedures/Drugs
}


def word_to_int(token: str) -> Optional[int]:
    token = token.lower().strip()
    if token.isdigit():
        return int(token)
    try:
        return w2n.word_to_num(token)
    except Exception:
        return None


def clean_token(tok: str) -> str:
    return tok.replace("##", "").strip()


def uniq_preserve_order(items: List[str]) -> List[str]:
    seen, out = set(), []
    for it in items:
        key = it.lower().strip()
        if key and key not in seen and not key.startswith('c0'):  # Filter CUI codes
            seen.add(key)
            out.append(it)
    return out


# ----------------- Core Extractor Class -----------------
class MedicalExtractor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_sci_md")
            print("✅ SciSpaCy model loaded.")
        except Exception:
            self.nlp = spacy.load("en_core_web_sm")
            print("⚠️ Fallback to en_core_web_sm")

        try:
            self.nlp.add_pipe("abbreviation_detector", last=True)
            print("✅ Abbreviation detector active.")
        except Exception as e:
            print(f"⚠️ Abbreviation detector failed: {e}")

        self.entity_linker = None
        if SCISPACY_LINKING_AVAILABLE:
            try:
                print("🔄 Loading UMLS linker...")
                self.nlp.add_pipe(
                    "scispacy_linker",
                    config={
                        "resolve_abbreviations": True,
                        "linker_name": "umls",
                        "threshold": 0.70
                    }
                )
                self.entity_linker = self.nlp.get_pipe("scispacy_linker")
                print("✅ UMLS linker integrated.")
            except Exception as e:
                print(f"⚠️ Could not initialize UMLS linker: {e}")

        self.matcher = Matcher(self.nlp.vocab)
        self._setup_patterns()

        self.ner = None
        if BIOBERT_AVAILABLE:
            try:
                self.ner = pipeline(
                    "ner",
                    model="HUMADEX/english_medical_ner",
                    aggregation_strategy="simple"
                )
                print("✅ BioBERT loaded successfully.")
            except Exception as e:
                print(f"⚠️ BioBERT initialization failed: {e}")

        self.biobert_map = {
            "SIGN_SYMPTOM": "SYMPTOM",
            "SYMPTOM": "SYMPTOM",
            "DISEASE_DISORDER": "DIAGNOSIS",
            "CONDITION": "DIAGNOSIS",
            "PROCEDURE": "TREATMENT",
            "DRUG": "TREATMENT",
            "MEDICATION": "TREATMENT",
            "TEST": "TREATMENT"
        }

    def _setup_patterns(self):
        # Symptom patterns
        symptom_patterns = [
            [{"POS": {"IN": ["ADJ", "NOUN"]}, "OP": "+"}, {"LEMMA": {"IN": ["pain", "ache", "discomfort", "stiffness"]}}],
            [{"LEMMA": {"IN": ["pain", "ache", "hurt"]}}, {"POS": "ADP", "OP": "?"}, {"POS": "DET", "OP": "?"}, {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"}],
        ]
        for pattern in symptom_patterns:
            self.matcher.add("SYMPTOM", [pattern])

        # Treatment patterns
        treatment_patterns = [
            [{"LIKE_NUM": True}, {"LOWER": {"IN": ["physiotherapy", "therapy", "session", "sessions"]}}],
            [{"LIKE_NUM": True}, {"LOWER": {"IN": ["session", "sessions"]}}, {"LOWER": "of"}, {"POS": "NOUN", "OP": "+"}],
            [{"LOWER": {"IN": ["physical", "occupational"]}}, {"LOWER": "therapy"}],
        ]
        for pattern in treatment_patterns:
            self.matcher.add("TREATMENT", [pattern])

    def parse_speakers(self, text: str) -> Dict[str, str]:
        speakers = defaultdict(list)
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split(":", 1)
            if len(parts) == 2:
                speaker, content = parts[0].lower(), parts[1].strip()
                if any(r in speaker for r in ["doctor", "physician", "dr"]):
                    speakers["DOCTOR"].append(content)
                elif "patient" in speaker:
                    speakers["PATIENT"].append(content)
                else:
                    speakers["OTHER"].append(content)
            else:
                speakers["OTHER"].append(line)
        return {k: " ".join(v) for k, v in speakers.items()}

    def extract_patient_name(self, text: str) -> str:
        """Extract patient name using multiple strategies"""
        patterns = [
            r"Patient\s*(?:Name)?\s*:\s*(?:Mr\.?|Mrs\.?|Ms\.?|Miss\.?|Dr\.?)?\s*([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)",
            r"(?:Mr\.?|Mrs\.?|Ms\.?|Miss\.?)\s+([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)",
            r"Name\s*(?:is|:)?\s*([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                name = match.group(1).strip()
                if name.lower() not in GENERIC_NOISE and len(name) > 3:
                    return name
        
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name = ent.text.strip()
                words = name.split()
                if (len(words) >= 2 and 
                    name.lower() not in GENERIC_NOISE and
                    all(w.lower() not in {"patient", "doctor", "physician"} for w in words)):
                    return name
        
        return "Unknown"

    def extract_biobert(self, text: str) -> Dict[str, List[str]]:
        """Extract entities using BioBERT NER"""
        out = defaultdict(list)
        if not self.ner:
            return out
        try:
            ents = self.ner(text)
            for e in ents:
                raw_label = e.get("entity_group", "").upper()
                mapped = self.biobert_map.get(raw_label)
                if not mapped:
                    continue
                term = clean_token(e.get("word", "")).strip()
                if term and term.lower() not in GENERIC_NOISE and len(term) > 2:
                    out[mapped].append(term.title())
        except Exception as ex:
            print(f"⚠️ BioBERT error: {ex}")
        return dict(out)

    def extract_with_matcher(self, doc: Doc) -> Dict[str, List[str]]:
        """Extract entities using spaCy matcher patterns"""
        results = defaultdict(list)
        seen_spans = set()  # Track seen text spans to avoid duplicates
        
        for match_id, start, end in self.matcher(doc):
            label = self.nlp.vocab.strings[match_id]
            span = doc[start:end]
            text = span.text.strip().lower()
            
            # Skip if already seen or too short
            if text in seen_spans or len(text) < 3 or text in GENERIC_NOISE:
                continue
            
            seen_spans.add(text)
            results[label].append(span.text.strip())
        
        return results

    def get_umls_canonical_name(self, cui: str) -> Optional[str]:
        """Get the canonical/preferred name for a UMLS CUI"""
        try:
            kb = self.entity_linker.kb
            
            if hasattr(kb, 'cui_to_entity') and cui in kb.cui_to_entity:
                concept = kb.cui_to_entity[cui]
                if concept and len(concept) > 0:
                    name = concept[0]
                    if isinstance(name, bytes):
                        name = name.decode('utf-8')
                    name = str(name).strip()
                    
                    if not name.startswith('C0') and len(name) > 2:
                        return name
            
            if hasattr(kb, 'cui_to_name') and cui in kb.cui_to_name:
                return kb.cui_to_name[cui]
            
            if hasattr(kb, 'get_concept_name'):
                name = kb.get_concept_name(cui)
                if name and not name.startswith('C0'):
                    return name
                    
        except Exception:
            pass
        
        return None

    def extract_umls_entities(self, doc: Doc) -> Dict[str, List[Dict[str, str]]]:
        """Extract and validate medical entities using UMLS"""
        entities = defaultdict(list)
        if not self.entity_linker:
            return entities

        for ent in doc.ents:
            if hasattr(ent._, "kb_ents") and ent._.kb_ents:
                best_cui, score = ent._.kb_ents[0]
                
                if score < 0.70:
                    continue
                
                try:
                    canonical_name = self.get_umls_canonical_name(best_cui)
                    
                    if not canonical_name or canonical_name.startswith('C0') or len(canonical_name) < 2:
                        canonical_name = ent.text
                    
                    kb = self.entity_linker.kb
                    concept = kb.cui_to_entity.get(best_cui)
                    semantic_types = concept[3] if concept and len(concept) > 3 else []
                    
                    if not semantic_types:
                        continue
                    
                    tui = semantic_types[0]
                    
                    category = None
                    if tui in {"T184", "T033"}:
                        category = "SYMPTOM"
                    elif tui in {"T047", "T048", "T191", "T046"}:
                        category = "DIAGNOSIS"
                    elif tui in {"T060", "T061", "T058", "T121", "T200"}:
                        category = "TREATMENT"
                    
                    if category:
                        entities[category].append({
                            "text": canonical_name.title(),
                            "original": ent.text,
                            "cui": best_cui,
                            "confidence": round(score, 3)
                        })
                        
                except Exception:
                    continue
        
        return dict(entities)

    def extract_numeric_treatments(self, doc: Doc) -> List[str]:
        """Extract treatments with numeric quantities"""
        treatments = []
        
        for i, token in enumerate(doc):
            if token.like_num or token.text.lower() in ["one", "two", "three", "four", "five", 
                                                         "six", "seven", "eight", "nine", "ten"]:
                num = word_to_int(token.text)
                if num is None:
                    continue
                
                for j in range(i + 1, min(i + 5, len(doc))):
                    next_token = doc[j]
                    if next_token.lemma_ in {"physiotherapy", "therapy", "session", "appointment"}:
                        therapy_type = "Physiotherapy"
                        for k in range(max(0, i), min(len(doc), j + 3)):
                            if doc[k].text.lower() in {"physiotherapy", "physical"}:
                                therapy_type = "Physiotherapy"
                                break
                        
                        treatments.append(f"{num} {therapy_type} Sessions")
                        break
        
        return treatments

    def extract_diagnosis(self, text: str, merged_diagnoses: List[str]) -> str:
        """Extract diagnosis using multiple strategies"""
        if merged_diagnoses:
            return merged_diagnoses[0].title()
        
        diagnosis_patterns = [
            r"Diagnosis\s*:\s*([A-Za-z\s\-]+?)(?:\.|$)",
            r"diagnosed\s+with\s+([A-Za-z\s\-]+?)(?:\.|,|$)",
            r"suffering\s+from\s+([A-Za-z\s\-]+?)(?:\.|,|$)",
        ]
        
        for pattern in diagnosis_patterns:
            match = re.search(pattern, text, re.I)
            if match:
                diagnosis = match.group(1).strip()
                if diagnosis.lower() not in GENERIC_NOISE and len(diagnosis) > 3:
                    return diagnosis.title()
        
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in {"DISEASE", "CONDITION", "DISORDER"}:
                return ent.text.title()
        
        return "Not specified"

    def extract_current_status(self, text: str) -> str:
        """Extract current patient status"""
        patterns = [
            r"(?:currently|now|presently|at present|still)\s+(?:experience|experiences|has|have|feel|feels)\s+(.+?)(?:\.|,|$)",
            r"(?:only|just)\s+(?:have|has)\s+(.+?)(?:\.|,|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                status = match.group(1).strip()
                status = re.sub(r'\s+', ' ', status)
                if len(status) > 5 and status.lower() not in GENERIC_NOISE:
                    return status.capitalize()
        
        return "Not specified"

    def extract_prognosis(self, text: str) -> str:
        """Extract prognosis information"""
        patterns = [
            r"((?:full|complete)\s+recovery\s+(?:is\s+)?(?:expected|anticipated)\s+(?:within|in)\s+[^.]+)",
            r"((?:should|will|expected to)\s+(?:recover|heal|improve)\s+(?:within|in)\s+[^.]+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                prognosis = match.group(1).strip()
                return prognosis.capitalize()
        
        return "Not specified"

    def summarize(self, text: str) -> Dict:
        """Extract and summarize medical information from conversation"""
        text = " ".join(text.split())
        speakers = self.parse_speakers(text)
        doc = self.nlp(text)

        patient_name = self.extract_patient_name(text)

        matcher_ents = self.extract_with_matcher(doc)
        biobert_ents = self.extract_biobert(text)
        umls_ents = self.extract_umls_entities(doc)
        numeric_treatments = self.extract_numeric_treatments(doc)

        merged = defaultdict(list)
        
        for category, entities in umls_ents.items():
            for entity_dict in entities:
                entity_text = entity_dict['text']
                if not entity_text.startswith('C0') and len(entity_text) > 2:
                    merged[category].append(entity_text)
                else:
                    merged[category].append(entity_dict['original'])
        
        for key in ["SYMPTOM", "DIAGNOSIS", "TREATMENT"]:
            merged[key].extend(biobert_ents.get(key, []))
            merged[key].extend(matcher_ents.get(key, []))
        
        merged["TREATMENT"].extend(numeric_treatments)

        symptoms = uniq_preserve_order([s.title() for s in merged["SYMPTOM"] if len(s) > 3])
        diagnosis = self.extract_diagnosis(text, merged["DIAGNOSIS"])
        treatments = uniq_preserve_order([t.title() for t in merged["TREATMENT"] if len(t) > 3])

        current_status = self.extract_current_status(text)
        prognosis = self.extract_prognosis(text)

        result = {
            "Patient_Name": patient_name,
            "Symptoms": symptoms or ["None identified"],
            "Diagnosis": diagnosis,
            "Treatment": treatments or ["None identified"],
            "Current_Status": current_status,
            "Prognosis": prognosis
        }

        return result


if __name__ == "__main__":
    extractor = MedicalExtractor()

    text = """
    Patient: Mrs. Janet Jones
    She reported neck pain and back pain after a head impact.
    Diagnosis: Whiplash injury.
    Treatment plan: 10 physiotherapy sessions and painkillers.
    She currently experiences occasional backache, but a full recovery is expected within six months.
    """

    result = extractor.summarize(text)
    print("\n" + "="*70)
    print("MEDICAL SUMMARY")
    print("="*70)
    print(json.dumps(result, indent=2))